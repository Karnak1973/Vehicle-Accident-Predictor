import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib

DATA_FILE = 'data/AP7_Final_Training_Set.csv'
MODEL_FILE = 'models/accident_xgboost.pkl'
print("Loading data...")
df = pd.read_csv(DATA_FILE)

# Delete columns not needed for XGBoost (training will handle categorical features internally)
X = df.drop(columns=['Y_ACCIDENT', 'timestamp_hora', 'station_id']) 
y = df['Y_ACCIDENT']

# Temporal split 
# No random shuffling to respect chronological order
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("Data split:", f"Train size: {len(X_train)}", f"Test size: {len(X_test)}")

print(f"Training with {len(X_train)} rows...")

# Caulculate ratio 
# scale_pos_weight = count(negative) / count(positive)
negatives = (y_train == 0).sum()
positives = (y_train == 1).sum()
base_ratio = negatives / positives
print(f"Ratio base: {base_ratio:.2f}")

# TRUCO: Multiplicamos el ratio por un factor para hacerlo más agresivo
# Si ratio es 7000, le ponemos 10000 para que le dé más importancia aún a los accidentes
# aggressive_ratio = ratio * 1.5 
# print(f"Original Ratio: {ratio:.2f} | Agressive Ratio used: {aggressive_ratio:.2f}")

# * GRID OF PARAMETERS TO TEST
# RandomizedSearchCV trys random combinations of these values
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],              # % of rows used by tree (reduce overfit)
    'colsample_bytree': [0.7, 0.8, 1.0],       # % of columns used by tree
    'scale_pos_weight': [base_ratio, base_ratio * 1.5, base_ratio * 2.0], # Jugar con la agresividad
    'reg_alpha': [0, 0.1, 1.0],                # Regularization L1 (reduce ruido)
    'reg_lambda': [1.0, 1.5, 2.0]              # Regularization L2
}

# Configuration XGBoost base
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    tree_method='hist' # Más rápido para datos grandes
)

# Config random search (Faster than exhaustive GridSearch)
# n_iter=20 will try 20 different combinations
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20, 
    scoring='roc_auc',
    cv=3, # Cross-Validation with 3 folds
    verbose=1,
    n_jobs=1, # 1 job here because XGBoost already uses all cores internally
    random_state=42
)
print(f"Starting hyperparameter search (20 combinations)...")
# We train ONLY with a sample of the Train set to go fast (eg. 20%)
#? For this example, we use a random sample of 50% of the train to speed up
sample_size = int(len(X_train) * 0.5)
indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[indices]
y_train_sample = y_train.iloc[indices]

search.fit(X_train_sample, y_train_sample)

print("Best hyperparameters found:")
print(search.best_params_)
print("Best AUC in validation: {search.best_score_:.4f}")

# FINAL TRAIN WITH THE BEST HYPERPARAMETERS (with all the train data)
print("Training XGBoost...")
model = search.best_estimator_
model.fit(X_train, y_train)

# EVALUATION
print("\n Evaluating...")
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f" ROC AUC Final Score: {auc:.4f}")

# Search the best threshold based on Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
# Minimum recall 
target_recall = 0.55
# Search the threshold that gives at least target_recall (60%)
idx = np.argmax(recalls <= target_recall) 
best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

print(f" Selected threshold: {best_threshold:.6f} (for Recall ~{target_recall*100}%)")
preds = (probs > best_threshold).astype(int)

print("\n--- FINAL RESULTS ---")
print("\n Confusion matrix:")
print(confusion_matrix(y_test, preds))
print("\nReport:")
print(classification_report(y_test, preds))

# Feature Importance (to know what variables matter most)
print("\nMost Important Variables:")
importances = model.feature_importances_
feature_names = X.columns
feat_importances = pd.Series(importances, index=feature_names)
print(feat_importances.nlargest(10))

# Guardar
joblib.dump(model, MODEL_FILE)

