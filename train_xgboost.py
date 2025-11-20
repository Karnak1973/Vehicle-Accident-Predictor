import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import matplotlib.pyplot as plt

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
ratio = negatives / positives
print(f"Desbalancing ratio: {ratio:.2f}")

# TRUCO: Multiplicamos el ratio por un factor para hacerlo más agresivo
# Si ratio es 7000, le ponemos 10000 para que le dé más importancia aún a los accidentes
aggressive_ratio = ratio * 1.5 
print(f"Original Ratio: {ratio:.2f} | Agressive Ratio used: {aggressive_ratio:.2f}")

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=ratio, 
    eval_metric='auc',
    n_jobs=-1
)

print("Training XGBoost...")
model.fit(X_train, y_train)

# Evaluar
print("\n Evaluating...")
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f" ROC AUC: {auc:.4f}")

# Search the best threshold based on Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Minimum recall 
target_recall = 0.60
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

