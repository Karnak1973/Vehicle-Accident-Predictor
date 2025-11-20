import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib

# --- CONFIG ---
DATA_FILE = 'data/AP7_Final_Training_Set.csv'
MODEL_FILE = 'models/accident_xgboost_tuned.pkl'

print("📂 Cargando datos...")
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=['Y_ACCIDENT', 'timestamp_hora', 'station_id']) 
y = df['Y_ACCIDENT']

# Split Cronológico
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Calcular ratio base
negatives = (y_train == 0).sum()
positives = (y_train == 1).sum()
base_ratio = negatives / positives
print(f"⚖️ Ratio base: {base_ratio:.2f}")

# --- GRID DE PARÁMETROS A PROBAR ---
# RandomizedSearchCV probará combinaciones aleatorias de estos valores
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],              # % de filas usadas por árbol (reduce overfit)
    'colsample_bytree': [0.7, 0.8, 1.0],       # % de columnas usadas por árbol
    'scale_pos_weight': [base_ratio, base_ratio * 1.5, base_ratio * 2.0], # Jugar con la agresividad
    'reg_alpha': [0, 0.1, 1.0],                # Regularización L1 (reduce ruido)
    'reg_lambda': [1.0, 1.5, 2.0]              # Regularización L2
}

# Configurar XGBoost base
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    tree_method='hist' # Más rápido para datos grandes
)

# Configurar búsqueda aleatoria (Más rápido que GridSearch exhaustivo)
# n_iter=20 significa que probará 20 combinaciones distintas
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20, 
    scoring='roc_auc',
    cv=3, # Validación cruzada de 3 pliegues (Cross-Validation)
    verbose=1,
    n_jobs=1, # Usamos 1 job aquí porque XGBoost ya usa todos los cores internamente
    random_state=42
)

print(f"🚀 Iniciando búsqueda de hiperparámetros (20 combinaciones)...")
# Entrenamos SOLO con una muestra del Train set para ir rápido (ej. 20%)
# O usamos todo el train si tienes tiempo/máquina potente.
# Para este ejemplo, usamos una muestra aleatoria del 50% del train para agilizar.
sample_size = int(len(X_train) * 0.5)
indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[indices]
y_train_sample = y_train.iloc[indices]

search.fit(X_train_sample, y_train_sample)

# --- RESULTADOS ---
print("\n✅ Mejores parámetros encontrados:")
print(search.best_params_)
print(f"Mejor AUC en validación: {search.best_score_:.4f}")

# --- ENTRENAMIENTO FINAL CON MEJORES PARÁMETROS ---
print("\n🚀 Re-entrenando modelo final con TODOS los datos de train...")
best_model = search.best_estimator_
best_model.fit(X_train, y_train)

# --- EVALUACIÓN ---
print("\n📝 Evaluando modelo optimizado...")
probs = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"🌟 ROC AUC Score Final: {auc:.4f}")

# Threshold Tuning
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
target_recall = 0.55 # Busquemos un punto medio (55%)
idx = np.argmax(recalls <= target_recall) 
best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

print(f"✅ Umbral seleccionado: {best_threshold:.6f} (Recall ~{target_recall*100}%)")

preds_optimized = (probs > best_threshold).astype(int)

print("\n--- RESULTADOS FINALES ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds_optimized))
print("\nReport:")
print(classification_report(y_test, preds_optimized))

joblib.dump(best_model, MODEL_FILE)
print(f"💾 Modelo optimizado guardado en {MODEL_FILE}")