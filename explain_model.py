import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_FILE = 'data/AP7_Final_Training_Set.csv'
MODEL_FILE = 'models/accident_xgboost.pkl'

print("Loading data and model...")
model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

# Prepare data (We use a sample of the test set to avoid RAM saturation)
X = df.drop(columns=['Y_ACCIDENT', 'timestamp_hora', 'station_id'])
# Random sample of 5000 rows for SHAP explanation (SHAP is slow)
X_sample = X.sample(5000, random_state=42)

print("Calculating SHAP values (Explainability)...")
print("   (This may take a few minutes...)")

# Create the explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

#  Global Summary (Beeswarm) 
print("Generating summary plot...")
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("Impact of Each Variable on Accident Risk")
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png')
print("   -> Saved: shap_summary_beeswarm.png")

#  Bar Importance
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("Mean Variable Importance")
plt.tight_layout()
plt.savefig('shap_importance_bar.png')
print("   -> Saved: shap_importance_bar.png")

# Partial Dependence (Example: Rain) 
# Analyze how rain affects risk (Is it linear? Does it spike after certain mm?)
try:
    print("Generating dependence plot for 'precipitation'...")
    plt.figure()
    shap.dependence_plot("precipitation", shap_values, X_sample, show=False)
    plt.title("Effect of Rain on Risk")
    plt.tight_layout()
    plt.savefig('shap_dependence_rain.png')
    print("   -> Saved: shap_dependence_rain.png")
except Exception as e:
    print(f"Could not generate dependence plot: {e}")

print("\nSHAP analysis completed.")
print("INTERPRETATION:")
print("- Beeswarm: Los puntos rojos a la derecha significan que valores altos de esa variable aumentan el riesgo.")
print("- Si ves puntos azules a la derecha, valores bajos aumentan el riesgo.")