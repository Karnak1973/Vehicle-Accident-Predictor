import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

DATA_FILE = 'data/AP7_Final_Training_Set.csv'
MODEL_FILE = 'models/accident_lstm_v2.keras'
SCALER_FILE = 'models/scaler.pkl'

# Hyperparameters
LOOKBACK_HOURS = 6   # Time window (in hours) the model looks back to predict
NEGATIVE_RATIO = 20  # Aggresive ratio: for every 1 accident, 20 no-accident samples
BATCH_SIZE = 64    # Batch size for training (adjust based on GPU/RAM)
EPOCHS = 30          # Max epochs
TEST_SIZE = 0.2      # 20% of data used for testing (chronological split)

# Columns to EXCLUDE from training (Metadata, not predictive features)
#* Note: 'station_id' is kept as a feature but needs to be Label Encoded properly.
COLS_TO_DROP = ['timestamp_hora', 'segmento_pk', 'station_id'] 


def create_sequences(X, y, time_steps=1):
    """
    Transforms 2D data (Samples, Features) into 3D sequences for LSTM.
    Input: (Total_Samples, Features) -> Output: (New_Samples, TimeSteps, Features)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # Sequence X: looks back 'time_steps' hours
        Xs.append(X[i:(i + time_steps)])
        # Target Y: the accident status at the time step immediately after the sequence
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def load_and_preprocess_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # CRITICAL: LSTM requires data sorted chronologically
    df['timestamp_hora'] = pd.to_datetime(df['timestamp_hora'])
    df = df.sort_values(['timestamp_hora', 'segmento_pk'])
    
    # Separate Target
    y_raw = df['Y_ACCIDENT'].values
    
    # Separate Features (X)
    X_raw = df.drop(columns=['Y_ACCIDENT'] + COLS_TO_DROP)
    
    feature_names = X_raw.columns.tolist()
    print(f"Features selected ({len(feature_names)}): {feature_names}")
    
    # Scale data (0 to 1) - Vital for neural networks
    print("Scaling features (MinMax)...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Save the scaler for use in future real-time predictions
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    
    return X_scaled, y_raw, feature_names

def perform_undersampling(X, y, ratio=20):
    """Selects all positive samples (1) and a subset of negative samples (0) to balance the dataset."""
    idx_1 = np.where(y == 1)[0]
    idx_0 = np.where(y == 0)[0]
    
    # Select a subset of negative samples randomly
    # Avoid selecting contiguous blocks to prevent temporal leakage,
    n_negatives = len(idx_1) * ratio
    idx_0_selected = np.random.choice(idx_0, size=n_negatives, replace=False)
    
    # Join indices and sort
    indices = np.concatenate([idx_1, idx_0_selected])
    indices.sort()
    
    X_sampled = X[indices]
    y_sampled = y[indices]
    
    print(f"   Original data: {len(y)}")
    print(f"   Balanced data: {len(y_sampled)} (Positives: {len(idx_1)})")
    
    return X_sampled, y_sampled

def build_model(input_shape):
    """Defines the LSTM architecture."""
    model = Sequential()
    
    # First LSTM layer (returns sequences for the next LSTM layer)
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization()) # Helps convergence
    model.add(Dropout(0.3)) # Increased dropout slightly for robustness
    
    # Second LSTM layer (returns only the final output vector)
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Dense layer for interpretation
    model.add(Dense(16, activation='relu'))
    
    # Output: 1 neuron with Sigmoid (Probability 0 to 1)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compiler: Use AUC as the primary metric for imbalanced data
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.AUC(name='auc'), 'precision', 'recall']
    )
    return model

if __name__ == "__main__":
    
    # 1. Load and Preprocess Data
    X_scaled, y_raw, features = load_and_preprocess_data()
    
    # 2. Create Sequences
    print(f"Generating temporal sequences (Lookback: {LOOKBACK_HOURS}h)...")
    X_seq, y_seq = create_sequences(X_scaled, y_raw, time_steps=LOOKBACK_HOURS)
    
    print(f"   X Dimensions: {X_seq.shape}")
    print(f"   Y Dimensions: {y_seq.shape}")
    
    # 3. Split Train/Test (Chronological Split)
    split_idx = int(len(X_seq) * (1 - TEST_SIZE))
    
    X_train_raw = X_seq[:split_idx]
    y_train_raw = y_seq[:split_idx]
    
    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]
    
    print(f"Test set (original):  {len(y_test)} samples")
    
    # 4. Apply Undersampling to Training Data
    # This is essential to prevent the model from always predicting 0.
    X_train_bal, y_train_bal = perform_undersampling(X_train_raw, y_train_raw, ratio=NEGATIVE_RATIO)
    
    # 5. Build and Train Model
    model = build_model((X_train_bal.shape[1], X_train_bal.shape[2]))
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'), # Increased patience
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_auc', mode='max')
    ]
    
    print("Starting Model Training...")
    history = model.fit(
        X_train_bal, y_train_bal,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluation
    print("\nEvaluating model...")
    predictions = model.predict(X_test)
    
    # Histogram of predicted probabilities
    plt.figure(figsize=(10, 5))
    plt.hist(predictions[y_test==0], bins=50, alpha=0.5, label='No Accident', log=True) 
    plt.hist(predictions[y_test==1], bins=50, alpha=0.5, label='Yes Accident', color='red')
    plt.title('Distribución de Probabilidades Predichas')
    plt.legend()
    plt.savefig('probability_distribution.png')

    auc = roc_auc_score(y_test, predictions)
    print(f"ROC AUC Score: {auc:.4f}")
    
    # Adjust threshold for better Recall (catching more accidents)
    # The default threshold of 0.5 often misses the rare positive class.

    # Find the best threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, predictions)
    # Buscamos un umbral que nos de al menos un 60-70% de Recall
    target_recall = 0.70
    idx = np.argmax(recalls <= target_recall) # Primer índice donde recall cae por debajo de 0.7
    best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    y_pred_binary = (predictions > best_threshold).astype(int)
    
    print(f"\n--- Results (Threshold={best_threshold}) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    print(f"Model saved to {MODEL_FILE}")
    print("Training phase complete!")