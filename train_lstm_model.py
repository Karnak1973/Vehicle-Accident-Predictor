import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_FILE = 'data/AP7_Final_Training_Set.csv'
MODEL_FILE = 'models/accident_lstm_v1.keras'
SCALER_FILE = 'models/scaler.pkl'

# Hyperparameters
LOOKBACK_HOURS = 6   # Time window (in hours) the model looks back to predict
BATCH_SIZE = 1024    # Batch size for training (adjust based on GPU/RAM)
EPOCHS = 20          # Max epochs
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
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    
    return X_scaled, y_raw, feature_names

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
    
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # 4. Calculate Class Weights (Handling Imbalance)
    # This is essential to prevent the model from always predicting 0.
    neg = np.bincount(y_train)[0]
    pos = np.bincount(y_train)[1]
    total = neg + pos
    
    # Weight calculation: balance the classes
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated Class Weights: Class 0: {weight_for_0:.4f}, Class 1 (Accident): {weight_for_1:.4f}")
    
    # 5. Build and Train Model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'), # Increased patience
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_auc', mode='max')
    ]
    
    print("Starting Model Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluation
    print("\nEvaluating model...")
    predictions = model.predict(X_test)
    
    auc = roc_auc_score(y_test, predictions)
    print(f"ROC AUC Score: {auc:.4f}")
    
    # Adjust threshold for better Recall (catching more accidents)
    # The default threshold of 0.5 often misses the rare positive class.
    THRESHOLD = 0.3 # Example: Lower the threshold to increase sensitivity
    y_pred_binary = (predictions > THRESHOLD).astype(int)
    
    print(f"\n--- Results (Threshold={THRESHOLD}) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    print(f"Model saved to {MODEL_FILE}")
    print("Training phase complete!")