"""
Train precision static landmark-based MLP for SignAlphaSet letters (25 classes: A-I, K-Z, excluding static J).
Uses 87-dimensional precision features:
- Canonical in-plane rotation normalization
- 8 relative fingertip distance ratios (resolves C vs G, U vs V)
- 6 depth-order / occlusion features
- 10 joint flexion angles
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ml.preprocess import compute_precision_features
from ml.dataset import load_letters_landmarks, split_d1_data
from ml.config import MODEL_DIR, STATIC_EPOCHS, STATIC_BATCH_SIZE

def build_precision_letters_mlp(input_dim=87, num_classes=25):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("=" * 60)
    print("PHASE 2 & 3: Training Precision Static Letters (25 classes, excluding J)")
    print("=" * 60)
    
    # Load cached static landmarks
    X_raw, y_raw, label_names_raw = load_letters_landmarks()
    print(f"Loaded raw data: {X_raw.shape}, {len(label_names_raw)} classes")
    
    # Filter out class 'J' (index 9)
    j_idx = label_names_raw.index('J') if 'J' in label_names_raw else -1
    mask = (y_raw != j_idx)
    X_filtered = X_raw[mask]
    y_filtered = y_raw[mask]
    
    # Remap y labels: indices after j_idx shift down by 1
    y_remapped = np.array([idx if idx < j_idx else idx - 1 for idx in y_filtered], dtype=np.int32)
    label_names = [l for l in label_names_raw if l != 'J']
    print(f"Filtered out 'J'. Remaining classes ({len(label_names)}): {label_names}")
    
    # Compute 87D precision features
    print("Computing 87D precision features...")
    X_feats = np.array([compute_precision_features(sample) for sample in X_filtered], dtype=np.float32)
    print(f"Computed features: {X_feats.shape}")
    
    # Train / Val / Test split (70 / 15 / 15)
    X_train, X_val, X_test, y_train, y_val, y_test = split_d1_data(X_feats, y_remapped, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model = build_precision_letters_mlp(input_dim=87, num_classes=len(label_names))
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=STATIC_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    # Check specific pairs: C vs G, U vs V
    c_idx = label_names.index('C')
    g_idx = label_names.index('G')
    u_idx = label_names.index('U')
    v_idx = label_names.index('V')
    
    print("\n--- PRECISION VERIFICATION ON CONFUSION PAIRS ---")
    print(f"C samples: {cm[c_idx].sum()}, C->G errors: {cm[c_idx, g_idx]}")
    print(f"G samples: {cm[g_idx].sum()}, G->C errors: {cm[g_idx, c_idx]}")
    print(f"U samples: {cm[u_idx].sum()}, U->V errors: {cm[u_idx, v_idx]}")
    print(f"V samples: {cm[v_idx].sum()}, V->U errors: {cm[v_idx, u_idx]}")
    
    # Save model and labels
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "landmark_letters.keras")
    model.save(model_path)
    print(f"[OK] Saved model to {model_path}")
    
    labels_path = os.path.join(MODEL_DIR, "landmark_letters_labels.json")
    with open(labels_path, "w") as f:
        json.dump(label_names, f)
    print(f"[OK] Saved labels to {labels_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Precision Static Letters Confusion Matrix (25 Classes)")
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(os.path.dirname(__file__), "..", "docs", "letters_precision_cm.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
