"""
Train precision temporal model for dynamic ASL phrase gestures from SignAlphaSet.
Uses velocity-augmented trajectories (32, 126) and Bi-directional LSTM.
Addresses:
- ASL dynamic 'NO' accuracy & sensitivity
- Trajectory velocity separation
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

from ml.preprocess import compute_temporal_precision_features
from ml.dataset import load_dynamic_asl_landmarks, split_d2_data
from ml.config import MODEL_DIR

def build_precision_asl_lstm(num_classes=5, seq_len=32, feat_dim=126):
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, feat_dim)),
        keras.layers.Masking(mask_value=0.0),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False, dropout=0.3)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def augment_asl_data(X, y, factor=8):
    augmented_X = [X]
    augmented_y = [y]
    
    for _ in range(factor - 1):
        X_aug = X.copy()
        noise = np.random.normal(0, 0.02, (X_aug.shape[0], X_aug.shape[1], 63)).astype(np.float32)
        X_aug[:, :, :63] = X_aug[:, :, :63] + noise
        
        scales = np.random.uniform(0.9, 1.1, (X_aug.shape[0], 1, 1)).astype(np.float32)
        X_aug[:, :, :63] = X_aug[:, :, :63] * scales
        
        for i in range(len(X_aug)):
            X_aug[i] = compute_temporal_precision_features(X_aug[i, :, :63])
            
        augmented_X.append(X_aug)
        augmented_y.append(y.copy())
        
    return np.concatenate(augmented_X), np.concatenate(augmented_y)

def main():
    print("=" * 60)
    print("PHASE 6: Training Precision Temporal Model for ASL Dynamic Phrases")
    print("=" * 60)
    
    classes = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
    X_raw, y, groups, label_names = load_dynamic_asl_landmarks(classes=classes)
    print(f"Loaded raw sequences: {X_raw.shape}, {len(label_names)} classes: {label_names}")
    
    # Compute velocity features (N, 32, 126)
    print("Computing velocity-augmented sequences (32, 126)...")
    X_feats = np.array([compute_temporal_precision_features(seq) for seq in X_raw], dtype=np.float32)
    
    # Video-level split
    X_train, X_val, y_train, y_val = split_d2_data(X_feats, y, groups, test_size=0.3, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Augment
    X_train_aug, y_train_aug = augment_asl_data(X_train, y_train, factor=8)
    print(f"Train after augmentation: {len(X_train_aug)}")
    
    model = build_precision_asl_lstm(num_classes=len(label_names), seq_len=32, feat_dim=126)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=18, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6
        )
    ]
    
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=70,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
    
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm = confusion_matrix(y_val, y_pred, labels=range(len(label_names)))
    
    no_idx = label_names.index('NO')
    print("\n--- PRECISION VERIFICATION ON ASL 'NO' ---")
    total_no = cm[no_idx].sum()
    correct_no = cm[no_idx, no_idx]
    print(f"ASL 'NO' Val Samples: {total_no}")
    print(f"ASL 'NO' Correct: {correct_no} ({correct_no / (total_no + 1e-7) * 100:.1f}%)")
    misclassified_as = {label_names[i]: int(cm[no_idx, i]) for i in range(len(label_names)) if i != no_idx and cm[no_idx, i] > 0}
    print(f"ASL 'NO' Misclassifications: {misclassified_as}")
    
    # Save model and labels
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "temporal_letters_phrases.keras")
    model.save(model_path)
    print(f"[OK] Saved model to {model_path}")
    
    labels_path = os.path.join(MODEL_DIR, "temporal_letters_phrases_labels.json")
    with open(labels_path, "w") as f:
        json.dump(label_names, f)
    print(f"[OK] Saved labels to {labels_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Precision ASL Dynamic Phrases Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(os.path.dirname(__file__), "..", "docs", "asl_gestures_precision_cm.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
