"""
Train precision static landmark-based MLP for Digits (0-9) using 87D precision features.
Addresses digit 3 registration issues via canonical rotation and thumb extension features.
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
from ml.dataset import load_d1_landmarks, split_d1_data
from ml.config import MODEL_DIR, STATIC_BATCH_SIZE

def build_precision_digits_mlp(input_dim=87, num_classes=10):
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
    print("PHASE 4: Training Precision Digits Classifier (0-9)")
    print("=" * 60)
    
    X_raw, y, label_names = load_d1_landmarks()
    print(f"Loaded raw digits: {X_raw.shape}, {len(label_names)} classes")
    
    # Compute 87D precision features
    print("Computing 87D precision features...")
    X_feats = np.array([compute_precision_features(sample) for sample in X_raw], dtype=np.float32)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_d1_data(X_feats, y, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model = build_precision_digits_mlp(input_dim=87, num_classes=len(label_names))
    
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
    
    idx_3 = label_names.index('3')
    print("\n--- PRECISION VERIFICATION ON DIGIT 3 ---")
    total_3 = cm[idx_3].sum()
    correct_3 = cm[idx_3, idx_3]
    print(f"Digit 3 Test Samples: {total_3}")
    print(f"Digit 3 Correct: {correct_3} ({correct_3 / total_3 * 100:.1f}%)")
    misclassified_as = {label_names[i]: int(cm[idx_3, i]) for i in range(len(label_names)) if i != idx_3 and cm[idx_3, i] > 0}
    print(f"Digit 3 Misclassifications: {misclassified_as}")
    
    # Save model and labels
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "landmark_digits.keras")
    model.save(model_path)
    print(f"[OK] Saved model to {model_path}")
    
    labels_path = os.path.join(MODEL_DIR, "landmark_digits_labels.json")
    with open(labels_path, "w") as f:
        json.dump(label_names, f)
    print(f"[OK] Saved labels to {labels_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Precision Digits Confusion Matrix (0-9)")
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(os.path.dirname(__file__), "..", "docs", "digits_precision_cm.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
