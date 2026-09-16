"""
Train GRU-based temporal model for ISL word recognition (D2).
Uses video-level group split to prevent data leakage.
Falls back to frame-level classification + majority voting if insufficient data.
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

from ml.dataset import load_d2_landmarks, split_d2_data
from ml.config import MODEL_DIR, TEMPORAL_EPOCHS, TEMPORAL_BATCH_SIZE, FEATURE_DIM, TEMPORAL_SEQUENCE_LENGTH


def build_gru_model(num_classes, seq_len=TEMPORAL_SEQUENCE_LENGTH):
    """
    GRU: (seq_len, 63) → GRU(64) → Dense(32) → softmax
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, FEATURE_DIM)),
        keras.layers.Masking(mask_value=0.0),
        keras.layers.GRU(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
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


def augment_temporal_data(X, y, factor=5):
    """
    Data augmentation for temporal sequences:
    - Random temporal jitter (shift frame indices)
    - Random landmark noise
    - Random scaling
    """
    augmented_X = [X]
    augmented_y = [y]
    
    for _ in range(factor - 1):
        X_aug = X.copy()
        
        # Add small Gaussian noise to landmarks
        noise = np.random.normal(0, 0.02, X_aug.shape).astype(np.float32)
        X_aug = X_aug + noise
        
        # Random scaling per sample
        scales = np.random.uniform(0.9, 1.1, (X_aug.shape[0], 1, 1)).astype(np.float32)
        X_aug = X_aug * scales
        
        # Random temporal shift (roll frames)
        for i in range(len(X_aug)):
            shift = np.random.randint(-3, 4)
            X_aug[i] = np.roll(X_aug[i], shift, axis=0)
        
        augmented_X.append(X_aug)
        augmented_y.append(y.copy())
    
    return np.concatenate(augmented_X), np.concatenate(augmented_y)


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 8: Training Temporal GRU Word Classifier")
    print("=" * 60)
    
    X, y, groups, label_names = load_d2_landmarks()
    print(f"\nTotal sequences: {X.shape[0]}, Shape: {X.shape}")
    print(f"Classes: {label_names}")
    
    # Video-level split (prevents data leakage)
    X_train, X_val, y_train, y_val = split_d2_data(X, y, groups, test_size=0.3)
    print(f"Train: {len(X_train)}, Val: {len(X_val)} (video-level split)")
    
    # Augment training data (7 videos/class is very small)
    X_train_aug, y_train_aug = augment_temporal_data(X_train, y_train, factor=8)
    print(f"After augmentation: {len(X_train_aug)} training sequences")
    
    # Build and train
    model = build_gru_model(num_classes=len(label_names))
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
        )
    ]
    
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=TEMPORAL_EPOCHS,
        batch_size=TEMPORAL_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"{'=' * 60}")
    
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(y_val, y_pred, target_names=label_names)
    print("\nClassification Report:")
    print(report)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "temporal_words.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    labels_path = os.path.join(MODEL_DIR, "temporal_words_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(label_names, f)
    print(f"Labels saved to {labels_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
    os.makedirs(docs_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Oranges')
    ax.set_title('Confusion Matrix — Temporal Word Classifier (GRU)', fontsize=14)
    fig.colorbar(im)
    tick_marks = np.arange(len(label_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(label_names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    cm_path = os.path.join(docs_dir, "confusion_matrix_words.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    return {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "train_samples": len(X_train),
        "augmented_samples": len(X_train_aug),
        "val_samples": len(X_val),
        "report": report
    }


if __name__ == "__main__":
    results = train()
    print("\n✓ Temporal model training complete!")
    print(json.dumps({k: v for k, v in results.items() if k != 'report'}, indent=2))
