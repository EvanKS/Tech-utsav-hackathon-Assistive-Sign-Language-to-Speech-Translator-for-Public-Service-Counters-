"""
Train the static landmark-based MLP model for SignAlphaSet letters (A-Z).
Mirrors train_static_landmarks.py exactly:
63 -> Dense(128) -> BatchNorm -> Dropout(0.3) -> Dense(64) -> BatchNorm -> Dropout(0.3) -> Dense(26, softmax).
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

from ml.dataset import load_letters_landmarks, split_d1_data
from ml.config import MODEL_DIR, STATIC_EPOCHS, STATIC_BATCH_SIZE, FEATURE_DIM


def build_letters_mlp(num_classes=26):
    """
    MLP architecture matching landmark_digits:
    63 -> 128 -> 64 -> 26
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(FEATURE_DIM,)),
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


def train(max_per_class=300):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 2: Training Static Landmark Letters Classifier (A-Z)")
    print("=" * 60)
    
    X, y, label_names = load_letters_landmarks(max_per_class=max_per_class)
    print(f"\nTotal samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes ({len(label_names)}): {label_names}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_d1_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model = build_letters_mlp(num_classes=len(label_names))
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=STATIC_EPOCHS,
        batch_size=STATIC_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'=' * 60}")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=label_names)
    print("\nClassification Report:")
    print(report)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "landmark_letters.keras")
    model.save(model_path)
    print(f"[OK] Model saved to {model_path}")
    
    # Save labels
    labels_path = os.path.join(MODEL_DIR, "landmark_letters_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(label_names, f)
    print(f"[OK] Labels saved to {labels_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
    os.makedirs(docs_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap='Purples')
    ax.set_title('Confusion Matrix — SignAlphaSet Static Letters (A-Z)', fontsize=14)
    fig.colorbar(im)
    tick_marks = np.arange(len(label_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(label_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(label_names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
    
    plt.tight_layout()
    cm_path = os.path.join(docs_dir, "confusion_matrix_letters.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved to {cm_path}")
    
    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Letters Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Letters Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    hist_path = os.path.join(docs_dir, "training_history_letters.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"[OK] Training history saved to {hist_path}")
    
    return {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "total_samples": len(X),
        "report": report
    }


if __name__ == "__main__":
    results = train()
    print("\n[OK] Static letters training complete!")
    print(json.dumps({k: v for k, v in results.items() if k != 'report'}, indent=2))
