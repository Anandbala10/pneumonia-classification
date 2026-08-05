"""
Pneumonia Detection using Chest X-Ray Images
Transfer Learning with VGG16 — Improved Pipeline
=================================================
Dataset structure expected:
    chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# 1. CONFIGURATION  ← update paths here only
# ─────────────────────────────────────────────

TRAIN_DIR  = r"D:\AnandItems\chest_xray\chest_xray\train"
TEST_DIR   = r"D:\AnandItems\chest_xray\chest_xray\test"

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16          # faster & more stable than 4  32
EPOCHS        = 3          # EarlyStopping will cut short if needed   15
FINETUNE_EPOCHS = 2
LR_INITIAL    = 1e-4
LR_FINETUNE   = 1e-5
MODEL_PATH    = "best_pneumonia_model.keras"

# ─────────────────────────────────────────────
# 2. DATA GENERATORS
# ─────────────────────────────────────────────

# Training: augmentation to improve generalisation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,   # 80% train / 20% val from TRAIN_DIR
)

# Test: only rescale — no augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

validation_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,           # must be False for correct label alignment
)

# Detect class count dynamically (same as your original approach)
NUM_CLASSES  = len(training_set.class_indices)
CLASS_NAMES  = list(training_set.class_indices.keys())

print(f"\nClasses      : {training_set.class_indices}")
print(f"Train samples: {training_set.samples}")
print(f"Val   samples: {validation_set.samples}")
print(f"Test  samples: {test_set.samples}\n")

# ─────────────────────────────────────────────
# 3. BUILD MODEL  (VGG16 + custom head)
# ─────────────────────────────────────────────

base_model = VGG16(
    input_shape=(*IMG_SIZE, 3),
    weights="imagenet",
    include_top=False,
)
base_model.trainable = False   # freeze all VGG16 layers initially

# Custom classifier on top
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)                          # regularisation → reduces overfitting
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output, name="Pneumonia_VGG16")

model.compile(
    optimizer=Adam(learning_rate=LR_INITIAL),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ─────────────────────────────────────────────
# 4. CALLBACKS
# ─────────────────────────────────────────────

callbacks = [
    # Stop if val_loss doesn't improve for 4 consecutive epochs
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    # Save only the best checkpoint
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

# ─────────────────────────────────────────────
# 5. PHASE 1 — Train with frozen base
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("PHASE 1: Training with frozen VGG16 base")
print("="*60 + "\n")

# No manual steps_per_epoch — Keras infers from generator automatically
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────
# 6. PHASE 2 — Fine-tune last VGG16 block
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("PHASE 2: Fine-tuning last VGG16 block (block5)")
print("="*60 + "\n")

# Unfreeze only the last convolutional block (last 4 layers of VGG16)
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with a much lower LR to avoid disrupting pretrained weights
model.compile(
    optimizer=Adam(learning_rate=LR_FINETUNE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

finetune_callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
]

history_fine = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=FINETUNE_EPOCHS,
    callbacks=finetune_callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────
# 7. PLOT TRAINING CURVES
# ─────────────────────────────────────────────

def plot_training_curves(h1, h2):
    """Merge phase-1 and phase-2 histories and plot accuracy + loss."""
    acc     = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss    = h1.history["loss"]         + h2.history["loss"]
    val_loss= h1.history["val_loss"]     + h2.history["val_loss"]
    epochs  = range(1, len(acc) + 1)
    phase1_end = len(h1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History (Phase 1 + Fine-tuning)", fontsize=13, fontweight="bold")

    # Accuracy
    ax1.plot(epochs, acc,     label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, val_acc, label="Val Accuracy",   linewidth=2, linestyle="--")
    ax1.axvline(x=phase1_end, color="gray", linestyle=":", label="Fine-tune start")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, loss,     label="Train Loss", linewidth=2)
    ax2.plot(epochs, val_loss, label="Val Loss",   linewidth=2, linestyle="--")
    ax2.axvline(x=phase1_end, color="gray", linestyle=":", label="Fine-tune start")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved: training_curves.png")

plot_training_curves(history, history_fine)

# ─────────────────────────────────────────────
# 8. EVALUATE ON TEST SET
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60 + "\n")

best_model = load_model(MODEL_PATH)

test_loss, test_acc = best_model.evaluate(test_set, verbose=1)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}\n")

# Predict probabilities then convert to class indices
test_set.reset()
y_pred_prob = best_model.predict(test_set, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = test_set.classes

# ── Confusion Matrix ──
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# ── Classification Report ──
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 9. PREDICT ON A SINGLE IMAGE
# ─────────────────────────────────────────────

def predict_image(image_path, model_path=MODEL_PATH):
    """
    Predict NORMAL or PNEUMONIA for a single chest X-ray image.

    Args:
        image_path : full path to the image file (.jpg / .jpeg / .png)
        model_path : path to the saved Keras model
    """
    mdl   = load_model(model_path)
    img   = load_img(image_path, target_size=IMG_SIZE)
    arr   = img_to_array(img) / 255.0
    arr   = np.expand_dims(arr, axis=0)          # shape: (1, 224, 224, 3)

    probs           = mdl.predict(arr, verbose=0)[0]
    predicted_idx   = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = probs[predicted_idx]

    print(f"\n{'─'*45}")
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {predicted_class}")
    print(f"  Confidence : {confidence:.2%}")
    print(f"{'─'*45}\n")
    return predicted_class, confidence


# ── Example — update to any real test image path ──
img_path = r"D:\AnandItems\chest_xray\chest_xray\test\PNEUMONIA\person29_virus_64.jpeg"
predict_image(img_path)

# ─────────────────────────────────────────────
# 10. DONE
# ─────────────────────────────────────────────

print("="*60)
print(f"  Best model saved : {MODEL_PATH}")
print(f"  Test accuracy    : {test_acc:.4f}")
print("  Plots saved      : training_curves.png, confusion_matrix.png")
print("="*60)