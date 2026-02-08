import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter

# ================= PATHS =================

DATASET_DIR = "visual_dataset"          # inside ai-model/
MODEL_DIR = "backend/models"             # backend will load from here

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= CONFIG =================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 40

# ================= DATA =================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.85, 1.15]
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

# ================= SANITY CHECKS =================

print("Classes:", train_gen.class_indices)
print("Class distribution:", Counter(train_gen.classes))

num_classes = len(train_gen.class_indices)

# ================= MODEL (TRANSFER LEARNING) =================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# ================= COMPILE =================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================= CLASS WEIGHTS =================
# (Adjust only if distribution is very skewed)

class_weights = {
    0: 1.0,   # Green
    1: 1.3,   # Yellow
    2: 1.8,   # Red
    3: 2.2    # Black
}

# ================= CALLBACKS =================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

# ================= TRAIN =================

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ================= SAVE MODEL =================

model_path = os.path.join(MODEL_DIR, "visual_cnn.h5")
model.save(model_path)

print("✅ Model saved at:", model_path)
