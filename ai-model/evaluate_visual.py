import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# PATHS
# ===============================
DATASET_DIR = "visual_dataset"
MODEL_PATH = "../backend/models/visual_cnn.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

CLASS_NAMES = ["black", "green", "red", "yellow"]

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ===============================
# DATA GENERATOR (NO AUGMENTATION)
# ===============================
datagen = ImageDataGenerator(rescale=1./255)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ===============================
# PREDICTIONS
# ===============================
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# ===============================
# CLASSIFICATION REPORT
# ===============================
print("\n📊 Classification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=3
))

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Visual CNN Confusion Matrix")
plt.tight_layout()
plt.show()
