# app/vision_model.py
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---- CONFIG ----
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Black", "Green", "Red", "Yellow"]

MODEL_PATH = "models/visual_cnn.h5"

# ---- LOAD MODEL ONCE ----
model = load_model(MODEL_PATH)
print("✅ Visual CNN loaded")

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_visual(image_bytes: bytes):
    x = preprocess_image(image_bytes)
    preds = model.predict(x, verbose=0)[0]

    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    label = CLASS_NAMES[int(np.argmax(preds))]
    confidence = float(np.max(preds))

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probs
    }
