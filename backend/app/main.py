import os
import io
import uuid
import tempfile
import numpy as np
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
from PIL import Image

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from security import get_current_user
from models import User
from auth import router as auth_router
from text_analyzer import analyze_text

import uvicorn


# ================= PATH SETTINGS =================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_cnn_ravdess.keras")
VISUAL_MODEL_PATH = os.path.join(MODEL_DIR, "visual_cnn.h5")

IMG_SIZE = (128, 128)

CLASS_NAMES = ["Black", "Green", "Red", "Yellow"]
AUDIO_CLASSES = ["normal", "distress", "silent"]

ALLOW_AUDIO = True


# ================= FASTAPI =================

app = FastAPI(title="Military Triage System - Backend")
app.include_router(auth_router, prefix="/auth", tags=["Auth"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= LOAD MODELS =================

audio_model = None
visual_model = None


def load_audio_model():
    global audio_model
    if os.path.exists(AUDIO_MODEL_PATH):
        audio_model = load_model(AUDIO_MODEL_PATH)
        print("✅ Audio model loaded")
    else:
        audio_model = None
        print("❌ Audio model not found")


def load_visual_model():
    global visual_model
    if os.path.exists(VISUAL_MODEL_PATH):
        visual_model = load_model(VISUAL_MODEL_PATH)
        print("✅ Visual model loaded")
    else:
        visual_model = None
        print("⚠️ Visual model not found")


load_audio_model()
load_visual_model()


# ================= PREPROCESS =================

def preprocess_image(image_bytes: bytes):
    if len(image_bytes) < 1000:
        raise HTTPException(400, "Invalid image")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def audio_to_mel(wav_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(wav_bytes)
        path = f.name

    y, sr = librosa.load(path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.Figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.subplots()
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0

    return np.expand_dims(arr, axis=0)


# ================= PREDICTION =================

def predict_audio(wav_bytes):
    if audio_model is None:
        return "normal", 1.0, [0.33, 0.33, 0.33]

    x = audio_to_mel(wav_bytes)
    preds = audio_model.predict(x, verbose=0)[0]

    if np.max(preds) < 0.6:
        return "normal", 0.5, preds.tolist()

    idx = int(np.argmax(preds))
    return AUDIO_CLASSES[idx], float(preds[idx]), preds.tolist()


def sharpen_probs(probs, temperature=0.7):
    probs = np.array(probs)
    probs = np.power(probs, 1 / temperature)
    return (probs / probs.sum()).tolist()


def predict_visual(image_bytes):
    if visual_model is None:
        return "none", 0.0, [0.25] * 4

    x = preprocess_image(image_bytes)
    preds = visual_model.predict(x, verbose=0)[0]
    preds = sharpen_probs(preds)

    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds


# ================= TRIAGE DISTRIBUTIONS =================

def audio_to_triage_dist(label):
    if label == "distress":
        return np.array([0.05, 0.25, 0.65, 0.05])
    if label == "silent":
        return np.array([0.15, 0.20, 0.25, 0.40])
    return np.array([0.80, 0.12, 0.06, 0.02])


# ================= FUSION =================

def fuse(audio_res, visual_res, text_res):
    audio_label, _, _ = audio_res
    visual_label, visual_conf, visual_probs = visual_res
    text_label, text_conf = text_res

    visual_probs = np.array(visual_probs)
    audio_dist = audio_to_triage_dist(audio_label)

    # ---------- TEXT DISTRIBUTION ----------
    if text_conf < 0.6:
        text_dist = np.array([0.25] * 4)
    elif text_label == "red" and "pain" in text_label:
        text_dist = np.array([0.55, 0.30, 0.10, 0.05])
    elif text_label == "red":
        text_dist = np.array([0.10, 0.20, 0.55, 0.15])
    elif text_label == "yellow":
        text_dist = np.array([0.15, 0.55, 0.25, 0.05])
    elif text_label == "green":
        text_dist = np.array([0.65, 0.25, 0.07, 0.03])
    else:
        text_dist = np.array([0.25] * 4)

    # ---------- WEIGHTS ----------
    if visual_conf > 0.85:
        vw, tw, aw = 0.85, 0.10, 0.05
    elif visual_conf < 0.5:
        vw, tw, aw = 0.45, 0.35, 0.20
    else:
        vw, tw, aw = 0.65, 0.20, 0.15

    fused = vw * visual_probs + tw * text_dist + aw * audio_dist

    # ---------- MEDICAL RULES ----------
    if visual_conf > 0.85 and visual_label != "Red":
        fused[2] *= 0.4  # suppress Red

    fused = fused / fused.sum()

    idx = int(np.argmax(fused))
    confidence = float(np.max(fused))

    return CLASS_NAMES[idx], fused.tolist(), confidence


# ================= ADVICE =================

def generate_advice(triage, text):
    advice = []

    if triage == "Black":
        advice += ["🚨 CRITICAL CONDITION", "Immediate evacuation required"]
    elif triage == "Red":
        advice += ["⚠️ Severe injury detected", "Seek medic immediately"]
    elif triage == "Yellow":
        advice += ["🟡 Moderate injury", "Monitor condition"]
    else:
        advice += ["🟢 Minor injury", "Clean wound"]

    if "pain" in text.lower():
        advice.append("Provide pain relief if available")

    return advice


# ================= API =================

@app.post("/predict")
async def predict(
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: str = Form(None),
    current_user: User = Depends(get_current_user)
):
    audio_out = predict_audio(await audio.read()) if audio else ("normal", 1.0, [])
    visual_out = predict_visual(await image.read()) if image else ("none", 0.0, [0.25] * 4)
    text_out = analyze_text(text) if text else ("none", 0.0)

    triage, probs, conf = fuse(audio_out, visual_out, text_out)

    return {
        "patient_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "triage_level": triage,
        "confidence": conf,
        "probabilities": dict(zip(CLASS_NAMES, probs)),
        "advice": generate_advice(triage, text or ""),
        "audio_raw": {"label": audio_out[0], "confidence": audio_out[1]},
        "visual_raw": {"label": visual_out[0], "confidence": visual_out[1], "probs": visual_out[2]},
        "text_raw": {"label": text_out[0], "confidence": text_out[1]},
    }


# ================= RUN =================

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
