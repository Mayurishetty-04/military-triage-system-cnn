import os
import io
import numpy as np
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
from PIL import Image

import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import uvicorn

from text_analyzer import analyze_text


# ---------------- PATH SETTINGS ----------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_cnn_ravdess.keras")
VISUAL_MODEL_PATH = os.path.join(MODEL_DIR, "visual_cnn.h5")

IMG_SIZE = (128, 128)

CLASS_NAMES = ["Green", "Yellow", "Red", "Black"]
AUDIO_CLASSES = ["normal", "distress", "silent"]


# ---------------- FASTAPI APP ----------------

app = FastAPI(title="Military Triage System - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- LOAD MODELS ----------------

audio_model = None
visual_model = None


def load_audio_model():
    global audio_model

    if os.path.exists(AUDIO_MODEL_PATH):
        audio_model = load_model(AUDIO_MODEL_PATH)
        print("✅ Audio model loaded")
    else:
        print("❌ Audio model not found")
        audio_model = None


def load_visual_model():
    global visual_model

    if os.path.exists(VISUAL_MODEL_PATH):
        visual_model = load_model(VISUAL_MODEL_PATH)
        print("✅ Visual model loaded")
    else:
        print("⚠️ Visual model not found (placeholder mode)")
        visual_model = None


# Load at startup
load_audio_model()
load_visual_model()


# ---------------- PREPROCESSING ----------------

def preprocess_image(image_bytes):

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img) / 255.0

    return np.expand_dims(arr, axis=0)


def audio_to_mel(wav_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(wav_bytes)
        path = temp.name

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


# ---------------- MODEL PREDICTION ----------------

def predict_audio(wav_bytes):

    if audio_model is None:
        raise Exception("Audio model not loaded")

    x = audio_to_mel(wav_bytes)

    preds = audio_model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))

    return AUDIO_CLASSES[idx], float(preds[idx]), preds.tolist()


def predict_visual(image_bytes):

    # Placeholder if no model yet
    if visual_model is None:

        probs = [0.25, 0.25, 0.25, 0.25]

        return "none", 0.0, probs

    x = preprocess_image(image_bytes)

    preds = visual_model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))

    return CLASS_NAMES[idx], float(preds[idx]), preds.tolist()


# ---------------- FUSION LOGIC ----------------

def audio_to_triage_dist(label):

    if label == "distress":
        return np.array([0.05, 0.25, 0.65, 0.05])

    elif label == "normal":
        return np.array([0.85, 0.10, 0.04, 0.01])

    else:  # silent
        return np.array([0.20, 0.20, 0.20, 0.40])


def text_to_triage_dist(label):

    if label == "severe":
        return np.array([0.05, 0.20, 0.65, 0.10])

    elif label == "moderate":
        return np.array([0.15, 0.45, 0.35, 0.05])

    elif label == "mild":
        return np.array([0.70, 0.20, 0.08, 0.02])

    else:
        return np.array([0.25, 0.25, 0.25, 0.25])


def fuse(audio_res, visual_res, text_res):

    audio_label, _, _ = audio_res
    audio_dist = audio_to_triage_dist(audio_label)

    visual_probs = np.array(visual_res[2])

    text_label, text_conf = text_res
    text_dist = text_to_triage_dist(text_label)

    # Weighted fusion
    fused = (
        0.4 * audio_dist +
        0.4 * visual_probs +
        0.2 * text_dist
    )

    fused = fused / fused.sum()

    idx = int(np.argmax(fused))

    return CLASS_NAMES[idx], fused.tolist(), float(fused[idx])


# ---------------- API ENDPOINTS ----------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "audio_model": audio_model is not None,
        "visual_model": visual_model is not None
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: str = Form(None)
):

    if image is None and audio is None and not text:
        raise HTTPException(
            status_code=400,
            detail="Send at least image, audio, or text"
        )


    # -------- AUDIO --------
    if audio:

        aud_bytes = await audio.read()

        audio_out = predict_audio(aud_bytes)

    else:
        audio_out = ("normal", 1.0, [1/3, 1/3, 1/3])


    # -------- IMAGE --------
    if image:

        img_bytes = await image.read()

        if len(img_bytes) > 0:
            visual_out = predict_visual(img_bytes)
        else:
            visual_out = ("none", 0.0, [0.25]*4)

    else:
        visual_out = ("none", 0.0, [0.25]*4)


    # -------- TEXT --------
    if text and len(text.strip()) > 0:

        text_label, text_conf = analyze_text(text)

        text_out = (text_label, text_conf)

    else:
        text_out = ("none", 0.0)


    # -------- FUSION --------
    triage, probs, conf = fuse(audio_out, visual_out, text_out)


    # -------- RESPONSE --------
    return {

        "triage_level": triage,

        "confidence": conf,

        "probabilities": {
            CLASS_NAMES[i]: probs[i] for i in range(4)
        },

        "audio_raw": {
            "label": audio_out[0],
            "confidence": audio_out[1],
            "probs": audio_out[2]
        },

        "visual_raw": {
            "label": visual_out[0],
            "confidence": visual_out[1],
            "probs": visual_out[2]
        },

        "text_raw": {
            "label": text_out[0],
            "confidence": text_out[1]
        }
    }


# ---------------- RUN ----------------

if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
