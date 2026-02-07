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
import uuid
from datetime import datetime

import uvicorn

from text_analyzer import analyze_text


# ================= PATH SETTINGS =================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_cnn_ravdess.keras")
VISUAL_MODEL_PATH = os.path.join(MODEL_DIR, "visual_cnn.h5")

IMG_SIZE = (128, 128)

CLASS_NAMES = ["Green", "Yellow", "Red", "Black"]
AUDIO_CLASSES = ["normal", "distress", "silent"]

ALLOW_AUDIO = True


# ================= FASTAPI =================

app = FastAPI(title="Military Triage System - Backend")

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
        print("⚠️ Visual model not found (placeholder)")


load_audio_model()
load_visual_model()


# ================= PREPROCESS =================

def preprocess_image(image_bytes):

    if len(image_bytes) < 1000:
        raise ValueError("Invalid image file")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image uploaded"
        )

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


# ================= PREDICTION =================

def predict_audio(wav_bytes):

    if audio_model is None:
        return "normal", 0.5, [0.33, 0.33, 0.33]

    x = audio_to_mel(wav_bytes)

    preds = audio_model.predict(x, verbose=0)[0]

    # Low confidence safeguard
    if np.max(preds) < 0.6:
        return "normal", 0.5, preds.tolist()

    idx = int(np.argmax(preds))

    return AUDIO_CLASSES[idx], float(preds[idx]), preds.tolist()


def predict_visual(image_bytes):

    if visual_model is None:
        return "none", 0.0, [0.25]*4

    x = preprocess_image(image_bytes)

    preds = visual_model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))

    return CLASS_NAMES[idx], float(preds[idx]), preds.tolist()


# ================= TRIAGE DISTRIBUTIONS =================

def audio_to_triage_dist(label):

    if label == "distress":
        return np.array([0.05, 0.25, 0.65, 0.05])

    if label == "silent":
        return np.array([0.15, 0.20, 0.25, 0.40])

    return np.array([0.80, 0.12, 0.06, 0.02])



def text_dist(label):

    if label == "red":
        return np.array([0.05, 0.15, 0.70, 0.10])

    if label == "yellow":
        return np.array([0.10, 0.60, 0.25, 0.05])

    if label == "green":
        return np.array([0.70, 0.20, 0.08, 0.02])

    return np.array([0.25, 0.25, 0.25, 0.25])


# ================= FUSION =================

def fuse(audio_res, visual_res, text_res):

    audio_label, audio_conf, _ = audio_res
    visual_label, visual_conf, visual_probs = visual_res
    text_label, text_conf = text_res


    visual_probs = np.array(visual_probs)


    # ---------------- BASE WEIGHTS ----------------
    visual_weight = 0.65   # MAIN SOURCE
    text_weight   = 0.20
    audio_weight  = 0.15


    # ---------------- AUDIO ----------------
    audio_dist = audio_to_triage_dist(audio_label)


    # ---------------- TEXT ----------------
    if text_conf < 0.6:
        # Ignore weak text
        text_dist = np.array([0.25, 0.25, 0.25, 0.25])

    else:
        if text_label == "red":
            text_dist = np.array([0.10, 0.20, 0.55, 0.15])

        elif text_label == "yellow":
            text_dist = np.array([0.15, 0.55, 0.25, 0.05])

        elif text_label == "green":
            text_dist = np.array([0.65, 0.25, 0.07, 0.03])

        else:
            text_dist = np.array([0.25, 0.25, 0.25, 0.25])


    # ---------------- DYNAMIC WEIGHTS ----------------

    # If image is confident → trust it more
    if visual_conf > 0.85:
        visual_weight = 0.75
        text_weight   = 0.15
        audio_weight  = 0.10

    # If image is weak → use text more
    elif visual_conf < 0.5:
        visual_weight = 0.45
        text_weight   = 0.35
        audio_weight  = 0.20


    # ---------------- FUSION ----------------
    fused = (
        visual_weight * visual_probs +
        text_weight   * text_dist +
        audio_weight  * audio_dist
    )


    # ---------------- MEDICAL RULES ----------------

    # Rule 1: Cut/Wound should never be Black alone
    if visual_label in ["Green", "Yellow"] and text_label != "black":
        fused[3] *= 0.2


    # Rule 2: Red requires agreement
    red_votes = (
        (visual_label == "Red") +
        (text_label == "red") +
        (audio_label == "distress")
    )

    if red_votes < 2:
        fused[2] *= 0.6


    # Rule 3: Black needs strong evidence
    black_votes = (
        (visual_label == "Black") +
        (audio_label == "silent") +
        (text_label == "black")
    )

    if black_votes < 2:
        fused[3] *= 0.3


    # Rule: Severe bleeding forces Red upward
    if "bleed" in text_label.lower() or text_label == "red":
        fused[2] *= 1.3   # boost Red

    # Normalize
    fused = fused / fused.sum()

    idx = int(np.argmax(fused))

    return CLASS_NAMES[idx], fused.tolist(), float(fused[idx])

# ================= ADVICE =================

def generate_advice(triage, text):

    advice = []

    if triage == "Black":
        advice += [
            "🚨 CRITICAL CONDITION",
            "Immediate evacuation required",
            "Call emergency medical team",
            "Do NOT move patient unnecessarily"
        ]

    elif triage == "Red":
        advice += [
            "⚠️ Severe injury detected",
            "Apply pressure to bleeding",
            "Seek medic immediately",
            "Immobilize affected area"
        ]

    elif triage == "Yellow":
        advice += [
            "🟡 Moderate injury",
            "Clean wound",
            "Apply sterile dressing",
            "Monitor condition"
        ]

    elif triage == "Green":
        advice += [
            "🟢 Minor injury",
            "Clean with antiseptic",
            "Apply ointment",
            "Rest and hydrate"
        ]

    t = text.lower()

    if "bleed" in t:
        advice.append("Apply direct pressure")

    if "burn" in t:
        advice.append("Cool burn with clean water")

    if "pain" in t:
        advice.append("Provide pain relief if available")

    return advice


# ================= API =================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "audio_model": audio_model is not None,
        "visual_model": visual_model is not None,
        "audio_enabled": ALLOW_AUDIO
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: str = Form(None)
):

    if image is None and audio is None and not text:
        raise HTTPException(400, "Send image, audio, or text")


    # -------- AUDIO --------
    if audio and ALLOW_AUDIO:
        aud_bytes = await audio.read()
        audio_out = predict_audio(aud_bytes)
    else:
        audio_out = ("normal", 1.0, [0.33]*3)


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
    if text and text.strip():

        text_label, text_conf = analyze_text(text)
        text_out = (text_label, text_conf)

    else:
        text_out = ("none", 0.0)


    # -------- FUSION --------
    triage, probs, conf = fuse(
        audio_out,
        visual_out,
        text_out
    )
    patient_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # -------- ADVICE --------
    advice = generate_advice(triage, text or "")

 

    # -------- RESPONSE --------
    return {
        "patient_id": patient_id,
        "timestamp": timestamp,
        "triage_level": triage,

        "confidence": conf,

        "probabilities": {
            CLASS_NAMES[i]: probs[i] for i in range(4)
        },

        "advice": advice,


        "audio_raw": {
            "label": audio_out[0],
            "confidence": audio_out[1]
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


# ================= RUN =================

if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
