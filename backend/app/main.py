import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import librosa, librosa.display
import matplotlib.pyplot as plt
import tempfile
import uvicorn

# ---------------- PATH SETTINGS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_cnn_ravdess.keras")
VISUAL_MODEL_PATH = os.path.join(MODEL_DIR, "visual_cnn.h5")  # not trained yet

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

def load_audio():
    global audio_model
    if os.path.exists(AUDIO_MODEL_PATH):
        audio_model = load_model(AUDIO_MODEL_PATH)
        print("Audio model loaded.")
    else:
        audio_model = None
        print("Audio model NOT found.")

def load_visual():
    global visual_model
    if os.path.exists(VISUAL_MODEL_PATH):
        visual_model = load_model(VISUAL_MODEL_PATH)
        print("Visual model loaded.")
    else:
        visual_model = None
        print("Visual model NOT found (placeholder mode).")

# load at startup
load_audio()
load_visual()

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

    img = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0

    return np.expand_dims(arr, 0)

# ---------------- MODEL PREDICTION ----------------
def predict_audio(wav_bytes):
    if audio_model is None:
        raise Exception("Audio model not loaded")

    x = audio_to_mel(wav_bytes)
    preds = audio_model.predict(x)[0]
    idx = int(np.argmax(preds))
    return AUDIO_CLASSES[idx], float(preds[idx]), preds.tolist()

def predict_visual(image_bytes):
    if visual_model is None:
        probs = [0.25, 0.25, 0.25, 0.25]
        return "none", 0.0, probs

    x = preprocess_image(image_bytes)
    preds = visual_model.predict(x)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds.tolist()

# ---------------- FUSION LOGIC ----------------
def audio_to_triage_dist(label):
    if label == "distress":
        return np.array([0.05, 0.25, 0.65, 0.05])
    elif label == "normal":
        return np.array([0.85, 0.1, 0.04, 0.01])
    else:
        return np.array([0.2, 0.2, 0.2, 0.4])

def fuse(audio_res, visual_res):
    audio_label, _, _ = audio_res
    audio_dist = audio_to_triage_dist(audio_label)
    visual_probs = np.array(visual_res[2])

    if visual_model is None:
        fused = audio_dist
    else:
        fused = 0.7 * visual_probs + 0.3 * audio_dist

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
async def predict(image: UploadFile = File(None), audio: UploadFile = File(None)):

    if image is None and audio is None:
        raise HTTPException(status_code=400, detail="Send at least image or audio")

    # audio
    if audio is not None:
        aud_bytes = await audio.read()
        audio_out = predict_audio(aud_bytes)
    else:
        audio_out = ("normal", 1.0, [1/3, 1/3, 1/3])

    # visual (SAFE HANDLING)
    if image is not None:
        img_bytes = await image.read()
        if len(img_bytes) > 0:
            visual_out = predict_visual(img_bytes)
        else:
            visual_out = ("none", 0.0, [0.25, 0.25, 0.25, 0.25])
    else:
        visual_out = ("none", 0.0, [0.25, 0.25, 0.25, 0.25])

    triage, probs, conf = fuse(audio_out, visual_out)

    return {
        "triage_level": triage,
        "confidence": conf,
        "probabilities": {CLASS_NAMES[i]: probs[i] for i in range(4)},
        "audio_raw": {
            "label": audio_out[0],
            "confidence": audio_out[1],
            "probs": audio_out[2]
        },
        "visual_raw": {
            "label": visual_out[0],
            "confidence": visual_out[1],
            "probs": visual_out[2]
        }
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
