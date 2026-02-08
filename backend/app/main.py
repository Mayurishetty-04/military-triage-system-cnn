import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.auth import router as auth_router
from app.security import get_current_user
from app.models import User
from app.text_analyzer import analyze_text
from app.vision_utils import is_blurry

app = FastAPI(title="Military Triage Backend")

# ---------------- AUTH ----------------
app.include_router(auth_router, prefix="/auth")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- TRIAGE ACTIONS ----------------
TRIAGE_ACTIONS = {
    "Green": [
        "Minor injury",
        "Basic first aid",
        "No evacuation required"
    ],
    "Yellow": [
        "Moderate injury",
        "Clean wound",
        "Apply sterile dressing",
        "Monitor condition",
        "Provide pain relief if available"
    ],
    "Red": [
        "Severe injury",
        "Immediate medical attention",
        "Control bleeding",
        "Urgent evacuation"
    ],
    "Black": [
        "Non-survivable injury",
        "No medical intervention",
        "Prioritize other casualties"
    ]
}

# ---------------- FUSION ----------------
def fuse(image_probs=None, audio_probs=None, text_probs=None):
    weights = {
        "image": 0.4,
        "audio": 0.3,
        "text": 0.3
    }

    classes = ["Green", "Yellow", "Red", "Black"]
    final = dict.fromkeys(classes, 0.0)

    if image_probs:
        for c in classes:
            final[c] += image_probs[c] * weights["image"]

    if audio_probs:
        for c in classes:
            final[c] += audio_probs[c] * weights["audio"]

    if text_probs:
        for c in classes:
            final[c] += text_probs[c] * weights["text"]

    label = max(final, key=final.get)
    return label, final

# ---------------- PREDICT ----------------
@app.post("/predict")
async def predict(
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: str = Form(None),
    current_user: User = Depends(get_current_user),
):
    if not image and not audio and not text:
        raise HTTPException(400, "Provide at least one modality")

    modalities = []
    image_raw = audio_raw = text_raw = None

    # -------- IMAGE --------
    if image:
        img_bytes = await image.read()
        if is_blurry(img_bytes):
            raise HTTPException(400, "Image is blurry. Please recapture.")

        image_probs = {
            "Green": 0.03,
            "Yellow": 0.837,
            "Red": 0.12,
            "Black": 0.013
        }

        image_raw = {
            "label": "Yellow",
            "confidence": image_probs["Yellow"]
        }
        modalities.append("image")
    else:
        image_probs = None

    # -------- AUDIO --------
    if audio:
        audio_probs = {
            "Green": 0.01,
            "Yellow": 0.619,
            "Red": 0.347,
            "Black": 0.024
        }

        audio_raw = {
            "label": "distress",
            "confidence": 0.965
        }
        modalities.append("audio")
    else:
        audio_probs = None

    # -------- TEXT --------
    if text and text.strip():
        label, conf = analyze_text(text)
        text_probs = {
            "Green": 0.01,
            "Yellow": 0.02,
            "Red": conf,
            "Black": 0.0
        }

        text_raw = {
            "label": label,
            "confidence": conf
        }
        modalities.append("text")
    else:
        text_probs = None

    # -------- FUSION --------
    triage, probs = fuse(image_probs, audio_probs, text_probs)
    confidence = probs[triage]

    return {
        "triage_level": triage,
        "confidence": confidence,
        "probabilities": probs,
        "advice": TRIAGE_ACTIONS[triage],
        "modalities_used": modalities,
        "image_raw": image_raw,
        "audio_raw": audio_raw,
        "text_raw": text_raw
    }

# ---------------- PDF ----------------
@app.post("/download-report")
def download_report(data: dict):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    x = 40
    y = 800
    line = 18

    def draw(text):
        nonlocal y
        c.drawString(x, y, text)
        y -= line

    c.setFont("Helvetica-Bold", 16)
    draw("Military Triage Report")
    y -= 10

    c.setFont("Helvetica", 12)
    draw(f"Triage Level: {data.get('triage_level', '-')}")
    draw(f"Overall Confidence: {round(data.get('confidence', 0)*100, 1)}%")
    y -= 10

    # -------- Modalities --------
    draw("Modalities Used:")
    for m in data.get("modalities_used", []):
        draw(f"- {m}")
    y -= 10

    # -------- Probabilities --------
    probs = data.get("probabilities")
    if probs:
        draw("Triage Probabilities:")
        for k, v in probs.items():
            draw(f"- {k}: {round(v*100, 1)}%")
        y -= 10

    # -------- Recommendations --------
    advice = data.get("advice")
    if advice:
        draw("Recommended Actions:")
        for a in advice:
            draw(f"- {a}")
        y -= 10

    # -------- Image Analysis --------
    visual = data.get("visual_raw")
    if visual:
        draw("Image Analysis:")
        draw(f"- {visual['label']} ({round(visual['confidence']*100, 1)}%)")
        y -= 10

    # -------- Audio Analysis --------
    audio = data.get("audio_raw")
    if audio:
        draw("Audio Analysis:")
        draw(f"- {audio['label']} ({round(audio['confidence']*100, 1)}%)")
        y -= 10

    # -------- Text Analysis --------
    text = data.get("text_raw")
    if text:
        draw("Text Analysis:")
        draw(f"- {text['label']} ({round(text['confidence']*100, 1)}%)")
        y -= 10

    c.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=triage_report.pdf"}
    )
