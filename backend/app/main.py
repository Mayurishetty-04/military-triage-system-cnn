import io
import random
import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from pydantic import BaseModel

from app.database import engine, Base
from app.auth import router as auth_router
from app.security import get_current_user
from app.models import User
import app.models as models
from app.text_analyzer import analyze_text
from app.vision_model import predict_visual
from app.audio_model import predict_audio
from app.vitals import vitals_override, calculate_dynamic_survival

# Create tables on startup
Base.metadata.create_all(bind=engine)

# ============================================================
# REALTIME VITALS MODEL
# ============================================================

class VitalData(BaseModel):
    heart_rate: int
    spo2: int
    systolic_bp: int
    diastolic_bp: int


class AIScores(BaseModel):
    image: float
    audio: float
    video: float

class PatientVitals(BaseModel):
    spo2: int
    heartRate: int

class PatientCreate(BaseModel):
    patientId: str
    status: str
    survivalProbability: float
    injuryType: str
    vitals: PatientVitals
    aiScores: AIScores
    recommendation: str

class PatientUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[int] = None
    recommendation: Optional[str] = None


# ============================================================
# APP INIT
# ============================================================

app = FastAPI(title="Military Triage Backend")

app.include_router(auth_router, prefix="/auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- GLOBAL STATE FOR LIVE VITALS ----
# Stores the last received vitals from the Android Bridge
latest_vitals = {
    "heart_rate": None,
    "spo2": None,
    "systolic_bp": None,
    "diastolic_bp": None,
    "triage": "N/A",
    "timestamp": None
}

@app.post("/realtime-vitals")
def receive_vitals(data: VitalData):
    """
    Receives real-time vitals from Android Bridge App.
    Performs fast triage classification and stores it for the frontend.
    """
    global latest_vitals
    triage = "GREEN"
    reason = "Vitals within normal range"

    # BLACK: Not survivable / completely unresponsive indicators
    if data.spo2 < 70 or data.heart_rate > 160 or data.systolic_bp < 60:
        triage = "BLACK"
        reason = f"Critical: SpO2={data.spo2}%, HR={data.heart_rate}, SysBP={data.systolic_bp}"

    # RED: Immediate life threat
    elif data.spo2 < 90 or data.heart_rate > 130 or data.heart_rate < 40 or data.systolic_bp < 80 or data.systolic_bp > 180:
        triage = "RED"
        reason = f"Severe: SpO2={data.spo2}%, HR={data.heart_rate}, SysBP={data.systolic_bp}"

    # YELLOW: Delayed/moderate concern
    elif data.spo2 < 94 or data.heart_rate > 100 or data.heart_rate < 60 or data.systolic_bp < 90 or data.systolic_bp > 140:
        triage = "YELLOW"
        reason = f"Moderate: SpO2={data.spo2}%, HR={data.heart_rate}, SysBP={data.systolic_bp}"

    # Store for frontend
    latest_vitals = {
        "heart_rate": data.heart_rate,
        "spo2": data.spo2,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "triage": triage,
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat()
    }

    return latest_vitals

@app.get("/latest-vitals")
def get_latest_vitals():
    """
    Endpoint for frontend to poll the latest vitals from Android bridge.
    """
    return latest_vitals


# ============================================================
# PATIENT MANAGEMENT ROUTES
# ============================================================

def get_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/patients")
def create_patient(patient: PatientCreate, db = Depends(get_db)):
    # Calculate priority based on status
    priority_map = {"RED": 1, "YELLOW": 2, "GREEN": 3, "BLACK": 4}
    priority = priority_map.get(patient.status.upper(), 4)

    db_patient = models.Patient(
        patientId=patient.patientId,
        status=patient.status.upper(),
        survivalProbability=patient.survivalProbability,
        injuryType=patient.injuryType,
        spo2=patient.vitals.spo2,
        heartRate=patient.vitals.heartRate,
        imageScore=patient.aiScores.image,
        audioScore=patient.aiScores.audio,
        videoScore=patient.aiScores.video,
        recommendation=patient.recommendation,
        priority=priority
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/patients")
def get_patients(db = Depends(get_db)):
    # Get only the latest record for each unique patientId by comparing timestamps
    subquery = db.query(
        models.Patient.patientId,
        func.max(models.Patient.timestamp).label('max_time')
    ).group_by(models.Patient.patientId).subquery()
    
    patients = db.query(models.Patient).join(
        subquery,
        (models.Patient.patientId == subquery.c.patientId) &
        (models.Patient.timestamp == subquery.c.max_time)
    ).order_by(models.Patient.priority.asc()).all()
    
    return patients

@app.get("/patients/{patient_id}/history")
def get_patient_history(patient_id: str, db = Depends(get_db)):
    # Return all records for a specific patient, sorted by time
    return db.query(models.Patient).filter(models.Patient.patientId == patient_id).order_by(models.Patient.timestamp.desc()).all()

@app.put("/patients/{patient_id}")
def update_patient(patient_id: str, patient_update: PatientUpdate, db = Depends(get_db)):
    db_patient = db.query(models.Patient).filter(models.Patient.patientId == patient_id).first()
    if not db_patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if patient_update.status:
        db_patient.status = patient_update.status.upper()
        # Update priority if status changed
        priority_map = {"RED": 1, "YELLOW": 2, "GREEN": 3, "BLACK": 4}
        db_patient.priority = priority_map.get(db_patient.status, 4)
    
    if patient_update.priority is not None:
        db_patient.priority = patient_update.priority
    
    if patient_update.recommendation:
        db_patient.recommendation = patient_update.recommendation
        
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.post("/patients/{id}/acknowledge")
def acknowledge_patient(id: int, db = Depends(get_db)):
    db_patient = db.query(models.Patient).filter(models.Patient.id == id).first()
    if not db_patient:
        raise HTTPException(status_code=404, detail="Patient record not found")
    
    db_patient.is_acknowledged = 1
    db.commit()
    db.refresh(db_patient)
    return {"message": "Patient alert acknowledged successfully", "patientId": db_patient.patientId}

# ============================================================
# CONSTANTS
# ============================================================

CLASSES = ["Green", "Yellow", "Red", "Black"]

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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def to_prob_dict(raw: Optional[dict]):
    """
    Convert model output into a natural probability distribution.
    If only a single label is provided, it simulates uncertainty 
    by distributing remaining probability to 'nearby' classes.
    """
    probs = {c: 0.0 for c in CLASSES}
    
    # Proximity Mapping: What other classes might be likely?
    PROXIMITY = {
        "Green":  {"Yellow": 0.6, "Red": 0.3, "Black": 0.1},
        "Yellow": {"Green": 0.4, "Red": 0.5, "Black": 0.1},
        "Red":    {"Yellow": 0.6, "Black": 0.3, "Green": 0.1},
        "Black":  {"Red": 0.7, "Yellow": 0.2, "Green": 0.1}
    }

    if not raw:
        return {c: 1.0/len(CLASSES) for c in CLASSES}

    if "probabilities" in raw:
        # If we already have a full distribution, just use it
        for k, v in raw["probabilities"].items():
            k_title = k.title()
            if k_title in probs:
                probs[k_title] = v
    elif "label" in raw and "confidence" in raw:
        label = str(raw["label"]).title()
        conf = raw["confidence"]
        
        if label in probs:
            # Primary label gets most of the confidence (but capped to leave room for others)
            primary_conf = min(conf, 0.92)
            probs[label] = primary_conf
            
            # Distribute the remainder (0.08 minimum) based on proximity
            remainder = 1.0 - primary_conf
            dist = PROXIMITY.get(label, {})
            for other, weight in dist.items():
                if other in probs:
                    probs[other] += remainder * weight

    # Final normalization to ensure sum is exactly 1.0
    total = sum(probs.values())
    if total > 0:
        for k in probs:
            probs[k] /= total

    return probs


def weighted_fusion(image_probs, audio_probs, text_probs, vitals_probs=None):
    """
    Adaptive weighted fusion.
    Works for 1, 2, 3, or 4 modalities.
    """

    weights = {
        "image": 0.3,
        "audio": 0.2,
        "text": 0.1,
        "vitals": 0.4
    }

    final = {c: 0.0 for c in CLASSES}
    total_weight = 0

    if image_probs:
        total_weight += weights["image"]
        for c in CLASSES:
            final[c] += image_probs.get(c, 0.0) * weights["image"]

    if audio_probs:
        total_weight += weights["audio"]
        for c in CLASSES:
            final[c] += audio_probs.get(c, 0.0) * weights["audio"]

    if text_probs:
        total_weight += weights["text"]
        for c in CLASSES:
            final[c] += text_probs.get(c, 0.0) * weights["text"]

    if vitals_probs:
        total_weight += weights["vitals"]
        for c in CLASSES:
            final[c] += vitals_probs.get(c, 0.0) * weights["vitals"]

    if total_weight == 0:
        return final

    # Normalize if fewer modalities used
    for c in CLASSES:
        final[c] = final[c] / total_weight

    return final


# ============================================================
# PREDICT ROUTE
# ============================================================

@app.post("/predict")
async def predict(
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: str = Form(None),
    pulse: int | None = Form(None),
    spo2: int | None = Form(None),
    systolic_bp: int | None = Form(None),
    unconscious: bool = Form(False),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    patient_id: str | None = Form(None),
    current_user: User = Depends(get_current_user),
):

    if not image and not audio and not text:
        raise HTTPException(400, "Provide at least one modality")

    modalities_used = []

    # ---------------- IMAGE ----------------
    visual_raw = None
    if image:
        img_bytes = await image.read()
        visual_raw = predict_visual(img_bytes)
        modalities_used.append("image")

    # ---------------- AUDIO ----------------
    audio_raw = None
    if audio:
        audio_bytes = await audio.read()
        audio_raw = predict_audio(audio_bytes)
        modalities_used.append("audio")

    # ---------------- TEXT ----------------
    text_raw = None
    if text:
        label, conf = analyze_text(text)
        text_raw = {"label": label, "confidence": conf}
        modalities_used.append("text")

    # ---------------- CONVERT TO PROBS ----------------
    image_probs = to_prob_dict(visual_raw)
    audio_probs = to_prob_dict(audio_raw)
    text_probs = to_prob_dict(text_raw)

    from app.vitals import get_vitals_probs
    vitals_probs = get_vitals_probs(pulse, spo2, systolic_bp, unconscious)

    # ---------------- MODEL FUSION ----------------
    model_probabilities = weighted_fusion(
        image_probs if visual_raw else None,
        audio_probs if audio_raw else None,
        text_probs if text_raw else None,
        vitals_probs
    )

    if unconscious:
        final_label = "Black"
        final_conf = 1.0
        override_reason = "Critical condition: Patient is unconscious"
        model_probabilities = {"Green": 0.0, "Yellow": 0.0, "Red": 0.0, "Black": 1.0}
    else:
        final_label = max(model_probabilities, key=model_probabilities.get)
        final_conf = model_probabilities[final_label]
        override_reason = None

    # ---------------- SAVE PATIENT (Auto) ----------------
    from app.database import SessionLocal
    db = SessionLocal()
    
    try:
        # 1. Determine stable patientId
        if not patient_id:
            if current_user and current_user.role == "patient" and current_user.patientId:
                patient_id = current_user.patientId
            else:
                # Anonymous or legacy fallback identity
                patient_id = f"PAT-{int(datetime.datetime.now().timestamp())}-{random.randint(100, 999)}"
    
        # Map AI scores (using probabilities for now or max confidence)
        ai_scores = {
            "image": image_probs.get(final_label, 0) if visual_raw else 0.0,
            "audio": audio_probs.get(final_label, 0) if audio_raw else 0.0,
            "text": text_probs.get(final_label, 0) if text_raw else 0.0,
            "video": 0.0
        }
        # Calculate Survival Probability based on sophisticated clinical logic
        vitals_payload = {
            "pulse": pulse,
            "spo2": spo2,
            "bp_systolic": systolic_bp,
            "resp_rate": 16 # Default for now
        }
        dynamic_survival = calculate_dynamic_survival(vitals_payload, final_label)

        priority_map = {"RED": 1, "YELLOW": 2, "GREEN": 3, "BLACK": 4}
        db_patient = models.Patient(
            patientId=patient_id,
            patientName=current_user.username if current_user else None,
            status=final_label.upper(),
            survivalProbability=dynamic_survival,
            injuryType=text if text else "Unknown",
            spo2=spo2 if spo2 else 0,
            heartRate=pulse if pulse else 0,
            imageScore=ai_scores["image"],
            audioScore=ai_scores["audio"],
            textScore=ai_scores["text"],
            videoScore=ai_scores["video"],
            recommendation=", ".join(TRIAGE_ACTIONS[final_label]),
            priority=priority_map.get(final_label.upper(), 4),
            latitude=latitude,
            longitude=longitude,
            user_id=current_user.id if current_user and current_user.role == "patient" else None
        )
        db.add(db_patient)
        db.commit()
    except Exception as e:
        print(f"Error auto-saving patient: {e}")
    finally:
        db.close()

    return {
        "patientId": patient_id,
        "patientName": current_user.username if current_user else None,
        "triage_level": final_label,
        "confidence": round(final_conf * 100, 1),
        "override_reason": override_reason,
        "probabilities": model_probabilities,
        "recommended_action": TRIAGE_ACTIONS.get(final_label, ["Monitor condition"]),
        "modalities_used": modalities_used,
        "vitals": {
            "pulse": pulse,
            "spo2": spo2,
            "systolic_bp": systolic_bp,
            "unconscious": unconscious
        }
    }


# ============================================================
# PDF DOWNLOAD
# ============================================================

@app.post("/download-report")
def download_report(data: dict):
    from reportlab.lib import colors
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=6,
        alignment=1,  # center
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#666666'),
        spaceAfter=20,
        alignment=1,
        fontName='Helvetica'
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#ffffff'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    # Header Section
    story.append(Paragraph("🪖 MILITARY TRIAGE REPORT", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 0.3*inch))

    # Triage Level Banner
    triage_level = data.get('triage_level', 'Unknown')
    confidence = data.get('confidence', 0)
    
    triage_colors = {
        'Green': colors.HexColor('#10b981'),
        'Yellow': colors.HexColor('#f59e0b'),
        'Red': colors.HexColor('#ef4444'),
        'Black': colors.HexColor('#6b7280')
    }
    
    triage_color = triage_colors.get(triage_level, colors.HexColor('#6b7280'))
    
    # Triage banner table
    triage_data = [
        [Paragraph(f"<b>TRIAGE LEVEL</b>", ParagraphStyle('banner', parent=styles['Normal'], fontSize=12, textColor=colors.whitesmoke, fontName='Helvetica-Bold')),
         Paragraph(f"<b>CONFIDENCE</b>", ParagraphStyle('banner', parent=styles['Normal'], fontSize=12, textColor=colors.whitesmoke, fontName='Helvetica-Bold'))],
        [Paragraph(f"<b>{triage_level}</b>", ParagraphStyle('banner', parent=styles['Normal'], fontSize=24, textColor=colors.whitesmoke, fontName='Helvetica-Bold')),
         Paragraph(f"<b>{confidence}%</b>", ParagraphStyle('banner', parent=styles['Normal'], fontSize=24, textColor=colors.whitesmoke, fontName='Helvetica-Bold'))]
    ]
    
    triage_table = Table(triage_data, colWidths=[3*inch, 2.5*inch])
    triage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), triage_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, 1), 20),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [triage_color, triage_color]),
    ]))
    
    story.append(triage_table)
    story.append(Spacer(1, 0.3*inch))

    # Override Reason
    if data.get('override_reason'):
        story.append(Paragraph("⚠️ OVERRIDE REASON", heading_style))
        override_style = ParagraphStyle(
            'override',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#dc2626'),
            spaceAfter=12,
            fontName='Helvetica'
        )
        story.append(Paragraph(data.get('override_reason'), override_style))
        story.append(Spacer(1, 0.2*inch))

    # Modalities Used
    story.append(Paragraph("📊 MODALITIES USED", heading_style))
    modalities_style = ParagraphStyle(
        'modalities',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=8,
        fontName='Helvetica'
    )
    modalities = data.get("modalities_used", [])
    for m in modalities:
        story.append(Paragraph(f"• {m.capitalize()}", modalities_style))
    story.append(Spacer(1, 0.2*inch))

    # Model Probabilities
    story.append(Paragraph("📈 MODEL CONFIDENCE BY TRIAGE LEVEL", heading_style))
    probs = data.get("probabilities", {})
    
    prob_data = [["Triage Level", "Confidence Score", "Percentage"]]
    for k, v in probs.items():
        prob_data.append([k, f"{round(v * 100, 1)}%", "█" * int(v * 20)])
    
    prob_table = Table(prob_data, colWidths=[1.8*inch, 1.5*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#374151')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f3f4f6'), colors.HexColor('#e5e7eb')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(prob_table)
    story.append(Spacer(1, 0.3*inch))

    # Recommended Actions
    advice = data.get("recommended_action", [])
    if advice:
        story.append(Paragraph("💡 RECOMMENDED ACTIONS", heading_style))
        action_style = ParagraphStyle(
            'actions',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=20,
            spaceAfter=10,
            fontName='Helvetica'
        )
        for i, a in enumerate(advice, 1):
            story.append(Paragraph(f"<b>{i}.</b> {a}", action_style))
        story.append(Spacer(1, 0.2*inch))

    # Footer
    story.append(Spacer(1, 0.3*inch))
    footer_style = ParagraphStyle(
        'footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#9ca3af'),
        alignment=1,
        fontName='Helvetica-Oblique'
    )
    story.append(Paragraph("Military Emergency Triage System © 2026 | Authorized Personnel Only", footer_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=triage_report.pdf"}
    )
