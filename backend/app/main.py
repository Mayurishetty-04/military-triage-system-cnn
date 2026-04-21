import io
import random
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import shutil
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from app.database import engine, Base
from app.auth import router as auth_router
from app.security import get_current_user
from app.models import User
import app.models as models
from app.text_analyzer import analyze_text
from app.vision_model import predict_visual
from app.audio_model import predict_audio
from app.vitals import (
    vitals_override, 
    calculate_dynamic_survival,
    get_hr_status,
    get_spo2_status,
    get_bp_status,
    get_physiological_syndrome
)

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

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


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
    reason = "Stable: Patient compensating well, normal vitals."

    # BLACK: Not survivable / completely unresponsive indicators
    if data.spo2 < 70 or data.heart_rate > 160 or data.systolic_bp < 60:
        triage = "BLACK"
        reason = f"Expectant: Catastrophic/terminal collapse (SpO2={data.spo2}%, HR={data.heart_rate}, SysBP={data.systolic_bp})"

    # RED: Immediate life threat
    elif data.spo2 < 90 or data.heart_rate > 130 or data.heart_rate < 40 or data.systolic_bp < 80 or data.systolic_bp > 180:
        triage = "RED"
        reason = f"Immediate: Severe oxygen starvation/collapse (SpO2={data.spo2}%, HR={data.heart_rate}, SysBP={data.systolic_bp})"

    # YELLOW: Delayed/moderate concern
    elif data.spo2 < 94 or data.heart_rate > 100 or data.heart_rate < 60 or data.systolic_bp < 90 or data.systolic_bp > 140:
        triage = "YELLOW"
        reason = f"Delayed: Early signs of stress/shock (SpO2={data.spo2}%, HR={data.heart_rate}, SysBP={data.systolic_bp})"

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
    ).order_by(models.Patient.priority.asc(), models.Patient.survivalProbability.asc()).all()
    
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

def generate_explanation(label, pulse, spo2, systolic_bp, unconscious, override_reason, modalities_used):
    if override_reason:
        return f"Triage level '{label}' prioritized due to critical safety override: {override_reason}. Immediate life-saving protocols are in effect."
        
    hr_status = get_hr_status(pulse)
    spo2_status = get_spo2_status(spo2)
    bp_status = get_bp_status(systolic_bp)
    syndrome = get_physiological_syndrome(pulse, spo2, systolic_bp)
    
    parts = []
    
    # Primary clinical assessment
    if syndrome:
        parts.append(f"AI Clinical Impression: {syndrome}.")
    
    # Vital-specific details
    vital_details = []
    if hr_status != "Normal" and hr_status != "Unknown": vital_details.append(f"{hr_status} ({pulse} bpm)")
    if spo2_status != "Normal" and spo2_status != "Unknown": vital_details.append(f"{spo2_status} ({spo2}%)")
    if bp_status != "Normal" and bp_status != "Unknown": vital_details.append(f"{bp_status} ({systolic_bp} mmHg sys)")
    
    if vital_details:
        parts.append(f"Vital signs show: {', '.join(vital_details)}.")
    else:
        parts.append("Physiological parameters currently within baseline ranges.")

    # Status-specific reasoning
    if label == "Black":
        parts.append("Condition is expectant; catastrophic system failure or non-survivable trauma indicated. Resources allocated to salvageable casualties.")
    elif label == "Red":
        parts.append("Immediate life-threat detected. Decompensated state requires instant medical intervention to prevent terminal collapse.")
    elif label == "Yellow":
        parts.append("Moderate physiological stress detected. Injuries require urgent but not immediate care. Patient requires close monitoring for decompensation.")
    else:
        parts.append("Patient is hemodynamically stable. Minor injuries confirmed by multi-modal analysis. Care can be delayed for more critical cases.")
        
    if modalities_used:
        parts.append(f"Assessment cross-verified using: {', '.join(modalities_used).upper()}.")
        
    return " ".join(parts)


def generate_class_explanations(pulse, spo2, systolic_bp, unconscious, override_reason, probabilities):
    explanations = {}
    
    hr_status = get_hr_status(pulse)
    spo2_status = get_spo2_status(spo2)
    bp_status = get_bp_status(systolic_bp)
    syndrome = get_physiological_syndrome(pulse, spo2, systolic_bp)
    
    if override_reason:
        for c in CLASSES:
            if c == "Black" and unconscious:
                explanations[c] = f"Override applied: {override_reason}"
            else:
                explanations[c] = "Clinical indicators indicate extreme instability requiring manual override."
        return explanations

    # --- RED REASONING ---
    if probabilities.get("Red", 0) > 0.15:
        if syndrome == "Decompensated Shock (Likely Hemorrhage)":
             explanations["Red"] = "High risk of hypovolemic shock due to critical hypotension and compensatory tachycardia."
        elif "Hypoxemia" in spo2_status:
             explanations["Red"] = f"Respiratory failure risk: {spo2_status} ({spo2}%) indicates urgent need for oxygenation."
        elif bp_status == "Critical Hypotension":
             explanations["Red"] = "Severe hypotension indicates lack of vital organ perfusion, requiring immediate resuscitation."
        else:
             explanations["Red"] = "Clinical markers and sensory AI suggest rapid deterioration potential requiring 'Immediate' category."
    else:
        explanations["Red"] = "Calculated risk of immediate life-threat is relatively low based on current data."

    # --- YELLOW REASONING ---
    if probabilities.get("Yellow", 0) > 0.15:
        if hr_status == "Mild Tachycardia" or bp_status == "Hypotension" or spo2_status == "Mild Hypoxemia":
            explanations["Yellow"] = f"Compensatory stress detected: {hr_status if hr_status != 'Normal' else ''} signs indicate moderate injury severity."
        else:
            explanations["Yellow"] = f"Moderate risk profile ({int(probabilities.get('Yellow', 0)*100)}%): Sensory inputs suggest injuries requiring urgent but non-immediate care."
    else:
         explanations["Yellow"] = "Symptoms do not align strongly with moderate, delayed-care profiles."
         
    # --- GREEN REASONING ---
    if probabilities.get("Green", 0) > 0.15:
        if hr_status == "Normal" and spo2_status == "Normal" and bp_status == "Normal":
            explanations["Green"] = "Patient exhibits physiological stability with normal hemodynamic parameters. Minor injury profile."
        else:
            explanations["Green"] = "Body is currently compensating well for injuries. Risk of near-term collapse remains low."
    else:
        explanations["Green"] = "Stability indicators are overshadowed by moderate or severe injury markers."
        
    # --- BLACK REASONING ---
    if probabilities.get("Black", 0) > 0.05:
        if unconscious or syndrome == "Severe Circulatory Collapse" or syndrome == "Critical Respiratory Failure":
             explanations["Black"] = f"Expectant state: Physiological markers ({syndrome if syndrome else 'Collapsing vitals'}) suggest catastrophic, non-survivable status."
        else:
             explanations["Black"] = "AI has flagged potential non-survivable injury patterns or extreme lack of responsiveness."
    else:
         explanations["Black"] = "No lethal or expectant indicators currently present."

    return explanations


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
    image_path = None
    if image:
        file_extension = os.path.splitext(image.filename)[1]
        if not file_extension:
            file_extension = ".jpg"
        filename = f"img_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}{file_extension}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        image_path = f"/uploads/{filename}"
        
        # Reset file pointer for predict_visual if needed (but we already read it)
        # Actually image.file was copied, so we need to read from filepath or buffer
        with open(filepath, "rb") as f:
            img_bytes = f.read()
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
            image_path=image_path,
            user_id=current_user.id if current_user and current_user.role == "patient" else None
        )
        db.add(db_patient)
        db.commit()
    except Exception as e:
        print(f"Error auto-saving patient: {e}")
    finally:
        db.close()

    explanation = generate_explanation(
        label=final_label,
        pulse=pulse,
        spo2=spo2,
        systolic_bp=systolic_bp,
        unconscious=unconscious,
        override_reason=override_reason,
        modalities_used=modalities_used
    )

    class_explanations = generate_class_explanations(
        pulse=pulse,
        spo2=spo2,
        systolic_bp=systolic_bp,
        unconscious=unconscious,
        override_reason=override_reason,
        probabilities=model_probabilities
    )

    return {
        "patientId": patient_id,
        "patientName": current_user.username if current_user else None,
        "triage_level": final_label,
        "explanation": explanation,
        "class_explanations": class_explanations,
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
        },
        "resource_advice": get_resource_optimization_advice(final_label, vitals_payload)
    }


def get_resource_optimization_advice(triage_level, vitals):
    """
    Generate dynamic resource optimization advice based on triage level and vitals.
    """
    level = triage_level.upper()
    hr = vitals.get('pulse')
    spo2 = vitals.get('spo2')
    bp = vitals.get('systolic_bp')
    
    advice = {
        "output": [],
        "why": "Decision-support system (dynamic analytics)"
    }
    
    if level == "RED":
        advice["output"].append("🚨 PRIORITY 1: Immediate life-saving evacuation required.")
        advice["output"].append("👨‍⚕️ Allocate trauma surgeon and senior nursing staff immediately.")
        if spo2 and spo2 < 90:
            advice["output"].append("💨 CRITICAL: Dispatch ALS unit with advanced airway management.")
        elif hr and (hr > 130 or hr < 40):
            advice["output"].append("💓 WARNING: High risk of cardiovascular collapse; prioritize telemetry transport.")
    
    elif level == "YELLOW":
        advice["output"].append("🟡 PRIORITY 2: Urgent evacuation within 2-4 hours.")
        advice["output"].append("🏥 Monitor vitals every 15-30 minutes for signs of decompensation.")
        if spo2 and spo2 < 94:
            advice["output"].append("🌬️ Administer supplemental oxygen; group with other respiratory-distressed casualties.")
        advice["output"].append("🚑 Multi-casualty ambulance transport suitable.")
        
    elif level == "GREEN":
        advice["output"].append("🟢 PRIORITY 3: Delayed evacuation (up to 24 hours).")
        advice["output"].append("🩹 Apply self-aid or buddy-aid protocols.")
        advice["output"].append("🚌 Utilize non-medical transport or multi-passenger vehicles for evacuation.")
        advice["why"] = "Patient physiologically stable; resources optimized for critical care."
        
    elif level == "BLACK":
        advice["output"].append("⚫ EXPECTANT: Provide palliative care and comfort measures.")
        advice["output"].append("🛡️ Redirect intensive resources (surgeons/vials) to salvageable casualties.")
        advice["output"].append("📝 Accurate biometric recording for subsequent identification.")
        advice["why"] = "Resource optimization strategy focused on maximal survival across the unit."
    
    else:
        advice["output"].append("Monitor clinical condition for changes.")
        advice["why"] = "Baseline assessment only."
        
    return advice


# ============================================================
# PDF DOWNLOAD
# ============================================================

@app.post("/download-report")
def download_report(data: dict):
    buffer = io.BytesIO()
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

    # Resource Optimization Engine
    vitals = data.get("vitals", {})
    opt_advice = get_resource_optimization_advice(triage_level, vitals)
    
    story.append(Paragraph("🚀 7. RESOURCE OPTIMIZATION ENGINE", heading_style))
    opt_style = ParagraphStyle(
        'opt_advice',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=10,
        fontName='Helvetica'
    )
    for a in opt_advice["output"]:
        story.append(Paragraph(f"• {a}", opt_style))
    
    story.append(Paragraph(f"<i><b>Why:</b> {opt_advice['why']}</i>", 
                           ParagraphStyle('opt_why', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#4b5563'), leftIndent=20, topMargin=5)))
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

# ============================================================
# CHAT & MESSAGING SYSTEM
# ============================================================

class MessageCreate(BaseModel):
    content: str
    recipient_id: int

class MessageResponse(BaseModel):
    id: int
    content: str
    sender_id: int
    sender_username: str
    timestamp: str
    is_read: int

class ConversationResponse(BaseModel):
    id: int
    patient_id: int
    patient_username: str
    doctor_id: Optional[int]
    doctor_username: Optional[str]
    last_message: Optional[str]
    updated_at: str
    unread_count: int

class DoctorUserResponse(BaseModel):
    id: int
    username: str

class PatientUserResponse(BaseModel):
    id: int
    username: str
    patientId: Optional[str]

def get_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/doctors", response_model=list[DoctorUserResponse])
def list_doctors(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    default_doctor_username = os.getenv("DEFAULT_DOCTOR_USERNAME", "doc1")
    doctors = db.query(User).filter(User.role == "doctor").order_by(User.id.asc()).all()
    doctors_sorted = sorted(
        doctors,
        key=lambda d: (0 if d.username == default_doctor_username else 1, d.id),
    )
    return [{"id": d.id, "username": d.username} for d in doctors_sorted]

@app.get("/users/patients", response_model=list[PatientUserResponse])
def list_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Only doctors should view full patient roster
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view patients")
    patients = db.query(User).filter(User.role == "patient").order_by(User.id.asc()).all()
    return [{"id": p.id, "username": p.username, "patientId": p.patientId} for p in patients]

@app.post("/messages/merge-to-default-doctor")
def merge_messages_to_default_doctor(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    For a patient, consolidate any prior conversations with other doctors
    into the conversation with the default doctor (by username).
    This prevents "missing history" when switching to the configured default.
    """
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can merge conversations")

    default_doctor_username = os.getenv("DEFAULT_DOCTOR_USERNAME", "doc1")
    default_doctor = db.query(User).filter(User.role == "doctor", User.username == default_doctor_username).first()
    if not default_doctor:
        raise HTTPException(status_code=404, detail=f"Default doctor '{default_doctor_username}' not found")

    # Ensure target conversation exists
    target_conv = db.query(models.Conversation).filter(
        models.Conversation.patient_id == current_user.id,
        models.Conversation.doctor_id == default_doctor.id,
    ).first()
    if not target_conv:
        target_conv = models.Conversation(patient_id=current_user.id, doctor_id=default_doctor.id)
        db.add(target_conv)
        db.commit()
        db.refresh(target_conv)

    other_convs = db.query(models.Conversation).filter(
        models.Conversation.patient_id == current_user.id,
        models.Conversation.doctor_id.isnot(None),
        models.Conversation.doctor_id != default_doctor.id,
    ).all()

    moved_messages = 0
    merged_conversations = 0
    for conv in other_convs:
        # Move messages
        updated = db.query(models.Message).filter(
            models.Message.conversation_id == conv.id
        ).update({"conversation_id": target_conv.id})
        moved_messages += int(updated or 0)

        # Delete old conversation
        db.delete(conv)
        merged_conversations += 1

    db.commit()

    return {
        "target_conversation_id": target_conv.id,
        "merged_conversations": merged_conversations,
        "moved_messages": moved_messages,
        "default_doctor_id": default_doctor.id,
        "default_doctor_username": default_doctor.username,
    }

@app.post("/messages/send")
def send_message(
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message to another user"""
    try:
        recipient = db.query(User).filter(User.id == message.recipient_id).first()
        if not recipient:
            raise HTTPException(status_code=404, detail="Recipient not found")

        # Enforce role-based chat (patient <-> doctor only)
        if current_user.role == "patient" and recipient.role != "doctor":
            raise HTTPException(status_code=400, detail="Patients can only message doctors")
        if current_user.role == "doctor" and recipient.role != "patient":
            raise HTTPException(status_code=400, detail="Doctors can only message patients")

        # Get or create conversation
        conv = db.query(models.Conversation).filter(
            ((models.Conversation.patient_id == current_user.id) & (models.Conversation.doctor_id == message.recipient_id)) |
            ((models.Conversation.patient_id == message.recipient_id) & (models.Conversation.doctor_id == current_user.id))
        ).first()
        
        if not conv:
            # Determine who is patient and who is doctor
            patient_id = current_user.id if current_user.role == "patient" else message.recipient_id
            doctor_id = message.recipient_id if current_user.role == "patient" else current_user.id
            
            conv = models.Conversation(
                patient_id=patient_id,
                doctor_id=doctor_id
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
        
        # Create message
        msg = models.Message(
            conversation_id=conv.id,
            sender_id=current_user.id,
            content=message.content
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        
        return {
            "id": msg.id,
            "content": msg.content,
            "sender_id": msg.sender_id,
            "sender_username": current_user.username,
            "timestamp": msg.timestamp.isoformat(),
            "is_read": msg.is_read
        }
    except Exception as e:
        print(f"Error sending message: {e}")
        raise HTTPException(500, f"Error sending message: {str(e)}")

@app.get("/messages/conversations")
def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all conversations for current user"""
    try:
        conversations = db.query(models.Conversation).filter(
            (models.Conversation.patient_id == current_user.id) | 
            (models.Conversation.doctor_id == current_user.id)
        ).order_by(models.Conversation.updated_at.desc()).all()
        
        result = []
        for conv in conversations:
            last_msg = db.query(models.Message).filter(
                models.Message.conversation_id == conv.id
            ).order_by(models.Message.timestamp.desc()).first()
            
            unread = db.query(func.count(models.Message.id)).filter(
                (models.Message.conversation_id == conv.id) &
                (models.Message.is_read == 0) &
                (models.Message.sender_id != current_user.id)
            ).scalar()
            
            patient_user = db.query(User).filter(User.id == conv.patient_id).first()
            doctor_user = db.query(User).filter(User.id == conv.doctor_id).first() if conv.doctor_id else None
            
            result.append({
                "id": conv.id,
                "patient_id": conv.patient_id,
                "patient_username": patient_user.username if patient_user else "Unknown",
                "doctor_id": conv.doctor_id,
                "doctor_username": doctor_user.username if doctor_user else None,
                "last_message": last_msg.content if last_msg else None,
                "updated_at": conv.updated_at.isoformat(),
                "unread_count": unread
            })
        
        return result
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        raise HTTPException(500, f"Error fetching conversations: {str(e)}")

@app.get("/messages/conversation/{conversation_id}")
def get_conversation_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all messages in a conversation"""
    try:
        conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
        if not conv:
            raise HTTPException(404, "Conversation not found")
        
        # Check authorization
        if not ((conv.patient_id == current_user.id) or (conv.doctor_id == current_user.id)):
            raise HTTPException(403, "Unauthorized")
        
        messages = db.query(models.Message).filter(
            models.Message.conversation_id == conversation_id
        ).order_by(models.Message.timestamp).all()
        
        # Mark as read
        db.query(models.Message).filter(
            (models.Message.conversation_id == conversation_id) &
            (models.Message.is_read == 0) &
            (models.Message.sender_id != current_user.id)
        ).update({"is_read": 1})
        db.commit()
        
        return [
            {
                "id": msg.id,
                "content": msg.content,
                "sender_id": msg.sender_id,
                "sender_username": msg.sender.username,
                "timestamp": msg.timestamp.isoformat(),
                "is_read": msg.is_read
            }
            for msg in messages
        ]
    except Exception as e:
        print(f"Error fetching messages: {e}")
        raise HTTPException(500, f"Error fetching messages: {str(e)}")
