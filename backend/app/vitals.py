import csv
import random
from pathlib import Path

CSV_PATH = Path(__file__).parent / "vitals_samples.csv"

TRIAGE_LABELS = ["Green", "Yellow", "Red", "Black"]

def load_random_vitals():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return random.choice(rows)


def vitals_override(probabilities, pulse=None, spo2=None, systolic_bp=None, unconscious=False):
    """
    Override triage if vitals indicate critical condition.
    Returns: (final_label, final_confidence, override_reason, final_probabilities)
    """
    reason = None
    final_probs = probabilities.copy()

    # Apply hard overrides
    if unconscious or (spo2 is not None and spo2 < 70) or (systolic_bp is not None and systolic_bp < 70):
        reason = "Critical vitals detected"
        # Boost Black probability significantly
        final_probs["Black"] = max(final_probs.get("Black", 0), 0.95)
        # Reduce others
        for k in final_probs:
            if k != "Black": final_probs[k] *= 0.1
    
    elif (pulse is not None and (pulse > 130 or pulse < 40)) or (spo2 is not None and spo2 < 88):
        reason = "Severe vital signs"
        # Boost Red probability
        final_probs["Red"] = max(final_probs.get("Red", 0), 0.85)
        # Reduce others
        for k in final_probs:
            if k != "Red": final_probs[k] *= 0.2

    # Normalize
    total = sum(final_probs.values())
    if total > 0:
        for k in final_probs: final_probs[k] /= total

    final_label = max(final_probs, key=final_probs.get)
    return final_label, final_probs[final_label], reason, final_probs


def get_vitals_probs(pulse, spo2, systolic_bp, unconscious):
    scores = {k: 0.0 for k in TRIAGE_LABELS}
    
    if unconscious:
        return {"Green": 0.0, "Yellow": 0.0, "Red": 0.0, "Black": 1.0}
        
    p = pulse if pulse is not None else 80
    s = spo2 if spo2 is not None else 98
    bp = systolic_bp if systolic_bp is not None else 120

    # BLACK conditions
    if s < 80 or bp < 60:
        scores["Black"] += 0.8
        scores["Red"] += 0.2
    # RED conditions
    elif p > 120 or p < 50 or s < 90 or bp < 80 or bp > 160:
        scores["Red"] += 0.8
        scores["Yellow"] += 0.2
    # YELLOW conditions
    elif p > 100 or p < 60 or s < 95 or bp < 90 or bp > 140:
        scores["Yellow"] += 0.7
        scores["Green"] += 0.3
    # GREEN conditions
    else:
        scores["Green"] += 0.9
        scores["Yellow"] += 0.1

    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total
            
    return scores


def calculate_dynamic_survival(v, final_label):
    """
    Calculate a granular survival probability based on vitals and triage status.
    """
    import random
    
    pulse = int(v.get("pulse", 80))
    spo2 = int(v.get("spo2", 98))
    bp = int(v.get("bp_systolic", 120))
    resp = int(v.get("resp_rate", 16))
    
    # Base survival by status
    bases = {
        "GREEN": 97.5,
        "YELLOW": 84.0,
        "RED": 42.0,
        "BLACK": 4.0
    }
    
    label = final_label.upper()
    survival = bases.get(label, 50.0)
    
    # Penalty calculation (Vital Stress)
    penalty = 0.0
    
    # SpO2 penalty (High impact)
    if spo2 < 95:
        penalty += (95 - spo2) * 1.5
        
    # Pulse penalty
    if pulse > 100:
        penalty += (pulse - 100) * 0.2
    elif pulse < 60:
        penalty += (60 - pulse) * 0.3
        
    # Resp Rate penalty
    if resp > 20:
        penalty += (resp - 20) * 0.5
    elif resp < 10:
        penalty += (10 - resp) * 0.8
        
    # BP penalty
    if bp < 100:
        penalty += (100 - bp) * 0.4
    elif bp > 150:
        penalty += (bp - 150) * 0.1

    # Apply penalty based on status sensitivity
    if label == "GREEN":
        # Green is very stable, penalty is minimal
        survival -= penalty * 0.2
    elif label == "YELLOW":
        survival -= penalty * 0.5
    else:
        # RED/BLACK are very sensitive to vitals
        survival -= penalty * 1.0
        
    # Add random variation for uniqueness (±1.5%)
    variation = random.uniform(-1.5, 1.5)
    survival += variation
    
    # Clamp results
    if label == "GREEN":
        survival = max(95.0, min(99.9, survival))
    elif label == "BLACK":
        survival = max(0.1, min(15.0, survival))
    else:
        survival = max(15.0, min(94.0, survival))
        
    return round(survival, 1)


def get_hr_status(hr):
    if hr is None: return "Unknown"
    if hr > 120: return "Severe Tachycardia"
    if hr > 100: return "Mild Tachycardia"
    if hr < 50: return "Severe Bradycardia"
    if hr < 60: return "Mild Bradycardia"
    return "Normal"

def get_spo2_status(spo2):
    if spo2 is None: return "Unknown"
    if spo2 < 85: return "Severe Hypoxemia"
    if spo2 < 92: return "Moderate Hypoxemia"
    if spo2 < 95: return "Mild Hypoxemia"
    return "Normal"

def get_bp_status(sys_bp):
    if sys_bp is None: return "Unknown"
    if sys_bp < 70: return "Critical Hypotension"
    if sys_bp < 90: return "Hypotension"
    if sys_bp > 160: return "Severe Hypertension"
    if sys_bp > 140: return "Hypertension"
    return "Normal"

def get_physiological_syndrome(hr, spo2, bp):
    """
    Identifies clinical syndromes based on vital patterns.
    """
    hr_s = get_hr_status(hr)
    spo2_s = get_spo2_status(spo2)
    bp_s = get_bp_status(bp)
    
    # Pattern matching
    if bp_s == "Critical Hypotension" and hr_s in ["Normal", "Mild Bradycardia"]:
        return "Potential Neurogenic Shock or Failing Compensation"
    if bp_s in ["Critical Hypotension", "Hypotension"] and hr_s in ["Severe Tachycardia", "Mild Tachycardia"]:
        return "Decompensated Shock (Likely Hemorrhage)"
    if spo2_s in ["Severe Hypoxemia", "Moderate Hypoxemia"] and hr_s in ["Severe Tachycardia", "Mild Tachycardia"]:
        return "Acute Respiratory Distress with Compensatory Tachycardia"
    if bp_s == "Severe Hypertension" and hr_s in ["Severe Bradycardia", "Mild Bradycardia"]:
        return "Cushing's Triad (Potential Intracranial Pressure)"
    
    if bp_s == "Critical Hypotension": return "Severe Circulatory Collapse"
    if spo2_s == "Severe Hypoxemia": return "Critical Respiratory Failure"
    
    return None
