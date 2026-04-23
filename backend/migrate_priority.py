import os
import sys

# Add the app directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models import Patient
from app.main import calculate_dynamic_priority

def migrate_priorities():
    db = SessionLocal()
    try:
        patients = db.query(Patient).all()
        for p in patients:
            new_priority = calculate_dynamic_priority(
                p.status,
                p.survivalProbability,
                p.spo2,
                p.heartRate
            )
            p.priority = new_priority
        db.commit()
        print(f"Successfully migrated {len(patients)} patients to dynamic priority.")
    except Exception as e:
        print(f"Error migrating priorities: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    migrate_priorities()
