from app.database import SessionLocal
from app.models import User
import sys

try:
    db = SessionLocal()
    print("DB Session created")
    count = db.query(User).count()
    print(f"User count: {count}")
    db.close()
    print("DB Success")
except Exception as e:
    print(f"DB Error: {e}")
    sys.exit(1)
