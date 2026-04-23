import sqlite3
import os

# Use absolute path for Windows reliability
db_path = os.path.join("c:\\military_triage_system_CNN\\backend", "triage.db")
print(f"Checking database at: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Clean up ALL patient users and ALL patients/records
    print("Cleaning up ALL users with role 'patient' and removing all existing patients...")
    cursor.execute("DELETE FROM users WHERE role = 'patient'")
    cursor.execute("DELETE FROM patients")
    cursor.execute("DELETE FROM triage_records")
    
    # 2. Add patientId column to users if missing
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    if "patientId" not in columns:
        print("Adding missing 'patientId' column to users table...")
        cursor.execute("ALTER TABLE users ADD COLUMN patientId TEXT")
    
    # 3. Add user_id to patients if missing
    cursor.execute("PRAGMA table_info(patients)")
    columns = [column[1] for column in cursor.fetchall()]
    if "user_id" not in columns:
        print("Adding missing 'user_id' column to patients...")
        cursor.execute("ALTER TABLE patients ADD COLUMN user_id INTEGER")

    conn.commit()
    print("Database cleanup and health checks completed successfully.")
    conn.close()
except Exception as e:
    print(f"Error during cleanup: {e}")
