import sqlite3
import os

def migrate():
    db_path = os.path.join(os.path.dirname(__file__), "triage.db")
    print(f"Connecting to database at: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if patientId column exists in users table
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "patientId" not in columns:
            print("Adding 'patientId' column to 'users' table...")
            cursor.execute("ALTER TABLE users ADD COLUMN patientId TEXT")
            conn.commit()
            print("Migration successful: 'patientId' column added.")
        else:
            print("Column 'patientId' already exists in 'users' table.")
            
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
