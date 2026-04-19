import sqlite3
import os

def migrate():
    db_path = os.path.join(os.path.dirname(__file__), "triage.db")
    print(f"Connecting to database at: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if explanation column exists in patients table
        cursor.execute("PRAGMA table_info(patients)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "explanation" not in columns:
            print("Adding 'explanation' column to 'patients' table...")
            cursor.execute("ALTER TABLE patients ADD COLUMN explanation TEXT")
            conn.commit()
            print("Migration successful: 'explanation' column added.")
        else:
            print("Column 'explanation' already exists in 'patients' table.")
            
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
