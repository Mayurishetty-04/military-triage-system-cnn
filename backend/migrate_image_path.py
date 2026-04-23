import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'triage.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE patients ADD COLUMN image_path VARCHAR;")
    conn.commit()
    print("Successfully added image_path column to patients table.")
except sqlite3.OperationalError as e:
    print(f"Error: {e}")
finally:
    conn.close()
