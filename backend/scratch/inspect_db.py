import sqlite3
import os

db_path = r"c:\military_triage_system_CNN\backend\triage.db"

if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    # Query user 'test1'
    cursor.execute("SELECT id, username, email, role FROM users WHERE username='test1';")
    user = cursor.fetchone()
    
    if user:
        print(f"User found: ID={user[0]}, Username='{user[1]}', Email='{user[2]}', Role='{user[3]}'")
    else:
        print("User 'test1' NOT found in the database.")
    
    # List all users
    cursor.execute("SELECT id, username, role FROM users;")
    all_users = cursor.fetchall()
    print(f"All users: {all_users}")
    
    conn.close()
