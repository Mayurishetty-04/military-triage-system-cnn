import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use absolute path to guarantee connection to the existing database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Since database.py is in app/ we need to go up one directory
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "triage.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

# connect_args={"check_same_thread": False} is required for SQLite and FastAPI
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()