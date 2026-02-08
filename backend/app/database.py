from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker, declarative_base 
DATABASE_URL = "postgresql://triage_user:password123@localhost:5432/triage_db" 
engine = create_engine(DATABASE_URL) 
SessionLocal = sessionmaker( autocommit=False, autoflush=False, bind=engine ) 
Base = declarative_base() 
def get_db(): 
    db = SessionLocal() 
    try: 
        yield db 
    finally: 
        db.close()