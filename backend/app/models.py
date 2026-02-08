from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey 
from sqlalchemy.orm import relationship 
from datetime import datetime 
from database import Base 
class User(Base): 
    __tablename__ = "users" 
    id = Column(Integer, primary_key=True, index=True) 
    username = Column(String, unique=True, index=True) 
    email = Column(String, unique=True, index=True) 
    hashed_password = Column(String) 
    records = relationship("TriageRecord", back_populates="user")

class TriageRecord(Base): 
    __tablename__ = "triage_records" 
    id = Column(Integer, primary_key=True, index=True) 
    triage_level = Column(String) 
    confidence = Column(Float) 
    timestamp = Column(DateTime, default=datetime.utcnow) 
    user_id = Column(Integer, ForeignKey("users.id")) 
    user = relationship("User", back_populates="records")