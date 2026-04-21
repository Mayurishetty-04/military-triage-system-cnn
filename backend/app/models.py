from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey 
from sqlalchemy.orm import relationship 
from datetime import datetime 
from app.database import Base 
class User(Base): 
    __tablename__ = "users" 
    id = Column(Integer, primary_key=True, index=True) 
    username = Column(String, unique=True, index=True) 
    email = Column(String, unique=True, index=True) 
    hashed_password = Column(String) 
    role = Column(String, default="patient") # "patient" or "doctor"
    patientId = Column(String, nullable=True, unique=True)  # Fixed ID assigned at registration
    records = relationship("TriageRecord", back_populates="user")

class TriageRecord(Base): 
    __tablename__ = "triage_records" 
    id = Column(Integer, primary_key=True, index=True) 
    triage_level = Column(String) 
    confidence = Column(Float) 
    timestamp = Column(DateTime, default=datetime.now) 
    user_id = Column(Integer, ForeignKey("users.id")) 
    user = relationship("User", back_populates="records")

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    patientId = Column(String, index=True)
    patientName = Column(String, nullable=True)   # username of the logged-in patient
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(String)  # RED, YELLOW, GREEN, BLACK
    survivalProbability = Column(Float)
    injuryType = Column(String)
    spo2 = Column(Integer)
    heartRate = Column(Integer)
    imageScore = Column(Float)
    audioScore = Column(Float)
    videoScore = Column(Float)
    textScore = Column(Float, nullable=True)
    recommendation = Column(String)
    priority = Column(Integer)  # 1 for RED, 2 for YELLOW, 3 for GREEN, 4 for BLACK
    explanation = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    image_path = Column(String, nullable=True)
    is_acknowledged = Column(Integer, default=0) # 0 for False, 1 for True (SQLite boolean)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    sender_id = Column(Integer, ForeignKey("users.id"))
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    is_read = Column(Integer, default=0)
    sender = relationship("User")
    conversation = relationship("Conversation", back_populates="messages")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    patient = relationship("User", foreign_keys=[patient_id])
    doctor = relationship("User", foreign_keys=[doctor_id])
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
