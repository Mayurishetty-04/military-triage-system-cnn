from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import User
from app.security import (
    hash_password,
    verify_password,
    create_access_token
)
import datetime
import random

router = APIRouter()

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str


class LoginRequest(BaseModel):
    username: str
    password: str
    role: str

from sqlalchemy.exc import IntegrityError

@router.post("/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    print(f"DEBUG: Registering user {data.username} as {data.role}")
    try:
        user = db.query(User).filter(User.username == data.username).first()
        if user:
            print(f"DEBUG: Username {data.username} already exists")
            raise HTTPException(400, "Username already exists")

        new_user = User(
            username=data.username,
            email=data.email,
            hashed_password=hash_password(data.password),
            role=data.role
        )

        if data.role == "patient":
            new_user.patientId = f"PAT-{int(datetime.datetime.now().timestamp())}-{random.randint(100, 999)}"

        db.add(new_user)
        db.commit()
        print(f"DEBUG: User {data.username} registered successfully")
        return {"msg": "Registered successfully"}

    except IntegrityError as e:
        db.rollback()
        print(f"DEBUG: Integrity error during registration: {str(e)}")
        raise HTTPException(400, "Registration failed. Username or email may already exist.")
    except Exception as e:
        db.rollback()
        print(f"DEBUG: Unexpected error during registration: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")



@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.username == data.username).first()

    if not user:
        raise HTTPException(400, "Invalid credentials")

    if user.role != data.role:
        raise HTTPException(400, f"Invalid credentials for {data.role} role")

    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(400, "Invalid credentials")

    token = create_access_token({"sub": user.username, "user_id": user.id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "patientId": user.patientId,
        "user_id": user.id
    }

