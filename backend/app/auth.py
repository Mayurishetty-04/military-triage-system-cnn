from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models import User
from security import (
    hash_password,
    verify_password,
    create_access_token
)

router = APIRouter()

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.username == data.username).first()

    if user:
        raise HTTPException(400, "Username already exists")

    new_user = User(
        username=data.username,
        email=data.email,
        hashed_password=hash_password(data.password)
    )

    db.add(new_user)
    db.commit()

    return {"msg": "Registered successfully"}



@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.username == data.username).first()

    if not user:
        raise HTTPException(400, "Invalid credentials")

    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(400, "Invalid credentials")

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }

