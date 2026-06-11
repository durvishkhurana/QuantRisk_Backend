from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth import create_access_token, hash_password, verify_password
from app.database import get_db
from app.models import User
from app.schemas import AuthLoginIn, AuthOut, AuthRegisterIn

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthOut)
async def register(payload: AuthRegisterIn, db: AsyncSession = Depends(get_db)) -> AuthOut:
    existing = await db.execute(select(User).where(User.email == payload.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(email=payload.email.lower(), password_hash=hash_password(payload.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token, expires_at = create_access_token(user_id=str(user.id), email=user.email)
    return AuthOut(user_id=user.id, email=user.email, token=token, expires_at=expires_at)


@router.post("/login", response_model=AuthOut)
async def login(payload: AuthLoginIn, db: AsyncSession = Depends(get_db)) -> AuthOut:
    result = await db.execute(select(User).where(User.email == payload.email.lower()))
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token, expires_at = create_access_token(user_id=str(user.id), email=user.email)
    return AuthOut(user_id=user.id, email=user.email, token=token, expires_at=expires_at)
