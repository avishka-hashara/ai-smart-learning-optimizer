"""Accounts: register / login / logout / me / delete account.

Sessions are a signed JWT in an httpOnly cookie (not readable by page JS, so
an XSS bug can't steal it). SameSite=Lax stops other sites from sending it on
cross-site POST/DELETE, which covers CSRF for this same-origin app. API
clients can send the same token as `Authorization: Bearer <token>`.
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User

log = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    log.warning("SECRET_KEY not set; using a random key. Sessions end when the server restarts.")

ALGORITHM = "HS256"
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "7"))
COOKIE_NAME = "session"
# Set COOKIE_SECURE=1 when served over HTTPS (always, in production)
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"

# bcrypt only reads the first 72 bytes of a password
MAX_PASSWORD_BYTES = 72

# Hash compared against when the email is unknown, so a failed login takes
# the same time whether or not the account exists
_DUMMY_HASH = bcrypt.hashpw(b"timing-equaliser", bcrypt.gensalt())

router = APIRouter(prefix="/api", tags=["auth"])


# ---------------------------------------------------------------- schemas
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)
    name: str = Field(min_length=1, max_length=80)

    @field_validator("password")
    @classmethod
    def password_fits_bcrypt(cls, v):
        if len(v.encode()) > MAX_PASSWORD_BYTES:
            raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
        return v

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be blank.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(max_length=MAX_PASSWORD_BYTES * 4)


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    user: UserOut
    token: str


# ---------------------------------------------------------------- helpers
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password, hashed):
    pw = password.encode()
    if len(pw) > MAX_PASSWORD_BYTES:
        return False
    return bcrypt.checkpw(pw, hashed.encode())


def create_token(user_id):
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(days=SESSION_DAYS)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _set_session(response, token):
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=SESSION_DAYS * 86400,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _token_from(request):
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.cookies.get(COOKIE_NAME)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in.",
                                 headers={"WWW-Authenticate": "Bearer"})
    token = _token_from(request)
    if not token:
        raise unauthorized
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise unauthorized
    user = db.get(User, user_id)
    if user is None:
        raise unauthorized
    return user


# ---------------------------------------------------------------- routes
@router.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    user = User(email=req.email.lower(), password_hash=hash_password(req.password), name=req.name)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")
    token = create_token(user.id)
    _set_session(response, token)
    return {"user": user, "token": token}


@router.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == req.email.lower()))
    if user is None:
        verify_password(req.password, _DUMMY_HASH.decode())
        ok = False
    else:
        ok = verify_password(req.password, user.password_hash)
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password.")
    token = create_token(user.id)
    _set_session(response, token)
    return {"user": user, "token": token}


@router.post("/auth/logout", status_code=204)
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.delete("/me", status_code=204)
def delete_account(response: Response, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Permanently deletes the account and all of its subjects, check-ins and plans."""
    db.delete(user)
    db.commit()
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")
