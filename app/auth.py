"""Accounts: register / login / logout / profile / password / delete account.

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
def _fits_bcrypt(v):
    if len(v.encode()) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")
    return v


def _not_blank(v):
    if v is None:
        return v
    v = v.strip()
    if not v:
        raise ValueError("Name cannot be blank.")
    return v


def _blank_to_none(v):
    if v is None:
        return v
    return v.strip() or None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)
    name: str = Field(min_length=1, max_length=80)

    check_password = field_validator("password")(_fits_bcrypt)
    check_name = field_validator("name")(_not_blank)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(max_length=MAX_PASSWORD_BYTES * 4)


class ProfileUpdate(BaseModel):
    """Only the fields sent are changed. Changing the email needs the current password."""
    name: str | None = Field(None, min_length=1, max_length=80)
    email: EmailStr | None = None
    university: str | None = Field(None, max_length=120)
    programme: str | None = Field(None, max_length=120)
    year_of_study: int | None = Field(None, ge=1, le=8)
    current_password: str | None = Field(None, max_length=MAX_PASSWORD_BYTES * 4)

    check_name = field_validator("name")(_not_blank)
    clean_optional = field_validator("university", "programme")(_blank_to_none)


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=MAX_PASSWORD_BYTES * 4)
    new_password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)

    check_password = field_validator("new_password")(_fits_bcrypt)


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    university: str | None
    programme: str | None
    year_of_study: int | None
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


def _password_stamp(user):
    """Changes whenever the password does; tokens carry it so old ones stop working."""
    changed = user.password_changed_at
    if changed is None:
        return 0
    if changed.tzinfo is None:  # SQLite hands datetimes back without a zone
        changed = changed.replace(tzinfo=timezone.utc)
    return int(changed.timestamp() * 1000)


def create_token(user):
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user.id), "pwd": _password_stamp(user),
               "iat": now, "exp": now + timedelta(days=SESSION_DAYS)}
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
    if user is None or payload.get("pwd", 0) != _password_stamp(user):
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
    token = create_token(user)
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
    token = create_token(user)
    _set_session(response, token)
    return {"user": user, "token": token}


@router.post("/auth/logout", status_code=204)
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/me", response_model=UserOut)
def update_profile(req: ProfileUpdate, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    changes = req.model_dump(include=req.model_fields_set - {"current_password"})
    if changes.get("name", "") is None:
        raise HTTPException(422, "Name cannot be blank.")
    if "email" in changes:
        if changes["email"] is None:
            raise HTTPException(422, "Email cannot be blank.")
        changes["email"] = changes["email"].lower()
        if changes["email"] == user.email:
            del changes["email"]
        elif not req.current_password or not verify_password(req.current_password, user.password_hash):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Enter your current password to change your email.")
    for field, value in changes.items():
        setattr(user, field, value)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")
    return user


@router.post("/me/password", status_code=204)
def change_password(req: PasswordChange, response: Response, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """Sets a new password and signs out every other session; this one gets a fresh token."""
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Current password is incorrect.")
    if req.new_password == req.current_password:
        raise HTTPException(422, "New password must be different.")
    user.password_hash = hash_password(req.new_password)
    user.password_changed_at = datetime.now(timezone.utc)
    db.commit()
    _set_session(response, create_token(user))


@router.delete("/me", status_code=204)
def delete_account(response: Response, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Permanently deletes the account and all of its subjects, check-ins and plans."""
    db.delete(user)
    db.commit()
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")
