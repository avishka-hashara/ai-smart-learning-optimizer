"""Per-user subjects and weekly check-ins.

A check-in is keyed by (user, week). Submitting again for the same week
replaces that week's entries and plan instead of adding a second one.
"""

import os
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.auth import get_current_user
from app.config import (CONF_RANGE, DEFAULT_DIFFICULTY, DIFFICULTY_RANGE, HOURS_RANGE,
                        MAX_SUBJECTS, SUBJECT_COLORS, run_pipeline, to_payload)
from app.db import get_db
from app.models import Checkin, CheckinEntry, Plan, Subject, User, utcnow
from src.prediction.predict import MODEL_PATH

MODEL_VERSION = os.path.splitext(os.path.basename(MODEL_PATH))[0]

router = APIRouter(prefix="/api", tags=["study"])


# ---------------------------------------------------------------- schemas
def _clean_name(v):
    v = " ".join(v.split())
    if not v:
        raise ValueError("Name cannot be blank.")
    return v


class SubjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    icon: str = Field(default="📘", min_length=1, max_length=8)
    difficulty: int = Field(default=DEFAULT_DIFFICULTY, ge=DIFFICULTY_RANGE[0], le=DIFFICULTY_RANGE[1])
    exam_date: date | None = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, v):
        return _clean_name(v)


class SubjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=60)
    icon: str | None = Field(default=None, min_length=1, max_length=8)
    difficulty: int | None = Field(default=None, ge=DIFFICULTY_RANGE[0], le=DIFFICULTY_RANGE[1])
    exam_date: date | None = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, v):
        return v if v is None else _clean_name(v)


class SubjectOut(BaseModel):
    id: int
    name: str
    icon: str
    color: str
    difficulty: int
    exam_date: date | None
    archived: bool

    model_config = {"from_attributes": True}


class EntryIn(BaseModel):
    subject_id: int
    hours: float = Field(ge=HOURS_RANGE[0], le=HOURS_RANGE[1])
    confidence: float = Field(ge=CONF_RANGE[0], le=CONF_RANGE[1])


class CheckinIn(BaseModel):
    # Any date in the week; normalised to that week's Monday. Defaults to this week.
    week_start: date | None = None
    entries: list[EntryIn] = Field(min_length=1, max_length=MAX_SUBJECTS)


class EntryOut(BaseModel):
    subject_id: int
    subject_name: str
    hours: float
    confidence: float


class PlanOut(BaseModel):
    model_version: str
    created_at: datetime
    result: dict


class CheckinOut(BaseModel):
    id: int
    week_start: date
    created_at: datetime
    updated_at: datetime
    entries: list[EntryOut]
    plan: PlanOut | None


class CheckinSummary(BaseModel):
    id: int
    week_start: date
    updated_at: datetime
    n_subjects: int
    budget: float
    avg_predicted: float
    fitness: float
    top_priority: str


class MetaOut(BaseModel):
    hours_range: tuple[float, float]
    conf_range: tuple[float, float]
    difficulty_range: tuple[int, int]
    max_subjects: int
    current_week: date


# ---------------------------------------------------------------- helpers
def monday_of(d):
    return d - timedelta(days=d.weekday())


def _get_subject(db, user, subject_id):
    subject = db.get(Subject, subject_id)
    if subject is None or subject.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Subject not found.")
    return subject


def _get_checkin(db, user, checkin_id):
    checkin = db.scalar(
        select(Checkin)
        .where(Checkin.id == checkin_id, Checkin.user_id == user.id)
        .options(selectinload(Checkin.entries).selectinload(CheckinEntry.subject),
                 selectinload(Checkin.plan)))
    if checkin is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Check-in not found.")
    return checkin


def _active_subjects(db, user):
    return db.scalars(
        select(Subject)
        .where(Subject.user_id == user.id, Subject.archived.is_(False))
        .order_by(Subject.created_at, Subject.id)).all()


def _name_taken(db, user, name, exclude_id=None):
    q = select(Subject).where(Subject.user_id == user.id, Subject.name.ilike(name))
    if exclude_id is not None:
        q = q.where(Subject.id != exclude_id)
    return db.scalar(q)


def _checkin_out(c):
    return {
        "id": c.id,
        "week_start": c.week_start,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
        "entries": [{"subject_id": e.subject_id, "subject_name": e.subject.name,
                     "hours": e.hours, "confidence": e.confidence} for e in c.entries],
        "plan": None if c.plan is None else {
            "model_version": c.plan.model_version,
            "created_at": c.plan.created_at,
            "result": c.plan.result,
        },
    }


# ---------------------------------------------------------------- meta
@router.get("/me/meta", response_model=MetaOut)
def user_meta(user: User = Depends(get_current_user)):
    return {
        "hours_range": HOURS_RANGE,
        "conf_range": CONF_RANGE,
        "difficulty_range": DIFFICULTY_RANGE,
        "max_subjects": MAX_SUBJECTS,
        "current_week": monday_of(date.today()),
    }


# ---------------------------------------------------------------- subjects
@router.get("/subjects", response_model=list[SubjectOut])
def list_subjects(include_archived: bool = False, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    q = select(Subject).where(Subject.user_id == user.id)
    if not include_archived:
        q = q.where(Subject.archived.is_(False))
    return db.scalars(q.order_by(Subject.created_at, Subject.id)).all()


@router.post("/subjects", response_model=SubjectOut, status_code=201)
def create_subject(req: SubjectIn, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    active = _active_subjects(db, user)
    if len(active) >= MAX_SUBJECTS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"You can track at most {MAX_SUBJECTS} subjects.")

    existing = _name_taken(db, user, req.name)
    if existing is not None and not existing.archived:
        raise HTTPException(status.HTTP_409_CONFLICT, f'You already have a subject called "{existing.name}".')

    used = {s.color for s in active}
    color = next((c for c in SUBJECT_COLORS if c not in used), SUBJECT_COLORS[0])

    if existing is not None:
        # Re-adding an archived subject restores it, keeping its history
        subject = existing
        subject.archived = False
        subject.name, subject.icon = req.name, req.icon
        subject.difficulty, subject.exam_date = req.difficulty, req.exam_date
        if subject.color in used:
            subject.color = color
    else:
        subject = Subject(user_id=user.id, color=color, **req.model_dump())
        db.add(subject)
    db.commit()
    return subject


@router.patch("/subjects/{subject_id}", response_model=SubjectOut)
def update_subject(subject_id: int, req: SubjectPatch, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    subject = _get_subject(db, user, subject_id)
    changes = req.model_dump(exclude_unset=True)
    if changes.get("name") is None:
        changes.pop("name", None)
    if "name" in changes and _name_taken(db, user, changes["name"], exclude_id=subject.id):
        raise HTTPException(status.HTTP_409_CONFLICT, f'You already have a subject called "{changes["name"]}".')
    for key in ("icon", "difficulty"):
        if key in changes and changes[key] is None:
            changes.pop(key)
    for key, value in changes.items():
        setattr(subject, key, value)
    db.commit()
    return subject


@router.delete("/subjects/{subject_id}", status_code=204)
def delete_subject(subject_id: int, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Archives a subject that appears in past check-ins; deletes it otherwise."""
    subject = _get_subject(db, user, subject_id)
    in_history = db.scalar(select(CheckinEntry.id).where(CheckinEntry.subject_id == subject.id).limit(1))
    if in_history:
        subject.archived = True
    else:
        db.delete(subject)
    db.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------- check-ins
@router.post("/checkins", response_model=CheckinOut)
def submit_checkin(req: CheckinIn, response: Response, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Saves this week's hours and confidence, runs the pipeline, stores the plan.

    Returns 201 for a new week and 200 when the week's check-in was replaced.
    """
    this_week = monday_of(date.today())
    week = monday_of(req.week_start) if req.week_start else this_week
    if week > this_week:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You can't check in for a future week.")

    ids = [e.subject_id for e in req.entries]
    if len(set(ids)) != len(ids):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Each subject can appear only once.")
    subjects = {s.id: s for s in _active_subjects(db, user)}
    unknown = [i for i in ids if i not in subjects]
    if unknown:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown or archived subject id(s): {unknown}.")

    ordered = [subjects[i] for i in ids]
    hours = [e.hours for e in req.entries]
    exam_dates = {s.name: s.exam_date.isoformat() for s in ordered
                  if s.exam_date is not None and s.exam_date >= week}
    try:
        res = run_pipeline(
            hours,
            [e.confidence for e in req.entries],
            subjects=[s.name for s in ordered],
            difficulty=[s.difficulty for s in ordered],
            exam_dates=exam_dates or None,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    payload = to_payload(res, hours, [{"name": s.name, "icon": s.icon, "color": s.color} for s in ordered])

    checkin = db.scalar(select(Checkin).where(Checkin.user_id == user.id, Checkin.week_start == week))
    created = checkin is None
    if created:
        checkin = Checkin(user_id=user.id, week_start=week)
        db.add(checkin)
    else:
        # Flush the orphan deletes before inserting, or the unique
        # (checkin, subject) / plan.checkin_id constraints would clash
        checkin.entries.clear()
        checkin.plan = None
        db.flush()
        # Set explicitly: onupdate only fires when a column on this row changes
        checkin.updated_at = utcnow()

    checkin.entries = [CheckinEntry(subject_id=e.subject_id, hours=e.hours, confidence=e.confidence)
                       for e in req.entries]
    checkin.plan = Plan(model_version=MODEL_VERSION, result=payload)
    db.commit()

    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return _checkin_out(_get_checkin(db, user, checkin.id))


@router.get("/checkins", response_model=list[CheckinSummary])
def list_checkins(limit: int = 52, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """Most recent weeks first, with headline numbers for the history list."""
    limit = max(1, min(limit, 520))
    rows = db.scalars(
        select(Checkin)
        .where(Checkin.user_id == user.id)
        .options(selectinload(Checkin.plan))
        .order_by(Checkin.week_start.desc())
        .limit(limit)).all()

    out = []
    for c in rows:
        if c.plan is None:
            continue
        r = c.plan.result
        prio = r["risk"]["priority"]
        out.append({
            "id": c.id,
            "week_start": c.week_start,
            "updated_at": c.updated_at,
            "n_subjects": len(r["subjects"]),
            "budget": r["budget"],
            "avg_predicted": round(sum(r["predicted"]) / len(r["predicted"]), 2),
            "fitness": r["allocation"]["fitness"],
            "top_priority": r["subjects"][prio.index(max(prio))],
        })
    return out


@router.get("/checkins/{checkin_id}", response_model=CheckinOut)
def get_checkin(checkin_id: int, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    return _checkin_out(_get_checkin(db, user, checkin_id))


@router.delete("/checkins/{checkin_id}", status_code=204)
def delete_checkin(checkin_id: int, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    db.delete(_get_checkin(db, user, checkin_id))
    db.commit()
    return Response(status_code=204)
