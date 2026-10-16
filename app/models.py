"""ORM models.

users 1─* subjects
users 1─* checkins 1─* checkin_entries *─1 subjects
checkins 1─1 plans

A check-in is one week's self-report (hours + confidence per subject). Each
check-in gets exactly one plan: the stored pipeline output for that input.
Subjects are archived rather than deleted so past check-ins keep their rows.
"""

from datetime import date, datetime, timezone

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer,
                        String, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(80))
    university: Mapped[str | None] = mapped_column(String(120))
    programme: Mapped[str | None] = mapped_column(String(120))
    year_of_study: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Sessions issued before this are rejected (set on password change)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    subjects: Mapped[list["Subject"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    checkins: Mapped[list["Checkin"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True)


class Subject(Base):
    __tablename__ = "subjects"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_subject_user_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(60))
    icon: Mapped[str] = mapped_column(String(8), default="📘")
    color: Mapped[str] = mapped_column(String(9))
    difficulty: Mapped[int] = mapped_column(Integer, default=2)
    exam_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="subjects")


class Checkin(Base):
    __tablename__ = "checkins"
    __table_args__ = (UniqueConstraint("user_id", "week_start", name="uq_checkin_user_week"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    week_start: Mapped[date] = mapped_column(Date)  # always a Monday
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="checkins")
    entries: Mapped[list["CheckinEntry"]] = relationship(
        back_populates="checkin", cascade="all, delete-orphan", passive_deletes=True)
    plan: Mapped["Plan | None"] = relationship(
        back_populates="checkin", cascade="all, delete-orphan", passive_deletes=True, uselist=False)


class CheckinEntry(Base):
    __tablename__ = "checkin_entries"
    __table_args__ = (UniqueConstraint("checkin_id", "subject_id", name="uq_entry_checkin_subject"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    checkin_id: Mapped[int] = mapped_column(ForeignKey("checkins.id", ondelete="CASCADE"), index=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    hours: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)

    checkin: Mapped[Checkin] = relationship(back_populates="entries")
    subject: Mapped[Subject] = relationship()


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    checkin_id: Mapped[int] = mapped_column(
        ForeignKey("checkins.id", ondelete="CASCADE"), unique=True)
    model_version: Mapped[str] = mapped_column(String(40))
    # Full pipeline payload exactly as the dashboard renders it, so old plans
    # display the same even after subjects are renamed or the model changes.
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    checkin: Mapped[Checkin] = relationship(back_populates="plan")
