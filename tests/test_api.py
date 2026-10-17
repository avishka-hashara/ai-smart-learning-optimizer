"""API tests for accounts, subjects and weekly check-ins.

Uses a throwaway SQLite file that is built by the real Alembic migrations
(through the app's startup hook), so the migration is tested too.
"""

import os
import sys
import tempfile
from datetime import date, timedelta

import pytest

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp, 'test.db')}"
os.environ["AUTO_MIGRATE"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402
from src.prediction.predict import MODEL_PATH  # noqa: E402

needs_model = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="trained model not present")

PASSWORD = "correct horse battery"


@pytest.fixture(scope="module")
def server():
    with TestClient(app) as c:  # context manager runs the startup migrations
        yield c


def new_client(server, email, name="Test"):
    c = TestClient(app)
    r = c.post("/api/auth/register", json={"email": email, "password": PASSWORD, "name": name})
    assert r.status_code == 201, r.text
    return c


def add_subjects(c, *names):
    ids = []
    for n in names:
        r = c.post("/api/subjects", json={"name": n})
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    return ids


# ---------------------------------------------------------------- auth
def test_register_login_logout(server):
    c = new_client(server, "alice@example.com", "Alice")
    assert "session" in c.cookies
    me = c.get("/api/me").json()
    assert me["email"] == "alice@example.com" and me["name"] == "Alice"
    assert "password_hash" not in me

    assert c.post("/api/auth/logout").status_code == 204
    c.cookies.clear()
    assert c.get("/api/me").status_code == 401

    # Email match is case-insensitive
    r = c.post("/api/auth/login", json={"email": "ALICE@example.com", "password": PASSWORD})
    assert r.status_code == 200
    assert c.get("/api/me").status_code == 200

    # Bearer token works for non-browser clients
    token = r.json()["token"]
    bare = TestClient(app)
    assert bare.get("/api/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_register_rejects_duplicates_and_weak_input(server):
    new_client(server, "dup@example.com")
    c = TestClient(app)
    r = c.post("/api/auth/register", json={"email": "Dup@Example.com", "password": PASSWORD, "name": "X"})
    assert r.status_code == 409
    assert c.post("/api/auth/register", json={"email": "x@example.com", "password": "short", "name": "X"}).status_code == 422
    assert c.post("/api/auth/register", json={"email": "not-an-email", "password": PASSWORD, "name": "X"}).status_code == 422
    assert c.post("/api/auth/register", json={"email": "y@example.com", "password": PASSWORD, "name": "   "}).status_code == 422


def test_bad_login_is_generic(server):
    new_client(server, "bob@example.com")
    c = TestClient(app)
    wrong_pw = c.post("/api/auth/login", json={"email": "bob@example.com", "password": "nope-nope-nope"})
    no_user = c.post("/api/auth/login", json={"email": "ghost@example.com", "password": PASSWORD})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_tampered_token_rejected(server):
    c = new_client(server, "tamper@example.com")
    c.cookies.set("session", c.cookies["session"][:-2] + "xx")
    assert c.get("/api/me").status_code == 401


def test_protected_routes_need_login(server):
    c = TestClient(app)
    assert c.get("/api/subjects").status_code == 401
    assert c.get("/api/checkins").status_code == 401
    assert c.post("/api/checkins", json={"entries": []}).status_code == 401


# ---------------------------------------------------------------- subjects
def test_subject_crud(server):
    c = new_client(server, "subjects@example.com")
    r = c.post("/api/subjects", json={"name": "  Operating   Systems ", "difficulty": 3, "exam_date": "2026-12-01"})
    assert r.status_code == 201
    s = r.json()
    assert s["name"] == "Operating Systems" and s["difficulty"] == 3 and s["color"]

    assert c.post("/api/subjects", json={"name": "operating systems"}).status_code == 409
    assert c.post("/api/subjects", json={"name": "X", "difficulty": 5}).status_code == 422

    other = c.post("/api/subjects", json={"name": "Compilers"}).json()
    assert other["color"] != s["color"]

    r = c.patch(f"/api/subjects/{s['id']}", json={"name": "OS", "exam_date": None})
    assert r.status_code == 200 and r.json()["name"] == "OS" and r.json()["exam_date"] is None
    assert c.patch(f"/api/subjects/{s['id']}", json={"name": "compilers"}).status_code == 409

    assert c.delete(f"/api/subjects/{other['id']}").status_code == 204
    names = [x["name"] for x in c.get("/api/subjects").json()]
    assert names == ["OS"]


def test_subject_limit(server):
    c = new_client(server, "limit@example.com")
    limit = c.get("/api/me/meta").json()["max_subjects"]
    add_subjects(c, *[f"S{i}" for i in range(limit)])
    assert c.post("/api/subjects", json={"name": "One too many"}).status_code == 400


def test_users_cannot_touch_each_others_subjects(server):
    a = new_client(server, "a-iso@example.com")
    b = new_client(server, "b-iso@example.com")
    (sid,) = add_subjects(a, "Private")
    assert b.patch(f"/api/subjects/{sid}", json={"name": "Hacked"}).status_code == 404
    assert b.delete(f"/api/subjects/{sid}").status_code == 404
    assert b.get("/api/subjects").json() == []


# ---------------------------------------------------------------- check-ins
@needs_model
def test_checkin_runs_pipeline_and_stores_plan(server):
    c = new_client(server, "checkin@example.com")
    ids = add_subjects(c, "Algorithms", "Databases", "Ethics")
    body = {"entries": [{"subject_id": i, "hours": h, "confidence": conf}
                        for i, h, conf in zip(ids, [10, 6, 4], [8, 4, 6])]}

    r = c.post("/api/checkins", json=body)
    assert r.status_code == 201, r.text
    ck = r.json()
    plan = ck["plan"]["result"]
    assert plan["subjects"] == ["Algorithms", "Databases", "Ethics"]
    assert [m["name"] for m in plan["subject_meta"]] == plan["subjects"]
    assert len(plan["predicted"]) == len(plan["allocation"]["hours"]) == 3
    assert abs(sum(plan["allocation"]["hours"]) - 20) < 0.05
    assert ck["week_start"] == (date.today() - timedelta(days=date.today().weekday())).isoformat()
    assert ck["plan"]["model_version"]

    # Same week again replaces rather than duplicates
    body["entries"][0]["hours"] = 12
    r2 = c.post("/api/checkins", json=body)
    assert r2.status_code == 200
    assert r2.json()["id"] == ck["id"]
    assert r2.json()["plan"]["result"]["budget"] == 22

    # A previous week is a separate check-in
    last_week = (date.today() - timedelta(days=7)).isoformat()
    r3 = c.post("/api/checkins", json={**body, "week_start": last_week})
    assert r3.status_code == 201 and r3.json()["id"] != ck["id"]

    history = c.get("/api/checkins").json()
    assert [h["id"] for h in history] == [ck["id"], r3.json()["id"]]  # newest week first
    assert history[0]["top_priority"] in plan["subjects"]

    full = c.get(f"/api/checkins/{ck['id']}").json()
    assert [e["subject_name"] for e in full["entries"]] == plan["subjects"]

    assert c.delete(f"/api/checkins/{r3.json()['id']}").status_code == 204
    assert len(c.get("/api/checkins").json()) == 1


@needs_model
def test_checkin_validation(server):
    c = new_client(server, "validate@example.com")
    (sid,) = add_subjects(c, "Physics")
    entry = {"subject_id": sid, "hours": 5, "confidence": 5}

    future = (date.today() + timedelta(days=7)).isoformat()
    assert c.post("/api/checkins", json={"entries": [entry], "week_start": future}).status_code == 400
    assert c.post("/api/checkins", json={"entries": [entry, entry]}).status_code == 400
    assert c.post("/api/checkins", json={"entries": [{**entry, "hours": 99}]}).status_code == 422
    assert c.post("/api/checkins", json={"entries": [{**entry, "confidence": 0}]}).status_code == 422
    assert c.post("/api/checkins", json={"entries": [{**entry, "subject_id": 999999}]}).status_code == 400
    assert c.post("/api/checkins", json={"entries": []}).status_code == 422


@needs_model
def test_checkins_are_private(server):
    a = new_client(server, "a-ck@example.com")
    b = new_client(server, "b-ck@example.com")
    (sid,) = add_subjects(a, "Secret")
    ck = a.post("/api/checkins", json={"entries": [{"subject_id": sid, "hours": 5, "confidence": 5}]}).json()

    assert b.get(f"/api/checkins/{ck['id']}").status_code == 404
    assert b.delete(f"/api/checkins/{ck['id']}").status_code == 404
    # Can't submit someone else's subject either
    assert b.post("/api/checkins", json={"entries": [{"subject_id": sid, "hours": 5, "confidence": 5}]}).status_code == 400


@needs_model
def test_deleting_used_subject_archives_it(server):
    c = new_client(server, "archive@example.com")
    sid, other = add_subjects(c, "Chemistry", "Biology")
    c.post("/api/checkins", json={"entries": [{"subject_id": sid, "hours": 5, "confidence": 5}]})

    assert c.delete(f"/api/subjects/{sid}").status_code == 204
    assert [s["name"] for s in c.get("/api/subjects").json()] == ["Biology"]
    archived = c.get("/api/subjects?include_archived=true").json()
    assert any(s["id"] == sid and s["archived"] for s in archived)

    # History still renders with the archived subject's name
    (ck,) = c.get("/api/checkins").json()
    assert c.get(f"/api/checkins/{ck['id']}").json()["entries"][0]["subject_name"] == "Chemistry"

    # Archived subjects can't be checked in; re-adding by name restores it
    assert c.post("/api/checkins", json={"entries": [{"subject_id": sid, "hours": 5, "confidence": 5}]}).status_code == 400
    restored = c.post("/api/subjects", json={"name": "chemistry"}).json()
    assert restored["id"] == sid and not restored["archived"]


@needs_model
def test_update_profile(server):
    c = new_client(server, "prof@example.com", "Prof")
    r = c.patch("/api/me", json={"name": "  Pat Smith ", "university": "UoM",
                                 "programme": "BSc AI", "year_of_study": 2})
    assert r.status_code == 200, r.text
    me = r.json()
    assert (me["name"], me["university"], me["programme"], me["year_of_study"]) == \
        ("Pat Smith", "UoM", "BSc AI", 2)

    # Unsent fields stay; blank optional text clears it
    me = c.patch("/api/me", json={"university": "  "}).json()
    assert me["university"] is None and me["programme"] == "BSc AI"

    assert c.patch("/api/me", json={"name": "   "}).status_code == 422
    assert c.patch("/api/me", json={"name": None}).status_code == 422
    assert c.patch("/api/me", json={"year_of_study": 0}).status_code == 422
    assert c.patch("/api/me", json={"university": "x" * 121}).status_code == 422
    assert TestClient(app).patch("/api/me", json={"name": "X"}).status_code == 401


def test_change_email_needs_password_and_must_be_free(server):
    c = new_client(server, "mover@example.com")
    new_client(server, "taken@example.com")

    assert c.patch("/api/me", json={"email": "moved@example.com"}).status_code == 403
    assert c.patch("/api/me", json={"email": "moved@example.com",
                                    "current_password": "wrong password"}).status_code == 403
    r = c.patch("/api/me", json={"email": "Taken@Example.com", "current_password": PASSWORD})
    assert r.status_code == 409
    # Same email (any case) needs no password
    assert c.patch("/api/me", json={"email": "MOVER@example.com"}).status_code == 200

    r = c.patch("/api/me", json={"email": "Moved@Example.com", "current_password": PASSWORD})
    assert r.status_code == 200 and r.json()["email"] == "moved@example.com"
    login = TestClient(app).post("/api/auth/login", json={"email": "moved@example.com", "password": PASSWORD})
    assert login.status_code == 200


def test_change_password_signs_out_other_sessions(server):
    c = new_client(server, "pw@example.com")
    other = TestClient(app)
    assert other.post("/api/auth/login", json={"email": "pw@example.com", "password": PASSWORD}).status_code == 200

    new_pw = "a brand new secret"
    assert c.post("/api/me/password", json={"current_password": "wrong", "new_password": new_pw}).status_code == 403
    assert c.post("/api/me/password", json={"current_password": PASSWORD, "new_password": "short"}).status_code == 422
    assert c.post("/api/me/password", json={"current_password": PASSWORD, "new_password": PASSWORD}).status_code == 422

    assert c.post("/api/me/password", json={"current_password": PASSWORD, "new_password": new_pw}).status_code == 204
    assert c.get("/api/me").status_code == 200       # this session got a fresh token
    assert other.get("/api/me").status_code == 401   # the other one is revoked

    anon = TestClient(app)
    assert anon.post("/api/auth/login", json={"email": "pw@example.com", "password": PASSWORD}).status_code == 401
    assert anon.post("/api/auth/login", json={"email": "pw@example.com", "password": new_pw}).status_code == 200


def test_delete_account_removes_everything(server):
    c = new_client(server, "leaving@example.com")
    uid = c.get("/api/me").json()["id"]
    (sid,) = add_subjects(c, "History")
    ck = c.post("/api/checkins", json={"entries": [{"subject_id": sid, "hours": 5, "confidence": 5}]}).json()

    assert c.delete("/api/me").status_code == 204
    assert c.get("/api/me").status_code == 401
    r = TestClient(app).post("/api/auth/login", json={"email": "leaving@example.com", "password": PASSWORD})
    assert r.status_code == 401

    from app.db import SessionLocal
    from app.models import Checkin, CheckinEntry, Plan, Subject
    with SessionLocal() as db:
        assert db.query(Subject).filter_by(user_id=uid).count() == 0
        assert db.query(Checkin).filter_by(user_id=uid).count() == 0
        assert db.query(CheckinEntry).filter_by(checkin_id=ck["id"]).count() == 0
        assert db.query(Plan).filter_by(checkin_id=ck["id"]).count() == 0


# ---------------------------------------------------------------- demo
@needs_model
def test_public_demo_still_works(server):
    c = TestClient(app)
    meta = c.get("/api/meta").json()
    first = next(iter(meta["personas"].values()))
    r = c.post("/api/run", json={"hours": first["hours"], "confidence": first["conf"]})
    assert r.status_code == 200
    assert len(r.json()["subject_meta"]) == len(meta["subjects"])
