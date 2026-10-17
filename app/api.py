"""JSON API + static host for the HTML dashboard.

Run from the project root (the model path is relative to it):
    uvicorn app.api:app --reload
then open http://localhost:8000

Database migrations run on startup (set AUTO_MIGRATE=0 to manage them
yourself with `alembic upgrade head`).

/api/meta, /api/run and /api/lms-sync are the public demo (personas, no
account). Signed-in users go through /api/auth, /api/subjects and
/api/checkins instead, which save every week's check-in and plan.
"""

import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure the project root is in the Python path for reliable module imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

from app import auth, study
from app.config import PERSONAS, SUBJECT_META, SUBJECTS, HOURS_RANGE, CONF_RANGE, run_pipeline, to_payload

WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')
N = len(SUBJECTS)


def migrate():
    from alembic import command
    from alembic.config import Config

    cfg = Config(os.path.join(ROOT, 'alembic.ini'))
    cfg.set_main_option('script_location', os.path.join(ROOT, 'migrations'))
    cfg.attributes['skip_logging_config'] = True
    command.upgrade(cfg, 'head')


@asynccontextmanager
async def lifespan(_app):
    if os.getenv('AUTO_MIGRATE', '1') == '1':
        migrate()
    yield


app = FastAPI(title="Smart Learning & Performance Optimizer", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(study.router)


class RunRequest(BaseModel):
    hours: list[float] = Field(min_length=N, max_length=N)
    confidence: list[float] = Field(min_length=N, max_length=N)


@app.get("/api/meta")
def meta():
    return {
        "personas": PERSONAS,
        "subjects": SUBJECT_META,
        "hours_range": HOURS_RANGE,
        "conf_range": CONF_RANGE,
        "lms_logins": np.random.randint(5, 16, size=N).tolist(),
    }


@app.post("/api/lms-sync")
def lms_sync():
    # Simulated LMS round-trip, mirrors the Streamlit sync button
    time.sleep(0.4)
    return {"lms_logins": np.random.randint(6, 18, size=N).tolist()}


@app.post("/api/run")
def run(req: RunRequest):
    lo, hi = HOURS_RANGE
    if any(not lo <= h <= hi for h in req.hours):
        raise HTTPException(400, f"Hours must be between {lo} and {hi}.")
    lo, hi = CONF_RANGE
    if any(not lo <= c <= hi for c in req.confidence):
        raise HTTPException(400, f"Confidence must be between {lo} and {hi}.")

    timings = {}
    started = time.perf_counter()
    try:
        res = run_pipeline(req.hours, req.confidence)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    timings['total_ms'] = round((time.perf_counter() - started) * 1000)

    payload = to_payload(res, req.hours, SUBJECT_META)
    payload["timings"] = timings
    return payload


# Mounted last so /api routes take precedence
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
