"""JSON API + static host for the HTML dashboard.

Run from the project root (the model path is relative to it):
    uvicorn app.api:app --reload
then open http://localhost:8000
"""

import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure the project root is in the Python path for reliable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import PERSONAS, SUBJECT_META, SUBJECTS, HOURS_RANGE, CONF_RANGE, run_pipeline

WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')
N = len(SUBJECTS)

app = FastAPI(title="Smart Learning & Performance Optimizer")


class RunRequest(BaseModel):
    hours: list[float] = Field(min_length=N, max_length=N)
    confidence: list[float] = Field(min_length=N, max_length=N)


def _round(values, digits=2):
    return [round(float(v), digits) for v in values]


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

    risk = res['risk_results']
    alloc = res['allocation_results']
    return {
        "subjects": SUBJECTS,
        "budget": res['budget'],
        "input_hours": _round(req.hours, 1),
        "confidence": _round(res['confidence_array'], 1),
        "predicted": _round(res['predicted_marks']),
        "risk": {
            "priority": _round(risk['priority']),
            "gap": _round(risk['gap']),
            "label": list(risk['label']),
            "explanation": list(risk['explanation']),
        },
        "allocation": {
            "hours": _round(alloc['hours']),
            "fitness": round(float(alloc['fitness']), 3),
            "history": _round(alloc['history'], 3),
            "baselines": {k: round(float(v), 3) for k, v in alloc['baselines'].items()},
        },
        "schedule": res['schedule'],
        "timings": timings,
    }


# Mounted last so /api routes take precedence
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
