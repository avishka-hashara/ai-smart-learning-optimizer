"""Shared dashboard configuration and pipeline runner used by both the
Streamlit app (app/main.py) and the HTML dashboard API (app/api.py)."""

import numpy as np
import pandas as pd

from src.prediction.predict import predict_scores
from src.fuzzy.risk import assess_risk
from src.optimizer.allocate import allocate_hours
from src.scheduler.schedule import build_schedule

# -----------------------------------------------------------------------------
# STUDENT PERSONAS & SUBJECT CONFIGURATION
# -----------------------------------------------------------------------------
PERSONAS = {
    "The Overconfident Student": {
        "description": "High self-rated confidence despite borderline predicted performance in technical subjects.",
        "avatar": "🦁",
        "conf": [9.0, 5.0, 6.0, 7.0, 5.0],
        "hours": [12.0, 15.0, 8.0, 20.0, 5.0]
    },
    "The Anxious High Achiever": {
        "description": "Consistently strong performance telemetry despite severe under-confidence in core subjects.",
        "avatar": "🦉",
        "conf": [3.0, 8.0, 7.0, 8.0, 9.0],
        "hours": [25.0, 20.0, 22.0, 30.0, 18.0]
    },
    "The Uniformly Weak Student": {
        "description": "Critical low attendance and engagement requiring urgent structured study reallocations.",
        "avatar": "🌱",
        "conf": [2.0, 3.0, 2.0, 4.0, 2.0],
        "hours": [5.0, 4.0, 6.0, 5.0, 3.0]
    }
}

SUBJECT_META = [
    {"name": "Databases", "icon": "💾", "color": "#38bdf8", "accent": "rgba(56, 189, 248, 0.4)"},
    {"name": "Networks", "icon": "🌐", "color": "#818cf8", "accent": "rgba(129, 140, 248, 0.4)"},
    {"name": "Statistics", "icon": "📊", "color": "#fb7185", "accent": "rgba(251, 113, 133, 0.4)"},
    {"name": "Maths", "icon": "📐", "color": "#facc15", "accent": "rgba(250, 204, 21, 0.4)"},
    {"name": "Ethics", "icon": "⚖️", "color": "#4ade80", "accent": "rgba(74, 222, 128, 0.4)"},
]
SUBJECTS = [s["name"] for s in SUBJECT_META]

# Input bounds exposed to the UI
HOURS_RANGE = (1.0, 40.0)
CONF_RANGE = (1.0, 10.0)


def build_features(hours):
    """Stage 1 feature frame; non-hour features use fixed demo defaults."""
    n = len(SUBJECTS)
    return pd.DataFrame({
        'subject': SUBJECTS,
        'study_hours_week': list(hours),
        'quiz_attempts': [5] * n,
        'consecutive_logins': [10] * n,
        'material_access_freq': [20] * n,
        'improvement_rate': [1.2] * n,
        'score_consistency': [0.6] * n,
        'time_per_question': [1.5] * n,
        'class_size': [30] * n,
        'resource_index': [50.0] * n,
        'faculty_ratio': [0.1] * n,
        'peak_study_time': [14] * n,
        'weekend_study_ratio': [0.5] * n,
        'difficulty_indicator': [2] * n
    })


def hour_bounds(budget, n):
    """Per-subject GA bounds that always admit a feasible allocation.

    Defaults are 2h min / 15h max, widened when the budget would otherwise
    fall outside [n * min, n * max] (e.g. a 115h budget with 5 subjects).
    """
    per_subject = budget / n
    min_h = np.full(n, min(2.0, per_subject))
    max_h = np.full(n, max(15.0, per_subject * 1.5))
    return min_h, max_h


def run_pipeline(hours, confidence):
    """Runs all four stages and returns raw (numpy) results."""
    hours = [float(h) for h in hours]
    confidence_array = np.asarray(confidence, dtype=float)

    predicted_marks = predict_scores(build_features(hours))
    risk_results = assess_risk(predicted_marks, confidence_array)

    budget = float(sum(hours))
    min_h, max_h = hour_bounds(budget, len(SUBJECTS))
    allocation_results = allocate_hours(
        priority=risk_results['priority'],
        predicted=predicted_marks,
        budget=budget,
        min_h=min_h,
        max_h=max_h,
        seed=42
    )

    schedule = build_schedule(
        hours=allocation_results['hours'],
        subjects=SUBJECTS,
        days=7,
        daily_limits=8.0,
        peak_window="09:00-11:00"
    )

    return {
        'predicted_marks': predicted_marks,
        'confidence_array': confidence_array,
        'risk_results': risk_results,
        'allocation_results': allocation_results,
        'schedule': schedule,
        'budget': budget
    }
