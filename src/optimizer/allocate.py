import numpy as np

def allocate_hours(priority, predicted, budget, min_h, max_h):
    """Stage 3 Stub: Simply divides the budget equally among subjects."""
    n = len(priority)
    hours = np.full(n, budget / n)
    return {
        'hours': hours,
        'fitness': 100.0,
        'history': [50.0, 75.0, 100.0],
        'baselines': {'equal': 80.0, 'proportional': 90.0, 'greedy': 85.0}
    }