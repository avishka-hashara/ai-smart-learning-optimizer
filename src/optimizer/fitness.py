import math
import numpy as np

DEFAULT_TAU = 7.0
DEFAULT_PENALTY_LAMBDA = 20.0  # Significant quadratic penalty for constraint violations

def expected_score_gain(hours, tau=DEFAULT_TAU):
    """
    Diminishing-returns model for study efficiency:
    Gain flattens out as hours increase (concave function).
    At hours = tau, ~63.2% of the potential gain is captured.
    """
    if hours <= 0:
        return 0.0
    return 1.0 - math.exp(-hours / tau)

def fitness_function(hours_allocated, priority, predicted, min_h, max_h, tau=DEFAULT_TAU, penalty_lambda=DEFAULT_PENALTY_LAMBDA):
    """
    Computes total fitness according to Implementation Handbook Section 3.4.1 & 3.4.3:
    Value_i = priority_i * headroom_i * (1 - exp(-h_i / tau)) - penalties

    Parameters
    ----------
    hours_allocated : array-like of shape (n,)
        Study hours assigned to each subject.
    priority : array-like of shape (n,)
        Fuzzy priority from Stage 2 (0 to 10 scale).
    predicted : array-like of shape (n,)
        Predicted mark from Stage 1 (0 to 100 scale).
    min_h : array-like of shape (n,)
        Minimum allowed hours per subject.
    max_h : array-like of shape (n,)
        Maximum allowed hours per subject.
    tau : float
        Saturation time constant in hours.
    penalty_lambda : float
        Quadratic penalty scaling factor for bound violations.

    Returns
    -------
    float : scalar fitness value to maximize.
    """
    hours = np.asarray(hours_allocated, dtype=float)
    prio = np.asarray(priority, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    min_bounds = np.asarray(min_h, dtype=float)
    max_bounds = np.asarray(max_h, dtype=float)

    # Headroom is the marks still available to be gained
    headroom = np.maximum(0.0, 100.0 - pred)

    total_value = 0.0
    penalty = 0.0

    for h, p, hr, mn, mx in zip(hours, prio, headroom, min_bounds, max_bounds):
        # Academic score improvement contribution
        gain = 1.0 - math.exp(-h / tau) if h > 0 else 0.0
        total_value += p * hr * gain

        # Penalty terms for violating lower or upper hour limits
        if h < mn:
            penalty += penalty_lambda * ((mn - h) ** 2)
        elif h > mx:
            penalty += penalty_lambda * ((h - mx) ** 2)

    return total_value - penalty
