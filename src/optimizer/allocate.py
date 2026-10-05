import numpy as np
from src.optimizer.ga import run_ga
from src.optimizer.baselines import compute_baselines
from src.optimizer.repair import check_feasibility

def allocate_hours(priority, predicted, budget, min_h, max_h, seed=42):
    """
    Stage 3: Genetic Algorithm Hour Allocation (Frozen Interface - Table 0.2).

    Parameters
    ----------
    priority : array-like of shape (n,)
        Fuzzy priority ratings from Stage 2 (0 to 10).
    predicted : array-like of shape (n,)
        Predicted marks from Stage 1 (0 to 100).
    budget : float
        Total weekly revision budget in hours.
    min_h : float or array-like of shape (n,)
        Minimum study hours per subject.
    max_h : float or array-like of shape (n,)
        Maximum study hours per subject.
    seed : int, optional (default 42)
        Random seed for reproducibility.

    Returns
    -------
    dict
        'hours': np.ndarray of shape (n,)
            Non-negative floats summing to budget within 0.01.
        'fitness': float
            Best achieved fitness score.
        'history': list of float
            Best fitness achieved at each generation.
        'baselines': dict
            Fitness scores for {'equal': ..., 'proportional': ..., 'greedy': ...}.
    """
    prio = np.asarray(priority, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    n = len(prio)

    if len(pred) != n:
        raise ValueError(
            f"Stage 3 Error: Length mismatch between priority ({n}) and predicted ({len(pred)})."
        )

    # Convert scalar min_h / max_h to length-n arrays if needed
    if np.isscalar(min_h):
        min_bounds = np.full(n, float(min_h))
    else:
        min_bounds = np.asarray(min_h, dtype=float)

    if np.isscalar(max_h):
        max_bounds = np.full(n, float(max_h))
    else:
        max_bounds = np.asarray(max_h, dtype=float)

    # Check feasibility early
    check_feasibility(budget, min_bounds, max_bounds)

    # Evaluate heuristic baselines
    baselines_info = compute_baselines(prio, pred, budget, min_bounds, max_bounds)
    baseline_scores = {
        'equal': baselines_info['equal'],
        'proportional': baselines_info['proportional'],
        'greedy': baselines_info['greedy']
    }

    # Run Genetic Algorithm
    best_hours, best_fitness, history = run_ga(
        priority=prio,
        predicted=pred,
        budget=budget,
        min_h=min_bounds,
        max_h=max_bounds,
        pop_size=80,
        generations=100,
        cxpb=0.8,
        mutpb=0.2,
        seed=seed
    )

    # Final sanity checks on constraints
    sum_hours = float(np.sum(best_hours))
    if abs(sum_hours - budget) > 0.01:
        # Scale one last time to satisfy exact 0.01 tolerance
        best_hours = best_hours * (budget / sum_hours)

    return {
        'hours': np.array(best_hours, dtype=float),
        'fitness': float(best_fitness),
        'history': history,
        'baselines': baseline_scores
    }