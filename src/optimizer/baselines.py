import numpy as np
from src.optimizer.repair import repair
from src.optimizer.fitness import fitness_function

def equal_split(n, budget, min_h, max_h):
    """
    Divides total budget equally across all subjects, repaired to bounds.
    """
    raw = np.full(n, budget / n, dtype=float)
    return repair(raw, budget, min_h, max_h)

def proportional_split(priority, budget, min_h, max_h):
    """
    Divides total budget in direct proportion to fuzzy priority, repaired to bounds.
    """
    prio = np.asarray(priority, dtype=float)
    total_prio = np.sum(prio)
    if total_prio <= 0:
        raw = np.full(len(prio), budget / len(prio), dtype=float)
    else:
        raw = budget * (prio / total_prio)
    return repair(raw, budget, min_h, max_h)

def greedy_split(priority, predicted, budget, min_h, max_h, tau=7.0, step=0.5):
    """
    Greedily allocates hours in small increments to whichever subject currently
    yields the highest marginal fitness improvement, respecting bounds.
    """
    n = len(priority)
    min_b = np.asarray(min_h, dtype=float)
    max_b = np.asarray(max_h, dtype=float)
    prio = np.asarray(priority, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    # Start with minimum required hours
    alloc = min_b.copy()
    remaining = budget - np.sum(alloc)

    while remaining >= 1e-4:
        current_step = min(step, remaining)
        best_subj = -1
        best_gain = -1e9

        for i in range(n):
            if alloc[i] + current_step <= max_b[i] + 1e-4:
                # Marginal gain = difference in fitness
                candidate = alloc.copy()
                candidate[i] += current_step
                gain = (
                    fitness_function(candidate, prio, pred, min_b, max_b, tau=tau) -
                    fitness_function(alloc, prio, pred, min_b, max_b, tau=tau)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_subj = i

        if best_subj == -1:
            # All modules at max bounds
            break

        alloc[best_subj] += current_step
        remaining -= current_step

    return repair(alloc, budget, min_h, max_h)

def compute_baselines(priority, predicted, budget, min_h, max_h, tau=7.0):
    """
    Evaluates equal, proportional, and greedy baselines against the fitness function.
    Returns dictionary with allocations and fitness scores.
    """
    n = len(priority)
    eq_alloc = equal_split(n, budget, min_h, max_h)
    prop_alloc = proportional_split(priority, budget, min_h, max_h)
    grd_alloc = greedy_split(priority, predicted, budget, min_h, max_h, tau=tau)

    eq_fit = float(fitness_function(eq_alloc, priority, predicted, min_h, max_h, tau=tau))
    prop_fit = float(fitness_function(prop_alloc, priority, predicted, min_h, max_h, tau=tau))
    grd_fit = float(fitness_function(grd_alloc, priority, predicted, min_h, max_h, tau=tau))

    return {
        'equal': eq_fit,
        'proportional': prop_fit,
        'greedy': grd_fit,
        '_allocations': {
            'equal': eq_alloc,
            'proportional': prop_alloc,
            'greedy': grd_alloc
        }
    }
