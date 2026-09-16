import numpy as np

def check_feasibility(budget, min_h, max_h):
    """
    Validates whether a feasible hour allocation exists for the given budget and bounds.
    Handbook Section 3.3.4.
    """
    min_sum = float(np.sum(min_h))
    max_sum = float(np.sum(max_h))

    if min_sum > budget + 1e-5:
        raise ValueError(
            f"Stage 3 Error: Infeasible budget. Sum of minimum hours ({min_sum:.2f}h) "
            f"exceeds total budget ({budget:.2f}h)."
        )
    if max_sum < budget - 1e-5:
        raise ValueError(
            f"Stage 3 Error: Infeasible budget. Sum of maximum hours ({max_sum:.2f}h) "
            f"is less than total budget ({budget:.2f}h)."
        )

def repair(hours, budget, min_h, max_h, max_iter=5):
    """
    Repair operator for candidate chromosomes (Handbook Section 3.3.4):
    Ensures that hours strictly sum to budget and conform to [min_h, max_h] bounds.
    Uses an alternating bounded loop (at most 5 iterations) to prevent infinite oscillation.
    """
    check_feasibility(budget, min_h, max_h)

    arr = np.asarray(hours, dtype=float).copy()
    min_b = np.asarray(min_h, dtype=float)
    max_b = np.asarray(max_h, dtype=float)

    # Replace negative or NaN values with lower bounds
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.maximum(min_b, arr)

    for _ in range(max_iter):
        total = np.sum(arr)
        if abs(total - budget) < 1e-4:
            break

        if total <= 0:
            arr = np.full_like(arr, budget / len(arr))
            break

        # Scale toward budget
        scale = budget / total
        arr = arr * scale

        # Clip into bounds
        arr = np.clip(arr, min_b, max_b)

    # Residual adjustment: distribute any small remaining discrepancy
    residual = budget - np.sum(arr)
    if abs(residual) > 1e-4:
        # Find genes not pegged strictly at their bound
        if residual > 0:
            room = max_b - arr
            unpegged = room > 1e-4
            if np.any(unpegged):
                share = residual / np.sum(room[unpegged])
                arr[unpegged] += room[unpegged] * min(share, 1.0)
            else:
                arr[np.argmax(max_b - arr)] += residual
        else:
            room = arr - min_b
            unpegged = room > 1e-4
            if np.any(unpegged):
                share = (-residual) / np.sum(room[unpegged])
                arr[unpegged] -= room[unpegged] * min(share, 1.0)
            else:
                arr[np.argmax(arr - min_b)] += residual

    return arr
