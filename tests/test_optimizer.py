import numpy as np
from src.optimizer.allocate import allocate_hours
from src.optimizer.fitness import fitness_function
from src.optimizer.repair import check_feasibility

def run_optimizer_tests():
    print("\n==========================================")
    print("MEMBER 3 - OPTIMIZER & GA ACCEPTANCE TESTS")
    print("==========================================\n")

    failures = []

    priority = np.array([8.5, 5.2, 6.2, 2.1, 5.8])
    predicted = np.array([41.8, 49.4, 45.6, 78.3, 46.9])
    budget = 40.0
    min_h = np.full(5, 2.0)
    max_h = np.full(5, 15.0)

    # Test 1: Standard Run and Contract Shape
    res = allocate_hours(priority, predicted, budget, min_h, max_h, seed=42)
    hours = res['hours']
    fitness = res['fitness']
    history = res['history']
    baselines = res['baselines']

    print(f"T1: Standard Run - Hours: {np.round(hours, 2)}")
    print(f"    Fitness: {fitness:.2f}, Baselines: {baselines}")

    # Check 1: Sum constraint within 0.01
    sum_diff = abs(np.sum(hours) - budget)
    if sum_diff > 0.01:
        failures.append(f"T1 FAILED: Sum discrepancy {sum_diff:.4f} > 0.01")
    else:
        print("T1.1: Sum constraint PASSED (|sum - budget| <= 0.01)")

    # Check 2: Bounds compliance
    if np.any(hours < min_h - 1e-4) or np.any(hours > max_h + 1e-4):
        failures.append("T1 FAILED: Hours violate min_h or max_h bounds")
    else:
        print("T1.2: Bounds compliance PASSED (all min_h <= hours <= max_h)")

    # Check 3: Beats Equal Split
    if fitness < baselines['equal'] - 1e-4:
        failures.append(f"T1 FAILED: GA fitness ({fitness:.2f}) does not beat equal baseline ({baselines['equal']:.2f})")
    else:
        print(f"T1.3: Beats Equal Split PASSED ({fitness:.2f} >= {baselines['equal']:.2f})")

    # Check 4: Beats Proportional Split
    if fitness < baselines['proportional'] - 1e-4:
        failures.append(f"T1 FAILED: GA fitness ({fitness:.2f}) does not beat proportional baseline ({baselines['proportional']:.2f})")
    else:
        print(f"T1.4: Beats Proportional Split PASSED ({fitness:.2f} >= {baselines['proportional']:.2f})")

    # Check 5: History convergence monotonicity (best fitness never drops)
    decreases = [history[i] - history[i-1] for i in range(1, len(history)) if history[i] < history[i-1] - 1e-4]
    if len(decreases) > 0:
        failures.append(f"T1 FAILED: Elitism violation, fitness dropped {len(decreases)} times")
    else:
        print("T1.5: Convergence Monotonicity PASSED (Elitism preserved)")

    # Test 2: Infeasible Budget Error Handling
    print("\nT2: Infeasible Case Testing...")
    try:
        allocate_hours(priority, predicted, budget=5.0, min_h=min_h, max_h=max_h)
        failures.append("T2 FAILED: Expected ValueError when sum(min_h) > budget")
    except ValueError as e:
        print(f"T2.1: Min-bound infeasibility correctly raised: {e}")

    try:
        allocate_hours(priority, predicted, budget=100.0, min_h=min_h, max_h=max_h)
        failures.append("T2 FAILED: Expected ValueError when sum(max_h) < budget")
    except ValueError as e:
        print(f"T2.2: Max-bound infeasibility correctly raised: {e}")

    # Test 3: Deterministic Reproducibility
    print("\nT3: Determinism Test...")
    res_repeat = allocate_hours(priority, predicted, budget, min_h, max_h, seed=42)
    if not np.allclose(res['hours'], res_repeat['hours'], atol=1e-4):
        failures.append("T3 FAILED: Re-running with identical seed produced different allocations")
    else:
        print("T3: Deterministic Reproducibility PASSED")

    # Test 4: Monotonicity with Priority
    print("\nT4: Monotonicity with Priority...")
    # Two identical subjects with different priorities
    p_mono = np.array([9.0, 3.0])
    pred_mono = np.array([50.0, 50.0])
    res_mono = allocate_hours(p_mono, pred_mono, budget=20.0, min_h=2.0, max_h=15.0, seed=42)
    if res_mono['hours'][0] < res_mono['hours'][1]:
        failures.append("T4 FAILED: Higher priority subject received fewer hours")
    else:
        print(f"T4: Priority Monotonicity PASSED (H_high={res_mono['hours'][0]:.2f}h >= H_low={res_mono['hours'][1]:.2f}h)")

    print("\n==========================================")
    if len(failures) == 0:
        print("ALL STAGE 3 OPTIMIZER TESTS PASSED!")
    else:
        print(f"{len(failures)} FAILURE(S) DETECTED:")
        for f in failures:
            print(f"- {f}")
    print("==========================================\n")
    return len(failures) == 0

if __name__ == "__main__":
    success = run_optimizer_tests()
    if not success:
        exit(1)
