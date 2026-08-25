import numpy as np
from src.fuzzy.risk import assess_risk


def run_case(case, predicted, confidence):
    result = assess_risk(
        np.array([predicted], dtype=float),
        np.array([confidence], dtype=float)
    )

    priority = float(result["priority"][0])
    label = result["label"][0]
    gap = float(result["gap"][0])

    print(
        f"{case}: Predicted={predicted}, "
        f"Confidence={confidence}, "
        f"Priority={priority:.3f}, "
        f"Label={label}, "
        f"Gap={gap:.1f}"
    )

    return priority, label


def run_corner_tests():

    print("\n==========================================")
    print("MEMBER 2 - FUZZY INFERENCE TEST RESULTS")
    print("==========================================\n")

    failures = []

    # -------------------------------------------------
    # C1
    # Low predicted + high confidence
    # -------------------------------------------------
    p, label = run_case("C1", 20, 9)

    if label != "overconfident":
        failures.append("C1 label should be overconfident")

    if p < 8:
        failures.append("C1 priority should be >= 8")


    # -------------------------------------------------
    # C2
    # Low predicted + low confidence
    # -------------------------------------------------
    p, label = run_case("C2", 20, 2)

    if label != "aligned":
        failures.append("C2 label should be aligned")

    if not (6 <= p <= 8.5):
        failures.append("C2 priority should be between 6 and 8.5")


    # -------------------------------------------------
    # C3
    # High predicted + high confidence
    # -------------------------------------------------
    p, label = run_case("C3", 85, 9)

    if label != "aligned":
        failures.append("C3 label should be aligned")

    if p > 2:
        failures.append("C3 priority should be <= 2")


    # -------------------------------------------------
    # C4
    # High predicted + low confidence
    # -------------------------------------------------
    p, label = run_case("C4", 85, 2)

    if label != "anxious":
        failures.append("C4 label should be anxious")

    if p > 3.5:
        failures.append("C4 priority should be <= 3.5")


    # -------------------------------------------------
    # C5
    # Neutral centre
    # -------------------------------------------------
    p, label = run_case("C5", 55, 5)

    if label != "aligned":
        failures.append("C5 label should be aligned")

    if not (4 <= p <= 6):
        failures.append("C5 priority should be between 4 and 6")


    # -------------------------------------------------
    # C6
    # Awkward middle
    # -------------------------------------------------
    p, label = run_case("C6", 50, 5)

    if label != "aligned":
        failures.append("C6 label should be aligned")

    if not (4 <= p <= 6):
        failures.append("C6 priority should be medium (4 to 6)")


    # -------------------------------------------------
    # C7
    # Partial membership case
    # -------------------------------------------------
    p, label = run_case("C7", 45, 7.5)

    if label != "overconfident":
        failures.append("C7 label should be overconfident")

    if p < 6:
        failures.append(
            "C7 priority should be high to very high"
        )


    # -------------------------------------------------
    # C8
    # Extreme low score + maximum confidence
    # -------------------------------------------------
    p, label = run_case("C8", 0, 10)

    if label != "overconfident":
        failures.append("C8 label should be overconfident")


    # -------------------------------------------------
    # C9
    # Extreme high score + minimum confidence
    # -------------------------------------------------
    p, label = run_case("C9", 100, 1)

    if label != "anxious":
        failures.append("C9 label should be anxious")


    # -------------------------------------------------
    # C10
    # Continuity when predicted score moves 35 -> 36
    # Handbook requirement: difference < 0.3
    # -------------------------------------------------
    result_35 = assess_risk([35], [5])
    result_36 = assess_risk([36], [5])

    p35 = float(result_35["priority"][0])
    p36 = float(result_36["priority"][0])

    label35 = result_35["label"][0]
    label36 = result_36["label"][0]

    difference = abs(p35 - p36)

    print(
        f"C10: Priority(35)={p35:.3f}, "
        f"Priority(36)={p36:.3f}, "
        f"Difference={difference:.3f}"
    )

    if label35 != label36:
        failures.append("C10 labels should remain the same")

    if difference >= 0.3:
        failures.append(
            f"C10 FAILED: difference {difference:.3f} must be < 0.3"
        )


    # -------------------------------------------------
    # C11
    # Confidence continuity 4.9 -> 5.1
    # Handbook requirement: difference < 0.2
    # -------------------------------------------------
    result_49 = assess_risk([60], [4.9])
    result_51 = assess_risk([60], [5.1])

    p49 = float(result_49["priority"][0])
    p51 = float(result_51["priority"][0])

    difference = abs(p49 - p51)

    print(
        f"C11: Priority(4.9)={p49:.3f}, "
        f"Priority(5.1)={p51:.3f}, "
        f"Difference={difference:.3f}"
    )

    if difference >= 0.2:
        failures.append(
            f"C11 FAILED: difference {difference:.3f} must be < 0.2"
        )


    # -------------------------------------------------
    # C12
    # Priority must never rise as predicted score rises
    # Confidence fixed at 5
    # -------------------------------------------------
    scores = np.arange(0, 101, 2, dtype=float)

    result = assess_risk(
        scores,
        np.full(len(scores), 5.0)
    )

    priorities = result["priority"]

    increases = np.where(
        np.diff(priorities) > 1e-6
    )[0]

    if len(increases) == 0:
        print("C12: Monotonicity PASSED")
    else:
        failures.append(
            f"C12 FAILED: priority increased "
            f"{len(increases)} time(s)"
        )


    # -------------------------------------------------
    # Final result
    # -------------------------------------------------
    print("\n==========================================")

    if len(failures) == 0:

        print("ALL 12 FUZZY TESTS PASSED")

    else:

        print(f"{len(failures)} TEST ISSUE(S) FOUND:\n")

        for failure in failures:
            print(f"- {failure}")

    print("==========================================\n")

    return failures


if __name__ == "__main__":
    run_corner_tests()