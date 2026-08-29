import numpy as np


# ============================================================
# MELANI - SECTION 2
# RISK LABEL THRESHOLD SENSITIVITY TEST
# ============================================================


def derive_label(predicted, confidence, threshold):
    gap = (confidence * 10.0) - predicted

    if gap > threshold:
        return "overconfident"

    if gap < -threshold:
        return "anxious"

    return "aligned"


def run_sensitivity_test():

    test_cases = [
        ("C1", 20, 9),
        ("C2", 20, 2),
        ("C3", 85, 9),
        ("C4", 85, 2),
        ("C5", 55, 5),
        ("C6", 50, 5),
        ("C7", 45, 7.5),
        ("C8", 0, 10),
        ("C9", 100, 1),
        ("S1", 50, 7),
        ("S2", 50, 3),
        ("S3", 60, 8),
        ("S4", 60, 4),
    ]

    thresholds = [15, 20, 25]

    print("\n======================================================")
    print("MELANI - FUZZY LABEL THRESHOLD SENSITIVITY TEST")
    print("======================================================\n")

    results = {}

    for name, predicted, confidence in test_cases:

        gap = (confidence * 10.0) - predicted

        labels = {}

        for threshold in thresholds:
            labels[threshold] = derive_label(
                predicted,
                confidence,
                threshold
            )

        results[name] = labels

        print(
            f"{name}: "
            f"Predicted={predicted:>3}, "
            f"Confidence={confidence:>4}, "
            f"Gap={gap:>6.1f} | "
            f"±15={labels[15]:<13} "
            f"±20={labels[20]:<13} "
            f"±25={labels[25]}"
        )

    print("\n------------------------------------------------------")

    baseline = 20

    for threshold in [15, 25]:

        changed = sum(
            results[name][threshold] != results[name][baseline]
            for name in results
        )

        print(
            f"Labels changed when threshold moved "
            f"from ±20 to ±{threshold}: "
            f"{changed}/{len(test_cases)}"
        )

    print("======================================================\n")


if __name__ == "__main__":
    run_sensitivity_test()