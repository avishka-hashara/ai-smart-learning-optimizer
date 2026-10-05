import numpy as np
from src.fuzzy.risk import assess_risk


# ============================================================
# MELANI - SECTION 2
# FIVE-SUBJECT SANITY TEST
# ============================================================


def run_five_subject_test():

    subjects = [
        "Databases",
        "Networks",
        "Statistics",
        "Maths",
        "Ethics"
    ]

    # Example Stage 1 predicted scores
    predicted = np.array([
        32.0,
        48.0,
        58.0,
        82.0,
        90.0
    ])

    # Student confidence ratings
    confidence = np.array([
        8.0,
        5.0,
        4.0,
        3.0,
        9.0
    ])

    result = assess_risk(
        predicted,
        confidence
    )

    print("\n==========================================================")
    print("MELANI - SECTION 2 FIVE-SUBJECT SANITY TEST")
    print("==========================================================\n")

    for i, subject in enumerate(subjects):

        print(f"Subject    : {subject}")
        print(f"Predicted  : {predicted[i]:.1f}")
        print(f"Confidence : {confidence[i]:.1f}")
        print(f"Priority   : {result['priority'][i]:.3f}")
        print(f"Gap        : {result['gap'][i]:.1f}")
        print(f"Label      : {result['label'][i]}")
        print(f"Explanation: {result['explanation'][i]}")

        print("-" * 58)

    # --------------------------------------------------------
    # Basic integration checks
    # --------------------------------------------------------

    assert len(result["priority"]) == 5
    assert len(result["label"]) == 5
    assert len(result["gap"]) == 5
    assert len(result["explanation"]) == 5

    assert np.all(result["priority"] >= 0)
    assert np.all(result["priority"] <= 10)

    valid_labels = {
        "overconfident",
        "aligned",
        "anxious"
    }

    assert all(
        label in valid_labels
        for label in result["label"]
    )

    # Weak + highly confident subject should be urgent.
    assert result["priority"][0] >= 8
    assert result["label"][0] == "overconfident"

    # Strong + high-confidence subject should be low priority.
    assert result["priority"][4] <= 2
    assert result["label"][4] == "aligned"

    print("\nALL FIVE-SUBJECT SANITY CHECKS PASSED")
    print("==========================================================\n")


if __name__ == "__main__":
    run_five_subject_test()