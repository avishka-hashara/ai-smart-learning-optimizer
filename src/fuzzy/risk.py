import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ============================================================
# MELANI - SECTION 2
# FUZZY INFERENCE RISK ASSESSOR
# ============================================================


# ------------------------------------------------------------
# 1. FUZZY UNIVERSES
# ------------------------------------------------------------

# Predicted score universe: 0 - 100
predicted_score = ctrl.Antecedent(
    np.arange(0, 100.5, 0.5),
    'predicted_score'
)

# Confidence universe: 0 - 10
confidence_score = ctrl.Antecedent(
    np.arange(0, 10.1, 0.1),
    'confidence'
)

# Priority universe: 0 - 10
priority_score = ctrl.Consequent(
    np.arange(0, 10.1, 0.1),
    'priority'
)

# Use centroid defuzzification
priority_score.defuzzify_method = 'centroid'


# ------------------------------------------------------------
# 2. MEMBERSHIP FUNCTIONS
# ------------------------------------------------------------

# -------------------------
# Predicted Score
# -------------------------

predicted_score['low'] = fuzz.trapmf(
    predicted_score.universe,
    [0, 0, 35, 50]
)

# Tuned from the initial [35, 55, 75]
# to improve continuity around scores 35 and 36.
predicted_score['medium'] = fuzz.trimf(
    predicted_score.universe,
    [40, 55, 75]
)

predicted_score['high'] = fuzz.trapmf(
    predicted_score.universe,
    [60, 75, 100, 100]
)


# -------------------------
# Confidence
# -------------------------

confidence_score['low'] = fuzz.trapmf(
    confidence_score.universe,
    [0, 0, 3, 5]
)

confidence_score['medium'] = fuzz.trimf(
    confidence_score.universe,
    [3, 5.5, 8]
)

confidence_score['high'] = fuzz.trapmf(
    confidence_score.universe,
    [6, 8, 10, 10]
)


# -------------------------
# Priority
# -------------------------

priority_score['very_low'] = fuzz.trimf(
    priority_score.universe,
    [0, 0, 2.5]
)

priority_score['low'] = fuzz.trimf(
    priority_score.universe,
    [0, 2.5, 5]
)

priority_score['medium'] = fuzz.trimf(
    priority_score.universe,
    [2.5, 5, 7.5]
)

priority_score['high'] = fuzz.trimf(
    priority_score.universe,
    [5, 7.5, 10]
)

priority_score['very_high'] = fuzz.trimf(
    priority_score.universe,
    [7.5, 10, 10]
)


# ------------------------------------------------------------
# 3. NINE MAMDANI FUZZY RULES
# ------------------------------------------------------------

# R1:
# Low predicted score + High confidence
# Failing but thinks they are safe.
rule1 = ctrl.Rule(
    predicted_score['low'] & confidence_score['high'],
    priority_score['very_high']
)

# R2:
# Low predicted score + Medium confidence
# Weak but still not concerned enough.
rule2 = ctrl.Rule(
    predicted_score['low'] & confidence_score['medium'],
    priority_score['very_high']
)

# R3:
# Low predicted score + Low confidence
# Weak and knows it, but still requires high study priority.
rule3 = ctrl.Rule(
    predicted_score['low'] & confidence_score['low'],
    priority_score['high']
)

# R4:
# Medium predicted score + High confidence
# Borderline but overconfident.
rule4 = ctrl.Rule(
    predicted_score['medium'] & confidence_score['high'],
    priority_score['high']
)

# R5:
# Medium predicted score + Medium confidence
# Neutral centre.
rule5 = ctrl.Rule(
    predicted_score['medium'] & confidence_score['medium'],
    priority_score['medium']
)

# R6:
# Medium predicted score + Low confidence
# Borderline and worried.
rule6 = ctrl.Rule(
    predicted_score['medium'] & confidence_score['low'],
    priority_score['medium']
)

# R7:
# High predicted score + High confidence
# Strong and accurately confident.
rule7 = ctrl.Rule(
    predicted_score['high'] & confidence_score['high'],
    priority_score['very_low']
)

# R8:
# High predicted score + Medium confidence
# Strong but mildly uncertain.
rule8 = ctrl.Rule(
    predicted_score['high'] & confidence_score['medium'],
    priority_score['low']
)

# R9:
# High predicted score + Low confidence
# Strong but anxious.
rule9 = ctrl.Rule(
    predicted_score['high'] & confidence_score['low'],
    priority_score['low']
)


# ------------------------------------------------------------
# 4. BUILD FUZZY CONTROL SYSTEM
# ------------------------------------------------------------

# Build the control system once.
# A fresh simulation will be created for every subject.
risk_control = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5,
    rule6,
    rule7,
    rule8,
    rule9
])


# ------------------------------------------------------------
# 5. CONFIDENCE GAP AND LABEL
# ------------------------------------------------------------

def _derive_label(predicted, confidence, threshold=20):
    """
    Determine whether the student's confidence is aligned
    with their predicted performance.

    Gap formula:
        gap = (confidence * 10) - predicted score

    Labels:
        gap > 20   -> overconfident
        gap < -20  -> anxious
        otherwise  -> aligned
    """

    gap = (confidence * 10.0) - predicted

    if gap > threshold:
        return 'overconfident', gap

    if gap < -threshold:
        return 'anxious', gap

    return 'aligned', gap


# ------------------------------------------------------------
# 6. HUMAN-READABLE EXPLANATION
# ------------------------------------------------------------

def _build_explanation(predicted, confidence, priority, label):
    """
    Generate a short explanation for the student.
    """

    if label == 'overconfident':
        return (
            f"You rated your confidence {confidence:.1f} out of 10, "
            f"but your predicted score is around {predicted:.1f}. "
            f"Your confidence is higher than your predicted performance, "
            f"so this subject receives stronger study attention. "
            f"Priority is {priority:.2f} out of 10."
        )

    if label == 'anxious':
        return (
            f"You rated your confidence {confidence:.1f} out of 10, "
            f"but your predicted score is around {predicted:.1f}. "
            f"You appear stronger than you think, so some study time "
            f"may be redirected to subjects that need it more. "
            f"Priority is {priority:.2f} out of 10."
        )

    # Aligned but academically weak
    if predicted < 50:
        return (
            f"Your predicted score is around {predicted:.1f} and your "
            f"confidence is {confidence:.1f} out of 10. "
            f"Your self-assessment is reasonably aligned, but this "
            f"subject still requires significant study attention. "
            f"Priority is {priority:.2f} out of 10."
        )

    # Aligned and academically strong
    if predicted >= 75:
        return (
            f"Your predicted score is around {predicted:.1f} and your "
            f"confidence is {confidence:.1f} out of 10. "
            f"Your self-assessment is reasonably aligned and this "
            f"subject is currently relatively secure. "
            f"Priority is {priority:.2f} out of 10."
        )

    # Aligned middle-performance case
    return (
        f"Your predicted score is around {predicted:.1f} and your "
        f"confidence is {confidence:.1f} out of 10. "
        f"Your self-assessment is reasonably aligned with your "
        f"current performance. "
        f"Priority is {priority:.2f} out of 10."
    )


# ------------------------------------------------------------
# 7. MAIN SECTION 2 INTEGRATION FUNCTION
# ------------------------------------------------------------

def assess_risk(predicted, confidence):
    """
    Melani - Section 2
    Mamdani Fuzzy Inference Risk Assessor.

    Parameters
    ----------
    predicted : array-like
        Predicted subject scores received from Section 1.
        Expected range: 0 to 100.

    confidence : array-like
        Student confidence ratings.
        Expected range: 1 to 10.

    Returns
    -------
    dict

        priority:
            NumPy array containing fuzzy study-priority
            values from 0 to 10.

        label:
            List containing:
            'overconfident', 'aligned', or 'anxious'.

        gap:
            NumPy array containing confidence-performance
            gap values.

        explanation:
            List containing a readable explanation
            for each subject.
    """

    # Convert inputs to NumPy arrays.
    predicted = np.asarray(predicted, dtype=float)
    confidence = np.asarray(confidence, dtype=float)

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if predicted.ndim != 1 or confidence.ndim != 1:
        raise ValueError(
            "Section 2 Error: predicted and confidence must "
            "be one-dimensional arrays."
        )

    # --------------------------------------------------------
    # Validate lengths
    # --------------------------------------------------------

    if len(predicted) != len(confidence):
        raise ValueError(
            "Section 2 Error: predicted and confidence must "
            "contain the same number of subjects."
        )

    priorities = []
    labels = []
    gaps = []
    explanations = []

    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------

    for p, c in zip(predicted, confidence):

        # Clamp inputs to the valid fuzzy universes.
        p = float(np.clip(p, 0, 100))
        c = float(np.clip(c, 0, 10))

        # Create a fresh simulation for this subject.
        simulation = ctrl.ControlSystemSimulation(
            risk_control
        )

        # Supply fuzzy inputs.
        simulation.input['predicted_score'] = p
        simulation.input['confidence'] = c

        # Run Mamdani fuzzy inference.
        simulation.compute()

        # Centroid-defuzzified priority value.
        priority = float(
            simulation.output['priority']
        )

        # Determine diagnostic label and gap.
        label, gap = _derive_label(
            predicted=p,
            confidence=c
        )

        # Create readable explanation.
        explanation = _build_explanation(
            predicted=p,
            confidence=c,
            priority=priority,
            label=label
        )

        # Store results.
        priorities.append(priority)
        labels.append(label)
        gaps.append(gap)
        explanations.append(explanation)

    # --------------------------------------------------------
    # RETURN SECTION 2 OUTPUT CONTRACT
    # --------------------------------------------------------

    return {
        'priority': np.asarray(
            priorities,
            dtype=float
        ),

        'label': labels,

        'gap': np.asarray(
            gaps,
            dtype=float
        ),

        'explanation': explanations
    }