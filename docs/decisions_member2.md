# Melani - Section 2 Decisions Log

## Module
Fuzzy Inference Risk Assessor

## Technique
Mamdani Fuzzy Inference System

## Purpose
The module receives predicted subject scores and student confidence values and produces a study priority, confidence-calibration label, confidence gap, and human-readable explanation.

## Input Universes

### Predicted Score
Range: 0 to 100  
Step: 0.5

### Confidence
Range: 0 to 10  
Step: 0.1

### Priority
Range: 0 to 10  
Step: 0.1

## Defuzzification Method
Centroid defuzzification was selected.

Reason:
Centroid uses the complete aggregated fuzzy output and provides a smooth priority value. This is appropriate because the priority is passed to the Genetic Algorithm as a weight.

## Predicted Score Membership Functions

### Low
Trapezoidal: [0, 0, 35, 50]

### Medium
Triangular: [40, 55, 75]

Initial handbook value:
[35, 55, 75]

Reason for tuning:
The original starting point produced a priority difference of 0.699 when the predicted score changed from 35 to 36 in test case C10.

The medium-set starting point was changed from 35 to 40. This maintained overlap between the Low and Medium sets while improving continuity around the lower-score boundary.

After tuning:

Priority at predicted score 35 = 9.139  
Priority at predicted score 36 = 9.139  
Difference = 0.000

Therefore, C10 passed.

### High
Trapezoidal: [60, 75, 100, 100]

## Confidence Membership Functions

### Low
Trapezoidal: [0, 0, 3, 5]

### Medium
Triangular: [3, 5.5, 8]

### High
Trapezoidal: [6, 8, 10, 10]

## Priority Membership Functions

### Very Low
Triangular: [0, 0, 2.5]

### Low
Triangular: [0, 2.5, 5]

### Medium
Triangular: [2.5, 5, 7.5]

### High
Triangular: [5, 7.5, 10]

### Very High
Triangular: [7.5, 10, 10]

## Rule Base

Nine Mamdani rules were implemented to cover every Low, Medium, and High combination of predicted score and confidence.

R1: Low predicted + High confidence -> Very High priority  
R2: Low predicted + Medium confidence -> Very High priority  
R3: Low predicted + Low confidence -> High priority  
R4: Medium predicted + High confidence -> High priority  
R5: Medium predicted + Medium confidence -> Medium priority  
R6: Medium predicted + Low confidence -> Medium priority  
R7: High predicted + High confidence -> Very Low priority  
R8: High predicted + Medium confidence -> Low priority  
R9: High predicted + Low confidence -> Low priority

## Risk Label Method

The diagnostic label is calculated separately from fuzzy priority.

Formula:

gap = (confidence x 10) - predicted score

Classification:

gap > +20 -> overconfident  
gap < -20 -> anxious  
otherwise -> aligned

Reason:
Priority measures the value of spending the next study hour on a subject, while the risk label measures whether the student's confidence is aligned with predicted performance. These are different concepts and should not be treated as the same output.

## Threshold Sensitivity Analysis

Thresholds tested:

±15  
±20  
±25

Results:

Changing from ±20 to ±15 changed 4 of 13 labels.

Changing from ±20 to ±25 changed 0 of 13 labels.

Interpretation:
The ±15 threshold is more sensitive and classifies borderline gaps of ±20 as overconfident or anxious.

The ±20 threshold avoids treating these borderline cases as severe miscalibration.

The ±25 threshold produced no label changes compared with ±20 for the selected test set.

Therefore, ±20 was retained as the final threshold.

## Testing

All 12 fuzzy-system acceptance tests passed.

C1-C9:
Expected corner labels and priority bands passed.

C10:
Predicted score continuity passed.

C11:
Confidence continuity passed.

C12:
Priority monotonicity passed.

## Generated Figures

The following report figures were generated:

- fuzzy_predicted_membership.png
- fuzzy_confidence_membership.png
- fuzzy_priority_membership.png
- fuzzy_control_surface.png

Location:

docs/figures/

## Final Status

The Section 2 fuzzy inference module satisfies the required input/output contract and all 12 acceptance tests.