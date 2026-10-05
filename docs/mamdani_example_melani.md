# Melani - Section 2
## Hand-Worked Mamdani Fuzzy Inference Example

### Example Scenario

Consider one subject with:

Predicted Score = 45  
Confidence = 7.5 / 10

The purpose of this example is to demonstrate how the fuzzy inference
system calculates the study priority.

---

## Step 1: Fuzzification

The predicted score of 45 belongs to the overlapping Low and Medium
predicted-score membership functions.

Predicted-score membership functions:

Low = trapmf [0, 0, 35, 50]

Medium = trimf [40, 55, 75]

High = trapmf [60, 75, 100, 100]

The confidence value of 7.5 belongs to the overlapping Medium and High
confidence membership functions.

Confidence membership functions:

Low = trapmf [0, 0, 3, 5]

Medium = trimf [3, 5.5, 8]

High = trapmf [6, 8, 10, 10]

---

## Step 2: Relevant Fuzzy Rules

For this input, the important activated rules are:

R2:
IF Predicted Score is Low
AND Confidence is Medium
THEN Priority is Very High

R4:
IF Predicted Score is Medium
AND Confidence is High
THEN Priority is High

Other overlapping rules may also contribute depending on the exact
membership degrees.

---

## Step 3: Rule Activation

For each rule, the AND operation is evaluated using the minimum
membership degree.

Example:

Rule Strength = min(
    Predicted Membership,
    Confidence Membership
)

The resulting firing strengths are used to clip the corresponding
priority membership functions.

---

## Step 4: Aggregation

The output membership functions produced by all activated rules are
combined into one aggregated fuzzy priority output.

The Mamdani inference system therefore considers all applicable rules
instead of selecting only one rule.

---

## Step 5: Defuzzification

Centroid defuzzification is used to convert the aggregated fuzzy output
into one numerical study-priority value.

For:

Predicted Score = 45
Confidence = 7.5

The implemented fuzzy system produces approximately:

Priority = 6.789 / 10

---

## Step 6: Confidence-Performance Gap

The diagnostic gap is calculated separately:

Gap = (Confidence × 10) - Predicted Score

Gap = (7.5 × 10) - 45

Gap = 75 - 45

Gap = +30

Since:

Gap > +20

the student is classified as:

Overconfident

---

## Final Result

Predicted Score: 45

Confidence: 7.5 / 10

Fuzzy Study Priority: approximately 6.789 / 10

Confidence Gap: +30

Risk Label: Overconfident

Interpretation:

The student has a relatively weak predicted performance but reports
high confidence. Therefore, the fuzzy system assigns substantial study
priority to the subject and identifies the student as overconfident.