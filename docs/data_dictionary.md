# Stage 1: Data Dictionary and Leakage Audit

## Dataset Transformation Decision
The original dataset (`Education_numerical.csv`) contains one row per student. Stage 1 requires one row per subject for the downstream pipeline. We will reshape the dataset using Pandas `melt` so that `math_score`, `science_score`, `english_score`, and `history_score` become a single `subject` column and a `target_score` column. 

**Target Column:** `target_score` (0-100). This represents the final mark a student achieves in a specific subject.

## Leakage Audit and Data Dictionary
*Rule applied: If a value would not be knowable one week before the exam, or if it is derived from the target, it must be removed.*

| Column | Type | Range | Missing | Candidate/Target? | Leakage Audit Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `student_id` | Integer | 0 - 9999 | 0 | Neither | Safe. Will be dropped before training. |
| `math_score` | Float | 30 - 100 | 0 | Target | Becomes `target_score` after reshaping. |
| `science_score` | Float | 40 - 100 | 0 | Target | Becomes `target_score` after reshaping. |
| `english_score` | Float | 50 - 100 | 0 | Target | Becomes `target_score` after reshaping. |
| `history_score` | Float | 35 - 100 | 0 | Target | Becomes `target_score` after reshaping. |
| `study_hours_week` | Float | 5 - 39 | 0 | Feature | Safe. |
| `quiz_attempts` | Integer | 0 - 16 | 0 | Feature | Safe. |
| `consecutive_logins`| Integer | 1 - 25 | 0 | Feature | Safe. |
| `material_access_freq`| Integer | 0 - 47 | 0 | Feature | Safe. |
| `improvement_rate` | Float | 0.5 - 2.5 | 0 | Feature | Safe. Proxy for recent quiz trends. |
| `score_consistency` | Float | 0.0 - 0.9 | 0 | Feature | Safe. |
| `time_per_question` | Float | 0.1 - 21 | 0 | Feature | Safe. |
| `class_size` | Integer | 20 - 59 | 0 | Feature | Safe. Contextual variable. |
| `resource_index` | Float | 10 - 100 | 0 | Feature | Safe. |
| `faculty_ratio` | Float | 0.05 - 0.20 | 0 | Feature | Safe. |
| `peak_study_time` | Integer | 0 - 22 | 0 | Feature | Safe. |
| `weekend_study_ratio`| Float | 0.0 - 1.0 | 0 | Feature | Safe. |
| `knowledge_gap` | Float | 5 - 60 | 0 | Feature | Safe. Proxy for difficulty indicator. |
| `predicted_score` | Float | 44 - 100 | 0 | Neither | **MUST REMOVE.** Leaked target variable. |
| `overall_avg` | Float | 50 - 89 | 0 | Neither | **MUST REMOVE.** Contains the target. |
| `science_math_diff` | Float | -54 - 63 | 0 | Neither | **MUST REMOVE.** Contains the target. |
| `eng_hist_ratio` | Float | 0.5 - 2.6 | 0 | Neither | **MUST REMOVE.** Contains the target. |