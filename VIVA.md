# VIVA PREPARATION — Avishka Hashara (Group 4)

> Smart Learning and Performance Optimization System — a hybrid AI pipeline (ANN → Fuzzy → GA → Rules)
>
> My areas (from my Individual Contribution Sheet):
> 1. Data preparation & leakage audit
> 2. Stage 1 — neural-network prediction engine
> 3. Model evaluation & baselines (incl. PyTorch version)
> 4. System integration (4-stage pipeline, smoke test, error propagation)
> 5. Backend, user interface & persistence (FastAPI, auth, SQLite/Alembic, web UI, Streamlit, tests), final report & presentation script
>
> How to use this file: read Part A tonight, Parts B–F in depth, then drill Part H (Q&A) out loud. Part G is your demo. Part I is the 10-minute cheat sheet for just before the viva.

---

## Table of contents

- **Part A** — Orientation: pitch, team, numbers to memorise
- **Part B** — Data preparation & leakage audit (`src/prediction/prepare_data.py`)
- **Part C** — Neural network theory you must be able to explain
- **Part D** — Stage 1 code walkthrough (`train_mlp.py`, `predict.py`)
- **Part E** — Evaluation, baselines, results, PyTorch (`baselines.py`, `pytorch_compare.py`)
- **Part F** — The other three stages (enough to defend integration)
- **Part G** — Integration (`app/config.py`, `smoke_test.py`, error propagation)
- **Part H** — Backend, database, auth, UI (`app/`)
- **Part I** — Demo script
- **Part J** — Limitations, ethics, known discrepancies (TRAPS)
- **Part K** — Viva question bank with answers
- **Part L** — Glossary
- **Part M** — Last-10-minutes cheat sheet

---

# PART A — Orientation

## A.1 The 30-second pitch

> "Students split revision time by feeling, not evidence. Our system fixes that with four AI stages. **I built Stage 1**: I audited the dataset, removed four columns that leaked the target, reshaped 10,000 students into 40,000 student–subject rows and trained an MLP regressor that predicts each subject's mark with a test MAE of 8.18 (target was under 10). **I also integrated the whole pipeline** — neural network → fuzzy risk → genetic algorithm → scheduler — through frozen interfaces, and built the FastAPI backend with user accounts, a SQLite database with Alembic migrations, and the web dashboard."

## A.2 The problem in one paragraph

Students are badly calibrated about their own performance (Dunning–Kruger: weak students overrate themselves; anxious strong students underrate themselves). Both mistakes waste revision time. The system: (1) **predicts** the mark per subject, (2) **compares** that with the student's confidence to flag overconfidence/anxiety and compute a study priority, (3) **optimises** how the weekly hours are split, (4) **schedules** the hours into a timetable.

## A.3 Team and ownership (know who did what — examiners ask)

| Stage / part | Owner | Evidence |
|---|---|---|
| Stage 1 — data prep, MLP, baselines, PyTorch | **Me (Member 1)** | `src/prediction/`, `docs/data_dictionary.md`, `docs/decisions_member1.md` |
| Stage 2 — Mamdani fuzzy risk | Melani | `src/fuzzy/risk.py`, `docs/decisions_member2.md` |
| Stage 3 — Genetic algorithm | Janithi | `src/optimizer/`, `genetic_algorithm/` (prototype) |
| Stage 4 — Rule-based scheduler | Member 4 (GitHub: kspanchami611) | `src/scheduler/schedule.py` |
| Integration, `app/config.py`, smoke test | **Me** | `smoke_test.py`, integration commits on `dev` |
| FastAPI backend, auth, DB, migrations, web UI, Streamlit, API tests | **Me** | `app/`, `migrations/`, `tests/test_api.py` |
| Final report, presentation script | **Me** | `REPORT.md`, `docs/presend.md` |

I also merged the teammates' pull requests (#3 fuzzy, #4 GA, #7 scheduler) and wrote the integration commits ("Integrate GA optimizer with full prediction and fuzzy pipeline", "Add Streamlit dashboard…", "Add FastAPI backend…", "Generalize study pipeline with user accounts and persistence").

## A.4 Numbers to memorise (the "must not hesitate" list)

| Fact | Value |
|---|---|
| Raw dataset | 10,000 students × 22 columns, no missing values |
| Leaking columns removed | **4**: `predicted_score`, `overall_avg`, `science_math_diff`, `eng_hist_ratio` |
| After reshape | **40,000** rows (10,000 × 4 subjects) |
| Train / test | 80/20 → **32,000 / 8,000**, `random_state=42` |
| Inputs to the network | **17** = 13 numeric + 4 one-hot subject columns |
| Architecture | **17 → 64 → 32 → 1**, ReLU hidden, linear output |
| Trainable parameters | **3,265** |
| Optimiser | Adam, lr 0.001, L2 α = 0.0001, batch 200 |
| Early stopping | 10 % validation, patience 10 → **stopped at epoch 47** |
| **MLP test MAE** | **8.18** (RMSE 10.34, R² 0.103) — target < 10 ✔ |
| Linear regression | MAE 8.13 (RMSE 10.28, R² 0.114) — tied with MLP |
| Global mean baseline | MAE 8.67, R² 0.00 |
| Student-average baseline | MAE **10.71**, R² −0.52 — MLP beats it ✔ |
| PyTorch MLP | MAE 8.88, R² −0.06 |
| Train vs test MAE | 8.175 vs 8.177 → no overfitting |
| Within 10 marks | 67.7 % of test predictions |
| Permutation importance | `subject` 1.102; next `improvement_rate` 0.009, `score_consistency` 0.008, `material_access_freq` 0.007 |
| Subject means | English 74.4, Science 69.6, History 67.4, Maths 64.5 |
| Synthetic confidence | 11.8 % overconfident, 16.4 % anxious (|gap| > 20) |
| Error propagation | ±10-mark error moves ≤ **~7 %** of the hours, **0** labels change |
| Seed | 42 everywhere |
| Tests now | `pytest tests` → **17 passed** (API) + smoke test passes all 3 personas |

---

# PART B — Data preparation & leakage audit

File: [src/prediction/prepare_data.py](src/prediction/prepare_data.py) · Output: `data/cleaned_features.csv` · Docs: [docs/data_dictionary.md](docs/data_dictionary.md)

## B.1 The dataset (`Education_numerical.csv`)

22 columns:
- `student_id`
- 4 **targets**: `math_score`, `science_score`, `english_score`, `history_score` (0–100)
- 13 behavioural/contextual features: `study_hours_week` (5–39), `quiz_attempts` (0–16), `consecutive_logins` (1–25), `material_access_freq` (0–47), `improvement_rate` (0.5–2.5), `score_consistency` (0–0.9), `time_per_question` (0.1–21), `class_size` (20–59), `resource_index` (10–100), `faculty_ratio` (0.05–0.20), `peak_study_time` (0–22, hour of day), `weekend_study_ratio` (0–1), `knowledge_gap` (5–60)
- 4 **derived columns** (the leak): `predicted_score`, `overall_avg`, `science_math_diff`, `eng_hist_ratio`

## B.2 Step 1 — Leakage audit

**What is data leakage?** When training data contains information that would not be available at prediction time — usually information derived from the answer itself. The model learns to "read the answer" and gets a fake, excellent test score that collapses in real use.

**The rule I applied to every column:**
> *"Would this value be knowable one week before the exam, and is it free of the target?"*

| Column dropped | Why it leaks |
|---|---|
| `predicted_score` | An existing prediction of the score — effectively the answer |
| `overall_avg` | Mean of the 4 subject scores → contains the target directly |
| `science_math_diff` | Science − Maths → arithmetic on two targets |
| `eng_hist_ratio` | English / History → arithmetic on two targets |

Example of why it matters: if `overall_avg` and the other three scores are known, the fourth score can be solved **exactly** (4 × avg − sum of the other three). Keeping it would give near-zero error and a meaningless model.

Code:
```python
leaked_cols = ['predicted_score', 'overall_avg', 'science_math_diff', 'eng_hist_ratio']
df = df.drop(columns=leaked_cols)
```

## B.3 Step 2 — Reshape with `pd.melt` (wide → long)

Before (1 row per student):
```
student_id | math_score | science_score | english_score | history_score | study_hours_week | ...
0          | 70         | 63            | 77            | 46            | 7                | ...
```
After (1 row per student-subject):
```
student_id | subject | target_score | study_hours_week | ...
0          | Maths   | 70           | 7                | ...
0          | Science | 63           | 7                | ...
0          | English | 77           | 7                | ...
0          | History | 46           | 7                | ...
```

```python
melted_df = pd.melt(df, id_vars=feature_cols, value_vars=target_cols,
                    var_name='subject', value_name='target_score')
```
- `id_vars` = columns copied onto every new row (the features)
- `value_vars` = the columns being "unpivoted" (the four scores)
- Then `subject_map` renames `math_score` → `Maths`, etc.

**Why reshape?** Stage 2 (fuzzy) needs **one predicted mark per subject**. So the *unit of prediction* is (student, subject). One model for all subjects, with `subject` as an input feature — instead of 4 separate models. Advantages: 4× more training rows per model, and one artifact to maintain.

Note: the behavioural features are **per student**, not per subject — so the four rows of the same student have identical features and differ only in `subject`. This is why `subject` dominates the model (Part E).

## B.4 Step 3 — Difficulty indicator from `knowledge_gap`

```python
melted_df['difficulty_indicator'] = pd.qcut(melted_df['knowledge_gap'], q=3, labels=[1, 2, 3]).astype(int)
melted_df = melted_df.drop(columns=['knowledge_gap'])
```
- `pd.qcut` = **quantile cut** = equal-frequency bins (each bin holds ~⅓ of rows). Different from `pd.cut`, which makes equal-width bins.
- Result: 1 = low, 2 = medium, 3 = high difficulty.
- **Why?** Matches the agreed data contract (handbook says 1/2/3), easy for a user to set ("how hard is this subject for you?"), and in the app a user picks difficulty 1–3 per subject — which feeds directly into this feature.
- **Known weakness:** bins were computed on the full data before the train/test split → a minor leak of the bin *boundaries* (not the target). Admit it if asked.

## B.5 Step 4 — Synthetic confidence column

The dataset has **no confidence rating**, but Stage 2 needs one. So I simulated it:

```
confidence = clip( round( target_score/10 + student_bias + noise ), 1, 10 )
student_bias ~ Normal(0, 2.0)   ← drawn ONCE per student
noise        ~ Normal(0, 0.5)   ← drawn per row
```

```python
np.random.seed(42)
student_bias = {sid: np.random.normal(loc=0, scale=2.0) for sid in unique_students}
def generate_confidence(row):
    base_conf = row['target_score'] / 10.0
    bias = student_bias[row['student_id']]
    noise = np.random.normal(loc=0, scale=0.5)
    return np.clip(np.round(base_conf + bias + noise), 1, 10)
```

Design reasoning:
- `target_score/10` puts confidence on the same scale as the mark (70 → 7) → a calibrated student rates ≈ mark/10.
- **Per-student bias** makes miscalibration *persistent*: an overconfident person is overconfident in every subject — realistic (it's a personality trait), and gives Stage 2 meaningful cases.
- Per-row noise adds small realistic variation.
- Clip 1–10 = the slider range in the UI. Round = people answer in whole numbers.
- Result printed by the script: **11.8 % overconfident, 16.4 % anxious** (gap = 10·confidence − mark, threshold ±20).

**Critical point — confidence is NOT a model feature.** In `train_mlp.py`: `X = df.drop(columns=['target_score', 'confidence'])`. Two reasons:
1. It was **generated from the target** → using it would be leakage.
2. By design it's an **input to Stage 2**, which compares it against the prediction. If the model used confidence, the comparison would be circular.

**Honesty point:** the miscalibration rates describe *my generator*, not real students. Stated explicitly in the report (abstract, methods, ethics).

## B.6 Data dictionary

[docs/data_dictionary.md](docs/data_dictionary.md) lists every column: type, range, missing count, role (feature/target/neither) and leakage verdict. Mention it as evidence that the audit was column-by-column, not ad hoc.

---

# PART C — Neural network theory (explain from first principles)

## C.1 What an MLP is

A **Multi-Layer Perceptron** is a feed-forward neural network: layers of neurons, each fully connected to the next.

One neuron: `output = φ(w₁x₁ + w₂x₂ + … + wₙxₙ + b)`
One layer (vector form): `a = φ(W·x + b)`
- `W` = weight matrix (learned), `b` = bias vector (learned), `φ` = activation function.

Our network:
```
input x (17)
  → hidden 1: a₁ = ReLU(W₁x + b₁)       W₁: 17×64
  → hidden 2: a₂ = ReLU(W₂a₁ + b₂)      W₂: 64×32
  → output:   ŷ  = W₃a₂ + b₃            W₃: 32×1   (no activation = linear, because regression)
```

**Parameter count (they love this question):**
- Layer 1: 17 × 64 + 64 = 1,152
- Layer 2: 64 × 32 + 32 = 2,080
- Output: 32 × 1 + 1 = 33
- **Total = 3,265**

Verified from the saved model: `coefs_` shapes are (17, 64), (64, 32), (32, 1).

**Why 64 → 32 (a "funnel")?** Wide first layer to learn many feature combinations, narrower second layer to compress them. Big enough to learn interactions, small enough (3,265 params vs 28,800 training rows) not to overfit. It was the handbook-recommended default; we did not run a hyperparameter search (admit it — future work).

## C.2 Activation: ReLU

`ReLU(z) = max(0, z)`
- **Non-linear** → stacking layers can model curved relationships (without a non-linearity, any number of layers collapses to one linear layer).
- **Cheap** to compute.
- **Avoids vanishing gradients**: for z > 0 the gradient is exactly 1, unlike sigmoid/tanh whose gradients shrink towards 0 at the extremes, which stalls learning in deep nets.
- Downside: "dying ReLU" (a neuron stuck at 0) — not a problem at this size.
- **Output is linear (identity)** because we predict a continuous unbounded number. A sigmoid output would squash to 0–1.

## C.3 Loss function

scikit-learn's `MLPRegressor` minimises **squared error** (half MSE) + L2 penalty:
```
Loss = (1/2n) Σ (yᵢ − ŷᵢ)²  +  (α/2n) Σ w²
```
- MSE punishes big errors more than small ones.
- **L2 penalty (weight decay)**, α = 0.0001 (default): discourages large weights → mild regularisation against overfitting.

## C.4 How it learns: backpropagation + gradient descent

1. **Forward pass**: compute ŷ for a batch.
2. **Loss**: compare ŷ with y.
3. **Backward pass (backpropagation)**: use the **chain rule** to compute ∂Loss/∂w for every weight, layer by layer from output back to input.
4. **Update**: move each weight a small step against its gradient: `w ← w − lr · ∂Loss/∂w`.
5. Repeat for every mini-batch; one pass over all training data = **one epoch**.

## C.5 Adam optimiser

Adam = **Ada**ptive **M**oment estimation. Keeps, for each weight, a running mean of the gradient (momentum, m) and of the squared gradient (v):
```
m ← β₁m + (1−β₁)g          (β₁ = 0.9)
v ← β₂v + (1−β₂)g²         (β₂ = 0.999)
m̂, v̂ = bias-corrected m, v
w ← w − lr · m̂ / (√v̂ + ε)   (lr = 0.001)
```
In words: **each weight gets its own step size**. Weights with consistently large gradients take smaller steps; rarely-updated weights take larger ones. Momentum smooths noisy gradients. → Works well with little tuning, which is why we chose it over plain SGD.

## C.6 Mini-batch training

- `batch_size='auto'` = min(200, n) = **200 rows per update**.
- Training rows actually used for fitting = 32,000 × 0.9 (10 % held out for early stopping) = **28,800**.
- 28,800 / 200 = **144 updates per epoch**; × 47 epochs ≈ **6,800 weight updates**.
- Remember this — it explains why the PyTorch version (full-batch, 1 update per epoch → only 1,000 updates) is worse.

## C.7 Early stopping

```python
early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=1000
```
- 10 % of the training set is held aside as a **validation set** (not the test set!).
- After each epoch, sklearn computes the **validation R²**.
- If it does not improve by more than `tol` (1e-4) for **10 consecutive epochs** (patience), training stops and the best weights are kept.
- **Stopped at epoch 47.**
- `max_iter=1000` was set **deliberately high** so that early stopping — not an arbitrary cap — decides when to stop.
- Purpose: stop **before the network starts memorising** training rows (overfitting).

Three-way data separation (be precise):
| Set | Size | Used for |
|---|---|---|
| Training (fit) | 28,800 | Updating weights |
| Validation | 3,200 | Deciding when to stop |
| **Test** | 8,000 | Final honest score — used **once** |

## C.8 Overfitting vs underfitting

- **Overfitting**: great on training, poor on test (memorised noise). Signs: train error ≪ test error.
- **Underfitting**: poor on both (model too simple or signal too weak).
- Ours: train MAE 8.175 ≈ test MAE 8.177 → **no overfitting**. The error is limited by the data's weak signal, not by the model memorising.

## C.9 Feature scaling — `StandardScaler`

`z = (x − mean) / std` → every numeric feature gets mean 0, std 1.
- **Why:** features have very different scales (`class_size` 20–59 vs `faculty_ratio` 0.05–0.20). Without scaling, large-unit features dominate the weighted sums and the gradients, and training is slow/unstable.
- **Fitted on the training rows only** (inside the Pipeline), then the same mean/std applied to test rows → no information from the test set leaks into training.

## C.10 Categorical encoding — `OneHotEncoder`

`subject` → 4 binary columns (categories learned alphabetically: English, History, Maths, Science):
```
Maths   → [0, 0, 1, 0]
English → [1, 0, 0, 0]
```
- **Why not integer codes (Maths=1, Science=2…)?** That implies a false order and distance (Science is "twice" Maths?). Subject is **nominal**.
- `handle_unknown='ignore'`: an unseen subject (e.g. "Databases") becomes **[0, 0, 0, 0]** instead of crashing. Added during integration because the app uses subjects not in the training data. (And `predict.py` now handles that case better — see D.3.)
- The linear-regression baseline uses `drop='first'` (3 columns) to avoid the **dummy-variable trap** (the 4 columns always sum to 1, which is perfectly collinear with the intercept). Neural nets don't need this.

## C.11 Why `student_id` disappears

`student_id` is still in `X`, but it is not listed in either transformer of the `ColumnTransformer`. The default `remainder='drop'` throws it away. So the model never sees the ID (correct — an ID has no meaning and would let the model memorise students).

## C.12 Why regression and not classification

- Marks are continuous. A pass/fail classifier treats a predicted 38 and 12 the same, but those students need very different study time.
- Stage 2 needs a 0–100 number for its fuzzy universe, and Stage 3 needs a **smooth** input (headroom = 100 − ŷ) so the optimisation landscape is not jagged.

## C.13 Why an MLP at all (and alternatives)

- Can model **non-linear interactions** automatically (e.g. extra hours may help more if quiz attempts are also high) without hand-engineering interaction terms.
- Module focus is neural networks.
- **Alternatives considered:** gradient-boosted trees (strong on tabular data, but produce step functions — not smooth — and less aligned with module), linear regression (trained as a control).
- **Honest finding:** the non-linearity did not help on this dataset (MLP ≈ linear regression). Reported as a finding, not hidden.

---

# PART D — Stage 1 code walkthrough

## D.1 `train_mlp.py` — step by step

[src/prediction/train_mlp.py](src/prediction/train_mlp.py)

```python
df = pd.read_csv("data/cleaned_features.csv")
X = df.drop(columns=['target_score', 'confidence'])   # confidence excluded on purpose
y = df['target_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 80/20 split, seed 42 → reproducible.

```python
numeric_features = X_train.drop(columns=['student_id', 'subject']).columns.tolist()   # 13 columns
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['subject'])
])
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                   max_iter=1000, early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=10, random_state=42)
pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', mlp)])
pipeline.fit(X_train, y_train)
```
- **`ColumnTransformer`** applies different preprocessing to different columns.
- **`Pipeline`** chains preprocessing + model into ONE object. Benefits:
  1. `fit` fits the scaler on training data only → no test leakage.
  2. `predict` automatically applies the exact same transformation.
  3. One object to save and load → no "forgot to scale at inference" bugs.

```python
y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index)
mae = mean_absolute_error(y_test, y_pred); rmse = sqrt(MSE); r2 = r2_score(...)
```
- Wrapped in a Series with the test index so the error analysis can look up the subject of each worst prediction.

Then:
- **Error analysis**: 20 largest absolute errors printed.
- **Three figures** saved to `docs/figures/`:
  - `loss_curve.png` — training loss per epoch + validation R² (shows convergence and the early-stop point)
  - `pred_vs_actual.png` — scatter with diagonal y = x; ours is a flat cloud (predictions squeezed into 60–80) → weak signal
  - `residuals.png` — histogram of (actual − predicted); centred near 0 and roughly symmetric → no systematic bias
- **Persist**: `joblib.dump(pipeline, "models/predictor_v1.joblib")`. `joblib` is efficient for objects containing big NumPy arrays. The `_v1` suffix = model versioning; the backend stores this version name with each saved plan.

## D.2 `predict.py` — the production wrapper (Stage 1's frozen interface)

[src/prediction/predict.py](src/prediction/predict.py)

`predict_scores(features_df) → np.ndarray` of marks in [0, 100]. Five jobs:

1. **Lazy, cached model loading**
   ```python
   _model = None
   def load_model():
       global _model
       if _model is None:
           _model = joblib.load(MODEL_PATH)
       return _model
   ```
   Loaded once on first use, then reused. **Why:** Streamlit reruns the whole script on every slider change → without caching it reloaded the model from disk every time. Also clear `FileNotFoundError` if the model hasn't been trained.

2. **Column validation (fail fast, readable error)** — checks all 14 expected columns (13 numeric + `subject`); raises `ValueError("Stage 1 Error: … missing required columns: [...]")`.

3. **Subject-name normalisation** — reads the trained categories from the fitted encoder and maps case/whitespace variants: `'maths '` → `'Maths'`.

4. **Unseen-subject fallback** (my integration fix)
   ```python
   unseen = ~features_df['subject'].isin(known_subjects)
   if unseen.any():
       per_subject = [model.predict(rows.assign(subject=s)) for s in known_subjects]
       predictions[unseen] = np.mean(per_subject, axis=0)
   ```
   Problem: an unknown subject one-hot encodes to all zeros — a pattern the network **never saw in training** (out of distribution). It predicted ~44–46 regardless of the student. Fix: predict the row **as each of the 4 known subjects and average** → a neutral prior that still responds to the student's other features. Now e.g. Databases ≈ 69.3.

5. **Clip to [0, 100]** — `np.clip(predictions, 0, 100)`. **Why in the wrapper:** it's an *integration contract*. Stage 2's fuzzy universe is 0–100; a value outside it cannot be fuzzified properly. The model itself is left untouched so evaluation reports its raw behaviour.

---

# PART E — Evaluation, baselines, results

## E.1 Metrics — definitions and intuition

| Metric | Formula | Meaning | Ours |
|---|---|---|---|
| **MAE** | (1/n) Σ \|y − ŷ\| | Average miss in marks. Easiest to explain to a student: "off by ~8 marks". Robust to outliers. | 8.18 |
| **RMSE** | √[(1/n) Σ (y − ŷ)²] | Squares errors first → punishes big misses more. Same units as marks. | 10.34 |
| **R²** | 1 − SS_res / SS_tot | Fraction of variance explained. 1 = perfect; **0 = no better than predicting the mean**; negative = worse than the mean. | 0.103 |

- RMSE > MAE always; a big gap means some large errors exist (our worst errors are ~34 marks).
- **Why MAE as the headline?** Interpretable in marks; the acceptance target was set in MAE (< 10).

## E.2 Baselines (`baselines.py`) — why they exist

A model's score means nothing without a reference point. Same split (seed 42), same test set.

| Baseline | How | Purpose |
|---|---|---|
| **Global mean** | Predict the training-set mean mark for everyone | The "zero-skill" floor; defines R² = 0 |
| **Student's own average** | Predict a student's mean from their *other* subjects in the training set (fallback: global mean) | The naive human heuristic: "I'm a 70 student" |
| **Linear regression** | Same preprocessing (`drop='first'`), `LinearRegression()` | Control: does the MLP's non-linearity actually help? |

## E.3 Results table (held-out 8,000 rows)

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| Global mean | 8.668 | 10.922 | 0.000 |
| Student's own average | 10.705 | 13.442 | −0.515 |
| Linear regression | **8.133** | **10.281** | **0.114** |
| **MLP (production)** | 8.177 | 10.342 | 0.103 |
| MLP (PyTorch) | 8.884 | 11.225 | −0.057 |

**Acceptance (objective O1):** MAE 8.18 < 10 ✔ and clearly better than student average (10.71) ✔ → **met**.

## E.4 Honest interpretation — the most important part to say confidently

**"The MLP is tied with linear regression and only slightly better than the mean. That's a property of the data, not a modelling failure."** Evidence:

1. **Subject scores are uncorrelated** with each other (pairwise r between −0.010 and 0.027). Knowing a student's Maths mark tells you nothing about their English mark.
2. → That's exactly why the **student-average baseline is worse than the global mean**: averaging a student's other subjects just adds noise. (Classic viva question!)
3. **Permutation importance**: shuffle one feature in the test set, measure how much MAE rises. `subject` = **1.102**; every behavioural feature ≈ 0.01 or less. The model essentially learned **per-subject means** (English 74.4, Science 69.6, History 67.4, Maths 64.5).
4. **No overfitting**: train 8.175 vs test 8.177.
5. **No non-linear structure** to exploit → MLP converges to roughly the linear solution.
6. **Prediction range is narrow**: all test predictions between 60.2 and 80.0 — the model shrinks towards subject means, which is the correct behaviour when signal is weak (it minimises expected error).
7. Conclusion: the dataset is probably **synthetic** (independent random columns). With real LMS data where behaviour actually drives marks, the same architecture would have signal to learn.

Live check I ran: Maths predicted **65.1 at 5 h/week vs 66.0 at 39 h/week** — study hours barely move the prediction. Same weak-signal story.

## E.5 Error analysis — the 20 worst predictions

- All are **Maths (15) or History (5)** — the two subjects with the widest spread (σ = 12.0 and 11.0).
- True marks at the **extremes** (30–37 floor, or 100 ceiling); predictions all in a 63–71 band; mean error on these 34.0.
- Explanation: **regression to the mean** — with weak features, the model can't identify outliers, so it predicts near the subject mean.

## E.6 Permutation importance — how it works

1. Compute baseline test MAE.
2. Randomly shuffle one column (breaks its relationship with the target, keeps its distribution).
3. Re-compute MAE. Increase = importance.
4. Repeat per feature.
Model-agnostic, uses the test set, measures what the *model* relies on (not causation).

## E.7 PyTorch comparison (`pytorch_compare.py`)

[src/prediction/pytorch_compare.py](src/prediction/pytorch_compare.py)

Same architecture written explicitly:
```python
nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
criterion = nn.MSELoss();  optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()           # 1. clear old gradients (PyTorch accumulates them)
    outputs = model(X_train_tensor) # 2. forward pass
    loss = criterion(outputs, y)    # 3. compute loss
    loss.backward()                 # 4. backpropagation → gradients
    optimizer.step()                # 5. update weights
model.eval(); with torch.no_grad(): predictions = model(X_test)   # inference, no gradient tracking
```
- Uses CUDA if available (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).
- `.view(-1, 1)` reshapes y to a column so it matches the output shape (n, 1).

**Why is it worse (MAE 8.88)?** Not the architecture — the **training regime**:
| | scikit-learn | PyTorch version |
|---|---|---|
| Batching | mini-batch 200 → 144 updates/epoch | **full batch** → 1 update/epoch |
| Total updates | ~6,800 | **1,000** |
| Early stopping | yes | **no** |
| L2 penalty | α = 1e-4 | **none** |
| Loss at end | converged | still falling (MSE ≈ 126) — under-trained |

**Lesson:** the optimisation regime matters as much as the architecture. Purpose of the script: make the training loop explicit (sklearn hides it) and show understanding of what `MLPRegressor` does internally.

How to fix it if asked: use a `DataLoader` with mini-batches (e.g. 200), add early stopping on a validation split, add `weight_decay=1e-4` to Adam.

---

# PART F — The other three stages (know enough to defend integration)

## F.1 Stage 2 — Mamdani fuzzy risk assessor (Melani) — `src/fuzzy/risk.py`

**Interface:** `assess_risk(predicted, confidence) → {priority (n,), label [n], gap (n,), explanation [n]}`

- Inputs: predicted score (0–100), confidence (0–10). Output: priority (0–10).
- Membership functions:
  - Predicted: low trap [0,0,35,50], medium tri **[40,55,75]** (tuned from [35,55,75] to fix a continuity jump), high trap [60,75,100,100]
  - Confidence: low trap [0,0,3,5], medium tri [3,5.5,8], high trap [6,8,10,10]
  - Priority: 5 triangles very_low…very_high peaking at 0, 2.5, 5, 7.5, 10
- **9 rules** (complete 3×3 grid):

| Predicted \ Confidence | low | medium | high |
|---|---|---|---|
| **low** | high (R3) | very high (R2) | **very high (R1)** — failing and unaware |
| **medium** | medium (R6) | medium (R5) | high (R4) |
| **high** | low (R9) | low (R8) | **very low (R7)** |

- Mamdani steps: **fuzzify → rule strength via min (AND) → clip consequents → aggregate via max → defuzzify via centroid**.
- Worked example (pred 45, conf 7.5) → priority **6.789**, gap +30 → overconfident.
- **Label** is crisp: `gap = 10·confidence − predicted`; > +20 overconfident, < −20 anxious, else aligned.
- Priority ≠ gap: a student predicted 20 with confidence 2 is calibrated (gap 0) but still gets priority 7.5 (R3) — honest weak students are not starved.
- **12/12 acceptance tests** pass, incl. continuity (4.9→5.1 confidence changes priority 0.000) and monotonicity. Priority range ≈ [0.83, 9.17] (centroid never reaches the edge).
- **Why fuzzy:** a crisp threshold would make priority jump, and priority is a *weight in the optimiser* → timetable would rearrange from a tiny slider change. **Why Mamdani not Sugeno:** no data to fit Sugeno outputs; Mamdani rules are human-readable → explainable.

## F.2 Stage 3 — Genetic algorithm (Janithi) — `src/optimizer/`

**Interface:** `allocate_hours(priority, predicted, budget, min_h, max_h, seed=42) → {hours, fitness, history, baselines}`

**Fitness (maximise):**
```
F(h) = Σ pᵢ · (100 − ŷᵢ) · (1 − e^(−hᵢ/τ))  −  λ · Σ [(minᵢ − hᵢ)₊² + (hᵢ − maxᵢ)₊²]
        priority   headroom   diminishing returns       quadratic bound penalty
τ = 7 h (≈63 % of gain captured by 7 h), λ = 20
```
- **This is where Stage 1 feeds in**: headroom = 100 − ŷ. A subject predicted 92 can't gain much.
- Concave gain → balanced splits beat extreme ones.
- Constraint: Σh = budget, via a **repair operator** (scale → clip, max 5 loops, then spread residual over unpinned genes). `check_feasibility` raises `ValueError` if Σmin > B or Σmax < B.
- DEAP config: real-valued genes (hours), population 80, 100 generations, tournament k=3, blend crossover α=0.5 p=0.8, Gaussian mutation σ=1 p=0.2, elitism 2 (Hall of Fame), seed 42. Initial population seeded with equal/proportional/greedy.
- Baselines: equal, proportional to priority, greedy (0.5 h steps by marginal gain).
- Results: beats equal & proportional on **30/30 seeds**, SD 0.0002; matches brute force within 0.002; ~0.35 s. Greedy is near-optimal because the objective is separable and concave — GA is never worse and still works if the objective becomes non-separable.

## F.3 Stage 4 — Rule-based scheduler (Member 4) — `src/scheduler/schedule.py`

**Interface:** `build_schedule(hours, subjects, days, daily_limits, peak_window, exam_dates=None) → [ {day, date, start, end, subject, minutes, type, reason} ]`

- Forward chaining: hours → minutes; order by earliest exam if given; place sessions of `min(60, remaining subject, remaining today)` minutes from the peak-window start (09:00); move to next day at the daily limit (8 h); stop when days run out.
- 500 random inputs: 0 violations of daily limit / 60-min sessions / overlaps.
- **Weakness:** if budget > 7 × 8 = 56 h, extra hours are **silently dropped** (both the 60 h and 115 h personas place only 56 h). No breaks, no spreading across days.

---

# PART G — Integration (my core contribution)

## G.1 Architecture — three layers

1. **AI core (`src/`)** — four independent packages; only plain lists/NumPy arrays cross boundaries.
2. **Orchestration (`app/config.py`)** — `run_pipeline()` is the **single entry point** used by both UIs and the API.
3. **Presentation** — Streamlit (`app/main.py`) and FastAPI + HTML/JS (`app/api.py`, `app/web/`).

## G.2 Frozen interfaces (data contract)

| Stage | Function | Receives | Returns |
|---|---|---|---|
| 1 | `predict_scores(features_df)` | DataFrame, one row per subject, unscaled | `ndarray (n,)` 0–100 |
| 2 | `assess_risk(predicted, confidence)` | two `(n,)` arrays | dict: priority, label, gap, explanation |
| 3 | `allocate_hours(priority, predicted, budget, min_h, max_h, seed)` | arrays + budget + bounds | dict: hours (Σ = budget ± 0.01), fitness, history, baselines |
| 4 | `build_schedule(hours, subjects, days, daily_limits, peak_window, exam_dates)` | hours + names + calendar params | list of session dicts |

**Design principles I set:**
- **Frozen contracts** → four people developed in parallel without breaking each other.
- **Stub-first integration**: early on I committed stub versions of all four functions returning correctly shaped fake data, so everyone could run the full pipeline from day one. (That's why `smoke_test.py` still has a stale comment about a "stub".)
- **Canonical subject order**: the row order of the input DataFrame is the ONLY ordering; no stage may sort internally. Stage 3 receives no names — just arrays — so a re-sort anywhere would silently attach hours to the wrong subject.
- **Fail fast, readable errors**: each stage validates its inputs ("Stage 1 Error: missing columns…", "Stage 3 Error: Infeasible budget…", "Stage 4 Error: received 4 hour values for 5 subjects").
- **Reproducibility**: seed 42 for split, network init, synthetic confidence and GA.
- **No cross-package class dependencies** — only plain types.

## G.3 `run_pipeline()` step by step — [app/config.py](app/config.py)

```python
def run_pipeline(hours, confidence, subjects=None, difficulty=None, exam_dates=None):
    # validate: ≥1 subject, equal lengths
    with _pipeline_lock:
        return _run_stages(...)

def _run_stages(subjects, hours, confidence_array, difficulty, exam_dates):
    predicted_marks = predict_scores(build_features(hours, subjects, difficulty))   # Stage 1
    risk_results    = assess_risk(predicted_marks, confidence_array)                # Stage 2
    budget          = float(sum(hours))
    min_h, max_h    = hour_bounds(budget, len(subjects))
    allocation      = allocate_hours(risk_results['priority'], predicted_marks,
                                     budget, min_h, max_h, seed=42)                 # Stage 3
    schedule        = build_schedule(allocation['hours'], subjects, days=7,
                                     daily_limits=8.0, peak_window="09:00-11:00",
                                     exam_dates=exam_dates)                          # Stage 4
    return {...all intermediate results...}
```

Key pieces:
- **`build_features(hours, subjects, difficulty)`** — builds the Stage 1 DataFrame. `study_hours_week` = the user's hours; `difficulty_indicator` = the subject's difficulty (1–3) set by the user; the other 11 behavioural features use **fixed demo defaults** (quiz_attempts 5, consecutive_logins 10, …) standing in for an LMS feed. The "Sync LMS" button is simulated.
- **Budget = sum of current hours** → the system **redistributes** existing effort; it never says "study more in total" (a limitation).
- **`hour_bounds(budget, n)`**: `min = min(2, B/n)`, `max = max(15, 1.5·B/n)`. Default [2 h, 15 h] per subject, widened so a feasible allocation **always** exists (e.g. B = 115, n = 5 → max 34.5).
- **`threading.Lock`**: FastAPI runs sync endpoints in a thread pool, so two requests can run at once. The GA seeds the **global** `random`/`np.random` generators; two concurrent runs would interleave random draws and lose reproducibility. The lock serialises pipeline runs.
- **`to_payload()`** converts NumPy results into rounded JSON for the UI and for storage; embeds subject name/icon/colour so stored plans render even if subjects are later renamed or archived.

## G.4 What broke during integration and how I fixed it (great viva material — shows real engineering)

| Problem | Fix |
|---|---|
| App subjects (Databases, Networks, Statistics, Ethics) weren't in the training data (Maths, Science, English, History). Default `OneHotEncoder` **crashed** on unknown categories. | Retrained with `handle_unknown='ignore'`. Later found the all-zero vector is out-of-distribution (always ~45) → added the **average-over-known-subjects fallback** in `predict.py`. |
| Persona budgets (115 h) were outside 5 × [2, 15] = [10, 75] → `check_feasibility` raised an error. | `hour_bounds()` derives bounds from the budget. |
| Two DEAP `creator` classes with the same name clashed when both the prototype GA (`genetic_algorithm/`) and the production GA were imported. | Renamed to `OptimizerFitnessMax` / `OptimizerIndividual`, guarded with `if not hasattr(creator, …)`. |
| Streamlit re-ran the script on every widget change → model reloaded every time. | Lazy cached loading in `predict.py`. |
| Two front ends could compute different results. | Both call one `run_pipeline()`. |
| Concurrent API requests broke GA reproducibility. | `threading.Lock`. |
| Out-of-range user inputs. | Range validation in FastAPI/Pydantic (HTTP 400/422), clamping in Stage 2, clipping in Stage 1. |

## G.5 Smoke test — [smoke_test.py](smoke_test.py)

Runs 3 profiles end to end with a 40 h budget and bounds [2, 15]:
1. **Overconfident Student** — confidence [9, 5, 6, 7, 5]
2. **Anxious High Achiever** — [3, 8, 7, 8, 9]
3. **Uniformly Weak Student** — [2, 3, 2, 4, 2]

Each prints the output of every stage and checks the hour sum (40.00 / 40.00) and that a schedule was produced. Command: `python smoke_test.py` → "ALL SMOKE TESTS COMPLETED SUCCESSFULLY!"

Current live output (verified): predicted ≈ [69.1, 72.0, 70.1, 78.3, 71.1]; e.g. profile 1 labels: overconfident, anxious, aligned, aligned, anxious; hours [11.35, 7.80, 8.84, 3.70, 8.31].

Note: the smoke test passes its own (non-default) feature values per subject; the dashboards use `build_features()` defaults — that's why their numbers differ.

## G.6 Live dashboard persona numbers (what the screen will show NOW)

| Persona | Budget | Predicted (DB, Net, Stat, Maths, Ethics) | Labels | GA hours | GA vs equal / prop / greedy |
|---|---|---|---|---|---|
| Overconfident | 60 h | 69.3, 69.3, 69.4, 64.8, 69.5 | Databases **overconfident** (gap +20.7), rest aligned | 12.79, 10.82, 10.81, **14.86**, 10.72 | 515.03 vs 511.85 / 514.46 / 514.98 |
| Anxious High Achiever | 115 h | 69.7, 69.5, 69.6, 65.2, 69.5 | Databases **anxious** (−39.7), Ethics overconfident (+20.5), rest aligned | 20.88, 22.86, 22.62, 25.71, 22.93 | 666.98 vs 666.33 / 665.40 / 666.98 |
| Uniformly Weak | 23 h | 69.5, 69.6, 69.5, 65.1, 69.7 | **all anxious** | 4.18, 4.14, 4.21, 6.38, 4.08 | 263.07 vs 260.65 / 262.20 / 262.98 |

How to read it: Maths gets the most hours in every persona because its predicted mark is lowest (most headroom) and its priority is highest. GA ≥ greedy ≥ proportional ≥ equal (approximately) everywhere. The schedule places at most 56 h (7 days × 8 h) — 60 h and 115 h personas lose hours (known limitation).

## G.7 Error-propagation experiment (the #1 architecture question)

**Question:** "If Stage 1 is wrong by 10 marks, what happens to the plan?"

Method: fixed profile ŷ = [38, 55, 62, 70, 48], confidence [8.5, 5, 6, 7, 4], B = 40 h. Baseline allocation [14.03, 7.56, 5.68, 3.38, 9.35]. Shift every prediction by ±5 and ±10 and re-run Stages 2–3.

| Perturbation | Max Δpriority | Max Δhours (1 subject) | Total hours moved | Labels changed |
|---|---|---|---|---|
| −10 | 2.25 | 2.06 h | 2.59 h (6.5 %) | 0/5 |
| −5 | 1.39 | 1.08 h | 1.76 h (4.4 %) | 0/5 |
| +5 | 1.98 | 1.38 h | 1.95 h (4.9 %) | 0/5 |
| +10 | 1.98 | 1.40 h | 2.71 h (6.8 %) | 0/5 |

**Answer:** "A 10-mark error — bigger than our MAE of 8.2 — moves at most about 7 % of the week's hours and changes no labels. Fuzzy membership and the concave fitness both respond smoothly, so the plan **degrades gracefully** rather than failing. The flip side: the plan can't be *better informed* than the prediction allows."
## G.8 Why a hybrid pipeline and not one big model

| Stage | Kind of uncertainty | Technique |
|---|---|---|
| 1 | Statistical — future mark not yet observed | Supervised learning (MLP) |
| 2 | Linguistic — vague self-report "I feel 7/10" | Fuzzy logic (Mamdani) |
| 3 | Combinatorial — ~1.9 M splits of 40 h over 5 subjects at 0.5 h steps | Genetic algorithm |
| 4 | None — known calendar rules | Rule-based expert system |

- A single end-to-end model would need **labelled "optimal timetables"** — they don't exist.
- It would lose interpretability (Stages 2 and 4 explain themselves) and couldn't **guarantee** hard constraints.
- Cost of hybrid: errors compound along the pipeline → measured in G.7.

---

# PART H — Backend, database, authentication, UI

## H.1 Stack

| Layer | Tech | Why |
|---|---|---|
| API | **FastAPI** (Python, ASGI) | Same language as the AI code → call `run_pipeline()` directly; automatic validation with Pydantic; automatic Swagger docs at `/docs` |
| Validation | **Pydantic** models | Declarative request schemas; wrong types/lengths → automatic **422**; my explicit range checks → **400** |
| ORM | **SQLAlchemy 2.0** (`Mapped`, `mapped_column`) | DB-agnostic: SQLite locally, Postgres by changing `DATABASE_URL` |
| DB | **SQLite** (`smartlearn.db`) | Zero setup for a demo |
| Migrations | **Alembic** | Versioned, repeatable schema changes |
| Auth | **bcrypt** + **JWT** (PyJWT, HS256) in an **httpOnly cookie** | Industry-standard password storage and stateless sessions |
| Front end | Vanilla HTML/CSS/JS + **Plotly 2.35.2** (CDN) | No build step; served by FastAPI as static files |
| Alt UI | **Streamlit** (`app/main.py`) + Plotly | Original rapid-prototype dashboard; still works |
| Tests | **pytest** + FastAPI `TestClient` | 17 API tests |

## H.2 `app/api.py` — app setup

[app/api.py](app/api.py)
- `lifespan` handler runs **Alembic migrations on startup** (`AUTO_MIGRATE=1` default) → fresh DB is created automatically.
- `app.include_router(auth.router)` and `study.router` — routes split by concern.
- Public demo routes (no account):
  - `GET /api/meta` — personas, subjects, input ranges, random LMS logins
  - `POST /api/lms-sync` — **simulated** LMS sync (sleeps 0.4 s, returns random login counts)
  - `POST /api/run` — `RunRequest{hours[5], confidence[5]}` with `Field(min_length=N, max_length=N)`; checks hours ∈ [1, 40], confidence ∈ [1, 10] → 400 otherwise; runs the pipeline; returns payload + `timings.total_ms`.
- `app.mount("/", StaticFiles(...))` **last**, so `/api/*` routes win over static files.

## H.3 All routes (know this table)

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /api/auth/register` | — | Create account (201), sets cookie, returns user + token |
| `POST /api/auth/login` | — | Login (200) or generic 401 |
| `POST /api/auth/logout` | — | Clears cookie (204) |
| `GET /api/me` | ✔ | Current user |
| `PATCH /api/me` | ✔ | Update profile (name, email, university, programme, year); changing email needs current password (403) |
| `POST /api/me/password` | ✔ | Change password → **signs out all other sessions** |
| `DELETE /api/me` | ✔ | Delete account + all data (cascade) |
| `GET /api/me/meta` | ✔ | Ranges, max subjects (10), current week |
| `GET/POST /api/subjects` | ✔ | List / create subject (max 10, unique name per user) |
| `PATCH/DELETE /api/subjects/{id}` | ✔ | Edit / delete-or-archive |
| `POST /api/checkins` | ✔ | Submit this week's hours + confidence → runs pipeline → stores plan (201 new, 200 replaced) |
| `GET /api/checkins`, `GET/DELETE /api/checkins/{id}` | ✔ | History / one check-in / delete |
| `GET /api/meta`, `POST /api/run`, `POST /api/lms-sync` | — | Public demo |

## H.4 Authentication in depth — [app/auth.py](app/auth.py)

**Passwords — bcrypt**
- `bcrypt.hashpw(password, gensalt())` — a **salted, deliberately slow** hash. Salt = random per password → identical passwords have different hashes, rainbow tables are useless. Slow → brute force is expensive.
- We never store the password, only the hash. Verification: `bcrypt.checkpw`.
- **72-byte limit**: bcrypt only reads the first 72 bytes, so longer passwords are rejected at registration (otherwise two different long passwords could match).
- Minimum length 8 (Pydantic `Field(min_length=8)`).

**Sessions — JWT**
- JWT = `header.payload.signature`, base64-encoded. Payload: `{"sub": user_id, "pwd": password_stamp, "iat": issued_at, "exp": +7 days}`.
- Signed with **HS256** (HMAC-SHA256) using `SECRET_KEY`. Anyone can *read* a JWT, but nobody can *modify* it without the key — `jwt.decode` verifies the signature and expiry. Test `test_tampered_token_rejected` proves it.
- Stateless: the server doesn't store sessions; it verifies the signature on each request.
- If `SECRET_KEY` isn't set, a random one is generated at startup (sessions end on restart; warning logged).

**Where the token lives — httpOnly cookie**
```python
response.set_cookie("session", token, max_age=7 days, httponly=True,
                    secure=COOKIE_SECURE, samesite="lax", path="/")
```
- `httponly` → page JavaScript **cannot read it** → an XSS bug can't steal the session (unlike localStorage).
- `samesite="lax"` → the browser won't send it on cross-site POST/DELETE → protects against **CSRF**.
- `secure` → HTTPS-only (enable with `COOKIE_SECURE=1` in production).
- API clients can alternatively send `Authorization: Bearer <token>`.

**`get_current_user`** (FastAPI dependency, used via `Depends`): read token → decode/verify → load user → check password stamp → else **401**.

**Anti-enumeration and timing attack defence (login)**
- Wrong email and wrong password return the **same generic message** "Invalid email or password." → attacker can't discover which emails are registered.
- If the email doesn't exist, we still run bcrypt against a **dummy hash** → the response takes the same time either way (otherwise a fast reply would reveal "no such account").

**Password change invalidates other sessions**
- `password_changed_at` timestamp → `_password_stamp(user)` (milliseconds) is put in every token as `pwd`.
- On password change the stamp changes → every **old token fails** the stamp check → all other devices are signed out; the current one receives a fresh token. (Stateless JWTs can't normally be revoked — this is how we revoke them.)
- Also: changing email requires the current password (stops someone with a stolen session from hijacking the account).

**Duplicate email** → DB unique constraint raises `IntegrityError` → rollback → **409 Conflict**. Emails are lower-cased first.

## H.5 Database schema — [app/models.py](app/models.py)

```
users 1─* subjects
users 1─* checkins 1─* checkin_entries *─1 subjects
checkins 1─1 plans
```

| Table | Key columns | Notes |
|---|---|---|
| `users` | id, email (unique, indexed), password_hash, name, university, programme, year_of_study, password_changed_at, created_at | |
| `subjects` | id, user_id (FK, cascade), name, icon, color, difficulty (1–3), exam_date, archived | Unique (user_id, name) |
| `checkins` | id, user_id, week_start (always a Monday), created/updated_at | Unique (user_id, week_start) → one check-in per week |
| `checkin_entries` | checkin_id, subject_id, hours, confidence | Unique (checkin_id, subject_id) |
| `plans` | checkin_id (unique → 1-to-1), model_version, **result (JSON)** | The full pipeline payload |

Design decisions:
- **Plan stores the full JSON result + `model_version`** ("predictor_v1"): old plans display exactly as they did, even after retraining the model or renaming subjects. Reproducibility/audit trail.
- **Subjects are archived, not deleted**, once used in a check-in → history keeps its rows. Re-adding an archived name restores it with its history. Unused subjects are really deleted.
- **Cascade deletes**: ORM `cascade="all, delete-orphan"` + DB `ON DELETE CASCADE`. Deleting a user deletes everything.
- **SQLite gotcha**: SQLite ignores foreign keys unless `PRAGMA foreign_keys=ON` — set on every connection in `db.py`.
- **`get_db()`** dependency: one session per request, always closed in `finally`.
- `check_same_thread=False` for SQLite because FastAPI uses multiple threads.
- Max 10 subjects per user = length of the colour palette; colours assigned to avoid duplicates.

## H.6 Check-in flow — `POST /api/checkins` ([app/study.py](app/study.py))

1. Compute week = Monday of the given/current date; **reject future weeks** (400).
2. Reject duplicate subject IDs; reject unknown or archived subjects (400). Lookup is restricted to *this user's* subjects.
3. Collect exam dates that are still ahead of that week.
4. `run_pipeline(hours, confidence, subjects=names, difficulty=…, exam_dates=…)` — user's real subject names and difficulties go into Stage 1.
5. `to_payload(...)`.
6. **Upsert**: if a check-in already exists for that week, clear its entries and plan, **flush** (so the unique constraints don't clash), then insert the new ones; set `updated_at` explicitly.
7. Store `Plan(model_version, result=payload)`; commit.
8. Return **201** if new, **200** if replaced.

**User isolation**: every lookup filters by `user_id`. Another user's subject/check-in returns **404, not 403** — doesn't reveal that the ID exists. Tested by `test_users_cannot_touch_each_others_subjects` and `test_checkins_are_private`.

## H.7 Alembic migrations

- `migrations/versions/0001_users_subjects_checkins_plans.py` — creates all tables.
- `migrations/versions/0002_user_profile.py` — adds `university`, `programme`, `year_of_study`, `password_changed_at` to `users`. `down_revision = '0001'` → a chain.
- Each migration has `upgrade()` and `downgrade()` → can roll back.
- Uses `op.batch_alter_table` because **SQLite can't ALTER TABLE most things** — batch mode recreates the table and copies data.
- Why migrations instead of `create_all()`? `create_all` only creates missing tables; it can't alter existing ones without losing data. Migrations evolve the schema safely on existing databases and are version-controlled.
- Commands: `alembic upgrade head`; runs automatically on app start.

## H.8 Tests — [tests/test_api.py](tests/test_api.py) (17 tests)

- Each run uses a **throwaway SQLite file built by the real Alembic migrations** (through the app's startup hook) → the migrations are tested too.
- `TestClient(app)` — in-process HTTP client; using it as a context manager triggers the `lifespan` startup.
- Covered: register/login/logout; duplicates & weak input; generic bad-login message; tampered token rejected; protected routes need login; subject CRUD; subject limit; users can't touch each other's subjects; check-in runs pipeline & stores plan; check-in validation; check-ins private; deleting a used subject archives it; change email needs password & must be free; change password signs out other sessions; delete account removes everything; public demo still works.
- `needs_model` skip marker if the trained model file is missing.
- Other test files: `tests/test_fuzzy.py`, `test_fuzzy_five_subjects.py`, `test_fuzzy_sensitivity.py`, `test_optimizer.py` (teammates' stages, run as modules).

## H.9 Front end — `app/web/` (index.html, app.js, styles.css)

- Single-page app, vanilla JS, Plotly for charts, light/dark theme.
- **Two modes**: **Demo** (`/#demo`, personas, uses `/api/run`, saves nothing) and **User** (account, own subjects, weekly check-ins, history).
- Key JS functions: `api()` (fetch wrapper with JSON + error handling), `runPipeline()`, `renderResults()` → `renderKpis`, `renderScores`, `renderRiskTable`, `renderExplanations`, `renderDeltas`, `renderBaselines`, `renderWeek` (timetable lanes), `renderCharts`, `renderRaw`; auth: `submitAuth`, `logout`, `deleteAccount`, `saveProfile`, `changePassword`; data: `saveCheckin`, `openCheckin`, `renderHistory`, `addSubject`.
- `esc()` HTML-escapes user text before inserting it → XSS protection on the rendering side.
- **Five result tabs** (same in Streamlit): **Predictions** (bar chart), **Risk & gaps** (table + calibration bubble chart: confidence vs predicted, bubble size = priority + explanations), **Reallocation** (current vs optimised hours, deltas, GA vs baselines, convergence curve), **Timetable**, **Raw data** (JSON for transparency).
- Design goal: "read the plan in under a minute" (not formally user-tested).

## H.10 Streamlit — `app/main.py`

Original dashboard: sidebar personas + simulated "Sync LMS", sliders for hours/confidence, five `st.tabs`, Plotly charts. Calls the same `run_pipeline()`. Run: `streamlit run app/main.py` → http://localhost:8501. Reason for moving to FastAPI: Streamlit reruns the whole script on every interaction and isn't suited to accounts, persistence or a proper REST API.

---

# PART I — Demo script (practise it once)

**Before the viva:** activate venv, run once to warm up, have two terminals ready. Don't run `train_mlp.py` live (slow; overwrites the model).

```powershell
cd "E:\University projects\Smart Learning & Performance Optimization System\Project"
.\venv\Scripts\Activate.ps1
```

**1. Data prep (30 s)** — open `src/prediction/prepare_data.py`, point at the 4 leaked columns, the melt, the qcut, the confidence generator. Optional: `python src/prediction/prepare_data.py` (prints dropped columns + "Overconfident 11.8 %, Anxious 16.4 %"). Safe: it only rewrites `data/cleaned_features.csv` with the same seed.

**2. Model & baselines (1 min)**
```powershell
python src/prediction/baselines.py
```
Show the table. Then open `docs/figures/loss_curve.png`, `pred_vs_actual.png`, `residuals.png` and explain each.

**3. End-to-end (1 min)**
```powershell
python smoke_test.py
```
Point at Stage 1 → 2 → 3 → 4 for one profile; "Hour sum: 40.00 / 40.00"; final SUCCESS.

**4. Tests (30 s)**
```powershell
python -m pytest tests -q
```
→ 17 passed.

**5. Web app (2–3 min)**
```powershell
uvicorn app.api:app --reload
```
- http://localhost:8000 → **Try the demo** → Overconfident Student → Run → walk through the 5 tabs. Say: "Databases is flagged overconfident — confidence 9 vs predicted 69; Maths gets the most hours because it has the lowest prediction and the highest priority."
- **Register** → add 2–3 subjects (one with a high difficulty and an exam date) → submit a check-in → show the stored plan and history.
- http://localhost:8000/docs → Swagger: show the routes and try `POST /api/run`.

**Fallbacks:** port busy → `--port 8001`. Missing model → `python src/prediction/train_mlp.py`. Fresh DB → stop server, delete `smartlearn.db`, restart. Streamlit backup → `streamlit run app/main.py`.

> ⚠️ Before the viva: the profile/password work (`app/auth.py`, `app/models.py`, `app/web/*`, `tests/test_api.py`, `migrations/versions/0002_user_profile.py`) was **uncommitted** when this file was written. Commit/push it so the GitHub evidence matches what you demo.

---

# PART J — Limitations, ethics, and TRAPS

## J.1 Limitations in my area (say them before the examiner does)

1. **Weak-signal dataset** — R² ≈ 0.10, scores uncorrelated, behaviour features ≈ zero importance → probably synthetic. Study hours barely move predictions (65.1 → 66.0).
2. **App subjects aren't in the training data** — handled by the averaging fallback, but it's still a neutral guess, not evidence. Real fix: training data for those subjects; flag low-confidence predictions in the UI.
3. **Non-hour features are fixed defaults** — a real system would read them from the LMS; "Sync LMS" is simulated.
4. **Row-level split** — the same student's other subjects can be in training. More rigorous: `GroupShuffleSplit` / `GroupKFold` on `student_id`. Small effect here (subjects uncorrelated). And the student-average baseline only works *because of* the row-level split.
5. **`qcut` bins computed before the split** — minor leak of bin edges.
6. **No cross-validation or hyperparameter search** — the proposal planned 5-fold CV; only the default architecture was evaluated.
7. **Synthetic confidence** — miscalibration rates describe the generator, not students.
8. **Budget = current hours** — the system redistributes, never recommends a different total.
9. **Scheduler drops hours** above 56 h (integration-level issue I know about).
10. **No formal usability study.**

## J.2 Ethics (short answers)

- **Self-fulfilling prediction**: "you're heading for 44" may demotivate → presented as a current trajectory the plan aims to change, always with actions.
- **The label judges a person** → the ±20 threshold is wider than the MAE (8.2), and the explanation shows its evidence.
- **Synthetic data** stated openly.
- **Bias**: no demographic columns, so fairness can't be measured; `class_size`, `resource_index`, `faculty_ratio` may proxy socio-economic status → needs a fairness audit.
- **Privacy**: study data is personal data → consent, minimisation, retention, encryption; GDPR / Sri Lanka PDPA No. 9 of 2022. Our backend: hashed passwords, httpOnly cookies, per-user isolation, full account deletion (right to erasure).
- **Advisory, not authoritative**; student can change any input.
- **AI-assistance disclosure**: AI tools were used for code suggestions/debugging/report editing; every result was run and verified; all numbers come from reproducible runs (seed 42).

## J.3 ⚠️ TRAPS — places where the report and the current code differ

| Where | Report says | Reality now | What to say |
|---|---|---|---|
| Unseen subjects (§8.5, §10.1 #2, ethics #5) | All-zero encoding → predicted ~44–46; that's why the Anxious persona looks overconfident | `predict.py` now **averages over the 4 known subjects** → ~69 | "We identified that the all-zero vector is out-of-distribution, so I added an averaging fallback after the report was written." |
| Table 14 persona results | Databases gap +45.6, priority 7.80, 14.15 h; Anxious persona "3 overconfident" | See **G.6** (e.g. Databases gap +20.7; Anxious persona: 1 anxious, 1 overconfident, 3 aligned) | Same explanation; the experiment design is unchanged. |
| §7.1 API routes | Only `/api/meta`, `/api/run`, `/api/lms-sync` | Plus accounts, subjects, check-ins, profile, password | "The report describes the first API version; I later added persistence and accounts." |
| Ethics #7 | "stores nothing beyond the trained model" | Stores accounts, subjects, check-ins, plans in SQLite | "Now it stores user data, protected with hashing, per-user isolation and full deletion." |
| `smoke_test.py` comment | "Stage 1 stub currently ignores features" | Real model used | "Leftover from stub-first integration." |
| Test count | — | 17 API tests pass | |

---

# PART K — Viva question bank (practise answering out loud)

### Data & leakage
1. **What is data leakage? Give an example from your project.** — Information unavailable at prediction time, or derived from the target, ending up in training. `overall_avg` is the mean of the four scores; with the other three you could solve the fourth exactly.
2. **How did you decide which columns leak?** — One rule for every column: knowable a week before the exam and free of the target? Four failed. Documented in the data dictionary.
3. **Why reshape the data?** — Stage 2 needs one mark per subject; the unit of prediction is (student, subject). One model with `subject` as a feature gives 4× the rows.
4. **What does `pd.melt` do?** — Wide → long: unpivots the four score columns into `subject` + `target_score`, copying the features onto each row.
5. **Why `qcut` instead of `cut`?** — Equal-frequency bins → balanced 1/2/3 groups; `cut` uses equal widths and can leave bins nearly empty.
6. **How was confidence generated? Why per-student bias?** — mark/10 + N(0, 2) per student + N(0, 0.5) per row, rounded and clipped 1–10. A per-student bias makes miscalibration persistent, like a real trait.
7. **Why isn't confidence a feature?** — It's generated from the target (leakage), and it's what Stage 2 compares the prediction against (circular).
8. **Is your synthetic data a problem?** — Yes for claims about real students; the rates describe the generator. Stated openly. The pipeline's logic doesn't depend on those rates.
9. **Any leakage remaining?** — The `qcut` edges were computed before the split (minor, no target info), and the row-level split puts the same student in both sets (small effect because subjects are uncorrelated). Fix: GroupShuffleSplit.

### Model
10. **Why an MLP?** — Learns non-linear interactions automatically; smooth output for the optimiser; module focus. Linear regression trained as a control.
11. **Explain the architecture.** — 17 → 64 ReLU → 32 ReLU → 1 linear, 3,265 parameters.
12. **Calculate the parameter count.** — 17·64+64 + 64·32+32 + 32+1 = 1,152 + 2,080 + 33 = 3,265.
13. **Where do 17 inputs come from?** — 13 scaled numeric features + 4 one-hot subject columns.
14. **Why ReLU?** — Non-linear, cheap, gradient 1 for positive inputs → avoids vanishing gradients.
15. **Why a linear output?** — Regression to an unbounded continuous value; clipping is done in the wrapper.
16. **What does Adam do?** — Per-weight adaptive step sizes from running averages of the gradient and squared gradient, plus momentum; robust with lr 0.001.
17. **Explain backpropagation.** — Chain rule computes the loss gradient for every weight from the output back; weights step against the gradient.
18. **What is early stopping, and what does it monitor?** — Holds out 10 % of training data; stops when the validation R² hasn't improved by 1e-4 for 10 epochs; stopped at 47.
19. **Why `max_iter=1000`?** — Deliberately high so early stopping, not a cap, ends training.
20. **What's the difference between the validation set and the test set?** — Validation (3,200 rows) decides when to stop during training; test (8,000) is touched once for the final score.
21. **Why scale features? Why fit the scaler on training data only?** — Different units would dominate gradients; fitting on test data would leak test statistics.
22. **Why one-hot and not label encoding?** — Subject is nominal; integers imply a false order.
23. **What does `handle_unknown='ignore'` do and why?** — Unknown subjects encode to all zeros instead of crashing; needed because the app uses subjects the model never saw.
24. **Is all-zero a good encoding for an unseen subject?** — No: it's out of distribution (always ~45). So I predict as each known subject and average.
25. **Why a `Pipeline`?** — Same preprocessing at train and inference, the scaler is fitted on training data only, one artifact to save.
26. **Why clip to 0–100? Why not inside the model?** — It's Stage 2's input contract (fuzzy universe 0–100). The raw model is kept for honest evaluation; the wrapper enforces the interface.
27. **Why cache the model?** — Streamlit reruns the script on every interaction; caching avoids reloading from disk.
28. **What's `random_state=42` for?** — Reproducible split, weight initialisation and results.
29. **What's `student_id` doing in X?** — Dropped by the ColumnTransformer (`remainder='drop'`); the model never sees it.
30. **Did you tune hyperparameters?** — No, we used recommended defaults; GroupKFold CV plus a search over layer sizes/α/lr is future work. Given the weak signal, gains would be small.
31. **Why not gradient boosting?** — It's a strong tabular baseline and would be a good addition, but it produces step functions (not smooth) and the module focus is neural networks.

### Evaluation
32. **Define MAE, RMSE, R².** — See E.1.
33. **Why MAE as the headline?** — It's in marks and interpretable, the target was set in MAE, and it's robust to outliers.
34. **Your MLP is no better than linear regression — so why use it?** — An honest finding: the data has no non-linear structure to exploit. The MLP meets the target, beats the student-average baseline, and with real data that has interactions it can learn them without hand-crafted terms. Reporting the tie is the scientifically correct thing to do.
35. **Why is the student-average baseline worse than the global mean?** — Subject scores are uncorrelated (r ≈ 0), so a student's other marks are noise; averaging three noisy numbers is worse than the population mean.
36. **What is R² = 0.10 telling you?** — The features explain only ~10 % of the variance — weak signal, probably synthetic data.
37. **Is the model overfitting?** — No: train 8.175 vs test 8.177.
38. **What is permutation importance?** — Shuffle one feature, measure the MAE increase. `subject` 1.10; the rest ≈ 0.01.
39. **What did the error analysis show?** — Worst errors are extreme Maths/History marks; predictions shrink to the subject mean (regression to the mean).
40. **What does the residual plot show?** — Centred at 0, roughly symmetric → no systematic bias.
41. **Why is the PyTorch model worse?** — Full batch → only 1,000 updates vs ~6,800; no early stopping, no L2; still under-trained. The regime matters, not just the architecture.
42. **Walk me through the PyTorch training loop.** — zero_grad → forward → loss → backward → step; eval with `no_grad`.
43. **How many updates per epoch in sklearn?** — 28,800 / 200 = 144.

### Integration
44. **Why a hybrid system?** — Four different kinds of uncertainty; no labelled optimal timetables; interpretability; hard constraints.
45. **What are the frozen interfaces and why?** — See G.2: parallel development, independent testing, no shape mismatches at integration.
46. **What is stub-first integration?** — Fake but correctly shaped functions committed early so the full pipeline ran from day one.
47. **What's the canonical subject order rule?** — Row order is the only ordering; the GA gets no names, so sorting anywhere would misalign hours.
48. **How does Stage 1's output affect Stage 3?** — Directly through headroom (100 − ŷ) in the fitness, and indirectly through Stage 2's priority.
49. **What if Stage 1 is wrong by 10 marks?** — ≤ ~7 % of hours move, 0 labels change; smooth degradation (G.7).
50. **What problems did you hit during integration?** — See G.4 (unseen subjects, infeasible bounds, DEAP name clash, Streamlit reloads, two UIs, thread safety).
51. **Why the threading lock?** — FastAPI's thread pool + the GA seeding global RNGs → interleaved draws break reproducibility.
52. **How are the budget and bounds chosen?** — Budget = Σ current hours; bounds [2, 15] widened to min(2, B/n) and max(15, 1.5·B/n) so a feasible split always exists.
53. **What does the smoke test check?** — All 4 stages run for 3 personas, hours sum to the budget, a schedule is produced.
54. **Why do the smoke test and dashboard give different predictions?** — Different behavioural feature values (the smoke test sets its own; the dashboard uses `build_features` defaults).

### Backend
55. **Why FastAPI?** — Python (calls the pipeline directly), Pydantic validation, auto docs, async-capable, fast.
56. **422 vs 400?** — 422 = Pydantic schema/type failure (automatic); 400 = my business-rule checks (ranges, future week, duplicates).
57. **How are passwords stored?** — bcrypt salted hashes, never plaintext; 72-byte limit enforced.
58. **What's a JWT and how do you stop tampering?** — Signed token (HS256 with a secret); the signature is verified on every request; expiry 7 days.
59. **Why a cookie and not localStorage?** — httpOnly → JS (and XSS) can't read it; SameSite=Lax → CSRF protection.
60. **How do you prevent account enumeration?** — Generic login error + dummy-hash timing equaliser.
61. **How do you log out other devices after a password change?** — Password timestamp stamp in the token; a changed stamp invalidates old tokens.
62. **How is one user stopped from reading another's data?** — Every query filters by `user_id`; foreign IDs return 404. Tested.
63. **Why store the full plan JSON?** — Old plans render identically after model/subject changes; `model_version` for audit.
64. **Why archive subjects?** — Past check-ins reference them; deleting would break history.
65. **What is Alembic and why use it?** — Versioned schema migrations with upgrade/downgrade; evolves existing DBs without data loss; tests run the real migrations.
66. **Why `PRAGMA foreign_keys=ON`?** — SQLite ignores FKs and cascades by default.
67. **Why `batch_alter_table`?** — SQLite has limited ALTER TABLE; batch mode rebuilds the table.
68. **How would you deploy to production?** — Postgres via `DATABASE_URL`, fixed `SECRET_KEY`, `COOKIE_SECURE=1` behind HTTPS, a real LMS integration, and a model loaded at startup (no lock needed if the GA used a local `random.Random(seed)` instead of global seeding).
69. **What do your API tests cover?** — See H.8; they use a temporary DB built by the real migrations.

### Reflection
70. **What would you do differently?** — Grouped split + CV + hyperparameter search; gradient-boosted baseline; real LMS data; local RNG in the GA instead of a global lock; a scheduler capacity warning; a usability study.
71. **What was the hardest part?** — Integration: domain mismatches (unseen subjects, budgets) rather than code shapes; the frozen contracts prevented the latter.
72. **How did you verify AI-assisted code?** — Ran everything, wrote tests, cross-checked numbers (e.g. the parameter count against the saved model, the fuzzy worked example against code output).

---

# PART L — Glossary

| Term | Meaning |
|---|---|
| ANN / MLP | Artificial neural network / multi-layer perceptron (fully connected feed-forward net) |
| Epoch | One pass over the whole training set |
| Batch / mini-batch | Rows used for one weight update (200 here) |
| Learning rate | Step size of weight updates (0.001) |
| Regularisation (L2, α) | Penalty on large weights to reduce overfitting |
| Early stopping / patience | Stop when validation score stalls for N epochs (10) |
| Generalisation | Performing well on unseen data |
| Baseline | Simple reference model to judge a real model against |
| Data leakage | Target or future information in the training features |
| One-hot encoding | Category → binary indicator columns |
| Standardisation | (x − mean)/std |
| Out-of-distribution (OOD) | Input unlike anything seen in training |
| Permutation importance | MAE increase when a feature is shuffled |
| Regression to the mean | Weak models predict near the average and miss extremes |
| Fuzzification / defuzzification | Crisp → membership degrees / aggregated fuzzy set → crisp (centroid) |
| Mamdani | Fuzzy inference with fuzzy-set consequents (vs Sugeno: functions) |
| GA terms | Chromosome (hour vector), fitness, tournament selection, crossover, mutation, elitism, repair |
| Forward chaining | Data-driven rule firing until no rule applies |
| REST API | HTTP endpoints operating on resources (GET/POST/PATCH/DELETE) |
| ORM | Object-relational mapper (SQLAlchemy): Python classes ↔ tables |
| Migration | Versioned script that changes the DB schema |
| JWT | JSON Web Token: signed claims (sub, exp…) |
| bcrypt / salt | Slow password hash / random per-password value |
| XSS / CSRF | Script injection that steals data / cross-site forged requests |
| httpOnly / SameSite | Cookie hidden from JS / not sent on cross-site requests |
| Idempotent upsert | Re-submitting a week's check-in replaces it instead of duplicating it |

---

# PART M — Last-10-minutes cheat sheet

- **4 leaks**: predicted_score, overall_avg, science_math_diff, eng_hist_ratio
- **10k → 40k** rows via melt; **32k/8k** split; seed **42**
- **17 → 64 → 32 → 1**, ReLU, Adam 0.001, α 1e-4, batch 200, early stop 10 %/patience 10 → **epoch 47**, **3,265 params**
- **MAE 8.18** (< 10 ✔) · LR 8.13 · mean 8.67 · student-avg **10.71** · PyTorch 8.88
- R² 0.10 → weak signal; scores uncorrelated; `subject` importance 1.10; train ≈ test → no overfit
- Confidence = mark/10 + N(0, 2)/student + N(0, 0.5), clip 1–10; **not a feature**; 11.8 % over / 16.4 % anxious
- predict.py: cache · validate columns · normalise names · **unseen → average of known subjects** · **clip 0–100**
- Pipeline: predict → fuzzy (priority, label, gap) → GA (Σh = B) → scheduler (7 d × 8 h, 09:00)
- hour_bounds: min(2, B/n), max(15, 1.5·B/n) · lock for GA global RNG
- Error propagation: ±10 marks → ≤ 7 % hours, 0 labels
- Backend: FastAPI + Pydantic · bcrypt · JWT HS256 in httpOnly SameSite=Lax cookie · dummy-hash timing · password stamp revokes tokens · 404 for other users' data · plan JSON + model_version · archive subjects · Alembic 0001 → 0002 · **17 tests pass**
- Honesty lines: "MLP ties with linear regression — the data, not the model." · "Report predates the unseen-subject fallback."
