<div align="center">

# Smart Learning and Performance Optimization System

### A Hybrid AI Pipeline for Predicting Performance, Detecting Miscalibrated Confidence and Optimising Revision Time

**Module:** Essentials of Artificial Intelligence · **Group 4** · October 2026

![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-%E2%89%A50.110-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-F7931E?logo=scikitlearn&logoColor=white)
![scikit-fuzzy](https://img.shields.io/badge/scikit--fuzzy-0.4.2-4B8BBE)
![DEAP](https://img.shields.io/badge/DEAP-1.4.1-6A5ACD)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C?logo=pytorch&logoColor=white)

</div>

---

## Abstract

Students revising for exams usually divide their time according to how they *feel* about each subject rather than how they are actually performing. A student who wrongly believes a subject is safe under-invests in it; an anxious student spends hours re-reading material they already know. This repository contains the **Smart Learning and Performance Optimization System**, a four-stage hybrid artificial-intelligence pipeline that addresses this problem. Each stage applies the AI technique best suited to the kind of uncertainty it faces:

1. a **multi-layer perceptron (MLP)** regressor estimates the mark a student is heading towards in each subject (*statistical uncertainty*);
2. a **Mamdani fuzzy inference system** compares that estimate with the student's self-reported confidence and produces a continuous 0–10 study priority, a calibration label (*overconfident / aligned / anxious*) and a plain-English explanation (*linguistic uncertainty*);
3. a **real-valued genetic algorithm (GA)** distributes a fixed weekly hour budget across subjects, maximising a priority-weighted, diminishing-returns model of mark gain under per-subject bounds (*combinatorial uncertainty*);
4. a **forward-chaining rule-based scheduler** converts the allocated hours into a day-by-day timetable (*procedural knowledge*).

The pipeline was trained on a dataset of 10,000 students, reshaped into 40,000 student–subject observations after four target-leaking columns were removed. On a held-out test set of 8,000 rows the MLP achieves a **mean absolute error (MAE) of 8.18 marks** (target < 10), outperforming the "student's own average" baseline (MAE 10.71) while being statistically tied with linear regression (MAE 8.13), a result traced to weak signal in the data rather than to a modelling fault. The fuzzy system passes **12 of 12 acceptance tests**, including continuity and monotonicity checks. The GA outperforms equal and proportional allocation on **30 of 30 random seeds** (σ = 0.0002), matches brute-force enumeration to within 0.002 fitness units, and runs in ≈ 0.35 s. The scheduler never violated its daily-limit, session-length or no-overlap rules across 500 randomised inputs. The system is delivered as a multi-user web application (FastAPI + SQLAlchemy + HTML/JS) with persistent weekly check-ins, alongside the original Streamlit prototype.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Background](#2-problem-background)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Methodology: Selected AI Techniques](#5-methodology-selected-ai-techniques)
6. [Justification of Techniques](#6-justification-of-techniques)
7. [System Architecture](#7-system-architecture)
8. [Implementation](#8-implementation)
9. [Web Application, Persistence and API](#9-web-application-persistence-and-api)
10. [Installation and Usage](#10-installation-and-usage)
11. [Evaluation and Results](#11-evaluation-and-results)
12. [Limitations](#12-limitations)
13. [Ethical and Social Considerations](#13-ethical-and-social-considerations)
14. [Future Work](#14-future-work)
15. [Conclusion](#15-conclusion)
16. [Repository Structure](#16-repository-structure)
17. [Project Team](#17-project-team)
18. [References](#18-references)

---

## 1. Introduction

Exam revision is a resource-allocation problem. A student has a finite number of hours before an examination period and several subjects competing for them. Two judgements determine how well those hours are used:

- *How well am I actually doing in each subject?*
- *How much will one additional hour in this subject help?*

People are known to be poorly calibrated on the first judgement: weaker performers tend to overestimate their ability (Kruger & Dunning, 1999), while anxious high achievers tend to underestimate theirs. Neither group has a principled way to answer the second.

This project builds a **decision-support system** that answers both questions on the student's behalf and converts the answers into an actionable plan. Each week the student records the hours they currently spend on each subject and a confidence rating from 1 to 10. The system then:

- **predicts** the mark the student is heading towards in each subject;
- **questions** the student's self-assessment, flagging overconfidence or anxiety and explaining why;
- **allocates** the weekly study budget across subjects to maximise expected marks gained; and
- **schedules** the allocated hours into a weekly timetable.

Every check-in and its resulting plan are stored, so a student can review previous weeks and track how their predicted marks change over time.

The four stages are implemented as loosely coupled modules with **frozen interfaces** (Table 1). This allowed four team members to develop in parallel against a shared contract and allows each stage to be tested in isolation.

**Table 1 – Frozen data contract between stages**

| Stage | Function | Receives | Returns |
|---|---|---|---|
| 1 | `predict_scores(features_df)` | DataFrame, one row per subject, unscaled features | `np.ndarray (n,)` of marks clipped to 0–100 |
| 2 | `assess_risk(predicted, confidence)` | Two arrays `(n,)`: marks 0–100, confidence 1–10 | dict: `priority (n,)` 0–10, `label` list, `gap (n,)`, `explanation` list |
| 3 | `allocate_hours(priority, predicted, budget, min_h, max_h, seed)` | Priority, marks, budget, per-subject bounds | dict: `hours (n,)` summing to budget ± 0.01, `fitness`, `history`, `baselines` |
| 4 | `build_schedule(hours, subjects, days, daily_limits, peak_window, exam_dates)` | Hours, names, days, daily cap, peak window, optional exam dates | list of session dicts (`day`, `start`, `end`, `subject`, `minutes`, `type`, `reason`) |

---

## 2. Problem Background

### 2.1 The revision-allocation problem

A typical undergraduate takes four to six modules concurrently. Revision time is constrained by lectures, employment, sleep and other commitments, so extra time for one subject necessarily means less for another. Three properties make this allocation harder than it appears:

1. **Students cannot observe their own trajectory.** The mark a student is heading towards depends on many behavioural signals (study hours, quiz attempts, login streaks, material-access frequency, score consistency, etc.) scattered across a learning management system (LMS). No student integrates these by hand.
2. **Self-assessment is biased.** Metacognition research shows that confidence and competence frequently diverge. The Dunning–Kruger effect describes low performers who overrate themselves (Kruger & Dunning, 1999); test anxiety leads strong students to over-revise subjects they have already secured. Both errors misallocate time, in opposite directions.
3. **Returns diminish.** The first hours spent on a weak topic close the largest gaps; later hours yield less. "Give everything to the weakest subject" is therefore a poor strategy, and because gains are non-linear the optimal split is not obvious.

### 2.2 Existing approaches

| Approach | Strength | Shortcoming addressed here |
|---|---|---|
| Manual planners / timetable apps | Simple; student retains control | No prediction; relies entirely on biased self-assessment |
| LMS learning-analytics dashboards | Expose activity data | Describe the past; do not predict, prescribe or explain |
| Early-warning (at-risk) systems | Predict failure | Usually a binary flag shown to staff; no plan given to the student |
| Single end-to-end ML recommender | Could, in principle, learn everything | Requires labelled "optimal timetables", which do not exist; opaque; cannot guarantee hard constraints |

No existing tool combines prediction, a calibration check, optimisation and explainable scheduling in a single pipeline. This project targets that gap.

### 2.3 Dataset

The system is trained on `Education_numerical.csv`: **10,000 students × 22 columns**, with no missing values. Each row holds four subject scores (Maths, Science, English, History) and 18 behavioural and contextual attributes. A leakage audit (§8.1.1) identified and removed four columns derived from the target. The dataset contains **no confidence column**, so one was simulated (§8.1.4); this is a significant caveat discussed in §12 and §13. A column-by-column description is given in [docs/data_dictionary.md](docs/data_dictionary.md).

---

## 3. Problem Statement

> *Given a student's behavioural learning data and their self-reported confidence in each of n subjects, and a fixed weekly study budget B hours, produce (a) a per-subject prediction of the mark the student is heading towards; (b) an interpretable assessment of whether the student's confidence is calibrated, together with a continuous study priority; (c) an allocation of B hours across subjects that maximises the expected priority-weighted mark gain subject to per-subject minimum and maximum hours; and (d) a timetable that places those hours into the available days without violating scheduling constraints — all within an interactive response time.*

Formally, Stage 3 solves

$$
\max_{h}\; F(h) = \sum_{i=1}^{n} p_i \,(100 - \hat{y}_i)\,\bigl(1 - e^{-h_i/\tau}\bigr)
\quad\text{s.t.}\quad \sum_{i=1}^{n} h_i = B,\qquad \min_i \le h_i \le \max_i
$$

where $\hat{y}_i$ is the Stage 1 prediction and $p_i$ the Stage 2 priority. Three properties rule out a single simple method:

- the optimiser's inputs ($\hat{y}$, $p$) are themselves uncertain estimates;
- one input — confidence — is a vague, linguistic self-report;
- the search space is large: at 0.5 h granularity, splitting 40 h across 5 subjects yields C(84, 4) = **1,929,501** allocations; 6 subjects yield **32,801,517**; 8 subjects yield **5,843,355,957**.

---

## 4. Objectives

Acceptance targets were fixed in the project proposal before implementation began.

| # | Objective | Acceptance target | Outcome (§11) |
|---|---|---|---|
| O1 | Predict each subject's mark from behavioural data without leakage | MAE < 10 on a held-out set **and** clearly better than the student-average baseline | **Met** – MAE 8.18 vs 10.71 |
| O2 | Assess confidence calibration; produce a smooth priority with an explanation | Expected label and priority band on every case of a hand-built corner-case suite | **Met** – 12/12 |
| O3 | Optimise the hour split | Fitness above equal and proportional baselines; interactive speed | **Met** – 30/30 seeds; ≈ 0.35 s |
| O4 | Produce a constraint-respecting timetable | No hard-constraint violation; all allocated hours placed | **Partially met** – zero violations of implemented rules; hours dropped when budget exceeds capacity |
| O5 | Deliver an integrated, usable interface | End-to-end pipeline callable from a UI; plan readable in under a minute | **Met (pipeline)**; not formally user-tested |
| O6 | Make the system explainable and transparent | Every parameter justified in a decisions log; limitations reported | **Met** – `docs/decisions_member*.md` |

---

## 5. Methodology: Selected AI Techniques

The system is a **hybrid intelligent system** that chains four AI paradigms (Figure 1).

```
 Student inputs: hours/week per subject, confidence (1–10)
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 1  Supervised learning – MLP regressor (scikit-learn) │──► predicted mark ŷᵢ ∈ [0,100]
 └─────────────────────────────────────────────────────────────┘
        │  ŷᵢ + confidence cᵢ
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 2  Mamdani fuzzy inference (scikit-fuzzy)             │──► priority pᵢ ∈ [0,10], label, gap, explanation
 └─────────────────────────────────────────────────────────────┘
        │  pᵢ, ŷᵢ, budget B, bounds
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 3  Real-valued genetic algorithm (DEAP)               │──► hours hᵢ, Σhᵢ = B
 └─────────────────────────────────────────────────────────────┘
        │  hᵢ
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 4  Forward-chaining rule-based scheduler              │──► timetable sessions
 └─────────────────────────────────────────────────────────────┘
```
*Figure 1 – Four-stage pipeline. Each arrow is a frozen interface (Table 1).*

### 5.1 Stage 1 – Multi-Layer Perceptron

An MLP is a feed-forward neural network in which each layer computes $a = \varphi(Wx + b)$ for learned weights $W$, bias $b$ and non-linear activation $\varphi$. It is trained by **backpropagation** (Rumelhart et al., 1986) with the **Adam** optimiser (Kingma & Ba, 2015). The architecture is `17 → 64 (ReLU) → 32 (ReLU) → 1 (linear)`, giving **3,265 trainable parameters**.

### 5.2 Stage 2 – Mamdani Fuzzy Inference System

Fuzzy set theory (Zadeh, 1965) replaces crisp membership with a degree of membership in [0, 1]. A Mamdani FIS (Mamdani & Assilian, 1975) proceeds in four steps: **fuzzification**; **rule evaluation** (AND = *min*, implication by clipping); **aggregation** (*max*); and **centroid defuzzification**. The system has two inputs (predicted score, confidence), one output (priority) and a complete 3 × 3 rule base of nine rules.

### 5.3 Stage 3 – Genetic Algorithm

A GA (Holland, 1975; Goldberg, 1989) is a population-based metaheuristic that evolves candidate solutions through **selection, crossover and mutation**. This implementation uses a **real-valued encoding** (one gene per subject = hours), **tournament selection**, **blend crossover (BLX-α)** (Eshelman & Schaffer, 1993), **Gaussian mutation** and **elitism**, with a **repair operator** for the budget equality and **quadratic penalties** for bounds (Michalewicz, 1996; Coello Coello, 2002). It is built on DEAP (Fortin et al., 2012).

### 5.4 Stage 4 – Rule-Based Expert System

A rule-based system encodes domain knowledge as explicit IF–THEN rules separate from the inference engine. **Forward chaining** is data-driven: starting from known facts (minutes to place, empty days), rules fire until no further rule applies (Giarratano & Riley, 2005). This is deliberately the only stage that learns nothing.

---

## 6. Justification of Techniques

### 6.1 Why a hybrid pipeline rather than one model

The four stages are not four ways of solving one problem; they are **four different problems, each with a different kind of uncertainty**.

| Stage | Kind of uncertainty | Why this technique fits |
|---|---|---|
| 1 | *Statistical* – a future mark not yet observable | Supervised learning estimates an unobserved value from correlated evidence |
| 2 | *Linguistic* – a vague self-report ("I feel 7/10") | Fuzzy sets model categories without sharp edges; Mamdani rules are self-explanatory |
| 3 | *Combinatorial* – millions of feasible splits, non-linear objective | Population-based search accepts any fitness function and constraint-handling scheme |
| 4 | *None* – rules about calendars and attention are known a priori | Explicit rules can be inspected, edited and guaranteed; no training data exists |

An end-to-end model would require **labelled optimal timetables, which do not exist**, would sacrifice the interpretability that Stages 2 and 4 must provide, and could not *guarantee* hard scheduling constraints. The cost of the hybrid design — that errors compound along the pipeline — was measured directly (§11.5).

### 6.2 Stage 1 – Why an MLP

- **Regression, not classification.** Marks are continuous; a pass/fail classifier would treat predicted marks of 38 and 12 identically. Stage 3 also needs a smoothly varying input.
- **Non-linear interactions** (e.g. extra hours being worth more to a student who also attempts quizzes) can be learned without hand-engineered interaction terms.
- **Linear regression was trained as a control** to test whether non-linearity helps. On this dataset it does not (§11.1), and this is reported as a finding.
- Gradient-boosted trees were considered, but the module's focus on neural networks and the need for a smooth function (trees produce step functions) favoured the MLP. A PyTorch re-implementation makes the training loop explicit (§8.1.7).

### 6.3 Stage 2 – Why fuzzy logic, and why Mamdani

A crisp rule such as "confidence < 5 is low" would place students rating 4.9 and 5.1 in **opposite categories**. Because priority is a **weight inside the optimiser**, the GA would then receive a discontinuous input and a 0.2 slider change could rearrange a whole timetable. With graded membership, priority is a continuous function of both inputs; test C11 confirms a change of 0.000 between confidence 4.9 and 5.1.

| Criterion | Mamdani | Sugeno |
|---|---|---|
| Consequent | Fuzzy set ("priority is *very high*") | Function of inputs (e.g. `0.8x + 2.1`) |
| Source of rules | Human expertise | Usually fitted to data |
| Explanation to a student | Follows directly from the rule | A coefficient explains nothing |
| Cost | Requires defuzzification | Cheaper weighted average |

Mamdani was selected because (1) **no dataset of "correct priorities" exists** to fit a Sugeno model, and (2) **interpretability is a functional requirement**. **Centroid defuzzification** uses every fired rule and varies smoothly; mean-of-maximum can jump discontinuously when the dominant rule changes.

Priority is also **not** the confidence gap. A student predicted at 20 who rates confidence 2/10 is perfectly calibrated (gap 0), yet still needs many hours; rule R3 gives this case priority **7.50**. A gap-only design would starve the weakest subject.

### 6.4 Stage 3 – Why a genetic algorithm

| Alternative | Reason rejected |
|---|---|
| Exhaustive enumeration | Feasible at 5 subjects (1.9 M points) but grows to 5.8 billion at 8, and fixes granularity artificially |
| Greedy marginal allocation | Implemented as a baseline. Because the objective is separable and concave, greedy is near-optimal here — a strong benchmark, not a straw man |
| Gradient methods | Quadratic bound penalties and the hard equality make the surface non-smooth at boundaries; would need projections or Lagrangian machinery |
| Linear programming | Exact only for a linear gain; $1 - e^{-h/\tau}$ is deliberately concave. Convex solvers could handle *this* objective, but the GA accepts *any* future fitness, including non-separable terms |

Design choices: **real-valued encoding** (hours are continuous); **tournament selection, k = 3** (moderate, scale-invariant pressure); **blend crossover** (children stay near the feasible hyperplane); **elitism** (best fitness never decreases).

### 6.5 Stage 4 – Why a rule-based system

Scheduling constraints derive from **knowledge about human attention and the student's own availability**, not from patterns in data — e.g. session-length caps, daily limits and the evidence that spaced practice improves retention (Cepeda et al., 2006; Dunlosky et al., 2013). No training data for "good timetables" exists, and a learned scheduler could neither guarantee a hard limit nor explain an individual placement.

---

## 7. System Architecture

### 7.1 Layered design

The system is organised in four layers:

1. **AI core (`src/`).** Four independent packages — `prediction`, `fuzzy`, `optimizer`, `scheduler` — each owned by one team member. Only plain Python types and NumPy arrays cross stage boundaries.
2. **Orchestration ([app/config.py](app/config.py)).** `run_pipeline(hours, confidence, subjects, difficulty, exam_dates)` builds the feature frame, calls the four stages in order, derives the budget and bounds, and returns all intermediate results. A process-wide lock serialises runs because the GA seeds the global random generators. Every user interface calls this one function, so they cannot drift apart.
3. **Service layer (`app/`).** A FastAPI application ([app/api.py](app/api.py)) exposing a public demo API, cookie/JWT authentication ([app/auth.py](app/auth.py)) and per-user subject and check-in management ([app/study.py](app/study.py)), backed by SQLAlchemy models ([app/models.py](app/models.py)) and Alembic migrations.
4. **Presentation.**
   - [app/web/](app/web/) – a single-page HTML/CSS/JS dashboard served by FastAPI (sign-in, subject management, weekly history, profile, light/dark theme, demo mode).
   - [app/main.py](app/main.py) – the original Streamlit dashboard.

```
┌────────────────────┐   ┌────────────────────┐
│ HTML/JS dashboard  │   │ Streamlit app      │
│ app/web/           │   │ app/main.py        │
└─────────┬──────────┘   └─────────┬──────────┘
          │ JSON (cookie / Bearer) │ direct call
┌─────────▼──────────────────────┐ │
│ FastAPI  app/api.py            │ │
│  ├─ auth.py   (users, JWT)     │ │
│  └─ study.py  (subjects,       │ │
│               check-ins, plans)│ │
└─────────┬───────────┬──────────┘ │
          │           │            │
   SQLAlchemy    ┌────▼────────────▼─────┐
   + Alembic     │ app/config.py         │
          │      │ run_pipeline()        │
   ┌──────▼──┐   └────┬──────────────────┘
   │ SQLite /│        │
   │ Postgres│   src/prediction → src/fuzzy → src/optimizer → src/scheduler
   └─────────┘
```
*Figure 2 – Deployment view.*

### 7.2 Data flow for one check-in

1. The client submits, for a given week, one entry per subject: `{subject_id, hours, confidence}`. Pydantic validation rejects hours outside [1, 40], confidence outside [1, 10], or more than 10 subjects.
2. `build_features()` creates one row per subject. `study_hours_week` and `difficulty_indicator` come from the user; the remaining 11 behavioural features use fixed defaults that stand in for an LMS feed (§12).
3. Stage 1 → $\hat{y}$ (clipped to 0–100) → Stage 2 → $p$, labels, gaps, explanations.
4. Budget $B = \sum$ hours. `hour_bounds(B, n)` sets default bounds [2 h, 15 h] and widens them so that a feasible allocation always exists: min = min(2, B/n), max = max(15, 1.5·B/n).
5. Stage 3 → $h$ (seed 42). Stage 4 → sessions over 7 days, 8 h daily cap, 09:00 start; subjects with upcoming exam dates are placed first.
6. The full result is serialised (`to_payload`) and stored as the week's plan together with the model version, then returned to the client.

### 7.3 Design principles

- **Canonical subject order.** Input row order is the only subject ordering; no stage may re-sort. Stage 3 receives no names, so this rule prevents silent misalignment.
- **Fail fast with readable errors.** Stage 1 names missing columns; Stage 2 checks shapes; Stage 3 raises a specific `ValueError` for infeasible budgets; Stage 4 checks one hour value per subject.
- **Reproducibility.** Seed 42 is used for the data split, network initialisation, synthetic confidence and the GA.
- **Stub-first integration.** Correctly shaped stubs of all four functions were committed early so each member could develop against a complete pipeline.
- **Immutable history.** Plans store the rendered payload and subject metadata, so past weeks display identically after subjects are renamed or the model is retrained.

---

## 8. Implementation

**Environment.** Python 3.10–3.12; pandas 2.2.1, NumPy 1.26.4, scikit-learn 1.4.1, scikit-fuzzy 0.4.2, DEAP 1.4.1, PyTorch 2.2.1, Streamlit 1.32.2, FastAPI ≥ 0.110, SQLAlchemy ≥ 2.0, Alembic ≥ 1.13, PyJWT, bcrypt. Versions of the AI libraries are pinned in [requirements.txt](requirements.txt) because scikit-fuzzy and DEAP have changed their APIs between releases.

### 8.1 Stage 1 – Prediction engine (`src/prediction/`)

| File | Purpose |
|---|---|
| [prepare_data.py](src/prediction/prepare_data.py) | Leakage removal, reshaping, difficulty binning, synthetic confidence → `data/cleaned_features.csv` |
| [baselines.py](src/prediction/baselines.py) | Global-mean, student-average and linear-regression baselines |
| [train_mlp.py](src/prediction/train_mlp.py) | Trains and saves the production pipeline → `models/predictor_v1.joblib` |
| [predict.py](src/prediction/predict.py) | Production inference wrapper (`predict_scores`) |
| [pytorch_compare.py](src/prediction/pytorch_compare.py) | Equivalent network in PyTorch for comparison |

#### 8.1.1 Leakage audit

Each column was tested against one rule: *"Would this value be knowable one week before the exam, and is it free of the target?"* Four columns failed:

| Column | Reason for removal |
|---|---|
| `predicted_score` | An existing prediction of the target |
| `overall_avg` | Mean of the four subject scores — contains the target |
| `science_math_diff` | Difference of two target scores |
| `eng_hist_ratio` | Ratio of two target scores |

#### 8.1.2 Reshaping

Stage 2 needs one prediction **per subject**, so the unit of prediction is *(student, subject)*. `pd.melt` converts the four score columns into `subject` and `target_score`: 10,000 × 4 = **40,000 rows**.

#### 8.1.3 Difficulty indicator

`knowledge_gap` (5–60) is binned into an ordinal `difficulty_indicator ∈ {1, 2, 3}` with equal-frequency bins (`pd.qcut`, q = 3). In the web application, each subject's difficulty is set by the student.

#### 8.1.4 Synthetic confidence (simulated data)

The dataset has no confidence rating, so one was generated:

```
confidence   = clip( round( target_score/10 + student_bias + ε ), 1, 10 )
student_bias ~ N(0, 2.0)   – one draw per student → persistent over/under-confidence
ε            ~ N(0, 0.5)   – per-row noise
```

This yields **11.8 % overconfident** and **16.4 % anxious** rows (|gap| > 20). **Confidence is deliberately not a model feature**: it is generated from the target (leakage), and by design it is an *input to Stage 2*.

#### 8.1.5 Preprocessing and model

- **Features:** 13 numeric features (`study_hours_week`, `quiz_attempts`, `consecutive_logins`, `material_access_freq`, `improvement_rate`, `score_consistency`, `time_per_question`, `class_size`, `resource_index`, `faculty_ratio`, `peak_study_time`, `weekend_study_ratio`, `difficulty_indicator`) plus one-hot `subject` → **17 inputs**.
- **`StandardScaler`** – features span very different scales (`class_size` 20–59 vs `faculty_ratio` 0.05–0.20).
- **`OneHotEncoder(handle_unknown='ignore')`** – subject is nominal; unseen subjects do not crash inference.
- **Single `Pipeline`** – the scaler is fitted on training rows only; preprocessing and model are persisted together as one `joblib` artefact.

**Table 2 – MLP hyperparameters**

| Hyperparameter | Value | Rationale |
|---|---|---|
| Hidden layers | (64, 32) | Funnel wide enough for interactions, small enough not to overfit 32 k rows |
| Activation | ReLU hidden, identity output | Non-saturating; linear output for regression |
| Solver | Adam, lr = 0.001 | Adaptive per-parameter steps |
| L2 penalty | α = 1e-4 | Light weight decay |
| Batch size | min(200, n) | ≈ 144 updates per epoch |
| Early stopping | 10 % validation, patience 10, tol 1e-4 | Stopped at **epoch 47** |
| `max_iter` | 1000 | High so that early stopping, not a cap, ends training |
| Split | 80/20, `random_state = 42` | 32,000 train / 8,000 test; test set used once |

#### 8.1.6 Production wrapper

`predict_scores()` **lazily loads and caches** the model, validates that all 14 expected columns are present, normalises subject names to the trained categories (case- and whitespace-insensitive), and **clips output to [0, 100]** to match Stage 2's universe.

**Unseen-subject fallback.** Students may register any subject (e.g. *Databases*). Such a subject one-hot encodes to all zeros, a point never seen in training. Rather than return that out-of-distribution value, the wrapper predicts the row **as each known subject and averages** the results — a neutral prior that still responds to the student's other features.

#### 8.1.7 PyTorch comparison

The same architecture (`Linear(17,64)–ReLU–Linear(64,32)–ReLU–Linear(32,1)`) is re-implemented in PyTorch with `MSELoss`, Adam (lr = 0.001), full-batch updates and 1,000 epochs, making the training loop (`zero_grad → forward → backward → step`) explicit.

### 8.2 Stage 2 – Fuzzy risk assessor ([src/fuzzy/risk.py](src/fuzzy/risk.py))

#### 8.2.1 Membership functions

Triangles are used for middle terms; **trapezoids** for end terms, whose plateaus express that, e.g., predicted scores of 5 and 15 are both unambiguously "low".

| Variable (universe, step) | Term | Shape | Parameters |
|---|---|---|---|
| Predicted score (0–100, 0.5) | low | trapezoid | [0, 0, 35, 50] |
| | medium | triangle | **[40, 55, 75]** (tuned from [35, 55, 75]) |
| | high | trapezoid | [60, 75, 100, 100] |
| Confidence (0–10, 0.1) | low | trapezoid | [0, 0, 3, 5] |
| | medium | triangle | [3, 5.5, 8] |
| | high | trapezoid | [6, 8, 10, 10] |
| Priority (0–10, 0.1) | very low / low / medium / high / very high | triangles | peaks at 0, 2.5, 5, 7.5, 10 |

**Tuning decision.** With the initial medium = [35, 55, 75], continuity test C10 failed (priority jumped by 0.699 between predicted 35 and 36). Moving the foot to 40 reduced the jump to **0.000**.

#### 8.2.2 Rule base

The 3 × 3 rule base is **complete by construction**. It separates two concepts: **priority** (*how valuable is the next hour here?*) and **label** (*can the self-assessment be trusted?*).

| # | Predicted | Confidence | → Priority | Reasoning |
|---|---|---|---|---|
| R1 | low | high | **very high** | Failing and unaware — the case the system exists for |
| R2 | low | medium | very high | Failing and insufficiently alarmed |
| R3 | low | low | high | Weak but calibrated — still needs many hours |
| R4 | medium | high | high | Borderline pass treated as safe |
| R5 | medium | medium | medium | Neutral centre |
| R6 | medium | low | medium | Borderline and worried; hours still useful |
| R7 | high | high | **very low** | Strong and correctly confident — maintenance only |
| R8 | high | medium | low | Strong, mildly unsure |
| R9 | high | low | low | Strong but anxious — hours reclaimed for R1-type subjects |

#### 8.2.3 Worked example (predicted = 45, confidence = 7.5)

1. **Fuzzification.** Predicted: μ_low = 0.333, μ_medium = 0.333. Confidence: μ_medium = 0.200, μ_high = 0.750.
2. **Rule evaluation (min).** R1 = 0.333, R2 = 0.200 (→ very high); R4 = 0.333 (→ high); R5 = 0.200 (→ medium).
3. **Aggregation (max).** *very high* clipped at 0.333, *high* at 0.333, *medium* at 0.200.
4. **Defuzzification (centroid)** → **6.789**, identical to the value returned by `assess_risk`.
5. **Label.** gap = 10 × 7.5 − 45 = +30 > +20 → **overconfident**.

See [docs/mamdani_example_melani.md](docs/mamdani_example_melani.md) and `docs/figures/fuzzy_worked_example.png`.

#### 8.2.4 Label, gap and explanation

The label is computed from a crisp gap rather than defuzzified:

```
gap = 10 · confidence − predicted
gap > +20 → overconfident     gap < −20 → anxious     otherwise → aligned
```

The crisp threshold applies only to a *reported diagnostic*, never to the optimiser's weight. Explanations are templated and retain the evidence, e.g. *"You rated your confidence 8.0 out of 10, but your predicted score is around 32.0 … Priority is 9.17 out of 10."*

#### 8.2.5 Engineering notes

The `ControlSystem` is built once; a fresh `ControlSystemSimulation` is created per subject to avoid state leakage. Inputs are clamped to their universes. Five subjects are assessed in ≈ 25 ms.

### 8.3 Stage 3 – GA optimiser (`src/optimizer/`)

#### 8.3.1 Fitness function ([fitness.py](src/optimizer/fitness.py))

$$
F(h) = \underbrace{\sum_i p_i \cdot \max(0, 100-\hat{y}_i) \cdot \bigl(1-e^{-h_i/\tau}\bigr)}_{\text{priority}\,\times\,\text{headroom}\,\times\,\text{diminishing returns}} \;-\; \lambda \sum_i \Bigl[(\min_i - h_i)_+^2 + (h_i - \max_i)_+^2\Bigr]
$$

- **Priority $p_i$** – value of a mark gained in this subject now.
- **Headroom $100-\hat{y}_i$** – marks still available.
- **Saturating gain, τ = 7 h** – ≈ 63.2 % of available gain captured by 7 h; concavity makes balanced splits beat extreme ones (Table 3).
- **Penalty λ = 20** – quadratic, so small violations are cheap and large ones prohibitive; a secondary safeguard behind repair.

**Table 3 – Concavity check** (two identical subjects, p = 5, ŷ = 50, B = 10 h)

| h₁ | h₂ | Fitness |
|---|---|---|
| 10 | 0 | 190.09 |
| 8 | 2 | 232.40 |
| 7 | 3 | 245.17 |
| 6 | 4 | 252.73 |
| **5** | **5** | **255.23** |

#### 8.3.2 Constraint handling ([repair.py](src/optimizer/repair.py))

- **Feasibility check** – if Σmin > B or Σmax < B, raise a specific error (e.g. *"Sum of minimum hours (10.00h) exceeds total budget (5.00h)"*).
- **Bounded repair** – alternate *scale → clip* at most five times, then distribute any residual only across genes not pinned at a bound, in proportion to their room. NaNs and negatives are replaced first. Repair runs after initialisation, after every variation, and before evaluation.

#### 8.3.3 GA configuration ([ga.py](src/optimizer/ga.py))

| Parameter | Value | Rationale |
|---|---|---|
| Encoding | Real vector, length n | Hours are continuous |
| Population | 80 | Diverse enough for 5–8 subjects (Table 9) |
| Generations | 100 | Best fitness plateaus within ≈ 10 generations |
| Selection | Tournament, k = 3 | Moderate, scale-invariant pressure |
| Crossover | `cxBlend`, α = 0.5, p = 0.8 | Children near the feasible hyperplane |
| Mutation | Gaussian, σ = 1.0 h, indpb = 0.25, p = 0.2 | Exploration without destroying good solutions |
| Elitism | Hall of Fame, best 2 | Convergence curve never falls |
| Initial population | Random repaired vectors **plus** equal, proportional and greedy baselines | GA can never return worse than the best heuristic |
| Seed | 42 | Reproducibility |

#### 8.3.4 Baselines ([baselines.py](src/optimizer/baselines.py))

- **Equal split** – B/n per subject, repaired.
- **Proportional split** – hours proportional to priority, repaired.
- **Greedy** – start at minima; repeatedly add 0.5 h to the subject with largest marginal gain.

`allocate_hours()` ([allocate.py](src/optimizer/allocate.py)) returns all three baseline fitness values alongside the GA result and rescales if the sum drifts by more than 0.01 h. An earlier prototype lives in [genetic_algorithm/](genetic_algorithm/) together with brute-force validation and convergence-plot scripts.

### 8.4 Stage 4 – Scheduler ([src/scheduler/schedule.py](src/scheduler/schedule.py))

- **Working memory:** minutes remaining per subject, current day, minutes used today.
- **Rules:**
  1. Convert hours to whole minutes.
  2. If exam dates are given, order subjects by earliest exam (conflict resolution); otherwise keep canonical order.
  3. IF today's minutes ≥ daily limit THEN advance to next day.
  4. IF no days remain THEN stop.
  5. Place a session of min(60, remaining for subject, remaining today) minutes, starting at the peak-window start (default 09:00) plus minutes already used.
- **Output:** each session records `day`, `start`, `end`, `subject`, `minutes`, `type = "study"` and `reason`.

### 8.5 Integration issues and resolutions

| Problem | Resolution |
|---|---|
| App subjects absent from training data caused `OneHotEncoder` errors | `handle_unknown='ignore'`, then an averaged-known-subject fallback in `predict_scores` (§8.1.6) |
| Persona budgets (e.g. 115 h) fell outside fixed bounds n × [2, 15] | `hour_bounds()` widens bounds from the budget |
| Duplicate DEAP `creator` class names between prototype and production GA | Production classes renamed `OptimizerFitnessMax` / `OptimizerIndividual`, guarded with `hasattr` |
| Streamlit reloaded the model on every rerun | Lazy, cached model loading |
| Two front ends could diverge | Both call a single `run_pipeline()` |
| Concurrent API requests broke GA reproducibility (global RNG) | Pipeline execution guarded by a `threading.Lock` |
| Out-of-range inputs | Pydantic range validation (HTTP 400/422) plus clamping in Stage 2 |

---

## 9. Web Application, Persistence and API

### 9.1 User workflow

1. **Register / sign in** (or open the read-only **demo** at `/#demo`, which uses three sample personas and saves nothing).
2. **Add subjects** – name, icon, difficulty (1–3) and optional exam date; up to 10 active subjects.
3. **Weekly check-in** – hours studied and confidence for each subject. The pipeline runs immediately.
4. **Review results** in five tabs: *Predictions*, *Risk & gaps* (calibration bubble chart and explanations), *Reallocation* (current vs optimised hours, GA vs baselines, convergence curve), *Timetable*, and *Raw data*.
5. **History** – previous weeks are listed in the sidebar and can be reopened or deleted.
6. **Profile** – name, university, programme, year of study, email and password; account deletion removes all data.

### 9.2 Data model

```
users ─┬─< subjects            name, icon, colour, difficulty (1–3), exam date, archived flag
       └─< checkins            one per user per week (week_start = Monday)
              ├─< checkin_entries   hours + confidence per subject
              └── plans             full pipeline payload (JSON) + model version
```

- Submitting a second check-in for the same week **replaces** that week's entries and plan.
- A subject referenced by past check-ins is **archived, not deleted**, so history stays intact; re-adding a subject with the same name restores it.
- Schema changes are managed with Alembic (`migrations/versions/0001_…`, `0002_user_profile.py`) and applied automatically at start-up when `AUTO_MIGRATE=1`.

### 9.3 Security

- Passwords hashed with **bcrypt** (72-byte limit enforced); login compares against a dummy hash for unknown emails to equalise timing.
- Sessions are **HS256 JWTs** delivered in an `httpOnly`, `SameSite=Lax` cookie (optionally `Secure`), or as `Authorization: Bearer <token>` for non-browser clients.
- Tokens embed a password-change stamp, so **changing the password revokes all other sessions**.
- Every subject and check-in query is scoped to the authenticated user; cross-user access returns 404.

### 9.4 REST API

Interactive OpenAPI documentation is served at `http://localhost:8000/docs`.

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/register` | `{email, password, name}` → create account and sign in |
| POST | `/api/auth/login` | `{email, password}` |
| POST | `/api/auth/logout` | Clear session cookie |
| GET / PATCH / DELETE | `/api/me` | Read profile / update profile / delete account and all data |
| POST | `/api/me/password` | Change password (signs out other sessions) |
| GET | `/api/me/meta` | Input ranges, subject limit, current week |
| GET / POST | `/api/subjects` | List (`?include_archived=true`) / add |
| PATCH / DELETE | `/api/subjects/{id}` | Edit / remove (archived if it has history) |
| POST | `/api/checkins` | `{week_start?, entries:[{subject_id, hours, confidence}]}` → run pipeline, store plan |
| GET | `/api/checkins` | History summaries, newest first |
| GET / DELETE | `/api/checkins/{id}` | Full check-in with plan / delete |
| GET | `/api/meta` | Public demo: personas and subjects |
| POST | `/api/run` | Public demo: run pipeline, nothing saved |
| POST | `/api/lms-sync` | Public demo: simulated LMS sync |

---

## 10. Installation and Usage

### 10.1 Prerequisites

- Python **3.10 – 3.12** (pinned NumPy 1.26 / pandas 2.2 may not build on newer versions)
- Git

### 10.2 Setup

```bash
git clone <repository-url>
cd Project

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 10.3 Data and model (only if missing)

`data/cleaned_features.csv` and `models/predictor_v1.joblib` are normally included. To regenerate them from `Education_numerical.csv`:

```bash
python src/prediction/prepare_data.py     # → data/cleaned_features.csv
python src/prediction/train_mlp.py        # → models/predictor_v1.joblib
python src/prediction/baselines.py        # optional: baseline comparison
python src/prediction/pytorch_compare.py  # optional: PyTorch comparison
```

### 10.4 Configuration

Copy [.env.example](.env.example) to `.env`, or set variables in the shell.

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | random per start | **Required in production**; signs session tokens. Without it, all users are signed out on restart. |
| `DATABASE_URL` | `sqlite:///./smartlearn.db` | Any SQLAlchemy URL, e.g. `postgresql+psycopg://user:pass@host:5432/smartlearn` |
| `COOKIE_SECURE` | `0` | Set to `1` behind HTTPS |
| `SESSION_DAYS` | `7` | Session lifetime |
| `AUTO_MIGRATE` | `1` | Run `alembic upgrade head` on start-up |

Generate a key with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

### 10.5 Running

```bash
uvicorn app.api:app --reload        # web app  → http://localhost:8000
                                    # API docs → http://localhost:8000/docs
streamlit run app/main.py           # Streamlit prototype → http://localhost:8501
```

Run commands from the project root; the model path is relative to it.

### 10.6 Database migrations

After changing [app/models.py](app/models.py):

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

### 10.7 Tests

```bash
python -m pytest tests                      # API: auth, user isolation, subjects, check-ins, profile, demo
python -m tests.test_fuzzy                  # Stage 2 corner-case acceptance tests
python -m tests.test_fuzzy_sensitivity      # Stage 2 label-threshold sensitivity
python -m tests.test_fuzzy_five_subjects    # Stage 2 + 3 five-subject sanity test
python -m tests.test_optimizer              # Stage 3 tests
python smoke_test.py                        # end-to-end pipeline on all personas
python -m genetic_algorithm.validate_brute_force   # GA vs exhaustive enumeration
```

### 10.8 Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn` not recognised | Activate the virtual environment |
| `ModuleNotFoundError` | `pip install -r requirements.txt` inside the venv |
| Missing model or features file | Run the commands in §10.3 |
| Port 8000 in use | `uvicorn app.api:app --reload --port 8001` |
| Signed out after every restart | Set a fixed `SECRET_KEY` |
| Schema errors after pulling | `alembic upgrade head` |
| Fresh database wanted | Stop the server, delete `smartlearn.db`, restart |

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for a step-by-step guide.

---

## 11. Evaluation and Results

All figures below were produced by running the project code (seed 42 unless stated). Full discussion is in [REPORT.md](REPORT.md).

### 11.1 Stage 1 – Prediction accuracy

**Table 4 – Held-out test set (8,000 rows)**

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| Global mean | 8.668 | 10.922 | 0.000 |
| Student's own average | 10.705 | 13.442 | −0.515 |
| Linear regression | **8.133** | **10.281** | **0.114** |
| **MLP (scikit-learn, production)** | 8.177 | 10.342 | 0.103 |
| MLP (PyTorch, full-batch, 1,000 epochs) | 8.884 | 11.225 | −0.057 |

67.7 % of test predictions fall within 10 marks of the true value. Train MAE (8.175) ≈ test MAE (8.177): no overfitting.

**Interpretation.** The MLP meets O1 but is tied with linear regression and only modestly better than the global mean. The evidence implicates the data, not the model:

1. Subject scores are mutually uncorrelated (pairwise r ∈ [−0.010, 0.027]), which is why the student-average baseline is *worse* than the global mean.
2. Permutation importance is dominated by `subject` (1.102); the next features contribute ≈ 0.009 or less. The model has mainly learned per-subject means (English 74.4, Science 69.6, History 67.4, Maths 64.5).
3. With no non-linear structure to exploit, the MLP converges near the linear solution.
4. The PyTorch model is weaker because it received ≈ 1,000 full-batch updates versus ≈ 6,800 mini-batch updates — evidence that the optimisation regime matters as much as architecture.

The 20 worst errors are all Maths or History rows whose true marks lie at the extremes (30–37 or 100), while predictions shrink toward the subject mean (all predictions lie in 60.2–80.0). Figures: `docs/figures/pred_vs_actual.png`, `residuals.png`, `loss_curve.png`.

### 11.2 Stage 2 – Fuzzy acceptance tests

**Table 5 – Corner-case suite ([tests/test_fuzzy.py](tests/test_fuzzy.py))**

| Case | Predicted | Confidence | Priority | Label | Expected | Result |
|---|---|---|---|---|---|---|
| C1 | 20 | 9 | 9.167 | overconfident | ≥ 8, overconfident | ✔ |
| C2 | 20 | 2 | 7.500 | aligned | 6–8.5, aligned | ✔ |
| C3 | 85 | 9 | 0.833 | aligned | ≤ 2, aligned | ✔ |
| C4 | 85 | 2 | 2.500 | anxious | ≤ 3.5, anxious | ✔ |
| C5 | 55 | 5 | 5.000 | aligned | 4–6, aligned | ✔ |
| C6 | 50 | 5 | 5.000 | aligned | 4–6, aligned | ✔ |
| C7 | 45 | 7.5 | 6.789 | overconfident | ≥ 6, overconfident | ✔ |
| C8 | 0 | 10 | 9.167 | overconfident | overconfident | ✔ |
| C9 | 100 | 1 | 2.500 | anxious | anxious | ✔ |
| C10 | 35 → 36 | 5 | 9.139 → 9.139 | — | Δ < 0.3 | ✔ Δ = 0.000 |
| C11 | 60 | 4.9 → 5.1 | 5.000 → 5.000 | — | Δ < 0.2 | ✔ Δ = 0.000 |
| C12 | 0…100 step 2 | 5 | non-increasing | — | monotonic | ✔ |

**12/12 pass.** The output range is ≈ [0.83, 9.17] rather than [0, 10], an expected property of centroid defuzzification.

**Threshold sensitivity** (13 cases): tightening ±20 → ±15 relabelled 4 borderline cases; loosening to ±25 changed none. ±20 was retained.

**Five-subject sanity test.** For ŷ = [32, 48, 58, 82, 90], confidence = [8, 5, 4, 3, 9]: priorities [9.17, 5.53, 5.00, 2.50, 0.83]; labels overconfident, aligned, aligned, anxious, aligned. With B = 40 h, the GA gives the overconfident subject the 15 h maximum and cuts the anxious-but-strong subject to the 2 h minimum (a proportional split would give it 4.41 h) — the intended "reclaim hours from reassurance" behaviour. Control surface: `docs/figures/fuzzy_control_surface.png`.

### 11.3 Stage 3 – GA optimisation

**Table 6 – Standard case** (p = [8.5, 5.2, 6.2, 2.1, 5.8], ŷ = [41.8, 49.4, 45.6, 78.3, 46.9], B = 40 h, bounds [2, 15])

| Method | Allocation (h) | Fitness |
|---|---|---|
| Equal | [8.0, 8.0, 8.0, 8.0, 8.0] | 986.67 |
| Proportional | [12.23, 7.48, 8.92, 3.02, 8.35] | 1054.70 |
| Greedy (0.5 h) | [12.0, 7.5, 9.5, 2.0, 9.0] | 1063.23 |
| **GA** | **[12.10, 7.69, 9.42, 2.00, 8.79]** | **1063.32** |

GA improves on equal by **+7.8 %** and on proportional by **+0.8 %**.

**Table 7 – Robustness over 30 seeds**

| Statistic | Value |
|---|---|
| Mean ± SD | 1063.321 ± 0.0002 |
| Beats equal / proportional | **30/30** / **30/30** |
| ≥ greedy | 30/30 |

Re-running all 30 seeds with a **purely random** initial population (no baselines injected) reached the same optimum on every seed, overtaking greedy by generation 8 — the seeding saves time but does not supply the answer.

**Table 8 – Brute-force validation** (3 subjects, B = 15 h, bounds [1, 10], 0.1 h grid)

| Method | Allocation | Fitness |
|---|---|---|
| Exhaustive enumeration | (9.2, 4.8, 1.0) | 498.436 |
| GA | (9.21, 4.79, 1.00) | 498.438 |

**Table 9 – Population sensitivity (10 seeds each)**

| Population | Mean fitness | SD | Median time |
|---|---|---|---|
| 20 | 1063.320 | 0.0008 | 0.09 s |
| 50 | 1063.321 | 0.0002 | 0.23 s |
| **80** | 1063.321 | 0.0002 | **≈ 0.35 s** |
| 150 | 1063.321 | 0.0000 | ≈ 5 s* |

\*Inflated by CPU throttling on the test machine; profiled compute was < 0.9 s.

Additional tests confirm: budget equality (±0.01), bound compliance, monotone best-fitness (elitism), clear errors on infeasible budgets, determinism under a fixed seed, monotonicity in priority, and correct handling of identical or zero priorities.

**Finding.** Because the objective is separable and concave, greedy is near-optimal and the GA's margin over it is small (+0.09). The GA is retained because it is **never worse** than greedy and **remains valid if the fitness becomes non-separable** (e.g. subject interactions), where greedy has no guarantee.

### 11.4 Stage 4 – Scheduler

**Table 10 – Constraint audit over 500 randomised inputs** (budget ~ U[10, 80] h, Dirichlet split, 7 days × 8 h)

| Check | Result |
|---|---|
| Day over 8 h limit | **0** violations |
| Session over 60 min | **0** violations |
| Overlapping sessions | **0** violations |
| All minutes placed when budget ≤ 56 h | **339/339** ✔ |
| Budget > capacity (> 56 h) | **161/500** cases: excess hours dropped without warning ✘ |
| Exam-date ordering | ✔ |

### 11.5 End-to-end integration and error propagation

[smoke_test.py](smoke_test.py) runs every persona through all four stages; across personas the ordering GA ≥ greedy ≥ proportional ≥ equal held throughout.

To test how Stage 1 error propagates, every prediction was perturbed by ±5 and ±10 marks for a profile with ŷ = [38, 55, 62, 70, 48], confidence = [8.5, 5, 6, 7, 4], B = 40 h.

**Table 11 – Sensitivity to Stage 1 error**

| Perturbation | Max Δpriority | Max Δhours (one subject) | Total hours moved | Labels changed |
|---|---|---|---|---|
| −10 | 2.25 | 2.06 h | 2.59 h (6.5 %) | 0/5 |
| −5 | 1.39 | 1.08 h | 1.76 h (4.4 %) | 0/5 |
| +5 | 1.98 | 1.38 h | 1.95 h (4.9 %) | 0/5 |
| +10 | 1.98 | 1.40 h | 2.71 h (6.8 %) | 0/5 |

A ten-mark error — larger than the model's MAE — moves **at most ≈ 7 % of the weekly budget** and changes no labels. Because fuzzification and the concave fitness both respond smoothly, the plan **degrades gracefully** rather than failing.

### 11.6 Application tests

[tests/test_api.py](tests/test_api.py) contains 17 integration tests run against a live server with a temporary database, covering registration and login, generic login errors, tampered tokens, protected routes, subject CRUD and limits, cross-user isolation of subjects and check-ins, archiving on deletion, check-in validation and plan storage, profile updates, email change, password change revoking other sessions, account deletion cascade, and the public demo.

---

## 12. Limitations

**Data and prediction**

1. **Weak signal.** R² ≈ 0.10, uncorrelated subject scores and near-zero behavioural importance suggest a synthetic dataset; `study_hours_week` barely moves predictions, weakening the "what if I study more" narrative.
2. **Out-of-distribution subjects.** Only Maths, Science, English and History were seen in training. Other subjects receive the averaged-known-subject prior (§8.1.6), which is neutral but not evidence-based.
3. **Fixed behavioural features.** Non-hour features are demo defaults; the LMS sync is simulated.
4. **Row-level split.** A grouped split on `student_id` would be more rigorous; `qcut` bin edges were computed before splitting.
5. **No hyperparameter search or cross-validation.**

**Fuzzy assessment**

6. **Synthetic confidence.** Miscalibration rates describe the generator, not real students; thresholds were sensitivity-tested but not validated on real self-reports.
7. **Hand-designed membership functions** based on expert judgement.

**Optimisation**

8. **The fitness function is a model, not a measurement.** The gain curve $1-e^{-h/\tau}$ and τ = 7 h are not empirically validated.
9. **Greedy is nearly as good** under the current separable objective.
10. **8-subject run-time target (< 1 s)** not consistently verified on the throttled test machine.

**Scheduling and interface**

11. **Silent hour loss** when the budget exceeds 7 × 8 h capacity.
12. **Unimplemented planned rules:** breaks, spacing across days, peak-window priority, pre-exam consolidation, specific reason strings. Output is valid but clumped.
13. **No formal usability study.**
14. **Fixed total budget.** The system redistributes existing effort; it never recommends studying more or less overall.

---

## 13. Ethical and Social Considerations

1. **Predictions about a person.** A predicted mark delivered during a stressful period could become self-fulfilling. The interface presents it as a *current trajectory that the plan is designed to change*, always paired with concrete actions.
2. **Calibration labels are judgements.** The ±20-mark threshold is wider than the model's ≈ 8-mark MAE, and every label displays its evidence so the student can check the reasoning. The system advises; it does not pass judgement.
3. **Synthetic data** is disclosed wherever results depend on it.
4. **Bias.** The dataset has no demographic attributes, so fairness cannot be measured; proxies such as `class_size`, `resource_index` and `faculty_ratio` may encode socio-economic differences. Deployment would require a fairness audit on representative data.
5. **Out-of-distribution predictions** should be flagged as lower-confidence in a production system.
6. **Advisory, not authoritative.** The system cannot know about illness, employment or caring duties; students can change any input and re-run.
7. **Privacy.** Study behaviour and marks are personal data. The application hashes passwords, scopes all data to its owner, and supports full account deletion. A real deployment would additionally need a lawful basis and consent (e.g. GDPR; Sri Lanka's Personal Data Protection Act No. 9 of 2022), data minimisation, defined retention and encryption at rest (Slade & Prinsloo, 2013).
8. **Staff use and power relations.** Any tutor-facing view should show students everything staff see, frame flags as conversation starters, and never feed assessment or progression decisions.
9. **Over-reliance.** Showing *why* alongside *what* is intended to build students' own calibration rather than replace it.
10. **AI-assistance disclosure.** AI assistants were used for code suggestions, debugging, test generation and editing. All AI-suggested components were reviewed and tested by the responsible member; design decisions are recorded in `docs/decisions_member*.md`. All numerical results come from reproducible runs of the project code.

---

## 14. Future Work

In priority order:

1. Scheduler capacity check with a user-facing warning; breaks, spaced practice and peak-window placement with an independent constraint checker.
2. Retrain on real multi-subject LMS data using a grouped split and cross-validated hyperparameter search; connect real LMS features in place of defaults.
3. Collect real confidence ratings to calibrate label thresholds.
4. Validate the gain model and τ empirically, and personalise τ per student.
5. Recommend a total weekly budget, not only its split.
6. Use stored weekly check-ins to measure plan adherence and outcome over time.
7. Conduct a formal usability study.

---

## 15. Conclusion

This project designed, implemented and evaluated a four-stage hybrid AI system that turns a student's learning data and self-reported confidence into a predicted mark, an explained calibration diagnosis, an optimised allocation of study hours and a weekly timetable, delivered through a multi-user web application with persistent weekly history. Each stage uses the technique suited to its uncertainty: a neural network for statistical estimation, Mamdani fuzzy inference for vague self-report, a genetic algorithm for combinatorial allocation, and a rule-based expert system for known procedural constraints.

Against its pre-registered targets, the MLP reached MAE 8.18 (< 10) and beat the student-average baseline; the fuzzy system passed 12/12 acceptance tests with a hand-verified worked example; the GA beat both naïve baselines on 30/30 seeds and matched brute force; and the scheduler never violated an implemented constraint. An error-propagation study showed that a 10-mark prediction error shifts at most ≈ 7 % of the weekly budget. The principal limitations — a weak-signal dataset, simulated confidence, an unvalidated gain model and an incomplete scheduler — are documented with their consequences and a remediation plan. The main contribution is a modular, testable and explainable pipeline in which every parameter is justified and every claim is backed by a reproducible number.

---

## 16. Repository Structure

```
Project/
├── app/
│   ├── api.py              FastAPI app, public demo endpoints, static hosting, auto-migration
│   ├── auth.py             Registration, login, JWT sessions, profile, password, account deletion
│   ├── study.py            Per-user subjects and weekly check-ins (runs and stores the pipeline)
│   ├── models.py           SQLAlchemy ORM: users, subjects, checkins, checkin_entries, plans
│   ├── db.py               Engine/session factory (SQLite default, Postgres-ready)
│   ├── config.py           run_pipeline(), feature builder, hour bounds, personas, input ranges
│   ├── main.py             Streamlit dashboard (original prototype)
│   └── web/                Single-page HTML/CSS/JS dashboard
├── src/
│   ├── prediction/         Stage 1 – data preparation, baselines, MLP training, inference, PyTorch comparison
│   ├── fuzzy/              Stage 2 – Mamdani FIS (risk.py) and membership/surface plots
│   ├── optimizer/          Stage 3 – fitness, repair, baselines, DEAP GA, allocate_hours()
│   └── scheduler/          Stage 4 – rule-based build_schedule()
├── genetic_algorithm/      GA prototype, brute-force validation, convergence plots, concavity check
├── migrations/             Alembic environment and schema versions
├── tests/                  API integration tests and stage acceptance tests
├── docs/                   Data dictionary, per-member decisions logs, worked example, figures
├── data/                   cleaned_features.csv (derived training data)
├── models/                 predictor_v1.joblib (fitted preprocessing + MLP pipeline)
├── Education_numerical.csv Source dataset (10,000 students × 22 columns)
├── smoke_test.py           End-to-end pipeline check
├── REPORT.md               Full project report
├── HOW_TO_RUN.md           Step-by-step local setup guide
├── alembic.ini             Alembic configuration
├── requirements.txt        Dependencies
└── .env.example            Configuration template
```

---

## 17. Project Team

**Group 4 — Essentials of Artificial Intelligence**

| Member | Responsibility |
|---|---|
| Avishka Hashara | Stage 1 – neural-network prediction engine, data layer, system integration, web application and persistence |
| Melani | Stage 2 – Mamdani fuzzy-inference risk assessor |
| Janithi | Stage 3 – genetic-algorithm study-hour optimiser |
| Panchami | Stage 4 – rule-based scheduler and user interface |

Design rationale for each stage is recorded in [docs/decisions_member1.md](docs/decisions_member1.md), [docs/decisions_member2.md](docs/decisions_member2.md) and [docs/decisions_member3.md](docs/decisions_member3.md).

This repository is an academic coursework submission. It is not intended for deployment with real student data without the safeguards described in §13.

---

## 18. References

Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380.

Coello Coello, C. A. (2002). Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: A survey of the state of the art. *Computer Methods in Applied Mechanics and Engineering, 191*(11–12), 1245–1287.

Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques. *Psychological Science in the Public Interest, 14*(1), 4–58.

Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. In *Foundations of Genetic Algorithms 2* (pp. 187–202). Morgan Kaufmann.

Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research, 13*, 2171–2175.

Giarratano, J. C., & Riley, G. D. (2005). *Expert Systems: Principles and Programming* (4th ed.). Thomson Course Technology.

Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.

Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), 1–21.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

Kruger, J., & Dunning, D. (1999). Unskilled and unaware of it: How difficulties in recognizing one's own incompetence lead to inflated self-assessments. *Journal of Personality and Social Psychology, 77*(6), 1121–1134.

Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies, 7*(1), 1–13.

Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs* (3rd ed.). Springer.

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Prechelt, L. (1998). Early stopping – but when? In G. B. Orr & K.-R. Müller (Eds.), *Neural Networks: Tricks of the Trade* (pp. 55–69). Springer.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*, 533–536.

Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

Slade, S., & Prinsloo, P. (2013). Learning analytics: Ethical issues and dilemmas. *American Behavioral Scientist, 57*(10), 1510–1529.

Warner, J., Sexauer, J., et al. (2019). *scikit-fuzzy: Fuzzy logic toolkit for SciPy* (Version 0.4.2) [Computer software]. https://github.com/scikit-fuzzy/scikit-fuzzy

Zadeh, L. A. (1965). Fuzzy sets. *Information and Control, 8*(3), 338–353.

Group 4. (2026). *Implementation Handbook – Smart Learning and Performance Optimization System* (Version 1.0). Internal project document.
