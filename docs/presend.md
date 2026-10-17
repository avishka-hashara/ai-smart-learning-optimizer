# Presentation Script — Member 1 (Stage 1: Neural Network Prediction Engine)

Target length: about 8 minutes (about 1 minute per slide).
Each slide below has: **On slide** (what to put on the slide) and **Say** (what to speak).
Speak slowly. Point at the diagram/table while talking. Do not read the slide word for word.

---

## Slide 1 — Stage 1: Neural Network Prediction Engine

**On slide**
- Member 1 — [Your Name]
- Predicts each student's mark per subject (0–100)
- Output feeds Fuzzy Logic → Genetic Algorithm → Scheduler
- Visual: 4 boxes — **Neural Network** → Fuzzy Logic → Genetic Algorithm → Scheduler (first box highlighted)

**Say**
> "Good morning. I'm [Your Name], and I built Stage 1 of our system — the neural network prediction engine.
> Our system has four stages. My model comes first: it takes a student's learning behaviour and predicts the mark they are likely to get in each subject, on a scale of 0 to 100.
> That prediction is the foundation for everything after it. The fuzzy logic stage uses it to decide study priority, the genetic algorithm uses it to allocate study hours, and the scheduler turns that into a timetable. So if my prediction is wrong, the error flows through the whole pipeline — that's why I focused heavily on correct data handling and honest evaluation."

---

## Slide 2 — Problem Definition

**On slide**
- Task: supervised regression (marks are continuous, 0–100)
- Input: student learning behaviour features
- Output: predicted mark for each subject
- Dataset: 10,000 students × 4 subjects = 40,000 rows
- Subjects: Maths, Science, English, History

**Say**
> "I framed this as a supervised regression problem. Marks are continuous values between 0 and 100, and the next stage needs an actual number, not a category like pass or fail — so regression, not classification.
> The dataset has 10,000 students, with one row per student and four subject scores. But the rest of our system works per subject, so the unit I predict is one student in one subject. After reshaping, that gives 40,000 training examples."

---

## Slide 3 — Data Preparation & Leakage Prevention

**On slide**
- Removed leakage columns: `overall_avg`, `predicted_score`, `science_math_diff`, `eng_hist_ratio`
- Rule: drop anything not known before the exam or derived from the target
- Reshaped data: one row per (student, subject)
- Created `difficulty_indicator`: knowledge gap split into 3 levels (low / medium / high)
- Confidence column generated for Stage 2 — **not used as a model input** (it comes from the target)

**Say**
> "The most important step before training was a data leakage audit. Data leakage means information about the answer sneaks into the inputs, which gives a model that looks excellent in testing but fails in real use.
> For example, `overall_avg` is the average of the four subject scores — it literally contains the answer. So I removed it, along with three other columns derived from the scores. My rule was simple: if a value would not be known before the exam, or it's calculated from the target, it is removed.
> I then reshaped the data to one row per student per subject, and created a difficulty indicator by splitting the knowledge gap into three equal-sized groups: low, medium and high.
> The dataset had no confidence column, but Stage 2 needs one, so I generated a synthetic confidence score. I deliberately did not use it as a model input, because it was generated from the real score — using it would be leakage."

---

## Slide 4 — Features & Preprocessing

**On slide**
- 13 numeric features → StandardScaler
- 1 categorical feature (subject) → One-Hot Encoding
- Total inputs: 17
- Scikit-learn Pipeline: scaler fitted on training data only; whole model saved as one file → identical preprocessing at prediction time
- Visual: Raw data → Scaler + One-Hot → MLP → Predicted mark

**Say**
> "The model uses 13 numeric features — things like study hours per week, quiz attempts, login streaks and improvement rate — plus the subject.
> Numeric features are standardised to mean zero and standard deviation one. Neural networks train poorly when features are on very different scales, because large values dominate the gradients. Scaling makes gradient descent faster and more stable.
> Subject is a category with no natural order, so I one-hot encoded it. If I had used numbers like 1, 2, 3, the model would think Maths is 'less than' Science, which is meaningless. That gives 17 inputs in total.
> I wrapped preprocessing and the model in a single scikit-learn Pipeline. This has two benefits: the scaler only learns from training data, so no test information leaks in; and the whole thing is saved as one file, so the app applies exactly the same preprocessing as during training."

---

## Slide 5 — Neural Network Architecture

**On slide**
- Model: Multi-Layer Perceptron (MLPRegressor)
- Layers: 17 inputs → 64 (ReLU) → 32 (ReLU) → 1 output (linear)
- Total parameters: 3,265
- Optimiser: Adam (learning rate 0.001)
- Loss: Mean Squared Error + L2 regularisation
- Batch size: 200
- Early stopping: 10% validation, stops after 10 epochs with no improvement
- Random seed: 42 (reproducible)
- Visual: network diagram 17 → 64 → 32 → 1

**Say**
> "The model is a multi-layer perceptron — a fully connected feed-forward neural network.
> It has 17 inputs, two hidden layers of 64 and 32 neurons, and one output neuron. That's 3,265 trainable parameters. The funnel shape — 64 then 32 — gives enough capacity to learn feature interactions, while staying small enough not to overfit 32,000 training rows.
> Hidden layers use ReLU activation. ReLU avoids the vanishing gradient problem of sigmoid and tanh, and is cheap to compute. The output layer is linear, because this is regression and the output must be a real number.
> Training minimises mean squared error, with a small L2 penalty for regularisation. I used the Adam optimiser, which combines momentum with a separate adaptive learning rate for each weight, so it converges well without manual tuning. Weights are updated in mini-batches of 200.
> To prevent overfitting, I used early stopping: 10% of the training data is held out as validation, and training stops once validation performance hasn't improved for 10 epochs. A fixed random seed of 42 makes every result reproducible."

---

## Slide 6 — Evaluation & Results

**On slide**
- Train/test split: 80% / 20% (32,000 / 8,000 rows)
- Metrics: MAE, RMSE, R²
- Table:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Global mean | 8.668 | 10.922 | 0.000 |
| Student average | 10.705 | 13.442 | −0.515 |
| Linear Regression | 8.133 | 10.281 | 0.114 |
| **MLP (ours)** | **8.177** | **10.342** | **0.103** |

- Visual: `docs/figures/loss_curve.png`

**Say**
> "I split the data 80/20: 32,000 rows for training and 8,000 for testing. The test set was used only once, for these final numbers, so they're an honest estimate of performance on unseen students.
> I report three metrics. MAE is the average error in marks — our model is about 8 marks off on average. RMSE penalises large errors more heavily. R² is the share of variance explained, where zero means no better than always predicting the average.
> A model number on its own means nothing, so I compared against three baselines: always predicting the global average, using the student's own average from other subjects, and linear regression.
> The MLP achieves an MAE of 8.18 and an R² of 0.10. It clearly beats the student-average baseline and improves on the global mean, and it performs on par with linear regression. The loss curve on the right shows training converging smoothly before early stopping."

---

## Slide 7 — Result Analysis

**On slide**
- MLP performs about the same as Linear Regression
- Reason 1: Subject scores are almost uncorrelated with each other (≈ 0.00–0.03)
- Reason 2: Subject is the main predictor; other features add very little
- No overfitting: Train MAE 8.175 vs Test MAE 8.177
- Conclusion: the limit is weak signal in the dataset, not the model
- Visual: `pred_vs_actual.png` and `residuals.png`

**Say**
> "I want to be transparent about these results. The R² is modest, and the neural network does not beat linear regression. I investigated why.
> First, the subject scores are almost completely uncorrelated — around 0.00 to 0.03. Knowing a student's Maths mark tells you nothing about their English mark. That's also why the student-average baseline does worse than the global mean.
> Second, I measured permutation importance. Subject is by far the strongest predictor; every other feature contributes very little. So the model mainly learns the typical score for each subject.
> Third, it is not overfitting: training error is 8.175 and test error is 8.177 — practically identical.
> A neural network's advantage is capturing non-linear relationships. When the data doesn't contain them, it converges to roughly the linear solution, which is exactly what we see. You can also see it in the prediction-versus-actual plot: predictions are compressed toward the mean. The residuals are centred on zero, so there is no systematic bias.
> So the limitation is the signal in the dataset, not the model. I chose to report that honestly rather than tune until the test score looked good."

---

## Slide 8 — Integration, Limitations & Future Work

**On slide**
- Integration: model loaded once and cached; input columns validated; output clipped to 0–100 to match the fuzzy input range
- Unseen subjects: predicted as the average over the 4 trained subjects (≈ 69 instead of an invalid ≈ 45)
- Limitations: unseen subjects get a neutral estimate, not a subject-specific one; split is by row, not by student
- Future work: real LMS data, group split by student, k-fold cross-validation, hyperparameter tuning

**Say**
> "For integration, I wrote a prediction wrapper used by the application. It loads the model once and caches it, so the app stays fast. It checks that all required input columns are present and gives a clear error if not. And it clips the output to 0–100, because that's the input range of the fuzzy logic stage — an out-of-range value would break the next stage.
> One important case: the model was trained on four subjects — Maths, Science, English and History — but users can add any subject. An unseen subject one-hot encodes to all zeros, which never appeared in training, so the raw model output was about 45 for every new subject, regardless of the name. That's an out-of-distribution input, not a real prediction. I fixed this with a fallback: for an unseen subject, the wrapper predicts the student as if it were each of the four known subjects and averages the results. That gives a neutral estimate of about 69 that still responds to the student's other features, and it also matches subject names regardless of capitalisation.
> There are limitations I want to be clear about. For new subjects this is a neutral estimate, not subject-specific knowledge — the real fix is training data that covers those subjects. And I split data by row rather than by student, so one student's other subjects can appear in training; the effect is small here because subjects are uncorrelated, but grouping by student is the more rigorous approach.
> For future work: train on real LMS data that covers the app's subjects, split by student, use k-fold cross-validation, and run a hyperparameter search over layer sizes, regularisation and learning rate."

---

## Slide 9 — Thank You / Questions

**Say**
> "To summarise: I built a leakage-free data pipeline, a neural network predictor, and evaluated it honestly against baselines. It's integrated as the first stage of our system. Thank you — I'm happy to take questions."

---

## Backup Slide — PyTorch Comparison (show only if asked)

**On slide**
- Same architecture rebuilt in PyTorch: 17 → 64 → 32 → 1
- Explicit training loop: zero_grad → forward → backward → step
- Comparable results to the scikit-learn MLP

**Say**
> "I also re-implemented the same architecture in PyTorch, where the training loop is written explicitly. Each step clears old gradients, does a forward pass, computes gradients with backpropagation, and updates weights with Adam. It produced comparable results, which confirms the scikit-learn model behaves as expected. The PyTorch version uses full-batch training without early stopping, so it's a validation exercise rather than the production model."

---

## Likely Questions — Short Answers

| Question | Answer |
|---|---|
| Why a neural network instead of linear regression? | It can model non-linear interactions. I kept linear regression as a baseline to test whether that helps — here it didn't, because the data has no strong non-linear signal. |
| How do you know it's not overfitting? | Train MAE 8.175 vs test MAE 8.177, plus early stopping and L2 regularisation. |
| What is an epoch? / a batch? | Epoch = one full pass over the training data. Batch = 200 rows used for one weight update. |
| Why Adam? | Momentum plus a per-weight adaptive learning rate; converges well without manual tuning. |
| Why ReLU? | No vanishing gradient for positive inputs, and cheap to compute. |
| Why a linear output? | Regression output must be an unrestricted real number. |
| Why scale features? | Faster, more stable gradient descent; stops large-valued features dominating. |
| Why one-hot encode subject? | No natural order between subjects; numbers would imply a false ranking. |
| What is data leakage? | Target information getting into the inputs, giving over-optimistic results. |
| Why clip the output? | The fuzzy stage expects 0–100. |
| What does the saved file contain? | The full pipeline: scaler means and standard deviations, encoder categories, network weights and biases. |
| How many parameters? | 3,265. |
| How do you predict a subject the model never saw (e.g. Databases)? | An unseen subject would one-hot encode to all zeros, which is out of distribution (raw output ≈ 45 for any name). The wrapper instead predicts it as each of the 4 known subjects and averages → ≈ 69, a neutral prior. Real fix: training data for those subjects, or update from the student's actual marks over time. |
| Why average and not just pick one subject? | No evidence which known subject is closest; the average is the least biased choice. A future option is letting the user pick a category (technical / theory) to map to. |
| How does your output affect other stages? | Fuzzy logic uses it for priority and the over/under-confidence label; the GA uses 100 − predicted as the marks still available to gain. |

## Do Not Say
- "The MLP outperforms all baselines." (It does not beat linear regression.)
- "It stopped at epoch 47." (Not verified — leave it out unless you re-run and check.)
