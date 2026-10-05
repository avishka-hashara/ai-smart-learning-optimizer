# Decisions Log - Member 1 (Prediction Engine & Integration)

| Decision | Chosen Value | Reasoning |
| :--- | :--- | :--- |
| **Unit of prediction** | 1 row per student per subject | Stage 2 requires one predicted mark per subject, so the dataset was reshaped (melted) to match this granularity. |
| **Target Variable** | `target_score` | Derived from the subject-specific score columns; represents the final exam mark being predicted. |
| **Hidden layers** | 64 then 32 neurons | Wide enough to capture interactions between features without severely overfitting our dataset size. |
| **Activation** | ReLU | Computationally cheap and avoids the vanishing gradient problem associated with sigmoids. Output layer remains linear for regression. |
| **Solver** | Adam | Provides adaptive learning rates per parameter, making it robust without requiring heavy manual tuning. |
| **Early stopping** | On (10% validation, patience 10) | Strictly necessary to prevent the network from simply memorizing the training rows. |
| **max_iter** | 1000 | Set deliberately high so that early stopping determines the halt point, not an arbitrary iteration cap. |
| **Evaluation protocol** | Single held-out test set (20%) | Provides an honest headline figure for the final MAE reporting on completely unseen data. |
| **Clipping the output** | Clipped to 0–100 in wrapper | Essential integration safety measure; prevents out-of-bounds predictions from crashing Stage 2's fuzzy universe. |
| **Random seed** | 42 | Used across data splitting, confidence generation, and model initialization to guarantee exactly reproducible report figures. |
| **Categorical Encoding** | `handle_unknown='ignore'` | Selected for the OneHotEncoder to ensure the production pipeline does not crash if a student inputs a subject (e.g., 'Databases') not seen in the training data. |