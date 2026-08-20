import joblib
import numpy as np
import pandas as pd
import os

# Cache the model so we don't load it from disk on every single Streamlit rerun
MODEL_PATH = os.path.join("models", "predictor_v1.joblib")
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_scores(features_df):
    """
    Stage 1: Predicts subject scores based on student features.
    Input: DataFrame with one row per subject.
    Output: NumPy array of predicted marks (0-100).
    """
    model = load_model()
    
    # The exact columns the pipeline expects (derived from the training script)
    expected_columns = [
        'study_hours_week', 'quiz_attempts', 'consecutive_logins', 
        'material_access_freq', 'improvement_rate', 'score_consistency', 
        'time_per_question', 'class_size', 'resource_index', 
        'faculty_ratio', 'peak_study_time', 'weekend_study_ratio', 
        'difficulty_indicator', 'subject'
    ]
    
    # 1. Validate Columns (Fail early with a readable error)
    missing_cols = [col for col in expected_columns if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Stage 1 Error: Input DataFrame is missing required columns: {missing_cols}")
        
    # 2. Predict
    predictions = model.predict(features_df)
    
    # 3. Clip output to the strict 0-100 universe required by Stage 2
    return np.clip(predictions, 0, 100)