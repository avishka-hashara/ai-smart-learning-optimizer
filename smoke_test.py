import pandas as pd
import numpy as np
from src.prediction.predict import predict_scores
from src.fuzzy.risk import assess_risk
from src.optimizer.allocate import allocate_hours
from src.scheduler.schedule import build_schedule

def test_profile(profile_name, subjects, features_df, confidence):
    print(f"\n{'='*50}")
    print(f"RUNNING PROFILE: {profile_name}")
    print(f"{'='*50}")
    
    print("\n--- STAGE 1: Prediction ---")
    predicted_marks = predict_scores(features_df)
    print(f"Predicted Marks: {np.round(predicted_marks, 2)}")
    
    print("\n--- STAGE 2: Fuzzy Risk Assessment ---")
    risk_results = assess_risk(predicted_marks, confidence)
    print(f"Priorities: {np.round(risk_results['priority'], 2)}")
    print(f"Labels: {risk_results['label']}")
    
    print("\n--- STAGE 3: Genetic Algorithm Allocation ---")
    budget = 40.0
    min_h = np.full(len(subjects), 2.0)
    max_h = np.full(len(subjects), 15.0)
    allocation_results = allocate_hours(risk_results['priority'], predicted_marks, budget, min_h, max_h)
    print(f"Allocated Hours: {np.round(allocation_results['hours'], 2)}")
    
    print("\n--- STAGE 4: Scheduling ---")
    schedule = build_schedule(
        hours=allocation_results['hours'],
        subjects=subjects,
        days=7,
        daily_limits=8.0,
        peak_window="09:00-11:00"
    )
    print(f"Schedule generated with {len(schedule)} blocks.")
    print(f"First block sample: {schedule[0]}")
    print(f"✅ {profile_name} passed!")

if __name__ == "__main__":
    # Standard agreed-upon subjects for testing
    subjects = ["Databases", "Networks", "Statistics", "Maths", "Ethics"]
    
    # We pass empty mock DataFrames because the Stage 1 stub currently ignores features
    mock_features = pd.DataFrame({
        'subject': subjects,
        'study_hours_week': [12.0, 15.0, 8.0, 20.0, 5.0],
        'quiz_attempts': [3, 5, 2, 8, 1],
        'consecutive_logins': [4, 10, 2, 14, 1],
        'material_access_freq': [10, 25, 5, 40, 2],
        'improvement_rate': [1.1, 1.5, 0.9, 2.1, 0.8],
        'score_consistency': [0.5, 0.7, 0.4, 0.8, 0.3],
        'time_per_question': [1.5, 1.2, 2.0, 1.0, 2.5],
        'class_size': [30] * 5,
        'resource_index': [50.0, 75.0, 40.0, 90.0, 30.0],
        'faculty_ratio': [0.1] * 5,
        'peak_study_time': [14, 10, 18, 9, 20],
        'weekend_study_ratio': [0.2, 0.5, 0.1, 0.8, 0.0],
        'difficulty_indicator': [2, 1, 3, 1, 2]
    })
    
    # 1. The Overconfident Student (Demo case: Failing a subject but rating it high)
    confidence_1 = np.array([9.0, 5.0, 6.0, 7.0, 5.0])
    test_profile("The Overconfident Student", subjects, mock_features, confidence_1)
    
    # 2. The Anxious High Achiever (Strong subject but rating it low)
    confidence_2 = np.array([3.0, 8.0, 7.0, 8.0, 9.0])
    test_profile("The Anxious High Achiever", subjects, mock_features, confidence_2)
    
    # 3. The Uniformly Weak Student (All subjects weak, all confidences low)
    confidence_3 = np.array([2.0, 3.0, 2.0, 4.0, 2.0])
    test_profile("The Uniformly Weak Student", subjects, mock_features, confidence_3)
    
    print("\n🚀 ALL SMOKE TESTS COMPLETED SUCCESSFULLY. PIPELINE IS INTEGRATED.")