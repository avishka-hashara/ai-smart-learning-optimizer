import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Ensure the src folder is in the path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction.predict import predict_scores
from src.fuzzy.risk import assess_risk
from src.optimizer.allocate import allocate_hours
from src.scheduler.schedule import build_schedule

st.set_page_config(page_title="Smart Learning - Baseline UI", layout="wide")

st.title("🧠 Smart Learning System - Baseline UI")
st.info("🟢 Stage 1 (Prediction) is LIVE. 🟡 Stages 2-4 are currently using Integration Stubs.")

# 1. Setup Default Mock Data (The Overconfident Student)
st.subheader("1. Student Profile (Edit data below)")

default_subjects = ["Databases", "Networks", "Statistics", "Maths", "Ethics"]
default_data = {
    'subject': default_subjects,
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
    'difficulty_indicator': [2, 1, 3, 1, 2],
    # Confidence is used by Stage 2
    'student_confidence': [9.0, 5.0, 6.0, 7.0, 5.0] 
}

df = pd.DataFrame(default_data)

# Let the user edit the dataframe in the UI
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# 2. Run Pipeline Button
if st.button("🚀 Run Pipeline", type="primary"):
    with st.spinner("Running 4-Stage Pipeline..."):
        
        # Split features for Stage 1 and confidence for Stage 2
        features_df = edited_df.drop(columns=['student_confidence'])
        confidence = edited_df['student_confidence'].values
        subjects = edited_df['subject'].tolist()
        
        # --- STAGE 1 ---
        predicted_marks = predict_scores(features_df)
        
        # --- STAGE 2 ---
        risk_results = assess_risk(predicted_marks, confidence)
        
        # --- STAGE 3 ---
        budget = 40.0
        min_h = np.full(len(subjects), 2.0)
        max_h = np.full(len(subjects), 15.0)
        allocation_results = allocate_hours(risk_results['priority'], predicted_marks, budget, min_h, max_h)
        
        # --- STAGE 4 ---
        schedule = build_schedule(
            hours=allocation_results['hours'],
            subjects=subjects,
            days=7,
            daily_limits=8.0,
            peak_window="09:00-11:00"
        )
        
        # 3. Display Results
        st.success("Pipeline executed successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.header("🎯 Stage 1: Predictions")
            st.dataframe(pd.DataFrame({
                'Subject': subjects,
                'Predicted Mark': np.round(predicted_marks, 2)
            }), use_container_width=True)
            
        with col2:
            st.header("⚠️ Stage 2: Risk (Stub)")
            st.dataframe(pd.DataFrame({
                'Subject': subjects,
                'Priority': np.round(risk_results['priority'], 2),
                'Risk Label': risk_results['label']
            }), use_container_width=True)
            
        with col3:
            st.header("⏱️ Stage 3: Allocation (Stub)")
            st.dataframe(pd.DataFrame({
                'Subject': subjects,
                'Allocated Hours': np.round(allocation_results['hours'], 2)
            }), use_container_width=True)
            
        st.header("📅 Stage 4: Generated Schedule (Stub)")
        st.dataframe(pd.DataFrame(schedule), use_container_width=True)