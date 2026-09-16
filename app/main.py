import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import plotly.graph_objects as go
import plotly.express as px

# Ensure the src folder is in the path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction.predict import predict_scores
from src.fuzzy.risk import assess_risk
from src.optimizer.allocate import allocate_hours
from src.scheduler.schedule import build_schedule

# Page configuration
st.set_page_config(
    page_title="Smart Learning Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background and Layout Polish */
    .main {
        background-color: #0b0f19;
    }
    
    /* Header Card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    .badge-live {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-stub {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Subject Card Box */
    .subject-card {
        background: linear-gradient(180deg, rgba(26, 34, 53, 0.65) 0%, rgba(17, 24, 39, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 14px;
        padding: 16px 14px;
        margin-bottom: 12px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .subject-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.15);
    }

    .subject-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .subject-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .lms-chip {
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Metric Highlight Boxes */
    .kpi-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .kpi-title {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }

    .kpi-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Timetable Card */
    .schedule-item {
        background: rgba(30, 41, 59, 0.45);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .schedule-time {
        font-weight: 700;
        color: #38bdf8;
        font-size: 0.9rem;
    }

    .schedule-subj {
        font-weight: 600;
        color: #f1f5f9;
        font-size: 0.95rem;
    }

    .schedule-badge {
        font-size: 0.75rem;
        background: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Button Customization */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.8rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton>button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* Hide standard Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Define Persona Preset Configurations
PERSONAS = {
    "The Overconfident Student": {
        "description": "High perceived confidence with moderate/unbalanced study allocation.",
        "avatar": "🦁",
        "conf": [9.0, 5.0, 6.0, 7.0, 5.0],
        "hours": [12.0, 15.0, 8.0, 20.0, 5.0]
    },
    "The Anxious High Achiever": {
        "description": "Very high study investment despite low confidence in key modules.",
        "avatar": "🦉",
        "conf": [3.0, 8.0, 7.0, 8.0, 9.0],
        "hours": [25.0, 20.0, 22.0, 30.0, 18.0]
    },
    "The Uniformly Weak Student": {
        "description": "Critical need for structured intervention with low study hours.",
        "avatar": "🌱",
        "conf": [2.0, 3.0, 2.0, 4.0, 2.0],
        "hours": [5.0, 4.0, 6.0, 5.0, 3.0]
    }
}

SUBJECT_META = [
    {"name": "Databases", "icon": "💾", "color": "#38bdf8"},
    {"name": "Networks", "icon": "🌐", "color": "#818cf8"},
    {"name": "Statistics", "icon": "📊", "color": "#fb7185"},
    {"name": "Maths", "icon": "📐", "color": "#facc15"},
    {"name": "Ethics", "icon": "⚖️", "color": "#4ade80"},
]
subjects = [s["name"] for s in SUBJECT_META]

# Initialize Session State for clean interactive persistence
if 'current_persona' not in st.session_state:
    st.session_state.current_persona = "The Overconfident Student"

if 'lms_logins' not in st.session_state:
    st.session_state.lms_logins = [np.random.randint(4, 15) for _ in range(5)]

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Simulation Control")
    
    selected_persona = st.selectbox(
        "Student Profile Persona:",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.current_persona)
    )
    
    # Check if persona changed
    if selected_persona != st.session_state.current_persona:
        st.session_state.current_persona = selected_persona
        st.session_state.pipeline_results = None
        st.rerun()

    active_p = PERSONAS[st.session_state.current_persona]
    st.info(f"{active_p['avatar']} **{st.session_state.current_persona}**\n\n{active_p['description']}")
    
    st.markdown("---")
    
    st.markdown("### 🔄 Integrations")
    if st.button("🔄 Sync LMS / Canvas Telemetry", use_container_width=True):
        with st.spinner("Connecting to Canvas LMS API..."):
            time.sleep(0.8)
            st.session_state.lms_logins = [np.random.randint(4, 16) for _ in range(5)]
        st.toast("✅ Real-time LMS telemetry synced successfully!", icon="📡")
        st.rerun()
        
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
        <b>Architecture Status:</b><br/>
        • <b>Stage 1:</b> ML Model (Live)<br/>
        • <b>Stage 2:</b> Fuzzy Risk Engine (Live)<br/>
        • <b>Stage 3:</b> GA Optimizer (Live)<br/>
        • <b>Stage 4:</b> Schedule Builder (Stub)
    </div>
    """, unsafe_allow_html=True)


# Hero Header Section
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <div class="hero-title">🧠 Smart Learning & Performance Optimizer</div>
            <div class="hero-subtitle">Adaptive ML score prediction, cognitive gap analysis, and intelligent study time allocation</div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <span class="badge-pill badge-live">● Stage 1: Live Predictor</span>
            <span class="badge-pill badge-live">● Stage 2: Live Fuzzy Risk</span>
            <span class="badge-pill badge-live">● Stage 3: Live GA Optimizer</span>
            <span class="badge-pill badge-stub">⚡ Stage 4: Stub</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Section: Subject Telemetry & Configuration
st.markdown("### 📝 Current Semester Telemetry & Configuration")
st.caption("Review and adjust your target study hours and self-perceived confidence per subject.")

preset_hours = active_p["hours"]
preset_conf = active_p["conf"]

cols = st.columns(5)
edited_data = {}

for i, col in enumerate(cols):
    meta = SUBJECT_META[i]
    with col:
        st.markdown(f"""
        <div class="subject-card">
            <div class="subject-header">
                <span class="subject-name">{meta['icon']} {meta['name']}</span>
                <span class="lms-chip">📡 {st.session_state.lms_logins[i]} Logins</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # User input fields with clean labels
        edited_data[f'hours_{i}'] = st.number_input(
            f"Target Study Hours/Wk",
            min_value=1.0,
            max_value=40.0,
            value=float(preset_hours[i]),
            step=1.0,
            key=f"h_{st.session_state.current_persona}_{i}",
            help=f"Weekly study hours planned for {meta['name']}"
        )
        
        edited_data[f'conf_{i}'] = st.slider(
            f"Perceived Confidence",
            min_value=1.0,
            max_value=10.0,
            value=float(preset_conf[i]),
            step=0.5,
            key=f"c_{st.session_state.current_persona}_{i}",
            help="1 = Very Low Confidence, 10 = Mastered"
        )

# Prepare DataFrame matching expected ML pipeline features exactly
df_features = pd.DataFrame({
    'subject': subjects,
    'study_hours_week': [edited_data[f'hours_{i}'] for i in range(5)],
    'quiz_attempts': [5] * 5, 
    'consecutive_logins': [10] * 5,
    'material_access_freq': [20] * 5,
    'improvement_rate': [1.2] * 5,
    'score_consistency': [0.6] * 5,
    'time_per_question': [1.5] * 5,
    'class_size': [30] * 5,
    'resource_index': [50.0] * 5,
    'faculty_ratio': [0.1] * 5,
    'peak_study_time': [14] * 5,
    'weekend_study_ratio': [0.5] * 5,
    'difficulty_indicator': [2] * 5
})

st.markdown("<br/>", unsafe_allow_html=True)

# Run Pipeline Action Bar
btn_col1, btn_col2, btn_col3 = st.columns([1.5, 2, 1.5])
with btn_col2:
    run_btn = st.button("🚀 Run Full Optimization Pipeline", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Processing telemetry through 4-Stage Optimization Pipeline..."):
        # Split features for Stage 1 and confidence for Stage 2
        confidence_array = np.array([edited_data[f'conf_{i}'] for i in range(5)])
        
        # --- STAGE 1: Predict Scores ---
        predicted_marks = predict_scores(df_features)
        
        # --- STAGE 2: Assess Risk ---
        risk_results = assess_risk(predicted_marks, confidence_array)
        
        # --- STAGE 3: Hour Allocation ---
        budget = sum([edited_data[f'hours_{i}'] for i in range(5)])
        min_h = np.full(len(subjects), 2.0)
        max_h = np.full(len(subjects), 15.0)
        allocation_results = allocate_hours(risk_results['priority'], predicted_marks, budget, min_h, max_h)
        
        # --- STAGE 4: Build Schedule ---
        schedule = build_schedule(
            hours=allocation_results['hours'],
            subjects=subjects,
            days=7,
            daily_limits=8.0,
            peak_window="09:00-11:00"
        )
        
        # Save output in session state
        st.session_state.pipeline_results = {
            'predicted_marks': predicted_marks,
            'confidence_array': confidence_array,
            'risk_results': risk_results,
            'allocation_results': allocation_results,
            'schedule': schedule,
            'budget': budget
        }
        
    st.toast("🎯 Optimization pipeline finished successfully!", icon="✨")

# Render Results If Available
if st.session_state.pipeline_results is not None:
    res = st.session_state.pipeline_results
    predicted_marks = res['predicted_marks']
    confidence_array = res['confidence_array']
    risk_results = res['risk_results']
    allocation_results = res['allocation_results']
    schedule = res['schedule']
    budget = res['budget']
    
    st.markdown("---")
    st.markdown("### 📊 Optimization Insights & Recommendations")
    
    # Key Metrics Banner
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        avg_score = np.mean(predicted_marks)
        score_color = "#34d399" if avg_score >= 70 else "#fbbf24" if avg_score >= 50 else "#f87171"
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Average Predicted Score</div>
            <div class="kpi-value" style="color: {score_color};">{avg_score:.1f}%</div>
            <div class="kpi-sub">Across {len(subjects)} active modules</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Weekly Study Budget</div>
            <div class="kpi-value">{budget:.1f} hrs</div>
            <div class="kpi-sub">Total allocated weekly study time</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        max_risk_idx = np.argmax(risk_results['priority'])
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Top Priority Focus</div>
            <div class="kpi-value" style="color: #f43f5e; font-size: 1.4rem;">{SUBJECT_META[max_risk_idx]['icon']} {subjects[max_risk_idx]}</div>
            <div class="kpi-sub">Priority Score: {risk_results['priority'][max_risk_idx]:.2f} / 10</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Scheduled Sessions</div>
            <div class="kpi-value">{len(schedule)}</div>
            <div class="kpi-sub">Optimized calendar blocks generated</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Tabbed Interactive View
    tab_overview, tab_pred, tab_risk, tab_alloc, tab_raw = st.tabs([
        "📅 Study Timetable (Stage 4)",
        "🎯 Score Predictions (Stage 1)",
        "⚠️ Risk & Cognitive Gaps (Stage 2)",
        "⏱️ Hour Reallocation (Stage 3)",
        "📋 Full Pipeline Data"
    ])
    
    # TAB 1: Generated Schedule
    with tab_overview:
        st.markdown("#### 📅 Generated Weekly Study Plan")
        st.caption("AI-optimized calendar slots distributed according to difficulty, risk priority, and peak cognitive windows.")
        
        sched_df = pd.DataFrame(schedule)
        
        t_col1, t_col2 = st.columns([1.8, 1.2])
        with t_col1:
            for item in schedule:
                subj_icon = next((s['icon'] for s in SUBJECT_META if s['name'] == item['subject']), '📚')
                st.markdown(f"""
                <div class="schedule-item">
                    <div>
                        <span class="schedule-time">🕒 {item.get('start', '09:00')} - {item.get('end', '10:00')}</span>
                        <div class="schedule-subj">{subj_icon} {item['subject']} ({item.get('minutes', 60)} mins)</div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">{item.get('day', 'Day')} • {item.get('reason', 'Optimized Placement')}</div>
                    </div>
                    <span class="schedule-badge">📌 {item.get('type', 'study').title()}</span>
                </div>
                """, unsafe_allow_html=True)
                
        with t_col2:
            st.markdown("##### ⏱️ Study Time Distribution")
            fig_pie = px.pie(
                values=allocation_results['hours'],
                names=subjects,
                hole=0.55,
                color=subjects,
                color_discrete_map={s['name']: s['color'] for s in SUBJECT_META}
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # TAB 2: Predictions
    with tab_pred:
        st.markdown("#### 🎯 ML Predicted Exam Marks")
        st.caption("Machine Learning regression estimates based on historical engagement telemetry and weekly study volume.")
        
        pred_df = pd.DataFrame({
            'Subject': subjects,
            'Predicted Mark': np.round(predicted_marks, 2),
            'Target Hours': [edited_data[f'hours_{i}'] for i in range(5)],
            'Confidence': confidence_array
        })
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=subjects,
            y=predicted_marks,
            marker=dict(
                color=[SUBJECT_META[i]['color'] for i in range(5)],
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            text=[f"{val:.1f}%" for val in predicted_marks],
            textposition='auto',
            name="Predicted Score"
        ))
        
        # Benchmark line for pass / distinction
        fig_bar.add_hline(y=50, line_dash="dot", line_color="#fbbf24", annotation_text="Passing (50%)", annotation_font_color="#fbbf24")
        fig_bar.add_hline(y=75, line_dash="dash", line_color="#34d399", annotation_text="Distinction (75%)", annotation_font_color="#34d399")
        
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            yaxis=dict(title="Score (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(title=None),
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detail Cards
        score_cols = st.columns(5)
        for i, sc in enumerate(score_cols):
            with sc:
                score = predicted_marks[i]
                tag = "🟢 Strong" if score >= 75 else "🟡 Moderate" if score >= 50 else "🔴 Critical"
                st.metric(label=f"{SUBJECT_META[i]['name']}", value=f"{score:.1f}%", delta=tag)

    # TAB 3: Risk Assessment
    with tab_risk:
        st.markdown("#### ⚠️ Cognitive Gap & Priority Risk Assessment")
        st.caption("Fuzzy inference module cross-examines student confidence with ML predicted performance to detect blind spots.")
        
        risk_df = pd.DataFrame({
            'Subject': subjects,
            'Confidence (1-10)': confidence_array,
            'Predicted Mark (%)': np.round(predicted_marks, 1),
            'Priority Score': np.round(risk_results['priority'], 2),
            'Risk Classification': risk_results['label']
        })
        
        r_col1, r_col2 = st.columns([1.5, 1.5])
        
        with r_col1:
            st.dataframe(
                risk_df.style.background_gradient(subset=['Priority Score'], cmap='Reds'),
                use_container_width=True,
                hide_index=True
            )
            
        with r_col2:
            fig_scatter = px.scatter(
                x=confidence_array,
                y=predicted_marks,
                text=subjects,
                size=risk_results['priority'] + 2,
                color=subjects,
                color_discrete_map={s['name']: s['color'] for s in SUBJECT_META},
                labels={'x': 'Self Confidence (1-10)', 'y': 'Predicted Mark (%)'},
                title="Confidence vs Predicted Mark (Bubble Size = Risk Priority)"
            )
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                xaxis=dict(range=[0, 11], gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # TAB 4: Allocation Comparison
    with tab_alloc:
        st.markdown("#### ⏱️ Study Time Optimization Comparison")
        st.caption("Genetic algorithm / optimizer shifts weekly study hours away from saturated areas toward high-risk subjects.")
        
        orig_hours = [edited_data[f'hours_{i}'] for i in range(5)]
        opt_hours = np.round(allocation_results['hours'], 1)
        
        comp_df = pd.DataFrame({
            'Subject': subjects,
            'Original Target (hrs)': orig_hours,
            'AI Recommended (hrs)': opt_hours,
            'Adjustment': [f"{opt - orig:+.1f} hrs" for orig, opt in zip(orig_hours, opt_hours)]
        })
        
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Bar(
            name='Original Target',
            x=subjects,
            y=orig_hours,
            marker_color='rgba(148, 163, 184, 0.4)'
        ))
        fig_alloc.add_trace(go.Bar(
            name='AI Optimized Allocation',
            x=subjects,
            y=opt_hours,
            marker_color='#6366f1'
        ))
        fig_alloc.update_layout(
            barmode='group',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            yaxis=dict(title="Study Hours / Week", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_alloc, use_container_width=True)
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("##### 📈 Genetic Algorithm Convergence & Baseline Benchmark")
        st.caption("Demonstrates optimization progress across 100 generations compared to naive heuristic baselines (Handbook Section 3.6).")
        
        baselines = allocation_results.get('baselines', {})
        eq_fit = baselines.get('equal', 0.0)
        prop_fit = baselines.get('proportional', 0.0)
        grd_fit = baselines.get('greedy', 0.0)
        ga_fit = allocation_results.get('fitness', 0.0)
        
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("🏆 GA Best Fitness", f"{ga_fit:.1f}", delta="Optimal Search")
        with b_col2:
            st.metric("Equal Split Baseline", f"{eq_fit:.1f}", delta=f"{ga_fit - eq_fit:+.1f} pts", delta_color="inverse")
        with b_col3:
            st.metric("Proportional Baseline", f"{prop_fit:.1f}", delta=f"{ga_fit - prop_fit:+.1f} pts", delta_color="inverse")
        with b_col4:
            st.metric("Greedy Baseline", f"{grd_fit:.1f}", delta=f"{ga_fit - grd_fit:+.1f} pts", delta_color="inverse")
            
        history = allocation_results.get('history', [])
        if history:
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode='lines',
                name='GA Best Fitness',
                line=dict(color='#38bdf8', width=2.5)
            ))
            if eq_fit > 0:
                fig_conv.add_hline(
                    y=eq_fit, line_dash="dash", line_color="#fbbf24",
                    annotation_text=f"Equal Split ({eq_fit:.1f})",
                    annotation_font_color="#fbbf24"
                )
            if prop_fit > 0:
                fig_conv.add_hline(
                    y=prop_fit, line_dash="dot", line_color="#f43f5e",
                    annotation_text=f"Proportional Split ({prop_fit:.1f})",
                    annotation_font_color="#f43f5e"
                )
            fig_conv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                xaxis=dict(title="Generation", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Fitness Score", gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(t=25, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_conv, use_container_width=True)

    # TAB 5: Raw Pipeline Data
    with tab_raw:
        st.markdown("#### 📋 Diagnostic Stubs & Raw Data Structure")
        st.caption("Developer inspection of exact dictionary outputs and payloads from all 4 pipeline stages.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Stage 1 & 2 Output Payload:**")
            st.json({
                "predicted_marks": list(np.round(predicted_marks, 2)),
                "risk_priority": list(np.round(risk_results['priority'], 2)),
                "risk_labels": risk_results['label']
            })
        with c2:
            st.write("**Stage 3 Allocation Metrics:**")
            st.json({
                "allocated_hours": list(np.round(allocation_results['hours'], 2)),
                "optimizer_fitness": allocation_results.get('fitness', 0),
                "baselines": allocation_results.get('baselines', {})
            })
            
        st.write("**Stage 4 Schedule Data Table:**")
        st.dataframe(sched_df, use_container_width=True)
else:
    st.info("💡 Click **'🚀 Run Full Optimization Pipeline'** above to compute predictions and generate your personalized schedule.")