import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import plotly.graph_objects as go
import plotly.express as px

# Ensure the project root is in the Python path for reliable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import PERSONAS, SUBJECT_META, SUBJECTS, run_pipeline

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Learning & Performance Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. ULTRA-PREMIUM MODERN DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Typography & Deep Obsidian Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, .hero-title, .kpi-value {
        font-family: 'Outfit', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(49, 46, 129, 0.25) 0%, transparent 45%),
                    radial-gradient(circle at 85% 75%, rgba(88, 28, 135, 0.2) 0%, transparent 45%),
                    #0b0f19 !important;
        color: #f1f5f9;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Hero Banner with Specular Highlight */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px 34px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 0%, #e2e8f0 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.03em;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 750px;
    }

    /* Live Status Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    
    .badge-live {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
    }
    
    .badge-stub {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 8px #34d399;
        animation: pulseAnimation 2s infinite ease-in-out;
    }

    @keyframes pulseAnimation {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.5; }
    }

    /* Subject Configuration Cards */
    .subject-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 16px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
    }
    
    .subject-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.4);
        border-color: rgba(255, 255, 255, 0.18);
    }

    .subject-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .subject-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .lms-pill {
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 3px 8px;
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* KPI Summary Cards */
    .kpi-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.55) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease;
    }

    .kpi-box:hover {
        transform: translateY(-2px);
    }

    .kpi-label {
        font-size: 0.76rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }

    .kpi-desc {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Timetable Items */
    .schedule-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
    }

    .schedule-card:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Modern Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.85);
        padding: 7px 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94a3b8;
        padding: 8px 18px;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4) !important;
    }

    /* Hero CTA Button */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 2.2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.25s ease !important;
    }

    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(139, 92, 246, 0.6) !important;
    }

    /* Diagnosis Pill Badges */
    .diag-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .diag-overconfident {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .diag-anxious {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .diag-aligned {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Clean Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. STUDENT PERSONAS & SUBJECT CONFIGURATION
# -----------------------------------------------------------------------------
subjects = SUBJECTS

# -----------------------------------------------------------------------------
# 4. PERSISTENT STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'current_persona' not in st.session_state:
    st.session_state.current_persona = "The Overconfident Student"

if 'lms_logins' not in st.session_state:
    st.session_state.lms_logins = [np.random.randint(5, 16) for _ in range(len(SUBJECTS))]

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None

# -----------------------------------------------------------------------------
# 5. SIDEBAR: SIMULATION & ENVIRONMENT CONTROL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Simulation Control")
    
    selected_persona = st.selectbox(
        "Student Profile Persona:",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.current_persona)
    )
    
    # Auto-switch persona telemetry when selector changes
    if selected_persona != st.session_state.current_persona:
        st.session_state.current_persona = selected_persona
        st.session_state.pipeline_results = None
        st.rerun()

    active_p = PERSONAS[st.session_state.current_persona]
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); border-radius: 12px; padding: 14px; border: 1px solid rgba(255, 255, 255, 0.08); margin-top: 10px;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
            {active_p['avatar']} {st.session_state.current_persona}
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
            {active_p['description']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔄 Integrations")
    if st.button("🔄 Sync LMS / Canvas Telemetry", use_container_width=True):
        with st.spinner("Connecting to LMS API..."):
            time.sleep(0.6)
            st.session_state.lms_logins = [np.random.randint(6, 18) for _ in range(len(SUBJECTS))]
        st.toast("✅ Real-time telemetry synced successfully!", icon="📡")
        st.rerun()
        
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.6;">
        <b style="color: #94a3b8;">System Pipeline Status:</b><br/>
        • <b>Stage 1:</b> ML Predictor <span style="color: #34d399; font-weight:700;">● Live</span><br/>
        • <b>Stage 2:</b> Fuzzy Risk Engine <span style="color: #34d399; font-weight:700;">● Live</span><br/>
        • <b>Stage 3:</b> GA Optimizer <span style="color: #34d399; font-weight:700;">● Live</span><br/>
        • <b>Stage 4:</b> Scheduler <span style="color: #fbbf24; font-weight:700;">⚡ Stub</span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div class="hero-title">🧠 Smart Learning & Performance Optimizer</div>
            <div class="hero-subtitle">Multi-stage pipeline combining ML neural regression, Mamdani fuzzy risk reasoning, and Genetic Algorithm study time optimization.</div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <span class="badge-pill badge-live"><span class="pulse-dot"></span> Stage 1: Live Predictor</span>
            <span class="badge-pill badge-live"><span class="pulse-dot"></span> Stage 2: Live Fuzzy Risk</span>
            <span class="badge-pill badge-live"><span class="pulse-dot"></span> Stage 3: Live GA Optimizer</span>
            <span class="badge-pill badge-stub">⚡ Stage 4: Stub</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. TELEMETRY CONFIGURATION GRID
# -----------------------------------------------------------------------------
st.markdown("### 📝 Current Semester Telemetry & Self-Assessment")
st.caption("Adjust weekly target revision hours and perceived confidence ratings per subject to simulate adaptive reallocations.")

preset_hours = active_p["hours"]
preset_conf = active_p["conf"]

cols = st.columns(5)
edited_data = {}

for i, col in enumerate(cols):
    meta = SUBJECT_META[i]
    with col:
        st.markdown(f"""
        <div class="subject-card" style="border-top: 3px solid {meta['color']};">
            <div class="subject-header">
                <span class="subject-title">{meta['icon']} {meta['name']}</span>
                <span class="lms-pill">📡 {st.session_state.lms_logins[i]} Logins</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        edited_data[f'hours_{i}'] = st.number_input(
            f"Target Hours / Wk",
            min_value=1.0,
            max_value=40.0,
            value=float(preset_hours[i]),
            step=1.0,
            key=f"h_{st.session_state.current_persona}_{i}",
            help=f"Target study hours for {meta['name']}"
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


st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. PIPELINE EXECUTION ACTION BAR
# -----------------------------------------------------------------------------
btn_col1, btn_col2, btn_col3 = st.columns([1.5, 2, 1.5])
with btn_col2:
    run_btn = st.button("🚀 Run Full Optimization Pipeline", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Executing 4-Stage AI Optimization Pipeline..."):
        st.session_state.pipeline_results = run_pipeline(
            hours=[edited_data[f'hours_{i}'] for i in range(len(SUBJECTS))],
            confidence=[edited_data[f'conf_{i}'] for i in range(len(SUBJECTS))]
        )
        
    st.toast("🎯 Optimization pipeline completed successfully!", icon="✨")

# -----------------------------------------------------------------------------
# 9. RESULTS DASHBOARD
# -----------------------------------------------------------------------------
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
    
    # KPI Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        avg_score = float(np.mean(predicted_marks))
        sc_color = "#34d399" if avg_score >= 70 else "#fbbf24" if avg_score >= 50 else "#f87171"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Average Predicted Score</div>
            <div class="kpi-value" style="color: {sc_color};">{avg_score:.1f}%</div>
            <div class="kpi-desc">Across {len(subjects)} active modules</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Weekly Study Budget</div>
            <div class="kpi-value" style="color: #38bdf8;">{budget:.1f} hrs</div>
            <div class="kpi-desc">Total weekly revision budget</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        max_idx = int(np.argmax(risk_results['priority']))
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Top Priority Focus</div>
            <div class="kpi-value" style="color: #fb7185; font-size: 1.55rem;">
                {SUBJECT_META[max_idx]['icon']} {subjects[max_idx]}
            </div>
            <div class="kpi-desc">Priority: {risk_results['priority'][max_idx]:.2f} / 10</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        ga_fit = allocation_results.get('fitness', 0.0)
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">GA Objective Fitness</div>
            <div class="kpi-value" style="color: #a78bfa;">{ga_fit:.1f}</div>
            <div class="kpi-desc">Score gain over baselines</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TABBED DETAILED VIEWS
    # -------------------------------------------------------------------------
    tab_pred, tab_risk, tab_alloc, tab_sched, tab_raw = st.tabs([
        "🎯 Score Predictions (Stage 1)",
        "⚠️ Risk & Cognitive Gaps (Stage 2)",
        "⏱️ Hour Reallocation (Stage 3)",
        "📅 Study Timetable (Stage 4)",
        "📋 Full Pipeline Data"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: PREDICTIONS (STAGE 1)
    # -------------------------------------------------------------------------
    with tab_pred:
        st.markdown("#### 🎯 Machine Learning Predicted Exam Marks")
        st.caption("Multilayer Perceptron regression estimates derived from historical telemetry, login consistency, and baseline hours.")
        
        # Plotly Bar Chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=subjects,
            y=predicted_marks,
            marker=dict(
                color=[SUBJECT_META[i]['color'] for i in range(len(SUBJECTS))],
                line=dict(color='rgba(255,255,255,0.2)', width=1.5)
            ),
            text=[f"{val:.1f}%" for val in predicted_marks],
            textposition='auto',
            textfont=dict(size=14, color='#ffffff', family="Outfit"),
            hoverinfo="x+y"
        ))
        
        # Benchmark threshold lines
        fig_bar.add_hline(
            y=50, line_dash="dash", line_color="#fbbf24",
            annotation_text="Passing Benchmark (50%)", annotation_font_color="#fbbf24",
            annotation_position="top left"
        )
        fig_bar.add_hline(
            y=75, line_dash="dash", line_color="#34d399",
            annotation_text="Distinction Benchmark (75%)", annotation_font_color="#34d399",
            annotation_position="top left"
        )
        
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
            yaxis=dict(
                title="Predicted Exam Mark (%)",
                range=[0, 105],
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.1)"
            ),
            xaxis=dict(title=None, showgrid=False),
            margin=dict(t=35, b=20, l=20, r=20),
            height=380
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Subject Status Metric Cards
        score_cols = st.columns(5)
        for i, sc in enumerate(score_cols):
            with sc:
                score = predicted_marks[i]
                tag = "🟢 Distinction" if score >= 75 else "🟡 Passing" if score >= 50 else "🔴 At Risk"
                st.metric(
                    label=f"{SUBJECT_META[i]['icon']} {SUBJECT_META[i]['name']}",
                    value=f"{score:.1f}%",
                    delta=tag
                )

    # -------------------------------------------------------------------------
    # TAB 2: RISK & COGNITIVE GAPS (STAGE 2)
    # -------------------------------------------------------------------------
    with tab_risk:
        st.markdown("#### ⚠️ Fuzzy Cognitive Gap & Priority Risk Assessment")
        st.caption("Mamdani fuzzy inference module cross-references student confidence with predicted marks to expose calibration blindspots.")
        
        r_col1, r_col2 = st.columns([1.4, 1.6])
        
        with r_col1:
            st.markdown("##### Diagnostic Classification Table")
            table_rows = []
            for i in range(len(subjects)):
                lbl = risk_results['label'][i]
                cls_tag = "diag-overconfident" if lbl == 'overconfident' else "diag-anxious" if lbl == 'anxious' else "diag-aligned"
                table_rows.append({
                    "Subject": f"{SUBJECT_META[i]['icon']} {subjects[i]}",
                    "Confidence": f"{confidence_array[i]:.1f} / 10",
                    "Predicted": f"{predicted_marks[i]:.1f}%",
                    "Priority": f"{risk_results['priority'][i]:.2f}",
                    "Classification": f"<span class='diag-pill {cls_tag}'>{lbl}</span>"
                })
            
            df_display = pd.DataFrame(table_rows)
            st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            
            # Natural-language Explanation Accordion
            with st.expander("💬 View Plain-English Diagnoses & Explanations"):
                for i in range(len(subjects)):
                    st.markdown(f"**{SUBJECT_META[i]['icon']} {subjects[i]}:** {risk_results['explanation'][i]}")
            
        with r_col2:
            fig_bubble = px.scatter(
                x=confidence_array,
                y=predicted_marks,
                text=subjects,
                size=risk_results['priority'] + 3,
                color=subjects,
                color_discrete_map={s['name']: s['color'] for s in SUBJECT_META},
                labels={'x': 'Self-Rated Confidence (1–10)', 'y': 'Predicted Mark (%)'},
                title="Confidence vs. Predicted Mark (Bubble Size = Fuzzy Priority)"
            )
            fig_bubble.update_traces(
                textposition='top center',
                textfont=dict(color='#f8fafc', size=12, family="Outfit"),
                marker=dict(line=dict(width=1.5, color='rgba(255,255,255,0.6)'))
            )
            fig_bubble.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.5)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                xaxis=dict(range=[0, 11], gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False,
                height=380
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: HOUR REALLOCATION (STAGE 3)
    # -------------------------------------------------------------------------
    with tab_alloc:
        st.markdown("#### ⏱️ Genetic Algorithm Revision Time Optimization")
        st.caption("Heuristic search shifts revision hours from saturated topics toward high-priority modules with diminishing returns.")
        
        orig_hours = [edited_data[f'hours_{i}'] for i in range(len(SUBJECTS))]
        opt_hours = np.round(allocation_results['hours'], 1)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Student Target Hours',
            x=subjects,
            y=orig_hours,
            marker=dict(color='rgba(148, 163, 184, 0.35)', line=dict(color='rgba(255,255,255,0.2)', width=1))
        ))
        fig_comp.add_trace(go.Bar(
            name='AI Optimized Allocation',
            x=subjects,
            y=opt_hours,
            marker=dict(color='#6366f1', line=dict(color='#818cf8', width=1))
        ))
        fig_comp.update_layout(
            barmode='group',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
            yaxis=dict(title="Study Hours / Week", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=25, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Benchmark Comparison Tiles
        st.markdown("##### 📈 GA Fitness Benchmark Against Heuristic Baselines")
        baselines = allocation_results.get('baselines', {})
        eq_fit = baselines.get('equal', 0.0)
        prop_fit = baselines.get('proportional', 0.0)
        grd_fit = baselines.get('greedy', 0.0)
        
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("🏆 GA Optimal Fitness", f"{ga_fit:.1f}", delta="Optimal Search")
        with b2:
            st.metric("Equal Split Baseline", f"{eq_fit:.1f}", delta=f"{ga_fit - eq_fit:+.1f} pts", delta_color="inverse")
        with b3:
            st.metric("Proportional Baseline", f"{prop_fit:.1f}", delta=f"{ga_fit - prop_fit:+.1f} pts", delta_color="inverse")
        with b4:
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
                plot_bgcolor="rgba(15,23,42,0.5)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                xaxis=dict(title="Generation", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Fitness Score", gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(t=25, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320
            )
            st.plotly_chart(fig_conv, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: STUDY TIMETABLE (STAGE 4)
    # -------------------------------------------------------------------------
    with tab_sched:
        st.markdown("#### 📅 AI-Generated Weekly Revision Timetable")
        st.caption("Daily calendar schedule formatted from allocated revision hours according to cognitive peak windows.")
        
        t1, t2 = st.columns([1.8, 1.2])
        with t1:
            for item in schedule:
                subj_icon = next((s['icon'] for s in SUBJECT_META if s['name'] == item['subject']), '📚')
                subj_color = next((s['color'] for s in SUBJECT_META if s['name'] == item['subject']), '#6366f1')
                st.markdown(f"""
                <div class="schedule-card" style="border-left: 4px solid {subj_color};">
                    <div>
                        <div style="font-weight: 700; color: #38bdf8; font-size: 0.9rem;">
                            🕒 {item.get('start', '09:00')} - {item.get('end', '10:00')} ({item.get('minutes', 60)} mins)
                        </div>
                        <div style="font-weight: 600; color: #f8fafc; font-size: 1rem; margin-top: 2px;">
                            {subj_icon} {item['subject']}
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                            {item.get('day', 'Day')} • {item.get('reason', 'Optimized Placement')}
                        </div>
                    </div>
                    <span class="lms-pill">📌 {item.get('type', 'study').title()}</span>
                </div>
                """, unsafe_allow_html=True)
                
        with t2:
            st.markdown("##### ⏱️ Study Time Share")
            fig_pie = px.pie(
                values=allocation_results['hours'],
                names=subjects,
                hole=0.6,
                color=subjects,
                color_discrete_map={s['name']: s['color'] for s in SUBJECT_META}
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=340
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: RAW PIPELINE TELEMETRY
    # -------------------------------------------------------------------------
    with tab_raw:
        st.markdown("#### 📋 Diagnostic JSON Payloads & Architecture Telemetry")
        st.caption("Developer view verifying exact frozen contract dictionary outputs across Stages 1 through 4.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Stage 1 & 2 Output Payload:**")
            st.json({
                "predicted_marks": list(np.round(predicted_marks, 2)),
                "risk_priority": list(np.round(risk_results['priority'], 2)),
                "risk_labels": risk_results['label']
            })
        with c2:
            st.write("**Stage 3 GA Allocation Metrics:**")
            st.json({
                "allocated_hours": list(np.round(allocation_results['hours'], 2)),
                "optimizer_fitness": allocation_results.get('fitness', 0),
                "baselines": allocation_results.get('baselines', {})
            })
            
        st.write("**Stage 4 Schedule Block Data:**")
        st.dataframe(pd.DataFrame(schedule), use_container_width=True)
else:
    st.info("💡 Click **'🚀 Run Full Optimization Pipeline'** above to compute predictions and generate your personalized schedule.")