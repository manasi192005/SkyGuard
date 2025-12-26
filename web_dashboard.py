"""
SkyGuard Professional Web Dashboard
Real-time monitoring of all 4 features with beautiful UI
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import time

from models.crowd_analysis_enhanced import CrowdAnalyzerEnhanced
from models.face_recognition_enhanced import FaceRecognitionEnhanced
from models.stampede_prediction_enhanced import StampedePredictorEnhanced
from models.emergency_detection_enhanced import EmergencyDetectorEnhanced
from models.database import (
    init_database, get_session, 
    get_recent_analytics, get_recent_suspects, get_active_emergencies
)

# Page Configuration
st.set_page_config(
    page_title="SkyGuard - Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #667eea;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: white;
        font-weight: bold;
    }
    .alert-critical { background-color: #ff4444; }
    .alert-high { background-color: #ff8800; }
    .alert-medium { background-color: #ffbb33; color: black; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# --- SYSTEM INITIALIZATION ---
@st.cache_resource
def init_skyguard_system():
    try:
        db_engine = init_database('data/database/skyguard.db')
        db_session = get_session(db_engine)
        
        return {
            'db_session': db_session,
            'crowd': CrowdAnalyzerEnhanced(use_simple_detection=True),
            'face': FaceRecognitionEnhanced(confidence_threshold=0.85),
            'stampede': StampedePredictorEnhanced(sequence_length=30, fps=30),
            'emergency': EmergencyDetectorEnhanced(fall_angle_threshold=60, immobility_seconds=300)
        }
    except Exception as e:
        st.error(f"System Error: {e}")
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # SOURCE SELECTION
    st.subheader("📹 Video Source")
    source_type = st.radio("Select Source", ["Webcam", "Upload Video", "IP Camera (iPhone)"])
    
    video_source = 0  # Default
    
    if source_type == "Webcam":
        # Allow choosing index 0, 1, 2 for different connected cameras
        cam_idx = st.selectbox("Select Camera Index", [0, 1, 2, 3], index=0)
        video_source = cam_idx
        
    elif source_type == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # SAVE TEMP FILE (Critical for OpenCV)
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_source = tfile.name

    elif source_type == "IP Camera (iPhone)":
        st.info("Use apps like 'DroidCam' or 'Iriun' and select Webcam Index, OR enter IP URL below.")
        ip_url = st.text_input("IP Camera URL", "http://192.168.1.XX:8080/video")
        use_url = st.checkbox("Use URL")
        if use_url:
            video_source = ip_url
        else:
            video_source = st.selectbox("Virtual Camera Index", [0, 1, 2, 3], index=1)

    st.markdown("---")
    
    # FEATURES
    st.subheader("🎛️ Active Features")
    enable_heatmap = st.checkbox("Heat Map", True)
    enable_face = st.checkbox("Face ID", True)
    enable_stampede = st.checkbox("Stampede", True)
    enable_emergency = st.checkbox("Emergency", True)
    
    st.markdown("---")
    
    # START / STOP
    if st.button("🚀 START MONITORING", type="primary"):
        st.session_state.run_camera = True
        st.rerun()
        
    if st.button("⏹️ STOP", type="secondary"):
        st.session_state.run_camera = False
        st.rerun()

# --- MAIN LAYOUT ---
st.markdown('<div class="main-header"><h1>🛡️ SkyGuard Command Center</h1></div>', unsafe_allow_html=True)

system = init_skyguard_system()

if st.session_state.run_camera and system:
    # Layout: Video on Left, Alerts/Stats on Right
    col_video, col_stats = st.columns([0.7, 0.3])
    
    with col_video:
        video_placeholder = st.empty()
        
    with col_stats:
        st.markdown("### 📊 Live Stats")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            stat_crowd = st.empty()
        with kpi2:
            stat_risk = st.empty()
            
        st.markdown("### 🚨 Alert Log")
        alert_container = st.container()
        
    # --- PROCESSING LOOP ---
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open video source {video_source}")
    else:
        frame_count = 0
        crowd_history = []
        
        while cap.isOpened() and st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Video Ended or Disconnected.")
                break
                
            frame_count += 1
            output = frame.copy()
            current_alerts = []
            
            # 1. HEATMAP & CROWD
            crowd_count = 0
            risk_level = "Low"
            zones = []
            
            if enable_heatmap:
                output, crowd_count, risk_level, _, zones = system['crowd'].analyze_crowd(output, draw_visualization=True)
                if zones:
                    current_alerts.append({"msg": f"Stampede Zone Detected!", "level": "critical"})

            # 2. FACE RECOGNITION
            if enable_face:
                output, _, suspects = system['face'].process_frame(output)
                for s in suspects:
                    current_alerts.append({"msg": f"SUSPECT: {s['name']}", "level": "critical"})

            # 3. STAMPEDE PREDICTION
            if enable_stampede and crowd_count > 0:
                warning = system['stampede'].generate_early_warning(crowd_count, zones)
                if warning and warning['risk_level'] in ['high', 'critical']:
                    current_alerts.append({"msg": "Stampede Risk High!", "level": "high"})

            # 4. EMERGENCY DETECTION
            if enable_emergency:
                output, detected, info = system['emergency'].detect_emergency(output, frame_count)
                if detected:
                    current_alerts.append({"msg": "Medical Emergency!", "level": "critical"})

            # --- UPDATE UI (Inside Loop) ---
            
            # Video
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            video_placeholder.image(output, channels="RGB", use_container_width=True)
            
            # KPIs
            stat_crowd.metric("Crowd", crowd_count)
            stat_risk.metric("Risk", risk_level.upper())
            
            # Alerts
            with alert_container:
                if current_alerts:
                    for alert in current_alerts:
                        color = "#ff4444" if alert['level'] == "critical" else "#ff8800"
                        st.markdown(f'<div style="background:{color};padding:10px;border-radius:5px;margin-bottom:5px;color:white;">{alert["msg"]}</div>', unsafe_allow_html=True)
                else:
                    st.empty() # Clear old alerts if safe

        cap.release()

elif not st.session_state.run_camera:
    st.info("System is ready. Select a source and click START to begin.")