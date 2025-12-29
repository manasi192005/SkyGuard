import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import time
from PIL import Image
import tempfile
import os

# --- Import SMS Sender (Ensure sms_sender.py exists) ---
try:
    from sms_sender import SMSSender
except ImportError:
    SMSSender = None  # Handle case where file is missing

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
    page_title="SkyGuard - Drone Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    
    .feature-card {
        background: white;
        border: 2px solid #667eea;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 15px rgba(255,68,68,0.4);
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ff8800 0%, #ff6600 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .drone-status {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
    }
    
    .video-source-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .video-source-card:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_started' not in st.session_state:
    st.session_state.system_started = False
    st.session_state.frame_count = 0
    st.session_state.alerts = []
    st.session_state.crowd_history = []
    st.session_state.risk_history = []
    st.session_state.suspect_count = 0
    st.session_state.emergency_count = 0
    st.session_state.video_source_type = None
    st.session_state.video_source = None

# --- SMS COOLDOWN STATE (Spam Prevention) ---
if 'sms_cooldown' not in st.session_state:
    st.session_state.sms_cooldown = {}

# Video connection functions
def connect_to_source(source_type, source_config):
    """Connect to video source"""
    try:
        if source_type == "Webcam":
            cap = cv2.VideoCapture(source_config['camera_index'])
        
        elif source_type == "Drone":
            connection_string = source_config['connection_string']
            if source_config['drone_type'] == "DJI Tello":
                cap = cv2.VideoCapture('udp://0.0.0.0:11111', cv2.CAP_FFMPEG)
            elif source_config['drone_type'] == "RTSP Stream":
                cap = cv2.VideoCapture(connection_string)
            elif source_config['drone_type'] == "UDP Stream":
                cap = cv2.VideoCapture(connection_string, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(connection_string)
        
        elif source_type == "Phone Camera":
            # IP Webcam URL
            cap = cv2.VideoCapture(source_config['phone_url'])
        
        elif source_type == "Upload Video":
            cap = cv2.VideoCapture(source_config['video_path'])
        
        else:
            return None, False, "Unknown source type"
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap, True, "Connected successfully"
            else:
                cap.release()
                return None, False, "Cannot read frames"
        else:
            return None, False, "Cannot open connection"
    
    except Exception as e:
        return None, False, f"Error: {str(e)}"

# Initialize system
@st.cache_resource
def init_skyguard_system():
    """Initialize all SkyGuard components"""
    try:
        db_engine = init_database('data/database/skyguard.db')
        db_session = get_session(db_engine)
        
        crowd_analyzer = CrowdAnalyzerEnhanced(use_simple_detection=True)
        face_recognition = FaceRecognitionEnhanced(confidence_threshold=0.85)
        stampede_predictor = StampedePredictorEnhanced(sequence_length=30, fps=30)
        emergency_detector = EmergencyDetectorEnhanced(
            fall_angle_threshold=60,
            immobility_seconds=300,
            fps=30
        )
        
        return {
            'db_session': db_session,
            'crowd': crowd_analyzer,
            'face': face_recognition,
            'stampede': stampede_predictor,
            'emergency': emergency_detector,
            'status': 'operational'
        }
    except Exception as e:
        st.error(f"System initialization error: {e}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ SkyGuard Surveillance System</h1>
    <p>AI-Powered Aerial Crowd Safety & Security Monitoring</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Team GDuo | VESIT, Mumbai</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎥 Video Source Selection")
    st.markdown("---")
    
    # Video source type
    source_type = st.radio(
        "Select Video Source",
        ["🚁 Drone", "📹 Webcam", "📱 Phone Camera", "📤 Upload Video"],
        key="source_type_radio"
    )
    
    # Clean up emoji from source type
    source_type_clean = source_type.split(" ", 1)[1]
    
    st.markdown("---")
    
    # Configuration based on source type
    source_config = {}
    
    if source_type_clean == "Webcam":
        st.markdown("### 📹 Webcam Settings")
        camera_index = st.selectbox("Camera", [0, 1, 2], index=0)
        source_config = {'camera_index': camera_index}
        
        st.info("💡 Using local webcam for monitoring")
    
    elif source_type_clean == "Drone":
        st.markdown("### 🚁 Drone Connection")
        
        drone_type = st.selectbox(
            "Drone Type",
            ["DJI Tello", "RTSP Stream", "UDP Stream", "HTTP/MJPEG", "Custom"]
        )
        
        if drone_type == "DJI Tello":
            st.info("📡 DJI Tello uses UDP on port 11111")
            connection_string = "udp://0.0.0.0:11111"
            st.caption("Ensure Tello is connected to WiFi")
        
        elif drone_type == "RTSP Stream":
            st.markdown("**Format:** `rtsp://ip:port/stream`")
            connection_string = st.text_input(
                "RTSP URL",
                value="rtsp://192.168.1.100:554/stream",
                placeholder="rtsp://192.168.1.100:554/stream"
            )
            st.caption("Example: rtsp://admin:pass@192.168.1.100:554/live")
        
        elif drone_type == "UDP Stream":
            st.markdown("**Format:** `udp://ip:port`")
            connection_string = st.text_input(
                "UDP URL",
                value="udp://0.0.0.0:5000",
                placeholder="udp://0.0.0.0:5000"
            )
        
        elif drone_type == "HTTP/MJPEG":
            st.markdown("**Format:** `http://ip:port/video`")
            connection_string = st.text_input(
                "HTTP URL",
                value="http://192.168.1.100:8080/video",
                placeholder="http://192.168.1.100:8080/video"
            )
        
        else:  # Custom
            connection_string = st.text_input(
                "Custom Stream URL",
                placeholder="Enter stream URL"
            )
        
        source_config = {
            'drone_type': drone_type,
            'connection_string': connection_string
        }
        
        if st.button("🔌 Test Connection", key="test_drone"):
            with st.spinner("Testing connection..."):
                cap, success, msg = connect_to_source(source_type_clean, source_config)
                if success:
                    st.success(f"✅ {msg}")
                    cap.release()
                else:
                    st.error(f"❌ {msg}")
    
    elif source_type_clean == "Phone Camera":
        st.markdown("### 📱 Phone Camera Setup")
        
        st.info("""
        **Install IP Webcam App:**
        - Android: "IP Webcam" from Play Store
        - iPhone: "EpocCam" or "iVCam"
        """)
        
        phone_ip = st.text_input(
            "Phone IP Address",
            value="192.168.1.100",
            placeholder="192.168.1.100"
        )
        
        phone_port = st.text_input(
            "Port",
            value="8080",
            placeholder="8080"
        )
        
        phone_url = f"http://{phone_ip}:{phone_port}/video"
        
        st.code(phone_url, language="text")
        
        source_config = {'phone_url': phone_url}
        
        st.caption("Make sure phone and computer are on same WiFi")
        
        if st.button("🔌 Test Connection", key="test_phone"):
            with st.spinner("Testing connection..."):
                cap, success, msg = connect_to_source(source_type_clean, source_config)
                if success:
                    st.success(f"✅ {msg}")
                    cap.release()
                else:
                    st.error(f"❌ {msg}")
    
    elif source_type_clean == "Upload Video":
        st.markdown("### 📤 Upload Video File")
        
        uploaded_file = st.file_uploader(
            "Choose video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            source_config = {'video_path': video_path}
            
            st.success("✅ Video uploaded successfully!")
            st.info(f"📁 File: {uploaded_file.name}")
        else:
            source_config = None
            st.warning("⚠️ Please upload a video file")
    
    st.markdown("---")
    
    # GPS Settings
    st.markdown("### 📍 GPS Location")
    gps_lat = st.number_input("Latitude", value=19.0760, format="%.6f", key="gps_lat")
    gps_lon = st.number_input("Longitude", value=72.8777, format="%.6f", key="gps_lon")
    
    st.caption(f"Current: {gps_lat:.4f}, {gps_lon:.4f}")
    
    st.markdown("---")

    # --- SMS CONFIGURATION ---
    st.markdown("### 📲 SMS Alerts")
    with st.expander("Configure Twilio (Optional)"):
        twilio_sid = st.text_input("Twilio SID", type="password")
        twilio_token = st.text_input("Twilio Token", type="password")
        twilio_from = st.text_input("From Number")
        auth_person = st.text_input("Authorized Person No.", value="+91")
    
    # Initialize SMS Sender
    sms_system = None
    if SMSSender and twilio_sid and twilio_token:
        sms_system = SMSSender(twilio_sid, twilio_token, twilio_from, auth_person)

    st.markdown("---")
    
    # Feature toggles
    st.markdown("### 🎛️ Features")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_heatmap = st.checkbox("🗺️ Heat Map", value=True, key="heat")
        enable_stampede = st.checkbox("⏰ Stampede", value=True, key="stamp")
    with col2:
        enable_face = st.checkbox("👤 Face ID", value=True, key="face")
        enable_emergency = st.checkbox("🚑 Emergency", value=True, key="emerg")
    
    st.markdown("---")
    
    # System controls
    st.markdown("### 🎮 Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 START", type="primary", key="start_btn"):
            if source_config or source_type_clean == "Webcam":
                st.session_state.system_started = True
                st.session_state.video_source_type = source_type_clean
                st.session_state.video_source = source_config
                st.session_state.frame_count = 0
                st.session_state.alerts = []
                st.rerun()
            else:
                st.error("Please configure video source first!")
    
    with col2:
        if st.button("⏹️ STOP", key="stop_btn"):
            st.session_state.system_started = False
            st.rerun()
    
    st.markdown("---")
    
    # Status
    st.markdown("### 📊 Status")
    if st.session_state.system_started:
        st.success("🟢 SYSTEM ACTIVE")
        st.metric("Frames", st.session_state.frame_count)
        st.metric("Alerts", len(st.session_state.alerts))
    else:
        st.error("🔴 SYSTEM INACTIVE")
    
    st.markdown("---")
    
    # Admin Management
    if st.button("👥 Manage Admins", key="manage_admins"):
        st.info("Run: `python3 manage_admins.py` in terminal")
    
    if st.button("➕ Add Suspect", key="add_suspect"):
        st.info("Run: `python3 add_suspect.py` in terminal")

# Main content
if not st.session_state.system_started:
    # Setup screen
    st.markdown("## 🎯 System Setup")
    
    # Video source cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚁 Drone Connection</h3>
            <p><strong>Professional aerial surveillance</strong></p>
            <ul>
                <li>DJI Tello support</li>
                <li>RTSP/UDP streaming</li>
                <li>Real-time GPS tracking</li>
                <li>High-altitude monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📱 Phone Camera</h3>
            <p><strong>Use your smartphone</strong></p>
            <ul>
                <li>No additional hardware needed</li>
                <li>WiFi streaming</li>
                <li>Mobile monitoring</li>
                <li>Easy setup</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📹 Webcam</h3>
            <p><strong>Local camera monitoring</strong></p>
            <ul>
                <li>Built-in camera support</li>
                <li>No network required</li>
                <li>Instant setup</li>
                <li>Testing mode</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📤 Upload Video</h3>
            <p><strong>Process recorded footage</strong></p>
            <ul>
                <li>MP4, AVI, MOV support</li>
                <li>Batch processing</li>
                <li>Historical analysis</li>
                <li>Demo mode</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features overview
    st.markdown("## 🎯 AI Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🗺️ Feature 1")
        st.markdown("**Density Heat Map**")
        st.markdown("- Crowd counting")
        st.markdown("- Red = High risk")
        st.markdown("- Stampede zones")
    
    with col2:
        st.markdown("### 👤 Feature 2")
        st.markdown("**Face Recognition**")
        st.markdown("- Suspect detection")
        st.markdown("- Admin alerts")
        st.markdown("- GPS tracking")
    
    with col3:
        st.markdown("### ⏰ Feature 3")
        st.markdown("**Stampede Prediction**")
        st.markdown("- 90-sec warning")
        st.markdown("- LSTM AI model")
        st.markdown("- Risk assessment")
    
    with col4:
        st.markdown("### 🚑 Feature 4")
        st.markdown("**Emergency Detection**")
        st.markdown("- Fall detection")
        st.markdown("- 5-min threshold")
        st.markdown("- Medical alerts")
    
    st.info("👆 **Configure video source in sidebar, then click START**")

else:
    # System running
    system = init_skyguard_system()
    
    if not system:
        st.error("Failed to initialize system")
        st.stop()
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎬 Frame", st.session_state.frame_count)
    
    with col2:
        crowd_count = st.session_state.crowd_history[-1] if st.session_state.crowd_history else 0
        st.metric("👥 People", crowd_count)
    
    with col3:
        st.metric("🎯 Suspects", st.session_state.suspect_count)
    
    with col4:
        st.metric("⚠️ Warnings", len([a for a in st.session_state.alerts if 'STAMPEDE' in a.get('type', '')]))
    
    with col5:
        st.metric("🚑 Emergencies", st.session_state.emergency_count)
    
    st.markdown("---")
    
    # Video and alerts
    col_video, col_alerts = st.columns([2.5, 1])
    
    with col_video:
        st.markdown(f"### 📹 Live Feed - {st.session_state.video_source_type}")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col_alerts:
        st.markdown("### 🚨 Active Alerts")
        alerts_placeholder = st.empty()
    
    # Connect and process
    cap, connected, msg = connect_to_source(
        st.session_state.video_source_type,
        st.session_state.video_source
    )
    
    if cap and connected:
        ret, frame = cap.read()
        
        if ret:
            st.session_state.frame_count += 1
            output = frame.copy()
            current_alerts = []
            
            crowd_count = 0
            risk_level = 'low'
            zones = []
            
            # Feature 1: Heat Map
            if enable_heatmap:
                try:
                    output, crowd_count, risk_level, heatmap, zones = \
                        system['crowd'].analyze_crowd(output, draw_visualization=True)
                    
                    st.session_state.crowd_history.append(crowd_count)
                    st.session_state.risk_history.append(risk_level)
                    
                    if len(st.session_state.crowd_history) > 100:
                        st.session_state.crowd_history.pop(0)
                        st.session_state.risk_history.pop(0)
                    
                    if zones:
                        current_alerts.append({
                            'type': 'STAMPEDE_ZONE',
                            'severity': 'critical',
                            'message': f'🔴 {len(zones)} stampede-prone zone(s) detected!',
                            'time': datetime.now()
                        })
                except:
                    pass
            
            # Feature 2: Face Recognition (WITH SMS LOGIC)
            if enable_face:
                try:
                    output, detections, suspects = \
                        system['face'].process_frame(
                            output,
                            latitude=gps_lat,
                            longitude=gps_lon,
                            draw_boxes=True
                        )
                    
                    for suspect in suspects:
                        name = suspect['name']
                        st.session_state.suspect_count += 1
                        
                        # Add Alert to UI
                        current_alerts.append({
                            'type': 'SUSPECT',
                            'severity': 'critical',
                            'message': f'👤 SUSPECT: {name} ({suspect["confidence"]:.0%})',
                            'time': datetime.now()
                        })

                        # --- SMS SPAM PREVENTION LOGIC ---
                        if sms_system:
                            last_sent = st.session_state.sms_cooldown.get(name, 0)
                            current_time = time.time()
                            
                            # Check if 5 minutes (300 seconds) have passed
                            if (current_time - last_sent) > 300:
                                success = sms_system.send_suspect_alert(name, gps_lat, gps_lon)
                                if success:
                                    st.toast(f"📨 SMS Alert sent for {name}!", icon="🚀")
                                    st.session_state.sms_cooldown[name] = current_time
                                else:
                                    st.toast(f"❌ Failed to send SMS for {name}", icon="⚠️")

                except Exception as e:
                    # In production, log this error
                    pass
            
            # Feature 3: Stampede Prediction
            if enable_stampede and crowd_count > 0:
                try:
                    warning = system['stampede'].generate_early_warning(crowd_count, zones)
                    
                    if warning and warning['risk_level'] in ['high', 'critical']:
                        current_alerts.append({
                            'type': 'STAMPEDE',
                            'severity': warning['risk_level'],
                            'message': f'⚠️ Stampede in 90s! ({warning["predicted_density"]} people)',
                            'time': datetime.now()
                        })
                except:
                    pass
            
            # Feature 4: Emergency Detection
            if enable_emergency:
                try:
                    output, emergency_detected, emergency_info = \
                        system['emergency'].detect_emergency(
                            output,
                            st.session_state.frame_count,
                            draw_visualization=True
                        )
                    
                    if emergency_detected:
                        st.session_state.emergency_count += 1
                        current_alerts.append({
                            'type': 'EMERGENCY',
                            'severity': 'critical',
                            'message': f'🚑 Medical emergency detected!',
                            'time': datetime.now()
                        })
                except:
                    pass
            
            # Add GPS to frame
            cv2.putText(output, f"GPS: {gps_lat:.4f}, {gps_lon:.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update alerts
            st.session_state.alerts.extend(current_alerts)
            if len(st.session_state.alerts) > 50:
                st.session_state.alerts = st.session_state.alerts[-50:]
            
            # Display video
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            video_placeholder.image(output_rgb, channels="RGB", use_container_width=True)
            
            # Display alerts
            with alerts_placeholder.container():
                if current_alerts or st.session_state.alerts:
                    recent = st.session_state.alerts[-10:]
                    for alert in reversed(recent):
                        severity = f"alert-{alert['severity']}"
                        st.markdown(
                            f'<div class="{severity}">'
                            f'{alert["message"]}<br>'
                            f'<small>{alert["time"].strftime("%H:%M:%S")}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.success("✅ No alerts")
            
            # Status
            status_placeholder.success(
                f"🎥 {st.session_state.video_source_type} | "
                f"Frame {st.session_state.frame_count} | "
                f"👥 {crowd_count} people"
            )
        
        cap.release()
    else:
        st.error(f"❌ {msg}")
    
    # Charts
    if st.session_state.crowd_history:
        st.markdown("---")
        st.markdown("## 📈 Analytics")
        
        fig_crowd = go.Figure()
        fig_crowd.add_trace(go.Scatter(
            y=st.session_state.crowd_history,
            mode='lines+markers',
            name='Crowd Density',
            line=dict(color='#667eea', width=3)
        ))
        fig_crowd.update_layout(
            title="Real-Time Crowd Density",
            xaxis_title="Frame",
            yaxis_title="People Count",
            height=300
        )
        st.plotly_chart(fig_crowd, use_container_width=True)
    
    # Auto-refresh
    time.sleep(0.033)
    st.rerun()