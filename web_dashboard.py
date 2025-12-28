import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
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
    
    .drone-status {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
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
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .video-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
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
    st.session_state.drone_connected = False

# Drone connection functions
def connect_to_drone(connection_type, connection_string):
    """
    Connect to drone based on type
    
    Supported types:
    - RTSP Stream
    - UDP Stream
    - HTTP/MJPEG Stream
    - DJI Drone
    - Custom IP Camera
    """
    try:
        if connection_type == "RTSP Stream":
            # RTSP format: rtsp://username:password@ip:port/stream
            cap = cv2.VideoCapture(connection_string)
        
        elif connection_type == "UDP Stream":
            # UDP format: udp://ip:port
            cap = cv2.VideoCapture(connection_string, cv2.CAP_FFMPEG)
        
        elif connection_type == "HTTP/MJPEG":
            # HTTP format: http://ip:port/video
            cap = cv2.VideoCapture(connection_string)
        
        elif connection_type == "DJI Tello":
            # DJI Tello uses UDP on port 11111
            cap = cv2.VideoCapture('udp://0.0.0.0:11111', cv2.CAP_FFMPEG)
        
        elif connection_type == "IP Camera":
            # Generic IP camera
            cap = cv2.VideoCapture(connection_string)
        
        else:
            # Fallback to webcam
            cap = cv2.VideoCapture(0)
        
        # Test if connection works
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return cap, True, "Connected successfully"
            else:
                cap.release()
                return None, False, "Cannot read frames from stream"
        else:
            return None, False, "Cannot open connection"
    
    except Exception as e:
        return None, False, f"Connection error: {str(e)}"

# Initialize system components
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
    <h1>🛡️ SkyGuard Drone Surveillance</h1>
    <p>AI-Powered Aerial Crowd Safety Monitoring</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Team GDuo | VESIT, Mumbai</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Control Panel
with st.sidebar:
    st.markdown("## 🚁 Drone Connection")
    st.markdown("---")
    
    # Connection type selection
    connection_type = st.selectbox(
        "Connection Type",
        [
            "Webcam (Testing)",
            "RTSP Stream",
            "UDP Stream", 
            "HTTP/MJPEG",
            "DJI Tello",
            "IP Camera",
            "Custom Stream"
        ]
    )
    
    # Connection string input based on type
    if connection_type == "Webcam (Testing)":
        camera_index = st.selectbox("Camera", [0, 1, 2], index=0)
        connection_string = camera_index
        st.info("Using local webcam for testing")
    
    elif connection_type == "RTSP Stream":
        st.markdown("**Format:** `rtsp://ip:port/stream`")
        connection_string = st.text_input(
            "RTSP URL",
            value="rtsp://192.168.1.100:554/stream",
            placeholder="rtsp://192.168.1.100:554/stream"
        )
        st.caption("Example: rtsp://admin:password@192.168.1.100:554/live")
    
    elif connection_type == "UDP Stream":
        st.markdown("**Format:** `udp://ip:port`")
        connection_string = st.text_input(
            "UDP URL",
            value="udp://0.0.0.0:5000",
            placeholder="udp://0.0.0.0:5000"
        )
    
    elif connection_type == "HTTP/MJPEG":
        st.markdown("**Format:** `http://ip:port/video`")
        connection_string = st.text_input(
            "HTTP URL",
            value="http://192.168.1.100:8080/video",
            placeholder="http://192.168.1.100:8080/video"
        )
    
    elif connection_type == "DJI Tello":
        st.info("DJI Tello uses UDP on port 11111")
        connection_string = "udp://0.0.0.0:11111"
        st.caption("Make sure Tello is connected to WiFi")
    
    elif connection_type == "IP Camera":
        st.markdown("**IP Camera Stream URL**")
        connection_string = st.text_input(
            "Camera URL",
            value="http://192.168.1.100/video.cgi",
            placeholder="http://192.168.1.100/video.cgi"
        )
    
    else:  # Custom Stream
        connection_string = st.text_input(
            "Custom Stream URL",
            placeholder="Enter full stream URL"
        )
    
    # Test connection button
    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            cap, success, message = connect_to_drone(connection_type, connection_string)
            if success:
                st.success(f"✅ {message}")
                st.session_state.drone_connected = True
                cap.release()
            else:
                st.error(f"❌ {message}")
                st.session_state.drone_connected = False
    
    st.markdown("---")
    
    # Drone status
    if st.session_state.drone_connected:
        st.markdown('<div class="drone-status">🟢 DRONE CONNECTED</div>', unsafe_allow_html=True)
    else:
        st.warning("🔴 Drone Not Connected")
    
    st.markdown("---")
    
    # Feature toggles
    st.markdown("### 🎛️ Feature Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_heatmap = st.checkbox("🗺️ Heat Map", value=True, key="heat")
        enable_stampede = st.checkbox("⏰ Stampede", value=True, key="stamp")
    with col2:
        enable_face = st.checkbox("👤 Face ID", value=True, key="face")
        enable_emergency = st.checkbox("🚑 Emergency", value=True, key="emerg")
    
    st.markdown("---")
    
    # System controls
    st.markdown("### 🎮 System Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 START", type="primary", use_container_width=True):
            st.session_state.system_started = True
            st.session_state.frame_count = 0
            st.session_state.alerts = []
            st.rerun()
    
    with col2:
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.system_started = False
            st.rerun()
    
    st.markdown("---")
    
    # GPS Coordinates (for drone)
    st.markdown("### 📍 GPS Location")
    gps_lat = st.number_input("Latitude", value=19.0760, format="%.4f")
    gps_lon = st.number_input("Longitude", value=72.8777, format="%.4f")
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    if st.session_state.system_started:
        st.success("🟢 MONITORING ACTIVE")
        st.metric("Frames Processed", st.session_state.frame_count)
    else:
        st.error("🔴 SYSTEM INACTIVE")

# Main content area
if not st.session_state.system_started:
    # Setup instructions
    st.markdown("## 🚁 Drone Setup Instructions")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📱 DJI Tello", "📡 RTSP Stream", "🌐 IP Camera", "🎥 Custom"])
    
    with tab1:
        st.markdown("""
        ### DJI Tello Drone Setup
        
        1. **Turn on Tello drone**
        2. **Connect to Tello WiFi** (TELLO-XXXXX)
        3. **Select "DJI Tello"** in connection type
        4. **Click "Test Connection"**
        5. **Click "START"** to begin monitoring
        
        **Port:** UDP 11111  
        **Resolution:** 720p @ 30fps
        """)
    
    with tab2:
        st.markdown("""
        ### RTSP Stream Setup
        
        **Format:** `rtsp://[username:password@]ip:port/stream`
        
        **Examples:**
        - `rtsp://192.168.1.100:554/stream`
        - `rtsp://admin:12345@192.168.1.100:554/live`
        - `rtsp://camera.local:8554/video`
        
        **Common Ports:**
        - 554 (Default RTSP)
        - 8554 (Alternative)
        """)
    
    with tab3:
        st.markdown("""
        ### IP Camera Setup
        
        **Format:** `http://ip:port/video` or `http://ip/video.cgi`
        
        **Examples:**
        - `http://192.168.1.100:8080/video`
        - `http://192.168.1.100/video.cgi`
        - `http://admin:pass@192.168.1.100/stream`
        
        **Common Ports:**
        - 80 (HTTP)
        - 8080 (Alternative HTTP)
        """)
    
    with tab4:
        st.markdown("""
        ### Custom Stream Setup
        
        **Supported Protocols:**
        - RTSP: `rtsp://...`
        - HTTP: `http://...`
        - UDP: `udp://...`
        - RTMP: `rtmp://...`
        
        **Tips:**
        - Make sure drone/camera is on same network
        - Check firewall settings
        - Test with VLC player first
        """)
    
    st.info("👆 **Configure drone connection in sidebar, then click START**")

else:
    # System is running
    system = init_skyguard_system()
    
    if not system:
        st.error("Failed to initialize system")
        st.stop()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎬 Frame", st.session_state.frame_count)
    
    with col2:
        crowd_count = st.session_state.crowd_history[-1] if st.session_state.crowd_history else 0
        st.metric("👥 People", crowd_count)
    
    with col3:
        st.metric("🎯 Suspects", st.session_state.suspect_count)
    
    with col4:
        stampede_count = len([a for a in st.session_state.alerts if 'STAMPEDE' in a.get('type', '')])
        st.metric("⚠️ Warnings", stampede_count)
    
    with col5:
        st.metric("🚑 Emergencies", st.session_state.emergency_count)
    
    st.markdown("---")
    
    # Main content: Video feed and alerts
    col_video, col_alerts = st.columns([2.5, 1])
    
    with col_video:
        st.markdown("### 🚁 Live Drone Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col_alerts:
        st.markdown("### 🚨 Active Alerts")
        alerts_placeholder = st.empty()
    
    # Connect to drone and process video
    cap, connected, msg = connect_to_drone(connection_type, connection_string)
    
    if cap and connected:
        ret, frame = cap.read()
        
        if ret:
            st.session_state.frame_count += 1
            output = frame.copy()
            current_alerts = []
            
            # Process through all features (same as before)
            crowd_count = 0
            risk_level = 'low'
            zones = []
            
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
                except Exception as e:
                    pass
            
            if enable_face:
                try:
                    output, detections, suspects = \
                        system['face'].process_frame(output, draw_boxes=True)
                    
                    for suspect in suspects:
                        st.session_state.suspect_count += 1
                        current_alerts.append({
                            'type': 'SUSPECT',
                            'severity': 'critical',
                            'message': f'👤 SUSPECT: {suspect["name"]}',
                            'time': datetime.now()
                        })
                except Exception as e:
                    pass
            
            if enable_stampede and crowd_count > 0:
                try:
                    warning = system['stampede'].generate_early_warning(crowd_count, zones)
                    
                    if warning and warning['risk_level'] in ['high', 'critical']:
                        current_alerts.append({
                            'type': 'STAMPEDE_WARNING',
                            'severity': warning['risk_level'],
                            'message': f'⚠️ Stampede in 90s! ({warning["predicted_density"]} people)',
                            'time': datetime.now()
                        })
                except Exception as e:
                    pass
            
            if enable_emergency:
                try:
                    output, emergency_detected, emergency_info = \
                        system['emergency'].detect_emergency(output, st.session_state.frame_count, draw_visualization=True)
                    
                    if emergency_detected:
                        st.session_state.emergency_count += 1
                        current_alerts.append({
                            'type': 'EMERGENCY',
                            'severity': 'critical',
                            'message': f'🚑 MEDICAL EMERGENCY!',
                            'time': datetime.now()
                        })
                except Exception as e:
                    pass
            
            # Add GPS coordinates to frame
            cv2.putText(output, f"GPS: {gps_lat:.4f}, {gps_lon:.4f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            st.session_state.alerts.extend(current_alerts)
            if len(st.session_state.alerts) > 50:
                st.session_state.alerts = st.session_state.alerts[-50:]
            
            # Display video
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            video_placeholder.image(output_rgb, channels="RGB", use_container_width=True)
            
            # Display alerts
            with alerts_placeholder.container():
                if current_alerts or st.session_state.alerts:
                    recent_alerts = st.session_state.alerts[-10:]
                    for alert in reversed(recent_alerts):
                        severity_class = f"alert-{alert['severity']}"
                        st.markdown(
                            f'<div class="{severity_class}">'
                            f'{alert["message"]}<br>'
                            f'<small>{alert["time"].strftime("%H:%M:%S")}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.success("✅ No alerts")
            
            status_placeholder.success(f"🚁 DRONE ACTIVE | Frame {st.session_state.frame_count} | "
                                     f"👥 {crowd_count} people | ⚠️ {risk_level.upper()}")
        
        cap.release()
    else:
        st.error(f"❌ {msg}")
        st.info("Check connection settings in sidebar")
    
    # Charts
    if st.session_state.crowd_history:
        st.markdown("---")
        st.markdown("## 📈 Real-Time Analytics")
        
        fig_crowd = go.Figure()
        fig_crowd.add_trace(go.Scatter(
            y=st.session_state.crowd_history,
            mode='lines+markers',
            name='Crowd Density',
            line=dict(color='#667eea', width=3)
        ))
        fig_crowd.update_layout(
            title="Crowd Density from Drone Feed",
            xaxis_title="Frame",
            yaxis_title="People Count",
            height=300
        )
        st.plotly_chart(fig_crowd, use_container_width=True)
    
    # Auto-refresh
    time.sleep(0.033)
    st.rerun()