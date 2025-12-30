import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from datetime import datetime

# Import Face Recognition System
from models.face_recognition_enhanced import FaceRecognitionEnhanced

# Page Configuration
st.set_page_config(
    page_title="SkyGuard - Face Recognition",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    
    .alert-high {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white; padding: 15px; border-radius: 8px;
        border-left: 5px solid #990000;
        animation: pulse 2s infinite;
        margin-bottom: 10px;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6600 100%);
        color: white; padding: 15px; border-radius: 8px;
        border-left: 5px solid #cc5200;
        margin-bottom: 10px;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #ffd700 0%, #ffaa00 100%);
        color: white; padding: 15px; border-radius: 8px;
        border-left: 5px solid #cc8800;
        margin-bottom: 10px;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.02); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .match-percentage {
        font-size: 24px;
        font-weight: bold;
        color: #ff4444;
    }
    
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'system_started' not in st.session_state: 
    st.session_state.system_started = False
if 'alerts' not in st.session_state: 
    st.session_state.alerts = []
if 'last_faces' not in st.session_state: 
    st.session_state.last_faces = [] 
if 'total_detections' not in st.session_state: 
    st.session_state.total_detections = 0
if 'frame_count' not in st.session_state: 
    st.session_state.frame_count = 0
if 'tolerance' not in st.session_state:
    st.session_state.tolerance = 0.5
if 'highest_match' not in st.session_state:
    st.session_state.highest_match = 0

# --- HELPER FUNCTIONS ---
@st.cache_resource
def init_system(tolerance):
    try:
        return FaceRecognitionEnhanced(tolerance=tolerance)
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return None

def get_alert_class(percentage):
    """Get CSS class based on match percentage"""
    if percentage >= 80:
        return "alert-high"
    elif percentage >= 60:
        return "alert-medium"
    else:
        return "alert-low"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Control Panel")
    
    # System Info
    st.info("🧠 Using: face_recognition library\n✅ Works with any angle")
    
    # Settings
    with st.expander("⚙️ Detection Settings", expanded=True):
        tolerance_percent = st.slider(
            "Match Sensitivity (%)",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            help="Lower = Stricter matching (fewer false positives)"
        )
        # Convert percentage to tolerance (inverse relationship)
        st.session_state.tolerance = 0.7 - (tolerance_percent / 100)
        
        st.info(f"Current tolerance: {st.session_state.tolerance:.2f}")
        
        process_frames = st.slider(
            "Process Every N Frames",
            min_value=1,
            max_value=10,
            value=3,
            help="Lower = More frequent detection (slower)"
        )
        st.session_state.process_interval = process_frames
    
    # Source Selection
    st.divider()
    st.subheader("📹 Video Source")
    
    source_type = st.selectbox(
        "Select Source",
        ["Webcam", "Upload Video", "Drone Stream"],
        help="Choose your video input"
    )
    
    source_config = {}
    
    if source_type == "Webcam":
        source_config['index'] = st.selectbox("Camera", [0, 1, 2], index=0)
        
    elif source_type == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload Video File", 
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            source_config['path'] = tfile.name
            st.success("✅ Video Loaded")
        else:
            st.warning("Please upload a video file")
            
    elif source_type == "Drone Stream":
        source_config['url'] = st.text_input(
            "Stream URL", 
            "udp://0.0.0.0:11111",
            help="Enter RTSP or UDP stream URL"
        )

    # Add Suspect
    st.divider()
    st.subheader("👤 Suspect Database")
    
    with st.expander("Add New Suspect", expanded=True):
        st.markdown("**📸 Upload Clear Photo**")
        st.caption("Tips: Front-facing, good lighting, single person")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            up_photo = st.file_uploader(
                "Face Photo", 
                type=['jpg', 'png', 'jpeg'],
                label_visibility="collapsed"
            )
        
        with col2:
            up_name = st.text_input(
                "Name",
                placeholder="John Doe",
                label_visibility="collapsed"
            )
        
        if st.button("➕ Add Suspect", type="primary", use_container_width=True):
            if up_photo and up_name:
                with st.spinner("🔄 Processing..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    tfile.write(up_photo.read())
                    tfile.close()
                    
                    sys = init_system(st.session_state.tolerance)
                    
                    if sys:
                        success = sys.add_suspect(up_name, tfile.name)
                        if success:
                            st.success(f"✅ {up_name} added!")
                            st.cache_resource.clear()
                        else:
                            st.error("❌ No face detected in photo")
                    
                    os.remove(tfile.name)
            else:
                st.error("⚠️ Please provide both photo and name")

    # Control Buttons
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 START", type="primary", use_container_width=True):
            if source_type == "Upload Video" and 'path' not in source_config:
                st.error("Please upload video first")
            else:
                st.session_state.system_started = True
                st.session_state.video_config = source_config
                st.session_state.source_type = source_type
                st.session_state.alerts = []
                st.session_state.total_detections = 0
                st.session_state.highest_match = 0
                st.rerun()
    
    with col2:
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.system_started = False
            st.rerun()

# --- MAIN DASHBOARD ---
st.title("🛡️ SkyGuard Face Recognition System")

if st.session_state.system_started:
    
    # Initialize System
    with st.spinner("🔄 Loading AI System..."):
        face_system = init_system(st.session_state.tolerance)
    
    if not face_system:
        st.error("❌ Failed to initialize system")
        st.info("Make sure you've installed: pip install face-recognition")
        st.stop()
    
    # Check if suspects exist
    if len(face_system.suspect_names) == 0:
        st.warning("⚠️ No suspects in database! Add suspects in the sidebar first.")
        st.stop()
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stats-box">🟢 ACTIVE<br><small>System Online</small></div>', 
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stats-box">{st.session_state.total_detections}<br><small>Total Detections</small></div>', 
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stats-box">{len(st.session_state.alerts)}<br><small>Alerts Triggered</small></div>', 
                    unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stats-box">{st.session_state.highest_match:.1f}%<br><small>Highest Match</small></div>', 
                    unsafe_allow_html=True)
    
    st.divider()
    
    # Main Layout
    col_video, col_alerts = st.columns([2, 1])
    
    with col_video:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col_alerts:
        st.subheader("🚨 Detection Log")
        alerts_placeholder = st.empty()

    # Video Capture
    cap = None
    
    try:
        if st.session_state.source_type == "Webcam":
            cap = cv2.VideoCapture(st.session_state.video_config['index'])
            
        elif st.session_state.source_type == "Upload Video":
            cap = cv2.VideoCapture(st.session_state.video_config['path'])
            
        elif st.session_state.source_type == "Drone Stream":
            cap = cv2.VideoCapture(st.session_state.video_config['url'])
            
    except Exception as e:
        st.error(f"❌ Cannot open video source: {e}")
        st.stop()

    if not cap or not cap.isOpened():
        st.error("❌ Failed to open video source")
        st.stop()
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_counter = 0
    
    # Main Processing Loop
    while st.session_state.system_started:
        ret, frame = cap.read()
        
        if not ret:
            if st.session_state.source_type == "Upload Video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            else:
                st.error("❌ Lost video connection")
                break
        
        frame_counter += 1
        
        # Resize for faster processing and display
        display_frame = cv2.resize(frame, (960, 540))
        output = display_frame.copy()
        
        # Draw existing detections (from previous processing)
        for face_data in st.session_state.last_faces:
            x, y, w, h = face_data['box']
            name = face_data['name']
            match_pct = face_data['match_percentage']
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Draw label with percentage
            label = f"{name}: {match_pct:.1f}%"
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Background for label
            cv2.rectangle(
                output, (x, y-label_h-15), (x+label_w+10, y), 
                (0, 0, 255), -1
            )
            
            # Text
            cv2.putText(
                output, label, (x+5, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        
        # Display frame
        out_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        video_placeholder.image(out_rgb, channels="RGB", use_container_width=True)
        
        # Process frame (every N frames)
        process_interval = st.session_state.get('process_interval', 3)
        
        if frame_counter % process_interval == 0:
            try:
                status_placeholder.info("🔍 Analyzing faces...")
                
                # --- SIMULATE DRONE GPS (Replace this with real drone SDK later) ---
                # Example: Coordinates for Jaipur, India
                drone_lat = 26.9124 + (np.random.random() * 0.001) 
                drone_long = 75.7873 + (np.random.random() * 0.001)
                
                # Run face recognition WITH COORDINATES
                _, detections, suspects = face_system.process_frame(
                    display_frame,
                    latitude=drone_lat,   # <--- Pass Lat
                    longitude=drone_long, # <--- Pass Long
                    draw_boxes=False
                )
                
                # Update tracking
                current_faces = []
                
                for suspect in suspects:
                    # Get box coordinates
                    if 'box' in suspect:
                        x, y, w, h = suspect['box']
                    else:
                        continue
                    
                    name = suspect['name']
                    match_pct = suspect['match_percentage']
                    
                    # Update highest match
                    if match_pct > st.session_state.highest_match:
                        st.session_state.highest_match = match_pct
                    
                    # Store face data
                    face_data = {
                        'box': [x, y, w, h],
                        'name': name,
                        'match_percentage': match_pct
                    }
                    current_faces.append(face_data)
                    
                    # Add to alerts
                    st.session_state.total_detections += 1
                    
                    alert_entry = {
                        "name": name,
                        "match_percentage": match_pct,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "frame": frame_counter
                    }
                    
                    st.session_state.alerts.insert(0, alert_entry)
                
                st.session_state.last_faces = current_faces
                
                # Status update
                status_placeholder.success(
                    f"✅ Frame {frame_counter} | "
                    f"Faces: {len(detections)} | "
                    f"Matched: {len(suspects)}"
                )
                
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
                print(f"Processing error: {e}")
        
        # Update alerts display
        with alerts_placeholder.container():
            if st.session_state.alerts:
                st.markdown("### Recent Detections")
                
                # Show last 8 alerts
                for alert in st.session_state.alerts[:8]:
                    alert_class = get_alert_class(alert['match_percentage'])
                    
                    st.markdown(
                        f'<div class="{alert_class}">'
                        f'<strong>🎯 {alert["name"]}</strong><br>'
                        f'<span class="match-percentage">{alert["match_percentage"]:.1f}%</span> Match<br>'
                        f'<small>Time: {alert["time"]} | Frame: {alert["frame"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("👀 Waiting for detections...")
        
        # Control frame rate
        time.sleep(0.08)  # ~12 FPS
    
    cap.release()
    
else:
    # Welcome Screen
    st.markdown("""
    ## 🎯 Professional Face Recognition System
    
    ### ✨ Features:
    - ✅ **High Accuracy** - State-of-the-art face matching
    - ✅ **Any Angle** - Works with side views, tilted faces
    - ✅ **Drone Ready** - Optimized for aerial footage
    - ✅ **Real-time** - Fast processing for live video
    - ✅ **Match Percentage** - Shows exact confidence score
    
    ### 🚀 Quick Start Guide:
    
    1. **Add Suspects** 👤
       - Upload clear face photos in the sidebar
       - System will analyze and store face patterns
       
    2. **Configure Settings** ⚙️
       - Adjust match sensitivity (50% recommended)
       - Set processing frequency
       
    3. **Select Video Source** 📹
       - Webcam for live testing
       - Upload video for analysis
       - Drone stream for real surveillance
       
    4. **Start Detection** 🚀
       - Click START button
       - System will detect and match faces
       - View match percentages in real-time
    
    ### 📊 Understanding Match Scores:
    - **80-100%** = High confidence match (RED alert)
    - **60-80%** = Medium confidence match (ORANGE alert)
    - **50-60%** = Low confidence match (YELLOW alert)
    
    ### 💡 Installation Required:
    ```bash
    pip install face-recognition
    pip install opencv-python
    pip install ultralytics
    ```
    
    ---
    
    👈 **Start by adding suspects in the sidebar!**
    """)
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔧 System Ready")
    with col2:
        st.info("👤 Add Suspects First")
    with col3:
        st.info("📹 Select Video Source")