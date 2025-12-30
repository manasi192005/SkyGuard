"""
SkyGuard - Authentication Web Interface (SQLite)
No Firebase - Completely Self-Contained
"""

import streamlit as st
import tempfile
import os
from datetime import datetime
from auth_system import SQLiteAuthSystem
from rbac_manager import Permission
import time

# Page config
st.set_page_config(
    page_title="SkyGuard - Secure Login",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
    .auth-container {
        background: white; padding: 50px; border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4); max-width: 600px; margin: 50px auto;
    }
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 25px; border-radius: 15px; text-align: center;
        margin: 20px 0; animation: fadeIn 0.5s;
    }
    .error-box {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white; padding: 25px; border-radius: 15px; margin: 20px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6600 100%);
        color: white; padding: 20px; border-radius: 10px; margin: 15px 0;
    }
    .info-box {
        background: #f0f4ff; padding: 20px; border-radius: 10px;
        border-left: 5px solid #667eea; margin: 15px 0;
    }
    .step-indicator { display: flex; justify-content: space-between; margin: 30px 0; }
    .step {
        flex: 1; text-align: center; padding: 10px; border-radius: 25px;
        margin: 0 5px; background: #e0e0e0; color: #666; font-weight: bold;
    }
    .step.active { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .step.completed { background: #38ef7d; color: white; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: bold; padding: 15px; border-radius: 10px;
        border: none; font-size: 16px; transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px); box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    .permission-badge {
        display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 8px 20px; border-radius: 25px; margin: 5px;
        font-size: 13px; font-weight: 600;
    }
    .agency-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 10px 25px; border-radius: 30px;
        font-size: 18px; font-weight: bold; display: inline-block;
    }
    .id-preview {
        border: 3px dashed #667eea; border-radius: 15px;
        padding: 20px; text-align: center; margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'registration_step' not in st.session_state:
    st.session_state.registration_step = 1
if 'reg_data' not in st.session_state:
    st.session_state.reg_data = {}

# Initialize auth system
if 'auth_system' not in st.session_state:
    with st.spinner("🔄 Initializing authentication system..."):
        try:
            st.session_state.auth_system = SQLiteAuthSystem()
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            st.info("💡 Make sure Tesseract OCR is installed")
            st.stop()

auth_system = st.session_state.auth_system


def show_login_page():
    """Login page"""
    
    st.markdown("""
    <div class='header-banner'>
        <h1>🛡️ SkyGuard</h1>
        <h3>Secure Government Access Portal</h3>
        <p>SQLite Database - No External Dependencies</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        # LOGIN TAB
        with tab1:
            st.markdown("### Secure Login")
            
            st.markdown("""
            <div class='info-box'>
                ℹ️ <strong>Security Notice:</strong><br>
                • Only authorized government personnel<br>
                • All login attempts are logged<br>
                • Account locked after 3 failed attempts (30 minutes)
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                email = st.text_input(
                    "📧 Official Email",
                    placeholder="officer@agency.gov.in"
                )
                
                password = st.text_input(
                    "🔒 Password",
                    type="password"
                )
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    submit = st.form_submit_button("🚀 Login", use_container_width=True)
                
                with col2:
                    if st.form_submit_button("❓ Help", use_container_width=True):
                        st.info("Contact: admin@skyguard.gov.in")
                
                if submit:
                    if not email or not password:
                        st.error("❌ Please enter both email and password")
                    else:
                        with st.spinner("🔐 Verifying credentials..."):
                            result = auth_system.authenticate_user(email, password)
                        
                        if result['success']:
                            st.session_state.authenticated = True
                            st.session_state.user_data = result
                            
                            st.markdown(f"""
                            <div class='success-box'>
                                <h2>✅ Login Successful!</h2>
                                <p>Welcome, {result['full_name']}</p>
                                <p>Redirecting...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            time.sleep(2)
                            st.rerun()
                        else:
                            if result.get('locked'):
                                st.markdown(f"""
                                <div class='error-box'>
                                    <h3>🔒 Account Locked</h3>
                                    <p>{result['error']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error(f"❌ {result['error']}")
        
        # REGISTRATION TAB
        with tab2:
            show_registration_form()
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_registration_form():
    """Multi-step registration"""
    
    st.markdown("### New User Registration")
    
    # Step indicator
    steps = ["Personal Info", "ID Verification", "Review & Submit"]
    step_html = '<div class="step-indicator">'
    for i, step in enumerate(steps, 1):
        if i < st.session_state.registration_step:
            step_html += f'<div class="step completed">✓ {step}</div>'
        elif i == st.session_state.registration_step:
            step_html += f'<div class="step active">{i}. {step}</div>'
        else:
            step_html += f'<div class="step">{i}. {step}</div>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)
    
    # STEP 1: Personal Information
    if st.session_state.registration_step == 1:
        st.markdown("#### Step 1: Personal Information")
        
        with st.form("personal_info_form"):
            full_name = st.text_input(
                "👤 Full Name",
                placeholder="As per government ID",
                value=st.session_state.reg_data.get('full_name', '')
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                email = st.text_input(
                    "📧 Official Email",
                    placeholder="officer@agency.gov.in",
                    value=st.session_state.reg_data.get('email', '')
                )
            
            with col2:
                phone = st.text_input(
                    "📱 Phone Number",
                    placeholder="+91XXXXXXXXXX",
                    value=st.session_state.reg_data.get('phone', '')
                )
            
            password = st.text_input("🔒 Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")
            
            if st.form_submit_button("Next →", use_container_width=True):
                errors = []
                
                if not full_name or len(full_name) < 3:
                    errors.append("Full name required")
                if not email or '@' not in email:
                    errors.append("Valid email required")
                if not phone or len(phone) < 10:
                    errors.append("Valid phone required")
                if not password or len(password) < 8:
                    errors.append("Password must be 8+ characters")
                if password != confirm_password:
                    errors.append("Passwords do not match")
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    st.session_state.reg_data.update({
                        'full_name': full_name,
                        'email': email,
                        'phone': phone,
                        'password': password
                    })
                    st.session_state.registration_step = 2
                    st.rerun()
    
    # STEP 2: ID Verification
    elif st.session_state.registration_step == 2:
        st.markdown("#### Step 2: Government ID Verification")
        
        st.markdown("""
        <div class='warning-box'>
            <strong>⚠️ ID Requirements</strong><br>
            • Valid government-issued ID<br>
            • Clear photo, readable text<br>
            • Accepted: Police, Defense, NDRF, Medical, Municipal
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("id_verification_form"):
            id_number = st.text_input("🆔 ID Number (Optional)", placeholder="e.g., MP123456")
            
            st.markdown("**📸 Upload Government ID Card**")
            id_card = st.file_uploader(
                "ID Card",
                type=['jpg', 'jpeg', 'png'],
                label_visibility="collapsed"
            )
            
            if id_card:
                st.markdown('<div class="id-preview">', unsafe_allow_html=True)
                st.image(id_card, caption="Uploaded ID", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.form_submit_button("← Back", use_container_width=True):
                    st.session_state.registration_step = 1
                    st.rerun()
            
            with col2:
                submit = st.form_submit_button("Verify →", use_container_width=True)
            
            if submit:
                if not id_card:
                    st.error("❌ Please upload ID card")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(id_card.read())
                        tmp_path = tmp.name
                    
                    try:
                        with st.spinner("🔍 Verifying ID (30-60 seconds)..."):
                            progress = st.progress(0)
                            status = st.empty()
                            
                            status.text("📄 Extracting text...")
                            progress.progress(33)
                            time.sleep(1)
                            
                            status.text("🎯 Matching patterns...")
                            progress.progress(66)
                            
                            is_valid, agency, role, confidence, details, extracted_id = \
                                auth_system.id_verifier.verify_government_id(tmp_path)
                            
                            progress.progress(100)
                            status.text("✅ Complete!")
                        
                        if is_valid:
                            st.markdown(f"""
                            <div class='success-box'>
                                <h3>✅ ID Verified!</h3>
                                <p><strong>Agency:</strong> {agency}</p>
                                <p><strong>Role:</strong> {role.name}</p>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.session_state.reg_data.update({
                                'id_card_path': tmp_path,
                                'id_number': extracted_id or id_number,
                                'agency': agency,
                                'role': role.name,
                                'confidence': confidence
                            })
                            
                            st.session_state.registration_step = 3
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.markdown(f"""
                            <div class='error-box'>
                                <h3>❌ Verification Failed</h3>
                                <p>Confidence: {confidence:.1f}%</p>
                                <p>Try scanning instead of photo</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    # STEP 3: Review
    elif st.session_state.registration_step == 3:
        st.markdown("#### Step 3: Review & Submit")
        
        reg = st.session_state.reg_data
        
        st.markdown(f"""
        <div class='info-box'>
            <h4>📋 Summary</h4>
            <p><strong>Name:</strong> {reg['full_name']}</p>
            <p><strong>Email:</strong> {reg['email']}</p>
            <p><strong>Agency:</strong> {reg['agency']}</p>
            <p><strong>Role:</strong> {reg['role']}</p>
            <p><strong>Confidence:</strong> {reg['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        agree = st.checkbox("I confirm all information is accurate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.registration_step = 2
                st.rerun()
        
        with col2:
            if st.button("✅ Complete", use_container_width=True, disabled=not agree):
                with st.spinner("🔄 Creating account..."):
                    result = auth_system.register_user(
                        email=reg['email'],
                        password=reg['password'],
                        full_name=reg['full_name'],
                        phone=reg['phone'],
                        id_card_path=reg['id_card_path'],
                        id_card_number=reg.get('id_number')
                    )
                
                if result['success']:
                    st.markdown(f"""
                    <div class='success-box'>
                        <h2>🎉 Registration Complete!</h2>
                        <p><strong>Agency:</strong> {result['agency']}</p>
                        <p><strong>Role:</strong> {result['role']}</p>
                        <p>You can now login!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if os.path.exists(reg['id_card_path']):
                        os.remove(reg['id_card_path'])
                    
                    st.session_state.registration_step = 1
                    st.session_state.reg_data = {}
                    time.sleep(3)
                    st.rerun()
                else:
                    st.markdown(f"""
                    <div class='error-box'>
                        <h3>❌ Registration Failed</h3>
                        <p>{result['error']}</p>
                    </div>
                    """, unsafe_allow_html=True)


def show_dashboard():
    """User dashboard"""
    
    user = st.session_state.user_data
    
    st.markdown("""
    <div class='header-banner'>
        <h2>🛡️ SkyGuard Surveillance System</h2>
        <p>Secure Government Portal</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 3, 1])
    
    with col1:
        st.markdown(f"### Welcome, {user['full_name']}!")
        st.caption(f"Last login: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 15px;'>
            <span class='agency-badge'>{user['agency']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.rerun()
    
    st.divider()
    
    # Info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 25px; border-radius: 15px; text-align: center;'>
            <h3>👤 Profile</h3>
            <p><strong>Role:</strong> {user['role']}</p>
            <p><strong>ID:</strong> {user.get('id_number', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    color: white; padding: 25px; border-radius: 15px; text-align: center;'>
            <h3>⛓️ Blockchain</h3>
            <p style='font-size: 10px;'>{user['eth_address']}</p>
            <p><strong>✅ Verified</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                    color: white; padding: 25px; border-radius: 15px; text-align: center;'>
            <h3>🔐 Access</h3>
            <p><strong>{len(user['permissions'])}</strong> Permissions</p>
            <p>{user['role']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Permissions
    st.markdown("### 🎯 Your Permissions")
    
    cols = st.columns(3)
    for idx, perm in enumerate(user['permissions']):
        with cols[idx % 3]:
            st.markdown(f'<span class="permission-badge">✅ {perm.replace("_", " ")}</span>', 
                       unsafe_allow_html=True)
    
    st.divider()
    
    # Actions
    st.markdown("### 🚀 System Access")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📹 Launch Surveillance", use_container_width=True, type="primary"):
            st.success("✅ Launching surveillance system...")
            # Add your app launch here
    
    with col2:
        if st.button("📊 Analytics", use_container_width=True):
            if Permission.VIEW_ANALYTICS.value in user['permissions']:
                st.success("✅ Opening analytics...")
            else:
                st.error("❌ Access denied")


def main():
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_dashboard()


if __name__ == '__main__':
    main()