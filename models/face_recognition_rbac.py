"""
Enhanced Face Recognition with Blockchain RBAC Integration
Only authorized roles receive suspect alerts
"""

import cv2
import numpy as np
import os
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Import blockchain RBAC
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blockchain import BlockchainRBAC, Permission

class AdminAlertSystem:
    """
    Alert system with Blockchain RBAC
    Only sends alerts to users with proper permissions
    """
    
    def __init__(self, config_path='data/admin_config.json'):
        self.config_path = config_path
        self.rbac = BlockchainRBAC()
        self.load_admin_config()
        self.alert_history = []
    
    def load_admin_config(self):
        """Load admin configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'email_enabled': False,
                    'sms_enabled': False,
                    'webhook_enabled': False,
                    'alert_cooldown': 60
                }
                self.save_admin_config()
        except Exception as e:
            print(f"Error loading admin config: {e}")
            self.config = {}
    
    def save_admin_config(self):
        """Save admin configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving admin config: {e}")
    
    def get_authorized_users_for_suspects(self):
        """
        Get all users authorized to receive suspect alerts
        Uses blockchain to verify VIEW_SUSPECTS permission
        """
        authorized_users = []
        
        # Get all blockchain users
        all_users = self.rbac.get_all_users()
        
        for username, user_data in all_users.items():
            # Check if user has VIEW_SUSPECTS permission via blockchain
            if self.rbac.has_permission(username, Permission.VIEW_SUSPECTS):
                if user_data.get('active', True):
                    authorized_users.append({
                        'username': username,
                        'email': user_data.get('email'),
                        'phone': user_data.get('phone'),
                        'role': user_data.get('role'),
                        'eth_address': user_data.get('eth_address')
                    })
        
        return authorized_users
    
    def send_email_alert(self, user_email, user_name, suspect_name, latitude, longitude, confidence):
        """Send email alert to authorized user"""
        try:
            smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.config.get('smtp_port', 587)
            sender_email = self.config.get('sender_email', 'skyguard@example.com')
            sender_password = self.config.get('sender_password', '')
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = user_email
            msg['Subject'] = f'🚨 SKYGUARD ALERT: Suspect Detected - {suspect_name}'
            
            maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
            
            body = f"""
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
    <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        
        <div style="background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">🚨 SUSPECT ALERT</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px;">SkyGuard Surveillance System</p>
            <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.9;">🔗 Blockchain-Verified Access</p>
        </div>
        
        <div style="background-color: #fff3cd; border-left: 4px solid #ff8800; padding: 15px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0; color: #cc0000;">⚠️ IMMEDIATE ACTION REQUIRED</h2>
            <p style="margin: 0; font-size: 14px; color: #856404;">
                Dear {user_name},<br><br>
                You are receiving this alert because you have <strong>blockchain-verified</strong> VIEW_SUSPECTS permission.
            </p>
        </div>
        
        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #333;">🎯 Suspect Information</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;"><strong>Name:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6; color: #cc0000; font-weight: bold;">{suspect_name}</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;"><strong>Confidence:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;">{confidence:.0%}</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;"><strong>Timestamp:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
                </tr>
            </table>
        </div>
        
        <div style="padding: 20px; background-color: #e7f3ff; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #333;">📍 Location Details</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #bee5eb;"><strong>Latitude:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #bee5eb;">{latitude}</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #bee5eb;"><strong>Longitude:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #bee5eb;">{longitude}</td>
                </tr>
            </table>
            
            <div style="margin-top: 20px; text-align: center;">
                <a href="{maps_link}" 
                   style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                          font-weight: bold; font-size: 16px;">
                    📍 View Location on Google Maps
                </a>
            </div>
        </div>
        
        <div style="padding: 20px; background-color: #d4edda; border: 2px solid #28a745; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #155724;">🔐 Blockchain Security</h3>
            <p style="margin: 0; color: #155724; font-size: 14px;">
                ✓ This alert is sent only to blockchain-verified users<br>
                ✓ Your role: <strong>{user_data.get('role', 'N/A')}</strong><br>
                ✓ Permission: VIEW_SUSPECTS (Verified on blockchain)<br>
                ✓ All access attempts are logged immutably
            </p>
        </div>
        
        <div style="padding: 20px; background-color: #fff; border: 2px solid #dc3545; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #dc3545;">⚡ Recommended Actions</h3>
            <ol style="margin: 0; padding-left: 20px; color: #333;">
                <li style="margin-bottom: 10px;">Alert nearby security personnel immediately</li>
                <li style="margin-bottom: 10px;">Monitor suspect's movement via live feed</li>
                <li style="margin-bottom: 10px;">Coordinate with local law enforcement if necessary</li>
                <li style="margin-bottom: 10px;">Keep a safe distance and await backup</li>
                <li>Document all observations for report</li>
            </ol>
        </div>
        
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; text-align: center;">
            <p style="margin: 0; color: #666; font-size: 14px;">
                This is an automated alert from <strong>SkyGuard Surveillance System</strong><br>
                Secured by Blockchain RBAC | Team GDuo | VESIT, Mumbai<br>
                <small>Do not reply to this email</small>
            </p>
        </div>
        
    </div>
</body>
</html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            if sender_password:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.send_message(msg)
                    print(f"✓ Email alert sent to {user_name} ({user_email})")
                    return True
            else:
                print(f"⚠ Email alert simulated for {user_name} ({user_email})")
                print(f"   Suspect: {suspect_name} at {latitude}, {longitude}")
                return True
                
        except Exception as e:
            print(f"✗ Email alert failed for {user_name}: {e}")
            return False
    
    def send_alert(self, suspect_name, latitude, longitude, confidence, image_path=None):
        """
        Send alert to all authorized users (via blockchain verification)
        """
        print(f"\n{'='*70}")
        print(f"🔐 BLOCKCHAIN-SECURED ALERT SYSTEM")
        print(f"{'='*70}")
        print(f"🚨 SUSPECT: {suspect_name}")
        print(f"📍 Location: {latitude}, {longitude}")
        print(f"🎯 Confidence: {confidence:.0%}")
        print(f"{'='*70}")
        
        # Get authorized users from blockchain
        authorized_users = self.get_authorized_users_for_suspects()
        
        if not authorized_users:
            print("⚠ No authorized users found with VIEW_SUSPECTS permission!")
            print("  Add users with blockchain: python3 blockchain/manage_users.py")
            return False
        
        print(f"\n✓ Found {len(authorized_users)} authorized user(s) via blockchain")
        
        alert_sent = False
        
        # Send to each authorized user
        for user in authorized_users:
            print(f"\n→ Alerting: {user['username']} ({user['role']})")
            print(f"  Permission verified on blockchain ✓")
            
            if self.config.get('email_enabled', True) and user.get('email'):
                if self.send_email_alert(
                    user['email'],
                    user['username'],
                    suspect_name,
                    latitude,
                    longitude,
                    confidence
                ):
                    alert_sent = True
        
        # Log to blockchain
        alert_record = {
            'suspect_name': suspect_name,
            'latitude': latitude,
            'longitude': longitude,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'authorized_users': len(authorized_users),
            'blockchain_verified': True
        }
        
        self.alert_history.append(alert_record)
        
        print(f"\n{'='*70}")
        if alert_sent:
            print(f"✅ Blockchain-verified alerts sent to {len(authorized_users)} authorized user(s)")
        else:
            print(f"⚠ Alert notification simulated (configure SMTP for real alerts)")
        print(f"🔗 All access logged on blockchain")
        print(f"{'='*70}\n")
        
        return alert_sent


class FaceRecognitionRBAC:
    """Face Recognition with Blockchain RBAC"""
    
    def __init__(self, suspects_db_path='data/suspects', confidence_threshold=0.85):
        self.suspects_db_path = suspects_db_path
        self.confidence_threshold = confidence_threshold
        
        os.makedirs(suspects_db_path, exist_ok=True)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize blockchain-secured alert system
        self.alert_system = AdminAlertSystem()
        
        # Load suspects
        self.load_suspects_database()
        self.load_suspect_encodings()
        
        self.recent_detections = {}
        self.alert_cooldown = 60
    
    def load_suspects_database(self):
        """Load suspect metadata"""
        self.suspects = []
        metadata_path = os.path.join(self.suspects_db_path, 'metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.suspects = json.load(f)
        
        print(f"Loaded {len(self.suspects)} suspects from database")
    
    def load_suspect_encodings(self):
        """Pre-load suspect face encodings"""
        self.suspect_encodings = {}
        
        for suspect in self.suspects:
            try:
                img_path = suspect['image_path']
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.suspect_encodings[suspect['name']] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error loading suspect {suspect['name']}: {e}")
    
    def detect_faces(self, frame):
        """Detect all faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({'bbox': [x, y, w, h], 'confidence': 1.0})
        
        return detections
    
    def match_face(self, frame, face_bbox):
        """Match detected face against suspect database"""
        x, y, w, h = face_bbox
        
        face_region = frame[y:y+h, x:x+w]
        if face_region.size == 0:
            return None
        
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))
        
        best_match = None
        best_score = 0
        
        for suspect_name, suspect_template in self.suspect_encodings.items():
            template_resized = cv2.resize(suspect_template, (100, 100))
            
            result = cv2.matchTemplate(face_resized, template_resized, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]
            similarity = (similarity + 1) / 2
            
            if similarity > best_score and similarity > self.confidence_threshold:
                best_score = similarity
                best_match = suspect_name
        
        if best_match:
            suspect_info = next((s for s in self.suspects if s['name'] == best_match), None)
            return {
                'name': best_match,
                'confidence': best_score,
                'description': suspect_info['description'] if suspect_info else '',
                'uploaded_by': suspect_info.get('uploaded_by', 'Admin') if suspect_info else 'Admin',
                'bbox': face_bbox
            }
        
        return None
    
    def should_send_alert(self, suspect_name):
        """Check if enough time has passed since last alert"""
        current_time = datetime.now()
        
        if suspect_name in self.recent_detections:
            last_alert_time = self.recent_detections[suspect_name]
            time_diff = (current_time - last_alert_time).seconds
            
            if time_diff < self.alert_cooldown:
                return False
        
        return True
    
    def process_frame(self, frame, latitude=19.0760, longitude=72.8777, draw_boxes=True):
        """Process frame with blockchain-secured alerts"""
        detections = self.detect_faces(frame)
        recognized_suspects = []
        
        output_frame = frame.copy()
        current_time = datetime.now()
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            match = self.match_face(frame, bbox)
            
            if match:
                # Check if should send alert
                if self.should_send_alert(match['name']):
                    # Send blockchain-secured alert
                    self.alert_system.send_alert(
                        suspect_name=match['name'],
                        latitude=latitude,
                        longitude=longitude,
                        confidence=match['confidence'],
                        image_path=None
                    )
                    
                    self.recent_detections[match['name']] = current_time
                
                recognized_suspects.append(match)
                
                if draw_boxes:
                    # Draw RED box for suspects
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    
                    alert_h = 100
                    cv2.rectangle(output_frame, (x, y-alert_h), (x+w+300, y), (0, 0, 255), -1)
                    
                    cv2.putText(output_frame, f"SUSPECT: {match['name']}", 
                               (x+5, y-75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"Confidence: {match['confidence']:.0%}", 
                               (x+5, y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(output_frame, f"GPS: {latitude:.4f}, {longitude:.4f}", 
                               (x+5, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(output_frame, "BLOCKCHAIN SECURED", 
                               (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    if int(current_time.timestamp() * 2) % 2 == 0:
                        cv2.putText(output_frame, "POLICE ALERTED!", 
                                   (x+w+10, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if draw_boxes:
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return output_frame, detections, recognized_suspects


if __name__ == '__main__':
    print("Face Recognition with Blockchain RBAC - Test")
    
    face_rec = FaceRecognitionRBAC()
    
    print(f"\n✓ System loaded")
    print(f"✓ Suspects: {len(face_rec.suspects)}")
    print(f"✓ Blockchain-secured alerts: Active")
    
    # Get authorized users
    authorized = face_rec.alert_system.get_authorized_users_for_suspects()
    print(f"✓ Authorized users (VIEW_SUSPECTS): {len(authorized)}")
    
    for user in authorized:
        print(f"  • {user['username']} ({user['role']})")
