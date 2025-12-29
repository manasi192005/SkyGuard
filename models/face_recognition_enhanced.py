
"""
Enhanced Face Recognition with Admin Alert System
Sends real-time location alerts to authorized admins when suspect detected
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

class AdminAlertSystem:
    """
    Alert system to notify admins when suspect is detected
    Sends location via email, SMS, and webhook
    """
    
    def __init__(self, config_path='data/admin_config.json'):
        """Initialize alert system"""
        self.config_path = config_path
        self.load_admin_config()
        self.alert_history = []
    
    def load_admin_config(self):
        """Load admin configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Default configuration
                self.config = {
                    'admins': [],
                    'email_enabled': False,
                    'sms_enabled': False,
                    'webhook_enabled': False,
                    'alert_cooldown': 60  # seconds between alerts for same suspect
                }
                self.save_admin_config()
        except Exception as e:
            print(f"Error loading admin config: {e}")
            self.config = {'admins': []}
    
    def save_admin_config(self):
        """Save admin configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving admin config: {e}")
    
    def add_admin(self, name, email=None, phone=None, role='security'):
        """Add an admin to the alert system"""
        admin = {
            'name': name,
            'email': email,
            'phone': phone,
            'role': role,
            'added_date': datetime.now().isoformat(),
            'active': True
        }
        
        self.config['admins'].append(admin)
        self.save_admin_config()
        print(f"✓ Admin added: {name}")
        return admin
    
    def send_email_alert(self, admin_email, suspect_name, latitude, longitude, confidence, image_path=None):
        """Send email alert to admin"""
        try:
            # Email configuration (use environment variables in production)
            smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.config.get('smtp_port', 587)
            sender_email = self.config.get('sender_email', 'skyguard@example.com')
            sender_password = self.config.get('sender_password', '')
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = admin_email
            msg['Subject'] = f'🚨 SKYGUARD ALERT: Suspect Detected - {suspect_name}'
            
            # Google Maps link
            maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
            
            # Email body
            body = f"""
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
    <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        
        <div style="background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">🚨 SUSPECT ALERT</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px;">SkyGuard Surveillance System</p>
        </div>
        
        <div style="background-color: #fff3cd; border-left: 4px solid #ff8800; padding: 15px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0; color: #cc0000;">⚠️ IMMEDIATE ACTION REQUIRED</h2>
            <p style="margin: 0; font-size: 14px; color: #856404;">A registered suspect has been detected by the SkyGuard system.</p>
        </div>
        
        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #333;">Suspect Information</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;"><strong>Name:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6; color: #cc0000;">{suspect_name}</td>
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
                <tr>
                    <td style="padding: 10px 0;"><strong>GPS Coordinates:</strong></td>
                    <td style="padding: 10px 0;">{latitude}, {longitude}</td>
                </tr>
            </table>
            
            <div style="margin-top: 20px; text-align: center;">
                <a href="{maps_link}" 
                   style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; 
                          font-weight: bold; font-size: 16px;">
                    📍 View on Google Maps
                </a>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background-color: white; border-radius: 5px; text-align: center;">
                <small style="color: #666;">Click the button above to see exact location on map</small>
            </div>
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
                Team GDuo | VESIT, Mumbai<br>
                <small>Do not reply to this email</small>
            </p>
        </div>
        
    </div>
</body>
</html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            if sender_password:  # Only send if password configured
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.send_message(msg)
                    print(f"✓ Email alert sent to {admin_email}")
                    return True
            else:
                print(f"⚠ Email alert simulated for {admin_email} (no SMTP configured)")
                print(f"   Suspect: {suspect_name} at {latitude}, {longitude}")
                return True
                
        except Exception as e:
            print(f"✗ Email alert failed: {e}")
            return False
    
    def send_sms_alert(self, admin_phone, suspect_name, latitude, longitude):
        """Send SMS alert (using Twilio or similar service)"""
        try:
            # Twilio configuration (example)
            twilio_sid = self.config.get('twilio_sid')
            twilio_token = self.config.get('twilio_token')
            twilio_from = self.config.get('twilio_from')
            
            if not all([twilio_sid, twilio_token, twilio_from]):
                print(f"⚠ SMS alert simulated for {admin_phone}")
                print(f"   Message: Suspect {suspect_name} detected at {latitude}, {longitude}")
                return True
            
            # Twilio API call would go here
            from twilio.rest import Client
            client = Client(twilio_sid, twilio_token)
            
            maps_link = f"https://maps.google.com?q={latitude},{longitude}"
            message_body = f"""
🚨 SKYGUARD ALERT

Suspect: {suspect_name}
Location: {latitude}, {longitude}
Time: {datetime.now().strftime("%H:%M:%S")}

View location: {maps_link}

- SkyGuard System
            """
            
            message = client.messages.create(
                body=message_body,
                from_=twilio_from,
                to=admin_phone
            )
            
            print(f"✓ SMS alert sent to {admin_phone}")
            return True
            
        except Exception as e:
            print(f"✗ SMS alert failed: {e}")
            return False
    
    def send_webhook_alert(self, suspect_name, latitude, longitude, confidence, image_path=None):
        """Send webhook notification (Slack, Discord, Teams, etc.)"""
        try:
            webhook_url = self.config.get('webhook_url')
            
            if not webhook_url:
                print("⚠ Webhook not configured")
                return False
            
            # Prepare payload
            payload = {
                'suspect_name': suspect_name,
                'latitude': latitude,
                'longitude': longitude,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'maps_link': f"https://maps.google.com?q={latitude},{longitude}"
            }
            
            # Send to webhook
            response = requests.post(webhook_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                print(f"✓ Webhook alert sent")
                return True
            else:
                print(f"✗ Webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Webhook alert failed: {e}")
            return False
    
    def send_alert(self, suspect_name, latitude, longitude, confidence, image_path=None):
        """
        Send alert to all active admins via all enabled channels
        """
        print(f"\n{'='*60}")
        print(f"🚨 SENDING ALERTS FOR SUSPECT: {suspect_name}")
        print(f"📍 Location: {latitude}, {longitude}")
        print(f"🎯 Confidence: {confidence:.0%}")
        print(f"{'='*60}")
        
        alert_sent = False
        
        # Get active admins
        active_admins = [a for a in self.config.get('admins', []) if a.get('active', True)]
        
        if not active_admins:
            print("⚠ No active admins configured!")
            return False
        
        # Send to each admin
        for admin in active_admins:
            print(f"\n→ Alerting admin: {admin['name']} ({admin.get('role', 'security')})")
            
            # Email alert
            if self.config.get('email_enabled', True) and admin.get('email'):
                if self.send_email_alert(
                    admin['email'], 
                    suspect_name, 
                    latitude, 
                    longitude, 
                    confidence, 
                    image_path
                ):
                    alert_sent = True
            
            # SMS alert
            if self.config.get('sms_enabled', False) and admin.get('phone'):
                if self.send_sms_alert(
                    admin['phone'], 
                    suspect_name, 
                    latitude, 
                    longitude
                ):
                    alert_sent = True
        
        # Webhook alert (sent once, not per admin)
        if self.config.get('webhook_enabled', False):
            if self.send_webhook_alert(
                suspect_name, 
                latitude, 
                longitude, 
                confidence, 
                image_path
            ):
                alert_sent = True
        
        # Log alert
        alert_record = {
            'suspect_name': suspect_name,
            'latitude': latitude,
            'longitude': longitude,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'admins_notified': len(active_admins),
            'channels_used': []
        }
        
        if self.config.get('email_enabled'):
            alert_record['channels_used'].append('email')
        if self.config.get('sms_enabled'):
            alert_record['channels_used'].append('sms')
        if self.config.get('webhook_enabled'):
            alert_record['channels_used'].append('webhook')
        
        self.alert_history.append(alert_record)
        
        print(f"\n{'='*60}")
        if alert_sent:
            print(f"✅ Alerts sent successfully to {len(active_admins)} admin(s)")
        else:
            print(f"⚠ Alert notification simulated (configure SMTP/SMS for real alerts)")
        print(f"{'='*60}\n")
        
        return alert_sent


class FaceRecognitionEnhanced:
    """Enhanced Face Recognition with Real-time Admin Alerts"""
    
    def __init__(self, suspects_db_path='data/suspects', confidence_threshold=0.50):
        """Initialize Enhanced Face Recognition System"""
        self.suspects_db_path = suspects_db_path
        self.confidence_threshold = confidence_threshold
        
        # Create suspects directory
        os.makedirs(suspects_db_path, exist_ok=True)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize alert system
        self.alert_system = AdminAlertSystem()
        
        # Load suspects
        self.load_suspects_database()
        self.load_suspect_encodings()
        
        # Detection history
        self.recent_detections = {}
        self.alert_cooldown = 60  # seconds
    
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
    
    def add_suspect(self, name, image_path, description='', uploaded_by='Admin'):
        """Add a new suspect to the database"""
        import shutil
        dest_path = os.path.join(self.suspects_db_path, f"{name}.jpg")
        
        if os.path.exists(image_path):
            shutil.copy(image_path, dest_path)
        else:
            dest_path = image_path
        
        suspect_info = {
            'name': name,
            'image_path': dest_path,
            'description': description,
            'uploaded_by': uploaded_by,
            'added_date': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.suspects.append(suspect_info)
        
        # Save metadata
        metadata_path = os.path.join(self.suspects_db_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.suspects, f, indent=2)
        
        # Load encoding
        img = cv2.imread(dest_path)
        if img is not None:
            self.suspect_encodings[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"✓ Added suspect: {name} (uploaded by: {uploaded_by})")
        return suspect_info
    
    def detect_faces(self, frame):
        """Detect all faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': 1.0
            })
        
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
        """
        Process frame: detect faces and identify suspects
        Send alerts with GPS location when suspect found
        
        Args:
            frame: Input video frame
            latitude: Current GPS latitude
            longitude: Current GPS longitude
            draw_boxes: Whether to draw detection boxes
            
        Returns:
            processed_frame, detections, recognized_suspects
        """
        detections = self.detect_faces(frame)
        recognized_suspects = []
        
        output_frame = frame.copy()
        current_time = datetime.now()
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Try to match face
            match = self.match_face(frame, bbox)
            
            if match:
                # Check if should send alert
                if self.should_send_alert(match['name']):
                    # Send alert to admins with GPS location
                    self.alert_system.send_alert(
                        suspect_name=match['name'],
                        latitude=latitude,
                        longitude=longitude,
                        confidence=match['confidence'],
                        image_path=None
                    )
                    
                    # Update detection time
                    self.recent_detections[match['name']] = current_time
                
                recognized_suspects.append(match)
                
                if draw_boxes:
                    # Draw RED box for suspects
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    
                    # Draw alert background
                    alert_h = 80
                    cv2.rectangle(output_frame, (x, y-alert_h), (x+w+250, y), (0, 0, 255), -1)
                    
                    # Suspect name
                    cv2.putText(output_frame, f"SUSPECT: {match['name']}", 
                               (x+5, y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Confidence
                    cv2.putText(output_frame, f"Confidence: {match['confidence']:.0%}", 
                               (x+5, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # GPS location
                    cv2.putText(output_frame, f"GPS: {latitude:.4f}, {longitude:.4f}", 
                               (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Blinking alert
                    if int(current_time.timestamp() * 2) % 2 == 0:
                        cv2.putText(output_frame, "!!! ALERT SENT !!!", 
                                   (x+w+10, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if draw_boxes:
                    # Draw GREEN box for unknown faces
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return output_frame, detections, recognized_suspects


# Test module
if __name__ == '__main__':
    print("Enhanced Face Recognition with Admin Alerts - Test")
    
    # Initialize system
    face_rec = FaceRecognitionEnhanced()
    
    # Add test admin
    face_rec.alert_system.add_admin(
        name="Rushil Patil",
        email="rushil.patil@vesit.edu.in",
        phone="+919876543210",
        role="Team Leader"
    )
    
    face_rec.alert_system.add_admin(
        name="Manasi Ghalsasi",
        email="manasi.ghalsasi@vesit.edu.in",
        phone="+919876543211",
        role="Team Member"
    )
    
    print(f"\n✓ System loaded")
    print(f"✓ Suspects in database: {len(face_rec.suspects)}")
    print(f"✓ Admins configured: {len(face_rec.alert_system.config['admins'])}")
    print(f"✓ Alert system ready")
    
    # Simulate detection
    print("\n--- Simulating Suspect Detection ---")
    if face_rec.suspects:
        test_suspect = face_rec.suspects[0]['name']
        face_rec.alert_system.send_alert(
            suspect_name=test_suspect,
            latitude=19.0760,
            longitude=72.8777,
            confidence=0.95
        )
