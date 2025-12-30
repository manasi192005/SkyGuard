"""
SkyGuard - SQLite Authentication System
Government ID Verification with Local Database
No Firebase required - completely self-contained
"""

import sqlite3
import hashlib
import secrets
import os
import json
from datetime import datetime, timedelta
from rbac_manager import BlockchainRBAC, Role
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re


class EnhancedIDVerifier:
    """
    Production-grade Government ID Verification
    - Multi-layer OCR validation
    - Pattern matching for Indian government IDs
    - Security features detection
    - Fraud detection
    """
    
    # Comprehensive ID patterns for Indian government agencies
    VALID_ID_PATTERNS = {
        'POLICE': {
            'required_keywords': ['POLICE', 'CONSTABLE', 'INSPECTOR', 'OFFICER'],
            'optional_keywords': ['STATE', 'CITY', 'COMMISSIONER', 'DEPARTMENT'],
            'id_formats': [
                r'[A-Z]{2}POL\d{6,8}',
                r'POL/[A-Z]{2}/\d{6}',
                r'[A-Z]{2}\d{7,8}'
            ],
            'security_features': ['HOLOGRAM', 'EMBOSSED', 'STAMP', 'SEAL'],
            'role': Role.POLICE,
            'min_confidence': 70
        },
        'DEFENSE': {
            'required_keywords': ['ARMY', 'NAVY', 'AIR FORCE', 'DEFENCE', 'DEFENSE'],
            'optional_keywords': ['INDIAN', 'ARMED FORCES', 'SOLDIER', 'OFFICER', 'CAPTAIN'],
            'id_formats': [
                r'[A-Z]{2}\d{7}',
                r'IC-\d{7}',
                r'IN-\d{6}'
            ],
            'security_features': ['MINISTRY OF DEFENCE', 'GOVT OF INDIA'],
            'role': Role.ADMIN,
            'min_confidence': 75
        },
        'NDRF': {
            'required_keywords': ['NDRF', 'NATIONAL DISASTER RESPONSE FORCE'],
            'optional_keywords': ['DISASTER', 'RESCUE', 'RELIEF', 'EMERGENCY'],
            'id_formats': [
                r'NDRF/\d{6,8}',
                r'DR/[A-Z]{2}/\d{5}'
            ],
            'security_features': ['MINISTRY OF HOME AFFAIRS'],
            'role': Role.DISASTER,
            'min_confidence': 70
        },
        'SDRF': {
            'required_keywords': ['SDRF', 'STATE DISASTER RESPONSE'],
            'optional_keywords': ['DISASTER', 'RESCUE', 'RELIEF'],
            'id_formats': [
                r'SDRF/[A-Z]{2}/\d{5,7}'
            ],
            'security_features': [],
            'role': Role.DISASTER,
            'min_confidence': 65
        },
        'MEDICAL_EMERGENCY': {
            'required_keywords': ['AMBULANCE', 'PARAMEDIC', 'EMERGENCY MEDICAL'],
            'optional_keywords': ['EMS', 'EMT', 'MEDICAL OFFICER', 'HEALTH'],
            'id_formats': [
                r'EMS/\d{6}',
                r'MED/[A-Z]{2}/\d{5}'
            ],
            'security_features': ['HEALTH DEPARTMENT'],
            'role': Role.MEDICAL,
            'min_confidence': 65
        },
        'MUNICIPAL': {
            'required_keywords': ['MUNICIPAL CORPORATION', 'CIVIC BODY', 'MUNICIPAL'],
            'optional_keywords': ['CITY', 'TOWN', 'ADMINISTRATOR', 'OFFICER'],
            'id_formats': [
                r'MC/[A-Z]{2,4}/\d{5}',
                r'MUN-\d{6,8}'
            ],
            'security_features': [],
            'role': Role.MUNICIPAL,
            'min_confidence': 60
        }
    }
    
    def __init__(self):
        print("✓ Enhanced ID Verifier initialized")
    
    def preprocess_image(self, image_path):
        """Advanced image preprocessing for better OCR"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None
            
            original = img.copy()
            
            max_dimension = 2000
            height, width = img.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned, original
            
        except Exception as e:
            print(f"⚠ Preprocessing error: {e}")
            return None, None
    
    def extract_text_multiple_methods(self, image_path):
        """Extract text using multiple OCR configurations"""
        processed, original = self.preprocess_image(image_path)
        
        if processed is None:
            return ""
        
        all_text = []
        
        try:
            text1 = pytesseract.image_to_string(processed, config='--psm 6')
            all_text.append(text1)
        except:
            pass
        
        try:
            text2 = pytesseract.image_to_string(processed, config='--psm 11')
            all_text.append(text2)
        except:
            pass
        
        try:
            text3 = pytesseract.image_to_string(
                cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                config='--psm 6'
            )
            all_text.append(text3)
        except:
            pass
        
        combined_text = "\n".join(all_text).upper()
        return combined_text
    
    def validate_security_features(self, text, security_features):
        """Check for security features in ID card"""
        found_features = []
        for feature in security_features:
            if feature.upper() in text:
                found_features.append(feature)
        return found_features
    
    def extract_id_number(self, text, id_formats):
        """Extract ID number from text using regex patterns"""
        for pattern in id_formats:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return None
    
    def calculate_confidence_score(self, text, agency_config):
        """Calculate verification confidence based on multiple factors"""
        score = 0
        details = []
        
        # Required keywords (40 points)
        required_found = 0
        for keyword in agency_config['required_keywords']:
            if keyword in text:
                required_found += 1
                details.append(f"✓ Found: {keyword}")
        
        required_score = (required_found / len(agency_config['required_keywords'])) * 40
        score += required_score
        
        # Optional keywords (20 points)
        optional_found = 0
        for keyword in agency_config['optional_keywords']:
            if keyword in text:
                optional_found += 1
        
        if agency_config['optional_keywords']:
            optional_score = (optional_found / len(agency_config['optional_keywords'])) * 20
            score += optional_score
        else:
            score += 10
        
        # ID number format (25 points)
        id_number = self.extract_id_number(text, agency_config['id_formats'])
        if id_number:
            score += 25
            details.append(f"✓ ID Number: {id_number}")
        else:
            details.append("✗ No valid ID number found")
        
        # Security features (15 points)
        security_features = self.validate_security_features(
            text, 
            agency_config['security_features']
        )
        
        if agency_config['security_features']:
            security_score = (len(security_features) / len(agency_config['security_features'])) * 15
            score += security_score
            for feature in security_features:
                details.append(f"✓ Security: {feature}")
        else:
            score += 7.5
        
        return min(score, 100), details, id_number
    
    def detect_fraud(self, image_path):
        """Basic fraud detection checks"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return True, "Could not read image"
            
            height, width = img.shape[:2]
            if height < 300 or width < 400:
                return True, "Image resolution too low (possible screenshot)"
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < 50:
                return True, "Image appears to be heavily processed or fake"
            
            aspect_ratio = width / height
            if aspect_ratio < 1.3 or aspect_ratio > 2.0:
                return False, "Warning: Unusual aspect ratio for ID card"
            
            return False, "No fraud indicators detected"
            
        except Exception as e:
            return False, f"Fraud check error: {e}"
    
    def verify_government_id(self, image_path):
        """Complete ID verification pipeline"""
        print(f"\n{'='*70}")
        print(f"🔍 Starting ID Verification")
        print(f"{'='*70}\n")
        
        # Fraud detection
        print("🛡️ Step 1: Fraud Detection...")
        is_fraud, fraud_msg = self.detect_fraud(image_path)
        print(f"   {fraud_msg}")
        
        if is_fraud:
            print("\n❌ VERIFICATION FAILED: Fraud detected")
            return False, None, None, 0, [fraud_msg], None
        
        # Extract text
        print("\n📄 Step 2: Text Extraction...")
        text = self.extract_text_multiple_methods(image_path)
        
        if not text or len(text) < 20:
            print("❌ Could not extract sufficient text from ID")
            return False, None, None, 0, ["Insufficient text extracted"], None
        
        print(f"   Extracted {len(text)} characters")
        
        # Match against agency patterns
        print("\n🎯 Step 3: Pattern Matching...")
        
        best_match = None
        best_score = 0
        best_details = []
        best_id_number = None
        
        for agency, config in self.VALID_ID_PATTERNS.items():
            score, details, id_number = self.calculate_confidence_score(text, config)
            
            print(f"\n   {agency}: {score:.1f}% (Required: {config['min_confidence']}%)")
            
            if score > best_score:
                best_score = score
                best_match = agency
                best_details = details
                best_id_number = id_number
        
        # Verification decision
        print(f"\n{'='*70}")
        
        if best_match:
            config = self.VALID_ID_PATTERNS[best_match]
            
            if best_score >= config['min_confidence']:
                print(f"✅ VERIFICATION SUCCESSFUL")
                print(f"{'='*70}")
                print(f"Agency: {best_match}")
                print(f"Role: {config['role'].name}")
                print(f"Confidence: {best_score:.1f}%")
                if best_id_number:
                    print(f"ID Number: {best_id_number}")
                print(f"{'='*70}\n")
                
                return True, best_match, config['role'], best_score, best_details, best_id_number
            else:
                print(f"❌ VERIFICATION FAILED")
                print(f"Best Match: {best_match} ({best_score:.1f}%)")
                print(f"Required: {config['min_confidence']}%")
                print(f"{'='*70}\n")
        
        return False, best_match, None, best_score, best_details, best_id_number


class SQLiteAuthSystem:
    """
    Complete Authentication System with SQLite Database
    No external dependencies - fully self-contained
    """
    
    def __init__(self, db_path='data/skyguard_auth.db'):
        
        print("\n" + "="*70)
        print("🔐 SkyGuard SQLite Authentication System")
        print("="*70 + "\n")
        
        self.db_path = db_path
        self.init_database()
        
        # Initialize Blockchain RBAC
        self.rbac = BlockchainRBAC()
        
        # Initialize ID Verifier
        self.id_verifier = EnhancedIDVerifier()
        
        # Security settings
        self.max_login_attempts = 3
        self.lockout_minutes = 30
        
        print("✅ Authentication system initialized\n")
    
    def init_database(self):
        """Initialize SQLite database with tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                phone TEXT,
                agency TEXT NOT NULL,
                role TEXT NOT NULL,
                eth_address TEXT NOT NULL,
                id_number TEXT,
                id_card_hash TEXT NOT NULL,
                verification_confidence REAL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                active INTEGER DEFAULT 1,
                verified INTEGER DEFAULT 1
            )
        ''')
        
        # Audit logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                user_id TEXT,
                email TEXT,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_token TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                active INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("✓ Database initialized")
    
    def hash_password(self, password, salt=None):
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return pwd_hash.hex(), salt
    
    def verify_password(self, password, password_hash, salt):
        """Verify password against hash"""
        pwd_hash, _ = self.hash_password(password, salt)
        return pwd_hash == password_hash
    
    def register_user(self, email, password, full_name, phone, id_card_path, 
                     id_card_number=None):
        """Register new user with government ID verification"""
        
        print(f"\n{'='*70}")
        print(f"📝 REGISTRATION REQUEST")
        print(f"{'='*70}")
        print(f"Email: {email}")
        print(f"Name: {full_name}")
        print(f"{'='*70}\n")
        
        # Validation
        if not all([email, password, full_name, id_card_path]):
            return {'success': False, 'error': 'All fields required'}
        
        if len(password) < 8:
            return {'success': False, 'error': 'Password must be at least 8 characters'}
        
        # Check if email already exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT email FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return {'success': False, 'error': 'Email already registered'}
        conn.close()
        
        # Step 1: Verify Government ID
        print("🔍 STEP 1: Government ID Verification\n")
        
        is_valid, agency, role, confidence, details, extracted_id = \
            self.id_verifier.verify_government_id(id_card_path)
        
        if not is_valid:
            return {
                'success': False,
                'error': 'Government ID verification failed',
                'confidence': confidence,
                'details': details
            }
        
        # Step 2: Register on Blockchain
        print("\n⛓️ STEP 2: Blockchain Registration\n")
        
        try:
            username = email.split('@')[0]
            blockchain_user = self.rbac.add_user(
                username=username,
                email=email,
                role=role,
                added_by='SYSTEM'
            )
            
            eth_address = blockchain_user['eth_address']
            print(f"✅ Blockchain: {eth_address}\n")
            
        except Exception as e:
            return {'success': False, 'error': f'Blockchain error: {str(e)}'}
        
        # Step 3: Store in Database
        print("💾 STEP 3: Storing User Data\n")
        
        try:
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Hash ID card
            with open(id_card_path, 'rb') as f:
                id_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Generate user ID
            user_id = secrets.token_urlsafe(16)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (
                    user_id, username, email, password_hash, salt,
                    full_name, phone, agency, role, eth_address,
                    id_number, id_card_hash, verification_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, username, email, password_hash, salt,
                full_name, phone, agency, role.name, eth_address,
                extracted_id or id_card_number, id_hash, confidence
            ))
            
            # Log registration
            cursor.execute('''
                INSERT INTO audit_logs (action, user_id, email, details)
                VALUES (?, ?, ?, ?)
            ''', ('USER_REGISTERED', user_id, email, f'Agency: {agency}, Role: {role.name}'))
            
            conn.commit()
            conn.close()
            
            print("✅ User data stored\n")
            
        except Exception as e:
            self.rbac.deactivate_user(username)
            return {'success': False, 'error': f'Database error: {str(e)}'}
        
        print(f"{'='*70}")
        print(f"✅ REGISTRATION SUCCESSFUL")
        print(f"{'='*70}")
        print(f"User: {full_name}")
        print(f"Email: {email}")
        print(f"Agency: {agency}")
        print(f"Role: {role.name}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"{'='*70}\n")
        
        return {
            'success': True,
            'user_id': user_id,
            'eth_address': eth_address,
            'role': role.name,
            'agency': agency,
            'confidence': confidence
        }
    
    def authenticate_user(self, email, password):
        """Authenticate user login"""
        
        print(f"\n{'='*70}")
        print(f"🔐 LOGIN REQUEST")
        print(f"{'='*70}")
        print(f"Email: {email}")
        print(f"{'='*70}\n")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if not user:
            self._log_failed_login(cursor, None, email, "User not found")
            conn.commit()
            conn.close()
            return {'success': False, 'error': 'Invalid email or password'}
        
        # Parse user data
        columns = [desc[0] for desc in cursor.description]
        user_data = dict(zip(columns, user))
        
        # Check if account is locked
        if user_data['locked_until']:
            locked_until = datetime.fromisoformat(user_data['locked_until'])
            if datetime.now() < locked_until:
                remaining = int((locked_until - datetime.now()).seconds / 60)
                conn.close()
                return {
                    'success': False,
                    'error': f'Account locked. Try again in {remaining} minutes.',
                    'locked': True
                }
        
        # Check if active
        if not user_data['active']:
            conn.close()
            return {'success': False, 'error': 'Account deactivated'}
        
        # Verify password
        if not self.verify_password(password, user_data['password_hash'], user_data['salt']):
            self._log_failed_login(cursor, user_data['user_id'], email, "Invalid password")
            
            # Increment attempts
            attempts = user_data['login_attempts'] + 1
            locked_until = None
            
            if attempts >= self.max_login_attempts:
                locked_until = (datetime.now() + timedelta(minutes=self.lockout_minutes)).isoformat()
                print(f"⚠️ Account locked until {locked_until}")
            
            cursor.execute('''
                UPDATE users 
                SET login_attempts = ?, locked_until = ?
                WHERE user_id = ?
            ''', (attempts, locked_until, user_data['user_id']))
            
            conn.commit()
            conn.close()
            return {'success': False, 'error': 'Invalid email or password'}
        
        # Verify blockchain
        role = self.rbac.get_user_role(user_data['username'])
        
        if not role:
            conn.close()
            return {'success': False, 'error': 'Blockchain verification failed'}
        
        # Reset attempts on successful login
        cursor.execute('''
            UPDATE users 
            SET login_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (user_data['user_id'],))
        
        # Log successful login
        cursor.execute('''
            INSERT INTO audit_logs (action, user_id, email)
            VALUES (?, ?, ?)
        ''', ('LOGIN_SUCCESS', user_data['user_id'], email))
        
        conn.commit()
        conn.close()
        
        # Get permissions
        permissions = self.rbac.get_user_permissions(user_data['username'])
        
        print(f"{'='*70}")
        print(f"✅ LOGIN SUCCESSFUL")
        print(f"{'='*70}\n")
        
        return {
            'success': True,
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'email': email,
            'full_name': user_data['full_name'],
            'role': role.name,
            'agency': user_data['agency'],
            'eth_address': user_data['eth_address'],
            'id_number': user_data['id_number'],
            'permissions': [p.value for p in permissions]
        }
    
    def _log_failed_login(self, cursor, user_id, email, reason):
        """Log failed login attempt"""
        cursor.execute('''
            INSERT INTO audit_logs (action, user_id, email, details)
            VALUES (?, ?, ?, ?)
        ''', ('LOGIN_FAILED', user_id, email, reason))
    
    def get_user_info(self, username):
        """Get user information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        conn.close()
        
        if not user:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        user_data = dict(zip(columns, user))
        
        role = self.rbac.get_user_role(username)
        permissions = self.rbac.get_user_permissions(username)
        
        return {
            'username': username,
            'email': user_data['email'],
            'full_name': user_data['full_name'],
            'role': role.name if role else 'NONE',
            'agency': user_data['agency'],
            'eth_address': user_data['eth_address'],
            'permissions': [p.value for p in permissions]
        }
    
    def verify_permission(self, username, permission):
        """Check if user has permission"""
        return self.rbac.has_permission(username, permission)


if __name__ == '__main__':
    # Quick test
    auth_system = SQLiteAuthSystem()
    print("✅ System ready!")