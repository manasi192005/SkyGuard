"""
SkyGuard Backend API
Handles face recognition and video processing
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import tempfile
from datetime import datetime
import sys

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.face_recognition_enhanced import FaceRecognitionEnhanced

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize face recognition system
face_system = None
SUSPECTS_DIR = "data/suspects"
os.makedirs(SUSPECTS_DIR, exist_ok=True)


def init_face_system(tolerance=0.5):
    """Initialize face recognition system"""
    global face_system
    face_system = FaceRecognitionEnhanced(tolerance=tolerance)
    return face_system


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/init', methods=['POST'])
def initialize_system():
    """Initialize the face recognition system"""
    try:
        data = request.json
        tolerance = data.get('tolerance', 0.5)
        
        system = init_face_system(tolerance)
        
        return jsonify({
            'success': True,
            'message': 'System initialized',
            'suspects_count': len(system.suspect_names) if system else 0
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/suspects', methods=['GET'])
def get_suspects():
    """Get all suspects"""
    if not face_system:
        return jsonify({'success': False, 'error': 'System not initialized'}), 400
    
    suspects = []
    for name in face_system.suspect_names:
        suspects.append({
            'id': name,
            'name': name,
            'photoUrl': f'/api/suspect-photo/{name}',
            'addedDate': datetime.now().isoformat()
        })
    
    return jsonify({
        'success': True,
        'suspects': suspects
    })


@app.route('/api/suspect-photo/<name>', methods=['GET'])
def get_suspect_photo(name):
    """Get suspect photo"""
    photo_path = os.path.join(SUSPECTS_DIR, f"{name}.jpg")
    if os.path.exists(photo_path):
        return send_file(photo_path, mimetype='image/jpeg')
    return jsonify({'error': 'Photo not found'}), 404


@app.route('/api/suspects/add', methods=['POST'])
def add_suspect():
    """Add new suspect"""
    if not face_system:
        return jsonify({'success': False, 'error': 'System not initialized'}), 400
    
    try:
        name = request.form.get('name')
        photo = request.files.get('photo')
        
        if not name or not photo:
            return jsonify({
                'success': False,
                'error': 'Name and photo required'
            }), 400
        
        # Save photo temporarily
        temp_path = tempfile.mktemp(suffix='.jpg')
        photo.save(temp_path)
        
        # Add to face recognition system
        success = face_system.add_suspect(name, temp_path)
        
        if success:
            # Save photo permanently
            photo_path = os.path.join(SUSPECTS_DIR, f"{name}.jpg")
            photo.save(photo_path)
            
            return jsonify({
                'success': True,
                'message': f'Suspect {name} added successfully',
                'suspect': {
                    'id': name,
                    'name': name,
                    'photoUrl': f'/api/suspect-photo/{name}',
                    'addedDate': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No face detected in photo'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # Cleanup temp file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    """Process video frame for face detection"""
    if not face_system:
        return jsonify({'success': False, 'error': 'System not initialized'}), 400
    
    try:
        data = request.json
        
        # Decode base64 frame
        frame_data = data.get('frame', '')
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get coordinates
        latitude = data.get('latitude', 26.9124)
        longitude = data.get('longitude', 75.7873)
        
        # Process frame
        output_frame, detections, suspects = face_system.process_frame(
            frame,
            latitude=latitude,
            longitude=longitude,
            draw_boxes=True
        )
        
        # Encode processed frame
        _, buffer = cv2.imencode('.jpg', output_frame)
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        
        # Format detections
        formatted_suspects = []
        for suspect in suspects:
            formatted_suspects.append({
                'name': suspect['name'],
                'matchPercentage': suspect['match_percentage'],
                'box': suspect.get('box', []),
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'success': True,
            'processedFrame': f'data:image/jpeg;base64,{processed_frame}',
            'detections': len(detections),
            'suspects': formatted_suspects
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/suspects/<name>', methods=['DELETE'])
def delete_suspect(name):
    """Delete a suspect"""
    if not face_system:
        return jsonify({'success': False, 'error': 'System not initialized'}), 400
    
    try:
        # Remove from system
        if name in face_system.suspect_names:
            idx = face_system.suspect_names.index(name)
            face_system.suspect_names.pop(idx)
            face_system.suspect_encodings.pop(idx)
        
        # Remove photo
        photo_path = os.path.join(SUSPECTS_DIR, f"{name}.jpg")
        if os.path.exists(photo_path):
            os.remove(photo_path)
        
        return jsonify({
            'success': True,
            'message': f'Suspect {name} deleted'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("="*70)
    print("🚀 SkyGuard Backend API")
    print("="*70)
    print("📍 Server: http://localhost:5000")
    print("🔧 CORS Enabled for React frontend")
    print("="*70)
    
    # Initialize with default tolerance
    init_face_system()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
```

