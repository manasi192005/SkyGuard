"""
Optimized Configuration for High Accuracy
"""

# Feature 1: Crowd Analysis Optimization
CROWD_CONFIG = {
    'detection_scale': 0.5,  # Balance between speed and accuracy
    'window_stride': (4, 4),  # Smaller stride = more accurate
    'padding': (8, 8),
    'scale_factor': 1.05,
    'gaussian_sigma': 30,  # Spread of density influence
    'risk_thresholds': {
        'low': 5,
        'medium': 15,
        'high': 30,
        'critical': 50
    }
}

# Feature 2: Face Recognition Optimization
FACE_CONFIG = {
    'confidence_threshold': 0.85,  # Higher = fewer false positives
    'scale_factor': 1.1,
    'min_neighbors': 5,  # Higher = more accurate but slower
    'min_face_size': (30, 30),
    'alert_cooldown': 5  # Seconds between repeated alerts
}

# Feature 3: Stampede Prediction Optimization
STAMPEDE_CONFIG = {
    'sequence_length': 30,  # Frames to analyze
    'prediction_horizon': 90,  # Seconds warning
    'stampede_threshold': 30,  # People count threshold
    'rapid_increase_rate': 0.5,  # 50% increase triggers warning
    'fps': 30
}

# Feature 4: Emergency Detection Optimization
EMERGENCY_CONFIG = {
    'fall_angle_threshold': 60,  # Degrees (lower = more horizontal)
    'immobility_seconds': 300,  # 5 minutes
    'position_tolerance': 20,  # Pixels movement threshold
    'fps': 30,
    'use_mediapipe': True  # Use MediaPipe if available
}

# Database Configuration
DATABASE_CONFIG = {
    'path': 'data/database/skyguard.db',
    'log_interval': 30,  # Log every 30 frames
    'batch_size': 10  # Batch database writes
}

# System Performance
PERFORMANCE_CONFIG = {
    'max_fps': 30,
    'display_width': 1280,
    'display_height': 720,
    'enable_gpu': True,  # Use GPU if available
    'buffer_size': 5
}
