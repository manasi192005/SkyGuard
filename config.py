"""
SkyGuard Unified Configuration
Central configuration for all AI, biometric, surveillance, and alert modules
Government-ready & production-friendly
"""

# ==========================================================
# GAIT RECOGNITION CONFIGURATION
# ==========================================================

GAIT_CAMERA_CONFIG = {
    'optimal_height_min': 6,
    'optimal_height_max': 8,
    'optimal_angle_min': 30,
    'optimal_angle_max': 45,
    'optimal_distance_min': 10,
    'optimal_distance_max': 15,
    'min_brightness': 80,
    'max_shadow_ratio': 0.3,
    'frame_width': 1280,
    'frame_height': 720,
    'fps': 30,
}

GAIT_CAPTURE_CONFIG = {
    'min_frames': 150,
    'min_walks': 3,
    'min_side_view_ratio': 0.6,
    'min_quality_score': 0.5,
    'profile_expiry_months': 6,
}

GAIT_RECOGNITION_CONFIG = {
    'min_frames': 90,
    'confidence_threshold': 0.80,
    'alert_cooldown': 10,
    'max_tracking_distance': 100,
}

GAIT_FEATURE_CONFIG = {
    'model_complexity': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'features': [
        'stride_length',
        'step_width',
        'hip_sway',
        'knee_angles',
        'arm_swing',
        'body_tilt',
        'step_height',
        'gait_symmetry',
    ],
    'feature_weights': {
        'stride_length': 1.5,
        'step_width': 1.2,
        'hip_sway': 1.0,
        'knee_angles': 1.3,
        'arm_swing': 1.0,
        'body_tilt': 0.8,
        'gait_symmetry': 1.4,
    }
}

GAIT_VISUALIZATION_CONFIG = {
    'color_optimal': (0, 255, 0),
    'color_warning': (0, 165, 255),
    'color_error': (0, 0, 255),
    'color_text': (255, 255, 255),
    'color_alert': (0, 0, 200),
    'show_pose_landmarks': True,
    'show_quality_metrics': True,
    'show_confidence_scores': True,
    'show_fps': True,
}

GAIT_STORAGE_CONFIG = {
    'profiles_dir': 'gait_profiles',
    'backup_dir': 'gait_profiles_backup',
    'auto_backup_days': 7,
}

GAIT_QUALITY_CONFIG = {
    'check_lighting': True,
    'check_shadows': True,
    'check_view_angle': True,
    'check_walking_speed': True,
    'auto_reject_low_quality': False,
    'quality_warning_threshold': 0.6,
}

# ==========================================================
# CROWD MONITORING CONFIGURATION
# ==========================================================

CROWD_CONFIG = {
    'detection_scale': 0.5,
    'window_stride': (4, 4),
    'padding': (8, 8),
    'scale_factor': 1.05,
    'gaussian_sigma': 30,
    'risk_thresholds': {
        'low': 5,
        'medium': 15,
        'high': 30,
        'critical': 50
    }
}

# ==========================================================
# FACE RECOGNITION CONFIGURATION
# ==========================================================

FACE_CONFIG = {
    'confidence_threshold': 0.85,
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_face_size': (30, 30),
    'alert_cooldown': 5
}

# ==========================================================
# STAMPEDE PREDICTION CONFIGURATION
# ==========================================================

STAMPEDE_CONFIG = {
    'sequence_length': 30,
    'prediction_horizon': 90,
    'stampede_threshold': 30,
    'rapid_increase_rate': 0.5,
    'fps': 30
}

# ==========================================================
# EMERGENCY / FALL DETECTION CONFIGURATION
# ==========================================================

EMERGENCY_CONFIG = {
    'fall_angle_threshold': 60,
    'immobility_seconds': 300,
    'position_tolerance': 20,
    'fps': 30,
    'use_mediapipe': True
}

# ==========================================================
# ALERT & LOGGING CONFIGURATION
# ==========================================================

ALERT_CONFIG = {
    'enable_audio': True,
    'enable_visual': True,
    'log_detections': True,
    'log_file': 'detection_log.txt',
}

# ==========================================================
# DATABASE CONFIGURATION
# ==========================================================

DATABASE_CONFIG = {
    'path': 'data/database/skyguard.db',
    'log_interval': 30,
    'batch_size': 10
}

# ==========================================================
# SYSTEM PERFORMANCE CONFIGURATION
# ==========================================================

SYSTEM_PERFORMANCE_CONFIG = {
    'max_fps': 30,
    'display_width': 1280,
    'display_height': 720,
    'enable_gpu': True,
    'buffer_size': 5,
    'frame_skip': 1,
    'max_history_length': 300,
}

# ==========================================================
# DEBUG CONFIGURATION
# ==========================================================

DEBUG_CONFIG = {
    'enabled': False,
    'save_frames': False,
    'debug_dir': 'debug_output',
    'verbose': False,
}

# ==========================================================
# GLOBAL CONFIG ACCESS
# ==========================================================

def get_config():
    """Return complete SkyGuard configuration"""
    return {
        'gait': {
            'camera': GAIT_CAMERA_CONFIG,
            'capture': GAIT_CAPTURE_CONFIG,
            'recognition': GAIT_RECOGNITION_CONFIG,
            'features': GAIT_FEATURE_CONFIG,
            'visualization': GAIT_VISUALIZATION_CONFIG,
            'storage': GAIT_STORAGE_CONFIG,
            'quality': GAIT_QUALITY_CONFIG,
        },
        'crowd': CROWD_CONFIG,
        'face': FACE_CONFIG,
        'stampede': STAMPEDE_CONFIG,
        'emergency': EMERGENCY_CONFIG,
        'alerts': ALERT_CONFIG,
        'database': DATABASE_CONFIG,
        'performance': SYSTEM_PERFORMANCE_CONFIG,
        'debug': DEBUG_CONFIG,
    }


def print_config():
    """Print full configuration (for admin/debug use)"""
    config = get_config()
    print("\n" + "=" * 80)
    print("⚙️  SKYGUARD SYSTEM CONFIGURATION")
    print("=" * 80)

    for section, settings in config.items():
        print(f"\n[{section.upper()}]")
        for k, v in settings.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    print_config()
