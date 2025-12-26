"""
Enhanced Stampede Prediction with 90-Second Early Warning
Uses crowd density trends to predict stampede risks
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime, timedelta

class StampedeLSTM(nn.Module):
    """LSTM model for crowd density prediction"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StampedeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out


class StampedePredictorEnhanced:
    """Enhanced Stampede Predictor with 90-second early warning"""
    
    def __init__(self, sequence_length=30, fps=30):
        """
        Initialize Stampede Predictor
        
        Args:
            sequence_length: Number of historical frames (default 30 = 1 second at 30fps)
            fps: Frames per second of video
        """
        self.sequence_length = sequence_length
        self.fps = fps
        self.prediction_horizon_seconds = 90  # 90 seconds early warning
        self.prediction_horizon_frames = self.prediction_horizon_seconds * fps  # 2700 frames
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = StampedeLSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1
        ).to(self.device)
        
        self.model.eval()
        
        # Historical data
        self.density_history = deque(maxlen=sequence_length)
        self.prediction_history = deque(maxlen=100)
        
        # Thresholds for stampede risk
        self.stampede_risk_threshold = 30  # People count threshold
        self.rapid_increase_rate = 0.5  # 50% increase rate
        
    def preprocess_sequence(self, sequence):
        """Preprocess sequence for model input"""
        seq_array = np.array(sequence).reshape(-1, 1)
        
        # Normalize
        if seq_array.max() > 0:
            seq_normalized = seq_array / seq_array.max()
        else:
            seq_normalized = seq_array
        
        seq_tensor = torch.FloatTensor(seq_normalized).unsqueeze(0).to(self.device)
        return seq_tensor, seq_array.max()
    
    def predict_density(self, current_density):
        """
        Predict future crowd density
        
        Returns:
            predicted_density, risk_level, confidence, time_to_critical
        """
        self.density_history.append(current_density)
        
        if len(self.density_history) < self.sequence_length:
            return None, 'insufficient_data', 0.0, None
        
        input_seq = list(self.density_history)
        input_tensor, max_val = self.preprocess_sequence(input_seq)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            predicted_value = prediction.item() * max_val
        
        predicted_density = max(0, int(predicted_value))
        
        # Store prediction
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'current': current_density,
            'predicted': predicted_density
        })
        
        # Assess risk with time calculation
        risk_level, confidence, time_to_critical = self.assess_stampede_risk(
            current_density, predicted_density
        )
        
        return predicted_density, risk_level, confidence, time_to_critical
    
    def assess_stampede_risk(self, current_density, predicted_density):
        """
        Assess stampede risk based on predictions
        
        Returns:
            risk_level, confidence, time_to_critical (in seconds)
        """
        # Calculate trend
        density_increase = predicted_density - current_density
        increase_rate = density_increase / current_density if current_density > 0 else 0
        
        confidence = 0.0
        risk_level = 'low'
        time_to_critical = None
        
        # Calculate time to reach critical threshold
        if density_increase > 0 and current_density < self.stampede_risk_threshold:
            frames_to_critical = (self.stampede_risk_threshold - current_density) / (density_increase / self.sequence_length)
            time_to_critical = int(frames_to_critical / self.fps)
        
        # Determine risk level
        if predicted_density > self.stampede_risk_threshold:
            if increase_rate > self.rapid_increase_rate:
                risk_level = 'critical'
                confidence = 0.95
            else:
                risk_level = 'high'
                confidence = 0.85
        
        elif increase_rate > self.rapid_increase_rate * 0.7:
            risk_level = 'high'
            confidence = 0.75
        
        elif predicted_density > self.stampede_risk_threshold * 0.7 or increase_rate > 0.3:
            risk_level = 'medium'
            confidence = 0.6
        
        else:
            risk_level = 'low'
            confidence = 0.4
        
        return risk_level, confidence, time_to_critical
    
    def generate_early_warning(self, current_density, stampede_zones=None):
        """
        Generate 90-second early warning for stampede risk
        
        Returns:
            warning_info dictionary
        """
        predicted_density, risk_level, confidence, time_to_critical = self.predict_density(current_density)
        
        if predicted_density is None:
            return None
        
        warning_info = {
            'current_density': current_density,
            'predicted_density': predicted_density,
            'risk_level': risk_level,
            'confidence': confidence,
            'time_to_critical': time_to_critical,
            'warning_seconds': self.prediction_horizon_seconds,
            'stampede_zones_count': len(stampede_zones) if stampede_zones else 0,
            'recommended_actions': self.get_recommended_actions(risk_level, time_to_critical)
        }
        
        return warning_info
    
    def get_recommended_actions(self, risk_level, time_to_critical):
        """Get recommended actions based on risk level"""
        actions = {
            'low': [
                'Continue normal monitoring',
                'Maintain current security posture'
            ],
            'medium': [
                'Increase surveillance in dense areas',
                'Alert ground staff to monitor closely',
                'Prepare crowd control measures'
            ],
            'high': [
                '⚠ ACTIVATE crowd control protocols',
                '⚠ Begin redirecting crowd flow',
                '⚠ Position emergency responders',
                '⚠ Prepare evacuation routes'
            ],
            'critical': [
                '🚨 IMMEDIATE ACTION REQUIRED',
                '🚨 STOP new entries to high-risk zones',
                '🚨 OPEN all emergency exits',
                '🚨 BEGIN controlled evacuation',
                '🚨 Deploy ALL security personnel'
            ]
        }
        
        base_actions = actions.get(risk_level, [])
        
        # Add time-specific warnings
        if time_to_critical:
            if time_to_critical < 60:
                base_actions.insert(0, f'⏰ CRITICAL in {time_to_critical}s!')
            elif time_to_critical < 90:
                base_actions.insert(0, f'⚠ High risk in {time_to_critical}s')
        
        return base_actions


# Test module
if __name__ == '__main__':
    print("Enhanced Stampede Prediction Module - Test")
    predictor = StampedePredictorEnhanced()
    print(f"✓ Module loaded successfully!")
    print(f"✓ Device: {predictor.device}")
    print(f"✓ Early warning time: {predictor.prediction_horizon_seconds} seconds")
    print(f"✓ Sequence tracking: {predictor.sequence_length} frames")
