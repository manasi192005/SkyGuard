
"""
LSTM-based Stampede Prediction Module
Uses time-series forecasting to predict crowd density and stampede risks
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
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers
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


class StampedePredictor:
    """Predict stampede risks using LSTM time-series forecasting"""
    
    def __init__(self, sequence_length=30, prediction_horizon=30, model_path=None):
        """
        Initialize Stampede Predictor
        
        Args:
            sequence_length: Number of historical frames to consider
            prediction_horizon: Number of frames to predict ahead
            model_path: Path to pretrained model (optional)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = StampedeLSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1
        ).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
        # Historical data
        self.density_history = deque(maxlen=sequence_length)
        self.prediction_history = deque(maxlen=100)
        
        # Thresholds
        self.stampede_risk_threshold = 100
        self.rapid_increase_threshold = 30
    
    def preprocess_sequence(self, sequence):
        """Preprocess sequence for model input"""
        seq_array = np.array(sequence).reshape(-1, 1)
        
        # Normalize
        if seq_array.max() > 0:
            seq_normalized = seq_array / seq_array.max()
        else:
            seq_normalized = seq_array
        
        seq_tensor = torch.FloatTensor(seq_normalized).unsqueeze(0).to(self.device)
        return seq_tensor
    
    def predict_density(self, current_density):
        """
        Predict future crowd density
        
        Returns:
            predicted_density, risk_level, confidence
        """
        self.density_history.append(current_density)
        
        if len(self.density_history) < self.sequence_length:
            return None, 'insufficient_data', 0.0
        
        input_seq = list(self.density_history)
        input_tensor = self.preprocess_sequence(input_seq)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            predicted_value = prediction.item() * max(input_seq)
        
        predicted_density = max(0, int(predicted_value))
        
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'current': current_density,
            'predicted': predicted_density
        })
        
        risk_level, confidence = self.assess_stampede_risk(
            current_density, predicted_density
        )
        
        return predicted_density, risk_level, confidence
    
    def assess_stampede_risk(self, current_density, predicted_density):
        """Assess stampede risk based on predictions"""
        density_increase = predicted_density - current_density
        rate_of_change = density_increase / self.sequence_length if self.sequence_length > 0 else 0
        
        confidence = 0.0
        risk_level = 'low'
        
        # Critical: High density + rapid increase
        if predicted_density > self.stampede_risk_threshold and density_increase > self.rapid_increase_threshold:
            risk_level = 'critical'
            confidence = 0.95
        
        # High: High predicted density or rapid increase
        elif predicted_density > self.stampede_risk_threshold * 0.8:
            risk_level = 'high'
            confidence = 0.8
        
        elif density_increase > self.rapid_increase_threshold * 0.7:
            risk_level = 'high'
            confidence = 0.75
        
        # Medium: Moderate concerns
        elif predicted_density > self.stampede_risk_threshold * 0.6 or density_increase > 15:
            risk_level = 'medium'
            confidence = 0.6
        
        else:
            risk_level = 'low'
            confidence = 0.4
        
        return risk_level, confidence
    
    def generate_early_warning(self, current_density):
        """Generate early warning for stampede risk (90 seconds ahead)"""
        predicted_density, risk_level, confidence = self.predict_density(current_density)
        
        if predicted_density is None:
            return None
        
        warning_info = {
            'current_density': current_density,
            'predicted_density': predicted_density,
            'risk_level': risk_level,
            'confidence': confidence,
            'time_to_critical': self.estimate_time_to_critical(current_density, predicted_density),
            'recommended_actions': self.get_recommended_actions(risk_level)
        }
        
        return warning_info
    
    def estimate_time_to_critical(self, current, predicted):
        """Estimate time until critical density is reached"""
        if predicted <= current:
            return None
        
        rate = (predicted - current) / self.prediction_horizon
        
        if rate <= 0:
            return None
        
        remaining = self.stampede_risk_threshold - current
        if remaining <= 0:
            return 0
        
        frames_to_critical = remaining / rate
        seconds_to_critical = frames_to_critical * 0.033  # Assuming 30 FPS
        
        return max(0, int(seconds_to_critical))
    
    def get_recommended_actions(self, risk_level):
        """Get recommended actions based on risk level"""
        actions = {
            'low': [
                'Continue monitoring',
                'Maintain normal operations'
            ],
            'medium': [
                'Increase surveillance',
                'Alert ground staff',
                'Prepare crowd control measures'
            ],
            'high': [
                'Activate crowd control protocols',
                'Redirect crowd flow',
                'Position emergency responders'
            ],
            'critical': [
                'IMMEDIATE ACTION REQUIRED',
                'Halt entry to area',
                'Open additional exits',
                'Activate emergency evacuation'
            ]
        }
        
        return actions.get(risk_level, [])


# Test the module
if __name__ == '__main__':
    print("Stampede Prediction Module - Test")
    predictor = StampedePredictor()
    print("Module loaded successfully!")
    print(f"Device: {predictor.device}")
    print(f"Sequence length: {predictor.sequence_length} frames")
    print(f"Prediction horizon: {predictor.prediction_horizon} frames (~{predictor.prediction_horizon * 0.033:.1f} seconds)")
