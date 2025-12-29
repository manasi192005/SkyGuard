import os

# Disable GPU + TensorFlow noise
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp

print("✅ MediaPipe loaded successfully")
print("Available attributes:")
print(dir(mp))

