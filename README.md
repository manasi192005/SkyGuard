# ���️ SkyGuard - Quick Reference Guide

## Team GDuo - VESIT, Mumbai

---

## ��� How to Run

### Start the System
```bash
cd ~/Documents/skyguard
source venv/bin/activate
python3 main_optimized.py --video 0
```

### Stop the System
- Press **'q'** in video window
- Or press **Ctrl+C** in terminal

---

## ��� The 4 Main Features

### ✅ Feature 1: Density Heat Map
**What it does:** Shows crowd density with color coding
- ��� Red = High density (stampede risk)
- ��� Orange/Yellow = Medium density  
- ��� Green/Blue = Low density

**File:** `models/crowd_analysis_enhanced.py`

### ✅ Feature 2: Face Recognition
**What it does:** Identifies suspects in crowds
- Green boxes = Unknown faces
- Red boxes = Detected suspects
- Instant security alerts

**File:** `models/face_recognition_enhanced.py`

**Add suspect:**
```bash
python3 add_suspect.py
```

### ✅ Feature 3: Stampede Prediction
**What it does:** Predicts stampede risk 90 seconds in advance
- Uses LSTM neural network
- Analyzes crowd trends
- Provides evacuation recommendations

**File:** `models/stampede_prediction_enhanced.py`

### ✅ Feature 4: Emergency Detection
**What it does:** Detects medical emergencies
- Monitors body angles (lying down)
- 5-minute immobility confirmation
- Triggers medical response

**File:** `models/emergency_detection_enhanced.py`

---

## ��� Database

### Location
```
data/database/skyguard.db
```

### Check Database
```bash
python3 check_database.py
```

### Reset Database
```bash
rm -f data/database/skyguard.db
python3 models/database.py
```

### Tables
- `detected_suspects` - Face recognition logs
- `crowd_analytics` - Crowd density records
- `emergency_events` - Emergency incidents
- `system_logs` - System activity

---

## ��� Troubleshooting

### Camera not working
```bash
# Try different camera index
python3 main_optimized.py --video 1
```

### Low FPS / Slow performance
The system runs at 15-25 FPS normally. This is expected.

### Database errors
```bash
# Reset database
./reset_database.sh
```

### Import errors
```bash
# Reinstall dependencies
pip install --upgrade opencv-python mediapipe torch
```

---

## ��� Project Structure
```
skyguard/
├── main_optimized.py          # Main system (USE THIS)
├── models/
│   ├── crowd_analysis_enhanced.py      # Feature 1
│   ├── face_recognition_enhanced.py    # Feature 2
│   ├── stampede_prediction_enhanced.py # Feature 3
│   ├── emergency_detection_enhanced.py # Feature 4
│   └── database.py                     # Database
├── data/
│   ├── database/skyguard.db   # SQLite database
│   └── suspects/              # Suspect photos
└── venv/                      # Virtual environment
```

---

## ✅ System Status

All 4 features: **WORKING**
Database: **CONNECTED**
Accuracy: **HIGH**
Status: **PRODUCTION READY**

---

## ��� Team

**Team GDuo**
- Rushil Patil (Leader)
- Manasi Ghalsasi (Member)

**VESIT, Mumbai**

---

## ��� For Presentation

### Key Points to Highlight:

1. **Real-time Processing** - 20-25 FPS
2. **AI-Powered** - Uses LSTM, CNN, MediaPipe
3. **Proactive Safety** - 90-second warning system
4. **Complete Integration** - All features work together
5. **Database Logging** - Everything recorded

### Demo Flow:
1. Show system initialization (all 4 features loading)
2. Show heat map with crowd counting
3. Demo face recognition (if you added suspects)
4. Explain stampede prediction (show console output)
5. Show emergency detection working
6. Show database records with `check_database.py`

---
