// App.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, Radio, Users, Settings, Play, Square, AlertTriangle, Target, Activity } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';
interface Suspect {
  id: string;
  name: string;
  photoUrl: string;
  addedDate: string;
}

interface Detection {
  id: string;
  name: string;
  matchPercentage: number;
  time: string;
  frame: number;
  coordinates?: { lat: number; lng: number };
}

interface Alert {
  id: string;
  name: string;
  matchPercentage: number;
  time: string;
  frame: number;
}

const App: React.FC = () => {
  // State Management
  const [isSystemActive, setIsSystemActive] = useState(false);
  const [sourceType, setSourceType] = useState<'webcam' | 'upload' | 'stream'>('webcam');
  const [tolerance, setTolerance] = useState(50);
  const [processInterval, setProcessInterval] = useState(3);
  const [suspects, setSuspects] = useState<Suspect[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [totalDetections, setTotalDetections] = useState(0);
  const [highestMatch, setHighestMatch] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [showAddSuspect, setShowAddSuspect] = useState(false);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Add Suspect Handler
  const handleAddSuspect = async (name: string, photo: File) => {
  try {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('photo', photo);

    const response = await axios.post(`${API_BASE_URL}/suspects/add`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });

    if (response.data.success) {
      const newSuspect: Suspect = {
        id: response.data.suspect.id,
        name: response.data.suspect.name,
        photoUrl: `${API_BASE_URL.replace('/api', '')}${response.data.suspect.photoUrl}`,
        addedDate: response.data.suspect.addedDate
      };
      setSuspects([...suspects, newSuspect]);
      setShowAddSuspect(false);
    }
  } catch (error) {
    console.error('Error adding suspect:', error);
    alert('Failed to add suspect');
  }
};

  // Get Alert Color
  const getAlertColor = (percentage: number): string => {
    if (percentage >= 80) return 'from-red-500 to-red-700';
    if (percentage >= 60) return 'from-orange-500 to-orange-700';
    return 'from-yellow-500 to-yellow-700';
  };

  const getAlertBorder = (percentage: number): string => {
    if (percentage >= 80) return 'border-red-500';
    if (percentage >= 60) return 'border-orange-500';
    return 'border-yellow-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-lg border-b border-white/10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Target className="w-7 h-7" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  SkyGuard
                </h1>
                <p className="text-xs text-gray-400">Face Recognition System</p>
              </div>
            </div>

            {/* System Status */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10">
                <div className={`w-2 h-2 rounded-full ${isSystemActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
                <span className="text-sm">{isSystemActive ? 'Active' : 'Standby'}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Sidebar */}
          <aside className="lg:col-span-1 space-y-6">
            
            {/* Stats Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-1 gap-4">
              <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 backdrop-blur-lg rounded-xl p-4 border border-blue-500/30">
                <div className="flex items-center justify-between mb-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  <span className="text-2xl font-bold">{totalDetections}</span>
                </div>
                <p className="text-xs text-gray-300">Total Detections</p>
              </div>

              <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
                <div className="flex items-center justify-between mb-2">
                  <AlertTriangle className="w-5 h-5 text-purple-400" />
                  <span className="text-2xl font-bold">{alerts.length}</span>
                </div>
                <p className="text-xs text-gray-300">Active Alerts</p>
              </div>

              <div className="bg-gradient-to-br from-pink-500/20 to-pink-600/20 backdrop-blur-lg rounded-xl p-4 border border-pink-500/30">
                <div className="flex items-center justify-between mb-2">
                  <Target className="w-5 h-5 text-pink-400" />
                  <span className="text-2xl font-bold">{highestMatch.toFixed(1)}%</span>
                </div>
                <p className="text-xs text-gray-300">Best Match</p>
              </div>

              <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 backdrop-blur-lg rounded-xl p-4 border border-green-500/30">
                <div className="flex items-center justify-between mb-2">
                  <Users className="w-5 h-5 text-green-400" />
                  <span className="text-2xl font-bold">{suspects.length}</span>
                </div>
                <p className="text-xs text-gray-300">Suspects</p>
              </div>
            </div>

            {/* Settings Panel */}
            <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 p-5">
              <div className="flex items-center gap-2 mb-4">
                <Settings className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold">Detection Settings</h3>
              </div>

              <div className="space-y-4">
                {/* Tolerance Slider */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">Match Sensitivity</label>
                    <span className="text-sm font-semibold text-purple-400">{tolerance}%</span>
                  </div>
                  <input
                    type="range"
                    min="30"
                    max="70"
                    step="5"
                    value={tolerance}
                    onChange={(e) => setTolerance(Number(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <p className="text-xs text-gray-400 mt-1">Lower = Stricter matching</p>
                </div>

                {/* Process Interval */}
                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm text-gray-300">Process Interval</label>
                    <span className="text-sm font-semibold text-purple-400">1/{processInterval}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={processInterval}
                    onChange={(e) => setProcessInterval(Number(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <p className="text-xs text-gray-400 mt-1">Process every N frames</p>
                </div>
              </div>
            </div>

            {/* Video Source */}
            <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 p-5">
              <div className="flex items-center gap-2 mb-4">
                <Camera className="w-5 h-5 text-blue-400" />
                <h3 className="font-semibold">Video Source</h3>
              </div>

              <div className="space-y-2">
                <button
                  onClick={() => setSourceType('webcam')}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                    sourceType === 'webcam'
                      ? 'bg-purple-500/30 border-2 border-purple-500'
                      : 'bg-white/5 border-2 border-transparent hover:bg-white/10'
                  }`}
                >
                  <Camera className="w-5 h-5" />
                  <span className="text-sm font-medium">Webcam</span>
                </button>

                <button
                  onClick={() => setSourceType('upload')}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                    sourceType === 'upload'
                      ? 'bg-purple-500/30 border-2 border-purple-500'
                      : 'bg-white/5 border-2 border-transparent hover:bg-white/10'
                  }`}
                >
                  <Upload className="w-5 h-5" />
                  <span className="text-sm font-medium">Upload Video</span>
                </button>

                <button
                  onClick={() => setSourceType('stream')}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                    sourceType === 'stream'
                      ? 'bg-purple-500/30 border-2 border-purple-500'
                      : 'bg-white/5 border-2 border-transparent hover:bg-white/10'
                  }`}
                >
                  <Radio className="w-5 h-5" />
                  <span className="text-sm font-medium">Drone Stream</span>
                </button>
              </div>

              {sourceType === 'upload' && (
                <div className="mt-3">
                  <input
                    type="file"
                    accept="video/*"
                    className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-500 file:text-white hover:file:bg-purple-600 file:cursor-pointer"
                  />
                </div>
              )}

              {sourceType === 'stream' && (
                <input
                  type="text"
                  placeholder="rtsp://stream-url"
                  className="w-full mt-3 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                />
              )}
            </div>

            {/* Control Buttons */}
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setIsSystemActive(!isSystemActive)}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold transition-all ${
                  isSystemActive
                    ? 'bg-red-500 hover:bg-red-600'
                    : 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600'
                }`}
              >
                {isSystemActive ? (
                  <>
                    <Square className="w-5 h-5" />
                    <span>Stop</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Start</span>
                  </>
                )}
              </button>

              <button
                onClick={() => setShowAddSuspect(true)}
                className="flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold bg-purple-500 hover:bg-purple-600 transition-all"
              >
                <Users className="w-5 h-5" />
                <span>Add</span>
              </button>
            </div>

            {/* Suspects List */}
            <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 p-5 max-h-64 overflow-y-auto">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Users className="w-5 h-5 text-green-400" />
                Suspect Database
              </h3>
              
              {suspects.length === 0 ? (
                <p className="text-sm text-gray-400 text-center py-4">No suspects added yet</p>
              ) : (
                <div className="space-y-2">
                  {suspects.map((suspect) => (
                    <div
                      key={suspect.id}
                      className="flex items-center gap-3 p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-all"
                    >
                      <img
                        src={suspect.photoUrl}
                        alt={suspect.name}
                        className="w-10 h-10 rounded-full object-cover border-2 border-purple-500"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{suspect.name}</p>
                        <p className="text-xs text-gray-400">
                          {new Date(suspect.addedDate).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>

          {/* Main Content */}
          <main className="lg:col-span-3 space-y-6">
            
            {/* Video Feed */}
            <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 overflow-hidden">
              <div className="bg-black/30 px-5 py-3 border-b border-white/10 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Camera className="w-5 h-5 text-blue-400" />
                  <h3 className="font-semibold">Live Video Feed</h3>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-400">
                  <span>Frame: {currentFrame}</span>
                  <span>•</span>
                  <span>{sourceType.toUpperCase()}</span>
                </div>
              </div>

              <div className="relative bg-black aspect-video">
                {isSystemActive ? (
                  <>
                    <video
                      ref={videoRef}
                      className="w-full h-full object-contain"
                      autoPlay
                      muted
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute top-0 left-0 w-full h-full"
                    />
                    
                    {/* Processing Indicator */}
                    <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-green-500/90 rounded-lg">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                      <span className="text-sm font-semibold">Processing</span>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <Camera className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                      <p className="text-gray-400">Press START to begin detection</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Detection Log */}
            <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10">
              <div className="bg-black/30 px-5 py-3 border-b border-white/10 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                  <h3 className="font-semibold">Detection Log</h3>
                </div>
                {alerts.length > 0 && (
                  <button
                    onClick={() => setAlerts([])}
                    className="text-sm text-gray-400 hover:text-white transition-colors"
                  >
                    Clear All
                  </button>
                )}
              </div>

              <div className="p-5 max-h-96 overflow-y-auto space-y-3">
                {alerts.length === 0 ? (
                  <div className="text-center py-8">
                    <Target className="w-12 h-12 mx-auto mb-3 text-gray-600" />
                    <p className="text-gray-400">No detections yet</p>
                    <p className="text-sm text-gray-500 mt-1">Alerts will appear here</p>
                  </div>
                ) : (
                  alerts.slice(0, 10).map((alert) => (
                    <div
                      key={alert.id}
                      className={`p-4 rounded-lg bg-gradient-to-r ${getAlertColor(alert.matchPercentage)} border-l-4 ${getAlertBorder(alert.matchPercentage)} animate-slideIn`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <Target className="w-5 h-5" />
                            <h4 className="font-bold text-lg">{alert.name}</h4>
                          </div>
                          <div className="flex items-center gap-3 text-sm">
                            <span className="font-semibold">{alert.matchPercentage.toFixed(1)}% Match</span>
                            <span>•</span>
                            <span>{alert.time}</span>
                            <span>•</span>
                            <span>Frame {alert.frame}</span>
                          </div>
                        </div>
                        <div className="text-3xl font-bold opacity-50">
                          {alert.matchPercentage >= 80 ? '🚨' : alert.matchPercentage >= 60 ? '⚠️' : '⚡'}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </main>
        </div>
      </div>

      {/* Add Suspect Modal */}
      {showAddSuspect && (
        <AddSuspectModal
          onClose={() => setShowAddSuspect(false)}
          onAdd={handleAddSuspect}
        />
      )}
    </div>
  );
};

// Add Suspect Modal Component
interface AddSuspectModalProps {
  onClose: () => void;
  onAdd: (name: string, photo: File) => void;
}

const AddSuspectModal: React.FC<AddSuspectModalProps> = ({ onClose, onAdd }) => {
  const [name, setName] = useState('');
  const [photo, setPhoto] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');

  const handlePhotoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPhoto(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = () => {
    if (name && photo) {
      onAdd(name, photo);
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 max-w-md w-full p-6 animate-scaleIn">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Add New Suspect</h2>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          {/* Photo Upload */}
          <div>
            <label className="block text-sm font-medium mb-2">Photo</label>
            <div className="border-2 border-dashed border-white/20 rounded-lg p-6 text-center hover:border-purple-500 transition-colors cursor-pointer">
              {preview ? (
                <img src={preview} alt="Preview" className="w-32 h-32 rounded-lg object-cover mx-auto" />
              ) : (
                <div>
                  <Upload className="w-12 h-12 mx-auto mb-2 text-gray-400" />
                  <p className="text-sm text-gray-400">Click to upload photo</p>
                </div>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handlePhotoChange}
                className="hidden"
                id="photo-upload"
              />
              <label htmlFor="photo-upload" className="cursor-pointer">
                {preview && (
                  <button className="mt-3 text-sm text-purple-400 hover:text-purple-300">
                    Change Photo
                  </button>
                )}
              </label>
            </div>
          </div>

          {/* Name Input */}
          <div>
            <label className="block text-sm font-medium mb-2">Suspect Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter name..."
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500 transition-colors"
            />
          </div>

          {/* Tips */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
            <p className="text-sm text-blue-300">
              <strong>Tips:</strong> Use a clear, front-facing photo with good lighting for best results.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 mt-6">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-3 rounded-lg bg-white/5 hover:bg-white/10 font-semibold transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={!name || !photo}
              className="flex-1 px-4 py-3 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Add Suspect
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;