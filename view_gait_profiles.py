"""
View all saved gait profiles
"""

import os
import json

print("\n" + "=" * 70)
print("📂 Saved Gait Profiles")
print("=" * 70)

gait_db_path = "gait_profiles"

if not os.path.exists(gait_db_path):
    print("\n⚠️  No profiles directory found")
    exit()

profiles = [f for f in os.listdir(gait_db_path) if f.endswith('.json')]

if not profiles:
    print("\n⚠️  No profiles saved yet")
    print("   Capture a profile: python3 capture_gait_profile.py")
    exit()

print(f"\n✅ Found {len(profiles)} profile(s):\n")

for profile_file in profiles:
    filepath = os.path.join(gait_db_path, profile_file)
    
    try:
        with open(filepath, 'r') as f:
            profile = json.load(f)
        
        print(f"📁 {profile['name']}")
        print(f"   Description: {profile.get('description', 'N/A')}")
        print(f"   Created: {profile.get('created_date', 'Unknown')}")
        print(f"   Frames: {profile.get('num_frames', 0)}")
        print(f"   Walks: {profile.get('num_walks', 0)}")
        print(f"   Quality: {profile.get('quality_score', 0)}")
        print(f"   File: {filepath}")
        
        # Show signature features
        sig = profile.get('signature', {})
        if sig:
            print(f"   Features: {', '.join(list(sig.keys())[:3])}...")
        
        print()
    
    except Exception as e:
        print(f"❌ Error reading {profile_file}: {e}\n")

print("=" * 70)
