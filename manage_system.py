"""
SkyGuard Unified Management System
- Admin Management (Alerts & Notifications)
- Gait Profile Management
"""

import os
import json
from datetime import datetime

# ===================== ADMIN MANAGEMENT =====================

from models.face_recognition_enhanced import AdminAlertSystem


def admin_management():
    print("\n" + "=" * 60)
    print("🛡️  SkyGuard Admin Management System")
    print("=" * 60)

    alert_system = AdminAlertSystem()

    while True:
        print("\n📋 Admin Menu:")
        print("1. View all admins")
        print("2. Add new admin")
        print("3. Remove admin")
        print("4. Configure email settings")
        print("5. Configure SMS settings")
        print("6. Test alert system")
        print("7. Back to main menu")

        choice = input("\nEnter choice (1-7): ")

        if choice == '1':
            admins = alert_system.config.get('admins', [])
            if admins:
                for i, admin in enumerate(admins, 1):
                    status = "🟢 Active" if admin.get('active', True) else "🔴 Inactive"
                    print(f"\n{i}. {admin['name']} ({admin.get('role', 'N/A')})")
                    print(f"   Email: {admin.get('email', 'N/A')}")
                    print(f"   Phone: {admin.get('phone', 'N/A')}")
                    print(f"   Status: {status}")
            else:
                print("No admins registered")

        elif choice == '2':
            name = input("Admin Name: ")
            email = input("Email: ")
            phone = input("Phone (+country code): ")
            role = input("Role: ")

            if name and (email or phone):
                alert_system.add_admin(name, email, phone, role)
                print("✅ Admin added successfully")
            else:
                print("❌ Name and contact required")

        elif choice == '3':
            admins = alert_system.config.get('admins', [])
            for i, admin in enumerate(admins, 1):
                print(f"{i}. {admin['name']}")

            try:
                idx = int(input("Select admin number to remove: ")) - 1
                removed = admins.pop(idx)
                alert_system.save_admin_config()
                print(f"✅ Removed admin: {removed['name']}")
            except:
                print("❌ Invalid selection")

        elif choice == '4':
            alert_system.config['email_enabled'] = True
            alert_system.config['smtp_server'] = input("SMTP Server: ") or "smtp.gmail.com"
            alert_system.config['smtp_port'] = int(input("SMTP Port: ") or "587")
            alert_system.config['sender_email'] = input("Sender Email: ")
            alert_system.config['sender_password'] = input("App Password: ")
            alert_system.save_admin_config()
            print("✅ Email configured")

        elif choice == '5':
            alert_system.config['sms_enabled'] = True
            alert_system.config['twilio_sid'] = input("Twilio SID: ")
            alert_system.config['twilio_token'] = input("Twilio Token: ")
            alert_system.config['twilio_from'] = input("Twilio Number: ")
            alert_system.save_admin_config()
            print("✅ SMS configured")

        elif choice == '6':
            alert_system.send_alert(
                suspect_name="Test Suspect",
                latitude=19.0760,
                longitude=72.8777,
                confidence=0.95
            )
            print("✅ Test alert sent")

        elif choice == '7':
            break

        else:
            print("❌ Invalid choice")


# ===================== GAIT PROFILE MANAGEMENT =====================

def load_gait_profiles():
    profiles_dir = 'gait_profiles'
    if not os.path.exists(profiles_dir):
        return {}

    profiles = {}
    for file in os.listdir(profiles_dir):
        if file.endswith('.json'):
            with open(os.path.join(profiles_dir, file)) as f:
                data = json.load(f)
                profiles[data['name']] = data
    return profiles


def gait_profile_management():
    print("\n" + "=" * 70)
    print("📁 Gait Profile Management System")
    print("=" * 70)

    while True:
        profiles = load_gait_profiles()

        print("\nOPTIONS:")
        print("1. List gait profiles")
        print("2. View profile details")
        print("3. Delete profile")
        print("4. Back to main menu")

        choice = input("\nSelect option (1-4): ")

        if choice == '1':
            if not profiles:
                print("No gait profiles found")
            for i, (name, data) in enumerate(profiles.items(), 1):
                created = datetime.fromisoformat(data['created_date'])
                print(f"{i}. {name} | Quality: {data.get('quality_score',0):.1%} | Created: {created.date()}")

        elif choice == '2':
            name = input("Enter profile name: ")
            data = profiles.get(name)
            if not data:
                print("❌ Profile not found")
                continue

            print("\nPROFILE DETAILS")
            print("-" * 60)
            print(f"Name: {data['name']}")
            print(f"Frames: {data.get('num_frames')}")
            print(f"Walks: {data.get('num_walks')}")
            print(f"Quality: {data.get('quality_score',0):.1%}")
            print("Signature Features:")
            for k, v in data.get('signature', {}).items():
                print(f"  {k}: {v:.4f}")

        elif choice == '3':
            name = input("Enter profile name to delete: ")
            file_path = os.path.join("gait_profiles", f"{name}.json")
            if os.path.exists(file_path):
                confirm = input("Type DELETE to confirm: ")
                if confirm == "DELETE":
                    os.remove(file_path)
                    print("✅ Profile deleted")
            else:
                print("❌ Profile not found")

        elif choice == '4':
            break

        else:
            print("❌ Invalid choice")


# ===================== MAIN MENU =====================

def main():
    while True:
        print("\n" + "=" * 70)
        print("🧠 SKYGUARD UNIFIED MANAGEMENT SYSTEM")
        print("=" * 70)
        print("1. Admin Management")
        print("2. Gait Profile Management")
        print("3. Exit")

        choice = input("\nSelect option (1-3): ")

        if choice == '1':
            admin_management()
        elif choice == '2':
            gait_profile_management()
        elif choice == '3':
            print("👋 Exiting system")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
