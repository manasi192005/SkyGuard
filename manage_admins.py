"""
Admin Management System
Add/remove admins who receive suspect alerts
"""

from models.face_recognition_enhanced import AdminAlertSystem
import json

def main():
    print("\n" + "="*60)
    print("🛡️  SkyGuard Admin Management System")
    print("="*60)
    
    alert_system = AdminAlertSystem()
    
    while True:
        print("\n📋 Menu:")
        print("1. View all admins")
        print("2. Add new admin")
        print("3. Remove admin")
        print("4. Configure email settings")
        print("5. Configure SMS settings")
        print("6. Test alert system")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ")
        
        if choice == '1':
            # View admins
            print("\n👥 Registered Admins:")
            print("-" * 60)
            
            admins = alert_system.config.get('admins', [])
            if admins:
                for i, admin in enumerate(admins, 1):
                    status = "🟢 Active" if admin.get('active', True) else "🔴 Inactive"
                    print(f"\n{i}. {admin['name']} ({admin.get('role', 'N/A')})")
                    print(f"   Email: {admin.get('email', 'Not provided')}")
                    print(f"   Phone: {admin.get('phone', 'Not provided')}")
                    print(f"   Status: {status}")
                    print(f"   Added: {admin.get('added_date', 'Unknown')}")
            else:
                print("No admins registered")
        
        elif choice == '2':
            # Add admin
            print("\n➕ Add New Admin")
            print("-" * 60)
            
            name = input("Admin Name: ")
            email = input("Email Address: ")
            phone = input("Phone Number (with country code, e.g., +919876543210): ")
            role = input("Role (e.g., Security, Team Leader, Manager): ")
            
            if name and (email or phone):
                alert_system.add_admin(name, email, phone, role)
                print(f"\n✅ Admin '{name}' added successfully!")
            else:
                print("\n❌ Name and at least one contact method required")
        
        elif choice == '3':
            # Remove admin
            print("\n➖ Remove Admin")
            print("-" * 60)
            
            admins = alert_system.config.get('admins', [])
            if admins:
                for i, admin in enumerate(admins, 1):
                    print(f"{i}. {admin['name']} ({admin.get('email', 'no email')})")
                
                try:
                    idx = int(input("\nEnter admin number to remove: ")) - 1
                    if 0 <= idx < len(admins):
                        removed = admins.pop(idx)
                        alert_system.save_admin_config()
                        print(f"\n✅ Removed admin: {removed['name']}")
                    else:
                        print("\n❌ Invalid selection")
                except:
                    print("\n❌ Invalid input")
            else:
                print("No admins to remove")
        
        elif choice == '4':
            # Configure email
            print("\n📧 Email Configuration")
            print("-" * 60)
            print("Configure SMTP settings for email alerts")
            print("(For Gmail, enable 'App Passwords' in Google Account settings)\n")
            
            alert_system.config['email_enabled'] = True
            alert_system.config['smtp_server'] = input("SMTP Server (default: smtp.gmail.com): ") or "smtp.gmail.com"
            alert_system.config['smtp_port'] = int(input("SMTP Port (default: 587): ") or "587")
            alert_system.config['sender_email'] = input("Sender Email: ")
            alert_system.config['sender_password'] = input("Sender Password/App Password: ")
            
            alert_system.save_admin_config()
            print("\n✅ Email settings configured")
        
        elif choice == '5':
            # Configure SMS
            print("\n📱 SMS Configuration (Twilio)")
            print("-" * 60)
            print("Get your Twilio credentials from: https://www.twilio.com\n")
            
            alert_system.config['sms_enabled'] = True
            alert_system.config['twilio_sid'] = input("Twilio Account SID: ")
            alert_system.config['twilio_token'] = input("Twilio Auth Token: ")
            alert_system.config['twilio_from'] = input("Twilio Phone Number: ")
            
            alert_system.save_admin_config()
            print("\n✅ SMS settings configured")
        
        elif choice == '6':
            # Test alert
            print("\n🧪 Test Alert System")
            print("-" * 60)
            
            alert_system.send_alert(
                suspect_name="Test Suspect",
                latitude=19.0760,
                longitude=72.8777,
                confidence=0.95
            )
            print("\n✅ Test alert sent!")
        
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice")

if __name__ == '__main__':
    main()
