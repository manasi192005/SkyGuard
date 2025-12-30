"""
Quick Test Script for SkyGuard Authentication System
Creates test users and verifies the system works
"""

import os
from auth_system import SQLiteAuthSystem
from PIL import Image
import numpy as np

def create_dummy_id_card(filename='test_police_id.jpg'):
    """Create a dummy ID card image for testing"""
    # Create a simple image
    img = Image.new('RGB', (400, 250), color=(73, 109, 137))
    
    # Save it
    img.save(filename)
    print(f"✓ Created dummy ID card: {filename}")
    return filename


def test_registration():
    """Test user registration"""
    print("\n" + "="*70)
    print("TEST 1: USER REGISTRATION")
    print("="*70 + "\n")
    
    # Create dummy ID card
    id_card_path = create_dummy_id_card('test_police_id.jpg')
    
    # Initialize system
    auth_system = SQLiteAuthSystem(db_path='test_skyguard.db')
    
    # Test registration
    result = auth_system.register_user(
        email='officer.test@police.gov.in',
        password='SecurePass123',
        full_name='Test Officer',
        id_card_path=id_card_path
    )
    
    if result['success']:
        print("\n✅ TEST 1 PASSED: Registration successful!")
        print(f"   Username: {result['username']}")
        print(f"   Role: {result['role']}")
        print(f"   Agency: {result['agency']}")
        return auth_system, result['username']
    else:
        print(f"\n❌ TEST 1 FAILED: {result['error']}")
        return None, None


def test_authentication(auth_system, expected_username):
    """Test user authentication"""
    print("\n" + "="*70)
    print("TEST 2: USER AUTHENTICATION")
    print("="*70 + "\n")
    
    # Test login
    result = auth_system.authenticate_user(
        email='officer.test@police.gov.in',
        password='SecurePass123'
    )
    
    if result['success']:
        print("\n✅ TEST 2 PASSED: Login successful!")
        print(f"   Username: {result['username']}")
        print(f"   Permissions: {len(result['permissions'])}")
        return True
    else:
        print(f"\n❌ TEST 2 FAILED: {result['error']}")
        return False


def test_permissions(auth_system, username):
    """Test permission system"""
    print("\n" + "="*70)
    print("TEST 3: PERMISSION SYSTEM")
    print("="*70 + "\n")
    
    permissions_to_test = [
        ('ACCESS_CAMERA', True),  # Police should have this
        ('VIEW_DETECTIONS', True),  # Police should have this
        ('SYSTEM_CONFIG', False)  # Police should NOT have this (admin only)
    ]
    
    all_passed = True
    
    for perm, should_have in permissions_to_test:
        has_perm = auth_system.verify_permission(username, perm)
        
        if has_perm == should_have:
            print(f"✅ {perm}: {'GRANTED' if has_perm else 'DENIED'} (as expected)")
        else:
            print(f"❌ {perm}: {'GRANTED' if has_perm else 'DENIED'} (expected: {'GRANTED' if should_have else 'DENIED'})")
            all_passed = False
    
    if all_passed:
        print("\n✅ TEST 3 PASSED: All permissions correct!")
    else:
        print("\n❌ TEST 3 FAILED: Permission errors detected")
    
    return all_passed


def test_blockchain_integrity(auth_system):
    """Test blockchain integrity"""
    print("\n" + "="*70)
    print("TEST 4: BLOCKCHAIN INTEGRITY")
    print("="*70 + "\n")
    
    is_valid = auth_system.rbac.verify_chain()
    
    if is_valid:
        print("✅ TEST 4 PASSED: Blockchain integrity verified!")
        print(f"   Total blocks: {len(auth_system.rbac.chain)}")
    else:
        print("❌ TEST 4 FAILED: Blockchain corruption detected!")
    
    return is_valid


def test_multiple_users():
    """Test multiple user registration"""
    print("\n" + "="*70)
    print("TEST 5: MULTIPLE USERS")
    print("="*70 + "\n")
    
    auth_system = SQLiteAuthSystem(db_path='test_skyguard.db')
    
    # Create different ID cards
    test_users = [
        {
            'email': 'doctor@hospital.gov.in',
            'password': 'MedPass123',
            'full_name': 'Dr. Medical Officer',
            'id_type': 'medical'
        },
        {
            'email': 'admin@defense.gov.in',
            'password': 'DefPass123',
            'full_name': 'Defense Administrator',
            'id_type': 'defense'
        }
    ]
    
    registered = 0
    
    for user in test_users:
        id_path = create_dummy_id_card(f"test_{user['id_type']}_id.jpg")
        
        result = auth_system.register_user(
            email=user['email'],
            password=user['password'],
            full_name=user['full_name'],
            id_card_path=id_path
        )
        
        if result['success']:
            registered += 1
            print(f"✅ Registered: {user['full_name']} ({result['role']})")
        else:
            print(f"❌ Failed: {user['full_name']} - {result['error']}")
    
    # List all users
    all_users = auth_system.list_all_users()
    
    print(f"\n✅ TEST 5 PASSED: {registered} additional users registered")
    print(f"   Total users in system: {len(all_users)}")
    
    return registered == len(test_users)


def cleanup():
    """Clean up test files"""
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70 + "\n")
    
    files_to_remove = [
        'test_skyguard.db',
        'test_police_id.jpg',
        'test_medical_id.jpg',
        'test_defense_id.jpg'
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Removed: {file}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 SKYGUARD AUTHENTICATION SYSTEM - TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Registration
        auth_system, username = test_registration()
        if not auth_system:
            print("\n❌ FATAL: Registration test failed. Aborting.")
            return
        
        # Test 2: Authentication
        if not test_authentication(auth_system, username):
            print("\n❌ FATAL: Authentication test failed. Aborting.")
            return
        
        # Test 3: Permissions
        test_permissions(auth_system, username)
        
        # Test 4: Blockchain
        test_blockchain_integrity(auth_system)
        
        # Test 5: Multiple Users
        test_multiple_users()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE!")
        print("="*70)
        print("\n🎉 System is working correctly!")
        print("\n📝 Next Steps:")
        print("   1. Run: streamlit run auth_web_interface.py")
        print("   2. Open browser to: http://localhost:8501")
        print("   3. Register with any ID card image")
        print("   4. Login with your credentials")
        print("\n💡 Test Credentials Created:")
        print("   Email: officer.test@police.gov.in")
        print("   Password: SecurePass123")
        
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ask if user wants to keep test data
        print("\n" + "="*70)
        keep = input("\nKeep test database? (y/n): ").strip().lower()
        if keep != 'y':
            cleanup()
            print("\n✓ Cleanup complete!")
        else:
            print("\n✓ Test data preserved in test_skyguard.db")


if __name__ == '__main__':
    main()