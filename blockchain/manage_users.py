"""
Blockchain User Management Interface
"""

from rbac_manager import BlockchainRBAC, Role, Permission

def main():
    print("\n" + "="*70)
    print("🔐 SkyGuard Blockchain User Management")
    print("="*70)
    
    rbac = BlockchainRBAC()
    
    while True:
        print("\n📋 Menu:")
        print("1. Add user")
        print("2. View users")
        print("3. Test permissions")
        print("4. View audit trail")
        print("5. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == '1':
            username = input("Username: ")
            email = input("Email: ")
            
            print("\nRoles:")
            for i, role in enumerate(Role):
                if role != Role.NONE:
                    print(f"  {role.value}. {role.name}")
            
            role_id = int(input("Role: "))
            role = Role(role_id)
            
            rbac.add_user(username, email, role)
            print(f"\n✅ User '{username}' added with role {role.name}")
        
        elif choice == '2':
            users = rbac.get_all_users()
            print(f"\n👥 Users ({len(users)}):")
            for username, user in users.items():
                status = "🟢" if user['active'] else "🔴"
                print(f"  {status} {username} - {user['role']} ({user['email']})")
        
        elif choice == '3':
            username = input("Username: ")
            
            print("\nPermissions:")
            for i, perm in enumerate(Permission):
                print(f"  {i+1}. {perm.name}")
            
            perm_idx = int(input("Permission: ")) - 1
            permission = list(Permission)[perm_idx]
            
            has = rbac.has_permission(username, permission)
            print(f"\n{'✅ GRANTED' if has else '❌ DENIED'}")
        
        elif choice == '4':
            username = input("Username (empty for all): ").strip() or None
            logs = rbac.get_audit_trail(username, 20)
            
            print(f"\n📜 Access Logs ({len(logs)}):")
            for log in logs:
                status = "✓" if log['granted'] else "✗"
                print(f"  [{log['timestamp']}] {log['username']} → {log['resource']}: {status}")
        
        elif choice == '5':
            break

if __name__ == '__main__':
    main()
