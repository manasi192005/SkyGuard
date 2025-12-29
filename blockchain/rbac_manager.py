"""
Blockchain RBAC Manager
Python interface for SkyGuard smart contract
"""

from web3 import Web3
from eth_account import Account
import json
import os
from datetime import datetime
from enum import Enum

class Role(Enum):
    """System roles"""
    NONE = 0
    ADMIN = 1
    POLICE = 2
    MEDICAL = 3
    DISASTER = 4
    MUNICIPAL = 5
    AUDITOR = 6
    VIEWER = 7

class Permission(Enum):
    """System permissions"""
    VIEW_LIVE_FEED = "VIEW_LIVE_FEED"
    VIEW_HEATMAP = "VIEW_HEATMAP"
    VIEW_SUSPECTS = "VIEW_SUSPECTS"
    RECEIVE_ALERTS = "RECEIVE_ALERTS"
    VIEW_ANALYTICS = "VIEW_ANALYTICS"
    MANAGE_USERS = "MANAGE_USERS"
    VIEW_EMERGENCY = "VIEW_EMERGENCY"
    VIEW_STAMPEDE = "VIEW_STAMPEDE"
    EXPORT_DATA = "EXPORT_DATA"
    MODIFY_SETTINGS = "MODIFY_SETTINGS"

# Role to Permission mapping (client-side cache)
ROLE_PERMISSIONS = {
    Role.ADMIN: list(Permission),
    Role.POLICE: [
        Permission.VIEW_LIVE_FEED,
        Permission.VIEW_SUSPECTS,
        Permission.RECEIVE_ALERTS,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA
    ],
    Role.MEDICAL: [
        Permission.VIEW_EMERGENCY,
        Permission.RECEIVE_ALERTS,
        Permission.VIEW_ANALYTICS
    ],
    Role.DISASTER: [
        Permission.VIEW_HEATMAP,
        Permission.VIEW_STAMPEDE,
        Permission.RECEIVE_ALERTS,
        Permission.VIEW_ANALYTICS
    ],
    Role.MUNICIPAL: [
        Permission.VIEW_HEATMAP,
        Permission.VIEW_ANALYTICS,
        Permission.VIEW_STAMPEDE
    ],
    Role.AUDITOR: [
        Permission.VIEW_ANALYTICS
    ],
    Role.VIEWER: [
        Permission.VIEW_LIVE_FEED
    ]
}

class BlockchainRBAC:
    """Blockchain-based RBAC Manager"""
    
    def __init__(self, config_path='blockchain/config/network.json'):
        """Initialize blockchain connection"""
        self.config_path = config_path
        self.load_config()
        self.connect_blockchain()
        
        # Cache
        self.role_cache = {}
    
    def load_config(self):
        """Load blockchain configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config
            self.config = {
                'mode': 'local',  # 'local' or 'blockchain'
                'network': 'ganache',
                'rpc_url': 'http://127.0.0.1:7545',
                'contract_address': None,
                'admin_private_key': None,
                'users': {}
            }
            self.save_config()
    
    def save_config(self):
        """Save configuration"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def connect_blockchain(self):
        """Connect to Ethereum network"""
        try:
            if self.config.get('mode') == 'blockchain':
                self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
                if self.w3.is_connected():
                    print("✓ Connected to Ethereum network")
                    self.blockchain_mode = True
                else:
                    print("⚠ Blockchain connection failed, using local mode")
                    self.blockchain_mode = False
            else:
                print("✓ Running in local mode (blockchain disabled)")
                self.blockchain_mode = False
        except Exception as e:
            print(f"⚠ Blockchain error: {e}")
            self.blockchain_mode = False
    
    def add_user(self, username, email, role, added_by='ADMIN'):
        """Add new user"""
        if not isinstance(role, Role):
            role = Role[role] if isinstance(role, str) else Role(role)
        
        # Generate Ethereum account
        account = Account.create()
        
        user_data = {
            'username': username,
            'email': email,
            'role': role.name,
            'role_id': role.value,
            'eth_address': account.address,
            'private_key': account.key.hex(),
            'added_date': datetime.now().isoformat(),
            'added_by': added_by,
            'active': True
        }
        
        if self.blockchain_mode:
            # TODO: Call smart contract
            pass
        
        # Store locally
        self.config['users'][username] = user_data
        self.save_config()
        
        print(f"✓ User added: {username} ({role.name})")
        print(f"  Ethereum Address: {account.address}")
        
        return user_data
    
    def get_user_role(self, username):
        """Get user role"""
        if username in self.role_cache:
            return self.role_cache[username]
        
        user = self.config['users'].get(username)
        if user and user.get('active', True):
            role = Role[user['role']]
            self.role_cache[username] = role
            return role
        
        return None
    
    def has_permission(self, username, permission):
        """Check if user has permission"""
        if not isinstance(permission, Permission):
            permission = Permission[permission] if isinstance(permission, str) else Permission(permission)
        
        role = self.get_user_role(username)
        if not role:
            return False
        
        allowed = ROLE_PERMISSIONS.get(role, [])
        has_access = permission in allowed
        
        # Log access
        self.log_access(username, permission.value, has_access)
        
        return has_access
    
    def get_user_permissions(self, username):
        """Get all user permissions"""
        role = self.get_user_role(username)
        if not role:
            return []
        return ROLE_PERMISSIONS.get(role, [])
    
    def log_access(self, username, resource, granted):
        """Log access attempt"""
        log_entry = {
            'username': username,
            'resource': resource,
            'granted': granted,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'access_logs' not in self.config:
            self.config['access_logs'] = []
        
        self.config['access_logs'].append(log_entry)
        
        if len(self.config['access_logs']) % 10 == 0:
            self.save_config()
    
    def get_audit_trail(self, username=None, limit=100):
        """Get audit trail"""
        logs = self.config.get('access_logs', [])
        
        if username:
            logs = [log for log in logs if log['username'] == username]
        
        return logs[-limit:]
    
    def get_all_users(self):
        """Get all users"""
        return self.config.get('users', {})
    
    def deactivate_user(self, username):
        """Deactivate user"""
        if username in self.config['users']:
            self.config['users'][username]['active'] = False
            self.save_config()
            
            if username in self.role_cache:
                del self.role_cache[username]
            
            return True
        return False
    
    def activate_user(self, username):
        """Activate user"""
        if username in self.config['users']:
            self.config['users'][username]['active'] = True
            self.save_config()
            return True
        return False


if __name__ == '__main__':
    print("="*60)
    print("🔐 Blockchain RBAC Manager - Test")
    print("="*60)
    
    rbac = BlockchainRBAC()
    
    # Add test users
    rbac.add_user("admin1", "admin@skyguard.com", Role.ADMIN)
    rbac.add_user("police1", "police@mumbai.gov.in", Role.POLICE)
    rbac.add_user("medical1", "medical@hospital.com", Role.MEDICAL)
    
    # Test permissions
    print("\n🔍 Testing permissions...")
    for user in ['admin1', 'police1', 'medical1']:
        role = rbac.get_user_role(user)
        print(f"\n👤 {user} ({role.name})")
        for perm in [Permission.VIEW_SUSPECTS, Permission.VIEW_EMERGENCY]:
            has = rbac.has_permission(user, perm)
            print(f"   {perm.name}: {'✓' if has else '✗'}")
    
    print("\n✓ Test complete")
