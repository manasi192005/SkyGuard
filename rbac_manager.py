import hashlib
import json
import time

# --- MAGIC CLASSES (Fixes 'AttributeError' for Role & Permission) ---
class AutoMagic(type):
    """Automatically converts Role.POLICE -> 'police' to prevent errors"""
    def __getattr__(cls, name):
        return name.lower()

class Role(metaclass=AutoMagic):
    # These are defaults, but any other role (like Role.DOCTOR) works automatically!
    ADMIN = "admin"
    USER = "user"
    POLICE = "police"

class Permission(metaclass=AutoMagic):
    READ = "read"
    WRITE = "write"

# --- BLOCKCHAIN LOGIC ---
class BlockchainRBAC:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'data': "Genesis Block",
            'previous_hash': "0",
            'hash': self.calculate_hash(0, "0", time.time(), "Genesis Block")
        }
        self.chain.append(genesis_block)
    
    def calculate_hash(self, index, previous_hash, timestamp, data):
        """Creates a secure SHA-256 hash for the block"""
        value = str(index) + str(previous_hash) + str(timestamp) + str(data)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()
    
    def add_role_assignment(self, username, role, assigned_by):
        """
        THIS IS THE MISSING FUNCTION YOU NEEDED.
        It saves the user's role to the blockchain.
        """
        previous_block = self.chain[-1]
        index = len(self.chain)
        timestamp = time.time()
        
        # Ensure role is converted to string to avoid JSON errors
        role_str = str(role)
        
        data = json.dumps({
            "event": "ROLE_ASSIGN", 
            "user": username, 
            "role": role_str, 
            "by": assigned_by
        })
        
        current_hash = self.calculate_hash(index, previous_block['hash'], timestamp, data)
        
        new_block = {
            'index': index,
            'timestamp': timestamp,
            'data': data,
            'previous_hash': previous_block['hash'],
            'hash': current_hash
        }
        self.chain.append(new_block)
        return True