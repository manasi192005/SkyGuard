# 🔗 SkyGuard Blockchain Module

Decentralized Role-Based Access Control using Ethereum Smart Contracts

## 📁 Structure
```
blockchain/
├── contracts/               # Solidity smart contracts
│   └── SkyGuardRBAC.sol    # Main RBAC contract
├── scripts/                # Deployment scripts
│   └── deploy.py           # Contract deployment
├── config/                 # Configuration files
│   ├── network.json        # Network settings
│   └── contract_abi.json   # Contract ABI
├── rbac_manager.py         # Python interface
└── __init__.py             # Module initialization
```

## 🚀 Quick Start

### Local Mode (No Blockchain Required)
```python
from blockchain import BlockchainRBAC, Role

# Initialize
rbac = BlockchainRBAC()

# Add user
rbac.add_user("police1", "police@city.gov", Role.POLICE)

# Check permission
has_access = rbac.has_permission("police1", "VIEW_SUSPECTS")
```

### Blockchain Mode (Requires Ganache)

1. Install Ganache: https://trufflesuite.com/ganache/
2. Start Ganache on port 7545
3. Update `config/network.json`: `"mode": "blockchain"`
4. Deploy contract: `python3 scripts/deploy.py`

## 🎭 Roles

- **ADMIN**: Full system access
- **POLICE**: Live alerts + suspect access
- **MEDICAL**: Emergency alerts only
- **DISASTER**: Stampede + crowd analytics
- **MUNICIPAL**: Public safety analytics
- **AUDITOR**: Read-only logs
- **VIEWER**: Basic view only

## 🔐 Permissions

- VIEW_LIVE_FEED
- VIEW_HEATMAP
- VIEW_SUSPECTS
- RECEIVE_ALERTS
- VIEW_ANALYTICS
- MANAGE_USERS
- VIEW_EMERGENCY
- VIEW_STAMPEDE
- EXPORT_DATA
- MODIFY_SETTINGS

## 📊 Usage Example
```python
from blockchain import BlockchainRBAC, Role, Permission

rbac = BlockchainRBAC()

# Add users
rbac.add_user("admin", "admin@skyguard.com", Role.ADMIN)
rbac.add_user("police", "police@mumbai.gov.in", Role.POLICE)
rbac.add_user("doctor", "doctor@hospital.com", Role.MEDICAL)

# Check permissions
print(rbac.has_permission("police", Permission.VIEW_SUSPECTS))  # True
print(rbac.has_permission("doctor", Permission.VIEW_SUSPECTS))  # False
print(rbac.has_permission("doctor", Permission.VIEW_EMERGENCY)) # True

# View audit trail
logs = rbac.get_audit_trail()
```

## 🔧 Integration with SkyGuard
```python
# In main system
from blockchain import BlockchainRBAC, Permission

rbac = BlockchainRBAC()

# Before showing suspects
if rbac.has_permission(current_user, Permission.VIEW_SUSPECTS):
    # Show suspect detection
    pass
else:
    # Access denied
    pass
```
