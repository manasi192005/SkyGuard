// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SkyGuardRBAC
 * @dev Role-Based Access Control Smart Contract for SkyGuard System
 * @notice Manages user roles and permissions on blockchain
 */
contract SkyGuardRBAC {
    
    // Role enumeration
    enum Role { 
        NONE,       // 0 - No role assigned
        ADMIN,      // 1 - Full system access
        POLICE,     // 2 - Live alerts + suspect access
        MEDICAL,    // 3 - Fall + injury alerts only
        DISASTER,   // 4 - Stampede + crowd analytics
        MUNICIPAL,  // 5 - Public safety analytics
        AUDITOR,    // 6 - Read-only logs
        VIEWER      // 7 - Basic view only
    }
    
    // User structure
    struct User {
        string username;
        string email;
        Role role;
        bool isActive;
        uint256 registeredAt;
        address registeredBy;
    }
    
    // Access log structure
    struct AccessLog {
        address user;
        string resource;
        bool granted;
        uint256 timestamp;
    }
    
    // State variables
    address public owner;
    mapping(address => User) public users;
    mapping(address => bool) public isRegistered;
    address[] public userAddresses;
    AccessLog[] public accessLogs;
    
    // Events
    event UserRegistered(address indexed userAddress, string username, Role role);
    event UserDeactivated(address indexed userAddress, string username);
    event UserActivated(address indexed userAddress, string username);
    event RoleChanged(address indexed userAddress, Role oldRole, Role newRole);
    event AccessAttempt(address indexed user, string resource, bool granted);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }
    
    modifier onlyAdmin() {
        require(users[msg.sender].role == Role.ADMIN, "Only admin can perform this action");
        require(users[msg.sender].isActive, "Admin account is not active");
        _;
    }
    
    modifier onlyActiveUser() {
        require(isRegistered[msg.sender], "User not registered");
        require(users[msg.sender].isActive, "User account is not active");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
        
        // Register owner as admin
        users[owner] = User({
            username: "System Admin",
            email: "admin@skyguard.com",
            role: Role.ADMIN,
            isActive: true,
            registeredAt: block.timestamp,
            registeredBy: owner
        });
        
        isRegistered[owner] = true;
        userAddresses.push(owner);
        
        emit UserRegistered(owner, "System Admin", Role.ADMIN);
    }
    
    /**
     * @dev Register a new user
     * @param userAddress Ethereum address of the user
     * @param username Username of the user
     * @param email Email of the user
     * @param role Role to assign to the user
     */
    function registerUser(
        address userAddress,
        string memory username,
        string memory email,
        Role role
    ) public onlyAdmin {
        require(!isRegistered[userAddress], "User already registered");
        require(userAddress != address(0), "Invalid address");
        require(role != Role.NONE, "Invalid role");
        
        users[userAddress] = User({
            username: username,
            email: email,
            role: role,
            isActive: true,
            registeredAt: block.timestamp,
            registeredBy: msg.sender
        });
        
        isRegistered[userAddress] = true;
        userAddresses.push(userAddress);
        
        emit UserRegistered(userAddress, username, role);
    }
    
    /**
     * @dev Deactivate a user account
     * @param userAddress Address of the user to deactivate
     */
    function deactivateUser(address userAddress) public onlyAdmin {
        require(isRegistered[userAddress], "User not registered");
        require(userAddress != owner, "Cannot deactivate owner");
        
        users[userAddress].isActive = false;
        
        emit UserDeactivated(userAddress, users[userAddress].username);
    }
    
    /**
     * @dev Activate a user account
     * @param userAddress Address of the user to activate
     */
    function activateUser(address userAddress) public onlyAdmin {
        require(isRegistered[userAddress], "User not registered");
        
        users[userAddress].isActive = true;
        
        emit UserActivated(userAddress, users[userAddress].username);
    }
    
    /**
     * @dev Change user role
     * @param userAddress Address of the user
     * @param newRole New role to assign
     */
    function changeUserRole(address userAddress, Role newRole) public onlyAdmin {
        require(isRegistered[userAddress], "User not registered");
        require(userAddress != owner, "Cannot change owner role");
        require(newRole != Role.NONE, "Invalid role");
        
        Role oldRole = users[userAddress].role;
        users[userAddress].role = newRole;
        
        emit RoleChanged(userAddress, oldRole, newRole);
    }
    
    /**
     * @dev Get user details
     * @param userAddress Address of the user
     */
    function getUser(address userAddress) public view returns (
        string memory username,
        string memory email,
        Role role,
        bool isActive,
        uint256 registeredAt
    ) {
        require(isRegistered[userAddress], "User not registered");
        
        User memory user = users[userAddress];
        return (user.username, user.email, user.role, user.isActive, user.registeredAt);
    }
    
    /**
     * @dev Check if user has specific permission
     * @param userAddress Address of the user
     * @param permission Permission to check (as string)
     */
    function hasPermission(address userAddress, string memory permission) public view returns (bool) {
        if (!isRegistered[userAddress] || !users[userAddress].isActive) {
            return false;
        }
        
        Role role = users[userAddress].role;
        
        // ADMIN has all permissions
        if (role == Role.ADMIN) {
            return true;
        }
        
        // Check specific permissions based on role
        bytes32 permHash = keccak256(bytes(permission));
        
        if (role == Role.POLICE) {
            return (
                permHash == keccak256(bytes("VIEW_LIVE_FEED")) ||
                permHash == keccak256(bytes("VIEW_SUSPECTS")) ||
                permHash == keccak256(bytes("RECEIVE_ALERTS")) ||
                permHash == keccak256(bytes("VIEW_ANALYTICS")) ||
                permHash == keccak256(bytes("EXPORT_DATA"))
            );
        }
        
        if (role == Role.MEDICAL) {
            return (
                permHash == keccak256(bytes("VIEW_EMERGENCY")) ||
                permHash == keccak256(bytes("RECEIVE_ALERTS")) ||
                permHash == keccak256(bytes("VIEW_ANALYTICS"))
            );
        }
        
        if (role == Role.DISASTER) {
            return (
                permHash == keccak256(bytes("VIEW_HEATMAP")) ||
                permHash == keccak256(bytes("VIEW_STAMPEDE")) ||
                permHash == keccak256(bytes("RECEIVE_ALERTS")) ||
                permHash == keccak256(bytes("VIEW_ANALYTICS"))
            );
        }
        
        if (role == Role.MUNICIPAL) {
            return (
                permHash == keccak256(bytes("VIEW_HEATMAP")) ||
                permHash == keccak256(bytes("VIEW_ANALYTICS")) ||
                permHash == keccak256(bytes("VIEW_STAMPEDE"))
            );
        }
        
        if (role == Role.AUDITOR) {
            return permHash == keccak256(bytes("VIEW_ANALYTICS"));
        }
        
        if (role == Role.VIEWER) {
            return permHash == keccak256(bytes("VIEW_LIVE_FEED"));
        }
        
        return false;
    }
    
    /**
     * @dev Log access attempt (called by system)
     * @param resource Resource being accessed
     * @param granted Whether access was granted
     */
    function logAccess(string memory resource, bool granted) public onlyActiveUser {
        accessLogs.push(AccessLog({
            user: msg.sender,
            resource: resource,
            granted: granted,
            timestamp: block.timestamp
        }));
        
        emit AccessAttempt(msg.sender, resource, granted);
    }
    
    /**
     * @dev Get total number of users
     */
    function getUserCount() public view returns (uint256) {
        return userAddresses.length;
    }
    
    /**
     * @dev Get access log count
     */
    function getAccessLogCount() public view returns (uint256) {
        return accessLogs.length;
    }
    
    /**
     * @dev Get access log by index
     */
    function getAccessLog(uint256 index) public view returns (
        address user,
        string memory resource,
        bool granted,
        uint256 timestamp
    ) {
        require(index < accessLogs.length, "Index out of bounds");
        
        AccessLog memory log = accessLogs[index];
        return (log.user, log.resource, log.granted, log.timestamp);
    }
    
    /**
     * @dev Get all user addresses (paginated)
     */
    function getUserAddresses(uint256 offset, uint256 limit) public view returns (address[] memory) {
        require(offset < userAddresses.length, "Offset out of bounds");
        
        uint256 end = offset + limit;
        if (end > userAddresses.length) {
            end = userAddresses.length;
        }
        
        address[] memory result = new address[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            result[i - offset] = userAddresses[i];
        }
        
        return result;
    }
}
