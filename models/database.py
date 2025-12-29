"""
Database models for SkyGuard system
Stores detected suspects, crowd analytics, and event logs
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class DetectedSuspect(Base):
    """Store information about detected suspects"""
    __tablename__ = 'detected_suspects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    suspect_name = Column(String(100))
    confidence = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String(255))
    status = Column(String(50), default='active')
    notes = Column(Text)

class CrowdAnalytics(Base):
    """Store crowd density and analytics data"""
    __tablename__ = 'crowd_analytics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    crowd_density = Column(Integer)
    risk_level = Column(String(20))
    latitude = Column(Float)
    longitude = Column(Float)
    predicted_density = Column(Integer, nullable=True)
    anomaly_detected = Column(Boolean, default=False)
    anomaly_type = Column(String(50))

class EmergencyEvent(Base):
    """Store emergency events and responses"""
    __tablename__ = 'emergency_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)
    severity = Column(String(20))
    response_time = Column(Float, nullable=True)
    resolved = Column(Boolean, default=False)
    image_path = Column(String(255))
    description = Column(Text)

class SystemLog(Base):
    """Store system logs and activities"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    log_level = Column(String(20))
    module = Column(String(50))
    message = Column(Text)
    details = Column(Text, nullable=True)

# Database initialization
def init_database(db_path='data/database/skyguard.db'):
    """Initialize the database and create tables"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    """Get a database session"""
    Session = sessionmaker(bind=engine)
    return Session()

# Utility functions
def add_detected_suspect(session, name, confidence, lat, lon, image_path):
    """Add a detected suspect to the database"""
    try:
        suspect = DetectedSuspect(
            suspect_name=name,
            confidence=confidence,
            latitude=lat,
            longitude=lon,
            image_path=image_path
        )
        session.add(suspect)
        session.commit()
        return suspect.id
    except Exception as e:
        session.rollback()
        print(f"Error adding suspect: {e}")
        return None

def add_crowd_analytics(session, density, risk_level, lat, lon, predicted=None, anomaly=False, anomaly_type='normal'):
    """Add crowd analytics data"""
    try:
        analytics = CrowdAnalytics(
            crowd_density=density,
            risk_level=risk_level,
            latitude=lat,
            longitude=lon,
            predicted_density=predicted,
            anomaly_detected=anomaly,
            anomaly_type=anomaly_type
        )
        session.add(analytics)
        session.commit()
        return analytics.id
    except Exception as e:
        session.rollback()
        print(f"Error adding crowd analytics: {e}")
        return None

def add_emergency_event(session, event_type, lat, lon, severity, image_path, description):
    """Add emergency event"""
    try:
        event = EmergencyEvent(
            event_type=event_type,
            latitude=lat,
            longitude=lon,
            severity=severity,
            image_path=image_path,
            description=description
        )
        session.add(event)
        session.commit()
        return event.id
    except Exception as e:
        session.rollback()
        print(f"Error adding emergency event: {e}")
        return None

def log_system_event(session, level, module, message, details=None):
    """Log system event"""
    try:
        log = SystemLog(
            log_level=level,
            module=module,
            message=message,
            details=details
        )
        session.add(log)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error logging system event: {e}")

# Query helper functions
def get_recent_suspects(session, limit=10):
    """Get recent detected suspects"""
    return session.query(DetectedSuspect).order_by(
        DetectedSuspect.timestamp.desc()
    ).limit(limit).all()

def get_recent_analytics(session, limit=100):
    """Get recent crowd analytics"""
    return session.query(CrowdAnalytics).order_by(
        CrowdAnalytics.timestamp.desc()
    ).limit(limit).all()

def get_active_emergencies(session):
    """Get unresolved emergencies"""
    return session.query(EmergencyEvent).filter(
        EmergencyEvent.resolved == False
    ).all()


if __name__ == '__main__':
    # Test database initialization
    print("Testing database...")
    engine = init_database()
    session = get_session(engine)
    
    # Test adding data
    print("Testing data insertion...")
    
    # Test crowd analytics
    analytics_id = add_crowd_analytics(session, 25, 'medium', 19.0760, 72.8777)
    print(f"✓ Added crowd analytics: {analytics_id}")
    
    # Test suspect
    suspect_id = add_detected_suspect(session, 'Test Suspect', 0.95, 19.0760, 72.8777, 'test.jpg')
    print(f"✓ Added suspect: {suspect_id}")
    
    # Test emergency
    emergency_id = add_emergency_event(session, 'test', 19.0760, 72.8777, 'high', 'test.jpg', 'Test emergency')
    print(f"✓ Added emergency: {emergency_id}")
    
    # Test queries
    suspects = get_recent_suspects(session, 5)
    print(f"✓ Found {len(suspects)} suspects")
    
    analytics = get_recent_analytics(session, 5)
    print(f"✓ Found {len(analytics)} analytics records")
    
    print("\n✓ Database test complete!")
    session.close()
