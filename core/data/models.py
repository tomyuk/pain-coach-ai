"""SQLAlchemy models for Pain Coach AI Pascal."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

Base = declarative_base()

class User(Base):
    """User profile model."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Basic information (encrypted)
    name_encrypted = Column(Text)
    birth_year = Column(Integer)
    gender = Column(String(50))
    
    # Medical information
    primary_condition = Column(String(100))
    diagnosis_date = Column(DateTime)
    medications = Column(JSON)
    
    # Settings
    timezone = Column(String(50), default="Asia/Tokyo")
    language = Column(String(10), default="ja")
    privacy_level = Column(Integer, default=2)
    
    # Metadata
    data_version = Column(Integer, default=1)
    consent_version = Column(Integer, default=1)
    last_backup = Column(DateTime)
    
    # Relationships
    pain_records = relationship("PainRecord", back_populates="user")
    conversations = relationship("AIConversation", back_populates="user")
    analysis_cache = relationship("AnalysisCache", back_populates="user")

class PainRecord(Base):
    """Pain record model."""
    __tablename__ = "pain_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    # Pain evaluation
    pain_level = Column(Integer)  # 0-10 scale
    pain_type = Column(JSON)  # Array of pain types
    pain_locations = Column(JSON)  # Body map coordinates
    pain_intensity_curve = Column(JSON)  # Time-series within day
    
    # Context
    weather_data = Column(JSON)
    activity_before = Column(String(200))
    sleep_quality = Column(Integer)  # 1-5 scale
    stress_level = Column(Integer)  # 1-5 scale
    
    # Interventions
    medications_taken = Column(JSON)
    non_med_interventions = Column(JSON)
    effectiveness_rating = Column(Integer)
    
    # Metadata
    input_method = Column(String(20))  # 'manual', 'voice', 'api'
    confidence_score = Column(Float)
    source_device = Column(String(50))
    
    # Relationships
    user = relationship("User", back_populates="pain_records")

class AIConversation(Base):
    """AI conversation model."""
    __tablename__ = "ai_conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    session_id = Column(UUID(as_uuid=True))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Conversation content (encrypted)
    user_message_encrypted = Column(Text)
    ai_response_encrypted = Column(Text)
    
    # Context
    conversation_type = Column(String(50))  # 'pain_check', 'counseling', 'analysis', 'emergency'
    mood_detected = Column(String(50))
    pain_context = Column(JSON)
    
    # AI processing info
    model_used = Column(String(50))
    processing_time_ms = Column(Integer)
    confidence_score = Column(Float)
    
    # Privacy
    retention_until = Column(DateTime)
    anonymized = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="conversations")

class AnalysisCache(Base):
    """Analysis results cache model."""
    __tablename__ = "analysis_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    analysis_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    valid_until = Column(DateTime)
    
    # Result data
    result_data = Column(JSON)
    confidence_interval = Column(JSON)
    
    # Computation metadata
    data_points_count = Column(Integer)
    algorithm_version = Column(String(20))
    computation_time_ms = Column(Integer)
    
    # Relationships
    user = relationship("User", back_populates="analysis_cache")

class HealthIntegration(Base):
    """Health data integration model."""
    __tablename__ = "health_integrations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    provider = Column(String(50))  # 'fitbit', 'apple_health', 'weather'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Integration data
    access_token_encrypted = Column(Text)
    refresh_token_encrypted = Column(Text)
    config_data = Column(JSON)
    
    # Status
    active = Column(Boolean, default=True)
    last_sync = Column(DateTime)
    sync_status = Column(String(20))  # 'success', 'error', 'pending'
    
    # Relationships
    user = relationship("User")

class UserPreference(Base):
    """User preferences model."""
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Notification preferences
    notifications_enabled = Column(Boolean, default=True)
    reminder_frequency = Column(String(20))  # 'hourly', 'daily', 'weekly'
    quiet_hours_start = Column(String(5))  # HH:MM format
    quiet_hours_end = Column(String(5))
    
    # AI preferences
    ai_personality = Column(String(20))  # 'gentle', 'encouraging', 'professional'
    voice_enabled = Column(Boolean, default=True)
    auto_analysis = Column(Boolean, default=True)
    
    # Privacy preferences
    data_retention_days = Column(Integer, default=365)
    share_anonymous_data = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User")