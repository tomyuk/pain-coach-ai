"""Pydantic schemas for Pain Coach AI Pascal."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class PainType(str, Enum):
    """Pain type enumeration."""
    SHARP = "sharp"
    DULL = "dull"
    BURNING = "burning"
    ACHING = "aching"
    SHOOTING = "shooting"
    THROBBING = "throbbing"
    CRAMPING = "cramping"

class ConversationType(str, Enum):
    """AI conversation type enumeration."""
    PAIN_CHECK = "pain_check"
    COUNSELING = "counseling"
    ANALYSIS = "analysis"
    EMERGENCY = "emergency"
    GENERAL = "general"

class Gender(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class UserSchema(BaseModel):
    """User schema for validation."""
    id: Optional[str] = None
    name: Optional[str] = None
    birth_year: Optional[int] = Field(None, ge=1900, le=2100)
    gender: Optional[Gender] = None
    primary_condition: Optional[str] = None
    diagnosis_date: Optional[datetime] = None
    medications: Optional[Dict[str, Any]] = None
    timezone: str = "Asia/Tokyo"
    language: str = "ja"
    privacy_level: int = Field(2, ge=1, le=3)
    
    class Config:
        use_enum_values = True

class PainRecordSchema(BaseModel):
    """Pain record schema for validation."""
    id: Optional[str] = None
    user_id: str
    pain_level: int = Field(..., ge=0, le=10)
    pain_type: Optional[List[PainType]] = None
    pain_locations: Optional[Dict[str, Any]] = None
    pain_intensity_curve: Optional[Dict[str, Any]] = None
    
    # Context
    weather_data: Optional[Dict[str, Any]] = None
    activity_before: Optional[str] = None
    sleep_quality: Optional[int] = Field(None, ge=1, le=5)
    stress_level: Optional[int] = Field(None, ge=1, le=5)
    
    # Interventions
    medications_taken: Optional[Dict[str, Any]] = None
    non_med_interventions: Optional[List[str]] = None
    effectiveness_rating: Optional[int] = Field(None, ge=1, le=5)
    
    # Metadata
    input_method: Optional[str] = "manual"
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_device: Optional[str] = None
    recorded_at: Optional[datetime] = None
    
    @validator('pain_type')
    def validate_pain_type(cls, v):
        if v is None:
            return v
        # Ensure all pain types are valid
        valid_types = [pt.value for pt in PainType]
        for pain_type in v:
            if pain_type not in valid_types:
                raise ValueError(f"Invalid pain type: {pain_type}")
        return v
    
    class Config:
        use_enum_values = True

class ConversationSchema(BaseModel):
    """AI conversation schema for validation."""
    id: Optional[str] = None
    user_id: str
    session_id: Optional[str] = None
    user_message: str
    ai_response: Optional[str] = None
    conversation_type: ConversationType = ConversationType.GENERAL
    mood_detected: Optional[str] = None
    pain_context: Optional[Dict[str, Any]] = None
    
    # AI processing info
    model_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Privacy
    retention_until: Optional[datetime] = None
    anonymized: bool = False
    timestamp: Optional[datetime] = None
    
    class Config:
        use_enum_values = True

class AnalysisResultSchema(BaseModel):
    """Analysis result schema."""
    user_id: str
    analysis_type: str
    result_data: Dict[str, Any]
    confidence_interval: Optional[Dict[str, Any]] = None
    data_points_count: int
    algorithm_version: str
    computation_time_ms: int
    created_at: Optional[datetime] = None
    valid_until: Optional[datetime] = None

class PainTrendSchema(BaseModel):
    """Pain trend analysis schema."""
    trends: List[Dict[str, Any]]
    correlations: Dict[str, float]
    statistics: Dict[str, float]
    analysis_date: datetime
    
class WeatherDataSchema(BaseModel):
    """Weather data schema."""
    pressure: Optional[float] = None
    humidity: Optional[float] = None
    temperature: Optional[float] = None
    pressure_change: Optional[float] = None
    weather_desc: Optional[str] = None
    date: Optional[datetime] = None

class HealthIntegrationSchema(BaseModel):
    """Health integration schema."""
    provider: str
    config_data: Dict[str, Any]
    active: bool = True
    last_sync: Optional[datetime] = None
    sync_status: str = "pending"

class UserPreferenceSchema(BaseModel):
    """User preference schema."""
    notifications_enabled: bool = True
    reminder_frequency: str = "daily"
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    
    # AI preferences
    ai_personality: str = "gentle"
    voice_enabled: bool = True
    auto_analysis: bool = True
    
    # Privacy preferences
    data_retention_days: int = 365
    share_anonymous_data: bool = False
    
    @validator('reminder_frequency')
    def validate_reminder_frequency(cls, v):
        valid_frequencies = ['hourly', 'daily', 'weekly', 'never']
        if v not in valid_frequencies:
            raise ValueError(f"Invalid reminder frequency: {v}")
        return v
    
    @validator('ai_personality')
    def validate_ai_personality(cls, v):
        valid_personalities = ['gentle', 'encouraging', 'professional', 'supportive']
        if v not in valid_personalities:
            raise ValueError(f"Invalid AI personality: {v}")
        return v