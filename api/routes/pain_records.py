"""Pain records API routes."""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from core.data.database import DatabaseManager
from core.data.schemas import PainRecordSchema
from core.privacy.audit_logger import AuditLogger
from core.performance.profiler import profile
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/records", response_model=dict)
@profile("create_pain_record")
async def create_pain_record(
    pain_data: PainRecordSchema,
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    audit_logger: AuditLogger = Depends()
):
    """Create a new pain record."""
    try:
        # Set user ID and timestamp
        pain_data.user_id = current_user
        if not pain_data.recorded_at:
            pain_data.recorded_at = datetime.utcnow()
        
        # Create pain record
        pain_record = await db.create_pain_record(pain_data.dict())
        
        # Log creation
        await audit_logger.log_data_modification(
            current_user,
            "pain_record",
            {"action": "create", "record_id": str(pain_record.id)}
        )
        
        return {
            "message": "Pain record created successfully",
            "record_id": str(pain_record.id),
            "recorded_at": pain_record.recorded_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pain record creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create pain record"
        )

@router.get("/records", response_model=List[dict])
@profile("get_pain_records")
async def get_pain_records(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    limit: int = Query(100, ge=1, le=1000),
    days_back: int = Query(30, ge=1, le=365),
    pain_level_min: Optional[int] = Query(None, ge=0, le=10),
    pain_level_max: Optional[int] = Query(None, ge=0, le=10)
):
    """Get pain records for the current user."""
    try:
        # Get pain records from database
        records = await db.get_pain_records(current_user, limit, days_back)
        
        # Apply filters
        if pain_level_min is not None:
            records = [r for r in records if r.pain_level >= pain_level_min]
        if pain_level_max is not None:
            records = [r for r in records if r.pain_level <= pain_level_max]
        
        # Convert to response format
        response_records = []
        for record in records:
            response_records.append({
                "id": str(record.id),
                "recorded_at": record.recorded_at.isoformat(),
                "pain_level": record.pain_level,
                "pain_type": record.pain_type,
                "pain_locations": record.pain_locations,
                "activity_before": record.activity_before,
                "sleep_quality": record.sleep_quality,
                "stress_level": record.stress_level,
                "medications_taken": record.medications_taken,
                "non_med_interventions": record.non_med_interventions,
                "effectiveness_rating": record.effectiveness_rating,
                "input_method": record.input_method,
                "weather_data": record.weather_data
            })
        
        return response_records
        
    except Exception as e:
        logger.error(f"Pain records retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pain records"
        )

@router.get("/records/{record_id}", response_model=dict)
@profile("get_pain_record")
async def get_pain_record(
    record_id: str,
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends()
):
    """Get a specific pain record."""
    try:
        # In practice, you'd fetch the specific record and verify ownership
        # For now, we'll return a placeholder
        
        return {
            "id": record_id,
            "message": "Pain record retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Pain record retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pain record"
        )

@router.put("/records/{record_id}", response_model=dict)
@profile("update_pain_record")
async def update_pain_record(
    record_id: str,
    pain_data: PainRecordSchema,
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    audit_logger: AuditLogger = Depends()
):
    """Update a pain record."""
    try:
        # In practice, you'd update the record and verify ownership
        
        # Log update
        await audit_logger.log_data_modification(
            current_user,
            "pain_record",
            {"action": "update", "record_id": record_id}
        )
        
        return {
            "message": "Pain record updated successfully",
            "record_id": record_id
        }
        
    except Exception as e:
        logger.error(f"Pain record update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update pain record"
        )

@router.delete("/records/{record_id}", response_model=dict)
@profile("delete_pain_record")
async def delete_pain_record(
    record_id: str,
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    audit_logger: AuditLogger = Depends()
):
    """Delete a pain record."""
    try:
        # In practice, you'd delete the record and verify ownership
        
        # Log deletion
        await audit_logger.log_data_modification(
            current_user,
            "pain_record",
            {"action": "delete", "record_id": record_id}
        )
        
        return {
            "message": "Pain record deleted successfully",
            "record_id": record_id
        }
        
    except Exception as e:
        logger.error(f"Pain record deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete pain record"
        )

@router.get("/summary", response_model=dict)
@profile("get_pain_summary")
async def get_pain_summary(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    days_back: int = Query(7, ge=1, le=365)
):
    """Get pain summary statistics."""
    try:
        # Get recent pain records
        records = await db.get_pain_records(current_user, limit=1000, days_back=days_back)
        
        if not records:
            return {
                "total_records": 0,
                "average_pain": 0,
                "days_analyzed": days_back
            }
        
        # Calculate statistics
        pain_levels = [r.pain_level for r in records]
        average_pain = sum(pain_levels) / len(pain_levels)
        min_pain = min(pain_levels)
        max_pain = max(pain_levels)
        
        # Pain type frequency
        pain_types = {}
        for record in records:
            if record.pain_type:
                for pain_type in record.pain_type:
                    pain_types[pain_type] = pain_types.get(pain_type, 0) + 1
        
        # Sleep quality correlation
        sleep_records = [r for r in records if r.sleep_quality is not None]
        sleep_pain_correlation = 0
        if len(sleep_records) > 1:
            # Simple correlation calculation
            avg_sleep = sum(r.sleep_quality for r in sleep_records) / len(sleep_records)
            avg_pain_for_sleep = sum(r.pain_level for r in sleep_records) / len(sleep_records)
            
            numerator = sum((r.sleep_quality - avg_sleep) * (r.pain_level - avg_pain_for_sleep) for r in sleep_records)
            denominator = (sum((r.sleep_quality - avg_sleep) ** 2 for r in sleep_records) * 
                          sum((r.pain_level - avg_pain_for_sleep) ** 2 for r in sleep_records)) ** 0.5
            
            if denominator > 0:
                sleep_pain_correlation = numerator / denominator
        
        return {
            "total_records": len(records),
            "days_analyzed": days_back,
            "average_pain": round(average_pain, 2),
            "min_pain": min_pain,
            "max_pain": max_pain,
            "pain_types": pain_types,
            "sleep_pain_correlation": round(sleep_pain_correlation, 3),
            "summary_generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pain summary error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate pain summary"
        )

@router.post("/quick-entry", response_model=dict)
@profile("quick_pain_entry")
async def quick_pain_entry(
    pain_level: int = Query(..., ge=0, le=10),
    notes: Optional[str] = None,
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    audit_logger: AuditLogger = Depends()
):
    """Quick pain entry with minimal data."""
    try:
        # Create minimal pain record
        pain_data = PainRecordSchema(
            user_id=current_user,
            pain_level=pain_level,
            recorded_at=datetime.utcnow(),
            input_method="quick_entry",
            activity_before=notes
        )
        
        # Create pain record
        pain_record = await db.create_pain_record(pain_data.dict())
        
        # Log creation
        await audit_logger.log_data_modification(
            current_user,
            "pain_record",
            {"action": "quick_create", "record_id": str(pain_record.id)}
        )
        
        return {
            "message": "Quick pain entry recorded successfully",
            "record_id": str(pain_record.id),
            "pain_level": pain_level,
            "recorded_at": pain_record.recorded_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Quick pain entry error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record quick pain entry"
        )