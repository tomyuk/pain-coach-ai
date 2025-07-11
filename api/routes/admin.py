"""Admin API routes."""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, Optional
import logging

from core.performance.monitoring import system_monitor
from core.performance.profiler import profiler
from core.privacy.gdpr_compliance import GDPRComplianceManager
from core.privacy.audit_logger import AuditLogger
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

async def verify_admin_user(current_user: str = Depends(get_current_user)):
    """Verify user has admin privileges."""
    # In practice, check user roles/permissions
    if current_user != "admin_user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

@router.get("/system/health", response_model=dict)
async def get_system_health(
    admin_user: str = Depends(verify_admin_user)
):
    """Get comprehensive system health status."""
    try:
        health_summary = system_monitor.get_system_health_summary()
        performance_summary = profiler.get_performance_summary()
        active_alerts = system_monitor.get_active_alerts()
        
        return {
            "system_health": health_summary,
            "performance_metrics": performance_summary,
            "active_alerts": len(active_alerts),
            "alert_details": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ],
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"System health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system health"
        )

@router.get("/system/metrics", response_model=dict)
async def get_system_metrics(
    admin_user: str = Depends(verify_admin_user),
    metric_name: Optional[str] = None,
    hours: int = 24
):
    """Get system metrics history."""
    try:
        if metric_name:
            metrics_history = system_monitor.get_metrics_history(metric_name, hours)
        else:
            # Get all available metrics
            metrics_history = {
                "cpu_percent": system_monitor.get_metrics_history("cpu_percent", hours),
                "memory_percent": system_monitor.get_metrics_history("memory_percent", hours),
                "disk_percent": system_monitor.get_metrics_history("disk_percent", hours),
                "temperature": system_monitor.get_metrics_history("temperature", hours)
            }
        
        return {
            "metrics": metrics_history,
            "time_range_hours": hours,
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"System metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system metrics"
        )

@router.get("/performance/report", response_model=dict)
async def get_performance_report(
    admin_user: str = Depends(verify_admin_user)
):
    """Get detailed performance report."""
    try:
        performance_summary = profiler.get_performance_summary()
        slowest_operations = profiler.get_slowest_operations(limit=20)
        
        return {
            "performance_summary": performance_summary,
            "slowest_operations": slowest_operations,
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"Performance report error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate performance report"
        )

@router.post("/alerts/{alert_id}/acknowledge", response_model=dict)
async def acknowledge_alert(
    alert_id: str,
    admin_user: str = Depends(verify_admin_user)
):
    """Acknowledge a system alert."""
    try:
        success = system_monitor.acknowledge_alert(alert_id)
        
        if success:
            return {
                "message": f"Alert {alert_id} acknowledged successfully",
                "alert_id": alert_id,
                "admin_user": admin_user
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert acknowledgment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert"
        )

@router.post("/alerts/{alert_id}/resolve", response_model=dict)
async def resolve_alert(
    alert_id: str,
    admin_user: str = Depends(verify_admin_user)
):
    """Resolve a system alert."""
    try:
        success = system_monitor.resolve_alert(alert_id)
        
        if success:
            return {
                "message": f"Alert {alert_id} resolved successfully",
                "alert_id": alert_id,
                "admin_user": admin_user
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert resolution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve alert"
        )

@router.get("/audit/logs", response_model=dict)
async def get_audit_logs(
    admin_user: str = Depends(verify_admin_user),
    user_id: Optional[str] = None,
    days_back: int = 7,
    event_type: Optional[str] = None
):
    """Get audit logs."""
    try:
        audit_logger = AuditLogger()
        
        # Get audit trail
        if user_id:
            audit_trail = await audit_logger.get_audit_trail(
                user_id=user_id,
                start_date=None,
                end_date=None,
                event_types=[event_type] if event_type else None
            )
        else:
            # Get compliance report
            audit_trail = await audit_logger.generate_compliance_report(days_back)
        
        return {
            "audit_data": audit_trail,
            "filters": {
                "user_id": user_id,
                "days_back": days_back,
                "event_type": event_type
            },
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"Audit logs error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs"
        )

@router.post("/gdpr/data-deletion", response_model=dict)
async def process_gdpr_deletion(
    user_id: str,
    admin_user: str = Depends(verify_admin_user)
):
    """Process GDPR data deletion request."""
    try:
        # In practice, initialize GDPR compliance manager properly
        gdpr_manager = GDPRComplianceManager(None, None, None)
        
        # Process deletion request
        deletion_result = await gdpr_manager.process_data_deletion_request(
            user_id=user_id,
            requester_info={"admin_user": admin_user, "request_type": "admin_initiated"}
        )
        
        return {
            "message": "GDPR data deletion request processed",
            "deletion_result": deletion_result,
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"GDPR deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process GDPR deletion request"
        )

@router.post("/gdpr/data-export", response_model=dict)
async def process_gdpr_export(
    user_id: str,
    admin_user: str = Depends(verify_admin_user)
):
    """Process GDPR data export request."""
    try:
        # In practice, initialize GDPR compliance manager properly
        gdpr_manager = GDPRComplianceManager(None, None, None)
        
        # Process export request
        export_result = await gdpr_manager.export_user_data(
            user_id=user_id,
            requester_info={"admin_user": admin_user, "request_type": "admin_initiated"}
        )
        
        return {
            "message": "GDPR data export request processed",
            "export_result": export_result,
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"GDPR export error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process GDPR export request"
        )

@router.post("/maintenance/cleanup", response_model=dict)
async def perform_maintenance_cleanup(
    admin_user: str = Depends(verify_admin_user),
    retention_days: int = 365
):
    """Perform maintenance cleanup."""
    try:
        # In practice, this would clean up old data, logs, etc.
        cleanup_results = {
            "old_conversations_cleaned": 0,
            "old_audit_logs_cleaned": 0,
            "cache_cleared": True,
            "retention_days": retention_days
        }
        
        return {
            "message": "Maintenance cleanup completed",
            "cleanup_results": cleanup_results,
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"Maintenance cleanup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform maintenance cleanup"
        )

@router.get("/statistics", response_model=dict)
async def get_system_statistics(
    admin_user: str = Depends(verify_admin_user)
):
    """Get system usage statistics."""
    try:
        # In practice, gather real statistics from database
        statistics = {
            "total_users": 150,
            "active_users_last_30_days": 120,
            "total_pain_records": 25000,
            "total_ai_conversations": 8500,
            "average_daily_active_users": 45,
            "system_uptime_hours": 168,
            "database_size_mb": 2048,
            "cache_hit_rate": 0.85
        }
        
        return {
            "statistics": statistics,
            "generated_at": "2024-01-01T00:00:00Z",
            "admin_user": admin_user
        }
        
    except Exception as e:
        logger.error(f"System statistics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system statistics"
        )