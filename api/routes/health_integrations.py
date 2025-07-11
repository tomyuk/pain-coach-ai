"""Health integrations API routes."""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from core.integrations.health_manager import HealthDataManager
from core.integrations.sync_scheduler import SyncScheduler, SyncFrequency
from core.data.schemas import HealthIntegrationSchema
from core.performance.profiler import profile
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

class IntegrationConfig(BaseModel):
    """Health integration configuration."""
    provider: str
    config: Dict[str, Any]
    sync_frequency: SyncFrequency = SyncFrequency.DAILY
    enabled: bool = True

class SyncTrigger(BaseModel):
    """Manual sync trigger."""
    provider: str
    days_back: int = 1

# Global health manager and scheduler (in practice, inject via dependencies)
health_manager = HealthDataManager()
sync_scheduler = SyncScheduler(health_manager)

@router.post("/configure", response_model=dict)
@profile("configure_integration")
async def configure_integration(
    config: IntegrationConfig,
    current_user: str = Depends(get_current_user)
):
    """Configure health data integration."""
    try:
        # Validate provider
        supported_providers = ["fitbit", "apple_health", "weather"]
        if config.provider not in supported_providers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider. Supported providers: {supported_providers}"
            )
        
        # Configure provider based on type
        if config.provider == "fitbit":
            from core.integrations.providers import FitbitProvider
            provider = FitbitProvider(
                access_token=config.config.get("access_token"),
                refresh_token=config.config.get("refresh_token")
            )
        elif config.provider == "apple_health":
            from core.integrations.providers import AppleHealthProvider
            provider = AppleHealthProvider(
                webhook_url=config.config.get("webhook_url")
            )
        elif config.provider == "weather":
            from core.integrations.providers import WeatherProvider
            provider = WeatherProvider(
                api_key=config.config.get("api_key"),
                location=(config.config.get("lat"), config.config.get("lon"))
            )
        
        # Register provider
        health_manager.register_provider(config.provider, provider)
        
        # Configure sync schedule
        sync_scheduler.configure_provider_sync(
            provider_name=config.provider,
            frequency=config.sync_frequency,
            user_id=current_user,
            enabled=config.enabled
        )
        
        return {
            "message": f"{config.provider} integration configured successfully",
            "provider": config.provider,
            "sync_frequency": config.sync_frequency.value,
            "enabled": config.enabled
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Integration configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure integration"
        )

@router.get("/status", response_model=dict)
@profile("integration_status")
async def get_integration_status(
    current_user: str = Depends(get_current_user)
):
    """Get status of all health integrations."""
    try:
        # Get provider info
        provider_info = health_manager.get_provider_info()
        
        # Get sync status
        sync_status = sync_scheduler.get_sync_status()
        
        # Test connections
        connection_status = await health_manager.test_all_connections()
        
        return {
            "user_id": current_user,
            "providers": provider_info,
            "sync_status": sync_status,
            "connection_status": connection_status,
            "total_providers": len(provider_info),
            "active_providers": len([p for p in provider_info.values() if p["status"]["status"] == "active"])
        }
        
    except Exception as e:
        logger.error(f"Integration status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get integration status"
        )

@router.post("/sync", response_model=dict)
@profile("manual_sync")
async def trigger_manual_sync(
    sync_request: SyncTrigger,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Trigger manual sync for a provider."""
    try:
        # Trigger sync in background
        background_tasks.add_task(
            perform_manual_sync,
            sync_request.provider,
            current_user,
            sync_request.days_back
        )
        
        return {
            "message": f"Manual sync triggered for {sync_request.provider}",
            "provider": sync_request.provider,
            "days_back": sync_request.days_back,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Manual sync error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger manual sync"
        )

async def perform_manual_sync(provider: str, user_id: str, days_back: int):
    """Perform manual sync (background task)."""
    try:
        result = await sync_scheduler.trigger_manual_sync(provider, user_id)
        logger.info(f"Manual sync completed for {provider}: {result.get('status', 'unknown')}")
    except Exception as e:
        logger.error(f"Manual sync background task error: {e}")

@router.get("/data", response_model=dict)
@profile("get_health_data")
async def get_health_data(
    current_user: str = Depends(get_current_user),
    days_back: int = 7,
    provider: Optional[str] = None
):
    """Get collected health data."""
    try:
        # Collect data from all or specific provider
        if provider:
            # Get data from specific provider
            result = await health_manager.sync_specific_provider(
                provider, current_user, days_back
            )
            return {
                "user_id": current_user,
                "provider": provider,
                "data": result,
                "days_back": days_back
            }
        else:
            # Get comprehensive data
            result = await health_manager.collect_comprehensive_data(
                current_user, days_back
            )
            return result
        
    except Exception as e:
        logger.error(f"Health data retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health data"
        )

@router.get("/permissions", response_model=dict)
@profile("get_permissions")
async def get_required_permissions(
    current_user: str = Depends(get_current_user)
):
    """Get required permissions for all providers."""
    try:
        permissions = health_manager.get_provider_permissions()
        
        return {
            "user_id": current_user,
            "provider_permissions": permissions,
            "setup_instructions": {
                "fitbit": "Visit developer.fitbit.com to create an app and get API credentials",
                "apple_health": "Set up iOS Shortcuts to send health data to webhook URL",
                "weather": "Get API key from openweathermap.org"
            }
        }
        
    except Exception as e:
        logger.error(f"Permissions retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve permissions"
        )

@router.delete("/provider/{provider_name}", response_model=dict)
@profile("remove_provider")
async def remove_provider(
    provider_name: str,
    current_user: str = Depends(get_current_user)
):
    """Remove a health data provider."""
    try:
        # Remove from sync scheduler
        sync_scheduler.update_provider_config(provider_name, enabled=False)
        
        # Remove from health manager
        if provider_name in health_manager.providers:
            del health_manager.providers[provider_name]
            del health_manager.sync_status[provider_name]
        
        return {
            "message": f"Provider {provider_name} removed successfully",
            "provider": provider_name,
            "user_id": current_user
        }
        
    except Exception as e:
        logger.error(f"Provider removal error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove provider"
        )

@router.put("/provider/{provider_name}/settings", response_model=dict)
@profile("update_provider_settings")
async def update_provider_settings(
    provider_name: str,
    settings: dict,
    current_user: str = Depends(get_current_user)
):
    """Update provider settings."""
    try:
        # Update sync scheduler settings
        valid_settings = {}
        
        if "sync_frequency" in settings:
            valid_settings["frequency"] = SyncFrequency(settings["sync_frequency"])
        
        if "enabled" in settings:
            valid_settings["enabled"] = settings["enabled"]
        
        if "sync_hours" in settings:
            valid_settings["sync_hours"] = settings["sync_hours"]
        
        if valid_settings:
            sync_scheduler.update_provider_config(provider_name, **valid_settings)
        
        return {
            "message": f"Settings updated for {provider_name}",
            "provider": provider_name,
            "updated_settings": valid_settings,
            "user_id": current_user
        }
        
    except Exception as e:
        logger.error(f"Provider settings update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update provider settings"
        )

@router.get("/sync-history", response_model=dict)
@profile("sync_history")
async def get_sync_history(
    current_user: str = Depends(get_current_user),
    days_back: int = 30
):
    """Get sync history for all providers."""
    try:
        # Get sync status for all providers
        sync_status = sync_scheduler.get_sync_status()
        
        # Format history
        history = {}
        for provider, status in sync_status.items():
            history[provider] = {
                "last_sync": status.get("last_sync"),
                "next_sync": status.get("next_sync"),
                "sync_frequency": status.get("frequency"),
                "enabled": status.get("enabled"),
                "error_count": status.get("retry_count", 0),
                "last_error": status.get("last_error"),
                "status": "healthy" if status.get("last_error") is None else "error"
            }
        
        return {
            "user_id": current_user,
            "sync_history": history,
            "days_back": days_back,
            "generated_at": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Sync history error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sync history"
        )