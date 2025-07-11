"""Sync scheduler for automated health data collection."""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum
from .health_manager import HealthDataManager

logger = logging.getLogger(__name__)

class SyncFrequency(str, Enum):
    """Sync frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MANUAL = "manual"

class SyncScheduler:
    """Automated sync scheduler for health data providers."""
    
    def __init__(self, health_manager: HealthDataManager):
        self.health_manager = health_manager
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        self.sync_configs: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
        self.sync_callbacks: List[Callable] = []
        
    def add_sync_callback(self, callback: Callable):
        """Add callback to be called after each sync."""
        self.sync_callbacks.append(callback)
        
    def configure_provider_sync(self, provider_name: str, 
                              frequency: SyncFrequency,
                              user_id: str,
                              enabled: bool = True,
                              sync_hours: Optional[List[int]] = None,
                              max_retries: int = 3):
        """Configure sync settings for a provider."""
        self.sync_configs[provider_name] = {
            "frequency": frequency,
            "user_id": user_id,
            "enabled": enabled,
            "sync_hours": sync_hours or [6, 12, 18],  # Default sync hours
            "max_retries": max_retries,
            "retry_count": 0,
            "last_sync": None,
            "last_error": None
        }
        
        logger.info(f"Configured sync for {provider_name}: {frequency.value}")
        
    async def start_scheduler(self):
        """Start the sync scheduler."""
        if self.is_running:
            logger.warning("Sync scheduler is already running")
            return
            
        self.is_running = True
        logger.info("Starting sync scheduler")
        
        # Start sync tasks for each configured provider
        for provider_name, config in self.sync_configs.items():
            if config["enabled"]:
                task = asyncio.create_task(
                    self._sync_loop(provider_name, config)
                )
                self.sync_tasks[provider_name] = task
                
        # Start main scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
    async def stop_scheduler(self):
        """Stop the sync scheduler."""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("Stopping sync scheduler")
        
        # Cancel all sync tasks
        for task in self.sync_tasks.values():
            task.cancel()
            
        # Cancel scheduler task
        if hasattr(self, 'scheduler_task'):
            self.scheduler_task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.sync_tasks.values(), return_exceptions=True)
        
        self.sync_tasks.clear()
        
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check for failed tasks and restart them
                await self._check_and_restart_failed_tasks()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)
                
    async def _check_and_restart_failed_tasks(self):
        """Check for failed sync tasks and restart them."""
        for provider_name, task in list(self.sync_tasks.items()):
            if task.done():
                # Task has completed (likely due to an error)
                config = self.sync_configs[provider_name]
                
                if config["enabled"]:
                    logger.warning(f"Restarting sync task for {provider_name}")
                    
                    # Create new task
                    new_task = asyncio.create_task(
                        self._sync_loop(provider_name, config)
                    )
                    self.sync_tasks[provider_name] = new_task
                    
    async def _sync_loop(self, provider_name: str, config: Dict[str, Any]):
        """Sync loop for a specific provider."""
        while self.is_running and config["enabled"]:
            try:
                # Calculate next sync time
                next_sync = self._calculate_next_sync_time(config)
                
                # Wait until next sync time
                now = datetime.utcnow()
                if next_sync > now:
                    wait_seconds = (next_sync - now).total_seconds()
                    await asyncio.sleep(wait_seconds)
                
                # Perform sync
                if self.is_running:
                    await self._perform_sync(provider_name, config)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error for {provider_name}: {e}")
                config["last_error"] = str(e)
                config["retry_count"] += 1
                
                # Exponential backoff for retries
                backoff_seconds = min(300, 30 * (2 ** config["retry_count"]))
                await asyncio.sleep(backoff_seconds)
                
    def _calculate_next_sync_time(self, config: Dict[str, Any]) -> datetime:
        """Calculate the next sync time for a provider."""
        now = datetime.utcnow()
        frequency = config["frequency"]
        
        if frequency == SyncFrequency.HOURLY:
            # Next hour
            next_sync = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            
        elif frequency == SyncFrequency.DAILY:
            # Next sync hour today or tomorrow
            sync_hours = config["sync_hours"]
            current_hour = now.hour
            
            # Find next sync hour
            next_hour = None
            for hour in sorted(sync_hours):
                if hour > current_hour:
                    next_hour = hour
                    break
                    
            if next_hour is None:
                # Next sync is tomorrow at first sync hour
                next_sync = now.replace(hour=sync_hours[0], minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_sync = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
                
        elif frequency == SyncFrequency.WEEKLY:
            # Next week at first sync hour
            days_ahead = 7 - now.weekday()
            next_sync = now.replace(hour=config["sync_hours"][0], minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
            
        else:  # MANUAL
            # Very far in the future (effectively disabled)
            next_sync = now + timedelta(days=365)
            
        return next_sync
        
    async def _perform_sync(self, provider_name: str, config: Dict[str, Any]):
        """Perform sync for a provider."""
        try:
            logger.info(f"Starting sync for {provider_name}")
            
            # Determine sync window
            if config["frequency"] == SyncFrequency.HOURLY:
                days_back = 1
            elif config["frequency"] == SyncFrequency.DAILY:
                days_back = 1
            else:  # WEEKLY
                days_back = 7
                
            # Perform sync
            result = await self.health_manager.sync_specific_provider(
                provider_name,
                config["user_id"],
                days_back
            )
            
            if result.get("status") == "success":
                config["last_sync"] = datetime.utcnow()
                config["retry_count"] = 0
                config["last_error"] = None
                
                logger.info(f"Sync completed successfully for {provider_name}")
                
                # Call sync callbacks
                for callback in self.sync_callbacks:
                    try:
                        await callback(provider_name, result)
                    except Exception as e:
                        logger.error(f"Sync callback error: {e}")
                        
            else:
                config["last_error"] = result.get("error", "Unknown error")
                config["retry_count"] += 1
                logger.error(f"Sync failed for {provider_name}: {config['last_error']}")
                
        except Exception as e:
            config["last_error"] = str(e)
            config["retry_count"] += 1
            logger.error(f"Sync error for {provider_name}: {e}")
            
    async def trigger_manual_sync(self, provider_name: str, user_id: str) -> Dict[str, Any]:
        """Trigger manual sync for a provider."""
        logger.info(f"Manual sync triggered for {provider_name}")
        
        result = await self.health_manager.sync_specific_provider(
            provider_name,
            user_id,
            days_back=1
        )
        
        # Update config if provider is configured
        if provider_name in self.sync_configs:
            config = self.sync_configs[provider_name]
            if result.get("status") == "success":
                config["last_sync"] = datetime.utcnow()
                config["retry_count"] = 0
                config["last_error"] = None
            else:
                config["last_error"] = result.get("error", "Unknown error")
                
        return result
        
    def get_sync_status(self) -> Dict[str, Dict[str, Any]]:
        """Get sync status for all configured providers."""
        status = {}
        
        for provider_name, config in self.sync_configs.items():
            next_sync = self._calculate_next_sync_time(config)
            
            status[provider_name] = {
                "frequency": config["frequency"],
                "enabled": config["enabled"],
                "last_sync": config["last_sync"].isoformat() if config["last_sync"] else None,
                "next_sync": next_sync.isoformat(),
                "retry_count": config["retry_count"],
                "last_error": config["last_error"],
                "task_running": provider_name in self.sync_tasks and not self.sync_tasks[provider_name].done()
            }
            
        return status
        
    def update_provider_config(self, provider_name: str, **kwargs):
        """Update configuration for a provider."""
        if provider_name not in self.sync_configs:
            raise ValueError(f"Provider {provider_name} not configured")
            
        config = self.sync_configs[provider_name]
        
        # Update config
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
                
        # Restart sync task if frequency changed
        if 'frequency' in kwargs or 'enabled' in kwargs:
            if provider_name in self.sync_tasks:
                self.sync_tasks[provider_name].cancel()
                
                if config["enabled"]:
                    new_task = asyncio.create_task(
                        self._sync_loop(provider_name, config)
                    )
                    self.sync_tasks[provider_name] = new_task
                    
        logger.info(f"Updated config for {provider_name}: {kwargs}")