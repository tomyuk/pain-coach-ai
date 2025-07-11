"""Health data integration manager."""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class HealthDataProvider(ABC):
    """Abstract base class for health data providers."""
    
    @abstractmethod
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Fetch health data for the specified date range."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to the health data provider."""
        pass
    
    @abstractmethod
    def get_required_permissions(self) -> List[str]:
        """Get list of required permissions."""
        pass

class HealthDataManager:
    """Manages health data collection from multiple sources."""
    
    def __init__(self):
        self.providers: Dict[str, HealthDataProvider] = {}
        self.sync_status: Dict[str, Dict[str, Any]] = {}
        
    def register_provider(self, name: str, provider: HealthDataProvider):
        """Register a health data provider."""
        self.providers[name] = provider
        self.sync_status[name] = {
            "last_sync": None,
            "status": "inactive",
            "error_count": 0,
            "last_error": None
        }
        logger.info(f"Registered health data provider: {name}")
        
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test connections to all registered providers."""
        results = {}
        
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.test_connection()
                if results[name]:
                    self.sync_status[name]["status"] = "active"
                else:
                    self.sync_status[name]["status"] = "error"
                    self.sync_status[name]["error_count"] += 1
            except Exception as e:
                results[name] = False
                self.sync_status[name]["status"] = "error"
                self.sync_status[name]["last_error"] = str(e)
                self.sync_status[name]["error_count"] += 1
                logger.error(f"Connection test failed for {name}: {e}")
        
        return results
    
    async def collect_comprehensive_data(self, 
                                       user_id: str,
                                       days_back: int = 7) -> Dict[str, Any]:
        """Collect comprehensive health data from all providers."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        collection_result = {
            "user_id": user_id,
            "collection_timestamp": end_date,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data": {},
            "errors": {},
            "provider_status": {}
        }
        
        # Collect data from all providers in parallel
        tasks = []
        for name, provider in self.providers.items():
            task = asyncio.create_task(
                self._safe_collect_data(name, provider, start_date, end_date)
            )
            tasks.append((name, task))
        
        # Wait for all tasks to complete
        for name, task in tasks:
            try:
                result = await task
                if "error" in result:
                    collection_result["errors"][name] = result["error"]
                    self.sync_status[name]["status"] = "error"
                    self.sync_status[name]["last_error"] = result["error"]
                    self.sync_status[name]["error_count"] += 1
                else:
                    collection_result["data"][name] = result
                    self.sync_status[name]["status"] = "success"
                    self.sync_status[name]["last_sync"] = datetime.utcnow()
                    self.sync_status[name]["error_count"] = 0
                    
                collection_result["provider_status"][name] = self.sync_status[name].copy()
                
            except Exception as e:
                collection_result["errors"][name] = str(e)
                logger.error(f"Error collecting data from {name}: {e}")
        
        return collection_result
    
    async def _safe_collect_data(self, name: str, provider: HealthDataProvider, 
                               start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Safely collect data from a provider with error handling."""
        try:
            data = await provider.fetch_data(start_date, end_date)
            return {
                "provider": name,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
        except Exception as e:
            logger.error(f"Error fetching data from {name}: {e}")
            return {"error": str(e)}
    
    async def sync_specific_provider(self, provider_name: str, 
                                   user_id: str, 
                                   days_back: int = 1) -> Dict[str, Any]:
        """Sync data from a specific provider."""
        if provider_name not in self.providers:
            return {"error": f"Provider {provider_name} not registered"}
        
        provider = self.providers[provider_name]
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            data = await provider.fetch_data(start_date, end_date)
            
            sync_result = {
                "provider": provider_name,
                "user_id": user_id,
                "sync_timestamp": datetime.utcnow().isoformat(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "data": data,
                "status": "success"
            }
            
            # Update sync status
            self.sync_status[provider_name]["last_sync"] = datetime.utcnow()
            self.sync_status[provider_name]["status"] = "success"
            self.sync_status[provider_name]["error_count"] = 0
            
            return sync_result
            
        except Exception as e:
            error_msg = f"Sync failed for {provider_name}: {e}"
            logger.error(error_msg)
            
            # Update error status
            self.sync_status[provider_name]["status"] = "error"
            self.sync_status[provider_name]["last_error"] = str(e)
            self.sync_status[provider_name]["error_count"] += 1
            
            return {
                "provider": provider_name,
                "user_id": user_id,
                "sync_timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": error_msg
            }
    
    def get_sync_status(self) -> Dict[str, Dict[str, Any]]:
        """Get sync status for all providers."""
        return self.sync_status.copy()
    
    def get_provider_permissions(self) -> Dict[str, List[str]]:
        """Get required permissions for all providers."""
        permissions = {}
        for name, provider in self.providers.items():
            permissions[name] = provider.get_required_permissions()
        return permissions
    
    async def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old cached data."""
        # This would clean up any cached data older than retention_days
        # Implementation depends on where cached data is stored
        pass
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered providers."""
        info = {}
        for name, provider in self.providers.items():
            info[name] = {
                "class": provider.__class__.__name__,
                "permissions": provider.get_required_permissions(),
                "status": self.sync_status[name],
                "last_sync": self.sync_status[name]["last_sync"].isoformat() if self.sync_status[name]["last_sync"] else None
            }
        return info