"""Health data provider implementations."""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from .health_manager import HealthDataProvider

logger = logging.getLogger(__name__)

class FitbitProvider(HealthDataProvider):
    """Fitbit API integration."""
    
    def __init__(self, access_token: str, refresh_token: Optional[str] = None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.base_url = "https://api.fitbit.com/1/user/-"
        
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Fetch comprehensive Fitbit data."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with aiohttp.ClientSession() as session:
            # Fetch multiple data types in parallel
            tasks = [
                self._fetch_heart_rate(session, headers, start_date, end_date),
                self._fetch_sleep_data(session, headers, start_date, end_date),
                self._fetch_activity_data(session, headers, start_date, end_date),
                self._fetch_steps_data(session, headers, start_date, end_date),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "heart_rate": results[0] if not isinstance(results[0], Exception) else None,
                "sleep": results[1] if not isinstance(results[1], Exception) else None,
                "activity": results[2] if not isinstance(results[2], Exception) else None,
                "steps": results[3] if not isinstance(results[3], Exception) else None,
                "source": "fitbit",
                "timestamp": datetime.utcnow().isoformat(),
                "errors": [str(r) for r in results if isinstance(r, Exception)]
            }
    
    async def _fetch_heart_rate(self, session: aiohttp.ClientSession, 
                              headers: Dict[str, str], 
                              start_date: datetime, 
                              end_date: datetime) -> Optional[List[Dict]]:
        """Fetch heart rate data."""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/activities/heart/date/{start_str}/{end_str}.json"
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("activities-heart", [])
                else:
                    logger.error(f"Heart rate fetch failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Heart rate fetch error: {e}")
            return None
    
    async def _fetch_sleep_data(self, session: aiohttp.ClientSession, 
                              headers: Dict[str, str], 
                              start_date: datetime, 
                              end_date: datetime) -> Optional[List[Dict]]:
        """Fetch sleep data."""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/sleep/date/{start_str}/{end_str}.json"
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("sleep", [])
                else:
                    logger.error(f"Sleep data fetch failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Sleep data fetch error: {e}")
            return None
    
    async def _fetch_activity_data(self, session: aiohttp.ClientSession, 
                                 headers: Dict[str, str], 
                                 start_date: datetime, 
                                 end_date: datetime) -> Optional[List[Dict]]:
        """Fetch activity data."""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/activities/date/{start_str}/{end_str}.json"
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("activities", [])
                else:
                    logger.error(f"Activity data fetch failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Activity data fetch error: {e}")
            return None
    
    async def _fetch_steps_data(self, session: aiohttp.ClientSession, 
                              headers: Dict[str, str], 
                              start_date: datetime, 
                              end_date: datetime) -> Optional[List[Dict]]:
        """Fetch steps data."""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/activities/steps/date/{start_str}/{end_str}.json"
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("activities-steps", [])
                else:
                    logger.error(f"Steps data fetch failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Steps data fetch error: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Test Fitbit API connection."""
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/profile.json"
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Fitbit connection test failed: {e}")
            return False
    
    def get_required_permissions(self) -> List[str]:
        """Get required Fitbit permissions."""
        return [
            "activity",
            "heartrate", 
            "sleep",
            "profile"
        ]

class AppleHealthProvider(HealthDataProvider):
    """Apple HealthKit integration via Shortcuts."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Fetch Apple Health data via Shortcuts webhook."""
        
        request_data = {
            "action": "fetch_health_data",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data_types": [
                "step_count",
                "heart_rate",
                "sleep_analysis",
                "mindful_minutes",
                "active_energy",
                "resting_heart_rate"
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "source": "apple_health",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": data
                        }
                    else:
                        logger.error(f"Apple Health fetch failed: {response.status}")
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Apple Health fetch error: {e}")
            return {"error": str(e)}
    
    async def test_connection(self) -> bool:
        """Test Apple Health connection."""
        try:
            test_data = {
                "action": "test_connection",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Apple Health connection test failed: {e}")
            return False
    
    def get_required_permissions(self) -> List[str]:
        """Get required Apple Health permissions."""
        return [
            "HKQuantityTypeIdentifierStepCount",
            "HKQuantityTypeIdentifierHeartRate",
            "HKCategoryTypeIdentifierSleepAnalysis",
            "HKQuantityTypeIdentifierMindfulSessionDuration",
            "HKQuantityTypeIdentifierActiveEnergyBurned",
            "HKQuantityTypeIdentifierRestingHeartRate"
        ]

class WeatherProvider(HealthDataProvider):
    """Weather data integration (OpenWeatherMap)."""
    
    def __init__(self, api_key: str, location: tuple):
        self.api_key = api_key
        self.lat, self.lon = location
        
    async def fetch_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Fetch historical weather data."""
        weather_data = []
        
        # Calculate days to fetch
        days_diff = (end_date - start_date).days
        
        async with aiohttp.ClientSession() as session:
            for i in range(days_diff + 1):
                target_date = start_date + timedelta(days=i)
                timestamp = int(target_date.timestamp())
                
                url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
                params = {
                    "lat": self.lat,
                    "lon": self.lon,
                    "dt": timestamp,
                    "appid": self.api_key,
                    "units": "metric"
                }
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Extract relevant weather data
                            weather_point = {
                                "date": target_date.isoformat(),
                                "pressure": data["current"]["pressure"],
                                "humidity": data["current"]["humidity"],
                                "temperature": data["current"]["temp"],
                                "weather_desc": data["current"]["weather"][0]["description"],
                                "wind_speed": data["current"]["wind_speed"],
                                "uv_index": data["current"].get("uvi", 0)
                            }
                            
                            weather_data.append(weather_point)
                        else:
                            logger.error(f"Weather fetch failed for {target_date}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Weather fetch error for {target_date}: {e}")
                    
                # Rate limiting
                await asyncio.sleep(0.1)
        
        # Calculate pressure changes
        for i in range(1, len(weather_data)):
            pressure_change = weather_data[i-1]["pressure"] - weather_data[i]["pressure"]
            weather_data[i]["pressure_change"] = pressure_change
            
        return {
            "source": "openweathermap",
            "timestamp": datetime.utcnow().isoformat(),
            "data": weather_data,
            "location": {"lat": self.lat, "lon": self.lon}
        }
    
    async def test_connection(self) -> bool:
        """Test weather API connection."""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": self.lat,
                "lon": self.lon,
                "appid": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Weather connection test failed: {e}")
            return False
    
    def get_required_permissions(self) -> List[str]:
        """Get required weather permissions."""
        return ["api_access"]