"""System monitoring and alerting for Pain Coach AI Pascal."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    acknowledged: bool = False
    resolved: bool = False

class SystemMonitor:
    """System monitoring and alerting."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.is_running = False
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Default alert thresholds
        self._setup_default_thresholds()
        
    def _setup_default_thresholds(self):
        """Set up default alert thresholds."""
        self.alert_thresholds = {
            "cpu_percent": {
                "medium": 80.0,
                "high": 90.0,
                "critical": 95.0
            },
            "memory_percent": {
                "medium": 85.0,
                "high": 90.0,
                "critical": 95.0
            },
            "disk_percent": {
                "medium": 80.0,
                "high": 90.0,
                "critical": 95.0
            },
            "temperature": {
                "medium": 70.0,
                "high": 80.0,
                "critical": 90.0
            },
            "response_time_ms": {
                "medium": 1000.0,
                "high": 2000.0,
                "critical": 5000.0
            },
            "error_rate": {
                "medium": 0.05,  # 5%
                "high": 0.10,    # 10%
                "critical": 0.20  # 20%
            }
        }
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric_name: str, severity: AlertSeverity, threshold: float):
        """Set custom threshold for a metric."""
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        
        self.alert_thresholds[metric_name][severity.value] = threshold
        logger.info(f"Set {severity.value} threshold for {metric_name}: {threshold}")
    
    async def start_monitoring(self):
        """Start system monitoring."""
        if self.is_running:
            logger.warning("System monitoring is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_system_health(self):
        """Check system health metrics."""
        try:
            # Get system metrics
            from .m3_optimizer import M3MaxOptimizer
            optimizer = M3MaxOptimizer()
            metrics = await optimizer.monitor_system_resources()
            
            # Store metrics history
            timestamp = datetime.utcnow()
            self._store_metrics(metrics, timestamp)
            
            # Check for alerts
            await self._check_alerts(metrics, timestamp)
            
        except Exception as e:
            logger.error(f"System health check error: {e}")
    
    def _store_metrics(self, metrics: Dict[str, Any], timestamp: datetime):
        """Store metrics in history."""
        metric_point = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics
        }
        
        # Store in history
        date_key = timestamp.strftime("%Y-%m-%d")
        if date_key not in self.metrics_history:
            self.metrics_history[date_key] = []
        
        self.metrics_history[date_key].append(metric_point)
        
        # Keep only last 7 days of metrics
        cutoff_date = timestamp - timedelta(days=7)
        cutoff_key = cutoff_date.strftime("%Y-%m-%d")
        
        for date_key in list(self.metrics_history.keys()):
            if date_key < cutoff_key:
                del self.metrics_history[date_key]
    
    async def _check_alerts(self, metrics: Dict[str, Any], timestamp: datetime):
        """Check metrics against alert thresholds."""
        
        # CPU usage
        cpu_percent = metrics.get("cpu", {}).get("percent", 0)
        await self._check_metric_alert("cpu_percent", cpu_percent, timestamp)
        
        # Memory usage
        memory_percent = metrics.get("memory", {}).get("percent", 0)
        await self._check_metric_alert("memory_percent", memory_percent, timestamp)
        
        # Disk usage
        disk_percent = metrics.get("disk", {}).get("percent", 0)
        await self._check_metric_alert("disk_percent", disk_percent, timestamp)
        
        # Temperature
        temperature = metrics.get("temperature", 0)
        await self._check_metric_alert("temperature", temperature, timestamp)
        
        # GPU metrics
        gpu_metrics = metrics.get("gpu", {})
        if gpu_metrics and "utilization" in gpu_metrics:
            await self._check_metric_alert("gpu_utilization", gpu_metrics["utilization"] * 100, timestamp)
    
    async def _check_metric_alert(self, metric_name: str, value: float, timestamp: datetime):
        """Check if a metric exceeds thresholds."""
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        
        # Check in order of severity
        severity = None
        threshold = None
        
        if "critical" in thresholds and value >= thresholds["critical"]:
            severity = AlertSeverity.CRITICAL
            threshold = thresholds["critical"]
        elif "high" in thresholds and value >= thresholds["high"]:
            severity = AlertSeverity.HIGH
            threshold = thresholds["high"]
        elif "medium" in thresholds and value >= thresholds["medium"]:
            severity = AlertSeverity.MEDIUM
            threshold = thresholds["medium"]
        
        if severity:
            await self._create_alert(metric_name, value, threshold, severity, timestamp)
    
    async def _create_alert(self, metric_name: str, value: float, threshold: float, 
                          severity: AlertSeverity, timestamp: datetime):
        """Create a new alert."""
        
        # Check if similar alert already exists and is not resolved
        for alert in self.alerts:
            if (alert.metric_name == metric_name and 
                alert.severity == severity and 
                not alert.resolved and
                timestamp - alert.timestamp < timedelta(minutes=5)):
                # Don't create duplicate alerts within 5 minutes
                return
        
        # Create alert
        alert_id = f"{metric_name}_{severity.value}_{int(timestamp.timestamp())}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=f"{metric_name.title()} {severity.value.title()} Alert",
            message=self._generate_alert_message(metric_name, value, threshold, severity),
            timestamp=timestamp,
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert.title} - {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _generate_alert_message(self, metric_name: str, value: float, 
                              threshold: float, severity: AlertSeverity) -> str:
        """Generate alert message."""
        messages = {
            "cpu_percent": f"CPU usage is {value:.1f}% (threshold: {threshold:.1f}%)",
            "memory_percent": f"Memory usage is {value:.1f}% (threshold: {threshold:.1f}%)",
            "disk_percent": f"Disk usage is {value:.1f}% (threshold: {threshold:.1f}%)",
            "temperature": f"Temperature is {value:.1f}°C (threshold: {threshold:.1f}°C)",
            "response_time_ms": f"Response time is {value:.1f}ms (threshold: {threshold:.1f}ms)",
            "error_rate": f"Error rate is {value:.1%} (threshold: {threshold:.1%})"
        }
        
        base_message = messages.get(metric_name, f"{metric_name} is {value} (threshold: {threshold})")
        
        if severity == AlertSeverity.CRITICAL:
            return f"CRITICAL: {base_message}. Immediate action required."
        elif severity == AlertSeverity.HIGH:
            return f"HIGH: {base_message}. Action required soon."
        elif severity == AlertSeverity.MEDIUM:
            return f"MEDIUM: {base_message}. Monitor closely."
        else:
            return f"LOW: {base_message}."
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_recent_alerts(24)
        
        # Calculate health score
        health_score = 100
        for alert in active_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                health_score -= 30
            elif alert.severity == AlertSeverity.HIGH:
                health_score -= 20
            elif alert.severity == AlertSeverity.MEDIUM:
                health_score -= 10
            else:
                health_score -= 5
        
        health_score = max(0, health_score)
        
        # Determine overall status
        if health_score >= 90:
            overall_status = "excellent"
        elif health_score >= 70:
            overall_status = "good"
        elif health_score >= 50:
            overall_status = "fair"
        else:
            overall_status = "poor"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "health_score": health_score,
            "active_alerts": len(active_alerts),
            "recent_alerts": len(recent_alerts),
            "alerts_by_severity": {
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
                "low": len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            },
            "monitoring_enabled": self.is_running
        }
    
    def get_metrics_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for a specific metric."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        history = []
        for date_key, metrics_list in self.metrics_history.items():
            for metric_point in metrics_list:
                timestamp = datetime.fromisoformat(metric_point["timestamp"])
                if timestamp > cutoff_time:
                    # Extract the specific metric
                    value = self._extract_metric_value(metric_point["metrics"], metric_name)
                    if value is not None:
                        history.append({
                            "timestamp": metric_point["timestamp"],
                            "value": value
                        })
        
        return sorted(history, key=lambda x: x["timestamp"])
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from metrics dict."""
        if metric_name == "cpu_percent":
            return metrics.get("cpu", {}).get("percent")
        elif metric_name == "memory_percent":
            return metrics.get("memory", {}).get("percent")
        elif metric_name == "disk_percent":
            return metrics.get("disk", {}).get("percent")
        elif metric_name == "temperature":
            return metrics.get("temperature")
        elif metric_name == "gpu_utilization":
            gpu_metrics = metrics.get("gpu", {})
            if gpu_metrics and "utilization" in gpu_metrics:
                return gpu_metrics["utilization"] * 100
        
        return None
    
    def export_alerts(self, filename: str):
        """Export alerts to JSON file."""
        try:
            alerts_data = []
            for alert in self.alerts:
                alerts_data.append({
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "acknowledged": alert.acknowledged,
                    "resolved": alert.resolved
                })
            
            with open(filename, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            logger.info(f"Alerts exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")

# Global system monitor instance
system_monitor = SystemMonitor()