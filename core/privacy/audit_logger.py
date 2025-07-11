"""Audit logging for GDPR compliance and security monitoring."""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import hashlib
import hmac
import os

class AuditEventType(str, Enum):
    """Audit event types."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    ENCRYPTION_KEY_ROTATION = "encryption_key_rotation"
    GDPR_REQUEST = "gdpr_request"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"

class AuditLogger:
    """Secure audit logger for compliance and security monitoring."""
    
    def __init__(self, log_file: str = "audit.log", 
                 integrity_key: Optional[str] = None):
        self.log_file = log_file
        self.integrity_key = integrity_key or os.getenv("AUDIT_INTEGRITY_KEY", "default_key")
        
        # Set up logging
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Ensure log file exists
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        
    async def log_event(self, event_type: str, user_id: str, 
                       details: Dict[str, Any], 
                       severity: str = "INFO") -> str:
        """Log an audit event with integrity protection."""
        
        # Create audit event
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
            "severity": severity,
            "source": "pain_coach_ai",
            "version": "1.0"
        }
        
        # Add integrity signature
        event_json = json.dumps(event, sort_keys=True)
        signature = self._generate_integrity_signature(event_json)
        event["integrity_signature"] = signature
        
        # Log the event
        final_json = json.dumps(event)
        self.logger.info(final_json)
        
        # Return event ID (hash of event)
        event_id = hashlib.sha256(final_json.encode()).hexdigest()[:16]
        return event_id
    
    async def log_data_access(self, user_id: str, resource: str, 
                            access_type: str, details: Optional[Dict] = None):
        """Log data access event."""
        event_details = {
            "resource": resource,
            "access_type": access_type,
            "ip_address": details.get("ip_address") if details else None,
            "user_agent": details.get("user_agent") if details else None
        }
        
        return await self.log_event(
            AuditEventType.DATA_ACCESS,
            user_id,
            event_details
        )
    
    async def log_data_modification(self, user_id: str, resource: str, 
                                  changes: Dict[str, Any], 
                                  details: Optional[Dict] = None):
        """Log data modification event."""
        event_details = {
            "resource": resource,
            "changes": changes,
            "ip_address": details.get("ip_address") if details else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.log_event(
            AuditEventType.DATA_MODIFICATION,
            user_id,
            event_details
        )
    
    async def log_security_violation(self, user_id: str, violation_type: str, 
                                   details: Dict[str, Any]):
        """Log security violation."""
        event_details = {
            "violation_type": violation_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.log_event(
            AuditEventType.SECURITY_VIOLATION,
            user_id,
            event_details,
            severity="ERROR"
        )
    
    async def log_gdpr_request(self, user_id: str, request_type: str, 
                             details: Dict[str, Any]):
        """Log GDPR request."""
        event_details = {
            "request_type": request_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.log_event(
            AuditEventType.GDPR_REQUEST,
            user_id,
            event_details,
            severity="INFO"
        )
    
    def _generate_integrity_signature(self, event_json: str) -> str:
        """Generate HMAC signature for event integrity."""
        return hmac.new(
            self.integrity_key.encode(),
            event_json.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_event_integrity(self, event: Dict[str, Any]) -> bool:
        """Verify event integrity."""
        if "integrity_signature" not in event:
            return False
        
        # Extract signature
        signature = event.pop("integrity_signature")
        
        # Recreate event JSON
        event_json = json.dumps(event, sort_keys=True)
        
        # Verify signature
        expected_signature = self._generate_integrity_signature(event_json)
        
        # Add signature back
        event["integrity_signature"] = signature
        
        return hmac.compare_digest(signature, expected_signature)
    
    async def get_audit_trail(self, user_id: str, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            event_types: Optional[list] = None) -> list:
        """Get audit trail for a user."""
        audit_events = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if "audit" in line:
                        try:
                            # Extract JSON from log line
                            json_part = line.split(" - INFO - ")[1]
                            event = json.loads(json_part)
                            
                            # Filter by user
                            if event.get("user_id") == user_id:
                                # Filter by date range
                                if start_date or end_date:
                                    event_time = datetime.fromisoformat(event["timestamp"])
                                    if start_date and event_time < start_date:
                                        continue
                                    if end_date and event_time > end_date:
                                        continue
                                
                                # Filter by event types
                                if event_types and event.get("event_type") not in event_types:
                                    continue
                                
                                audit_events.append(event)
                                
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                            
        except FileNotFoundError:
            pass
        
        return audit_events
    
    async def generate_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate compliance report."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        report = {
            "report_date": end_date.isoformat(),
            "period_days": days,
            "event_summary": {},
            "security_events": [],
            "gdpr_requests": [],
            "data_access_summary": {}
        }
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if "audit" in line:
                        try:
                            json_part = line.split(" - INFO - ")[1]
                            event = json.loads(json_part)
                            
                            event_time = datetime.fromisoformat(event["timestamp"])
                            if start_date <= event_time <= end_date:
                                # Count events by type
                                event_type = event.get("event_type", "unknown")
                                report["event_summary"][event_type] = report["event_summary"].get(event_type, 0) + 1
                                
                                # Collect security events
                                if event.get("severity") == "ERROR":
                                    report["security_events"].append(event)
                                
                                # Collect GDPR requests
                                if event_type == AuditEventType.GDPR_REQUEST:
                                    report["gdpr_requests"].append(event)
                                
                        except (json.JSONDecodeError, KeyError):
                            continue
                            
        except FileNotFoundError:
            pass
        
        return report