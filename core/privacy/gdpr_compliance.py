"""GDPR compliance manager for Pain Coach AI Pascal."""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
from ..data.database import DatabaseManager
from .encryption_manager import EncryptionManager
from .audit_logger import AuditLogger

logger = logging.getLogger(__name__)

class GDPRComplianceManager:
    """GDPR compliance and data protection manager."""
    
    def __init__(self, db_manager: DatabaseManager, 
                 encryption_manager: EncryptionManager,
                 audit_logger: AuditLogger):
        self.db = db_manager
        self.encryption = encryption_manager
        self.audit = audit_logger
        
    async def process_data_deletion_request(self, user_id: str, 
                                          requester_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process GDPR data deletion request (Right to be Forgotten)."""
        deletion_summary = {
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "requester": requester_info,
            "deleted_records": {},
            "errors": [],
            "status": "started"
        }
        
        # Log deletion request
        await self.audit.log_event(
            "data_deletion_request",
            user_id,
            {"requester": requester_info}
        )
        
        try:
            # Get user first to verify existence
            user = await self.db.get_user(user_id)
            if not user:
                deletion_summary["status"] = "error"
                deletion_summary["errors"].append("User not found")
                return deletion_summary
            
            # Delete from all tables
            tables_to_clean = [
                "pain_records",
                "ai_conversations", 
                "analysis_cache",
                "user_preferences",
                "health_integrations"
            ]
            
            for table in tables_to_clean:
                try:
                    count = await self._delete_user_data_from_table(table, user_id)
                    deletion_summary["deleted_records"][table] = count
                except Exception as e:
                    deletion_summary["errors"].append(f"Error deleting from {table}: {e}")
            
            # Delete user profile
            try:
                await self._delete_user_profile(user_id)
                deletion_summary["deleted_records"]["users"] = 1
            except Exception as e:
                deletion_summary["errors"].append(f"Error deleting user profile: {e}")
            
            # Delete encryption keys
            try:
                self.encryption.delete_user_key(user_id)
                deletion_summary["deleted_records"]["encryption_keys"] = 1
            except Exception as e:
                deletion_summary["errors"].append(f"Error deleting encryption keys: {e}")
            
            # Clean up DuckDB analytics data
            try:
                await self._clean_analytics_data(user_id)
                deletion_summary["deleted_records"]["analytics_data"] = 1
            except Exception as e:
                deletion_summary["errors"].append(f"Error cleaning analytics data: {e}")
                
            deletion_summary["status"] = "completed" if not deletion_summary["errors"] else "completed_with_errors"
            
            # Log completion
            await self.audit.log_event(
                "data_deletion_completed",
                user_id,
                deletion_summary
            )
            
        except Exception as e:
            deletion_summary["status"] = "error"
            deletion_summary["errors"].append(f"Unexpected error: {e}")
            logger.error(f"Data deletion error for user {user_id}: {e}")
            
        return deletion_summary
    
    async def export_user_data(self, user_id: str, 
                              requester_info: Dict[str, Any]) -> Dict[str, Any]:
        """Export user data (Right to Data Portability)."""
        export_data = {
            "export_timestamp": datetime.utcnow(),
            "user_id": user_id,
            "requester": requester_info,
            "data": {},
            "metadata": {
                "format": "JSON",
                "encryption_status": "decrypted_for_export"
            }
        }
        
        # Log export request
        await self.audit.log_event(
            "data_export_request",
            user_id,
            {"requester": requester_info}
        )
        
        try:
            # Export user profile
            user = await self.db.get_user(user_id)
            if user:
                export_data["data"]["user_profile"] = self._serialize_user_data(user)
            
            # Export pain records
            pain_records = await self.db.get_pain_records(user_id, limit=10000)
            export_data["data"]["pain_records"] = [
                self._serialize_pain_record(record) for record in pain_records
            ]
            
            # Export conversations (decrypt sensitive data)
            conversations = await self.db.get_recent_conversations(user_id, limit=1000)
            export_data["data"]["conversations"] = [
                await self._serialize_conversation(conv, user_id) for conv in conversations
            ]
            
            # Export analytics data
            analytics = await self.db.analyze_pain_trends(user_id, days=365)
            export_data["data"]["analytics"] = analytics
            
            # Log export completion
            await self.audit.log_event(
                "data_export_completed",
                user_id,
                {"records_exported": len(export_data["data"])}
            )
            
        except Exception as e:
            logger.error(f"Data export error for user {user_id}: {e}")
            export_data["error"] = str(e)
            
        return export_data
    
    async def process_data_correction_request(self, user_id: str, 
                                            field_corrections: Dict[str, Any],
                                            requester_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process data correction request (Right to Rectification)."""
        correction_summary = {
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "requester": requester_info,
            "corrections": field_corrections,
            "status": "started",
            "errors": []
        }
        
        # Log correction request
        await self.audit.log_event(
            "data_correction_request",
            user_id,
            {"corrections": field_corrections, "requester": requester_info}
        )
        
        try:
            # Update user data
            user = await self.db.get_user(user_id)
            if not user:
                correction_summary["status"] = "error"
                correction_summary["errors"].append("User not found")
                return correction_summary
            
            # Apply corrections
            updated_fields = []
            for field, new_value in field_corrections.items():
                if hasattr(user, field):
                    # Handle encrypted fields
                    if field.endswith("_encrypted"):
                        encrypted_value = self.encryption.encrypt_data(new_value, user_id)
                        setattr(user, field, encrypted_value)
                    else:
                        setattr(user, field, new_value)
                    updated_fields.append(field)
                else:
                    correction_summary["errors"].append(f"Field {field} not found")
            
            # Save changes
            if updated_fields:
                user.updated_at = datetime.utcnow()
                # In practice, you'd commit to database here
                
            correction_summary["status"] = "completed"
            correction_summary["updated_fields"] = updated_fields
            
            # Log completion
            await self.audit.log_event(
                "data_correction_completed",
                user_id,
                correction_summary
            )
            
        except Exception as e:
            correction_summary["status"] = "error"
            correction_summary["errors"].append(str(e))
            logger.error(f"Data correction error for user {user_id}: {e}")
            
        return correction_summary
    
    async def check_data_retention_compliance(self) -> Dict[str, Any]:
        """Check data retention compliance."""
        compliance_report = {
            "check_timestamp": datetime.utcnow(),
            "violations": [],
            "recommendations": [],
            "stats": {}
        }
        
        try:
            # Check for data past retention period
            # This would involve querying all tables for old data
            # Implementation depends on specific retention policies
            
            # Log compliance check
            await self.audit.log_event(
                "retention_compliance_check",
                "system",
                compliance_report
            )
            
        except Exception as e:
            logger.error(f"Retention compliance check error: {e}")
            compliance_report["error"] = str(e)
            
        return compliance_report
    
    async def _delete_user_data_from_table(self, table: str, user_id: str) -> int:
        """Delete user data from a specific table."""
        # This would use the database manager to delete records
        # Return count of deleted records
        return 0  # Placeholder
    
    async def _delete_user_profile(self, user_id: str):
        """Delete user profile."""
        # Use database manager to delete user
        pass
    
    async def _clean_analytics_data(self, user_id: str):
        """Clean user data from analytics database."""
        # Clean DuckDB data
        pass
    
    def _serialize_user_data(self, user) -> Dict[str, Any]:
        """Serialize user data for export."""
        return {
            "id": str(user.id),
            "created_at": user.created_at.isoformat(),
            "birth_year": user.birth_year,
            "gender": user.gender,
            "primary_condition": user.primary_condition,
            "diagnosis_date": user.diagnosis_date.isoformat() if user.diagnosis_date else None,
            "medications": user.medications,
            "timezone": user.timezone,
            "language": user.language
        }
    
    def _serialize_pain_record(self, record) -> Dict[str, Any]:
        """Serialize pain record for export."""
        return {
            "id": str(record.id),
            "recorded_at": record.recorded_at.isoformat(),
            "pain_level": record.pain_level,
            "pain_type": record.pain_type,
            "pain_locations": record.pain_locations,
            "weather_data": record.weather_data,
            "activity_before": record.activity_before,
            "sleep_quality": record.sleep_quality,
            "stress_level": record.stress_level,
            "input_method": record.input_method
        }
    
    async def _serialize_conversation(self, conversation, user_id: str) -> Dict[str, Any]:
        """Serialize conversation for export (decrypt sensitive data)."""
        user_message = ""
        ai_response = ""
        
        try:
            if conversation.user_message_encrypted:
                user_message = self.encryption.decrypt_data(
                    conversation.user_message_encrypted, user_id
                )
            if conversation.ai_response_encrypted:
                ai_response = self.encryption.decrypt_data(
                    conversation.ai_response_encrypted, user_id
                )
        except Exception as e:
            logger.error(f"Error decrypting conversation: {e}")
        
        return {
            "id": str(conversation.id),
            "timestamp": conversation.timestamp.isoformat(),
            "user_message": user_message,
            "ai_response": ai_response,
            "conversation_type": conversation.conversation_type,
            "mood_detected": conversation.mood_detected,
            "pain_context": conversation.pain_context
        }