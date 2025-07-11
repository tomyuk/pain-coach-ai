"""AI chat API routes."""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import asyncio
import logging
from datetime import datetime

from core.ai.mlx_engine import PainCoachMLXEngine
from core.data.database import DatabaseManager
from core.data.schemas import ConversationSchema, ConversationType
from core.privacy.audit_logger import AuditLogger
from core.privacy.encryption_manager import EncryptionManager
from core.performance.profiler import profile
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    conversation_type: ConversationType = ConversationType.GENERAL
    pain_context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str
    processing_time_ms: int
    confidence_score: Optional[float] = None
    suggestions: Optional[list] = None

@router.post("/chat", response_model=ChatResponse)
@profile("ai_chat")
async def chat_with_ai(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    ai_engine: PainCoachMLXEngine = Depends(),
    db: DatabaseManager = Depends(),
    audit_logger: AuditLogger = Depends(),
    encryption_manager: EncryptionManager = Depends()
):
    """Chat with AI assistant."""
    try:
        start_time = datetime.utcnow()
        
        # Get recent pain context if not provided
        if not chat_request.pain_context:
            recent_records = await db.get_pain_records(current_user, limit=5, days_back=7)
            if recent_records:
                latest_record = recent_records[0]
                chat_request.pain_context = {
                    "current_pain": latest_record.pain_level,
                    "recent_pattern": "stable",  # Could be calculated
                    "main_locations": latest_record.pain_locations
                }
        
        # Generate AI response
        response_text = ""
        async for token in ai_engine.generate_response(
            chat_request.message,
            pain_context=chat_request.pain_context
        ):
            response_text += token
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Create conversation record
        conversation_data = ConversationSchema(
            user_id=current_user,
            session_id=chat_request.session_id,
            user_message=chat_request.message,
            ai_response=response_text,
            conversation_type=chat_request.conversation_type,
            pain_context=chat_request.pain_context,
            model_used="ELYZA-japanese-Llama-3-8B-Instruct",
            processing_time_ms=int(processing_time),
            confidence_score=0.85,  # Placeholder
            timestamp=end_time
        )
        
        # Save conversation in background
        background_tasks.add_task(
            save_conversation,
            conversation_data,
            current_user,
            db,
            encryption_manager,
            audit_logger
        )
        
        return ChatResponse(
            response=response_text,
            conversation_id="temp_id",  # Will be updated when saved
            processing_time_ms=int(processing_time),
            confidence_score=0.85,
            suggestions=generate_suggestions(chat_request.conversation_type)
        )
        
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request"
        )

async def save_conversation(
    conversation_data: ConversationSchema,
    user_id: str,
    db: DatabaseManager,
    encryption_manager: EncryptionManager,
    audit_logger: AuditLogger
):
    """Save conversation to database (background task)."""
    try:
        # Encrypt sensitive data
        encrypted_user_message = encryption_manager.encrypt_data(
            conversation_data.user_message, user_id
        )
        encrypted_ai_response = encryption_manager.encrypt_data(
            conversation_data.ai_response, user_id
        )
        
        # Create conversation record
        conversation_dict = conversation_data.dict()
        conversation_dict["user_message_encrypted"] = encrypted_user_message
        conversation_dict["ai_response_encrypted"] = encrypted_ai_response
        conversation_dict["retention_until"] = datetime.utcnow() + timedelta(days=365)
        
        # Remove plaintext messages
        del conversation_dict["user_message"]
        del conversation_dict["ai_response"]
        
        # Save to database
        conversation = await db.create_conversation(conversation_dict)
        
        # Log conversation
        await audit_logger.log_event(
            "ai_conversation",
            user_id,
            {
                "conversation_id": str(conversation.id),
                "type": conversation_data.conversation_type.value,
                "processing_time_ms": conversation_data.processing_time_ms
            }
        )
        
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")

@router.get("/chat/stream")
@profile("ai_chat_stream")
async def stream_chat_with_ai(
    message: str,
    current_user: str = Depends(get_current_user),
    ai_engine: PainCoachMLXEngine = Depends(),
    db: DatabaseManager = Depends(),
    pain_context: Optional[str] = None
):
    """Stream chat with AI assistant."""
    try:
        # Parse pain context if provided
        parsed_context = None
        if pain_context:
            try:
                parsed_context = json.loads(pain_context)
            except json.JSONDecodeError:
                pass
        
        # Get recent pain context if not provided
        if not parsed_context:
            recent_records = await db.get_pain_records(current_user, limit=5, days_back=7)
            if recent_records:
                latest_record = recent_records[0]
                parsed_context = {
                    "current_pain": latest_record.pain_level,
                    "recent_pattern": "stable",
                    "main_locations": latest_record.pain_locations
                }
        
        async def generate_stream():
            """Generate streaming response."""
            try:
                async for token in ai_engine.generate_response(
                    message,
                    pain_context=parsed_context
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Stream chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start streaming chat"
        )

@router.get("/conversations", response_model=list)
@profile("get_conversations")
async def get_conversations(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    encryption_manager: EncryptionManager = Depends(),
    limit: int = 10
):
    """Get recent conversations."""
    try:
        # Get conversations from database
        conversations = await db.get_recent_conversations(current_user, limit)
        
        # Decrypt and format conversations
        formatted_conversations = []
        for conv in conversations:
            try:
                user_message = encryption_manager.decrypt_data(
                    conv.user_message_encrypted, current_user
                ) if conv.user_message_encrypted else ""
                
                ai_response = encryption_manager.decrypt_data(
                    conv.ai_response_encrypted, current_user
                ) if conv.ai_response_encrypted else ""
                
                formatted_conversations.append({
                    "id": str(conv.id),
                    "session_id": str(conv.session_id) if conv.session_id else None,
                    "timestamp": conv.timestamp.isoformat(),
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "conversation_type": conv.conversation_type,
                    "processing_time_ms": conv.processing_time_ms,
                    "confidence_score": conv.confidence_score
                })
                
            except Exception as e:
                logger.error(f"Error decrypting conversation {conv.id}: {e}")
                continue
        
        return formatted_conversations
        
    except Exception as e:
        logger.error(f"Conversations retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )

@router.post("/analyze-mood", response_model=dict)
@profile("analyze_mood")
async def analyze_mood(
    text: str,
    current_user: str = Depends(get_current_user),
    audit_logger: AuditLogger = Depends()
):
    """Analyze mood from text."""
    try:
        # Simple mood analysis (in practice, use NLP models)
        mood_keywords = {
            "happy": ["happy", "joy", "excited", "good", "great", "wonderful"],
            "sad": ["sad", "depressed", "down", "low", "terrible", "awful"],
            "anxious": ["anxious", "worried", "nervous", "scared", "afraid"],
            "frustrated": ["frustrated", "angry", "annoyed", "irritated"],
            "neutral": []
        }
        
        text_lower = text.lower()
        mood_scores = {}
        
        for mood, keywords in mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score
        
        # Determine primary mood
        primary_mood = "neutral"
        if mood_scores:
            primary_mood = max(mood_scores, key=mood_scores.get)
        
        # Log mood analysis
        await audit_logger.log_event(
            "mood_analysis",
            current_user,
            {"detected_mood": primary_mood, "text_length": len(text)}
        )
        
        return {
            "primary_mood": primary_mood,
            "mood_scores": mood_scores,
            "confidence": 0.7,  # Placeholder
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Mood analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze mood"
        )

@router.post("/emergency-check", response_model=dict)
@profile("emergency_check")
async def emergency_check(
    message: str,
    current_user: str = Depends(get_current_user),
    audit_logger: AuditLogger = Depends()
):
    """Check for emergency situations in message."""
    try:
        # Simple emergency detection
        emergency_keywords = [
            "suicide", "kill myself", "end it all", "can't go on",
            "emergency", "911", "help me", "crisis"
        ]
        
        message_lower = message.lower()
        emergency_detected = any(keyword in message_lower for keyword in emergency_keywords)
        
        if emergency_detected:
            # Log emergency detection
            await audit_logger.log_event(
                "emergency_detected",
                current_user,
                {"message_length": len(message), "immediate_action": "crisis_resources_provided"}
            )
            
            return {
                "emergency_detected": True,
                "severity": "high",
                "crisis_resources": [
                    {
                        "name": "National Suicide Prevention Lifeline",
                        "phone": "988",
                        "description": "24/7 crisis support"
                    },
                    {
                        "name": "Crisis Text Line",
                        "text": "Text HOME to 741741",
                        "description": "24/7 text-based crisis support"
                    }
                ],
                "message": "If you are in crisis, please reach out for help immediately. Your life matters."
            }
        
        return {
            "emergency_detected": False,
            "severity": "low",
            "message": "No emergency indicators detected."
        }
        
    except Exception as e:
        logger.error(f"Emergency check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform emergency check"
        )

def generate_suggestions(conversation_type: ConversationType) -> list:
    """Generate conversation suggestions based on type."""
    suggestions = {
        ConversationType.PAIN_CHECK: [
            "How has your pain been today?",
            "What activities have you tried for pain relief?",
            "How is your sleep quality affecting your pain?"
        ],
        ConversationType.COUNSELING: [
            "Tell me about your feelings regarding your pain",
            "What coping strategies work best for you?",
            "How can I support you better today?"
        ],
        ConversationType.ANALYSIS: [
            "Show me my pain patterns",
            "What factors might be affecting my pain?",
            "How effective are my current treatments?"
        ],
        ConversationType.GENERAL: [
            "How are you feeling today?",
            "What would you like to talk about?",
            "Is there anything I can help you with?"
        ]
    }
    
    return suggestions.get(conversation_type, suggestions[ConversationType.GENERAL])