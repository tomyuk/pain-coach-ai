"""Authentication routes."""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import jwt
from datetime import datetime, timedelta
import logging

from core.data.database import DatabaseManager
from core.data.schemas import UserSchema
from core.privacy.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# JWT settings
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class TokenRequest(BaseModel):
    """Token request model."""
    username: str
    password: str

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int
    user_id: str

class UserRegistration(BaseModel):
    """User registration model."""
    username: str
    password: str
    email: Optional[str] = None
    name: Optional[str] = None
    birth_year: Optional[int] = None
    gender: Optional[str] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=dict)
async def register_user(
    user_data: UserRegistration,
    db: DatabaseManager = Depends(),
    audit_logger: AuditLogger = Depends()
):
    """Register a new user."""
    try:
        # Check if user already exists
        # In practice, you'd check against your user database
        
        # Create user profile
        user_profile = UserSchema(
            name=user_data.name,
            birth_year=user_data.birth_year,
            gender=user_data.gender,
            primary_condition=None,
            diagnosis_date=None,
            medications=None
        )
        
        # Create user in database
        user = await db.create_user(user_profile.dict())
        
        # Log registration
        await audit_logger.log_event(
            "user_registration",
            str(user.id),
            {"username": user_data.username, "email": user_data.email}
        )
        
        return {
            "message": "User registered successfully",
            "user_id": str(user.id)
        }
        
    except Exception as e:
        logger.error(f"User registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    token_request: TokenRequest,
    audit_logger: AuditLogger = Depends()
):
    """Login and get access token."""
    try:
        # In practice, verify credentials against database
        # For demo purposes, we'll use a simple check
        if token_request.username == "demo" and token_request.password == "demo":
            user_id = "demo_user_id"
            
            # Create access token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user_id}, expires_delta=access_token_expires
            )
            
            # Log login
            await audit_logger.log_event(
                "user_login",
                user_id,
                {"username": token_request.username, "login_method": "password"}
            )
            
            return TokenResponse(
                access_token=access_token,
                token_type="bearer",
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                user_id=user_id
            )
        else:
            # Log failed login attempt
            await audit_logger.log_event(
                "login_failed",
                "unknown",
                {"username": token_request.username, "reason": "invalid_credentials"}
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/logout")
async def logout(
    current_user: str = Depends(get_current_user),
    audit_logger: AuditLogger = Depends()
):
    """Logout user."""
    try:
        # Log logout
        await audit_logger.log_event(
            "user_logout",
            current_user,
            {"logout_method": "explicit"}
        )
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/profile")
async def get_user_profile(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends()
):
    """Get user profile."""
    try:
        user = await db.get_user(current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Return user profile (without sensitive data)
        return {
            "user_id": str(user.id),
            "birth_year": user.birth_year,
            "gender": user.gender,
            "primary_condition": user.primary_condition,
            "diagnosis_date": user.diagnosis_date.isoformat() if user.diagnosis_date else None,
            "language": user.language,
            "timezone": user.timezone,
            "created_at": user.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )

@router.put("/profile")
async def update_user_profile(
    profile_data: UserSchema,
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    audit_logger: AuditLogger = Depends()
):
    """Update user profile."""
    try:
        # Get existing user
        user = await db.get_user(current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update user profile
        # In practice, you'd update the user in the database
        # For now, we'll just log the update
        
        # Log profile update
        await audit_logger.log_data_modification(
            current_user,
            "user_profile",
            profile_data.dict(exclude_none=True)
        )
        
        return {"message": "Profile updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )