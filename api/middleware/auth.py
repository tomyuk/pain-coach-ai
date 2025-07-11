"""Authentication middleware."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication."""
        
        # Skip auth for public endpoints
        public_endpoints = [
            "/health",
            "/",
            "/api/docs",
            "/api/redoc",
            "/api/auth/register",
            "/api/auth/token"
        ]
        
        if request.url.path in public_endpoints:
            response = await call_next(request)
            return response
        
        # For now, just pass through
        # In practice, you'd verify JWT tokens here
        response = await call_next(request)
        return response