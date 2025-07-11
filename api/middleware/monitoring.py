"""Monitoring middleware."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

logger = logging.getLogger(__name__)

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Monitoring middleware for request tracking."""
    
    async def dispatch(self, request: Request, call_next):
        """Track request metrics."""
        
        start_time = time.time()
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s - "
            f"Client: {client_ip}"
        )
        
        # Add headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Powered-By"] = "Pain Coach AI Pascal"
        
        return response