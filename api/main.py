"""Main FastAPI application for Pain Coach AI Pascal."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import logging
from typing import Optional

from core.data.database import DatabaseManager
from core.ai.mlx_engine import PainCoachMLXEngine, PainChatConfig
from core.privacy.encryption_manager import EncryptionManager
from core.privacy.audit_logger import AuditLogger
from core.integrations.health_manager import HealthDataManager
from core.performance.monitoring import system_monitor
from core.performance.profiler import profiler

from .routes import auth, pain_records, ai_chat, analytics, health_integrations, admin
from .middleware.auth import AuthMiddleware
from .middleware.rate_limiting import RateLimitMiddleware
from .middleware.monitoring import MonitoringMiddleware

logger = logging.getLogger(__name__)

# Global dependencies
db_manager = DatabaseManager()
ai_engine = PainCoachMLXEngine(PainChatConfig())
encryption_manager = EncryptionManager()
audit_logger = AuditLogger()
health_manager = HealthDataManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Pain Coach AI Pascal API")
    
    # Initialize database
    await db_manager.initialize()
    
    # Initialize AI engine
    await ai_engine.initialize()
    
    # Start system monitoring
    await system_monitor.start_monitoring()
    
    # Enable profiling
    profiler.enable()
    
    logger.info("Pain Coach AI Pascal API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Pain Coach AI Pascal API")
    
    # Stop monitoring
    await system_monitor.stop_monitoring()
    
    # Close database connections
    await db_manager.close()
    
    # Export metrics
    profiler.export_metrics("performance_metrics.json")
    
    logger.info("Pain Coach AI Pascal API shut down complete")

# Create FastAPI app
app = FastAPI(
    title="Pain Coach AI Pascal",
    description="AI-powered chronic pain management assistant",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(MonitoringMiddleware)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(pain_records.router, prefix="/api/pain", tags=["pain records"])
app.include_router(ai_chat.router, prefix="/api/ai", tags=["ai chat"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(health_integrations.router, prefix="/api/health", tags=["health integrations"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

# Dependency injection
async def get_db():
    """Get database manager."""
    return db_manager

async def get_ai_engine():
    """Get AI engine."""
    return ai_engine

async def get_encryption_manager():
    """Get encryption manager."""
    return encryption_manager

async def get_audit_logger():
    """Get audit logger."""
    return audit_logger

async def get_health_manager():
    """Get health data manager."""
    return health_manager

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }

# System status endpoint
@app.get("/api/system/status")
async def system_status():
    """Get system status."""
    health_summary = system_monitor.get_system_health_summary()
    performance_summary = profiler.get_performance_summary()
    
    return {
        "health": health_summary,
        "performance": performance_summary,
        "database": "connected",
        "ai_engine": "ready"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Pain Coach AI Pascal API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)