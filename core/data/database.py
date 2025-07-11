"""Database manager for Pain Coach AI Pascal."""

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete, update
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from .models import Base, User, PainRecord, AIConversation, AnalysisCache
import duckdb
import sqlite3

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages both SQLite (user data) and DuckDB (analytics) databases."""
    
    def __init__(self, sqlite_url: str = "sqlite+aiosqlite:///./pain_coach.db", 
                 duckdb_path: str = "./analytics.duckdb"):
        # SQLite for user data (encrypted)
        self.sqlite_engine = create_async_engine(sqlite_url, echo=False)
        self.async_session = sessionmaker(
            self.sqlite_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # DuckDB for analytics
        self.duckdb_path = duckdb_path
        self.duckdb_conn = None
        
    async def initialize(self):
        """Initialize databases."""
        # Create SQLite tables
        async with self.sqlite_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        # Initialize DuckDB
        self.duckdb_conn = duckdb.connect(self.duckdb_path)
        self._create_duckdb_tables()
        
        logger.info("Database initialized successfully")
    
    def _create_duckdb_tables(self):
        """Create DuckDB tables for analytics."""
        # Pain timeseries table
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS pain_timeseries (
                user_id VARCHAR,
                hour_bucket TIMESTAMP,
                avg_pain FLOAT,
                min_pain INTEGER,
                max_pain INTEGER,
                record_count INTEGER,
                pain_types_agg JSON,
                weather_snapshot JSON
            )
        """)
        
        # Pain patterns table
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS pain_patterns (
                user_id VARCHAR,
                date DATE,
                daily_avg_pain FLOAT,
                daily_max_pain INTEGER,
                pain_episodes INTEGER,
                sleep_quality FLOAT,
                stress_level FLOAT,
                weather_pressure FLOAT,
                weather_humidity FLOAT
            )
        """)
        
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user."""
        async with self.async_session() as session:
            user = User(**user_data)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    async def create_pain_record(self, pain_data: Dict[str, Any]) -> PainRecord:
        """Create a new pain record."""
        async with self.async_session() as session:
            pain_record = PainRecord(**pain_data)
            session.add(pain_record)
            await session.commit()
            await session.refresh(pain_record)
            
            # Also insert into DuckDB for analytics
            await self._insert_pain_analytics(pain_record)
            
            return pain_record
    
    async def get_pain_records(self, user_id: str, limit: int = 100, 
                              days_back: int = 30) -> List[PainRecord]:
        """Get pain records for a user."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        async with self.async_session() as session:
            result = await session.execute(
                select(PainRecord)
                .where(PainRecord.user_id == user_id)
                .where(PainRecord.recorded_at >= cutoff_date)
                .order_by(PainRecord.recorded_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def create_conversation(self, conversation_data: Dict[str, Any]) -> AIConversation:
        """Create a new AI conversation record."""
        async with self.async_session() as session:
            conversation = AIConversation(**conversation_data)
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation
    
    async def get_recent_conversations(self, user_id: str, limit: int = 10) -> List[AIConversation]:
        """Get recent conversations for a user."""
        async with self.async_session() as session:
            result = await session.execute(
                select(AIConversation)
                .where(AIConversation.user_id == user_id)
                .order_by(AIConversation.timestamp.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def _insert_pain_analytics(self, pain_record: PainRecord):
        """Insert pain record into DuckDB for analytics."""
        try:
            hour_bucket = pain_record.recorded_at.replace(minute=0, second=0, microsecond=0)
            
            # Insert into pain_timeseries
            self.duckdb_conn.execute("""
                INSERT INTO pain_timeseries 
                (user_id, hour_bucket, avg_pain, min_pain, max_pain, record_count, pain_types_agg, weather_snapshot)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """, [
                str(pain_record.user_id),
                hour_bucket,
                pain_record.pain_level,
                pain_record.pain_level,
                pain_record.pain_level,
                pain_record.pain_type,
                pain_record.weather_data
            ])
            
            # Insert into pain_patterns (daily aggregation)
            date = pain_record.recorded_at.date()
            self.duckdb_conn.execute("""
                INSERT OR REPLACE INTO pain_patterns 
                (user_id, date, daily_avg_pain, daily_max_pain, pain_episodes, sleep_quality, stress_level, weather_pressure, weather_humidity)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
            """, [
                str(pain_record.user_id),
                date,
                pain_record.pain_level,
                pain_record.pain_level,
                pain_record.sleep_quality,
                pain_record.stress_level,
                pain_record.weather_data.get('pressure', 0) if pain_record.weather_data else 0,
                pain_record.weather_data.get('humidity', 0) if pain_record.weather_data else 0
            ])
            
        except Exception as e:
            logger.error(f"Error inserting pain analytics: {e}")
    
    async def analyze_pain_trends(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze pain trends using DuckDB."""
        try:
            # Get pain trends
            trends = self.duckdb_conn.execute("""
                SELECT 
                    date,
                    daily_avg_pain,
                    daily_max_pain,
                    pain_episodes,
                    sleep_quality,
                    stress_level
                FROM pain_patterns 
                WHERE user_id = ? 
                    AND date >= CURRENT_DATE - INTERVAL '{days} days'
                ORDER BY date
            """.format(days=days), [user_id]).fetchall()
            
            # Calculate correlations
            correlations = self.duckdb_conn.execute("""
                SELECT 
                    corr(daily_avg_pain, sleep_quality) as pain_sleep_corr,
                    corr(daily_avg_pain, stress_level) as pain_stress_corr,
                    corr(daily_avg_pain, weather_pressure) as pain_weather_corr
                FROM pain_patterns 
                WHERE user_id = ? 
                    AND date >= CURRENT_DATE - INTERVAL '{days} days'
            """.format(days=days), [user_id]).fetchone()
            
            # Calculate statistics
            stats = self.duckdb_conn.execute("""
                SELECT 
                    AVG(daily_avg_pain) as avg_pain,
                    MIN(daily_avg_pain) as min_pain,
                    MAX(daily_avg_pain) as max_pain,
                    STDDEV(daily_avg_pain) as pain_variability,
                    COUNT(*) as total_days
                FROM pain_patterns 
                WHERE user_id = ? 
                    AND date >= CURRENT_DATE - INTERVAL '{days} days'
            """.format(days=days), [user_id]).fetchone()
            
            return {
                "trends": [dict(zip([col[0] for col in self.duckdb_conn.description], row)) for row in trends],
                "correlations": dict(zip([col[0] for col in self.duckdb_conn.description], correlations)) if correlations else {},
                "statistics": dict(zip([col[0] for col in self.duckdb_conn.description], stats)) if stats else {},
                "analysis_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pain trends: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, retention_days: int = 365):
        """Clean up old data based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        async with self.async_session() as session:
            # Delete old conversations
            await session.execute(
                delete(AIConversation).where(
                    AIConversation.retention_until < cutoff_date
                )
            )
            
            # Delete old analysis cache
            await session.execute(
                delete(AnalysisCache).where(
                    AnalysisCache.valid_until < cutoff_date
                )
            )
            
            await session.commit()
    
    async def close(self):
        """Close database connections."""
        await self.sqlite_engine.dispose()
        if self.duckdb_conn:
            self.duckdb_conn.close()