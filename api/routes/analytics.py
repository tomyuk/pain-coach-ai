"""Analytics API routes."""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from core.data.database import DatabaseManager
from core.performance.profiler import profile
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/pain-trends", response_model=dict)
@profile("pain_trends_analysis")
async def get_pain_trends(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    days: int = Query(30, ge=7, le=365)
):
    """Get pain trend analysis."""
    try:
        # Get pain trends from database
        trends = await db.analyze_pain_trends(current_user, days)
        
        return {
            "user_id": current_user,
            "analysis_period_days": days,
            "trends": trends,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pain trends analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze pain trends"
        )

@router.get("/correlations", response_model=dict)
@profile("correlation_analysis")
async def get_correlations(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    days: int = Query(30, ge=7, le=365)
):
    """Get correlation analysis between pain and other factors."""
    try:
        # Get pain records
        records = await db.get_pain_records(current_user, limit=1000, days_back=days)
        
        if len(records) < 10:
            return {
                "error": "Insufficient data for correlation analysis",
                "minimum_records_needed": 10,
                "current_records": len(records)
            }
        
        # Calculate correlations
        correlations = {}
        
        # Sleep quality correlation
        sleep_records = [r for r in records if r.sleep_quality is not None]
        if len(sleep_records) > 5:
            correlations["sleep_quality"] = calculate_correlation(
                [r.pain_level for r in sleep_records],
                [r.sleep_quality for r in sleep_records]
            )
        
        # Stress level correlation
        stress_records = [r for r in records if r.stress_level is not None]
        if len(stress_records) > 5:
            correlations["stress_level"] = calculate_correlation(
                [r.pain_level for r in stress_records],
                [r.stress_level for r in stress_records]
            )
        
        # Weather correlation (if available)
        weather_records = [r for r in records if r.weather_data and r.weather_data.get('pressure')]
        if len(weather_records) > 10:
            correlations["weather_pressure"] = calculate_correlation(
                [r.pain_level for r in weather_records],
                [r.weather_data.get('pressure', 0) for r in weather_records]
            )
        
        return {
            "user_id": current_user,
            "analysis_period_days": days,
            "correlations": correlations,
            "records_analyzed": len(records),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform correlation analysis"
        )

@router.get("/patterns", response_model=dict)
@profile("pattern_analysis")
async def get_patterns(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    days: int = Query(30, ge=7, le=365)
):
    """Get pain pattern analysis."""
    try:
        # Get pain records
        records = await db.get_pain_records(current_user, limit=1000, days_back=days)
        
        if not records:
            return {
                "error": "No pain records found",
                "suggestion": "Start recording your pain to see patterns"
            }
        
        # Analyze patterns
        patterns = {}
        
        # Time of day patterns
        hourly_pain = {}
        for record in records:
            hour = record.recorded_at.hour
            if hour not in hourly_pain:
                hourly_pain[hour] = []
            hourly_pain[hour].append(record.pain_level)
        
        # Calculate average pain by hour
        hourly_averages = {}
        for hour, pain_levels in hourly_pain.items():
            hourly_averages[hour] = sum(pain_levels) / len(pain_levels)
        
        patterns["time_of_day"] = hourly_averages
        
        # Day of week patterns
        daily_pain = {}
        for record in records:
            day = record.recorded_at.strftime("%A")
            if day not in daily_pain:
                daily_pain[day] = []
            daily_pain[day].append(record.pain_level)
        
        # Calculate average pain by day of week
        daily_averages = {}
        for day, pain_levels in daily_pain.items():
            daily_averages[day] = sum(pain_levels) / len(pain_levels)
        
        patterns["day_of_week"] = daily_averages
        
        # Pain type patterns
        pain_type_freq = {}
        for record in records:
            if record.pain_type:
                for pain_type in record.pain_type:
                    pain_type_freq[pain_type] = pain_type_freq.get(pain_type, 0) + 1
        
        patterns["pain_types"] = pain_type_freq
        
        # Activity patterns
        activity_pain = {}
        for record in records:
            if record.activity_before:
                activity = record.activity_before.lower()
                if activity not in activity_pain:
                    activity_pain[activity] = []
                activity_pain[activity].append(record.pain_level)
        
        # Calculate average pain by activity
        activity_averages = {}
        for activity, pain_levels in activity_pain.items():
            if len(pain_levels) >= 3:  # Only include activities with 3+ records
                activity_averages[activity] = sum(pain_levels) / len(pain_levels)
        
        patterns["activities"] = activity_averages
        
        return {
            "user_id": current_user,
            "analysis_period_days": days,
            "patterns": patterns,
            "records_analyzed": len(records),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pattern analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform pattern analysis"
        )

@router.get("/insights", response_model=dict)
@profile("insights_generation")
async def get_insights(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    days: int = Query(30, ge=7, le=365)
):
    """Get AI-generated insights from pain data."""
    try:
        # Get pain records
        records = await db.get_pain_records(current_user, limit=1000, days_back=days)
        
        if len(records) < 7:
            return {
                "insights": ["Record more pain data to get personalized insights"],
                "recommendations": ["Try to record your pain daily for better analysis"],
                "data_sufficiency": "insufficient"
            }
        
        insights = []
        recommendations = []
        
        # Calculate basic statistics
        pain_levels = [r.pain_level for r in records]
        avg_pain = sum(pain_levels) / len(pain_levels)
        
        # Pain level insights
        if avg_pain > 7:
            insights.append(f"Your average pain level is {avg_pain:.1f}/10, which is quite high")
            recommendations.append("Consider discussing pain management strategies with your healthcare provider")
        elif avg_pain < 4:
            insights.append(f"Your average pain level is {avg_pain:.1f}/10, which is relatively well-controlled")
            recommendations.append("Continue with your current pain management approach")
        
        # Trend analysis
        if len(records) > 14:
            recent_records = records[:7]  # Last 7 days
            older_records = records[7:14]  # Previous 7 days
            
            recent_avg = sum(r.pain_level for r in recent_records) / len(recent_records)
            older_avg = sum(r.pain_level for r in older_records) / len(older_records)
            
            if recent_avg > older_avg + 0.5:
                insights.append("Your pain levels have increased in the past week")
                recommendations.append("Monitor for potential triggers and consult your healthcare provider")
            elif recent_avg < older_avg - 0.5:
                insights.append("Your pain levels have improved in the past week")
                recommendations.append("Continue with current strategies that may be helping")
        
        # Sleep analysis
        sleep_records = [r for r in records if r.sleep_quality is not None]
        if len(sleep_records) > 5:
            avg_sleep = sum(r.sleep_quality for r in sleep_records) / len(sleep_records)
            if avg_sleep < 3:
                insights.append("Your sleep quality appears to be poor")
                recommendations.append("Focus on sleep hygiene as poor sleep can worsen pain")
        
        # Stress analysis
        stress_records = [r for r in records if r.stress_level is not None]
        if len(stress_records) > 5:
            avg_stress = sum(r.stress_level for r in stress_records) / len(stress_records)
            if avg_stress > 3:
                insights.append("You're experiencing elevated stress levels")
                recommendations.append("Consider stress management techniques like meditation or breathing exercises")
        
        # Consistency insights
        days_with_records = len(set(r.recorded_at.date() for r in records))
        if days_with_records < days * 0.5:
            insights.append("You're tracking pain less than half the days")
            recommendations.append("Try to record your pain daily for better insights")
        
        return {
            "user_id": current_user,
            "analysis_period_days": days,
            "insights": insights,
            "recommendations": recommendations,
            "data_sufficiency": "sufficient" if len(records) >= 14 else "limited",
            "records_analyzed": len(records),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate insights"
        )

@router.get("/export", response_model=dict)
@profile("data_export")
async def export_data(
    current_user: str = Depends(get_current_user),
    db: DatabaseManager = Depends(),
    format: str = Query("json", regex="^(json|csv)$"),
    days: int = Query(90, ge=1, le=365)
):
    """Export pain data for external analysis."""
    try:
        # Get pain records
        records = await db.get_pain_records(current_user, limit=10000, days_back=days)
        
        if not records:
            return {
                "error": "No data available for export",
                "records_found": 0
            }
        
        # Format data for export
        export_data = []
        for record in records:
            export_data.append({
                "date": record.recorded_at.isoformat(),
                "pain_level": record.pain_level,
                "pain_type": record.pain_type,
                "sleep_quality": record.sleep_quality,
                "stress_level": record.stress_level,
                "activity_before": record.activity_before,
                "medications_taken": record.medications_taken,
                "interventions": record.non_med_interventions,
                "effectiveness": record.effectiveness_rating,
                "weather_pressure": record.weather_data.get('pressure') if record.weather_data else None,
                "weather_humidity": record.weather_data.get('humidity') if record.weather_data else None
            })
        
        return {
            "user_id": current_user,
            "export_format": format,
            "records_exported": len(export_data),
            "date_range": {
                "start": records[-1].recorded_at.isoformat() if records else None,
                "end": records[0].recorded_at.isoformat() if records else None
            },
            "data": export_data,
            "exported_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data export error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export data"
        )

def calculate_correlation(x_values: list, y_values: list) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    
    n = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    sum_y2 = sum(y * y for y in y_values)
    
    denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
    if denominator == 0:
        return 0.0
    
    numerator = n * sum_xy - sum_x * sum_y
    return numerator / denominator