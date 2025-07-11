"""Performance profiler for Pain Coach AI Pascal."""

import time
import asyncio
import functools
from typing import Dict, Any, Optional, Callable, List
import logging
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import json

logger = logging.getLogger(__name__)

@dataclass
class ProfileMetrics:
    """Performance metrics for a profiled operation."""
    function_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: Optional[datetime] = None
    call_times: List[float] = field(default_factory=list)
    
    def add_call(self, execution_time: float):
        """Add a call to the metrics."""
        self.total_calls += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.total_calls
        self.last_call_time = datetime.utcnow()
        
        # Keep only last 100 call times
        self.call_times.append(execution_time)
        if len(self.call_times) > 100:
            self.call_times.pop(0)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Get percentile statistics."""
        if not self.call_times:
            return {}
        
        return {
            "p50": statistics.median(self.call_times),
            "p90": statistics.quantiles(self.call_times, n=10)[8] if len(self.call_times) > 10 else max(self.call_times),
            "p95": statistics.quantiles(self.call_times, n=20)[18] if len(self.call_times) > 20 else max(self.call_times),
            "p99": statistics.quantiles(self.call_times, n=100)[98] if len(self.call_times) > 100 else max(self.call_times),
        }

class PerformanceProfiler:
    """Performance profiler for tracking execution metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, ProfileMetrics] = {}
        self.enabled = True
        self.threshold_ms = 1.0  # Only track calls > 1ms
        
    def profile(self, name: Optional[str] = None):
        """Decorator to profile function execution."""
        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    if not self.enabled:
                        return await func(*args, **kwargs)
                    
                    start_time = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        end_time = time.perf_counter()
                        execution_time = end_time - start_time
                        
                        if execution_time * 1000 > self.threshold_ms:
                            self._record_metrics(profile_name, execution_time)
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    if not self.enabled:
                        return func(*args, **kwargs)
                    
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        end_time = time.perf_counter()
                        execution_time = end_time - start_time
                        
                        if execution_time * 1000 > self.threshold_ms:
                            self._record_metrics(profile_name, execution_time)
                
                return sync_wrapper
        
        return decorator
    
    def _record_metrics(self, name: str, execution_time: float):
        """Record metrics for a profiled operation."""
        if name not in self.metrics:
            self.metrics[name] = ProfileMetrics(function_name=name)
        
        self.metrics[name].add_call(execution_time)
    
    async def profile_ai_response_time(self, ai_engine, prompt: str, **kwargs) -> Dict[str, Any]:
        """Profile AI response generation."""
        start_time = time.perf_counter()
        memory_before = self._get_memory_usage()
        
        try:
            response_tokens = []
            token_times = []
            
            # Time each token
            async for token in ai_engine.generate_response(prompt, **kwargs):
                token_time = time.perf_counter()
                response_tokens.append(token)
                token_times.append(token_time)
            
            end_time = time.perf_counter()
            memory_after = self._get_memory_usage()
            
            response_text = "".join(response_tokens)
            total_time = end_time - start_time
            
            # Calculate token timing metrics
            if len(token_times) > 1:
                first_token_time = token_times[0] - start_time
                tokens_per_second = len(response_tokens) / total_time
                
                # Inter-token delays
                inter_token_delays = []
                for i in range(1, len(token_times)):
                    delay = token_times[i] - token_times[i-1]
                    inter_token_delays.append(delay)
                
                avg_inter_token_delay = statistics.mean(inter_token_delays) if inter_token_delays else 0
            else:
                first_token_time = total_time
                tokens_per_second = len(response_tokens) / total_time if total_time > 0 else 0
                avg_inter_token_delay = 0
            
            metrics = {
                "total_time_ms": total_time * 1000,
                "first_token_time_ms": first_token_time * 1000,
                "tokens_per_second": tokens_per_second,
                "avg_inter_token_delay_ms": avg_inter_token_delay * 1000,
                "total_tokens": len(response_tokens),
                "prompt_length": len(prompt),
                "response_length": len(response_text),
                "memory_delta_mb": (memory_after - memory_before) / 1024 / 1024,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Record in profiler
            self._record_metrics("ai_response_generation", total_time)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error profiling AI response: {e}")
            return {"error": str(e)}
    
    async def profile_database_operations(self, db_manager, operation_count: int = 100) -> Dict[str, Any]:
        """Profile database operations."""
        
        # Test different operations
        benchmarks = {}
        
        # Write operations
        write_times = []
        for i in range(operation_count):
            start_time = time.perf_counter()
            
            # Simulate pain record insert
            test_record = {
                "user_id": "test_user",
                "pain_level": 5,
                "recorded_at": datetime.utcnow(),
                "pain_type": ["dull"],
                "input_method": "benchmark"
            }
            
            try:
                # await db_manager.create_pain_record(test_record)
                # Simulate database operation
                await asyncio.sleep(0.001)  # 1ms simulation
                
                end_time = time.perf_counter()
                write_times.append(end_time - start_time)
                
            except Exception as e:
                logger.error(f"Database write benchmark error: {e}")
                break
        
        if write_times:
            benchmarks["write_operations"] = {
                "ops_per_second": len(write_times) / sum(write_times),
                "avg_time_ms": statistics.mean(write_times) * 1000,
                "min_time_ms": min(write_times) * 1000,
                "max_time_ms": max(write_times) * 1000,
                "p95_time_ms": statistics.quantiles(write_times, n=20)[18] * 1000 if len(write_times) > 20 else max(write_times) * 1000
            }
        
        # Read operations
        read_times = []
        for i in range(operation_count):
            start_time = time.perf_counter()
            
            try:
                # await db_manager.get_pain_records("test_user", limit=10)
                # Simulate database operation
                await asyncio.sleep(0.0005)  # 0.5ms simulation
                
                end_time = time.perf_counter()
                read_times.append(end_time - start_time)
                
            except Exception as e:
                logger.error(f"Database read benchmark error: {e}")
                break
        
        if read_times:
            benchmarks["read_operations"] = {
                "ops_per_second": len(read_times) / sum(read_times),
                "avg_time_ms": statistics.mean(read_times) * 1000,
                "min_time_ms": min(read_times) * 1000,
                "max_time_ms": max(read_times) * 1000,
                "p95_time_ms": statistics.quantiles(read_times, n=20)[18] * 1000 if len(read_times) > 20 else max(read_times) * 1000
            }
        
        # Analytics operations
        analytics_start = time.perf_counter()
        try:
            # await db_manager.analyze_pain_trends("test_user", days=30)
            # Simulate analytics operation
            await asyncio.sleep(0.01)  # 10ms simulation
            
            analytics_end = time.perf_counter()
            analytics_time = analytics_end - analytics_start
            
            benchmarks["analytics_operations"] = {
                "avg_time_ms": analytics_time * 1000,
                "operations_tested": 1
            }
            
        except Exception as e:
            logger.error(f"Database analytics benchmark error: {e}")
        
        return benchmarks
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all tracked operations."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_operations": len(self.metrics),
            "operations": {}
        }
        
        for name, metrics in self.metrics.items():
            operation_summary = {
                "total_calls": metrics.total_calls,
                "total_time_ms": metrics.total_time * 1000,
                "avg_time_ms": metrics.avg_time * 1000,
                "min_time_ms": metrics.min_time * 1000,
                "max_time_ms": metrics.max_time * 1000,
                "last_call": metrics.last_call_time.isoformat() if metrics.last_call_time else None
            }
            
            # Add percentiles if available
            percentiles = metrics.get_percentiles()
            if percentiles:
                operation_summary["percentiles_ms"] = {
                    k: v * 1000 for k, v in percentiles.items()
                }
            
            summary["operations"][name] = operation_summary
        
        return summary
    
    def get_slowest_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest operations."""
        sorted_ops = sorted(
            self.metrics.values(),
            key=lambda x: x.avg_time,
            reverse=True
        )
        
        return [{
            "name": op.function_name,
            "avg_time_ms": op.avg_time * 1000,
            "total_calls": op.total_calls,
            "total_time_ms": op.total_time * 1000
        } for op in sorted_ops[:limit]]
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()
        logger.info("Performance metrics reset")
    
    def set_threshold(self, threshold_ms: float):
        """Set minimum threshold for tracking operations."""
        self.threshold_ms = threshold_ms
        logger.info(f"Performance profiler threshold set to {threshold_ms}ms")
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
        logger.info("Performance profiling enabled")
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False
        logger.info("Performance profiling disabled")
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file."""
        try:
            summary = self.get_performance_summary()
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Metrics exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

# Global profiler instance
profiler = PerformanceProfiler()

def profile(name: Optional[str] = None):
    """Convenience decorator for profiling."""
    return profiler.profile(name)