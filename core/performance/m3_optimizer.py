"""M3 Max performance optimizer."""

import psutil
import torch
import mlx.core as mx
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Dict, Any, Optional
import time
import logging
import os

logger = logging.getLogger(__name__)

class M3MaxOptimizer:
    """M3 Max specific performance optimizations."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.gpu_available = torch.backends.mps.is_available()
        
        # MLX optimization
        if mx.metal.is_available():
            mx.set_default_device(mx.gpu)
            logger.info("MLX Metal GPU acceleration enabled")
        else:
            logger.warning("MLX Metal not available, using CPU")
            
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(8, self.cpu_count)
        )
        
        # Performance metrics
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "inference_times": [],
            "temperature": []
        }
        
    async def optimize_ai_inference(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize AI inference settings for M3 Max."""
        
        # Get current system state
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        optimization_settings = {
            "batch_size": 1,  # Real-time conversation
            "max_workers": min(4, self.cpu_count),
            "memory_pool_size": "auto",
            "gpu_utilization_target": 0.8,
            "quantization": "auto",
            "model_offload": False
        }
        
        # Memory-based optimizations
        available_memory_gb = memory_info.available / (1024**3)
        
        if available_memory_gb < 8:
            # Low memory optimizations
            optimization_settings.update({
                "quantization": "int8",
                "model_offload": True,
                "cache_size": "small",
                "max_sequence_length": 1024
            })
            logger.info("Applied low memory optimizations")
            
        elif available_memory_gb > 16:
            # High memory optimizations
            optimization_settings.update({
                "quantization": "int4",
                "model_offload": False,
                "cache_size": "large",
                "max_sequence_length": 4096,
                "batch_size": 2
            })
            logger.info("Applied high memory optimizations")
            
        # CPU utilization optimizations
        if cpu_percent > 80:
            optimization_settings.update({
                "max_workers": max(2, self.cpu_count // 2),
                "inference_threads": 2
            })
            logger.info("Applied high CPU usage optimizations")
            
        # GPU optimization
        if self.gpu_available:
            optimization_settings.update({
                "use_metal": True,
                "metal_optimization": True,
                "gpu_memory_fraction": 0.8
            })
            
        return optimization_settings
    
    async def optimize_database_operations(self) -> Dict[str, Any]:
        """Optimize database operations for M3 Max."""
        
        optimization_settings = {
            "connection_pool_size": min(20, self.cpu_count * 2),
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "echo": False,
            "query_cache_size": 1000
        }
        
        # DuckDB specific optimizations
        duckdb_settings = {
            "memory_limit": f"{int(self.memory_total * 0.3 / (1024**3))}GB",
            "threads": min(self.cpu_count, 8),
            "max_memory": f"{int(self.memory_total * 0.5 / (1024**3))}GB",
            "enable_object_cache": True,
            "enable_http_metadata_cache": True
        }
        
        optimization_settings["duckdb"] = duckdb_settings
        return optimization_settings
    
    async def optimize_audio_processing(self) -> Dict[str, Any]:
        """Optimize audio processing for M3 Max."""
        
        optimization_settings = {
            "sample_rate": 16000,
            "chunk_size": 1024,
            "buffer_size": 8192,
            "channels": 1,
            "dtype": "float32",
            "device": "default"
        }
        
        # Whisper optimization
        whisper_settings = {
            "model_size": "large-v3",
            "device": "auto",  # Auto-detect best device
            "compute_type": "int8",
            "beam_size": 1,  # Faster inference
            "best_of": 1,
            "patience": 1.0,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "prompt_reset_on_temperature": 0.5,
            "initial_prompt": None,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": False,
            "prepend_punctuations": "\"'"¿([{-",
            "append_punctuations": "\"'.。,，!！?？:：")]}、"
        }
        
        optimization_settings["whisper"] = whisper_settings
        return optimization_settings
    
    async def monitor_system_resources(self) -> Dict[str, Any]:
        """Monitor system resources."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # GPU metrics (if available)
        gpu_metrics = {}
        if self.gpu_available:
            gpu_metrics = await self._get_gpu_metrics()
            
        # Temperature (macOS)
        temperature = await self._get_cpu_temperature()
        
        metrics = {
            "timestamp": time.time(),
            "cpu": {
                "percent": cpu_percent,
                "frequency": cpu_freq.current if cpu_freq else 0,
                "count": self.cpu_count
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "disk": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": (disk_usage.used / disk_usage.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            "network": {
                "bytes_sent": network_io.bytes_sent if network_io else 0,
                "bytes_recv": network_io.bytes_recv if network_io else 0,
                "packets_sent": network_io.packets_sent if network_io else 0,
                "packets_recv": network_io.packets_recv if network_io else 0
            },
            "gpu": gpu_metrics,
            "temperature": temperature
        }
        
        # Store metrics for trending
        self.metrics["cpu_usage"].append(cpu_percent)
        self.metrics["memory_usage"].append(memory.percent)
        self.metrics["temperature"].append(temperature)
        
        # Keep only last 100 metrics
        for key in self.metrics:
            if len(self.metrics[key]) > 100:
                self.metrics[key] = self.metrics[key][-100:]
                
        return metrics
    
    async def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics for M3 Max."""
        try:
            if mx.metal.is_available():
                # MLX Metal metrics
                active_memory = mx.metal.get_active_memory()
                peak_memory = mx.metal.get_peak_memory()
                cache_memory = mx.metal.get_cache_memory()
                
                return {
                    "active_memory": active_memory,
                    "peak_memory": peak_memory,
                    "cache_memory": cache_memory,
                    "utilization": active_memory / peak_memory if peak_memory > 0 else 0,
                    "type": "Apple Silicon GPU"
                }
            else:
                return {"available": False}
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return {"error": str(e)}
    
    async def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (macOS specific)."""
        try:
            # Use powermetrics to get temperature
            import subprocess
            result = subprocess.run(
                ["sudo", "powermetrics", "-n", "1", "-s", "cpu_power", "--format", "plist"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse temperature from output (simplified)
                # In practice, you'd parse the plist output
                return 50.0  # Placeholder
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error getting CPU temperature: {e}")
            return 0.0
    
    async def optimize_memory_usage(self):
        """Optimize memory usage."""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear MLX cache if available
            if mx.metal.is_available():
                mx.metal.clear_cache()
                
            # Clear PyTorch cache if available
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current metrics."""
        recommendations = []
        
        # Check recent CPU usage
        if len(self.metrics["cpu_usage"]) > 10:
            avg_cpu = sum(self.metrics["cpu_usage"][-10:]) / 10
            if avg_cpu > 90:
                recommendations.append({
                    "type": "cpu",
                    "severity": "high",
                    "message": "High CPU usage detected. Consider reducing AI model complexity or increasing batch processing interval.",
                    "action": "reduce_ai_complexity"
                })
        
        # Check memory usage
        if len(self.metrics["memory_usage"]) > 10:
            avg_memory = sum(self.metrics["memory_usage"][-10:]) / 10
            if avg_memory > 85:
                recommendations.append({
                    "type": "memory",
                    "severity": "high",
                    "message": "High memory usage detected. Consider enabling model quantization or reducing cache size.",
                    "action": "enable_quantization"
                })
        
        # Check temperature
        if len(self.metrics["temperature"]) > 10:
            avg_temp = sum(self.metrics["temperature"][-10:]) / 10
            if avg_temp > 80:
                recommendations.append({
                    "type": "temperature",
                    "severity": "medium",
                    "message": "High temperature detected. Consider reducing processing frequency or improving cooling.",
                    "action": "reduce_processing_frequency"
                })
        
        return {
            "timestamp": time.time(),
            "recommendations": recommendations,
            "system_health": "good" if not recommendations else "warning"
        }
    
    async def apply_optimization_profile(self, profile: str) -> Dict[str, Any]:
        """Apply a predefined optimization profile."""
        profiles = {
            "power_saving": {
                "ai_quantization": "int8",
                "max_workers": 2,
                "gpu_utilization": 0.5,
                "cache_size": "small",
                "sync_frequency": "daily"
            },
            "balanced": {
                "ai_quantization": "int4",
                "max_workers": 4,
                "gpu_utilization": 0.7,
                "cache_size": "medium",
                "sync_frequency": "hourly"
            },
            "performance": {
                "ai_quantization": "float16",
                "max_workers": 8,
                "gpu_utilization": 0.9,
                "cache_size": "large",
                "sync_frequency": "continuous"
            }
        }
        
        if profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}")
        
        config = profiles[profile]
        logger.info(f"Applied optimization profile: {profile}")
        
        return config
    
    def close(self):
        """Clean up resources."""
        self.thread_pool.shutdown(wait=True)