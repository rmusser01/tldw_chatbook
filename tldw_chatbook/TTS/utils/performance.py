# performance.py  
# Description: Performance tracking utilities for TTS operations
#
# Imports
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from loguru import logger

#######################################################################################################################
#
# Performance Tracking Classes

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: datetime
    duration: float  # seconds
    tokens: int
    characters: int
    audio_duration: float  # seconds
    format: str
    backend: str
    voice: str
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second"""
        return self.tokens / self.duration if self.duration > 0 else 0
    
    @property
    def realtime_factor(self) -> float:
        """Calculate realtime factor (audio duration / generation time)"""
        return self.audio_duration / self.duration if self.duration > 0 else 0
    
    @property
    def characters_per_second(self) -> float:
        """Calculate characters per second"""
        return self.characters / self.duration if self.duration > 0 else 0


class PerformanceTracker:
    """Track TTS performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize tracker.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics: List[PerformanceMetric] = []
        self._current_timers: Dict[str, float] = {}
    
    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation"""
        self._current_timers[operation_id] = time.time()
    
    def end_timer(self, operation_id: str) -> float:
        """
        End timing and return duration.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Duration in seconds
        """
        if operation_id not in self._current_timers:
            logger.warning(f"No timer found for operation: {operation_id}")
            return 0.0
        
        start_time = self._current_timers.pop(operation_id)
        return time.time() - start_time
    
    def record_metric(
        self,
        duration: float,
        tokens: int,
        characters: int,
        audio_duration: float,
        format: str,
        backend: str,
        voice: str
    ) -> PerformanceMetric:
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            duration=duration,
            tokens=tokens,
            characters=characters,
            audio_duration=audio_duration,
            format=format,
            backend=backend,
            voice=voice
        )
        
        self.metrics.append(metric)
        
        # Trim history if needed
        if len(self.metrics) > self.max_history:
            self.metrics = self.metrics[-self.max_history:]
        
        # Log performance info
        logger.info(
            f"TTS Performance - Backend: {backend}, Voice: {voice}, "
            f"Duration: {duration:.2f}s, Audio: {audio_duration:.2f}s, "
            f"Speed: {metric.realtime_factor:.1f}x realtime, "
            f"Rate: {metric.tokens_per_second:.1f} tokens/s"
        )
        
        return metric
    
    def get_statistics(self, backend: Optional[str] = None) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Args:
            backend: Filter by backend name
            
        Returns:
            Dictionary of statistics
        """
        # Filter metrics
        metrics = self.metrics
        if backend:
            metrics = [m for m in metrics if m.backend == backend]
        
        if not metrics:
            return {
                "count": 0,
                "avg_tokens_per_second": 0,
                "avg_realtime_factor": 0,
                "avg_duration": 0,
                "total_audio_generated": 0,
            }
        
        # Calculate statistics
        tokens_per_second = [m.tokens_per_second for m in metrics]
        realtime_factors = [m.realtime_factor for m in metrics]
        durations = [m.duration for m in metrics]
        
        return {
            "count": len(metrics),
            "avg_tokens_per_second": statistics.mean(tokens_per_second),
            "min_tokens_per_second": min(tokens_per_second),
            "max_tokens_per_second": max(tokens_per_second),
            "avg_realtime_factor": statistics.mean(realtime_factors),
            "min_realtime_factor": min(realtime_factors),
            "max_realtime_factor": max(realtime_factors),
            "avg_duration": statistics.mean(durations),
            "total_audio_generated": sum(m.audio_duration for m in metrics),
            "total_time_spent": sum(durations),
        }
    
    def get_recent_metrics(self, count: int = 10) -> List[PerformanceMetric]:
        """Get recent performance metrics"""
        return list(reversed(self.metrics[-count:]))
    
    def clear_history(self) -> None:
        """Clear performance history"""
        self.metrics.clear()
        self._current_timers.clear()


# Global tracker instance
_global_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


# Convenience functions
def start_tts_timer(operation_id: str) -> None:
    """Start timing a TTS operation"""
    get_performance_tracker().start_timer(operation_id)


def end_tts_timer(operation_id: str) -> float:
    """End timing and return duration"""
    return get_performance_tracker().end_timer(operation_id)


def record_tts_performance(
    duration: float,
    text_length: int,
    audio_duration: float,
    format: str,
    backend: str,
    voice: str,
    token_estimate: Optional[int] = None
) -> PerformanceMetric:
    """
    Record TTS performance metric.
    
    Args:
        duration: Generation time in seconds
        text_length: Length of input text in characters
        audio_duration: Duration of generated audio in seconds
        format: Audio format
        backend: Backend name
        voice: Voice name
        token_estimate: Estimated token count (will estimate if not provided)
        
    Returns:
        Performance metric
    """
    # Estimate tokens if not provided (rough approximation)
    if token_estimate is None:
        token_estimate = text_length // 4  # Rough estimate
    
    return get_performance_tracker().record_metric(
        duration=duration,
        tokens=token_estimate,
        characters=text_length,
        audio_duration=audio_duration,
        format=format,
        backend=backend,
        voice=voice
    )


def get_tts_statistics(backend: Optional[str] = None) -> Dict[str, float]:
    """Get TTS performance statistics"""
    return get_performance_tracker().get_statistics(backend)

#
# End of performance.py
#######################################################################################################################