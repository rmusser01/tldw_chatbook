# dictation_metrics.py
"""
Performance monitoring and metrics for dictation functionality.
Tracks accuracy, performance, and usage patterns.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics
from loguru import logger

from ..config import get_cli_setting, save_setting_to_cli_config


@dataclass
class DictationMetrics:
    """Metrics for a single dictation session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    words_transcribed: int = 0
    audio_bytes_processed: int = 0
    
    # Performance metrics
    initialization_time: float = 0.0
    first_word_latency: float = 0.0
    average_latency: float = 0.0
    processing_gaps: List[float] = field(default_factory=list)
    
    # Quality metrics
    corrections_made: int = 0
    commands_detected: int = 0
    silence_periods: int = 0
    
    # Resource metrics
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Provider info
    provider: str = ""
    model: str = ""
    language: str = ""
    
    def calculate_wpm(self) -> float:
        """Calculate words per minute."""
        if self.duration > 0:
            return (self.words_transcribed / self.duration) * 60
        return 0.0
    
    def calculate_efficiency(self) -> float:
        """Calculate processing efficiency (0-1)."""
        if not self.processing_gaps:
            return 1.0
        
        # Efficiency based on processing gaps
        avg_gap = statistics.mean(self.processing_gaps)
        # Consider < 100ms gaps as efficient
        return min(1.0, 100.0 / max(avg_gap, 1.0))


class DictationPerformanceMonitor:
    """
    Monitors dictation performance and provides analytics.
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of recent sessions to keep
        """
        self.history_size = history_size
        self.sessions: deque[DictationMetrics] = deque(maxlen=history_size)
        self.current_session: Optional[DictationMetrics] = None
        
        # Real-time metrics
        self.latency_buffer: deque[float] = deque(maxlen=50)
        self.last_process_time = 0.0
        
        # Load saved statistics
        self._load_statistics()
    
    def start_session(
        self,
        session_id: str,
        provider: str,
        model: str = "",
        language: str = "en"
    ) -> DictationMetrics:
        """Start monitoring a new session."""
        self.current_session = DictationMetrics(
            session_id=session_id,
            start_time=datetime.now(),
            provider=provider,
            model=model,
            language=language
        )
        
        # Track initialization start
        self._init_start_time = time.perf_counter()
        
        return self.current_session
    
    def mark_initialized(self):
        """Mark that initialization is complete."""
        if self.current_session and self._init_start_time:
            self.current_session.initialization_time = (
                time.perf_counter() - self._init_start_time
            )
    
    def mark_first_word(self):
        """Mark when first word is transcribed."""
        if self.current_session and self.current_session.first_word_latency == 0:
            self.current_session.first_word_latency = (
                time.perf_counter() - self._init_start_time
            )
    
    def record_transcription(
        self,
        text: str,
        processing_time: float,
        audio_size: int = 0
    ):
        """Record a transcription event."""
        if not self.current_session:
            return
        
        # Update word count
        word_count = len(text.split())
        self.current_session.words_transcribed += word_count
        
        # Track first word
        if word_count > 0 and self.current_session.first_word_latency == 0:
            self.mark_first_word()
        
        # Update audio processed
        self.current_session.audio_bytes_processed += audio_size
        
        # Track latency
        self.latency_buffer.append(processing_time)
        
        # Track processing gaps
        current_time = time.perf_counter()
        if self.last_process_time > 0:
            gap = (current_time - self.last_process_time) * 1000  # Convert to ms
            self.current_session.processing_gaps.append(gap)
        self.last_process_time = current_time
    
    def record_correction(self):
        """Record that a correction was made."""
        if self.current_session:
            self.current_session.corrections_made += 1
    
    def record_command(self, command: str):
        """Record a voice command detection."""
        if self.current_session:
            self.current_session.commands_detected += 1
    
    def record_silence(self):
        """Record a silence period."""
        if self.current_session:
            self.current_session.silence_periods += 1
    
    def record_resource_usage(self, cpu_percent: float, memory_mb: float):
        """Record resource usage."""
        if self.current_session:
            self.current_session.peak_cpu_percent = max(
                self.current_session.peak_cpu_percent,
                cpu_percent
            )
            self.current_session.peak_memory_mb = max(
                self.current_session.peak_memory_mb,
                memory_mb
            )
    
    def end_session(self) -> Optional[DictationMetrics]:
        """End the current session and return metrics."""
        if not self.current_session:
            return None
        
        # Finalize metrics
        self.current_session.end_time = datetime.now()
        self.current_session.duration = (
            self.current_session.end_time - self.current_session.start_time
        ).total_seconds()
        
        # Calculate average latency
        if self.latency_buffer:
            self.current_session.average_latency = statistics.mean(self.latency_buffer)
        
        # Add to history
        self.sessions.append(self.current_session)
        
        # Save statistics
        self._save_statistics()
        
        # Return and clear
        metrics = self.current_session
        self.current_session = None
        self.latency_buffer.clear()
        self.last_process_time = 0.0
        
        return metrics
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of recent sessions."""
        if not self.sessions:
            return {
                'total_sessions': 0,
                'total_words': 0,
                'total_duration': 0,
                'average_wpm': 0,
                'average_latency': 0,
                'average_efficiency': 0,
                'providers_used': {}
            }
        
        # Aggregate metrics
        total_words = sum(s.words_transcribed for s in self.sessions)
        total_duration = sum(s.duration for s in self.sessions)
        total_latency = sum(s.average_latency for s in self.sessions if s.average_latency > 0)
        latency_count = sum(1 for s in self.sessions if s.average_latency > 0)
        
        # Provider usage
        provider_counts = {}
        for session in self.sessions:
            provider_counts[session.provider] = provider_counts.get(session.provider, 0) + 1
        
        # Calculate averages
        avg_wpm = (total_words / total_duration * 60) if total_duration > 0 else 0
        avg_latency = (total_latency / latency_count) if latency_count > 0 else 0
        avg_efficiency = statistics.mean(s.calculate_efficiency() for s in self.sessions)
        
        return {
            'total_sessions': len(self.sessions),
            'total_words': total_words,
            'total_duration': total_duration,
            'average_wpm': avg_wpm,
            'average_latency': avg_latency,
            'average_efficiency': avg_efficiency,
            'providers_used': provider_counts,
            'recent_sessions': [
                {
                    'timestamp': s.start_time.isoformat(),
                    'duration': s.duration,
                    'words': s.words_transcribed,
                    'wpm': s.calculate_wpm(),
                    'provider': s.provider
                }
                for s in list(self.sessions)[-10:]  # Last 10 sessions
            ]
        }
    
    def get_provider_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across providers."""
        provider_metrics = {}
        
        for session in self.sessions:
            provider = session.provider
            if provider not in provider_metrics:
                provider_metrics[provider] = {
                    'sessions': [],
                    'total_words': 0,
                    'total_duration': 0,
                    'latencies': []
                }
            
            provider_metrics[provider]['sessions'].append(session)
            provider_metrics[provider]['total_words'] += session.words_transcribed
            provider_metrics[provider]['total_duration'] += session.duration
            if session.average_latency > 0:
                provider_metrics[provider]['latencies'].append(session.average_latency)
        
        # Calculate averages per provider
        comparison = {}
        for provider, data in provider_metrics.items():
            if data['total_duration'] > 0:
                comparison[provider] = {
                    'session_count': len(data['sessions']),
                    'average_wpm': (data['total_words'] / data['total_duration']) * 60,
                    'average_latency': statistics.mean(data['latencies']) if data['latencies'] else 0,
                    'average_efficiency': statistics.mean(
                        s.calculate_efficiency() for s in data['sessions']
                    )
                }
        
        return comparison
    
    def get_time_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance over time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_sessions = [s for s in self.sessions if s.start_time > cutoff_time]
        
        if not recent_sessions:
            return {'no_data': True}
        
        # Group by hour
        hourly_stats = {}
        for session in recent_sessions:
            hour_key = session.start_time.strftime("%Y-%m-%d %H:00")
            if hour_key not in hourly_stats:
                hourly_stats[hour_key] = {
                    'sessions': 0,
                    'words': 0,
                    'duration': 0
                }
            
            hourly_stats[hour_key]['sessions'] += 1
            hourly_stats[hour_key]['words'] += session.words_transcribed
            hourly_stats[hour_key]['duration'] += session.duration
        
        return {
            'hourly_stats': hourly_stats,
            'peak_usage_hour': max(hourly_stats.items(), key=lambda x: x[1]['sessions'])[0],
            'total_sessions': len(recent_sessions),
            'total_words': sum(s.words_transcribed for s in recent_sessions)
        }
    
    def _load_statistics(self):
        """Load saved statistics from config."""
        stats_data = get_cli_setting('dictation.performance_stats', [])
        
        for stat_dict in stats_data[-self.history_size:]:
            try:
                # Convert dict back to DictationMetrics
                stat_dict['start_time'] = datetime.fromisoformat(stat_dict['start_time'])
                if stat_dict.get('end_time'):
                    stat_dict['end_time'] = datetime.fromisoformat(stat_dict['end_time'])
                
                metrics = DictationMetrics(**stat_dict)
                self.sessions.append(metrics)
            except Exception as e:
                logger.error(f"Failed to load metric: {e}")
    
    def _save_statistics(self):
        """Save statistics to config."""
        # Convert to serializable format
        stats_data = []
        for session in self.sessions:
            stat_dict = {
                'session_id': session.session_id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'duration': session.duration,
                'words_transcribed': session.words_transcribed,
                'audio_bytes_processed': session.audio_bytes_processed,
                'initialization_time': session.initialization_time,
                'first_word_latency': session.first_word_latency,
                'average_latency': session.average_latency,
                'processing_gaps': session.processing_gaps,
                'corrections_made': session.corrections_made,
                'commands_detected': session.commands_detected,
                'silence_periods': session.silence_periods,
                'provider': session.provider,
                'model': session.model,
                'language': session.language
            }
            stats_data.append(stat_dict)
        
        # Save to config
        save_setting_to_cli_config('dictation', 'performance_stats', stats_data)


# Global instance
_monitor_instance = None

def get_performance_monitor() -> DictationPerformanceMonitor:
    """Get the global performance monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = DictationPerformanceMonitor()
    return _monitor_instance