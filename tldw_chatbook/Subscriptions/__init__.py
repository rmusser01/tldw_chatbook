# __init__.py
# Subscriptions module - Content subscription and monitoring system
#
# This module provides comprehensive subscription management including:
# - RSS/Atom feed monitoring
# - URL change detection
# - Automated content ingestion
# - LLM analysis integration
# - Briefing generation
# - Security features (XXE/SSRF protection)
#

# Try importing dependencies - some may be optional
try:
    from .monitoring_engine import FeedMonitor, URLMonitor, RateLimiter, CircuitBreaker, ContentExtractor
    from .security import SecurityValidator, SSRFProtector, CredentialEncryptor, InputValidator
    from .scheduler import SubscriptionScheduler, TextualSchedulerWorker, create_scheduler
    from .content_processor import ContentProcessor, KeywordExtractor, ContentSummarizer
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

# These require optional dependencies
try:
    from .briefing_generator import BriefingGenerator, BriefingSchedule
    _BRIEFING_AVAILABLE = True
except ImportError:
    BriefingGenerator = None
    BriefingSchedule = None
    _BRIEFING_AVAILABLE = False

__all__ = []

if _CORE_AVAILABLE:
    __all__.extend([
        # Monitoring
        'FeedMonitor',
        'URLMonitor', 
        'RateLimiter',
        'CircuitBreaker',
        'ContentExtractor',
        
        # Security
        'SecurityValidator',
        'SSRFProtector',
        'CredentialEncryptor',
        'InputValidator',
        
        # Scheduling
        'SubscriptionScheduler',
        'TextualSchedulerWorker',
        'create_scheduler',
        
        # Content Processing
        'ContentProcessor',
        'KeywordExtractor',
        'ContentSummarizer',
    ])

if _BRIEFING_AVAILABLE:
    __all__.extend([
        # Briefing Generation
        'BriefingGenerator',
        'BriefingSchedule',
    ])

# Version info
__version__ = '1.0.0'
__author__ = 'TLDW ChatBook Team'