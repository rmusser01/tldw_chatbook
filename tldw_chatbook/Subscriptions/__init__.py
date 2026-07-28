# __init__.py
# Subscriptions module - Content subscription and monitoring system
#
# This module provides comprehensive subscription management including:
# - RSS/Atom feed monitoring
# - URL change detection
# - Automated content ingestion
# - LLM analysis integration
# - Security features (XXE/SSRF protection)
#
# Scheduling (ADR-019, TASK-1211):
# Watchlist checks run on the unified scheduler
# (tldw_chatbook.Scheduling.scheduler.loop.SchedulerLoop) via WatchlistCheckHandler,
# which delegates to monitoring_engine below. The legacy SubscriptionScheduler,
# SubscriptionSchedulerWorker and the briefing subsystem they drove have been
# removed -- they were unreachable, and the dual-run this package's deprecation
# notice described was never implemented.
#

from .local_watchlists_service import LocalWatchlistsService
from .server_watchlists_service import ServerWatchlistsService
from .watchlist_normalizers import (
    build_watchlist_item_id,
    normalize_local_subscription_row,
    normalize_server_delete_response,
    normalize_server_watchlist_source,
    normalize_watchlist_alert_rule,
    normalize_watchlist_run,
)
from .watchlist_scope_service import WatchlistBackend, WatchlistScopeService

# Optional core subsystems (feed/URL monitoring, security, content processing).
# These are re-exported when available; noqa is needed because ruff cannot
# resolve the dynamic __all__ entries guarded by _CORE_AVAILABLE.
try:  # noqa: SIM105
    from .monitoring_engine import (  # noqa: F401
        FeedMonitor,
        URLMonitor,
        RateLimiter,
        CircuitBreaker,
        ContentExtractor,
    )
    from .security import (  # noqa: F401
        SecurityValidator,
        SSRFProtector,
        CredentialEncryptor,
        InputValidator,
    )
    from .content_processor import (  # noqa: F401
        ContentProcessor,
        KeywordExtractor,
        ContentSummarizer,
    )

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

__all__ = (
    (
        [
            # Monitoring
            "FeedMonitor",
            "URLMonitor",
            "RateLimiter",
            "CircuitBreaker",
            "ContentExtractor",
            # Security
            "SecurityValidator",
            "SSRFProtector",
            "CredentialEncryptor",
            "InputValidator",
            # Content Processing
            "ContentProcessor",
            "KeywordExtractor",
            "ContentSummarizer",
        ]
        if _CORE_AVAILABLE
        else []
    )
    + [
        # Watchlists
        "LocalWatchlistsService",
        "ServerWatchlistsService",
        "WatchlistBackend",
        "WatchlistScopeService",
        "build_watchlist_item_id",
        "normalize_local_subscription_row",
        "normalize_server_delete_response",
        "normalize_server_watchlist_source",
        "normalize_watchlist_alert_rule",
        "normalize_watchlist_run",
    ]
)

# Version info
__version__ = "1.0.0"
__author__ = "TLDW ChatBook Team"
