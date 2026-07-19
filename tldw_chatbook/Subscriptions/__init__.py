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
# Deprecation notice (ADR-019):
# The legacy scheduling symbols SubscriptionScheduler, TextualSchedulerWorker,
# and create_scheduler are deprecated. They remain accessible by direct attribute
# access (e.g., ``from tldw_chatbook.Subscriptions import SubscriptionScheduler``)
# for backward compatibility during the dual-run validation period, but are no
# longer exported by ``from tldw_chatbook.Subscriptions import *``. Watchlist
# checks are migrating to the unified Scheduling scheduler
# (tldw_chatbook.Scheduling.scheduler.loop.SchedulerLoop).
#

from typing import Any

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

# Optional briefing generation subsystem.
try:  # noqa: SIM105
    from .briefing_generator import BriefingGenerator, BriefingSchedule  # noqa: F401
    _BRIEFING_AVAILABLE = True
except ImportError:
    BriefingGenerator = None  # type: ignore[assignment,misc]
    BriefingSchedule = None  # type: ignore[assignment,misc]
    _BRIEFING_AVAILABLE = False

__all__ = (
    ([
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
    ] if _CORE_AVAILABLE else [])
    + ([
        # Briefing Generation
        "BriefingGenerator",
        "BriefingSchedule",
    ] if _BRIEFING_AVAILABLE else [])
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
__version__ = '1.0.0'
__author__ = 'TLDW ChatBook Team'

# Legacy scheduler symbols are loaded lazily so that importing the package does
# not emit deprecation warnings for consumers that do not need them.
_DEPRECATED_SCHEDULER_SYMBOLS = {
    "SubscriptionScheduler",
    "TextualSchedulerWorker",
    "create_scheduler",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_SCHEDULER_SYMBOLS:
        from . import scheduler
        return getattr(scheduler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
