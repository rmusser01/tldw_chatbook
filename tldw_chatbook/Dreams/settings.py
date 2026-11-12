"""[dreams] config defaults and accessor. Spec §Configuration; off by default."""
from __future__ import annotations

from typing import Any

from tldw_chatbook.config import get_cli_setting

DREAMS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "provider": None,
    "model": None,
    "stories_per_cycle": 5,
    "queries_per_cycle": 3,
    "exploration_slots": 1,
    "search_engine": "duckduckgo",
    "web_search_enabled": True,
    "region": "",
    "cadence_hours": 24,
    "catchup_enabled": True,
    "watchlist_freshness_hours": 48,
    "seen_item_ttl_days": 90,
    "max_searches_per_day": 30,
    "max_llm_calls_per_day": 60,
    # --- Track loop caps/floors (Phase 2 Task 6); the single source the
    # track service reads -- its pre-Task-6 fallback literals are gone.
    "tracked_item_cap": 20,
    "track_min_check_interval_hours": 12,
    "track_quiet_retire_count": 14,
}


def dreams_setting(key: str, default: Any = None) -> Any:
    """Read one ``[dreams]`` key live from the CLI config.

    Args:
        key: Setting name (see ``DREAMS_DEFAULTS``).
        default: Fallback when the key is unset; ``None`` means the
            ``DREAMS_DEFAULTS`` value for ``key``.

    Returns:
        The configured value, else the default.
    """
    if default is None and key in DREAMS_DEFAULTS:
        default = DREAMS_DEFAULTS[key]
    return get_cli_setting("dreams", key, default)


def dreams_region() -> str:
    """The configured ``[dreams] region``, or ``""`` when it is not usable.

    The region goes verbatim into the query-synthesis prompt, a tracked
    question's search query and a story's location, and it can reach the
    config through Settings, Advanced Config or a text editor. Every read
    therefore goes through here and applies the rule a topic or goal must
    pass (review of #2921, item 8): one visible line, bounded, with no
    control or invisible formatting characters. A value that fails is
    treated as no region rather than passed on.

    Returns:
        The trimmed region, or an empty string when unset or refused.
    """
    # Imported at call time: this module is on the boot path and the
    # validator's module is not this one's to load there.
    from tldw_chatbook.Utils.input_validation import validate_dreams_profile_text

    try:
        return validate_dreams_profile_text(dreams_setting("region", ""))
    except ValueError:
        return ""


def dreams_execution_allowed(database: Any) -> bool:
    """Require current recovery authority for this selected runtime store."""
    from tldw_chatbook.Backup_Recovery.activation import execution_allowed

    return execution_allowed(("db.dreams",), getattr(database, "db_path", None))
