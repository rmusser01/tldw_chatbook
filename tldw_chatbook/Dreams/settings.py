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
}


def dreams_setting(key: str, default: Any = None) -> Any:
    if default is None and key in DREAMS_DEFAULTS:
        default = DREAMS_DEFAULTS[key]
    return get_cli_setting("dreams", key, default)
