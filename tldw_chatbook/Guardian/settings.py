"""``[guardian]`` config defaults and accessor. Spec §Settings; off by default.

Mirrors ``Dreams/settings.py``: ``enabled`` is FIRST and ``False`` (ADR-204
contract 1 -- a default install runs nothing, records nothing), and every
read goes live through ``get_cli_setting`` so a config-file change applies
without a restart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw_chatbook.config import get_cli_setting

GUARDIAN_DEFAULTS: dict[str, Any] = {
    # Off by default: no storage, no checks, no latency (ADR-204 contract 1).
    "enabled": False,
    "alert_retention_days": 180,
    "fixation_share_threshold": 0.6,
    "fixation_window_days": 7,
    "fixation_min_hits": 30,
    "doomloop_hits_per_day": 20,
}


def guardian_setting(key: str, default: Any = None) -> Any:
    """Read one ``[guardian]`` key live from the CLI config.

    Args:
        key: Setting name (see ``GUARDIAN_DEFAULTS``).
        default: Fallback when the key is unset; ``None`` means the
            ``GUARDIAN_DEFAULTS`` value for ``key``.

    Returns:
        The configured value, else the default.
    """
    if default is None and key in GUARDIAN_DEFAULTS:
        default = GUARDIAN_DEFAULTS[key]
    return get_cli_setting("guardian", key, default)


def guardian_db_path() -> Path:
    """Return the canonical path for the Guardian database.

    Dreams precedent (``config.get_dreams_db_path``): a custom
    ``guardian_db_path`` override wins, else ``guardian.sqlite`` under the
    current profile's user data directory. Resolved at call time so profile
    switches are honored; only ``get_guardian_db`` (enabled-gated) calls it,
    so a disabled Guardian never even resolves -- let alone creates -- the
    file.
    """
    from tldw_chatbook.config import _get_custom_database_path, get_user_data_dir

    return (
        _get_custom_database_path("guardian_db_path")
        or get_user_data_dir() / "guardian.sqlite"
    )
