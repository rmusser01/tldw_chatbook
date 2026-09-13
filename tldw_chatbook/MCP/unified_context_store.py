from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.Backup_Recovery import mcp_source_participants as mcp_sources

from .unified_control_models import UnifiedMCPContext

_UNIFIED_MCP_CONTEXT_FILENAME = "unified_mcp_context.json"


def _default_unified_mcp_context_path() -> Path:
    """Return this store's path when constructed with no explicit argument.

    Derives from ``config.get_user_data_dir()`` -- the same directory every
    real construction site (``app.py``) already passes explicitly -- rather
    than a stale, eagerly-computed module constant. See
    ``local_store._default_local_mcp_store_path`` for why this is resolved
    lazily instead of baked in at import time (TASK-855).

    Returns:
        ``get_user_data_dir() / "unified_mcp_context.json"``.
    """
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / _UNIFIED_MCP_CONTEXT_FILENAME


class UnifiedMCPContextStore:
    @mcp_sources.guarded
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _default_unified_mcp_context_path()
        from .recovery_activation import select

        select(self)

    @mcp_sources.guarded
    def load(self) -> UnifiedMCPContext:
        from .recovery_activation import readable

        if not readable(self, "mcp.context"):
            return UnifiedMCPContext()
        payload = self._read_payload()
        if not isinstance(payload, dict):
            return UnifiedMCPContext()
        return UnifiedMCPContext.from_dict(payload)

    @mcp_sources.guarded
    def save(self, context: UnifiedMCPContext) -> None:
        from .recovery_activation import require_store_write

        require_store_write(self, "mcp.context")
        try:
            payload = context.to_dict()
            payload["updated_at"] = _datetime_to_iso(datetime.now(timezone.utc))
            mcp_sources.write_json(self, payload)
        except OSError as exc:
            logger.warning(
                f"Unable to persist Unified MCP context to {self.path}: {exc}"
            )

    @mcp_sources.guarded
    def _read_payload(self) -> Any:
        try:
            with mcp_sources.reader(self) as handle:
                return json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}


def _datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
