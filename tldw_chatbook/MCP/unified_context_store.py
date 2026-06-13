from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.config import DEFAULT_CONFIG_PATH

from .unified_control_models import UnifiedMCPContext

DEFAULT_UNIFIED_MCP_CONTEXT_PATH = DEFAULT_CONFIG_PATH.parent / "unified_mcp_context.json"


class UnifiedMCPContextStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_UNIFIED_MCP_CONTEXT_PATH)

    def load(self) -> UnifiedMCPContext:
        payload = self._read_payload()
        if not isinstance(payload, dict):
            return UnifiedMCPContext()
        return UnifiedMCPContext.from_dict(payload)

    def save(self, context: UnifiedMCPContext) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            payload = context.to_dict()
            payload["updated_at"] = _datetime_to_iso(datetime.now(timezone.utc))

            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

            temp_path.replace(self.path)
        except OSError as exc:
            logger.warning(f"Unable to persist Unified MCP context to {self.path}: {exc}")

    def _read_payload(self) -> Any:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
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
