from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tldw_chatbook.config import DEFAULT_CONFIG_PATH

from .unified_control_models import ConfiguredServerTarget

DEFAULT_SERVER_TARGETS_PATH = DEFAULT_CONFIG_PATH.parent / "mcp_server_targets.json"


class ConfiguredServerTargetStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_SERVER_TARGETS_PATH)

    def load(self) -> list[ConfiguredServerTarget]:
        payload = self._read_payload()
        if not isinstance(payload, list):
            payload = payload.get("targets") if isinstance(payload, Mapping) else []
        if not isinstance(payload, list):
            return []
        return [ConfiguredServerTarget.from_dict(item) for item in payload if isinstance(item, Mapping)]

    def list_targets(self) -> list[ConfiguredServerTarget]:
        return self.load()

    def save_targets(self, targets: Sequence[ConfiguredServerTarget]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = {
            "targets": [target.to_dict() for target in targets],
            "updated_at": _datetime_to_iso(datetime.now(timezone.utc)),
        }

        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        temp_path.replace(self.path)

    def bootstrap_from_legacy_config(self, app_config: Mapping[str, Any] | None) -> bool:
        if self.list_targets():
            return False

        target = ConfiguredServerTarget.from_legacy_tldw_api_config(app_config or {})
        if target is None:
            return False

        self.save_targets([target])
        return True

    def _read_payload(self) -> Any:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return []
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []


def _datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
