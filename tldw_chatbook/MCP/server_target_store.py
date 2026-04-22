from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tldw_chatbook.config import DEFAULT_CONFIG_PATH

from .unified_control_models import ConfiguredServerTarget, TargetStatusMetadata

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

    def get_target(self, server_id: str) -> ConfiguredServerTarget | None:
        normalized_server_id = str(server_id or "").strip()
        if not normalized_server_id:
            return None
        for target in self.list_targets():
            if target.server_id == normalized_server_id:
                return target
        return None

    def resolve_active_target(self, server_id: str | None = None) -> ConfiguredServerTarget | None:
        targets = self.list_targets()
        if not targets:
            return None

        if server_id is not None:
            return self.get_target(server_id)

        default_targets = [target for target in targets if target.is_default]
        if default_targets:
            return default_targets[0]

        if len(targets) == 1:
            return targets[0]

        return targets[0]

    def set_default_target(self, server_id: str) -> ConfiguredServerTarget | None:
        normalized_server_id = str(server_id or "").strip()
        if not normalized_server_id:
            return None

        targets = self.list_targets()
        updated_targets: list[ConfiguredServerTarget] = []
        selected_target: ConfiguredServerTarget | None = None

        for target in targets:
            is_default = target.server_id == normalized_server_id
            updated_target = replace(target, is_default=is_default)
            updated_targets.append(updated_target)
            if is_default:
                selected_target = updated_target

        if selected_target is None:
            return None

        self.save_targets(updated_targets)
        return selected_target

    def update_target_status(
        self,
        server_id: str,
        *,
        last_known_server_label: str | None = None,
        last_known_reachability: str | None = None,
        last_known_auth_state: str | None = None,
        last_connected_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> ConfiguredServerTarget:
        normalized_server_id = str(server_id or "").strip()
        if not normalized_server_id:
            raise KeyError("server_id is required")

        targets = self.list_targets()
        updated_targets: list[ConfiguredServerTarget] = []
        updated_target: ConfiguredServerTarget | None = None

        for target in targets:
            if target.server_id != normalized_server_id:
                updated_targets.append(target)
                continue

            status = TargetStatusMetadata(
                last_known_server_label=last_known_server_label
                if last_known_server_label is not None
                else target.last_known_server_label,
                last_known_reachability=last_known_reachability
                if last_known_reachability is not None
                else target.last_known_reachability,
                last_known_auth_state=last_known_auth_state
                if last_known_auth_state is not None
                else target.last_known_auth_state,
                last_connected_at=last_connected_at if last_connected_at is not None else target.last_connected_at,
                updated_at=updated_at if updated_at is not None else target.updated_at,
            )
            updated_target = target.with_status(status)
            updated_targets.append(updated_target)

        if updated_target is None:
            raise KeyError(f"Unknown server_id: {normalized_server_id}")

        self.save_targets(updated_targets)
        return updated_target

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
