from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tldw_chatbook.config import DEFAULT_CONFIG_PATH

DEFAULT_LOCAL_MCP_STORE_PATH = DEFAULT_CONFIG_PATH.parent / "local_mcp_store.json"

_ENV_PLACEHOLDER_PATTERN = re.compile(r"^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$")
_SECRET_KEY_PATTERN = re.compile(r"(secret|token|password|passwd|api[_-]?key|access[_-]?key)", re.IGNORECASE)
_SECRET_VALUE_PATTERN = re.compile(
    r"^(?:sk-[A-Za-z0-9_\-]{8,}|ghp_[A-Za-z0-9]{8,}|xox[baprs]-[A-Za-z0-9\-]{8,})$"
)


def _datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_to_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _sanitize_env(env: Mapping[str, Any] | None) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in (env or {}).items():
        key = _text(raw_key)
        value = _text(raw_value)
        if not key or not value:
            continue
        if _ENV_PLACEHOLDER_PATTERN.fullmatch(value):
            sanitized[key] = value
            continue
        if _SECRET_KEY_PATTERN.search(key) or _SECRET_VALUE_PATTERN.fullmatch(value):
            raise ValueError(f"Refusing to persist raw secret for env key '{key}'")
        sanitized[key] = value
    return sanitized


@dataclass(frozen=True)
class LocalExternalMCPProfile:
    profile_id: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id))
        object.__setattr__(self, "command", _text(self.command))
        object.__setattr__(self, "args", tuple(_text(item) for item in self.args if _text(item)))
        object.__setattr__(self, "env", _sanitize_env(self.env))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LocalExternalMCPProfile":
        if not isinstance(data, Mapping):
            return cls(profile_id="", command="")
        raw_args = data.get("args")
        args = tuple(str(item).strip() for item in raw_args) if isinstance(raw_args, list) else ()
        return cls(
            profile_id=_text(data.get("profile_id")),
            command=_text(data.get("command")),
            args=args,
            env=_coerce_mapping(data.get("env")),
            created_at=_iso_to_datetime(data.get("created_at")),
            updated_at=_iso_to_datetime(data.get("updated_at")),
        )


@dataclass(frozen=True)
class LocalGovernanceRule:
    rule_id: str
    capability_id: str
    decision: str
    notes: str | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id))
        object.__setattr__(self, "capability_id", _text(self.capability_id))
        object.__setattr__(self, "decision", _text(self.decision))
        object.__setattr__(self, "notes", _text(self.notes) or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "capability_id": self.capability_id,
            "decision": self.decision,
            "notes": self.notes,
            "updated_at": _datetime_to_iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LocalGovernanceRule":
        if not isinstance(data, Mapping):
            return cls(rule_id="", capability_id="", decision="")
        return cls(
            rule_id=_text(data.get("rule_id")),
            capability_id=_text(data.get("capability_id")),
            decision=_text(data.get("decision")),
            notes=_text(data.get("notes")) or None,
            updated_at=_iso_to_datetime(data.get("updated_at")),
        )


@dataclass(frozen=True)
class LocalMCPStoreState:
    profiles: tuple[LocalExternalMCPProfile, ...] = ()
    discovery_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    governance_rules: tuple[LocalGovernanceRule, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [profile.to_dict() for profile in self.profiles],
            "discovery_snapshots": {
                server_id: dict(snapshot)
                for server_id, snapshot in self.discovery_snapshots.items()
            },
            "governance_rules": [rule.to_dict() for rule in self.governance_rules],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LocalMCPStoreState":
        if not isinstance(data, Mapping):
            return cls()
        profiles_raw = data.get("profiles")
        governance_raw = data.get("governance_rules")
        snapshots_raw = data.get("discovery_snapshots")
        profiles = tuple(
            profile
            for profile in (
                LocalExternalMCPProfile.from_dict(item)
                for item in (profiles_raw if isinstance(profiles_raw, list) else [])
            )
            if profile.profile_id and profile.command
        )
        governance_rules = tuple(
            rule
            for rule in (
                LocalGovernanceRule.from_dict(item)
                for item in (governance_raw if isinstance(governance_raw, list) else [])
            )
            if rule.rule_id and rule.capability_id and rule.decision
        )
        discovery_snapshots = (
            {
                str(server_id): dict(snapshot)
                for server_id, snapshot in snapshots_raw.items()
                if str(server_id).strip() and isinstance(snapshot, Mapping)
            }
            if isinstance(snapshots_raw, Mapping)
            else {}
        )
        return cls(
            profiles=profiles,
            discovery_snapshots=discovery_snapshots,
            governance_rules=governance_rules,
        )


class LocalMCPStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_LOCAL_MCP_STORE_PATH)

    def load(self) -> LocalMCPStoreState:
        payload = self._read_payload()
        if not isinstance(payload, Mapping):
            return LocalMCPStoreState()
        return LocalMCPStoreState.from_dict(payload)

    def save(self, state: LocalMCPStoreState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = state.to_dict()
        payload["updated_at"] = _datetime_to_iso(datetime.now(timezone.utc))

        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        temp_path.replace(self.path)

    def list_profiles(self) -> list[LocalExternalMCPProfile]:
        return list(self.load().profiles)

    def get_profile(self, profile_id: str) -> LocalExternalMCPProfile | None:
        normalized_profile_id = _text(profile_id)
        for profile in self.list_profiles():
            if profile.profile_id == normalized_profile_id:
                return profile
        return None

    def save_profile(self, profile: LocalExternalMCPProfile) -> LocalExternalMCPProfile:
        current = self.load()
        now = datetime.now(timezone.utc)
        existing_profile = next(
            (item for item in current.profiles if item.profile_id == profile.profile_id),
            None,
        )
        saved_profile = LocalExternalMCPProfile(
            profile_id=profile.profile_id,
            command=profile.command,
            args=profile.args,
            env=profile.env,
            created_at=profile.created_at or (existing_profile.created_at if existing_profile else now),
            updated_at=now,
        )
        profiles = [item for item in current.profiles if item.profile_id != saved_profile.profile_id]
        profiles.append(saved_profile)
        self.save(
            LocalMCPStoreState(
                profiles=tuple(profiles),
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=current.governance_rules,
            )
        )
        return saved_profile

    def delete_profile(self, profile_id: str) -> bool:
        normalized_profile_id = _text(profile_id)
        current = self.load()
        profiles = [profile for profile in current.profiles if profile.profile_id != normalized_profile_id]
        if len(profiles) == len(current.profiles):
            return False
        discovery_snapshots = dict(current.discovery_snapshots)
        discovery_snapshots.pop(normalized_profile_id, None)
        self.save(
            LocalMCPStoreState(
                profiles=tuple(profiles),
                discovery_snapshots=discovery_snapshots,
                governance_rules=current.governance_rules,
            )
        )
        return True

    def save_discovery_snapshot(self, profile_id: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        normalized_profile_id = _text(profile_id)
        current = self.load()
        discovery_snapshots = dict(current.discovery_snapshots)
        discovery_snapshots[normalized_profile_id] = dict(snapshot)
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=discovery_snapshots,
                governance_rules=current.governance_rules,
            )
        )
        return discovery_snapshots[normalized_profile_id]

    def get_discovery_snapshot(self, profile_id: str) -> dict[str, Any] | None:
        return self.load().discovery_snapshots.get(_text(profile_id))

    def list_governance_rules(self) -> list[LocalGovernanceRule]:
        return list(self.load().governance_rules)

    def save_governance_rule(self, rule: LocalGovernanceRule) -> LocalGovernanceRule:
        current = self.load()
        now = datetime.now(timezone.utc)
        saved_rule = LocalGovernanceRule(
            rule_id=rule.rule_id,
            capability_id=rule.capability_id,
            decision=rule.decision,
            notes=rule.notes,
            updated_at=now,
        )
        rules = [item for item in current.governance_rules if item.rule_id != saved_rule.rule_id]
        rules.append(saved_rule)
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=tuple(rules),
            )
        )
        return saved_rule

    def _read_payload(self) -> Any:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
