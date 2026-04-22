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
_SECRET_VALUE_PATTERNS = (
    re.compile(r"^sk-[A-Za-z0-9._-]{12,}$"),
    re.compile(r"^gh[pousr]_[A-Za-z0-9]{12,}$"),
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]{12,}$"),
    re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+$"),
)
_SAFE_LITERAL_VALUES = {
    "0",
    "1",
    "true",
    "false",
    "yes",
    "no",
    "on",
    "off",
    "enabled",
    "disabled",
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "critical",
    "trace",
}
_SAFE_INTEGER_PATTERN = re.compile(r"^[0-9]{1,5}$")
_SAFE_DECIMAL_PATTERN = re.compile(r"^[0-9]{1,4}\.[0-9]{1,2}$")
_SAFE_URL_LITERAL_PATTERN = re.compile(r"^https?://[A-Za-z0-9.-]+(?::[0-9]{1,5})?(?:/[^\s]*)?$", re.IGNORECASE)
_LEGACY_SAFE_URL_LITERAL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^\s]{1,255}$")
_LEGACY_SAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LEGACY_SAFE_PATH_PATTERN = re.compile(r"^(?:~|/)[A-Za-z0-9._/@:+-]{1,255}$")


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
        if _SECRET_KEY_PATTERN.search(key):
            raise ValueError(f"Secret-bearing env key '{key}' must use a placeholder")
        sanitized[key] = value
    return sanitized


def _is_secret_bearing_env_key(key: str) -> bool:
    return bool(_SECRET_KEY_PATTERN.search(key))


def _looks_like_raw_secret_value(value: str) -> bool:
    return any(pattern.fullmatch(value) for pattern in _SECRET_VALUE_PATTERNS)


def _is_safe_literal_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _SAFE_LITERAL_VALUES:
        return True
    if _SAFE_INTEGER_PATTERN.fullmatch(normalized):
        return True
    if _SAFE_DECIMAL_PATTERN.fullmatch(normalized):
        return True
    if _SAFE_URL_LITERAL_PATTERN.fullmatch(value.strip()):
        return True
    return False


def _coerce_legacy_env(legacy_env: Mapping[str, Any] | None) -> tuple[dict[str, str], dict[str, str]]:
    placeholders: dict[str, str] = {}
    literals: dict[str, str] = {}
    for raw_key, raw_value in (legacy_env or {}).items():
        key = _text(raw_key)
        value = _text(raw_value)
        if not key or not value:
            continue
        if _ENV_PLACEHOLDER_PATTERN.fullmatch(value):
            placeholders[key] = value
            continue
        if _is_secret_bearing_env_key(key):
            continue
        if _looks_like_raw_secret_value(value):
            continue
        if _is_legacy_safe_literal_value(value):
            literals[key] = value
    return placeholders, literals


def _is_legacy_safe_literal_value(value: str) -> bool:
    stripped = value.strip()
    if _is_safe_literal_value(stripped):
        return True
    if _LEGACY_SAFE_URL_LITERAL_PATTERN.fullmatch(stripped):
        return True
    if _LEGACY_SAFE_PATH_PATTERN.fullmatch(stripped):
        return True
    if _LEGACY_SAFE_TOKEN_PATTERN.fullmatch(stripped):
        return True
    return False


def _sanitize_legacy_env_literals(env: Mapping[str, Any] | None) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in (env or {}).items():
        key = _text(raw_key)
        value = _text(raw_value)
        if not key or not value:
            continue
        if _is_secret_bearing_env_key(key):
            continue
        if _looks_like_raw_secret_value(value):
            continue
        if _is_legacy_safe_literal_value(value):
            sanitized[key] = value
    return sanitized


def _sanitize_env_placeholders(env: Mapping[str, Any] | None) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in (env or {}).items():
        key = _text(raw_key)
        value = _text(raw_value)
        if not key or not value:
            continue
        if not _ENV_PLACEHOLDER_PATTERN.fullmatch(value):
            raise ValueError(f"Env placeholder '{key}' must use $NAME or ${'{'}NAME{'}'} syntax")
        sanitized[key] = value
    return sanitized


def _sanitize_env_literals(env: Mapping[str, Any] | None) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in (env or {}).items():
        key = _text(raw_key)
        value = _text(raw_value)
        if not key or not value:
            continue
        if _is_secret_bearing_env_key(key):
            raise ValueError(f"Secret-bearing env key '{key}' cannot be stored as a literal")
        if _looks_like_raw_secret_value(value):
            raise ValueError(f"Literal env key '{key}' looks like a raw secret and cannot be persisted")
        if not _is_safe_literal_value(value):
            raise ValueError(
                f"Literal env key '{key}' must use an explicit safe operational literal or an env placeholder"
            )
        sanitized[key] = value
    return sanitized


@dataclass(frozen=True)
class LocalExternalMCPProfile:
    profile_id: str
    command: str
    args: tuple[str, ...] = ()
    env_placeholders: dict[str, str] = field(default_factory=dict)
    env_literals: dict[str, str] = field(default_factory=dict)
    legacy_env_literals: dict[str, str] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id))
        object.__setattr__(self, "command", _text(self.command))
        object.__setattr__(self, "args", tuple(_text(item) for item in self.args if _text(item)))
        env_placeholders = _sanitize_env_placeholders(self.env_placeholders)
        env_literals = _sanitize_env_literals(self.env_literals)
        legacy_env_literals = _sanitize_legacy_env_literals(self.legacy_env_literals)
        duplicate_keys = (set(env_placeholders) & set(env_literals)) | (set(env_placeholders) & set(legacy_env_literals)) | (set(env_literals) & set(legacy_env_literals))
        if duplicate_keys:
            raise ValueError(f"Duplicate env keys across placeholders and literals: {sorted(duplicate_keys)}")
        object.__setattr__(self, "env_placeholders", env_placeholders)
        object.__setattr__(self, "env_literals", env_literals)
        object.__setattr__(self, "legacy_env_literals", legacy_env_literals)

    @property
    def env(self) -> dict[str, str]:
        merged = dict(self.legacy_env_literals)
        merged.update(self.env_literals)
        merged.update(self.env_placeholders)
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "command": self.command,
            "args": list(self.args),
            "env": self.env,
            "env_placeholders": dict(self.env_placeholders),
            "env_literals": dict(self.env_literals),
            "legacy_env_literals": dict(self.legacy_env_literals),
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
        }

    def to_storage_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "command": self.command,
            "args": list(self.args),
            "env_placeholders": dict(self.env_placeholders),
            "env_literals": dict(self.env_literals),
            "legacy_env_literals": dict(self.legacy_env_literals),
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
        }

    def to_input_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "command": self.command,
            "args": list(self.args),
            "env_placeholders": dict(self.env_placeholders),
            "env_literals": dict(self.env_literals),
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
        }

    @classmethod
    def from_input_dict(cls, data: Any) -> "LocalExternalMCPProfile":
        if not isinstance(data, Mapping):
            return cls(profile_id="", command="")
        raw_args = data.get("args")
        args = tuple(str(item).strip() for item in raw_args) if isinstance(raw_args, list) else ()
        return cls(
            profile_id=_text(data.get("profile_id")),
            command=_text(data.get("command")),
            args=args,
            env_placeholders=_coerce_mapping(data.get("env_placeholders")),
            env_literals=_coerce_mapping(data.get("env_literals")),
            created_at=_iso_to_datetime(data.get("created_at")),
            updated_at=_iso_to_datetime(data.get("updated_at")),
        )

    @classmethod
    def from_storage_dict(cls, data: Any) -> "LocalExternalMCPProfile":
        if not isinstance(data, Mapping):
            return cls(profile_id="", command="")
        raw_args = data.get("args")
        args = tuple(str(item).strip() for item in raw_args) if isinstance(raw_args, list) else ()
        raw_env_placeholders = _coerce_mapping(data.get("env_placeholders"))
        raw_env_literals = _coerce_mapping(data.get("env_literals"))
        raw_legacy_env_literals = _coerce_mapping(data.get("legacy_env_literals"))
        legacy_env = _coerce_mapping(data.get("env"))
        if legacy_env:
            legacy_placeholders, legacy_literals = _coerce_legacy_env(legacy_env)
            for key, value in legacy_placeholders.items():
                raw_env_placeholders.setdefault(key, value)
            for key, value in legacy_literals.items():
                raw_legacy_env_literals.setdefault(key, value)
        return cls(
            profile_id=_text(data.get("profile_id")),
            command=_text(data.get("command")),
            args=args,
            env_placeholders=raw_env_placeholders,
            env_literals=raw_env_literals,
            legacy_env_literals=raw_legacy_env_literals,
            created_at=_iso_to_datetime(data.get("created_at")),
            updated_at=_iso_to_datetime(data.get("updated_at")),
        )

    @classmethod
    def from_dict(cls, data: Any) -> "LocalExternalMCPProfile":
        return cls.from_input_dict(data)


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
            "profiles": [profile.to_storage_dict() for profile in self.profiles],
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
                LocalExternalMCPProfile.from_storage_dict(item)
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
        canonical_profile = LocalExternalMCPProfile.from_input_dict(profile.to_input_dict())
        existing_profile = next(
            (item for item in current.profiles if item.profile_id == canonical_profile.profile_id),
            None,
        )
        saved_profile = LocalExternalMCPProfile(
            profile_id=canonical_profile.profile_id,
            command=canonical_profile.command,
            args=canonical_profile.args,
            env_placeholders=canonical_profile.env_placeholders,
            env_literals=canonical_profile.env_literals,
            legacy_env_literals={},
            created_at=canonical_profile.created_at or (existing_profile.created_at if existing_profile else now),
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
