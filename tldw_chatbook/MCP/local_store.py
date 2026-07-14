from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

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


class LocalMCPStoreLoadError(RuntimeError):
    def __init__(self, path: Path, reason: Exception) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Failed to load local MCP store from '{self.path}': {reason}")


def _require_non_empty_field(value: str, field_name: str, record_type: str) -> str:
    normalized = _text(value)
    if not normalized:
        raise ValueError(f"{record_type} requires non-empty {field_name}")
    return normalized


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
class LocalApprovalRequest:
    request_id: str
    action_name: str
    resolved_action_id: str
    registry_capability_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    payload_fingerprint: str = ""
    status: str = "pending"
    matched_rule_id: str | None = None
    notes: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id))
        object.__setattr__(self, "action_name", _text(self.action_name))
        object.__setattr__(self, "resolved_action_id", _text(self.resolved_action_id))
        object.__setattr__(self, "registry_capability_id", _text(self.registry_capability_id) or None)
        object.__setattr__(self, "payload", _coerce_mapping(self.payload))
        object.__setattr__(self, "payload_fingerprint", _text(self.payload_fingerprint))
        object.__setattr__(self, "status", _text(self.status))
        object.__setattr__(self, "matched_rule_id", _text(self.matched_rule_id) or None)
        object.__setattr__(self, "notes", _text(self.notes) or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action_name": self.action_name,
            "resolved_action_id": self.resolved_action_id,
            "registry_capability_id": self.registry_capability_id,
            "payload": dict(self.payload),
            "payload_fingerprint": self.payload_fingerprint,
            "status": self.status,
            "matched_rule_id": self.matched_rule_id,
            "notes": self.notes,
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
            "resolved_at": _datetime_to_iso(self.resolved_at),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LocalApprovalRequest":
        if not isinstance(data, Mapping):
            return cls(request_id="", action_name="", resolved_action_id="")
        return cls(
            request_id=_text(data.get("request_id")),
            action_name=_text(data.get("action_name")),
            resolved_action_id=_text(data.get("resolved_action_id")),
            registry_capability_id=_text(data.get("registry_capability_id")) or None,
            payload=_coerce_mapping(data.get("payload")),
            payload_fingerprint=_text(data.get("payload_fingerprint")),
            status=_text(data.get("status")) or "pending",
            matched_rule_id=_text(data.get("matched_rule_id")) or None,
            notes=_text(data.get("notes")) or None,
            created_at=_iso_to_datetime(data.get("created_at")),
            updated_at=_iso_to_datetime(data.get("updated_at")),
            resolved_at=_iso_to_datetime(data.get("resolved_at")),
        )


@dataclass(frozen=True)
class LocalRuntimeActivity:
    activity_id: str
    action_name: str
    target: str
    ok: bool
    blocked: bool = False
    error: str | None = None
    resolved_action_id: str | None = None
    decision: str | None = None
    matched_rule_id: str | None = None
    approval_request_id: str | None = None
    approval_status: str | None = None
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity_id", _text(self.activity_id))
        object.__setattr__(self, "action_name", _text(self.action_name))
        object.__setattr__(self, "target", _text(self.target))
        object.__setattr__(self, "ok", bool(self.ok))
        object.__setattr__(self, "blocked", bool(self.blocked))
        object.__setattr__(self, "error", _text(self.error) or None)
        object.__setattr__(self, "resolved_action_id", _text(self.resolved_action_id) or None)
        object.__setattr__(self, "decision", _text(self.decision) or None)
        object.__setattr__(self, "matched_rule_id", _text(self.matched_rule_id) or None)
        object.__setattr__(self, "approval_request_id", _text(self.approval_request_id) or None)
        object.__setattr__(self, "approval_status", _text(self.approval_status) or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "action_name": self.action_name,
            "target": self.target,
            "ok": self.ok,
            "blocked": self.blocked,
            "error": self.error,
            "resolved_action_id": self.resolved_action_id,
            "decision": self.decision,
            "matched_rule_id": self.matched_rule_id,
            "approval_request_id": self.approval_request_id,
            "approval_status": self.approval_status,
            "occurred_at": _datetime_to_iso(self.occurred_at),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LocalRuntimeActivity":
        if not isinstance(data, Mapping):
            return cls(activity_id="", action_name="", target="", ok=False)
        return cls(
            activity_id=_text(data.get("activity_id")),
            action_name=_text(data.get("action_name")),
            target=_text(data.get("target")),
            ok=bool(data.get("ok")),
            blocked=bool(data.get("blocked")),
            error=_text(data.get("error")) or None,
            resolved_action_id=_text(data.get("resolved_action_id")) or None,
            decision=_text(data.get("decision")) or None,
            matched_rule_id=_text(data.get("matched_rule_id")) or None,
            approval_request_id=_text(data.get("approval_request_id")) or None,
            approval_status=_text(data.get("approval_status")) or None,
            occurred_at=_iso_to_datetime(data.get("occurred_at")),
        )


@dataclass(frozen=True)
class LocalMCPStoreState:
    profiles: tuple[LocalExternalMCPProfile, ...] = ()
    discovery_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    governance_rules: tuple[LocalGovernanceRule, ...] = ()
    approval_requests: tuple[LocalApprovalRequest, ...] = ()
    runtime_activity: tuple[LocalRuntimeActivity, ...] = ()
    profile_runtime_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [profile.to_storage_dict() for profile in self.profiles],
            "discovery_snapshots": {
                server_id: dict(snapshot)
                for server_id, snapshot in self.discovery_snapshots.items()
            },
            "governance_rules": [rule.to_dict() for rule in self.governance_rules],
            "approval_requests": [request.to_dict() for request in self.approval_requests],
            "runtime_activity": [activity.to_dict() for activity in self.runtime_activity],
            "profile_runtime_state": dict(self.profile_runtime_state),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LocalMCPStoreState":
        if not isinstance(data, Mapping):
            return cls()
        profiles_raw = data.get("profiles")
        governance_raw = data.get("governance_rules")
        approvals_raw = data.get("approval_requests")
        activity_raw = data.get("runtime_activity")
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
        approval_requests = tuple(
            request
            for request in (
                LocalApprovalRequest.from_dict(item)
                for item in (approvals_raw if isinstance(approvals_raw, list) else [])
            )
            if request.request_id and request.action_name and request.resolved_action_id and request.status
        )
        runtime_activity = tuple(
            activity
            for activity in (
                LocalRuntimeActivity.from_dict(item)
                for item in (activity_raw if isinstance(activity_raw, list) else [])
            )
            if activity.activity_id and activity.action_name
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
        runtime_state_raw = data.get("profile_runtime_state")
        profile_runtime_state = (
            {
                str(profile_id): dict(record)
                for profile_id, record in runtime_state_raw.items()
                if str(profile_id).strip() and isinstance(record, Mapping)
            }
            if isinstance(runtime_state_raw, Mapping)
            else {}
        )
        return cls(
            profiles=profiles,
            discovery_snapshots=discovery_snapshots,
            governance_rules=governance_rules,
            approval_requests=approval_requests,
            runtime_activity=runtime_activity,
            profile_runtime_state=profile_runtime_state,
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
        profile_id = _require_non_empty_field(canonical_profile.profile_id, "profile_id", "Local MCP profile")
        command = _require_non_empty_field(canonical_profile.command, "command", "Local MCP profile")
        existing_profile = next(
            (item for item in current.profiles if item.profile_id == profile_id),
            None,
        )
        saved_profile = LocalExternalMCPProfile(
            profile_id=profile_id,
            command=command,
            args=canonical_profile.args,
            env_placeholders=canonical_profile.env_placeholders,
            env_literals=canonical_profile.env_literals,
            legacy_env_literals={},
            created_at=canonical_profile.created_at or (existing_profile.created_at if existing_profile else now),
            updated_at=now,
        )
        profiles = [item for item in current.profiles if item.profile_id != saved_profile.profile_id]
        profiles.append(saved_profile)
        discovery_snapshots = dict(current.discovery_snapshots)
        profile_runtime_state = dict(current.profile_runtime_state)
        if existing_profile and self._launch_config_changed(existing_profile, saved_profile):
            discovery_snapshots.pop(saved_profile.profile_id, None)
            profile_runtime_state.pop(saved_profile.profile_id, None)
        self.save(
            LocalMCPStoreState(
                profiles=tuple(profiles),
                discovery_snapshots=discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=current.approval_requests,
                runtime_activity=current.runtime_activity,
                profile_runtime_state=profile_runtime_state,
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
        profile_runtime_state = dict(current.profile_runtime_state)
        profile_runtime_state.pop(normalized_profile_id, None)
        self.save(
            LocalMCPStoreState(
                profiles=tuple(profiles),
                discovery_snapshots=discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=current.approval_requests,
                runtime_activity=current.runtime_activity,
                profile_runtime_state=profile_runtime_state,
            )
        )
        return True

    def save_discovery_snapshot(self, profile_id: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        normalized_profile_id = _require_non_empty_field(
            profile_id,
            "profile_id",
            "Local MCP discovery snapshot",
        )
        current = self.load()
        discovery_snapshots = dict(current.discovery_snapshots)
        discovery_snapshots[normalized_profile_id] = dict(snapshot)
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=current.approval_requests,
                runtime_activity=current.runtime_activity,
                profile_runtime_state=current.profile_runtime_state,
            )
        )
        return discovery_snapshots[normalized_profile_id]

    def get_discovery_snapshot(self, profile_id: str) -> dict[str, Any] | None:
        return self.load().discovery_snapshots.get(_text(profile_id))

    def get_profile_runtime_state(self, profile_id: str) -> dict[str, Any] | None:
        """Return the persisted lifecycle-attempt record for a profile.

        Args:
            profile_id: Stable local profile identifier.

        Returns:
            The stored record dict, or None if no attempt has been recorded.
        """
        state = self.load()
        record = state.profile_runtime_state.get(_text(profile_id))
        return dict(record) if record is not None else None

    def save_profile_runtime_state(self, profile_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        """Persist the lifecycle-attempt record for a profile (last-write-wins).

        Args:
            profile_id: Stable local profile identifier.
            record: Attempt record (see module convention: last_attempt_at,
                last_action, ok, last_ok_at, last_error).

        Returns:
            The stored record dict.
        """
        normalized_profile_id = _require_non_empty_field(
            profile_id,
            "profile_id",
            "Local MCP profile runtime state",
        )
        current = self.load()
        profile_runtime_state = dict(current.profile_runtime_state)
        profile_runtime_state[normalized_profile_id] = dict(record)
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=current.approval_requests,
                runtime_activity=current.runtime_activity,
                profile_runtime_state=profile_runtime_state,
            )
        )
        return profile_runtime_state[normalized_profile_id]

    def get_catalog_bundle(self) -> dict[str, Any]:
        """Return the catalog-relevant state in one read.

        Batches the profile list, discovery snapshots, and lifecycle-attempt
        (runtime state) records that a full external-server catalog view
        needs. Callers that would otherwise issue one `load()` per profile
        for each of `get_discovery_snapshot()` and `get_profile_runtime_state()`
        (2N+1 total loads across N profiles) can use this single-`load()`
        accessor instead.

        Returns:
            Mapping with `profiles` (list of profile dicts via `to_dict()`),
            `discovery_snapshots` (profile_id -> snapshot dict), and
            `profile_runtime_state` (profile_id -> lifecycle record dict).
        """
        state = self.load()
        return {
            "profiles": [profile.to_dict() for profile in state.profiles],
            "discovery_snapshots": dict(state.discovery_snapshots),
            "profile_runtime_state": dict(state.profile_runtime_state),
        }

    def list_governance_rules(self) -> list[LocalGovernanceRule]:
        return list(self.load().governance_rules)

    def save_governance_rule(self, rule: LocalGovernanceRule) -> LocalGovernanceRule:
        current = self.load()
        now = datetime.now(timezone.utc)
        rule_id = _require_non_empty_field(rule.rule_id, "rule_id", "Local MCP governance rule")
        capability_id = _require_non_empty_field(
            rule.capability_id,
            "capability_id",
            "Local MCP governance rule",
        )
        decision = _require_non_empty_field(rule.decision, "decision", "Local MCP governance rule")
        saved_rule = LocalGovernanceRule(
            rule_id=rule_id,
            capability_id=capability_id,
            decision=decision,
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
                approval_requests=current.approval_requests,
                runtime_activity=current.runtime_activity,
                profile_runtime_state=current.profile_runtime_state,
            )
        )
        return saved_rule

    def delete_governance_rule(self, rule_id: str) -> bool:
        current = self.load()
        normalized_rule_id = _text(rule_id)
        if not normalized_rule_id:
            return False
        rules = [item for item in current.governance_rules if item.rule_id != normalized_rule_id]
        if len(rules) == len(current.governance_rules):
            return False
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=tuple(rules),
                approval_requests=current.approval_requests,
                runtime_activity=current.runtime_activity,
                profile_runtime_state=current.profile_runtime_state,
            )
        )
        return True

    def list_approval_requests(self) -> list[LocalApprovalRequest]:
        return list(self.load().approval_requests)

    def save_approval_request(self, request: LocalApprovalRequest) -> LocalApprovalRequest:
        current = self.load()
        now = datetime.now(timezone.utc)
        request_id = _require_non_empty_field(request.request_id, "request_id", "Local MCP approval request")
        action_name = _require_non_empty_field(request.action_name, "action_name", "Local MCP approval request")
        resolved_action_id = _require_non_empty_field(
            request.resolved_action_id,
            "resolved_action_id",
            "Local MCP approval request",
        )
        payload_fingerprint = _require_non_empty_field(
            request.payload_fingerprint,
            "payload_fingerprint",
            "Local MCP approval request",
        )
        status = _require_non_empty_field(request.status, "status", "Local MCP approval request")
        existing_request = next(
            (item for item in current.approval_requests if item.request_id == request_id),
            None,
        )
        saved_request = LocalApprovalRequest(
            request_id=request_id,
            action_name=action_name,
            resolved_action_id=resolved_action_id,
            registry_capability_id=request.registry_capability_id,
            payload=request.payload,
            payload_fingerprint=payload_fingerprint,
            status=status,
            matched_rule_id=request.matched_rule_id,
            notes=request.notes,
            created_at=request.created_at or (existing_request.created_at if existing_request else now),
            updated_at=now,
            resolved_at=request.resolved_at or (existing_request.resolved_at if existing_request else None),
        )
        approval_requests = [item for item in current.approval_requests if item.request_id != saved_request.request_id]
        approval_requests.append(saved_request)
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=tuple(approval_requests),
                runtime_activity=current.runtime_activity,
                profile_runtime_state=current.profile_runtime_state,
            )
        )
        return saved_request

    def resolve_approval_request(self, request_id: str, status: str) -> LocalApprovalRequest | None:
        normalized_request_id = _text(request_id)
        normalized_status = _text(status)
        if not normalized_request_id or not normalized_status:
            return None
        current = self.load()
        existing_request = next(
            (item for item in current.approval_requests if item.request_id == normalized_request_id),
            None,
        )
        if existing_request is None:
            return None
        now = datetime.now(timezone.utc)
        resolved_request = LocalApprovalRequest(
            request_id=existing_request.request_id,
            action_name=existing_request.action_name,
            resolved_action_id=existing_request.resolved_action_id,
            registry_capability_id=existing_request.registry_capability_id,
            payload=existing_request.payload,
            payload_fingerprint=existing_request.payload_fingerprint,
            status=normalized_status,
            matched_rule_id=existing_request.matched_rule_id,
            notes=existing_request.notes,
            created_at=existing_request.created_at,
            updated_at=now,
            resolved_at=now,
        )
        approval_requests = [
            item if item.request_id != normalized_request_id else resolved_request
            for item in current.approval_requests
        ]
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=tuple(approval_requests),
                runtime_activity=current.runtime_activity,
                profile_runtime_state=current.profile_runtime_state,
            )
        )
        return resolved_request

    def delete_approval_request(self, request_id: str) -> bool:
        normalized_request_id = _text(request_id)
        if not normalized_request_id:
            return False
        current = self.load()
        approval_requests = [
            item
            for item in current.approval_requests
            if item.request_id != normalized_request_id
        ]
        if len(approval_requests) == len(current.approval_requests):
            return False
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=tuple(approval_requests),
                runtime_activity=current.runtime_activity,
                profile_runtime_state=current.profile_runtime_state,
            )
        )
        return True

    def list_runtime_activity(self, limit: int = 20) -> list[dict[str, Any]]:
        normalized_limit = max(1, int(limit or 20))
        return [
            activity.to_dict()
            for activity in reversed(self.load().runtime_activity[-normalized_limit:])
        ]

    def record_runtime_activity(
        self,
        entry: Mapping[str, Any],
        limit: int = 50,
    ) -> dict[str, Any]:
        current = self.load()
        now = datetime.now(timezone.utc)
        normalized_limit = max(1, int(limit or 50))
        parsed_entry = LocalRuntimeActivity.from_dict(entry)
        action_name = _require_non_empty_field(
            parsed_entry.action_name,
            "action_name",
            "Local MCP runtime activity",
        )
        saved_entry = LocalRuntimeActivity(
            activity_id=parsed_entry.activity_id or f"activity-{uuid4().hex[:12]}",
            action_name=action_name,
            target=parsed_entry.target,
            ok=parsed_entry.ok,
            blocked=parsed_entry.blocked,
            error=parsed_entry.error,
            resolved_action_id=parsed_entry.resolved_action_id,
            decision=parsed_entry.decision,
            matched_rule_id=parsed_entry.matched_rule_id,
            approval_request_id=parsed_entry.approval_request_id,
            approval_status=parsed_entry.approval_status,
            occurred_at=parsed_entry.occurred_at or now,
        )
        runtime_activity = list(current.runtime_activity)
        runtime_activity.append(saved_entry)
        runtime_activity = runtime_activity[-normalized_limit:]
        self.save(
            LocalMCPStoreState(
                profiles=current.profiles,
                discovery_snapshots=current.discovery_snapshots,
                governance_rules=current.governance_rules,
                approval_requests=current.approval_requests,
                runtime_activity=tuple(runtime_activity),
                profile_runtime_state=current.profile_runtime_state,
            )
        )
        return saved_entry.to_dict()

    def _read_payload(self) -> Any:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LocalMCPStoreLoadError(self.path, exc) from exc

    def _launch_config_changed(
        self,
        existing_profile: LocalExternalMCPProfile,
        updated_profile: LocalExternalMCPProfile,
    ) -> bool:
        return self._launch_config_signature(existing_profile) != self._launch_config_signature(updated_profile)

    def _launch_config_signature(
        self,
        profile: LocalExternalMCPProfile,
    ) -> tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]]:
        return (
            profile.command,
            profile.args,
            tuple(sorted(profile.env.items())),
        )
