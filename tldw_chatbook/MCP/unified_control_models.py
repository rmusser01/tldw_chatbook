from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

_VALID_AUTH_MODES = {"api_key", "bearer", "custom_token"}
_VALID_REACHABILITY_STATES = {"unknown", "reachable", "unreachable"}
_VALID_AUTH_STATES = {"unknown", "authenticated", "auth_required", "session_invalid"}
_VALID_SCOPES = {"personal", "team", "org", "system_admin"}


def _datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_to_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
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


def _coerce_choice(value: Any, *, valid_values: set[str], default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in valid_values:
            return normalized
    return default


def _optional_choice(value: Any, *, valid_values: set[str]) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in valid_values:
            return normalized
    return None


def _normalize_server_identity(raw_url: str) -> tuple[str | None, str | None]:
    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.hostname:
        normalized = raw_url.rstrip("/") or None
        return normalized, normalized

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError:
        return None, None
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)

    netloc = hostname
    if port and not default_port:
        netloc = f"{hostname}:{port}"

    path = parsed.path.rstrip("/")
    normalized = urlunsplit((scheme, netloc, path, "", ""))
    return normalized, netloc


@dataclass(frozen=True)
class TargetStatusMetadata:
    last_known_server_label: str | None = None
    last_known_reachability: str | None = None
    last_known_auth_state: str | None = None
    last_connected_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_known_server_label": self.last_known_server_label,
            "last_known_reachability": self.last_known_reachability,
            "last_known_auth_state": self.last_known_auth_state,
            "last_connected_at": _datetime_to_iso(self.last_connected_at),
            "updated_at": _datetime_to_iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TargetStatusMetadata":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            last_known_server_label=_text_or_none(data.get("last_known_server_label")),
            last_known_reachability=_optional_choice(
                data.get("last_known_reachability"),
                valid_values=_VALID_REACHABILITY_STATES,
            ),
            last_known_auth_state=_optional_choice(
                data.get("last_known_auth_state"),
                valid_values=_VALID_AUTH_STATES,
            ),
            last_connected_at=_iso_to_datetime(data.get("last_connected_at")),
            updated_at=_iso_to_datetime(data.get("updated_at")),
        )


@dataclass(frozen=True)
class SectionCapabilityFlags:
    overview: bool = False
    inventory: bool = False
    catalogs: bool = False
    external_servers: bool = False
    governance: bool = False
    advanced: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "overview": self.overview,
            "inventory": self.inventory,
            "catalogs": self.catalogs,
            "external_servers": self.external_servers,
            "governance": self.governance,
            "advanced": self.advanced,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SectionCapabilityFlags":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            overview=bool(data.get("overview", False)),
            inventory=bool(data.get("inventory", False)),
            catalogs=bool(data.get("catalogs", False)),
            external_servers=bool(data.get("external_servers", False)),
            governance=bool(data.get("governance", False)),
            advanced=bool(data.get("advanced", False)),
        )


@dataclass(frozen=True)
class NormalizedPanelRecord:
    panel_id: str
    section: str | None = None
    title: str | None = None
    is_visible: bool = True
    is_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "section": self.section,
            "title": self.title,
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "NormalizedPanelRecord":
        if not isinstance(data, Mapping):
            return cls(panel_id="")
        raw_metadata = data.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        return cls(
            panel_id=str(data.get("panel_id") or "").strip(),
            section=_text_or_none(data.get("section")),
            title=_text_or_none(data.get("title")),
            is_visible=bool(data.get("is_visible", True)),
            is_enabled=bool(data.get("is_enabled", True)),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ConfiguredServerTarget:
    server_id: str
    label: str
    base_url: str
    auth_mode: str = "api_key"
    auth_reference: str | None = None
    is_default: bool = False
    last_known_server_label: str | None = None
    last_known_reachability: str | None = None
    last_known_auth_state: str | None = None
    last_connected_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        normalized_base_url = self.base_url.rstrip("/")
        object.__setattr__(self, "base_url", normalized_base_url)
        object.__setattr__(self, "auth_mode", _coerce_choice(self.auth_mode, valid_values=_VALID_AUTH_MODES, default="api_key"))
        object.__setattr__(
            self,
            "last_known_reachability",
            _optional_choice(self.last_known_reachability, valid_values=_VALID_REACHABILITY_STATES),
        )
        object.__setattr__(
            self,
            "last_known_auth_state",
            _optional_choice(self.last_known_auth_state, valid_values=_VALID_AUTH_STATES),
        )

    def with_status(self, status: TargetStatusMetadata) -> "ConfiguredServerTarget":
        return replace(
            self,
            last_known_server_label=status.last_known_server_label,
            last_known_reachability=status.last_known_reachability,
            last_known_auth_state=status.last_known_auth_state,
            last_connected_at=status.last_connected_at,
            updated_at=status.updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "label": self.label,
            "base_url": self.base_url,
            "auth_mode": self.auth_mode,
            "auth_reference": self.auth_reference,
            "is_default": self.is_default,
            "last_known_server_label": self.last_known_server_label,
            "last_known_reachability": self.last_known_reachability,
            "last_known_auth_state": self.last_known_auth_state,
            "last_connected_at": _datetime_to_iso(self.last_connected_at),
            "updated_at": _datetime_to_iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ConfiguredServerTarget":
        if not isinstance(data, Mapping):
            return cls(server_id="", label="", base_url="")

        status = TargetStatusMetadata.from_dict(data.get("status"))
        base_url = _text_or_none(data.get("base_url")) or ""
        return cls(
            server_id=_text_or_none(data.get("server_id")) or "",
            label=_text_or_none(data.get("label")) or "",
            base_url=base_url,
            auth_mode=_coerce_choice(
                data.get("auth_mode"),
                valid_values=_VALID_AUTH_MODES,
                default="api_key",
            ),
            auth_reference=_text_or_none(data.get("auth_reference")),
            is_default=bool(data.get("is_default", False)),
            last_known_server_label=_text_or_none(data.get("last_known_server_label")) or status.last_known_server_label,
            last_known_reachability=_optional_choice(
                data.get("last_known_reachability")
                if data.get("last_known_reachability") is not None
                else status.last_known_reachability,
                valid_values=_VALID_REACHABILITY_STATES,
            ),
            last_known_auth_state=_optional_choice(
                data.get("last_known_auth_state")
                if data.get("last_known_auth_state") is not None
                else status.last_known_auth_state,
                valid_values=_VALID_AUTH_STATES,
            ),
            last_connected_at=_iso_to_datetime(data.get("last_connected_at")) or status.last_connected_at,
            updated_at=_iso_to_datetime(data.get("updated_at")) or status.updated_at,
        )

    @classmethod
    def from_legacy_tldw_api_config(cls, app_config: Mapping[str, Any]) -> "ConfiguredServerTarget" | None:
        if not isinstance(app_config, Mapping):
            return None

        api_config = app_config.get("tldw_api", {})
        if not isinstance(api_config, Mapping):
            api_config = {}

        raw_url = str(api_config.get("base_url") or api_config.get("api_url") or api_config.get("url") or "").strip()
        if not raw_url:
            return None

        server_id, label = _normalize_server_identity(raw_url)
        if server_id is None:
            return None

        auth_mode = str(api_config.get("auth_mode") or "").strip().lower()
        if auth_mode not in _VALID_AUTH_MODES:
            auth_mode = "bearer" if api_config.get("bearer_token") and not api_config.get("api_key") else "api_key"

        now = datetime.now(timezone.utc)
        return cls(
            server_id=server_id,
            label=label or server_id,
            base_url=server_id,
            auth_mode=auth_mode,
            auth_reference="legacy:tldw_api",
            is_default=True,
            last_known_server_label=label or server_id,
            last_known_reachability="unknown",
            last_known_auth_state="unknown",
            updated_at=now,
        )


@dataclass(frozen=True)
class ServerAccessContext:
    server_id: str
    selected_scope: str | None = None
    selected_section: str | None = None
    section_capabilities: SectionCapabilityFlags = field(default_factory=SectionCapabilityFlags)
    target_status: TargetStatusMetadata = field(default_factory=TargetStatusMetadata)
    panel_records: tuple[NormalizedPanelRecord, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "selected_scope": self.selected_scope,
            "selected_section": self.selected_section,
            "section_capabilities": self.section_capabilities.to_dict(),
            "target_status": self.target_status.to_dict(),
            "panel_records": [record.to_dict() for record in self.panel_records],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ServerAccessContext":
        if not isinstance(data, Mapping):
            return cls(server_id="")
        panel_records = data.get("panel_records")
        if isinstance(panel_records, list):
            normalized_panels = tuple(
                record
                for record in (NormalizedPanelRecord.from_dict(item) for item in panel_records)
                if record.panel_id
            )
        else:
            normalized_panels = ()
        return cls(
            server_id=_text_or_none(data.get("server_id")) or "",
            selected_scope=_scope_or_none(data.get("selected_scope")),
            selected_section=_text_or_none(data.get("selected_section")),
            section_capabilities=SectionCapabilityFlags.from_dict(data.get("section_capabilities")),
            target_status=TargetStatusMetadata.from_dict(data.get("target_status")),
            panel_records=normalized_panels,
        )


@dataclass(frozen=True)
class UnifiedMCPContext:
    selected_source: str = "local"
    selected_active_server_id: str | None = None
    selected_scope: str | None = None
    selected_section: str | None = None
    per_server_state: dict[str, ServerAccessContext] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_source": self.selected_source,
            "selected_active_server_id": self.selected_active_server_id,
            "selected_scope": self.selected_scope,
            "selected_section": self.selected_section,
            "per_server_state": {
                server_id: context.to_dict()
                for server_id, context in self.per_server_state.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UnifiedMCPContext":
        if not isinstance(data, Mapping):
            return cls()

        per_server_state = data.get("per_server_state")
        if isinstance(per_server_state, Mapping):
            restored_state = {
                str(server_id): ServerAccessContext.from_dict(context)
                for server_id, context in per_server_state.items()
                if str(server_id).strip()
            }
        else:
            restored_state = {}

        return cls(
            selected_source=_selected_source_or_default(data.get("selected_source")),
            selected_active_server_id=_text_or_none(data.get("selected_active_server_id")),
            selected_scope=_scope_or_none(data.get("selected_scope")),
            selected_section=_text_or_none(data.get("selected_section")),
            per_server_state=restored_state,
        )


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _scope_or_none(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    if text in _VALID_SCOPES:
        return text
    return text


def _selected_source_or_default(value: Any) -> str:
    text = _text_or_none(value)
    if text in {"local", "server"}:
        return text
    return "local"
