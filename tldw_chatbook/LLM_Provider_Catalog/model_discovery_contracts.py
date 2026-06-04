"""Value contracts for local OpenAI-compatible model discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal


DiscoverySource = Literal["saved", "runtime_discovered", "persisted_discovered"]
CapabilityStatus = Literal["known", "inferred", "unknown"]
ProviderKeyResolutionStatus = Literal["resolved", "missing", "ambiguous"]
DiscoveryErrorKind = Literal[
    "unsupported_endpoint",
    "missing_endpoint",
    "missing_credentials",
    "ambiguous_provider_key",
    "request_failed",
    "invalid_response",
]


@dataclass(frozen=True)
class ProviderModelListKeyResolution:
    """Resolution state for an exact top-level ``[providers]`` model-list key."""

    requested_provider: str
    normalized_provider: str
    provider_list_key: str | None
    status: ProviderKeyResolutionStatus
    matches: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveredModel:
    """Normalized model record returned by a provider discovery endpoint."""

    provider: str
    provider_list_key: str
    model_id: str
    display_name: str
    source: DiscoverySource
    endpoint_fingerprint: str
    discovered_at: str
    metadata_raw_safe: Mapping[str, Any] = field(default_factory=dict)
    capability_status: CapabilityStatus = "unknown"
    persisted: bool = False

    def __post_init__(self) -> None:
        """Freeze a caller-independent shallow copy of safe endpoint metadata."""
        object.__setattr__(
            self,
            "metadata_raw_safe",
            MappingProxyType(dict(self.metadata_raw_safe)),
        )


@dataclass(frozen=True)
class ModelDiscoveryError:
    """Safe user-facing discovery failure details."""

    kind: DiscoveryErrorKind
    message: str
    recovery_hint: str


@dataclass(frozen=True)
class ModelDiscoveryResult:
    """Result of one manual local model discovery attempt."""

    provider: str
    provider_list_key: str | None
    endpoint_fingerprint: str | None
    status: Literal["success", "unsupported", "error"]
    models: tuple[DiscoveredModel, ...] = ()
    error: ModelDiscoveryError | None = None
    policy_action: str = "llm.catalog.models.discover.local"


@dataclass(frozen=True)
class MergedModelEntry:
    """Model selector entry merged from saved and runtime-discovered sources."""

    provider: str
    provider_list_key: str
    model_id: str
    display_name: str
    source: DiscoverySource
    capability_status: CapabilityStatus
    persisted: bool


@dataclass(frozen=True)
class PersistenceResult:
    """Result of explicitly saving discovered model IDs to provider config."""

    provider: str
    provider_list_key: str | None
    status: Literal["saved", "missing_provider_key", "ambiguous_provider_key", "error"]
    saved_model_ids: tuple[str, ...] = ()
    message: str = ""
