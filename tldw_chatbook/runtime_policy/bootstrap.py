from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from tldw_chatbook.config import DEFAULT_CONFIG_PATH
from tldw_chatbook.tldw_api import TLDWAPIClient

from .source_state import RuntimeSourceStateStore
from .types import RuntimeSourceState

DEFAULT_RUNTIME_POLICY_PATH = DEFAULT_CONFIG_PATH.parent / "runtime_policy.json"
_VALID_RUNTIME_SOURCES = {"local", "server"}


@dataclass(slots=True)
class RuntimePolicyContext:
    state: RuntimeSourceState
    store: RuntimeSourceStateStore

    def persist(self) -> None:
        self.store.save(self.state)


@dataclass(frozen=True, slots=True)
class ConfiguredServerBinding:
    active_server_id: str | None
    server_configured: bool
    last_known_server_label: str | None


def build_runtime_api_client(
    *,
    app_config: Mapping[str, Any] | None = None,
    endpoint_url: str | None = None,
    auth_token: str | None = None,
    auth_method: str | None = None,
) -> TLDWAPIClient:
    api_config: dict[str, Any] = {}
    if isinstance(app_config, Mapping):
        api_config = dict(app_config.get("tldw_api", {}) or {})
        if not api_config:
            # load_settings() keeps the raw CLI config (and its [tldw_api]
            # section) nested under COMPREHENSIVE_CONFIG_RAW.
            raw_config = app_config.get("COMPREHENSIVE_CONFIG_RAW", {})
            if isinstance(raw_config, Mapping):
                api_config = dict(raw_config.get("tldw_api", {}) or {})

    resolved_endpoint = str(
        endpoint_url
        or api_config.get("base_url")
        or api_config.get("api_url")
        or api_config.get("url")
        or ""
    ).strip()
    if not resolved_endpoint:
        raise ValueError("TLDW API base URL is not configured.")

    resolved_auth_method = str(auth_method or api_config.get("auth_mode") or "").strip().lower()
    resolved_auth_token = auth_token
    if resolved_auth_token is None:
        resolved_auth_token = (
            api_config.get("auth_token")
            or api_config.get("api_key")
            or api_config.get("bearer_token")
        )

    if not resolved_auth_method:
        resolved_auth_method = "bearer" if api_config.get("bearer_token") and not api_config.get("api_key") else "api_key"

    if resolved_auth_method in {"bearer", "custom_token"}:
        client = TLDWAPIClient(base_url=resolved_endpoint)
        client.bearer_token = resolved_auth_token
        return client

    return TLDWAPIClient(base_url=resolved_endpoint, token=resolved_auth_token)


def build_runtime_api_client_from_config(app_config: Mapping[str, Any] | None) -> TLDWAPIClient:
    return build_runtime_api_client(app_config=app_config)


@dataclass(slots=True)
class LegacyConfigServerClientProvider:
    app_config: Mapping[str, Any] | None
    _cached_client: TLDWAPIClient | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(app_config=<redacted>)"

    def build_client(self) -> TLDWAPIClient:
        if self._cached_client is None:
            self._cached_client = build_runtime_api_client_from_config(self.app_config)
        return self._cached_client

    async def close_cached_client(self) -> None:
        cached_client = self._cached_client
        self._cached_client = None
        if cached_client is not None:
            await cached_client.close()


def build_runtime_api_client_provider_from_config(
    app_config: Mapping[str, Any] | None,
) -> LegacyConfigServerClientProvider:
    return LegacyConfigServerClientProvider(app_config=app_config)


def build_server_chatbook_service(
    *,
    app_config: Mapping[str, Any] | None,
    policy_enforcer: Any | None = None,
    allow_unconfigured: bool = False,
) -> Any:
    from ..Chatbooks.server_chatbook_service import ServerChatbookService

    try:
        client = build_runtime_api_client(app_config=app_config)
    except ValueError:
        if not allow_unconfigured:
            raise
        client = None
    return ServerChatbookService(client, policy_enforcer=policy_enforcer)


def load_runtime_policy_for_app(
    app: Any,
    *,
    store: RuntimeSourceStateStore | None = None,
    path: str | Path | None = None,
) -> RuntimePolicyContext:
    runtime_store = store or RuntimeSourceStateStore(path or DEFAULT_RUNTIME_POLICY_PATH)
    loaded_state = runtime_store.load()
    synchronized_state = synchronize_runtime_source_state_with_app_config(
        loaded_state,
        getattr(app, "app_config", None),
    )
    context = RuntimePolicyContext(
        state=synchronized_state,
        store=runtime_store,
    )
    if synchronized_state != loaded_state:
        context.persist()
    setattr(app, "runtime_policy", context)
    _apply_runtime_policy_to_app(app, context.state)
    return context


def ensure_runtime_policy_for_app(
    app: Any,
    *,
    store: RuntimeSourceStateStore | None = None,
    path: str | Path | None = None,
) -> RuntimePolicyContext:
    context = getattr(app, "runtime_policy", None)
    if isinstance(context, RuntimePolicyContext):
        return context
    return load_runtime_policy_for_app(app, store=store, path=path)


def set_authoritative_runtime_source(app: Any, active_source: str) -> RuntimeSourceState:
    normalized_source = str(active_source or "").strip().lower()
    context = ensure_runtime_policy_for_app(app)
    if normalized_source not in _VALID_RUNTIME_SOURCES:
        return context.state

    configured_binding = derive_configured_server_binding(getattr(app, "app_config", None))
    resolved_source = normalized_source
    if resolved_source == "server" and not configured_binding.server_configured:
        resolved_source = "local"

    base_state = _clear_server_probe_state_if_binding_changed(context.state, configured_binding)
    updated_state = replace(
        base_state,
        active_source=resolved_source,
        active_server_id=configured_binding.active_server_id,
        server_configured=configured_binding.server_configured,
        last_known_server_label=configured_binding.last_known_server_label,
    )
    context.state = updated_state
    context.persist()
    _apply_runtime_policy_to_app(app, updated_state)
    return updated_state


def add_runtime_policy_snapshot(saved_screen_state: dict[str, Any], state: RuntimeSourceState) -> dict[str, Any]:
    snapshot_state = dict(saved_screen_state)
    snapshot_state["runtime_policy_snapshot"] = runtime_policy_snapshot_from_state(state)
    return snapshot_state


def reconcile_saved_screen_state(
    saved_screen_state: dict[str, Any] | None,
    authoritative_state: RuntimeSourceState,
) -> dict[str, Any] | None:
    if not isinstance(saved_screen_state, dict):
        return None

    restored_state = dict(saved_screen_state)
    snapshot = restored_state.pop("runtime_policy_snapshot", None)
    if not isinstance(snapshot, dict):
        return restored_state

    snapshot_source = snapshot.get("active_source")
    if snapshot_source in _VALID_RUNTIME_SOURCES and snapshot_source != authoritative_state.active_source:
        return None

    if authoritative_state.active_source != "server":
        return restored_state

    authoritative_server_id = authoritative_state.active_server_id
    snapshot_server_id = snapshot.get("active_server_id")
    if authoritative_server_id and snapshot_server_id != authoritative_server_id:
        return None

    return restored_state


def runtime_policy_snapshot_from_state(state: RuntimeSourceState) -> dict[str, Any]:
    return {
        "active_source": state.active_source,
        "active_server_id": state.active_server_id,
    }


def derive_configured_server_binding(app_config: Mapping[str, Any] | None) -> ConfiguredServerBinding:
    if not isinstance(app_config, Mapping):
        return ConfiguredServerBinding(
            active_server_id=None,
            server_configured=False,
            last_known_server_label=None,
        )

    api_config = app_config.get("tldw_api", {})
    if not isinstance(api_config, Mapping) or not api_config:
        # The app's app_config comes from load_settings(), which normalizes
        # sections and keeps the raw CLI config nested under
        # COMPREHENSIVE_CONFIG_RAW; [tldw_api] only exists there.
        raw_config = app_config.get("COMPREHENSIVE_CONFIG_RAW", {})
        api_config = raw_config.get("tldw_api", {}) if isinstance(raw_config, Mapping) else {}
    if not isinstance(api_config, Mapping):
        api_config = {}

    raw_url = str(api_config.get("base_url") or api_config.get("api_url") or api_config.get("url") or "").strip()
    if not raw_url:
        return ConfiguredServerBinding(
            active_server_id=None,
            server_configured=False,
            last_known_server_label=None,
        )

    active_server_id, last_known_server_label = _normalize_server_identity(raw_url)
    return ConfiguredServerBinding(
        active_server_id=active_server_id,
        server_configured=active_server_id is not None,
        last_known_server_label=last_known_server_label,
    )


def synchronize_runtime_source_state_with_app_config(
    state: RuntimeSourceState,
    app_config: Mapping[str, Any] | None,
) -> RuntimeSourceState:
    configured_binding = derive_configured_server_binding(app_config)
    resolved_source = state.active_source
    if resolved_source == "server" and not configured_binding.server_configured:
        resolved_source = "local"

    base_state = _clear_server_probe_state_if_binding_changed(state, configured_binding)
    return replace(
        base_state,
        active_source=resolved_source,
        active_server_id=configured_binding.active_server_id,
        server_configured=configured_binding.server_configured,
        last_known_server_label=configured_binding.last_known_server_label,
    )


def _clear_server_probe_state_if_binding_changed(
    state: RuntimeSourceState,
    configured_binding: ConfiguredServerBinding,
) -> RuntimeSourceState:
    if (
        state.active_server_id == configured_binding.active_server_id
        and state.server_configured == configured_binding.server_configured
    ):
        return state

    return replace(
        state,
        server_reachability="unknown",
        server_reachability_checked_at=None,
        server_auth_state="unknown",
        server_auth_checked_at=None,
    )


def _apply_runtime_policy_to_app(app: Any, state: RuntimeSourceState) -> None:
    setattr(app, "current_runtime_backend", state.active_source)
    setattr(app, "runtime_backend", state.active_source)
    setattr(app, "active_server_id", state.active_server_id)

    app_state = getattr(app, "app_state", None)
    if app_state is not None:
        app_state.runtime_source = state


def _normalize_server_identity(raw_url: str) -> tuple[str | None, str | None]:
    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.hostname:
        normalized = raw_url.rstrip("/") or None
        return normalized, normalized

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    port = parsed.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)

    netloc = hostname
    if port and not default_port:
        netloc = f"{hostname}:{port}"

    path = parsed.path.rstrip("/")
    normalized = urlunsplit((scheme, netloc, path, "", ""))
    return normalized, netloc
