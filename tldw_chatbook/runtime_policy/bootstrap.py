from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from tldw_chatbook.Utils.private_paths import lexical_path
from tldw_chatbook.config import (
    application_owned_config_directory,
    get_cli_config_path,
    resolve_tldw_api_config,
)

from .source_state import RuntimeSourceStateStore
from .types import RuntimeSourceState

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import TLDWAPIClient

_VALID_RUNTIME_SOURCES = {"local", "server"}


class RuntimePolicyContext:
    __slots__ = (
        "_owner_thread_id",
        "_snapshot",
        "__runtime_policy_projection_callback",
        "__runtime_policy_state_store",
    )

    def __init__(
        self,
        state: RuntimeSourceState,
        store: RuntimeSourceStateStore,
        *,
        publish: Callable[[RuntimeSourceState], None] | None = None,
    ) -> None:
        self._snapshot = (state, 0)
        self._owner_thread_id = threading.get_ident()
        self.__runtime_policy_projection_callback = publish
        self.__runtime_policy_state_store = store

    @property
    def state(self) -> RuntimeSourceState:
        return self._snapshot[0]

    def snapshot(self) -> tuple[RuntimeSourceState, int]:
        return self._snapshot

    def commit_state(
        self,
        candidate: RuntimeSourceState,
        *,
        expected_revision: int,
    ) -> bool:
        self._assert_owner_thread()
        _, current_revision = self._snapshot
        if expected_revision != current_revision:
            return False

        self.__runtime_policy_state_store.save(candidate)
        self._snapshot = (candidate, current_revision + 1)
        if self.__runtime_policy_projection_callback is not None:
            try:
                self.__runtime_policy_projection_callback(candidate)
            except Exception as exc:
                logger.warning(
                    "Runtime policy projection failed after durable commit "
                    "(exception_category={}).",
                    type(exc).__name__,
                )
        return True

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("runtime policy mutation requires the owner thread")


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
    # Deferred import: TLDWAPIClient's home module (tldw_api/client.py) eagerly
    # imports the full ~54-submodule schema surface (~450ms). Importing it here,
    # at actual client-construction time, keeps `import tldw_chatbook.app` from
    # paying that cost in local-only sessions (task-285).
    from tldw_chatbook.tldw_api import TLDWAPIClient

    api_config: dict[str, Any] = resolve_tldw_api_config(app_config)

    resolved_endpoint = str(
        endpoint_url
        or api_config.get("base_url")
        or api_config.get("api_url")
        or api_config.get("url")
        or ""
    ).strip()
    if not resolved_endpoint:
        raise ValueError("TLDW API base URL is not configured.")

    resolved_auth_method = (
        str(auth_method or api_config.get("auth_mode") or "").strip().lower()
    )
    resolved_auth_token = auth_token
    if resolved_auth_token is None:
        resolved_auth_token = (
            api_config.get("auth_token")
            or api_config.get("api_key")
            or api_config.get("bearer_token")
        )

    if not resolved_auth_method:
        resolved_auth_method = (
            "bearer"
            if api_config.get("bearer_token") and not api_config.get("api_key")
            else "api_key"
        )

    if resolved_auth_method in {"bearer", "custom_token"}:
        client = TLDWAPIClient(base_url=resolved_endpoint)
        client.bearer_token = resolved_auth_token
        return client

    return TLDWAPIClient(base_url=resolved_endpoint, token=resolved_auth_token)


def build_runtime_api_client_from_config(
    app_config: Mapping[str, Any] | None,
) -> TLDWAPIClient:
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


def _prepare_runtime_policy_context(
    *,
    app_config: Mapping[str, Any] | None,
    publish: Callable[[RuntimeSourceState], None],
    store: RuntimeSourceStateStore | None = None,
    path: str | Path | None = None,
) -> RuntimePolicyContext:
    if store is None:
        effective_config_path = get_cli_config_path()
        selected_path = (
            lexical_path(path)
            if path is not None
            else lexical_path(effective_config_path.parent / "runtime_policy.json")
        )
        runtime_store = RuntimeSourceStateStore(
            selected_path,
            application_owned_directory=(
                application_owned_config_directory(effective_config_path)
                if path is None
                else None
            ),
        )
    else:
        runtime_store = store
    loaded_state = runtime_store.load()
    synchronized_state = synchronize_runtime_source_state_with_app_config(
        loaded_state,
        app_config,
    )
    context = RuntimePolicyContext(
        state=loaded_state,
        store=runtime_store,
        publish=publish,
    )
    if synchronized_state != loaded_state:
        context.commit_state(synchronized_state, expected_revision=0)
    else:
        publish(loaded_state)
    return context


def load_runtime_policy_for_app(
    app: Any,
    *,
    store: RuntimeSourceStateStore | None = None,
    path: str | Path | None = None,
) -> RuntimePolicyContext:
    if isinstance(getattr(app, "runtime_policy", None), RuntimePolicyContext):
        raise RuntimeError("runtime policy context is already installed")

    context = _prepare_runtime_policy_context(
        app_config=getattr(app, "app_config", None),
        publish=lambda state: _apply_runtime_policy_to_app(app, state),
        store=store,
        path=path,
    )
    app.runtime_policy = context
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


def set_authoritative_runtime_source(
    context: RuntimePolicyContext,
    active_source: str,
    *,
    app_config: Mapping[str, Any] | None,
) -> RuntimeSourceState:
    normalized_source = str(active_source or "").strip().lower()
    state, revision = context.snapshot()
    if normalized_source not in _VALID_RUNTIME_SOURCES:
        return state

    configured_binding = derive_configured_server_binding(app_config)
    resolved_source = normalized_source
    if resolved_source == "server" and not configured_binding.server_configured:
        resolved_source = "local"

    base_state = _clear_server_probe_state_if_binding_changed(state, configured_binding)
    updated_state = replace(
        base_state,
        active_source=resolved_source,
        active_server_id=configured_binding.active_server_id,
        server_configured=configured_binding.server_configured,
        last_known_server_label=configured_binding.last_known_server_label,
    )
    if not context.commit_state(updated_state, expected_revision=revision):
        raise RuntimeError("runtime policy commit was rejected")
    return updated_state


def add_runtime_policy_snapshot(
    saved_screen_state: dict[str, Any], state: RuntimeSourceState
) -> dict[str, Any]:
    snapshot_state = dict(saved_screen_state)
    snapshot_state["runtime_policy_snapshot"] = runtime_policy_snapshot_from_state(
        state
    )
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
    if (
        snapshot_source in _VALID_RUNTIME_SOURCES
        and snapshot_source != authoritative_state.active_source
    ):
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


def derive_configured_server_binding(
    app_config: Mapping[str, Any] | None,
) -> ConfiguredServerBinding:
    if not isinstance(app_config, Mapping):
        return ConfiguredServerBinding(
            active_server_id=None,
            server_configured=False,
            last_known_server_label=None,
        )

    api_config = resolve_tldw_api_config(app_config)

    raw_url = str(
        api_config.get("base_url")
        or api_config.get("api_url")
        or api_config.get("url")
        or ""
    ).strip()
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
    publisher = getattr(app, "_publish_runtime_policy_projection")
    if not callable(publisher):
        raise TypeError("runtime policy projection publisher is not callable")
    publisher(state)


def _normalize_server_identity(raw_url: str) -> tuple[str | None, str | None]:
    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.hostname:
        normalized = raw_url.rstrip("/") or None
        return normalized, normalized

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    port = parsed.port
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )

    netloc = hostname
    if port and not default_port:
        netloc = f"{hostname}:{port}"

    path = parsed.path.rstrip("/")
    normalized = urlunsplit((scheme, netloc, path, "", ""))
    return normalized, netloc
