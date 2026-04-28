from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.tldw_api import TLDWAPIClient

from .bootstrap import RuntimePolicyContext, build_runtime_api_client, derive_configured_server_binding
from .server_credentials import (
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_API_KEY,
    SERVER_CREDENTIAL_BEARER_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    ServerCredentialStore,
)


@dataclass(frozen=True, slots=True)
class ActiveServerContext:
    active_server_id: str
    label: str | None
    base_url: str
    auth_method: str
    auth_token: str | None
    credential_source: str
    target: ConfiguredServerTarget
    server_headers: Mapping[str, str]
    capabilities: Mapping[str, Any]


class ServerContextUnavailable(RuntimeError):
    reason_code = "server_context_unavailable"


class ServerCredentialsUnavailable(RuntimeError):
    reason_code = "server_credentials_unavailable"


class RuntimeServerContextProvider:
    def __init__(
        self,
        *,
        runtime_context: RuntimePolicyContext,
        target_store: ConfiguredServerTargetStore,
        credential_store: ServerCredentialStore,
        app_config: Mapping[str, Any] | None,
    ) -> None:
        self.runtime_context = runtime_context
        self.target_store = target_store
        self.credential_store = credential_store
        self.app_config = app_config or {}

    def get_active_context(self) -> ActiveServerContext:
        active_server_id = self._require_active_server_id()
        target = self.resolve_target()
        using_legacy_fallback_target = False
        if target is None:
            target = self._legacy_target_for_active_server(active_server_id)
            using_legacy_fallback_target = target is not None
        if target is None:
            raise ServerContextUnavailable(f"Active server is not configured: {active_server_id}")

        auth_token, credential_source = self._resolve_auth_token(
            active_server_id,
            target,
            allow_legacy_config=using_legacy_fallback_target or target.auth_reference == "legacy:tldw_api",
        )
        return ActiveServerContext(
            active_server_id=active_server_id,
            label=target.label or None,
            base_url=target.base_url,
            auth_method=target.auth_mode,
            auth_token=auth_token,
            credential_source=credential_source,
            target=target,
            server_headers=self._build_server_headers(target.auth_mode, auth_token),
            capabilities=self._build_capabilities(target),
        )

    def build_client(self) -> TLDWAPIClient:
        context = self.get_active_context()
        return build_runtime_api_client(
            endpoint_url=context.base_url,
            auth_method=context.auth_method,
            auth_token=context.auth_token,
        )

    def clear_active_server_credentials(self) -> None:
        self.credential_store.clear_server(self._require_active_server_id())

    def clear_server_credentials(self, server_id: str) -> None:
        self.credential_store.clear_server(server_id)

    def clear_active_server_auth_tokens(self) -> None:
        active_server_id = self._require_active_server_id()
        self.credential_store.delete_secret(active_server_id, SERVER_CREDENTIAL_ACCESS_TOKEN)
        self.credential_store.delete_secret(active_server_id, SERVER_CREDENTIAL_REFRESH_TOKEN)

    def store_auth_tokens(
        self,
        *,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ) -> None:
        context = self.get_active_context()
        if access_token:
            self.credential_store.set_secret(
                context.active_server_id,
                SERVER_CREDENTIAL_ACCESS_TOKEN,
                access_token,
            )
        if refresh_token:
            self.credential_store.set_secret(
                context.active_server_id,
                SERVER_CREDENTIAL_REFRESH_TOKEN,
                refresh_token,
            )

    def resolve_target(self) -> ConfiguredServerTarget | None:
        active_server_id = self._require_active_server_id()
        target = self.target_store.get_target(active_server_id)
        if target is not None:
            return target

        resolved_target = self.target_store.resolve_active_target()
        if resolved_target is not None and resolved_target.server_id == active_server_id:
            return resolved_target
        return None

    def _require_active_server_id(self) -> str:
        state = self.runtime_context.state
        active_server_id = str(state.active_server_id or "").strip()
        if state.active_source != "server" or not state.server_configured or not active_server_id:
            raise ServerContextUnavailable("Runtime policy does not have an active configured server")
        return active_server_id

    def _legacy_target_for_active_server(self, active_server_id: str) -> ConfiguredServerTarget | None:
        legacy_binding = derive_configured_server_binding(self.app_config)
        if not legacy_binding.server_configured or legacy_binding.active_server_id != active_server_id:
            return None
        return ConfiguredServerTarget.from_legacy_tldw_api_config(self.app_config)

    def _resolve_auth_token(
        self,
        server_id: str,
        target: ConfiguredServerTarget,
        *,
        allow_legacy_config: bool,
    ) -> tuple[str | None, str]:
        purpose = self._purpose_from_auth_reference(target.auth_reference)
        if purpose is not None:
            secret = self.credential_store.get_secret(server_id, purpose)
            if secret is not None:
                return secret, f"credential_store:{purpose}"
            return None, "none"

        for candidate_purpose in self._purposes_for_auth_mode(target.auth_mode):
            secret = self.credential_store.get_secret(server_id, candidate_purpose)
            if secret is not None:
                return secret, f"credential_store:{candidate_purpose}"

        if allow_legacy_config:
            legacy_token = self._legacy_config_token()
            if legacy_token is not None:
                return legacy_token, "legacy:tldw_api"
        return None, "none"

    def _legacy_config_token(self) -> str | None:
        api_config = self._legacy_api_config()
        token = api_config.get("auth_token") or api_config.get("api_key") or api_config.get("bearer_token")
        if token is None:
            return None
        normalized = str(token).strip()
        return normalized or None

    def _legacy_api_config(self) -> Mapping[str, Any]:
        if not isinstance(self.app_config, Mapping):
            return {}
        api_config = self.app_config.get("tldw_api", {})
        return api_config if isinstance(api_config, Mapping) else {}

    def _build_capabilities(self, target: ConfiguredServerTarget) -> dict[str, Any]:
        state = self.runtime_context.state
        return {
            "server_configured": state.server_configured,
            "reachability": state.server_reachability,
            "auth_state": state.server_auth_state,
            "last_known_server_label": state.last_known_server_label,
            "target_last_known_reachability": target.last_known_reachability,
            "target_last_known_auth_state": target.last_known_auth_state,
            "target_last_known_server_label": target.last_known_server_label,
        }

    @staticmethod
    def _build_server_headers(auth_method: str, auth_token: str | None) -> dict[str, str]:
        if not auth_token:
            return {}
        if auth_method in {"bearer", "custom_token"}:
            return {"Authorization": f"Bearer {auth_token}"}
        if auth_method == "api_key":
            return {"X-API-KEY": auth_token}
        return {}

    @staticmethod
    def _purpose_from_auth_reference(auth_reference: str | None) -> str | None:
        if not auth_reference:
            return None
        prefix = "keyring:"
        if not auth_reference.startswith(prefix):
            return None
        purpose = auth_reference[len(prefix) :].strip()
        return purpose or None

    @staticmethod
    def _purposes_for_auth_mode(auth_mode: str) -> tuple[str, ...]:
        if auth_mode == "bearer":
            return (SERVER_CREDENTIAL_BEARER_TOKEN, SERVER_CREDENTIAL_ACCESS_TOKEN)
        if auth_mode == "api_key":
            return (SERVER_CREDENTIAL_API_KEY,)
        if auth_mode == "custom_token":
            return (SERVER_CREDENTIAL_BEARER_TOKEN, SERVER_CREDENTIAL_ACCESS_TOKEN)
        return ()


__all__ = [
    "ActiveServerContext",
    "RuntimeServerContextProvider",
    "ServerContextUnavailable",
    "ServerCredentialsUnavailable",
]
