from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.tldw_api import TLDWAPIClient

from .bootstrap import RuntimePolicyContext, build_runtime_api_client, derive_configured_server_binding
from .server_credentials import (
    CredentialStoreUnavailable,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_API_KEY,
    SERVER_CREDENTIAL_BEARER_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    ServerCredentialStore,
)
from .types import ServerContextFailure, ServerContextFailureReason


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


@dataclass(frozen=True, slots=True)
class _CachedClientKey:
    active_server_id: str
    base_url: str
    auth_method: str
    credential_source: str
    token_fingerprint: str | None


class ServerContextError(RuntimeError):
    reason_code = "server_context_unavailable"

    def __init__(
        self,
        message: str,
        *,
        reason_code: ServerContextFailureReason | str | None = None,
        recoverable: bool = True,
        active_server_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code or self.reason_code
        self.recoverable = recoverable
        self.active_server_id = active_server_id

    def to_contract(self) -> dict[str, object]:
        return ServerContextFailure(
            reason_code=self.reason_code,
            message=str(self),
            recoverable=self.recoverable,
            active_server_id=self.active_server_id,
        ).to_contract()


class ServerContextUnavailable(ServerContextError):
    reason_code = "server_not_configured"


class ServerCredentialsUnavailable(ServerContextError):
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
        self._legacy_cleared_server_ids: set[str] = set()
        self._cached_client_key: _CachedClientKey | None = None
        self._cached_client: TLDWAPIClient | None = None
        self._pending_client_close_tasks: set[asyncio.Task[None]] = set()

    def get_active_context(self) -> ActiveServerContext:
        active_server_id = self._require_active_server_id()
        target = self.resolve_target()
        using_legacy_fallback_target = False
        if target is None:
            target = self._legacy_target_for_active_server(active_server_id)
            using_legacy_fallback_target = target is not None
        if target is None:
            raise ServerContextUnavailable(
                "Active server profile is unavailable.",
                reason_code="server_profile_missing",
                active_server_id=active_server_id,
            )

        auth_token, credential_source = self._resolve_auth_token(
            active_server_id,
            target,
            allow_legacy_config=self._should_allow_legacy_config(
                active_server_id,
                target,
                using_legacy_fallback_target=using_legacy_fallback_target,
            ),
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
        cache_key = self._client_cache_key(context)
        if self._cached_client is not None and self._cached_client_key == cache_key:
            return self._cached_client

        self._invalidate_cached_client()
        self._cached_client_key = cache_key
        self._cached_client = build_runtime_api_client(
            endpoint_url=context.base_url,
            auth_method=context.auth_method,
            auth_token=context.auth_token,
        )
        return self._cached_client

    async def close_cached_client(self) -> None:
        cached_client = self._cached_client
        self._cached_client = None
        self._cached_client_key = None
        if cached_client is not None:
            await cached_client.close()
        pending_tasks = list(self._pending_client_close_tasks)
        if pending_tasks:
            await asyncio.gather(*pending_tasks)

    def clear_active_server_credentials(self) -> None:
        active_server_id = self._require_active_server_id()
        self.credential_store.clear_server(active_server_id)
        self._mark_legacy_server_id_cleared(active_server_id)
        self._invalidate_cached_client()

    def clear_server_credentials(self, server_id: str) -> None:
        self.credential_store.clear_server(server_id)
        self._mark_legacy_server_id_cleared(server_id)
        self._invalidate_cached_client()

    def clear_all_credentials(self) -> None:
        self.credential_store.clear_all()
        self._legacy_cleared_server_ids.update(self._legacy_server_ids_for_signout())
        self._invalidate_cached_client()

    def clear_active_server_auth_tokens(self) -> None:
        active_server_id = self._require_active_server_id()
        self.credential_store.delete_secret(active_server_id, SERVER_CREDENTIAL_ACCESS_TOKEN)
        self.credential_store.delete_secret(active_server_id, SERVER_CREDENTIAL_REFRESH_TOKEN)
        self._invalidate_cached_client()

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
        if access_token or refresh_token:
            self._legacy_cleared_server_ids.discard(context.active_server_id)
            self._invalidate_cached_client()

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
            raise ServerContextUnavailable(
                "Runtime policy does not have an active configured server.",
                reason_code="server_not_configured",
            )
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
            secret = self._get_credential_secret(server_id, purpose)
            if secret is not None:
                return secret, f"credential_store:{purpose}"
            return None, "none"

        credential_error: ServerCredentialsUnavailable | None = None
        for candidate_purpose in self._purposes_for_auth_mode(target.auth_mode):
            try:
                secret = self._get_credential_secret(server_id, candidate_purpose)
            except ServerCredentialsUnavailable as exc:
                credential_error = exc
                break
            if secret is not None:
                return secret, f"credential_store:{candidate_purpose}"

        if allow_legacy_config:
            if server_id in self._legacy_cleared_server_ids:
                if credential_error is not None:
                    raise credential_error
                raise ServerCredentialsUnavailable(
                    "The active server profile is no longer authorized.",
                    reason_code="profile_no_longer_authorized",
                    active_server_id=server_id,
                )
            legacy_token = self._legacy_config_token()
            if legacy_token is not None:
                imported_purpose = self._import_legacy_token(server_id, target.auth_mode, legacy_token)
                if imported_purpose is not None:
                    return legacy_token, f"credential_store:{imported_purpose}"
                return legacy_token, "legacy:tldw_api"
        if credential_error is not None:
            raise credential_error
        return None, "none"

    def _get_credential_secret(self, server_id: str, purpose: str) -> str | None:
        try:
            return self.credential_store.get_secret(server_id, purpose)
        except CredentialStoreUnavailable as exc:
            raise ServerCredentialsUnavailable(
                "Credential store is unavailable for the active server.",
                reason_code=exc.reason_code,
                active_server_id=server_id,
            ) from exc
        except Exception as exc:
            raise ServerCredentialsUnavailable(
                "Server credentials are unavailable for the active server.",
                reason_code="server_credentials_unavailable",
                active_server_id=server_id,
            ) from exc

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

    def _should_allow_legacy_config(
        self,
        active_server_id: str,
        target: ConfiguredServerTarget,
        *,
        using_legacy_fallback_target: bool,
    ) -> bool:
        if not using_legacy_fallback_target and target.auth_reference != "legacy:tldw_api":
            return False

        legacy_binding = derive_configured_server_binding(self.app_config)
        return legacy_binding.server_configured and legacy_binding.active_server_id == active_server_id

    def _mark_legacy_server_id_cleared(self, server_id: str) -> None:
        normalized_server_id = str(server_id or "").strip()
        if not normalized_server_id:
            return
        if normalized_server_id in self._legacy_server_ids_for_signout():
            self._legacy_cleared_server_ids.add(normalized_server_id)

    def _legacy_server_ids_for_signout(self) -> set[str]:
        server_ids = {
            target.server_id
            for target in self.target_store.list_targets()
            if target.auth_reference == "legacy:tldw_api"
        }
        legacy_binding = derive_configured_server_binding(self.app_config)
        if legacy_binding.server_configured and legacy_binding.active_server_id:
            server_ids.add(legacy_binding.active_server_id)
        return server_ids

    def _import_legacy_token(self, server_id: str, auth_mode: str, token: str) -> str | None:
        purposes = self._purposes_for_auth_mode(auth_mode)
        if not purposes:
            return None

        purpose = purposes[0]
        try:
            self.credential_store.set_secret(server_id, purpose, token)
        except Exception:
            return None
        return purpose

    def _invalidate_cached_client(self) -> None:
        cached_client = self._cached_client
        self._cached_client = None
        self._cached_client_key = None
        if cached_client is not None:
            self._close_client_sync_safe(cached_client)

    def _close_client_sync_safe(self, client: TLDWAPIClient) -> None:
        async def _close() -> None:
            await client.close()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_close())
            return

        task = loop.create_task(_close())
        self._pending_client_close_tasks.add(task)
        task.add_done_callback(self._pending_client_close_tasks.discard)

    @classmethod
    def _client_cache_key(cls, context: ActiveServerContext) -> _CachedClientKey:
        return _CachedClientKey(
            active_server_id=context.active_server_id,
            base_url=context.base_url,
            auth_method=context.auth_method,
            credential_source=context.credential_source,
            token_fingerprint=cls._token_fingerprint(context.auth_token),
        )

    @staticmethod
    def _token_fingerprint(auth_token: str | None) -> str | None:
        if not auth_token:
            return None
        return hashlib.sha256(auth_token.encode("utf-8")).hexdigest()

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
    "ServerContextError",
    "ServerContextUnavailable",
    "ServerCredentialsUnavailable",
]
