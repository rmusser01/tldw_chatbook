"""Source-aware routing for server-owned auth/profile/account operations."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from tldw_chatbook.runtime_policy.server_credentials import SERVER_CREDENTIAL_BEARER_TOKEN


class AuthAccountBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "auth_account.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server auth/profile/session state is unavailable in local/offline mode; Chatbook local identity remains single-user and separate.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "auth_account.durable_credential_storage.server",
        "source": "server",
        "supported": False,
        "reason_code": "runtime_credential_context_unavailable",
        "user_message": "Durable credential storage requires an active runtime server context provider with credential-store hooks; this scope cannot store or clear credentials until that provider is wired.",
        "affected_action_ids": [],
    },
    {
        "operation_id": "auth_account.admin_user_management.server",
        "source": "server",
        "supported": False,
        "reason_code": "out_of_scope_admin_surface",
        "user_message": "Admin/ops user-management surfaces are intentionally outside Chatbook client parity.",
        "affected_action_ids": [],
    },
]


class AuthAccountScopeService:
    """Route account actions to the active server without creating local account authority."""

    def __init__(
        self,
        *,
        server_service: Any = None,
        policy_enforcer: Any = None,
        server_context_provider: Any = None,
    ):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer
        self.server_context_provider = server_context_provider

    def _normalize_mode(self, mode: AuthAccountBackend | str | None) -> AuthAccountBackend:
        if mode is None:
            return AuthAccountBackend.SERVER
        if isinstance(mode, AuthAccountBackend):
            return mode
        try:
            return AuthAccountBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid auth/account backend: {mode}") from exc

    def _require_server_service(self, mode: AuthAccountBackend) -> Any:
        if mode == AuthAccountBackend.LOCAL:
            raise ValueError(
                "Auth/profile/session operations are server-owned; Chatbook local identity remains separate."
            )
        if self.server_service is None:
            raise ValueError("Server auth/profile/account backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _with_backend(mode: AuthAccountBackend, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:auth:{kind}")
        return record

    def _store_auth_tokens(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self.store_login_tokens(
            access_token=payload.get("access_token"),
            refresh_token=payload.get("refresh_token"),
        )

    def _clear_active_server_auth_tokens(self) -> None:
        self.clear_login_tokens()

    def store_login_tokens(
        self,
        *,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ) -> None:
        if not access_token and not refresh_token:
            return
        store = getattr(self.server_context_provider, "store_auth_tokens", None)
        if callable(store):
            store(access_token=access_token, refresh_token=refresh_token)
        self._set_effective_bearer_token(access_token)

    def clear_login_tokens(self) -> None:
        clear = getattr(self.server_context_provider, "clear_active_server_auth_tokens", None)
        if callable(clear):
            clear()
        self._delete_effective_bearer_token()

    def _set_effective_bearer_token(self, access_token: str | None) -> None:
        if not access_token:
            return
        credential_store, active_server_id = self._resolve_provider_credential_store_context()
        if credential_store is None or not active_server_id:
            return
        credential_store.set_secret(
            active_server_id,
            SERVER_CREDENTIAL_BEARER_TOKEN,
            access_token,
        )

    def _delete_effective_bearer_token(self) -> None:
        credential_store, active_server_id = self._resolve_provider_credential_store_context()
        if credential_store is None or not active_server_id:
            return
        credential_store.delete_secret(
            active_server_id,
            SERVER_CREDENTIAL_BEARER_TOKEN,
        )

    def _resolve_provider_credential_store_context(self) -> tuple[Any | None, str | None]:
        provider = self.server_context_provider
        if provider is None:
            return None, None

        credential_store = getattr(provider, "credential_store", None)
        if credential_store is None:
            return None, None

        active_server_id = getattr(provider, "active_server_id", None)
        if active_server_id:
            return credential_store, str(active_server_id)

        get_active_context = getattr(provider, "get_active_context", None)
        if callable(get_active_context):
            try:
                active_context = get_active_context()
            except Exception:
                active_context = None
            resolved_server_id = getattr(active_context, "active_server_id", None)
            if resolved_server_id:
                return credential_store, str(resolved_server_id)

        runtime_context = getattr(provider, "runtime_context", None)
        state = getattr(runtime_context, "state", None)
        resolved_server_id = getattr(state, "active_server_id", None)
        if resolved_server_id:
            return credential_store, str(resolved_server_id)
        return credential_store, None

    def _has_durable_credential_storage(self) -> bool:
        credential_store, active_server_id = self._resolve_provider_credential_store_context()
        if credential_store is None or not active_server_id:
            return False
        return callable(getattr(self.server_context_provider, "store_auth_tokens", None)) and callable(
            getattr(self.server_context_provider, "clear_active_server_auth_tokens", None)
        )

    @staticmethod
    def _normalize_session(mode: AuthAccountBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        session_id = record.get("id") or record.get("session_id")
        if session_id is not None:
            record.setdefault("record_id", f"{mode.value}:auth_session:{session_id}")
        return record

    @staticmethod
    def _normalize_provider_key(mode: AuthAccountBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        provider = record.get("provider")
        if provider is not None:
            record.setdefault("record_id", f"{mode.value}:provider_key:{provider}")
        return record

    @staticmethod
    def _normalize_storage_file(mode: AuthAccountBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        file_id = record.get("id") or record.get("file_id") or record.get("uuid")
        if file_id is not None:
            record.setdefault("record_id", f"{mode.value}:storage_file:{file_id}")
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == AuthAccountBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        unsupported = []
        durable_storage_available = self._has_durable_credential_storage()
        for item in _SERVER_UNSUPPORTED_CAPABILITIES:
            if (
                item["operation_id"] == "auth_account.durable_credential_storage.server"
                and durable_storage_available
            ):
                continue
            unsupported.append(dict(item))
        return unsupported

    async def _call(
        self,
        *,
        mode: AuthAccountBackend | str | None,
        action_id: str,
        method_name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[AuthAccountBackend, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        result = await self._maybe_await(getattr(service, method_name)(**(kwargs or {})))
        return normalized_mode, result

    async def login(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        username: str,
        password: str,
        set_bearer_token: bool = True,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.identity.launch.server",
            method_name="login",
            kwargs={
                "username": username,
                "password": password,
                **({"set_bearer_token": set_bearer_token} if not set_bearer_token else {}),
            },
        )
        self._store_auth_tokens(result)
        return self._with_backend(normalized_mode, "identity", result)

    async def refresh_auth_token(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
        set_bearer_token: bool = True,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.identity.update.server",
            method_name="refresh_auth_token",
            kwargs={
                "request_data": request_data,
                **({"set_bearer_token": set_bearer_token} if not set_bearer_token else {}),
            },
        )
        self._store_auth_tokens(result)
        return self._with_backend(normalized_mode, "identity", result)

    async def logout(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        all_devices: bool = False,
        clear_bearer_token: bool = True,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.identity.delete.server",
            method_name="logout",
            kwargs={
                "all_devices": all_devices,
                **({"clear_bearer_token": clear_bearer_token} if not clear_bearer_token else {}),
            },
        )
        if clear_bearer_token:
            self._clear_active_server_auth_tokens()
        return self._with_backend(normalized_mode, "identity_logout", result)

    async def list_auth_sessions(self, *, mode: AuthAccountBackend | str | None = None) -> list[dict[str, Any]]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.sessions.list.server",
            method_name="list_auth_sessions",
        )
        return [
            self._normalize_session(normalized_mode, item) if isinstance(item, dict) else item
            for item in list(result or [])
        ]

    async def revoke_auth_session(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        session_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.sessions.delete.server",
            method_name="revoke_auth_session",
            kwargs={"session_id": session_id},
        )
        return self._with_backend(normalized_mode, "session_revoke", result)

    async def revoke_all_auth_sessions(self, *, mode: AuthAccountBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.sessions.delete.server",
            method_name="revoke_all_auth_sessions",
        )
        return self._with_backend(normalized_mode, "session_revoke_all", result)

    async def get_user_profile_catalog(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        if_none_match: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.profile.list.server",
            method_name="get_user_profile_catalog",
            kwargs={"if_none_match": if_none_match},
        )
        return self._with_backend(normalized_mode, "profile_catalog", result)

    async def get_current_user_profile(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        sections: str | list[str] | None = None,
        include_sources: bool = False,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.profile.detail.server",
            method_name="get_current_user_profile",
            kwargs={"sections": sections, "include_sources": include_sources},
        )
        record = self._with_backend(normalized_mode, "profile", result)
        record["record_id"] = f"{normalized_mode.value}:auth_profile:self"
        return record

    async def update_current_user_profile(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.profile.update.server",
            method_name="update_current_user_profile",
            kwargs={"request_data": request_data},
        )
        record = self._with_backend(normalized_mode, "profile_update", result)
        record["record_id"] = f"{normalized_mode.value}:auth_profile:self"
        return record

    async def register_user(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.registration.create.server",
            method_name="register_user",
            kwargs={"request_data": request_data},
        )
        return self._with_backend(normalized_mode, "registration", result)

    async def change_password(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.update.server",
            method_name="change_password",
            kwargs={"request_data": request_data},
        )
        return self._with_backend(normalized_mode, "security", result)

    async def request_password_reset(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.launch.server",
            method_name="request_password_reset",
            kwargs={"request_data": request_data},
        )
        return self._with_backend(normalized_mode, "security_reset_request", result)

    async def reset_password(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.update.server",
            method_name="reset_password",
            kwargs={"request_data": request_data},
        )
        return self._with_backend(normalized_mode, "security_reset", result)

    async def verify_email(self, *, mode: AuthAccountBackend | str | None = None, token: str) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.update.server",
            method_name="verify_email",
            kwargs={"token": token},
        )
        return self._with_backend(normalized_mode, "email_verify", result)

    async def resend_verification(self, *, mode: AuthAccountBackend | str | None = None, email: str) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.launch.server",
            method_name="resend_verification",
            kwargs={"email": email},
        )
        return self._with_backend(normalized_mode, "email_resend", result)

    async def request_magic_link(self, *, mode: AuthAccountBackend | str | None = None, email: str) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.launch.server",
            method_name="request_magic_link",
            kwargs={"email": email},
        )
        return self._with_backend(normalized_mode, "magic_link_request", result)

    async def verify_magic_link(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        token: str,
        set_bearer_token: bool = True,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.identity.launch.server",
            method_name="verify_magic_link",
            kwargs={
                "token": token,
                **({"set_bearer_token": set_bearer_token} if not set_bearer_token else {}),
            },
        )
        self._store_auth_tokens(result)
        return self._with_backend(normalized_mode, "identity", result)

    async def setup_mfa(self, *, mode: AuthAccountBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.launch.server",
            method_name="setup_mfa",
        )
        return self._with_backend(normalized_mode, "mfa_setup", result)

    async def verify_mfa_setup(self, *, mode: AuthAccountBackend | str | None = None, token: str) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.update.server",
            method_name="verify_mfa_setup",
            kwargs={"token": token},
        )
        return self._with_backend(normalized_mode, "mfa_verify", result)

    async def disable_mfa(self, *, mode: AuthAccountBackend | str | None = None, password: str) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.security.update.server",
            method_name="disable_mfa",
            kwargs={"password": password},
        )
        return self._with_backend(normalized_mode, "mfa_disable", result)

    async def complete_mfa_login(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        session_token: str,
        mfa_token: str,
        set_bearer_token: bool = True,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.identity.launch.server",
            method_name="complete_mfa_login",
            kwargs={
                "session_token": session_token,
                "mfa_token": mfa_token,
                **({"set_bearer_token": set_bearer_token} if not set_bearer_token else {}),
            },
        )
        self._store_auth_tokens(result)
        return self._with_backend(normalized_mode, "identity", result)

    async def list_user_api_keys(self, *, mode: AuthAccountBackend | str | None = None) -> list[dict[str, Any]]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.api_keys.list.server",
            method_name="list_user_api_keys",
        )
        return [self._with_backend(normalized_mode, "api_key", item) for item in list(result or [])]

    async def create_user_api_key(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.api_keys.create.server",
            method_name="create_user_api_key",
            kwargs={"request_data": request_data},
        )
        return self._with_backend(normalized_mode, "api_key", result)

    async def create_virtual_api_key(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.api_keys.create.server",
            method_name="create_virtual_api_key",
            kwargs={"request_data": request_data, **kwargs},
        )
        return self._with_backend(normalized_mode, "api_key", result)

    async def rotate_user_api_key(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        key_id: int,
        request_data: Any = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.api_keys.update.server",
            method_name="rotate_user_api_key",
            kwargs={"key_id": key_id, "request_data": request_data},
        )
        return self._with_backend(normalized_mode, "api_key", result)

    async def revoke_user_api_key(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        key_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.api_keys.delete.server",
            method_name="revoke_user_api_key",
            kwargs={"key_id": key_id},
        )
        return self._with_backend(normalized_mode, "api_key_revoke", result)

    async def upsert_user_provider_key(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.provider_keys.create.server",
            method_name="upsert_user_provider_key",
            kwargs={"request_data": request_data},
        )
        return self._normalize_provider_key(normalized_mode, result)

    async def list_user_provider_keys(self, *, mode: AuthAccountBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.provider_keys.list.server",
            method_name="list_user_provider_keys",
        )
        record = self._with_backend(normalized_mode, "provider_keys", result)
        if isinstance(record.get("items"), list):
            record["items"] = [
                self._normalize_provider_key(normalized_mode, item) if isinstance(item, dict) else item
                for item in record["items"]
            ]
        return record

    async def test_user_provider_key(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.provider_keys.validate.server",
            method_name="test_user_provider_key",
            kwargs={"request_data": request_data},
        )
        return self._normalize_provider_key(normalized_mode, result)

    async def get_openai_oauth_status(self, *, mode: AuthAccountBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.provider_keys.detail.server",
            method_name="get_openai_oauth_status",
        )
        return self._normalize_provider_key(normalized_mode, result)

    async def get_user_storage_quota(self, *, mode: AuthAccountBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.storage.detail.server",
            method_name="get_user_storage_quota",
        )
        return self._with_backend(normalized_mode, "storage_quota", result)

    async def list_storage_files(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        offset: int = 0,
        limit: int = 50,
        file_category: Any = None,
        source_feature: Any = None,
        folder_tag: str | None = None,
        search: str | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.storage.list.server",
            method_name="list_storage_files",
            kwargs={
                "offset": offset,
                "limit": limit,
                "file_category": file_category,
                "source_feature": source_feature,
                "folder_tag": folder_tag,
                "search": search,
                "include_deleted": include_deleted,
            },
        )
        record = self._with_backend(normalized_mode, "storage_files", result)
        if isinstance(record.get("items"), list):
            record["items"] = [
                self._normalize_storage_file(normalized_mode, item) if isinstance(item, dict) else item
                for item in record["items"]
            ]
        return record

    async def get_storage_file(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        file_id: int,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.storage.detail.server",
            method_name="get_storage_file",
            kwargs={"file_id": file_id},
        )
        return self._normalize_storage_file(normalized_mode, result)

    async def update_storage_file(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        file_id: int,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.storage.update.server",
            method_name="update_storage_file",
            kwargs={"file_id": file_id, "request_data": request_data},
        )
        return self._normalize_storage_file(normalized_mode, result)

    async def delete_storage_file(
        self,
        *,
        mode: AuthAccountBackend | str | None = None,
        file_id: int,
        hard_delete: bool = False,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="auth.storage.delete.server",
            method_name="delete_storage_file",
            kwargs={"file_id": file_id, "hard_delete": hard_delete},
        )
        return self._with_backend(normalized_mode, "storage_delete", result)
