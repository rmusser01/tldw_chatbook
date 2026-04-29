"""Policy-gated remote auth/profile/account service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    APIKeyCreateRequest,
    APIKeyRotateRequest,
    BulkDeleteRequest,
    BulkMoveRequest,
    GeneratedFileUpdate,
    OpenAICredentialSourceSwitchRequest,
    OpenAIOAuthAuthorizeRequest,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    ProviderKeyTestRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TLDWAPIClient,
    UserProfileUpdateRequest,
    UserProviderKeyUpsertRequest,
    VirtualAPIKeyCreateRequest,
)


@dataclass(slots=True)
class _ConfigBackedClientProvider:
    app_config: Mapping[str, Any]
    _client: TLDWAPIClient | None = None

    def build_client(self) -> TLDWAPIClient:
        if self._client is None:
            self._client = build_runtime_api_client_from_config(self.app_config)
        return self._client


class ServerAuthAccountService:
    """Execute explicit active-server account operations without local identity mirroring."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient] = None,
        *,
        policy_enforcer: Any | None = None,
        client_provider: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer
        self.client_provider = client_provider

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerAuthAccountService":
        return cls(
            client_provider=_ConfigBackedClientProvider(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_app_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerAuthAccountService":
        return cls.from_config(app_config, policy_enforcer=policy_enforcer)

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerAuthAccountService":
        return cls(client_provider=provider, policy_enforcer=policy_enforcer)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server auth/profile/account operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server account action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @classmethod
    def _dump(cls, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [cls._dump(item) for item in response]
        if isinstance(response, dict):
            return dict(response)
        return response

    async def login(self, *, username: str, password: str, set_bearer_token: bool = True) -> dict[str, Any]:
        self._enforce("auth.identity.launch.server")
        return self._dump(
            await self._require_client().login(
                username,
                password,
                set_bearer_token=set_bearer_token,
            )
        )

    async def refresh_auth_token(
        self,
        request_data: RefreshTokenRequest,
        *,
        set_bearer_token: bool = True,
    ) -> dict[str, Any]:
        self._enforce("auth.identity.update.server")
        return self._dump(
            await self._require_client().refresh_auth_token(
                request_data,
                set_bearer_token=set_bearer_token,
            )
        )

    async def logout(self, *, all_devices: bool = False, clear_bearer_token: bool = True) -> dict[str, Any]:
        self._enforce("auth.identity.delete.server")
        return self._dump(
            await self._require_client().logout(
                all_devices=all_devices,
                clear_bearer_token=clear_bearer_token,
            )
        )

    async def list_auth_sessions(self) -> list[dict[str, Any]]:
        self._enforce("auth.sessions.list.server")
        return self._dump(await self._require_client().list_auth_sessions())

    async def revoke_auth_session(self, session_id: int) -> dict[str, Any]:
        self._enforce("auth.sessions.delete.server")
        return self._dump(await self._require_client().revoke_auth_session(session_id))

    async def revoke_all_auth_sessions(self) -> dict[str, Any]:
        self._enforce("auth.sessions.delete.server")
        return self._dump(await self._require_client().revoke_all_auth_sessions())

    async def get_user_profile_catalog(self, *, if_none_match: str | None = None) -> dict[str, Any]:
        self._enforce("auth.profile.list.server")
        return self._dump(await self._require_client().get_user_profile_catalog(if_none_match=if_none_match))

    async def get_current_user_profile(
        self,
        *,
        sections: str | list[str] | None = None,
        include_sources: bool = False,
    ) -> dict[str, Any]:
        self._enforce("auth.profile.detail.server")
        return self._dump(
            await self._require_client().get_current_user_profile(
                sections=sections,
                include_sources=include_sources,
            )
        )

    async def update_current_user_profile(self, request_data: UserProfileUpdateRequest) -> dict[str, Any]:
        self._enforce("auth.profile.update.server")
        return self._dump(await self._require_client().update_current_user_profile(request_data))

    async def register_user(self, request_data: RegisterRequest) -> dict[str, Any]:
        self._enforce("auth.registration.create.server")
        return self._dump(await self._require_client().register_user(request_data))

    async def change_password(self, request_data: PasswordChangeRequest) -> dict[str, Any]:
        self._enforce("auth.security.update.server")
        return self._dump(await self._require_client().change_password(request_data))

    async def request_password_reset(self, request_data: PasswordResetRequest) -> dict[str, Any]:
        self._enforce("auth.security.launch.server")
        return self._dump(await self._require_client().request_password_reset(request_data))

    async def reset_password(self, request_data: PasswordResetConfirm) -> dict[str, Any]:
        self._enforce("auth.security.update.server")
        return self._dump(await self._require_client().reset_password(request_data))

    async def verify_email(self, token: str) -> dict[str, Any]:
        self._enforce("auth.security.update.server")
        return self._dump(await self._require_client().verify_email(token))

    async def resend_verification(self, email: str) -> dict[str, Any]:
        self._enforce("auth.security.launch.server")
        return self._dump(await self._require_client().resend_verification(email))

    async def request_magic_link(self, email: str) -> dict[str, Any]:
        self._enforce("auth.security.launch.server")
        return self._dump(await self._require_client().request_magic_link(email))

    async def verify_magic_link(self, token: str, *, set_bearer_token: bool = True) -> dict[str, Any]:
        self._enforce("auth.identity.launch.server")
        return self._dump(
            await self._require_client().verify_magic_link(token, set_bearer_token=set_bearer_token)
        )

    async def setup_mfa(self) -> dict[str, Any]:
        self._enforce("auth.security.launch.server")
        return self._dump(await self._require_client().setup_mfa())

    async def verify_mfa_setup(self, token: str) -> dict[str, Any]:
        self._enforce("auth.security.update.server")
        return self._dump(await self._require_client().verify_mfa_setup(token))

    async def disable_mfa(self, password: str) -> dict[str, Any]:
        self._enforce("auth.security.update.server")
        return self._dump(await self._require_client().disable_mfa(password))

    async def complete_mfa_login(
        self,
        *,
        session_token: str,
        mfa_token: str,
        set_bearer_token: bool = True,
    ) -> dict[str, Any]:
        self._enforce("auth.identity.launch.server")
        return self._dump(
            await self._require_client().complete_mfa_login(
                session_token=session_token,
                mfa_token=mfa_token,
                set_bearer_token=set_bearer_token,
            )
        )

    async def list_user_api_keys(self) -> list[dict[str, Any]]:
        self._enforce("auth.api_keys.list.server")
        return self._dump(await self._require_client().list_user_api_keys())

    async def create_user_api_key(self, request_data: APIKeyCreateRequest) -> dict[str, Any]:
        self._enforce("auth.api_keys.create.server")
        return self._dump(await self._require_client().create_user_api_key(request_data))

    async def create_virtual_api_key(
        self,
        request_data: VirtualAPIKeyCreateRequest | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._enforce("auth.api_keys.create.server")
        return self._dump(await self._require_client().create_virtual_api_key(request_data, **kwargs))

    async def rotate_user_api_key(
        self,
        key_id: int,
        request_data: APIKeyRotateRequest | None = None,
    ) -> dict[str, Any]:
        self._enforce("auth.api_keys.update.server")
        return self._dump(await self._require_client().rotate_user_api_key(key_id, request_data))

    async def revoke_user_api_key(self, key_id: int) -> dict[str, Any]:
        self._enforce("auth.api_keys.delete.server")
        return self._dump(await self._require_client().revoke_user_api_key(key_id))

    async def upsert_user_provider_key(self, request_data: UserProviderKeyUpsertRequest) -> dict[str, Any]:
        self._enforce("auth.provider_keys.create.server")
        return self._dump(await self._require_client().upsert_user_provider_key(request_data))

    async def list_user_provider_keys(self) -> dict[str, Any]:
        self._enforce("auth.provider_keys.list.server")
        return self._dump(await self._require_client().list_user_provider_keys())

    async def test_user_provider_key(self, request_data: ProviderKeyTestRequest) -> dict[str, Any]:
        self._enforce("auth.provider_keys.validate.server")
        return self._dump(await self._require_client().test_user_provider_key(request_data))

    async def authorize_openai_oauth(
        self,
        request_data: OpenAIOAuthAuthorizeRequest | None = None,
    ) -> dict[str, Any]:
        self._enforce("auth.provider_keys.create.server")
        return self._dump(await self._require_client().authorize_openai_oauth(request_data))

    async def complete_openai_oauth_callback(
        self,
        *,
        code: str,
        state: str,
        redirect: bool = False,
    ) -> dict[str, Any]:
        self._enforce("auth.provider_keys.update.server")
        return self._dump(
            await self._require_client().complete_openai_oauth_callback(
                code=code,
                state=state,
                redirect=redirect,
            )
        )

    async def get_openai_oauth_status(self) -> dict[str, Any]:
        self._enforce("auth.provider_keys.detail.server")
        return self._dump(await self._require_client().get_openai_oauth_status())

    async def refresh_openai_oauth(self) -> dict[str, Any]:
        self._enforce("auth.provider_keys.update.server")
        return self._dump(await self._require_client().refresh_openai_oauth())

    async def disconnect_openai_oauth(self) -> bool:
        self._enforce("auth.provider_keys.delete.server")
        return bool(await self._require_client().disconnect_openai_oauth())

    async def switch_openai_credential_source(
        self,
        request_data: OpenAICredentialSourceSwitchRequest,
    ) -> dict[str, Any]:
        self._enforce("auth.provider_keys.update.server")
        return self._dump(await self._require_client().switch_openai_credential_source(request_data))

    async def delete_user_provider_key(self, provider: str) -> bool:
        self._enforce("auth.provider_keys.delete.server")
        return bool(await self._require_client().delete_user_provider_key(provider))

    async def get_user_storage_quota(self) -> dict[str, Any]:
        self._enforce("auth.storage.detail.server")
        return self._dump(await self._require_client().get_user_storage_quota())

    async def recalculate_user_storage_quota(self) -> dict[str, Any]:
        self._enforce("auth.storage.update.server")
        return self._dump(await self._require_client().recalculate_user_storage_quota())

    async def list_storage_files(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("auth.storage.list.server")
        return self._dump(await self._require_client().list_storage_files(**kwargs))

    async def get_storage_file(self, file_id: int) -> dict[str, Any]:
        self._enforce("auth.storage.detail.server")
        return self._dump(await self._require_client().get_storage_file(file_id))

    async def download_storage_file(self, file_id: int) -> Any:
        self._enforce("auth.storage.export.server")
        return await self._require_client().download_storage_file(file_id)

    async def update_storage_file(self, file_id: int, request_data: GeneratedFileUpdate) -> dict[str, Any]:
        self._enforce("auth.storage.update.server")
        return self._dump(await self._require_client().update_storage_file(file_id, request_data))

    async def delete_storage_file(self, file_id: int, *, hard_delete: bool = False) -> dict[str, Any]:
        self._enforce("auth.storage.delete.server")
        return self._dump(await self._require_client().delete_storage_file(file_id, hard_delete=hard_delete))

    async def bulk_delete_storage_files(self, request_data: BulkDeleteRequest) -> dict[str, Any]:
        self._enforce("auth.storage.delete.server")
        return self._dump(await self._require_client().bulk_delete_storage_files(request_data))

    async def bulk_move_storage_files(self, request_data: BulkMoveRequest) -> dict[str, Any]:
        self._enforce("auth.storage.update.server")
        return self._dump(await self._require_client().bulk_move_storage_files(request_data))

    async def list_storage_folders(self) -> dict[str, Any]:
        self._enforce("auth.storage.list.server")
        return self._dump(await self._require_client().list_storage_folders())

    async def create_storage_folder(self, name: str) -> dict[str, Any]:
        self._enforce("auth.storage.create.server")
        return self._dump(await self._require_client().create_storage_folder(name))

    async def list_least_accessed_storage_files(self, *, limit: int = 20) -> dict[str, Any]:
        self._enforce("auth.storage.list.server")
        return self._dump(await self._require_client().list_least_accessed_storage_files(limit=limit))

    async def get_storage_usage(self) -> dict[str, Any]:
        self._enforce("auth.storage.detail.server")
        return self._dump(await self._require_client().get_storage_usage())

    async def get_storage_usage_breakdown(self) -> dict[str, Any]:
        self._enforce("auth.storage.detail.server")
        return self._dump(await self._require_client().get_storage_usage_breakdown())

    async def list_storage_trash(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        self._enforce("auth.storage.list.server")
        return self._dump(await self._require_client().list_storage_trash(offset=offset, limit=limit))

    async def restore_storage_file(self, file_id: int) -> dict[str, Any]:
        self._enforce("auth.storage.update.server")
        return self._dump(await self._require_client().restore_storage_file(file_id))

    async def permanently_delete_storage_file(self, file_id: int) -> dict[str, Any]:
        self._enforce("auth.storage.delete.server")
        return self._dump(await self._require_client().permanently_delete_storage_file(file_id))
