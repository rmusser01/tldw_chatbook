"""Server-backed workspace sharing and share-link service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerSharingService:
    """Policy-gated access to remote sharing APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerSharingService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerSharingService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server sharing operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server sharing action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    @staticmethod
    def _dump_list(response: Any) -> list[dict[str, Any]]:
        return [ServerSharingService._dump(item) for item in list(response or [])]

    async def create_link(
        self,
        *,
        resource_type: str,
        resource_id: str,
        access_level: str = "view_chat",
        allow_clone: bool = True,
        password: str | None = None,
        max_uses: int | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import CreateTokenRequest

        self._enforce("sharing.links.create.server")
        request = CreateTokenRequest(
            resource_type=resource_type,  # type: ignore[arg-type]
            resource_id=resource_id,
            access_level=access_level,  # type: ignore[arg-type]
            allow_clone=allow_clone,
            password=password,
            max_uses=max_uses,
            expires_at=expires_at,
        )
        return self._dump(await self._require_client().create_share_token(request))

    async def list_links(self) -> dict[str, Any]:
        self._enforce("sharing.links.list.server")
        return self._dump(await self._require_client().list_share_tokens())

    async def revoke_link(self, token_id: int) -> dict[str, Any]:
        self._enforce("sharing.links.revoke.server")
        return await self._require_client().revoke_share_token(token_id)

    async def inspect_public_link(self, token: str) -> dict[str, Any]:
        self._enforce("sharing.links.inspect.server")
        return self._dump(await self._require_client().preview_public_share(token))

    async def verify_public_link_password(self, token: str, *, password: str) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import VerifyPasswordRequest

        self._enforce("sharing.links.launch.server")
        request = VerifyPasswordRequest(password=password)
        return self._dump(await self._require_client().verify_public_share_password(token, request))

    async def import_public_link(self, token: str) -> dict[str, Any]:
        self._enforce("sharing.links.launch.server")
        return self._dump(await self._require_client().import_public_share(token))

    async def observe_link_events(
        self,
        *,
        owner_user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce("sharing.links.observe.server")
        return self._dump(
            await self._require_client().list_sharing_audit_events(
                owner_user_id=owner_user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                limit=limit,
                offset=offset,
            )
        )

    async def share_workspace(
        self,
        *,
        workspace_id: str,
        share_scope_type: str,
        share_scope_id: int,
        access_level: str = "view_chat",
        allow_clone: bool = True,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import ShareWorkspaceRequest

        self._enforce("sharing.permissions.configure.server")
        request = ShareWorkspaceRequest(
            share_scope_type=share_scope_type,  # type: ignore[arg-type]
            share_scope_id=share_scope_id,
            access_level=access_level,  # type: ignore[arg-type]
            allow_clone=allow_clone,
        )
        return self._dump(await self._require_client().share_workspace(workspace_id, request))

    async def list_workspace_shares(
        self,
        workspace_id: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any]:
        self._enforce("sharing.permissions.configure.server")
        return self._dump(
            await self._require_client().list_workspace_shares(
                workspace_id,
                include_revoked=include_revoked,
            )
        )

    async def update_share(
        self,
        share_id: int,
        *,
        access_level: str | None = None,
        allow_clone: bool | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import UpdateShareRequest

        self._enforce("sharing.permissions.configure.server")
        request = UpdateShareRequest(
            access_level=access_level,  # type: ignore[arg-type]
            allow_clone=allow_clone,
        )
        return self._dump(await self._require_client().update_share(share_id, request))

    async def revoke_share(self, share_id: int) -> dict[str, Any]:
        self._enforce("sharing.permissions.configure.server")
        return await self._require_client().revoke_share(share_id)

    async def list_shared_with_me(self) -> dict[str, Any]:
        self._enforce("sharing.links.list.server")
        return self._dump(await self._require_client().list_shared_with_me())

    async def get_shared_workspace(self, share_id: int) -> dict[str, Any]:
        self._enforce("sharing.links.inspect.server")
        return self._dump(await self._require_client().get_shared_workspace(share_id))

    async def clone_shared_workspace(self, share_id: int, *, new_name: str | None = None) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import CloneWorkspaceRequest

        self._enforce("sharing.links.launch.server")
        request = CloneWorkspaceRequest(new_name=new_name)
        return self._dump(await self._require_client().clone_shared_workspace(share_id, request))

    async def list_shared_workspace_sources(self, share_id: int) -> list[dict[str, Any]]:
        self._enforce("sharing.links.inspect.server")
        return self._dump_list(await self._require_client().list_shared_workspace_sources(share_id))

    async def get_shared_workspace_media(self, share_id: int, media_id: int) -> dict[str, Any]:
        self._enforce("sharing.links.inspect.server")
        return self._dump(await self._require_client().get_shared_workspace_media(share_id, media_id))

    async def chat_with_shared_workspace(
        self,
        share_id: int,
        *,
        query: str,
        model: str | None = None,
        api_name: str | None = None,
        system_message: str | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SharedChatRequest

        self._enforce("sharing.links.launch.server")
        request = SharedChatRequest(
            query=query,
            model=model,
            api_name=api_name,
            system_message=system_message,
        )
        return await self._require_client().chat_with_shared_workspace(share_id, request)
