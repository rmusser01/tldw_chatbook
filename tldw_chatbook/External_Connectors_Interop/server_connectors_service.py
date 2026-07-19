"""Server-backed external connector service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerConnectorsService:
    """Policy-gated access to server connector account/source APIs."""

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
    ) -> "ServerConnectorsService":
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
    ) -> "ServerConnectorsService":
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
        raise ValueError("TLDW API client is required for server connector operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server connector action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerConnectorsService._dump(item) for item in response]
        if isinstance(response, (dict, bool)):
            return response
        return dict(response or {})

    async def list_providers(self) -> list[dict[str, Any]]:
        self._enforce("connectors.providers.list.server")
        return self._dump(await self._require_client().list_connector_providers())

    async def authorize_provider(
        self,
        provider: str,
        *,
        state: str | None = None,
        scopes: list[str] | str | None = None,
    ) -> dict[str, Any]:
        self._enforce("connectors.providers.launch.server")
        return self._dump(
            await self._require_client().authorize_connector_provider(provider, state=state, scopes=scopes)
        )

    async def complete_oauth_callback(
        self,
        provider: str,
        *,
        code: str | None = None,
        oauth_token: str | None = None,
        oauth_verifier: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("connectors.providers.launch.server")
        return self._dump(
            await self._require_client().complete_connector_oauth_callback(
                provider,
                code=code,
                oauth_token=oauth_token,
                oauth_verifier=oauth_verifier,
                state=state,
            )
        )

    async def list_accounts(self) -> list[dict[str, Any]]:
        self._enforce("connectors.accounts.list.server")
        return self._dump(await self._require_client().list_connector_accounts())

    async def delete_account(self, account_id: int) -> bool:
        self._enforce("connectors.accounts.delete.server")
        return bool(await self._require_client().delete_connector_account(int(account_id)))

    async def browse_sources(
        self,
        provider: str,
        *,
        account_id: int,
        parent_remote_id: str | None = None,
        page_size: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("connectors.sources.list.server")
        return self._dump(
            await self._require_client().browse_connector_sources(
                provider,
                account_id=account_id,
                parent_remote_id=parent_remote_id,
                page_size=page_size,
                cursor=cursor,
            )
        )

    async def create_source(
        self,
        *,
        account_id: int,
        provider: str,
        remote_id: str,
        type: str,
        path: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import ConnectorSourceCreateRequest

        self._enforce("connectors.sources.create.server")
        request = ConnectorSourceCreateRequest(
            account_id=account_id,
            provider=provider,  # type: ignore[arg-type]
            remote_id=remote_id,
            type=type,  # type: ignore[arg-type]
            path=path,
            options=options or {},
        )
        return self._dump(await self._require_client().create_connector_source(request))

    async def list_sources(self) -> list[dict[str, Any]]:
        self._enforce("connectors.sources.list.server")
        return self._dump(await self._require_client().list_connector_sources())

    async def update_source(
        self,
        source_id: int,
        *,
        enabled: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import ConnectorSourcePatchRequest

        self._enforce("connectors.sources.update.server")
        request = ConnectorSourcePatchRequest(enabled=enabled, options=options)
        return self._dump(await self._require_client().update_connector_source(int(source_id), request))

    async def import_source(self, source_id: int) -> dict[str, Any]:
        self._enforce("connectors.sources.launch.server")
        return self._dump(await self._require_client().import_connector_source(int(source_id)))

    async def get_source_sync_status(self, source_id: int) -> dict[str, Any]:
        self._enforce("connectors.sources.observe.server")
        return self._dump(await self._require_client().get_connector_source_sync_status(int(source_id)))

    async def trigger_source_sync(self, source_id: int) -> dict[str, Any]:
        self._enforce("connectors.sources.launch.server")
        return self._dump(await self._require_client().trigger_connector_source_sync(int(source_id)))

    async def get_job_status(self, job_id: int | str) -> dict[str, Any]:
        self._enforce("connectors.jobs.observe.server")
        return self._dump(await self._require_client().get_connector_job_status(job_id))
