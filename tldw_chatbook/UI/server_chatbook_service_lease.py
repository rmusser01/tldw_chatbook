from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

from ..Chatbooks.server_chatbook_service import ServerChatbookService
from ..runtime_policy.bootstrap import derive_configured_server_binding


@dataclass(frozen=True, slots=True)
class ServerChatbookServiceLease:
    service: Any
    close_owner: Any | None = None


def server_chatbook_service_lease(
    app_instance: Any,
    *,
    config: Mapping[str, Any] | None,
    policy_enforcer: Any | None = None,
    client_provider: Any | None = None,
) -> ServerChatbookServiceLease:
    app_service = getattr(app_instance, "server_chatbook_service", None)
    if app_service is not None:
        return ServerChatbookServiceLease(service=app_service)

    app_provider = getattr(app_instance, "server_context_provider", None)
    if app_provider is not None:
        return ServerChatbookServiceLease(
            service=ServerChatbookService.from_server_context_provider(
                app_provider,
                policy_enforcer=policy_enforcer,
            )
        )

    if client_provider is not None:
        return ServerChatbookServiceLease(
            service=ServerChatbookService.from_config(
                config or {},
                client_provider=client_provider,
                policy_enforcer=policy_enforcer,
            ),
            close_owner=client_provider,
        )

    if not derive_configured_server_binding(config).server_configured:
        raise ValueError("TLDW API base URL is not configured.")

    service = ServerChatbookService.from_config(
        config or {},
        policy_enforcer=policy_enforcer,
    )
    return ServerChatbookServiceLease(service=service, close_owner=service)


async def close_server_chatbook_service_lease(lease: ServerChatbookServiceLease) -> None:
    close_owner = lease.close_owner
    if close_owner is None:
        return

    client = getattr(close_owner, "client", None)
    if client is not None:
        close = getattr(client, "close", None)
        if callable(close):
            close_result = close()
            if inspect.isawaitable(close_result):
                await close_result

    provider = getattr(close_owner, "client_provider", close_owner)
    close_cached_client = getattr(provider, "close_cached_client", None)
    if callable(close_cached_client):
        close_result = close_cached_client()
        if inspect.isawaitable(close_result):
            await close_result
