"""Server client for scheduling reminders."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError
except ImportError:  # pragma: no cover
    PolicyDeniedError = None  # type: ignore[assignment,misc]


class ServerClientError(Exception):
    """Base class for all server-client failures."""


class ServerUnavailableError(ServerClientError):
    """Raised when the server client is invoked while no server is connected."""


class ServerClientTimeoutError(ServerClientError):
    """Request to the server timed out."""


class ServerClientNotFoundError(ServerClientError):
    """Server returned 404; the task was deleted server-side."""


class ServerClientValidationError(ServerClientError):
    """Server returned 4xx other than 404, or a local policy denied the action."""


class ServerClientServerError(ServerClientError):
    """Server returned 5xx."""


@dataclass(slots=True)
class ServerClientConfig:
    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0


class SchedulingServerClient:
    """Async client that delegates scheduling operations to a notifications service.

    The client is a thin wrapper around an injected notifications service. All
    methods raise :class:`ServerUnavailableError` when no service has been
    configured, so callers can distinguish "server missing" from actual request
    failures.
    """

    def __init__(
        self,
        notifications_service: Any | None = None,
        config: ServerClientConfig | None = None,
    ) -> None:
        self.notifications_service = notifications_service
        self.config = config or ServerClientConfig()

    def set_notifications_service(self, notifications_service: Any | None) -> None:
        """Inject or refresh the underlying notifications service."""
        self.notifications_service = notifications_service

    def _is_available(self) -> bool:
        return self.notifications_service is not None

    @staticmethod
    def _strip_local_only_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove kwargs that the server service does not accept."""
        return {k: v for k, v in kwargs.items() if k != "idempotency_key"}

    async def _call_with_retry(
        self,
        method_name: str,
        *args: Any,
        retry: bool = True,
        is_read: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        service = self.notifications_service
        if service is None:
            raise ServerUnavailableError("server not available")

        kwargs = self._strip_local_only_kwargs(kwargs)
        method = getattr(service, method_name)
        timeout = self.config.timeout if is_read else self.config.timeout * 3
        last_error: Exception | None = None
        error_cls: type[ServerClientError] = ServerClientError

        attempts = self.config.max_retries + 1 if retry else 1
        for attempt in range(attempts):
            try:
                coro = method(*args, **kwargs)
                return await asyncio.wait_for(coro, timeout=timeout)
            except PolicyDeniedError as exc:
                raise ServerClientValidationError(str(exc)) from exc
            except ServerClientNotFoundError:
                raise
            except ServerClientValidationError:
                raise
            except ServerClientServerError as exc:
                last_error = exc
                error_cls = ServerClientServerError
            except ServerClientTimeoutError as exc:
                last_error = exc
                error_cls = ServerClientTimeoutError
            except asyncio.TimeoutError as exc:
                last_error = exc
                error_cls = ServerClientTimeoutError
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status == 404:
                    raise ServerClientNotFoundError(str(exc)) from exc
                if status is not None and 400 <= status < 500:
                    raise ServerClientValidationError(str(exc)) from exc
                if status is not None and 500 <= status < 600:
                    error_cls = ServerClientServerError
                elif httpx is not None and isinstance(exc, httpx.TimeoutException):
                    error_cls = ServerClientTimeoutError
                elif httpx is not None and isinstance(
                    exc, (httpx.ConnectError, httpx.NetworkError)
                ):
                    error_cls = ServerClientServerError
                else:
                    error_cls = ServerClientError

            if not retry or attempt == attempts - 1:
                raise error_cls(str(last_error)) from last_error

            delay = self.config.retry_delay * (2**attempt)
            delay += random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay)

        raise ServerClientError("unexpected end of retry loop")

    async def create_reminder(self, **payload: Any) -> dict[str, Any]:
        return await self._call_with_retry("create_reminder", retry=False, **payload)

    async def update_reminder(self, task_id: str, **payload: Any) -> dict[str, Any]:
        return await self._call_with_retry("update_reminder", task_id, **payload)

    async def delete_reminder(self, task_id: str) -> dict[str, Any]:
        return await self._call_with_retry("delete_reminder", task_id)

    async def list_reminders(self) -> dict[str, Any]:
        return await self._call_with_retry("list_reminders", is_read=True)

    async def get_reminder(self, task_id: str) -> dict[str, Any]:
        return await self._call_with_retry("get_reminder", task_id, is_read=True)
