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

from tldw_chatbook.runtime_policy.types import PolicyDeniedError


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


class ServerClientPolicyError(ServerClientValidationError):
    """A local runtime-mode policy refused the action before any network I/O.

    Subclass of :class:`ServerClientValidationError` so existing catches keep
    working; the refined type lets consumers distinguish "not applicable in
    this runtime mode" from a real failure — the sync engine must not persist
    a refusal as a sync error (task-2722).
    """


class ServerClientServerError(ServerClientError):
    """Server returned 5xx."""


@dataclass(slots=True)
class ServerClientConfig:
    """Configuration for the scheduling server client."""

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
        """Initialize the client.

        Args:
            notifications_service: Service that implements the reminder CRUD
                contract, or ``None`` if no scheduling server is connected.
            config: Client configuration. Defaults to a new
                :class:`ServerClientConfig` instance.
        """
        self.notifications_service = notifications_service
        self.config = config or ServerClientConfig()

    def set_notifications_service(self, notifications_service: Any | None) -> None:
        """Inject or refresh the underlying notifications service.

        Args:
            notifications_service: The service to use for future reminder
                operations, or ``None`` to disconnect the server.
        """
        self.notifications_service = notifications_service

    @staticmethod
    def _strip_local_only_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove kwargs that are only meaningful to the local scheduler.

        Args:
            kwargs: Keyword arguments destined for the notifications service.

        Returns:
            A copy of ``kwargs`` with local-only keys removed.
        """
        return {k: v for k, v in kwargs.items() if k != "idempotency_key"}

    async def _call_with_retry(
        self,
        method_name: str,
        *args: Any,
        retry: bool = True,
        is_read: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call a notifications-service method with retries and error mapping.

        Args:
            method_name: Name of the method to invoke on the injected service.
            *args: Positional arguments for the service method.
            retry: Whether to retry on retriable failures. Defaults to ``True``.
            is_read: Whether this is a read operation, which uses a shorter
                timeout. Defaults to ``False``.
            **kwargs: Keyword arguments for the service method.

        Returns:
            The dictionary returned by the service method.

        Raises:
            ServerUnavailableError: If no notifications service is configured.
            ServerClientNotFoundError: If the server reports the task was not found.
            ServerClientValidationError: If the request is rejected by policy or
                the server returns a client error.
            ServerClientServerError: If the server returns a server error and
                retries are exhausted.
            ServerClientTimeoutError: If the request times out and retries are
                exhausted.
            ServerClientError: For other failures after retries are exhausted.
        """
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
                raise ServerClientPolicyError(str(exc)) from exc
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
                if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                    status = getattr(exc.response, "status_code", None)
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
        """Create a new reminder.

        Args:
            **payload: Reminder fields to pass to the notifications service.

        Returns:
            The created reminder as returned by the service.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientValidationError: If the request is rejected locally or by
                the server.
            ServerClientTimeoutError: If the request times out.
        """
        return await self._call_with_retry("create_reminder", retry=False, **payload)

    async def update_reminder(self, task_id: str, **payload: Any) -> dict[str, Any]:
        """Update an existing reminder.

        Args:
            task_id: Identifier of the reminder to update.
            **payload: Reminder fields to update.

        Returns:
            The updated reminder as returned by the service.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientNotFoundError: If the reminder does not exist server-side.
            ServerClientValidationError: If the request is rejected locally or by
                the server.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry("update_reminder", task_id, **payload)

    async def delete_reminder(self, task_id: str) -> dict[str, Any]:
        """Delete a reminder.

        Args:
            task_id: Identifier of the reminder to delete.

        Returns:
            The service response after deletion.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientNotFoundError: If the reminder does not exist server-side.
            ServerClientValidationError: If the request is rejected locally or by
                the server.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry("delete_reminder", task_id)

    async def list_reminders(self) -> dict[str, Any]:
        """List all reminders.

        Returns:
            The service response containing the reminder list.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry("list_reminders", is_read=True)

    async def get_reminder(self, task_id: str) -> dict[str, Any]:
        """Fetch a single reminder.

        Args:
            task_id: Identifier of the reminder to retrieve.

        Returns:
            The requested reminder as returned by the service.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientNotFoundError: If the reminder does not exist server-side.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry("get_reminder", task_id, is_read=True)

    async def list_automation_definitions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List the server's automation definitions (ADR-077 control plane).

        Args:
            limit: Page size to request from the server.
            offset: Pagination offset to request from the server.

        Returns:
            The definition list response (``items``/``total``/pagination).

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientValidationError: If the request is rejected by policy
                or the server.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry(
            "list_scheduled_automations", limit=limit, offset=offset, is_read=True
        )

    async def list_automation_definition_audit(
        self,
        definition_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """List one definition's durable execution-audit trail (ADR-077 AC#4).

        Args:
            definition_id: The server definition whose trail to fetch.
            limit: Page size to request from the server.
            offset: Pagination offset to request from the server.
            event_type: Optional event-type filter (e.g. ``run_succeeded``);
                kept for parity with the notifications service and API
                client so callers through this layer can filter too.

        Returns:
            The audit list response (``items`` carrying ``run_{status}``
            events with run ids for correlating results, plus pagination).

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientNotFoundError: If the definition does not exist
                server-side.
            ServerClientValidationError: If the request is rejected by policy
                or the server.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry(
            "list_scheduled_automation_audit",
            definition_id,
            limit=limit,
            offset=offset,
            event_type=event_type,
            is_read=True,
        )

    async def run_automation_definition_now(self, definition_id: str) -> dict[str, Any]:
        """Trigger one immediate server-side execution of a definition.

        Args:
            definition_id: The server definition to dispatch.

        Returns:
            The run reference (``definition_id``/``run_slot_utc``/``job_id``/
            ``deduped``) for correlating with the eventual result
            notification.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientNotFoundError: If the definition does not exist
                server-side.
            ServerClientValidationError: If the definition refuses the run
                (paused/archived lifecycle) or policy denies it.
            ServerClientServerError: If the server returns a server error --
                NOT retried: a retried trigger could enqueue a second run,
                so the caller sees the failure instead (the server-side
                run-slot dedupe only collapses triggers sharing a slot).
            ServerClientTimeoutError: If the request times out (not retried,
                for the same reason).

        No ``idempotency_key`` is threaded through: this layer never retries
        a trigger, each user-initiated Run-now is intentionally a distinct
        run, and ``_strip_local_only_kwargs`` would drop the key before it
        reached the network anyway.
        """
        return await self._call_with_retry(
            "run_scheduled_automation_now",
            definition_id,
            retry=False,
        )

    async def list_automation_results(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        definition_id: str | None = None,
        review_state: str | None = None,
    ) -> dict[str, Any]:
        """List the server's scheduled-task results (spec §4.2).

        Args:
            limit: Page size to request from the server.
            offset: Pagination offset to request from the server.
            definition_id: Optional filter to one definition's results.
            review_state: Optional review-state filter.

        Returns:
            The result list response (``items``/``total``/pagination).

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientValidationError: If the request is rejected by policy
                or the server.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry(
            "list_scheduled_automation_results",
            limit=limit,
            offset=offset,
            definition_id=definition_id,
            review_state=review_state,
            is_read=True,
        )

    async def review_automation_result(
        self,
        result_id: str,
        review_state: str,
        *,
        review_note: str | None = None,
    ) -> dict[str, Any]:
        """Set one result's review state, retried on failure.

        Unlike ``run_automation_definition_now``, replaying the same review
        state is idempotent server-side -- a retry cannot double-fire
        anything, so this call keeps the default retry behavior.

        Args:
            result_id: The server result to update.
            review_state: New review state (``read``/``dismissed``/etc).
            review_note: Optional free-text note attached to the review.

        Returns:
            The updated result row.

        Raises:
            ServerUnavailableError: If no scheduling server is connected.
            ServerClientNotFoundError: If the result does not exist
                server-side (e.g. retired).
            ServerClientValidationError: If the request is rejected by policy
                or the server.
            ServerClientServerError: If the server returns a server error after
                retries are exhausted.
            ServerClientTimeoutError: If the request times out after retries.
        """
        return await self._call_with_retry(
            "review_scheduled_automation_result",
            result_id,
            review_state,
            review_note=review_note,
        )
