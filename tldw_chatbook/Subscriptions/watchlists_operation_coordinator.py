"""Application-owned supervision for durable manual Watchlists operations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from .briefing_service import (
    INTERRUPTED_ERROR,
    accept_briefing,
    execute_accepted_briefing,
)

if TYPE_CHECKING:
    from ..DB.Subscriptions_DB import SubscriptionsDB
    from .local_watchlists_service import LocalWatchlistsService


_CHECK_FAILURE = "Watchlists source check failed. Try again."
_BRIEFING_FAILURE = "Watchlists briefing generation failed. Try again."
_TERMINAL_STATUSES = {
    "check": frozenset({"completed", "failed", "cancelled"}),
    "briefing": frozenset({"complete", "empty", "failed"}),
}
_TERMINAL_RETRY_LIMIT = 3


class WatchlistsOperationCoordinator:
    """Own accepted manual work for the lifetime of the running app loop."""

    def __init__(
        self,
        *,
        local_service: "LocalWatchlistsService",
        briefing_db: "SubscriptionsDB",
    ) -> None:
        self._local_service = local_service
        self._briefing_db = briefing_db
        self._check_slots = asyncio.Semaphore(4)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._kinds: dict[str, str] = {}
        self._briefing_terminal_errors: dict[str, str] = {}
        self._reconcile_tasks: dict[str, asyncio.Task[None]] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._accepting = True
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def active_receipt_ids(self) -> tuple[str, ...]:
        """Canonical identities for tasks still owned by this coordinator."""
        return tuple(self._tasks)

    def bind_running_loop(self) -> None:
        """Bind ownership to the caller's currently running application loop."""
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("Watchlists coordinator is bound to another loop")
        self._loop = loop

    def _assert_accepting(self) -> None:
        if not self._accepting:
            raise RuntimeError("Watchlists operation coordinator is shutting down")
        self.bind_running_loop()

    def _start(self, receipt_id: str, kind: str, coroutine: Any) -> None:
        current = self._tasks.get(receipt_id)
        if current is not None and not current.done():
            coroutine.close()
            return
        reconcile = self._reconcile_tasks.pop(receipt_id, None)
        if reconcile is not None and not reconcile.done():
            reconcile.cancel()
            self._retain_background(reconcile)
        task = asyncio.create_task(coroutine, name=f"watchlists:{receipt_id}")
        self._tasks[receipt_id] = task
        self._kinds[receipt_id] = kind
        task.add_done_callback(
            lambda finished, key=receipt_id: self._consume_terminal(key, finished)
        )

    def _consume_terminal(self, receipt_id: str, task: asyncio.Task[None]) -> None:
        """Consume the executor terminal and reconcile its durable receipt."""
        self._consume_exception(task)
        if self._tasks.get(receipt_id) is not task:
            return
        self._start_reconciliation(receipt_id, task)

    def _start_reconciliation(
        self,
        receipt_id: str,
        owner: asyncio.Task[None],
    ) -> None:
        """Resume terminal persistence without repeating the owned effect."""
        current = self._reconcile_tasks.get(receipt_id)
        if current is not None and not current.done():
            return
        reconcile = asyncio.create_task(
            self._reconcile_terminal(receipt_id, owner),
            name=f"watchlists:reconcile:{receipt_id}",
        )
        self._reconcile_tasks[receipt_id] = reconcile
        reconcile.add_done_callback(
            lambda finished, key=receipt_id: self._consume_reconciliation(
                key, finished
            )
        )

    @staticmethod
    def _consume_exception(task: asyncio.Task[Any]) -> None:
        try:
            if not task.cancelled():
                task.exception()
        except BaseException:  # noqa: BLE001 - loop callback must never escape
            logger.error("Watchlists operation task ended unexpectedly.")

    def _consume_reconciliation(
        self,
        receipt_id: str,
        task: asyncio.Task[None],
    ) -> None:
        self._consume_exception(task)
        if self._reconcile_tasks.get(receipt_id) is task:
            self._reconcile_tasks.pop(receipt_id, None)

    def _retain_background(self, task: asyncio.Task[Any]) -> None:
        """Retain and consume a task that may outlive a bounded wait."""
        self._background_tasks.add(task)

        def consume(finished: asyncio.Task[Any]) -> None:
            self._consume_exception(finished)
            self._background_tasks.discard(finished)

        task.add_done_callback(consume)

    def _has_live_executor(self, receipt_id: str) -> bool:
        task = self._tasks.get(receipt_id)
        return task is not None and not task.done()

    async def accept_checks(
        self, source_ids: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Accept validated checks and schedule only newly won receipts."""
        self._assert_accepting()
        receipts = await self._local_service.accept_source_checks(source_ids)
        for receipt in receipts:
            run_id = int(receipt["run_id"])
            canonical_id = f"local:watchlist_run:{run_id}"
            claim_acquired = bool(receipt.pop("_claim_acquired"))
            if claim_acquired or not self._has_live_executor(canonical_id):
                self._start(canonical_id, "check", self._run_check(run_id))
        return receipts

    async def accept_briefing(
        self, watchlist_id: int, preset_id: int | None = None
    ) -> dict[str, Any]:
        """Accept one briefing and schedule only the durable claim winner."""
        self._assert_accepting()
        receipt = await accept_briefing(
            self._briefing_db, int(watchlist_id), preset_id=preset_id
        )
        briefing_id = int(receipt["id"])
        canonical_id = f"local:briefing:{briefing_id}"
        claim_acquired = bool(receipt.pop("_claim_acquired"))
        owner = self._tasks.get(canonical_id)
        if claim_acquired:
            self._briefing_terminal_errors[canonical_id] = _BRIEFING_FAILURE
            self._start(
                canonical_id,
                "briefing",
                self._run_briefing(briefing_id),
            )
        elif owner is None:
            self._briefing_terminal_errors[canonical_id] = INTERRUPTED_ERROR
            self._start(
                canonical_id,
                "briefing",
                self._recover_orphaned_briefing(canonical_id),
            )
        elif owner.done():
            self._start_reconciliation(canonical_id, owner)
        return receipt

    def submit_checks(self, source_ids: Sequence[int]) -> list[dict[str, Any]]:
        """Thread-safe Console provider bridge for durable check acceptance."""
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("Watchlists operation coordinator is unavailable")
        return asyncio.run_coroutine_threadsafe(
            self.accept_checks(source_ids), loop
        ).result()

    def submit_briefing(
        self, watchlist_id: int, preset_id: int | None = None
    ) -> dict[str, Any]:
        """Thread-safe Console provider bridge for briefing acceptance."""
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("Watchlists operation coordinator is unavailable")
        return asyncio.run_coroutine_threadsafe(
            self.accept_briefing(watchlist_id, preset_id), loop
        ).result()

    async def reconcile_startup(self, boundary: Any) -> dict[str, int]:
        """Terminalize active receipts proven to predate this process."""
        from .startup_reconcile import reconcile_interrupted_subscription_work

        return await asyncio.to_thread(
            reconcile_interrupted_subscription_work,
            self._briefing_db,
            boundary,
        )

    async def _run_check(self, run_id: int) -> None:
        async with self._check_slots:
            try:
                await self._local_service.execute_accepted_run(run_id)
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - durable boundary owns terminal state
                try:
                    await self._local_service.record_run_failure(
                        run_id,
                        error=_CHECK_FAILURE,
                    )
                except Exception:  # noqa: BLE001 - terminal may already exist
                    logger.error("Could not terminalize a failed Watchlists check.")

    async def _run_briefing(self, briefing_id: int) -> None:
        try:
            await execute_accepted_briefing(
                self._briefing_db,
                briefing_id,
                scrub_failures=True,
            )
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - durable boundary owns terminal state
            await asyncio.to_thread(self._fail_briefing_if_active, briefing_id)

    async def _recover_orphaned_briefing(self, receipt_id: str) -> None:
        """Terminalize an unowned generating row without replaying its provider."""
        await self._ensure_terminal(
            receipt_id,
            "briefing",
            cancel_check=False,
        )

    def _fail_briefing_if_active(
        self,
        briefing_id: int,
        error: str = _BRIEFING_FAILURE,
    ) -> None:
        try:
            self._briefing_db.transition_briefing(
                briefing_id,
                status="failed",
                error=error,
            )
        except Exception:  # noqa: BLE001 - shutdown/recovery remains best effort
            logger.error("Could not terminalize a failed Watchlists briefing.")

    async def _is_durable_terminal(self, receipt_id: str, kind: str) -> bool:
        numeric_id = int(receipt_id.rsplit(":", 1)[-1])
        try:
            if kind == "check":
                row = await asyncio.to_thread(
                    self._briefing_db.get_watchlist_run_for_agent,
                    numeric_id,
                )
            else:
                row = await asyncio.to_thread(
                    self._briefing_db.get_briefing_for_agent,
                    numeric_id,
                )
        except Exception:  # noqa: BLE001 - transient storage failure is retried
            return False
        return bool(row and row.get("status") in _TERMINAL_STATUSES[kind])

    async def _write_terminal(
        self,
        receipt_id: str,
        kind: str,
        *,
        cancel_check: bool,
    ) -> None:
        numeric_id = int(receipt_id.rsplit(":", 1)[-1])
        try:
            if kind == "check" and cancel_check:
                await self._local_service.cancel_run(numeric_id)
            elif kind == "check":
                await self._local_service.record_run_failure(
                    numeric_id,
                    error=_CHECK_FAILURE,
                )
            else:
                await asyncio.to_thread(
                    self._fail_briefing_if_active,
                    numeric_id,
                    self._briefing_terminal_errors.get(
                        receipt_id,
                        _BRIEFING_FAILURE,
                    ),
                )
        except Exception:  # noqa: BLE001 - verified and retried by the caller
            logger.error("Could not persist a Watchlists terminal receipt.")

    async def _ensure_terminal(
        self,
        receipt_id: str,
        kind: str,
        *,
        cancel_check: bool,
    ) -> bool:
        for attempt in range(_TERMINAL_RETRY_LIMIT):
            if await self._is_durable_terminal(receipt_id, kind):
                return True
            await self._write_terminal(
                receipt_id,
                kind,
                cancel_check=cancel_check,
            )
            if await self._is_durable_terminal(receipt_id, kind):
                return True
            if attempt + 1 < _TERMINAL_RETRY_LIMIT:
                await asyncio.sleep(0)
        return False

    async def _reconcile_terminal(
        self,
        receipt_id: str,
        owner: asyncio.Task[None],
    ) -> None:
        kind = self._kinds.get(receipt_id)
        if kind is None or self._tasks.get(receipt_id) is not owner:
            return
        durable = await self._ensure_terminal(
            receipt_id,
            kind,
            cancel_check=not self._accepting and owner.cancelled(),
        )
        if durable and owner.done() and self._tasks.get(receipt_id) is owner:
            self._tasks.pop(receipt_id, None)
            self._kinds.pop(receipt_id, None)
            self._briefing_terminal_errors.pop(receipt_id, None)

    async def wait_idle(self, *, timeout: float) -> None:
        """Wait boundedly until every currently owned operation settles."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        while True:
            await asyncio.sleep(0)
            pending = {
                task
                for task in (*self._tasks.values(), *self._reconcile_tasks.values())
                if not task.done()
            }
            if not pending:
                await asyncio.sleep(0)
                pending = {
                    task
                    for task in (
                        *self._tasks.values(),
                        *self._reconcile_tasks.values(),
                    )
                    if not task.done()
                }
                if not pending:
                    return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            _done, pending = await asyncio.wait(pending, timeout=remaining)
            if pending:
                raise asyncio.TimeoutError

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Stop acceptance, persist interruption, and settle owned tasks."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        self._accepting = False
        executor_tasks = tuple(self._tasks.items())
        terminalizers: list[asyncio.Task[bool]] = []
        for receipt_id, task in executor_tasks:
            kind = self._kinds.get(receipt_id)
            if kind is not None:
                terminalizer = asyncio.create_task(
                    self._ensure_terminal(
                        receipt_id,
                        kind,
                        cancel_check=True,
                    )
                )
                terminalizers.append(terminalizer)
                self._retain_background(terminalizer)

        pending_terminalizers = {
            task for task in terminalizers if not task.done()
        }
        if pending_terminalizers:
            remaining = max(0.0, deadline - loop.time())
            done, pending_terminalizers = await asyncio.wait(
                pending_terminalizers,
                timeout=remaining,
            )
            for task in done:
                self._consume_exception(task)

        for _receipt_id, task in executor_tasks:
            if not task.done():
                task.cancel()

        reconciliations = tuple(self._reconcile_tasks.values())
        for task in reconciliations:
            if not task.done():
                task.cancel()

        pending = {
            task
            for task in (
                *(task for _receipt_id, task in executor_tasks),
                *reconciliations,
                *terminalizers,
            )
            if not task.done()
        }
        if not pending:
            return
        remaining = max(0.0, deadline - loop.time())
        done, pending = await asyncio.wait(
            pending,
            timeout=remaining,
        )
        for task in done:
            self._consume_exception(task)
        if pending:
            logger.warning("Watchlists operation shutdown timed out.")
