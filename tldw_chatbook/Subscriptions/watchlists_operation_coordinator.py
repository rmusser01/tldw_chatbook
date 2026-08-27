"""Application-owned supervision for durable manual Watchlists operations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from .briefing_service import accept_briefing, execute_accepted_briefing

if TYPE_CHECKING:
    from ..DB.Subscriptions_DB import SubscriptionsDB
    from .local_watchlists_service import LocalWatchlistsService


_CHECK_FAILURE = "Watchlists source check failed. Try again."
_BRIEFING_FAILURE = "Watchlists briefing generation failed. Try again."


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
        if receipt_id in self._tasks:
            coroutine.close()
            return
        task = asyncio.create_task(coroutine, name=f"watchlists:{receipt_id}")
        self._tasks[receipt_id] = task
        self._kinds[receipt_id] = kind
        task.add_done_callback(
            lambda finished, key=receipt_id: self._consume_terminal(key, finished)
        )

    def _consume_terminal(self, receipt_id: str, task: asyncio.Task[None]) -> None:
        """Consume every terminal and release its strong reference once durable."""
        try:
            if not task.cancelled():
                task.exception()
        except BaseException:  # noqa: BLE001 - loop callback must never escape
            logger.error("Watchlists operation task ended unexpectedly.")
        self._tasks.pop(receipt_id, None)
        self._kinds.pop(receipt_id, None)

    async def accept_checks(
        self, source_ids: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Accept validated checks and schedule only newly won receipts."""
        self._assert_accepting()
        receipts = await self._local_service.accept_source_checks(source_ids)
        for receipt in receipts:
            run_id = int(receipt["run_id"])
            canonical_id = f"local:watchlist_run:{run_id}"
            if receipt.pop("_claim_acquired"):
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
        if receipt.pop("_claim_acquired"):
            self._start(
                canonical_id,
                "briefing",
                self._run_briefing(briefing_id),
            )
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

    def _fail_briefing_if_active(self, briefing_id: int) -> None:
        try:
            self._briefing_db.transition_briefing(
                briefing_id,
                status="failed",
                error=_BRIEFING_FAILURE,
            )
        except Exception:  # noqa: BLE001 - shutdown/recovery remains best effort
            logger.error("Could not terminalize a failed Watchlists briefing.")

    async def wait_idle(self, *, timeout: float) -> None:
        """Wait boundedly until every currently owned operation settles."""
        tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        await asyncio.sleep(0)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Stop acceptance, persist interruption, and settle owned tasks."""
        self._accepting = False
        tasks = tuple(self._tasks.items())
        for receipt_id, task in tasks:
            kind = self._kinds.get(receipt_id)
            numeric_id = int(receipt_id.rsplit(":", 1)[-1])
            try:
                if kind == "check":
                    await self._local_service.cancel_run(numeric_id)
                elif kind == "briefing":
                    await asyncio.to_thread(
                        self._fail_briefing_if_active, numeric_id
                    )
            except Exception:  # noqa: BLE001 - task may have terminalized first
                pass
            task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(task for _receipt_id, task in tasks),
                        return_exceptions=True,
                    ),
                    timeout=max(0.0, float(timeout)),
                )
            except asyncio.TimeoutError:
                logger.warning("Watchlists operation shutdown timed out.")
        await asyncio.sleep(0)
