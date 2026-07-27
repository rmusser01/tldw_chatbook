"""Grid execution.

Row-major: complete, comparable rows appear while the run is still going,
which is the point of the grid doubling as the progress view. Fail-fast on a
dead target is preflight's job, not the fill order's.

Sequential within and across targets by default -- local servers are
frequently single-slot, and concurrent requests either queue or 503.
Parallelism is opt-in through ``BenchConfig.concurrency``: a value of 1 (the
default) runs the original sequential loop, byte-for-byte. A value > 1 fans
out ONE ROW (all of that row's targets) at a time, bounded by an
``asyncio.Semaphore`` so a single target never receives two in-flight
requests at once and the grid never shows a half-filled row -- results are
always saved back in the row's target order regardless of which request
actually completed first (``asyncio.gather`` preserves input order), so the
"row-major, comparable rows" contract holds at any concurrency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from loguru import logger

from ...DB.Evals_DB import EvalsDB
from .models import (
    BenchConfig, CellCapture, CellError, PreflightResult, PromptMode, Snippet, Target,
)
from .storage import create_run_group, save_cell

ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class RunOutcome:
    """What a run produced, beyond the cells themselves.

    ``preflight`` is returned rather than discarded because the screen renders
    readiness badges, recovery callouts, and the effective-K header from it.
    Re-running preflight to recover this would double the canary calls and
    could report a verdict the run itself never saw.
    """

    group_id: str
    preflight: dict[str, PreflightResult]


class CaptureClientLike(Protocol):
    async def preflight(
        self, target: Target, mode: PromptMode, top_k: int
    ) -> PreflightResult: ...

    async def capture(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> CellCapture | CellError: ...


class CancelToken:
    """Cancels a whole run group, not a single run."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class WordBenchRunner:
    """Executes a bench over a snippet set and a target set."""

    def __init__(
        self, db: EvalsDB, client_factory: Callable[[Target], CaptureClientLike]
    ) -> None:
        self._db = db
        self._client_factory = client_factory

    async def run(
        self,
        config: BenchConfig,
        targets: Sequence[Target],
        snippets: Sequence[Snippet],
        task_id: str,
        progress: Optional[ProgressFn] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> RunOutcome:
        """Execute the grid and return its run group id and preflight verdicts."""
        for target in targets:
            if not target.is_valid_for_mode(config.prompt_mode):
                raise ValueError(
                    f"Target {target.name!r} is not valid for {config.prompt_mode!r} mode: "
                    "raw mode takes a prefix, chat mode takes a system_prompt."
                )

        clients = {t.id: self._client_factory(t) for t in targets}
        try:
            return await self._run_with_clients(
                config, targets, snippets, task_id, clients, progress, cancel_token
            )
        finally:
            # Every client this run created is done being used -- by
            # success, by a returned cancellation, or by a raised exception
            # (including a hard asyncio.CancelledError, re-raised from
            # _run_with_clients below) -- before this line ever runs, so
            # closing here can never race a concurrent in-flight capture()
            # call. Duck-typed: fakes used throughout this test suite carry
            # no aclose() and are left alone; only WordBenchCaptureClient's
            # pooled httpx.AsyncClient actually holds a resource to release.
            await self._close_clients(clients)

    async def _run_with_clients(
        self,
        config: BenchConfig,
        targets: Sequence[Target],
        snippets: Sequence[Snippet],
        task_id: str,
        clients: dict[str, CaptureClientLike],
        progress: Optional[ProgressFn],
        cancel_token: Optional[CancelToken],
    ) -> RunOutcome:
        # Preflight before any measurement, so a dead or degenerate target is
        # known up front rather than discovered N cells in.
        results: dict[str, PreflightResult] = {}
        canaries: dict[str, str] = {}
        for target in targets:
            result = await clients[target.id].preflight(
                target, config.prompt_mode, config.top_k
            )
            results[target.id] = result
            canaries[target.id] = result.canary
            if result.is_warned:
                logger.warning(
                    "Word bench target {} preflighted degenerate; its column "
                    "carries a warning.",
                    target.name,
                )

        group_id, run_ids = create_run_group(
            self._db, task_id, config, targets, snippets, preflight=results
        )
        for run_id in run_ids.values():
            self._db.update_run_status(run_id, "running")

        total = len(snippets) * len(targets)
        # A plain mutable holder, not a local `done`/`progress` pair, because
        # the parallel path's per-row helper needs to read and mutate both
        # across `await` points without `nonlocal` scattered through two
        # nested closures.
        state = {"done": 0, "progress": progress}

        def _report(cell_target: Target, result: CellCapture | CellError) -> None:
            stamped = self._stamp_canary(result, canaries[cell_target.id])
            save_cell(self._db, run_ids[cell_target.id], snippet, stamped)
            state["done"] += 1
            fn = state["progress"]
            if fn is None:
                return
            try:
                fn(state["done"], total)
            except Exception:
                # A broken UI-supplied callback must degrade to no progress
                # reporting, not kill the run -- otherwise it escapes to
                # run()'s caller and is indistinguishable from a real
                # cancellation, leaving this run group's rows stranded at
                # "running" forever (the same failure class the
                # asyncio.CancelledError handling below exists to close,
                # just for a different exception type). Stop calling it
                # after the first failure rather than logging once per
                # remaining cell.
                logger.opt(exception=True).warning(
                    "Word bench progress callback raised for run group "
                    "{}; continuing without progress reporting.",
                    group_id,
                )
                state["progress"] = None

        def _mark_cancelled(reason: str) -> RunOutcome:
            logger.info(
                "Word bench run group {} cancelled ({}) after {}/{} cells",
                group_id, reason, state["done"], total,
            )
            for run_id in run_ids.values():
                self._db.update_run_status(run_id, "cancelled")
            return RunOutcome(group_id=group_id, preflight=results)

        semaphore = asyncio.Semaphore(config.concurrency) if config.concurrency > 1 else None

        try:
            for snippet in snippets:  # row-major
                if cancel_token is not None and cancel_token.is_cancelled:
                    return _mark_cancelled("cooperative")

                if semaphore is None:
                    # concurrency == 1: the original sequential loop,
                    # unchanged -- per-cell cancel check, deterministic call
                    # order, no asyncio.gather involved.
                    for target in targets:
                        if cancel_token is not None and cancel_token.is_cancelled:
                            return _mark_cancelled("cooperative")
                        result = await clients[target.id].capture(
                            snippet.text, target, config.prompt_mode, config.top_k
                        )
                        _report(target, result)
                else:
                    # concurrency > 1: fan out this row's targets, bounded by
                    # `semaphore` so no target ever has two in-flight
                    # requests and at most `config.concurrency` requests are
                    # in flight across the whole row. Cancellation is
                    # checked once per ROW rather than once per cell here --
                    # a row already dispatched is allowed to finish so the
                    # grid never persists a half-captured row (see the
                    # module docstring).
                    async def _capture_one(
                        target: Target,
                    ) -> tuple[Target, CellCapture | CellError]:
                        async with semaphore:
                            captured = await clients[target.id].capture(
                                snippet.text, target, config.prompt_mode, config.top_k
                            )
                        return target, captured

                    # asyncio.gather preserves input order in its returned
                    # list regardless of completion order, so saving/
                    # reporting below always happens in `targets` order --
                    # the row-major guarantee holds even though the
                    # underlying requests may complete out of order.
                    row_results = await asyncio.gather(
                        *(_capture_one(target) for target in targets)
                    )
                    for target, result in row_results:
                        _report(target, result)
        except asyncio.CancelledError:
            # A HARD cancellation (e.g. the Task running this coroutine is
            # cancelled directly) is a BaseException, not caught by the
            # `except Exception` above -- it must still leave no run row
            # stranded at "running". Mark them cancelled and let the
            # cancellation propagate; it must never be swallowed here.
            logger.info(
                "Word bench run group {} hard-cancelled after {}/{} cells",
                group_id, state["done"], total,
            )
            for run_id in run_ids.values():
                self._db.update_run_status(run_id, "cancelled")
            raise

        for run_id in run_ids.values():
            self._db.update_run_status(run_id, "completed")

        return RunOutcome(group_id=group_id, preflight=results)

    @staticmethod
    async def _close_clients(clients: dict[str, CaptureClientLike]) -> None:
        """Best-effort cleanup for every client this run created.

        ``CaptureClientLike`` does not require an ``aclose()`` -- most test
        fakes have none -- so this is duck-typed rather than part of the
        Protocol. A client that fails to close cleanly must not prevent the
        run's own outcome (already computed by the caller) from being
        returned, or turn a successful/cancelled run into a raised
        exception.
        """
        for client in clients.values():
            aclose = getattr(client, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception:
                logger.opt(exception=True).warning(
                    "Word bench capture client failed to close cleanly; continuing."
                )

    @staticmethod
    def _stamp_canary(
        result: CellCapture | CellError, canary: str
    ) -> CellCapture | CellError:
        """Carry the target's preflight verdict onto the cell.

        Without this the warning is lost between preflight and the grid, and a
        divergence produced by out-of-distribution behaviour reads as a
        finding about the model's content.
        """
        if isinstance(result, CellError):
            return result
        return CellCapture(
            prompt_mode=result.prompt_mode,
            k_requested=result.k_requested,
            k_returned=result.k_returned,
            content_offset=result.content_offset,
            top_k=result.top_k,
            canary=canary,
            captured_at=result.captured_at,
            schema=result.schema,
        )
