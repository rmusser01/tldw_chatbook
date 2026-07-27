"""Grid execution.

Row-major: complete, comparable rows appear while the run is still going,
which is the point of the grid doubling as the progress view. Fail-fast on a
dead target is preflight's job, not the fill order's.

Sequential within and across targets by default -- local servers are
frequently single-slot, and concurrent requests either queue or 503.
"""

from __future__ import annotations

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
        done = 0

        for snippet in snippets:  # row-major
            for target in targets:
                if cancel_token is not None and cancel_token.is_cancelled:
                    logger.info(
                        "Word bench run group {} cancelled after {}/{} cells",
                        group_id, done, total,
                    )
                    for run_id in run_ids.values():
                        self._db.update_run_status(run_id, "cancelled")
                    return RunOutcome(group_id=group_id, preflight=results)

                result = await clients[target.id].capture(
                    snippet.text, target, config.prompt_mode, config.top_k
                )
                result = self._stamp_canary(result, canaries[target.id])
                save_cell(self._db, run_ids[target.id], snippet, result)

                done += 1
                if progress is not None:
                    try:
                        progress(done, total)
                    except Exception:
                        # A broken UI-supplied callback must degrade to no
                        # progress reporting, not kill the run -- otherwise
                        # it escapes to run()'s caller and is indistinguishable
                        # from a real cancellation, leaving this run group's
                        # rows stranded at "running" forever (the same failure
                        # class the asyncio.CancelledError handler around
                        # runner.run() exists to close, just for a different
                        # exception type). Stop calling it after the first
                        # failure rather than logging once per remaining cell.
                        logger.opt(exception=True).warning(
                            "Word bench progress callback raised for run group "
                            "{}; continuing without progress reporting.",
                            group_id,
                        )
                        progress = None

        for run_id in run_ids.values():
            self._db.update_run_status(run_id, "completed")

        return RunOutcome(group_id=group_id, preflight=results)

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
