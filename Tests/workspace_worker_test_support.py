"""Real workspace dispatch without one-shot process containment in pytest."""

from collections.abc import Callable
from types import FunctionType
from typing import BinaryIO

from tldw_chatbook.Tools import workspace_tool_worker


def run_in_process_workspace_worker(
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: BinaryIO,
    *,
    bundle_sha256: str = "",
    worker: Callable[..., int] = workspace_tool_worker.run_workspace_worker,
) -> int:
    """Run unchanged worker code with process-only watchdogs disabled locally."""
    # Keep real watchdogs intact in other threads and forked children.
    worker_globals = dict(worker.__globals__)
    worker_globals.update(
        arm_watchdog=lambda _budget, _temps: None,
        disarm_watchdog=lambda: None,
    )
    in_process_worker = FunctionType(
        worker.__code__,
        worker_globals,
        worker.__name__,
        worker.__defaults__,
        worker.__closure__,
    )
    in_process_worker.__kwdefaults__ = worker.__kwdefaults__
    return in_process_worker(stdin, stdout, stderr, bundle_sha256=bundle_sha256)
