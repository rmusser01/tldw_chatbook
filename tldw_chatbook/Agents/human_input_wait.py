"""Run-scoped marks for waits on a HUMAN decision (ADR-067).

Three blocking prompts arm a card and then poll for the user's answer:
the tool approval round (``ConsoleChatController.request_mcp_approvals``),
the skill install confirm, and the skill script confirm. The approval
round -- and any other wait dispatched inside a tool call -- blocks on a
thread hosted by ``AgentService._call_with_timeout``, whose wall-clock
deadline exists so a single wedged tool cannot hang a cooperative-cancel
run forever.

Waiting for a dinner-and-back human is not a wedged tool. While a mark
is held for a run, that run's per-call deadline PAUSES (re-arms each poll
slice, see ``_call_with_timeout``'s ``pauses_deadline``): the budget
counts machine time -- actual tool execution -- not human deliberation.

Why process state and not another ContextVar (contrast
``run_context``): the mark is SET on the round's worker/daemon thread and
POLLED on the wrapper's calling thread. A ContextVar set on the worker is
invisible where the deadline is checked -- the exact trap ``run_context``
documents for its own bindings. The run id itself still travels via
``run_context`` (the round captures ``current_run_id()`` at arm time);
only the mark crosses threads here.

Refcounted per run: a run may legitimately hold TWO overlapping waits
(a batch review-hook round plus a single-call fallback approval), so the
mark must survive until the last one clears. Stdlib only, no project
imports: this is consumed by ``agent_service`` and ``Chat``, and stays
dependency-light like ``run_context``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

#: Guards ``_WAIT_REFCOUNTS``: the set/clear run on the round's thread,
#: the active-check runs on the wrapper's poll thread.
_WAIT_LOCK = threading.Lock()
#: run_id -> number of overlapping waits currently held for that run.
_WAIT_REFCOUNTS: dict[str, int] = {}


@contextmanager
def use_human_input_wait(run_id: str | None) -> Iterator[None]:
    """Mark ``run_id`` as waiting on a human decision for the block.

    Args:
        run_id: The run whose per-call clock should pause. ``None`` or an
            empty string normalizes to ``""`` (the no-run key a round
            armed outside any agent run carries -- same convention as
            ``run_context``).

    Yields:
        None. The mark is refcounted and drops on exit, including on an
        exception (a round torn down mid-wait must not leak a frozen
        clock for the run's next tool call).
    """
    key = run_id or ""
    with _WAIT_LOCK:
        _WAIT_REFCOUNTS[key] = _WAIT_REFCOUNTS.get(key, 0) + 1
    try:
        yield
    finally:
        with _WAIT_LOCK:
            remaining = _WAIT_REFCOUNTS.get(key, 0) - 1
            if remaining > 0:
                _WAIT_REFCOUNTS[key] = remaining
            else:
                _WAIT_REFCOUNTS.pop(key, None)


def human_input_wait_active(run_id: str | None) -> bool:
    """Whether a human decision is pending for ``run_id`` right now.

    Args:
        run_id: The run to check. ``None`` or an empty string checks the
            ``""`` slot, mirroring ``use_human_input_wait``'s
            normalization.

    Returns:
        True while at least one wait holds the mark. Never raises.
    """
    with _WAIT_LOCK:
        return (_WAIT_REFCOUNTS.get(run_id or "", 0) > 0)
