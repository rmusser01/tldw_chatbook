"""Two-tier hard-timeout watchdog for the pinned workspace worker.

Part of the worker's stdlib-only import closure (Phase 0c; Task 8
flattens this module into the remote worker bundle). Shared by the
LOCAL worker and the shipped bundle: ``run_workspace_worker`` arms both
tiers with the request's ``timeout_seconds`` immediately after decoding
and disarms them once the exchange's frames are out (Task 12).

Why two tiers — the ``fs_grep`` catastrophic-regex case
------------------------------------------------------

``fs_grep`` runs caller-supplied patterns through ``re`` in the worker
process. A pattern like ``(a+)+$`` against a near-miss line (60KB of
``a`` plus one trailing ``b``) drives the C regex engine into quadratic-
or-worse backtracking, and that engine does not release the GIL while
it spins:

* **Tier 1 — graceful ``threading.Timer``** fires at the budget,
  sweeps every path in :data:`TEMP_REGISTRY` (best-effort ``unlink``),
  writes the fixed stderr line, and ``os._exit(75)``. It produces a
  clean, attributable death — but it is PYTHON code: it needs the GIL,
  and a starving regex can hold that GIL forever, so tier 1 alone can
  be starved.
* **Tier 2 — OS backstop ``signal.alarm(budget + 2)`` with the DEFAULT
  action**. No Python handler is installed (a handler would need the
  GIL and starve exactly like the Timer); the kernel terminates the
  process on delivery, GIL or no GIL. The two-second grace exists so
  the graceful tier wins whenever it can.

An optional RLIMIT_CPU ceiling (soft == hard == ``ceil(budget * 2)``)
backs both for CPU-spin cases on platforms that have ``resource``:
exceeding it raises SIGXCPU with its default action. It is guarded —
``resource`` and its limits are not universal — and NOT restored on
disarm: ``setrlimit`` is per-process and cannot be un-set, which is
fine because the worker is one-shot (one exchange per process).

The transport side (Task 11) buckets "admitted marker + exit 75" (tier
1) and "admitted marker + completion-deadline kill" (tier 2, seen by
the caller as death by signal) as the same failure kind, OP_TIMEOUT.

Exit-code reservation: ``os._exit(WATCHDOG_EXIT_CODE)`` below is the
ONLY place in the worker's closure that may hard-exit with 75; worker
failure paths exit 2. Task 11's transport keys on it.
"""

from __future__ import annotations

import math
import os
import signal
import threading

#: Reserved process exit code for a watchdog death (tier 1). No other
#: worker path may exit with 75 — the transport maps exactly this code
#: (and its own deadline kill) to the OP_TIMEOUT failure kind.
WATCHDOG_EXIT_CODE = 75

#: The fixed single stderr line tier 1 writes before exiting — bounded,
#: request-independent, and the line Task 11's transport greps for a
#: failure reason. Written with one unbuffered ``os.write`` on fd 2 so
#: it survives ``os._exit`` (no interpreter flush happens after it).
WATCHDOG_STDERR_MARKER = b"tldw-worker-watchdog\n"

#: Live registry of absolute paths of temp files created by in-flight
#: operations (the atomic-write temp of ``fs_write``/``fs_edit``/CAS
#: writes). Creation registers, success/failure-cleanup unregisters;
#: tier 1's sweep unlinks whatever is still registered.
TEMP_REGISTRY: list[str] = []

#: The armed tier-1 timer, or ``None`` when disarmed/never armed.
_armed_timer: threading.Timer | None = None


def register_temp(path: str) -> None:
    """Record one temp-file path for the watchdog's cleanup sweep."""
    if path not in TEMP_REGISTRY:
        TEMP_REGISTRY.append(path)


def unregister_temp(path: str) -> None:
    """Drop one temp-file path (its operation completed or cleaned up)."""
    if path in TEMP_REGISTRY:
        TEMP_REGISTRY.remove(path)


def _watchdog_fire(temp_registry: list[str]) -> None:
    """Tier-1 expiry: sweep temps, mark stderr, hard-exit 75.

    Everything here is best-effort by construction — the process is
    already past its budget, and ``os._exit`` skips ``finally`` blocks
    and interpreter cleanup, so THIS callback is the one place the
    sweep can run. It must never raise before the ``os._exit``.
    """
    for path in list(temp_registry):
        try:
            os.unlink(path)
        except OSError:
            pass  # raced, already consumed, or never linked — best effort
    try:
        os.write(2, WATCHDOG_STDERR_MARKER)
    except OSError:
        pass  # closed/redirected stderr must not prevent the exit
    # Reserved exit code; see WATCHDOG_EXIT_CODE. The ONLY os._exit(75)
    # call site in the worker closure.
    os._exit(WATCHDOG_EXIT_CODE)


def arm_watchdog(budget_seconds: float, temp_registry: list[str]) -> None:
    """Arm both hard-timeout tiers for one exchange.

    Args:
        budget_seconds: The request's ``timeout_seconds`` — over ssh the
            transport writes the REMAINING budget into that field; this
            function just consumes it. Values ``<= 0`` arm NOTHING: the
            wire decoder and the transport both require a positive
            budget, so reaching here with a non-positive one is already
            a caller bug, and skipping is the defensive choice (a 0
            budget would otherwise fire tier 1 instantly).
        temp_registry: The live registry tier 1 sweeps (by reference —
            registrations after arming are still seen; the worker passes
            ``TEMP_REGISTRY`` itself).
    """
    global _armed_timer
    disarm_watchdog()  # one-shot worker, but never leave a stale tier alive
    try:
        budget = float(budget_seconds)
    except (TypeError, ValueError):
        return
    if not math.isfinite(budget) or budget <= 0:
        return
    timer = threading.Timer(budget, _watchdog_fire, args=(temp_registry,))
    timer.daemon = True  # a cancelled-but-joinable timer must not delay exit
    timer.start()
    _armed_timer = timer
    # Tier 2: default-action alarm (NO Python handler — installing one
    # would reintroduce the GIL dependency tier 2 exists to escape).
    if hasattr(signal, "alarm"):
        signal.alarm(max(1, int(budget) + 2))
    # Optional RLIMIT_CPU backstop; not every platform has resource,
    # and a refused rlimit must never fail the arming.
    try:
        import resource

        cpu_seconds = max(1, math.ceil(budget * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ImportError, OSError, ValueError):
        pass


def disarm_watchdog() -> None:
    """Cancel tier 1 and zero tier 2 once the exchange completed.

    Called after the final response frame is emitted: a completed op
    must not die late. The RLIMIT_CPU ceiling is per-process and cannot
    be restored — it simply persists for the worker's remaining life,
    which is fine (the worker is one-shot: one exchange, then exit).
    """
    global _armed_timer
    if _armed_timer is not None:
        _armed_timer.cancel()
        _armed_timer = None
    if hasattr(signal, "alarm"):
        signal.alarm(0)


__all__ = [
    "TEMP_REGISTRY",
    "WATCHDOG_EXIT_CODE",
    "WATCHDOG_STDERR_MARKER",
    "arm_watchdog",
    "disarm_watchdog",
    "register_temp",
    "unregister_temp",
]
