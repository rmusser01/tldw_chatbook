"""Routine Console transcript-poll ticks: live progress between full passes.

Lever L2 (owner-approved; ``Docs/Development/console-optimization-review-list.md``
OPT-60, proposed ADR-226). ``ChatScreen._start_console_transcript_sync_timer``
ticks every 0.2 s while any turn, run, custody, wake delivery or queued review
publication is live. Each tick used to run the whole
``ChatScreen._sync_native_console_chat_ui`` reconciliation: core live state and
roleplay under fresh native config admission, scope/dictionary/world-book/
avatar/character-context refreshes, workspace context, rails and controls,
settings-recovery and readiness surfaces, rail visibility, read-ack scheduling.
Measured on Windows during a warm Send, that cost ~1 thread-second of
main-thread native admission per 2 s, 0.2-0.4 s UI stalls, held the global
config locks the Send's own reads then waited on (0.18 s on one hook read), and
fanned out background refreshes that competed with the Send's critical path.

``poll_wants_full_sync`` keeps the ORIGINAL full pass for a tick when:

* **the screen is torn down** -- the full pass's own entry guard owns that tick
  (it renders nothing and drops coalesced demand), before any runtime read;
* **the poll is about to stop** -- the original poll-needed predicate is false.
  The poll's own tail (persisted-rows invalidation, timer stop, fleet survivor
  handoff) still only follows a pass that returned True, so a deferred final
  refresh keeps the poll alive (integration ``116628850f``);
* **a run settled** -- the viewed run left an active status, the in-flight run
  count dropped, or turn custody ended. A viewed turn settling while another
  run keeps the poll alive gets its full terminal transition on that tick, not
  at the stop edge (ADR-226 item 6). Only a full pass ADMITTED after the edge
  was observed discharges it, since an earlier one may have read pre-settle
  state;
* **full demand is deferred or stranded** -- the maintenance pause, the
  control-bar whole-sync replay, or a ``_console_sync_requested`` with no pass
  running to replay it; those keep their original owners and return paths;
* **the owner is unproven** -- no full pass has completed yet, or the store,
  active session, or session membership (identity and order) differs from what
  the last completed full pass reconciled, or the active session has no
  settings (the tab helper would create them: a live effect, not display)
  (ADR-226 item 4);
* **the cadence is due** -- ``CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS`` have
  passed since the later of this poll's first tick and the start of the last
  completed full pass from ANY caller, and no full pass is already running
  (that pass resets the clock itself). Counting from the poll's first tick
  keeps the routine cadence out of a Send's first two seconds (the window
  this lever exists for) while still reconciling at least that often.

Every other tick runs ``sync_console_poll_display``: session tabs (run/stream/
queue markers), transcript publication and the run chip -- the surfaces that
show live progress. Rails, inspector, the Agent fleet line, controls and the
other full-pass surfaces catch up at the next full pass. Direct callers of
``_sync_native_console_chat_ui`` (send/stop transitions, session switches,
attach, settings and readiness publication, ...) are untouched and always run
the full pass.

The light pass shares the full pass's exclusion and replay machinery instead
of adding an owner: it holds ``_console_sync_in_progress`` so a full request
arriving across its awaits coalesces into ``_console_sync_requested`` and is
handed to one replay worker when the light pass ends, exactly as the full
pass's own ``finally`` does (maintenance and the control-bar replay keep it for
their owners). It never completes attachment.

Nothing here is authority. The record says only when a full pass started and
which store/session/membership it reconciled, so the poll can decide WHEN to
run the original checks -- never whether an action may skip one. No permission
cache, no TTL on authority, no lock or lease held across an await.

This module is imported lazily by the poll callback, so it stays off the boot
import leg (ADR-097); the full pass records its completion as a plain tuple.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

#: Longest a live poll goes without the original full reconciliation. Read at
#: call time so a test can pin the cadence explicitly.
CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS = 2.0

#: Diagnostic worker label for the light pass, distinct from ``console-sync``
#: so responsiveness records attribute stalls to the right pass.
CONSOLE_POLL_DISPLAY_WORKER = "console-sync-poll-display"

#: The clock ``ChatScreen._sync_native_console_chat_ui`` stamps its
#: ``started_at`` with: high resolution, so a full pass admitted right after a
#: settle edge orders after it even on Windows' 15.6 ms ``monotonic`` tick.
_clock = time.perf_counter


@dataclass(slots=True)
class ConsolePollCadence:
    """Screen-owned routing memory for the transcript poll.

    Attributes:
        timer: The poll timer whose first tick ``first_tick_at`` records.
        first_tick_at: ``time.perf_counter()`` at that timer's first tick.
        live: The previous tick's ``(viewed_active, in_flight, custody)``.
        settled_at: When a settle edge was observed and not yet discharged
            by a full pass admitted after it.
    """

    timer: Any = None
    first_tick_at: float = 0.0
    live: tuple[bool, int, bool] | None = None
    settled_at: float | None = None


def poll_cadence_for(screen: Any) -> ConsolePollCadence:
    """Return the screen's poll routing memory, creating it on first use.

    Read through ``vars`` so a spec'd mock never answers with a child mock.

    Args:
        screen: The Console ``ChatScreen`` (or a fixture standing in for it).

    Returns:
        The screen's ``ConsolePollCadence``.
    """
    cadence = vars(screen).get("_console_poll_cadence")
    if type(cadence) is not ConsolePollCadence:
        cadence = ConsolePollCadence()
        screen._console_poll_cadence = cadence
    return cadence


def _completed_full_sync(screen: Any) -> tuple[float, Any, tuple] | None:
    """Return the last completed full pass's ``(started, store, owner)``."""
    record = vars(screen).get("_console_full_sync_completed")
    if (
        type(record) is not tuple
        or len(record) != 3
        or type(record[2]) is not tuple
        or len(record[2]) != 2
    ):
        return None
    return record


def _live_shape(screen: Any) -> tuple[bool, int, bool] | None:
    """Read the in-memory facts whose decrease marks a settled run."""
    from ...Chat.console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES

    controller = getattr(screen, "_console_chat_controller", None)
    if controller is None:
        return None
    # The same reads the original poll-needed predicate makes every tick:
    # controller/runtime memory only, no store, config or native IO.
    return (
        controller.run_state.status in FEEDBACK_ACTIVE_RUN_STATUSES,
        int(controller.in_flight_run_count()),
        bool(screen._console_runtime().has_custodied_turns()),
    )


def poll_wants_full_sync(screen: Any) -> bool:
    """Route one poll tick to the original full pass or the light pass.

    Args:
        screen: The Console screen whose transcript poll is ticking.

    Returns:
        True when this tick must run ``_sync_native_console_chat_ui``.
    """
    from ..Screens.chat_screen import _console_screen_is_torn_down

    if _console_screen_is_torn_down(screen):
        # Before any runtime read: the full pass's own entry guard owns a
        # torn-down tick (renders nothing, drops coalesced demand), exactly
        # as every tick did before this routing existed.
        return True
    cadence = poll_cadence_for(screen)
    now = _clock()
    timer = getattr(screen, "_console_transcript_sync_timer", None)
    if cadence.timer is not timer:
        cadence.timer, cadence.first_tick_at = timer, now
    if not screen._console_transcript_poll_needed():
        # Forget this poll's live shape: the next poll's first Preparing
        # tick must not read as a settle of this poll's last run.
        cadence.live = None
        return True
    shape = _live_shape(screen)
    previous, cadence.live = cadence.live, shape
    if (
        previous is not None
        and shape is not None
        and (
            (previous[0] and not shape[0])
            or shape[1] < previous[1]
            or (previous[2] and not shape[2])
        )
    ):
        cadence.settled_at = now
    running = screen._console_sync_in_progress
    if (
        getattr(screen, "_console_sync_maintenance_paused", False)
        or getattr(screen, "_console_control_bar_replay_whole_sync", False)
        or (screen._console_sync_requested and not running)
    ):
        # Deferred or stranded full demand keeps the original route, whose
        # deferral re-requests it for the original resume/replay owners. A
        # request behind a RUNNING pass already has its trailing replay.
        return True
    record = _completed_full_sync(screen)
    if record is None:
        return True
    started, store, (session_id, sessions) = record
    if cadence.settled_at is not None:
        if started <= cadence.settled_at:
            return True
        cadence.settled_at = None
    current = getattr(screen, "_console_chat_store", None)
    if (
        current is None
        or current is not store
        or current.active_session_id != session_id
    ):
        return True
    current_sessions = current.sessions()
    if len(current_sessions) != len(sessions) or any(
        now_session is not then_session
        for now_session, then_session in zip(current_sessions, sessions)
    ):
        return True
    active = next(
        (session for session in current_sessions if session.id == session_id), None
    )
    if active is None or active.settings is None:
        return True
    # A tick that lands while a (direct) full pass runs leaves the cadence to
    # that pass: it records its own completion, and coalescing behind it would
    # only queue a redundant replay. Edges and unproven owners above still
    # request one, because a pass admitted before them cannot discharge them.
    elapsed = now - max(started, cadence.first_tick_at)
    return not running and elapsed >= CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS


def sync_console_run_chip(screen: Any) -> None:
    """Publish the viewed session's run chip from in-memory controller state.

    The same two calls ``ChatScreen._sync_console_mode_bar`` ends with; the
    light pass skips that method's mode-bar half, a hidden compat static whose
    copy needs the control-state build.

    Args:
        screen: The Console screen owning ``#console-status-chips``.
    """
    from textual.css.query import QueryError

    from ...Widgets.Console.console_status_chips import ConsoleStatusChips

    try:
        status_chips = screen.query_one("#console-status-chips", ConsoleStatusChips)
    except QueryError:
        return
    active_run_copy = screen._console_active_run_copy()
    status_chips.sync_run_chip(bool(active_run_copy), active_run_copy)


def _rearm_requested_console_sync(screen: Any) -> None:
    """Replay full demand that coalesced while the light pass ran.

    Mirrors the ``finally`` of ``ChatScreen._sync_native_console_chat_ui``:
    maintenance and the control-bar whole-sync replay keep the request for
    their own resume/replay owners, and a dead screen must not re-arm itself
    (``run_worker`` here runs after Textual's unmount sweep, so the worker it
    created would outlive the screen that owns it).
    """
    from ..Screens.chat_screen import _console_screen_is_torn_down

    if (
        screen._console_sync_requested
        and not getattr(screen, "_console_sync_maintenance_paused", False)
        and not getattr(screen, "_console_control_bar_replay_whole_sync", False)
    ):
        screen._console_sync_requested = False
        if not _console_screen_is_torn_down(screen):
            screen.run_worker(
                screen._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )


async def sync_console_poll_display(screen: Any) -> bool:
    """Publish live run progress for one routine poll tick.

    Runs, in the full pass's own order, only the original owners of visible
    live progress: session tabs (run/stream/queue markers and their existing
    seen-mark path), transcript publication (new rows, streaming text, the
    turn-activity line, terminal-receipt acknowledgement under its original
    runtime owner) and the run chip. It never enters ``sync_live_state``/
    ``run_console_config_sync`` or any other config-locked or storage-
    admission step of the full pass. The one config-adjacent read left is the
    transcript's own: its memory banner reads the shared
    ``ConsoleContextReadSnapshot`` presentation memo, whose expiry schedules
    that memo's existing finite, deduplicated worker refresh off the main
    thread. That is unchanged from the full pass and is part of publishing
    the transcript, so it stays.

    Args:
        screen: The Console screen whose transcript poll is ticking.

    Returns:
        True when this pass published; False when it was skipped because a
        sync pass is already running (that pass publishes the same surfaces),
        the screen is torn down, or full demand is deferred.
    """
    from ..Screens.chat_screen import _console_screen_is_torn_down

    if (
        _console_screen_is_torn_down(screen)
        or screen._console_sync_in_progress
        or getattr(screen, "_console_sync_maintenance_paused", False)
        or getattr(screen, "_console_control_bar_replay_whole_sync", False)
    ):
        return False
    screen._console_sync_in_progress = True
    screen._record_ui_worker_started(CONSOLE_POLL_DISPLAY_WORKER)
    try:
        await screen._sync_console_native_session_tabs()
        await screen._sync_native_console_transcript()
        sync_console_run_chip(screen)
        return True
    except Exception:
        # Teardown-scoped, like the full pass: a tick mid-flight when the
        # screen closes is querying removed widgets, and this runs on a timer
        # whose error would take the app down. A live screen's error raises.
        if not _console_screen_is_torn_down(screen):
            raise
        return False
    finally:
        screen._record_ui_worker_finished(CONSOLE_POLL_DISPLAY_WORKER)
        screen._console_sync_in_progress = False
        _rearm_requested_console_sync(screen)
