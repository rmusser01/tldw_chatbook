"""The Console's synchronous checked configuration refresh lifetime.

Also decides how often the 0.2 s poll pays for it (TASK-33620.15.1).
"""

import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from contextvars import ContextVar
from typing import Any

#: TASK-33620.15.1 (owner decision 2026-10-09): during a run, the poll's
#: config projections run at least this often even when no input changed.
RUN_BACKSTOP_SECONDS = 1.0
_clock = time.monotonic
_POLL_TICK: ContextVar[bool] = ContextVar("console_poll_tick", default=False)


def run_console_config_sync(
    sync: Callable[[], None],
    *,
    maintenance_paused: bool,
    defer: Callable[[], None],
    checked_projection: object | None = None,
) -> bool:
    """Render under current configuration, or defer without waiting on its locks.

    Args:
        sync: Synchronous render callback; never awaited under native locks.
        maintenance_paused: Current Console maintenance gate.
        defer: Request the existing coalesced, fresh-state trailing refresh.

    Returns:
        True after rendering; False when the existing retry owns the refresh.

    Raises:
        BaseException: Original checked entry, render or native cleanup failure.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.config_participants import operation

    if maintenance_paused:
        defer()
        return False
    if checked_projection is not None:
        from .console_spend_projection import _run_checked_display_sync

        displayed = _run_checked_display_sync(checked_projection, sync, defer)
        if displayed is not None:
            return displayed
    failure: BaseException | None = None
    entered = False
    try:
        with ExitStack() as acquired:
            # Same native locks/order as checked entry; retain them so another
            # thread cannot win between the nonblocking probe and entry.
            for lock in (config._settings_rebuild_lock(), config._config_file_lock()):
                if not lock.acquire(blocking=False):
                    defer()
                    return False
                acquired.callback(lock.release)
            # Nested readers still check the source. Reentrant native locks
            # preserve the unchanged operation through render and retirement.
            with operation(config):
                entered = True
                try:
                    sync()
                except BaseException as error:  # noqa: BLE001 - re-raised after native owner exit.
                    # A UI error must not mark config persistence as failed.
                    # Nested config failures retain their own failure state.
                    failure = error
    except BaseException as error:
        if (
            not entered
            and type(error) is RecoveryRequired
            and error.args == ("storage_locally_paused",)
        ):
            # Native intent can precede the local monitor. Recompute on
            # one trailing timer even if that intent is canceled unseen.
            defer()
            return False
        if failure is not None and error is not failure:
            raise error from failure
        raise
    if failure is not None:
        raise failure
    return True


def run_console_presentation_sync(screen, sync: Callable[[], None]) -> bool:
    """Select the existing screen's checked display, retaining live fallback."""
    from .console_spend_projection import (
        ConsoleReadinessConfigProjection,
        _screen_readiness_config,
    )

    def render(projection=None):
        return run_console_config_sync(
            sync,
            maintenance_paused=getattr(screen, "_console_sync_maintenance_paused", False),
            defer=lambda: screen._request_console_control_bar_sync(delayed=True),
            checked_projection=projection,
        )

    if getattr(screen, "_console_sync_maintenance_paused", False) or getattr(
        screen, "_closing", False
    ):
        screen._request_console_control_bar_sync(delayed=True)
        return False
    value = _screen_readiness_config(screen)
    if not value or not screen._console_config_snapshot_is_disk_loaded(value):
        return render()
    projection = ConsoleReadinessConfigProjection.for_screen(screen)
    return projection.run(lambda: render(projection))


def replay_whole_console_sync(screen) -> bool:
    """Hand the rest of a full UI sync to the coalesced control-bar replay.

    Returns:
        False, for a full sync that stops here to return; a caller that lets
        the pass continue (``sync_console_core_state``) ignores it.
    """
    screen._console_control_bar_replay_whole_sync = True
    screen._request_console_control_bar_sync(delayed=True)
    return False


def sync_console_core_state(screen) -> None:
    """Push the live provider selection unless that read would wait.

    TASK-35645: the full UI sync's push reads ``load_settings()`` live, which
    waited out any config rebuild on the UI loop. While a rebuild holds a cold
    config the push hands the whole sync to the replay instead, and the rest of
    this pass still runs. Otherwise it is the same live read a send's own push
    makes (``_ensure_console_chat_controller``).
    """
    from tldw_chatbook.config import config_read_would_wait

    if config_read_would_wait():
        replay_whole_console_sync(screen)
        return
    screen._sync_console_chat_core_state()


def sync_console_rail_visibility(screen) -> bool:
    """Apply rail visibility from checked display config, never waiting.

    TASK-35444: the full UI sync ran this after its transcript await with a
    live ``load_settings()``, which waited out any config rebuild that began
    in between, on the UI loop. It now reads like the other rail-visibility
    sync (``wiring.py``): checked display data, or a deferral that replays the
    whole sync.

    Returns:
        True after applying; False when the replay owns it.
    """
    if run_console_presentation_sync(
        screen,
        lambda: screen._sync_console_rail_visibility_if_changed(
            screen._current_console_rail_state()
        ),
    ):
        return True
    return replay_whole_console_sync(screen)


class _Cadence:
    """When the sync tick last ran its projections, and from which inputs."""

    def __init__(self) -> None:
        self.requested = 0  # Syncs other than the poll's, counted at entry.
        self.served = 0  # Requests a completed projection has covered.
        self.objects: tuple = ()  # Compared by identity (and kept alive).
        self.values: tuple | None = None
        self.at = 0.0


def _cadence(screen: Any) -> _Cadence:
    cadence = getattr(screen, "_console_config_cadence", None)
    if cadence is None:
        cadence = _Cadence()
        screen._console_config_cadence = cadence
    return cadence


async def run_periodic_console_sync(sync: Callable[[], Awaitable[Any]]) -> Any:
    """Run the 0.2 s poll's sync, whose projections may skip their operation.

    Args:
        sync: Starts the poll's sync (``sync_poll_display_ui`` on the screen).

    Returns:
        Whatever the sync returned.
    """
    token = _POLL_TICK.set(True)
    try:
        return await sync()
    finally:
        _POLL_TICK.reset(token)


def begin_console_sync(screen: Any) -> None:
    """Classify a sync at its entry; every sync but the poll's asks to project.

    Called before the sync's first await, so the poll's mark is consumed by
    the sync it was set for: work that sync starts (a worker, a replay) does
    not inherit it and counts as a request of its own.

    Args:
        screen: The Console screen whose sync is starting.
    """
    if _POLL_TICK.get():
        _POLL_TICK.set(False)
    else:
        _cadence(screen).requested += 1


def _projection_inputs(screen: Any) -> tuple[tuple, tuple, bool]:
    """What the tick's config projections read, cheaply; and whether a run is on.

    Returns:
        Objects compared by identity (the installed config, settings and
        persistence error: every install is a new object), values compared by
        equality, and whether any chat has a run going (the backstop's gate).
    """
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Chat.console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES

    store = getattr(screen, "_console_chat_store", None)
    controller = getattr(screen, "_console_chat_controller", None)
    session_id = getattr(store, "active_session_id", None)
    states = controller.run_states() if controller is not None else {}
    viewed_active = controller is not None and bool(screen._console_run_active())
    run_active = viewed_active or any(
        state.status in FEEDBACK_ACTIVE_RUN_STATUSES for state in states.values()
    )
    approvals = (
        controller.pending_round_count(session_id or "")
        if controller is not None
        else 0
    )
    objects = (
        config._CONFIG_CACHE,
        config._SETTINGS_CACHE,
        config._CONFIG_PERSISTENCE_ERROR,
    )
    values = (
        config._CONFIG_GENERATION,
        config._get_effective_config_path(),
        storage_admission._pause is not None,
        session_id,
        tuple(states.items()),
        viewed_active,
        approvals,
    )
    return objects, values, run_active


def project_console_tick(screen: Any) -> bool:
    """Run the tick's checked config projections unless nothing they read moved.

    TASK-33620.15.1 (owner decision 2026-10-09): the 0.2 s poll paid one
    checked config operation per tick (median 6-10 ms, max 44-80 ms live,
    over half of it the operation's entry and exit) to re-render unchanged
    state. The projections now run when a sync other than the poll's asked,
    when an input changed since the last projection, or once a second while
    a run is going. When they run, ``_sync_console_rail_and_controls`` is
    unchanged: same locks, refusal and deferral. A deferred or failed
    projection records nothing, so the next tick tries again. A config file
    edited outside the app is noticed by the projection's own config reads,
    so during a run it reaches the controls about a second later than it
    did; outside a run it waits for the next other sync or input change
    (the poll itself stops once no run work is left).

    Args:
        screen: The Console screen whose sync tick is running.

    Returns:
        False when the projection was deferred (the caller's existing replay
        owns it); True when it rendered or was not needed.
    """
    cadence = _cadence(screen)
    objects, values, run_active = _projection_inputs(screen)
    requested = cadence.requested
    unchanged = (
        requested == cadence.served
        and values == cadence.values
        and len(objects) == len(cadence.objects)
        and all(now is then for now, then in zip(objects, cadence.objects))
    )
    if unchanged and not (
        run_active and _clock() - cadence.at >= RUN_BACKSTOP_SECONDS
    ):
        return True
    if screen._sync_console_rail_and_controls() is False:
        return False
    cadence.served = requested
    cadence.objects, cadence.values, cadence.at = objects, values, _clock()
    return True
