"""The Console's synchronous checked configuration refresh lifetime.

Also decides how often the 0.2 s poll pays for it (TASK-33620.15.1).
"""

import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, nullcontext
from contextvars import ContextVar
from typing import Any

#: TASK-33620.15.1 (owner decision 2026-10-09): while the poll ticks (it
#: ticks only while run work is live), its config projections run at least
#: this often even when no input changed.
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
        self.owed = False  # A poll tick skipped since the last projection.


def _cadence(screen: Any) -> _Cadence:
    cadence = getattr(screen, "_console_config_cadence", None)
    if cadence is None:
        cadence = _Cadence()
        screen._console_config_cadence = cadence
    return cadence


async def run_periodic_console_sync(
    sync: Callable[..., Awaitable[Any]], *args: Any
) -> Any:
    """Run the 0.2 s poll's sync, whose projections may skip their operation.

    Args:
        sync: Starts the poll's sync (the screen's ``sync_poll_display_ui``).
        *args: Passed to ``sync``.

    Returns:
        Whatever the sync returned (False: a deferred display, which the
        poll leaves to its replay).
    """
    token = _POLL_TICK.set(True)
    try:
        return await sync(*args)
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


def _pending_rounds(store: Any, controller: Any) -> tuple:
    """Every chat's outstanding interrupt rounds, by kind, and the fleet counts.

    Approvals, questions and confirmations on any chat: the active chat's
    show in the controls and Inspector, a background chat's in the Agent
    rail's fleet line (``fleet_summary_counts``).
    """
    sessions = store.sessions() if store is not None else ()
    rounds = tuple(
        (session.id, kind, controller.pending_round_count(session.id, kind=kind))
        for session in sessions
        for kind in sorted(controller.pending_round_kinds(session.id))
    )
    return rounds, controller.fleet_summary_counts()


def _legacy_approval(screen: Any) -> tuple | bool:
    """The resume state's approval, read by value as the approval count reads it.

    A legacy unscoped tool round mounts its card with no kind-registry
    entry, so no pending-round count moves; ``ChatScreen.
    _console_pending_approval_count`` counts it from here for the controls,
    rail and Inspector, dropping one whose phase is ``finishing`` or that
    another chat owns. The interrupt host can set a retained round's phase
    in place, so a held reference would compare equal to itself.
    """
    state = getattr(screen, "_task_resume_state", None)
    approval = getattr(state, "pending_approval", None)
    if isinstance(approval, dict):
        return tuple(approval.get(key) for key in ("round_id", "phase", "session_id"))
    return bool(approval)


def _cost_inputs(
    store: Any, controller: Any, agent: Any, session_id: str, in_flight: bool
) -> tuple:
    """What moves the cost chip: settled rows and usage, fleet spend, cache TTL.

    The chip leaves streaming rows out of its total. Rows added or settled
    move the payload revision. The display revision also moves on every
    streamed chunk (the speech fence), so it counts only while the chat has
    no run in flight -- where a Stop's late usage lands. The cache TTL
    contributes its displayed second while warm, on the cadence's clock.
    """
    warm_until, had_activity = controller.cache_ttl_snapshot(session_id)
    remaining = None
    if had_activity and warm_until is not None:
        remaining = max(int(warm_until - _clock()), -1)
    return (
        None if in_flight else store.display_projection_revision(session_id),
        store.payload_revision(session_id),
        store.session_settings_revision(session_id),
        agent._console_agent_fleet_token_total() if agent is not None else 0,
        controller.unattributed_fleet_tokens(session_id),
        had_activity,
        remaining,
    )


def _projection_inputs(screen: Any) -> tuple[tuple, tuple]:
    """What the tick's projections read, cheaply, including what moves on its own.

    Review round (2026-10-10): besides config, the projections render things
    that change during a run with no sync asking -- the Agent rail's status,
    step and fleet rows (elapsed time and token totals included), the cost
    chip's running total, the readiness evidence and the tool counts. Those
    are inputs too, as the text or counter the screen shows, so they keep
    the parent's tick rate. No private module global is read (a renamed one
    must not reach a live tick) except the installed config and settings,
    by identity with a default; the Agent rail's rows come from the rail's
    own readers, the ones the projection renders from.

    Returns:
        Objects compared by identity (the installed config and settings, the
        dictionary and world-book summaries) and values compared by equality.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Chat.console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES
    from tldw_chatbook.Chat.provider_test_evidence import shared_connection_evidence

    store = getattr(screen, "_console_chat_store", None)
    controller = getattr(screen, "_console_chat_controller", None)
    agent = getattr(screen, "_agent", None)
    app = getattr(screen, "app_instance", None)
    session_id = getattr(store, "active_session_id", None)
    states = controller.run_states() if controller is not None else {}
    viewed_active = controller is not None and bool(screen._console_run_active())
    error = getattr(config, "_CONFIG_PERSISTENCE_ERROR", None)
    # Instance-held owners only: screen doubles resolve class attributes
    # (``app``, the summary descriptors) as bound methods.
    evidence = shared_connection_evidence(lambda: app) if app is not None else None
    retrieval = getattr(screen, "_retrieval", None)
    run: tuple = ()
    if controller is not None:
        run = (_pending_rounds(store, controller),)
        if session_id is not None:
            state = states.get(session_id)
            in_flight = viewed_active or (
                state is not None and state.status in FEEDBACK_ACTIVE_RUN_STATUSES
            )
            run += (_cost_inputs(store, controller, agent, session_id, in_flight),)
    if agent is not None:
        with agent.historical_reads_held():  # The projection starts those.
            run += (
                agent._console_agent_section_lines(),
                agent._console_agent_fleet_rows(),
            )
    objects = (
        getattr(config, "_CONFIG_CACHE", None),
        getattr(config, "_SETTINGS_CACHE", None),
        getattr(retrieval, "_active_dictionaries_summary", None),
        getattr(retrieval, "_active_world_books_summary", None),
    )
    values = (
        config.current_config_identity(),
        None if error is None else (type(error).__name__, str(error)),
        storage_admission.local_pause_active(),
        session_id,
        tuple(states.items()),
        viewed_active,
        tuple(
            getattr(app, name, None)
            for name in (
                "console_tool_count",
                "console_mcp_tool_count",
                "console_mcp_not_connected_count",
                "console_provider_ready",
            )
        ),
        evidence.version if evidence is not None else None,
        _legacy_approval(screen),
        run,
    )
    return objects, values


def project_console_tick(screen: Any, *, force: bool = False) -> bool:
    """Run the tick's checked config projections unless nothing they read moved.

    TASK-33620.15.1 (owner decision 2026-10-09): the 0.2 s poll paid one
    checked config operation per tick (median 6-10 ms, max 44-80 ms live,
    over half of it the operation's entry and exit) to re-render unchanged
    state. The projections now run when a sync other than the poll's asked,
    when an input changed since the last projection, or at least once a
    second while the poll ticks, which it does only while run work is live
    (a run, and its tail: a custodied turn, a wake delivery, a review
    publication, a finishing run worker); ``settle_stopped_poll`` renders
    what a skipped last tick left. When they run,
    ``_sync_console_rail_and_controls`` is unchanged: same locks, refusal and
    deferral. A deferred or failed projection records nothing, so the next
    tick tries again. What changes on its own during a run (the Agent rail,
    the cost chip, readiness evidence, tool counts, pending rounds and a
    legacy unscoped approval) is an input, so it renders on the next tick
    as before. What can reach the
    screen up to a second later is whatever else the projections read that
    no input sees change: a config value read fresh inside them, the
    context estimate (itself refreshed once a second while streaming), and
    state with its own syncs or that moves with an input (the workbench's
    image-edit and prepared-start state, whether the Agent rail offers its
    full log, the cost chip's helper usage and pricing catalog). When the
    poll does not tick there was no poll to render it before either.

    Since PR #3023 a projection renders from a checked display, with no
    operation, while that display is under a second old, and pays the
    native operation (or defers) once it is stale; its presentation reads
    can also publish a follow-up whole sync. A skipped tick pays none of
    that: no render, no display bookkeeping, no follow-up. Counted with a
    fake clock (one 0.2 s tick each, a 15-chunk streamed run), a run made
    25-43 projections without this cadence and 9-18 with it.

    Args:
        screen: The Console screen whose sync tick is running.
        force: Run the projections even when nothing moved.

    Returns:
        False when the projection was deferred (handed to the delayed
        whole-sync replay here); True when it rendered or was not needed.
    """
    cadence = _cadence(screen)
    try:
        objects, values = _projection_inputs(screen)
    except Exception:  # noqa: BLE001 -- the tick worker exits the app on a raise.
        # An unreadable input is a change: project (the parent's tick read
        # these inside the cost chip's best-effort build), and record none.
        objects, values = (), None
    requested = cadence.requested
    unchanged = (
        values is not None
        and requested == cadence.served
        and values == cadence.values
        and len(objects) == len(cadence.objects)
        and all(now is then for now, then in zip(objects, cadence.objects))
    )
    if not force and unchanged and _clock() - cadence.at < RUN_BACKSTOP_SECONDS:
        cadence.owed = True
        return True
    if screen._sync_console_rail_and_controls() is False:
        # The tick's deferral: the coalesced whole-sync replay owns the
        # render. A cold checked display defers without asking for one.
        return replay_whole_console_sync(screen)
    cadence.served = requested
    cadence.objects, cadence.values, cadence.at = objects, values, _clock()
    cadence.owed = False
    return True


def settle_stopped_poll(screen: Any, torn_down: Callable[[Any], bool]) -> None:
    """Render, as the poll stops itself, what its skipped ticks left unrendered.

    The poll's last tick used to render; once it stops nothing renders until
    some other sync, so a skipped last tick would leave the controls as they
    were up to a second before the run work ended, for as long as the
    Console then sits idle. A running sync, a pending whole-sync replay or a
    maintenance pause owns that render instead (the pause's resume replays a
    requested sync), exactly as for a tick's own deferral.

    Args:
        screen: The Console screen whose poll is stopping itself.
        torn_down: The screen's teardown predicate; a closing screen renders
            nothing.
    """
    if not _cadence(screen).owed or torn_down(screen):
        return
    if getattr(screen, "_console_sync_maintenance_paused", False):
        screen._console_sync_requested = True
        return
    if getattr(screen, "_console_sync_in_progress", False) or getattr(
        screen, "_console_control_bar_replay_whole_sync", False
    ):
        return
    workspace = getattr(screen, "_workspace", None)
    scope = getattr(workspace, "tick_workspace_build_scope", nullcontext)
    with scope():  # One workspace-context build, as on a tick.
        project_console_tick(screen, force=True)  # Defers as a tick does.
