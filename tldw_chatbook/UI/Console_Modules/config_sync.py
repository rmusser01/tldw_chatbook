"""The Console's synchronous checked configuration refresh lifetime.

Also decides how often the 0.2 s poll pays for it (TASK-33620.15.1).
"""

import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

#: TASK-33620.15.1 (owner decision 2026-10-09): while the poll ticks (it
#: ticks only while run work is live), its config projections run at least
#: this often even when no input changed.
RUN_BACKSTOP_SECONDS = 1.0
#: TASK-33620.15.4: while the run poll ticks, an explicit sync that lands
#: within this long of the last projection leaves its projection to the
#: poll's next tick (this is the poll's own interval), so the syncs a run
#: start requests share one projection.
RUN_COALESCE_SECONDS = 0.2
#: TASK-39145 (owner decisions 2026-11-07 and 2026-11-09): while config.toml
#: cannot be parsed, refused core-state pushes keep owing whole replays for
#: this long from the episode's first refusal, then stop and name the file on
#: the status strip until the next good load.
CORE_REFUSAL_EPISODE_SECONDS = 3.0
_clock = time.monotonic
_POLL_TICK: ContextVar[bool] = ContextVar("console_poll_tick", default=False)
#: The running sync is the poll's (set at its entry, for its own task).
_POLL_SYNC: ContextVar[bool] = ContextVar("console_poll_sync", default=False)


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
        from .console_spend_projection import (
            _CheckedDisplayRefused,
            _run_checked_display_sync,
        )

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
    if checked_projection is not None and type(failure) is _CheckedDisplayRefused:
        # An expired display may have selected this native route. A later
        # metadata refusal still requests replay, but only after native exit.
        # Cleanup errors propagate through the exception path above.
        defer()
        return False
    if failure is not None:
        raise failure
    return True



@dataclass(eq=False, repr=False)
class _ConsoleConfigHandoff:
    screen: Any
    owner: str
    source: dict
    token: Any = None
    projection: Any = None
    receipt: tuple | None = None

    def retire(self, *, owed: bool = False) -> None:
        from ...Canvas.config_policy import _end_canvas_poll_handoffs

        if self.token is not None:
            _end_canvas_poll_handoffs(self.screen, self.token)
        self.token = self.projection = self.receipt = None
        pending = getattr(self.screen, "_console_config_handoffs", {})
        if not owed and pending.get(self.owner) is self:
            pending.pop(self.owner)

    def current(self) -> bool:
        from .composer_run_controls import poll_source_is_current

        return not getattr(
            self.screen, "_console_sync_maintenance_paused", False
        ) and poll_source_is_current(self.screen, self.source)

    def begin(self, projection: Any) -> Any:
        from ...Canvas.config_policy import _begin_canvas_poll_handoff
        from .console_spend_projection import _handoff_projection_is_stock

        if not self.current() or not _handoff_projection_is_stock(projection):
            self.retire()
            return None
        if self.token is None or not self.token.active:
            self.token = _begin_canvas_poll_handoff(self.screen)
            self.projection, self.receipt = projection, None
        if self.owner == "identity" and projection.pending:
            _bind_identity_config_read(
                self.screen, projection, projection._handoff_read
            )
        return self.token

    def accept(self, projection: Any, token: Any) -> None:
        from .console_spend_projection import (
            _checked_display_proofs,
            _handoff_projection_is_stock,
            _same_readiness_key,
        )

        if self.token is not token:
            return
        proof = projection._display_proof
        if (
            token is None
            or not token.active
            or not self.current()
            or self.projection is not projection
            or not _handoff_projection_is_stock(projection)
            or projection.pending
            or projection._reads
            or projection._runtime_owner is not self.source["runtime"]
            or proof is None
            or proof not in _checked_display_proofs
            or proof.projection is not projection
            or proof.screen is not self.screen
            or proof.value is not projection.value
            or proof.reader is not projection.read_current
            or not _same_readiness_key(projection._key(), projection.key)
            or not _same_readiness_key(proof.owner, projection.key)
            or time.monotonic() - projection.at >= projection.max_age
        ):
            self.retire()
            return
        self.receipt = (
            projection.key,
            projection.value,
            projection.at,
            proof,
            projection.max_age,
        )

    def eligible(self) -> bool:
        if self.token is not None and (not self.token.active or not self.current()):
            self.retire()
            return False
        if self.receipt is None:
            read = getattr(self.projection, "_handoff_read", None)
            worker = read[2] if read is not None else None
            if worker is not None and (worker.is_cancelled or worker.is_finished):
                self.retire()
            return False
        receipt = self.receipt
        self.accept(self.projection, self.token)
        same = self.receipt is not None and (
            self.receipt[0] is receipt[0]
            and self.receipt[1] is receipt[1]
            and self.receipt[2] == receipt[2]
            and self.receipt[3] is receipt[3]
            and self.receipt[4] == receipt[4]
        )
        if not same:
            self.retire()
        return same


def _owe_console_config_sync(screen: Any, owner: str) -> _ConsoleConfigHandoff | None:
    from .composer_run_controls import poll_source

    pending = getattr(screen, "_console_config_handoffs", None)
    if pending is None:
        pending = screen._console_config_handoffs = {}
    current = pending.get(owner)
    if current is not None:
        if current.current():
            return current
        current.retire()
    try:
        source = poll_source(screen)
    except BaseException:
        return None  # A custom/incomplete observer keeps the original live route.
    if source is None:
        return None
    current = _ConsoleConfigHandoff(screen, owner, source)
    pending[owner] = current
    return current


def _begin_console_config_handoffs(screen: Any, projection: Any) -> tuple:
    request = getattr(screen, "_console_config_handoffs", {}).get("core")
    return () if request is None else ((request, request.begin(projection)),)


def _bind_identity_config_read(screen: Any, projection: Any, read: Any) -> None:
    """Join one issued logical read generation, never a later screen request."""
    from .console_spend_projection import _same_readiness_key

    request = getattr(screen, "_console_config_handoffs", {}).get("identity")
    if (
        read is not None
        and request is not None
        and request.projection is projection
        and request.token is not None
        and request.token.active
        and request.current()
        and _same_readiness_key(read[0], projection._key())
    ):
        read[1]["identity"] = (request, request.token)


def _finish_identity_config_read(projection: Any, read: Any, accepted: bool) -> None:
    if read is None:
        return
    for request, token in tuple(read[1].values()):
        if request.token is not token:
            continue
        if accepted:
            request.accept(projection, token)
        else:
            request.retire()


def _abandon_console_config_handoffs(screen: Any) -> None:
    for request in tuple(getattr(screen, "_console_config_handoffs", {}).values()):
        request.retire()


async def prepare_console_presentation_sync(screen: Any) -> bool:
    """Await the existing stock display owner before FULL's synchronous render.

    No native operation spans this await. Direct/custom presentation keeps its
    existing route; the caller owns whole-FULL replay when preparation refuses.

    Args:
        screen: Current Console view with its existing display projection.

    Returns:
        Whether FULL may attempt its original synchronous presentation.
    """
    from .console_spend_projection import (
        ConsoleReadinessConfigProjection,
        _same_readiness_key,
        _screen_readiness_config,
        _standard_readiness_projections,
    )

    if getattr(screen, "_console_sync_maintenance_paused", False) or getattr(
        screen, "_closing", False
    ):
        return False
    value = _screen_readiness_config(screen)
    warmed, key = True, None
    if value and screen._console_config_snapshot_is_disk_loaded(value):
        projection = ConsoleReadinessConfigProjection.for_screen(screen)
        if (
            type(projection) is ConsoleReadinessConfigProjection
            and projection in _standard_readiness_projections
        ):
            key = projection._key()
            handoffs = _begin_console_config_handoffs(screen, projection)
            try:
                warmed = await projection.warm()
            except BaseException:
                for request, token in handoffs:
                    if request.token is token and request.receipt is None:
                        request.retire()
                raise
            for request, token in handoffs:
                if warmed:
                    request.accept(projection, token)
                elif request.token is token:
                    request.retire()
            if not warmed and not _same_readiness_key(key, projection._key()):
                key = None  # A chat or model switch moved it mid-read.
    # TASK-38696: the replay backs off only while one unmoved owner keeps
    # refusing; a refusal for a switched chat or model starts over.
    previous = getattr(screen, "_console_warm_refused_key", None)
    refusals = getattr(screen, "_console_warm_refusals", 0)
    if not _same_readiness_key(key, previous):
        refusals = 0
    screen._console_warm_refused_key = None if warmed else key
    screen._console_warm_refusals = 0 if warmed else refusals + 1
    return warmed


async def rewarm_console_presentation(screen: Any) -> None:
    """Re-read an expired checked display before a FULL sync's rail step.

    The FULL warms the display before its projection, then awaits the
    transcript and the archive view. Under load those awaits outlived the
    display's 1 s life, the rail step refused it, and the whole sync replayed,
    itself and then its replay, about once a second (r11c-rev2). One read of
    the expired display replaces that replay; a live display reads nothing.
    While reads fail nothing is read either, as in ``warm()``: the rail step's
    own refusal keeps its backoff, and a second failing read in one sync
    doubled it (r11c-rev3).
    """
    from .console_spend_projection import (
        ConsoleReadinessConfigProjection,
        _standard_readiness_projections,
    )

    projection = getattr(screen, "_console_readiness_config_projection", None)
    if (
        type(projection) is ConsoleReadinessConfigProjection
        and projection in _standard_readiness_projections
        and not projection._retry_delay
    ):
        projection.run(lambda: None, life=projection.max_age)  # Expired only.
        if projection.pending:
            await projection._settled.wait()


def full_sync_replay_delay(screen: Any) -> bool | float:
    """Delay for the whole-sync replay of a FULL that could not render.

    TASK-38696: a refused warm-up used to replay every 0.2 s, starting another
    config read each time, for as long as the refusal lasted. Each further
    consecutive refusal now doubles the wait, on TASK-38649's schedule (0.2 s
    up to 5 s). The first replay keeps the fixed 0.2 s timer (a cold mount's
    one refusal replays exactly as before), and so does a maintenance pause:
    that timer is its resume path.

    Returns:
        True for the fixed 0.2 s timer, else the backed-off delay in seconds.
    """
    from .console_spend_projection import RETRY_FIRST_DELAY, RETRY_MAX_DELAY

    refusals = getattr(screen, "_console_warm_refusals", 0)
    if refusals <= 1 or getattr(screen, "_console_sync_maintenance_paused", False):
        return True
    return min(RETRY_FIRST_DELAY * 2 ** (refusals - 1), RETRY_MAX_DELAY)


def cut_replay_backoff(screen: Any, held: bool) -> None:
    """Let an explicit FULL that defers to a warm-up backoff run that replay now.

    TASK-38696 review: about 90 direct callers ask for a full sync (a user's
    switch, a publication of a fresh readiness read); deferring them to the
    backoff stalled the Console for up to 5 s after a refusal cleared. The
    run poll's own ticks do not cut: they come every 0.2 s, so they would
    bring the 0.2 s replay loop back for any refusal that overlaps a run, and
    a refused FULL renders nothing anyway. A fixed 0.2 s replay (a first
    refusal, a pause, a cold core-state push) stays as it was.

    Args:
        screen: The Console screen whose full sync was just deferred.
        held: Whether a send's durable commit deferred it instead.
    """
    if held or _POLL_SYNC.get() or getattr(
        screen, "_console_sync_maintenance_paused", False
    ):
        return
    if getattr(screen, "_console_replay_backoff_timer", None) is not None:
        screen._request_console_control_bar_sync()


def request_control_bar_sync(screen: Any, delayed: bool | float) -> None:
    """Coalesce a live screen's control-bar syncs into one trailing run.

    ``delayed=True`` waits 0.2 s and a number waits that long (a refused
    warm-up's backoff). A pending run absorbs later requests, except that an
    immediate request, such as the readiness projection's own retry or a
    maintenance resume, cuts a backoff wait short (TASK-38696).
    """
    backoff = getattr(screen, "_console_replay_backoff_timer", None)
    if getattr(screen, "_console_control_bar_sync_scheduled", False):
        if delayed or backoff is None:
            return
        backoff.stop()  # Stopping a fired timer is a no-op; its run finds no replay.
    screen._console_replay_backoff_timer = None
    screen._console_control_bar_sync_scheduled = True
    run = screen._run_coalesced_control_bar_sync
    if not delayed:
        screen.call_after_refresh(run)
    elif delayed is True:
        screen.set_timer(0.2, run)
    else:
        screen._console_replay_backoff_timer = screen.set_timer(delayed, run)


def run_console_presentation_sync(screen, sync: Callable[[], None]) -> bool:
    """Select the existing screen's checked display, retaining live fallback."""
    from .console_spend_projection import (
        ConsoleReadinessConfigProjection,
        _screen_readiness_config,
    )

    def render(projection=None):
        return run_console_config_sync(
            sync,
            maintenance_paused=getattr(
                screen, "_console_sync_maintenance_paused", False
            ),
            defer=lambda: screen._request_console_control_bar_sync(delayed=True),
            checked_projection=projection,
        )

    sync_config_unreadable_notice(screen)
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


def terminal_projection_owner(screen: Any) -> tuple[Any, str | None] | None:
    """Capture the owner of controls that settle after the terminal transcript."""
    from ...Chat.console_chat_models import ConsoleRunStatus

    if screen._current_console_run_status_value() not in (
        ConsoleRunStatus.COMPLETED.value,
        ConsoleRunStatus.FAILED.value,
        ConsoleRunStatus.STOPPED.value,
    ):
        return None
    store = screen._console_chat_store
    return store, getattr(store, "active_session_id", None)


def settle_terminal_projection(
    screen: Any,
    owner: tuple[Any, str | None],
    transcript_synced: bool | None,
    torn_down: Callable[[Any], bool],
) -> bool:
    """Publish the terminal controls after the current transcript settles."""
    if torn_down(screen):
        return False
    store, session_id = owner
    if (
        transcript_synced is False
        or screen._console_chat_store is not store
        or getattr(store, "active_session_id", None) != session_id
    ):
        return replay_whole_console_sync(screen)
    return project_console_tick(screen)


def sync_console_core_state(screen) -> None:
    """Push the live provider selection only when that read is warm and ours.

    TASK-35645: the full UI sync's push reads ``load_settings()`` live, which
    waited out any config rebuild on the UI loop. While a rebuild holds a cold
    config the push hands the whole sync to the replay instead, and the rest of
    this pass still runs. Otherwise it is the same live read a send's own push
    makes (``_ensure_console_chat_controller``).

    TASK-36511: the push now holds both config locks, and replays once when a
    live read would be cold. A worker read that took the locks after the old
    would-wait probe held the UI loop 238 ms; a cold read rebuilt config on it.
    The replayed pass pushes after its preparation warmed config off-loop, and
    pushes live if config is still cold (an unparseable file stays cold).

    TASK-39145: with config.toml unparseable, failing reads hold both locks
    through each rebuild, and under load every replay lost its probe. A refusal
    still owes the whole replay and its CORE handoff, as on dev (ADR-121
    amendment, 0aa96f8d1b), but only for ``CORE_REFUSAL_EPISODE_SECONDS`` of an
    unreadable file (owner decision 2026-11-09); then the status strip says so
    until the next good load. A parseable file never stops replaying.
    """
    from tldw_chatbook import config

    from .console_spend_projection import defer_cold_live_read

    sync_config_unreadable_notice(screen)
    with ExitStack() as held:
        for lock in (config._settings_rebuild_lock(), config._config_file_lock()):
            if not lock.acquire(blocking=False):
                break
            held.callback(lock.release)
        else:
            if not defer_cold_live_read(screen, "sync_console_core_state"):
                screen._console_core_refused_since = None  # The episode ends.
                request = getattr(screen, "_console_config_handoffs", {}).get("core")
                try:
                    screen._sync_console_chat_core_state()
                finally:
                    if request is not None:
                        request.retire()
                return
    if not _refusal_owes_replay(screen):
        # No replay owes it: a dropped replay retires the handoff and its
        # Canvas poll token (ADR-121 staged handoff; r12p-rev4).
        request = getattr(screen, "_console_config_handoffs", {}).get("core")
        if request is not None:
            request.retire()
        return
    request = _owe_console_config_sync(screen, "core")
    if request is not None and request.token is not None:
        request.retire(owed=True)  # One guarded attempt; whole replay still owns it.
    replay_whole_console_sync(screen)


def _refusal_owes_replay(screen) -> bool:
    """Whether a refused core-state push still owes its whole replay (TASK-39145)."""
    from tldw_chatbook import config

    if getattr(screen, "_console_config_unreadable_installs", None) is not None:
        return False  # Stopped and said so: nothing more until a good load.
    installs = config._CONFIG_CACHE_INSTALLS  # Before the failure, never after.
    if config.get_config_load_failure() is None:
        screen._console_core_refused_since = None
        return True  # A parseable file never stops replaying.
    since = getattr(screen, "_console_core_refused_since", None)
    if since is None:
        screen._console_core_refused_since = _clock()
        return True
    if _clock() - since < CORE_REFUSAL_EPISODE_SECONDS:
        return True
    screen._console_core_refused_since = None
    screen._console_config_unreadable_installs = installs  # Once per episode.
    sync_config_unreadable_notice(screen)
    return False


def sync_config_unreadable_notice(screen) -> None:
    """Paint the unreadable-config notice, or clear it after a good load.

    TASK-39145 (owner decision 2026-11-07). The notice stands from the install
    count recorded when the push stopped until any good load installs config
    (``_CONFIG_CACHE_INSTALLS`` moves); that also starts a new episode. Called
    by FULL and control-bar syncs and by a successful readiness read.
    """
    from tldw_chatbook import config

    installs = getattr(screen, "_console_config_unreadable_installs", None)
    if installs is not None and installs != config._CONFIG_CACHE_INSTALLS:
        installs = screen._console_config_unreadable_installs = None
        screen._console_core_refused_since = None
    shown = installs is not None
    if not shown and not getattr(screen, "_console_config_unreadable_painted", False):
        return  # Nothing shown or to clear: no DOM query on the sync path.
    from ...Widgets.Console.console_status_chips import ConsoleStatusChips

    for chips in screen.query(ConsoleStatusChips):
        chips.sync_config_unreadable(shown)
    screen._console_config_unreadable_painted = shown



def run_coalesced_control_bar_sync(screen: Any, is_torn_down: Callable) -> None:
    """Service one accepted CORE handoff without reordering the original FULL."""
    screen._console_control_bar_sync_scheduled = False
    if is_torn_down(screen):
        _abandon_console_config_handoffs(screen)
        screen._console_control_bar_replay_whole_sync = False
        return
    paused = getattr(screen, "_console_sync_maintenance_paused", False)
    if paused:
        _abandon_console_config_handoffs(screen)
    whole = getattr(screen, "_console_control_bar_replay_whole_sync", False)
    if not screen._console_sync_in_progress and not whole:
        request = getattr(screen, "_console_config_handoffs", {}).get("core")
        if request is not None:
            request.retire()
    if whole:
        if screen._console_sync_in_progress or paused:
            request = getattr(screen, "_console_config_handoffs", {}).get("core")
            if not paused and request is not None and request.eligible():
                sync_console_core_state(screen)
            screen._request_console_control_bar_sync(delayed=True)
            return
        screen._console_control_bar_replay_whole_sync = False
        screen._console_sync_requested = False
        screen.run_worker(
            screen._sync_native_console_chat_ui, exclusive=True, group="console-sync"
        )
    else:
        screen._sync_console_control_bar()


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
        self.published = 0  # Snapshot publications handed to the poll (an input).
        self.waiting_since: float | None = None  # First request left to the poll.
        self.commit_synced: float | None = None  # The commit a sync already ran in.
        self.commit_holding = False  # Syncs wait until the poll's next tick.


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
        *args: Passed to ``sync``; the first is the Console screen (or
            ``sync`` is a method bound to it).

    Returns:
        Whatever the sync returned (False: a deferred display, which the
        poll leaves to its replay). False, with no sync, for a tick that
        waits out a send's durable commit (``_held_for_durable_commit``).
    """
    screen = args[0] if args else getattr(sync, "__self__", None)
    if screen is not None and _held_for_durable_commit(screen, poll=True):
        return False
    token = _POLL_TICK.set(True)
    try:
        return await sync(*args)
    finally:
        _POLL_TICK.reset(token)


def _durable_commit_since(screen: Any) -> float | None:
    """When the screen's store began the durable commit now in flight, if any.

    Only while the run poll ticks: a sync held for the commit is served by
    a later poll tick, so with no poll nothing is held. Read without an
    import: the commit's window lives in ``console_native_commit``, and a
    process that never loaded it has never committed.
    """
    if getattr(screen, "_console_transcript_sync_timer", None) is None:
        return None
    commits = sys.modules.get("tldw_chatbook.Chat.console_native_commit")
    if commits is None:
        return None
    return commits.durable_commit_started_at(getattr(screen, "_console_chat_store", None))


def _held_for_durable_commit(screen: Any, *, poll: bool) -> bool:
    """Whether a whole sync waits out the send's durable commit now in flight.

    TASK-33620.15.4 (measured live 2026-10-23, 30 sends per arm): a send's
    durable turn commit runs on a worker thread for 140-180 ms while the
    send waits on it, and UI-loop work in that window lengthens it. On the
    base build the poll's first tick in the commit deferred to a whole-sync
    replay 0.2 s later, and every sync until that replay returned at once:
    the loop stayed quiet for the rest of the commit by accident. Without
    that deferral, the first tick's re-armed sync and the next tick painted
    inside the commit, which then took 218 ms instead of 180 ms, and warm
    sends reached the provider a median 31 ms later.

    So, while the run poll ticks and a commit is in flight, the commit's
    first whole sync runs (as the base's first tick did) and opens a hold
    that lasts until the poll's next tick: explicit syncs started meanwhile
    leave their request (``_console_sync_requested``) for later, and that
    next tick skips and ends the hold. The tick after it runs and serves
    what was held, two poll intervals at most after the first sync. Against
    the base that is one interval later where the base deferred to its
    replay (0.83 replays per measured send window) and up to two where it
    did not, since the base then rendered an explicit sync at once. Each
    commit holds once; the hold ends early if the commit does.

    The window is per store, and one store holds every Console session, so
    a commit in one session holds one tick for all of them, a streaming one
    included (once per commit, never two ticks in a row).

    Args:
        screen: The Console screen whose sync is starting.
        poll: The run poll's own tick (else an explicit sync).

    Returns:
        True when this sync waits for a later tick.
    """
    cadence = _cadence(screen)
    since = _durable_commit_since(screen)
    if since is None:
        cadence.commit_synced, cadence.commit_holding = None, False
        return False
    if cadence.commit_synced != since:  # The commit's first sync runs.
        cadence.commit_synced, cadence.commit_holding = since, True
        return False
    if not cadence.commit_holding:
        return False
    if poll:
        cadence.commit_holding = False  # This tick skips; the next one runs.
    return True


def poll_sync_running() -> bool:
    """Whether the calling task runs the 0.2 s poll's own sync (TASK-36571.1)."""
    return _POLL_SYNC.get()


def begin_console_sync(screen: Any) -> bool:
    """Classify a sync at its entry; every sync but the poll's asks to project.

    Called before the sync's first await, so the poll's mark is consumed by
    the sync it was set for: work that sync starts (a worker, a replay) does
    not inherit it and counts as a request of its own.

    Args:
        screen: The Console screen whose sync is starting.

    Returns:
        True when an explicit sync waits for a later poll tick because a
        send's durable commit is in flight (``_held_for_durable_commit``);
        the caller records the request and returns. The poll's own tick was
        decided before its sync started.
    """
    poll = _POLL_TICK.get()
    if poll:
        _POLL_TICK.set(False)
    else:
        _cadence(screen).requested += 1
    _POLL_SYNC.set(poll)
    return not poll and _held_for_durable_commit(screen, poll=False)


#: TASK-33620.15.4: a send's start (see ``send_start_in_progress``) ends at
#: its provider request, or this long after acceptance if it never makes one.
SEND_START_LIMIT_SECONDS = 3.0


def send_start_in_progress(screen: Any) -> bool:
    """Whether a send on the screen's store has not made its provider request yet.

    TASK-33620.15.4 (measured live 2026-10-23, 8 first sends per arm): an
    off-loop display read opens its own SQLite connection (34 ms wall, 10 ms
    CPU measured alone; 20-60 ms CPU each in the app), and the Library
    activity, registry and appearance re-reads started at acceptance and right
    after the durable commit took that CPU from the send's own work, so first
    sends reached the provider a median 66 ms later than with the same reads
    made on the loop (0.3-3 ms each, as on the base). During a send's start
    those reads stay synchronous; elsewhere they stay off the loop.

    TASK-36610: each send owns its start (``console_send_start``), opened when
    the runtime accepts it (any origin: the composer's sends, a queued
    prompt's dispatch, an agent wake, a goal iteration) or when an agent's
    chat start submits, and ended by that send's own provider request, its
    end, or ``SEND_START_LIMIT_SECONDS``. Another session's request ends
    nothing here.

    Args:
        screen: The Console screen.

    Returns:
        True while any send's start on the screen's store is open.
    """
    starts = sys.modules.get("tldw_chatbook.Chat.console_send_start")
    if starts is None:  # Loaded by the first acceptance.
        return False
    store = getattr(screen, "_console_chat_store", None)
    return store is not None and starts.send_start_open(store, SEND_START_LIMIT_SECONDS)


def warm_run_presentation(screen: Any) -> None:
    """Re-read an aging checked display as run work starts (TASK-33620.15.4).

    The poll's first tick comes 0.2 s after this. An idle Console projects
    nothing, so its checked display had always expired by then: the first
    projection after a send ran the native config operation on the UI loop,
    and the same tick's later presentation reads deferred to a whole-sync
    replay. Starting the read here lets it land before that tick.

    Args:
        screen: The Console screen whose run poll is starting.
    """
    from .console_spend_projection import (
        ConsoleReadinessConfigProjection,
        _screen_readiness_config,
    )

    if getattr(screen, "_closing", False):
        return
    value = _screen_readiness_config(screen)
    if value and screen._console_config_snapshot_is_disk_loaded(value):
        ConsoleReadinessConfigProjection.for_screen(screen).run(lambda: None)


def publish_console_presentation(screen: Any, group: str) -> None:
    """Hand a refreshed presentation snapshot to the next projection.

    TASK-33620.15.4: each snapshot publication used to start a whole sync,
    which projected again however recently another sync had (one to three
    times in a send's run start). While the run poll ticks, a publication
    is a cadence input instead: the next tick, within one poll interval,
    projects it together with whatever else that interval asked for. One
    landing after the tick's projection, on the tick that stops the poll,
    is owed, so the stop renders it (``settle_stopped_poll``). With no
    poll, a whole sync renders it at once, as before.

    Args:
        screen: The Console screen the snapshot belongs to.
        group: The worker group of the whole sync started without a poll.
    """
    if getattr(screen, "_console_transcript_sync_timer", None) is not None:
        cadence = _cadence(screen)
        cadence.published += 1
        cadence.owed = True
        return
    screen.run_worker(
        screen._sync_native_console_chat_ui, exclusive=False, group=group
    )


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
    from tldw_chatbook.Chat.console_chat_models import (
        FEEDBACK_ACTIVE_RUN_STATUSES,
        ConsoleRunStatus,
    )
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
            # A terminal run state can paint before its queue releases. Observe
            # that final ownership change even when the run state stays put.
            if (
                state is not None
                and state.status is not ConsoleRunStatus.IDLE
                and state.is_send_allowed
            ):
                lifecycle_revision = getattr(
                    controller, "_lifecycle_revision_for", None
                )
                if callable(lifecycle_revision):
                    run += (lifecycle_revision(session_id),)
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
        _cadence(screen).published,
        run,
    )
    return objects, values


def _left_to_the_poll(screen: Any, cadence: _Cadence) -> bool:
    """Whether an explicit sync's projection waits for the run poll's next tick.

    TASK-33620.15.4 (measured 2026-10-23): a send's run start asked for one to
    four whole syncs (a deferral's replay, a sync re-armed behind the poll's
    own, snapshot publications), and each projected at once, 40-120 ms after
    the poll's tick had, before its next tick projected again. While the poll
    ticks, a request that lands within ``RUN_COALESCE_SECONDS`` of the last
    projection now stays requested (``requested`` ahead of ``served``), so
    the next tick projects it with everything else of that interval; a
    request already waiting that long projects in whichever sync comes
    first. With no poll, every explicit sync projects at once, as before.
    """
    if getattr(screen, "_console_transcript_sync_timer", None) is None:
        cadence.waiting_since = None
        return False
    now = _clock()
    if cadence.waiting_since is None:
        if now - cadence.at >= RUN_COALESCE_SECONDS:
            return False
        cadence.waiting_since = now
        return True
    return now - cadence.waiting_since < RUN_COALESCE_SECONDS


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

    TASK-33620.15.4: while the poll ticks, an explicit sync within a poll
    interval of the last projection leaves its projection to the next tick
    (``_left_to_the_poll``), and a snapshot published meanwhile is an input
    (``publish_console_presentation``), so a run start's syncs share one
    projection.

    Args:
        screen: The Console screen whose sync tick is running.
        force: Run the projections even when nothing moved.

    Returns:
        False when the projection was deferred (handed to the delayed
        whole-sync replay here); True when it rendered or was not needed.
    """
    cadence = _cadence(screen)
    # TASK-33620.15.1.1: the input read and the render below share one read
    # of the Agent rail's drilled sub-agent run (the agent's own scope).
    agent = getattr(screen, "_agent", None)
    shared_reads = getattr(type(agent), "tick_reads_shared", None)
    with shared_reads(agent) if shared_reads is not None else nullcontext():
        try:
            objects, values = _projection_inputs(screen)
        except Exception:  # noqa: BLE001 -- the tick worker exits the app on a raise.
            # An unreadable input is a change: project (the parent's tick read
            # these inside the cost chip's best-effort build), and record none.
            objects, values = (), None
        if not force and not _POLL_SYNC.get() and _left_to_the_poll(screen, cadence):
            cadence.owed = True
            return True
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
    cadence.waiting_since = None
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
        if _cadence(screen).waiting_since is not None:
            # TASK-33620.15.4: the running sync left its projection to a poll
            # that is stopping; its re-armed sync (with no poll) projects.
            screen._console_sync_requested = True
        return
    workspace = getattr(screen, "_workspace", None)
    scope = getattr(workspace, "tick_workspace_build_scope", nullcontext)
    with scope():  # One workspace-context build, as on a tick.
        project_console_tick(screen, force=True)  # Defers as a tick does.
