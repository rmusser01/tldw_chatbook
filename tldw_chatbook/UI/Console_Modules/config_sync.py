"""The Console's synchronous checked configuration refresh lifetime."""

from collections.abc import Callable
from contextlib import ExitStack


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
    """Push the provider selection from checked display config, never waiting.

    TASK-35645: the full UI sync pushed it with a live ``load_settings()``,
    which waited out any config rebuild on the UI loop. A deferral hands the
    whole sync to the replay; the rest of this pass still runs, and each later
    step defers itself. A send pushes the live selection itself
    (``_ensure_console_chat_controller``).
    """
    if not run_console_presentation_sync(screen, screen._sync_console_chat_core_state):
        replay_whole_console_sync(screen)


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
