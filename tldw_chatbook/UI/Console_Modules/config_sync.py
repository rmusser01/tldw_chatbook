"""Checked Console config preparation and synchronous presentation lifetime."""

from collections.abc import Callable
from contextlib import ExitStack
from typing import Any


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
        _screen_readiness_config,
        _standard_readiness_projections,
    )

    if getattr(screen, "_console_sync_maintenance_paused", False) or getattr(
        screen, "_closing", False
    ):
        return False
    value = _screen_readiness_config(screen)
    if not value or not screen._console_config_snapshot_is_disk_loaded(value):
        return True
    projection = ConsoleReadinessConfigProjection.for_screen(screen)
    if (
        type(projection) is not ConsoleReadinessConfigProjection
        or projection not in _standard_readiness_projections
    ):
        return True
    return await projection.warm()


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
