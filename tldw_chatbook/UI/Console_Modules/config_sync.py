"""The Console's synchronous checked configuration refresh lifetime."""

from collections.abc import Callable
from contextlib import ExitStack


def run_console_config_sync(
    sync: Callable[[], None],
    *,
    maintenance_paused: bool,
    defer: Callable[[], None],
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
