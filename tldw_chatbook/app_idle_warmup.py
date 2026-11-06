"""UI-loop completion of the app's speculative first-send warm-up."""

from typing import Any


def request_idle_heap_refreeze(app: Any) -> None:
    """Re-freeze at once, or request maintenance, for the current UI-loop attachment."""
    try:
        shutting_down = app._shutting_down
    except AttributeError:
        shutting_down = False
    if shutting_down:
        return
    try:
        runtime = app.console_runtime
    except AttributeError:
        runtime = None
    view = runtime.view if runtime is not None else None
    if view is None:
        return
    from tldw_chatbook.UI.Console_Modules.transcript import (
        finish_console_heap_warm_up,
    )

    finish_console_heap_warm_up(view, "first_send_preimport")
