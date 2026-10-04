from typing import Any, Callable


def build_console_controllers(
    screen: Any,
    *,
    read_trace_recovery_dispatch: Callable[[], Callable[..., Any]],
    read_trace_recovery_state: Callable[[], Callable[..., Any]],
) -> None:
    """Add these two parameters and four arguments to the existing wiring."""
    screen._session = ConsoleSessionController(
        screen,
        read_trace_recovery_dispatch=read_trace_recovery_dispatch,
        read_trace_recovery_state=read_trace_recovery_state,
        read_trace_recovery_started=lambda: screen._start_console_transcript_sync_timer,
        read_trace_recovery_finished=lambda: screen._sync_native_console_chat_ui,
    )
