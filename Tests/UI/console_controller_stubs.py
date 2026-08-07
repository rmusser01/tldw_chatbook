"""Controller stubs for bypassed-``__init__`` ``ChatScreen`` test shells.

Many Console tests build their screen with ``ChatScreen.__new__(ChatScreen)``
and hand-set the handful of attributes the code under test reads, instead of
mounting a real app. That shell never runs ``ChatScreen.__init__``, so it
never gets the sub-controllers the Console decomposition introduced -- and
any call into a method that was moved to one (reached through the screen's
delegation under its original name) fails with
``AttributeError: 'ChatScreen' object has no attribute '_message'``.

``stub_message_controller`` closes that gap. Every constructor callable
defaults to a raiser, so a shell only gets working behaviour for the seams
the caller explicitly wires -- a test that wanders into an unwired branch
fails loudly at the seam instead of silently taking a no-op path.

Wave-3 console decomposition, task 1. Waves 2 and 3 keep expanding the set
of moved methods, so prefer extending this module over hand-rolling another
copy of the constructor call inside a test file.
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController

#: Every keyword-only dependency of ``ConsoleMessageController.__init__``
#: except ``app_instance`` (a plain value, not a callable).
MESSAGE_CONTROLLER_CALLABLES = (
    "chat_store_accessor",
    "current_chat_store_accessor",
    "ensure_console_chat_controller",
    "current_chat_controller_accessor",
    "sync_native_console_chat_ui",
    "active_session_is_ephemeral",
    "active_native_console_session",
    "current_console_conversation_id",
    "active_console_provider_model_display",
    "console_initial_session_title_for_workspace",
    "console_change_review_run_id",
    "open_change_review",
    "start_console_transcript_sync_timer",
    "clear_native_console_message_selection",
    "regenerate_console_generation_variant",
    "select_console_generation_variant",
    "keep_console_generation_variant",
    "handle_console_toggle_image_view",
    "invalidate_console_persisted_rows_cache",
)


def _raiser(name: str, context: str):
    def _unreached(*_args, **_kwargs):
        raise AssertionError(
            f"{context}: constructor callable {name!r} is not wired for real "
            "-- the scenario reaching it needs its own stub."
        )

    return _unreached


def stub_message_controller(
    screen: Any,
    *,
    context: str = "stub_message_controller",
    app_instance: Any = None,
    **wired: Any,
) -> ConsoleMessageController:
    """Attach a ``ConsoleMessageController`` to a bare ``ChatScreen`` shell.

    Args:
        screen: The ``ChatScreen.__new__(ChatScreen)`` shell. Gets its
            ``_message`` attribute set as a side effect.
        context: Label used in the failure message of unwired callables, so
            a fail-loud trip names the fixture it came from.
        app_instance: Value for the controller's ``app_instance`` snapshot.
            Defaults to the shell's own ``app_instance`` when it has one.
        **wired: Any subset of ``MESSAGE_CONTROLLER_CALLABLES``, wired for
            real. Everything omitted raises ``AssertionError`` when called.

    Returns:
        The controller, already assigned to ``screen._message``.

    Raises:
        TypeError: If ``wired`` names something that is not a constructor
            callable -- a typo would otherwise silently leave that seam
            raising.
    """
    unknown = set(wired) - set(MESSAGE_CONTROLLER_CALLABLES)
    if unknown:
        raise TypeError(
            f"stub_message_controller got unknown callable(s) {sorted(unknown)}; "
            f"expected a subset of {list(MESSAGE_CONTROLLER_CALLABLES)}"
        )

    kwargs = {
        name: wired.get(name, _raiser(name, context))
        for name in MESSAGE_CONTROLLER_CALLABLES
    }
    controller = ConsoleMessageController(
        screen,
        app_instance=(
            app_instance
            if app_instance is not None
            else getattr(screen, "app_instance", None)
        ),
        **kwargs,
    )
    screen._message = controller
    return controller
