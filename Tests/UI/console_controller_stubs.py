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

from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController
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

IMAGE_CONTROLLER_CALLABLES = (
    "ensure_console_image_view",
    "recent_console_image_messages",
    "console_image_default_mode",
    "console_generation_browse",
    "sync_native_console_chat_ui",
    "ensure_console_chat_store",
    "build_console_provider_selection",
    "ensure_console_provider_gateway",
    "console_image_preparing",
    "current_console_chat_store",
    "console_composer_or_none",
    "console_visible_draft_session_id",
    "append_native_console_system_message",
    "request_console_control_bar_sync",
    "default_console_session_settings",
    "clear_console_composer_draft",
)


def _raiser(name: str, context: str):
    def _unreached(*_args, **_kwargs):
        raise AssertionError(
            f"{context}: constructor callable {name!r} is not wired for real "
            "-- the scenario reaching it needs its own stub."
        )

    return _unreached


#: Sentinel for "this fixture deliberately has no app". Distinct from the
#: `None` a not-yet-wired shell yields, which is the bug this separates out.
NO_APP: Any = object()


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
            Defaults to the shell's own ``app_instance``. Pass the sentinel
            ``NO_APP`` to assert deliberately that this fixture has no app.
        **wired: Any subset of ``MESSAGE_CONTROLLER_CALLABLES``, wired for
            real. Everything omitted raises ``AssertionError`` when called.

    Returns:
        The controller, already assigned to ``screen._message``.

    Raises:
        TypeError: If ``wired`` names something that is not a constructor
            callable -- a typo would otherwise silently leave that seam
            raising.
        AssertionError: If no ``app_instance`` can be resolved and ``NO_APP``
            was not passed. The controller SNAPSHOTS this value, so a fixture
            that attaches before its harness app exists captures ``None``
            forever -- and every moved body reads it through
            ``getattr(self.app_instance, ..., None)``, so the test then takes
            a silent default branch instead of failing. That already happened
            once (a fixture attaching the controller a line too early), and
            it is the one silent-default hole in an otherwise fail-loud
            factory. Making it explicit costs one argument and closes it.
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
    resolved_app = (
        app_instance
        if app_instance is not None
        else getattr(screen, "app_instance", None)
    )
    assert resolved_app is not None, (
        f"{context}: no app_instance to snapshot. Attach the controller AFTER "
        "the harness app sets screen.app_instance, or pass app_instance=NO_APP "
        "to state that this fixture deliberately has none."
    )
    controller = ConsoleMessageController(
        screen,
        app_instance=None if resolved_app is NO_APP else resolved_app,
        **kwargs,
    )
    screen._message = controller
    return controller


def stub_image_controller(
    screen: Any,
    *,
    context: str = "stub_image_controller",
    app_instance: Any = None,
    **wired: Any,
) -> ConsoleImageController:
    """Attach a fail-loud image controller to a bare screen shell."""
    unknown = set(wired) - set(IMAGE_CONTROLLER_CALLABLES)
    if unknown:
        raise TypeError(
            f"stub_image_controller got unknown callable(s) {sorted(unknown)}; "
            f"expected a subset of {list(IMAGE_CONTROLLER_CALLABLES)}"
        )
    resolved_app = (
        app_instance
        if app_instance is not None
        else getattr(screen, "app_instance", None)
    )
    assert resolved_app is not None, (
        f"{context}: no app_instance to snapshot. Attach the controller after "
        "the harness app exists, or pass app_instance=NO_APP."
    )
    controller = ConsoleImageController(
        screen,
        app_instance=None if resolved_app is NO_APP else resolved_app,
        **{
            name: wired.get(name, _raiser(name, context))
            for name in IMAGE_CONTROLLER_CALLABLES
        },
    )
    screen._image = controller
    return controller
