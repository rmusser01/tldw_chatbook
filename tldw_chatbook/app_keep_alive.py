"""What the handler-error keep-alive does with the pump that died (TASK-33621.13).

TASK-32533 made ``TldwCli._handle_exception`` (``app_lifecycle.py``) swallow an
exception raised inside one pump's own handler -- a widget's or a screen's --
instead of exiting the app. Textual still ends that pump's message loop right
after the call and detaches it from the DOM. GAP4-01 (Console UX review
2026-09-29) found the two ways that left a live process behind a dead UI:

* The pump was a SCREEN (the Conversation Inspector, whose handler awaited
  ``push_screen_wait`` outside a worker). It stayed on the screen stack, with
  the folder picker its handler had just pushed on top. Once the picker
  closed, every key, click and resize went to a screen that no longer
  processed anything, and Ctrl+Q was ignored too: ``Screen._binding_chain``
  is built from the dead screen's detached focused widget, whose ancestors no
  longer reach the App.
* The pump was a WIDGET that held focus. The live screen's ``focused`` still
  named the detached widget, so keys were forwarded to it and dropped, and
  the binding chain again stopped short of the App.

:func:`retire_dead_pump` removes the dead screen and every screen above it,
or moves focus off the dead widget's subtree. When neither is possible -- the
dead screen is the content screen itself, with only Textual's blank
placeholder beneath it -- it returns ``None`` and the caller takes Textual's
loud exit: a crash is better than a frozen app that cannot even quit.
"""

from __future__ import annotations

from typing import Any, Literal

from textual.screen import Screen
from textual.widget import Widget

#: ``App.get_default_screen()``'s id. Textual's blank default screen sits at
#: the bottom of every stack; ``TldwCli`` pushes its routed content screen on
#: top of it (``_navigation_outgoing_screen`` documents the layout), so it is
#: never a screen to hand the user back to.
_TEXTUAL_PLACEHOLDER_SCREEN_ID = "_default"

DeadPumpKind = Literal["screen", "widget"]


def retire_dead_pump(app: Any, pump: Any) -> DeadPumpKind | None:
    """Leave no dead pump in charge of input once the keep-alive returns.

    Called synchronously from ``_handle_exception`` while ``pump`` is still the
    active message pump; Textual breaks its loop (and detaches it) right
    after that call returns.

    Args:
        app: The running ``TldwCli``.
        pump: The widget or screen whose own handler raised.

    Returns:
        ``"screen"`` when a dead screen was taken off the stack, ``"widget"``
        when the app can stay up around a dead widget, or ``None`` when no
        live screen would be left in charge -- the caller must then exit.
    """
    try:
        if isinstance(pump, Screen):
            return "screen" if _discard_dead_screen(app, pump) else None
        if isinstance(pump, Widget):
            _move_focus_off_dead_widget(app, pump)
        return "widget"
    except Exception:  # noqa: BLE001 -- a recovery that fails takes the loud exit
        return None


def keep_alive_notice(
    site: tuple[str, str, int | None], pump: Any, raised: BaseException, kind: str
) -> str:
    """The toast for a kept-alive handler error. Never carries the message.

    Args:
        site: ``(module, function, line)`` of the innermost Chatbook frame.
        pump: The pump whose handler raised, named when no Chatbook frame is.
        raised: The exception; only its class name is ever used.
        kind: What :func:`retire_dead_pump` did.

    Returns:
        User-facing copy naming where it failed and what happened to it.
    """
    if site[0].startswith("tldw_chatbook."):
        where = site[1]
    elif pump is not None:
        pump_id = getattr(pump, "id", None)
        where = type(pump).__name__ + (f"#{pump_id}" if pump_id else "")
    else:
        where = site[1] or type(raised).__name__
    if kind == "screen":
        return (
            f"Something went wrong in {where} — that view was closed so the "
            "app keeps responding; details are in the log file."
        )
    return (
        f"Something went wrong in {where} — the screen was kept open. "
        "That panel may stop responding or disappear until you "
        "reopen it; details are in the log file."
    )


def _is_live(screen: Screen) -> bool:
    return bool(screen.is_running and not screen._closing and not screen._closed)


def _discard_dead_screen(app: Any, screen: Screen) -> bool:
    """Pop ``screen`` and everything pushed above it; False if that is unsafe.

    Popping only the modal the dead screen had pushed is not enough (the dead
    screen is then on top), and neither is a priority Ctrl+Q binding (the
    chain is built from a detached widget). Each popped screen's pending
    result is resolved with ``None`` first -- the value Esc and a bare
    ``dismiss()`` already deliver -- so a worker awaiting one of them through
    ``push_screen_wait`` resumes instead of hanging forever
    (``_dismiss_navigation_overlays`` in ``app_navigation.py`` has the same
    reasoning for navigation).
    """
    stack = app._screen_stack
    if screen not in stack:
        # Already off the current stack: nothing dead is in charge here --
        # unless another mode's stack still holds it, which we cannot fix.
        return not any(screen in other for other in app._screen_stacks.values())
    index = stack.index(screen)
    floor = 2 if stack and stack[0].id == _TEXTUAL_PLACEHOLDER_SCREEN_ID else 1
    if index < floor or not _is_live(stack[index - 1]):
        return False
    for doomed in reversed(stack[index:]):
        _release_pending_result(doomed)
        app.pop_screen()
    return True


def _release_pending_result(screen: Screen) -> None:
    callbacks = getattr(screen, "_result_callbacks", None)
    if not callbacks:
        return
    callback = callbacks[-1]
    future = getattr(callback, "future", None)
    if future is not None and future.done():
        return
    callback(None)


def _move_focus_off_dead_widget(app: Any, widget: Widget) -> None:
    """Refocus a live widget if focus sits in the dying widget's subtree."""
    try:
        screen = widget.screen
    except Exception:  # noqa: BLE001 -- not on a screen: nothing to refocus
        return
    doomed = list(widget.walk_children(with_self=True))
    doomed_ids = {id(node) for node in doomed}
    focused = screen.focused
    if focused is not None and id(focused) in doomed_ids:
        # The same call Textual's own `App._prune` makes for a removed subtree.
        screen._reset_focus(focused, doomed)
    captured = getattr(app, "mouse_captured", None)
    if captured is not None and id(captured) in doomed_ids:
        app.capture_mouse(None)
