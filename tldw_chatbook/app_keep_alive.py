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

Not every kept-alive error kills its pump. :func:`pump_loop_ended` reads the
traceback to tell the two apart, so a failed ``call_next`` or
``call_after_refresh`` callback -- which leaves the pump running -- never
closes a live screen or exits the app.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from loguru import logger
from textual.screen import Screen
from textual.widget import Widget

#: ``App.get_default_screen()``'s id. Textual's blank default screen sits at
#: the bottom of every stack; ``TldwCli`` pushes its routed content screen on
#: top of it (``_navigation_outgoing_screen`` documents the layout), so it is
#: never a screen to hand the user back to.
_TEXTUAL_PLACEHOLDER_SCREEN_ID = "_default"

#: Textual's own module, as ``_exception_frames`` in ``app_lifecycle`` names it.
_PUMP_MODULE = "textual.message_pump"

DeadPumpKind = Literal["screen", "widget", "alive"]


def pump_loop_ended(frames: Sequence[tuple[str, str, int | None]]) -> bool:
    """Whether the error that reached ``_handle_exception`` ended its pump.

    ``frames`` is the traceback, outermost first, so ``frames[0]`` is the
    frame that caught the error. In Textual 8.2.8 (``message_pump.py``) only
    two catches end the pump's message loop: ``_pre_process`` (mount never
    finished) and ``_process_messages_loop`` after a failed
    ``_dispatch_message``, which ``break``s. A failed ``call_next`` or
    ``call_after_refresh`` callback (caught in ``_flush_next_callbacks``) or
    ``on_idle`` handler (caught in the loop, but around ``invoke``) only
    leaves an inner loop, and the pump keeps running.
    ``test_pump_loop_ended_matches_what_textual_does_to_the_pump`` pins this
    against the installed Textual.

    Args:
        frames: ``(module, function, line)`` per traceback frame.

    Returns:
        True only for the two loop-ending catches; False for anything else.
    """
    if not frames or frames[0][0] != _PUMP_MODULE:
        return False
    if frames[0][1] == "_pre_process":
        return True
    return (
        frames[0][1] == "_process_messages_loop"
        and len(frames) > 1
        and frames[1][:2] == (_PUMP_MODULE, "_dispatch_message")
    )


def retire_dead_pump(
    app: Any, pump: Any, frames: Sequence[tuple[str, str, int | None]]
) -> DeadPumpKind | None:
    """Leave no dead pump in charge of input once the keep-alive returns.

    Called synchronously from ``_handle_exception`` while ``pump`` is still the
    active message pump; when the error ended its loop (:func:`pump_loop_ended`)
    Textual breaks the loop (and detaches the pump) right after that call
    returns.

    Args:
        app: The running ``TldwCli``.
        pump: The widget or screen whose own handler raised.
        frames: The error's traceback frames, outermost first.

    Returns:
        ``"alive"`` when the error did not end the pump (nothing to retire),
        ``"screen"`` when a dead screen was taken off the stack (or out of the
        reusable-screen cache), ``"widget"`` when the app can stay up around a
        dead widget, or ``None`` when no live screen would be left in charge
        -- the caller must then exit. A recovery that itself raises also
        returns ``None``, after a warning naming its class and site.
    """
    if not pump_loop_ended(frames):
        return "alive"
    try:
        if isinstance(pump, Screen):
            return "screen" if _discard_dead_screen(app, pump) else None
        if isinstance(pump, Widget):
            _move_focus_off_dead_widget(app, pump)
        return "widget"
    except Exception as exc:  # noqa: BLE001 -- a recovery that fails takes the loud exit
        # Without this line the exit is indistinguishable from "nothing live
        # left": only the original error reaches the diagnostics log.
        logger.warning(
            "Dead pump recovery failed: {} at {}", type(exc).__name__, _raise_site(exc)
        )
        return None


def _raise_site(error: BaseException) -> str:
    """``module.qualname:line`` of the frame that raised ``error`` and, when
    that frame is outside the package, of the innermost Chatbook frame too.

    Identifiers only, never the message or a file path -- the shape
    ``_handle_exception`` records for the original error (TASK-32533).
    """
    raised = chatbook = None
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        raised = (
            f"{frame.f_globals.get('__name__', '?')}."
            f"{frame.f_code.co_qualname}:{tb.tb_lineno}"
        )
        if raised.startswith("tldw_chatbook."):
            chatbook = raised
        tb = tb.tb_next
    if raised is None:
        return "unknown"
    return raised if chatbook in (None, raised) else f"{raised} via {chatbook}"


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
    if kind == "alive":
        return (
            f"Something went wrong in {where} — the app kept running; "
            "details are in the log file."
        )
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
    reasoning for navigation). A dead screen already off the stack (a
    suspended reusable screen) is only forgotten, never popped.
    """
    stack = app._screen_stack
    if screen not in stack:
        # Already off the current stack (a suspended screen): nothing dead is
        # in charge now -- unless another mode's stack still holds it, which
        # we cannot fix -- but it must not come back on the next visit.
        if any(screen in other for other in app._screen_stacks.values()):
            return False
        _forget_dead_screen(app, screen)
        return True
    index = stack.index(screen)
    floor = 2 if stack and stack[0].id == _TEXTUAL_PLACEHOLDER_SCREEN_ID else 1
    if index < floor or not _is_live(stack[index - 1]):
        return False
    for doomed in reversed(stack[index:]):
        _release_pending_result(doomed)
        app.pop_screen()
    _forget_dead_screen(app, screen)
    return True


def _forget_dead_screen(app: Any, screen: Screen) -> None:
    """Make sure no later navigation hands the dead screen back.

    Reusable routes (``ScreenRoute.reusable``: Home, Console, Library) keep
    one INSTALLED instance in ``app._reusable_screen_instances`` that survives
    navigation and keeps processing messages while suspended
    (``_reusable_navigation_screen`` in ``app_navigation.py``). If that
    instance dies, returning to the route would restart the dead instance with
    focus still on a detached widget: the GAP4-01 wedge, Ctrl+Q ignored. Drop
    it from the cache and uninstall it, so the next visit builds a fresh one.
    Only called once ``screen`` is on no stack (``uninstall_screen`` raises
    otherwise, and that raise takes the loud exit).
    """
    cache = getattr(app, "_reusable_screen_instances", None) or {}
    for route, (_identity, cached) in list(cache.items()):
        if cached is screen:
            cache.pop(route, None)
    if app.is_screen_installed(screen):
        app.uninstall_screen(screen)


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
