# test_chat_screen_suspend.py
# Description: RED-first regression coverage for task-247 (duplicate Console save_state on screen suspend).
"""
Task-247: ``app.py`` (~4611-4624) already calls ``current_screen.save_state()``
explicitly and stores the returned state before switching screens away from
Console. ``ChatScreen.on_screen_suspend`` used to call ``self.save_state()``
a *second* time and discard the result -- a full O(sessions x messages)
native-console serialization wasted on every tab switch away from Console.

Note on Textual's dispatch: ``Screen`` itself defines a *private*
``_on_screen_suspend`` (adds/removes the suspended-screen CSS class, clears
mouse-over/tooltip state) which is a different attribute name and is
untouched by this fix -- it keeps firing via Textual's message dispatch
regardless of whether a *public* ``on_screen_suspend`` override exists.
Removing ``ChatScreen``'s override therefore only removes the redundant
``save_state()`` call, not screen-suspend behavior in general.
"""

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _bare_chat_screen():
    """A ChatScreen instance without running __init__.

    ``on_screen_suspend`` (pre-fix) only touches ``self.save_state()``, so a
    bare ``__new__()`` instance plus a recording stand-in for ``save_state``
    is enough to exercise it in isolation, without needing a live app/mount.
    """
    return ChatScreen.__new__(ChatScreen)


async def _run_suspend_with_stubs(screen) -> list[str]:
    """Drive the real on_screen_suspend inside a loop, all seams stubbed.

    Args:
        screen: A bare (``__new__``) ChatScreen with stub attributes.

    Returns:
        The ordered names of the seams the hook invoked.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    calls: list[str] = []

    class _Timer:
        def __init__(self, name: str) -> None:
            self._name = name

        def stop(self) -> None:
            calls.append(f"stop:{self._name}")

    screen.save_state = lambda: calls.append("save_state")
    screen._release_claimed_conversation_settings_return = lambda: calls.append(
        "release_claim"
    )

    async def _flush() -> None:
        calls.append("sidebar_flush")

    screen._flush_sidebar_state_now = _flush
    screen._message = SimpleNamespace(
        invalidate_console_speech_context=lambda: calls.append("speech_stop")
    )
    screen._console_auto_speak = SimpleNamespace(
        unmount=lambda: calls.append("auto_speak_unmount")
    )
    screen._stop_console_transcript_sync_timer = lambda: calls.append(
        "stop:sync"
    )
    screen._fleet = SimpleNamespace(
        _stop_console_fleet_survivor_tick=lambda: calls.append("stop:survivor")
    )
    screen._stop_console_cost_ttl_timer = lambda: calls.append("stop:cost_ttl")
    screen._console_draft_spend_refresh = SimpleNamespace(
        stop=lambda: calls.append("stop:draft_spend")
    )
    screen._hands_free = SimpleNamespace(
        teardown=lambda: calls.append("hands_free_teardown")
    )

    async def _async_teardown(name: str) -> None:
        calls.append(name)

    screen._realtime = SimpleNamespace(
        teardown=lambda: _async_teardown("realtime_teardown")
    )
    screen._dictation = SimpleNamespace(
        teardown=lambda: _async_teardown("dictation_teardown")
    )
    screen._console_resume_handoff_timers = [_Timer("handoff")]
    screen.__dict__.pop("_console_suspend_sidebar_flush", None)

    screen.on_screen_suspend()
    # Drain the tasks the hook scheduled (sidebar flush, audio teardowns).
    for task in [screen._console_suspend_sidebar_flush] + list(
        screen._console_suspend_flush_tasks
    ):
        await task
    assert Mock  # keep the import referenced for future stub growth
    return calls


def test_on_screen_suspend_does_not_call_save_state():
    """The task-247 regression, preserved through TASK-31520's rewrite.

    The hook exists again (the chat route is reusable, so suspend must
    quiesce per-visit work), but the original waste it was deleted for --
    a second O(sessions x messages) ``save_state()`` per tab switch --
    must never come back: app.py's navigation seam owns that call.
    """
    import asyncio

    screen = _bare_chat_screen()
    calls = asyncio.run(_run_suspend_with_stubs(screen))
    assert "save_state" not in calls, "on_screen_suspend must not save_state()"


def test_on_screen_suspend_quiesces_every_visit_seam():
    """Unit contract for the TASK-31520 hook, no Textual app involved.

    Every per-visit seam the 2026-09-04 audit dispositioned to suspend is
    invoked exactly once: the handoff-claim release, the serialized
    sidebar flush, TTS stop, auto-speak unmount, all four timer stops plus
    the tracked resume-handoff timers, and the three audio teardowns.
    """
    import asyncio

    screen = _bare_chat_screen()
    calls = asyncio.run(_run_suspend_with_stubs(screen))
    for expected in (
        "release_claim",
        "sidebar_flush",
        "speech_stop",
        "auto_speak_unmount",
        "stop:sync",
        "stop:survivor",
        "stop:cost_ttl",
        "stop:draft_spend",
        "stop:handoff",
        "hands_free_teardown",
        "realtime_teardown",
        "dictation_teardown",
    ):
        assert calls.count(expected) == 1, (
            f"{expected}: expected exactly one invocation, got "
            f"{calls.count(expected)} in {calls}"
        )
    assert screen._console_resume_handoff_timers == []
