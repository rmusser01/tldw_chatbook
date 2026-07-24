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


def test_on_screen_suspend_does_not_call_save_state():
    """Whatever fires for screen-suspend on a ChatScreen instance must never
    invoke save_state() -- app.py already did that before the screen switch.
    """
    screen = _bare_chat_screen()
    calls = []
    screen.save_state = lambda: calls.append(1)

    # Post-fix, ChatScreen no longer defines a public on_screen_suspend at
    # all (Screen doesn't define one either -- only the differently-named
    # _on_screen_suspend), so resolve it the same way Python attribute
    # lookup / Textual's dispatch would: absent means "nothing to call".
    handler = getattr(screen, "on_screen_suspend", None)
    if handler is not None:
        handler()

    assert calls == [], "on_screen_suspend must not invoke save_state()"


def test_chat_screen_no_longer_overrides_on_screen_suspend():
    """Pin the actual fix shape: the override is removed outright (not just
    made a no-op), so it doesn't shadow a future base-class implementation
    and doesn't leave dead code behind."""
    assert "on_screen_suspend" not in ChatScreen.__dict__
