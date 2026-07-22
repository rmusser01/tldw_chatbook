"""P3c Task 1: active-character rail accessors on ``ChatScreen``.

Resolves the active character ONLY off the live native Console session
(``_active_native_console_session().character_id`` / ``.character_name`` --
#754 sets these at Start-Chat, on DB-resume, and on screen-state restore);
never from the legacy ``app.current_chat_*`` reactives (the documented
Console<->Library split-brain -- see
``Tests/UI/test_console_dictionaries_attach.py``).

Uses the same ``_bare_console_screen`` pattern as
``Tests/UI/test_console_native_chat_flow.py``: builds a native-console
screen shell directly (bypassing ``ChatScreen.__init__``, which requires a
mounted Textual app), so these are plain, fast unit-level checks rather than
a full pilot-driven screen.
"""

import pytest
import pytest_asyncio

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


def _bare_console_screen(store: ConsoleChatStore) -> ChatScreen:
    """Build a native-console screen shell for direct accessor calls.

    See ``Tests/UI/test_console_native_chat_flow.py::_bare_console_screen``
    for the rationale (bypasses ``ChatScreen.__init__``).
    """
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_chat_store = store
    screen._console_visible_draft_session_id = None
    screen._console_composer_or_none = lambda: None
    return screen


def _store_with_session(session: ConsoleChatSession) -> ConsoleChatStore:
    store = ConsoleChatStore()
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    return store


def test_current_console_rail_character_id_reads_active_session():
    session = ConsoleChatSession(id="session-a", character_id=7, character_name="Ada")
    screen = _bare_console_screen(_store_with_session(session))

    assert screen._current_console_rail_character_id() == 7
    assert screen._current_console_rail_character_name() == "Ada"


def test_current_console_rail_character_id_none_for_generic_session():
    session = ConsoleChatSession(id="session-a")
    screen = _bare_console_screen(_store_with_session(session))

    assert screen._current_console_rail_character_id() is None
    assert screen._current_console_rail_character_name() is None


def test_p3c_leaves_dictionary_scope_ids_unchanged():
    """Pin: P3c must NOT make ``_active_console_dictionary_scope_ids``
    character-aware -- that would change the dictionary/world-book "what's
    in play" content it feeds. ``character_id`` stays ``None`` there even
    for a character-bound native session.
    """
    session = ConsoleChatSession(
        id="session-a",
        character_id=7,
        character_name="Ada",
        persisted_conversation_id="conv-1",
    )
    screen = _bare_console_screen(_store_with_session(session))

    conversation_id, character_id = screen._active_console_dictionary_scope_ids()
    assert conversation_id == "conv-1"
    assert character_id is None


# --- P3c Task 2: config-gated "Character" rail section ----------------------
#
# `_build_character_avatar_widget` is a T2 stub -- it only renders the
# text/empty state from a cache spec; T3 extends it with real image
# rendering. The rail section itself composes reading only
# `self._active_character_avatar` / `_active_character_avatar_name`, which
# T2 always seeds empty in `__init__` -- T3 wires the real active-character
# lookup that fills them. So the "with character" fixture below only needs
# to exercise the default-config (`show_character_avatar` True) path, not an
# actually character-bound session.


def test_build_character_avatar_widget_empty_state_no_spec():
    screen = _bare_console_screen(ConsoleChatStore())
    widget = screen._build_character_avatar_widget(None)
    assert str(widget.renderable) == "No character in this chat"


def test_build_character_avatar_widget_spec_without_image():
    screen = _bare_console_screen(ConsoleChatStore())
    widget = screen._build_character_avatar_widget(
        {"character_id": 7, "name": "Ada", "pil": None, "pixels": None}
    )
    assert str(widget.renderable) == "no avatar"


@pytest_asyncio.fixture
async def console_screen_with_character():
    """Mounted Console screen under the default config (avatar rail on)."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(
            screen, pilot, "#console-rail-section-header-details"
        )
        yield screen


@pytest_asyncio.fixture
async def console_screen_avatar_off():
    """Mounted Console screen with ``chat.images.show_character_avatar`` off."""
    app = _build_test_app()
    app.app_config["chat"] = {"images": {"show_character_avatar": False}}
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(
            screen, pilot, "#console-rail-section-header-details"
        )
        yield screen


@pytest_asyncio.fixture
async def console_screen_generic():
    """Mounted Console screen, default config, generic (no-character) session."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(
            screen, pilot, "#console-rail-section-header-details"
        )
        yield screen


@pytest.mark.asyncio
async def test_character_section_composes_when_config_on(
    console_screen_with_character,
):
    screen = console_screen_with_character  # config default -> show_character_avatar True
    assert screen.query("#console-rail-section-body-character")  # section present
    assert screen.query("#console-character-name")


@pytest.mark.asyncio
async def test_character_section_absent_when_config_off(console_screen_avatar_off):
    # console_screen_avatar_off: app_config has chat.images.show_character_avatar = False
    screen = console_screen_avatar_off
    assert not screen.query("#console-rail-section-body-character")


@pytest.mark.asyncio
async def test_character_section_empty_state_for_generic_session(
    console_screen_generic,
):
    screen = console_screen_generic
    name = screen.query_one("#console-character-name")
    assert "No character" in str(name.renderable)  # empty-state copy
