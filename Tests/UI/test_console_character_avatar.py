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
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
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


def test_build_character_avatar_widget_pixels_failure_falls_back_to_text(monkeypatch):
    """FIX A: `_build_character_avatar_widget` must NEVER raise, even when the
    ``rich_pixels`` build fails. It is reached from
    ``_render_character_avatar_into_section``, which runs outside
    ``_refresh_active_character_avatar_if_scope_changed``'s try/except -- and
    that refresh itself must never raise into the 0.2s Console sync poll. A
    decode/build failure here must degrade to the same text placeholder as
    the no-image case, not propagate.
    """
    from PIL import Image as PILImage
    import rich_pixels

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rich_pixels.Pixels, "from_image", staticmethod(_boom))

    screen = _bare_console_screen(ConsoleChatStore())
    spec = {
        "character_id": 7,
        "name": "Ada",
        "mode": "pixels",  # skip the graphics branch, hit the pixels fallback
        "pil": PILImage.new("RGB", (32, 32), (200, 10, 10)),
        "pixels": None,
    }

    widget = screen._build_character_avatar_widget(spec)  # must not raise

    assert isinstance(widget, Static)
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


# --- P3c Task 3: avatar cache + scope-guarded off-thread refresh + render ---
#
# Real screen + real ``CharactersRAGDB``: only a real DB round-trip proves
# `_refresh_active_character_avatar_if_scope_changed` genuinely decodes the
# stored character-card image bytes into the cache (a fake DB can't catch a
# broken `get_character_card_by_id(...)["image"]` read, and a fake cache
# can't catch a broken `ConsoleImageRenderCache.prepare` call).


@pytest.fixture
def avatar_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "console_character_avatar.db", "test-client")
    yield db
    db.close_connection()


@pytest_asyncio.fixture
async def console_screen_with_db(avatar_db):
    """Mounted Console screen wired to a real ``CharactersRAGDB``."""
    app = _build_test_app()
    app.chachanotes_db = avatar_db
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(
            screen, pilot, "#console-rail-section-header-details"
        )
        yield app, screen, avatar_db


@pytest_asyncio.fixture
async def console_screen_with_db_avatar_off(avatar_db):
    """Mounted Console screen wired to a real DB, with the avatar rail
    section config-off (``chat.images.show_character_avatar = False``).
    """
    app = _build_test_app()
    app.chachanotes_db = avatar_db
    app.app_config["chat"] = {"images": {"show_character_avatar": False}}
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(
            screen, pilot, "#console-rail-section-header-details"
        )
        yield app, screen, avatar_db


def _set_active_console_character(screen, character_id, character_name) -> None:
    """Bind the active native Console session to a character (or clear it)."""
    session = screen._active_native_console_session()
    assert session is not None, "no active native Console session"
    session.character_id = character_id
    session.character_name = character_name


@pytest.mark.asyncio
async def test_refresh_populates_avatar_cache_and_mounts(console_screen_with_db):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar.get("character_id") == char_id
    assert screen._active_character_avatar.get("pil") is not None or \
           screen._active_character_avatar.get("pixels") is not None

    # FIX B: prove the widget actually landed in the DOM (not just the
    # cached spec dict) right after the refresh awaits the mount.
    holder = screen.query_one("#console-character-avatar")
    mounted_ids = {child.id for child in holder.children}
    assert "console-character-avatar-image" in mounted_ids

    # unchanged scope -> no re-fetch (spy the DB fetch)
    calls = []
    orig = screen._fetch_character_card_for_avatar   # the off-thread fetch wrapper
    screen._fetch_character_card_for_avatar = lambda cid: (calls.append(cid), orig(cid))[1]
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert calls == []   # scope guard short-circuits before any fetch


@pytest.mark.asyncio
async def test_refresh_clears_avatar_for_generic_session(console_screen_with_db):
    app, screen, db = console_screen_with_db
    _set_active_console_character(screen, None, None)
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is None


@pytest.mark.asyncio
async def test_refresh_never_raises_on_bad_image(console_screen_with_db):
    app, screen, db = console_screen_with_db
    char_id = db.add_character_card({"name": "Bad", "image": b"not-an-image"})
    _set_active_console_character(screen, char_id, "Bad")
    await screen._refresh_active_character_avatar_if_scope_changed()  # must not raise
    # decode failed -> empty/text spec, name still set
    assert screen._active_character_avatar_name == "Bad"


@pytest.mark.asyncio
async def test_refresh_never_raises_when_mount_fails(console_screen_with_db, monkeypatch):
    """Whole-branch review, FIX 1: `_render_character_avatar_into_section`'s
    ``holder.mount(...)`` runs outside `_refresh_active_character_avatar_if_
    scope_changed`'s own try/except, at two call sites, and that refresh runs
    unconditionally on every 0.2s Console sync tick -- some worker dispatch
    sites run with ``exit_on_error=True``, so an escaping mount failure (e.g.
    a transient layout race on a session-switch/resume tick) could crash the
    app. The refresh must never raise even when the mount itself blows up.
    """
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    holder = screen.query_one("#console-character-avatar")

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(holder, "mount", _boom)

    await screen._refresh_active_character_avatar_if_scope_changed()  # must not raise


# --- P3c Task 4: wire the refresh into the Console sync tick -----------------
#
# Unlike `test_refresh_populates_avatar_cache_and_mounts` above (which calls
# `_refresh_active_character_avatar_if_scope_changed` directly), this proves
# the wire: the real Console sync entrypoint `_sync_native_console_chat_ui`
# -- the same tick that already refreshes the dictionary/world-book "what's
# in play" summaries -- also refreshes the character avatar.


@pytest.mark.asyncio
async def test_sync_tick_refreshes_avatar(console_screen_with_db):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    await screen._sync_native_console_chat_ui()  # the real sync entrypoint, not the refresh directly

    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar_name == "Ada"
    name = screen.query_one("#console-character-name")
    assert "Ada" in str(name.renderable)


# --- Whole-branch review fixes (P3c) -----------------------------------------


@pytest.mark.asyncio
async def test_refresh_skips_db_fetch_when_config_off(console_screen_with_db_avatar_off):
    """Whole-branch review, FIX 2: per the spec's Error-handling section,
    `_refresh_active_character_avatar_if_scope_changed` must early-return
    when `resolve_show_character_avatar(...)` is False -- the rail section
    isn't even composed in that case, so the off-thread DB fetch + PIL
    decode must not run at all.
    """
    app, screen, db = console_screen_with_db_avatar_off
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    calls = []
    orig = screen._fetch_character_card_for_avatar
    screen._fetch_character_card_for_avatar = lambda cid: (calls.append(cid), orig(cid))[1]

    await screen._refresh_active_character_avatar_if_scope_changed()

    assert calls == []  # config-off short-circuits before any DB fetch
    assert screen._active_character_avatar is None


@pytest.mark.asyncio
async def test_refresh_repopulates_after_config_toggle_off_then_on(console_screen_with_db):
    """Qodo #782-3 regression: the config-off branch clears the cache AND must
    invalidate the scope guard, otherwise re-enabling the feature without
    changing the active character hits the scope-equality early-return and the
    Character section sticks in the empty state forever.
    """
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (10, 180, 60)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    # (1) feature on (default): populates + records scope (char_id,)
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None

    # (2) toggle off: clears the cache AND invalidates the scope guard
    app.app_config["chat"] = {"images": {"show_character_avatar": False}}
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is None
    assert screen._last_console_avatar_scope is None  # guard invalidated

    # (3) toggle back on, SAME character: must repopulate (was stuck empty pre-fix)
    app.app_config["chat"] = {"images": {"show_character_avatar": True}}
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar.get("character_id") == char_id


# --- P3d-1 Task 3: reactive avatar scope (character_id, state) --------------
#
# Widens the P3c `(character_id,)` scope guard to `(character_id, state)` so
# the avatar swaps as the character thinks/speaks/errors, and adds a
# per-state decode cache so revisiting a state already seen this session is
# served instantly.


@pytest.mark.asyncio
async def test_avatar_swaps_across_expression_states(console_screen_with_db, monkeypatch):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    def _png(color):
        buf = BytesIO(); PILImage.new("RGB", (32, 32), color).save(buf, format="PNG"); return buf.getvalue()
    char_id = db.add_character_card({"name": "Ada", "image": _png((10, 10, 10))})
    db.set_character_expression_image(char_id, "speaking", _png((0, 200, 0)))
    _set_active_console_character(screen, char_id, "Ada")

    # Drive the derived state directly (the pure resolver is unit-tested separately);
    # here we assert the refresh reacts to the state it computes.
    import tldw_chatbook.UI.Screens.chat_screen as cs
    state_box = {"v": "idle"}
    monkeypatch.setattr(cs, "resolve_console_expression_state", lambda *a, **k: state_box["v"])

    state_box["v"] = "idle"
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._last_console_avatar_scope == (char_id, "idle")

    state_box["v"] = "speaking"
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope == (char_id, "speaking")

    # Revisiting a state is served from the per-state cache (no re-decode).
    assert (char_id, "speaking") in screen._console_expression_spec_cache


@pytest.mark.asyncio
async def test_expression_state_falls_back_to_idle_image(console_screen_with_db, monkeypatch):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    buf = BytesIO(); PILImage.new("RGB", (32, 32), (5, 5, 5)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})   # idle image only
    _set_active_console_character(screen, char_id, "Ada")
    import tldw_chatbook.UI.Screens.chat_screen as cs
    monkeypatch.setattr(cs, "resolve_console_expression_state", lambda *a, **k: "thinking")

    await screen._refresh_active_character_avatar_if_scope_changed()   # no thinking image -> idle image
    assert screen._active_character_avatar is not None   # rendered the idle fallback, did not crash
    assert screen._last_console_avatar_scope == (char_id, "thinking")


# --- P3d-1 Task 3 review fixes ------------------------------------------------


@pytest.mark.asyncio
async def test_expression_spec_cache_is_bounded(console_screen_with_db, monkeypatch):
    """Review FIX 1: ``_console_expression_spec_cache`` is written on every
    new ``(character_id, state)`` decode and never evicted -- over a long
    session visiting many characters this retains unbounded ``PILImage.Image``
    references (the ``_console_image_cache`` render LRU does NOT protect this
    cache, since the spec dicts hold their own independent PIL references).
    Visit 6 characters x 3 states each (18 distinct scopes, more than the
    16-entry cap) and assert the cache never grows past the cap.
    """
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO

    def _png(color):
        buf = BytesIO()
        PILImage.new("RGB", (32, 32), color).save(buf, format="PNG")
        return buf.getvalue()

    char_ids = [
        db.add_character_card({"name": f"Char{i}", "image": _png((i * 10, 10, 10))})
        for i in range(6)
    ]

    import tldw_chatbook.UI.Screens.chat_screen as cs
    state_box = {"v": "idle"}
    monkeypatch.setattr(cs, "resolve_console_expression_state", lambda *a, **k: state_box["v"])

    for char_id in char_ids:
        _set_active_console_character(screen, char_id, f"Char{char_id}")
        for state in ("idle", "thinking", "speaking"):
            state_box["v"] = state
            await screen._refresh_active_character_avatar_if_scope_changed()

    assert len(screen._console_expression_spec_cache) <= 16


# --- P3d-1 Task 5: end-to-end integration + fail-soft ------------------------
#
# Regression guards locking the react-off gate and the corrupt-image
# fail-soft path all the way through the real refresh entrypoint (not the
# pure resolver, which is unit-tested separately).


@pytest.mark.asyncio
async def test_react_off_pins_idle_even_when_streaming(console_screen_with_db, monkeypatch):
    """A genuinely "streaming" session (real assistant message, real
    ``store``, no ``resolve_console_expression_state`` monkeypatch -- unlike
    the P3d-1 Task 3 tests above) still resolves to "idle" once
    ``react_character_expressions`` is off, proving the config gate is wired
    all the way through the real refresh entrypoint, not just the pure
    resolver (already locked at the unit level by
    ``Tests/Chat/test_console_expression_state.py::test_react_disabled_pins_idle``).
    """
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    def _png(c):
        b = BytesIO(); PILImage.new("RGB", (16, 16), c).save(b, format="PNG"); return b.getvalue()
    char_id = db.add_character_card({"name": "Ada", "image": _png((1, 1, 1))})
    db.set_character_expression_image(char_id, "speaking", _png((0, 255, 0)))
    _set_active_console_character(screen, char_id, "Ada")

    # Put a genuinely-streaming assistant message on the active session so
    # the raw status really would say "streaming" (-> "speaking") if react
    # were on.
    controller = screen._ensure_console_chat_controller()
    session = screen._active_native_console_session()
    message = controller.store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    controller.store.append_stream_chunk(message.id, "partial reply")

    app.app_config["chat"] = {"images": {"react_character_expressions": False}}
    # Even though the raw status is "streaming", react-off must pin idle.
    # (resolve_console_expression_state honors react_enabled=False internally.)
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope == (char_id, "idle")


@pytest.mark.asyncio
async def test_reactive_avatar_never_raises_on_corrupt_expression(console_screen_with_db, monkeypatch):
    app, screen, db = console_screen_with_db
    char_id = db.add_character_card({"name": "Bad"})
    db.set_character_expression_image(char_id, "speaking", b"not-an-image")
    _set_active_console_character(screen, char_id, "Ada")
    import tldw_chatbook.UI.Screens.chat_screen as cs
    monkeypatch.setattr(cs, "resolve_console_expression_state", lambda *a, **k: "speaking")
    # Must not raise into the sync tick even though the image is corrupt.
    await screen._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope == (char_id, "speaking")
