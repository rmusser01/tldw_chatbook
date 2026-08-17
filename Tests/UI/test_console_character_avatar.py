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

import asyncio
import threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from PIL import Image as PILImage
from textual.app import ComposeResult
from textual.widgets import Static

import tldw_chatbook.UI.Console_Modules.session as session_module

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    VisualIdentityPublicationResult,
    VisualIdentityResolution,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.character import ConsoleCharacterController
from tldw_chatbook.UI.Console_Modules.session import (
    CharacterSessionPromptSeed,
    ConsoleSessionController,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.Console.console_image_viewer_modal import ClickableAvatarBox
from tldw_chatbook.Widgets.Console.console_reaction_picker_modal import (
    ConsoleReactionPickerModal,
    ReactionOption,
)


def _avatar_png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    PILImage.new("RGB", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


def _resolution(
    character_id: int,
    *,
    requested: str,
    manual: str | None,
    source: str,
    identity_suffix: str,
    image: bytes | None,
) -> VisualIdentityResolution:
    return VisualIdentityResolution(
        actor_kind="character",
        actor_id=str(character_id),
        requested_expression_key=requested,
        manual_expression_key=manual,
        resolved_expression_key=manual or requested,
        pack_id=1,
        pack_version_id=1,
        asset_id=1,
        expression_id=None,
        storage_source="builtin",
        storage_relpath=None,
        content_type="image/png" if image else None,
        is_animated=False,
        resolution_source=source,
        fallback_reason="none",
        cache_identity=(
            "visual-identity-v1",
            "actor_kind=character",
            f"actor_id={character_id}",
            f"requested={requested}",
            f"manual={manual or ''}",
            f"source={source}",
            identity_suffix,
        ),
        image_bytes=image,
    )


def _bare_console_screen(store: ConsoleChatStore) -> ChatScreen:
    """Build a native-console screen shell for direct accessor calls.

    See ``Tests/UI/test_console_native_chat_flow.py::_bare_console_screen``
    for the rationale (bypasses ``ChatScreen.__init__``).
    """
    screen = ChatScreen.__new__(ChatScreen)
    screen._character = ConsoleCharacterController.__new__(ConsoleCharacterController)
    screen._character._active_character_avatar = None
    screen._character._active_character_avatar_name = None
    screen._character._last_console_avatar_scope = None
    screen._character._console_expression_spec_cache = {}
    screen._console_chat_store = store
    screen._session = ConsoleSessionController.__new__(ConsoleSessionController)
    screen._session._chat_store_accessor = lambda: screen._console_chat_store
    screen._session._current_chat_store_accessor = lambda: screen._console_chat_store
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
    session = ConsoleChatSession(
        id="session-a",
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
        character_name="Ada",
    )
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
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
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

    from tldw_chatbook.Utils import mosaic_render

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mosaic_render, "mosaic_from_image", _boom)

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
        await _wait_for_selector(screen, pilot, "#console-rail-section-header-details")
        yield screen


def _set_chat_images_setting(app, key: str, value) -> None:
    """Set a `[chat.images]` value where the shipping app actually reads it.

    task-15270. These tests used to assign ``app.app_config["chat"]``
    wholesale, which only worked against the old three-key test config:
    `console_image_view._chat_images_config` prefers the RAW TOML nested
    under ``COMPREHENSIVE_CONFIG_RAW`` whenever the snapshot carries it --
    and every real `load_settings()` snapshot does -- falling back to the top
    level only when it does not. So the same edit in the shipping app would
    have been ignored, and these tests were pinning the fallback shape rather
    than the one a user has. Write both, raw nest first.
    """
    raw = app.app_config.setdefault("COMPREHENSIVE_CONFIG_RAW", {})
    raw.setdefault("chat", {}).setdefault("images", {})[key] = value
    app.app_config.setdefault("chat", {}).setdefault("images", {})[key] = value


@pytest_asyncio.fixture
async def console_screen_avatar_off():
    """Mounted Console screen with ``chat.images.show_character_avatar`` off."""
    app = _build_test_app()
    _set_chat_images_setting(app, "show_character_avatar", False)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-rail-section-header-details")
        yield screen


@pytest_asyncio.fixture
async def console_screen_generic():
    """Mounted Console screen, default config, generic (no-character) session."""
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-rail-section-header-details")
        yield screen


@pytest.mark.asyncio
async def test_character_section_composes_when_config_on(
    console_screen_with_character,
):
    screen = (
        console_screen_with_character  # config default -> show_character_avatar True
    )
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


@pytest.mark.asyncio
async def test_initial_and_recomposed_avatar_caption_treat_rich_tags_as_literal(
    console_screen_with_character,
):
    screen = console_screen_with_character
    raw_initial = "Nyx\n\t\x00[/broken]"
    screen._active_character_avatar_name = raw_initial

    await screen.recompose()
    initial = screen.query_one("#console-character-name", Static)
    initial_visual = initial.visual
    assert initial._render_markup is False
    assert initial_visual.plain == "Nyx ?[/broken]"
    assert "\n" not in initial_visual.plain
    assert "\t" not in initial_visual.plain

    raw_recomposed = "Lady\t[bold]Nyx[/bold]"
    screen._active_character_avatar_name = raw_recomposed
    await screen.recompose()
    recomposed = screen.query_one("#console-character-name", Static)
    assert recomposed._render_markup is False
    assert recomposed.visual.plain == "Lady [bold]Nyx[/bold]"
    assert screen._active_character_avatar_name == raw_recomposed


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
        await _wait_for_selector(screen, pilot, "#console-rail-section-header-details")
        yield app, screen, avatar_db


@pytest_asyncio.fixture
async def console_screen_with_db_avatar_off(avatar_db):
    """Mounted Console screen wired to a real DB, with the avatar rail
    section config-off (``chat.images.show_character_avatar = False``).
    """
    app = _build_test_app()
    app.chachanotes_db = avatar_db
    _set_chat_images_setting(app, "show_character_avatar", False)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-rail-section-header-details")
        yield app, screen, avatar_db


def _set_active_console_character(screen, character_id, character_name) -> None:
    """Bind the active native Console session to a character (or clear it)."""
    session = screen._session._active_native_console_session()
    assert session is not None, "no active native Console session"
    session.character_id = character_id
    session.character_name = character_name
    session.runtime_backend = "local"
    session.assistant_authority_id = None
    if type(character_id) is int and character_id > 0:
        session.assistant_kind = "character"
        session.assistant_id = str(character_id)
    else:
        session.assistant_kind = "generic"
        session.assistant_id = "console"


@pytest.mark.asyncio
async def test_refresh_populates_avatar_cache_and_mounts(console_screen_with_db):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar.get("character_id") == char_id
    assert (
        screen._active_character_avatar.get("pil") is not None
        or screen._active_character_avatar.get("pixels") is not None
    )

    # FIX B: prove the widget actually landed in the DOM (not just the
    # cached spec dict) right after the refresh awaits the mount.
    holder = screen.query_one("#console-character-avatar")
    mounted_ids = {child.id for child in holder.children}
    assert "console-character-avatar-image" in mounted_ids

    # unchanged scope -> no re-fetch (spy the DB fetch)
    calls = []
    orig = screen._fetch_character_card_for_avatar  # the off-thread fetch wrapper
    screen._fetch_character_card_for_avatar = lambda cid: (
        calls.append(cid),
        orig(cid),
    )[1]
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert calls == []  # scope guard short-circuits before any fetch


@pytest.mark.asyncio
async def test_refresh_clears_avatar_for_generic_session(console_screen_with_db):
    app, screen, db = console_screen_with_db
    _set_active_console_character(screen, None, None)
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is None


@pytest.mark.asyncio
async def test_refresh_never_raises_on_bad_image(console_screen_with_db):
    app, screen, db = console_screen_with_db
    char_id = db.add_character_card({"name": "Bad", "image": b"not-an-image"})
    _set_active_console_character(screen, char_id, "Bad")
    await (
        screen._character._refresh_active_character_avatar_if_scope_changed()
    )  # must not raise
    # decode failed -> empty/text spec, name still set
    assert screen._active_character_avatar_name == "Bad"


@pytest.mark.asyncio
async def test_refresh_never_raises_when_mount_fails(
    console_screen_with_db, monkeypatch
):
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

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    holder = screen.query_one("#console-character-avatar")

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(holder, "mount", _boom)

    await (
        screen._character._refresh_active_character_avatar_if_scope_changed()
    )  # must not raise


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

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    await (
        screen._sync_native_console_chat_ui()
    )  # the real sync entrypoint, not the refresh directly

    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar_name == "Ada"
    name = screen.query_one("#console-character-name")
    assert "Ada" in str(name.renderable)


@pytest.mark.asyncio
async def test_avatar_caption_projects_raw_character_name_to_one_line(
    console_screen_with_db,
):
    _app, screen, db = console_screen_with_db
    raw_name = "Nyx\n\tAdmin\x00[/bold]"
    char_id = db.add_character_card({"name": raw_name})
    _set_active_console_character(screen, char_id, raw_name)

    await screen._character._refresh_active_character_avatar_if_scope_changed()

    caption = screen.query_one("#console-character-name")
    assert str(caption.renderable) == "Nyx Admin?[/bold]"
    assert screen._active_character_avatar_name == raw_name
    assert screen._session._active_native_console_session().character_name == raw_name


# --- Whole-branch review fixes (P3c) -----------------------------------------


@pytest.mark.asyncio
async def test_refresh_skips_db_fetch_when_config_off(
    console_screen_with_db_avatar_off,
):
    """Whole-branch review, FIX 2: per the spec's Error-handling section,
    `_refresh_active_character_avatar_if_scope_changed` must early-return
    when `resolve_show_character_avatar(...)` is False -- the rail section
    isn't even composed in that case, so the off-thread DB fetch + PIL
    decode must not run at all.
    """
    app, screen, db = console_screen_with_db_avatar_off
    from PIL import Image as PILImage
    from io import BytesIO

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    calls = []
    orig = screen._fetch_character_card_for_avatar
    screen._fetch_character_card_for_avatar = lambda cid: (
        calls.append(cid),
        orig(cid),
    )[1]

    await screen._character._refresh_active_character_avatar_if_scope_changed()

    assert calls == []  # config-off short-circuits before any DB fetch
    assert screen._active_character_avatar is None


@pytest.mark.asyncio
async def test_refresh_repopulates_after_config_toggle_off_then_on(
    console_screen_with_db,
):
    """Qodo #782-3 regression: the config-off branch clears the cache AND must
    invalidate the scope guard, otherwise re-enabling the feature without
    changing the active character hits the scope-equality early-return and the
    Character section sticks in the empty state forever.
    """
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (10, 180, 60)).save(buf, format="PNG")
    char_id = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, char_id, "Ada")

    # (1) feature on (default): populates + records scope (char_id,)
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None

    # (2) toggle off: clears the cache AND invalidates the scope guard
    _set_chat_images_setting(app, "show_character_avatar", False)
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is None
    assert screen._last_console_avatar_scope is None  # guard invalidated

    # (3) toggle back on, SAME character: must repopulate (was stuck empty pre-fix)
    _set_chat_images_setting(app, "show_character_avatar", True)
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._active_character_avatar.get("character_id") == char_id


# --- P3d-1 Task 3: reactive avatar scope (character_id, state) --------------
#
# Widens the P3c `(character_id,)` scope guard to `(character_id, state)` so
# the avatar swaps as the character thinks/speaks/errors, and adds a
# per-state decode cache so revisiting a state already seen this session is
# served instantly.


@pytest.mark.asyncio
async def test_avatar_swaps_across_expression_states(
    console_screen_with_db, monkeypatch
):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO

    def _png(color):
        buf = BytesIO()
        PILImage.new("RGB", (32, 32), color).save(buf, format="PNG")
        return buf.getvalue()

    char_id = db.add_character_card({"name": "Ada", "image": _png((10, 10, 10))})
    db.set_character_expression_image(char_id, "speaking", _png((0, 200, 0)))
    _set_active_console_character(screen, char_id, "Ada")

    # Drive the derived state directly (the pure resolver is unit-tested separately);
    # here we assert the refresh reacts to the state it computes.
    import tldw_chatbook.UI.Console_Modules.character as cs

    state_box = {"v": "idle"}
    monkeypatch.setattr(
        cs, "resolve_console_expression_state", lambda *a, **k: state_box["v"]
    )

    state_box["v"] = "idle"
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._active_character_avatar is not None
    assert screen._last_console_avatar_scope[0][2] == str(char_id)
    assert screen._last_console_avatar_scope[1:] == ("idle", None)

    state_box["v"] = "speaking"
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope[1:] == ("speaking", None)

    # Revisiting a state is served from the per-state cache (no re-decode).
    assert (
        screen._active_character_avatar["resolution_cache_identity"]
        in screen._console_expression_spec_cache
    )


@pytest.mark.asyncio
async def test_expression_state_falls_back_to_idle_image(
    console_screen_with_db, monkeypatch
):
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (5, 5, 5)).save(buf, format="PNG")
    char_id = db.add_character_card(
        {"name": "Ada", "image": buf.getvalue()}
    )  # idle image only
    _set_active_console_character(screen, char_id, "Ada")
    import tldw_chatbook.UI.Console_Modules.character as cs

    monkeypatch.setattr(
        cs, "resolve_console_expression_state", lambda *a, **k: "thinking"
    )

    await (
        screen._character._refresh_active_character_avatar_if_scope_changed()
    )  # no thinking image -> idle image
    assert (
        screen._active_character_avatar is not None
    )  # rendered the idle fallback, did not crash
    assert screen._last_console_avatar_scope[1:] == ("thinking", None)


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

    import tldw_chatbook.UI.Console_Modules.character as cs

    state_box = {"v": "idle"}
    monkeypatch.setattr(
        cs, "resolve_console_expression_state", lambda *a, **k: state_box["v"]
    )

    for char_id in char_ids:
        _set_active_console_character(screen, char_id, f"Char{char_id}")
        for state in ("idle", "thinking", "speaking"):
            state_box["v"] = state
            await screen._character._refresh_active_character_avatar_if_scope_changed()

    assert len(screen._console_expression_spec_cache) <= 16


# --- P3d-1 Task 5: end-to-end integration + fail-soft ------------------------
#
# Regression guards locking the react-off gate and the corrupt-image
# fail-soft path all the way through the real refresh entrypoint (not the
# pure resolver, which is unit-tested separately).


@pytest.mark.asyncio
async def test_react_off_pins_idle_even_when_streaming(
    console_screen_with_db, monkeypatch
):
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
        b = BytesIO()
        PILImage.new("RGB", (16, 16), c).save(b, format="PNG")
        return b.getvalue()

    char_id = db.add_character_card({"name": "Ada", "image": _png((1, 1, 1))})
    db.set_character_expression_image(char_id, "speaking", _png((0, 255, 0)))
    _set_active_console_character(screen, char_id, "Ada")

    # Put a genuinely-streaming assistant message on the active session so
    # the raw status really would say "streaming" (-> "speaking") if react
    # were on.
    controller = screen._ensure_console_chat_controller()
    session = screen._session._active_native_console_session()
    message = controller.store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    controller.store.append_stream_chunk(message.id, "partial reply")

    _set_chat_images_setting(app, "react_character_expressions", False)
    # Even though the raw status is "streaming", react-off must pin idle.
    # (resolve_console_expression_state honors react_enabled=False internally.)
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope[1:] == ("idle", None)


@pytest.mark.asyncio
async def test_reactive_avatar_never_raises_on_corrupt_expression(
    console_screen_with_db, monkeypatch
):
    app, screen, db = console_screen_with_db
    char_id = db.add_character_card({"name": "Bad"})
    db.set_character_expression_image(char_id, "speaking", b"not-an-image")
    _set_active_console_character(screen, char_id, "Ada")
    import tldw_chatbook.UI.Console_Modules.character as cs

    monkeypatch.setattr(
        cs, "resolve_console_expression_state", lambda *a, **k: "speaking"
    )
    # Must not raise into the sync tick even though the image is corrupt.
    await screen._character._refresh_active_character_avatar_if_scope_changed()
    assert screen._last_console_avatar_scope[1:] == ("speaking", None)


@pytest.mark.parametrize(
    ("second_source", "second_identity_suffix"),
    (
        ("pack_operational", "pack_version_id=2|asset_id=9|sha256=bbb"),
        ("card_avatar", "pack_version_id=1|asset_id=1|sha256=aaa"),
    ),
)
@pytest.mark.asyncio
async def test_visual_identity_cache_uses_the_complete_resolution_identity(
    console_screen_with_db,
    monkeypatch,
    second_source,
    second_identity_suffix,
):
    """Version/asset/digest or source-only changes miss after invalidation."""

    _app, screen, db = console_screen_with_db
    character_id = db.add_character_card(
        {"name": "Samira", "image": _avatar_png((1, 1, 1))}
    )
    _set_active_console_character(screen, character_id, "Samira")
    identity = {"value": "pack_version_id=1|asset_id=1|sha256=aaa"}
    source = {"value": "pack_operational"}
    calls: list[str] = []

    def resolve(_scope, requested_state, manual_key):
        calls.append(identity["value"])
        return _resolution(
            character_id,
            requested="thinking",
            manual=manual_key,
            source=source["value"],
            identity_suffix=identity["value"],
            image=_avatar_png((10, len(calls), 20)),
        )

    monkeypatch.setattr(screen._session, "_resolve_visual_identity", resolve)
    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.character.resolve_console_expression_state",
        lambda *_args, **_kwargs: "thinking",
    )

    await screen._character._refresh_active_character_avatar_if_scope_changed()
    first_identity = screen._active_character_avatar["resolution_cache_identity"]
    assert first_identity in screen._console_expression_spec_cache
    assert (character_id, "thinking") not in screen._console_expression_spec_cache

    source["value"] = second_source
    identity["value"] = second_identity_suffix
    await screen._session.invalidate_visual_identity_actor("character", character_id)
    second_identity = screen._active_character_avatar["resolution_cache_identity"]

    assert second_identity != first_identity
    assert first_identity not in screen._console_expression_spec_cache
    assert second_identity in screen._console_expression_spec_cache
    assert len(calls) >= 2


@pytest.mark.asyncio
async def test_personas_publication_targets_mounted_console_cache_before_return(
    console_screen_with_db, monkeypatch
):
    _app, console, _db = console_screen_with_db
    invalidate = AsyncMock()
    monkeypatch.setattr(console._session, "invalidate_visual_identity_actor", invalidate)
    owner = SimpleNamespace(app=SimpleNamespace(screen_stack=(console,)))
    result = VisualIdentityPublicationResult(
        actor_kind="character",
        actor_id="42",
        old_pack_id=1,
        old_version_id=1,
        new_pack_id=2,
        new_version_id=2,
        version_directory=Path("unused"),
    )

    await PersonasScreen._invalidate_visual_identity_publication(owner, result)

    invalidate.assert_awaited_once_with("character", "42")


@pytest.mark.parametrize(
    "context_change", ("actor", "session", "manual", "state", "config", "store")
)
@pytest.mark.asyncio
async def test_decode_completion_live_fences_every_avatar_request_input(
    console_screen_with_db, monkeypatch, context_change
):
    """B completes before blocked A; no stale request may paint afterward."""

    app, screen, db = console_screen_with_db
    character_id = db.add_character_card(
        {"name": "Samira", "image": _avatar_png((1, 1, 1))}
    )
    _set_active_console_character(screen, character_id, "Samira")
    controller = screen._console_chat_controller
    original_store = controller.store
    state = {"value": "thinking"}

    def expression_state(store, _session_id, *, react_enabled):
        if not react_enabled:
            return "idle"
        if store is not original_store:
            return "speaking"
        return state["value"]

    def resolve(scope, requested_state, manual_key):
        actor_id = int(scope[2])
        return _resolution(
            actor_id,
            requested=requested_state,
            manual=manual_key,
            source="pack_manual" if manual_key else "pack_operational",
            identity_suffix=f"sha256={scope[0]}:{requested_state}:{manual_key or '-'}",
            image=_avatar_png((actor_id % 255, 20, 30)),
        )

    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.character.resolve_console_expression_state",
        expression_state,
    )
    monkeypatch.setattr(screen._session, "_resolve_visual_identity", resolve)
    _view, cache = screen._ensure_console_image_view()
    original_prepare = cache.prepare
    started = threading.Event()
    release = threading.Event()
    first = {"blocked": False}

    def prepare(cache_key, image_bytes):
        if not first["blocked"]:
            first["blocked"] = True
            started.set()
            assert release.wait(timeout=5)
        return original_prepare(cache_key, image_bytes)

    monkeypatch.setattr(cache, "prepare", prepare)
    painted: list[tuple[str, ...] | None] = []
    original_build = screen._build_character_avatar_widget

    def build(spec):
        painted.append(
            tuple(spec["resolution_cache_identity"]) if spec is not None else None
        )
        return original_build(spec)

    monkeypatch.setattr(screen, "_build_character_avatar_widget", build)

    stale = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed(force=True)
    )
    assert await asyncio.to_thread(started.wait, 5)
    actor_scope = screen._session._current_visual_identity_actor_scope()
    assert actor_scope is not None
    if context_change == "actor":
        replacement_id = db.add_character_card(
            {"name": "Mira", "image": _avatar_png((2, 2, 2))}
        )
        _set_active_console_character(screen, replacement_id, "Mira")
    elif context_change == "session":
        original_store.create_session(
            session_id="replacement-session",
            character_id=character_id,
            character_name="Samira",
            assistant_kind="character",
            assistant_id=str(character_id),
        )
    elif context_change == "manual":
        screen._session._set_manual_reaction(actor_scope, "custom:relief")
    elif context_change == "state":
        state["value"] = "speaking"
    elif context_change == "config":
        _set_chat_images_setting(app, "react_character_expressions", False)
    else:
        replacement_store = type(
            "ReplacementStore",
            (),
            {"active_session_id": original_store.active_session_id},
        )()
        controller.store = replacement_store

    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    current_identity = tuple(
        screen._active_character_avatar["resolution_cache_identity"]
    )
    current_paint = len(painted)
    release.set()
    await stale
    controller.store = original_store

    assert tuple(screen._active_character_avatar["resolution_cache_identity"]) == (
        current_identity
    )
    assert painted[current_paint:] == []


@pytest.mark.parametrize("blocked_await", ("remove", "mount"))
@pytest.mark.asyncio
async def test_render_awaits_never_resume_a_stale_avatar_paint(
    console_screen_with_db, monkeypatch, blocked_await
):
    """Once B paints, A may not mount or update labels after either await."""

    _app, screen, db = console_screen_with_db
    character_id = db.add_character_card(
        {"name": "Samira", "image": _avatar_png((1, 1, 1))}
    )
    _set_active_console_character(screen, character_id, "Samira")
    actor_scope = screen._session._current_visual_identity_actor_scope()
    assert actor_scope is not None

    def resolve(_scope, requested_state, manual_key):
        return _resolution(
            character_id,
            requested=requested_state,
            manual=manual_key,
            source="pack_manual" if manual_key else "pack_operational",
            identity_suffix=f"sha256={manual_key or 'automatic'}",
            image=_avatar_png((20 if manual_key else 10, 30, 40)),
        )

    monkeypatch.setattr(screen._session, "_resolve_visual_identity", resolve)
    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.character.resolve_console_expression_state",
        lambda *_args, **_kwargs: "thinking",
    )
    holder = screen.query_one("#console-character-avatar")
    started = asyncio.Event()
    release = asyncio.Event()
    first = {"blocked": False}
    original_remove = holder.remove_children
    original_mount = holder.mount

    async def remove_children():
        await original_remove()
        if blocked_await == "remove" and not first["blocked"]:
            first["blocked"] = True
            started.set()
            await release.wait()

    async def mount(*widgets, **kwargs):
        await original_mount(*widgets, **kwargs)
        if blocked_await == "mount" and not first["blocked"]:
            first["blocked"] = True
            started.set()
            await release.wait()

    monkeypatch.setattr(holder, "remove_children", remove_children)
    monkeypatch.setattr(holder, "mount", mount)
    builds: list[tuple[str, ...] | None] = []
    original_build = screen._build_character_avatar_widget

    def build(spec):
        builds.append(
            tuple(spec["resolution_cache_identity"]) if spec is not None else None
        )
        return original_build(spec)

    monkeypatch.setattr(screen, "_build_character_avatar_widget", build)
    name = screen.query_one("#console-character-name", Static)
    reaction = screen.query_one("#console-character-reaction-state", Static)
    original_name_update = name.update
    original_reaction_update = reaction.update
    updates: list[str] = []

    def update_name(value):
        updates.append("name")
        return original_name_update(value)

    def update_reaction(value):
        updates.append("reaction")
        return original_reaction_update(value)

    monkeypatch.setattr(name, "update", update_name)
    monkeypatch.setattr(reaction, "update", update_reaction)

    stale = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed(force=True)
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    screen._session._set_manual_reaction(actor_scope, "custom:relief")
    await screen._character._refresh_active_character_avatar_if_scope_changed(
        force=True
    )
    current_identity = tuple(
        screen._active_character_avatar["resolution_cache_identity"]
    )
    builds_after_current = len(builds)
    updates_after_current = len(updates)
    release.set()
    await stale

    assert tuple(screen._active_character_avatar["resolution_cache_identity"]) == (
        current_identity
    )
    assert builds[builds_after_current:] == []
    assert updates[updates_after_current:] == []


def test_visual_identity_expected_errors_fail_soft(monkeypatch):
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._visual_identity_db_accessor = object
    scope = ("session-a", "character", "7")

    def expected_error(*_args, **_kwargs):
        raise ValueError("invalid visual identity metadata")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.session.resolve_visual_identity",
        expected_error,
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.session.VisualIdentityRepository.get_active_actor_pack",
        expected_error,
    )

    assert controller._resolve_visual_identity(scope, "idle", None) is None
    assert controller._visual_identity_options(scope) == ()


def test_visual_identity_unexpected_programming_errors_propagate(monkeypatch):
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._visual_identity_db_accessor = object
    scope = ("session-a", "character", "7")

    def programming_error(*_args, **_kwargs):
        raise RuntimeError("programming defect")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.session.resolve_visual_identity",
        programming_error,
    )
    with pytest.raises(RuntimeError, match="programming defect"):
        controller._resolve_visual_identity(scope, "idle", None)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.session.VisualIdentityRepository.get_active_actor_pack",
        programming_error,
    )
    with pytest.raises(RuntimeError, match="programming defect"):
        controller._visual_identity_options(scope)


@pytest.mark.asyncio
async def test_late_avatar_load_never_overwrites_a_new_manual_reaction(
    console_screen_with_db, monkeypatch
):
    """Load A, publish B, finish B then A: A must never be applied."""

    _app, screen, db = console_screen_with_db
    character_id = db.add_character_card(
        {"name": "Samira", "image": _avatar_png((1, 1, 1))}
    )
    _set_active_console_character(screen, character_id, "Samira")
    actor_scope = screen._session._current_visual_identity_actor_scope()
    assert actor_scope is not None
    started = threading.Event()
    release = threading.Event()
    first = {"blocked": False}

    def resolve(_scope, requested_state, manual_key):
        if manual_key is None and not first["blocked"]:
            first["blocked"] = True
            started.set()
            assert release.wait(timeout=5)
        return _resolution(
            character_id,
            requested="thinking",
            manual=manual_key,
            source="pack_manual" if manual_key else "pack_operational",
            identity_suffix=f"sha256={manual_key or 'automatic'}",
            image=_avatar_png((200, 20 if manual_key else 10, 10)),
        )

    monkeypatch.setattr(screen._session, "_resolve_visual_identity", resolve)
    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.character.resolve_console_expression_state",
        lambda *_args, **_kwargs: "thinking",
    )

    stale = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed()
    )
    assert await asyncio.to_thread(started.wait, 5)
    screen._session._set_manual_reaction(actor_scope, "custom:relief")
    screen._last_console_avatar_scope = None
    current = asyncio.create_task(
        screen._character._refresh_active_character_avatar_if_scope_changed()
    )
    await current
    release.set()
    await stale

    assert screen._active_character_avatar["manual_expression_key"] == "custom:relief"
    assert (
        "manual=custom:relief"
        in screen._active_character_avatar["resolution_cache_identity"]
    )


@pytest.mark.asyncio
async def test_console_reaction_picker_selects_and_clears_session_override(
    console_screen_with_db, monkeypatch
):
    _app, screen, db = console_screen_with_db
    character_id = db.add_character_card(
        {"name": "Samira", "image": _avatar_png((20, 20, 20))}
    )
    _set_active_console_character(screen, character_id, "Samira")
    option = ReactionOption("custom:relief", "Relief", "image/webp", False)
    monkeypatch.setattr(
        session_module,
        "_visual_identity_options_for_db",
        lambda _db, _scope: (option,),
    )
    await screen.recompose()

    screen.query_one("#console-character-reaction-open").press()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if isinstance(screen.app.screen, ConsoleReactionPickerModal):
            break
    assert isinstance(screen.app.screen, ConsoleReactionPickerModal)
    first_modal = screen.app.screen
    screen.app.screen.query_one("#console-reaction-picker-select").press()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if screen.app.screen is screen:
            break

    actor_scope = screen._session._current_visual_identity_actor_scope()
    assert actor_scope is not None
    for _ in range(50):
        await asyncio.sleep(0.01)
        if screen._session._manual_reaction_key(actor_scope) == "custom:relief":
            break
    assert screen._session._manual_reaction_key(actor_scope) == "custom:relief"
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "Relief (manual)" in str(
            screen.query_one("#console-character-reaction-state").renderable
        ):
            break
    assert "Relief (manual)" in str(
        screen.query_one("#console-character-reaction-state").renderable
    )

    for _ in range(50):
        await asyncio.sleep(0.01)
        if screen.app.screen is screen:
            break
    assert screen.app.screen is screen
    screen.query_one("#console-character-reaction-open").press()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if (
            isinstance(screen.app.screen, ConsoleReactionPickerModal)
            and screen.app.screen is not first_modal
            and screen.app.screen.is_mounted
        ):
            break
    modal = screen.app.screen
    assert isinstance(modal, ConsoleReactionPickerModal)
    modal.query_one("#console-reaction-picker-clear").press()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if screen.app.screen is screen:
            break

    assert screen._session._manual_reaction_key(actor_scope) is None
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "Automatic" in str(
            screen.query_one("#console-character-reaction-state").renderable
        ):
            break
    assert "Automatic" in str(
        screen.query_one("#console-character-reaction-state").renderable
    )


@pytest.mark.asyncio
async def test_invalid_reaction_selection_preserves_prior_override_and_reports_error(
    console_screen_with_db, monkeypatch
):
    app, screen, db = console_screen_with_db
    character_id = db.add_character_card(
        {"name": "Samira", "image": _avatar_png((20, 20, 20))}
    )
    _set_active_console_character(screen, character_id, "Samira")
    actor_scope = screen._session._current_visual_identity_actor_scope()
    assert actor_scope is not None
    screen._session._set_manual_reaction(actor_scope, "custom:relief")
    monkeypatch.setattr(
        session_module,
        "_visual_identity_options_for_db",
        lambda _db, _scope: (
            ReactionOption("neutral", "Neutral", "image/webp", False),
        ),
    )
    notifications: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, severity=None, **_kwargs: notifications.append(
            (message, severity)
        ),
    )

    await screen._session._select_console_reaction(
        ReactionOption("custom:alarm", "Alarm", "image/webp", False)
    )

    assert screen._session._manual_reaction_key(actor_scope) == "custom:relief"
    assert notifications == [("That reaction is no longer available.", "error")]


@pytest.mark.asyncio
async def test_successful_session_close_clears_only_that_sessions_reactions(
    console_screen_with_db,
):
    _app, screen, _db = console_screen_with_db
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="Samira")
    scope = (session.id, "character", "7")
    survivor = ("other-session", "character", "7")
    screen._session._set_manual_reaction(scope, "custom:relief")
    screen._session._set_manual_reaction(survivor, "custom:alarm")

    await screen._session._close_console_session_tab(session.id)

    assert screen._session._manual_reaction_key(scope) is None
    assert screen._session._manual_reaction_key(survivor) == "custom:alarm"


@pytest.mark.asyncio
async def test_failed_session_close_preserves_reaction_override(
    console_screen_with_db, monkeypatch
):
    _app, screen, _db = console_screen_with_db
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="Samira")
    scope = (session.id, "character", "7")
    screen._session._set_manual_reaction(scope, "custom:relief")
    controller = screen._ensure_console_chat_controller()
    monkeypatch.setattr(
        controller,
        "close_session",
        lambda _session_id: (_ for _ in ()).throw(RuntimeError("close failed")),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await screen._session._close_console_session_tab(session.id)

    assert screen._session._manual_reaction_key(scope) == "custom:relief"


@pytest.mark.asyncio
async def test_successful_actor_replacement_clears_old_actor_override(
    console_screen_with_db, monkeypatch
):
    _app, screen, _db = console_screen_with_db
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="Old")
    object.__setattr__(session, "runtime_backend", "local")
    object.__setattr__(session, "assistant_kind", "character")
    object.__setattr__(session, "assistant_id", "7")
    object.__setattr__(session, "character_id", 7)
    old_scope = (session.id, "character", "7")
    screen._session._set_manual_reaction(old_scope, "custom:relief")
    monkeypatch.setattr(
        store,
        "swap_session_character_roleplay",
        lambda *_args, **_kwargs: (session, None, True),
    )

    assert screen._session._swap_console_session_character(
        store,
        8,
        CharacterSessionPromptSeed("New", "", "", "", ""),
        global_default="User",
    )

    assert screen._session._manual_reaction_key(old_scope) is None


# ---- task-1661: rail-derived avatar box + hugging holder ----


def test_avatar_box_scales_with_rail_width():
    """A wider rail yields a bigger box, clamped at both ends.

    task-1661: the box was hard-coded 16x8 regardless of rail width, so a
    ~50-column rail showed a 16-column thumb.
    """
    from tldw_chatbook.UI.Screens.chat_screen import (
        CHARACTER_AVATAR_COLS,
        CHARACTER_AVATAR_LINES,
        CHARACTER_AVATAR_MAX_COLS,
        CHARACTER_AVATAR_MAX_LINES,
        character_avatar_box,
    )

    # Unsettled layout (0) falls back to the historical minimum.
    assert character_avatar_box(0) == (CHARACTER_AVATAR_COLS, CHARACTER_AVATAR_LINES)
    # A real rail width scales up...
    cols, lines = character_avatar_box(48)
    assert cols > CHARACTER_AVATAR_COLS and lines > CHARACTER_AVATAR_LINES
    assert lines == round(cols / 2)
    # ...but never past the clamp.
    wide_cols, wide_lines = character_avatar_box(400)
    assert wide_cols == CHARACTER_AVATAR_MAX_COLS
    assert wide_lines <= CHARACTER_AVATAR_MAX_LINES


def test_mosaic_fallback_contains_rather_than_crops():
    """The non-graphics path must show the whole portrait (user choice).

    task-1661: the fallback baked with fit="cover", cropping the edges;
    graphics uses a contain fit, so the two paths disagreed on framing.
    """
    import inspect

    from tldw_chatbook.UI.Screens import chat_screen

    src = inspect.getsource(chat_screen._character_avatar_fallback_renderable)
    assert 'fit="contain"' in src
    assert 'fit="cover"' not in src


@pytest.mark.asyncio
async def test_avatar_holder_hugs_its_content():
    """The holder must not claim the whole rail section.

    task-1661: ClickableAvatarBox is a bare Container (width/height 1fr),
    so it expanded to fill the rail -- the portrait sat in the corner of a
    tall empty box with the character name pushed to the bottom.
    """
    import inspect

    # wave-1 console decomposition, task 3: the Character section's
    # `compose()` code (including the avatar holder) moved out of
    # `ChatScreen.compose_content` onto `ConsoleLeftRail`. Retargeted per
    # the screen-decomposition design's testing rule -- the source-location
    # assertion moves with the code, the width/height assertions stay
    # byte-for-byte.
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail

    src = inspect.getsource(ConsoleLeftRail.compose)
    holder = src.split("avatar_holder = ClickableAvatarBox", 1)[1][:900]
    assert 'avatar_holder.styles.width = "auto"' in holder
    assert 'avatar_holder.styles.height = "auto"' in holder


@pytest.mark.asyncio
async def test_available_cols_measures_the_section_not_the_holder(
    console_screen_with_db,
):
    """Width must come from the rail section body, never the holder.

    task-1661 regression: the holder is ``width: auto`` so it hugs the
    portrait; measuring IT to size the portrait is circular -- it fed the
    previous child's width back in (13 cols observed) and pinned the box
    at the 16-column minimum no matter how wide the rail was.
    """
    app, screen, db = console_screen_with_db
    from PIL import Image as PILImage
    from io import BytesIO

    buf = BytesIO()
    PILImage.new("RGB", (400, 500), (200, 10, 10)).save(buf, format="PNG")
    cid = db.add_character_card({"name": "Ada", "image": buf.getvalue()})
    _set_active_console_character(screen, cid, "Ada")
    await screen._character._refresh_active_character_avatar_if_scope_changed()

    body = screen.query_one("#console-rail-section-body-character")
    holder = screen.query_one("#console-character-avatar")
    measured = screen._character_avatar_available_cols()

    assert measured == body.content_size.width
    assert holder.content_size.width < body.content_size.width, (
        "holder should hug its content, so this test proves the two differ"
    )
    assert measured != holder.content_size.width


@pytest.mark.unit
def test_expanding_the_character_section_reallows_a_rail_width_avatar():
    """Re-opening the collapsed Character section must re-render the avatar.

    A user reported "no character image in the Console" and turned out to
    have the Character rail section COLLAPSED. Clicking it open is the fix
    -- but two mechanisms interact badly on that path:

    * a collapsed body has ``display: none``, so
      ``_character_avatar_available_cols()`` measures 0 and
      ``character_avatar_box(0)`` clamps to the 16-column MINIMUM;
    * ``_refresh_active_character_avatar_if_scope_changed`` early-returns
      while ``(character_id, state)`` is unchanged.

    So an avatar first rendered while collapsed stays pinned at 16 columns
    forever -- exactly the "~50-column rail showing a 16-column portrait"
    defect task-1661 fixed for a different trigger. Toggling the section
    open must invalidate the scope guard so the next sync re-measures.
    """
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen, character_avatar_box

    # The sizing half of the trap, in isolation: a hidden body measures 0.
    assert character_avatar_box(0) == (16, 8), "collapsed body clamps to minimum"
    assert character_avatar_box(48)[0] > 16, "an open rail should size larger"

    screen = ChatScreen.__new__(ChatScreen)
    screen._character = ConsoleCharacterController.__new__(ConsoleCharacterController)
    screen._last_console_avatar_scope = (7, "idle")
    applied: list[tuple[str, bool]] = []
    screen._set_console_rail_preference = lambda **kw: applied.append(("pref", True))

    # wave-1 console decomposition, task 3: section-open DOM sync moved onto
    # `ConsoleLeftRail` (`apply_section_open`), so `_toggle_console_rail_
    # section` now reaches it via `self.query_one("#console-left-rail",
    # ConsoleLeftRail)` instead of calling a same-class private method.
    # Retargeted per the screen-decomposition design's testing rule: a test
    # that reaches into a moved private method gets its plumbing retargeted,
    # the assertion stays byte-for-byte.
    class _FakeLeftRail:
        def apply_section_open(self, section_id, section_open):
            applied.append((section_id, section_open))

    def _fake_query_one(selector, expect_type=None):
        # Final review finding 5: a wildcard fake here would still pass even
        # if `_toggle_console_rail_section` queried the wrong id or type --
        # assert the selector/type match the real rail's before handing back
        # the fake, so this test still fails if that call site regresses.
        assert selector == "#console-left-rail", selector
        assert expect_type is ConsoleLeftRail, expect_type
        return _FakeLeftRail()

    screen.query_one = _fake_query_one
    screen._current_console_rail_state = lambda: type(
        "S", (), {"character_open": False}
    )()

    screen._toggle_console_rail_section("character")

    assert ("character", True) in applied, "section did not open"
    assert screen._last_console_avatar_scope is None, (
        "opening the Character section must clear the avatar scope guard, or "
        "the portrait stays at the collapsed 16-column size forever"
    )


# --- task-3793: explicit sizing regression pins -----------------------------
#
# The rail's avatar holder is width/height auto (task-1661); a default-width
# (100%) Static inside an auto container resolves to 0x0 under Textual 8.x,
# so the avatar -- and even the no-character placeholder -- mounted but
# painted nothing. That was the Console's "fully broken" character portrait.
# The builder now sets explicit width/height from the baked renderable's
# grid. These mount the builder's REAL output into the production holder
# shape and assert the painted region is non-zero.


class _AvatarHolderApp(ConsolidatedCSSApp):
    """Host mirroring the rail's auto/auto avatar holder (task-1661 shape)."""

    def __init__(self, screen: ChatScreen, spec: dict | None):
        super().__init__()
        self._avatar_screen = screen
        self._avatar_spec = spec

    def compose(self) -> ComposeResult:
        holder = ClickableAvatarBox(id="console-character-avatar")
        holder.styles.width = "auto"
        holder.styles.height = "auto"
        with holder:
            # Built HERE, inside the active app context: the pixels fallback
            # reads `self.app.no_color` for its monochrome guard, which needs
            # a running app (a bare unit call degrades to the placeholder).
            yield self._avatar_screen._build_character_avatar_widget(self._avatar_spec)


@pytest.mark.asyncio
async def test_pixels_avatar_paints_nonzero_region_in_auto_holder():
    """task-3793 regression: the pixels avatar must paint a non-zero region.

    Mounts the real builder output in an auto/auto holder replicating
    ``ClickableAvatarBox`` and asserts the painted region is non-zero with
    the explicit cell size clamped to the avatar box; before the fix the
    default-width Static collapsed to 0x0 and painted nothing.
    """
    from PIL import Image as PILImage

    from tldw_chatbook.UI.Screens.chat_screen import character_avatar_box

    screen = _bare_console_screen(ConsoleChatStore())
    spec = {
        "character_id": 7,
        "name": "Ada",
        "mode": "pixels",  # skip the graphics branch, exercise the mosaic
        "pil": PILImage.new("RGB", (64, 96), (10, 180, 200)),
        "pixels": None,
    }
    app = _AvatarHolderApp(screen, spec)
    async with app.run_test(size=(60, 30)):
        widget = app.query_one("#console-character-avatar-image", Static)
        # Before task-3793 this region was 0x0: mounted, painted nothing.
        assert widget.region.width > 0
        assert widget.region.height > 0
        # Explicit cell size from the mosaic grid, clamped to the avatar box
        # (the fallback box: no rail section body exists in this host).
        box_cols, box_lines = character_avatar_box(0)
        assert 1 <= widget.styles.width.value <= box_cols
        assert 1 <= widget.styles.height.value <= box_lines


@pytest.mark.asyncio
async def test_avatar_placeholder_paints_nonzero_region_in_auto_holder():
    """task-3793 regression: the no-character placeholder must stay visible.

    Same auto/auto holder as the pixels case: the placeholder Static needs
    ``width auto`` so it cannot collapse to 0x0 under Textual 8.
    """
    screen = _bare_console_screen(ConsoleChatStore())
    app = _AvatarHolderApp(screen, None)
    async with app.run_test(size=(60, 30)):
        widget = app.query_one("#console-character-avatar-empty", Static)
        assert str(widget.renderable) == "No character in this chat"
        # Same 0x0 collapse hit the placeholder; width auto is the guard.
        assert widget.region.width > 0
        assert widget.region.height > 0
