"""Roleplay P3d-1 Task 4: character expression authoring slots
(thinking/speaking/error) in the character editor.

Mirrors ``test_personas_character_editor_avatar.py``'s screen-level harness
(real ``CharactersRAGDB``, a character seeded via ``add_character_card``, its
id fed back into the stubbed ``ccp_character_handler`` module functions, then
driven through the real ``PersonasScreen``) - wrapped in a
``pytest_asyncio.fixture`` (the ``console_screen_with_db``-style pattern in
``test_console_character_avatar.py``) so each test starts with the editor
already open for a saved character, rather than repeating the mount/select/
open-editor boilerplate per test.
"""

import asyncio
from dataclasses import replace
import gc
from io import BytesIO
from pathlib import Path
import sqlite3
from threading import Event, Lock
import weakref

import pytest
import pytest_asyncio
from PIL import Image
from textual.app import ComposeResult
from textual.pilot import Pilot
from textual.screen import Screen

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Input, Static, TextArea

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Persona_Modules.personas_preview_coordinator import (
    PersonasPreviewCoordinator,
    get_personas_preview_coordinator,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Utils.paths import get_user_data_dir
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_visual_identity_pack_widget import (
    PersonasVisualIdentityPackWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterExpressionUploadRequested,
    EditCharacterRequested,
)

from Tests.UI.test_personas_dictionaries import PersonasTestApp, patch_character_paging


@pytest.fixture
def expr_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "personas_char_expression.db", "test-client")
    yield db
    db.close_connection()


async def _select_character(pilot, char_id):
    await pilot.pause()
    await pilot.click(f"#personas-library-row-character-{char_id}")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen


async def _open_editor_for(pilot, screen, char_id):
    screen.post_message(EditCharacterRequested(str(char_id)))
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


@pytest_asyncio.fixture
async def personas_editor_with_saved_character(mock_app_instance, monkeypatch, expr_db):
    """Mounted ``PersonasScreen`` with the character editor open for a saved
    character, wired to a real file-backed ``CharactersRAGDB``."""
    mock_app_instance.chachanotes_db = expr_db
    mock_app_instance.chat_dictionary_scope_service = None
    char_id = expr_db.add_character_card({"name": "Expressive"})

    record = {
        "id": char_id,
        "name": "Expressive",
        "description": "",
        "first_message": "Hi.",
        "version": 1,
    }
    monkeypatch.setattr(
        character_handler_module, "fetch_all_characters", lambda: [dict(record)]
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: (
            dict(record) if str(character_id) == str(char_id) else None
        ),
    )
    patch_character_paging(monkeypatch)

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _select_character(pilot, char_id)
        await _open_editor_for(pilot, screen, char_id)
        yield app, screen, expr_db, char_id


def _pack_asset(label: str, index: int) -> dict:
    from tldw_chatbook.Character_Chat.visual_identity import SAMIRA_EXPRESSION_KEYS

    return {
        "expression_key": SAMIRA_EXPRESSION_KEYS[label],
        "original_expression_key": label,
        "display_label": label.title(),
        "source_filename": f"{label}.webp",
        "storage_relpath": f"characters/samira/expressions/{label}.webp",
        "content_type": "image/webp",
        "bytes": 10 + index,
        "sha256": f"fixture-{index}",
        "width": 1024,
        "height": 1024,
        "source_context": {"fixture": True},
        "is_animated": False,
        "frame_count": 1,
    }


def _visual_identity_load_snapshot(screen, editor, character_id):
    return personas_screen_module._CharacterVisualIdentityLoadSnapshot(
        editor_ref=weakref.ref(editor),
        db=getattr(screen.app_instance, "chachanotes_db", None),
        character_id=character_id,
        screen_generation=screen._character_editor_generation,
        editor_session_token=editor.visual_identity_session_token,
    )


def _assert_visual_identity_unavailable(editor) -> None:
    host = editor.query_one("#personas-char-editor-visual-identity-host")
    assert host.display
    assert not editor.query_one("#personas-char-editor-legacy-expressions").display
    assert not editor.query(PersonasVisualIdentityPackWidget)
    assert any(
        isinstance(child, Static) and str(child.renderable) == "Unavailable"
        for child in host.children
    )


@pytest_asyncio.fixture
async def personas_editor_with_bound_pack(mock_app_instance, monkeypatch, expr_db):
    """Saved character whose repository graph has an active 31-asset pack."""

    from tldw_chatbook.Character_Chat.visual_identity import SAMIRA_REACTION_LABELS

    mock_app_instance.chachanotes_db = expr_db
    mock_app_instance.chat_dictionary_scope_service = None
    char_id = expr_db.add_character_card({"name": "Packed"})
    VisualIdentityRepository(expr_db).activate_pack(
        pack={
            "title": "Samira Reactions",
            "default_expression_key": "neutral",
            "source_kind": "builtin",
            "source_context": {"source_id": "fixture.pack"},
        },
        manifest={"schema_id": "fixture/v1"},
        assets=[
            _pack_asset(label, index)
            for index, label in enumerate(SAMIRA_REACTION_LABELS, start=1)
        ],
        actor_kind="character",
        actor_id=char_id,
    )
    record = {
        "id": char_id,
        "name": "Packed",
        "description": "",
        "first_message": "Hi.",
        "version": 1,
    }
    monkeypatch.setattr(
        character_handler_module, "fetch_all_characters", lambda: [dict(record)]
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: (
            dict(record) if str(character_id) == str(char_id) else None
        ),
    )
    patch_character_paging(monkeypatch)

    def _legacy_read_forbidden(*_args, **_kwargs):
        raise AssertionError("bound pack attempted a legacy expression-image read")

    monkeypatch.setattr(
        expr_db, "get_character_expression_image", _legacy_read_forbidden
    )

    preview_calls: list[str] = []

    def _resolve_preview(_db, **kwargs):
        from tldw_chatbook.Character_Chat.visual_identity import (
            SAMIRA_EXPRESSION_KEYS,
            VisualIdentityResolution,
        )

        key = kwargs["manual_expression_key"]
        preview_calls.append(key)
        asset_id = list(SAMIRA_EXPRESSION_KEYS.values()).index(key) + 1
        buf = BytesIO()
        Image.new("RGB", (8, 8), (asset_id, 20, 30)).save(buf, format="PNG")
        return VisualIdentityResolution(
            actor_kind="character",
            actor_id=str(char_id),
            requested_expression_key="neutral",
            manual_expression_key=key,
            resolved_expression_key=key,
            pack_id=1,
            pack_version_id=1,
            asset_id=asset_id,
            expression_id=None,
            storage_source="builtin",
            storage_relpath="redacted",
            content_type="image/webp",
            is_animated=False,
            resolution_source="pack_manual",
            fallback_reason="none",
            cache_identity=("fixture", key),
            image_bytes=buf.getvalue(),
        )

    monkeypatch.setattr(
        personas_screen_module, "resolve_visual_identity", _resolve_preview
    )

    app = PersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _select_character(pilot, char_id)
        await _open_editor_for(pilot, screen, char_id)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        yield app, screen, expr_db, char_id, preview_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", (False, True))
async def test_discard_keeps_saved_pack_and_pending_load_across_cached_return(
    personas_editor_with_bound_pack, monkeypatch, pending
):
    app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    pilot = Pilot(app)
    editor = screen.query_one(PersonasCharacterEditorWidget)
    browser = editor.query_one(PersonasVisualIdentityPackWidget)
    pack = browser.pack
    started = Event()
    release = Event()
    original_read = VisualIdentityRepository.get_active_actor_pack

    def delayed_read(repository, actor_kind, actor_id):
        started.set()
        assert release.wait(10)
        return original_read(repository, actor_kind, actor_id)

    try:
        if pending:
            monkeypatch.setattr(
                VisualIdentityRepository, "get_active_actor_pack", delayed_read
            )
            screen.post_message(EditCharacterRequested(str(char_id)))
            assert await asyncio.to_thread(started.wait, 2)
        token = editor.visual_identity_session_token
        before = editor.get_character_data()
        editor.query_one("#personas-char-editor-name", Input).value = "Discard me"
        editor.query_one("#personas-char-editor-description", TextArea).text = "Draft"
        await pilot.pause()
        assert screen.state.has_unsaved_changes
        decision = screen.run_worker(screen.confirm_navigation())
        await pilot.pause()
        await pilot.click("#roleplay-draft-discard-continue")
        assert await decision.wait() is True
        await app.push_screen(Screen())
        await app.pop_screen()
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.screen is screen
        assert editor.get_character_data() == before
        assert not screen.state.has_unsaved_changes
        assert editor.query_one("#personas-char-editor-visual-identity-host").display
        assert not editor.query_one("#personas-char-editor-legacy-expressions").display
        current_browser = editor.query_one(PersonasVisualIdentityPackWidget)
        assert current_browser.pack == pack
        if not pending:
            assert current_browser is browser
        assert editor.visual_identity_session_token == token
        assert (
            VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
            is not None
        )
    finally:
        release.set()
        await app.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_expression_slots_present_for_saved_character(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    for state in ("thinking", "speaking", "error"):
        assert screen.query_one(f"#char-expression-slot-{state}") is not None


@pytest.mark.asyncio
async def test_unbound_character_keeps_legacy_controls_and_mounts_no_pack_browser(
    personas_editor_with_saved_character,
):
    _app, screen, _db, _char_id = personas_editor_with_saved_character
    assert not screen.query(PersonasVisualIdentityPackWidget)
    assert screen.query_one("#personas-char-editor-legacy-expressions").display
    for state in ("thinking", "speaking", "error"):
        slot = screen.query_one(f"#char-expression-slot-{state}")
        assert slot.display
        assert screen.query_one(f"#personas-char-editor-expr-{state}-upload", Button)
        assert screen.query_one(f"#personas-char-editor-expr-{state}-generate", Button)
        assert screen.query_one(f"#personas-char-editor-expr-{state}-clear", Button)


@pytest.mark.asyncio
async def test_bound_character_mounts_pack_browser_and_hides_legacy_controls(
    personas_editor_with_bound_pack,
):
    _app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    assert browser.pack is not None
    assert len(browser.pack.assets) == 31
    assert browser.pack.source_kind == "builtin"
    assert not screen.query_one("#personas-char-editor-legacy-expressions").display
    for state in ("thinking", "speaking", "error"):
        # The legacy subtree remains byte-for-behavior and is hidden at its
        # owning wrapper; children keep their own display value for a later
        # unbound character session.
        assert screen.query_one(f"#char-expression-slot-{state}").display


@pytest.mark.asyncio
async def test_new_character_clears_bound_pack_and_restores_unsaved_legacy_state(
    personas_editor_with_bound_pack,
):
    _app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    assert editor.query_one("#personas-char-editor-visual-identity-host").display

    editor.new_character()

    assert editor.query_one("#personas-char-editor-legacy-expressions").display
    assert not editor.query_one("#personas-char-editor-visual-identity-host").display
    assert editor.query_one(
        "#personas-char-editor-expr-thinking-upload", Button
    ).disabled


@pytest.mark.asyncio
async def test_late_discard_of_detached_pack_cannot_replace_new_character_legacy_state(
    personas_editor_with_bound_pack,
):
    _app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    old_browser = editor.query_one(PersonasVisualIdentityPackWidget)

    editor.new_character()
    host = editor.query_one("#personas-char-editor-visual-identity-host")
    await host.remove_children()
    assert old_browser.parent is None

    await editor.discard_visual_identity_pack(old_browser)

    assert editor.query_one("#personas-char-editor-legacy-expressions").display
    assert not host.display
    assert not host.children
    assert editor.query_one(
        "#personas-char-editor-expr-thinking-upload", Button
    ).disabled


@pytest.mark.asyncio
async def test_bound_pack_decodes_only_the_selected_lazy_preview(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    await app.workers.wait_for_complete()
    assert preview_calls == ["custom:admiration", "custom:admiration"]
    assert (
        len(browser.query_one("#personas-visual-identity-preview-image").children) == 1
    )

    def _path_read_forbidden(*_args, **_kwargs):
        raise AssertionError("lazy preview attempted a direct path read")

    monkeypatch.setattr(Path, "open", _path_read_forbidden)
    monkeypatch.setattr(Path, "read_bytes", _path_read_forbidden)

    browser.apply_filter("joy")
    await asyncio.sleep(0.1)
    await app.workers.wait_for_complete()
    assert preview_calls == [
        "custom:admiration",
        "custom:admiration",
        "happy",
        "happy",
    ]
    assert (
        len(browser.query_one("#personas-visual-identity-preview-image").children) == 1
    )


@pytest.mark.asyncio
async def test_failed_selected_preview_replaces_old_pixels_with_unavailable(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    holder = browser.query_one("#personas-visual-identity-preview-image")
    original_resolve = personas_screen_module.resolve_visual_identity
    started = Event()
    release = Event()

    def fail_joy(*args, **kwargs):
        if kwargs["manual_expression_key"] == "happy":
            started.set()
            assert release.wait(2)
            raise ValueError("corrupt selected asset")
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(personas_screen_module, "resolve_visual_identity", fail_joy)
    browser.apply_filter("joy")
    assert await asyncio.to_thread(started.wait, 2)
    try:
        await asyncio.sleep(0)
        assert (
            str(browser.query_one("#personas-visual-identity-label", Static).renderable)
            == "Joy"
        )
        assert len(holder.children) == 1
        assert str(holder.children[0].renderable) == "Loading…"
    finally:
        release.set()
    await app.workers.wait_for_complete()

    assert len(holder.children) == 1
    assert str(holder.children[0].renderable) == "Unavailable"


@pytest.mark.asyncio
async def test_failed_stale_preview_cannot_clear_newer_selection(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    holder = browser.query_one("#personas-visual-identity-preview-image")
    original_resolve = personas_screen_module.resolve_visual_identity
    started = Event()
    release = Event()

    def fail_joy(*args, **kwargs):
        if kwargs["manual_expression_key"] == "happy":
            started.set()
            assert release.wait(2)
            raise ValueError("corrupt stale asset")
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(personas_screen_module, "resolve_visual_identity", fail_joy)
    browser.apply_filter("joy")
    assert await asyncio.to_thread(started.wait, 2)
    browser.apply_filter("fear")
    release.set()
    await app.workers.wait_for_complete()

    assert (
        str(browser.query_one("#personas-visual-identity-label", Static).renderable)
        == "Fear"
    )
    assert len(holder.children) == 1
    assert str(holder.children[0].renderable) != "Unavailable"


@pytest.mark.asyncio
async def test_rapid_preview_selection_never_overlaps_sync_resolution(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    original_resolve = personas_screen_module.resolve_visual_identity
    guard = Lock()
    first_started = Event()
    release = Event()
    overlap_seen = Event()
    active = 0
    peak = 0

    def blocked_resolve(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            if active > 1:
                overlap_seen.set()
        first_started.set()
        try:
            assert release.wait(2)
            return original_resolve(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(
        personas_screen_module, "resolve_visual_identity", blocked_resolve
    )

    browser.apply_filter("joy")
    assert await asyncio.to_thread(first_started.wait, 2)
    browser.apply_filter("fear")
    overlapped = await asyncio.to_thread(overlap_seen.wait, 0.5)
    release.set()
    await asyncio.sleep(0.5)

    assert not overlapped
    assert peak == 1


@pytest.mark.asyncio
async def test_fresh_personas_screen_reentry_never_overlaps_sync_resolution(
    personas_editor_with_bound_pack, monkeypatch
):
    app, first_screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    first_browser = first_screen.query_one(PersonasVisualIdentityPackWidget)
    second_screen = PersonasScreen(app)
    await app.push_screen(second_screen)
    for _ in range(100):
        await asyncio.sleep(0.01)
        if second_screen.query("#personas-library-rows"):
            break
    else:
        pytest.fail("second PersonasScreen shell never finished mounting")
    await second_screen._select_character(str(char_id), "Packed")
    await second_screen._handle_edit_requested(EditCharacterRequested(str(char_id)))
    await app.workers.wait_for_complete()
    second_browser = second_screen.query_one(PersonasVisualIdentityPackWidget)
    original_resolve = personas_screen_module.resolve_visual_identity
    guard = Lock()
    first_started = Event()
    release = Event()
    overlap_seen = Event()
    active = 0
    peak = 0

    def blocked_resolve(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            if active > 1:
                overlap_seen.set()
        first_started.set()
        try:
            assert release.wait(3)
            return original_resolve(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(
        personas_screen_module, "resolve_visual_identity", blocked_resolve
    )

    first_browser.apply_filter("joy")
    assert await asyncio.to_thread(first_started.wait, 2)
    second_browser.apply_filter("fear")
    overlapped = await asyncio.to_thread(overlap_seen.wait, 1.5)
    release.set()
    await asyncio.sleep(0.5)

    assert not overlapped
    assert peak == 1


@pytest.mark.asyncio
async def test_distinct_apps_have_independent_preview_coordinators():
    class AppOwner:
        pass

    first_app = AppOwner()
    second_app = AppOwner()
    first = get_personas_preview_coordinator(first_app)
    second = get_personas_preview_coordinator(second_app)
    assert first is get_personas_preview_coordinator(first_app)
    assert first is not second

    guard = Lock()
    both_active = Event()
    release = Event()
    active = 0

    def blocked_stage():
        nonlocal active
        with guard:
            active += 1
            if active == 2:
                both_active.set()
        try:
            assert release.wait(2)
        finally:
            with guard:
                active -= 1

    async def run(coordinator):
        async with coordinator.serialize():
            await coordinator.run_sync(blocked_stage)

    tasks = [asyncio.create_task(run(first)), asyncio.create_task(run(second))]
    assert await asyncio.to_thread(both_active.wait, 1)
    release.set()
    await asyncio.gather(*tasks)


def test_preview_coordinator_rebinds_after_drained_sequential_event_loops():
    coordinator = PersonasPreviewCoordinator()
    calls: list[str] = []

    async def run_once(value):
        async with coordinator.serialize():
            await coordinator.run_sync(calls.append, value)

    asyncio.run(run_once("first"))
    asyncio.run(run_once("second"))

    assert calls == ["first", "second"]


def test_preview_coordinator_does_not_retain_its_app_owner():
    class AppOwner:
        pass

    owner = AppOwner()
    owner_ref = weakref.ref(owner)
    get_personas_preview_coordinator(owner)
    del owner
    gc.collect()

    assert owner_ref() is None


@pytest.mark.asyncio
async def test_late_preview_cannot_paint_a_newer_editor_session(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    original_resolve = personas_screen_module.resolve_visual_identity
    started = Event()
    release = Event()

    def delayed_resolve(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original_resolve(*args, **kwargs)

    applied: list[str] = []
    monkeypatch.setattr(
        personas_screen_module, "resolve_visual_identity", delayed_resolve
    )
    monkeypatch.setattr(
        PersonasVisualIdentityPackWidget,
        "set_preview",
        lambda self, _renderable, *, asset_id: applied.append(str(asset_id)),
    )

    browser.apply_filter("joy")
    assert await asyncio.to_thread(started.wait, 2)
    screen._character_editor_generation += 1
    release.set()
    await app.workers.wait_for_complete()

    assert applied == []


@pytest.mark.asyncio
async def test_preview_cannot_paint_after_character_editor_mode_exit(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    original_resolve = personas_screen_module.resolve_visual_identity
    started = Event()
    release = Event()

    def delayed_resolve(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original_resolve(*args, **kwargs)

    applied: list[str] = []
    monkeypatch.setattr(
        personas_screen_module, "resolve_visual_identity", delayed_resolve
    )
    monkeypatch.setattr(
        PersonasVisualIdentityPackWidget,
        "set_preview",
        lambda self, _renderable, *, asset_id: applied.append(str(asset_id)),
    )

    browser.apply_filter("joy")
    assert await asyncio.to_thread(started.wait, 2)
    screen.state.active_mode = "personas"
    screen._show_center("#ccp-persona-card-view")
    release.set()
    await app.workers.wait_for_complete()

    assert applied == []


@pytest.mark.asyncio
async def test_preview_cannot_paint_after_same_character_editor_session_reload(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    original_resolve = personas_screen_module.resolve_visual_identity
    started = Event()
    release = Event()

    def delayed_resolve(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original_resolve(*args, **kwargs)

    applied: list[str] = []
    monkeypatch.setattr(
        personas_screen_module, "resolve_visual_identity", delayed_resolve
    )
    monkeypatch.setattr(editor, "_reset_visual_identity_browser", lambda: None)
    monkeypatch.setattr(
        PersonasVisualIdentityPackWidget,
        "set_preview",
        lambda self, _renderable, *, asset_id: applied.append(str(asset_id)),
    )

    browser.apply_filter("joy")
    assert await asyncio.to_thread(started.wait, 2)
    editor.load_character({"id": char_id, "name": "Reloaded same character"})
    release.set()
    await app.workers.wait_for_complete()

    assert applied == []


@pytest.mark.asyncio
async def test_preview_rechecks_active_asset_identity_after_decode(
    personas_editor_with_bound_pack, monkeypatch
):
    from tldw_chatbook.Chat.console_image_view import ConsoleImageRenderCache

    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    original_resolve = personas_screen_module.resolve_visual_identity
    original_prepare = ConsoleImageRenderCache.prepare
    decode_started = Event()
    release_decode = Event()
    identity_changed = Event()

    def changing_resolve(*args, **kwargs):
        resolution = original_resolve(*args, **kwargs)
        if not identity_changed.is_set():
            return resolution
        return replace(
            resolution,
            pack_version_id=(resolution.pack_version_id or 0) + 1,
            asset_id=(resolution.asset_id or 0) + 100,
            storage_source="manual",
            storage_relpath="redacted-new-source",
            cache_identity=("changed-during-decode",),
        )

    def delayed_prepare(self, cache_key, image_bytes):
        decode_started.set()
        assert release_decode.wait(2)
        return original_prepare(self, cache_key, image_bytes)

    applied: list[str] = []
    monkeypatch.setattr(
        personas_screen_module, "resolve_visual_identity", changing_resolve
    )
    monkeypatch.setattr(ConsoleImageRenderCache, "prepare", delayed_prepare)
    monkeypatch.setattr(
        PersonasVisualIdentityPackWidget,
        "set_preview",
        lambda self, _renderable, *, asset_id: applied.append(str(asset_id)),
    )

    browser.apply_filter("joy")
    assert await asyncio.to_thread(decode_started.wait, 2)
    identity_changed.set()
    release_decode.set()
    await app.workers.wait_for_complete()

    assert applied == []


@pytest.mark.asyncio
async def test_preview_rejects_binding_transition_during_decode(
    personas_editor_with_bound_pack, monkeypatch
):
    from tldw_chatbook.Chat.console_image_view import ConsoleImageRenderCache

    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(PersonasVisualIdentityPackWidget)
    original_prepare = ConsoleImageRenderCache.prepare
    decode_started = Event()
    release_decode = Event()

    def delayed_prepare(self, cache_key, image_bytes):
        decode_started.set()
        assert release_decode.wait(2)
        return original_prepare(self, cache_key, image_bytes)

    applied: list[str] = []
    monkeypatch.setattr(ConsoleImageRenderCache, "prepare", delayed_prepare)
    monkeypatch.setattr(
        PersonasVisualIdentityPackWidget,
        "set_preview",
        lambda self, _renderable, *, asset_id: applied.append(str(asset_id)),
    )

    browser.apply_filter("joy")
    assert await asyncio.to_thread(decode_started.wait, 2)
    assert browser.pack is not None
    browser.pack = replace(browser.pack, binding_id=browser.pack.binding_id + 1)
    release_decode.set()
    await app.workers.wait_for_complete()

    assert applied == []


@pytest.mark.asyncio
async def test_late_pack_metadata_cannot_remount_over_a_newer_character(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    assert graph is not None
    started = Event()
    release = Event()

    def delayed_graph(_self, _kind, _actor_id):
        started.set()
        assert release.wait(2)
        return graph

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", delayed_graph
    )
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    screen._character_editor_generation += 1
    editor.load_character(
        {"id": char_id + 1, "name": "Newer"}, visual_identity_pending=True
    )
    host = editor.query_one("#personas-char-editor-visual-identity-host")
    assert host.display
    assert any(
        isinstance(child, Static)
        and str(child.renderable) == "Loading visual identity…"
        for child in host.children
    )
    release.set()
    await task
    await asyncio.sleep(0.1)

    assert not editor.query(PersonasVisualIdentityPackWidget)


@pytest.mark.asyncio
async def test_pack_metadata_read_cannot_mount_after_character_mode_exit(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    started = Event()
    release = Event()

    def delayed_graph(_self, _kind, _actor_id):
        started.set()
        assert release.wait(2)
        return graph

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", delayed_graph
    )
    editor._reset_visual_identity_browser()
    await editor.query_one(
        "#personas-char-editor-visual-identity-host"
    ).remove_children()
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    screen.state.active_mode = "personas"
    screen._show_center("#ccp-persona-card-view")
    release.set()
    await task

    assert not editor.query(PersonasVisualIdentityPackWidget)


@pytest.mark.asyncio
async def test_pack_metadata_read_cannot_mount_after_same_character_reload(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    started = Event()
    release = Event()

    def delayed_graph(_self, _kind, _actor_id):
        started.set()
        assert release.wait(2)
        return graph

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", delayed_graph
    )
    editor._reset_visual_identity_browser()
    await editor.query_one(
        "#personas-char-editor-visual-identity-host"
    ).remove_children()
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    editor.load_character({"id": char_id, "name": "Reloaded same character"})
    release.set()
    await task

    assert not editor.query(PersonasVisualIdentityPackWidget)


@pytest.mark.asyncio
async def test_pack_metadata_read_rejects_changed_live_binding(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    assert graph is not None
    changed_graph = {
        **graph,
        "binding": {**graph["binding"], "id": int(graph["binding"]["id"]) + 1},
    }
    started = Event()
    release = Event()
    calls = 0

    def changing_graph(_self, _kind, _actor_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(2)
            return graph
        return changed_graph

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", changing_graph
    )
    editor._reset_visual_identity_browser()
    await editor.query_one(
        "#personas-char-editor-visual-identity-host"
    ).remove_children()
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    release.set()
    await task

    assert calls >= 2
    assert not editor.query(PersonasVisualIdentityPackWidget)
    _assert_visual_identity_unavailable(editor)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_call", [1, 2, 3])
async def test_metadata_identity_mismatch_is_unavailable_at_every_read(
    personas_editor_with_bound_pack, monkeypatch, changed_call
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    assert graph is not None
    changed_graph = {
        **graph,
        "version": {**graph["version"], "id": int(graph["version"]["id"]) + 1},
    }
    calls = 0

    def changing_graph(_self, _kind, _actor_id):
        nonlocal calls
        calls += 1
        return changed_graph if calls == changed_call else graph

    async def no_preview(_snapshot):
        return None

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", changing_graph
    )
    monkeypatch.setattr(screen, "_render_visual_identity_pack_preview", no_preview)
    editor.load_character(
        {"id": char_id, "name": "Metadata mismatch"},
        visual_identity_pending=True,
    )
    await asyncio.sleep(0)
    await screen._configure_character_visual_identity(
        _visual_identity_load_snapshot(screen, editor, char_id)
    )

    _assert_visual_identity_unavailable(editor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_call", "error"),
    [
        (1, sqlite3.DatabaseError("private initial database detail")),
        (2, sqlite3.OperationalError("private live database detail")),
        (3, sqlite3.OperationalError("private final database detail")),
    ],
)
async def test_metadata_database_error_is_unavailable_at_every_read(
    personas_editor_with_bound_pack, monkeypatch, failed_call, error
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    calls = 0
    messages: list[str] = []

    def failing_graph(_self, _kind, _actor_id):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise error
        return graph

    async def no_preview(_snapshot):
        return None

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", failing_graph
    )
    monkeypatch.setattr(
        personas_screen_module.logger,
        "debug",
        lambda message, *_args, **_kwargs: messages.append(str(message)),
    )
    monkeypatch.setattr(screen, "_render_visual_identity_pack_preview", no_preview)
    editor.load_character(
        {"id": char_id, "name": "Metadata database failure"},
        visual_identity_pending=True,
    )
    await asyncio.sleep(0)
    await screen._configure_character_visual_identity(
        _visual_identity_load_snapshot(screen, editor, char_id)
    )

    _assert_visual_identity_unavailable(editor)
    assert any("category=database" in message for message in messages)
    assert "private" not in " ".join(messages)


@pytest.mark.asyncio
async def test_stale_final_read_failure_cannot_replace_new_character_legacy_state(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    assert graph is not None
    final_started = Event()
    release_final = Event()
    calls = 0

    def fail_final_read(_self, _kind, _actor_id):
        nonlocal calls
        calls += 1
        if calls == 3:
            final_started.set()
            assert release_final.wait(2)
            raise sqlite3.OperationalError("private stale final-read detail")
        return graph

    async def no_preview(_snapshot):
        return None

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", fail_final_read
    )
    monkeypatch.setattr(screen, "_render_visual_identity_pack_preview", no_preview)
    editor.load_character(
        {"id": char_id, "name": "Blocked final read"},
        visual_identity_pending=True,
    )
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    assert await asyncio.to_thread(final_started.wait, 2)

    editor.new_character()
    host = editor.query_one("#personas-char-editor-visual-identity-host")
    await host.remove_children()
    release_final.set()
    await task

    assert editor.query_one("#personas-char-editor-legacy-expressions").display
    assert not host.display
    assert not host.children
    assert editor.query_one(
        "#personas-char-editor-expr-thinking-upload", Button
    ).disabled


@pytest.mark.asyncio
async def test_stale_final_read_failure_discards_exact_mount_after_mode_exit(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    assert graph is not None
    final_started = Event()
    release_final = Event()
    calls = 0

    def fail_final_read(_self, _kind, _actor_id):
        nonlocal calls
        calls += 1
        if calls == 3:
            final_started.set()
            assert release_final.wait(2)
            raise sqlite3.OperationalError("private stale mode-exit detail")
        return graph

    async def no_preview(_snapshot):
        return None

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", fail_final_read
    )
    monkeypatch.setattr(screen, "_render_visual_identity_pack_preview", no_preview)
    editor.load_character(
        {"id": char_id, "name": "Blocked mode-exit read"},
        visual_identity_pending=True,
    )
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    assert await asyncio.to_thread(final_started.wait, 2)
    old_browser = editor.query_one(PersonasVisualIdentityPackWidget)

    screen.state.active_mode = "personas"
    screen._show_center("#ccp-persona-card-view")
    release_final.set()
    await task

    assert old_browser.parent is None
    assert screen.state.active_mode == "personas"
    assert screen.query_one("#ccp-persona-card-view").display


@pytest.mark.asyncio
async def test_pack_browser_mounted_during_mode_exit_is_removed(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    original_show = editor.show_visual_identity_pack
    mounted = asyncio.Event()
    release = asyncio.Event()

    async def delayed_show(pack):
        browser = await original_show(pack)
        mounted.set()
        await release.wait()
        return browser

    monkeypatch.setattr(editor, "show_visual_identity_pack", delayed_show)
    editor._reset_visual_identity_browser()
    await editor.query_one(
        "#personas-char-editor-visual-identity-host"
    ).remove_children()
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    await asyncio.wait_for(mounted.wait(), 2)
    screen.state.active_mode = "personas"
    screen._show_center("#ccp-persona-card-view")
    release.set()
    await task

    assert not editor.query(PersonasVisualIdentityPackWidget)
    assert editor.query_one("#personas-char-editor-visual-identity-host").display


@pytest.mark.asyncio
async def test_pack_browser_mounted_during_binding_change_is_removed(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    assert graph is not None
    changed_graph = {
        **graph,
        "version": {**graph["version"], "id": int(graph["version"]["id"]) + 1},
    }
    binding_changed = False

    def changing_graph(_self, _kind, _actor_id):
        return changed_graph if binding_changed else graph

    original_show = editor.show_visual_identity_pack
    mounted = asyncio.Event()
    release = asyncio.Event()

    async def delayed_show(pack):
        browser = await original_show(pack)
        mounted.set()
        await release.wait()
        return browser

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", changing_graph
    )
    monkeypatch.setattr(editor, "show_visual_identity_pack", delayed_show)
    editor._reset_visual_identity_browser()
    await editor.query_one(
        "#personas-char-editor-visual-identity-host"
    ).remove_children()
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    await asyncio.wait_for(mounted.wait(), 2)
    binding_changed = True
    release.set()
    await task

    assert not editor.query(PersonasVisualIdentityPackWidget)
    assert editor.query_one("#personas-char-editor-visual-identity-host").display
    _assert_visual_identity_unavailable(editor)


@pytest.mark.asyncio
async def test_active_binding_read_hides_and_disables_legacy_authoring(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)
    graph = VisualIdentityRepository(db).get_active_actor_pack("character", char_id)
    started = Event()
    release = Event()

    def delayed_graph(_self, _kind, _actor_id):
        started.set()
        assert release.wait(2)
        return graph

    monkeypatch.setattr(
        VisualIdentityRepository, "get_active_actor_pack", delayed_graph
    )
    editor.load_character(
        {"id": char_id, "name": "Reloaded bound character"},
        visual_identity_pending=True,
    )
    task = asyncio.create_task(
        screen._configure_character_visual_identity(
            _visual_identity_load_snapshot(screen, editor, char_id)
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    captured: list[object] = []
    original_post_message = editor.post_message
    monkeypatch.setattr(editor, "post_message", captured.append)
    upload = editor.query_one("#personas-char-editor-expr-thinking-upload", Button)
    try:
        assert not editor.query_one("#personas-char-editor-legacy-expressions").display
        assert upload.disabled
        upload.press()
        await asyncio.sleep(0)
        assert not any(
            isinstance(message, CharacterExpressionUploadRequested)
            for message in captured
        )
    finally:
        monkeypatch.setattr(editor, "post_message", original_post_message)
        release.set()
    await task
    await app.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_active_binding_read_failure_shows_non_authoring_unavailable(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = screen.query_one(PersonasCharacterEditorWidget)

    def failed_graph(_self, _kind, _actor_id):
        raise ValueError("visual identity graph unavailable")

    monkeypatch.setattr(VisualIdentityRepository, "get_active_actor_pack", failed_graph)
    editor.load_character(
        {"id": char_id, "name": "Reloaded bound character"},
        visual_identity_pending=True,
    )
    await asyncio.sleep(0)
    await screen._configure_character_visual_identity(
        _visual_identity_load_snapshot(screen, editor, char_id)
    )

    legacy = editor.query_one("#personas-char-editor-legacy-expressions")
    host = editor.query_one("#personas-char-editor-visual-identity-host")
    assert not legacy.display
    assert host.display
    assert not editor.query(PersonasVisualIdentityPackWidget)
    assert any(
        isinstance(child, Static) and str(child.renderable) == "Unavailable"
        for child in host.children
    )
    assert editor.query_one(
        "#personas-char-editor-expr-thinking-upload", Button
    ).disabled
    await app.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_upload_writes_expression_row(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    buf = BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    result = await screen._apply_expression_upload(
        char_id, "speaking", buf.getvalue(), "image/png"
    )
    assert db.get_character_expression_image(char_id, "speaking") is not None
    # task-563 AC4: callers that aggregate multiple slots (Generate-all)
    # need an honest success signal, not just "this call didn't raise".
    assert result is True


@pytest.mark.asyncio
async def test_apply_expression_upload_returns_false_on_db_write_failure(
    personas_editor_with_saved_character, monkeypatch
):
    """task-563 AC4: a DB-write failure inside _apply_expression_upload must
    be reported to the caller (not just swallowed behind an error notify) so
    an aggregating caller (the Generate-all sweep) doesn't count this slot as
    a success."""
    app, screen, db, char_id = personas_editor_with_saved_character

    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(db, "set_character_expression_image", _boom)

    result = await screen._apply_expression_upload(
        char_id, "speaking", b"not-really-png", "image/png"
    )

    assert result is False
    assert db.get_character_expression_image(char_id, "speaking") is None


@pytest.mark.asyncio
async def test_clear_soft_deletes_expression_row(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    db.set_character_expression_image(char_id, "error", b"x")
    await screen._clear_expression_slot(char_id, "error")
    assert db.get_character_expression_image(char_id, "error") is None


@pytest.mark.asyncio
async def test_apply_expression_set_stages_idle_and_writes_three(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    import io as _io
    from PIL import Image as _Img

    def _png(c=(1, 2, 3)):
        b = _io.BytesIO()
        _Img.new("RGB", (8, 8), c).save(b, format="PNG")
        return b.getvalue()

    result = await screen._apply_expression_set(
        char_id, {"idle": _png((9, 9, 9)), "speaking": _png(), "thinking": _png()}
    )
    # idle STAGED in the editor (not the table); three -> table
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
        PersonasCharacterEditorWidget,
    )

    editor = screen.query_one(PersonasCharacterEditorWidget)
    assert editor.current_avatar_bytes() == _png((9, 9, 9))  # idle staged
    assert db.get_character_expression_image(char_id, "speaking") is not None
    assert db.get_character_expression_image(char_id, "idle") is None
    assert set(result.applied) >= {"idle", "speaking", "thinking"}


# ===== Roleplay P3d-2 Task 4: import/export expression-set buttons + workers =====


@pytest.mark.asyncio
async def test_import_expression_set_from_zip_path(
    personas_editor_with_saved_character, tmp_path
):
    app, screen, db, char_id = personas_editor_with_saved_character
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    import io as _io
    from PIL import Image as _Img

    def _png():
        b = _io.BytesIO()
        _Img.new("RGB", (8, 8)).save(b, format="PNG")
        return b.getvalue()

    z = tmp_path / "set.zip"
    z.write_bytes(build_expression_set_zip("Ada", {"idle": _png(), "speaking": _png()}))

    await screen._import_expression_set_from_path(char_id, str(z))

    assert db.get_character_expression_image(char_id, "speaking") is not None
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
        PersonasCharacterEditorWidget,
    )

    assert (
        screen.query_one(PersonasCharacterEditorWidget).current_avatar_bytes()
        is not None
    )  # idle staged


@pytest.mark.asyncio
async def test_import_vpack_from_path(personas_editor_with_saved_character, tmp_path):
    app, screen, db, char_id = personas_editor_with_saved_character
    from Tests.Character_Chat.test_expression_set_io import simple_vpack, _png

    z = tmp_path / "pack.tldw-persona-vpack"
    z.write_bytes(
        simple_vpack({"idle": _png(), "speaking": _png(), "thinking": _png()})
    )

    await screen._import_expression_set_from_path(char_id, str(z))

    assert db.get_character_expression_image(char_id, "speaking") is not None
    assert db.get_character_expression_image(char_id, "thinking") is not None
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
        PersonasCharacterEditorWidget,
    )

    assert (
        screen.query_one(PersonasCharacterEditorWidget).current_avatar_bytes()
        is not None
    )  # idle staged


@pytest.mark.asyncio
async def test_export_expression_set_writes_a_zip(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    import io as _io
    from PIL import Image as _Img

    def _png():
        b = _io.BytesIO()
        _Img.new("RGB", (8, 8)).save(b, format="PNG")
        return b.getvalue()

    db.set_character_expression_image(char_id, "speaking", _png())
    target = await screen._export_expression_set(char_id, "Ada")
    assert target is not None
    from pathlib import Path
    import zipfile

    assert zipfile.is_zipfile(Path(target))
    assert "speaking.png" in zipfile.ZipFile(target).namelist()


@pytest.mark.asyncio
async def test_import_export_buttons_present_for_saved_character(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    assert screen.query_one("#personas-char-editor-expr-import") is not None
    assert screen.query_one("#personas-char-editor-expr-export") is not None


# ===== Review fix 1: _export_expression_set cleans up its temp file on
# failure (mirrors _dictionary_export_worker's try/except OSError +
# temp.unlink(missing_ok=True) idiom, which the initial implementation
# omitted). =====


@pytest.mark.asyncio
async def test_export_expression_set_cleans_up_temp_on_replace_failure(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character

    def _png():
        buf = BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        return buf.getvalue()

    db.set_character_expression_image(char_id, "speaking", _png())

    def _boom(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", _boom)

    # Diff before/after rather than asserting the dir is empty: it's the
    # shared test-home exports dir, which other test runs may have already
    # left debris in.
    exports_dir = get_user_data_dir() / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    before = set(exports_dir.glob("*.tmp"))

    with pytest.raises(OSError):
        await screen._export_expression_set(char_id, "Ada")

    after = set(exports_dir.glob("*.tmp"))
    assert after - before == set()


# ===== Review fix 2: the Import set…/Export set… buttons must be disabled
# for an unsaved character, same as the per-slot Upload/Clear buttons
# (_sync_expression_slots_enabled). =====


class _UnsavedEditorHost(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()


@pytest.mark.asyncio
async def test_import_export_buttons_disabled_for_unsaved_character():
    app = _UnsavedEditorHost()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A"})  # no "id" key -> unsaved
        await pilot.pause()
        assert ed.expression_character_id() is None
        assert (
            ed.query_one("#personas-char-editor-expr-import", Button).disabled is True
        )
        assert (
            ed.query_one("#personas-char-editor-expr-export", Button).disabled is True
        )


# ===== Qodo review fix 1: _import_expression_set_from_path validates the
# picked path at the screen boundary (mirrors _read_avatar_image_bytes'
# use of validate_path_simple), instead of handing an unvalidated path
# straight to the pure resolver. =====


@pytest.mark.asyncio
async def test_import_expression_set_nonexistent_path_notifies_and_does_not_crash(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    # Shadow the delegating test App's notify (like TestDictionaryImport's
    # _capture_notifications) rather than mock_app_instance.notify --
    # screen.app_instance is the PersonasTestApp, and its real (Textual App)
    # notify() shadows the mock's via normal attribute lookup.
    captured: list[tuple[str, str]] = []
    app.notify = lambda message, severity="information", **kwargs: captured.append(
        (str(message), severity)
    )

    # Must not raise -- the invalid path is rejected before it ever reaches
    # resolve_local_expression_set.
    await screen._import_expression_set_from_path(char_id, "/no/such/path/set.zip")

    assert captured, "expected a notification for a rejected path"
    assert captured[-1][1] == "error"


# ===== Qodo review fix 7: the export handler must honor the same
# _io_dialog_active gate as the import handler and _dictionary_export_worker,
# so a queued export cannot race a second worker onto the same filename. =====


@pytest.mark.asyncio
async def test_export_handler_blocked_while_io_dialog_active(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        CharacterExpressionSetExportRequested,
    )

    calls: list[int] = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    screen._io_dialog_active = True

    screen._handle_expression_set_export_requested(
        CharacterExpressionSetExportRequested()
    )

    assert calls == []  # gate blocked a second worker from starting


@pytest.mark.asyncio
async def test_export_handler_starts_worker_and_sets_gate_when_clear(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        CharacterExpressionSetExportRequested,
    )

    calls: list[int] = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()  # avoid a "coroutine was never awaited" warning

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)
    screen._io_dialog_active = False

    screen._handle_expression_set_export_requested(
        CharacterExpressionSetExportRequested()
    )

    assert calls == [1]
    assert screen._io_dialog_active is True  # gate set before the worker starts


@pytest.mark.asyncio
async def test_import_export_buttons_enabled_for_saved_character(
    personas_editor_with_saved_character,
):
    app, screen, db, char_id = personas_editor_with_saved_character
    assert (
        screen.query_one("#personas-char-editor-expr-import", Button).disabled is False
    )
    assert (
        screen.query_one("#personas-char-editor-expr-export", Button).disabled is False
    )
