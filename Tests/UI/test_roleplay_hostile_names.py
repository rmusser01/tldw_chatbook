"""Untrusted names render literally on every Roleplay surface (TASK-34400).

A character, persona, dictionary, lore book, entry key, tag or conversation
title shaped like Textual markup must paint as typed: no ``MarkupError`` (an
escaped crash ends ``run_test`` and fails the test; in the real app a
render-time ``MarkupError`` bypasses the TASK-32533 keep-alive and exits the
whole app), no ``@click`` meta on any painted cell, and clicking every painted
copy of the name runs nothing. ``[/`` and ``[TODO] y`` pin the escaper choice:
``textual.markup.escape`` gets both wrong on Textual 8.2.8, while
``Utils.input_validation.escape_markup`` and literal ``Content`` get them right.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import ListView, Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (bootstrap profile)
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app

# Roleplay frame B1: the paint helpers and the character seam live in the
# frame harness, their one home (test_roleplay_hostile_text_surfaces.py
# imports them from there too).
from Tests.UI.roleplay_frame_harness import (
    StyledRoleplayMockApp,
    click_meta_cells,
    open_styled_roleplay,
    painted_rows,
    seed_mock_characters,
    settle,
    wait_until,
)
from Tests.UI.test_personas_dictionaries import (
    FakeDictScopeService,
    make_dict_record,
)
from Tests.UI.test_personas_workbench import (
    PersonasTestApp,
    _conversation_record,
    _install_conversation_db,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditCharacterRequested,
    PersonaProfileSaveRequested,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

HOSTILE_NAMES = (
    "[/]",
    "[b]x",
    "[@click=app.record('x')]N[/]",
    "[/",
    "[TODO] y",
)
SIZE = (220, 55)


class _RecordingApp(PersonasTestApp):
    """Gives ``[@click=app.record(...)]`` a real target, so a click that ran
    it would be observable (a missing action is a silent no-op)."""

    def __init__(self, mock_app_instance):
        super().__init__(mock_app_instance)
        self.recorded: list[str] = []

    def action_record(self, value: str) -> None:
        self.recorded.append(value)


class _RecordingStyledApp(_RecordingApp, StyledRoleplayMockApp):
    """``_RecordingApp`` under styled tier 1 (Roleplay frame B1): only the lazy
    Roleplay sheet gives the header's item label and chips their row."""


def _painted(screen) -> str:
    return "\n".join(painted_rows(screen))


async def _assert_literal_and_inert(pilot, name: str, *, at_least: int) -> None:
    """The name paints literally at least ``at_least`` times, carries no
    ``@click`` meta, and clicking each painted copy runs nothing."""
    app = pilot.app
    screen = app.screen
    await app.run_action("record('positive-control')")
    assert app.recorded == ["positive-control"]
    app.recorded.clear()
    assert click_meta_cells(screen) == []
    copies = [
        (x, y)
        for y, row in enumerate(painted_rows(screen))
        for x in range(len(row))
        if row.startswith(name, x)
    ]
    assert len(copies) >= at_least, painted_rows(screen)
    for x, y in copies:
        await pilot.click(offset=(x, y))
        await settle(pilot)
    assert app.recorded == []
    assert click_meta_cells(app.screen) == []


async def _select_first_row(pilot) -> None:
    rows = pilot.app.screen.query_one("#personas-library-rows", ListView)
    rows.index = 0
    rows.action_select_cursor()
    await settle(pilot)


async def _enter_mode(pilot, mode: str) -> None:
    assert await pilot.click(f"#personas-mode-{mode}")
    await settle(pilot)


def _no_splash(section, key=None, default=None):
    """``get_cli_setting`` for the real app: no splash screen, else defaults."""
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def test_the_real_app_keeps_running_with_a_character_named_like_markup(
    monkeypatch,
):
    """The user-visible failure: on dev the whole app exited when Roleplay
    showed a character named ``[/]`` (a render-time ``MarkupError`` from the
    Inspector's ``Selected:`` line, which the keep-alive does not cover)."""
    seed_mock_characters(
        monkeypatch, [{"id": 1, "name": "[/]", "description": "d", "version": 1}]
    )
    app = _build_test_app(configured_default="personas")
    with patch_app_global("get_cli_setting", side_effect=_no_splash):
        async with app.run_test(size=SIZE) as pilot:
            await wait_until(
                pilot,
                lambda: (
                    type(app.screen).__name__ == "PersonasScreen"
                    and bool(app.screen.query("#personas-library-row-character-1"))
                ),
                what="Roleplay's character row",
            )
            await settle(pilot)
            await pilot.click("#personas-library-row-character-1")
            await settle(pilot)
            assert app.is_running
            assert app.screen.state.selected_entity_id == "1"
            assert "Selected: [/]" in _painted(app.screen)


@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_a_character_name_tag_and_conversation_title(
    name, mock_app_instance, monkeypatch
):
    """Library row, card, Inspector, conversation row, the Tag filter button,
    a toast and the header's item label while editing."""
    seed_mock_characters(
        monkeypatch,
        [{"id": 1, "name": name, "description": "d", "tags": [name], "version": 1}],
    )
    monkeypatch.setattr(
        character_handler_module, "_default_character_db", lambda: object()
    )
    _install_conversation_db(monkeypatch, [_conversation_record(1, title=name)])
    app = _RecordingApp(mock_app_instance)
    async with app.run_test(size=SIZE, notifications=True) as pilot:
        await settle(pilot)
        await _select_first_row(pilot)
        screen = app.screen
        assert screen.state.selected_entity_id == "1"
        assert f"Selected: {name}" in _painted(screen)
        # The Tag filter button names the active tag. Checked, then cleared,
        # so the click sweep below never opens the tag picker over the copies.
        await screen._apply_tag_filter(name)
        await settle(pilot)
        assert f"Tag: {name}" in _painted(screen)
        assert click_meta_cells(screen) == []
        await screen._apply_tag_filter(None)
        await settle(pilot)
        screen._notify(f"Imported '{name}'.", "information")
        await settle(pilot)
        assert f"Imported '{name}'." in _painted(screen)
        # Row, card name, card tags, Inspector, conversation row and toast.
        await _assert_literal_and_inert(pilot, name, at_least=6)
        # Roleplay frame B1: editing names the item in the header's literal
        # item label, never in the shared markup-on subtitle (the kind). This
        # unstyled tier gives the label no row to paint on, so the painted
        # and clicked copy is pinned under styled tier 1 by
        # test_the_header_item_label_and_server_label.
        screen.post_message(EditCharacterRequested("1"))
        await settle(pilot)
        assert screen._edit_mode == "edit"
        item = screen.query_one("#personas-header-item", FittedText)
        assert item.value == (name, True)
        subtitle = screen.query_one("#workbench-header-subtitle", Static)
        assert str(subtitle.render()) == "Characters"
        assert click_meta_cells(screen) == []


@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_the_header_item_label_and_server_label(
    name, mock_app_instance, monkeypatch
):
    """Roleplay frame B1's header surfaces: the item label (a literal
    ``FittedText``, viewing and editing) and the status chip's server label
    (escaped by ``build_header_view`` into the shared markup-on header)."""
    seed_mock_characters(
        monkeypatch, [{"id": 1, "name": name, "description": "d", "version": 1}]
    )
    async with open_styled_roleplay(
        "mock", mock_app_instance, size=SIZE, app_class=_RecordingStyledApp
    ) as pilot:
        screen = pilot.app.screen
        item = screen.query_one("#personas-header-item", FittedText)
        # Alone in the library, so the first-paint auto-selection (F-031) picks it.
        await wait_until(
            pilot,
            lambda: item.value == (name, False),
            what="the hostile name in the header",
        )
        pilot.app.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(last_known_server_label=name, active_server_id=None)
        )
        screen._set_persona_editor_runtime_source("server")
        screen._update_title()
        await settle(pilot)
        assert item.fitted_text == f"› {name}"
        painted = _painted(screen)
        assert f"› {name}" in painted
        assert f"Server: {name} · read-only" in painted
        # Library row, card name, Inspector, header item label and status.
        await _assert_literal_and_inert(pilot, name, at_least=5)
        # Editing is a local-only action: back to the local source first.
        screen._set_persona_editor_runtime_source("local")
        screen._update_title()
        await settle(pilot)
        screen.post_message(EditCharacterRequested("1"))
        await settle(pilot)
        assert screen._edit_mode == "edit"
        assert f"› {name} · editing" in _painted(screen)
        assert click_meta_cells(screen) == []


@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_a_persona_name(name, mock_app_instance, monkeypatch):
    seed_mock_characters(monkeypatch, [])
    record = {
        "id": "p-1",
        "name": name,
        "description": "d",
        "system_prompt": "You are terse.",
    }
    service = Mock()
    service.list_persona_profiles = AsyncMock(
        return_value={"items": [dict(record)], "total": 1}
    )
    service.get_persona_profile = AsyncMock(return_value=dict(record))
    mock_app_instance.character_persona_scope_service = service
    app = _RecordingApp(mock_app_instance)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await _enter_mode(pilot, "personas")
        await _select_first_row(pilot)
        assert f"Selected: {name}" in _painted(app.screen)
        # Row, profile card and Inspector.
        await _assert_literal_and_inert(pilot, name, at_least=3)


@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_a_chat_dictionary_and_its_entry_key(
    name, mock_app_instance, monkeypatch
):
    seed_mock_characters(monkeypatch, [])
    entry = {
        "pattern": name,
        "replacement": "replacement",
        "probability": 1.0,
        "group": None,
        "timed_effects": None,
        "max_replacements": 1,
        "type": "literal",
        "enabled": True,
        "case_sensitive": False,
        "priority": 0,
    }
    mock_app_instance.chat_dictionary_scope_service = FakeDictScopeService(
        [make_dict_record(1, name, entries=[entry])]
    )
    app = _RecordingApp(mock_app_instance)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await _enter_mode(pilot, "dictionaries")
        await _select_first_row(pilot)
        assert f"Selected: {name}" in _painted(app.screen)
        # Row, the entry's key in the entries table and the Inspector.
        await _assert_literal_and_inert(pilot, name, at_least=3)


@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_a_lore_book_and_its_entry_key(
    name, mock_app_instance, monkeypatch, tmp_path
):
    seed_mock_characters(monkeypatch, [])
    db = CharactersRAGDB(tmp_path / "hostile_lore.db", "test-client")
    try:
        manager = WorldBookManager(db)
        book_id = manager.create_world_book(name, description="d")
        manager.create_world_book_entry(book_id, keys=[name], content="content")
        mock_app_instance.chachanotes_db = db
        app = _RecordingApp(mock_app_instance)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await _enter_mode(pilot, "lore")
            await _select_first_row(pilot)
            assert f"Selected: {name}" in _painted(app.screen)
            # Row, the entry's key in the entries table and the Inspector.
            await _assert_literal_and_inert(pilot, name, at_least=3)
    finally:
        db.close_connection()


async def test_a_persona_save_that_fails_validation_keeps_the_app_running(
    mock_app_instance, monkeypatch
):
    """TASK-33790's crash: a 205-character name fails the model's 200-character
    limit, and the validation text (``[type=string_too_long, ...]``) reached a
    markup-parsing toast."""
    seed_mock_characters(monkeypatch, [])
    service = Mock()
    service.list_persona_profiles = AsyncMock(return_value={"items": [], "total": 0})
    service.create_persona_profile = AsyncMock(return_value={"id": "p-9"})
    mock_app_instance.character_persona_scope_service = service
    app = _RecordingApp(mock_app_instance)
    async with app.run_test(size=SIZE, notifications=True) as pilot:
        await settle(pilot)
        await _enter_mode(pilot, "personas")
        assert await pilot.click("#personas-library-new")
        await settle(pilot)
        screen = app.screen
        assert screen._edit_mode == "create"
        screen.post_message(PersonaProfileSaveRequested({"name": "N" * 205}))
        await settle(pilot)
        assert app.is_running
        service.create_persona_profile.assert_not_awaited()
        assert "[type=" in _painted(screen)
