"""Mounted geometry contracts for the Console Character browser."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.console_rail_section_helpers import open_rail_section
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_character_avatar import _set_chat_images_setting
from Tests.UI.test_console_character_context import _groups, _resolved
from Tests.UI.test_console_inspector_compact_access import _stored_rail_preferences
from Tests.UI.test_console_left_rail import make_console_pilot
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    ResolvedLocalCharacterKey,
    UnresolvedConversationKey,
)
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationResultKind,
    ConsoleConversationActivationResult,
)
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextState,
)
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    LibraryCharacterRepairContext,
    LibraryUnavailableConversationInspection,
    LibraryUnavailableConversationsBrowse,
    RoleplayCharacterConversationLink,
)
from tldw_chatbook.Widgets.Console.console_character_context import (
    ConsoleCharacterContext,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((52, 20), (72, 35), (80, 24), (120, 50)))
async def test_character_context_stays_inside_rail_and_never_claims_task5(
    size, tmp_path
) -> None:
    async with make_console_pilot(size=size, production_styles=True) as pilot:
        screen = pilot.app.screen
        if size[0] >= 72:
            screen._set_console_rail_preference(
                left_open=True,
                notify_on_failure=False,
            )
            await pilot.pause(0.4)
            rail = screen.query_one("#console-left-rail")
            assert rail.display
            await open_rail_section(screen, pilot, "character")
            await pilot.pause(0.4)
            browse_state = ConsoleCharacterContextState(
                groups=_groups(),
                expanded_key=_groups()[0].key,
                data_revision=7,
            )
            screen._character_context._publish(browse_state)
            await pilot.pause()
            character = screen.query_one("#console-character-context")
            character_geometry = screen.find_widget(character)
            assert character_geometry.clip
            assert rail.region.contains_region(character_geometry.clip)
            assert len(screen.query(".console-character-group")) == 4
            assert len(screen.query(".console-character-row")) == 5
            view_all = screen.query_one(
                f"#{ConsoleCharacterContext.action_dom_id('view-all', _groups()[0].key)}"
            )
            assert str(view_all.label) == "View all 9 in Roleplay"
            for action in character.query("Button, Input"):
                action.focus()
                for _ in range(20):
                    action.scroll_visible(animate=False, immediate=True, force=True)
                    await pilot.pause(0.05)
                    geometry = screen.find_widget(action)
                    if geometry.clip.contains_region(action.region):
                        break
                assert geometry.clip.contains_region(action.region)

                if isinstance(action, Input):
                    painted_search = "\n".join(
                        strip.text[action.region.x : action.region.right]
                        for strip in screen._compositor.render_strips()[
                            action.region.y : action.region.bottom
                        ]
                    )
                    assert "Search chats" in painted_search
                    assert (
                        action.name
                        == "Global Keyword search over local character chats"
                    )

            view_all.focus()
            for _ in range(20):
                view_all.scroll_visible(animate=False, immediate=True, force=True)
                await pilot.pause(0.05)
                painted = "\n".join(
                    strip.text for strip in screen._compositor.render_strips()
                )
                if "View all 9 in Roleplay" in painted:
                    break
            assert "View all 9 in Roleplay" in painted
            pilot.app.save_screenshot(
                filename=f"character-browse-{size[0]}x{size[1]}.svg",
                path=str(tmp_path),
            )
            (tmp_path / "browse-paint.txt").write_text(painted, encoding="utf-8")

            search_state = ConsoleCharacterContextState(
                groups=_groups(),
                query="needle",
                search_rows=tuple(
                    replace(
                        _resolved(1, f"search-{index}"),
                        last_modified=datetime.now(UTC).isoformat(),
                    )
                    for index in range(8)
                ),
                data_revision=7,
            )
            screen._character_context._publish(search_state)
            await pilot.pause()
            assert len(character.query(".console-character-row")) == 8
            for action in character.query("Button, Input"):
                action.focus()
                for _ in range(20):
                    action.scroll_visible(animate=False, immediate=True, force=True)
                    await pilot.pause(0.05)
                    geometry = screen.find_widget(action)
                    if geometry.clip.contains_region(action.region):
                        break
                assert geometry.clip.contains_region(action.region), [
                    (
                        str(parent),
                        parent.region,
                        parent.virtual_size,
                        parent.scroll_y,
                        parent.max_scroll_y,
                    )
                    for parent in action.ancestors
                    if hasattr(parent, "scroll_y")
                ]
                if action.has_class("console-character-row"):
                    strips = screen._compositor.render_strips()
                    row_paint = "\n".join(
                        strip.text[action.region.x : action.region.right]
                        for strip in strips[action.region.y : action.region.bottom]
                    )
                    assert "Character 1" in row_paint
                    assert "Local" in row_paint
                    assert "now" in row_paint
        else:
            character = screen.query_one("#console-character-context")
            character_controls = set(character.query(Button)) | set(
                character.query(Input)
            )
            assert character_controls.isdisjoint(screen.focus_chain)
        rendered = str(screen.render())
        assert "Continue search in Character chats" not in rendered
        painted = "\n".join(strip.text for strip in screen._compositor.render_strips())
        assert "Continue search in Character chats" not in painted
        pilot.app.save_screenshot(
            filename=f"character-final-{size[0]}x{size[1]}.svg",
            path=str(tmp_path),
        )
        (tmp_path / "final-paint.txt").write_text(painted, encoding="utf-8")


@pytest.fixture
def character_walkthrough_database(tmp_path: Path, record_property):
    """Close the test-owned handle after the Pilot host and workers settle."""
    database = CharactersRAGDB(
        tmp_path / "task4-walkthrough.sqlite",
        client_id="task4-walkthrough",
    )
    try:
        yield database
    finally:
        record_property(
            "context_db_connections_before_close",
            database.registered_connection_count(),
        )
        database.close()
        record_property(
            "context_db_connections_after_close",
            database.registered_connection_count(),
        )


@pytest.mark.asyncio
async def test_real_console_walkthrough_uses_synthetic_local_database(
    character_walkthrough_database,
) -> None:
    """Walk every Step 10 state in one production-composed Textual host."""

    app = _build_test_app()
    database = character_walkthrough_database
    app.chachanotes_db = database
    character_id = database.add_character_card({"name": "Synthetic Ada"})
    assert character_id is not None
    authority = database.get_local_authority_id()
    assert (
        database.add_conversation(
            {
                "id": "task4-synthetic-chat",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": "Lantern archive",
            }
        )
        == "task4-synthetic-chat"
    )
    assert (
        database.add_message(
            {
                "id": "task4-synthetic-message",
                "conversation_id": "task4-synthetic-chat",
                "sender": "user",
                "role": "user",
                "content": "SYNTHETIC_LANTERN_CANARY",
                "timestamp": "2026-09-03T12:00:00Z",
            }
        )
        == "task4-synthetic-message"
    )
    database.set_conversation_active_leaf(
        "task4-synthetic-chat", "task4-synthetic-message"
    )
    assert (
        database.add_conversation(
            {
                "id": "task4-unavailable-chat",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": "Historical unavailable chat",
            }
        )
        == "task4-unavailable-chat"
    )
    assert (
        database.add_message(
            {
                "id": "task4-unavailable-message",
                "conversation_id": "task4-unavailable-chat",
                "sender": "user",
                "role": "user",
                "content": "historical unavailable",
                "timestamp": "2026-09-03T11:00:00Z",
            }
        )
        == "task4-unavailable-message"
    )
    database.set_conversation_active_leaf(
        "task4-unavailable-chat", "task4-unavailable-message"
    )
    with database.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET assistant_authority_id = NULL, "
            "assistant_id = 'historical-unknown' WHERE id = ?",
            ("task4-unavailable-chat",),
        )
    _configure_native_ready_console(app)
    _set_chat_images_setting(app, "show_character_avatar", False)

    class ProductionStyledConsoleHarness(ConsoleHarness):
        CSS_PATH = str(BUNDLED_STYLESHEET)

    host = ProductionStyledConsoleHarness(app)
    async with host.run_test(size=(120, 50)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        for _ in range(80):
            if screen._character_context.state.groups:
                break
            await pilot.pause(0.05)
        labels = {
            group.character_label for group in screen._character_context.state.groups
        }
        assert labels == {"Synthetic Ada", "Chats with unavailable characters"}
        await open_rail_section(screen, pilot, "character")
        await pilot.pause()

        # First-use and returning-user disclosure are exercised through the
        # canonical screen writer in this production composition.
        assert screen._current_console_rail_state().character_open
        screen._toggle_console_rail_section("character", next_open=False)
        await pilot.pause()
        stored = _stored_rail_preferences(app)
        assert stored["character_open"] is False
        assert stored[CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY] is True
        screen._toggle_console_rail_section("character", next_open=True)
        await pilot.pause()
        assert _stored_rail_preferences(app)["character_open"] is True
        assert not screen.query_one("#console-character-avatar-frame").display

        controller = screen._character_context
        activations: list[CharacterConversationActivationRequest] = []
        roleplay_links: list[RoleplayCharacterConversationLink] = []
        repair_contexts: list[LibraryCharacterRepairContext] = []
        inspection_links: list[LibraryUnavailableConversationInspection] = []
        unavailable_browses: list[LibraryUnavailableConversationsBrowse] = []
        roleplay_home: list[bool] = []

        async def activate(request, _cancellation):
            activations.append(request)
            return ConsoleConversationActivationResult(
                ConsoleActivationResultKind.OPENED,
                request.target,
                True,
            )

        controller._activate_target = activate
        controller._navigate_roleplay = roleplay_links.append
        controller._navigate_repair = repair_contexts.append
        controller._navigate_inspection = inspection_links.append
        controller._navigate_unavailable_browse = unavailable_browses.append
        controller._navigate_roleplay_home = lambda: roleplay_home.append(True)

        resolved_group = next(
            group
            for group in controller.state.groups
            if isinstance(group.key, ResolvedLocalCharacterKey)
        )
        if controller.state.expanded_key != resolved_group.key:
            screen.query_one(
                f"#{ConsoleCharacterContext.group_dom_id(resolved_group.key)}",
                Button,
            ).press()
            await pilot.pause()

        row_key = resolved_group.rows[0].row_key
        row_id = ConsoleCharacterContext.row_dom_id(row_key)
        row = screen.query_one(f"#{row_id}")
        row.press()
        for _ in range(40):
            if activations:
                break
            await pilot.pause(0.05)
        assert activations and activations[0].target == resolved_group.rows[0].target

        await pilot.pause()
        screen.query_one(
            f"#{ConsoleCharacterContext.action_dom_id('view-all', resolved_group.key)}",
            Button,
        ).press()
        await pilot.pause()
        assert roleplay_links and roleplay_links[0].character == resolved_group.key

        row = screen.query_one(f"#{row_id}")
        row.focus()
        await pilot.pause()
        search = screen.query_one("#console-character-search", Input)
        search.focus()
        await pilot.pause()
        search.value = "SYNTHETIC_LANTERN_CANARY"
        for _ in range(80):
            state = screen._character_context.state
            if state.query and not state.loading:
                break
            await pilot.pause(0.05)
        state = screen._character_context.state
        assert len(state.search_rows) == 1
        assert state.search_rows[0].selected_excerpt == ""
        await _wait_for_selector(
            screen,
            pilot,
            f"#{row_id}",
        )
        search_row = screen.query_one(f"#{row_id}")
        assert "Local" in str(search_row.label)

        await pilot.press("escape")
        for _ in range(80):
            if (
                not screen._character_context.state.query
                and getattr(screen.focused, "id", None) == row_id
            ):
                break
            await pilot.pause(0.05)
        assert screen._character_context.state.query == ""
        assert getattr(screen.focused, "id", None) == row_id

        unavailable_group = next(
            group
            for group in controller.state.groups
            if isinstance(group.key, UnresolvedConversationKey)
        )
        screen.query_one(
            f"#{ConsoleCharacterContext.group_dom_id(unavailable_group.key)}",
            Button,
        ).press()
        await pilot.pause()
        unavailable_row = unavailable_group.rows[0]
        screen.query_one(
            f"#{ConsoleCharacterContext.row_dom_id(unavailable_row.row_key)}",
            Button,
        ).focus()
        await pilot.pause()
        reason = screen.query_one("#console-character-unavailable-reason")
        assert str(reason.renderable) == "Historical identity incomplete"
        await controller.refresh_unavailable_details((unavailable_group,))
        await pilot.pause()
        assert screen.query_one("#console-character-repair-library", Button)
        screen.query_one("#console-character-open-library", Button).press()
        for _ in range(40):
            if inspection_links:
                break
            await pilot.pause(0.05)
        assert inspection_links[0].unresolved == unavailable_row.unresolved
        assert repair_contexts == []
        await _wait_for_selector(
            screen,
            pilot,
            "#console-character-repair-library",
        )
        screen.query_one("#console-character-repair-library", Button).press()
        for _ in range(40):
            if repair_contexts:
                break
            await pilot.pause(0.05)
        assert repair_contexts[0].unresolved == unavailable_row.unresolved
        unavailable_view_all_id = ConsoleCharacterContext.action_dom_id(
            "view-all", unavailable_group.key
        )
        await _wait_for_selector(screen, pilot, f"#{unavailable_view_all_id}")
        screen.query_one(
            f"#{unavailable_view_all_id}",
            Button,
        ).press()
        for _ in range(40):
            if unavailable_browses:
                break
            await pilot.pause(0.05)
        assert unavailable_browses[0].selected == unavailable_row.unresolved

        # Overall empty recovery and the narrow fallback remain truthful and
        # do not advertise the dormant Task 5 capability.
        controller._publish(ConsoleCharacterContextState())
        await pilot.pause()
        screen.query_one("#console-character-open-roleplay", Button).press()
        await pilot.pause()
        assert roleplay_home == [True]
        await pilot.resize_terminal(52, 20)
        await pilot.pause()
        character = screen.query_one("#console-character-context")
        controls = set(character.query(Button)) | set(character.query(Input))
        assert controls.isdisjoint(screen.focus_chain)
        assert "Continue search in Character chats" not in str(screen.render())
