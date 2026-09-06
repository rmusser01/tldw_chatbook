"""Exact Character activation must preserve reusable Console ownership."""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_console_screen_reuse import (
    _boot_settled,
    _press_until_screen,
    _scratch_env,
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationResultKind,
)
from tldw_chatbook.Constants import TAB_CHAT, TAB_PERSONAS
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_kind,size",
    [("inspection", (52, 20)), ("inspection", (80, 24)), ("browse", (140, 42))],
)
async def test_unavailable_route_returns_to_originating_console_character(
    monkeypatch, tmp_path, route_kind, size
):
    from copy import deepcopy

    from textual.widgets import Button

    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        UnresolvedConversationKey,
    )
    from tldw_chatbook.Constants import (
        LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE,
        LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION,
        TAB_LIBRARY,
    )
    from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
        LibraryUnavailableConversationInspection,
        LibraryUnavailableConversationsBrowse,
        RoleplayReturnTarget,
        serialize_library_unavailable_browse,
        serialize_library_unavailable_inspection,
    )

    _scratch_env(monkeypatch, tmp_path)
    app = TldwCli()
    _configure_native_ready_console(app)
    async with app.run_test(size=size) as pilot:
        await asyncio.wait_for(_boot_settled(app, pilot), 30)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        database = app.chachanotes_db
        actor = database.add_character_card({"name": "Synthetic lost actor"})
        authority = database.get_local_authority_id()
        database.add_conversation(
            {
                "id": "synthetic-return-chat",
                "title": "Synthetic unavailable chat",
                "character_id": actor,
                "assistant_kind": "character",
                "assistant_id": str(actor),
                "assistant_authority_id": authority,
            }
        )
        with database.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE conversations SET assistant_id = 'Historical actor', assistant_authority_id = NULL WHERE id = ?",
                ("synthetic-return-chat",),
            )
        key = UnresolvedConversationKey(authority, "synthetic-return-chat")
        anchor = RoleplayReturnTarget.console_context_character()
        if route_kind == "inspection":
            context = {
                LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION: serialize_library_unavailable_inspection(
                    LibraryUnavailableConversationInspection(key, anchor)
                )
            }
        else:
            context = {
                LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE: serialize_library_unavailable_browse(
                    LibraryUnavailableConversationsBrowse(key, anchor)
                )
            }
        console._set_console_rail_preference(
            left_open=False, section_updates={"character": False}
        )
        prior_preferences = deepcopy(
            app.app_config.get("console", {}).get("rail_state", {})
        )
        app.post_message(NavigateToScreen(TAB_LIBRARY, context))
        for _ in range(100):
            await pilot.pause(0.05)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-character-back-console"
            ):
                break
        library = app.screen
        back = library.query_one("#library-character-back-console", Button)
        back.focus()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in library._compositor.render_strips())
        assert "Back to Console" in painted
        app.save_screenshot(
            filename=f"library-return-{route_kind}-{size[0]}x{size[1]}.svg",
            path=str(tmp_path),
        )
        (tmp_path / "return-paint.txt").write_text(painted)
        if size == (80, 24):
            original_flush = library.flush_pending_work
            vetoed = asyncio.Event()

            async def veto_departure():
                vetoed.set()
                return False

            library.flush_pending_work = veto_departure
            back.press()
            await asyncio.wait_for(vetoed.wait(), 2)
            await pilot.pause()
            assert app.screen is library
            assert library.query_one("#library-character-back-console") is back
            library.flush_pending_work = original_flush
        back.press()
        for _ in range(100):
            await pilot.pause(0.05)
            if app.screen is console and (
                size[0] == 52
                or getattr(app.focused, "id", None) == "console-character-search"
            ):
                break
        assert app.screen is console
        await pilot.pause(0.3)
        if size[0] == 52:
            assert not console.query_one("#console-left-rail").display
            assert getattr(app.focused, "id", None) != "console-character-search"
            assert app.focused is not None and app.focused.visible
            assert console.query_one("#console-native-composer") in (
                app.focused,
                *app.focused.ancestors,
            )
            assert (
                console._pending_character_return_focus_id
                == "console-context-character"
            )
            app.save_screenshot(
                filename="console-return-fallback-52x20.svg", path=str(tmp_path)
            )
            fallback = app.focused
            await pilot.resize_terminal(80, 24)
            await pilot.pause(0.3)
            assert app.focused is fallback
            assert (
                console._pending_character_return_focus_id
                == "console-context-character"
            )
        else:
            # Refresh-on-resume may replace the first focused Input. Wait for
            # the final owned node to paint, without scrolling it from the test.
            from textual.errors import NoWidget

            for _ in range(60):
                await pilot.pause(0.05)
                search = console.query_one("#console-character-search")
                try:
                    geometry = console.find_widget(search)
                except NoWidget:
                    continue
                if app.focused is search and geometry.clip.contains_region(
                    search.region
                ):
                    break
            search = console.query_one("#console-character-search")
            assert app.focused is search
            geometry = console.find_widget(search)
            assert geometry.clip.contains_region(search.region)
            painted_focus = "\n".join(
                strip.text[search.region.x : search.region.right]
                for strip in console._compositor.render_strips()[
                    search.region.y : search.region.bottom
                ]
            )
            assert "Search chats" in painted_focus
            app.save_screenshot(
                filename=f"console-return-{route_kind}-{size[0]}x{size[1]}.svg",
                path=str(tmp_path),
            )
        assert (
            app.app_config.get("console", {}).get("rail_state", {}) == prior_preferences
        )
        if size[0] == 52:
            console._toggle_console_rail_section("character", next_open=True)
            await pilot.pause(0.3)
            assert console._pending_character_return_focus_id is None
            assert app.focused is console.query_one("#console-character-search")
        console._toggle_console_rail_section("character", next_open=False)
        assert not console._character_context.return_reveal
        assert not console.query_one("#console-rail-section-body-character").display
        await pilot.pause()
        app.post_message(NavigateToScreen(TAB_LIBRARY, {"mode": "conversations"}))
        for _ in range(100):
            await pilot.pause(0.05)
            if app.screen is library:
                break
        assert not library.query("#library-character-back-console")
        settled_frames = 0
        for _ in range(40):
            await pilot.pause(0.05)
            if (
                not library._recompose_required
                and not library._conversations_state.loading
            ):
                settled_frames += 1
                if settled_frames == 3:
                    break
            else:
                settled_frames = 0
        assert settled_frames == 3
        assert not library._recompose_required
        assert not library._conversations_state.loading


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fail_visibility,prior_resume_gate", [(False, False), (True, False), (True, True)]
)
async def test_character_activation_uses_cached_console_and_preserves_rollback(
    monkeypatch, tmp_path, fail_visibility, prior_resume_gate
):
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    # Configure synthetic readiness, not a server probe or a generated send.
    # The real setup modal intentionally prevents composer focus otherwise.
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console

    _configure_native_ready_console(app)
    async with app.run_test(size=(170, 48)) as pilot:
        await asyncio.wait_for(_boot_settled(app, pilot), timeout=30)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        runtime = app.console_runtime
        store = console._workspace._ensure_console_chat_store()
        prior_session = store.active_session_id
        database = app.chachanotes_db
        character_id = database.add_character_card({"name": "Exact cached actor"})
        authority = database.get_local_authority_id()
        conversation_id = "cached-character-target"
        assert database.add_conversation(
            {
                "id": conversation_id,
                "title": "Exact cached target",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
            }
        )
        request = CharacterConversationActivationRequest(
            LocalCharacterConversationTarget(
                ResolvedLocalCharacterKey(authority, character_id), conversation_id
            ),
            authority,
            database.get_character_conversation_search_revision(),
        )
        app.post_message(NavigateToScreen(TAB_PERSONAS))
        for _ in range(100):
            await pilot.pause(0.05)
            if type(app.screen).__name__ == "PersonasScreen":
                break
        roleplay = app.screen
        assert type(roleplay).__name__ == "PersonasScreen"
        assert runtime.view is console
        if fail_visibility:
            monkeypatch.setattr(
                console._workspace,
                "_character_conversation_target_visible",
                lambda _request: False,
            )

        console._resume_navigation_startup_in_progress = prior_resume_gate
        result = await app.activate_character_conversation_from_roleplay(
            request, asyncio.Event(), lambda _phase: None
        )
        await pilot.pause()

        assert (
            app._reusable_navigation_screen(TAB_CHAT, app._current_runtime_identity())
            is console
        )
        assert runtime.view is console
        assert console._resume_navigation_startup_in_progress is prior_resume_gate
        if fail_visibility:
            assert result.kind is ConsoleActivationResultKind.FAILED
            assert app.screen is roleplay
            assert store.active_session_id == prior_session
            assert all(
                session.persisted_conversation_id != conversation_id
                for session in store.sessions()
            )
        else:
            assert result.kind is ConsoleActivationResultKind.OPENED
            assert app.screen is console
            assert console._workspace._character_conversation_target_visible(request)
            await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
            await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
            assert app.screen is console
            assert console._workspace._character_conversation_target_visible(request)


@pytest.mark.asyncio
async def test_reused_library_accepts_character_repair_and_returns_to_roleplay(
    monkeypatch, tmp_path
):
    from textual.widgets import Button, Select

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        UnresolvedConversationKey,
    )
    from tldw_chatbook.Constants import (
        LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR,
        TAB_LIBRARY,
    )
    from tldw_chatbook.UI.Library_Modules.library_character_repair_controller import (
        LibraryCharacterRepairDialog,
    )
    from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
        LibraryCharacterRepairContext,
        RoleplayReturnTarget,
        serialize_library_character_repair_context,
    )

    _scratch_env(monkeypatch, tmp_path)
    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await asyncio.wait_for(_boot_settled(app, pilot), timeout=30)
        await _press_until_screen(pilot, "ctrl+3", "LibraryScreen")
        library = app.screen
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        database = app.chachanotes_db
        character_id = database.add_character_card({"name": "Repair target"})
        authority = database.get_local_authority_id()
        database.add_conversation(
            {
                "id": "reused-library-repair",
                "title": "Repair target",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
            }
        )
        with database.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE conversations SET assistant_id = 'Historical target', assistant_authority_id = NULL WHERE id = ?",
                ("reused-library-repair",),
            )
        original = database.get_conversation_by_id("reused-library-repair")
        context = LibraryCharacterRepairContext(
            UnresolvedConversationKey(authority, "reused-library-repair"),
            original["version"],
            "Historical target",
            RoleplayReturnTarget.personas_filter(),
        )
        app.post_message(
            NavigateToScreen(
                TAB_LIBRARY,
                {
                    LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR: serialize_library_character_repair_context(
                        context
                    )
                },
            )
        )
        for _ in range(100):
            await pilot.pause(0.05)
            if isinstance(app.screen, LibraryCharacterRepairDialog):
                break
        dialog = app.screen
        assert isinstance(dialog, LibraryCharacterRepairDialog)
        assert app.screen_stack[-2] is library
        for _ in range(100):
            await pilot.pause(0.05)
            if library._navigation_controller.repair_controller.context is None:
                continue
            if not dialog.query_one(
                "#library-character-repair-candidate", Select
            ).disabled:
                break
        assert library._navigation_controller.repair_controller.context == context
        dialog.query_one("#library-character-repair-candidate", Select).value = str(
            character_id
        )
        await pilot.pause()
        apply = dialog.query_one("#library-character-repair-apply", Button)
        apply.press()
        await pilot.pause()
        assert str(apply.label) == "Confirm repair"
        apply.press()
        for _ in range(100):
            await pilot.pause(0.05)
            if type(app.screen).__name__ == "PersonasScreen":
                break
        assert type(app.screen).__name__ == "PersonasScreen"
        assert library._navigation_controller.pending_repair_context is None
        assert (
            database.get_conversation_by_id("reused-library-repair")[
                "assistant_authority_id"
            ]
            == authority
        )
