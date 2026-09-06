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
