"""Home exact Resume across the real reusable Console and independent Buddy."""

import asyncio

import pytest
from textual.widgets import Button, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_screen_reuse import _boot_settled, _press_until_screen


async def _until(predicate, *, timeout=8):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, (
            "Home Resume did not settle"
        )
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "reused"])
@private_profile_test
async def test_home_resume_keeps_saved_identity_and_independent_buddy_draft(
    request: pytest.FixtureRequest, warm: bool
):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyCandidate,
    )
    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    assert save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "general": {"default_tab": "home"},
            "model_catalog": {"auto_refresh_enabled": False},
        }
    )
    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        ids = []
        for title in ("Buddy owner A", "Home resume B"):
            conversation_id = app.local_chat_conversation_service.create_conversation(
                title=title, runtime_backend="local", scope_type="global"
            )
            ChatPersistenceService(
                app.chachanotes_db
            ).console_library_policy_repository.insert(
                conversation_id,
                ConsoleLibraryPolicyCandidate(
                    ConsoleAutoRetrieve.NEVER, ConsoleAssistantLibraryAccess.BLOCKED
                ),
            )
            app.chachanotes_db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "role": "assistant",
                    "content": f"Saved history for {title}",
                }
            )
            ids.append(conversation_id)
        buddy_id, resume_id = ids
        prior_console = None
        prior_session = None
        if warm:
            await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
            prior_console = app.screen
            store = prior_console._ensure_console_chat_store()
            prior_session = store._session_or_raise(store.active_session_id)
            prior_console.query_one("#console-native-composer").load_draft(
                "Unrelated Console draft stays exact."
            )
            await pilot.pause()
            await _press_until_screen(pilot, "ctrl+1", "HomeScreen")

        binding = BuddyBinding(
            "conversation", "saved:" + buddy_id, conversation_id=buddy_id
        )
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await _until(lambda: modal.coordinator.resolve(binding) is not None)
        await pilot.pause()
        buddy_session = modal.coordinator.resolve(binding)
        modal.query_one("#buddy-reply", TextArea).load_text("Buddy A unsent draft.")
        await pilot.pause()
        modal.query_one("#buddy-close", Button).press()
        await _until(lambda: type(app.screen).__name__ == "HomeScreen")
        home = app.screen
        await home._refresh_home_content_snapshot().wait()
        await pilot.pause()
        resume = home.query_one("#home-resume-latest", Button)
        assert "Home resume B" in str(resume.label)
        resume.press()
        await _until(lambda: type(app.screen).__name__ == "ChatScreen")
        console = app.screen
        store = console._ensure_console_chat_store()
        await _until(
            lambda: (
                store.active_session_id is not None
                and store._session_or_raise(
                    store.active_session_id
                ).persisted_conversation_id
                == resume_id
            )
        )
        await _until(lambda: not console._resume_navigation_startup_in_progress)
        await pilot.pause()
        target = store._session_or_raise(store.active_session_id)
        assert target.persisted_conversation_id == resume_id
        assert (
            store.messages_for_session(target.id)[0].content
            == "Saved history for Home resume B"
        )
        assert (
            len(
                [
                    s
                    for s in store.sessions()
                    if s.persisted_conversation_id == resume_id
                ]
            )
            == 1
        )
        assert buddy_session.persisted_conversation_id == buddy_id
        assert modal.coordinator.drafts[binding] == "Buddy A unsent draft."
        if warm:
            assert console is prior_console
            assert prior_session.draft == "Unrelated Console draft stays exact."
            await console._session._activate_native_console_session(prior_session.id)
            await _until(
                lambda: (
                    console.query_one("#console-native-composer").draft_text()
                    == "Unrelated Console draft stays exact."
                )
            )
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        reopened = open_buddy_conversation(app, binding, allow_voice=False)
        await _until(lambda: reopened.coordinator.resolve(binding) is buddy_session)
        await pilot.pause()
        assert (
            reopened.query_one("#buddy-reply", TextArea).text == "Buddy A unsent draft."
        )
        reopened.query_one("#buddy-close", Button).press()
        await _until(lambda: type(app.screen).__name__ == "HomeScreen")
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.ui
@private_profile_test
async def test_shutdown_settles_resume_presentation_before_runtime_disposal(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    from textual.worker import WorkerState, get_current_worker

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyCandidate,
    )
    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    assert save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "general": {"default_tab": "home"},
            "model_catalog": {"auto_refresh_enabled": False},
        }
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original_sync = ChatScreen._sync_native_console_chat_ui

    async def held_resume_sync(screen):
        if (
            get_current_worker().group == "console-resume-navigation-startup"
            and not entered.is_set()
        ):
            entered.set()
            await release.wait()
        await original_sync(screen)

    monkeypatch.setattr(ChatScreen, "_sync_native_console_chat_ui", held_resume_sync)
    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        conversation_id = app.local_chat_conversation_service.create_conversation(
            title="Shutdown during Home Resume",
            runtime_backend="local",
            scope_type="global",
        )
        ChatPersistenceService(
            app.chachanotes_db
        ).console_library_policy_repository.insert(
            conversation_id,
            ConsoleLibraryPolicyCandidate(
                ConsoleAutoRetrieve.NEVER, ConsoleAssistantLibraryAccess.BLOCKED
            ),
        )
        await app.screen._refresh_home_content_snapshot().wait()
        await pilot.pause()
        app.screen.query_one("#home-resume-latest", Button).press()
        await asyncio.wait_for(entered.wait(), timeout=15)
        console = app.screen
        resume_worker = next(
            worker
            for worker in app.workers
            if worker.group == "console-resume-navigation-startup"
        )
        unrelated = app.run_worker(release.wait(), group="unrelated-owned-work")
        try:
            await app._shutdown_console_runtime()
            assert resume_worker.state == WorkerState.CANCELLED
            assert not console._resume_navigation_startup_in_progress
            assert not any(
                worker.group in {"console-sync", "console-resume-navigation-startup"}
                for worker in app.workers
            )
            assert not unrelated.is_cancelled
            console._start_resume_navigation_startup()
            await original_sync(console)
            await pilot.pause()
            assert not any(
                worker.group in {"console-sync", "console-resume-navigation-startup"}
                for worker in app.workers
            )
        finally:
            release.set()
            await pilot.pause()
