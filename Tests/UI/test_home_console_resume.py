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
        # Drain the queued ScreenResume before starting its exclusive worker group.
        await asyncio.wait_for(pilot.pause(), timeout=8)
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


def _seed_resume_target(app, title):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyCandidate,
    )

    conversation_id = app.local_chat_conversation_service.create_conversation(
        title=title, runtime_backend="local", scope_type="global"
    )
    ChatPersistenceService(app.chachanotes_db).console_library_policy_repository.insert(
        conversation_id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.NEVER, ConsoleAssistantLibraryAccess.BLOCKED
        ),
    )
    return conversation_id


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.parametrize("stage", ["load", "presentation"])
@private_profile_test
async def test_suspended_resume_settles_and_retries_exact_target(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, stage: str
):
    from textual.worker import WorkerState

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.UI.Console_Modules import workspace
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

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
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        store = console._ensure_console_chat_store()
        prior = store.active_session_id
        target_id = _seed_resume_target(app, "Suspended exact Home target")
        entered, release = asyncio.Event(), asyncio.Event()
        original_load = workspace.load_console_conversation_tree
        original_sync = ChatScreen._sync_native_console_chat_ui

        async def held_load(owner, conversation_id):
            if (
                stage == "load"
                and conversation_id == target_id
                and not entered.is_set()
            ):
                entered.set()
                await release.wait()
            return await original_load(owner, conversation_id)

        async def held_sync(screen):
            active = (
                store._session_or_raise(store.active_session_id)
                if store.active_session_id is not None
                else None
            )
            if (
                stage == "presentation"
                and active is not None
                and active.persisted_conversation_id == target_id
                and not entered.is_set()
            ):
                entered.set()
                await release.wait()
            await original_sync(screen)

        monkeypatch.setattr(workspace, "load_console_conversation_tree", held_load)
        monkeypatch.setattr(ChatScreen, "_sync_native_console_chat_ui", held_sync)
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        await app.screen._refresh_home_content_snapshot().wait()
        app.screen.query_one("#home-resume-latest", Button).press()
        await asyncio.wait_for(entered.wait(), 15)
        worker = next(
            worker
            for worker in app.workers
            if worker.group == "console-resume-navigation-startup"
        )
        try:
            await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
            assert worker.is_cancelled, "hidden Console retained its Resume worker"
            await asyncio.wait_for(
                asyncio.gather(worker.wait(), return_exceptions=True), 10
            )
            assert worker.state is WorkerState.CANCELLED
            assert store.active_session_id == prior
            assert console._pending_resume_local_conversation_id == target_id
            assert all(
                session.persisted_conversation_id != target_id
                for session in store.sessions()
            )
            await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
            await _until(
                lambda: (
                    store.active_session_id is not None
                    and store._session_or_raise(
                        store.active_session_id
                    ).persisted_conversation_id
                    == target_id
                )
            )
            await _until(lambda: not console._resume_navigation_startup_in_progress)
            assert app.screen is console
            assert console._pending_resume_local_conversation_id is None
            assert (
                len(
                    [
                        session
                        for session in store.sessions()
                        if session.persisted_conversation_id == target_id
                    ]
                )
                == 1
            )
        finally:
            release.set()
            await asyncio.gather(worker.wait(), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.ui
@private_profile_test
async def test_character_commit_supersedes_queued_home_resume(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        LocalCharacterConversationTarget,
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.Chat.console_conversation_activation import (
        CharacterConversationActivationRequest,
        ConsoleActivationResultKind,
    )
    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.Constants import TAB_PERSONAS
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    assert save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "general": {"default_tab": "home"},
            "model_catalog": {"auto_refresh_enabled": False},
        }
    )
    app = TldwCli()
    _configure_native_ready_console(app)
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        db = app.chachanotes_db
        character_id = db.add_character_card({"name": "New chosen character"})
        authority = db.get_local_authority_id()
        character_chat = "new-character-over-retained-home"
        assert db.add_conversation(
            {
                "id": character_chat,
                "title": "Newer character choice",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
            }
        )
        old_home = _seed_resume_target(app, "Older queued Home target")
        release_hedge = asyncio.Event()
        actual_timer = console.set_timer

        def gate_ordered_timer(interval, callback, **kwargs):
            if getattr(callback, "__name__", "") == "_start_resume_navigation_startup":

                async def after_character_commit():
                    await release_hedge.wait()
                    callback()

                return actual_timer(interval, after_character_commit, **kwargs)
            return actual_timer(interval, callback, **kwargs)

        monkeypatch.setattr(console, "set_timer", gate_ordered_timer)
        monkeypatch.setattr(console, "CONSUMER_SETTLE_HEDGE_SECONDS", 30)
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        await app.screen._refresh_home_content_snapshot().wait()
        app.screen.query_one("#home-resume-latest", Button).press()
        await _until(
            lambda: app.screen is console and console._console_resume_handoff_timers
        )
        assert console._pending_resume_local_conversation_id == old_home
        app.post_message(NavigateToScreen(TAB_PERSONAS))
        await _until(lambda: type(app.screen).__name__ == "PersonasScreen")
        assert console._pending_resume_local_conversation_id == old_home
        monkeypatch.setattr(console, "CONSUMER_SETTLE_HEDGE_SECONDS", 0.15)
        activation = CharacterConversationActivationRequest(
            LocalCharacterConversationTarget(
                ResolvedLocalCharacterKey(authority, character_id), character_chat
            ),
            authority,
            db.get_character_conversation_search_revision(),
        )
        result = await app.activate_character_conversation_from_roleplay(
            activation, asyncio.Event(), lambda _phase: None
        )
        release_hedge.set()
        await _until(lambda: not console._resume_navigation_startup_in_progress)
        await pilot.pause()
        assert result.kind is ConsoleActivationResultKind.OPENED
        assert app.screen is console
        assert console._workspace._character_conversation_target_visible(activation)
        assert console._pending_resume_local_conversation_id is None
        assert not console._resume_navigation_startup_in_progress
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert app.screen is console
        assert console._workspace._character_conversation_target_visible(activation)


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.parametrize("successor", ["ordinary_return", "character_commit"])
@private_profile_test
async def test_cancelled_resume_rollback_survives_later_navigation(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, successor: str
):
    from textual.worker import NoActiveWorker, get_current_worker

    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        LocalCharacterConversationTarget,
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.Chat.console_conversation_activation import (
        CharacterConversationActivationRequest,
        ConsoleActivationResultKind,
    )
    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.Constants import TAB_PERSONAS
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    assert save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "general": {"default_tab": "home"},
            "model_catalog": {"auto_refresh_enabled": False},
        }
    )
    app = TldwCli()
    _configure_native_ready_console(app)
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        store = console._ensure_console_chat_store()
        prior = store.active_session_id
        target_id = _seed_resume_target(app, "Interrupted Home owner")
        entered, rollback_entered = asyncio.Event(), asyncio.Event()
        release, rollback_release = asyncio.Event(), asyncio.Event()
        original_sync = ChatScreen._sync_native_console_chat_ui

        async def held_sync(screen):
            try:
                resume_sync = (
                    get_current_worker().group == "console-resume-navigation-startup"
                )
            except NoActiveWorker:
                resume_sync = False
            if resume_sync:
                active = store._session_or_raise(store.active_session_id)
                if (
                    active.persisted_conversation_id == target_id
                    and not entered.is_set()
                ):
                    entered.set()
                    await release.wait()
                elif (
                    active.id == prior
                    and entered.is_set()
                    and not rollback_entered.is_set()
                ):
                    rollback_entered.set()
                    await rollback_release.wait()
            await original_sync(screen)

        monkeypatch.setattr(ChatScreen, "_sync_native_console_chat_ui", held_sync)
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        await app.screen._refresh_home_content_snapshot().wait()
        app.screen.query_one("#home-resume-latest", Button).press()
        await asyncio.wait_for(entered.wait(), 15)
        worker = console._resume_navigation_startup_worker
        operation = None
        try:
            await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
            await asyncio.wait_for(rollback_entered.wait(), 10)
            assert store.active_session_id == prior
            if successor == "ordinary_return":
                await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
                await pilot.pause(0.3)
                await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
                assert not worker.is_finished, "later visit cancelled rollback again"
                assert console._pending_resume_local_conversation_id == target_id
                rollback_release.set()
                await asyncio.gather(worker.wait(), return_exceptions=True)
                await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
                await _until(
                    lambda: (
                        store._session_or_raise(
                            store.active_session_id
                        ).persisted_conversation_id
                        == target_id
                    )
                )
            else:
                db = app.chachanotes_db
                actor = db.add_character_card({"name": "Rollback successor"})
                authority = db.get_local_authority_id()
                character_id = "rollback-successor-chat"
                assert db.add_conversation(
                    {
                        "id": character_id,
                        "title": "Rollback successor",
                        "character_id": actor,
                        "assistant_kind": "character",
                        "assistant_id": str(actor),
                        "assistant_authority_id": authority,
                    }
                )
                app.post_message(NavigateToScreen(TAB_PERSONAS))
                await _until(lambda: type(app.screen).__name__ == "PersonasScreen")
                activation = CharacterConversationActivationRequest(
                    LocalCharacterConversationTarget(
                        ResolvedLocalCharacterKey(authority, actor), character_id
                    ),
                    authority,
                    db.get_character_conversation_search_revision(),
                )
                retiring = asyncio.Event()
                actual_retire = console._retire_resume_navigation_startup

                async def observe_retirement():
                    retiring.set()
                    return await actual_retire()

                monkeypatch.setattr(
                    console, "_retire_resume_navigation_startup", observe_retirement
                )
                operation = asyncio.create_task(
                    app.activate_character_conversation_from_roleplay(
                        activation, asyncio.Event(), lambda _phase: None
                    )
                )
                await asyncio.wait_for(retiring.wait(), 15)
                await pilot.pause(0.2)
                assert not worker.is_finished, (
                    "new character interrupted prior rollback"
                )
                assert not operation.done()
                assert store.active_session_id == prior
                rollback_release.set()
                result = await asyncio.wait_for(operation, 15)
                assert result.kind is ConsoleActivationResultKind.OPENED
                assert console._workspace._character_conversation_target_visible(
                    activation
                )
                assert console._pending_resume_local_conversation_id is None
            await _until(lambda: not console._resume_navigation_startup_in_progress)
        finally:
            release.set()
            rollback_release.set()
            await asyncio.gather(worker.wait(), return_exceptions=True)
            if operation is not None:
                await asyncio.gather(operation, return_exceptions=True)
