"""Installed switcher activation must reveal an exact real Console target."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
import pytest_asyncio
from textual.widgets import Button, Input

from Tests.UI.test_console_workbench_contract import (
    ConsoleHarness,
    _configure_native_ready_console,
)
from Tests.UI.test_library_inspection_admission import library  # noqa: F401
from tldw_chatbook.Chat.console_conversation_activation import (
    ConsoleActivationResultKind,
)
from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)


@pytest_asyncio.fixture
async def activation_library(library):  # noqa: F811 - imported pytest fixture
    """End this Console fixture's runtime before its borrowed file owner closes."""
    owner, _, _ = library
    try:
        yield library
    finally:
        runtime = owner.console_runtime
        await asyncio.wait_for(runtime.dispose(), 5)
        # The fixture owns this runtime's entire temporary AgentRuns file.
        from Tests.conftest import _close_database_instance

        _close_database_instance(runtime._agent_runs_db)


async def _until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


def _seed(owner, db):
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
        LocalChatDictionaryService,
    )

    _configure_native_ready_console(owner)
    owner.chat_dictionary_scope_service.local_service = LocalChatDictionaryService(db)
    card = db.add_character_card({"name": "Activation fixture"})
    record = db.get_conversation_by_id("exact")
    db.update_conversation(
        "exact",
        {
            "character_id": card,
            "assistant_kind": "character",
            "assistant_id": str(card),
            "assistant_authority_id": db.get_local_authority_id(),
        },
        record["version"],
    )
    db.add_message(
        {
            "id": "activation-message",
            "conversation_id": "exact",
            "role": "user",
            "sender": "user",
            "content": "Exact selected content",
        }
    )


async def _open_switcher(chat, host):
    await chat.action_open_console_session_switcher(
        initial_mode=SwitcherMode.CHARACTER_CHATS,
        initial_character_query="Exact",
    )
    await _until(
        lambda: (
            isinstance(host.screen, ConsoleSessionSwitcherModal)
            and not host.screen._query_pending
            and host.screen._entries
        )
    )
    return host.screen


@pytest.mark.asyncio
async def test_installed_owner_opens_cold_reuses_warm_and_keeps_composer_focus(
    activation_library,
):
    owner, _, db = activation_library
    _seed(owner, db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        # A non-composer opener reproduces Continue-search focus restoration.
        opener = Button("Continue search", id="activation-opener")
        await chat.mount(opener)
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        target_session = None
        for _ in range(2):
            chat.set_focus(opener)
            modal = await _open_switcher(chat, host)
            target = modal._entries[0].target
            await pilot.press("enter")
            await _until(
                lambda modal=modal: (
                    modal._activation_task is not None and modal._activation_task.done()
                )
            )
            assert host.screen is chat, modal._activation_failure_kind
            await pilot.pause()
            assert chat._workspace._character_conversation_target_visible(target)
            assert chat.focused is chat.query_one("#console-native-composer")
            matching = [
                s for s in store.sessions() if s.persisted_conversation_id == "exact"
            ]
            assert len(matching) == 1
            if target_session is not None:
                assert matching[0] is target_session
            target_session = matching[0]
            assert target_session.id != prior


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["settings", "target", "payload", "none"])
async def test_token_preparation_is_off_loop_before_activation_and_fences_settings(
    activation_library,
    monkeypatch,
    change,
):
    owner, _, db = activation_library
    _seed(owner, db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)):
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        # Seed a real inactive runtime through the canonical hydrator, then
        # exercise the existing warm-session open path with a held estimator.
        from tldw_chatbook.Chat import console_session_settings
        from tldw_chatbook.Chat.console_conversation_hydration import (
            hydrate_console_session,
            load_console_conversation_tree,
        )

        tree = await load_console_conversation_tree(owner, "exact")
        hydration = chat._workspace._console_session_settings_for_resume(
            tree["conversation"]
        )
        session = await hydrate_console_session(
            app=owner,
            store=store,
            conversation_id="exact",
            tree=tree,
            settings=hydration.settings,
            generation_durable_snapshot=hydration.durable_snapshot,
            generation_metadata_status=hydration.metadata_status,
            activate=False,
        )
        entered, release = threading.Event(), threading.Event()
        main_thread = threading.get_ident()
        estimate = console_session_settings._estimate_tokens_locally

        def held(messages, model, provider):
            assert threading.get_ident() != main_thread
            entered.set()
            assert release.wait(3), "bounded estimator release"
            return estimate(messages, model, provider)

        monkeypatch.setattr(console_session_settings, "_estimate_tokens_locally", held)
        task = asyncio.create_task(
            chat._workspace.open_console_workspace_conversation("exact")
        )
        try:
            await _until(lambda: entered.is_set() or task.done())
            assert entered.is_set(), "open never prepared the tokenizer off-loop"
            assert store.active_session_id == prior
            await asyncio.sleep(0.02)  # loop remains actionable while worker held
            if change == "settings":
                store.replace_session_settings(
                    session.id,
                    replace(store.session_settings(session.id), model="new-model"),
                )
            elif change == "target":
                session.persisted_conversation_id = "changed"
            elif change == "payload":
                store._bump_payload_revision(session.id)
        finally:
            release.set()
            await asyncio.wait_for(task, 5)
        if change == "none":
            assert store.active_session_id == session.id
            assert task.result() is True
        else:
            assert store.active_session_id == prior
            assert task.result() is None  # incumbent warm-open transient-failure result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "cancel",
        "open-failure",
        "transcript",
        "overlay",
        "away-back",
        "generation",
    ],
)
async def test_installed_activation_rejects_stale_or_failed_owner_and_preserves_visit(
    activation_library,
    monkeypatch,
    fault,
):
    from textual.screen import Screen

    from tldw_chatbook.Chat.console_conversation_activation import (
        ConsoleActivationCommit,
    )

    owner, _, db = activation_library
    _seed(owner, db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        modal = await _open_switcher(chat, host)
        workspace = chat._workspace
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        entered, release = asyncio.Event(), asyncio.Event()
        name = (
            "_revalidate_character_conversation_target"
            if fault == "cancel"
            else "_open_character_conversation_activation"
        )
        original = getattr(workspace, name)

        async def held(request):
            result = await original(request)
            entered.set()
            await asyncio.wait_for(release.wait(), 3)
            if fault == "open-failure":
                return ConsoleActivationCommit(False, result.owned_runtime_token)
            return result

        monkeypatch.setattr(workspace, name, held)
        await pilot.press("enter")
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if fault == "cancel":
                await pilot.press("escape")
            elif fault == "transcript":
                chat.query_one("#console-native-transcript")._session_identity = "wrong"
            elif fault in {"overlay", "away-back"}:
                await host.push_screen(Screen())
                if fault == "away-back":
                    await host.pop_screen()
            elif fault == "generation":
                modal._request_generation += 1
        finally:
            release.set()
            await asyncio.wait_for(modal._activation_task, 5)
        assert store.active_session_id == prior
        assert not [
            s for s in store.sessions() if s.persisted_conversation_id == "exact"
        ]
        assert modal.is_mounted
        assert modal.query_one("#console-switcher-query", Input).value == "Exact"
        assert modal._entries[modal._candidate_index].target.conversation_id == "exact"
        assert not modal._activation_completion_consumed
        if fault != "overlay":
            assert host.screen is modal


@pytest.mark.asyncio
async def test_ordinary_caller_cannot_claim_underlay_and_source_consumes_once(
    activation_library,
    monkeypatch,
):
    from tldw_chatbook.Chat.console_conversation_activation import (
        CharacterConversationActivationRequest,
    )

    owner, _, db = activation_library
    _seed(owner, db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        modal = await _open_switcher(chat, host)
        target = modal._entries[0].target
        request = CharacterConversationActivationRequest(
            target,
            db.get_local_authority_id(),
            modal._character_data_revision,
        )
        result = await chat._workspace.activate_character_conversation(request)
        assert result.kind is ConsoleActivationResultKind.FAILED
        assert host.screen is modal
        completion = modal.complete_character_activation
        consumes = []

        def observed(request, result, **kwargs):
            consumes.append(completion(request, result, **kwargs))
            consumes.append(completion(request, result, **kwargs))
            return consumes[0]

        monkeypatch.setattr(modal, "complete_character_activation", observed)
        await pilot.press("enter")
        await _until(
            lambda: modal._activation_task is not None and modal._activation_task.done()
        )
        assert host.screen is chat
        assert consumes == [True, False]
        assert not completion(request, result, console=chat)
