"""Installed switcher activation must reveal an exact real Console target."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
import pytest_asyncio
from textual.widgets import Button, Input, Static

from Tests.UI.test_console_workbench_contract import (
    ConsoleHarness,
    _configure_native_ready_console,
)
from Tests.UI.test_library_inspection_admission import library  # noqa: F401
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationPhase,
    ConsoleActivationResultKind,
)
from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)

pytestmark = pytest.mark.bootstrap_profile
_additional_activation_databases = pytest.StashKey[list[object]]()


def register_activation_database(
    request: pytest.FixtureRequest, database: object
) -> None:
    """Register an exact additional test owner after activation runtime disposal.

    Args:
        request: The test node using the activation_library fixture.
        database: Explicitly created disposable database; never inferred from app state.
    """
    request.node.stash[_additional_activation_databases].append(database)


@pytest_asyncio.fixture
async def activation_library(library, request):  # noqa: F811 - imported pytest fixture
    """End this Console fixture's runtime before its borrowed file owner closes."""
    owner, _, _ = library
    databases = []
    request.node.stash[_additional_activation_databases] = databases
    try:
        yield library
    finally:
        runtime = owner.console_runtime
        await asyncio.wait_for(runtime.dispose(), 5)
        # The fixture owns this runtime's entire temporary AgentRuns file.
        from Tests.conftest import _close_database_instance

        _close_database_instance(runtime._agent_runs_db)
        for database in databases:
            _close_database_instance(database)


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
@pytest.mark.parametrize(
    ("cause", "size"),
    [
        ("owned-quit", (120, 50)),
        ("owned-quit", (52, 20)),
        ("owned-enter", (120, 50)),
        ("owned-click", (120, 50)),
        ("foreign-question", (120, 50)),
        ("open-failure", (120, 50)),
        ("stale-owner", (120, 50)),
    ],
)
async def test_character_open_under_quit_question_explains_only_owned_interruption(
    activation_library, monkeypatch, tmp_path, cause, size
):
    """An intentional quit fence is recoverable, not an unexplained open failure."""
    from tldw_chatbook.Chat.console_conversation_activation import (
        ConsoleActivationCommit,
    )
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    owner, _, db = activation_library
    _seed(owner, db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=size) as pilot:
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        modal = await _open_switcher(chat, host)
        store = chat._ensure_console_chat_store()
        prior = store.active_session_id
        selected = modal._entries[modal._candidate_index].target
        entered, release = asyncio.Event(), asyncio.Event()
        original = chat._workspace._open_character_conversation_activation

        async def held(request):
            opened = await original(request)
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
            if cause == "open-failure":
                return ConsoleActivationCommit(False, opened.owned_runtime_token)
            return opened

        monkeypatch.setattr(
            chat._workspace, "_open_character_conversation_activation", held
        )
        await pilot.press("enter")
        await asyncio.wait_for(entered.wait(), 5)
        await _until(
            lambda: modal._activation_phase is ConsoleActivationPhase.COMMITTING
        )
        quit_worker = None
        if cause == "foreign-question":
            # Identical words are not evidence that this switcher owns the question.
            question = ConfirmationDialog(
                title="Quit while still working?", cancel_label="Wait"
            )
            await host.push_screen(question)
        else:
            assert await modal.confirm_quit() is False
            quit_worker = host.run_worker(modal.confirm_quit(), exit_on_error=False)
            await _until(lambda: isinstance(host.screen, ConfirmationDialog))
            question = host.screen
        if cause == "stale-owner":
            modal._request_generation += 1
        release.set()
        await asyncio.wait_for(modal._activation_task, 5)
        assert host.screen is question
        assert modal in host.screen_stack
        assert store.active_session_id == prior
        assert not [
            s for s in store.sessions() if s.persisted_conversation_id == "exact"
        ]
        expected = (
            "Open interrupted by quit confirmation"
            if cause.startswith("owned-")
            else "Could not open chat"
        )
        assert (
            str(modal.query_one("#console-switcher-status", Static).renderable)
            == expected
        )

        await pilot.press("escape")  # Wait, through the question's own safe action.
        if quit_worker is not None:
            assert await quit_worker.wait() is False
        await _until(lambda: host.screen is modal)
        await pilot.pause()
        assert modal.query_one("#console-switcher-query", Input).value == "Exact"
        assert modal._entries[modal._candidate_index].target == selected
        if cause.startswith("owned-"):
            modal._poll_active_projection()
            selected_button = modal._button_for_key(modal._candidate_key())
            assert "RESUME CHAT" in str(selected_button.label)
            recovery = modal.query_one("#console-switcher-recovery", Button)
            assert recovery.display and not recovery.disabled
            assert str(recovery.label) == "Retry"
            assert (
                expected
                in modal.query_one(
                    "#console-switcher-selected-detail", Static
                ).renderable.plain
            )
            host.save_screenshot(path=str(tmp_path), filename="quit-interruption.svg")
            # Retry is explicit and starts a fresh exact attempt; no automatic replay.
            monkeypatch.setattr(
                chat._workspace, "_open_character_conversation_activation", original
            )
            previous_attempt = modal._activation_task
            if cause == "owned-enter":
                modal.query_one("#console-switcher-query", Input).focus()
                await pilot.press("enter")
            elif cause == "owned-click":
                await pilot.click(selected_button)
            else:
                await pilot.click("#console-switcher-recovery")
            await _until(lambda: modal._activation_task is not previous_attempt)
            await asyncio.wait_for(modal._activation_task, 5)
            assert host.screen is chat, (
                modal._activation_failure_kind,
                str(modal.query_one("#console-switcher-status", Static).renderable),
            )
            assert chat._workspace._character_conversation_target_visible(selected)
            assert (
                len(
                    [
                        s
                        for s in store.sessions()
                        if s.persisted_conversation_id == "exact"
                    ]
                )
                == 1
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [None, "deleted", "query", "mode", "overlay", "close"]
)
async def test_interrupted_retry_refreshes_only_the_same_live_target(change):
    """A refreshed neighbor or a changed visit cannot inherit the Retry action."""
    from textual.screen import Screen

    from Tests.UI.test_console_character_switcher import (
        _character_row,
        _CharacterSwitcherApp,
    )
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationPage,
    )
    from tldw_chatbook.Chat.console_conversation_activation import (
        ConsoleConversationActivationResult,
    )

    exact = _character_row("exact", "Exact", "2026-09-01T12:00:00Z")
    neighbor = _character_row("neighbor", "Exact neighbor", "2026-09-01T12:00:00Z")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    opened = []

    async def load(**_kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
        rows = (neighbor,) if calls > 1 and change == "deleted" else (exact, neighbor)
        return CharacterConversationPage(rows, len(rows), None, calls)

    async def activate(request, _cancellation):
        opened.append(request)
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED, request.target, False
        )

    host = _CharacterSwitcherApp(
        character_loader=load,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
        initial_character_query="Exact",
    )
    async with host.run_test() as pilot:
        modal = host.screen
        await _until(lambda: modal._entries and not modal._query_pending)
        modal._committed_character_result = modal._entries[0]
        modal._activation_interrupted_by_quit = True
        modal._show_activation_failure(ConsoleActivationResultKind.FAILED)
        retry = asyncio.create_task(
            modal._recover_character_activation(
                Button.Pressed(modal.query_one("#console-switcher-recovery", Button))
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            refresh_attempt = modal._activation_task
            if change == "query":
                modal.query_one("#console-switcher-query", Input).value = "Other"
            elif change == "mode":
                modal._mode = SwitcherMode.ACTIVE
            elif change == "overlay":
                await host.push_screen(Screen())
            elif change == "close":
                modal.dismiss_safe_once(None)
            release.set()
            await asyncio.wait_for(retry, 5)
            await asyncio.wait_for(refresh_attempt, 5)
            await pilot.pause()
            if change is None:
                await asyncio.wait_for(modal._activation_task, 5)
                assert len(opened) == 1
                assert opened[0].target == exact.target
                assert opened[0].data_revision == 2
            else:
                assert not opened
        finally:
            release.set()
            await asyncio.gather(retry, return_exceptions=True)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_queued_switcher_acknowledges_only_its_own_commit(
    activation_library, monkeypatch, cancel
):
    owner, _, db = activation_library
    _seed(owner, db)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        # Complete genuine first-use hydration before capturing both equal
        # requests: it persists settings and legitimately advances DB revision.
        await chat._workspace.open_console_workspace_conversation("exact")
        modal = await _open_switcher(chat, host)
        workspace = chat._workspace
        request = CharacterConversationActivationRequest(
            modal._entries[0].target,
            db.get_local_authority_id(),
            modal._character_data_revision,
        )
        a_open, b_validate, b_open = (asyncio.Event() for _ in range(3))
        release_a, release_validate, release_b = (asyncio.Event() for _ in range(3))
        actual_open = workspace._open_character_conversation_activation
        actual_validate = workspace._revalidate_character_conversation_target
        opens = []

        async def validate(candidate):
            result = await actual_validate(candidate)
            if candidate is not request:
                b_validate.set()
                await asyncio.wait_for(release_validate.wait(), 5)
            return result

        async def open_target(candidate):
            opens.append(candidate)
            if candidate is request:
                result = await actual_open(candidate)
                a_open.set()
                await asyncio.wait_for(release_a.wait(), 5)
                return result
            b_open.set()
            await asyncio.wait_for(release_b.wait(), 5)
            return await actual_open(candidate)

        monkeypatch.setattr(
            workspace, "_revalidate_character_conversation_target", validate
        )
        monkeypatch.setattr(
            workspace, "_open_character_conversation_activation", open_target
        )
        ordinary = asyncio.create_task(
            workspace.activate_character_conversation(request)
        )
        try:
            await asyncio.wait_for(a_open.wait(), 5)
            await pilot.press("enter")
            await pilot.pause()
            assert modal._activation_request == request
            assert modal._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE
            assert len(opens) == 1
            if cancel:
                await pilot.press("escape")
                assert modal._activation_cancellation.is_set()
            release_a.set()
            await asyncio.wait_for(b_validate.wait(), 5)
            assert modal._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE
            release_validate.set()
            if not cancel:
                await asyncio.wait_for(b_open.wait(), 5)
                await _until(
                    lambda: modal._activation_phase is ConsoleActivationPhase.COMMITTING
                )
                await pilot.press("escape")
                assert not modal._activation_cancellation.is_set()
        finally:
            release_a.set()
            release_validate.set()
            release_b.set()
            await asyncio.wait_for(ordinary, 5)
            if modal._activation_task is not None:
                await asyncio.wait_for(modal._activation_task, 5)
        assert ordinary.result().kind is ConsoleActivationResultKind.FAILED
        if cancel:
            assert len(opens) == 1
            assert host.screen is modal
            assert not modal._activation_completion_consumed
        else:
            assert len(opens) == 2
            assert host.screen is chat
            assert workspace._character_conversation_target_visible(request)
