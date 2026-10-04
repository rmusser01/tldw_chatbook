"""Bounded native chat creation: approval, custody and dispatch boundaries."""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Chat.test_console_chat_create_integration import _FakeConfirm, _FakeExecutor
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_agent_bridge import build_chat_create_tool_closures


def _prepare(payload):
    return {
        **payload,
        "_grant_scope": (
            "incarnation-1",
            "new_chat",
            "global" if payload.get("destination") == "casual" else "workspace",
            None if payload.get("destination") == "casual" else "workspace-1",
            payload.get("mode", "draft"),
        ),
        "_creation_token": object(),
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"destination": "other-workspace"},
        {"destination": 1},
        {"mode": "repeat"},
        {"mode": False},
        {"title": None},
        {"title": "x" * 121},
        {"opening_prompt": {}},
        {"opening_prompt": "x" * 20_001},
        {"instructions": []},
        {"instructions": "x" * 20_001},
        {"mode": "start", "opening_prompt": " \n\t"},
    ],
)
def test_invalid_new_chat_never_asks_or_executes(arguments):
    """Bad public fields must fail before acquiring any approval authority."""
    confirm = _FakeConfirm([{"allow": True, "remember": False}])
    executor = _FakeExecutor()
    prepared = []

    def prepare(payload):
        prepared.append(payload)
        return _prepare(payload)

    _, new_chat = build_chat_create_tool_closures(
        confirm=confirm,
        execute=executor,
        prepare=prepare,
        session_id="s1",
        run_id="message-1",
    )
    assert not new_chat(arguments).ok
    assert not prepared
    assert not confirm.payloads
    assert not executor.calls


def test_remembered_draft_does_not_grant_start_or_casual():
    confirm = _FakeConfirm([{"allow": True, "remember": True}])
    executor = _FakeExecutor()
    _, new_chat = build_chat_create_tool_closures(
        confirm=confirm,
        execute=executor,
        prepare=_prepare,
        session_id="s1",
        run_id="message-1",
    )
    assert new_chat({"opening_prompt": "draft"}).ok
    assert new_chat({"opening_prompt": "later draft"}).ok
    assert not new_chat({"opening_prompt": "run", "mode": "start"}).ok
    assert not new_chat({"destination": "casual"}).ok
    assert len(executor.calls) == 2


def test_new_chat_uses_invocation_run_and_drops_model_internal_keys():
    """A forged model field cannot replace run lineage or the prepared token."""
    executor = _FakeExecutor()
    confirm = _FakeConfirm([{"allow": True, "remember": False}])
    _, new_chat = build_chat_create_tool_closures(
        confirm=confirm,
        execute=executor,
        prepare=_prepare,
        session_id="s1",
        run_id="message-1",
    )
    with use_run_id("actual-primary-run"):
        result = new_chat(
            {
                "opening_prompt": "go",
                "mode": "start",
                "source_run_id": "forged-run",
                "session_id": "forged-session",
                "_grant_scope": "forged-grant",
                "_creation_token": "forged-token",
            }
        )
    assert result.ok
    executed = executor.calls[0]
    assert executed["source_run_id"] == "actual-primary-run"
    assert executed["source_message_id"] == "message-1"
    assert executed["session_id"] == "s1"
    assert executed["_creation_token"] != "forged-token"
    assert executed["_grant_scope"] == (
        "incarnation-1",
        "new_chat",
        "workspace",
        "workspace-1",
        "start",
    )


def test_new_chat_requires_trusted_preparation_even_with_an_allow():
    confirm = _FakeConfirm([{"allow": True, "remember": True}])
    executor = _FakeExecutor()
    _, new_chat = build_chat_create_tool_closures(
        confirm=confirm, execute=executor, session_id="s1", run_id="message-1"
    )
    assert not new_chat({"opening_prompt": "go"}).ok
    assert not confirm.payloads
    assert not executor.calls


def test_remembered_creation_rechecks_preparation_and_denials_cross_modes():
    confirm = _FakeConfirm([{"allow": True, "remember": True}])
    executor = _FakeExecutor()
    live = True
    prepares = 0

    def prepare(payload):
        nonlocal prepares
        prepares += 1
        if not live:
            raise PermissionError("source_stopped")
        return _prepare(payload)

    _, new_chat = build_chat_create_tool_closures(
        confirm=confirm,
        execute=executor,
        prepare=prepare,
        session_id="s1",
        run_id="message-1",
    )
    assert new_chat({}).ok
    live = False
    assert not new_chat({}).ok
    assert prepares == 2
    assert len(executor.calls) == 1
    live = True
    assert not new_chat({"mode": "start", "opening_prompt": "go"}).ok
    assert not new_chat({"destination": "casual"}).ok
    refused = new_chat({})
    assert not refused.ok
    assert "denied_repeatedly" in refused.error


@pytest.mark.parametrize(
    "status", ["draft", "not_started", "started", "review_required"]
)
def test_known_creation_outcome_keeps_launch_status_and_no_repeat_instruction(status):
    executor = _FakeExecutor(
        {
            "ok": True,
            "conversation_id": "created",
            "destination": "casual",
            "scope_type": "global",
            "workspace_id": None,
            "mode": "draft" if status == "draft" else "start",
            "launch_status": status,
            "reason": "capacity" if status == "not_started" else None,
        }
    )
    _, new_chat = build_chat_create_tool_closures(
        confirm=_FakeConfirm([{"allow": True}]),
        execute=executor,
        prepare=_prepare,
        session_id="s1",
        run_id="message-1",
    )
    result = new_chat({"opening_prompt": "go"})
    assert result.ok
    body = json.loads(result.content)
    assert body["conversation_id"] == "created"
    assert body["launch_status"] == status
    assert body["destination"] == "casual"
    assert body["scope_type"] == "global"
    assert "repeat" in body["note"].lower()


@pytest.fixture
def handoff_store(tmp_path):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "handoff.sqlite", client_id="handoff-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    yield store, db
    db.close_connection()


def _restore_handoff(store, conversation_id, *, activate=False):
    return store.restore_persisted_session(
        title="Created",
        workspace_id="global",
        persisted_conversation_id=conversation_id,
        all_nodes=[],
        activate=activate,
    )


def _create_handoff(store, **changes):
    handoff = {
        "version": 2,
        "created_via": "new_chat",
        "state": "pending",
        "draft_revision": 1,
        "draft": "original",
        "source_run_id": "source-run",
    }
    handoff.update(changes)
    return store.persistence.create_conversation(
        conversation_title="Created", metadata={"console_agent_handoff": handoff}
    )


def test_pending_v2_draft_survives_activation_and_reopen(handoff_store):
    store, db = handoff_store
    conversation = _create_handoff(store)
    target = _restore_handoff(store, conversation, activate=True)
    assert target.draft == "original"
    stored = json.loads(db.get_conversation_by_id(conversation)["metadata"])
    assert stored["console_agent_handoff"]["state"] == "pending"
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    reopened = ConsoleChatStore(persistence=store.persistence)
    assert _restore_handoff(reopened, conversation, activate=True).draft == "original"


@pytest.mark.parametrize(
    "edited",
    ["edited opening", "", pytest.param("x" * 20_001, id="human-edit-over-tool-cap")],
)
@pytest.mark.asyncio
async def test_v2_edit_or_clear_survives_reopen(handoff_store, edited):
    store, db = handoff_store
    conversation = _create_handoff(store)
    target = _restore_handoff(store, conversation)
    store.set_session_draft(target.id, edited)
    assert await store.drain_agent_handoff(target.id)
    saved = json.loads(db.get_conversation_by_id(conversation)["metadata"])
    assert saved["console_agent_handoff"]["draft"] == edited
    assert saved["console_agent_handoff"]["draft_revision"] == 2
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    reopened = ConsoleChatStore(persistence=store.persistence)
    assert _restore_handoff(reopened, conversation, activate=True).draft == edited


def test_unknown_handoff_version_cannot_restore_an_authorized_draft(handoff_store):
    store, db = handoff_store
    conversation = _create_handoff(store, version=99)
    target = _restore_handoff(store, conversation, activate=True)
    assert target.draft == ""
    saved = json.loads(db.get_conversation_by_id(conversation)["metadata"])
    assert saved["console_agent_handoff"]["version"] == 99


def test_session_reuse_never_reuses_creation_incarnation(handoff_store):
    store, _ = handoff_store
    first = store.create_session(session_id="slot")
    from dataclasses import replace

    copied = replace(first)
    assert copied.id == first.id
    assert copied.incarnation_id != first.incarnation_id


def test_automatic_primary_claims_share_two_slots_and_exact_release(handoff_store):
    """A start and a wake cannot each obtain their own automatic quota."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, _ = handoff_store
    controller = ConsoleChatController(store=store, provider_gateway=object())
    one = store.create_session()
    two = store.create_session()
    three = store.create_session()
    owner = controller._fleet_wake
    tokens = [object(), object(), object()]
    try:
        assert owner.try_claim_automatic_primary(one.id, tokens[0])
        assert owner.try_claim_automatic_primary(two.id, tokens[1])
        assert not owner.try_claim_automatic_primary(three.id, tokens[2])
        assert not owner.release_automatic_primary(one.id, object())
        assert not owner.try_claim_automatic_primary(three.id, tokens[2])
        assert owner.release_automatic_primary(one.id, tokens[0])
        assert owner.try_claim_automatic_primary(three.id, tokens[2])
        assert not owner.release_automatic_primary(three.id, tokens[0])
    finally:
        controller.begin_shutdown()


def test_automatic_claims_keep_manual_reserve_without_double_count(handoff_store):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus

    store, _ = handoff_store
    controller = ConsoleChatController(store=store, provider_gateway=object())
    manual = store.create_session()
    automatic = store.create_session()
    other = store.create_session()
    owner = controller._fleet_wake
    token = object()
    try:
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING), session_id=manual.id
        )
        assert owner.try_claim_automatic_primary(automatic.id, token)
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING), session_id=automatic.id
        )
        assert owner.try_claim_automatic_primary(automatic.id, token)
        assert not owner.try_claim_automatic_primary(other.id, object())
    finally:
        controller.begin_shutdown()


@pytest.fixture
def creation_controller(handoff_store):
    from dataclasses import replace
    from types import SimpleNamespace
    from threading import Event
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )

    store, db = handoff_store
    defaults = replace(
        blank_console_session_settings({}),
        provider="openai",
        model="destination-model",
        system_prompt="destination instructions",
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        default_session_settings=lambda: defaults,
    )
    app = SimpleNamespace(
        app_config={},
        call_from_thread=lambda callback, *args, **kwargs: callback(*args, **kwargs),
    )
    runtime = SimpleNamespace(_app=app)
    runtime._resolve_new_console_assistant = lambda workspace, settings: (
        ConsoleRuntime._resolve_new_console_assistant(runtime, workspace, settings)
    )
    app.console_runtime = runtime
    controller.app = app
    source = store.create_session(
        title="source",
        workspace_id="global",
        settings=replace(
            defaults, model="source-model", system_prompt="source instructions"
        ),
        activate=True,
    )
    source.persisted_conversation_id = store.persistence.create_conversation(
        conversation_title="source"
    )
    controller._active_cancel_events[source.id] = Event()
    controller._active_assistant_message_ids[source.id] = "source-message"
    run = {
        "id": "source-run",
        "conversation_id": source.persisted_conversation_id,
        "agent_kind": "primary",
        "status": "running",
    }
    controller._agent_bridge = SimpleNamespace(
        live_primary_run_id=lambda conversation: "source-run",
        runs_db=SimpleNamespace(get_run=lambda run_id: run),
    )
    # Confirmation runs within the invoking primary worker context.
    with use_run_id("source-run"):
        yield controller, source, db
    controller.begin_shutdown()


def _prepared_creation(controller, source, **changes):
    payload = dict(
        tool="new_chat",
        session_id=source.id,
        source_run_id="source-run",
        source_message_id="source-message",
        opening_prompt="opening",
    )
    payload.update(changes)
    with use_run_id("source-run"):
        return controller.prepare_agent_chat_create(payload)


def test_preparation_is_frozen_and_mutation_needs_approved_token(creation_controller):
    controller, source, db = creation_controller
    prepared = _prepared_creation(controller, source, destination="casual")
    assert prepared["_grant_scope"] == (
        source.incarnation_id,
        "new_chat",
        "global",
        None,
        "draft",
    )
    assert not controller.execute_agent_chat_create(prepared)["ok"]
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM conversations WHERE deleted=0")
        .fetchone()[0]
        == 1
    )


def test_remembered_creation_works_detached_and_restores_fresh_defaults(
    creation_controller,
):
    controller, source, db = creation_controller
    controller.store.set_session_draft(source.id, "keep my composer")
    prepared = _prepared_creation(controller, source, destination="casual")
    controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    result = controller.execute_agent_chat_create(prepared)
    assert result["ok"], result
    assert result["launch_status"] == "draft"
    row = db.get_conversation_by_id(result["conversation_id"])
    assert row["scope_type"] == "global" and row["workspace_id"] is None
    assert row["system_prompt"] == "destination instructions"
    target = next(
        item
        for item in controller.store.sessions()
        if item.persisted_conversation_id == result["conversation_id"]
    )
    assert target.settings.model == "destination-model"
    assert target.assistant_kind == "generic"
    assert target.draft == "opening"
    assert controller.store.active_session_id == source.id
    assert source.draft == "keep my composer"
    assert (
        json.loads(row["metadata"])["console_agent_handoff"]["source_run_id"]
        == "source-run"
    )
    assert not controller._chat_creation_records


def test_changed_payload_or_stopped_source_cannot_reuse_approval(creation_controller):
    controller, source, _db = creation_controller
    prepared = _prepared_creation(controller, source)
    controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    assert not controller.execute_agent_chat_create({**prepared, "mode": "start"})["ok"]
    controller._active_cancel_events[source.id].set()
    assert not controller.execute_agent_chat_create(prepared)["ok"]
    with pytest.raises(PermissionError):
        _prepared_creation(controller, source)


@pytest.mark.parametrize("mismatch", [False, True])
def test_durable_acceptance_consumes_exact_handoff_with_receipt(
    handoff_store, mismatch
):
    from dataclasses import replace
    from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
        _acceptance,
    )
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate
    from tldw_chatbook.Chat.message_metadata import AgentChatStartMetadata

    store, db = handoff_store
    conversation = _create_handoff(store)
    acceptance = _acceptance(conversation)
    candidate = ConsoleLibraryPolicyCandidate(
        acceptance.frozen_authority.policy.auto_retrieve,
        acceptance.frozen_authority.policy.assistant_access,
    )
    store.persistence.console_library_policy_repository.insert(conversation, candidate)
    authority = replace(
        acceptance.frozen_authority,
        policy=replace(acceptance.frozen_authority.policy, policy_revision=1),
    )
    acceptance = replace(
        acceptance,
        user_content="original",
        frozen_authority=authority,
        origin="agent_chat_start",
        agent_chat_start_attempt_id="start-1",
        agent_chat_start=AgentChatStartMetadata("start-1", "run-1", "source-1"),
        handoff_draft_revision=2 if mismatch else 1,
    )

    def commit():
        return store.persistence.commit_durable_turn(
            acceptance=acceptance,
            policy_candidate=candidate,
            conversation_kwargs={"scope_type": "global", "workspace_id": None},
        )

    if mismatch:
        with pytest.raises(ValueError, match="handoff"):
            commit()
        assert not db.get_messages_for_conversation(conversation)
        assert (
            json.loads(db.get_conversation_by_id(conversation)["metadata"])[
                "console_agent_handoff"
            ]["state"]
            == "pending"
        )
    else:
        receipt = commit()
        assert commit() == receipt
        handoff = json.loads(db.get_conversation_by_id(conversation)["metadata"])[
            "console_agent_handoff"
        ]
        assert handoff["state"] == "consumed" and handoff["draft"] == ""
        assert handoff["draft_revision"] == 2
        assert handoff["accepted_attempt_id"] == "start-1"
        assert not store.persistence.update_agent_handoff_draft(
            conversation, expected_revision=1, draft_revision=2, draft="resurrect"
        )
        assert len(db.get_messages_for_conversation(conversation)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["hello", "/help", "@file"])
async def test_native_start_uses_both_receipts_and_literal_machine_request(
    tmp_path, prompt
):
    import asyncio
    from threading import Event
    from Tests.Chat.test_console_agent_swap import _controller
    from tldw_chatbook.Chat.console_chat_start import (
        AgentChatStartRequest,
        ConsoleChatStartCoordinator,
    )

    controller, store, runs = _controller(tmp_path, [["target answer"]])
    source = store.create_session(workspace_id="global", activate=True)
    source.persisted_conversation_id = store.persistence.create_conversation(
        conversation_title="source"
    )
    chain = runs.automatic_work.create_chain(
        source.persisted_conversation_id, root_submission_id="human-root"
    )
    source_run = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="primary",
        work_chain_id=chain,
    )
    controller._agent_bridge._live_primary_runs[source.persisted_conversation_id] = (
        source_run
    )
    controller._active_cancel_events[source.id] = Event()
    controller._active_assistant_message_ids[source.id] = "source-message"
    conversation = _create_handoff(store, draft=prompt, source_run_id=source_run)
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate

    defaults = store._library_policy_defaults
    store.persistence.console_library_policy_repository.insert(
        conversation,
        ConsoleLibraryPolicyCandidate(
            defaults.auto_retrieve, defaults.assistant_access
        ),
    )
    target = _restore_handoff(store, conversation)
    coordinator = ConsoleChatStartCoordinator(controller)
    controller._chat_start = coordinator
    await store.library_policy_coordinator.capture_for_execution(target.id)
    request = AgentChatStartRequest(
        "native-start",
        source_run,
        source.id,
        source.incarnation_id,
        conversation,
        target.id,
        target.incarnation_id,
        1,
        store.conversation_context_epoch(target.id),
        prompt,
        controller.resolve_turn_configuration_snapshot(target.id),
        target.workspace_id,
    )
    commit_errors = []
    committed_receipts = []
    original_commit = store.commit_durable_turn

    def recording_commit(acceptance):
        try:
            committed = original_commit(acceptance)
            committed_receipts.append(committed.checkpoint)
            return committed
        except Exception as exc:
            commit_errors.append(
                (
                    str(exc),
                    acceptance.frozen_authority.policy,
                    store.session_library_policy_candidate(target.id),
                    dict(
                        store.persistence.db.get_connection()
                        .execute(
                            "SELECT * FROM console_conversation_library_policy WHERE conversation_id=?",
                            (conversation,),
                        )
                        .fetchone()
                        or {}
                    ),
                )
            )
            raise

    store.commit_durable_turn = recording_commit
    try:
        outcome = await coordinator.start(request)
        assert outcome.launch_status == "started", (outcome, commit_errors)
        await asyncio.gather(*coordinator.tasks())
        attempt = runs.automatic_work.read_chat_start_attempt(
            "native-start", owner_id=controller._fleet_wake.runtime_owner_id
        )
        assert attempt.state == "completed"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
        rows = store.persistence.db.get_messages_for_conversation(conversation)
        user = next(row for row in rows if row["sender"] == "user")
        assert user["content"] == prompt
        metadata = json.loads(user["metadata_json"])
        assert metadata["origin"] == "agent_chat_start"
        assert metadata["agent_chat_start"]["attempt_id"] == "native-start"
        assert committed_receipts[0].agent_chat_start_attempt_id == "native-start"
        assert any(
            row["sender"] == "assistant" and row["content"] == "target answer"
            for row in rows
        )
        assert store.active_session_id == source.id
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


def test_new_chat_card_discloses_destination_mode_and_complete_instructions():
    from tldw_chatbook.Widgets.Chat_Widgets.chat_create_confirm_card import (
        ChatCreateConfirmCard,
    )

    card = ChatCreateConfirmCard()
    instructions = "[bold]literal[/bold]\n" + "all instructions " * 100
    card.set_payload(
        dict(
            tool="new_chat",
            title="Created",
            destination="casual",
            mode="start",
            scope_type="global",
            workspace_id=None,
            assistant="console",
            model="local-model",
            instructions=instructions,
            opening_prompt="@file",
            source_run_id="true-run",
        )
    )
    body = card._body_text()
    assert "Casual" in body and "background" in body
    assert "local-model" in body and "console" in body
    assert instructions in body and "true-run" in body


@pytest.mark.asyncio
async def test_manual_send_withdraws_prepared_start_before_busy_gate(tmp_path):
    import asyncio
    from threading import Event
    from Tests.Chat.test_console_agent_swap import _controller
    from tldw_chatbook.Chat.console_chat_start import AgentChatStartRequest
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate

    controller, store, runs = _controller(tmp_path, [["manual answer"]])
    source = store.create_session(workspace_id="global")
    source.persisted_conversation_id = store.persistence.create_conversation(
        conversation_title="source"
    )
    chain = runs.automatic_work.create_chain(
        source.persisted_conversation_id, root_submission_id="human-root"
    )
    source_run = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="primary",
        work_chain_id=chain,
    )
    controller._agent_bridge._live_primary_runs[source.persisted_conversation_id] = (
        source_run
    )
    controller._active_cancel_events[source.id] = Event()
    controller._active_assistant_message_ids[source.id] = "source-message"
    conversation = _create_handoff(store)
    defaults = store._library_policy_defaults
    store.persistence.console_library_policy_repository.insert(
        conversation,
        ConsoleLibraryPolicyCandidate(
            defaults.auto_retrieve, defaults.assistant_access
        ),
    )
    target = _restore_handoff(store, conversation)
    await store.library_policy_coordinator.capture_for_execution(target.id)
    request = AgentChatStartRequest(
        "withdraw-start",
        source_run,
        source.id,
        source.incarnation_id,
        conversation,
        target.id,
        target.incarnation_id,
        1,
        store.conversation_context_epoch(target.id),
        "original",
        controller.resolve_turn_configuration_snapshot(target.id),
        target.workspace_id,
    )
    reached = asyncio.Event()
    original_resolve = controller._resolve_for_send_bounded
    calls = 0

    async def paused_resolve(selection):
        nonlocal calls
        calls += 1
        if calls == 1:
            reached.set()
            await asyncio.Event().wait()
        return await original_resolve(selection)

    controller._resolve_for_send_bounded = paused_resolve
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        await asyncio.wait_for(reached.wait(), 5)
        assert controller._chat_start.is_prepared(target.id)
        assert controller.send_refusal_copy(target.id) is None
        result = await controller.submit_draft(
            "human replacement", session_id=target.id
        )
        assert result.accepted, result
        assert (await start).launch_status == "not_started"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 0
        rows = store.persistence.db.get_messages_for_conversation(conversation)
        assert [row["content"] for row in rows if row["sender"] == "user"] == [
            "human replacement"
        ]
        assert not controller._chat_start.tasks()
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.parametrize("valid", [True, False])
def test_machine_request_presentation_never_uses_human_name(valid):
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_roleplay_identity import (
        ConsolePresentationContext,
        resolve_console_message_presentation,
    )
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    payload = {
        "origin": "agent_chat_start",
        "agent_chat_start": {
            "attempt_id": "attempt",
            "source_run_id": "source",
            "source_conversation_id": "conversation",
        }
        if valid
        else {},
    }
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="private opening",
        metadata=MessageMetadata.from_json(json.dumps(payload)),
    )
    presentation = resolve_console_message_presentation(
        message, ConsolePresentationContext(user_name="Human")
    )
    assert presentation.speaker_label == (
        "Agent handoff" if valid else "Unverified handoff"
    )


def test_unconfigured_destination_saves_absent_snapshot_and_reopens(
    creation_controller,
):
    from dataclasses import replace
    from tldw_chatbook.Chat.console_generation_settings_metadata import (
        ConsoleGenerationSettingsReadStatus,
    )

    controller, source, db = creation_controller
    settings = replace(controller._default_session_settings(), provider="", model=None)
    controller._default_session_settings = lambda: settings
    prepared = _prepared_creation(controller, source, mode="start")
    controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    outcome = controller.execute_agent_chat_create(prepared)
    assert outcome["ok"] and outcome["launch_status"] == "not_started"
    row = db.get_conversation_by_id(outcome["conversation_id"])
    assert "console_generation_settings" not in json.loads(row["metadata"])
    reopened = _restore_handoff(controller.store, outcome["conversation_id"])
    assert reopened.draft == "opening"
    assert (
        reopened.generation_metadata_status
        is ConsoleGenerationSettingsReadStatus.ABSENT
    )
    assert row["system_prompt"] == settings.system_prompt


@pytest.mark.parametrize("destination", ["same_workspace", "casual"])
@pytest.mark.parametrize("instructions", ["", "explicit override"])
def test_workspace_destination_resolves_persona_without_source_identity(
    creation_controller, tmp_path, destination, instructions
):
    from dataclasses import replace
    from types import SimpleNamespace
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

    controller, source, db = creation_controller
    workspaces = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="creation")
    registry = LocalWorkspaceRegistryService(workspaces)
    registry.create_workspace(
        workspace_id="named",
        name="Named",
        assistant_defaults=WorkspaceAssistantDefaults(
            assistant_id="destination-persona"
        ),
    )
    controller.store.persistence.workspace_registry = registry
    controller.app.workspace_registry_service = registry
    controller.app.local_character_persona_service = SimpleNamespace(
        get_persona_profile=lambda identifier: {
            "id": identifier,
            "name": "Destination persona",
            "system_prompt": "Destination standing instructions",
        }
    )
    source.workspace_id = "named"
    source.assistant_kind = "persona"
    source.assistant_id = "source-persona"
    settings = replace(controller._default_session_settings(), system_prompt=None)
    controller._default_session_settings = lambda: settings
    prepared = _prepared_creation(
        controller, source, destination=destination, instructions=instructions
    )
    controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    outcome = controller.execute_agent_chat_create(prepared)
    assert outcome["ok"], outcome
    row = db.get_conversation_by_id(outcome["conversation_id"])
    if destination == "casual":
        assert row["workspace_id"] is None and row["scope_type"] == "global"
        assert row["assistant_kind"] == "generic"
    else:
        assert row["workspace_id"] == "named" and row["scope_type"] == "workspace"
        assert row["assistant_id"] == (
            "console" if instructions else "destination-persona"
        )
        assert any(
            member.item_id == outcome["conversation_id"]
            for member in registry.list_workspace_conversations("named")
        )
    if instructions:
        assert row["system_prompt"] == instructions
    assert row["assistant_id"] != "source-persona"
    workspaces.close()


@pytest.mark.asyncio
async def test_reopened_machine_retry_is_explicit_and_retains_provenance(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from Tests.Chat.test_console_dispatch_recovery import (
        _database,
        _acceptance,
        _insert,
        _restored_store,
        _NoReplayGateway,
        _patch_exact_retry_context,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.message_metadata import (
        AgentChatStartMetadata,
        MessageMetadata,
    )
    from tldw_chatbook.Agents.agent_models import WorkOrigin

    db, conversation, repository = _database(tmp_path / "retry-machine.sqlite")
    provenance = AgentChatStartMetadata("old-start", "old-source", "source-chat")
    acceptance = replace(
        _acceptance(conversation),
        origin="agent_chat_start",
        agent_chat_start_attempt_id="old-start",
        agent_chat_start=provenance,
        handoff_draft_revision=1,
    )
    _insert(db, repository, acceptance)
    store, session_id = _restored_store(db, conversation)
    gateway = _NoReplayGateway(db)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        base_url="http://127.0.0.1:9099",
        agent_runtime_enabled=False,
    )
    await _patch_exact_retry_context(monkeypatch, controller, gateway)
    assert not gateway.provider_states
    captured = []
    original_stream = controller._stream_assistant_response

    async def record_stream(**kwargs):
        captured.append(kwargs)
        return await original_stream(**kwargs)

    monkeypatch.setattr(controller, "_stream_assistant_response", record_stream)
    try:
        result = await controller.retry_dispatch_recovery(session_id)
        assert result.accepted
        assert captured[0].get("work_origin", WorkOrigin.MANUAL) is WorkOrigin.MANUAL
        assert captured[0].get("trusted_profile_user_message_id") is None
        assert (
            MessageMetadata.from_json(
                db.get_message_by_id(acceptance.user_message_id)["metadata_json"]
            ).agent_chat_start
            == provenance
        )
    finally:
        await controller.shutdown()
        db.close_connection()


async def _native_start_rig(tmp_path):
    from threading import Event
    from Tests.Chat.test_console_agent_swap import _controller
    from tldw_chatbook.Chat.console_chat_start import AgentChatStartRequest
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate

    controller, store, runs = _controller(tmp_path, [["target answer"]])
    source = store.create_session(workspace_id="global", activate=True)
    source.persisted_conversation_id = store.persistence.create_conversation(
        conversation_title="source"
    )
    chain = runs.automatic_work.create_chain(
        source.persisted_conversation_id, root_submission_id="human-root"
    )
    run = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="primary",
        work_chain_id=chain,
    )
    controller._agent_bridge._live_primary_runs[source.persisted_conversation_id] = run
    controller._active_cancel_events[source.id] = Event()
    controller._active_assistant_message_ids[source.id] = "source-message"
    conversation = _create_handoff(store, source_run_id=run)
    defaults = store._library_policy_defaults
    store.persistence.console_library_policy_repository.insert(
        conversation,
        ConsoleLibraryPolicyCandidate(
            defaults.auto_retrieve, defaults.assistant_access
        ),
    )
    target = _restore_handoff(store, conversation)
    await store.library_policy_coordinator.capture_for_execution(target.id)
    request = AgentChatStartRequest(
        "barrier-start",
        run,
        source.id,
        source.incarnation_id,
        conversation,
        target.id,
        target.incarnation_id,
        1,
        store.conversation_context_epoch(target.id),
        "original",
        controller.resolve_turn_configuration_snapshot(target.id),
        target.workspace_id,
    )
    return controller, store, runs, source, target, chain, request


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["source_stop", "source_close", "edit", "clear"])
async def test_before_cutoff_withdrawal_preserves_latest_draft_and_refunds(
    tmp_path, action
):
    import asyncio

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    if action == "source_close":
        _seed_close_source_message(controller, source)
    reached = asyncio.Event()

    async def paused(_selection):
        reached.set()
        await asyncio.Event().wait()

    controller._resolve_for_send_bounded = paused
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        await asyncio.wait_for(reached.wait(), 5)
        if action in {"source_stop", "source_close"}:
            if action == "source_close":
                controller.begin_session_close(
                    source.id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=source.id
                    ).revision,
                )
            else:
                controller._signal_stop(session_id=source.id)
            expected = "original"
        else:
            expected = "edited" if action == "edit" else ""
            store.set_session_draft(target.id, expected)
        assert (await start).launch_status == "not_started"
        await asyncio.gather(*controller._chat_start.tasks())
        assert await store.drain_agent_handoff(target.id)
        assert target.draft == expected
        row = store.persistence.db.get_conversation_by_id(request.conversation_id)
        assert json.loads(row["metadata"])["console_agent_handoff"]["draft"] == expected
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert runs.automatic_work.snapshot(chain).used["generation"] == 0
        assert not controller._fleet_wake._automatic_primary_claims
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["source_stop", "source_close", "write_failure", "target_stop"]
)
async def test_ledger_cutoff_retains_charge_and_requires_conversation_receipt(
    tmp_path, action
):
    import asyncio
    from threading import Event

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    if action == "source_close":
        _seed_close_source_message(controller, source)
    entered, release = Event(), Event()
    original_commit = store.commit_durable_turn

    def paused_commit(acceptance):
        entered.set()
        assert release.wait(10)
        if action == "write_failure":
            raise OSError("injected write refusal")
        return original_commit(acceptance)

    store.commit_durable_turn = paused_commit
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
        if action in {"source_stop", "source_close"}:
            if action == "source_close":
                controller.begin_session_close(
                    source.id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=source.id
                    ).revision,
                )
            else:
                controller._signal_stop(session_id=source.id)
        if action == "target_stop":
            store.switch_session(target.id)
            assert controller.stop_active_run()
        release.set()
        outcome = await start
        await asyncio.gather(*controller._chat_start.tasks())
        assert outcome.launch_status == (
            "started"
            if action in {"source_stop", "source_close"}
            else "review_required"
        ), outcome
        rows = store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert any(
            row["sender"] == "assistant" and row["content"] == "target answer"
            for row in rows
        ) == (action in {"source_stop", "source_close"})
        if action == "source_close":
            attempt = runs.automatic_work.read_chat_start_attempt(
                request.attempt_id, owner_id=controller.fleet_wake.runtime_owner_id
            )
            assert attempt.state == "completed"
            user = next(row for row in rows if row["sender"] == "user")
            assert (
                json.loads(user["metadata_json"])["agent_chat_start"]["attempt_id"]
                == request.attempt_id
            )
            handoff = json.loads(
                store.persistence.db.get_conversation_by_id(request.conversation_id)[
                    "metadata"
                ]
            )["console_agent_handoff"]
            assert handoff["state"] == "consumed" and handoff["draft"] == ""
            assert target.agent_handoff_state == "consumed" and target.draft == ""
        if action == "write_failure":
            assert not rows and target.draft == "original"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
        if action not in {"source_stop", "source_close"}:
            assert (
                runs.automatic_work.read_chat_start_attempt(
                    request.attempt_id, owner_id=controller.fleet_wake.runtime_owner_id
                ).state
                == "review_required"
            )
            assert runs.automatic_work.snapshot(chain).status == "review_required"
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

            with pytest.raises(AutomaticWorkRefused, match="interrupted_work"):
                runs.automatic_work.check_active(
                    chain, owner_id=controller.fleet_wake.runtime_owner_id
                )
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_live_machine_retry_starts_manual_work_without_reassigning_old_allowance(
    tmp_path,
):
    from tldw_chatbook.Agents.agent_models import WorkOrigin
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    original_effect = controller._run_durable_postcommit_effect

    async def fail_once(preparation_id, name, effect, **kwargs):
        if name == "provider_entry":
            raise OSError("injected before dispatch")
        return await original_effect(preparation_id, name, effect, **kwargs)

    controller._run_durable_postcommit_effect = fail_once
    try:
        assert (await controller._chat_start.start(request)).launch_status == "started"
        await __import__("asyncio").gather(*controller._chat_start.tasks())
        assert store.dispatch_recovery_for_session(target.id) is not None
        controller._run_durable_postcommit_effect = original_effect
        original_stream = controller._stream_assistant_response
        seen = []

        async def record_stream(**kwargs):
            seen.append((kwargs, current_automatic_work()))
            return await original_stream(**kwargs)

        controller._stream_assistant_response = record_stream
        result = await controller.retry_dispatch_recovery(target.id)
        assert result.accepted
        assert seen and seen[0][0]["work_origin"] is WorkOrigin.MANUAL
        assert seen[0][0]["work_chain_id"] is None and seen[0][1] is None
        assert seen[0][0]["trusted_profile_user_message_id"] is None
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
        assert (
            store.messages_for_session(target.id)[0].metadata.origin
            == "agent_chat_start"
        )
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.parametrize(
    "status,copy",
    [
        ("draft", "Draft"),
        ("started", "Started"),
        ("not_started", "Not started"),
        ("review_required", "Review required"),
    ],
)
def test_completion_observer_distinguishes_launch_outcomes(status, copy):
    from types import SimpleNamespace
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    notices = []
    screen = SimpleNamespace(
        _console_chat_controller=SimpleNamespace(store=object()),
        _workspace=SimpleNamespace(
            _invalidate_console_persisted_rows_cache=lambda: None
        ),
        run_worker=lambda *args, **kwargs: None,
        _sync_native_console_chat_ui=lambda: None,
        app_instance=SimpleNamespace(notify=notices.append),
    )
    ChatScreen._complete_agent_chat_create(
        screen,
        session_id="source",
        conversation_id="target",
        title="New target",
        tool="new_chat",
        opening_prompt="private prompt",
        workspace_id="global",
        launch_status=status,
        reason="capacity" if status == "not_started" else None,
    )
    assert copy in notices[0]
    assert "private prompt" not in notices[0]


@pytest.mark.parametrize("case", ["no_view", "deny", "stopped", "destination_changed"])
def test_prepared_approval_round_releases_denied_records(creation_controller, case):
    import threading
    from Tests.Chat.test_console_skill_script_confirm import _wait_until

    controller, source, _db = creation_controller
    prepared = _prepared_creation(controller, source)
    if case == "no_view":
        decision = controller.request_chat_create_confirm(
            prepared, session_id=source.id
        )
    else:
        cards = []
        controller.set_pending_chat_create = cards.append
        decisions = []
        thread = threading.Thread(
            target=lambda: decisions.append(
                controller.request_chat_create_confirm(prepared, session_id=source.id)
            )
        )
        thread.start()
        _wait_until(lambda: bool(controller.pending_chat_create_ids()))
        request_id = controller.pending_chat_create_ids()[0]
        controller.resolve_pending_chat_create(True, True, request_id="stale-request")
        assert thread.is_alive()
        if case == "stopped":
            controller._signal_stop(session_id=source.id)
        if case == "destination_changed":
            source.workspace_id = "different-workspace"
        controller.resolve_pending_chat_create(
            case != "deny", True, request_id=request_id
        )
        thread.join(5)
        assert not thread.is_alive()
        decision = decisions[0]
    assert not decision["allow"]
    assert not controller._chat_creation_records
    assert not controller._chat_create_session_grants.get(source.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["disabled", "capacity", "staged", "unready"])
async def test_refused_start_preserves_draft_without_dispatch_or_timer(
    tmp_path, monkeypatch, case
):
    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    if case == "disabled":
        controller._agent_runtime_enabled = False
    elif case == "capacity":
        monkeypatch.setattr(
            type(controller), "max_parallel_runs", property(lambda _self: 1)
        )
    elif case == "staged":
        controller._staged_evidence_provider = lambda session_id: True
    else:

        async def unready(_selection):
            return None

        controller._resolve_for_send_bounded = unready
    try:
        outcome = await controller._chat_start.start(request)
        await __import__("asyncio").gather(*controller._chat_start.tasks())
        assert outcome.launch_status == "not_started"
        assert target.draft == "original"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert runs.automatic_work.snapshot(chain).used["generation"] == 0
        assert not controller._chat_start.tasks()
        assert not controller._fleet_wake._automatic_primary_claims
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_handoff_writer_coalesces_edits_and_consumption_preserves_new_composer(
    handoff_store,
):
    import asyncio
    from threading import Event

    store, db = handoff_store
    conversation = _create_handoff(store)
    target = _restore_handoff(store, conversation)
    entered, release = Event(), Event()
    writes = []
    original = store.persistence.update_agent_handoff_draft

    def paused_write(*args, **kwargs):
        writes.append(kwargs["draft_revision"])
        if len(writes) == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    store.persistence.update_agent_handoff_draft = paused_write
    store.set_session_draft(target.id, "first")
    assert await asyncio.to_thread(entered.wait, 5)
    task = store._agent_handoff_writes[target.id]["task"]
    for index in range(100):
        store.set_session_draft(target.id, f"latest {index}")
        assert store._agent_handoff_writes[target.id]["task"] is task
    release.set()
    assert await store.drain_agent_handoff(target.id)
    assert len(writes) == 2
    assert _restore_handoff(store, conversation).draft == "latest 99"
    accepted_revision = target.agent_handoff_revision
    # A newer composer value arriving after the receipt does not belong to its draft.
    store.set_session_draft(target.id, "new human input")
    store.publish_agent_handoff_consumed(target.id, accepted_revision)
    assert target.draft == "new human input"
    assert target.agent_handoff_state == "consumed"
    assert target.id not in store._agent_handoff_writes
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_handoff_writer_releases_its_thread_connection(handoff_store):
    store, db = handoff_store
    target = _restore_handoff(store, _create_handoff(store))
    baseline = len(db._connection_quiescence._connections)
    store.set_session_draft(target.id, "saved edit")
    assert await store.drain_agent_handoff(target.id)
    assert len(db._connection_quiescence._connections) == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "false"])
async def test_unconfirmed_preaccept_settlement_requires_review(
    tmp_path, monkeypatch, failure
):
    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )

    async def unready(_selection):
        return None

    controller._resolve_for_send_bounded = unready

    def cannot_settle(*args, **kwargs):
        if failure == "false":
            return False
        raise OSError("injected settlement uncertainty")

    monkeypatch.setattr(runs.automatic_work, "abort_chat_start", cannot_settle)
    try:
        outcome = await controller._chat_start.start(request)
        await __import__("asyncio").gather(*controller._chat_start.tasks())
        assert outcome.launch_status == "review_required"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert runs.automatic_work.snapshot(chain).reserved["generation"] == 1
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "replacement"])
async def test_visible_pending_handoff_persists_every_composer_edit(
    handoff_store, text
):
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    store, db = handoff_store
    conversation = _create_handoff(store)
    target = _restore_handoff(store, conversation)
    store.switch_session(target.id)
    screen = SimpleNamespace(
        _console_chat_store=store,
        _console_visible_draft_session_id=target.id,
        _console_composer_or_none=lambda: SimpleNamespace(draft_text=lambda: text),
        _console_chat_controller=SimpleNamespace(
            fleet_wake=SimpleNamespace(retry_soon=lambda: None),
            run_state_for=lambda _id: SimpleNamespace(status=ConsoleRunStatus.IDLE),
        ),
        _console_draft_spend_refresh=SimpleNamespace(route_edit=lambda **kwargs: None),
        _console_runtime=lambda: SimpleNamespace(
            has_custodied_turns=lambda session_id: False
        ),
    )
    ChatScreen._on_console_composer_draft_changed(screen, SimpleNamespace(value=text))
    assert await store.drain_agent_handoff(target.id)
    assert _restore_handoff(store, conversation).draft == text


def test_configured_creation_snapshot_survives_changed_reopen_defaults(
    creation_controller,
):
    from tldw_chatbook.Chat.console_conversation_hydration import (
        hydrate_console_generation_settings,
    )
    from tldw_chatbook.Chat.console_generation_settings_metadata import (
        ConsoleGenerationSettingsReadStatus,
    )

    controller, source, db = creation_controller
    prepared = _prepared_creation(controller, source)
    controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    outcome = controller.execute_agent_chat_create(prepared)
    row = db.get_conversation_by_id(outcome["conversation_id"])
    restored = hydrate_console_generation_settings(
        {
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": "changed-model",
                "temperature": 1.9,
            }
        },
        row,
    )
    assert restored.metadata_status is ConsoleGenerationSettingsReadStatus.VALID
    assert restored.settings.provider == "openai"
    assert restored.settings.model == "destination-model"
    assert restored.settings.system_prompt == "destination instructions"
    assert restored.durable_snapshot == next(
        s.generation_durable_snapshot
        for s in controller.store.sessions()
        if s.persisted_conversation_id == outcome["conversation_id"]
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_next_send_builder_previews_creation_schemas_without_execution(
    tmp_path, enabled, monkeypatch
):
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "false")
    from Tests.Chat.test_console_agent_bridge import (
        _bridge_with_gateway,
        _ChunkGateway,
        _native_resolution,
    )
    from Tests.Chat.test_console_agent_project_instructions import _candidate

    gateway = _ChunkGateway([])
    bridge, runs, store, session, assistant_id = _bridge_with_gateway(tmp_path, gateway)
    before = tuple(store.sessions())
    try:
        result = bridge.build_project_instruction_preview_request(
            candidate=_candidate(tmp_path),
            session_id=session.id,
            resolution=_native_resolution(),
            fallback_model="gpt-test",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "preview"}],
            fork_chat_enabled=enabled,
            new_chat_enabled=enabled,
        )
        assert result is not None
        request, snapshot = result
        names = {tool["function"]["name"] for tool in request.get("tools", [])}
        assert ("new_chat" in names) is enabled
        assert ("fork_chat" in names) is enabled
        if enabled:
            tool = next(
                tool
                for tool in request["tools"]
                if tool["function"]["name"] == "new_chat"
            )
            assert tool["function"]["parameters"]["properties"]["mode"]["enum"] == [
                "draft",
                "start",
            ]
        assert tuple(store.sessions()) == before
        assert not gateway.messages_seen
        assert runs.list_runs(session.id) == []
    finally:
        runs.close()


@pytest.mark.asyncio
async def test_stop_keeps_native_claim_until_actual_bridge_worker_exits(tmp_path):
    import asyncio
    from threading import Event

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    entered, release, exited = Event(), Event(), Event()
    original = controller._agent_bridge.run_reply

    def held(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(30)
            return original(*args, **kwargs)
        finally:
            exited.set()

    controller._agent_bridge.run_reply = held
    try:
        assert (await controller._chat_start.start(request)).launch_status == "started"
        assert await asyncio.to_thread(entered.wait, 5)
        store.switch_session(target.id)
        assert controller.stop_active_run()
        owned = tuple(controller._chat_start.tasks())
        assert owned
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(asyncio.gather(*owned)), 0.5)
        assert not exited.is_set()
        assert target.id in controller.fleet_wake._automatic_primary_claims
        assert controller._chat_start.tasks()
        assert (
            runs.automatic_work.read_chat_start_attempt(
                request.attempt_id, owner_id=controller.fleet_wake.runtime_owner_id
            ).state
            == "accepted"
        )
        release.set()
        await asyncio.wait_for(asyncio.gather(*controller._chat_start.tasks()), 5)
        assert exited.is_set()
        assert target.id not in controller.fleet_wake._automatic_primary_claims
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("refund", ["false", "raise"])
@pytest.mark.parametrize("withdrawal", ["source_stop", "caller_cancel", "shutdown"])
async def test_initial_preparation_keeps_owner_until_uncertain_refund(
    tmp_path, refund, withdrawal
):
    import asyncio
    from threading import Event

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    entered, release = Event(), Event()
    ledger = runs.automatic_work
    prepare, abort = ledger.prepare_chat_start, ledger.abort_chat_start

    def held(**kwargs):
        attempt = prepare(**kwargs)
        entered.set()
        assert release.wait(30)
        return attempt

    def uncertain(*args, **kwargs):
        if refund == "raise":
            raise RuntimeError("injected settlement failure")
        return False

    ledger.prepare_chat_start, ledger.abort_chat_start = held, uncertain
    start = asyncio.create_task(controller._chat_start.start(request))
    shutdown = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        item = controller._chat_start._active[target.id]
        if withdrawal == "source_stop":
            controller._signal_stop(session_id=source.id)
        elif withdrawal == "caller_cancel":
            start.cancel()
        else:
            shutdown = asyncio.create_task(controller.shutdown())
        await asyncio.sleep(0.05)
        assert controller._chat_start.tasks()
        assert target.id in controller.fleet_wake._automatic_primary_claims
        release.set()
        outcome = await asyncio.wait_for(asyncio.shield(item.outcome), 5)
        assert outcome.launch_status == "review_required"
        assert outcome.reason == "settlement_unconfirmed"
        assert ledger.snapshot(chain).reserved["generation"] == 1
        await asyncio.gather(start, return_exceptions=True)
        if shutdown:
            await shutdown
    finally:
        release.set()
        ledger.abort_chat_start = abort
        await asyncio.gather(start, return_exceptions=True)
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("folder_count", [1, 2])
async def test_native_project_decision_refuses_before_both_fences(
    tmp_path, folder_count
):
    import asyncio
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import Mock, AsyncMock
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    workspaces = WorkspaceDB(tmp_path / "projects.sqlite", client_id="project-start")
    registry = LocalWorkspaceRegistryService(workspaces)
    registry.create_workspace(workspace_id="w1", name="Project")
    store.persistence.workspace_registry = registry
    for session in (source, target):
        session.workspace_id = "w1"
        with store.persistence.db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET scope_type = ?, workspace_id = ? WHERE id = ?",
                ("workspace", "w1", session.persisted_conversation_id),
            )
    target.project_instruction_state = ProjectInstructionControlState.new_session()
    for i in range(folder_count):
        root = tmp_path / f"project-{i}"
        root.mkdir()
        (root / "AGENTS.md").write_text("Private project instruction body")
        registry.add_folder_binding("w1", root, allow_write=True)
    controller.app = SimpleNamespace(workspace_registry_service=registry)
    selection = AsyncMock(return_value=("cancel", None))
    consent = Mock(return_value="cancel")
    controller._select_project_instruction_binding = selection
    controller._confirm_project_instruction_dispatch = consent
    request = replace(
        request,
        configuration=controller.resolve_turn_configuration_snapshot(target.id),
        workspace_id=target.workspace_id,
    )
    try:
        outcome = await controller._chat_start.start(request)
        assert outcome.launch_status == "not_started"
        assert outcome.reason in {
            "project_binding_required",
            "project_consent_required",
        }
        await asyncio.gather(*controller._chat_start.tasks())
        selection.assert_not_called()
        consent.assert_not_called()
        assert target.draft == "original" and target.agent_handoff_state == "pending"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 0
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()
        workspaces.close()


@pytest.mark.asyncio
async def test_refused_native_outcome_reopens_as_display_only_status(tmp_path):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    controller._agent_runtime_enabled = False
    try:
        outcome = await controller._chat_start.start(request)
        assert outcome.launch_status == "not_started"
        row = store.persistence.db.get_conversation_by_id(request.conversation_id)
        launch = json.loads(row["metadata"])["console_agent_handoff"]["launch"]
        assert launch == {
            "mode": "start",
            "status": "not_started",
            "reason": "runtime_disabled",
        }
        assert controller.activity_for(target.id).agent_handoff_status == "Not started"
        reopened = ConsoleChatStore(persistence=store.persistence)
        restored = _restore_handoff(reopened, request.conversation_id)
        assert restored.agent_handoff_launch.status == "not_started"
        assert restored.agent_handoff_launch.label == "Not started"
        assert restored.draft == "original"
        assert not controller._chat_start.tasks()
        assert not runs.list_runs(request.conversation_id)
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_limit", ["generation", "deadline"])
async def test_native_start_child_and_wake_share_original_allowance(
    tmp_path, monkeypatch, stop_limit
):
    import asyncio
    import time
    from dataclasses import replace
    from threading import Event
    from Tests.Chat.test_console_agent_swap import _Gateway
    from Tests.Chat.test_console_agent_bridge import _fence
    from Tests.Chat.test_console_fleet_wake import _settle
    from Tests.Chat.test_fleet_attention import _AppStub
    from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicyCandidate

    monkeypatch.setenv(
        "TLDW_AGENTS_MAX_AUTOWAKE_GENERATIONS",
        "2" if stop_limit == "generation" else "5",
    )
    monkeypatch.setenv("TLDW_AGENTS_MAX_AUTOWAKE_CHILD_LAUNCHES", "1")
    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    child_entered, child_release = Event(), Event()
    import httpx
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    calls = {"parent": 0, "child": 0}

    async def adapter(http_request):
        messages = json.loads(http_request.content)["messages"]
        is_child = str(messages[0].get("content", "")).startswith(
            SUBAGENT_PROMPT_PREFIX
        )
        if is_child:
            calls["child"] += 1
            child_entered.set()
            assert await asyncio.to_thread(child_release.wait, 30)
            content = "child finished"
        else:
            calls["parent"] += 1
            content = [
                _fence("spawn_subagent", {"task": "finish one child task"}),
                "native finished",
                "wake finished",
            ][calls["parent"] - 1]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(adapter))
    gateway = ConsoleProviderGateway(http_client=client)

    async def resolve(selection):
        return replace(
            await _Gateway([]).resolve_for_send(selection),
            streaming=False,
            base_url="http://127.0.0.1:9099",
        )

    gateway.resolve_for_send = resolve
    controller.provider_gateway = gateway
    controller._agent_bridge._gateway = gateway
    app = _AppStub(store.persistence.db)
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    app.local_chat_conversation_service = ChatConversationService(store.persistence.db)
    controller.app = app
    runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    custody_errors = []
    run_custodied = runtime._run_custodied_turn

    async def record_custody_error(*args, **kwargs):
        try:
            return await run_custodied(*args, **kwargs)
        except Exception as exc:
            custody_errors.append(repr(exc))
            raise

    runtime._run_custodied_turn = record_custody_error
    controller.fleet_wake.wire(app=app)
    controller._agent_bridge.on_fleet_drained(
        "native-integration", controller.fleet_wake.on_fleet_drained
    )
    try:
        assert (await controller._chat_start.start(request)).launch_status == "started"
        assert await asyncio.to_thread(child_entered.wait, 5), (
            calls,
            runs.list_runs(request.conversation_id),
            controller.run_state_for(target.id),
        )
        await asyncio.gather(*controller._chat_start.tasks())
        original = runs.automatic_work.snapshot(chain)
        assert original.used["generation"] == 1 and original.used["child_launch"] == 1
        assert original.deadline_at is not None
        child_release.set()
        assert await _settle(lambda: calls["parent"] == 3, seconds=10), (
            custody_errors,
            controller.fleet_wake._paused,
            runs.automatic_work.snapshot(chain),
        )
        assert await _settle(
            lambda: not controller.fleet_wake._delivery_tasks, seconds=10
        )
        records = runs.list_runs(request.conversation_id)
        child = next(row for row in records if row["agent_kind"] == "subagent")
        assert (
            runs.get_run(child["parent_run_id"])["conversation_id"]
            == request.conversation_id
        )
        assert all(row["conversation_id"] == request.conversation_id for row in records)
        assert len([row for row in records if row["agent_kind"] == "primary"]) == 2
        descendant = runs.automatic_work.snapshot(child["work_chain_id"])
        root = runs.automatic_work.snapshot(chain)
        assert descendant.used == root.used
        assert root.used["generation"] == 2 and root.used["child_launch"] == 1
        assert root.used["model_call"] == 4
        assert root.used["tokens"] == 12
        assert root.deadline_at == original.deadline_at == descendant.deadline_at
        assert root.limits == original.limits == descendant.limits
        messages = store.messages_for_session(target.id)
        assert any(m.content == "wake finished" for m in messages)
        assert any(getattr(m.metadata, "origin", "") == "agent_wake" for m in messages)
        # Looser current config cannot renew the root or its original deadline.
        monkeypatch.setenv("TLDW_AGENTS_MAX_AUTOWAKE_GENERATIONS", "99")
        if stop_limit == "deadline":
            runs.automatic_work._wall_clock = lambda: original.deadline_at + 1
            runs.automatic_work._monotonic_clock = lambda: time.monotonic() + 1000
        extra_conversation = _create_handoff(store, source_run_id=request.source_run_id)
        defaults = store._library_policy_defaults
        store.persistence.console_library_policy_repository.insert(
            extra_conversation,
            ConsoleLibraryPolicyCandidate(
                defaults.auto_retrieve, defaults.assistant_access
            ),
        )
        extra = _restore_handoff(store, extra_conversation)
        await store.library_policy_coordinator.capture_for_execution(extra.id)
        next_request = replace(
            request,
            attempt_id="root-exhausted",
            conversation_id=extra_conversation,
            session_id=extra.id,
            session_incarnation=extra.incarnation_id,
            context_epoch=store.conversation_context_epoch(extra.id),
            configuration=controller.resolve_turn_configuration_snapshot(extra.id),
        )
        assert (
            await controller._chat_start.start(next_request)
        ).launch_status == "not_started"
        await asyncio.gather(*controller._chat_start.tasks())
        assert calls == {"parent": 3, "child": 1}
        assert runs.automatic_work.snapshot(chain).used["generation"] == 2
    finally:
        child_release.set()
        await controller.shutdown()
        await client.aclose()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_stop_during_started_status_write_does_not_dispatch_provider(tmp_path):
    import asyncio
    from threading import Event

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    entered, release = Event(), Event()
    original = store.persistence.update_agent_handoff_launch

    def held(conversation_id, launch):
        if launch.status == "started":
            entered.set()
            assert release.wait(30)
        return original(conversation_id, launch)

    store.persistence.update_agent_handoff_launch = held
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        store.switch_session(target.id)
        assert controller.stop_active_run()
        assert controller._chat_start.tasks()
        release.set()
        outcome = await asyncio.wait_for(start, 5)
        await asyncio.gather(*controller._chat_start.tasks())
        assert controller.provider_gateway.parent_calls == 0
        assert outcome.launch_status == "review_required"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_saved_launch_status_projects_to_native_and_persisted_rows(tmp_path):
    from types import SimpleNamespace
    from Tests.UI.test_console_workspace_controller import _workspace_controller
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    controller._agent_runtime_enabled = False
    try:
        await controller._chat_start.start(request)
        workspace = _workspace_controller(
            app_instance=SimpleNamespace(
                local_chat_conversation_service=ChatConversationService(
                    store.persistence.db
                )
            ),
            chat_store_accessor=lambda: store,
            current_chat_store_accessor=lambda: store,
            current_chat_controller_accessor=lambda: controller,
        )
        for rows in (
            workspace._native_console_browser_rows(),
            workspace._native_console_switcher_rows(),
        ):
            row = next(row for row in rows if row.native_session_id == target.id)
            assert row.status == "Not started"
            assert "original" not in row.status
        rows, _, error = await workspace._persisted_console_browser_rows(
            "", scopes=(("global", None),)
        )
        assert not error
        row = next(
            row for row in rows if row.conversation_id == request.conversation_id
        )
        assert row.status == "Not started"
        entry = next(
            row
            for row in workspace.console_session_switcher_active_entries()
            if row.conversation_id == request.conversation_id
        )
        assert entry.activity_state == "blocked"
        history = await workspace.load_console_session_switcher_history(
            query="", offset=0, limit=20
        )
        entry = next(
            row
            for row in history.entries
            if row.conversation_id == request.conversation_id
        )
        assert "not started" in entry.subtitle.lower()
        from Tests.UI.test_console_activity_switcher import _ActivitySwitcherApp
        from textual.widgets import Input

        workspace.app_instance.console_runtime = SimpleNamespace(
            profile_authority="profile-a", authority_token="runtime-a"
        )
        host = _ActivitySwitcherApp(
            history_loader=workspace.load_console_session_switcher_history
        )
        async with host.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            host.screen.query_one("#console-switcher-query", Input).value = "Created"
            await pilot.pause(0.6)
            labels = [
                str(button.label)
                for button in host.screen.query(".console-switcher-result")
            ]
            assert any("not started" in label.lower() for label in labels), labels
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"mode": "start", "status": "started", "reason": "prompt body"},
        {"mode": "start", "status": "started", "reason": None, "authorize": True},
        {"mode": [], "status": "started", "reason": None},
        {"mode": "start", "status": "started", "reason": "x" * 65},
    ],
)
def test_malformed_launch_display_metadata_is_ignored(bad):
    from tldw_chatbook.Chat.message_metadata import AgentHandoffLaunchMetadata

    assert AgentHandoffLaunchMetadata.read(bad) is None


def test_creation_without_owner_loop_persists_not_started(creation_controller):
    controller, source, db = creation_controller
    prepared = _prepared_creation(
        controller, source, destination="casual", mode="start"
    )
    controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    result = controller.execute_agent_chat_create(prepared)
    assert result["ok"] and result["launch_status"] == "not_started"
    row = db.get_conversation_by_id(result["conversation_id"])
    assert json.loads(row["metadata"])["console_agent_handoff"]["launch"] == {
        "mode": "start",
        "status": "not_started",
        "reason": "runtime_unavailable",
    }
    target = next(
        session
        for session in controller.store.sessions()
        if session.persisted_conversation_id == result["conversation_id"]
    )
    assert target.agent_handoff_launch.status == "not_started"


@pytest.mark.asyncio
async def test_stop_keeps_native_claim_until_generic_provider_adapter_exits(tmp_path):
    import asyncio
    from dataclasses import replace
    from threading import Event
    from Tests.Chat.test_console_agent_swap import _Gateway
    from Tests.console_provider_doubles import with_destination
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    entered, release, exited = Event(), Event(), Event()

    def adapter(**kwargs):
        entered.set()
        try:
            assert release.wait(30)
            return {
                "choices": [{"message": {"content": "finished"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
        finally:
            exited.set()

    gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)

    async def resolve(selection):
        return with_destination(
            replace(
                await _Gateway([]).resolve_for_send(selection),
                provider="openai",
                execution_key="openai",
                readiness_key="openai",
                base_url="https://api.openai.com/v1",
                api_key="test-only",
                streaming=False,
            )
        )

    gateway.resolve_for_send = resolve
    controller.provider_gateway = gateway
    controller._agent_bridge._gateway = gateway
    try:
        assert (await controller._chat_start.start(request)).launch_status == "started"
        assert await asyncio.to_thread(entered.wait, 5)
        store.switch_session(target.id)
        assert controller.stop_active_run()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*controller._chat_start.tasks())), 0.75
            )
        assert not exited.is_set()
        assert target.id in controller.fleet_wake._automatic_primary_claims
        release.set()
        await asyncio.wait_for(asyncio.gather(*controller._chat_start.tasks()), 5)
        assert exited.is_set()
        assert target.id not in controller.fleet_wake._automatic_primary_claims
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "launch_status,label,reason,reserved_generation",
    [
        ("not_started", "Not started", "runtime_disabled", 0),
        ("review_required", "Review required", "settlement_unconfirmed", 1),
    ],
)
async def test_manual_recovery_clears_handoff_attention_but_retains_launch_history(
    tmp_path, monkeypatch, launch_status, label, reason, reserved_generation
):
    """A consumed launch refusal must not keep successful human work blocked."""
    import asyncio
    from types import SimpleNamespace
    from Tests.UI.test_console_workspace_controller import _workspace_controller
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunStatus,
        ConsoleSubmissionOrigin,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_switcher_state import ActivityGroup

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    workspace = _workspace_controller(
        app_instance=SimpleNamespace(
            local_chat_conversation_service=ChatConversationService(
                store.persistence.db
            )
        ),
        chat_store_accessor=lambda: store,
        current_chat_store_accessor=lambda: store,
        current_chat_controller_accessor=lambda: controller,
    )

    def active_entry():
        return next(
            entry
            for entry in workspace.console_session_switcher_active_entries()
            if entry.conversation_id == request.conversation_id
        )

    async def unready(_selection):
        return None

    try:
        with monkeypatch.context() as failures:
            if launch_status == "not_started":
                failures.setattr(controller, "_agent_runtime_enabled", False)
            else:
                failures.setattr(controller, "_resolve_for_send_bounded", unready)
                failures.setattr(
                    runs.automatic_work,
                    "abort_chat_start",
                    lambda *args, **kwargs: False,
                )
            outcome = await controller._chat_start.start(request)
            await asyncio.gather(*controller._chat_start.tasks())
        assert (outcome.launch_status, outcome.reason) == (launch_status, reason)
        assert target.agent_handoff_state == "pending"
        assert active_entry().activity_state == "blocked"
        assert active_entry().group is ActivityGroup.WAITING_FOR_YOU
        launch = target.agent_handoff_launch
        assert launch.label == label

        result = await controller.submit_draft(target.draft, session_id=target.id)
        await asyncio.gather(*controller._active_stream_tasks.values())
        assert result.accepted and result.provider_started
        assert result.origin is ConsoleSubmissionOrigin.MANUAL
        assert controller.run_state_for(target.id).status is ConsoleRunStatus.COMPLETED
        assert target.agent_handoff_state == "consumed"
        activity = controller.activity_for(target.id)
        assert not activity.accepted_live_turn and activity.queued_count == 0
        assert active_entry().activity_state != "blocked"
        assert active_entry().group is not ActivityGroup.WAITING_FOR_YOU

        # Human recovery changes live custody, not the old launch or allowance.
        assert target.agent_handoff_launch == launch
        assert (
            runs.automatic_work.snapshot(chain).reserved["generation"]
            == reserved_generation
        )
        handoff = json.loads(
            store.persistence.db.get_conversation_by_id(request.conversation_id)[
                "metadata"
            ]
        )["console_agent_handoff"]
        assert handoff["state"] == "consumed" and handoff["draft"] == ""
        assert handoff["launch"] == {
            "mode": "start",
            "status": launch_status,
            "reason": reason,
        }
        restored = _restore_handoff(
            ConsoleChatStore(persistence=store.persistence), request.conversation_id
        )
        assert restored.agent_handoff_state == "consumed"
        assert restored.agent_handoff_launch == launch
        assert restored.draft == ""
        history = await workspace.load_console_session_switcher_history(
            query="", offset=0, limit=20
        )
        entry = next(
            entry
            for entry in history.entries
            if entry.conversation_id == request.conversation_id
        )
        assert label.lower() in entry.subtitle.lower()
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.fixture
def creation_workspace(creation_controller, tmp_path):
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    controller, source, _db = creation_controller
    database = WorkspaceDB(tmp_path / "availability.sqlite", client_id="availability")
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="named", name="Named")
    controller.store.persistence.workspace_registry = registry
    controller.app.workspace_registry_service = registry
    source.workspace_id = "named"
    yield registry
    database.close()


def _remove_destination(registry, change):
    if change == "archived":
        registry.archive_workspace("named")
    else:
        with registry.db.transaction() as connection:
            connection.execute(
                "DELETE FROM workspace_records WHERE workspace_id = ?", ("named",)
            )


@pytest.mark.parametrize("change", ["archived", "removed"])
@pytest.mark.parametrize("phase", ["before_approval", "after_approval"])
def test_unavailable_destination_refuses_creation_before_mutation(
    creation_controller, creation_workspace, change, phase
):
    controller, source, database = creation_controller
    before = (
        database.get_connection()
        .execute("SELECT COUNT(*) FROM conversations")
        .fetchone()[0]
    )
    if phase == "before_approval":
        _remove_destination(creation_workspace, change)
        with pytest.raises((ValueError, PermissionError)):
            _prepared_creation(controller, source)
    else:
        prepared = _prepared_creation(controller, source)
        controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
        assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
            "allow"
        ]
        _remove_destination(creation_workspace, change)
        outcome = controller.execute_agent_chat_create(prepared)
        assert outcome["ok"] is False, outcome
    assert (
        database.get_connection()
        .execute("SELECT COUNT(*) FROM conversations")
        .fetchone()[0]
        == before
    )
    assert controller.store.active_session_id == source.id
    assert not controller._chat_creation_records


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["archived", "removed", "unchanged"])
@pytest.mark.parametrize("phase", ["before_start", "readiness"])
async def test_destination_availability_is_rechecked_before_native_acceptance(
    tmp_path, change, phase
):
    import asyncio
    from dataclasses import replace
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    database = WorkspaceDB(tmp_path / "availability.sqlite", client_id="availability")
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="named", name="Named")
    store.persistence.workspace_registry = registry
    for session in (source, target):
        session.workspace_id = "named"
        with store.persistence.db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET scope_type = ?, workspace_id = ? WHERE id = ?",
                ("workspace", "named", session.persisted_conversation_id),
            )
    request = replace(
        request,
        configuration=controller.resolve_turn_configuration_snapshot(target.id),
        workspace_id=target.workspace_id,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    resolve = controller._resolve_for_send_bounded

    async def held_readiness(selection):
        entered.set()
        await release.wait()
        return await resolve(selection)

    controller._resolve_for_send_bounded = held_readiness
    try:
        if phase == "before_start":
            if change != "unchanged":
                _remove_destination(registry, change)
            release.set()
        pending = asyncio.create_task(controller._chat_start.start(request))
        if phase == "readiness":
            await asyncio.wait_for(entered.wait(), 5)
            if change != "unchanged":
                _remove_destination(registry, change)
            release.set()
        outcome = await asyncio.wait_for(pending, 10)
        await asyncio.gather(*controller._chat_start.tasks())
        if change == "unchanged":
            assert outcome.launch_status == "started", outcome
            assert controller.provider_gateway.calls == 1
        else:
            assert outcome.launch_status == "not_started", outcome
            assert outcome.reason == "destination_unavailable"
            assert controller.provider_gateway.calls == 0
            assert target.draft == "original"
            assert target.agent_handoff_state == "pending"
            assert runs.automatic_work.snapshot(chain).used["generation"] == 0
            assert not store.persistence.db.get_messages_for_conversation(
                request.conversation_id
            )
            assert (
                not store.persistence.db.get_connection()
                .execute(
                    "SELECT 1 FROM console_dispatch_checkpoints WHERE conversation_id = ?",
                    (request.conversation_id,),
                )
                .fetchall()
            )
        row = store.persistence.db.get_conversation_by_id(request.conversation_id)
        assert row["scope_type"] == "workspace" and row["workspace_id"] == "named"
        assert not controller._fleet_wake._automatic_primary_claims
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_runtime_update_during_readiness_rechecks_native_acceptance(
    tmp_path, enabled
):
    import asyncio

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    entered, release = asyncio.Event(), asyncio.Event()
    resolve = controller._resolve_for_send_bounded

    async def held_readiness(selection):
        entered.set()
        await release.wait()
        return await resolve(selection)

    controller._resolve_for_send_bounded = held_readiness
    try:
        pending = asyncio.create_task(controller._chat_start.start(request))
        await asyncio.wait_for(entered.wait(), 5)
        controller.update_agent_runtime(
            enabled=enabled, bridge=controller._agent_bridge
        )
        release.set()
        outcome = await asyncio.wait_for(pending, 10)
        await asyncio.gather(*controller._chat_start.tasks())
        await asyncio.sleep(0)  # Let the owned task's done callback remove it.
        if enabled:
            assert outcome.launch_status == "started", outcome
            assert controller.provider_gateway.calls == 1
        else:
            assert outcome.launch_status == "not_started", outcome
            assert outcome.reason == "runtime_disabled"
            assert controller.provider_gateway.calls == 0
            assert (
                target.draft == "original" and target.agent_handoff_state == "pending"
            )
            assert runs.automatic_work.snapshot(chain).used["generation"] == 0
            assert not store.persistence.db.get_messages_for_conversation(
                request.conversation_id
            )
            assert (
                not store.persistence.db.get_connection()
                .execute(
                    "SELECT 1 FROM console_dispatch_checkpoints WHERE conversation_id = ?",
                    (request.conversation_id,),
                )
                .fetchall()
            )
        assert not controller._fleet_wake._automatic_primary_claims
        assert not controller._chat_start.tasks()
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.parametrize("remembered", [False, True])
def test_missing_persona_notice_reaches_preview_and_normal_target_notice(
    creation_controller, creation_workspace, remembered
):
    from dataclasses import replace
    from types import SimpleNamespace
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

    controller, source, database = creation_controller
    creation_workspace.set_assistant_defaults(
        "named", WorkspaceAssistantDefaults(assistant_id="deleted-persona")
    )
    controller.app.local_character_persona_service = SimpleNamespace(
        get_persona_profile=lambda _identifier: None
    )
    settings = replace(controller._default_session_settings(), system_prompt=None)
    controller._default_session_settings = lambda: settings
    notices, cards = [], []
    controller.store._on_assistant_default_notice = notices.append
    expected = (
        "Workspace default Persona unavailable (persona_deleted). Started with None."
    )
    prepared = _prepared_creation(controller, source)
    if remembered:
        controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    else:

        def approve(payload):
            if payload:
                cards.append(payload)
                controller.resolve_pending_chat_create(
                    True, False, request_id=payload["request_id"]
                )

        controller.set_pending_chat_create = approve
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    outcome = controller.execute_agent_chat_create(prepared)
    assert outcome["ok"], outcome
    target = next(
        item
        for item in controller.store.sessions()
        if item.persisted_conversation_id == outcome["conversation_id"]
    )
    assert target.assistant_id == "console"
    assert target.assistant_default_notice == expected
    assert notices == [expected]
    assert prepared["assistant_default_notice"] == expected
    if remembered:
        assert not cards
    else:
        assert cards[0]["assistant_default_notice"] == expected
    assert (
        "persona_deleted"
        not in database.get_conversation_by_id(outcome["conversation_id"])["metadata"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["backend", "bridge", "owner", "destination"])
async def test_prepared_native_start_keeps_exact_runtime_and_destination(
    tmp_path, change
):
    import asyncio
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    entered, release = asyncio.Event(), asyncio.Event()
    resolve = controller._resolve_for_send_bounded
    original_owner = controller._fleet_wake.runtime_owner_id

    async def held_readiness(selection):
        entered.set()
        await release.wait()
        return await resolve(selection)

    controller._resolve_for_send_bounded = held_readiness
    try:
        pending = asyncio.create_task(controller._chat_start.start(request))
        await asyncio.wait_for(entered.wait(), 5)
        if change == "backend":
            target.runtime_backend = "server"
        elif change == "bridge":
            bridge = ConsoleAgentBridge(
                agent_runs_db=runs,
                store=store,
                provider_gateway=controller.provider_gateway,
            )
            bridge._live_primary_runs[source.persisted_conversation_id] = (
                request.source_run_id
            )
            controller.update_agent_runtime(enabled=True, bridge=bridge)
        elif change == "owner":
            controller._fleet_wake._owner_id = "replacement-owner"
        else:
            target.workspace_id = "different-workspace"
        release.set()
        outcome = await asyncio.wait_for(pending, 10)
        await asyncio.gather(*controller._chat_start.tasks())
        assert outcome.launch_status == "not_started", outcome
        assert controller.provider_gateway.calls == 0
        assert target.draft == "original" and target.agent_handoff_state == "pending"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 0
        attempt = runs.automatic_work.read_chat_start_attempt(
            request.attempt_id, owner_id=original_owner
        )
        assert attempt.state == "aborted"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert not controller._fleet_wake._automatic_primary_claims
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("moved", [False, True])
async def test_created_start_keeps_approved_destination_during_library_capture(
    tmp_path, moved
):
    import asyncio
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    workspaces = WorkspaceDB(tmp_path / "moved.sqlite", client_id="moved-start")
    registry = LocalWorkspaceRegistryService(workspaces)
    registry.create_workspace(workspace_id="named", name="Named")
    store.persistence.workspace_registry = registry
    controller._owner_loop = asyncio.get_running_loop()
    capture = store.library_policy_coordinator.capture_for_execution
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_capture(session_id):
        entered.set()
        await release.wait()
        return await capture(session_id)

    store.library_policy_coordinator.capture_for_execution = held_capture
    approved = {
        "source_run_id": request.source_run_id,
        "session_id": source.id,
        "source_incarnation": source.incarnation_id,
        "opening_prompt": "original",
        "scope_type": "global",
        "workspace_id": None,
    }
    try:
        pending = asyncio.create_task(
            asyncio.to_thread(controller._start_created_chat, approved, target)
        )
        await asyncio.wait_for(entered.wait(), 5)
        if moved:
            target.workspace_id = "named"
            with store.persistence.db.transaction() as cursor:
                cursor.execute(
                    "UPDATE conversations SET scope_type = ?, workspace_id = ? WHERE id = ?",
                    ("workspace", "named", target.persisted_conversation_id),
                )
        release.set()
        outcome = await asyncio.wait_for(pending, 10)
        await asyncio.gather(*controller._chat_start.tasks())
        if not moved:
            assert outcome["launch_status"] == "started", outcome
            assert controller.provider_gateway.calls == 1
            assert target.agent_handoff_state == "consumed"
        else:
            assert outcome["launch_status"] == "not_started", outcome
            assert outcome["reason"] == "target_changed"
            assert target.workspace_id == "named"
            assert (
                target.draft == "original" and target.agent_handoff_state == "pending"
            )
            assert controller.provider_gateway.calls == 0
            assert runs.automatic_work.snapshot(chain).used["generation"] == 0
            assert not store.persistence.db.get_messages_for_conversation(
                request.conversation_id
            )
    finally:
        release.set()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()
        workspaces.close()


@pytest.mark.asyncio
async def test_native_maintenance_refuses_before_attempt_or_shared_claim(tmp_path):
    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    try:
        controller.maintenance_close_admission()
        assert not controller._fleet_wake.try_claim_automatic_primary(
            target.id, object()
        )
        result = await controller._chat_start.start(request)
        assert result.launch_status == "not_started"
        assert result.reason == "maintenance"
        assert target.draft == "original"
        with runs.connection() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM automatic_chat_start_attempts"
                ).fetchone()[0]
                == 0
            )
        assert runs.automatic_work.snapshot(chain).used["generation"] == 0
    finally:
        controller.maintenance_resume()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close()


@pytest.mark.asyncio
async def test_native_start_never_enrolls_queue_stop_parent(tmp_path):
    import asyncio

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    try:
        result = await controller._chat_start.start(request)
        assert result.launch_status == "started"
        await asyncio.gather(*controller._chat_start.tasks())
        assert target.id not in controller.prompt_queue_coordinator._chains
        assert target.id not in controller.prompt_queue_coordinator._stop_parents
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
        assert (
            store.persistence.db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == 0
        )
        assert controller.provider_gateway.calls == 1
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_initialization", [False, True])
async def test_native_configured_v2_initialization_and_stop_proposal(
    tmp_path, allow_initialization
):
    import asyncio
    from types import SimpleNamespace
    from Tests.Agents.test_hooks_v2_execution import command
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    app = SimpleNamespace(app_config={})
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    controller.app = app
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    initialized = tmp_path / "native-initialized"
    stopped = tmp_path / "native-stop"
    init = command(
        "from pathlib import Path;"
        + f"Path({str(initialized)!r}).write_text('initialized');"
        + f"raise SystemExit({0 if allow_initialization else 1})",
        name="SessionStart",
        required=True,
    )
    stop = command(
        "from pathlib import Path;"
        + f"Path({str(stopped)!r}).write_text('stop');"
        + 'print(\'{"version":2,"decision":"pass","continuation":{"message":"escaped follow up"}}\')',
        name="Stop",
        effects=["continuation"],
    ).model_copy(update={"id": "native-stop"})
    runtime.ensure_hooks_v2(target.id, (init, stop), lambda *_: True)
    try:
        result = await controller._chat_start.start(request)
        await asyncio.gather(*controller._chat_start.tasks())
        assert initialized.exists(), "required native initialization never ran"
        assert result.launch_status == (
            "started" if allow_initialization else "not_started"
        )
        assert controller.provider_gateway.calls == int(allow_initialization)
        assert runs.automatic_work.snapshot(chain).used["generation"] == int(
            allow_initialization
        )
        assert target.id not in controller.prompt_queue_coordinator._chains
        assert target.id not in controller.prompt_queue_coordinator._stop_parents
        assert not stopped.exists(), (
            "native start escaped into queue-owned Stop scheduling"
        )
        assert (
            store.persistence.db.get_connection()
            .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
            .fetchone()[0]
            == 0
        )
    finally:
        await runtime.close_hooks_v2()
        await runtime.dispose()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_native_commit_keeps_exact_worker_and_capacity_until_repeated_cancel_drains(
    tmp_path,
):
    import asyncio
    from threading import Event

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    entered, release, exited = Event(), Event(), Event()
    original = store.commit_durable_turn
    baseline = len(store.persistence.db._connection_quiescence._connections)

    def held(acceptance):
        entered.set()
        try:
            assert release.wait(10)
            return original(acceptance)
        finally:
            exited.set()

    store.commit_durable_turn = held
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        item = controller._chat_start._active[target.id]
        store.switch_session(target.id)
        assert controller.stop_active_run()
        await asyncio.sleep(0.05)
        item.task.cancel()
        await asyncio.sleep(0.05)
        assert target.id in controller.fleet_wake._automatic_primary_claims
        assert not item.outcome.done()
        assert controller._chat_start.tasks() and not exited.is_set()
        release.set()
        assert (await start).launch_status == "review_required"
        await asyncio.gather(*controller._chat_start.tasks())
        assert exited.is_set()
        assert not controller.fleet_wake._automatic_primary_claims
        assert runs.automatic_work.snapshot(chain).status == "review_required"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
        rows = store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert rows and not any(
            row["sender"] == "assistant" and row["content"] for row in rows
        )
        assert len(store.persistence.db._connection_quiescence._connections) == baseline
    finally:
        release.set()
        await asyncio.gather(start, return_exceptions=True)
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["preaccept", "accepted_cleanup"])
async def test_unexpected_start_failure_is_type_only_and_accepted_cleanup_is_review(
    tmp_path, monkeypatch, phase
):
    import asyncio
    from loguru import logger

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    secret = "private-prompt-api-key-should-never-appear"
    records = []
    sink = logger.add(lambda message: records.append(message.record))
    if phase == "preaccept":

        async def fail(_selection):
            raise ValueError(secret)

        monkeypatch.setattr(controller, "_resolve_for_send_bounded", fail)
    else:
        accept = runs.automatic_work.accept_chat_start

        def fail(*args, **kwargs):
            assert accept(*args, **kwargs)
            raise ValueError(secret)

        monkeypatch.setattr(runs.automatic_work, "accept_chat_start", fail)
    try:
        outcome = await controller._chat_start.start(request)
        await asyncio.gather(*controller._chat_start.tasks())
        assert (outcome.launch_status, outcome.reason) == (
            ("not_started", "start_failed")
            if phase == "preaccept"
            else ("review_required", "receipt_unconfirmed")
        )
        failures = [
            record for record in records if "Chat start failed" in record["message"]
        ]
        assert failures and all(
            "ValueError" in record["message"] for record in failures
        )
        assert all(
            secret not in str(record) and record["exception"] is None
            for record in records
        )
        assert target.draft == "original"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert runs.automatic_work.snapshot(chain).used["generation"] == int(
            phase == "accepted_cleanup"
        )
        assert runs.automatic_work.read_chat_start_attempt(
            request.attempt_id, owner_id=controller.fleet_wake.runtime_owner_id
        ).state == ("aborted" if phase == "preaccept" else "review_required")
    finally:
        logger.remove(sink)
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("settlement_result", ["raise", "false"])
async def test_failed_review_settlement_blocks_siblings_across_ledger_handles(
    tmp_path, monkeypatch, alias, settlement_result
):
    import asyncio
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    owner = controller.fleet_wake.runtime_owner_id

    def fail_commit(acceptance):
        raise OSError("private commit failure")

    from threading import Event

    entered, release = Event(), Event()

    def fail_settlement(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        if settlement_result == "raise":
            raise OSError("private ledger failure")
        return False

    monkeypatch.setattr(store, "commit_durable_turn", fail_commit)
    settle = runs.automatic_work.mark_chat_start_review_required
    monkeypatch.setattr(
        runs.automatic_work, "mark_chat_start_review_required", fail_settlement
    )
    peer_path = runs.db_path
    if alias:
        alias_dir = tmp_path / "alias-dir"
        alias_dir.mkdir()
        peer_path = alias_dir / ".." / runs.db_path.name
    peer = AgentRunsDB(peer_path, client_id="peer", reconcile_on_init=False)
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        # Denial starts before uncertain settlement I/O returns.
        with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
            peer.automatic_work.check_active(chain, owner_id=owner)
        release.set()
        outcome = await start
        await asyncio.gather(*controller._chat_start.tasks())
        assert (outcome.launch_status, outcome.reason) == (
            "review_required",
            "settlement_unconfirmed",
        )
        assert peer.automatic_work.snapshot(chain).status == "active"
        with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
            peer.automatic_work.check_active(chain, owner_id=owner)
        with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
            peer.automatic_work.prepare_chat_start(
                attempt_id="sibling",
                source_run_id=request.source_run_id,
                target_conversation_id="sibling",
                target_session_id="sibling",
                target_session_incarnation="sibling",
                owner_id=owner,
                draft_revision=1,
                context_epoch=0,
                request_fingerprint="a" * 64,
            )
        # Verified durable reconciliation replaces transient denial with the root pause.
        assert settle(request.attempt_id, owner_id=owner)
        with pytest.raises(AutomaticWorkRefused, match="interrupted_work"):
            peer.automatic_work.check_active(chain, owner_id=owner)
    finally:
        release.set()
        await asyncio.gather(start, return_exceptions=True)
        peer.close()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_native_acceptance_refuses_real_writer_contention_and_restores_timeout(
    tmp_path, monkeypatch
):
    import asyncio
    import sqlite3
    import time

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    original_accept = controller._chat_start.accept
    blocker = sqlite3.connect(runs.db_path)
    # Warm and borrow the real loop-thread owner before taking the independent lock.
    with runs.connection() as connection:
        connection.execute("PRAGMA busy_timeout=1739")
        original_sync = connection.execute("PRAGMA synchronous").fetchone()[0]
    duration = []
    ticker = []
    running = True

    async def tick():
        while running:
            ticker.append(time.perf_counter())
            await asyncio.sleep(0.005)

    async def contended(item):
        blocker.execute("BEGIN IMMEDIATE")
        started = time.perf_counter()
        try:
            return await original_accept(item)
        finally:
            duration.append(time.perf_counter() - started)
            blocker.rollback()

    monkeypatch.setattr(controller._chat_start, "accept", contended)
    ticking = asyncio.create_task(tick())
    try:
        outcome = await controller._chat_start.start(request)
        await asyncio.gather(*controller._chat_start.tasks())
        await asyncio.sleep(0.02)
        assert duration[0] < 0.25, duration
        assert max(b - a for a, b in zip(ticker, ticker[1:])) < 0.25
        print(
            f"contention_cutoff_seconds={duration[0]:.6f}; ticker_max_gap_seconds={max(b - a for a, b in zip(ticker, ticker[1:])):.6f}"
        )
        assert (outcome.launch_status, outcome.reason) == (
            "not_started",
            "ledger_contended",
        )
        with runs.connection() as same:
            assert same is connection
            assert same.execute("PRAGMA busy_timeout").fetchone()[0] == 1739
            assert same.execute("PRAGMA synchronous").fetchone()[0] == original_sync
        assert target.draft == "original"
        assert runs.automatic_work.snapshot(chain).used["generation"] == 0
        assert (
            runs.automatic_work.read_chat_start_attempt(
                request.attempt_id, owner_id=controller.fleet_wake.runtime_owner_id
            ).state
            == "aborted"
        )
    finally:
        running = False
        await ticking
        blocker.close()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_native_uncontended_acceptance_restores_full_policy_and_retires_owned_handle(
    tmp_path, monkeypatch
):
    import asyncio
    import time

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    original = controller._chat_start.accept
    timings = []

    # No borrowed handle: the acceptance operation must retire its own handle.
    async def measured(item):
        runs.close()
        started = time.perf_counter()
        try:
            return await original(item)
        finally:
            timings.append(time.perf_counter() - started)
            assert getattr(runs._thread_local, "conn", None) is None

    monkeypatch.setattr(controller._chat_start, "accept", measured)
    try:
        assert (await controller._chat_start.start(request)).launch_status == "started"
        await asyncio.gather(*controller._chat_start.tasks())
        assert runs.automatic_work.snapshot(chain).used["generation"] == 1
        print(f"uncontended_cutoff_seconds={timings[0]:.6f}")
    finally:
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


def _seed_close_source_message(controller, source):
    """Give actual Close the live message owned by the original rig's turn."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole

    controller.store.append_message(
        source.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        message_id=controller._active_assistant_message_ids[source.id],
    )


@pytest.mark.parametrize("close_before_decision", [False, True])
def test_prepared_worker_decision_and_close_terminate(
    creation_controller, close_before_decision
):
    """Exercise the actual decision lock with a prepared primary worker."""
    import threading
    from Tests.Chat.test_console_skill_script_confirm import _wait_until

    controller, source, _db = creation_controller
    _seed_close_source_message(controller, source)
    prepared = _prepared_creation(controller, source)
    controller.set_pending_chat_create = lambda payload: None
    decisions = []

    def decide():
        with use_run_id("source-run"):
            decisions.append(
                controller.request_chat_create_confirm(prepared, session_id=source.id)
            )

    worker = threading.Thread(target=decide, daemon=True)
    worker.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    request_id = controller.pending_chat_create_ids()[0]
    if close_before_decision:
        controller.begin_session_close(
            source.id,
            expected_revision=controller.lifecycle_impact(
                session_id=source.id
            ).revision,
        )
    controller.resolve_pending_chat_create(True, True, request_id=request_id)
    worker.join(5)
    assert not worker.is_alive(), (
        "prepared decision recursively acquired the plain lock"
    )
    assert decisions == [
        {"allow": not close_before_decision, "remember": not close_before_decision}
    ]
    if close_before_decision:
        assert not controller._chat_creation_records
        assert not controller._chat_create_session_grants.get(source.id)
    else:
        assert controller._chat_creation_record(prepared)["approved"]
        assert controller._chat_create_session_grants[source.id] == {
            prepared["_grant_scope"]
        }
        prepared["_creation_token"].close()


@pytest.mark.parametrize("mode", ["draft", "start"])
@pytest.mark.parametrize("phase", ["before_save", "saved", "queued_restore"])
def test_prepared_close_preserves_only_successfully_saved_draft(
    creation_controller, monkeypatch, mode, phase
):
    controller, source, db = creation_controller
    _seed_close_source_message(controller, source)
    prepared = _prepared_creation(controller, source, mode=mode)
    controller._chat_create_session_grants[source.id] = {prepared["_grant_scope"]}
    assert controller.request_chat_create_confirm(prepared, session_id=source.id)[
        "allow"
    ]
    completions, starts, saves = [], [], []
    controller.complete_agent_chat_create = lambda **kwargs: completions.append(kwargs)
    monkeypatch.setattr(
        controller, "_start_created_chat", lambda *args: starts.append(args)
    )

    def close():
        controller.begin_session_close(
            source.id,
            expected_revision=controller.lifecycle_impact(
                session_id=source.id
            ).revision,
        )

    original_save = (
        controller.store.persistence.persist_console_conversation_with_policy
    )

    def save(**kwargs):
        original_save(**kwargs)
        saves.append(kwargs["conversation_id"])
        if phase == "saved":
            close()

    monkeypatch.setattr(
        controller.store.persistence, "persist_console_conversation_with_policy", save
    )
    if phase == "before_save":
        available = controller._chat_creation_destination_available

        def retire_before_save(**kwargs):
            result = available(**kwargs)
            close()
            return result

        monkeypatch.setattr(
            controller, "_chat_creation_destination_available", retire_before_save
        )
    elif phase == "queued_restore":
        original_marshal = controller.app.call_from_thread

        def marshal(callback, *args, **kwargs):
            if callback.__name__ == "restore":
                close()
            return original_marshal(callback, *args, **kwargs)

        controller.app.call_from_thread = marshal
    result = controller.execute_agent_chat_create(prepared)
    assert not starts and not completions
    assert not controller._chat_creation_records
    assert len(controller.store.sessions()) == 1
    if phase == "before_save":
        assert not result["ok"]
        assert not saves
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM conversations")
            .fetchone()[0]
            == 1
        )
        return
    assert result["ok"] and result["conversation_id"] == saves[0]
    assert result["launch_status"] == ("draft" if mode == "draft" else "not_started")
    assert result["reason"] == "source_unavailable"
    row = db.get_conversation_by_id(saves[0])
    assert row and not row["deleted"]
    handoff = json.loads(row["metadata"])["console_agent_handoff"]
    assert handoff == {
        "version": 2,
        "created_via": "new_chat",
        "state": "pending",
        "draft_revision": 1,
        "draft": "opening",
        "source_run_id": "source-run",
        "launch": {
            "mode": mode,
            "status": result["launch_status"],
            "reason": "source_unavailable",
        },
    }
    assert not db.get_messages_for_conversation(saves[0])
    assert not controller.execute_agent_chat_create(prepared)["ok"]
    assert len(saves) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("registration_failure", [False, True])
@pytest.mark.parametrize("another_restriction", [False, True])
async def test_failed_preparation_with_no_attempt_clears_only_its_restriction(
    tmp_path, monkeypatch, registration_failure, another_restriction
):
    """A rolled-back reservation must not leave every unrelated root blocked."""
    import asyncio
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    ledger = runs.automatic_work
    owner = controller.fleet_wake.runtime_owner_id
    ledger.recover(current_owner_id=owner)
    unrelated = ledger.create_chain("unrelated", root_submission_id="unrelated")
    peer = AgentRunsDB(runs.db_path, client_id="peer", reconcile_on_init=False)
    with runs.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER deny_start BEFORE INSERT ON automatic_chat_start_attempts BEGIN SELECT RAISE(ABORT, 'private rollback'); END"
        )
    register = ledger._restrict_chat_start

    def register_with_peer(*args, **kwargs):
        if another_restriction:
            register("another", owner_id=owner, chain_id=chain)
        if registration_failure:
            raise OSError("private identity failure")
        return register(*args, **kwargs)

    monkeypatch.setattr(ledger, "_restrict_chat_start", register_with_peer)
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        await asyncio.sleep(0)
        item = controller._chat_start._active[target.id]
        results = await asyncio.gather(item.task, return_exceptions=True)
        assert results == [None]
        assert item.outcome.done()
        outcome = await start
        assert (outcome.launch_status, outcome.reason) == (
            "not_started",
            "start_failed",
        )
        with runs.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_chat_start_attempts"
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_work_reservations"
                ).fetchone()[0]
                == 0
            )
        assert target.draft == "original"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        peer.automatic_work.check_active(unrelated, owner_id=owner)
        if another_restriction:
            with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
                peer.automatic_work.check_active(chain, owner_id=owner)
        else:
            peer.automatic_work.check_active(chain, owner_id=owner)
        assert target.id not in controller.fleet_wake._automatic_primary_claims
        assert target.id not in controller._chat_start._active
    finally:
        start.cancel()
        await asyncio.gather(start, return_exceptions=True)
        monkeypatch.setattr(ledger, "_restrict_chat_start", register)
        ledger._clear_chat_start_restriction("another", owner_id=owner)
        ledger._clear_chat_start_restriction(request.attempt_id, owner_id=owner)
        peer.close()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["prepared", "accepted", "observation_failure", "missing_runtime", "stale_runtime"],
)
async def test_preparation_exception_reconciles_persisted_authority_conservatively(
    tmp_path, monkeypatch, case
):
    """A lost preparation return is not evidence that committed authority vanished."""
    import asyncio
    import sqlite3
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    ledger = runs.automatic_work
    owner = controller.fleet_wake.runtime_owner_id
    ledger.recover(current_owner_id=owner)
    unrelated = ledger.create_chain("unrelated", root_submission_id="unrelated")
    peer = AgentRunsDB(runs.db_path, client_id="peer", reconcile_on_init=False)
    prepare = ledger.prepare_chat_start

    def commit_then_raise(**kwargs):
        if case in {"prepared", "accepted"}:
            prepare(**kwargs)
            if case == "accepted":
                assert ledger.accept_chat_start(request.attempt_id, owner_id=owner)
        else:
            with ledger.transaction() as conn:
                if case == "missing_runtime":
                    conn.execute("DELETE FROM automatic_work_runtime_owner")
                elif case == "stale_runtime":
                    conn.execute(
                        "UPDATE automatic_work_runtime_owner SET owner_id='replacement'"
                    )
        raise OSError("private lost preparation return")

    monkeypatch.setattr(ledger, "prepare_chat_start", commit_then_raise)
    if case == "observation_failure":

        def fail_observation(*args, **kwargs):
            raise sqlite3.OperationalError("private observation failure")

        monkeypatch.setattr(
            ledger, "_confirm_chat_start_absent", fail_observation, raising=False
        )
    try:
        outcome = await controller._chat_start.start(request)
        await asyncio.gather(*controller._chat_start.tasks())
        expected_reason = {
            "prepared": "preparation_unconfirmed",
            "accepted": "receipt_unconfirmed",
        }.get(case, "settlement_unconfirmed")
        assert (outcome.launch_status, outcome.reason) == (
            "review_required",
            expected_reason,
        )
        assert ledger.snapshot(chain).used["generation"] == int(case == "accepted")
        assert ledger.snapshot(chain).reserved["generation"] == 0
        if case in {"prepared", "accepted"}:
            assert ledger.read_chat_start_attempt(
                request.attempt_id, owner_id=owner
            ).state == ("aborted" if case == "prepared" else "review_required")
            peer.automatic_work.check_active(unrelated, owner_id=owner)
            if case == "accepted":
                with pytest.raises(AutomaticWorkRefused, match="interrupted_work"):
                    peer.automatic_work.check_active(chain, owner_id=owner)
        elif case != "stale_runtime":
            with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
                peer.automatic_work.check_active(unrelated, owner_id=owner)
        else:
            with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
                peer.automatic_work.check_active(unrelated, owner_id=owner)
        assert target.draft == "original"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert target.id not in controller.fleet_wake._automatic_primary_claims
    finally:
        ledger._clear_chat_start_restriction(request.attempt_id, owner_id=owner)
        peer.close()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("registration_site", ["first", "second", "first_uncertain"])
@pytest.mark.parametrize("worker", ["commit", "provider"])
async def test_restriction_registration_failure_still_drains_exact_native_owners(
    tmp_path, monkeypatch, registration_site, worker
):
    """Registration failure must not strand native physical custody or its result."""
    import asyncio
    from threading import Event
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    ledger = runs.automatic_work
    owner = controller.fleet_wake.runtime_owner_id
    ledger.recover(current_owner_id=owner)
    peer = AgentRunsDB(runs.db_path, client_id="peer", reconcile_on_init=False)
    entered, release, exited = Event(), Event(), Event()
    original = (
        store.commit_durable_turn
        if worker == "commit"
        else controller._agent_bridge.run_reply
    )

    def held(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(10)
            return original(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(
        store if worker == "commit" else controller._agent_bridge,
        "commit_durable_turn" if worker == "commit" else "run_reply",
        held,
    )
    register = ledger._restrict_chat_start
    registrations = []

    def broken_registration(*args, **kwargs):
        registrations.append(kwargs["chain_id"])
        if len(registrations) == (2 if registration_site == "second" else 1):
            raise OSError("private registration failure")
        return register(*args, **kwargs)

    monkeypatch.setattr(ledger, "_restrict_chat_start", broken_registration)
    if registration_site != "first":

        def fail_settlement(*args, **kwargs):
            raise OSError("private settlement failure")

        monkeypatch.setattr(ledger, "mark_chat_start_review_required", fail_settlement)
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        item = controller._chat_start._active[target.id]
        captured_owner, captured_token = item.capacity_owner, item.token
        # The real provider has passed the two commits; simulate a lost local receipt
        # flag while retaining its already-published outcome and physical worker.
        if worker == "provider":
            assert (await start).launch_status == "started"
            item.receipted = False
        item.task.cancel()
        await asyncio.sleep(0.05)
        item.task.cancel()
        await asyncio.sleep(0.05)
        assert captured_owner._automatic_primary_claims[target.id] is captured_token
        assert not exited.is_set()
        assert not item.task.done()
        if worker == "commit":
            assert not item.outcome.done()
        if registration_site == "second":
            with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
                peer.automatic_work.check_active(chain, owner_id=owner)
        release.set()
        assert await asyncio.gather(item.task, return_exceptions=True) == [None]
        assert exited.is_set()
        assert item.outcome.done()
        assert target.id not in captured_owner._automatic_primary_claims
        assert target.id not in controller._chat_start._active
        assert ledger.snapshot(chain).used["generation"] == 1
        if registration_site == "first":
            assert ledger.snapshot(chain).status == "review_required"
        else:
            with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
                peer.automatic_work.check_active(chain, owner_id=owner)
    finally:
        release.set()
        start.cancel()
        await asyncio.gather(start, return_exceptions=True)
        await asyncio.gather(*controller._chat_start.tasks(), return_exceptions=True)
        ledger._clear_chat_start_restriction(request.attempt_id, owner_id=owner)
        peer.close()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_contended_preparation_confirms_absence_after_writer_exits(
    tmp_path, monkeypatch
):
    """A real failed BEGIN and drained writer leave no automatic charge or denial."""
    import asyncio
    import sqlite3
    from threading import Event
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    ledger = runs.automatic_work
    owner = controller.fleet_wake.runtime_owner_id
    ledger.recover(current_owner_id=owner)
    unrelated = ledger.create_chain("unrelated", root_submission_id="unrelated")
    peer = AgentRunsDB(runs.db_path, client_id="peer", reconcile_on_init=False)
    blocker = sqlite3.connect(runs.db_path)
    failed, released = Event(), Event()
    prepare = ledger.prepare_chat_start

    def contended(**kwargs):
        with runs.connection() as conn:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.execute("PRAGMA busy_timeout=0")
            try:
                return prepare(**kwargs)
            except sqlite3.OperationalError:
                failed.set()
                assert released.wait(10)
                raise
            finally:
                conn.execute(f"PRAGMA busy_timeout={timeout}")

    monkeypatch.setattr(ledger, "prepare_chat_start", contended)
    blocker.execute("BEGIN IMMEDIATE")
    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        assert await asyncio.to_thread(failed.wait, 5)
        item = controller._chat_start._active[target.id]
        assert not item.outcome.done()
        assert target.id in controller.fleet_wake._automatic_primary_claims
        blocker.rollback()
        released.set()
        outcome = await start
        await asyncio.gather(*controller._chat_start.tasks())
        assert (outcome.launch_status, outcome.reason) == (
            "not_started",
            "start_failed",
        )
        with runs.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_chat_start_attempts"
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_work_reservations"
                ).fetchone()[0]
                == 0
            )
        peer.automatic_work.check_active(chain, owner_id=owner)
        peer.automatic_work.check_active(unrelated, owner_id=owner)
        assert target.draft == "original"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
        assert target.id not in controller.fleet_wake._automatic_primary_claims
    finally:
        blocker.rollback()
        blocker.close()
        released.set()
        await asyncio.gather(start, return_exceptions=True)
        ledger._clear_chat_start_restriction(request.attempt_id, owner_id=owner)
        peer.close()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["publication", "abandonment", "run_state", "replacement"]
)
async def test_post_drain_cleanup_keeps_outcome_and_exact_capacity_release(
    tmp_path, monkeypatch, fault
):
    """Fallible UI cleanup cannot retain a drained slot or steal a replacement."""
    import asyncio
    from copy import copy
    from dataclasses import replace
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    coordinator = controller._chat_start
    ledger = runs.automatic_work
    abort = ledger.abort_chat_start
    abandon = controller._abandon_preparation
    set_state = controller._set_run_state
    saved = {}

    def fail(*args, **kwargs):
        raise OSError("private cleanup failure")

    async def refuse_accept(item):
        saved["item"] = item
        saved["preparation"] = store.preparation_for_session(target.id)
        assert saved["preparation"] is not None
        if fault == "abandonment":
            monkeypatch.setattr(controller, "_abandon_preparation", fail)
        raise OSError("private preaccept refusal")

    monkeypatch.setattr(coordinator, "accept", refuse_accept)

    def refunded(*args, **kwargs):
        result = abort(*args, **kwargs)
        assert result
        item = saved["item"]
        if fault == "run_state":
            # Re-establish the exact validation state to exercise its cleanup fence.
            set_state(item.validating_state, session_id=target.id)
            monkeypatch.setattr(controller, "_set_run_state", fail)
        return result

    monkeypatch.setattr(ledger, "abort_chat_start", refunded)
    publish = coordinator._publish_outcome

    async def publication(*args, **kwargs):
        item = saved["item"]
        if fault == "publication":
            raise OSError("private publication failure")
        if fault == "replacement":
            replacement = copy(item)
            replacement.token = object()
            replacement.preparation_id = "replacement-preparation"
            coordinator._active[target.id] = replacement
            item.capacity_owner._automatic_primary_claims[target.id] = replacement.token
            saved["replacement"] = replacement
            controller._rollback_committing_preparation(
                saved["preparation"].preparation_id
            )
            assert store.preparation_for_session(target.id) is None
            replacement_preparation = replace(
                saved["preparation"], preparation_id="replacement-preparation"
            )
            assert (
                store.begin_preparation(replacement_preparation)
                is replacement_preparation
            )
            saved["replacement_preparation"] = replacement_preparation
            saved["state"] = ConsoleRunState(ConsoleRunStatus.VALIDATING, "replacement")
            set_state(saved["state"], session_id=target.id)
        return await publish(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_publish_outcome", publication)
    start = asyncio.create_task(coordinator.start(request))
    try:
        await asyncio.sleep(0)
        item = coordinator._active[target.id]
        results = await asyncio.gather(item.task, return_exceptions=True)
        assert item.outcome.done()
        assert results == [None]
        outcome = await start
        assert outcome.launch_status == (
            "review_required" if fault == "publication" else "not_started"
        )
        assert ledger.snapshot(chain).reserved["generation"] == 0
        if fault == "replacement":
            replacement = saved["replacement"]
            assert coordinator._active[target.id] is replacement
            assert (
                item.capacity_owner._automatic_primary_claims[target.id]
                is replacement.token
            )
            assert controller.run_state_for(target.id) is saved["state"]
            assert (
                store.preparation_for_session(target.id)
                is saved["replacement_preparation"]
            )
        else:
            assert target.id not in coordinator._active
            assert target.id not in item.capacity_owner._automatic_primary_claims
    finally:
        monkeypatch.setattr(controller, "_abandon_preparation", abandon)
        monkeypatch.setattr(controller, "_set_run_state", set_state)
        coordinator._active.pop(target.id, None)
        controller.fleet_wake._automatic_primary_claims.pop(target.id, None)
        start.cancel()
        await asyncio.gather(start, return_exceptions=True)
        if "preparation" in saved:
            abandon(saved["preparation"].preparation_id)
        if "replacement_preparation" in saved:
            abandon(saved["replacement_preparation"].preparation_id)
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["canonical_resolution", "persistent_registration"])
@pytest.mark.parametrize("worker", ["commit", "provider"])
@pytest.mark.parametrize("replacement", [False, True])
async def test_persistent_registration_and_settlement_failure_retains_native_denial(
    tmp_path, monkeypatch, failure, worker, replacement
):
    """Drained accepted work remains charged and denied despite both cleanup faults."""
    import asyncio
    from copy import copy
    from pathlib import Path
    import sys
    from threading import Event
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    ledger = runs.automatic_work
    owner = controller.fleet_wake.runtime_owner_id
    ledger.recover(current_owner_id=owner)
    unrelated = ledger.create_chain("unrelated", root_submission_id="unrelated")
    alias = tmp_path / "alias"
    alias.mkdir()
    peer = AgentRunsDB(
        alias / ".." / runs.db_path.name, client_id="peer", reconcile_on_init=False
    )
    peer_ledger = peer.automatic_work
    entered, release, exited = Event(), Event(), Event()
    original = (
        store.commit_durable_turn
        if worker == "commit"
        else controller._agent_bridge.run_reply
    )

    def held(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(10)
            return original(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(
        store if worker == "commit" else controller._agent_bridge,
        "commit_durable_turn" if worker == "commit" else "run_reply",
        held,
    )
    # Real SQLite rejects the durable settlement transaction, after acceptance.
    with runs.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER deny_review BEFORE UPDATE OF state ON automatic_chat_start_attempts WHEN NEW.state='review_required' BEGIN SELECT RAISE(ABORT, 'private settlement failure'); END"
        )
    resolve = Path.resolve

    def fail_resolution(path, *args, **kwargs):
        if (
            path == runs.db_path
            and sys._getframe(1).f_globals.get("__name__")
            == "tldw_chatbook.DB.automatic_work"
        ):
            raise OSError("private canonical identity failure")
        return resolve(path, *args, **kwargs)

    registrations = []

    def fail_registration(*args, **kwargs):
        registrations.append(kwargs["chain_id"])
        raise OSError("private persistent registration failure")

    start = asyncio.create_task(controller._chat_start.start(request))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        item = controller._chat_start._active[target.id]
        captured_owner, token = item.capacity_owner, item.token
        if worker == "provider":
            assert (await start).launch_status == "started"
            item.receipted = False
        # Identity is known before this real filesystem failure starts.
        monkeypatch.setattr(Path, "resolve", fail_resolution)
        if failure == "persistent_registration":
            monkeypatch.setattr(ledger, "_restrict_chat_start", fail_registration)
        item.task.cancel()
        await asyncio.sleep(0.05)
        item.task.cancel()
        await asyncio.sleep(0.05)
        assert captured_owner._automatic_primary_claims[target.id] is token
        assert not item.task.done() and not exited.is_set()
        if worker == "commit":
            assert not item.outcome.done()
        if replacement:
            replacement_item = copy(item)
            replacement_item.token = object()
            controller._chat_start._active[target.id] = replacement_item
            captured_owner._automatic_primary_claims[target.id] = replacement_item.token
        release.set()
        assert await asyncio.gather(item.task, return_exceptions=True) == [None]
        assert exited.is_set() and item.outcome.done()
        if worker == "commit":
            outcome = await start
            assert (outcome.launch_status, outcome.reason) == (
                "review_required",
                "settlement_unconfirmed",
            )
        if replacement:
            assert controller._chat_start._active[target.id] is replacement_item
            assert (
                captured_owner._automatic_primary_claims[target.id]
                is replacement_item.token
            )
        else:
            assert target.id not in controller._chat_start._active
            assert target.id not in captured_owner._automatic_primary_claims
        if failure == "persistent_registration":
            assert len(registrations) == 2
        assert (
            ledger.read_chat_start_attempt(request.attempt_id, owner_id=owner).state
            == "accepted"
        )
        assert ledger.snapshot(chain).used["generation"] == 1
        assert ledger.snapshot(chain).status == "active"
        # Only transient same-key denial can protect this healthy alias handle.
        with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
            peer_ledger.check_active(chain, owner_id=owner)
        peer_ledger.check_active(unrelated, owner_id=owner)
    finally:
        release.set()
        monkeypatch.setattr(Path, "resolve", resolve)
        start.cancel()
        await asyncio.gather(start, return_exceptions=True)
        await asyncio.gather(*controller._chat_start.tasks(), return_exceptions=True)
        controller._chat_start._active.pop(target.id, None)
        controller.fleet_wake._automatic_primary_claims.pop(target.id, None)
        ledger._clear_chat_start_restriction(request.attempt_id, owner_id=owner)
        peer.close()
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()


@pytest.mark.asyncio
async def test_initial_identity_failure_precedes_native_start_authority(
    tmp_path, monkeypatch
):
    """An unresolved canonical key refuses start before a slot or attempt exists."""
    from pathlib import Path
    import sys

    controller, store, runs, source, target, chain, request = await _native_start_rig(
        tmp_path
    )
    # The real start boundary must construct a fresh ledger, as on first access.
    ledger = runs.__dict__.pop("automatic_work")
    resolve = Path.resolve

    def fail_resolution(path, *args, **kwargs):
        if (
            path == runs.db_path
            and sys._getframe(1).f_globals.get("__name__")
            == "tldw_chatbook.DB.automatic_work"
        ):
            raise OSError("private initial identity failure")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_resolution)
    try:
        with pytest.raises(OSError):
            await controller._chat_start.start(request)
        assert not controller._chat_start._active
        assert not controller._chat_start.tasks()
        assert not controller.fleet_wake._automatic_primary_claims
        with runs.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_chat_start_attempts"
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_work_reservations"
                ).fetchone()[0]
                == 0
            )
        assert target.draft == "original"
        assert not store.persistence.db.get_messages_for_conversation(
            request.conversation_id
        )
    finally:
        monkeypatch.setattr(Path, "resolve", resolve)
        runs.__dict__["automatic_work"] = ledger
        await controller.shutdown()
        runs.close()
        store.persistence.db.close_connection()
