from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
def append(name,text):
 p=root/name;p.write_text(p.read_text()+text)
append('Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py','''\n\ndef test_native_replay_rejects_added_hook_receipt_before_dedupe(tmp_path):
    from dataclasses import replace
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt
    from tldw_chatbook.Chat.message_metadata import AgentChatStartMetadata

    db, conversation = _db_and_conversation(tmp_path / "native-replay.sqlite")
    try:
        native = replace(
            _acceptance(conversation), origin="agent_chat_start",
            agent_chat_start_attempt_id="native-start",
            agent_chat_start=AgentChatStartMetadata("native-start", "source-run", "source-chat"),
            handoff_draft_revision=1,
        )
        repository = ConsoleDispatchRepository(db)
        checkpoint = _insert(db, repository, native)
        assert _insert(db, repository, native) == checkpoint
        messages = db.get_messages_for_conversation(conversation)
        mixed = replace(native, continuation_receipt=ContinuationReceipt(
            "parent", "stop", "assistant", "scheduler", 1))
        with pytest.raises(ConsoleDispatchCheckpointValidationError):
            _insert(db, repository, mixed)
        assert db.get_messages_for_conversation(conversation) == messages
        assert len(messages) == 2
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM console_dispatch_checkpoints").fetchone()[0] == 1
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM console_hook_continuation_receipts").fetchone()[0] == 0
        assert _insert(db, repository, native) == checkpoint
    finally:
        db.close()
''')
append('Tests/DB/test_chachanotes_v76_agent_chat_starts_migration.py','''\n\ndef test_mixed_hook_replay_keeps_native_receipt_and_messages_exact(tmp_path):
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt

    db, conversation = _db_and_conversation(tmp_path / "mixed-replay.sqlite")
    try:
        acceptance = replace(
            _acceptance(conversation), origin="agent_chat_start",
            agent_chat_start_attempt_id="start",
            agent_chat_start=message_metadata.AgentChatStartMetadata("start", "run", "source"),
            handoff_draft_revision=1,
        )
        repository = ConsoleDispatchRepository(db)
        checkpoint = _insert(db, repository, acceptance)
        assert _insert(db, repository, acceptance) == checkpoint
        before = dict(db.get_connection().execute(
            "SELECT * FROM console_dispatch_checkpoints").fetchone())
        messages = db.get_messages_for_conversation(conversation)
        with pytest.raises(ValueError):
            _insert(db, repository, replace(acceptance, continuation_receipt=ContinuationReceipt(
                "parent", "stop", "assistant", "scheduler", 1)))
        assert dict(db.get_connection().execute(
            "SELECT * FROM console_dispatch_checkpoints").fetchone()) == before
        assert db.get_messages_for_conversation(conversation) == messages
        assert len(messages) == 2
        assert db.get_connection().execute(
            "SELECT COUNT(*) FROM console_hook_continuation_receipts").fetchone()[0] == 0
    finally:
        db.close()
''')
append('Tests/Chat/test_console_chat_create_integration.py','''\n\n@pytest.fixture
def child_new_chat_rig(real_db_controller, tmp_path):
    """A trusted child actor backed by genuine native run and conversation rows."""
    from threading import Event
    from types import SimpleNamespace
    from tldw_chatbook.Agents.run_context import CurrentRunActor
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    controller, db = real_db_controller
    source = controller.store.create_session(title="Parent chat")
    source.persisted_conversation_id = controller.store.persistence.create_conversation(
        conversation_title="Parent chat")
    runs = AgentRunsDB(tmp_path / "child-runs.sqlite")
    parent = runs.create_run(conversation_id=source.persisted_conversation_id,
        agent_kind="primary", assistant_message_id="parent-message")
    child = runs.create_run(conversation_id=source.persisted_conversation_id,
        agent_kind="subagent", parent_run_id=parent, task="Follow-up work")
    controller._agent_bridge = SimpleNamespace(runs_db=runs, agent_runs_db=runs,
        live_primary_run_id=lambda conversation: parent)
    controller._active_cancel_events[source.id] = Event()
    controller._active_assistant_message_ids[source.id] = "parent-message"
    controller.app.app_config = _FAKE_APP_CONFIG
    runtime = SimpleNamespace(_app=controller.app)
    runtime._resolve_new_console_assistant = lambda workspace, settings: (
        ConsoleRuntime._resolve_new_console_assistant(runtime, workspace, settings))
    controller.app.console_runtime = runtime
    controller._default_session_settings = lambda: ConsoleSessionSettings(
        provider="llama_cpp", model="destination-model")
    actor = CurrentRunActor("subagent", child, parent)
    payload = {"tool": "new_chat", "session_id": source.id,
        "source_run_id": child, "source_message_id": "parent-message",
        "title": "Child draft", "opening_prompt": "literal /new @helper", "instructions": ""}
    try:
        yield controller, db, runs, source, actor, payload
    finally:
        runs.close()


def test_child_new_chat_prepares_confirms_executes_and_requires_fresh_approval(child_new_chat_rig):
    from tldw_chatbook.Agents.run_context import use_run_actor

    controller, db, runs, source, actor, payload = child_new_chat_rig
    cards = []
    def approve(card):
        if card:
            cards.append(dict(card))
            controller.resolve_pending_chat_create(True, True, card["request_id"])
    controller.set_pending_chat_create = approve
    controller._start_created_chat = lambda *args: pytest.fail("child draft acquired start authority")
    _, new_chat = build_chat_create_tool_closures(
        confirm=lambda prepared: controller.request_chat_create_confirm(prepared, session_id=source.id),
        execute=controller.execute_agent_chat_create,
        prepare=controller.prepare_agent_chat_create,
        session_id=source.id, run_id="parent-message")
    source.draft = "parent composer custody"
    with use_run_actor(actor):
        first = new_chat({"title": "Child draft", "opening_prompt": "literal /new @helper"})
        assert first.ok, first.error
        second = new_chat({"title": "Child draft 2", "opening_prompt": "second draft"})
        assert second.ok, second.error
    assert len(cards) == 2
    assert cards[0]["request_id"] != cards[1]["request_id"]
    for card in cards:
        assert card["run_id"] == actor.run_id
        assert card["agent_kind"] == "subagent" and card["parent_run_id"] == actor.parent_run_id
        assert card["destination"] == "same_workspace" and card["mode"] == "draft"
    for result, prompt in ((first, "literal /new @helper"), (second, "second draft")):
        outcome = json.loads(result.content)
        assert outcome["launch_status"] == "draft"
        assert outcome["draft_set"] and outcome["copied_messages"] == 0
        row = db.get_conversation_by_id(outcome["conversation_id"])
        metadata = json.loads(row["metadata"])
        assert metadata["console_agent_handoff"]["draft"] == prompt
        assert metadata["console_agent_handoff"]["source_run_id"] == actor.run_id
        assert not db.get_messages_for_conversation(outcome["conversation_id"])
    assert controller.store.active_session_id == source.id
    assert source.draft == "parent composer custody"
    assert not controller._chat_create_session_grants.get(source.id)
    assert not controller._chat_creation_records and not controller.pending_chat_create_ids()
    with runs.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM automatic_chat_start_attempts").fetchone()[0] == 0


@pytest.mark.parametrize("extra", [{"destination": "casual"}, {"mode": "start"}])
def test_child_new_chat_refuses_new_destination_and_start_authority(child_new_chat_rig, extra):
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    with use_run_actor(actor), pytest.raises(PermissionError):
        controller.prepare_agent_chat_create({**payload, **extra})
    assert not controller._chat_creation_records


@pytest.mark.parametrize("mutation", ["terminal_child", "wrong_actor", "source_incarnation"])
def test_child_prepared_create_rechecks_currentness_before_execution(child_new_chat_rig, mutation):
    from dataclasses import replace
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    controller.set_pending_chat_create = lambda card: (
        controller.resolve_pending_chat_create(True, False, card["request_id"]) if card else None)
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(payload)
        assert controller.request_chat_create_confirm(prepared)["allow"]
        if mutation == "terminal_child":
            with runs.transaction() as connection:
                connection.execute("UPDATE agent_runs SET status='cancelled' WHERE id=?", (actor.run_id,))
        elif mutation == "source_incarnation":
            source.incarnation_id = "replacement-incarnation"
        acting = replace(actor, parent_run_id="other-parent") if mutation == "wrong_actor" else actor
        with use_run_actor(acting):
            outcome = controller.execute_agent_chat_create(prepared)
    assert not outcome["ok"] and outcome["kind"] == "approval_required"
    assert not controller._chat_creation_records
''')
append('Tests/Chat/test_console_chat_create_confirm.py','''\n\n# Genuine child preparation uses the same native fixture as execution controls.
from Tests.Chat.test_console_chat_create_integration import child_new_chat_rig, real_db_controller


def test_child_new_chat_cancelled_during_confirmation_closes_authority(child_new_chat_rig):
    from tldw_chatbook.Agents.run_context import use_run_actor
    controller, db, runs, source, actor, payload = child_new_chat_rig
    cards = []
    def revoke(card):
        if card:
            cards.append(dict(card))
            assert controller.revoke_approval_rounds_for_run(actor.run_id) == 1
            controller.resolve_pending_chat_create(True, True, card["request_id"])
    controller.set_pending_chat_create = revoke
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(payload)
        decision = controller.request_chat_create_confirm(prepared, session_id=source.id)
    assert decision == {"allow": False, "remember": False}
    assert cards[0]["run_id"] == actor.run_id
    assert not controller._chat_creation_records and not controller.pending_chat_create_ids()
    assert not controller._chat_create_session_grants.get(source.id)
''')
