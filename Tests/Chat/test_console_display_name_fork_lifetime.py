"""Display-name changes retain fork ownership until detached work settles."""

import asyncio
import json
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.settings_durability import (
    ConsoleSettingsDurabilityController,
)


@pytest.fixture
def name_source(tmp_path, request):
    db = CharactersRAGDB(tmp_path / "name.sqlite", client_id="name-fork-lifetime")

    def close_database():
        with db.quiesce_connections(timeout_seconds=2):
            pass
        db.close_connection()
        assert db.registered_connection_count() == 0

    request.addfinalizer(close_database)
    character_id = db.add_character_card({"name": "Nova"})
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="openai", model="test-model", system_prompt="Help User."
        ),
        assistant_kind="character",
        assistant_id=str(character_id),
        assistant_authority_id=db.get_local_authority_id(),
        character_id=character_id,
        character_name="Nova",
    )
    session.character_system_template = "Help {{user}}."
    session.library_policy_hydrated = True
    message = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="A question", persist=True
    )
    submission = ConsoleSettingsSubmission(
        submission_id="name-change",
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        surface=ConsoleSettingsSurface.FULL_SETTINGS,
        origin=store.capture_console_settings_origin(session.id),
        draft=ConsoleSettingsDraftState(
            settings=session.settings,
            context_policy_overrides=ConsoleContextPolicyOverrides(),
            field_drafts=(),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override="Rowan",
        default_field_mask=frozenset(),
    )
    commit = store.commit_console_settings_live(submission)
    assert store.fork_eligibility(message.id).eligible
    yield db, store, session, message, submission, commit


def _coordinator(store, recovery=lambda: None):
    notifications = []
    app = SimpleNamespace(notify=lambda text, **kwargs: notifications.append(text))

    def unused(*args, **kwargs):
        raise AssertionError("unrelated controller dependency entered")

    return ConsoleSettingsDurabilityController(
        app_instance_accessor=lambda: app,
        _ensure_console_chat_controller=unused,
        _ensure_console_chat_store=lambda: store,
        _global_chat_display_name=lambda: "User",
        _provider_readiness_app_config=unused,
        _sync_console_identity_surfaces=lambda: None,
        _sync_console_settings_recovery_surfaces=recovery,
        _sync_native_console_chat_ui=unused,
        run_worker=unused,
    )


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("outcome", ("success", "stale", "error", "prestart_cancel"))
async def test_display_name_coordinator_releases_terminal_plans(
    name_source, monkeypatch, outcome
):
    _, store, session, message, submission, commit = name_source
    controller = _coordinator(store)
    serialize = store.persist_roleplay_projection_plan_serialized
    entered = []

    async def persist(plan):
        entered.append(plan)
        if outcome == "stale":
            session.identity_revision += 1
        if outcome == "error":
            raise RuntimeError("persistence entry failed")
        return await serialize(plan)

    monkeypatch.setattr(store, "persist_roleplay_projection_plan_serialized", persist)
    loop = asyncio.get_running_loop()
    create_task = loop.create_task
    cancelled = []

    def cancel_before_start(coro, *args, **kwargs):
        task = create_task(coro, *args, **kwargs)
        if coro.cr_code.co_name == "persist_display_name":
            cancelled.append(task)
            task.cancel()
        return task

    if outcome == "prestart_cancel":
        monkeypatch.setattr(loop, "create_task", cancel_before_start)
    try:
        operation = controller._coordinate_console_settings_submission(
            ConsoleSettingsCommittedSubmission(submission, commit), None
        )
        if outcome == "prestart_cancel":
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert len(cancelled) == 1
            assert entered == []
        else:
            await operation
            assert len(entered) == 1
        assert store._roleplay_fork_transition_leases == {}
        assert store._fork_source_transitions == {}
        assert store.fork_eligibility(message.id).eligible
    finally:
        await asyncio.get_running_loop().shutdown_default_executor()


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("outcome", ("cancel", "sibling_error"))
async def test_display_name_lease_outlives_a_blocked_writer(
    name_source, monkeypatch, outcome
):
    db, store, session, message, submission, commit = name_source
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    cancellation_delivered = asyncio.Event()
    release = Event()
    write = ConsoleChatStore.persist_roleplay_projection_plan

    def blocked_write(plan):
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(10), "test never released the real writer"
            return write(plan)
        finally:
            db.close_connection()

    monkeypatch.setattr(
        ConsoleChatStore,
        "persist_roleplay_projection_plan",
        staticmethod(blocked_write),
    )
    create_task = loop.create_task
    display_tasks = []

    def observe_task(coro, *args, **kwargs):
        task = create_task(coro, *args, **kwargs)
        if coro.cr_code.co_name == "persist_display_name":
            display_tasks.append(task)
        return task

    monkeypatch.setattr(loop, "create_task", observe_task)
    shield = asyncio.shield

    def observe_shield(task):
        future = shield(task)
        if (
            isinstance(task, asyncio.Task)
            and task.get_coro().cr_code.co_name == "to_thread"
        ):

            def cancelled(done):
                if done.cancelled():
                    cancellation_delivered.set()

            future.add_done_callback(cancelled)
        return future

    monkeypatch.setattr(asyncio, "shield", observe_shield)
    persist_settings = store.persist_console_settings_commit_serialized

    async def settings_after_writer(*args, **kwargs):
        await started.wait()
        return await persist_settings(*args, **kwargs)

    monkeypatch.setattr(
        store, "persist_console_settings_commit_serialized", settings_after_writer
    )

    def recovery():
        if outcome == "sibling_error":
            raise RuntimeError("sibling publication failed")

    controller = _coordinator(store, recovery)
    coordination = asyncio.create_task(
        controller._coordinate_console_settings_submission(
            ConsoleSettingsCommittedSubmission(submission, commit), None
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert len(display_tasks) == 1
        if outcome == "cancel":
            coordination.cancel()
            await asyncio.wait_for(cancellation_delivered.wait(), 5)
        else:
            with pytest.raises(RuntimeError, match="sibling publication failed"):
                await coordination
        assert not display_tasks[0].done()
        assert not store.fork_eligibility(message.id).eligible
        with pytest.raises(ValueError, match="changing"):
            store.issue_fork_fence(message.id)
        release.set()
        await asyncio.gather(coordination, *display_tasks, return_exceptions=True)
        assert store._roleplay_fork_transition_leases == {}
        assert store._fork_source_transitions == {}
        assert store.fork_eligibility(message.id).eligible
        row = db.get_conversation_by_id(session.persisted_conversation_id)
        assert row["system_prompt"] == "Help Rowan."
    finally:
        release.set()
        if not coordination.done():
            coordination.cancel()
        await asyncio.gather(coordination, *display_tasks, return_exceptions=True)
        await loop.shutdown_default_executor()


def test_display_name_blocks_fork_through_real_durable_acceptance(
    name_source, monkeypatch
):
    db, store, session, message, _, commit = name_source
    other = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="other-model"),
        ephemeral=True,
    )
    other_message = store.append_message(
        other.id, role=ConsoleMessageRole.USER, content="Independent", persist=False
    )
    materialize = store._materialize_roleplay_projections_live
    observed = []

    def observe(*args, **kwargs):
        observed.append(store.fork_eligibility(message.id).eligible)
        assert store.fork_eligibility(other_message.id).eligible
        return materialize(*args, **kwargs)

    monkeypatch.setattr(store, "_materialize_roleplay_projections_live", observe)
    _, plan = store.prepare_session_user_display_name_override_for_commit(
        commit, "Rowan", global_default="User"
    )
    assert observed == [False]
    assert plan is not None
    assert session.settings.system_prompt == "Help Rowan."
    assert not store.fork_eligibility(message.id).eligible
    assert store.fork_eligibility(other_message.id).eligible
    with pytest.raises(ValueError, match="changing"):
        store.issue_fork_fence(message.id)
    result = store.persist_roleplay_projection_plan(plan)
    assert result.persisted
    assert not store.fork_eligibility(message.id).eligible
    assert store.accept_roleplay_projection_persistence_result(result)
    assert store.fork_eligibility(message.id).eligible
    assert store._roleplay_fork_transition_leases == {}
    assert store._fork_source_transitions == {}
    db.close_connection()
    row = db.get_conversation_by_id(session.persisted_conversation_id)
    assert row["system_prompt"] == "Help Rowan."
    assert (
        json.loads(row["metadata"])["console_roleplay_context"]["user_name_override"]
        == "Rowan"
    )


@pytest.mark.parametrize("case", ("noop", "stale", "exception"))
def test_display_name_preparation_without_a_plan_leaves_no_fork_lease(
    name_source, monkeypatch, case
):
    _, store, session, message, _, commit = name_source
    if case == "stale":
        commit = replace(commit, conversation_binding_revision=-1)
    if case == "exception":

        def fail(*args, **kwargs):
            raise RuntimeError("materialization failed")

        monkeypatch.setattr(store, "_materialize_roleplay_projections_live", fail)
        with pytest.raises(RuntimeError, match="materialization failed"):
            store.prepare_session_user_display_name_override_for_commit(
                commit, "Rowan", global_default="User"
            )
    else:
        _, plan = store.prepare_session_user_display_name_override_for_commit(
            commit, None if case == "noop" else "Rowan", global_default="User"
        )
        assert plan is None
        assert session.user_display_name_override is None
    assert store._roleplay_fork_transition_leases == {}
    assert store._fork_source_transitions == {}
    assert store.fork_eligibility(message.id).eligible
