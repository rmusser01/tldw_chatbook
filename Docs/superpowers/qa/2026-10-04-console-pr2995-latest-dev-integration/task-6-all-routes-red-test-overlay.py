"""Public mutation routes hold the existing fork owner through live publication."""

from dataclasses import replace
from threading import Event, Thread

import pytest

from Tests.Chat.test_console_chat_fork import _fork_store
from Tests.Chat.test_console_settings_apply_store import _submission
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


def _other_selection(store, settings):
    other = store.create_session(title="Independent", settings=settings)
    store.append_message(other.id, role=ConsoleMessageRole.USER, content="Question")
    return store.append_message(other.id, role=ConsoleMessageRole.ASSISTANT, content="Answer")


@pytest.mark.parametrize("route,field", (
    ("publish", "persisted_conversation_id"),
    ("rebind", "persisted_conversation_id"),
    ("settings", "context_policy_overrides"),
    ("display_name", "user_display_name_override"),
    ("persona", "persona_system_template"),
    ("assistant", "assistant_name"),
))
@pytest.mark.parametrize("fail", (False, True))
def test_public_mutation_blocks_eligible_fork_at_actual_assignment(monkeypatch, route, field, fail):
    store, persistence, session, _, _, _, selected, _ = _fork_store(durable=route in {"publish", "rebind"})
    if route == "publish":
        session.persisted_conversation_id = None
    if route == "rebind":
        persistence.active_leaf_by_conversation["new-id"] = persistence.active_leaf_by_conversation["conversation-1"]
    other = _other_selection(store, session.settings)
    submission = _submission(store, session.id, submission_id="settings", model="new-model")
    commit = store.commit_console_settings_live(submission) if route == "display_name" else None
    assert store.fork_eligibility(selected.id).eligible
    assert store.fork_eligibility(other.id).eligible
    entered, release = Event(), Event()
    errors, results = [], []
    original = type(session).__setattr__

    def publish_then_pause(target, name, value):
        original(target, name, value)
        if target is session and name == field:
            entered.set()
            assert release.wait(5), "mutation was not released"
            if fail:
                raise RuntimeError("controlled publication failure")

    monkeypatch.setattr(type(session), "__setattr__", publish_then_pause)

    def mutate():
        try:
            if route == "publish":
                result = store.publish_first_persisted_conversation(session.id, "conversation-1")
            elif route == "rebind":
                result = store.rebind_persisted_conversation(session.id, "new-id")
            elif route == "settings":
                result = store.commit_console_settings_live(submission)
            elif route == "display_name":
                result = store.prepare_session_user_display_name_override_for_commit(commit, "Taylor", global_default="Riley")
            elif route == "persona":
                result = store.seed_persona_roleplay(session.id, system_template="Help {{user}}", global_default="Riley")
            else:
                result = store.set_session_assistant_name(session.id, "Nova", global_default="Riley")
            results.append(result)
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=mutate)
    worker.start()
    try:
        assert entered.wait(5), errors
        assert store.fork_eligibility(selected.id).reason == "Console fork source is changing; retry after the update finishes."
        assert store.fork_eligibility(other.id).eligible is True
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    if fail:
        assert len(errors) == 1 and str(errors[0]) == "controlled publication failure"
    else:
        assert errors == []
        if route == "display_name":
            plan = results[0][1]
            assert plan.fork_transition_token is not None
            assert not store.fork_eligibility(selected.id).eligible
            assert store.abandon_roleplay_projection_plan(plan)
    assert store._fork_source_transitions == {}
    assert store._roleplay_fork_transition_leases == {}
    assert store.fork_eligibility(other.id).eligible


@pytest.mark.parametrize("outcome", ("accept", "abandon", "noop", "stale", "materialize_none", "materialize_error"))
def test_display_name_plan_exact_lease_lifetime(monkeypatch, outcome):
    store, _, session, _, _, _, selected, _ = _fork_store()
    commit = store.commit_console_settings_live(_submission(store, session.id, submission_id="name", model="new"))
    if outcome == "stale":
        commit = replace(commit, conversation_binding_revision=-1)
    if outcome.startswith("materialize_"):
        def materialize(*args, **kwargs):
            if outcome == "materialize_error":
                raise RuntimeError("projection failed")
            return None
        monkeypatch.setattr(store, "_materialize_roleplay_projections_live", materialize)
    if outcome == "materialize_error":
        with pytest.raises(RuntimeError, match="projection failed"):
            store.prepare_session_user_display_name_override_for_commit(commit, "Taylor", global_default="Riley")
    else:
        _, plan = store.prepare_session_user_display_name_override_for_commit(commit, "Riley" if outcome == "noop" else "Taylor", global_default="Riley")
        if outcome in {"noop", "stale"}:
            assert plan is None
        else:
            assert plan is not None and plan.fork_transition_token is not None
            assert not store.fork_eligibility(selected.id).eligible
            if outcome == "accept":
                result = store.persist_roleplay_projection_plan(plan)
                assert not store.fork_eligibility(selected.id).eligible
                assert store.accept_roleplay_projection_persistence_result(result)
                assert not store.abandon_roleplay_projection_plan(plan)
            else:
                with store.fork_source_transition(session.id):
                    assert store.abandon_roleplay_projection_plan(plan)
                    assert store._fork_source_transitions[session.id] == 1
                assert not store.abandon_roleplay_projection_plan(plan)
    assert store._fork_source_transitions == {}
    assert store._roleplay_fork_transition_leases == {}
    assert store.fork_eligibility(selected.id).eligible


@pytest.mark.parametrize("initial", ("default", "explicit", "invalid"))
def test_restore_initial_project_controls_are_constructor_owned(monkeypatch, initial):
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_project_instructions import ProjectInstructionControlState, encode_project_context_json

    durable = ProjectInstructionControlState.legacy_disabled()
    fresh = ProjectInstructionControlState.new_session()
    store = ConsoleChatStore(persistence=SimpleNamespace(
        get_conversation_console_project_context=lambda **kwargs: encode_project_context_json(durable),
    ))
    seen = []
    original = store.create_session
    def create(**kwargs):
        seen.append(kwargs["project_instruction_state"])
        return original(**kwargs)
    monkeypatch.setattr(store, "create_session", create)
    kwargs = dict(title="Restored", workspace_id=None, persisted_conversation_id="conversation", all_nodes=[], activate=False)
    if initial != "default":
        kwargs["initial_project_instruction_state"] = fresh if initial == "explicit" else "untrusted"
    if initial == "invalid":
        with pytest.raises(TypeError, match="initial_project_instruction_state"):
            store.restore_persisted_session(**kwargs)
        assert seen == []
        assert store._sessions == {}
    else:
        session = store.restore_persisted_session(**kwargs)
        expected = fresh if initial == "explicit" else durable
        assert seen == [expected]
        assert session.project_instruction_state == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("accept", "none", "error", "cancel"))
async def test_settings_caller_releases_name_lease_after_physical_drain(monkeypatch, outcome):
    import asyncio
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_settings_apply import ConsoleSettingsCommittedSubmission
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    store, _, session, _, _, _, selected, _ = _fork_store(durable=True)
    submission = replace(_submission(store, session.id, submission_id="caller", model="new"), user_display_name_override="Taylor")
    committed = ConsoleSettingsCommittedSubmission(submission, store.commit_console_settings_live(submission))
    entered, release, finished = Event(), Event(), Event()
    original = ConsoleChatStore.persist_roleplay_projection_plan
    def persist(plan):
        entered.set()
        try:
            assert release.wait(5)
            if outcome == "error":
                raise RuntimeError("controlled persistence failure")
            return original(plan)
        finally:
            finished.set()
    monkeypatch.setattr(ConsoleChatStore, "persist_roleplay_projection_plan", staticmethod(persist))
    async def no_conversation(*args, **kwargs):
        return None
    monkeypatch.setattr(store, "persist_console_settings_commit_serialized", no_conversation)
    if outcome == "none":
        monkeypatch.setattr(store, "is_roleplay_projection_plan_current", lambda plan: False)
    screen = SimpleNamespace(
        _ensure_console_chat_store=lambda: store,
        _global_chat_display_name=lambda: "Riley",
        _sync_console_settings_recovery_surfaces=lambda: None,
        _sync_console_identity_surfaces=lambda: None,
        app_instance=SimpleNamespace(notify=lambda *args, **kwargs: None),
    )
    task = asyncio.create_task(ChatScreen._coordinate_console_settings_submission(screen, committed, None))
    if outcome != "none":
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert not store.fork_eligibility(selected.id).eligible
            if outcome == "cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not finished.is_set()
                assert store._roleplay_fork_transition_leases
                assert not task.done()
        finally:
            release.set()
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    if outcome != "none":
        assert finished.is_set()
    assert store._roleplay_fork_transition_leases == {}
    assert store._fork_source_transitions == {}
    assert store._roleplay_persistence_locks == {}
    assert store.fork_eligibility(selected.id).eligible
