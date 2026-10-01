"""Deterministic fork rejection during exact Store publication boundaries."""

from dataclasses import replace
from threading import Event, Thread

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsDraftState,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)


def _stable_source():
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="fixture-model"),
        ephemeral=True,
        assistant_kind="persona",
        assistant_id="fixture-persona",
        assistant_name="Fixture Assistant",
    )
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Question")
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Answer"
    )
    assert store.fork_eligibility(answer.id).eligible
    return store, session, answer


def _submission(store, session):
    return ConsoleSettingsSubmission(
        submission_id="fixture-submission",
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        surface=ConsoleSettingsSurface.FULL_SETTINGS,
        origin=store.capture_console_settings_origin(session.id),
        draft=ConsoleSettingsDraftState(
            settings=replace(session.settings, temperature=0.25),
            context_policy_overrides=ConsoleContextPolicyOverrides(),
            field_drafts=(),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override="New Human",
        default_field_mask=frozenset(),
    )


@pytest.mark.parametrize(
    ("route", "pause_owner"),
    (
        ("commit_console_settings_live", "_replace_session_context_policy_live"),
        (
            "prepare_session_user_display_name_override_for_commit",
            "_snapshot_roleplay_context_write",
        ),
        (
            "publish_first_persisted_conversation",
            "_record_console_settings_binding_revision",
        ),
        ("rebind_persisted_conversation", "_advance_console_settings_binding_revision"),
        ("seed_persona_roleplay", "_persist_roleplay_context"),
        ("set_session_assistant_name", "_materialize_roleplay_projections"),
    ),
)
def test_live_store_publication_rejects_a_fork_until_complete(
    monkeypatch, route, pause_owner
):
    store, session, answer = _stable_source()
    other = store.create_session(settings=session.settings, ephemeral=True)
    other_answer = store.append_message(
        other.id, role=ConsoleMessageRole.ASSISTANT, content="Other answer"
    )
    submission = _submission(store, session)
    commit = store.commit_console_settings_live(submission)
    entered, release = Event(), Event()
    outcomes, failures = [], []
    original = getattr(store, pause_owner)

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5), "reader did not release the controlled writer"
        return original(*args, **kwargs)

    monkeypatch.setattr(store, pause_owner, paused)

    def mutate():
        try:
            if route == "commit_console_settings_live":
                value = store.commit_console_settings_live(
                    replace(submission, submission_id="next-submission")
                )
            elif route == "prepare_session_user_display_name_override_for_commit":
                value = store.prepare_session_user_display_name_override_for_commit(
                    commit, "New Human", global_default="Human"
                )
            elif route == "publish_first_persisted_conversation":
                value = store.publish_first_persisted_conversation(
                    session.id, "durable"
                )
            elif route == "rebind_persisted_conversation":
                value = store.rebind_persisted_conversation(session.id, "rebound")
            elif route == "seed_persona_roleplay":
                value = store.seed_persona_roleplay(
                    session.id, system_template="Help {{user}}.", global_default="Human"
                )
            else:
                value = store.set_session_assistant_name(
                    session.id, "New Assistant", global_default="Human"
                )
            outcomes.append(value)
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(exc)

    writer = Thread(target=mutate)
    writer.start()
    try:
        assert entered.wait(5), "writer did not reach the selected live mutation"
        eligibility = store.fork_eligibility(answer.id)
        assert not eligibility.eligible, route
        assert "source is changing" in eligibility.reason
        with pytest.raises(ValueError, match="source is changing"):
            store.issue_fork_fence(answer.id)
        assert store.fork_eligibility(other_answer.id).eligible
    finally:
        release.set()
        writer.join(5)
    assert not writer.is_alive()
    assert not failures
    assert outcomes
    if route == "prepare_session_user_display_name_override_for_commit":
        _, plan = outcomes[0]
        if plan is not None:
            store.abandon_roleplay_projection_plan(plan)
    assert not store._fork_source_transitions
    assert store.fork_eligibility(answer.id).eligible


@pytest.mark.parametrize("terminal", ("accept", "abandon"))
@pytest.mark.parametrize("named", (True, False))
def test_committed_name_plan_retains_exact_fork_custody_until_terminal(terminal, named):
    from types import SimpleNamespace

    store, session, answer = _stable_source()
    if not named:
        session.assistant_name = None
    session.persona_system_template = "Help {{user}}."
    session.persisted_conversation_id = "fixture-conversation"
    store.persistence = SimpleNamespace(
        update_conversation_roleplay_context=lambda **_kwargs: True,
        update_conversation_system_prompt=lambda **_kwargs: True,
    )
    commit = store.commit_console_settings_live(_submission(store, session))
    _, plan = store.prepare_session_user_display_name_override_for_commit(
        commit, "New Human", global_default="Human"
    )
    assert plan is not None
    try:
        assert plan.fork_transition_token is not None
        assert store._roleplay_fork_transition_leases == {
            plan.fork_transition_token: session.id
        }
        assert "source is changing" in store.fork_eligibility(answer.id).reason
        if terminal == "accept":
            result = store.persist_roleplay_projection_plan(plan)
            assert store.accept_roleplay_projection_persistence_result(result)
        else:
            assert store.abandon_roleplay_projection_plan(plan)
        assert not store._fork_source_transitions
        assert not store._roleplay_fork_transition_leases
        assert store.fork_eligibility(answer.id).eligible
    finally:
        store.abandon_roleplay_projection_plan(plan)


def test_voice_pair_cannot_be_forked_before_publication_rollback(monkeypatch):
    from tldw_chatbook.Chat.console_voice_promotion import (
        VoicePromotionContext,
        derive_voice_promotion_identities,
    )

    store, session, answer = _stable_source()
    origin, native_leaf, persisted_leaf = store.snapshot_voice_promotion_origin(
        session.id
    )
    context = VoicePromotionContext(
        promotion_id="fixture-promotion",
        attempt_id="fixture-attempt",
        origin=origin,
        expected_native_leaf_id=native_leaf,
        expected_persisted_leaf_id=persisted_leaf,
        user_text="Voice question",
        assistant_text="Voice answer",
        usage_json=None,
        terminal_boundary_id="fixture-terminal",
        capture_eligible_at_dispatch=False,
    )
    claim = store.try_claim_voice_promotion(context)
    assert claim.lease is not None
    identities = derive_voice_promotion_identities(context.promotion_id)
    entered, release = Event(), Event()
    failures = []

    def fail_before_publication_finishes(_session_id):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("controlled publication rollback")

    monkeypatch.setattr(
        store, "_bump_payload_revision", fail_before_publication_finishes
    )

    def publish():
        try:
            store.publish_temporary_voice_pair(claim.lease, context)
        except RuntimeError as exc:
            failures.append(str(exc))

    writer = Thread(target=publish)
    writer.start()
    snapshot = None
    try:
        assert entered.wait(5)
        eligibility = store.fork_eligibility(identities.assistant_message_id)
        if eligibility.eligible:
            fence = store.issue_fork_fence(identities.assistant_message_id)
            snapshot = store.stage_fork_snapshot(
                fence,
                title="Controlled fork",
                fork_session_id="fixture-fork",
                fork_conversation_id=None,
            )
        else:
            assert "source is changing" in eligibility.reason
    finally:
        release.set()
        writer.join(5)
    assert not writer.is_alive()
    assert failures == ["controlled publication rollback"]
    assert identities.assistant_message_id not in store._nodes_by_session[session.id]
    assert store.active_leaf(session.id) == answer.id
    assert snapshot is None, "a fork copied the pair that publication rolled back"
    assert not store._fork_source_transitions


def _name_plan(*, durable=True):
    from types import SimpleNamespace

    store, session, answer = _stable_source()
    if durable:
        session.persisted_conversation_id = "fixture-conversation"
        store.persistence = SimpleNamespace(
            update_conversation_roleplay_context=lambda **_kwargs: True,
        )
    commit = store.commit_console_settings_live(_submission(store, session))
    _, plan = store.prepare_session_user_display_name_override_for_commit(
        commit, "New Human", global_default="Human"
    )
    assert plan is not None
    return store, session, answer, plan


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", (True, False))
async def test_serialized_stale_name_plan_releases_its_exact_token(durable):
    store, session, answer, plan = _name_plan(durable=durable)
    session.identity_revision += 1
    assert await store.persist_roleplay_projection_plan_serialized(plan) is None
    assert not store._fork_source_transitions
    assert not store._roleplay_fork_transition_leases
    assert not store._roleplay_persistence_locks
    assert store.fork_eligibility(answer.id).eligible


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", (True, False))
async def test_serialized_name_plan_error_releases_its_exact_token(
    monkeypatch, durable
):
    store, _, answer, plan = _name_plan(durable=durable)

    def failed(_plan):
        raise RuntimeError("controlled finite writer failure")

    monkeypatch.setattr(
        ConsoleChatStore, "persist_roleplay_projection_plan", staticmethod(failed)
    )
    with pytest.raises(RuntimeError, match="controlled finite writer failure"):
        await store.persist_roleplay_projection_plan_serialized(plan)
    assert not store._fork_source_transitions
    assert not store._roleplay_fork_transition_leases
    assert not store._roleplay_persistence_locks
    assert store.fork_eligibility(answer.id).eligible


@pytest.mark.asyncio
async def test_cancelled_serialized_name_plan_releases_only_after_worker_finishes(
    monkeypatch,
):
    import asyncio

    store, session, answer, plan = _name_plan()
    entered, release, finished = Event(), Event(), Event()
    original = ConsoleChatStore.persist_roleplay_projection_plan

    def finite_writer(frozen_plan):
        entered.set()
        try:
            assert release.wait(5)
            return original(frozen_plan)
        finally:
            finished.set()

    monkeypatch.setattr(
        ConsoleChatStore,
        "persist_roleplay_projection_plan",
        staticmethod(finite_writer),
    )
    task = asyncio.create_task(store.persist_roleplay_projection_plan_serialized(plan))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
        assert store._fork_source_transitions[session.id] == 1
        assert "source is changing" in store.fork_eligibility(answer.id).reason
        task.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    assert not store._fork_source_transitions
    assert not store._roleplay_fork_transition_leases
    assert not store._roleplay_persistence_locks
    assert store.fork_eligibility(answer.id).eligible


@pytest.mark.asyncio
async def test_serialized_success_keeps_name_plan_token_until_owner_acceptance():
    store, _, answer, plan = _name_plan()
    result = await store.persist_roleplay_projection_plan_serialized(plan)
    assert result is not None
    assert result.fork_transition_token == plan.fork_transition_token
    assert "source is changing" in store.fork_eligibility(answer.id).reason
    assert store.accept_roleplay_projection_persistence_result(result)
    assert not store._fork_source_transitions
    assert not store._roleplay_fork_transition_leases
    assert store.fork_eligibility(answer.id).eligible


@pytest.mark.parametrize("outcome", ("noop", "materialization_error"))
def test_committed_name_preparation_balances_only_its_own_transition(
    monkeypatch, outcome
):
    (
        store,
        session,
        _,
    ) = _stable_source()
    commit = store.commit_console_settings_live(_submission(store, session))
    with store.fork_source_transition(session.id):
        if outcome == "noop":
            _, plan = store.prepare_session_user_display_name_override_for_commit(
                commit, None, global_default="Human"
            )
            assert plan is None
        else:

            def failed_materialization(*_args, **_kwargs):
                raise RuntimeError("controlled live materialization failure")

            monkeypatch.setattr(
                store, "_materialize_roleplay_projections_live", failed_materialization
            )
            with pytest.raises(
                RuntimeError, match="controlled live materialization failure"
            ):
                store.prepare_session_user_display_name_override_for_commit(
                    commit, "New Human", global_default="Human"
                )
        assert store._fork_source_transitions == {session.id: 1}
        assert not store._roleplay_fork_transition_leases
    assert not store._fork_source_transitions


def test_attachment_extraction_preserves_the_store_class_patch_seam(monkeypatch):
    from tldw_chatbook.Chat.console_chat_fork import (
        fingerprint_console_fork_attachments,
    )

    assert (
        ConsoleChatStore._fork_attachment_fingerprint
        is fingerprint_console_fork_attachments
    )
    store, _, answer = _stable_source()
    calls = []

    def patched(attachments, generation):
        calls.append((attachments, generation))
        return fingerprint_console_fork_attachments(attachments, generation)

    monkeypatch.setattr(
        ConsoleChatStore, "_fork_attachment_fingerprint", staticmethod(patched)
    )
    store.issue_fork_fence(answer.id)
    assert calls
