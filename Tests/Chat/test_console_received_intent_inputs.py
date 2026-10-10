"""Exact authored input revisions for a received Console turn."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from Tests.Chat import test_console_chat_start as handoff_controls
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

pytestmark = pytest.mark.bootstrap_profile
handoff_store = handoff_controls.handoff_store


@pytest.fixture
def input_store():
    store = ConsoleChatStore()
    session = store.create_session(session_id="intent-inputs", ephemeral=True)
    try:
        yield store, session
    finally:
        for live in store.sessions():
            store.close_session(live.id)
        store.end_app_runtime()


def _attachment(identifier):
    return PendingAttachment(
        file_path=f"/{identifier}",
        display_name=identifier,
        file_type="image",
        insert_mode="attachment",
        attachment_id=identifier,
        data=b"private attachment bytes",
    )


def test_snapshot_retains_exact_inputs_without_exposing_bodies(input_store):
    store, session = input_store
    attachment = _attachment("original")
    store.set_session_draft(session.id, "private authored draft")
    store.add_pending_attachment(session.id, attachment)
    store.set_session_one_shot_prefill(session.id, "private prefill")

    snapshot = store.session_input_snapshot(session.id)

    assert snapshot.session_id == session.id
    assert snapshot.incarnation_id == session.incarnation_id
    assert snapshot.draft == "private authored draft"
    assert snapshot.attachments == (attachment,)
    assert snapshot.attachments[0] is attachment
    assert snapshot.one_shot_prefill == "private prefill"
    assert snapshot.prefill_revision == session.one_shot_prefill_revision
    assert store.session_inputs_are_current(snapshot) is True
    assert "private" not in repr(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.draft = "replacement"


def test_authored_identity_changes_invalidate_identical_text_before_queued_event(
    input_store,
):
    store, session = input_store
    store.set_session_draft(session.id, "same draft", authored_token=(7, 1))
    captured = store.session_input_snapshot(session.id)

    # A clear/retype can return to identical text before DraftChanged is delivered.
    store.set_session_draft(session.id, "same draft", authored_token=(8, 3))

    assert store.session_draft(session.id) == captured.draft
    assert (
        store.session_input_snapshot(session.id).draft_revision
        > captured.draft_revision
    )
    assert store.session_inputs_are_current(captured) is False
    assert store.commit_session_input_draft(captured) is False
    assert store.session_draft(session.id) == "same draft"


def test_repeated_token_and_legacy_poll_do_not_invalidate_snapshot(input_store):
    store, session = input_store
    store.set_session_draft(session.id, "draft", authored_token=(2, 1))
    captured = store.session_input_snapshot(session.id)

    store.set_session_draft(session.id, "draft", authored_token=(2, 1))
    store.set_session_draft(session.id, "draft")

    assert store.session_inputs_are_current(captured) is True
    assert (
        store.session_input_snapshot(session.id).draft_revision
        == captured.draft_revision
    )


def test_legacy_clear_retype_still_invalidates_received_input(input_store):
    store, session = input_store
    store.set_session_draft(session.id, "same draft")
    captured = store.session_input_snapshot(session.id)
    store.set_session_draft(session.id, "")
    store.set_session_draft(session.id, "same draft")

    assert store.session_inputs_are_current(captured) is False
    assert store.commit_session_input_draft(captured) is False
    assert store.session_draft(session.id) == "same draft"


@pytest.mark.parametrize("change", ["binding", "ephemeral", "incarnation", "foreign"])
def test_input_checks_and_clear_refuse_changed_session_owner(input_store, change):
    store, session = input_store
    store.set_session_draft(session.id, "original")
    captured = store.session_input_snapshot(session.id)
    target = store
    other_store = None
    if change == "binding":
        session.conversation_binding_revision += 1
    elif change == "ephemeral":
        session.ephemeral = False
    elif change == "incarnation":
        store.close_session(session.id)
        store.create_session(session_id=session.id, ephemeral=True)
        store.set_session_draft(session.id, "original")
    else:
        other_store = ConsoleChatStore()
        other_store.create_session(session_id=session.id, ephemeral=True)
        other_store.set_session_draft(session.id, "original")
        target = other_store
    try:
        assert target.session_inputs_are_current(captured) is False
        assert target.commit_session_input_draft(captured) is False
        assert target.session_draft(session.id) == "original"
    finally:
        if other_store is not None:
            other_store.close_session(session.id)
            other_store.end_app_runtime()


@pytest.mark.parametrize("mutation", ["append", "remove", "clear_rearm", "replace"])
def test_attachment_changes_invalidate_received_snapshot(input_store, mutation):
    store, session = input_store
    original = _attachment("original")
    store.add_pending_attachment(session.id, original)
    captured = store.session_input_snapshot(session.id)
    if mutation == "append":
        store.add_pending_attachment(session.id, _attachment("later"))
    elif mutation == "remove":
        assert store.consume_pending_attachment(session.id, "original") is True
    elif mutation == "clear_rearm":
        store.clear_pending_attachments(session.id)
        store.add_pending_attachment(session.id, original)
    else:
        # Reusing an ID does not authorize replacing the original staged object.
        store.set_pending_attachment(session.id, _attachment("original"))

    assert store.session_inputs_are_current(captured) is False
    assert (
        store.session_input_snapshot(session.id).attachment_revision
        > captured.attachment_revision
    )


def test_transfer_restore_retains_legacy_prefix_and_advances_revision(input_store):
    store, session = input_store
    original = _attachment("original")
    later = _attachment("later")
    store.add_pending_attachment(session.id, original)
    captured = store.session_input_snapshot(session.id)
    store.add_pending_attachment(session.id, later)

    transferred = store.transfer_pending_attachments_to_turn(
        session.id, "accepted-turn", (original.attachment_id,)
    )

    assert transferred[0] is original
    assert store.pending_attachments(session.id) == [later]
    assert store.session_inputs_are_current(captured) is False
    after_transfer = store.session_input_snapshot(session.id)
    store.restore_transferred_pending_attachments(session.id, transferred)
    assert store.pending_attachments(session.id) == [original, later]
    assert store.pending_attachments(session.id)[0] is original
    assert (
        store.session_input_snapshot(session.id).attachment_revision
        > after_transfer.attachment_revision
    )


def test_rearming_identical_prefill_invalidates_received_snapshot(input_store):
    store, session = input_store
    store.set_session_one_shot_prefill(session.id, "same prefill")
    captured = store.session_input_snapshot(session.id)
    store.set_session_one_shot_prefill(session.id, "same prefill")

    assert store.session_inputs_are_current(captured) is False
    assert (
        store.session_input_snapshot(session.id).prefill_revision
        > captured.prefill_revision
    )


def test_draft_acceptance_cas_succeeds_after_exact_staged_input_transfer(input_store):
    store, session = input_store
    attachment = _attachment("original")
    store.set_session_draft(session.id, "accepted", authored_token=(5, 1))
    store.add_pending_attachment(session.id, attachment)
    store.set_session_one_shot_prefill(session.id, "prefill")
    captured = store.session_input_snapshot(session.id)
    assert store.session_inputs_are_current(captured) is True
    store.transfer_pending_attachments_to_turn(session.id, "turn", ("original",))
    store.consume_session_one_shot_prefill(session.id, captured.prefill_revision)

    assert store.commit_session_input_draft(captured) is True
    assert store.session_draft(session.id) == ""
    assert store.commit_session_input_draft(captured) is False


def test_draft_after_promotion_survives_old_acceptance_clear(input_store):
    store, session = input_store
    store.set_session_draft(session.id, "accepted", authored_token=(4, 1))
    captured = store.session_input_snapshot(session.id)
    store.set_session_draft(session.id, "accepted new suffix", authored_token=(4, 2))

    assert store.commit_session_input_draft(captured) is False
    assert store.session_draft(session.id) == "accepted new suffix"


def test_received_intent_contains_detached_selection_and_no_view_callback(input_store):
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_configuration_preparation import (
        ConsoleTurnCaptureSelection,
    )
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent

    store, session = input_store
    store.set_session_draft(session.id, "private intent draft")
    selected = ConsoleTurnCaptureSelection(
        ConsoleProviderSelection(provider="llama_cpp", explicit_model="test-model"),
        None,
        {"source_types": ["notes"]},
        {},
        None,
        False,
        False,
    )
    intent = ConsoleReceivedTurnIntent(
        turn_id="received-turn",
        session_id=session.id,
        inputs=store.session_input_snapshot(session.id),
        selection=selected,
        staged_evidence_launch=None,
        staged_evidence_revision=0,
        view_attachment_generation=3,
        queue_revision=None,
    )

    assert intent.inputs.draft == "private intent draft"
    assert intent.selection.rag_defaults["source_types"] == ("notes",)
    assert intent.queue_revision is None
    assert "private intent draft" not in repr(intent)
    with pytest.raises(FrozenInstanceError):
        intent.turn_id = "successor"


@pytest.mark.parametrize(
    "change",
    [
        "settings_revision",
        "generation_settings_revision",
        "context_policy_revision",
        "identity_revision",
        "workspace_id",
        "settings_owner",
    ],
)
def test_receipt_checks_original_configuration_witness(input_store, change):
    from dataclasses import replace
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    store, session = input_store
    if change == "settings_owner":
        store.replace_session_settings(
            session.id, ConsoleSessionSettings(provider="llama_cpp", model="test-model")
        )
    store.set_session_draft(session.id, "captured")
    captured = store.session_input_snapshot(session.id)
    if change == "workspace_id":
        session.workspace_id = "other-workspace"
    elif change == "settings_owner":
        assert session.settings is not None
        session.settings = replace(session.settings)
    else:
        setattr(session, change, getattr(session, change) + 1)

    assert store.session_inputs_are_current(captured) is False
    assert store.session_draft(session.id) == "captured"


def test_postpromotion_settings_change_does_not_block_exact_draft_clear(input_store):
    store, session = input_store
    store.set_session_draft(session.id, "accepted", authored_token=(3, 1))
    captured = store.session_input_snapshot(session.id)
    session.settings_revision += 1
    session.identity_revision += 1
    session.workspace_id = "next-workspace"

    assert store.session_inputs_are_current(captured) is False
    assert store.commit_session_input_draft(captured) is True
    assert store.session_draft(session.id) == ""


def _received_preparation(session, *, draft="captured", state=None, pause_kind=None):
    from dataclasses import replace

    from Tests.Chat.test_console_automatic_library_preparation import _preparation
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState

    return replace(
        _preparation(
            session_id=session.id,
            draft=draft,
            state=state or ConsoleTurnPreparationState.PREPARING,
            pause_kind=pause_kind,
        ),
        ephemeral=session.ephemeral,
    )


def test_received_claim_refuses_stale_authored_revision_without_occupying_slot(
    input_store,
):
    store, session = input_store
    store.set_session_draft(session.id, "captured", authored_token=(4, 1))
    captured = store.session_input_snapshot(session.id)
    store.set_session_draft(session.id, "captured", authored_token=(5, 3))

    assert (
        store.claim_received_turn(
            session.id, "turn", draft_revision=captured.draft_revision
        )
        is None
    )
    assert store.received_turn_for_session(session.id) is None


def test_promoted_draft_revision_survives_pause_and_resume(input_store):
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsolePreparationPauseKind,
        ConsolePreparationTransition,
        ConsoleTurnPreparationState,
    )

    store, session = input_store
    store.set_session_draft(session.id, "captured", authored_token=(4, 1))
    captured = store.session_input_snapshot(session.id)
    claim = store.claim_received_turn(
        session.id, "turn", draft_revision=captured.draft_revision
    )
    assert claim is not None
    preparation = _received_preparation(session)
    promoted = store.promote_received_turn(claim, preparation)
    assert promoted is not None
    assert promoted.input_draft_revision == captured.draft_revision
    paused = store.compare_and_set_preparation(
        session.id,
        ConsolePreparationTransition(
            preparation.preparation_id,
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.PAUSED,
            ConsolePreparationPauseKind.RETRIEVAL,
            None,
        ),
    )
    assert paused is not None
    assert paused.input_draft_revision == captured.draft_revision
    resumed = store.compare_and_set_preparation(
        session.id,
        ConsolePreparationTransition(
            preparation.preparation_id,
            ConsoleTurnPreparationState.PAUSED,
            ConsoleTurnPreparationState.PREPARING,
            None,
            "received-resumed-attempt",
        ),
    )
    assert resumed is not None
    assert resumed.attempt_id != paused.attempt_id
    assert resumed.execution_context.library_authority.attempt_id == resumed.attempt_id
    assert resumed.input_draft_revision == captured.draft_revision


def test_received_cancel_recovery_cannot_overwrite_newer_draft(input_store):
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState

    store, session = input_store
    store.set_session_draft(session.id, "captured", authored_token=(4, 1))
    captured = store.session_input_snapshot(session.id)
    claim = store.claim_received_turn(
        session.id, "turn", draft_revision=captured.draft_revision
    )
    assert claim is not None
    preparation = store.promote_received_turn(claim, _received_preparation(session))
    assert preparation is not None
    store.set_session_draft(session.id, "next independent draft", authored_token=(4, 2))
    revision = store.session_input_snapshot(session.id).draft_revision

    cancelled = store.cancel_preparation(
        session.id,
        preparation.preparation_id,
        expected_state=ConsoleTurnPreparationState.PREPARING,
    )

    assert cancelled is not None
    assert cancelled.input_draft_revision == captured.draft_revision
    assert store.session_draft(session.id) == "next independent draft"
    assert store.session_input_snapshot(session.id).draft_revision == revision


def test_legacy_manual_cancel_keeps_original_draft_restore(input_store):
    from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState

    store, session = input_store
    store.set_session_draft(session.id, "later")
    preparation = _received_preparation(session, draft="legacy executed")
    assert store.begin_preparation(preparation) is preparation
    before = store.session_input_snapshot(session.id).draft_revision

    cancelled = store.cancel_preparation(
        session.id,
        preparation.preparation_id,
        expected_state=ConsoleTurnPreparationState.PREPARING,
    )

    assert cancelled is not None
    assert cancelled.input_draft_revision is None
    assert store.session_draft(session.id) == "legacy executed"
    assert store.session_input_snapshot(session.id).draft_revision > before


@pytest.mark.asyncio
async def test_identical_authored_retype_survives_old_handoff_consumption(
    handoff_store,
):
    from Tests.Chat.test_console_chat_start import _create_handoff, _restore_handoff

    store, database = handoff_store
    session = _restore_handoff(store, _create_handoff(store))
    original = store.session_input_snapshot(session.id)
    accepted_revision = session.agent_handoff_revision
    assert session.agent_handoff_state == "pending"
    assert original.draft == "original"

    store.set_session_draft(session.id, "original", authored_token=(19, 3))
    assert session.draft_revision > original.draft_revision
    assert session.agent_handoff_revision > accepted_revision
    assert await store.drain_agent_handoff(session.id)
    stored = database.get_conversation_by_id(session.persisted_conversation_id)
    import json

    handoff = json.loads(stored["metadata"])["console_agent_handoff"]
    assert handoff["draft"] == "original"
    assert handoff["draft_revision"] == session.agent_handoff_revision

    store.publish_agent_handoff_consumed(session.id, accepted_revision)

    assert session.draft == "original"
    assert session.agent_handoff_state == "consumed"
    assert store.commit_session_input_draft(original) is False
