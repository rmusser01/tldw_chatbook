"""Contracts for the device-private one-time Notes import receipt ledger."""

from __future__ import annotations

import inspect
import os
import sqlite3
import stat
from dataclasses import FrozenInstanceError, replace
from itertools import islice, repeat
from pathlib import Path

import pytest

from tldw_chatbook import config
from tldw_chatbook.Notes.note_import_execution_models import (
    ImportEffectState,
    ImportItemOutcome,
    ImportSessionState,
    approve_note_import_plan,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    ImportClassification,
    ImportMatch,
    ImportMatchKind,
    ImportPreviewItem,
    ImportSource,
    ImportSourceKind,
    NoteImportPlan,
    ParsedNotePayload,
    ProposedFolderMembership,
    RootCollisionChoice,
    RootCollisionState,
)
from tldw_chatbook.Notes.note_import_receipts import (
    EFFECT_STATE_TRANSITIONS,
    ITEM_OUTCOME_TRANSITIONS,
    SESSION_STATE_TRANSITIONS,
    EffectTransition,
    ImportReceiptConflictError,
    ImportReceiptError,
    ImportReceiptTransitionError,
    ItemTransition,
    NoteImportReceiptRepository,
)

_APPROVAL_ID = "00000000-0000-4000-8000-000000000011"
_PRIVATE_SOURCE = Path("/private/alice/Project/notes.json")
_PRIVATE_TITLE = "Private quarterly title"
_PRIVATE_BODY = "Body secret which must never be persisted"
_PRIVATE_KEYWORD = "confidential-keyword"
_PRIVATE_TEMPLATE = "Private journal template"
_RAW_EXCEPTION = "raw exception /private/alice/Project/notes.json"


def _item(
    *,
    item_id: str = "item-1",
    content: str = _PRIVATE_BODY,
    selected_action: ImportAction = ImportAction.UPDATE_EXISTING,
) -> ImportPreviewItem:
    payload = ParsedNotePayload(
        title=_PRIVATE_TITLE,
        content=content,
        keywords=(_PRIVATE_KEYWORD,),
        template_name=_PRIVATE_TEMPLATE,
    )
    if selected_action is ImportAction.UPDATE_EXISTING:
        classification = ImportClassification.CHANGED_REPEAT
        match = ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id="opaque-note-7",
            note_version=7,
        )
        allowed_actions = (
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        )
        replace_content = True
    else:
        classification = ImportClassification.NEW
        match = None
        allowed_actions = (ImportAction.SKIP, ImportAction.CREATE_NEW)
        replace_content = False
    return ImportPreviewItem(
        item_id=item_id,
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="Project/notes.json",
            source_path=_PRIVATE_SOURCE,
        ),
        payloads=(payload,),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Imported Project", "Meetings"),
            ),
        ),
        classification=classification,
        reason="Ready after explicit review.",
        default_action=ImportAction.CREATE_NEW,
        selected_action=selected_action,
        allowed_actions=allowed_actions,
        match=match,
        replace_content=replace_content,
        add_membership=True,
    )


def _plan(*, content: str = _PRIVATE_BODY) -> NoteImportPlan:
    return NoteImportPlan(
        bounds=ImportBounds(
            max_files=50,
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
            max_depth=8,
            max_entries=1_000,
            max_notes_per_file=100,
            max_keywords_per_note=50,
        ),
        items=(_item(content=content),),
        proposed_folder_paths=(
            ("Imported Project",),
            ("Imported Project", "Meetings"),
        ),
        root_collision=RootCollisionState(
            proposed_label="Project",
            collides=True,
            choice=RootCollisionChoice.RENAMED_ROOT,
            resolved_label="Imported Project",
        ),
    )


def _approved(*, content: str = _PRIVATE_BODY):
    return approve_note_import_plan(
        _plan(content=content),
        approval_id=_APPROVAL_ID,
    )


def _create_item_with_payloads(*, payload_count: int = 2) -> ImportPreviewItem:
    payloads = tuple(
        ParsedNotePayload(
            title=f"Private title {index}",
            content=f"Private body {index}",
            keywords=(f"private-keyword-{index}",),
        )
        for index in range(payload_count)
    )
    return ImportPreviewItem(
        item_id="multi-create",
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="Project/multi.json",
            source_path=Path("/private/alice/Project/multi.json"),
        ),
        payloads=payloads,
        memberships=tuple(
            ProposedFolderMembership(
                payload_index=index,
                folder_segments=("Imported Project", "Meetings"),
            )
            for index in range(payload_count)
        ),
        classification=ImportClassification.NEW,
        reason="Ready to create.",
        default_action=ImportAction.CREATE_NEW,
        selected_action=ImportAction.CREATE_NEW,
        allowed_actions=(ImportAction.SKIP, ImportAction.CREATE_NEW),
        match=None,
        replace_content=False,
        add_membership=True,
    )


def _approved_for_item(
    item: ImportPreviewItem,
    *,
    approval_id: str = _APPROVAL_ID,
    proposed_folder_paths: tuple[tuple[str, ...], ...] = (
        ("Imported Project",),
        ("Imported Project", "Meetings"),
    ),
):
    return approve_note_import_plan(
        replace(
            _plan(),
            items=(item,),
            proposed_folder_paths=proposed_folder_paths,
        ),
        approval_id=approval_id,
    )


def _applied_transition(effect, *, note_id: str = "opaque-note-1"):
    if effect.table == "import_payload_effects":
        return EffectTransition(
            table=effect.table,
            effect_id=effect.effect_id,
            state=ImportEffectState.APPLIED,
            target_note_id=note_id,
            observed_version=1,
        )
    if effect.table == "import_folder_effects":
        return EffectTransition(
            table=effect.table,
            effect_id=effect.effect_id,
            state=ImportEffectState.APPLIED,
            target_folder_id="opaque-folder-1",
        )
    return EffectTransition(
        table=effect.table,
        effect_id=effect.effect_id,
        state=ImportEffectState.APPLIED,
        target_note_id=note_id,
        target_folder_id="opaque-folder-1",
    )


def _repository(tmp_path: Path) -> NoteImportReceiptRepository:
    return NoteImportReceiptRepository(tmp_path / "notes-sync.sqlite3")


def test_notes_sync_state_path_is_profile_local_and_not_a_generic_database_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    assert config.get_notes_sync_state_db_path() == (
        tmp_path / "tldw_chatbook_notes_sync_state.db"
    )
    generic_paths = (
        Path(__file__).parents[2] / "tldw_chatbook/Chatbooks/database_paths.py"
    ).read_text(encoding="utf-8")
    settings_backup = (
        Path(__file__).parents[2] / "tldw_chatbook/UI/Tools_Settings_Window.py"
    ).read_text(encoding="utf-8")
    for source in (generic_paths, settings_backup):
        assert "get_notes_sync_state_db_path" not in source
        assert "tldw_chatbook_notes_sync_state.db" not in source


def test_receipt_repository_creates_v1_normalized_schema_without_private_text(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    snapshot = repository.begin(_approved(), batch_size=25)
    schema = repository._test_schema_snapshot()

    assert snapshot.batch_size == 25
    assert schema.user_version == 1
    assert set(schema.tables) == {
        "import_sessions",
        "import_items",
        "import_payload_effects",
        "import_folder_effects",
        "import_membership_effects",
    }
    database_text = (
        (tmp_path / "notes-sync.sqlite3").read_bytes().decode("utf-8", errors="ignore")
    )
    for forbidden in (
        str(_PRIVATE_SOURCE),
        "Project/notes.json",
        _PRIVATE_TITLE,
        _PRIVATE_BODY,
        _PRIVATE_KEYWORD,
        _PRIVATE_TEMPLATE,
        _RAW_EXCEPTION,
    ):
        assert forbidden not in database_text


def test_schema_column_census_excludes_content_and_exception_fields(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)

    columns = {
        column
        for table_columns in repository._test_schema_snapshot().columns.values()
        for column in table_columns
    }
    assert {
        "approval_id",
        "session_id",
        "item_id",
        "plan_digest",
        "source_locator_digest",
        "payload_digest",
        "selected_action",
        "effect_kind",
        "state",
        "target_note_id",
        "target_folder_id",
        "expected_version",
        "observed_version",
        "reason_code",
        "retryable",
    } <= columns
    forbidden_column_fragments = {
        "absolute_path",
        "display_path",
        "source_path",
        "title",
        "content",
        "keyword",
        "template",
        "exception",
        "raw_error",
    }
    assert not {
        column
        for column in columns
        if any(fragment in column for fragment in forbidden_column_fragments)
    }


def test_begin_is_idempotent_durable_and_rejects_digest_or_batch_substitution(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    first = repository.begin(_approved(), batch_size=10)

    assert repository.begin(_approved(), batch_size=10) == first
    assert _repository(tmp_path).get_session(_APPROVAL_ID) == first
    with pytest.raises(ImportReceiptConflictError) as digest_conflict:
        repository.begin(_approved(content="substituted private body"), batch_size=10)
    with pytest.raises(ImportReceiptConflictError) as batch_conflict:
        repository.begin(_approved(), batch_size=11)
    for error in (digest_conflict.value, batch_conflict.value):
        assert _APPROVAL_ID not in str(error)
        assert "substituted private body" not in str(error)


def test_begin_validates_bounded_batch_size_without_creating_a_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    repository = NoteImportReceiptRepository(database)

    for invalid in (0, 101, True, 1.5, "25"):
        with pytest.raises((TypeError, ValueError)):
            repository.begin(_approved(), batch_size=invalid)  # type: ignore[arg-type]
    assert not database.exists()


def test_begin_seeds_only_approved_effects_and_skip_has_no_mutation_effects(
    tmp_path: Path,
) -> None:
    plan = _plan()
    create_item = _item(item_id="item-create", selected_action=ImportAction.CREATE_NEW)
    skip_item = replace(
        create_item,
        item_id="item-skip",
        selected_action=ImportAction.SKIP,
        add_membership=False,
    )
    approved = approve_note_import_plan(
        replace(plan, items=(plan.items[0], create_item, skip_item)),
        approval_id=_APPROVAL_ID,
    )
    repository = _repository(tmp_path)

    session = repository.begin(approved, batch_size=25)
    durable = repository.load_session_snapshot(session.approval_id)

    assert len(durable.items) == 3
    assert len(durable.payload_effects) == 2
    assert {effect.effect_kind for effect in durable.payload_effects} == {
        "create_note",
        "replace_content",
    }
    assert len(durable.folder_effects) == 2
    assert len(durable.membership_effects) == 2
    assert all(effect.item_id != "item-skip" for effect in durable.payload_effects)
    assert all(effect.item_id != "item-skip" for effect in durable.membership_effects)


def test_all_skip_and_zero_work_sessions_complete_without_folder_effects(
    tmp_path: Path,
) -> None:
    create_item = _item(selected_action=ImportAction.CREATE_NEW)
    skip_item = replace(
        create_item,
        selected_action=ImportAction.SKIP,
        add_membership=False,
    )
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(skip_item), batch_size=25)

    skipped = repository.load_session_snapshot(_APPROVAL_ID)
    assert skipped.payload_effects == ()
    assert skipped.folder_effects == ()
    assert skipped.membership_effects == ()
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    repository.transition_item(
        _APPROVAL_ID,
        skip_item.item_id,
        ImportItemOutcome.SKIPPED,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    skip_receipt = repository.aggregate_receipt(_APPROVAL_ID)
    assert (skip_receipt.total, skip_receipt.skipped) == (1, 1)

    empty_id = "00000000-0000-4000-8000-000000000012"
    empty = approve_note_import_plan(
        replace(_plan(), items=()),
        approval_id=empty_id,
    )
    repository.begin(empty, batch_size=25)
    assert repository.load_session_snapshot(empty_id).folder_effects == ()
    repository.transition_session(empty_id, ImportSessionState.RUNNING)
    repository.transition_session(empty_id, ImportSessionState.COMPLETED)
    empty_receipt = repository.aggregate_receipt(empty_id)
    assert (empty_receipt.total, empty_receipt.completed) == (0, 0)


def test_multi_payload_create_counts_notes_and_reopens_all_target_ids(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    session = repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    assert session.total == 2
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    durable = repository.load_session_snapshot(_APPROVAL_ID)
    note_ids = ("opaque-created-note-1", "opaque-created-note-2")
    transitions = [
        EffectTransition(
            table=effect.table,
            effect_id=effect.effect_id,
            state=ImportEffectState.APPLIED,
            target_note_id=note_ids[effect.payload_index],
            observed_version=1,
        )
        for effect in durable.payload_effects
    ]
    transitions.extend(_applied_transition(effect) for effect in durable.folder_effects)
    transitions.extend(
        _applied_transition(
            effect,
            note_id=note_ids[effect.payload_index],
        )
        for effect in durable.membership_effects
    )
    repository.transition_effects(_APPROVAL_ID, transitions)
    repository.transition_item(
        _APPROVAL_ID,
        "multi-create",
        ImportItemOutcome.IMPORTED,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)

    receipt = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (receipt.total, receipt.completed, receipt.imported) == (2, 2, 2)
    assert receipt._note_ids == note_ids


def test_multi_payload_failure_uses_note_weight_for_failed_and_retryable_counts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(
        _approved_for_item(_create_item_with_payloads()),
        batch_size=25,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    repository.transition_item(
        _APPROVAL_ID,
        "multi-create",
        ImportItemOutcome.FAILED,
        reason_code="database_busy",
        retryable=True,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)

    receipt = _repository(tmp_path).aggregate_receipt(_APPROVAL_ID)
    assert (receipt.total, receipt.completed, receipt.failed, receipt.retryable) == (
        2,
        2,
        2,
        2,
    )


def test_exact_state_transition_allowlists_are_closed_and_immutable() -> None:
    assert SESSION_STATE_TRANSITIONS == {
        ImportSessionState.PENDING: frozenset(
            {ImportSessionState.RUNNING, ImportSessionState.CANCELLED}
        ),
        ImportSessionState.RUNNING: frozenset(
            {
                ImportSessionState.CANCELLED,
                ImportSessionState.COMPLETED,
                ImportSessionState.NEEDS_ATTENTION,
            }
        ),
        ImportSessionState.NEEDS_ATTENTION: frozenset(
            {ImportSessionState.RUNNING, ImportSessionState.CANCELLED}
        ),
        ImportSessionState.COMPLETED: frozenset(),
        ImportSessionState.CANCELLED: frozenset(),
    }
    assert ITEM_OUTCOME_TRANSITIONS == {
        ImportItemOutcome.PENDING: frozenset(
            {
                ImportItemOutcome.IMPORTED,
                ImportItemOutcome.UPDATED,
                ImportItemOutcome.SKIPPED,
                ImportItemOutcome.FAILED,
            }
        ),
        ImportItemOutcome.IMPORTED: frozenset(),
        ImportItemOutcome.UPDATED: frozenset(),
        ImportItemOutcome.SKIPPED: frozenset(),
        ImportItemOutcome.FAILED: frozenset(),
    }
    assert EFFECT_STATE_TRANSITIONS == {
        ImportEffectState.PENDING: frozenset(
            {ImportEffectState.APPLIED, ImportEffectState.FAILED}
        ),
        ImportEffectState.APPLIED: frozenset(),
        ImportEffectState.FAILED: frozenset(),
    }
    with pytest.raises(TypeError):
        SESSION_STATE_TRANSITIONS[ImportSessionState.PENDING] = frozenset()  # type: ignore[index]


def test_session_item_and_effect_illegal_transitions_fail_without_mutation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    session = repository.begin(_approved(), batch_size=25)
    durable = repository.load_session_snapshot(session.approval_id)
    effect = durable.payload_effects[0]

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.PENDING,
        )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    table=effect.table,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.PENDING,
                ),
            ),
        )
    after = repository.load_session_snapshot(_APPROVAL_ID)
    assert after.state is ImportSessionState.PENDING
    assert after.items[0].outcome is ImportItemOutcome.PENDING
    assert after.payload_effects[0].state is ImportEffectState.PENDING


def test_completion_rejects_pending_failed_or_unapplied_work_and_rolls_back(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING

    repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.UPDATED,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_session(_APPROVAL_ID, ImportSessionState.COMPLETED)
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.RUNNING
    assert reopened.items[0].outcome is ImportItemOutcome.UPDATED
    assert any(
        effect.state is ImportEffectState.PENDING
        for effect in (
            *reopened.payload_effects,
            *reopened.folder_effects,
            *reopened.membership_effects,
        )
    )


def test_cancelled_session_rejects_item_effect_and_retry_mutations_durably(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.CANCELLED)
    effect = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.FAILED,
            retryable=True,
            reason_code="cancelled_work",
        )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    table=effect.table,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.FAILED,
                    retryable=True,
                    reason_code="cancelled_work",
                ),
            ),
        )
    with pytest.raises(ImportReceiptTransitionError):
        repository.reset_retryable_effect(
            _APPROVAL_ID,
            table=effect.table,
            effect_id=effect.effect_id,
        )

    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert reopened.state is ImportSessionState.CANCELLED
    assert reopened.items[0].outcome is ImportItemOutcome.PENDING
    assert reopened.payload_effects[0].state is ImportEffectState.PENDING


@pytest.mark.parametrize(
    ("action", "invalid_outcome"),
    [
        (ImportAction.SKIP, ImportItemOutcome.IMPORTED),
        (ImportAction.CREATE_NEW, ImportItemOutcome.UPDATED),
        (ImportAction.UPDATE_EXISTING, ImportItemOutcome.IMPORTED),
    ],
)
def test_item_outcomes_must_match_the_approved_action(
    tmp_path: Path,
    action: ImportAction,
    invalid_outcome: ImportItemOutcome,
) -> None:
    item = _item(selected_action=ImportAction.CREATE_NEW)
    if action is ImportAction.SKIP:
        item = replace(item, selected_action=action, add_membership=False)
    elif action is ImportAction.UPDATE_EXISTING:
        item = _item(selected_action=action)
    repository = _repository(tmp_path)
    repository.begin(_approved_for_item(item), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_item(
            _APPROVAL_ID,
            item.item_id,
            invalid_outcome,
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0].outcome
        is ImportItemOutcome.PENDING
    )


def test_multi_item_effect_transition_is_atomic_when_one_row_is_invalid(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    effects = repository.load_session_snapshot(_APPROVAL_ID).folder_effects
    assert len(effects) == 2

    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                table=effects[1].table,
                effect_id=effects[1].effect_id,
                state=ImportEffectState.APPLIED,
                target_folder_id="opaque-folder-2",
            ),
        ),
    )
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id="item-1",
                    outcome=ImportItemOutcome.UPDATED,
                ),
            ),
            effect_transitions=(
                EffectTransition(
                    table=effects[0].table,
                    effect_id=effects[0].effect_id,
                    state=ImportEffectState.APPLIED,
                    target_folder_id="opaque-folder-1",
                ),
                EffectTransition(
                    table=effects[1].table,
                    effect_id=effects[1].effect_id,
                    state=ImportEffectState.FAILED,
                    reason_code="folder_conflict",
                ),
            ),
        )

    after = repository.load_session_snapshot(_APPROVAL_ID)
    assert after.items[0].outcome is ImportItemOutcome.PENDING
    assert [effect.state for effect in after.folder_effects] == [
        ImportEffectState.PENDING,
        ImportEffectState.APPLIED,
    ]


def test_failed_effect_retry_requires_explicit_reset_and_safe_reason_codes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    effect = repository.load_session_snapshot(_APPROVAL_ID).payload_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                table=effect.table,
                effect_id=effect.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
            ),
        ),
    )

    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (
                EffectTransition(
                    table=effect.table,
                    effect_id=effect.effect_id,
                    state=ImportEffectState.APPLIED,
                ),
            ),
        )
    reset = repository.reset_retryable_effect(
        _APPROVAL_ID,
        table=effect.table,
        effect_id=effect.effect_id,
    )
    assert reset.state is ImportEffectState.PENDING
    assert reset.reason_code is None
    assert reset.retryable is False
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    with pytest.raises(ValueError) as caught:
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.FAILED,
            reason_code=_RAW_EXCEPTION,
            retryable=True,
        )
    assert _RAW_EXCEPTION not in str(caught.value)


def test_preseeded_update_target_and_observed_version_cannot_be_rebound(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptConflictError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.FAILED,
            reason_code="version_conflict",
            retryable=True,
            target_note_id="substituted-note",
            observed_version=8,
        )
    original = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert original.target_note_id == "opaque-note-7"
    assert original.observed_version is None
    assert original.outcome is ImportItemOutcome.PENDING

    failed = repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.FAILED,
        reason_code="database_busy",
        retryable=True,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    assert (failed.target_note_id, failed.observed_version) == ("opaque-note-7", 8)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    reset = repository.reset_retryable_item(_APPROVAL_ID, item_id="item-1")
    assert (reset.target_note_id, reset.observed_version) == ("opaque-note-7", 8)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptConflictError):
        repository.transition_item(
            _APPROVAL_ID,
            "item-1",
            ImportItemOutcome.UPDATED,
            target_note_id="opaque-note-7",
            observed_version=9,
        )
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).items[0]
    assert reopened.outcome is ImportItemOutcome.PENDING
    assert (reopened.target_note_id, reopened.observed_version) == (
        "opaque-note-7",
        8,
    )


def test_effect_bindings_survive_retry_and_mixed_conflict_rolls_back_after_reopen(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    durable = repository.load_session_snapshot(_APPROVAL_ID)
    payload = durable.payload_effects[0]
    folder = durable.folder_effects[0]
    membership = durable.membership_effects[0]
    repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                table=payload.table,
                effect_id=payload.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            EffectTransition(
                table=folder.table,
                effect_id=folder.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_folder_id="opaque-effect-folder",
            ),
            EffectTransition(
                table=membership.table,
                effect_id=membership.effect_id,
                state=ImportEffectState.FAILED,
                reason_code="database_busy",
                retryable=True,
                target_note_id="opaque-note-7",
                target_folder_id="opaque-effect-folder",
            ),
        ),
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)
    for effect in (payload, folder, membership):
        repository.reset_retryable_effect(
            _APPROVAL_ID,
            table=effect.table,
            effect_id=effect.effect_id,
        )
    reopened = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    rebound = {
        effect.table: effect
        for effect in (
            reopened.payload_effects[0],
            reopened.folder_effects[0],
            reopened.membership_effects[0],
        )
    }
    assert (
        rebound[payload.table].target_note_id,
        rebound[payload.table].observed_version,
    ) == ("opaque-note-7", 8)
    assert rebound[folder.table].target_folder_id == "opaque-effect-folder"
    assert (
        rebound[membership.table].target_note_id,
        rebound[membership.table].target_folder_id,
    ) == ("opaque-note-7", "opaque-effect-folder")
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)

    with pytest.raises(ImportReceiptConflictError):
        repository.transition_batch(
            _APPROVAL_ID,
            item_transitions=(
                ItemTransition(
                    item_id="item-1",
                    outcome=ImportItemOutcome.UPDATED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
            ),
            effect_transitions=(
                EffectTransition(
                    table=payload.table,
                    effect_id=payload.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_note_id="opaque-note-7",
                    observed_version=8,
                ),
                EffectTransition(
                    table=folder.table,
                    effect_id=folder.effect_id,
                    state=ImportEffectState.APPLIED,
                    target_folder_id="substituted-folder",
                ),
            ),
        )
    rolled_back = _repository(tmp_path).load_session_snapshot(_APPROVAL_ID)
    assert rolled_back.items[0].outcome is ImportItemOutcome.PENDING
    assert rolled_back.payload_effects[0].state is ImportEffectState.PENDING
    assert rolled_back.folder_effects[0].state is ImportEffectState.PENDING
    applied = repository.transition_effects(
        _APPROVAL_ID,
        (
            EffectTransition(
                table=payload.table,
                effect_id=payload.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                observed_version=8,
            ),
            EffectTransition(
                table=folder.table,
                effect_id=folder.effect_id,
                state=ImportEffectState.APPLIED,
                target_folder_id="opaque-effect-folder",
            ),
            EffectTransition(
                table=membership.table,
                effect_id=membership.effect_id,
                state=ImportEffectState.APPLIED,
                target_note_id="opaque-note-7",
                target_folder_id="opaque-effect-folder",
            ),
        ),
    )
    assert all(effect.state is ImportEffectState.APPLIED for effect in applied)


@pytest.mark.parametrize(
    "transition",
    [
        EffectTransition(
            table="import_payload_effects",
            effect_id="opaque-effect",
            state=ImportEffectState.APPLIED,
        ),
        EffectTransition(
            table="import_folder_effects",
            effect_id="opaque-effect",
            state=ImportEffectState.APPLIED,
        ),
        EffectTransition(
            table="import_membership_effects",
            effect_id="opaque-effect",
            state=ImportEffectState.APPLIED,
        ),
    ],
)
def test_applied_effects_require_their_reconciliation_identities(
    tmp_path: Path,
    transition: EffectTransition,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    snapshot = repository.load_session_snapshot(_APPROVAL_ID)
    effects = {
        "import_payload_effects": snapshot.payload_effects[0],
        "import_folder_effects": snapshot.folder_effects[0],
        "import_membership_effects": snapshot.membership_effects[0],
    }
    seeded = effects[transition.table]

    with pytest.raises(ImportReceiptTransitionError):
        repository.transition_effects(
            _APPROVAL_ID,
            (replace(transition, effect_id=seeded.effect_id),),
        )
    assert (
        _repository(tmp_path).load_session_snapshot(_APPROVAL_ID).state
        is ImportSessionState.RUNNING
    )


def test_transition_collection_ceiling_fails_before_database_open(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    repository = NoteImportReceiptRepository(database)
    transition = EffectTransition(
        table="import_payload_effects",
        effect_id="opaque-effect",
        state=ImportEffectState.FAILED,
        reason_code="database_busy",
        retryable=True,
    )

    with pytest.raises(ValueError, match="ceiling"):
        repository.transition_effects(
            _APPROVAL_ID,
            islice(repeat(transition), 100_001),
        )
    assert not database.exists()


def test_hostile_transition_iterator_is_sanitized_before_database_open(
    tmp_path: Path,
) -> None:
    database = tmp_path / "notes-sync.sqlite3"
    repository = NoteImportReceiptRepository(database)
    secret = "hostile iterator detail must not leak"
    transition = EffectTransition(
        table="import_payload_effects",
        effect_id="opaque-effect",
        state=ImportEffectState.FAILED,
        reason_code="database_busy",
        retryable=True,
    )

    def hostile_transitions():
        yield transition
        raise RuntimeError(secret)

    with pytest.raises(ValueError, match="ceiling") as caught:
        repository.transition_effects(_APPROVAL_ID, hostile_transitions())
    assert secret not in str(caught.value)
    assert caught.value.__context__ is None
    assert not database.exists()


def test_begin_sanitizes_stateful_private_canonicalization_failure(
    tmp_path: Path,
) -> None:
    secret = "hostile source and content must not leak"
    approved = _approved()
    concrete_path_type = type(Path("/"))

    class StatefulHostilePath(concrete_path_type):  # type: ignore[misc,valid-type]
        def __str__(self) -> str:
            raise RuntimeError(secret)

    object.__setattr__(
        approved.plan.items[0].source,
        "source_path",
        StatefulHostilePath("/private/alice/hostile.json"),
    )

    with pytest.raises(ImportReceiptError) as caught:
        _repository(tmp_path).begin(approved, batch_size=25)
    assert secret not in str(caught.value)
    assert caught.value.__context__ is None
    database = tmp_path / "notes-sync.sqlite3"
    if database.exists():
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (0,)


def test_aggregate_receipt_uses_frozen_projection_and_hides_private_values(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.begin(_approved(), batch_size=25)
    repository.transition_session(_APPROVAL_ID, ImportSessionState.RUNNING)
    repository.transition_item(
        _APPROVAL_ID,
        "item-1",
        ImportItemOutcome.FAILED,
        reason_code="version_conflict",
        retryable=False,
        target_note_id="opaque-note-7",
        observed_version=8,
    )
    repository.transition_session(_APPROVAL_ID, ImportSessionState.NEEDS_ATTENTION)

    receipt = repository.aggregate_receipt(_APPROVAL_ID)

    assert receipt.failed == 1
    assert receipt.reason_code == "version_conflict"
    assert receipt.to_diagnostic().failed == 1
    for rendered in (repr(repository), repr(receipt), repr(receipt.to_diagnostic())):
        assert _APPROVAL_ID not in rendered
        assert "opaque-note-7" not in rendered
        assert _PRIVATE_BODY not in rendered
        assert _PRIVATE_SOURCE.as_posix() not in rendered


def test_public_repository_methods_return_frozen_models_not_sqlite_rows(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    session = repository.begin(_approved(), batch_size=25)
    durable = repository.load_session_snapshot(_APPROVAL_ID)
    transitioned = repository.transition_session(
        _APPROVAL_ID, ImportSessionState.RUNNING
    )

    assert not isinstance(session, sqlite3.Row)
    assert not isinstance(durable, sqlite3.Row)
    assert not isinstance(transitioned, sqlite3.Row)
    with pytest.raises(FrozenInstanceError):
        session.batch_size = 50  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        durable.items[0].outcome = ImportItemOutcome.SKIPPED  # type: ignore[misc]
    public_methods = {
        name
        for name, _method in inspect.getmembers(
            NoteImportReceiptRepository, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert {
        "begin",
        "get_session",
        "load_session_snapshot",
        "transition_session",
        "transition_item",
        "transition_effects",
        "reset_retryable_effect",
        "aggregate_receipt",
    } <= public_methods


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-file contract")
def test_receipt_database_and_parent_have_private_modes(tmp_path: Path) -> None:
    database = tmp_path / "private-parent" / "notes-sync.sqlite3"
    database.parent.mkdir(mode=0o700)
    database.parent.chmod(0o700)

    NoteImportReceiptRepository(database).begin(_approved(), batch_size=25)

    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(database.stat().st_mode) == 0o600
