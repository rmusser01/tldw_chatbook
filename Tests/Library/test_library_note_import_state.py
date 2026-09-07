"""Pure workflow-state contracts for reviewed one-time Notes imports."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tldw_chatbook.Library.library_note_import_state import (
    LibraryNoteImportItemSnapshot,
    LibraryNoteImportSnapshot,
    MAX_IMPORT_REVIEW_PAGE_SIZE,
    NoteImportPhase,
    NoteImportReviewEffect,
    add_selected_file,
    apply_import_progress,
    begin_checking,
    begin_importing,
    begin_retry,
    initial_note_import_snapshot,
    request_import_cancellation,
    project_library_note_import_snapshot,
    set_collision_rename,
    set_destination_input,
    revisit_latest_receipt,
    select_folder,
    set_approved_plan,
    set_destination_segments,
    set_item_decision,
    set_review_page,
    set_root_collision_resolution,
    settle_import,
    show_review,
)
from tldw_chatbook.Notes.note_import_execution_models import (
    ImportExecutionProgress,
    ImportExecutionReceipt,
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

_APPROVAL_ID = "00000000-0000-4000-8000-000000000003"
_PRIVATE_ROOT = Path("/private/alice/Tax records")
_SECRET_CONTENT = "bank account 9999 and medical note"
_SECRET_KEYWORD = "confidential-keyword"


def _bounds() -> ImportBounds:
    return ImportBounds(
        max_files=100,
        max_file_bytes=1_000_000,
        max_total_bytes=5_000_000,
        max_depth=8,
    )


def _item(
    item_number: int = 1,
    *,
    classification: ImportClassification = ImportClassification.NEW,
    selected_action: ImportAction | None = None,
    match_kind: ImportMatchKind | None = None,
    replace_content: bool = False,
    add_membership: bool = True,
) -> ImportPreviewItem:
    if classification is ImportClassification.NEW:
        default_action = ImportAction.CREATE_NEW
        allowed_actions = (ImportAction.SKIP, ImportAction.CREATE_NEW)
        match = None
        action = selected_action or ImportAction.CREATE_NEW
    else:
        default_action = (
            ImportAction.CREATE_NEW
            if classification
            in {
                ImportClassification.CHANGED_REPEAT,
                ImportClassification.UNCERTAIN_MATCH,
            }
            else ImportAction.SKIP
        )
        if match_kind is None:
            match_kind = (
                ImportMatchKind.UNCERTAIN
                if classification is ImportClassification.UNCERTAIN_MATCH
                else ImportMatchKind.EXACT
            )
        allowed = [ImportAction.SKIP, ImportAction.CREATE_NEW]
        if match_kind in {ImportMatchKind.EXACT, ImportMatchKind.USER_CONFIRMED}:
            allowed.append(ImportAction.UPDATE_EXISTING)
        allowed_actions = tuple(allowed)
        match = ImportMatch(
            kind=match_kind,
            note_id=f"note-{item_number}",
            note_version=7,
        )
        action = selected_action or default_action
    if action is ImportAction.SKIP:
        replace_content = False
        add_membership = False
    elif action is ImportAction.CREATE_NEW:
        replace_content = False
        add_membership = True
    return ImportPreviewItem(
        item_id=f"item-{item_number}",
        source=ImportSource(
            kind=ImportSourceKind.SELECTED_FILE,
            display_path=f"record-{item_number}.md",
            source_path=_PRIVATE_ROOT / f"record-{item_number}.md",
        ),
        payloads=(
            ParsedNotePayload(
                title=f"Private {item_number}",
                content=_SECRET_CONTENT,
                keywords=(_SECRET_KEYWORD,),
            ),
        ),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Imported",),
            ),
        ),
        classification=classification,
        reason="Ready after review.",
        default_action=default_action,
        selected_action=action,
        allowed_actions=allowed_actions,
        match=match,
        replace_content=replace_content,
        add_membership=add_membership,
    )


def _plan(
    *items: ImportPreviewItem,
    collision: RootCollisionState | None = None,
) -> NoteImportPlan:
    return NoteImportPlan(
        bounds=_bounds(),
        items=items or (_item(),),
        proposed_folder_paths=(("Imported",),),
        root_collision=collision,
    )


def _file_review(
    plan: NoteImportPlan | None = None,
    *,
    page_size: int = 20,
):
    state = add_selected_file(
        initial_note_import_snapshot(page_size=page_size),
        _PRIVATE_ROOT / "record.md",
    )
    state = set_destination_segments(state, ("Imported",))
    state = begin_checking(state)
    return show_review(state, plan or _plan())


def _progress(**overrides: object) -> ImportExecutionProgress:
    values: dict[str, object] = {
        "state": ImportSessionState.RUNNING,
        "total": 4,
        "completed": 1,
        "imported": 1,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "retryable": 0,
    }
    values.update(overrides)
    return ImportExecutionProgress(**values)  # type: ignore[arg-type]


def _receipt(**overrides: object) -> ImportExecutionReceipt:
    values: dict[str, object] = {
        "approval_id": _APPROVAL_ID,
        "state": ImportSessionState.NEEDS_ATTENTION,
        "total": 4,
        "completed": 3,
        "imported": 1,
        "updated": 1,
        "skipped": 0,
        "failed": 1,
        "retryable": 1,
        "reason_code": "target_conflict",
        "_raw_errors": (f"failed at {_PRIVATE_ROOT}",),
    }
    values.update(overrides)
    return ImportExecutionReceipt(**values)  # type: ignore[arg-type]


def test_snapshot_is_frozen_and_repr_and_diagnostic_are_redacted() -> None:
    state = _file_review()

    with pytest.raises(FrozenInstanceError):
        state.phase = NoteImportPhase.SELECT  # type: ignore[misc]

    rendered = repr(state)
    diagnostic = repr(state.to_diagnostic())
    for secret in (
        str(_PRIVATE_ROOT),
        "record.md",
        _SECRET_CONTENT,
        _SECRET_KEYWORD,
        "Ready after review.",
    ):
        assert secret not in rendered
        assert secret not in diagnostic
    assert state.to_diagnostic().selected_count == 1
    assert state.to_diagnostic().review_item_count == 1


def test_phase_flow_requires_selection_and_file_destination() -> None:
    state = initial_note_import_snapshot()
    assert state.phase is NoteImportPhase.SELECT
    assert state.can_check is False

    state = add_selected_file(state, _PRIVATE_ROOT / "a.md")
    state = add_selected_file(state, _PRIVATE_ROOT / "b.md")
    assert state.selected_count == 2
    assert state.can_add_file is True
    assert state.requires_destination is True
    assert state.phase is NoteImportPhase.DESTINATION
    assert state.can_check is False

    state = set_destination_segments(state, ("Imported", "Archive"))
    assert state.can_check is True
    assert begin_checking(state).phase is NoteImportPhase.CHECKING


def test_file_selection_accumulates_without_duplicates_and_folder_is_exclusive() -> (
    None
):
    first = _PRIVATE_ROOT / "a.md"
    state = add_selected_file(initial_note_import_snapshot(), first)
    assert add_selected_file(state, first).selected_count == 1

    with pytest.raises(ValueError, match="folder"):
        select_folder(state, _PRIVATE_ROOT)

    folder_state = select_folder(initial_note_import_snapshot(), _PRIVATE_ROOT)
    assert folder_state.selected_count == 1
    assert folder_state.selection_is_folder is True
    assert folder_state.requires_destination is False
    assert folder_state.can_add_file is False
    assert folder_state.can_check is True
    with pytest.raises(ValueError, match="file"):
        add_selected_file(folder_state, first)


def test_destination_segments_are_required_safe_and_invalidate_review_authority() -> (
    None
):
    state = add_selected_file(initial_note_import_snapshot(), _PRIVATE_ROOT / "a.md")
    for segments in ((), ("",), ("..",), ("bad/name",)):
        if not segments:
            assert set_destination_segments(state, segments).can_check is False
        else:
            with pytest.raises(ValueError, match="destination"):
                set_destination_segments(state, segments)

    review = _file_review()
    approved = approve_note_import_plan(review.plan, approval_id=_APPROVAL_ID)
    approved_state = set_approved_plan(review, approved)
    changed = set_destination_segments(approved_state, ("Elsewhere",))
    assert changed.approved_plan is None
    assert changed.phase is NoteImportPhase.DESTINATION


def test_destination_input_retains_raw_text_and_exposes_inline_validation() -> None:
    state = add_selected_file(initial_note_import_snapshot(), _PRIVATE_ROOT / "a.md")

    valid = set_destination_input(state, " Inbox / Archive ")
    assert valid.destination_input == " Inbox / Archive "
    assert valid.destination_segments == ()
    assert valid.destination_error == (
        "Remove leading or trailing spaces from each folder name."
    )
    assert valid.can_check is False
    assert project_library_note_import_snapshot(valid).destination == (
        " Inbox / Archive "
    )

    valid = set_destination_input(state, "Inbox/Archive")
    assert valid.destination_input == "Inbox/Archive"
    assert valid.destination_segments == ("Inbox", "Archive")
    assert valid.destination_error == ""
    assert valid.can_check is True

    invalid = set_destination_input(valid, "Inbox//Archive")
    assert invalid.destination_input == "Inbox//Archive"
    assert invalid.destination_segments == ()
    assert invalid.destination_error == "Remove the empty folder between separators."
    assert invalid.can_check is False

    too_long = set_destination_input(state, "x" * 256)
    assert "255" in too_long.destination_error
    assert too_long.can_check is False

    nfkc_separator = set_destination_input(state, "Inbox／Archive")
    assert "valid folder name" in nfkc_separator.destination_error
    assert nfkc_separator.can_check is False


def test_review_page_is_bounded_and_clamps_after_plan_changes() -> None:
    plan = _plan(*(_item(number) for number in range(1, 58)))
    state = _file_review(plan, page_size=10_000)

    assert state.page.page_size == MAX_IMPORT_REVIEW_PAGE_SIZE
    assert len(state.page.items) == MAX_IMPORT_REVIEW_PAGE_SIZE
    assert state.page.total_items == 57
    assert state.page.page_count == 3
    assert state.page.has_previous is False
    assert state.page.has_next is True

    last = set_review_page(state, 999)
    assert last.page.page_number == 3
    assert len(last.page.items) == 7


def test_unresolved_collision_blocks_approval_until_an_explicit_resolution() -> None:
    collision = RootCollisionState(proposed_label="Imported", collides=True)
    state = _file_review(_plan(collision=collision))

    assert state.can_approve is False
    assert state.approval_blocker == "Choose how to handle the folder name collision."
    projected = project_library_note_import_snapshot(state)
    assert projected.collision_rename_input == "Imported"
    assert "already exists" in projected.collision_rename_error

    resolved = set_root_collision_resolution(
        state,
        RootCollisionChoice.UNIQUE_SIBLING,
        resolved_label="Imported 2",
    )
    assert resolved.can_approve is True


def test_update_only_progress_names_the_updated_count() -> None:
    review = _file_review()
    state = begin_importing(
        set_approved_plan(review, approve_note_import_plan(review.plan))
    )
    state = apply_import_progress(
        state,
        _progress(completed=1, imported=0, updated=1),
    )

    assert "1 updated" in project_library_note_import_snapshot(state).progress_detail


def test_uncertain_create_new_is_approvable_but_update_requires_confirmation() -> None:
    uncertain = _item(classification=ImportClassification.UNCERTAIN_MATCH)
    state = _file_review(_plan(uncertain))

    assert state.plan.items[0].selected_action is ImportAction.CREATE_NEW
    assert state.can_approve is True
    with pytest.raises(ValueError, match="confirmed"):
        set_item_decision(
            state,
            "item-1",
            action=ImportAction.UPDATE_EXISTING,
            replace_content=True,
            add_membership=False,
        )

    confirmed_plan = _plan(
        _item(
            classification=ImportClassification.UNCERTAIN_MATCH,
            match_kind=ImportMatchKind.USER_CONFIRMED,
        )
    )
    confirmed = show_review(begin_checking(state), confirmed_plan)
    assert confirmed.can_approve is True


def test_collision_rename_input_invalidates_rename_authority() -> None:
    collision = RootCollisionState(
        proposed_label="Imported",
        collides=True,
        choice=RootCollisionChoice.RENAMED_ROOT,
        resolved_label="Fresh",
    )
    state = _file_review(_plan(collision=collision))

    edited = set_collision_rename(
        state,
        "Imported",
        error="That folder name already exists. Enter a different name.",
    )

    assert edited.collision_rename_input == "Imported"
    assert edited.collision_rename_error.startswith("That folder name")
    assert edited.plan.root_collision.choice is None
    assert edited.can_approve is False


def test_update_existing_requires_an_explicit_content_and_membership_decision() -> None:
    repeat = _item(
        classification=ImportClassification.CHANGED_REPEAT,
        selected_action=ImportAction.CREATE_NEW,
    )
    state = _file_review(_plan(repeat))

    with pytest.raises(ValueError, match="replace content or add membership"):
        set_item_decision(
            state,
            "item-1",
            action=ImportAction.UPDATE_EXISTING,
            replace_content=False,
            add_membership=False,
        )

    membership_only = set_item_decision(
        state,
        "item-1",
        action=ImportAction.UPDATE_EXISTING,
        replace_content=False,
        add_membership=True,
    )
    item = membership_only.plan.items[0]
    assert item.selected_action is ImportAction.UPDATE_EXISTING
    assert item.replace_content is False
    assert item.add_membership is True
    assert membership_only.decision_item_ids == frozenset({"item-1"})


def test_each_review_mutation_invalidates_exact_approval_authority() -> None:
    repeat = _item(classification=ImportClassification.CHANGED_REPEAT)
    state = _file_review(_plan(repeat))
    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    state = set_approved_plan(state, approved)
    assert state.approved_plan is approved

    changed = set_item_decision(
        state,
        "item-1",
        action=ImportAction.SKIP,
        replace_content=False,
        add_membership=False,
    )
    assert changed.approved_plan is None
    assert changed.revision == state.revision + 1

    with pytest.raises(ValueError, match="exact current plan"):
        set_approved_plan(changed, approved)


def test_importing_requires_exact_approval_and_progress_cannot_regress() -> None:
    state = _file_review()
    with pytest.raises(ValueError, match="approved"):
        begin_importing(state)

    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    importing = begin_importing(set_approved_plan(state, approved))
    assert importing.phase is NoteImportPhase.IMPORTING
    assert importing.progress.state is ImportSessionState.PENDING

    advanced = apply_import_progress(importing, _progress())
    with pytest.raises(ValueError, match="regress"):
        apply_import_progress(
            advanced,
            _progress(
                completed=0,
                imported=0,
            ),
        )
    with pytest.raises(ValueError, match="total"):
        apply_import_progress(advanced, _progress(total=5))


def test_cancellation_is_cooperative_and_does_not_fabricate_a_receipt() -> None:
    state = _file_review()
    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    importing = begin_importing(set_approved_plan(state, approved))
    cancelling = request_import_cancellation(importing)

    assert cancelling.cancel_requested is True
    assert cancelling.phase is NoteImportPhase.IMPORTING
    assert cancelling.receipt is None
    assert cancelling.can_cancel is False
    assert request_import_cancellation(cancelling) is cancelling


def test_partial_cancelled_receipt_reports_truth_and_is_not_retryable_without_failures() -> (
    None
):
    state = _file_review()
    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    importing = begin_importing(set_approved_plan(state, approved))
    receipt = _receipt(
        state=ImportSessionState.CANCELLED,
        completed=2,
        imported=1,
        updated=0,
        skipped=1,
        failed=0,
        retryable=0,
        reason_code="cancelled",
    )
    settled = settle_import(importing, receipt)

    assert settled.phase is NoteImportPhase.RECEIPT
    assert settled.is_partial is True
    assert settled.can_retry is True
    assert settled.latest_receipt is receipt


def test_receipt_must_belong_to_current_approval() -> None:
    state = _file_review()
    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    importing = begin_importing(set_approved_plan(state, approved))
    foreign = _receipt(approval_id="00000000-0000-4000-8000-000000000004")

    with pytest.raises(ValueError, match="approval"):
        settle_import(importing, foreign)


def test_retry_is_gated_by_retryable_failures_and_retains_exact_authority() -> None:
    state = _file_review()
    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    settled = settle_import(
        begin_importing(set_approved_plan(state, approved)),
        _receipt(),
    )

    retrying = begin_retry(settled)
    assert retrying.phase is NoteImportPhase.IMPORTING
    assert retrying.approved_plan is approved
    assert retrying.cancel_requested is False
    assert retrying.receipt is settled.receipt

    no_retry = replace(
        settled,
        receipt=_receipt(
            state=ImportSessionState.COMPLETED,
            completed=4,
            imported=2,
            updated=1,
            skipped=1,
            failed=0,
            retryable=0,
            reason_code=None,
        ),
    )
    with pytest.raises(ValueError, match="retryable"):
        begin_retry(no_retry)


def test_latest_receipt_can_be_revisited_during_the_same_session() -> None:
    state = _file_review()
    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    receipt_state = settle_import(
        begin_importing(set_approved_plan(state, approved)),
        _receipt(),
    )

    selecting = add_selected_file(
        initial_note_import_snapshot(
            latest_receipt=receipt_state.latest_receipt,
        ),
        _PRIVATE_ROOT / "next.md",
    )
    assert selecting.can_revisit_receipt is False
    revisited = revisit_latest_receipt(
        initial_note_import_snapshot(
            latest_receipt=receipt_state.latest_receipt,
        )
    )
    assert revisited.phase is NoteImportPhase.RECEIPT
    assert revisited.receipt is receipt_state.latest_receipt


def test_presentation_projection_is_frozen_path_safe_and_redacted() -> None:
    workflow = _file_review()

    projected = project_library_note_import_snapshot(workflow)

    assert isinstance(projected, LibraryNoteImportSnapshot)
    assert isinstance(projected.preview_items[0], LibraryNoteImportItemSnapshot)
    assert projected.selected_names == ("record.md",)
    assert projected.preview_items[0].name == "record-1.md"
    assert projected.phase == "review"
    assert projected.can_import is True
    with pytest.raises(FrozenInstanceError):
        projected.phase = "select"  # type: ignore[misc]
    rendered = repr(projected)
    for secret in (
        str(_PRIVATE_ROOT),
        "record.md",
        "record-1.md",
        _SECRET_CONTENT,
        _SECRET_KEYWORD,
    ):
        assert secret not in rendered


def test_review_projection_exposes_relative_source_membership_and_bounded_effects() -> (
    None
):
    item = replace(
        _item(
            classification=ImportClassification.CHANGED_REPEAT,
            selected_action=ImportAction.UPDATE_EXISTING,
            replace_content=True,
            add_membership=True,
        ),
        source=replace(_item().source, display_path="alpha/record.md"),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Imported", "Alpha"),
            ),
        ),
    )
    state = _file_review(_plan(item))
    effect = NoteImportReviewEffect(
        item_id=item.item_id,
        target_title="Existing title",
        target_version=7,
        content_diff="--- Existing\n+++ Imported\n-Old\n+New",
    )
    state = replace(state, review_effects=(effect,))

    projected = project_library_note_import_snapshot(state).preview_items[0]

    assert projected.name == "alpha/record.md"
    assert projected.target_label == "Existing note: Existing title (version 7)."
    assert projected.membership_summary == "Folder placement: add Imported / Alpha."
    assert projected.effect_summary == "Content: replace existing content."
    assert "-Old" in projected.content_diff
    assert "alpha/record.md" not in repr(projected)
    assert "Existing title" not in repr(projected)


@pytest.mark.parametrize(
    ("receipt_state", "completed", "failed", "expected_status", "detail_fragment"),
    (
        (ImportSessionState.COMPLETED, 4, 0, "Import completed.", "settled"),
        (
            ImportSessionState.CANCELLED,
            2,
            0,
            "Import cancelled after the current item.",
            "not rolled back",
        ),
        (
            ImportSessionState.NEEDS_ATTENTION,
            4,
            1,
            "Import needs attention.",
            "failed",
        ),
    ),
)
def test_receipt_projection_distinguishes_durable_session_state(
    receipt_state: ImportSessionState,
    completed: int,
    failed: int,
    expected_status: str,
    detail_fragment: str,
) -> None:
    state = _file_review()
    approved = approve_note_import_plan(state.plan, approval_id=_APPROVAL_ID)
    receipt = _receipt(
        state=receipt_state,
        completed=completed,
        imported=max(completed - failed, 0),
        updated=0,
        skipped=0,
        failed=failed,
        retryable=failed,
        reason_code="target_conflict"
        if failed
        else "cancelled"
        if receipt_state is ImportSessionState.CANCELLED
        else None,
    )
    settled = settle_import(
        begin_importing(set_approved_plan(state, approved)), receipt
    )

    projected = project_library_note_import_snapshot(settled)

    assert projected.status_line == expected_status
    assert detail_fragment in projected.receipt_detail.casefold()
