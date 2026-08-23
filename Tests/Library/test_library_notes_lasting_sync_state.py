"""Pure presentation contracts for inert lasting Notes sync surfaces."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from tldw_chatbook.Library import library_notes_lasting_sync_state as lasting_state
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncReviewRow,
    LastingSyncSetup,
    build_reconciliation_review,
    initial_lasting_sync_snapshot,
    set_setup_value,
)
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictComparison,
    ConflictSelection,
    NotesSyncConflictChoice,
)
from tldw_chatbook.Notes.notes_sync_models import NotesSyncAction, NotesSyncActionKind
from tldw_chatbook.Notes.notes_sync_reconciler import (
    DeletionGroup,
    ManagedPlacementEffect,
    ManagedPlacementEffectKind,
    ReconciliationAttention,
    ReconciliationAttentionKind,
    ReconciliationPlan,
    ReconciliationSkip,
    ReconciliationSkipKind,
)


TOKEN = "a" * 64


def test_setup_defaults_to_bidirectional_and_requires_name_folder_and_local_scope() -> (
    None
):
    snapshot = initial_lasting_sync_snapshot(lasting_available=True)

    assert snapshot.phase == "choose"
    assert snapshot.setup.direction == "bidirectional"
    assert snapshot.setup.destination == "local"
    assert snapshot.setup.can_check is False

    snapshot = replace(snapshot, phase="configure")
    snapshot = set_setup_value(snapshot, "display_name", "Research [2026]")
    snapshot = set_setup_value(snapshot, "folder", "/tmp/research")
    snapshot = set_setup_value(snapshot, "note_scope_id", "local-notes")

    assert snapshot.setup.can_check is True
    assert snapshot.setup.validation_message == ""


def test_server_destination_is_visibly_unavailable_and_never_checkable() -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(lasting_available=True), phase="configure"
    )
    snapshot = set_setup_value(snapshot, "destination", "server")

    assert snapshot.setup.can_check is False
    assert snapshot.setup.server_available is False
    assert snapshot.setup.server_disabled_reason == (
        "Unavailable - server sync-folder capability not installed"
    )
    assert "local" in snapshot.setup.validation_message.casefold()


def test_setup_contract_rejects_global_winner_and_automatic_interval_policy() -> None:
    names = {field.name for field in fields(LastingSyncSetup)}

    assert "conflict_winner" not in names
    assert "auto_sync_minutes" not in names
    assert "auto_sync" not in names


def test_review_projects_bounded_safe_attention_skip_and_managed_move_rows() -> None:
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=(
            NotesSyncAction("act-1", NotesSyncActionKind.UPDATE_NOTE, "bind-1"),
        ),
        attention=(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "both_sides_changed",
                "bind-2",
            ),
        ),
        skips=(ReconciliationSkip(ReconciliationSkipKind.OFFLINE, "root_offline"),),
        managed_placement_effects=(
            ManagedPlacementEffect(ManagedPlacementEffectKind.FILE_MOVE, "bind-3"),
        ),
        deletion_groups=(),
        page_size=2,
    )

    review = build_reconciliation_review(plan, page=1)

    assert (review.safe_count, review.attention_count, review.skip_count) == (1, 1, 1)
    assert review.observation_token == TOKEN
    assert review.page == 1
    assert review.page_count == 2
    assert any("Keep file" in row.choices for row in review.rows)
    assert any("Keep note" in row.choices for row in review.rows)
    assert any("Keep both" in row.choices for row in review.rows)
    assert any("Skip for now" in row.choices for row in review.rows)
    all_rows = review.rows + build_reconciliation_review(plan, page=2).rows
    assert any("filesystem move" in row.effect.casefold() for row in all_rows)
    assert all("/" not in row.item_id for row in review.rows)


def test_review_page_is_bounded() -> None:
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=tuple(
            NotesSyncAction(
                f"act-{index}", NotesSyncActionKind.UPDATE_NOTE, f"b-{index}"
            )
            for index in range(5)
        ),
        attention=(),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
        page_size=2,
    )

    assert [row.item_id for row in build_reconciliation_review(plan, page=99).rows] == [
        "b-4"
    ]


@pytest.mark.parametrize(
    ("reason_code", "effect"),
    [
        ("both_sides_changed", "Both file and note changed"),
        (
            "out_of_direction_change",
            "This change is outside the root direction",
        ),
    ],
)
def test_exact_content_conflicts_are_selectable(reason_code: str, effect: str) -> None:
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=(),
        attention=(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                reason_code,
                "bind-1",
            ),
        ),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )

    row = build_reconciliation_review(plan).rows[0]

    assert row.effect == effect
    assert row.choices == (
        "Keep file",
        "Keep note",
        "Keep both",
        "Skip for now",
    )


@pytest.mark.parametrize(
    "reason_code",
    [
        "duplicate_authority",
        "out_of_direction_create",
        "out_of_direction_move",
        "out_of_direction_representation",
        "ambiguous_identity",
        "note_implied_filesystem_move",
    ],
)
def test_other_conflicts_are_not_selectable(reason_code: str) -> None:
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=(),
        attention=(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                reason_code,
                "bind-1",
            ),
        ),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )

    row = build_reconciliation_review(plan).rows[0]

    assert row.effect == "Both file and note changed"
    assert row.choices == ("Keep file", "Keep note", "Keep both")


def test_managed_placement_conflict_is_not_selectable() -> None:
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=(),
        attention=(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "both_sides_changed",
                "bind-1",
            ),
        ),
        skips=(),
        managed_placement_effects=(
            ManagedPlacementEffect(
                ManagedPlacementEffectKind.FILE_MOVE,
                "bind-1",
            ),
        ),
        deletion_groups=(),
    )

    conflict_row = build_reconciliation_review(plan).rows[0]

    assert conflict_row.effect == "Both file and note changed"
    assert conflict_row.choices == ("Keep file", "Keep note", "Keep both")


def test_deletion_review_has_bounded_explicit_choices_and_no_global_winner() -> None:
    deletion = ReconciliationAttention(
        ReconciliationAttentionKind.DELETION_REVIEW,
        "file_missing",
        "bind-1",
    )
    plan = ReconciliationPlan(
        root_id="root:remote.v1",
        observation_token=TOKEN,
        safe_actions=(),
        attention=(),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(DeletionGroup((deletion,)),),
    )

    review = build_reconciliation_review(plan)

    assert review.attention_count == 1
    assert review.rows[0].choices == (
        "Restore missing sides",
        "Delete/archive counterparts",
        "Disconnect items",
    )
    assert "winner" not in repr(review).casefold()


def test_ordinary_root_rows_and_repr_do_not_carry_absolute_paths_or_content() -> None:
    snapshot = replace(
        initial_lasting_sync_snapshot(),
        setup=LastingSyncSetup(folder="/Users/private/Research"),
        review=replace(
            initial_lasting_sync_snapshot().review,
            observation_token="f" * 64,
        ),
    )

    assert "/Users/private" not in repr(snapshot.review)
    assert "/Users/private" not in repr(snapshot)
    assert "f" * 64 not in repr(snapshot)
    assert "note body" not in repr(snapshot.review)


def test_root_row_rejects_path_like_opaque_identifier() -> None:
    from tldw_chatbook.Library.library_notes_lasting_sync_state import (
        LastingSyncRootRow,
    )

    with pytest.raises(ValueError, match="root_id"):
        LastingSyncRootRow(
            "private/root",
            "Research",
            "paused",
            "resume_sync",
            "Ⅱ Paused",
            "Resume",
        )


def test_snapshot_rejects_unknown_phase_and_untyped_root_collection() -> None:
    with pytest.raises(ValueError, match="phase"):
        replace(initial_lasting_sync_snapshot(), phase="automatic")
    with pytest.raises(TypeError, match="roots"):
        replace(initial_lasting_sync_snapshot(), roots=[object()])


def test_review_rejects_mutable_choices_boolean_counts_and_unbounded_selection_cache() -> (
    None
):
    from tldw_chatbook.Library.library_notes_lasting_sync_state import (
        LastingSyncReview,
        LastingSyncReviewRow,
    )

    with pytest.raises(TypeError, match="choices"):
        LastingSyncReviewRow("bind-1", "attention", "Changed", ["Keep file"])
    with pytest.raises(ValueError, match="choices"):
        LastingSyncReviewRow("bind-1", "attention", "Changed", ("",))
    with pytest.raises(ValueError, match="choices"):
        LastingSyncReviewRow(
            "bind-1", "attention", "Changed", ("Keep file\nwithout review",)
        )
    with pytest.raises(ValueError, match="counts"):
        LastingSyncReview(safe_count=True)
    assert "selected_action_ids" not in {
        field.name for field in fields(LastingSyncReview)
    }


def test_setup_rejects_unknown_fields_instead_of_growing_policy_surface() -> None:
    with pytest.raises(ValueError, match="unknown setup field"):
        set_setup_value(initial_lasting_sync_snapshot(), "auto_sync_minutes", "5")


def test_conflict_review_projects_typed_selection_and_skip_only_apply_blocker() -> None:
    plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=(),
        attention=(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT,
                "both_sides_changed",
                "bind-1",
            ),
        ),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )

    unselected = build_reconciliation_review(plan)
    skipped = build_reconciliation_review(
        plan,
        selections=(ConflictSelection("bind-1", NotesSyncConflictChoice.SKIP),),
    )
    mutating = build_reconciliation_review(
        plan,
        selections=(ConflictSelection("bind-1", NotesSyncConflictChoice.KEEP_FILE),),
    )

    assert unselected.rows[0].conflict_eligible is True
    assert unselected.rows[0].selected_choice is None
    assert unselected.can_apply is False
    assert unselected.apply_blocker.value == "nothing_selected"
    assert skipped.rows[0].selected_choice is NotesSyncConflictChoice.SKIP
    assert skipped.rows[0].selected_label == "Selected: Skip for now"
    assert skipped.can_apply is False
    assert skipped.apply_blocker.value == "nothing_selected"
    assert mutating.rows[0].selected_label == "Selected: Keep file"
    assert mutating.can_apply is True
    assert mutating.apply_blocker.value == "none"


def test_conflict_row_and_snapshot_bound_review_labels_and_focus_request() -> None:
    row = LastingSyncReviewRow(
        "bind-1",
        "attention",
        "Both file and note changed",
        conflict_eligible=True,
        conflict_title="Release [red]note[/red]",
        conflict_relative_path="notes/release.md",
    )
    snapshot = replace(
        initial_lasting_sync_snapshot(),
        history_available=True,
        conflict_focus_binding_id="bind-1",
    )

    assert row.conflict_title == "Release [red]note[/red]"
    assert row.conflict_relative_path == "notes/release.md"
    assert snapshot.history_available is True
    assert snapshot.conflict_focus_binding_id == "bind-1"
    with pytest.raises(ValueError, match="conflict_title"):
        replace(row, conflict_title="x" * 161)
    with pytest.raises(ValueError, match="relative path"):
        replace(row, conflict_relative_path="../private.md")


def test_review_apply_blockers_derive_from_typed_plan_facts() -> None:
    base = dict(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=(
            NotesSyncAction("act-1", NotesSyncActionKind.UPDATE_NOTE, "bind-safe"),
        ),
        attention=(),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
    )
    safe = build_reconciliation_review(ReconciliationPlan(**base))
    deletion = build_reconciliation_review(
        ReconciliationPlan(
            **{
                **base,
                "attention": (
                    ReconciliationAttention(
                        ReconciliationAttentionKind.DELETION_REVIEW,
                        "file_missing",
                        "bind-1",
                    ),
                ),
            }
        )
    )
    capability = build_reconciliation_review(
        ReconciliationPlan(
            **{
                **base,
                "skips": (
                    ReconciliationSkip(
                        ReconciliationSkipKind.CAPABILITY,
                        "write_capability_missing",
                    ),
                ),
            }
        )
    )

    assert safe.can_apply is True
    assert safe.apply_blocker.value == "none"
    assert deletion.can_apply is False
    assert deletion.apply_blocker.value == "deletion_review"
    assert capability.can_apply is False
    assert capability.apply_blocker.value == "root_or_capability"


def test_snapshot_conflict_projection_is_immutable_bounded_and_typed() -> None:
    assert hasattr(lasting_state, "LastingSyncReceiptRow")
    assert hasattr(lasting_state, "LastingSyncHistory")
    receipt_type = lasting_state.LastingSyncReceiptRow
    history_row_type = lasting_state.LastingSyncHistoryRow
    history_type = lasting_state.LastingSyncHistory
    comparison = ConflictComparison(
        binding_id="bind-1",
        note_title="Note",
        relative_path="note.md",
        note_version=1,
        note_updated_at=None,
        file_modified_ns=2,
        note_character_count=4,
        note_line_count=1,
        file_character_count=4,
        file_line_count=1,
        diff="",
        input_elided=False,
        output_elided=False,
    )
    receipt = receipt_type(
        "operation-1",
        "Note · note.md",
        NotesSyncConflictChoice.KEEP_FILE,
        "completed",
        True,
    )
    history_row = history_row_type(
        "operation-1",
        "Note · note.md",
        NotesSyncConflictChoice.KEEP_FILE,
        "completed",
        "2026-08-22T12:00:00+00:00",
        "2026-08-22T12:00:00+00:00",
        True,
    )
    snapshot = replace(
        initial_lasting_sync_snapshot(),
        comparison=comparison,
        receipts=(receipt,),
        history=history_type("root-1", (history_row,), 1, False),
    )

    assert snapshot.comparison is comparison
    assert snapshot.receipts == (receipt,)
    assert snapshot.history.rows == (history_row,)
    with pytest.raises(ValueError, match="bounded"):
        replace(snapshot, receipts=(receipt,) * 101)
    with pytest.raises(TypeError, match="history"):
        replace(snapshot, history=object())


def test_history_page_rejects_bool_and_offsets_outside_sqlite_integer_range() -> None:
    history_type = lasting_state.LastingSyncHistory
    largest_page = ((2**63 - 1) // 100) + 1

    assert history_type(page=largest_page).page == largest_page
    for invalid in (True, largest_page + 1, 10**100):
        with pytest.raises(ValueError, match="history page|SQLite"):
            history_type(page=invalid)


def test_receipt_unavailable_projection_is_an_exact_boolean() -> None:
    snapshot = initial_lasting_sync_snapshot()

    assert snapshot.receipts_unavailable is False
    assert replace(snapshot, receipts_unavailable=True).receipts_unavailable is True
    with pytest.raises(TypeError, match="receipts_unavailable"):
        replace(snapshot, receipts_unavailable=1)


def test_conflict_review_row_repr_does_not_expose_title_or_relative_path() -> None:
    row = LastingSyncReviewRow(
        "bind-1",
        "attention",
        "Both file and note changed",
        ("Keep file",),
        conflict_eligible=True,
        conflict_title="Private release title",
        conflict_relative_path="private/releases/note.md",
    )

    projected = repr(row)

    assert "Private release title" not in projected
    assert "private/releases/note.md" not in projected
    assert projected == "LastingSyncReviewRow(<private>)"


@pytest.mark.parametrize("field", ("conflict_title", "conflict_relative_path"))
@pytest.mark.parametrize("invalid", (0, False))
def test_conflict_review_labels_reject_falsey_non_strings(
    field: str, invalid: object
) -> None:
    values = {"conflict_title": "", "conflict_relative_path": ""}
    values[field] = invalid

    with pytest.raises((TypeError, ValueError)):
        LastingSyncReviewRow(
            "bind-1",
            "attention",
            "Both changed",
            conflict_eligible=True,
            **values,
        )


@pytest.mark.parametrize(
    "factory", (lasting_state.LastingSyncReview, lasting_state.LastingSyncHistory)
)
@pytest.mark.parametrize("invalid", (0, False))
def test_review_and_history_roots_reject_falsey_non_strings(
    factory: object, invalid: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(root_id=invalid)


@pytest.mark.parametrize("invalid", (0, False))
def test_review_token_rejects_falsey_non_strings(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        lasting_state.LastingSyncReview(observation_token=invalid)


def test_empty_optional_review_and_history_identifiers_remain_valid() -> None:
    LastingSyncReviewRow("bind-1", "safe", "No change")
    lasting_state.LastingSyncReview()
    lasting_state.LastingSyncHistory()
