"""Pure presentation contracts for inert lasting Notes sync surfaces."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncSetup,
    build_reconciliation_review,
    initial_lasting_sync_snapshot,
    set_setup_value,
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
