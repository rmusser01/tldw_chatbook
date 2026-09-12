from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncDirection,
    NotesSyncSerializationProfile,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    BindingObservation,
    DeletionGroup,
    ManagedPlacementEffect,
    ReconciliationPlan,
    ReconciliationSkip,
    ReconciliationAttentionKind,
    ReconciliationInput,
    ReconciliationSkipKind,
    assert_review_current,
    assert_review_token,
    plan_reconciliation,
)


BASE_FILE = "1" * 64
BASE_NOTE = "2" * 64
NEW_FILE = "3" * 64
NEW_NOTE = "4" * 64
IDENTITY = "5" * 64
MOVED_IDENTITY = "6" * 64


def _binding(
    *,
    file_digest: str | None = BASE_FILE,
    note_digest: str | None = BASE_NOTE,
    file_identity: str | None = IDENTITY,
    baseline_file: str = BASE_FILE,
    baseline_note: str = BASE_NOTE,
    baseline_identity: str = IDENTITY,
    relative_path: str = "note.md",
    baseline_path: str = "note.md",
    note_implied_path: str | None = None,
    duplicate_authority: bool = False,
    baseline_serialization: NotesSyncSerializationProfile | None = None,
    serialization: NotesSyncSerializationProfile | None = None,
) -> BindingObservation:
    return BindingObservation(
        binding_id="binding-1",
        baseline_file_digest=baseline_file,
        baseline_note_digest=baseline_note,
        baseline_identity_digest=baseline_identity,
        baseline_relative_path=baseline_path,
        file_digest=file_digest,
        note_digest=note_digest,
        file_identity_digest=file_identity,
        relative_path=relative_path,
        note_scope_id="scope-1",
        note_id="note-1",
        note_version=7,
        note_implied_relative_path=note_implied_path,
        duplicate_authority=duplicate_authority,
        baseline_serialization=baseline_serialization,
        serialization=serialization,
    )


def _plan(
    direction: NotesSyncDirection,
    binding: BindingObservation,
    **kwargs: object,
):
    request = ReconciliationInput(
        root_id="root-1",
        direction=direction,
        bindings=(binding,),
        observation_generation=7,
        expected_generation=7,
        **kwargs,
    )
    return plan_reconciliation(request)


@pytest.mark.parametrize(
    ("direction", "file_digest", "note_digest", "expected"),
    [
        (
            NotesSyncDirection.BIDIRECTIONAL,
            BASE_FILE,
            BASE_NOTE,
            NotesSyncActionKind.NO_CHANGE,
        ),
        (
            NotesSyncDirection.BIDIRECTIONAL,
            NEW_FILE,
            BASE_NOTE,
            NotesSyncActionKind.UPDATE_NOTE,
        ),
        (
            NotesSyncDirection.BIDIRECTIONAL,
            BASE_FILE,
            NEW_NOTE,
            NotesSyncActionKind.UPDATE_FILE,
        ),
        (
            NotesSyncDirection.FOLDER_TO_NOTES,
            NEW_FILE,
            BASE_NOTE,
            NotesSyncActionKind.UPDATE_NOTE,
        ),
        (
            NotesSyncDirection.NOTES_TO_FOLDER,
            BASE_FILE,
            NEW_NOTE,
            NotesSyncActionKind.UPDATE_FILE,
        ),
        (
            NotesSyncDirection.FOLDER_TO_NOTES,
            BASE_FILE,
            BASE_NOTE,
            NotesSyncActionKind.NO_CHANGE,
        ),
        (
            NotesSyncDirection.NOTES_TO_FOLDER,
            BASE_FILE,
            BASE_NOTE,
            NotesSyncActionKind.NO_CHANGE,
        ),
    ],
)
def test_safe_direction_matrix(
    direction: NotesSyncDirection,
    file_digest: str,
    note_digest: str,
    expected: NotesSyncActionKind,
) -> None:
    plan = _plan(direction, _binding(file_digest=file_digest, note_digest=note_digest))
    assert [action.kind for action in plan.safe_actions] == [expected]
    assert not plan.attention


@pytest.mark.parametrize(
    ("direction", "file_digest", "note_digest", "reason"),
    [
        (
            NotesSyncDirection.FOLDER_TO_NOTES,
            BASE_FILE,
            NEW_NOTE,
            "out_of_direction_change",
        ),
        (
            NotesSyncDirection.NOTES_TO_FOLDER,
            NEW_FILE,
            BASE_NOTE,
            "out_of_direction_change",
        ),
        (NotesSyncDirection.BIDIRECTIONAL, NEW_FILE, NEW_NOTE, "both_sides_changed"),
        (NotesSyncDirection.FOLDER_TO_NOTES, NEW_FILE, NEW_NOTE, "both_sides_changed"),
        (NotesSyncDirection.NOTES_TO_FOLDER, NEW_FILE, NEW_NOTE, "both_sides_changed"),
    ],
)
def test_attention_direction_matrix_never_selects_a_winner(
    direction: NotesSyncDirection,
    file_digest: str,
    note_digest: str,
    reason: str,
) -> None:
    plan = _plan(direction, _binding(file_digest=file_digest, note_digest=note_digest))
    assert not plan.safe_actions
    assert [(item.kind, item.reason_code) for item in plan.attention] == [
        (ReconciliationAttentionKind.CONFLICT, reason)
    ]


@pytest.mark.parametrize("direction", list(NotesSyncDirection))
@pytest.mark.parametrize(
    ("file_digest", "note_digest", "reason"),
    [
        (None, BASE_NOTE, "file_missing"),
        (BASE_FILE, None, "note_missing"),
        (None, None, "both_missing"),
    ],
)
def test_every_missing_side_is_deletion_review(
    direction: NotesSyncDirection,
    file_digest: str | None,
    note_digest: str | None,
    reason: str,
) -> None:
    plan = _plan(
        direction,
        _binding(
            file_digest=file_digest,
            note_digest=note_digest,
            file_identity=None if file_digest is None else IDENTITY,
        ),
    )
    assert not plan.safe_actions
    assert plan.attention[0].kind is ReconciliationAttentionKind.DELETION_REVIEW
    assert plan.attention[0].reason_code == reason


def test_unbound_one_sided_discovery_creates_only_in_configured_direction() -> None:
    discovered_file = _binding(
        file_digest=NEW_FILE,
        note_digest=None,
        baseline_file="0" * 64,
        baseline_note="0" * 64,
        baseline_identity="0" * 64,
    ).as_unbound()
    discovered_note = _binding(
        file_digest=None,
        note_digest=NEW_NOTE,
        file_identity=None,
        baseline_file="0" * 64,
        baseline_note="0" * 64,
        baseline_identity="0" * 64,
    ).as_unbound()

    assert (
        _plan(NotesSyncDirection.FOLDER_TO_NOTES, discovered_file).safe_actions[0].kind
        is NotesSyncActionKind.CREATE_NOTE
    )
    assert (
        _plan(NotesSyncDirection.NOTES_TO_FOLDER, discovered_note).safe_actions[0].kind
        is NotesSyncActionKind.CREATE_FILE
    )
    assert (
        _plan(NotesSyncDirection.BIDIRECTIONAL, discovered_file).safe_actions[0].kind
        is NotesSyncActionKind.CREATE_NOTE
    )
    assert (
        _plan(NotesSyncDirection.BIDIRECTIONAL, discovered_note).safe_actions[0].kind
        is NotesSyncActionKind.CREATE_FILE
    )
    assert (
        _plan(NotesSyncDirection.NOTES_TO_FOLDER, discovered_file)
        .attention[0]
        .reason_code
        == "out_of_direction_create"
    )
    assert (
        _plan(NotesSyncDirection.FOLDER_TO_NOTES, discovered_note)
        .attention[0]
        .reason_code
        == "out_of_direction_create"
    )


def test_identity_proven_move_precedes_missing_old_path_classification() -> None:
    moved = _binding(relative_path="moved/note.md")
    plan = _plan(NotesSyncDirection.BIDIRECTIONAL, moved)

    assert [action.kind for action in plan.safe_actions] == [
        NotesSyncActionKind.NO_CHANGE
    ]
    assert [effect.kind for effect in plan.managed_placement_effects] == ["file_move"]
    assert not plan.attention


def test_external_move_respects_direction_authority() -> None:
    moved = _binding(relative_path="moved/note.md")

    assert not _plan(NotesSyncDirection.FOLDER_TO_NOTES, moved).attention
    notes_owned = _plan(NotesSyncDirection.NOTES_TO_FOLDER, moved)
    assert notes_owned.attention[0].reason_code == "out_of_direction_move"


def test_ambiguous_or_note_implied_move_requires_attention() -> None:
    ambiguous = _binding(relative_path="moved/note.md", file_identity=MOVED_IDENTITY)
    implied = _binding(note_implied_path="renamed.md")

    assert (
        _plan(NotesSyncDirection.BIDIRECTIONAL, ambiguous).attention[0].reason_code
        == "ambiguous_identity"
    )
    assert (
        _plan(NotesSyncDirection.BIDIRECTIONAL, implied).attention[0].reason_code
        == "note_implied_filesystem_move"
    )


def test_representation_only_change_is_observed_and_direction_aware() -> None:
    baseline = NotesSyncSerializationProfile(False, "lf", True, 0o640)
    changed_profile = NotesSyncSerializationProfile(True, "crlf", False, 0o600)
    changed = _binding(
        baseline_serialization=baseline,
        serialization=changed_profile,
    )

    folder_owned = _plan(NotesSyncDirection.FOLDER_TO_NOTES, changed)
    assert folder_owned.managed_placement_effects[0].kind == "representation_refresh"
    notes_owned = _plan(NotesSyncDirection.NOTES_TO_FOLDER, changed)
    assert notes_owned.attention[0].reason_code == "out_of_direction_representation"


@pytest.mark.parametrize(
    ("kwargs", "skip_kind", "reason"),
    [
        ({"root_available": False}, ReconciliationSkipKind.OFFLINE, "root_offline"),
        ({"root_overlap": True}, ReconciliationSkipKind.UNSAFE_ROOT, "root_overlap"),
        (
            {"write_capable": False},
            ReconciliationSkipKind.CAPABILITY,
            "capability_loss",
        ),
    ],
)
def test_root_level_guards_skip_without_item_actions(
    kwargs: dict[str, object],
    skip_kind: ReconciliationSkipKind,
    reason: str,
) -> None:
    plan = _plan(NotesSyncDirection.BIDIRECTIONAL, _binding(), **kwargs)
    assert not plan.safe_actions and not plan.attention
    assert [(item.kind, item.reason_code) for item in plan.skips] == [
        (skip_kind, reason)
    ]


def test_read_only_capability_still_allows_folder_to_notes_work() -> None:
    plan = _plan(
        NotesSyncDirection.FOLDER_TO_NOTES,
        _binding(file_digest=NEW_FILE),
        write_capable=False,
    )

    assert [action.kind for action in plan.safe_actions] == [
        NotesSyncActionKind.UPDATE_NOTE
    ]
    assert not plan.skips


def test_offline_root_suppresses_missing_side_deletion_rows() -> None:
    plan = _plan(
        NotesSyncDirection.BIDIRECTIONAL,
        _binding(file_digest=None, file_identity=None),
        root_available=False,
    )

    assert not plan.attention
    assert not plan.deletion_groups
    assert plan.skips[0].reason_code == "root_offline"


def test_duplicate_authority_and_stale_observation_require_attention() -> None:
    duplicate = _plan(
        NotesSyncDirection.BIDIRECTIONAL,
        _binding(duplicate_authority=True),
    )
    stale = plan_reconciliation(
        ReconciliationInput(
            root_id="root-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            bindings=(_binding(),),
            observation_generation=8,
            expected_generation=7,
        )
    )

    assert duplicate.attention[0].reason_code == "duplicate_authority"
    assert stale.attention[0].reason_code == "stale_observation"
    assert not stale.safe_actions


def test_planning_is_frozen_deterministic_and_mutation_free() -> None:
    binding = _binding(file_digest=NEW_FILE)
    request = ReconciliationInput(
        root_id="root-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(binding,),
        observation_generation=7,
        expected_generation=7,
    )
    before = repr(request)
    first = plan_reconciliation(request)
    second = plan_reconciliation(request)

    assert first == second
    assert repr(request) == before
    assert len(first.observation_token) == 64
    with pytest.raises(FrozenInstanceError):
        first.safe_actions = ()  # type: ignore[misc]
    assert_review_token(first, first.observation_token)
    with pytest.raises(ValueError, match="stale_review"):
        assert_review_token(first, "0" * 64)


def test_action_identity_is_stable_per_plan_and_changes_with_observation() -> None:
    first = _plan(
        NotesSyncDirection.BIDIRECTIONAL,
        _binding(file_digest=NEW_FILE),
    )
    repeated = _plan(
        NotesSyncDirection.BIDIRECTIONAL,
        _binding(file_digest=NEW_FILE),
    )
    later = _plan(
        NotesSyncDirection.BIDIRECTIONAL,
        _binding(file_digest="9" * 64),
    )

    assert first.safe_actions[0].action_id == repeated.safe_actions[0].action_id
    assert first.safe_actions[0].action_id != later.safe_actions[0].action_id


def test_fresh_review_comparison_fences_version_and_capability_changes() -> None:
    request = ReconciliationInput(
        root_id="root-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(_binding(),),
        observation_generation=7,
        expected_generation=7,
        capability_generation=3,
    )
    plan = plan_reconciliation(request)
    version_changed = ReconciliationInput(
        root_id="root-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(replace(_binding(), note_version=8),),
        observation_generation=7,
        expected_generation=7,
        capability_generation=3,
    )
    capability_changed = replace(request, capability_generation=4)

    assert_review_current(plan, request)
    with pytest.raises(ValueError, match="stale_review"):
        assert_review_current(plan, version_changed)
    with pytest.raises(ValueError, match="stale_review"):
        assert_review_current(plan, capability_changed)


def test_plans_are_sorted_paged_and_group_large_deletion_bursts() -> None:
    bindings = tuple(
        BindingObservation(
            binding_id=f"binding-{index:04d}",
            baseline_file_digest=BASE_FILE,
            baseline_note_digest=BASE_NOTE,
            baseline_identity_digest=IDENTITY,
            baseline_relative_path=f"{index:04d}.md",
            file_digest=None,
            note_digest=BASE_NOTE,
            file_identity_digest=None,
            relative_path=f"{index:04d}.md",
            note_scope_id="scope-1",
            note_id=f"note-{index:04d}",
            note_version=1,
        )
        for index in reversed(range(200))
    )
    plan = plan_reconciliation(
        ReconciliationInput(
            root_id="root-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            bindings=bindings,
            observation_generation=1,
            expected_generation=1,
        )
    )

    assert 0 < plan.page_size < len(bindings)
    assert not plan.attention
    assert len(plan.deletion_groups) == 1
    assert plan.deletion_groups[0].binding_ids == tuple(
        f"binding-{index:04d}" for index in range(200)
    )
    assert {item.reason_code for item in plan.deletion_groups[0].items} == {
        "file_missing"
    }


def test_public_planner_outputs_validate_and_redact_private_observations() -> None:
    request = ReconciliationInput(
        root_id="root-1",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=(
            replace(
                _binding(file_digest=NEW_FILE),
                relative_path="PRIVATE-sentinel/note.md",
            ),
        ),
        observation_generation=1,
        expected_generation=1,
        capability_generation=9,
    )
    plan = plan_reconciliation(request)

    assert "PRIVATE-sentinel" not in repr(plan)
    assert plan.observation_token not in repr(plan)
    assert NEW_FILE not in repr(plan)
    with pytest.raises(TypeError):
        ReconciliationSkip("offline", "root_offline")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ManagedPlacementEffect("file_move", "binding-1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeletionGroup(())
    with pytest.raises(TypeError):
        ReconciliationPlan(
            root_id="root-1",
            observation_token="a" * 64,
            safe_actions=[],  # type: ignore[arg-type]
            attention=(),
            skips=(),
            managed_placement_effects=(),
            deletion_groups=(),
        )
