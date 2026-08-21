from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncAction,
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncFileIdentity,
    NotesSyncFileObservation,
    NotesSyncNoteObservation,
    NotesSyncOperationState,
    NotesSyncPlan,
    NotesSyncRecoveryAdmission,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
    normalize_notes_sync_relative_path,
)


def test_sync_models_are_frozen_and_validate_opaque_ids() -> None:
    plan = NotesSyncPlan(
        root_id="root-1",
        observation_token="observation-1",
        actions=(),
    )

    with pytest.raises(FrozenInstanceError):
        plan.root_id = "replacement"  # type: ignore[misc]
    for invalid in ("", "/private/root", "../root", "contains space", "x" * 257):
        with pytest.raises(ValueError, match="opaque"):
            NotesSyncPlan(root_id=invalid, observation_token="token-1", actions=())


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("note.md", "note.md"),
        ("folder/note.md", "folder/note.md"),
        ("folder//note.md", "folder/note.md"),
    ],
)
def test_relative_paths_are_normalized(candidate: str, expected: str) -> None:
    assert normalize_notes_sync_relative_path(candidate) == expected


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        ".",
        "..",
        "../note.md",
        "folder/../note.md",
        "/note.md",
        "C:/note.md",
        "C:note.md",
        "folder\\note.md",
        "folder/\x00note.md",
        "folder/\nprivate.md",
        "x" * 4097,
        f"{'x' * 256}/note.md",
    ],
)
def test_relative_paths_reject_absolute_alias_and_parent_traversal(
    candidate: str,
) -> None:
    with pytest.raises(ValueError, match="relative path"):
        normalize_notes_sync_relative_path(candidate)


def test_representation_and_observations_retain_private_values_without_repr_leaks() -> (
    None
):
    profile = NotesSyncSerializationProfile(
        utf8_bom=True,
        newline="crlf",
        final_newline=False,
        mode=0o640,
    )
    identity = NotesSyncFileIdentity(device=12, inode=34, link_count=1)
    file_observation = NotesSyncFileObservation(
        relative_path="private/project.md",
        identity=identity,
        content_digest="a" * 64,
        size_bytes=321,
        serialization=profile,
    )
    note_observation = NotesSyncNoteObservation(
        note_scope_id="scope-1",
        note_id="note-1",
        version=7,
        content_digest="b" * 64,
    )

    assert file_observation.relative_path == "private/project.md"
    assert file_observation.serialization is profile
    assert note_observation.version == 7
    for private_value in ("private/project.md", "a" * 64, "b" * 64, "12", "34"):
        assert private_value not in repr(file_observation)
        assert private_value not in repr(note_observation)


def test_action_plan_and_recovery_admission_are_typed_and_privacy_safe() -> None:
    action = NotesSyncAction(
        action_id="action-1",
        kind=NotesSyncActionKind.UPDATE_NOTE,
        binding_id="binding-1",
        reason_code="file_changed",
    )
    plan = NotesSyncPlan(
        root_id="root-1",
        observation_token="token-1",
        actions=(action,),
    )
    admission = NotesSyncRecoveryAdmission(
        admitted=False,
        reason_code="capacity_exceeded",
        required_bytes=200,
        available_bytes=100,
    )

    assert plan.actions == (action,)
    assert admission.required_bytes == 200
    assert "capacity_exceeded" in repr(admission)
    with pytest.raises(ValueError, match="reason code"):
        NotesSyncRecoveryAdmission(admitted=False, reason_code="/private/error")


def test_sync_state_enums_pin_storage_values() -> None:
    assert {item.value for item in NotesSyncDirection} == {
        "bidirectional",
        "folder_to_notes",
        "notes_to_folder",
    }
    assert "active" in {item.value for item in NotesSyncRootState}
    assert "active" in {item.value for item in NotesSyncBindingState}
    assert "completed" in {item.value for item in NotesSyncOperationState}
