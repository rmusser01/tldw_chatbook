"""Pure contracts for reviewed lasting Notes sync conflicts."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields

import pytest

from tldw_chatbook.Notes.notes_sync_conflicts import (
    COMPARISON_MAX_INPUT_CHARS,
    COMPARISON_MAX_INPUT_LINES,
    COMPARISON_MAX_OUTPUT_CHARS,
    COMPARISON_MAX_OUTPUT_LINES,
    ConflictApplyResult,
    ConflictHistoryRow,
    ConflictReceipt,
    ConflictSelection,
    NotesSyncConflictChoice,
    build_conflict_comparison,
    conflict_copies_folder_id,
    conflict_copy_note_id,
    conflict_resolution_operation_id,
    conflict_root_folder_id,
    eligible_conflict_reason,
    linked_undo_operation_id,
)


TOKEN = "a" * 64


def test_conflict_choice_has_exact_wire_values() -> None:
    assert tuple(NotesSyncConflictChoice) == (
        NotesSyncConflictChoice.KEEP_FILE,
        NotesSyncConflictChoice.KEEP_NOTE,
        NotesSyncConflictChoice.KEEP_BOTH,
        NotesSyncConflictChoice.SKIP,
    )
    assert tuple(choice.value for choice in NotesSyncConflictChoice) == (
        "keep_file",
        "keep_note",
        "keep_both",
        "skip",
    )


def test_conflict_selection_requires_a_typed_choice_and_opaque_binding() -> None:
    selection = ConflictSelection("binding-1", NotesSyncConflictChoice.KEEP_FILE)

    assert selection.choice is NotesSyncConflictChoice.KEEP_FILE
    with pytest.raises(TypeError, match="NotesSyncConflictChoice"):
        ConflictSelection("binding-1", "keep_file")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="binding_id"):
        ConflictSelection("/private/binding", NotesSyncConflictChoice.KEEP_FILE)
    with pytest.raises(FrozenInstanceError):
        selection.choice = NotesSyncConflictChoice.SKIP  # type: ignore[misc]


def test_only_bound_content_change_reasons_are_selectable() -> None:
    assert eligible_conflict_reason("both_sides_changed", managed=False)
    assert eligible_conflict_reason("out_of_direction_change", managed=False)
    for reason in (
        "duplicate_authority",
        "out_of_direction_create",
        "out_of_direction_move",
        "out_of_direction_representation",
        "ambiguous_identity",
        "note_implied_filesystem_move",
    ):
        assert not eligible_conflict_reason(reason, managed=False)
    assert not eligible_conflict_reason("both_sides_changed", managed=True)


@pytest.mark.parametrize("managed", [0, None])
def test_conflict_eligibility_rejects_untyped_management_flags(
    managed: object,
) -> None:
    with pytest.raises(TypeError, match="managed must be a boolean"):
        eligible_conflict_reason(
            "both_sides_changed",
            managed=managed,  # type: ignore[arg-type]
        )


def test_conflict_ids_are_canonical_deterministic_and_domain_separated() -> None:
    expected_parent = hashlib.sha256(b"conflict_copies_folder_v1\0scope-1").hexdigest()
    expected_child = hashlib.sha256(
        b"conflict_root_folder_v1\0scope-1\0root-1"
    ).hexdigest()
    expected_copy = hashlib.sha256(
        f"conflict_copy_note_v1\0root-1\0binding-1\0{TOKEN}".encode()
    ).hexdigest()
    expected_operation = hashlib.sha256(
        f"conflict_resolution_v1\0root-1\0binding-1\0{TOKEN}\0keep_file".encode()
    ).hexdigest()
    expected_undo = hashlib.sha256(
        f"undo_resolution_v1\0root-1\0{expected_operation}".encode()
    ).hexdigest()

    assert conflict_copies_folder_id("scope-1") == expected_parent
    assert conflict_root_folder_id("scope-1", "root-1") == expected_child
    assert conflict_copy_note_id("root-1", "binding-1", TOKEN) == expected_copy
    assert (
        conflict_resolution_operation_id(
            "root-1",
            "binding-1",
            TOKEN,
            NotesSyncConflictChoice.KEEP_FILE,
        )
        == expected_operation
    )
    assert linked_undo_operation_id("root-1", expected_operation) == expected_undo
    assert (
        len({expected_parent, expected_child, expected_copy, expected_operation}) == 4
    )
    with pytest.raises(ValueError, match="does not create"):
        conflict_resolution_operation_id(
            "root-1", "binding-1", TOKEN, NotesSyncConflictChoice.SKIP
        )


def test_comparison_diff_is_oriented_from_note_to_file() -> None:
    comparison = build_conflict_comparison(
        binding_id="binding-1",
        title="Title",
        relative_path="folder/note.md",
        note_text="note\n",
        file_text="file\n",
        note_version=3,
        note_updated_at="2026-08-22T12:30:00+00:00",
        file_modified_ns=7,
    )

    assert comparison.diff.startswith("--- Note\n+++ File\n")
    assert "-note" in comparison.diff
    assert "+file" in comparison.diff
    assert comparison.input_elided is False
    assert comparison.output_elided is False


def test_comparison_missing_note_timestamp_is_unavailable() -> None:
    comparison = build_conflict_comparison(
        binding_id="binding-1",
        title="Title",
        relative_path="folder/note.md",
        note_text="same",
        file_text="same",
        note_version=3,
        note_updated_at=None,
        file_modified_ns=7,
    )

    assert comparison.note_updated_label == "Unavailable"


def test_comparison_measures_complete_inputs_before_omitting_large_diff() -> None:
    private_note = "note-private:" + ("n" * COMPARISON_MAX_INPUT_CHARS)
    private_file = "file-private\n" * (COMPARISON_MAX_INPUT_LINES + 1)

    comparison = build_conflict_comparison(
        binding_id="binding-1",
        title="Title",
        relative_path="folder/note.md",
        note_text=private_note,
        file_text=private_file,
        note_version=3,
        note_updated_at=None,
        file_modified_ns=7,
    )

    assert comparison.note_character_count == len(private_note)
    assert comparison.note_line_count == len(private_note.splitlines())
    assert comparison.file_character_count == len(private_file)
    assert comparison.file_line_count == len(private_file.splitlines())
    assert comparison.input_elided is True
    assert comparison.output_elided is False
    assert "Diff omitted" in comparison.diff
    assert f"{len(private_note):,}" in comparison.diff
    assert f"{len(private_file.splitlines()):,}" in comparison.diff
    assert "note-private" not in comparison.diff
    assert "file-private" not in comparison.diff
    assert len(comparison.diff) < 1_000


def test_comparison_bounds_generated_diff_with_one_elision_marker() -> None:
    note_text = "".join(
        f"note-{index:04d}-" + ("n" * 40) + "\n" for index in range(3_000)
    )
    file_text = "".join(
        f"file-{index:04d}-" + ("f" * 40) + "\n" for index in range(3_000)
    )

    comparison = build_conflict_comparison(
        binding_id="binding-1",
        title="Title",
        relative_path="folder/note.md",
        note_text=note_text,
        file_text=file_text,
        note_version=3,
        note_updated_at=None,
        file_modified_ns=7,
    )

    assert len(note_text) <= COMPARISON_MAX_INPUT_CHARS
    assert len(note_text.splitlines()) <= COMPARISON_MAX_INPUT_LINES
    assert len(comparison.diff) <= COMPARISON_MAX_OUTPUT_CHARS
    assert len(comparison.diff.splitlines()) <= COMPARISON_MAX_OUTPUT_LINES
    assert comparison.diff.count("comparison output elided") == 1
    assert comparison.input_elided is False
    assert comparison.output_elided is True


def test_comparison_boundaries_are_inclusive() -> None:
    note_text = "n" * COMPARISON_MAX_INPUT_CHARS
    file_text = ("f\n" * (COMPARISON_MAX_INPUT_LINES - 1)) + "f"

    comparison = build_conflict_comparison(
        binding_id="binding-1",
        title="Title",
        relative_path="folder/note.md",
        note_text=note_text,
        file_text=file_text,
        note_version=3,
        note_updated_at=None,
        file_modified_ns=7,
    )

    assert comparison.input_elided is False


def test_conflict_contract_reprs_do_not_expose_private_values() -> None:
    private_path = "private/secret.md"
    comparison = build_conflict_comparison(
        binding_id="binding-secret",
        title="private title",
        relative_path=private_path,
        note_text="private note body",
        file_text="private file body",
        note_version=3,
        note_updated_at=None,
        file_modified_ns=7,
    )
    selection = ConflictSelection("binding-secret", NotesSyncConflictChoice.KEEP_NOTE)
    result = ConflictApplyResult(
        results=(),
        safe_completed=0,
        conflicts_resolved=0,
        unresolved_conflicts=1,
        attention_remains=True,
        partial=False,
        needs_recovery=False,
        fresh_plan=None,
    )
    receipt = ConflictReceipt(
        operation_id="operation-secret",
        choice=NotesSyncConflictChoice.KEEP_NOTE,
        state="completed",
        undo_available=True,
    )
    history = ConflictHistoryRow(
        operation_id="operation-secret",
        choice=NotesSyncConflictChoice.KEEP_NOTE,
        state="completed",
        completed_at="2026-08-22T12:30:00+00:00",
        updated_at="2026-08-22T12:30:00+00:00",
        undo_available=True,
    )

    rendered = " ".join(
        repr(value) for value in (comparison, selection, result, receipt, history)
    )
    for private in (
        private_path,
        "binding-secret",
        "operation-secret",
        "private title",
        "private note body",
        "private file body",
        hashlib.sha256(b"private note body").hexdigest(),
    ):
        assert private not in rendered
    assert {field.name for field in fields(type(comparison))}.isdisjoint(
        {"absolute_path", "content_digest", "file_identity", "root_path"}
    )


def test_conflict_receipt_requires_a_typed_mutating_choice_and_state() -> None:
    with pytest.raises(TypeError, match="mutating NotesSyncConflictChoice"):
        ConflictReceipt(
            operation_id="operation-1",
            choice=NotesSyncConflictChoice.SKIP,
            state="completed",
            undo_available=False,
        )
    with pytest.raises(TypeError, match="state"):
        ConflictReceipt(
            operation_id="operation-1",
            choice=NotesSyncConflictChoice.KEEP_FILE,
            state=None,  # type: ignore[arg-type]
            undo_available=False,
        )


def test_conflict_apply_result_requires_typed_executor_results() -> None:
    with pytest.raises(TypeError, match="NotesSyncExecutionResult"):
        ConflictApplyResult(
            results=(object(),),  # type: ignore[arg-type]
            safe_completed=0,
            conflicts_resolved=0,
            unresolved_conflicts=1,
            attention_remains=True,
            partial=False,
            needs_recovery=False,
            fresh_plan=None,
        )
