"""Contract tests for one-time Database Notes import planning."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _source() -> ImportSource:
    return ImportSource(
        kind=ImportSourceKind.DIRECTORY_MEMBER,
        display_path="Project/notes.json",
        source_path=Path("/private/user/Project/notes.json"),
    )


def _payload() -> ParsedNotePayload:
    return ParsedNotePayload(
        title="First note",
        content="Body",
        keywords=["project", "draft"],
        template_name="Meeting",
    )


def _new_item(**overrides: object) -> ImportPreviewItem:
    values: dict[str, object] = {
        "item_id": "item-1",
        "source": _source(),
        "payloads": [_payload()],
        "memberships": [
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=["Project", "Meetings"],
            )
        ],
        "classification": ImportClassification.NEW,
        "reason": "Ready to import.",
        "default_action": ImportAction.CREATE_NEW,
        "selected_action": ImportAction.CREATE_NEW,
        "allowed_actions": [ImportAction.SKIP, ImportAction.CREATE_NEW],
        "match": None,
        "replace_content": False,
        "add_membership": True,
    }
    values.update(overrides)
    return ImportPreviewItem(**values)  # type: ignore[arg-type]


def test_preview_models_copy_nested_collections_into_immutable_tuples() -> None:
    """Frozen records must not retain mutable collections supplied by callers."""
    keywords = ["project", "draft"]
    segments = ["Project", "Meetings"]
    payloads = [
        ParsedNotePayload(
            title="First note",
            content="Body",
            keywords=keywords,
        )
    ]
    memberships = [ProposedFolderMembership(payload_index=0, folder_segments=segments)]
    allowed_actions = [ImportAction.SKIP, ImportAction.CREATE_NEW]
    items = [
        _new_item(
            payloads=payloads,
            memberships=memberships,
            allowed_actions=allowed_actions,
        )
    ]
    proposed_paths = [["Project"], ["Project", "Meetings"]]

    plan = NoteImportPlan(
        bounds=ImportBounds(
            max_files=50,
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
            max_depth=8,
        ),
        items=items,
        proposed_folder_paths=proposed_paths,
        root_collision=RootCollisionState(
            proposed_label="Project",
            collides=False,
        ),
    )

    keywords.append("later")
    segments.append("later")
    payloads.clear()
    memberships.clear()
    allowed_actions.clear()
    items.clear()
    proposed_paths.clear()

    assert plan.items[0].payloads[0].keywords == ("project", "draft")
    assert plan.items[0].memberships[0].folder_segments == (
        "Project",
        "Meetings",
    )
    assert plan.items[0].allowed_actions == (
        ImportAction.SKIP,
        ImportAction.CREATE_NEW,
    )
    assert plan.proposed_folder_paths == (
        ("Project",),
        ("Project", "Meetings"),
    )

    with pytest.raises(FrozenInstanceError):
        plan.items[0].selected_action = ImportAction.SKIP  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.items[0].payloads[0].keywords[0] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.proposed_folder_paths[0][0] = "changed"  # type: ignore[index]


def test_source_keeps_diagnostics_relative_and_internal_path_out_of_repr() -> None:
    """A source's public representation must not expose its execution path."""
    source = _source()

    assert source.display_path == "Project/notes.json"
    assert source.source_path == Path("/private/user/Project/notes.json")
    assert "/private/user" not in repr(source)

    with pytest.raises(ValueError, match="relative"):
        ImportSource(
            kind=ImportSourceKind.SELECTED_FILE,
            display_path="/private/user/secret.md",
            source_path=Path("/private/user/secret.md"),
        )


def test_match_and_collision_records_are_immutable_and_fingerprint_free() -> None:
    """Public matching/collision state carries identities, not private hashes."""
    match = ImportMatch(
        kind=ImportMatchKind.EXACT,
        note_id="note-7",
        note_version=4,
    )
    collision = RootCollisionState(
        proposed_label="Project",
        collides=True,
        choice=RootCollisionChoice.RENAMED_ROOT,
        resolved_label="Imported Project",
    )

    assert match.note_id == "note-7"
    assert not hasattr(match, "fingerprint")
    assert collision.resolved_label == "Imported Project"
    with pytest.raises(FrozenInstanceError):
        match.note_id = "note-8"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        collision.choice = RootCollisionChoice.USE_EXISTING  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_action": ImportAction.SKIP},
        {"selected_action": ImportAction.UPDATE_EXISTING},
        {"allowed_actions": [ImportAction.SKIP, ImportAction.SKIP]},
    ],
)
def test_preview_item_rejects_actions_outside_its_valid_action_set(
    kwargs: dict[str, object],
) -> None:
    """Default, selected, and allowed actions must form one coherent choice set."""
    with pytest.raises(ValueError):
        _new_item(**kwargs)


@pytest.mark.parametrize(
    "match",
    [
        None,
        ImportMatch(
            kind=ImportMatchKind.UNCERTAIN,
            note_id="note-7",
            note_version=4,
        ),
    ],
)
def test_update_requires_an_exact_or_user_confirmed_match(
    match: ImportMatch | None,
) -> None:
    """An uncertain candidate cannot authorize replacement by itself."""
    with pytest.raises(ValueError, match="exact or user-confirmed"):
        _new_item(
            classification=ImportClassification.CHANGED_REPEAT,
            default_action=ImportAction.CREATE_NEW,
            selected_action=ImportAction.UPDATE_EXISTING,
            allowed_actions=[
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ],
            match=match,
            replace_content=True,
        )


def test_update_rejects_a_mutable_exact_match_lookalike() -> None:
    """Only the immutable public match record can authorize an update."""
    mutable_match = SimpleNamespace(
        kind=ImportMatchKind.EXACT,
        note_id="note-7",
        note_version=4,
        fingerprint="private-fingerprint",
    )

    with pytest.raises(TypeError, match="ImportMatch"):
        _new_item(
            classification=ImportClassification.CHANGED_REPEAT,
            selected_action=ImportAction.UPDATE_EXISTING,
            allowed_actions=[
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ],
            match=mutable_match,
            replace_content=True,
        )


@pytest.mark.parametrize(
    "classification",
    [ImportClassification.UNSUPPORTED, ImportClassification.FAILED],
)
def test_unimportable_items_can_only_skip(
    classification: ImportClassification,
) -> None:
    """Unsupported and failed sources cannot acquire a mutating action."""
    with pytest.raises(ValueError, match="must only allow Skip"):
        _new_item(
            classification=classification,
            default_action=ImportAction.CREATE_NEW,
            selected_action=ImportAction.CREATE_NEW,
        )


def test_classifications_enforce_their_match_and_default_contracts() -> None:
    """Repeat categories cannot be constructed without the required match state."""
    exact = ImportMatch(
        kind=ImportMatchKind.EXACT,
        note_id="note-7",
        note_version=4,
    )
    item = _new_item(
        classification=ImportClassification.CHANGED_REPEAT,
        allowed_actions=[
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        ],
        match=exact,
    )

    assert item.match == exact

    with pytest.raises(ValueError, match="exact match"):
        _new_item(
            classification=ImportClassification.UNCHANGED_REPEAT,
            default_action=ImportAction.SKIP,
            selected_action=ImportAction.SKIP,
            allowed_actions=[
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ],
            match=None,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_files": 0},
        {"max_file_bytes": 0},
        {"max_total_bytes": 0},
        {"max_depth": -1},
        {"max_reason_length": 0},
    ],
)
def test_import_bounds_reject_non_positive_capacity(
    kwargs: dict[str, int],
) -> None:
    """Every planner resource bound must be finite and usable."""
    values = {
        "max_files": 50,
        "max_file_bytes": 1_000_000,
        "max_total_bytes": 5_000_000,
        "max_depth": 8,
        "max_reason_length": 240,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        ImportBounds(**values)


def test_memberships_reference_an_existing_payload() -> None:
    """A membership cannot target a note payload absent from its source item."""
    with pytest.raises(ValueError, match="payload index"):
        _new_item(
            memberships=[
                ProposedFolderMembership(
                    payload_index=1,
                    folder_segments=["Project"],
                )
            ]
        )


def test_collision_resolution_requires_a_resolved_root_label() -> None:
    """Rename and unique-sibling choices must identify the chosen folder label."""
    with pytest.raises(ValueError, match="resolved label"):
        RootCollisionState(
            proposed_label="Project",
            collides=True,
            choice=RootCollisionChoice.UNIQUE_SIBLING,
        )
