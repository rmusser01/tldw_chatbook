"""Contract tests for one-time Database Notes import planning."""

import csv
import inspect
import os
from collections import Counter
from dataclasses import FrozenInstanceError
from dataclasses import replace as dataclass_replace
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Self

import pytest

from tldw_chatbook.Notes import (
    note_import_discovery,
    note_import_parsers,
    note_import_plan_models,
    note_import_planner,
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
from tldw_chatbook.Notes.note_import_planner import (
    ImportDiscovery,
    ImportSelectionError,
    discover_import_sources,
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


@pytest.mark.parametrize(
    ("field_name", "limit_name"),
    [
        ("title", "MAX_IMPORT_TITLE_LENGTH"),
        ("template_name", "MAX_IMPORT_TEMPLATE_NAME_LENGTH"),
        ("keyword", "MAX_IMPORT_KEYWORD_LENGTH"),
    ],
)
def test_parsed_note_payload_enforces_public_scalar_limits(
    field_name: str,
    limit_name: str,
) -> None:
    limit = getattr(note_import_plan_models, limit_name)
    values: dict[str, object] = {
        "title": "T" * limit if field_name == "title" else "Title",
        "content": "Body",
        "keywords": ("K" * limit,) if field_name == "keyword" else (),
        "template_name": "M" * limit if field_name == "template_name" else None,
    }

    payload = ParsedNotePayload(**values)  # type: ignore[arg-type]

    if field_name == "title":
        accepted_value = payload.title
    elif field_name == "template_name":
        accepted_value = payload.template_name
    else:
        accepted_value = payload.keywords[0]
    assert accepted_value is not None
    assert len(accepted_value) == limit

    if field_name == "keyword":
        values["keywords"] = ("K" * (limit + 1),)
    elif field_name == "title":
        values["title"] = "T" * (limit + 1)
    else:
        values["template_name"] = "M" * (limit + 1)
    with pytest.raises(ValueError, match="safety ceiling"):
        ParsedNotePayload(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("title", ""),
        ("title", " \t\n"),
        ("content", ""),
        ("content", " \t\n"),
    ],
)
def test_parsed_note_payload_rejects_blank_required_text(
    field_name: str,
    value: str,
) -> None:
    values = {"title": "Title", "content": "Body"}
    values[field_name] = value

    with pytest.raises(ValueError, match="non-blank"):
        ParsedNotePayload(**values)


@pytest.mark.parametrize(
    "paths",
    [
        "Folder",
        b"Folder",
        ("Folder",),
        (b"Folder",),
    ],
)
def test_parsed_import_batch_rejects_text_in_folder_path_collections(
    paths: object,
) -> None:
    with pytest.raises(TypeError, match="collection, not text"):
        note_import_planner.ParsedImportBatch(
            parsed=(),
            issues=(),
            proposed_folder_paths=paths,  # type: ignore[arg-type]
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


@pytest.mark.parametrize(
    "item_id",
    [
        "contains whitespace",
        "café",
        "x" * 257,
        "/path-like",
    ],
)
def test_preview_item_rejects_unsafe_or_inoperable_identifiers(item_id: str) -> None:
    with pytest.raises(ValueError, match="safe opaque"):
        _new_item(item_id=item_id)


@pytest.mark.parametrize("item_id", [17, b"item-1", None])
def test_preview_item_rejects_coerced_identifier_types(item_id: object) -> None:
    with pytest.raises(TypeError, match="item_id"):
        _new_item(item_id=item_id)


def _plan_with_item(
    item: ImportPreviewItem,
    *,
    max_reason_length: int = 240,
) -> NoteImportPlan:
    return NoteImportPlan(
        bounds=ImportBounds(
            max_files=50,
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
            max_depth=8,
            max_reason_length=max_reason_length,
        ),
        items=[item],
        proposed_folder_paths=[["Project"], ["Project", "Meetings"]],
        root_collision=RootCollisionState(
            proposed_label="Project",
            collides=False,
        ),
    )


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


def test_import_resource_and_scalar_absolute_ceilings_are_named() -> None:
    assert note_import_plan_models.MAX_IMPORT_FILES == 10_000
    assert note_import_plan_models.MAX_IMPORT_FILE_BYTES == 64 * 1024 * 1024
    assert note_import_plan_models.MAX_IMPORT_TOTAL_BYTES == 512 * 1024 * 1024
    assert note_import_plan_models.MAX_IMPORT_ENTRIES == 100_000
    assert note_import_plan_models.MAX_IMPORT_NOTES_PER_FILE == 10_000
    assert note_import_plan_models.MAX_IMPORT_KEYWORDS_PER_NOTE == 1_000
    assert note_import_plan_models.MAX_IMPORT_TITLE_LENGTH == 4_096
    assert note_import_plan_models.MAX_IMPORT_TEMPLATE_NAME_LENGTH == 1_024
    assert note_import_plan_models.MAX_IMPORT_KEYWORD_LENGTH == 512


def _absolute_ceiling_bounds(**overrides: int) -> ImportBounds:
    values = {
        "max_files": 1,
        "max_file_bytes": 1,
        "max_total_bytes": note_import_plan_models.MAX_IMPORT_TOTAL_BYTES,
        "max_depth": 1,
        "max_entries": 1,
        "max_notes_per_file": 1,
        "max_keywords_per_note": 1,
    }
    values.update(overrides)
    return ImportBounds(**values)


@pytest.mark.parametrize(
    ("field_name", "ceiling"),
    [
        ("max_files", 10_000),
        ("max_file_bytes", 64 * 1024 * 1024),
        ("max_total_bytes", 512 * 1024 * 1024),
        ("max_entries", 100_000),
        ("max_notes_per_file", 10_000),
        ("max_keywords_per_note", 1_000),
    ],
)
def test_import_bounds_accept_each_absolute_resource_ceiling(
    field_name: str,
    ceiling: int,
) -> None:
    bounds = _absolute_ceiling_bounds(**{field_name: ceiling})

    assert getattr(bounds, field_name) == ceiling


@pytest.mark.parametrize(
    ("field_name", "ceiling"),
    [
        ("max_files", 10_000),
        ("max_file_bytes", 64 * 1024 * 1024),
        ("max_total_bytes", 512 * 1024 * 1024),
        ("max_entries", 100_000),
        ("max_notes_per_file", 10_000),
        ("max_keywords_per_note", 1_000),
    ],
)
def test_import_bounds_reject_values_above_absolute_resource_ceilings(
    field_name: str,
    ceiling: int,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _absolute_ceiling_bounds(**{field_name: ceiling + 1})


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


@pytest.mark.parametrize("reason", ["", "   ", "unsafe\x00reason"])
@pytest.mark.parametrize(
    "classification",
    [ImportClassification.UNSUPPORTED, ImportClassification.FAILED],
)
def test_unimportable_items_require_a_safe_nonblank_reason(
    classification: ImportClassification,
    reason: str,
) -> None:
    """Skipped failures still need bounded, display-safe user guidance."""
    with pytest.raises(ValueError, match="reason"):
        _new_item(
            payloads=[],
            memberships=[],
            classification=classification,
            reason=reason,
            default_action=ImportAction.SKIP,
            selected_action=ImportAction.SKIP,
            allowed_actions=[ImportAction.SKIP],
            add_membership=False,
        )


def test_plan_enforces_its_configured_reason_bound() -> None:
    """The aggregate bound must constrain every item's public diagnostic reason."""
    item = _new_item(reason="Sixteen chars ok")

    with pytest.raises(ValueError, match="max_reason_length"):
        _plan_with_item(item, max_reason_length=8)


@pytest.mark.parametrize(
    "classification",
    [
        ImportClassification.NEW,
        ImportClassification.UNCHANGED_REPEAT,
        ImportClassification.CHANGED_REPEAT,
        ImportClassification.UNCERTAIN_MATCH,
    ],
)
def test_importable_classifications_require_a_payload(
    classification: ImportClassification,
) -> None:
    """No import action can be approved without a parsed note payload."""
    match = None
    default_action = ImportAction.CREATE_NEW
    selected_action = ImportAction.CREATE_NEW
    allowed_actions = [ImportAction.SKIP, ImportAction.CREATE_NEW]
    add_membership = True
    if classification in {
        ImportClassification.UNCHANGED_REPEAT,
        ImportClassification.CHANGED_REPEAT,
    }:
        match = ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id="note-7",
            note_version=4,
        )
        allowed_actions.append(ImportAction.UPDATE_EXISTING)
    if classification is ImportClassification.UNCHANGED_REPEAT:
        default_action = ImportAction.SKIP
        selected_action = ImportAction.SKIP
        add_membership = False
    if classification is ImportClassification.UNCERTAIN_MATCH:
        match = ImportMatch(
            kind=ImportMatchKind.UNCERTAIN,
            note_id="note-7",
            note_version=4,
        )

    with pytest.raises(ValueError, match="at least one payload"):
        _new_item(
            payloads=[],
            memberships=[],
            classification=classification,
            default_action=default_action,
            selected_action=selected_action,
            allowed_actions=allowed_actions,
            match=match,
            add_membership=add_membership,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"add_membership": False},
        {
            "payloads": [_payload(), _payload()],
            "memberships": [
                ProposedFolderMembership(
                    payload_index=0,
                    folder_segments=["Project"],
                )
            ],
        },
    ],
)
def test_create_new_requires_membership_coverage_for_every_payload(
    overrides: dict[str, object],
) -> None:
    """Every created note must have its previewed manual destination."""
    with pytest.raises(ValueError, match="membership"):
        _new_item(**overrides)


def test_add_membership_requires_coverage_for_every_payload() -> None:
    """An update cannot claim placement while omitting a structured payload."""
    exact = ImportMatch(
        kind=ImportMatchKind.EXACT,
        note_id="note-7",
        note_version=4,
    )

    with pytest.raises(ValueError, match="membership"):
        _new_item(
            payloads=[_payload(), _payload()],
            memberships=[
                ProposedFolderMembership(
                    payload_index=0,
                    folder_segments=["Project"],
                )
            ],
            classification=ImportClassification.CHANGED_REPEAT,
            selected_action=ImportAction.UPDATE_EXISTING,
            allowed_actions=[
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ],
            match=exact,
            replace_content=True,
            add_membership=True,
        )


def test_update_existing_rejects_a_noop_selection() -> None:
    """An Update choice must approve content, membership, or both."""
    exact = ImportMatch(
        kind=ImportMatchKind.EXACT,
        note_id="note-7",
        note_version=4,
    )

    with pytest.raises(ValueError, match="replace content or add membership"):
        _new_item(
            classification=ImportClassification.CHANGED_REPEAT,
            selected_action=ImportAction.UPDATE_EXISTING,
            allowed_actions=[
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ],
            match=exact,
            replace_content=False,
            add_membership=False,
        )


@pytest.mark.parametrize(
    ("replace_content", "add_membership"),
    [(True, False), (False, True), (True, True)],
)
def test_update_existing_accepts_independent_content_and_membership_choices(
    replace_content: bool,
    add_membership: bool,
) -> None:
    """Content-only, membership-only, and combined updates are all meaningful."""
    exact = ImportMatch(
        kind=ImportMatchKind.EXACT,
        note_id="note-7",
        note_version=4,
    )

    item = _new_item(
        classification=ImportClassification.CHANGED_REPEAT,
        selected_action=ImportAction.UPDATE_EXISTING,
        allowed_actions=[
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        ],
        match=exact,
        replace_content=replace_content,
        add_membership=add_membership,
    )

    assert item.replace_content is replace_content
    assert item.add_membership is add_membership


def test_skip_is_always_a_noop() -> None:
    """Skipping preserves preview information but approves no mutation."""
    item = _new_item(
        selected_action=ImportAction.SKIP,
        replace_content=False,
        add_membership=False,
    )

    assert item.payloads
    assert item.memberships
    assert item.selected_action is ImportAction.SKIP


def test_plan_repr_redacts_nested_payload_and_execution_path_values() -> None:
    """Nested repr output cannot accidentally become a content-bearing log line."""
    payload = ParsedNotePayload(
        title="PRIVATE TITLE",
        content="PRIVATE BODY",
        keywords=["PRIVATE KEYWORD"],
        template_name="PRIVATE TEMPLATE",
    )
    item = _new_item(payloads=[payload])
    rendered = repr(_plan_with_item(item))

    assert "PRIVATE TITLE" not in rendered
    assert "PRIVATE BODY" not in rendered
    assert "PRIVATE KEYWORD" not in rendered
    assert "PRIVATE TEMPLATE" not in rendered
    assert "/private/user" not in rendered


def test_plan_exposes_only_an_immutable_redacted_diagnostic_projection() -> None:
    """Logging uses the explicit projection rather than dataclass serialization."""
    payload = ParsedNotePayload(
        title="PRIVATE TITLE",
        content="PRIVATE BODY",
        keywords=["PRIVATE KEYWORD"],
        template_name="PRIVATE TEMPLATE",
    )
    plan = _plan_with_item(
        _new_item(item_id="HASH-FINGERPRINT-SECRET", payloads=[payload])
    )

    diagnostic = plan.to_diagnostic()

    assert diagnostic.item_count == 1
    assert diagnostic.proposed_folder_count == 2
    assert diagnostic.items[0].source_display_path == "Project/notes.json"
    assert diagnostic.items[0].classification is ImportClassification.NEW
    assert diagnostic.items[0].selected_action is ImportAction.CREATE_NEW
    assert diagnostic.items[0].payload_count == 1
    assert diagnostic.items[0].membership_count == 1
    assert not hasattr(diagnostic.items[0], "reason")
    rendered = repr(diagnostic)
    assert "PRIVATE" not in rendered
    assert "/private/user" not in rendered
    assert "fingerprint" not in rendered
    assert "HASH-FINGERPRINT-SECRET" not in rendered
    with pytest.raises(FrozenInstanceError):
        diagnostic.item_count = 2  # type: ignore[misc]


def test_diagnostic_projection_excludes_free_form_user_reason() -> None:
    """User-facing reasons cannot leak sensitive details into persistent logs."""
    sensitive_values = (
        "/Users/alice/private.md",
        "SOURCE_SECRET_BODY",
        "sha256:deadbeef",
        "PermissionError: denied",
    )
    item = _new_item(
        payloads=[],
        memberships=[],
        classification=ImportClassification.FAILED,
        reason=" | ".join(sensitive_values),
        default_action=ImportAction.SKIP,
        selected_action=ImportAction.SKIP,
        allowed_actions=[ImportAction.SKIP],
        add_membership=False,
    )

    rendered = repr(_plan_with_item(item).to_diagnostic())

    for sensitive_value in sensitive_values:
        assert sensitive_value not in rendered


@pytest.mark.parametrize(
    (
        "classification",
        "match_kind",
        "default_action",
        "allowed_actions",
        "selected_action",
        "replace_content",
        "add_membership",
    ),
    [
        (
            ImportClassification.NEW,
            None,
            ImportAction.CREATE_NEW,
            (ImportAction.SKIP, ImportAction.CREATE_NEW),
            ImportAction.CREATE_NEW,
            False,
            True,
        ),
        (
            ImportClassification.UNCHANGED_REPEAT,
            ImportMatchKind.EXACT,
            ImportAction.SKIP,
            (
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ),
            ImportAction.SKIP,
            False,
            False,
        ),
        (
            ImportClassification.CHANGED_REPEAT,
            ImportMatchKind.EXACT,
            ImportAction.CREATE_NEW,
            (
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ),
            ImportAction.UPDATE_EXISTING,
            True,
            False,
        ),
        (
            ImportClassification.UNCERTAIN_MATCH,
            ImportMatchKind.UNCERTAIN,
            ImportAction.CREATE_NEW,
            (ImportAction.SKIP, ImportAction.CREATE_NEW),
            ImportAction.CREATE_NEW,
            False,
            True,
        ),
        (
            ImportClassification.UNCERTAIN_MATCH,
            ImportMatchKind.USER_CONFIRMED,
            ImportAction.CREATE_NEW,
            (
                ImportAction.SKIP,
                ImportAction.CREATE_NEW,
                ImportAction.UPDATE_EXISTING,
            ),
            ImportAction.UPDATE_EXISTING,
            False,
            True,
        ),
        (
            ImportClassification.UNSUPPORTED,
            None,
            ImportAction.SKIP,
            (ImportAction.SKIP,),
            ImportAction.SKIP,
            False,
            False,
        ),
        (
            ImportClassification.FAILED,
            None,
            ImportAction.SKIP,
            (ImportAction.SKIP,),
            ImportAction.SKIP,
            False,
            False,
        ),
    ],
)
def test_classification_action_matrix_accepts_only_valid_contracts(
    classification: ImportClassification,
    match_kind: ImportMatchKind | None,
    default_action: ImportAction,
    allowed_actions: tuple[ImportAction, ...],
    selected_action: ImportAction,
    replace_content: bool,
    add_membership: bool,
) -> None:
    """All classifications retain their approved defaults and update eligibility."""
    match = (
        ImportMatch(kind=match_kind, note_id="note-7", note_version=4)
        if match_kind is not None
        else None
    )
    importable = classification not in {
        ImportClassification.UNSUPPORTED,
        ImportClassification.FAILED,
    }

    item = _new_item(
        payloads=[_payload()] if importable else [],
        memberships=(
            [
                ProposedFolderMembership(
                    payload_index=0,
                    folder_segments=["Project"],
                )
            ]
            if importable
            else []
        ),
        classification=classification,
        reason=f"{classification.value} reason",
        default_action=default_action,
        selected_action=selected_action,
        allowed_actions=allowed_actions,
        match=match,
        replace_content=replace_content,
        add_membership=add_membership,
    )

    assert item.default_action is default_action
    assert item.allowed_actions == allowed_actions


@pytest.mark.parametrize(
    ("classification", "match_kind"),
    [
        (ImportClassification.NEW, ImportMatchKind.EXACT),
        (ImportClassification.UNCHANGED_REPEAT, ImportMatchKind.UNCERTAIN),
        (ImportClassification.CHANGED_REPEAT, ImportMatchKind.USER_CONFIRMED),
        (ImportClassification.UNCERTAIN_MATCH, ImportMatchKind.EXACT),
        (ImportClassification.UNSUPPORTED, ImportMatchKind.EXACT),
        (ImportClassification.FAILED, ImportMatchKind.UNCERTAIN),
    ],
)
def test_classification_action_matrix_rejects_invalid_match_states(
    classification: ImportClassification,
    match_kind: ImportMatchKind,
) -> None:
    """Classification labels cannot be paired with a contradictory match state."""
    match = ImportMatch(kind=match_kind, note_id="note-7", note_version=4)
    unimportable = classification in {
        ImportClassification.UNSUPPORTED,
        ImportClassification.FAILED,
    }
    default_action = (
        ImportAction.SKIP
        if classification
        in {
            ImportClassification.UNCHANGED_REPEAT,
            ImportClassification.UNSUPPORTED,
            ImportClassification.FAILED,
        }
        else ImportAction.CREATE_NEW
    )

    with pytest.raises(ValueError):
        _new_item(
            payloads=[] if unimportable else [_payload()],
            memberships=(
                []
                if unimportable
                else [
                    ProposedFolderMembership(
                        payload_index=0,
                        folder_segments=["Project"],
                    )
                ]
            ),
            classification=classification,
            default_action=default_action,
            selected_action=ImportAction.SKIP if unimportable else default_action,
            allowed_actions=(
                [ImportAction.SKIP]
                if unimportable
                else [ImportAction.SKIP, ImportAction.CREATE_NEW]
            ),
            match=match,
            add_membership=not unimportable,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ProposedFolderMembership(
            payload_index=True,
            folder_segments=["Project"],
        ),
        lambda: ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id="note-7",
            note_version=True,
        ),
        lambda: ImportBounds(
            max_files=True,
            max_file_bytes=100,
            max_total_bytes=100,
            max_depth=1,
        ),
        lambda: ImportBounds(
            max_files=1,
            max_file_bytes=100,
            max_total_bytes=100,
            max_depth=False,
        ),
    ],
)
def test_integer_fields_reject_booleans(factory: object) -> None:
    """Boolean values cannot masquerade as resource counts or record indexes."""
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize("field_name", ["replace_content", "add_membership"])
def test_mutation_flags_are_type_checked_before_action_semantics(
    field_name: str,
) -> None:
    """Truthiness cannot smuggle a non-boolean mutation choice into the plan."""
    with pytest.raises(TypeError, match="must be booleans"):
        _new_item(**{field_name: 1})


@pytest.mark.parametrize("display_path", [".", "./"])
def test_source_rejects_normalized_empty_display_paths(display_path: str) -> None:
    """The public relative path must identify a real source entry."""
    with pytest.raises(ValueError, match="relative"):
        ImportSource(
            kind=ImportSourceKind.SELECTED_FILE,
            display_path=display_path,
            source_path=Path("/private/user/note.md"),
        )


def test_plan_rejects_duplicate_proposed_folder_paths() -> None:
    """A proposed tree cannot carry the same logical folder twice."""
    with pytest.raises(ValueError, match="duplicate"):
        NoteImportPlan(
            bounds=ImportBounds(
                max_files=50,
                max_file_bytes=1_000_000,
                max_total_bytes=5_000_000,
                max_depth=8,
            ),
            items=[_new_item()],
            proposed_folder_paths=[["Project"], ["Project"]],
        )


def _discovery_bounds(**overrides: int) -> ImportBounds:
    values = {
        "max_files": 20,
        "max_file_bytes": 1_000,
        "max_total_bytes": 5_000,
        "max_depth": 4,
        "max_reason_length": 120,
        "max_entries": 100,
    }
    values.update(overrides)
    return ImportBounds(**values)


def test_discovery_accepts_several_individual_regular_files(tmp_path: Path) -> None:
    """Several file selections become deterministic selected-file candidates."""
    second = tmp_path / "second.md"
    first = tmp_path / "first.txt"
    second.write_text("two", encoding="utf-8")
    first.write_text("one", encoding="utf-8")

    discovery = discover_import_sources([second, first], _discovery_bounds())

    assert discovery.root_label is None
    assert [candidate.source.display_path for candidate in discovery.candidates] == [
        "first.txt",
        "second.md",
    ]
    assert all(
        candidate.source.kind is ImportSourceKind.SELECTED_FILE
        for candidate in discovery.candidates
    )
    assert [candidate.size_bytes for candidate in discovery.candidates] == [3, 3]
    assert discovery.total_bytes == 6
    assert discovery.failures == ()


def test_discovery_accepts_one_directory_and_scans_it_recursively(
    tmp_path: Path,
) -> None:
    """A directory selection retains its root label and relative hierarchy."""
    root = tmp_path / "Project"
    nested = root / "child" / "deeper"
    nested.mkdir(parents=True)
    (root / "root.md").write_text("root", encoding="utf-8")
    (nested / "note.txt").write_text("nested", encoding="utf-8")

    discovery = discover_import_sources([root], _discovery_bounds())

    assert discovery.root_label == "Project"
    assert [candidate.source.display_path for candidate in discovery.candidates] == [
        "Project/child/deeper/note.txt",
        "Project/root.md",
    ]
    assert all(
        candidate.source.kind is ImportSourceKind.DIRECTORY_MEMBER
        for candidate in discovery.candidates
    )
    assert discovery.entry_count == 4


@pytest.mark.parametrize(
    ("selection_factory", "reason_code"),
    [
        (lambda root: [], "empty_selection"),
        (
            lambda root: [root / "note.md", root / "folder"],
            "mixed_selection",
        ),
        (
            lambda root: [root / "folder", root / "other-folder"],
            "multiple_directories",
        ),
    ],
)
def test_discovery_rejects_invalid_selection_shapes(
    tmp_path: Path,
    selection_factory: object,
    reason_code: str,
) -> None:
    """Selection is files-only or exactly one directory, never empty or mixed."""
    (tmp_path / "note.md").write_text("note", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    (tmp_path / "other-folder").mkdir()
    selection = selection_factory(tmp_path)  # type: ignore[operator]

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources(selection, _discovery_bounds())

    assert raised.value.reason_code == reason_code
    assert str(tmp_path) not in str(raised.value)


@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_discovery_rejects_a_selected_symlink(
    tmp_path: Path,
    target_kind: str,
) -> None:
    """A selected link is fatal even when its target is otherwise admissible."""
    target = tmp_path / "target"
    if target_kind == "file":
        target.write_text("note", encoding="utf-8")
    else:
        target.mkdir()
    selected = tmp_path / "selected-link"
    selected.symlink_to(target, target_is_directory=target_kind == "directory")

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([selected], _discovery_bounds())

    assert raised.value.reason_code == "selected_symlink"
    assert str(target) not in repr(raised.value)


def test_nested_symlinks_are_visible_failures_and_are_never_traversed(
    tmp_path: Path,
) -> None:
    """Nested links become Skip-ready failures without admitting their targets."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    root = tmp_path / "Project"
    root.mkdir()
    (root / "safe.md").write_text("safe", encoding="utf-8")
    (root / "file-link.md").symlink_to(root / "safe.md")
    (root / "folder-link").symlink_to(outside, target_is_directory=True)

    discovery = discover_import_sources([root], _discovery_bounds())

    assert [candidate.source.display_path for candidate in discovery.candidates] == [
        "Project/safe.md"
    ]
    assert [failure.display_path for failure in discovery.failures] == [
        "Project/file-link.md",
        "Project/folder-link",
    ]
    assert {failure.reason_code for failure in discovery.failures} == {"nested_symlink"}
    assert all("secret.md" not in repr(failure) for failure in discovery.failures)


def test_missing_and_non_regular_top_level_selections_fail_safely(
    tmp_path: Path,
) -> None:
    """Missing and special top-level entries reject the whole selection."""
    missing = tmp_path / "missing.md"

    with pytest.raises(ImportSelectionError) as missing_error:
        discover_import_sources([missing], _discovery_bounds())

    assert missing_error.value.reason_code == "selection_missing"
    assert str(missing) not in str(missing_error.value)

    fifo = tmp_path / "selected-fifo"
    os.mkfifo(fifo)
    with pytest.raises(ImportSelectionError) as fifo_error:
        discover_import_sources([fifo], _discovery_bounds())

    assert fifo_error.value.reason_code == "selection_not_regular"
    assert str(fifo) not in str(fifo_error.value)


def test_nested_non_regular_entries_become_bounded_safe_failures(
    tmp_path: Path,
) -> None:
    """A nested special entry is categorized without aborting safe candidates."""
    root = tmp_path / "Project"
    root.mkdir()
    fifo = root / "events"
    os.mkfifo(fifo)

    discovery = discover_import_sources(
        [root],
        _discovery_bounds(max_reason_length=24),
    )

    assert discovery.candidates == ()
    assert len(discovery.failures) == 1
    failure = discovery.failures[0]
    assert failure.display_path == "Project/events"
    assert failure.reason_code == "nested_not_regular"
    assert 0 < len(failure.user_message) <= 24
    assert str(root) not in repr(failure)


def test_nested_unsafe_names_are_escaped_in_visible_failures(tmp_path: Path) -> None:
    """A rejected source name cannot inject an ambiguous diagnostic path."""
    root = tmp_path / "Project"
    root.mkdir()
    (root / "unsafe\\name.md").write_text("note", encoding="utf-8")

    discovery = discover_import_sources([root], _discovery_bounds())

    assert discovery.candidates == ()
    assert (
        discovery.failures[0].display_path == "Project/.unsafe-entry/unsafe%5Cname.md"
    )
    assert "\\" not in repr(discovery.failures[0])


def test_directory_discovery_order_is_deterministic(tmp_path: Path) -> None:
    """Filesystem enumeration order cannot affect public candidate/failure order."""
    root = tmp_path / "Project"
    (root / "z-dir").mkdir(parents=True)
    (root / "a-dir").mkdir()
    for relative_path in ("z.md", "a-dir/z.md", "a.md", "z-dir/a.md"):
        path = root / relative_path
        path.write_text(relative_path, encoding="utf-8")

    first = discover_import_sources([root], _discovery_bounds())
    second = discover_import_sources([root], _discovery_bounds())

    expected = sorted(candidate.source.display_path for candidate in first.candidates)
    assert [candidate.source.display_path for candidate in first.candidates] == expected
    assert second == first


def test_discovery_redacts_internal_paths_and_identity_from_repr(
    tmp_path: Path,
) -> None:
    """Only relative source names and the selected root label are diagnostic."""
    private_parent = tmp_path / "PRIVATE-ABSOLUTE-PARENT"
    root = private_parent / "Project"
    root.mkdir(parents=True)
    note = root / "child.md"
    note.write_text("private body", encoding="utf-8")

    discovery = discover_import_sources([root], _discovery_bounds())

    candidate = discovery.candidates[0]
    assert discovery.root_label == "Project"
    assert candidate.source.display_path == "Project/child.md"
    assert candidate.source.source_path == note.absolute()
    assert candidate.identity.size == len("private body")
    rendered = repr(discovery)
    assert str(private_parent) not in rendered
    assert "private body" not in rendered
    assert "st_dev" not in rendered


@pytest.mark.parametrize(
    ("tree_factory", "bounds_overrides", "reason_code"),
    [
        (
            lambda root: (root / "child").mkdir(),
            {"max_depth": 0},
            "max_depth_exceeded",
        ),
        (
            lambda root: [
                (root / name).write_text(name, encoding="utf-8")
                for name in ("one.md", "two.md")
            ],
            {"max_files": 1},
            "max_files_exceeded",
        ),
        (
            lambda root: (root / "large.md").write_bytes(b"1234"),
            {"max_file_bytes": 3},
            "max_file_bytes_exceeded",
        ),
        (
            lambda root: [
                (root / name).write_bytes(b"12") for name in ("one.md", "two.md")
            ],
            {"max_file_bytes": 3, "max_total_bytes": 3},
            "max_total_bytes_exceeded",
        ),
        (
            lambda root: [(root / name).mkdir() for name in ("one", "two", "three")],
            {"max_entries": 2},
            "max_entries_exceeded",
        ),
    ],
)
def test_directory_discovery_limits_fail_closed(
    tmp_path: Path,
    tree_factory: object,
    bounds_overrides: dict[str, int],
    reason_code: str,
) -> None:
    """Any resource-limit breach rejects discovery instead of returning a prefix."""
    root = tmp_path / "Project"
    root.mkdir()
    tree_factory(root)  # type: ignore[operator]

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _discovery_bounds(**bounds_overrides))

    assert raised.value.reason_code == reason_code
    assert str(root) not in str(raised.value)


def test_nested_unsafe_entries_count_toward_the_breadth_limit(
    tmp_path: Path,
) -> None:
    """Links cannot bypass the directory-entry budget just because they are skipped."""
    root = tmp_path / "Project"
    root.mkdir()
    target = tmp_path / "target.md"
    target.write_text("target", encoding="utf-8")
    (root / "one-link").symlink_to(target)
    (root / "two-link").symlink_to(target)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _discovery_bounds(max_entries=1))

    assert raised.value.reason_code == "max_entries_exceeded"


def test_entry_limit_stops_directory_enumeration_before_unbounded_sort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breadth rejection occurs while enumerating, before sorting an oversized list."""
    root = tmp_path / "Project"
    root.mkdir()

    class GuardedScandir:
        yielded = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self) -> Self:
            return self

        def __next__(self) -> SimpleNamespace:
            self.yielded += 1
            if self.yielded > 3:
                raise AssertionError("discovery consumed beyond the entry bound")
            return SimpleNamespace(name=f"entry-{self.yielded}")

    guarded = GuardedScandir()
    monkeypatch.setattr(os, "scandir", lambda _directory_fd: guarded)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _discovery_bounds(max_entries=2))

    assert raised.value.reason_code == "max_entries_exceeded"
    assert guarded.yielded == 3


def test_selected_file_count_and_size_limits_fail_closed(tmp_path: Path) -> None:
    """Direct file selections obey the same candidate and byte ceilings."""
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_bytes(b"12")
    second.write_bytes(b"34")

    with pytest.raises(ImportSelectionError) as count_error:
        discover_import_sources(
            [first, second],
            _discovery_bounds(max_files=1),
        )
    assert count_error.value.reason_code == "max_files_exceeded"

    with pytest.raises(ImportSelectionError) as total_error:
        discover_import_sources(
            [first, second],
            _discovery_bounds(max_file_bytes=3, max_total_bytes=3),
        )
    assert total_error.value.reason_code == "max_total_bytes_exceeded"


def test_duplicate_selected_file_display_names_are_rejected(tmp_path: Path) -> None:
    """Separate files cannot produce an ambiguous manual-selection display path."""
    first = tmp_path / "first" / "note.md"
    second = tmp_path / "second" / "note.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([first, second], _discovery_bounds())

    assert raised.value.reason_code == "ambiguous_display_path"


@pytest.mark.parametrize("max_entries", [0, -1])
def test_import_bounds_require_a_positive_entry_limit(max_entries: int) -> None:
    """Breadth must have its own finite, explicit capacity."""
    with pytest.raises(ValueError, match="max_entries"):
        _discovery_bounds(max_entries=max_entries)


def test_import_bounds_entry_limit_rejects_booleans() -> None:
    """A boolean cannot masquerade as a directory-entry budget."""
    with pytest.raises(TypeError, match="max_entries"):
        _discovery_bounds(max_entries=True)


def test_discovery_performs_no_app_level_filesystem_mutation(
    tmp_path: Path,
) -> None:
    """Discovery reads metadata only and leaves the selected tree byte-identical."""
    root = tmp_path / "Project"
    nested = root / "child"
    nested.mkdir(parents=True)
    note = nested / "note.md"
    note.write_bytes(b"unchanged")
    before = {
        path.relative_to(root).as_posix(): (
            "directory" if path.is_dir() else path.read_bytes()
        )
        for path in root.rglob("*")
    }

    discovery = discover_import_sources([root], _discovery_bounds())

    after = {
        path.relative_to(root).as_posix(): (
            "directory" if path.is_dir() else path.read_bytes()
        )
        for path in root.rglob("*")
    }
    assert discovery.candidates
    assert after == before


def test_discovery_aggregate_copies_collections_into_immutable_tuples() -> None:
    """Frozen discovery results cannot retain mutable caller-owned collections."""
    candidates: list[object] = []
    failures: list[object] = []

    discovery = ImportDiscovery(
        candidates=candidates,  # type: ignore[arg-type]
        failures=failures,  # type: ignore[arg-type]
        root_label=None,
        total_bytes=0,
        entry_count=0,
    )
    candidates.append(object())
    failures.append(object())

    assert discovery.candidates == ()
    assert discovery.failures == ()
    with pytest.raises(FrozenInstanceError):
        discovery.total_bytes = 1  # type: ignore[misc]


def _source_identity(**overrides: object) -> object:
    values: dict[str, object] = {
        "device": 1,
        "inode": 2,
        "mode": 0o100644,
        "size": 4,
        "modified_ns": 5,
        "changed_ns": 6,
    }
    values.update(overrides)
    return note_import_planner.SourceIdentity(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    ["device", "inode", "mode", "size", "modified_ns", "changed_ns"],
)
def test_source_identity_integer_fields_reject_booleans(field_name: str) -> None:
    with pytest.raises(TypeError, match=field_name):
        _source_identity(**{field_name: True})


@pytest.mark.parametrize("field_name", ["device", "inode", "mode", "size"])
def test_source_identity_nonnegative_fields_reject_negative_values(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _source_identity(**{field_name: -1})


def test_source_identity_timestamps_may_be_negative_integers() -> None:
    identity = _source_identity(modified_ns=-2, changed_ns=-1)

    assert identity.modified_ns == -2
    assert identity.changed_ns == -1


def test_discovered_source_defensively_copies_parent_identity_collections() -> None:
    source = ImportSource(
        kind=ImportSourceKind.SELECTED_FILE,
        display_path="note.txt",
        source_path=Path("/private/note.txt"),
    )
    identity = _source_identity(size=4)
    parents = [_source_identity(mode=0o040755, size=0)]

    candidate = note_import_planner.DiscoveredImportSource(
        source=source,
        size_bytes=4,
        identity=identity,
        parent_identities=parents,
    )
    parents.clear()

    assert len(candidate.parent_identities) == 1
    assert isinstance(candidate.parent_identities, tuple)


@pytest.mark.parametrize(
    "overrides",
    [
        {"source": object()},
        {"size_bytes": True},
        {"size_bytes": -1},
        {"identity": object()},
        {"parent_identities": []},
        {"parent_identities": [object()]},
        {"parent_identities": "not-identities"},
        {"size_bytes": 3},
    ],
)
def test_discovered_source_rejects_invalid_exported_record_state(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "source": ImportSource(
            kind=ImportSourceKind.SELECTED_FILE,
            display_path="note.txt",
            source_path=Path("/private/note.txt"),
        ),
        "size_bytes": 4,
        "identity": _source_identity(size=4),
        "parent_identities": [_source_identity(mode=0o040755, size=0)],
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        note_import_planner.DiscoveredImportSource(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("selected_kind", ["file", "directory"])
def test_selected_paths_reject_symlinked_parent_components(
    tmp_path: Path,
    selected_kind: str,
) -> None:
    """Every selected path component is checked without following parent links."""
    real_parent = tmp_path / "real-parent"
    project = real_parent / "Project"
    project.mkdir(parents=True)
    note = project / "note.md"
    note.write_text("note", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    selected = alias / "Project"
    if selected_kind == "file":
        selected /= "note.md"

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([selected], _discovery_bounds())

    assert raised.value.reason_code == "selected_symlink"
    assert str(real_parent) not in str(raised.value)


def test_surrogate_escaped_nested_name_becomes_a_total_safe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undecodable filename bytes cannot crash failure-display encoding."""
    root = tmp_path / "Project"
    root.mkdir()

    class SurrogateScandir:
        yielded = False

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self) -> Self:
            return self

        def __next__(self) -> SimpleNamespace:
            if self.yielded:
                raise StopIteration
            self.yielded = True
            return SimpleNamespace(name="unsafe-\udcff.md")

    monkeypatch.setattr(os, "scandir", lambda _directory_fd: SurrogateScandir())

    discovery = discover_import_sources([root], _discovery_bounds())

    assert discovery.candidates == ()
    assert len(discovery.failures) == 1
    assert discovery.failures[0].reason_code == "nested_unsafe_name"
    assert "%FF" in discovery.failures[0].display_path
    assert str(root) not in repr(discovery.failures[0])


def test_unsafe_failure_display_cannot_collide_with_literal_percent_filename(
    tmp_path: Path,
) -> None:
    """Encoded unsafe names occupy a distinct, deterministic display namespace."""
    root = tmp_path / "Project"
    root.mkdir()
    (root / "unsafe\\name.md").write_text("unsafe", encoding="utf-8")
    (root / "unsafe%5Cname.md").write_text("literal", encoding="utf-8")

    discovery = discover_import_sources([root], _discovery_bounds())

    candidate_path = discovery.candidates[0].source.display_path
    failure_path = discovery.failures[0].display_path
    assert candidate_path == "Project/unsafe%5Cname.md"
    assert failure_path == "Project/.unsafe-entry/unsafe%5Cname.md"
    assert failure_path != candidate_path


@pytest.mark.parametrize("operation", ["fstat", "close"])
def test_root_descriptor_errors_are_normalized_without_raw_os_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Root descriptor inspection and cleanup expose only stable selection errors."""
    root = tmp_path / "Project"
    root.mkdir()
    private_error = "PRIVATE ROOT DESCRIPTOR ERROR"

    if operation == "fstat":
        monkeypatch.setattr(
            note_import_discovery.os,
            "fstat",
            lambda _descriptor: (_ for _ in ()).throw(OSError(private_error)),
        )
    else:
        real_close = os.close

        def close_then_fail(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError(private_error)

        monkeypatch.setattr(note_import_discovery.os, "close", close_then_fail)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _discovery_bounds())

    assert raised.value.reason_code == "selection_unreadable"
    assert private_error not in str(raised.value)
    assert str(root) not in str(raised.value)


def test_unsupported_descriptor_api_fails_closed_with_a_stable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Platforms lacking secure descriptor operations never fall back to path walks."""
    note = tmp_path / "note.md"
    note.write_text("note", encoding="utf-8")
    monkeypatch.setattr(
        note_import_discovery.os,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            NotImplementedError("PRIVATE UNSUPPORTED API")
        ),
    )

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([note], _discovery_bounds())

    assert raised.value.reason_code == "secure_discovery_unavailable"
    assert "PRIVATE" not in str(raised.value)


def test_close_failure_does_not_mask_a_stable_selected_symlink_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup errors cannot replace the safer primary selection diagnosis."""
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    real_close = os.close

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("PRIVATE CLOSE ERROR")

    monkeypatch.setattr(note_import_discovery.os, "close", close_then_fail)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([alias / "missing.md"], _discovery_bounds())

    assert raised.value.reason_code == "selected_symlink"
    assert "PRIVATE" not in str(raised.value)


@pytest.mark.parametrize("operation", ["open", "fstat", "close", "scandir"])
def test_child_descriptor_errors_become_one_safe_nested_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Every child-descriptor failure is normalized and never aborts with OS text."""
    root = tmp_path / "Project"
    child = root / "child"
    child.mkdir(parents=True)
    private_error = "PRIVATE CHILD DESCRIPTOR ERROR"
    real_open = os.open
    real_fstat = os.fstat
    real_close = os.close
    real_scandir = os.scandir
    child_descriptors: set[int] = set()

    def tracked_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == "child" and dir_fd is not None:
            if operation == "open":
                raise OSError(private_error)
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            child_descriptors.add(descriptor)
            return descriptor
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def injected_fstat(descriptor: int) -> os.stat_result:
        if operation == "fstat" and descriptor in child_descriptors:
            raise OSError(private_error)
        return real_fstat(descriptor)

    def injected_close(descriptor: int) -> None:
        if operation == "close" and descriptor in child_descriptors:
            child_descriptors.discard(descriptor)
            real_close(descriptor)
            raise OSError(private_error)
        real_close(descriptor)

    def injected_scandir(descriptor: int) -> "os.ScandirIterator[str]":
        if operation == "scandir" and descriptor in child_descriptors:
            raise OSError(private_error)
        return real_scandir(descriptor)

    monkeypatch.setattr(note_import_discovery.os, "open", tracked_open)
    monkeypatch.setattr(note_import_discovery.os, "fstat", injected_fstat)
    monkeypatch.setattr(note_import_discovery.os, "close", injected_close)
    monkeypatch.setattr(note_import_discovery.os, "scandir", injected_scandir)

    discovery = discover_import_sources([root], _discovery_bounds())

    assert discovery.candidates == ()
    assert len(discovery.failures) == 1
    failure = discovery.failures[0]
    assert failure.display_path == "Project/child"
    assert failure.reason_code == "nested_unavailable"
    assert private_error not in failure.user_message
    assert private_error not in repr(discovery)


@pytest.mark.parametrize("path_text", ["bad\x00name", "bad\ud800name"])
def test_invalid_selected_path_text_is_normalized_without_leaking(
    path_text: str,
) -> None:
    """Invalid lexical or filesystem text exposes only a stable generic error."""
    bounds = _discovery_bounds(max_reason_length=32)

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([Path(path_text)], bounds)

    assert raised.value.reason_code == "invalid_selection"
    assert 0 < len(raised.value.user_message) <= 32
    assert path_text not in str(raised.value)
    assert path_text not in repr(raised.value)


def test_selected_file_names_reject_canonical_unicode_ambiguity(
    tmp_path: Path,
) -> None:
    """Canonically equivalent basenames cannot produce two manual display paths."""
    composed = tmp_path / "composed" / "Caf\N{LATIN SMALL LETTER E WITH ACUTE}.md"
    decomposed = tmp_path / "decomposed" / "Cafe\N{COMBINING ACUTE ACCENT}.md"
    composed.parent.mkdir()
    decomposed.parent.mkdir()
    composed.write_text("one", encoding="utf-8")
    decomposed.write_text("two", encoding="utf-8")

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([composed, decomposed], _discovery_bounds())

    assert raised.value.reason_code == "ambiguous_display_path"
    assert str(tmp_path) not in str(raised.value)


class _DiscoveryInterruption(BaseException):
    """Synthetic non-Exception interruption for descriptor cleanup coverage."""


@pytest.mark.parametrize(
    ("boundary", "error_type"),
    [
        ("inspect", RuntimeError),
        ("verified_open", RuntimeError),
        ("root_scan", RuntimeError),
        ("child_scan", RuntimeError),
        ("root_scan", _DiscoveryInterruption),
    ],
)
def test_unexpected_discovery_errors_close_every_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    error_type: type[BaseException],
) -> None:
    """Every lexical descriptor owner cleans up before unexpected propagation."""
    root = tmp_path / "Project"
    (root / "child").mkdir(parents=True)
    real_open = os.open
    real_fstat = os.fstat
    real_close = os.close
    real_scandir = os.scandir
    opened: list[int] = []
    closed: list[int] = []
    project_fds: set[int] = set()
    child_fds: set[int] = set()
    inspect_injected = False

    def tracked_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        if os.fspath(path) == "Project" and dir_fd is not None:
            project_fds.add(descriptor)
        if os.fspath(path) == "child" and dir_fd is not None:
            child_fds.add(descriptor)
        return descriptor

    def injected_fstat(descriptor: int) -> os.stat_result:
        nonlocal inspect_injected
        should_raise = False
        if boundary == "inspect" and not inspect_injected:
            inspect_injected = True
            should_raise = True
        elif (boundary == "verified_open" and descriptor in project_fds) or (
            boundary == "child_scan" and descriptor in child_fds
        ):
            should_raise = True
        if should_raise:
            raise error_type("unexpected discovery failure")
        return real_fstat(descriptor)

    def tracked_close(descriptor: int) -> None:
        real_close(descriptor)
        closed.append(descriptor)

    def injected_scandir(descriptor: int) -> "os.ScandirIterator[str]":
        if boundary == "root_scan" and descriptor in project_fds:
            raise error_type("unexpected discovery failure")
        return real_scandir(descriptor)

    monkeypatch.setattr(note_import_discovery.os, "open", tracked_open)
    monkeypatch.setattr(note_import_discovery.os, "fstat", injected_fstat)
    monkeypatch.setattr(note_import_discovery.os, "close", tracked_close)
    monkeypatch.setattr(note_import_discovery.os, "scandir", injected_scandir)

    with pytest.raises(error_type):
        discover_import_sources([root], _discovery_bounds())

    assert Counter(closed) == Counter(opened)


@pytest.mark.parametrize(
    "folder_name",
    [
        "Project\N{FULLWIDTH SOLIDUS}Archive",
        "Project\N{FULLWIDTH REVERSE SOLIDUS}Archive",
    ],
)
def test_selected_root_uses_canonical_folder_name_validation(
    tmp_path: Path,
    folder_name: str,
) -> None:
    """A root whose NFKC key becomes a path segment is rejected fatally."""
    root = tmp_path / folder_name
    root.mkdir()

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _discovery_bounds())

    assert raised.value.reason_code == "unsafe_display_path"
    assert str(root) not in str(raised.value)


@pytest.mark.parametrize(
    "folder_name",
    ["nested\N{FULLWIDTH SOLIDUS}name", "nested\N{FULLWIDTH REVERSE SOLIDUS}name"],
)
def test_invalid_canonical_nested_folder_is_one_untraversed_failure(
    tmp_path: Path,
    folder_name: str,
) -> None:
    """Invalid nested folder segments are visible once and never traversed."""
    root = tmp_path / "Project"
    nested = root / folder_name
    nested.mkdir(parents=True)
    (nested / "secret.md").write_text("secret", encoding="utf-8")

    discovery = discover_import_sources([root], _discovery_bounds())

    assert discovery.candidates == ()
    assert len(discovery.failures) == 1
    assert discovery.failures[0].reason_code == "nested_unsafe_name"
    assert "secret.md" not in repr(discovery)


def test_canonically_equivalent_sibling_folders_reject_the_whole_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling folders cannot silently merge through normalized folder keys."""
    root = tmp_path / "Project"
    root.mkdir()
    directory_metadata = root.stat()

    class FolderEntry:
        def __init__(self, name: str) -> None:
            self.name = name

        def stat(self, *, follow_symlinks: bool) -> os.stat_result:
            assert follow_symlinks is False
            return directory_metadata

    class FolderScandir:
        def __init__(self) -> None:
            self.entries = iter(
                [
                    FolderEntry("Caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
                    FolderEntry("Cafe\N{COMBINING ACUTE ACCENT}"),
                ]
            )

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self) -> Self:
            return self

        def __next__(self) -> FolderEntry:
            return next(self.entries)

    monkeypatch.setattr(os, "scandir", lambda _descriptor: FolderScandir())

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([root], _discovery_bounds())

    assert raised.value.reason_code == "ambiguous_folder_path"
    assert str(root) not in str(raised.value)


def test_import_bounds_reject_depth_above_absolute_runtime_ceiling() -> None:
    """Configured depth cannot exceed the recursion-safe absolute ceiling."""
    assert note_import_plan_models.MAX_IMPORT_DEPTH == 64
    with pytest.raises(ValueError, match="max_depth"):
        _discovery_bounds(max_depth=65)


@pytest.mark.parametrize("race_target", ["selected_parent", "nested_child"])
def test_directory_identity_replacement_races_fail_safely_and_close_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_target: str,
) -> None:
    """Stat/open directory replacement races never admit the replacement tree."""
    root = tmp_path / "Project"
    watched_parent = tmp_path / "watched-parent"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    if race_target == "selected_parent":
        root = watched_parent / "Project"
        root.mkdir(parents=True)
        selected = root
        replaced_name = "watched-parent"
    else:
        (root / "child").mkdir(parents=True)
        selected = root
        replaced_name = "child"

    real_open = os.open
    real_close = os.close
    opened: list[int] = []
    closed: list[int] = []

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == replaced_name and dir_fd is not None:
            descriptor = real_open(replacement, flags)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor: int) -> None:
        real_close(descriptor)
        closed.append(descriptor)

    monkeypatch.setattr(note_import_discovery.os, "open", racing_open)
    monkeypatch.setattr(note_import_discovery.os, "close", tracked_close)

    if race_target == "selected_parent":
        with pytest.raises(ImportSelectionError) as raised:
            discover_import_sources([selected], _discovery_bounds())
        assert raised.value.reason_code == "selection_changed"
    else:
        discovery = discover_import_sources([selected], _discovery_bounds())
        assert discovery.candidates == ()
        assert len(discovery.failures) == 1
        assert discovery.failures[0].reason_code == "nested_unavailable"

    assert Counter(closed) == Counter(opened)


def test_missing_secure_discovery_capability_has_a_distinct_stable_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability absence is distinguishable from an unreadable selected path."""
    note = tmp_path / "note.md"
    note.write_text("note", encoding="utf-8")
    monkeypatch.delattr(note_import_discovery.os, "O_NOFOLLOW")

    with pytest.raises(ImportSelectionError) as raised:
        discover_import_sources([note], _discovery_bounds())

    assert raised.value.reason_code == "secure_discovery_unavailable"
    assert str(note) not in str(raised.value)


def test_failed_entry_stat_is_not_retried_into_an_ambiguous_sibling_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One scan pass uses one metadata snapshot and never traverses a reappeared entry."""
    root = tmp_path / "Project"
    root.mkdir()
    directory_metadata = root.stat()
    decomposed_name = "Cafe\N{COMBINING ACUTE ACCENT}"

    class RacingEntry:
        def __init__(self, name: str, *, fail_first: bool = False) -> None:
            self.name = name
            self.fail_first = fail_first
            self.stat_calls = 0

        def stat(self, *, follow_symlinks: bool) -> os.stat_result:
            assert follow_symlinks is False
            self.stat_calls += 1
            if self.fail_first and self.stat_calls == 1:
                raise FileNotFoundError("entry disappeared")
            return directory_metadata

    composed = RacingEntry("Caf\N{LATIN SMALL LETTER E WITH ACUTE}")
    decomposed = RacingEntry(decomposed_name, fail_first=True)

    class RacingScandir:
        def __init__(self) -> None:
            self.entries = iter([composed, decomposed])

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self) -> Self:
            return self

        def __next__(self) -> RacingEntry:
            return next(self.entries)

    real_open = os.open

    def guarded_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == decomposed_name and dir_fd is not None:
            raise AssertionError("reappeared entry was traversed")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "scandir", lambda _descriptor: RacingScandir())
    monkeypatch.setattr(note_import_discovery.os, "open", guarded_open)

    discovery = discover_import_sources([root], _discovery_bounds())

    assert discovery.candidates == ()
    assert composed.stat_calls == 1
    assert decomposed.stat_calls == 1
    assert [failure.reason_code for failure in discovery.failures] == [
        "nested_unavailable",
        "nested_unavailable",
    ]


def _parse_selection(
    paths: list[Path],
    *,
    bounds: ImportBounds | None = None,
    destination: tuple[str, ...] | None = None,
) -> object:
    active_bounds = bounds or _discovery_bounds()
    discovery = discover_import_sources(paths, active_bounds)
    return note_import_planner.parse_import_sources(
        discovery,
        active_bounds,
        destination_folder_segments=destination,
    )


def test_text_and_markdown_parsing_retains_content_and_uses_safe_titles(
    tmp_path: Path,
) -> None:
    """Text uses the stem while Markdown may use an early level-one heading."""
    root = tmp_path / "Project"
    root.mkdir()
    text = root / "plain.txt"
    markdown = root / "guide.md"
    text.write_text("Full plain content", encoding="utf-8")
    markdown_content = "intro\n# Guide title\nbody\n"
    markdown.write_text(markdown_content, encoding="utf-8")

    batch = _parse_selection([root])

    assert [source.payloads[0].title for source in batch.parsed] == [
        "Guide title",
        "plain",
    ]
    assert batch.parsed[0].payloads[0].content == markdown_content
    assert batch.parsed[1].payloads[0].content == "Full plain content"


@pytest.mark.parametrize("extension", [".json", ".yaml", ".yml"])
def test_structured_sources_expand_atomically_with_bounded_metadata(
    tmp_path: Path,
    extension: str,
) -> None:
    """JSON and YAML mappings expand into immutable note payload tuples."""
    root = tmp_path / "Project"
    child = root / "child"
    child.mkdir(parents=True)
    source = child / f"notes{extension}"
    if extension == ".json":
        source.write_text(
            '[{"title":"One","content":"First","tags":["a","b"],'
            '"template":"Meeting"},{"name":"Two","body":"Second"}]',
            encoding="utf-8",
        )
    else:
        source.write_text(
            "- title: One\n  content: First\n  tags: [a, b]\n"
            "  template: Meeting\n- name: Two\n  body: Second\n",
            encoding="utf-8",
        )

    batch = _parse_selection([root])

    assert batch.issues == ()
    parsed = batch.parsed[0]
    assert tuple(payload.title for payload in parsed.payloads) == ("One", "Two")
    assert parsed.payloads[0].keywords == ("a", "b")
    assert parsed.payloads[0].template_name == "Meeting"
    assert tuple(membership.payload_index for membership in parsed.memberships) == (
        0,
        1,
    )
    assert all(
        membership.folder_segments == ("Project", "child")
        for membership in parsed.memberships
    )
    assert isinstance(parsed.payloads, tuple)
    assert isinstance(parsed.memberships, tuple)


def test_csv_uses_recognized_columns_and_rejects_invalid_rows_atomically(
    tmp_path: Path,
) -> None:
    """CSV recognizes semantic headers and never returns a partial row import."""
    valid = tmp_path / "valid.csv"
    invalid = tmp_path / "invalid.csv"
    valid.write_text(
        "BODY,NAME,TAGS\nFirst,One,alpha\nSecond,Two,beta\n",
        encoding="utf-8",
    )
    invalid.write_text("title,content\nGood,Body\nBad,\n", encoding="utf-8")

    batch = _parse_selection(
        [valid, invalid],
        destination=("Imported",),
    )

    assert len(batch.parsed) == 1
    assert tuple(payload.title for payload in batch.parsed[0].payloads) == (
        "One",
        "Two",
    )
    assert batch.parsed[0].payloads[0].keywords == ("alpha",)
    assert len(batch.issues) == 1
    assert batch.issues[0].classification is ImportClassification.FAILED
    assert batch.issues[0].reason_code == "invalid_content"


def test_csv_content_field_above_stdlib_default_parses_within_file_bounds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large-field.csv"
    content = "B" * 131_073
    source.write_text(f"title,content\nOne,{content}\n", encoding="utf-8")
    bounds = _discovery_bounds(max_file_bytes=200_000, max_total_bytes=300_000)
    original_limit = csv.field_size_limit()
    csv.field_size_limit(131_072)
    try:
        batch = _parse_selection(
            [source],
            bounds=bounds,
            destination=("Imported",),
        )
    finally:
        csv.field_size_limit(original_limit)

    assert batch.issues == ()
    assert batch.parsed[0].payloads[0].content == content


@pytest.mark.parametrize("valid_csv", [True, False])
def test_csv_field_limit_is_restored_after_success_and_parser_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_csv: bool,
) -> None:
    source = tmp_path / "field-limit.csv"
    field = "B" * 2_048
    if valid_csv:
        source.write_text(f"title,content\nOne,{field}\n", encoding="utf-8")
    else:
        source.write_text(f'title,content\nOne,"{field}', encoding="utf-8")
    bounds = _discovery_bounds(max_file_bytes=10_000, max_total_bytes=20_000)
    real_field_size_limit = csv.field_size_limit
    process_limit = real_field_size_limit()
    real_field_size_limit(1_024)
    calls: list[int] = []

    def tracking_field_size_limit(new_limit: int | None = None) -> int:
        if new_limit is None:
            return real_field_size_limit()
        calls.append(new_limit)
        return real_field_size_limit(new_limit)

    monkeypatch.setattr(
        note_import_parsers.csv,
        "field_size_limit",
        tracking_field_size_limit,
    )
    try:
        batch = _parse_selection(
            [source],
            bounds=bounds,
            destination=("Imported",),
        )
        restored_limit = real_field_size_limit()
    finally:
        real_field_size_limit(process_limit)

    assert calls == [bounds.max_file_bytes, 1_024]
    assert restored_limit == 1_024
    if valid_csv:
        assert batch.issues == ()
        assert batch.parsed[0].payloads[0].content == field
    else:
        assert batch.parsed == ()
        assert batch.issues[0].reason_code == "invalid_content"


def test_csv_falls_back_to_first_two_distinct_columns(tmp_path: Path) -> None:
    source = tmp_path / "notes.csv"
    source.write_text("subject,text\nOne,Body\n", encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed[0].payloads[0] == ParsedNotePayload(
        title="One",
        content="Body",
    )


def test_csv_generic_fallback_skips_reserved_metadata_columns(tmp_path: Path) -> None:
    source = tmp_path / "notes.csv"
    source.write_text(
        "tags,template,subject,text\nalpha,Meeting,One,Body\n",
        encoding="utf-8",
    )

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.issues == ()
    assert batch.parsed[0].payloads[0] == ParsedNotePayload(
        title="One",
        content="Body",
        keywords=("alpha",),
        template_name="Meeting",
    )


@pytest.mark.parametrize(
    "csv_content",
    [
        "tags,template\nalpha,Meeting\n",
        "tags,subject\nalpha,One\n",
    ],
)
def test_csv_generic_fallback_requires_two_unreserved_columns(
    tmp_path: Path,
    csv_content: str,
) -> None:
    source = tmp_path / "notes.csv"
    source.write_text(csv_content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "invalid_content"


@pytest.mark.parametrize(
    ("csv_content", "expected_title", "expected_content"),
    [
        ("body,subject\nBody,One\n", "One", "Body"),
        ("text,name\nBody,One\n", "One", "Body"),
    ],
)
def test_csv_partial_semantic_headers_preserve_the_recognized_role(
    tmp_path: Path,
    csv_content: str,
    expected_title: str,
    expected_content: str,
) -> None:
    source = tmp_path / "notes.csv"
    source.write_text(csv_content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    payload = batch.parsed[0].payloads[0]
    assert payload.title == expected_title
    assert payload.content == expected_content


@pytest.mark.parametrize(
    ("content", "reason_code"),
    [
        ('{"title":"One","title":"Two","content":"Body"}', "invalid_content"),
        ("title: One\ntitle: Two\ncontent: Body\n", "invalid_content"),
        ("base: &base {content: Body}\nnote: *base\n", "invalid_content"),
        ("[]", "empty_structured_source"),
    ],
)
def test_structured_sources_reject_duplicates_aliases_and_empty_results(
    tmp_path: Path,
    content: str,
    reason_code: str,
) -> None:
    suffix = ".json" if content.startswith("{") or content == "[]" else ".yaml"
    source = tmp_path / f"notes{suffix}"
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == reason_code


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        (
            "notes.json",
            '{"title":"One","name":"One","content":"Body"}',
        ),
        (
            "notes.json",
            '{"title":"One","content":"Body","body":"Body"}',
        ),
        (
            "notes.json",
            '{"content":"Body","keywords":["a"],"tags":["a"]}',
        ),
        ("notes.yaml", "title: One\nname: One\ncontent: Body\n"),
        ("notes.yaml", "title: One\ncontent: Body\nbody: Body\n"),
        ("notes.yaml", "content: Body\nkeywords: [a]\ntags: [a]\n"),
    ],
)
def test_structured_mapping_rejects_conflicting_semantic_aliases_atomically(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    source = tmp_path / filename
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "invalid_content"


def test_duplicate_normalized_csv_headers_fail_atomically(tmp_path: Path) -> None:
    source = tmp_path / "notes.csv"
    source.write_text("Title, title ,content\nOne,Other,Body\n", encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "invalid_content"


@pytest.mark.parametrize(
    "csv_content",
    [
        "title,name,content\nOne,One,Body\n",
        "title,content,body\nOne,Body,Body\n",
        "title,content,keywords,tags\nOne,Body,a,a\n",
    ],
)
def test_csv_rejects_multiple_headers_for_one_semantic_role(
    tmp_path: Path,
    csv_content: str,
) -> None:
    source = tmp_path / "notes.csv"
    source.write_text(csv_content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "invalid_content"


@pytest.mark.parametrize(
    "csv_content",
    [
        "title,tags\nOne,alpha\n",
        "body,template\nBody,Meeting\n",
    ],
)
def test_csv_partial_semantic_role_requires_an_unreserved_fallback_column(
    tmp_path: Path,
    csv_content: str,
) -> None:
    source = tmp_path / "notes.csv"
    source.write_text(csv_content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "invalid_content"


@pytest.mark.parametrize(
    ("csv_content", "expected_title", "expected_content"),
    [
        ("body,tags,subject\nBody,alpha,One\n", "One", "Body"),
        ("name,template,text\nOne,Meeting,Body\n", "One", "Body"),
    ],
)
def test_csv_partial_semantic_role_skips_reserved_fallback_columns(
    tmp_path: Path,
    csv_content: str,
    expected_title: str,
    expected_content: str,
) -> None:
    source = tmp_path / "notes.csv"
    source.write_text(csv_content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    payload = batch.parsed[0].payloads[0]
    assert payload.title == expected_title
    assert payload.content == expected_content


def test_unsupported_candidates_are_not_opened_and_have_safe_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "private.bin"
    source.write_bytes(b"secret")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([source], bounds)

    monkeypatch.setattr(
        note_import_discovery.os,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported source was opened")
        ),
    )
    batch = note_import_planner.parse_import_sources(
        discovery,
        bounds,
        destination_folder_segments=("Imported",),
    )

    assert batch.parsed == ()
    assert batch.proposed_folder_paths == ()
    assert batch.issues[0].classification is ImportClassification.UNSUPPORTED
    assert batch.issues[0].reason_code == "unsupported_extension"
    assert str(tmp_path) not in repr(batch)
    assert "secret" not in repr(batch)


@pytest.mark.parametrize(
    ("raw_content", "reason_code"),
    [
        (b"\xff\xfe", "invalid_utf8"),
        (b"{malformed", "invalid_content"),
    ],
)
def test_invalid_utf8_and_malformed_content_become_safe_failures(
    tmp_path: Path,
    raw_content: bytes,
    reason_code: str,
) -> None:
    suffix = ".txt" if reason_code == "invalid_utf8" else ".json"
    source = tmp_path / f"private{suffix}"
    source.write_bytes(raw_content)

    batch = _parse_selection([source], destination=("Imported",))

    issue = batch.issues[0]
    assert issue.classification is ImportClassification.FAILED
    assert issue.reason_code == reason_code
    assert len(issue.user_message) <= _discovery_bounds().max_reason_length
    assert str(tmp_path) not in repr(issue)
    assert repr(raw_content) not in repr(issue)


def test_utf8_bom_is_accepted_without_entering_note_content(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_bytes(b"\xef\xbb\xbfBody")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed[0].payloads[0].content == "Body"


@pytest.mark.parametrize("extension", [".txt", ".md", ".markdown"])
@pytest.mark.parametrize("content", ["", " \n\t "])
def test_plain_sources_reject_empty_or_whitespace_only_content(
    tmp_path: Path,
    extension: str,
    content: str,
) -> None:
    source = tmp_path / f"note{extension}"
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].classification is ImportClassification.FAILED
    assert batch.issues[0].reason_code == "invalid_content"


def _classification_plan(
    batch: note_import_planner.ParsedImportBatch,
    *observations: note_import_planner.PriorImportObservation,
) -> NoteImportPlan:
    return note_import_planner.classify_import_batch(
        batch,
        _discovery_bounds(),
        prior_observations=observations,
    )


def _prior_observation(
    display_path: str,
    *,
    fingerprint: str | None,
    kind: ImportMatchKind = ImportMatchKind.EXACT,
    note_id: str = "note-7",
    note_version: int | None = 4,
) -> note_import_planner.PriorImportObservation:
    return note_import_planner.PriorImportObservation(
        display_path=display_path,
        match_kind=kind,
        note_id=note_id,
        note_version=note_version,
        payload_fingerprint=fingerprint,
    )


def test_classify_import_batch_applies_safe_defaults_for_new_and_repeat_sources(
    tmp_path: Path,
) -> None:
    unchanged = tmp_path / "unchanged.md"
    changed = tmp_path / "changed.md"
    new = tmp_path / "new.md"
    unchanged.write_text("# Same\nBody", encoding="utf-8")
    changed.write_text("# Changed\nNew body", encoding="utf-8")
    new.write_text("# New\nBody", encoding="utf-8")
    batch = _parse_selection([new, unchanged, changed], destination=("Imported",))
    parsed = {source.candidate.source.display_path: source for source in batch.parsed}

    plan = _classification_plan(
        batch,
        _prior_observation(
            "unchanged.md",
            fingerprint=note_import_planner._private_payload_fingerprint(
                parsed["unchanged.md"].payloads
            ),
            note_id="same-note",
        ),
        _prior_observation(
            "changed.md",
            fingerprint=note_import_planner._private_payload_fingerprint(
                (ParsedNotePayload(title="Old", content="Old body"),)
            ),
            note_id="changed-note",
        ),
    )

    items = {item.source.display_path: item for item in plan.items}
    assert items["new.md"].classification is ImportClassification.NEW
    assert items["new.md"].default_action is ImportAction.CREATE_NEW
    assert items["new.md"].selected_action is ImportAction.CREATE_NEW
    assert items["new.md"].allowed_actions == (
        ImportAction.SKIP,
        ImportAction.CREATE_NEW,
    )
    assert items["new.md"].match is None

    unchanged_item = items["unchanged.md"]
    assert unchanged_item.classification is ImportClassification.UNCHANGED_REPEAT
    assert unchanged_item.default_action is ImportAction.SKIP
    assert unchanged_item.selected_action is ImportAction.SKIP
    assert unchanged_item.allowed_actions == (
        ImportAction.SKIP,
        ImportAction.CREATE_NEW,
        ImportAction.UPDATE_EXISTING,
    )
    assert unchanged_item.match == ImportMatch(
        kind=ImportMatchKind.EXACT,
        note_id="same-note",
        note_version=4,
    )
    assert not unchanged_item.replace_content
    assert not unchanged_item.add_membership

    changed_item = items["changed.md"]
    assert changed_item.classification is ImportClassification.CHANGED_REPEAT
    assert changed_item.default_action is ImportAction.CREATE_NEW
    assert changed_item.selected_action is ImportAction.CREATE_NEW
    assert changed_item.allowed_actions == (
        ImportAction.SKIP,
        ImportAction.CREATE_NEW,
        ImportAction.UPDATE_EXISTING,
    )
    assert changed_item.match is not None
    assert changed_item.match.kind is ImportMatchKind.EXACT
    assert not changed_item.replace_content
    assert changed_item.add_membership


def test_classify_import_batch_keeps_uncertain_matches_create_only_by_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "possible.md"
    source.write_text("# Possible\nBody", encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))

    plan = _classification_plan(
        batch,
        _prior_observation(
            "possible.md",
            kind=ImportMatchKind.UNCERTAIN,
            fingerprint=None,
            note_id="possible-note",
        ),
    )

    item = plan.items[0]
    assert item.classification is ImportClassification.UNCERTAIN_MATCH
    assert item.default_action is ImportAction.CREATE_NEW
    assert item.selected_action is ImportAction.CREATE_NEW
    assert item.allowed_actions == (ImportAction.SKIP, ImportAction.CREATE_NEW)
    assert item.match == ImportMatch(
        kind=ImportMatchKind.UNCERTAIN,
        note_id="possible-note",
        note_version=4,
    )
    assert ImportAction.UPDATE_EXISTING not in item.allowed_actions


def test_classify_import_batch_turns_parse_issues_into_skip_only_items(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "archive.bin"
    malformed = tmp_path / "broken.json"
    unsupported.write_bytes(b"not imported")
    malformed.write_text("{private malformed body", encoding="utf-8")
    batch = _parse_selection([unsupported, malformed], destination=("Imported",))

    plan = _classification_plan(batch)

    items = {item.source.display_path: item for item in plan.items}
    assert items["archive.bin"].classification is ImportClassification.UNSUPPORTED
    assert items["broken.json"].classification is ImportClassification.FAILED
    for item in items.values():
        assert item.default_action is ImportAction.SKIP
        assert item.selected_action is ImportAction.SKIP
        assert item.allowed_actions == (ImportAction.SKIP,)
        assert item.payloads == ()
        assert item.memberships == ()
        assert item.match is None
        assert not item.replace_content
        assert not item.add_membership
        assert len(item.reason) <= _discovery_bounds().max_reason_length


def test_private_payload_fingerprint_is_exact_and_payload_sensitive() -> None:
    first = ParsedNotePayload(
        title="Cafe\u0301",
        content="Body",
        keywords=("one", "two"),
        template_name="Meeting",
    )
    canonically_equivalent = ParsedNotePayload(
        title="Caf\u00e9",
        content="Body",
        keywords=("one", "two"),
        template_name="Meeting",
    )
    changed = ParsedNotePayload(
        title="Caf\u00e9",
        content="Changed body",
        keywords=("one", "two"),
        template_name="Meeting",
    )

    fingerprint = note_import_planner._private_payload_fingerprint((first,))

    assert fingerprint != note_import_planner._private_payload_fingerprint(
        [canonically_equivalent]
    )
    assert fingerprint != note_import_planner._private_payload_fingerprint((changed,))
    assert len(fingerprint) == 64
    assert fingerprint == fingerprint.casefold()


def test_private_payload_fingerprint_preserves_structural_type_boundaries() -> None:
    keyword_value = (
        ParsedNotePayload(
            title="Title",
            content="Body",
            keywords=("Meeting",),
            template_name=None,
        ),
    )
    template_value = (
        ParsedNotePayload(
            title="Title",
            content="Body",
            keywords=(),
            template_name="Meeting",
        ),
    )
    one_payload_with_delimiters = (
        ParsedNotePayload(title='One"},{"title":"Two', content="A|B"),
    )
    two_payloads = (
        ParsedNotePayload(title="One", content="A"),
        ParsedNotePayload(title="Two", content="B"),
    )

    fingerprints = {
        note_import_planner._private_payload_fingerprint(keyword_value),
        note_import_planner._private_payload_fingerprint(template_value),
        note_import_planner._private_payload_fingerprint(one_payload_with_delimiters),
        note_import_planner._private_payload_fingerprint(two_payloads),
    }

    assert len(fingerprints) == 4


@pytest.mark.parametrize(
    "observations",
    [
        "not-an-observation-collection",
        (object(),),
    ],
)
def test_classify_import_batch_rejects_invalid_observation_collection_shape(
    tmp_path: Path,
    observations: object,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))

    with pytest.raises((TypeError, ValueError), match="observation"):
        note_import_planner.classify_import_batch(
            batch,
            _discovery_bounds(),
            prior_observations=observations,  # type: ignore[arg-type]
        )


def test_prior_observations_reject_duplicates_and_unknown_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))
    fingerprint = note_import_planner._private_payload_fingerprint(
        batch.parsed[0].payloads
    )
    observation = _prior_observation("note.md", fingerprint=fingerprint)

    with pytest.raises(ValueError, match="duplicate"):
        _classification_plan(batch, observation, observation)
    with pytest.raises(ValueError, match="unknown"):
        _classification_plan(
            batch,
            _prior_observation("other.md", fingerprint=fingerprint),
        )


def test_prior_observation_keys_match_and_deduplicate_by_nfc_form(
    tmp_path: Path,
) -> None:
    decomposed_name = "Cafe\u0301.md"
    source = tmp_path / decomposed_name
    source.write_text("Body", encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))
    fingerprint = note_import_planner._private_payload_fingerprint(
        batch.parsed[0].payloads
    )
    decomposed = _prior_observation(decomposed_name, fingerprint=fingerprint)
    composed = _prior_observation("Caf\u00e9.md", fingerprint=fingerprint)

    plan = _classification_plan(batch, composed)

    assert plan.items[0].classification is ImportClassification.UNCHANGED_REPEAT
    with pytest.raises(ValueError, match="duplicate"):
        _classification_plan(batch, decomposed, composed)


@pytest.mark.parametrize(
    "values",
    [
        {
            "display_path": "/private/absolute.md",
            "match_kind": ImportMatchKind.EXACT,
            "payload_fingerprint": "a" * 64,
        },
        {
            "display_path": "note.md",
            "match_kind": ImportMatchKind.EXACT,
            "payload_fingerprint": "not-a-sha256",
        },
        {
            "display_path": "note.md",
            "match_kind": ImportMatchKind.UNCERTAIN,
            "payload_fingerprint": "a" * 64,
        },
        {
            "display_path": "note.md",
            "match_kind": ImportMatchKind.USER_CONFIRMED,
            "payload_fingerprint": None,
        },
    ],
)
def test_prior_observation_rejects_unsafe_or_contradictory_shape(
    values: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        note_import_planner.PriorImportObservation(
            note_id="note-7",
            note_version=1,
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("match_kind", "exact"),
        ("note_id", "/private/note-id"),
        ("note_id", "note id with spaces"),
    ],
)
def test_prior_observation_rejects_coerced_enums_and_unsafe_note_ids(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "display_path": "note.md",
        "match_kind": ImportMatchKind.EXACT,
        "note_id": "note-7",
        "note_version": 1,
        "payload_fingerprint": "a" * 64,
    }
    values[field_name] = value

    with pytest.raises((TypeError, ValueError)):
        note_import_planner.PriorImportObservation(**values)  # type: ignore[arg-type]


def test_observation_cardinality_is_bounded_before_consuming_extra_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))
    fingerprint = note_import_planner._private_payload_fingerprint(
        batch.parsed[0].payloads
    )

    def observations() -> object:
        yield _prior_observation("note.md", fingerprint=fingerprint)
        yield _prior_observation("other.md", fingerprint=fingerprint)
        raise AssertionError("classifier consumed observations past its bound")

    with pytest.raises(ValueError, match="too many"):
        note_import_planner.classify_import_batch(
            batch,
            _discovery_bounds(),
            prior_observations=observations(),  # type: ignore[arg-type]
        )


def test_observation_iterator_errors_are_sanitized(
    tmp_path: Path,
) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))
    private_error = f"SOURCE-SECRET {tmp_path} sha256:deadbeef"

    def broken_observations() -> object:
        raise ValueError(private_error)
        yield  # pragma: no cover

    with pytest.raises(ValueError, match="could not be read safely") as raised:
        note_import_planner.classify_import_batch(
            batch,
            _discovery_bounds(),
            prior_observations=broken_observations(),  # type: ignore[arg-type]
        )

    assert private_error not in str(raised.value)
    assert private_error not in repr(raised.value)


def test_observation_iter_creation_errors_are_sanitized(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_text("Body", encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))
    private_error = f"SOURCE-SECRET {tmp_path} sha256:deadbeef"

    class BrokenObservations:
        def __iter__(self) -> object:
            raise ValueError(private_error)

    with pytest.raises(ValueError, match="could not be read safely") as raised:
        note_import_planner.classify_import_batch(
            batch,
            _discovery_bounds(),
            prior_observations=BrokenObservations(),  # type: ignore[arg-type]
        )

    assert private_error not in repr(raised.value)


def test_private_fingerprint_helper_is_not_part_of_public_planner_exports() -> None:
    assert "_private_payload_fingerprint" not in note_import_planner.__all__
    assert "private_payload_fingerprint" not in note_import_planner.__all__


def test_multi_note_source_level_observation_is_fail_safe_uncertain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "many.json"
    source.write_text(
        '[{"title":"One","content":"A"},{"title":"Two","content":"B"}]',
        encoding="utf-8",
    )
    batch = _parse_selection([source], destination=("Imported",))
    observation = _prior_observation(
        "many.json",
        fingerprint=note_import_planner._private_payload_fingerprint(
            batch.parsed[0].payloads
        ),
    )

    item = _classification_plan(batch, observation).items[0]

    assert len(item.payloads) == 2
    assert item.classification is ImportClassification.UNCERTAIN_MATCH
    assert item.match is not None
    assert item.match.kind is ImportMatchKind.UNCERTAIN
    assert item.selected_action is ImportAction.CREATE_NEW
    assert item.allowed_actions == (ImportAction.SKIP, ImportAction.CREATE_NEW)


def test_classification_is_deterministic_and_preserves_hierarchy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project"
    nested = root / "Nested"
    nested.mkdir(parents=True)
    (root / "z.md").write_text("Z", encoding="utf-8")
    (nested / "a.md").write_text("A", encoding="utf-8")
    batch = _parse_selection([root])
    scrambled = note_import_planner.ParsedImportBatch(
        parsed=tuple(reversed(batch.parsed)),
        issues=tuple(reversed(batch.issues)),
        proposed_folder_paths=batch.proposed_folder_paths,
    )

    first = _classification_plan(batch)
    second = _classification_plan(scrambled)

    assert tuple(item.item_id for item in first.items) == tuple(
        item.item_id for item in second.items
    )
    assert tuple(item.source.display_path for item in first.items) == tuple(
        item.source.display_path for item in second.items
    )
    assert tuple(item.memberships for item in first.items) == tuple(
        item.memberships for item in second.items
    )
    assert first.proposed_folder_paths == batch.proposed_folder_paths
    with pytest.raises(FrozenInstanceError):
        first.items[0].classification = ImportClassification.FAILED  # type: ignore[misc]


def test_classification_privacy_excludes_private_values_from_repr_diagnostics_and_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "private.md"
    private_content = "SOURCE-CONTENT-SECRET"
    source.write_text(private_content, encoding="utf-8")
    batch = _parse_selection([source], destination=("Imported",))
    private_fingerprint = note_import_planner._private_payload_fingerprint(
        batch.parsed[0].payloads
    )
    observation = _prior_observation(
        "private.md",
        fingerprint=private_fingerprint,
    )

    plan = _classification_plan(batch, observation)
    rendered = f"{plan!r} {plan.to_diagnostic()!r} {observation!r}"
    log_text = " ".join(record.getMessage() for record in caplog.records)

    for private_value in (
        private_fingerprint,
        str(tmp_path),
        private_content,
        "PermissionError",
    ):
        assert private_value not in rendered
        assert private_value not in log_text


def test_classification_replaces_caller_issue_text_with_bounded_safe_reason(
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "private.json"
    raw_exception = f"PermissionError: denied {private_path} SOURCE-SECRET"
    issue = note_import_planner.ImportParseIssue(
        display_path="private.json",
        source_path=private_path,
        classification=ImportClassification.FAILED,
        reason_code="source_unavailable",
        user_message=raw_exception,
    )
    batch = note_import_planner.ParsedImportBatch(
        parsed=(),
        issues=(issue,),
        proposed_folder_paths=(),
    )
    bounds = ImportBounds(
        max_files=10,
        max_file_bytes=1_000,
        max_total_bytes=10_000,
        max_depth=4,
        max_reason_length=12,
    )

    plan = note_import_planner.classify_import_batch(batch, bounds)

    assert 0 < len(plan.items[0].reason) <= 12
    rendered = repr(plan)
    assert raw_exception not in rendered
    assert str(tmp_path) not in rendered
    assert "SOURCE-SECRET" not in rendered


def test_exact_observation_fingerprinting_handles_json_lone_surrogates_privately(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "surrogate.json"
    source.write_text(
        '{"title":"Private","content":"SOURCE-SECRET\\ud800"}',
        encoding="utf-8",
    )
    batch = _parse_selection([source], destination=("Imported",))
    observation = _prior_observation(
        "surrogate.json",
        fingerprint="a" * 64,
        note_version=3,
    )

    plan = _classification_plan(batch, observation)

    item = plan.items[0]
    assert item.classification is ImportClassification.CHANGED_REPEAT
    assert ImportAction.UPDATE_EXISTING in item.allowed_actions
    rendered = f"{plan!r} {plan.to_diagnostic()!r} {observation!r}"
    log_text = " ".join(record.getMessage() for record in caplog.records)
    for private_value in (
        "SOURCE-SECRET",
        str(tmp_path),
        "UnicodeEncodeError",
        "\\ud800",
    ):
        assert private_value not in rendered
        assert private_value not in log_text


@pytest.mark.parametrize("changed", [False, True])
def test_exact_observation_without_current_version_cannot_authorize_update(
    changed: bool,
) -> None:
    payload = ParsedNotePayload(title="Title", content="Current")
    fingerprint_payload = (
        ParsedNotePayload(title="Title", content="Prior") if changed else payload
    )
    fingerprint = note_import_planner._private_payload_fingerprint(
        (fingerprint_payload,)
    )

    with pytest.raises(ValueError, match="version"):
        _prior_observation(
            "note.md",
            fingerprint=fingerprint,
            note_version=None,
        )

    uncertain = _prior_observation(
        "note.md",
        kind=ImportMatchKind.UNCERTAIN,
        fingerprint=None,
        note_version=None,
    )
    assert uncertain.note_version is None


@pytest.mark.parametrize("second_version", [7, 8])
def test_duplicate_exact_note_targets_are_rejected_before_update_authorization(
    tmp_path: Path,
    second_version: int,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("First", encoding="utf-8")
    second.write_text("Second", encoding="utf-8")
    batch = _parse_selection([first, second], destination=("Imported",))
    parsed = {source.candidate.source.display_path: source for source in batch.parsed}
    private_note_id = "SOURCE-SECRET-NOTE"
    observations = (
        _prior_observation(
            "first.md",
            fingerprint=note_import_planner._private_payload_fingerprint(
                parsed["first.md"].payloads
            ),
            note_id=private_note_id,
            note_version=7,
        ),
        _prior_observation(
            "second.md",
            fingerprint=note_import_planner._private_payload_fingerprint(
                parsed["second.md"].payloads
            ),
            note_id=private_note_id,
            note_version=second_version,
        ),
    )

    with pytest.raises(ValueError, match="duplicate exact note target") as raised:
        _classification_plan(batch, *observations)

    assert private_note_id not in str(raised.value)
    assert private_note_id not in repr(raised.value)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("notes.json", '{"content":""}'),
        ("notes.json", '{"content":"   \\n"}'),
        ("notes.yaml", 'content: ""\n'),
        ("notes.yaml", 'content: "   "\n'),
        ("notes.csv", "title,content\nOne,\n"),
        ("notes.csv", "title,content\nOne,   \n"),
    ],
)
def test_structured_sources_reject_empty_or_whitespace_only_bodies_atomically(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    source = tmp_path / filename
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].classification is ImportClassification.FAILED
    assert batch.issues[0].reason_code == "invalid_content"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("notes.json", '{"title":"   ","content":"  Body \\n"}'),
        ("notes.yaml", 'name: "   "\nbody: "  Body "\n'),
        ("notes.csv", "title,content\n   ,  Body \n"),
    ],
)
def test_structured_blank_titles_fall_back_without_stripping_content(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    source = tmp_path / filename
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    payload = batch.parsed[0].payloads[0]
    assert payload.title == "Untitled"
    assert payload.content.startswith("  Body ")


def _parse_scalar_limit_source(
    tmp_path: Path,
    filename: str,
    content: str,
) -> object:
    source = tmp_path / filename
    source.write_text(content, encoding="utf-8")
    bounds = _discovery_bounds(max_file_bytes=20_000, max_total_bytes=50_000)
    return _parse_selection(
        [source],
        bounds=bounds,
        destination=("Imported",),
    )


@pytest.mark.parametrize("format_name", ["markdown", "json", "yaml", "csv"])
@pytest.mark.parametrize("extra_length", [0, 1])
def test_parsed_title_scalar_limit_boundary(
    tmp_path: Path,
    format_name: str,
    extra_length: int,
) -> None:
    limit = note_import_plan_models.MAX_IMPORT_TITLE_LENGTH
    title = "T" * (limit + extra_length)
    if format_name == "markdown":
        filename, content = "note.md", f"# {title}\nBody"
    elif format_name == "json":
        filename, content = "note.json", f'{{"title":"{title}","content":"Body"}}'
    elif format_name == "yaml":
        filename, content = "note.yaml", f'title: "{title}"\ncontent: Body\n'
    else:
        filename, content = "note.csv", f"title,content\n{title},Body\n"

    batch = _parse_scalar_limit_source(tmp_path, filename, content)

    if extra_length:
        assert batch.parsed == ()
        assert batch.issues[0].reason_code == "invalid_content"
    else:
        assert batch.parsed[0].payloads[0].title == title


@pytest.mark.parametrize("format_name", ["json", "yaml", "csv"])
@pytest.mark.parametrize("extra_length", [0, 1])
def test_parsed_template_scalar_limit_boundary(
    tmp_path: Path,
    format_name: str,
    extra_length: int,
) -> None:
    limit = note_import_plan_models.MAX_IMPORT_TEMPLATE_NAME_LENGTH
    template = "M" * (limit + extra_length)
    if format_name == "json":
        filename = "note.json"
        content = f'{{"content":"Body","template":"{template}"}}'
    elif format_name == "yaml":
        filename = "note.yaml"
        content = f'content: Body\ntemplate: "{template}"\n'
    else:
        filename = "note.csv"
        content = f"title,content,template\nOne,Body,{template}\n"

    batch = _parse_scalar_limit_source(tmp_path, filename, content)

    if extra_length:
        assert batch.parsed == ()
        assert batch.issues[0].reason_code == "invalid_content"
    else:
        assert batch.parsed[0].payloads[0].template_name == template


@pytest.mark.parametrize("format_name", ["json", "yaml", "csv"])
@pytest.mark.parametrize("extra_length", [0, 1])
def test_parsed_keyword_scalar_limit_boundary(
    tmp_path: Path,
    format_name: str,
    extra_length: int,
) -> None:
    limit = note_import_plan_models.MAX_IMPORT_KEYWORD_LENGTH
    keyword = "K" * (limit + extra_length)
    if format_name == "json":
        filename = "note.json"
        content = f'{{"content":"Body","keywords":"{keyword}"}}'
    elif format_name == "yaml":
        filename = "note.yaml"
        content = f'content: Body\nkeywords: "{keyword}"\n'
    else:
        filename = "note.csv"
        content = f"title,content,keywords\nOne,Body,{keyword}\n"

    batch = _parse_scalar_limit_source(tmp_path, filename, content)

    if extra_length:
        assert batch.parsed == ()
        assert batch.issues[0].reason_code == "invalid_content"
    else:
        assert batch.parsed[0].payloads[0].keywords == (keyword,)


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("max_notes_per_file", 0, ValueError),
        ("max_notes_per_file", True, TypeError),
        ("max_keywords_per_note", 0, ValueError),
        ("max_keywords_per_note", False, TypeError),
    ],
)
def test_parser_bounds_are_strict_positive_integers(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "max_files": 20,
        "max_file_bytes": 1_000,
        "max_total_bytes": 5_000,
        "max_depth": 4,
        field_name: value,
    }

    with pytest.raises(error_type):
        ImportBounds(**values)  # type: ignore[arg-type]


def test_structured_note_and_keyword_expansion_respects_independent_bounds(
    tmp_path: Path,
) -> None:
    too_many_notes = tmp_path / "many.json"
    too_many_keywords = tmp_path / "keywords.json"
    too_many_notes.write_text(
        '[{"content":"One"},{"content":"Two"}]',
        encoding="utf-8",
    )
    too_many_keywords.write_text(
        '{"content":"Body","tags":["one","two"]}',
        encoding="utf-8",
    )
    bounds = _discovery_bounds(max_notes_per_file=1, max_keywords_per_note=1)

    batch = _parse_selection(
        [too_many_notes, too_many_keywords],
        bounds=bounds,
        destination=("Imported",),
    )

    assert batch.parsed == ()
    assert {issue.reason_code for issue in batch.issues} == {
        "too_many_keywords",
        "too_many_notes",
    }


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        (
            "notes.json",
            '{"content":"Body","keywords":" alpha, beta "}',
        ),
        ("notes.yaml", 'content: Body\ntags: " alpha, beta "\n'),
        ("notes.csv", 'title,content,tags\nOne,Body," alpha, beta "\n'),
    ],
)
def test_keyword_strings_split_and_trim_across_structured_formats(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    source = tmp_path / filename
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed[0].payloads[0].keywords == ("alpha", "beta")


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        (
            "notes.json",
            '{"content":"Body","keywords":[" alpha "," beta"]}',
        ),
        ("notes.yaml", 'content: Body\ntags: [" alpha ", " beta"]\n'),
    ],
)
def test_keyword_lists_are_trimmed_across_structured_formats(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    source = tmp_path / filename
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed[0].payloads[0].keywords == ("alpha", "beta")


@pytest.mark.parametrize(
    "content",
    [
        '{"content":"Body","tags":"alpha,,beta"}',
        '{"content":"Body","tags":["alpha","   "]}',
    ],
)
def test_empty_keyword_elements_reject_the_structured_source_atomically(
    tmp_path: Path,
    content: str,
) -> None:
    source = tmp_path / "notes.json"
    source.write_text(content, encoding="utf-8")

    batch = _parse_selection([source], destination=("Imported",))

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "invalid_content"


def test_keyword_bound_counts_split_string_elements(tmp_path: Path) -> None:
    source = tmp_path / "notes.json"
    source.write_text(
        '{"content":"Body","tags":"alpha,beta"}',
        encoding="utf-8",
    )
    bounds = _discovery_bounds(max_keywords_per_note=1)

    batch = _parse_selection(
        [source],
        bounds=bounds,
        destination=("Imported",),
    )

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "too_many_keywords"


def test_replaced_source_and_parent_directory_are_rejected_as_races(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    source = root / "note.md"
    source.write_text("original", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([root], bounds)
    old_root = tmp_path / "old-project"
    root.rename(old_root)
    root.mkdir()
    (root / "note.md").write_text("replacement", encoding="utf-8")

    batch = note_import_planner.parse_import_sources(discovery, bounds)

    assert batch.parsed == ()
    assert batch.proposed_folder_paths == ()
    assert batch.issues[0].reason_code == "source_changed"


def test_changed_leaf_identity_is_rejected_before_content_is_parsed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("original", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([source], bounds)
    source.unlink()
    source.write_text("replacement", encoding="utf-8")

    batch = note_import_planner.parse_import_sources(
        discovery,
        bounds,
        destination_folder_segments=("Imported",),
    )

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "source_changed"


def test_leaf_change_during_bounded_read_is_rejected_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("original", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([source], bounds)
    real_read = os.read
    changed = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, count)
        if chunk and not changed:
            changed = True
            source.write_text("modified", encoding="utf-8")
        return chunk

    monkeypatch.setattr(note_import_discovery.os, "read", racing_read)

    batch = note_import_planner.parse_import_sources(
        discovery,
        bounds,
        destination_folder_segments=("Imported",),
    )

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "source_changed"


def test_lexical_parent_replacement_during_read_is_rejected_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "selected"
    parent.mkdir()
    source = parent / "note.txt"
    source.write_text("original", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([source], bounds)
    moved_parent = tmp_path / "moved-selected"
    real_read = os.read
    real_open = os.open
    real_close = os.close
    replaced = False
    opened: list[int] = []
    closed: list[int] = []

    def tracked_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor: int) -> None:
        real_close(descriptor)
        closed.append(descriptor)

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, count)
        if chunk and not replaced:
            replaced = True
            parent.rename(moved_parent)
            parent.mkdir()
            source.write_text("replacement", encoding="utf-8")
        return chunk

    monkeypatch.setattr(note_import_discovery.os, "open", tracked_open)
    monkeypatch.setattr(note_import_discovery.os, "close", tracked_close)
    monkeypatch.setattr(note_import_discovery.os, "read", racing_read)

    batch = note_import_planner.parse_import_sources(
        discovery,
        bounds,
        destination_folder_segments=("Imported",),
    )

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "source_changed"
    assert opened
    assert Counter(opened) == Counter(closed)


def test_bounded_read_does_not_trust_discovered_stat_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.txt"
    source.write_bytes(b"12345")
    bounds = _discovery_bounds(max_file_bytes=5)
    discovery = discover_import_sources([source], bounds)
    real_read = os.read
    injected = False

    def oversized_read(descriptor: int, count: int) -> bytes:
        nonlocal injected
        chunk = real_read(descriptor, count)
        if chunk and not injected:
            injected = True
            return chunk + b"x"
        return chunk

    monkeypatch.setattr(note_import_discovery.os, "read", oversized_read)

    batch = note_import_planner.parse_import_sources(
        discovery,
        bounds,
        destination_folder_segments=("Imported",),
    )

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "max_file_bytes_exceeded"


def test_leaf_fifo_swap_between_stat_and_open_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("original", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([source], bounds)
    real_stat = os.stat
    swapped = False

    def swap_after_stat(
        path: object,
        *args: object,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal swapped
        metadata = real_stat(
            path,
            *args,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
            **kwargs,
        )
        if os.fspath(path) == source.name and dir_fd is not None and not swapped:
            swapped = True
            source.unlink()
            os.mkfifo(source)
        return metadata

    monkeypatch.setattr(note_import_discovery.os, "stat", swap_after_stat)
    results: list[object] = []
    errors: list[BaseException] = []

    def parse() -> None:
        try:
            results.append(
                note_import_planner.parse_import_sources(
                    discovery,
                    bounds,
                    destination_folder_segments=("Imported",),
                )
            )
        except BaseException as error:  # noqa: BLE001 - test must join the worker.
            errors.append(error)

    worker = Thread(target=parse, daemon=True)
    worker.start()
    worker.join(1.0)
    completed_promptly = not worker.is_alive()
    if worker.is_alive():
        writer = os.open(source, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
        worker.join(1.0)

    assert not worker.is_alive()
    assert completed_promptly
    assert errors == []
    assert len(results) == 1
    batch = results[0]
    assert batch.parsed == ()
    assert batch.issues[0].reason_code in {"source_changed", "source_unavailable"}


def test_missing_nonblocking_leaf_open_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Body", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([source], bounds)
    monkeypatch.delattr(note_import_discovery.os, "O_NONBLOCK")

    batch = note_import_planner.parse_import_sources(
        discovery,
        bounds,
        destination_folder_segments=("Imported",),
    )

    assert batch.parsed == ()
    assert batch.issues[0].reason_code == "secure_read_unavailable"


def test_directory_hierarchy_proposes_only_successful_ancestor_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project"
    child = root / "child"
    empty = root / "empty"
    unsupported = root / "unsupported"
    child.mkdir(parents=True)
    empty.mkdir()
    unsupported.mkdir()
    (child / "note.md").write_text("Body", encoding="utf-8")
    (unsupported / "asset.bin").write_bytes(b"private")

    batch = _parse_selection([root])

    assert batch.proposed_folder_paths == (
        ("Project",),
        ("Project", "child"),
    )
    assert batch.parsed[0].memberships[0].folder_segments == (
        "Project",
        "child",
    )


def test_empty_directory_and_unsupported_only_branch_propose_no_folders(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "Empty"
    empty.mkdir()
    unsupported = tmp_path / "Unsupported"
    unsupported.mkdir()
    (unsupported / "asset.bin").write_bytes(b"private")

    empty_batch = _parse_selection([empty])
    unsupported_batch = _parse_selection([unsupported])

    assert empty_batch.parsed == empty_batch.issues == ()
    assert empty_batch.proposed_folder_paths == ()
    assert unsupported_batch.parsed == ()
    assert unsupported_batch.proposed_folder_paths == ()


def test_selected_files_require_a_canonical_manual_destination(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("First", encoding="utf-8")
    second.write_text("Second", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([first, second], bounds)

    with pytest.raises(ImportSelectionError) as missing:
        note_import_planner.parse_import_sources(discovery, bounds)
    assert missing.value.reason_code == "destination_required"

    with pytest.raises(ImportSelectionError) as invalid:
        note_import_planner.parse_import_sources(
            discovery,
            bounds,
            destination_folder_segments=(" Imported ",),
        )
    assert invalid.value.reason_code == "invalid_destination"

    batch = note_import_planner.parse_import_sources(
        discovery,
        bounds,
        destination_folder_segments=("Inbox", "Imported"),
    )
    assert batch.proposed_folder_paths == (("Inbox",), ("Inbox", "Imported"))
    assert all(
        source.memberships[0].folder_segments == ("Inbox", "Imported")
        for source in batch.parsed
    )


def test_directory_import_rejects_a_separate_destination(tmp_path: Path) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    (root / "note.md").write_text("Body", encoding="utf-8")
    bounds = _discovery_bounds()
    discovery = discover_import_sources([root], bounds)

    with pytest.raises(ImportSelectionError) as raised:
        note_import_planner.parse_import_sources(
            discovery,
            bounds,
            destination_folder_segments=("Imported",),
        )

    assert raised.value.reason_code == "destination_not_allowed"


def test_discovery_failures_are_carried_forward_without_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    target = tmp_path / "outside.md"
    target.write_text("private", encoding="utf-8")
    (root / "linked.md").symlink_to(target)

    batch = _parse_selection([root])

    assert batch.parsed == ()
    assert batch.issues[0].classification is ImportClassification.FAILED
    assert batch.issues[0].reason_code == "nested_symlink"
    assert batch.proposed_folder_paths == ()


def _task5_directory_plan(
    *,
    root_label: str = "Work",
    item: ImportPreviewItem | None = None,
) -> NoteImportPlan:
    root_source = ImportSource(
        kind=ImportSourceKind.DIRECTORY_MEMBER,
        display_path=f"{root_label}/note.md",
        source_path=Path("/private/source/note.md"),
    )
    root_memberships = (
        ProposedFolderMembership(
            payload_index=0,
            folder_segments=(root_label,),
        ),
    )
    first_item = (
        dataclass_replace(item, source=root_source, memberships=root_memberships)
        if item is not None
        else _new_item(
            item_id="item-root",
            source=root_source,
            memberships=root_memberships,
        )
    )
    nested_item = _new_item(
        item_id="item-nested",
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path=f"{root_label}/Ideas/nested.md",
            source_path=Path("/private/source/Ideas/nested.md"),
        ),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=(root_label, "Ideas"),
            ),
        ),
    )
    return NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(first_item, nested_item),
        proposed_folder_paths=((root_label,), (root_label, "Ideas")),
    )


def _task5_exact_item(
    *,
    item_id: str = "item-exact",
    selected_action: ImportAction = ImportAction.CREATE_NEW,
    replace_content: bool = False,
    add_membership: bool = True,
) -> ImportPreviewItem:
    return _new_item(
        item_id=item_id,
        classification=ImportClassification.CHANGED_REPEAT,
        default_action=ImportAction.CREATE_NEW,
        selected_action=selected_action,
        allowed_actions=(
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        ),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id="note-17",
            note_version=9,
        ),
        replace_content=replace_content,
        add_membership=add_membership,
    )


def _task5_uncertain_item(
    *,
    payloads: tuple[ParsedNotePayload, ...] | None = None,
    memberships: tuple[ProposedFolderMembership, ...] | None = None,
    note_version: int | None = 12,
) -> ImportPreviewItem:
    return _new_item(
        item_id="item-uncertain",
        payloads=payloads or (_payload(),),
        memberships=memberships
        or (
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Project", "Meetings"),
            ),
        ),
        classification=ImportClassification.UNCERTAIN_MATCH,
        default_action=ImportAction.CREATE_NEW,
        selected_action=ImportAction.CREATE_NEW,
        allowed_actions=(ImportAction.SKIP, ImportAction.CREATE_NEW),
        match=ImportMatch(
            kind=ImportMatchKind.UNCERTAIN,
            note_id="note-possible",
            note_version=note_version,
        ),
    )


def _task5_update_capable_item(
    *,
    match_kind: ImportMatchKind,
    classification: ImportClassification,
    note_version: int | None,
    payload_count: int,
) -> ImportPreviewItem:
    payloads = tuple(
        ParsedNotePayload(title=f"Note {index}", content=f"Body {index}")
        for index in range(payload_count)
    )
    memberships = tuple(
        ProposedFolderMembership(
            payload_index=index,
            folder_segments=("Project", "Meetings"),
        )
        for index in range(payload_count)
    )
    return _new_item(
        payloads=payloads,
        memberships=memberships,
        classification=classification,
        default_action=ImportAction.CREATE_NEW,
        selected_action=ImportAction.CREATE_NEW,
        allowed_actions=(
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        ),
        match=ImportMatch(
            kind=match_kind,
            note_id="note-update-target",
            note_version=note_version,
        ),
    )


@pytest.mark.parametrize(
    ("match_kind", "classification"),
    [
        (ImportMatchKind.EXACT, ImportClassification.CHANGED_REPEAT),
        (ImportMatchKind.USER_CONFIRMED, ImportClassification.UNCERTAIN_MATCH),
    ],
)
@pytest.mark.parametrize(
    ("note_version", "payload_count"),
    [(None, 1), (7, 2)],
)
def test_update_capable_item_requires_one_payload_and_a_current_version(
    match_kind: ImportMatchKind,
    classification: ImportClassification,
    note_version: int | None,
    payload_count: int,
) -> None:
    with pytest.raises(ValueError, match="one payload and a current note version"):
        _task5_update_capable_item(
            match_kind=match_kind,
            classification=classification,
            note_version=note_version,
            payload_count=payload_count,
        )


def test_single_versioned_exact_and_confirmed_items_remain_update_capable() -> None:
    exact = _task5_update_capable_item(
        match_kind=ImportMatchKind.EXACT,
        classification=ImportClassification.CHANGED_REPEAT,
        note_version=4,
        payload_count=1,
    )
    confirmed = _task5_update_capable_item(
        match_kind=ImportMatchKind.USER_CONFIRMED,
        classification=ImportClassification.UNCERTAIN_MATCH,
        note_version=5,
        payload_count=1,
    )

    assert ImportAction.UPDATE_EXISTING in exact.allowed_actions
    assert ImportAction.UPDATE_EXISTING in confirmed.allowed_actions


def test_multi_payload_versionless_uncertain_item_remains_create_only() -> None:
    payloads = (
        ParsedNotePayload(title="One", content="First"),
        ParsedNotePayload(title="Two", content="Second"),
    )
    memberships = (
        ProposedFolderMembership(
            payload_index=0,
            folder_segments=("Project", "Meetings"),
        ),
        ProposedFolderMembership(
            payload_index=1,
            folder_segments=("Project", "Meetings"),
        ),
    )

    item = _task5_uncertain_item(
        payloads=payloads,
        memberships=memberships,
        note_version=None,
    )

    assert item.allowed_actions == (ImportAction.SKIP, ImportAction.CREATE_NEW)
    assert item.selected_action is ImportAction.CREATE_NEW


@pytest.mark.parametrize(
    ("match_kind", "classification", "note_version", "payload_count"),
    [
        (
            ImportMatchKind.EXACT,
            ImportClassification.CHANGED_REPEAT,
            None,
            1,
        ),
        (
            ImportMatchKind.USER_CONFIRMED,
            ImportClassification.UNCERTAIN_MATCH,
            8,
            2,
        ),
    ],
)
def test_caller_cannot_build_an_override_plan_that_bypasses_update_authorization(
    match_kind: ImportMatchKind,
    classification: ImportClassification,
    note_version: int | None,
    payload_count: int,
) -> None:
    with pytest.raises(ValueError, match="one payload and a current note version"):
        item = _task5_update_capable_item(
            match_kind=match_kind,
            classification=classification,
            note_version=note_version,
            payload_count=payload_count,
        )
        plan = _plan_with_item(item)
        note_import_planner.apply_item_override(
            plan,
            item.item_id,
            ImportAction.UPDATE_EXISTING,
            replace_content=True,
        )


@pytest.mark.parametrize(
    "existing_name",
    [
        "work",
        "Ｗｏｒｋ",
        "WoRK",
    ],
)
def test_root_collision_detection_uses_folder_canonical_semantics(
    existing_name: str,
) -> None:
    plan = _task5_directory_plan()

    analyzed = note_import_planner.analyze_root_collision(plan, (existing_name,))

    assert analyzed.root_collision == RootCollisionState(
        proposed_label="Work",
        collides=True,
    )
    assert analyzed.proposed_folder_paths == plan.proposed_folder_paths
    assert analyzed.items == plan.items


def test_root_collision_detection_uses_unicode_normalization() -> None:
    plan = _task5_directory_plan(root_label="Café")

    analyzed = note_import_planner.analyze_root_collision(plan, ("Cafe\u0301",))

    assert analyzed.root_collision is not None
    assert analyzed.root_collision.collides


def test_noncolliding_root_is_explicitly_analyzed_but_cannot_be_fake_resolved() -> None:
    plan = note_import_planner.analyze_root_collision(
        _task5_directory_plan(),
        ("Archive",),
    )

    assert plan.root_collision == RootCollisionState(
        proposed_label="Work",
        collides=False,
    )
    with pytest.raises(ValueError, match="colliding root"):
        note_import_planner.resolve_root_collision(
            plan,
            RootCollisionChoice.USE_EXISTING,
            existing_top_level_names=("Archive",),
        )


def test_use_existing_resolves_collision_without_rewriting_the_root() -> None:
    original = _task5_directory_plan()
    analyzed = note_import_planner.analyze_root_collision(original, ("work",))

    resolved = note_import_planner.resolve_root_collision(
        analyzed,
        RootCollisionChoice.USE_EXISTING,
        existing_top_level_names=("work",),
    )

    assert resolved.root_collision == RootCollisionState(
        proposed_label="Work",
        collides=True,
        choice=RootCollisionChoice.USE_EXISTING,
    )
    assert resolved.proposed_folder_paths == original.proposed_folder_paths
    assert tuple(item.memberships for item in resolved.items) == tuple(
        item.memberships for item in original.items
    )
    assert analyzed.root_collision is not None
    assert analyzed.root_collision.choice is None


def test_unique_sibling_uses_first_genuinely_noncolliding_canonical_name() -> None:
    analyzed = note_import_planner.analyze_root_collision(
        _task5_directory_plan(),
        ("work", "WORK (2)", "Ｗｏｒｋ (3)"),
    )

    resolved = note_import_planner.resolve_root_collision(
        analyzed,
        RootCollisionChoice.UNIQUE_SIBLING,
        existing_top_level_names=("work", "WORK (2)", "Ｗｏｒｋ (3)"),
    )

    assert resolved.root_collision == RootCollisionState(
        proposed_label="Work",
        collides=True,
        choice=RootCollisionChoice.UNIQUE_SIBLING,
        resolved_label="Work (4)",
    )
    assert resolved.proposed_folder_paths == (
        ("Work (4)",),
        ("Work (4)", "Ideas"),
    )


def test_renamed_root_rewrites_every_folder_reference_but_not_sources_or_payloads() -> (
    None
):
    original = _task5_directory_plan()
    analyzed = note_import_planner.analyze_root_collision(original, ("work",))

    resolved = note_import_planner.resolve_root_collision(
        analyzed,
        RootCollisionChoice.RENAMED_ROOT,
        existing_top_level_names=("work",),
        renamed_root="Archive",
    )

    assert resolved.proposed_folder_paths == (
        ("Archive",),
        ("Archive", "Ideas"),
    )
    assert tuple(
        membership.folder_segments
        for item in resolved.items
        for membership in item.memberships
    ) == (("Archive",), ("Archive", "Ideas"))
    assert tuple(item.source for item in resolved.items) == tuple(
        item.source for item in original.items
    )
    assert tuple(item.payloads for item in resolved.items) == tuple(
        item.payloads for item in original.items
    )
    assert tuple(
        membership.payload_index
        for item in resolved.items
        for membership in item.memberships
    ) == (0, 0)
    assert original.proposed_folder_paths == (("Work",), ("Work", "Ideas"))
    assert original.items[0].memberships[0].folder_segments == ("Work",)


@pytest.mark.parametrize(
    ("choice", "renamed_root"),
    [
        ("use_existing", None),
        (RootCollisionChoice.RENAMED_ROOT, " "),
        (RootCollisionChoice.RENAMED_ROOT, "work"),
        (RootCollisionChoice.RENAMED_ROOT, "Ｗｏｒｋ"),
        (RootCollisionChoice.UNIQUE_SIBLING, "Caller supplied"),
        (RootCollisionChoice.USE_EXISTING, "Caller supplied"),
    ],
)
def test_root_collision_resolution_rejects_coercion_invalid_or_inapplicable_names(
    choice: object,
    renamed_root: str | None,
) -> None:
    analyzed = note_import_planner.analyze_root_collision(
        _task5_directory_plan(),
        ("work",),
    )

    with pytest.raises((TypeError, ValueError)):
        note_import_planner.resolve_root_collision(
            analyzed,
            choice,  # type: ignore[arg-type]
            existing_top_level_names=("work",),
            renamed_root=renamed_root,
        )


def test_collision_name_inputs_are_bounded_and_iterator_errors_are_sanitized() -> None:
    plan = _task5_directory_plan()
    private_error = "SECRET /private/folder-name"

    def too_many_names() -> object:
        for index in range(plan.bounds.max_entries + 1):
            yield f"Folder {index}"
        raise AssertionError("collision analysis over-consumed its bounded input")

    with pytest.raises(ValueError, match="too many"):
        note_import_planner.analyze_root_collision(plan, too_many_names())  # type: ignore[arg-type]

    def broken_names() -> object:
        raise RuntimeError(private_error)
        yield  # pragma: no cover

    with pytest.raises(ValueError, match="read safely") as raised:
        note_import_planner.analyze_root_collision(plan, broken_names())  # type: ignore[arg-type]
    assert private_error not in str(raised.value)
    assert private_error not in repr(raised.value)


def test_collision_analysis_rejects_unsafe_existing_names_without_echoing_them() -> (
    None
):
    unsafe_name = "SECRET/private-root"

    with pytest.raises(ValueError, match="valid folder names") as raised:
        note_import_planner.analyze_root_collision(
            _task5_directory_plan(),
            (unsafe_name,),
        )

    assert unsafe_name not in str(raised.value)
    assert unsafe_name not in repr(raised.value)


def test_selected_file_destination_is_not_mistaken_for_a_directory_root() -> None:
    plan = NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(
            _new_item(
                source=ImportSource(
                    kind=ImportSourceKind.SELECTED_FILE,
                    display_path="note.md",
                    source_path=Path("/private/source/note.md"),
                )
            ),
        ),
        proposed_folder_paths=(("Imported",),),
    )

    analyzed = note_import_planner.analyze_root_collision(plan, ("Imported",))

    assert analyzed.root_collision is None
    with pytest.raises(ValueError, match="directory root"):
        note_import_planner.resolve_root_collision(
            NoteImportPlan(
                bounds=plan.bounds,
                items=plan.items,
                proposed_folder_paths=plan.proposed_folder_paths,
                root_collision=RootCollisionState(
                    proposed_label="Imported",
                    collides=True,
                ),
            ),
            RootCollisionChoice.USE_EXISTING,
            existing_top_level_names=("Imported",),
        )


def test_empty_or_skip_only_plan_has_no_collision_or_folder_creation() -> None:
    unsupported = _new_item(
        item_id="item-failed",
        payloads=(),
        memberships=(),
        classification=ImportClassification.UNSUPPORTED,
        reason="Unsupported.",
        default_action=ImportAction.SKIP,
        selected_action=ImportAction.SKIP,
        allowed_actions=(ImportAction.SKIP,),
        match=None,
        replace_content=False,
        add_membership=False,
    )
    plan = NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(unsupported,),
        proposed_folder_paths=(),
    )

    analyzed = note_import_planner.analyze_root_collision(plan, ("Project",))

    assert analyzed is plan
    assert analyzed.proposed_folder_paths == ()
    assert analyzed.root_collision is None


def test_collision_analysis_ignores_a_directory_with_no_selected_membership() -> None:
    skipped_item = _task5_exact_item(
        item_id="item-root",
        selected_action=ImportAction.SKIP,
        add_membership=False,
    )
    plan = NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(skipped_item,),
        proposed_folder_paths=(("Project",), ("Project", "Meetings")),
    )

    analyzed = note_import_planner.analyze_root_collision(plan, ("Project",))

    assert analyzed.root_collision is None


def test_skipping_the_last_membership_clears_obsolete_collision_state() -> None:
    item = _task5_exact_item(item_id="item-root")
    plan = NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(item,),
        proposed_folder_paths=(("Project",), ("Project", "Meetings")),
        root_collision=RootCollisionState(
            proposed_label="Project",
            collides=True,
            choice=RootCollisionChoice.USE_EXISTING,
        ),
    )

    skipped = note_import_planner.apply_item_override(
        plan,
        "item-root",
        ImportAction.SKIP,
    )

    assert skipped.root_collision is None
    assert skipped.proposed_folder_paths == ()


def _task5_branch_plan() -> NoteImportPlan:
    alpha = _task5_exact_item(item_id="item-alpha")
    alpha = note_import_plan_models.ImportPreviewItem(
        item_id=alpha.item_id,
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="Work/Alpha/note.md",
            source_path=Path("/private/source/Alpha/note.md"),
        ),
        payloads=alpha.payloads,
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Work", "Alpha"),
            ),
        ),
        classification=alpha.classification,
        reason=alpha.reason,
        default_action=alpha.default_action,
        selected_action=alpha.selected_action,
        allowed_actions=alpha.allowed_actions,
        match=alpha.match,
        replace_content=alpha.replace_content,
        add_membership=alpha.add_membership,
    )
    beta = _task5_exact_item(item_id="item-beta")
    beta = note_import_plan_models.ImportPreviewItem(
        item_id=beta.item_id,
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="Work/Beta/Deep/note.md",
            source_path=Path("/private/source/Beta/Deep/note.md"),
        ),
        payloads=beta.payloads,
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Work", "Beta", "Deep"),
            ),
        ),
        classification=beta.classification,
        reason=beta.reason,
        default_action=beta.default_action,
        selected_action=beta.selected_action,
        allowed_actions=beta.allowed_actions,
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id="note-18",
            note_version=10,
        ),
        replace_content=beta.replace_content,
        add_membership=beta.add_membership,
    )
    return NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(beta, alpha),
        proposed_folder_paths=(
            ("Work",),
            ("Work", "Beta"),
            ("Work", "Beta", "Deep"),
            ("Work", "Alpha"),
            ("Work", "Stale"),
        ),
        root_collision=RootCollisionState(
            proposed_label="Work",
            collides=True,
            choice=RootCollisionChoice.USE_EXISTING,
        ),
    )


def test_item_override_recomputes_sorted_folder_ancestor_closure() -> None:
    original = _task5_branch_plan()

    updated = note_import_planner.apply_item_override(
        original,
        "item-beta",
        ImportAction.CREATE_NEW,
    )

    assert updated.proposed_folder_paths == (
        ("Work",),
        ("Work", "Alpha"),
        ("Work", "Beta"),
        ("Work", "Beta", "Deep"),
    )
    assert updated.root_collision == original.root_collision
    assert original.proposed_folder_paths[-1] == ("Work", "Stale")
    assert original.root_collision is not None


def test_skip_and_restore_prune_and_restore_only_the_affected_branch() -> None:
    original = _task5_branch_plan()

    skipped = note_import_planner.apply_item_override(
        original,
        "item-alpha",
        ImportAction.SKIP,
    )
    restored = note_import_planner.apply_item_override(
        skipped,
        "item-alpha",
        ImportAction.CREATE_NEW,
    )

    assert skipped.proposed_folder_paths == (
        ("Work",),
        ("Work", "Beta"),
        ("Work", "Beta", "Deep"),
    )
    assert restored.proposed_folder_paths == (
        ("Work",),
        ("Work", "Alpha"),
        ("Work", "Beta"),
        ("Work", "Beta", "Deep"),
    )
    assert skipped.root_collision == original.root_collision
    assert restored.root_collision == original.root_collision
    assert original.items[1].selected_action is ImportAction.CREATE_NEW


def test_branch_override_preserves_unresolved_collision_for_the_same_root() -> None:
    original = _task5_branch_plan()
    unresolved = NoteImportPlan(
        bounds=original.bounds,
        items=original.items,
        proposed_folder_paths=original.proposed_folder_paths,
        root_collision=RootCollisionState(proposed_label="Work", collides=True),
    )

    updated = note_import_planner.apply_item_override(
        unresolved,
        "item-alpha",
        ImportAction.SKIP,
    )

    assert updated.root_collision == unresolved.root_collision


def test_branch_override_discards_collision_when_effective_root_changes() -> None:
    item = _task5_exact_item(item_id="item-root")
    item = dataclass_replace(
        item,
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="Archive/note.md",
            source_path=Path("/private/source/Archive/note.md"),
        ),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=("Archive",),
            ),
        ),
    )
    stale = NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(item,),
        proposed_folder_paths=(("Work",),),
        root_collision=RootCollisionState(
            proposed_label="Work",
            collides=True,
            choice=RootCollisionChoice.USE_EXISTING,
        ),
    )

    updated = note_import_planner.apply_item_override(
        stale,
        "item-root",
        ImportAction.CREATE_NEW,
    )

    assert updated.proposed_folder_paths == (("Archive",),)
    assert updated.root_collision is None


def test_content_only_update_prunes_all_folder_proposals() -> None:
    item = _task5_exact_item(item_id="item-root")
    plan = NoteImportPlan(
        bounds=_discovery_bounds(),
        items=(item,),
        proposed_folder_paths=(("Project",), ("Project", "Meetings")),
        root_collision=RootCollisionState(
            proposed_label="Project",
            collides=False,
        ),
    )

    updated = note_import_planner.apply_item_override(
        plan,
        "item-root",
        ImportAction.UPDATE_EXISTING,
        replace_content=True,
        add_membership=False,
    )

    assert updated.items[0].replace_content
    assert not updated.items[0].add_membership
    assert updated.proposed_folder_paths == ()
    assert updated.root_collision is None


def test_membership_closure_deduplicates_shared_ancestors_deterministically() -> None:
    original = _task5_branch_plan()
    reversed_plan = NoteImportPlan(
        bounds=original.bounds,
        items=tuple(reversed(original.items)),
        proposed_folder_paths=original.proposed_folder_paths,
        root_collision=original.root_collision,
    )

    first = note_import_planner.apply_item_override(
        original,
        "item-alpha",
        ImportAction.CREATE_NEW,
    )
    second = note_import_planner.apply_item_override(
        reversed_plan,
        "item-alpha",
        ImportAction.CREATE_NEW,
    )

    assert first.proposed_folder_paths == second.proposed_folder_paths
    assert first.proposed_folder_paths.count(("Work",)) == 1


def test_reenabled_membership_restores_rebased_folder_paths_for_reanalysis() -> None:
    analyzed = note_import_planner.analyze_root_collision(
        _task5_directory_plan(item=_task5_exact_item(item_id="item-root")),
        ("Work",),
    )
    resolved = note_import_planner.resolve_root_collision(
        analyzed,
        RootCollisionChoice.RENAMED_ROOT,
        existing_top_level_names=("Work",),
        renamed_root="Archive",
    )
    skipped_root = note_import_planner.apply_item_override(
        resolved,
        "item-root",
        ImportAction.SKIP,
    )
    skipped_all = note_import_planner.apply_item_override(
        skipped_root,
        "item-nested",
        ImportAction.SKIP,
    )

    restored = note_import_planner.apply_item_override(
        skipped_all,
        "item-root",
        ImportAction.CREATE_NEW,
    )

    assert skipped_all.proposed_folder_paths == ()
    assert restored.proposed_folder_paths == (("Archive",),)
    assert restored.items[0].memberships[0].folder_segments == ("Archive",)
    assert restored.root_collision is None


def test_confirm_uncertain_match_adds_update_without_changing_classification() -> None:
    uncertain = _task5_uncertain_item()
    original = _plan_with_item(uncertain)

    confirmed = note_import_planner.confirm_uncertain_match(
        original,
        "item-uncertain",
    )

    item = confirmed.items[0]
    assert item.classification is ImportClassification.UNCERTAIN_MATCH
    assert item.match == ImportMatch(
        kind=ImportMatchKind.USER_CONFIRMED,
        note_id="note-possible",
        note_version=12,
    )
    assert item.allowed_actions == (
        ImportAction.SKIP,
        ImportAction.CREATE_NEW,
        ImportAction.UPDATE_EXISTING,
    )
    assert item.selected_action is ImportAction.CREATE_NEW
    assert item.default_action is ImportAction.CREATE_NEW
    assert original.items[0].match is not None
    assert original.items[0].match.kind is ImportMatchKind.UNCERTAIN


def test_confirm_uncertain_match_rejects_an_exact_update_target_collision() -> None:
    """One exact match and one confirmation cannot authorize the same target."""
    private_note_id = "PRIVATE-SHARED-NOTE"
    exact = dataclass_replace(
        _task5_exact_item(),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=private_note_id,
            note_version=9,
        ),
    )
    uncertain = dataclass_replace(
        _task5_uncertain_item(),
        match=ImportMatch(
            kind=ImportMatchKind.UNCERTAIN,
            note_id=private_note_id,
            note_version=12,
        ),
    )
    plan = dataclass_replace(_plan_with_item(exact), items=(exact, uncertain))

    with pytest.raises(ValueError, match="duplicate update target") as raised:
        note_import_planner.confirm_uncertain_match(plan, "item-uncertain")

    assert private_note_id not in str(raised.value)


def test_confirm_uncertain_match_rejects_a_second_uncertain_update_target() -> None:
    """Two uncertain sources cannot both gain update authority for one note."""
    private_note_id = "PRIVATE-SHARED-NOTE"
    first = dataclass_replace(
        _task5_uncertain_item(),
        item_id="item-uncertain-1",
        match=ImportMatch(
            kind=ImportMatchKind.UNCERTAIN,
            note_id=private_note_id,
            note_version=12,
        ),
    )
    second = dataclass_replace(
        _task5_uncertain_item(),
        item_id="item-uncertain-2",
        match=ImportMatch(
            kind=ImportMatchKind.UNCERTAIN,
            note_id=private_note_id,
            note_version=13,
        ),
    )
    plan = dataclass_replace(_plan_with_item(first), items=(first, second))
    once_confirmed = note_import_planner.confirm_uncertain_match(
        plan,
        "item-uncertain-1",
    )

    with pytest.raises(ValueError, match="duplicate update target") as raised:
        note_import_planner.confirm_uncertain_match(
            once_confirmed,
            "item-uncertain-2",
        )

    assert private_note_id not in str(raised.value)


def _duplicate_authorized_update_plan() -> NoteImportPlan:
    private_note_id = "PRIVATE-SHARED-NOTE"
    first = dataclass_replace(
        _task5_exact_item(item_id="item-update-1"),
        match=ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id=private_note_id,
            note_version=12,
        ),
    )
    second = dataclass_replace(
        _task5_uncertain_item(),
        item_id="item-update-2",
        match=ImportMatch(
            kind=ImportMatchKind.USER_CONFIRMED,
            note_id=private_note_id,
            note_version=13,
        ),
        allowed_actions=(
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        ),
    )
    return dataclass_replace(_plan_with_item(first), items=(first, second))


def test_update_override_rejects_a_second_selected_update_target() -> None:
    """A crafted authorized plan still cannot select two updates for one note."""
    plan = _duplicate_authorized_update_plan()
    first_update = note_import_planner.apply_item_override(
        plan,
        "item-update-1",
        ImportAction.UPDATE_EXISTING,
        replace_content=True,
    )

    with pytest.raises(ValueError, match="duplicate update target") as raised:
        note_import_planner.apply_item_override(
            first_update,
            "item-update-2",
            ImportAction.UPDATE_EXISTING,
            replace_content=True,
        )

    assert "PRIVATE-SHARED-NOTE" not in str(raised.value)


def test_note_import_plan_rejects_duplicate_selected_update_targets() -> None:
    """The aggregate model prevents direct construction from bypassing the guard."""
    plan = _duplicate_authorized_update_plan()
    selected = tuple(
        dataclass_replace(
            item,
            selected_action=ImportAction.UPDATE_EXISTING,
            replace_content=True,
            add_membership=False,
        )
        for item in plan.items
    )

    with pytest.raises(ValueError, match="duplicate update target") as raised:
        dataclass_replace(plan, items=selected)

    assert "PRIVATE-SHARED-NOTE" not in str(raised.value)


def test_confirm_uncertain_match_rejects_a_versionless_target() -> None:
    original = _plan_with_item(_task5_uncertain_item(note_version=None))

    with pytest.raises(ValueError, match="cannot be confirmed"):
        note_import_planner.confirm_uncertain_match(original, "item-uncertain")

    assert original.items[0].match is not None
    assert original.items[0].match.kind is ImportMatchKind.UNCERTAIN
    assert ImportAction.UPDATE_EXISTING not in original.items[0].allowed_actions


def test_confirm_uncertain_match_rejects_a_multi_payload_source() -> None:
    payloads = (
        ParsedNotePayload(title="One", content="First"),
        ParsedNotePayload(title="Two", content="Second"),
    )
    memberships = (
        ProposedFolderMembership(
            payload_index=0,
            folder_segments=("Project", "Meetings"),
        ),
        ProposedFolderMembership(
            payload_index=1,
            folder_segments=("Project", "Meetings"),
        ),
    )
    original = _plan_with_item(
        _task5_uncertain_item(payloads=payloads, memberships=memberships)
    )

    with pytest.raises(ValueError, match="cannot be confirmed"):
        note_import_planner.confirm_uncertain_match(original, "item-uncertain")

    assert original.items[0].classification is ImportClassification.UNCERTAIN_MATCH
    assert original.items[0].match is not None
    assert original.items[0].match.kind is ImportMatchKind.UNCERTAIN


@pytest.mark.parametrize("item_id", ["missing", "", "unsafe/path", 17])
def test_confirm_uncertain_match_rejects_missing_or_unsafe_item_ids(
    item_id: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="item"):
        note_import_planner.confirm_uncertain_match(
            _plan_with_item(_task5_uncertain_item()),
            item_id,  # type: ignore[arg-type]
        )


def test_only_uncertain_matches_can_be_confirmed() -> None:
    with pytest.raises(ValueError, match="uncertain match"):
        note_import_planner.confirm_uncertain_match(
            _plan_with_item(_task5_exact_item()),
            "item-exact",
        )


def test_uncertain_update_requires_confirmation_first() -> None:
    plan = _plan_with_item(_task5_uncertain_item())

    with pytest.raises(ValueError, match="allowed"):
        note_import_planner.apply_item_override(
            plan,
            "item-uncertain",
            ImportAction.UPDATE_EXISTING,
            replace_content=True,
            add_membership=False,
        )

    confirmed = note_import_planner.confirm_uncertain_match(plan, "item-uncertain")
    updated = note_import_planner.apply_item_override(
        confirmed,
        "item-uncertain",
        ImportAction.UPDATE_EXISTING,
        replace_content=True,
        add_membership=False,
    )
    assert updated.items[0].selected_action is ImportAction.UPDATE_EXISTING


@pytest.mark.parametrize(
    ("replace_content", "add_membership"),
    [(True, False), (False, True), (True, True)],
)
def test_update_override_keeps_content_and_membership_choices_independent(
    replace_content: bool,
    add_membership: bool,
) -> None:
    original = _plan_with_item(_task5_exact_item())

    updated = note_import_planner.apply_item_override(
        original,
        "item-exact",
        ImportAction.UPDATE_EXISTING,
        replace_content=replace_content,
        add_membership=add_membership,
    )

    item = updated.items[0]
    assert item.selected_action is ImportAction.UPDATE_EXISTING
    assert item.replace_content is replace_content
    assert item.add_membership is add_membership
    assert item.match == original.items[0].match
    assert item.match is not None and item.match.note_version == 9
    assert item.source is original.items[0].source
    assert item.payloads is original.items[0].payloads
    assert item.memberships is original.items[0].memberships
    assert original.items[0].selected_action is ImportAction.CREATE_NEW


def test_update_override_requires_at_least_one_effect() -> None:
    with pytest.raises(ValueError, match="replace content or add membership"):
        note_import_planner.apply_item_override(
            _plan_with_item(_task5_exact_item()),
            "item-exact",
            ImportAction.UPDATE_EXISTING,
            replace_content=False,
            add_membership=False,
        )


def test_skip_and_create_overrides_set_their_only_valid_effects() -> None:
    original = _plan_with_item(_task5_exact_item())

    skipped = note_import_planner.apply_item_override(
        original,
        "item-exact",
        ImportAction.SKIP,
    )
    recreated = note_import_planner.apply_item_override(
        skipped,
        "item-exact",
        ImportAction.CREATE_NEW,
    )

    assert not skipped.items[0].replace_content
    assert not skipped.items[0].add_membership
    assert not recreated.items[0].replace_content
    assert recreated.items[0].add_membership
    assert original.items[0].selected_action is ImportAction.CREATE_NEW


@pytest.mark.parametrize(
    ("action", "replace_content", "add_membership"),
    [
        ("skip", False, False),
        (ImportAction.UPDATE_EXISTING, 1, False),
        (ImportAction.UPDATE_EXISTING, False, 0),
        (ImportAction.CREATE_NEW, True, False),
    ],
)
def test_item_override_rejects_coerced_actions_bools_and_invalid_effects(
    action: object,
    replace_content: object,
    add_membership: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        note_import_planner.apply_item_override(
            _plan_with_item(_task5_exact_item()),
            "item-exact",
            action,  # type: ignore[arg-type]
            replace_content=replace_content,  # type: ignore[arg-type]
            add_membership=add_membership,  # type: ignore[arg-type]
        )


def test_skip_only_items_reject_create_and_update_overrides() -> None:
    unsupported = _new_item(
        item_id="item-unsupported",
        payloads=(),
        memberships=(),
        classification=ImportClassification.UNSUPPORTED,
        reason="Unsupported.",
        default_action=ImportAction.SKIP,
        selected_action=ImportAction.SKIP,
        allowed_actions=(ImportAction.SKIP,),
        match=None,
        replace_content=False,
        add_membership=False,
    )
    plan = _plan_with_item(unsupported)

    for action in (ImportAction.CREATE_NEW, ImportAction.UPDATE_EXISTING):
        with pytest.raises(ValueError, match="allowed"):
            note_import_planner.apply_item_override(
                plan,
                "item-unsupported",
                action,
                replace_content=action is ImportAction.UPDATE_EXISTING,
            )


def test_item_override_returns_new_plan_and_preserves_unaffected_frozen_values() -> (
    None
):
    original = _task5_directory_plan(item=_task5_exact_item(item_id="item-root"))

    updated = note_import_planner.apply_item_override(
        original,
        "item-root",
        ImportAction.SKIP,
    )

    assert updated is not original
    assert updated.items[0] is not original.items[0]
    assert updated.items[1] is original.items[1]
    assert updated.bounds is original.bounds
    assert updated.proposed_folder_paths == (
        ("Work",),
        ("Work", "Ideas"),
    )
    assert original.items[0].selected_action is ImportAction.CREATE_NEW


def test_collision_and_override_transforms_do_not_write_to_disk(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    analyzed = note_import_planner.analyze_root_collision(
        _task5_directory_plan(),
        ("work",),
    )
    resolved = note_import_planner.resolve_root_collision(
        analyzed,
        RootCollisionChoice.UNIQUE_SIBLING,
        existing_top_level_names=("work",),
    )
    overridden = note_import_planner.apply_item_override(
        resolved,
        "item-root",
        ImportAction.SKIP,
    )
    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))

    assert overridden.items[0].selected_action is ImportAction.SKIP
    assert before == after
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_discovery_uses_the_shared_path_validator_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact shared-validator result, not the raw selection, reaches the OS."""
    selected = tmp_path / "selected.md"
    selected.write_text("selected", encoding="utf-8")
    validated = tmp_path / "validated.md"
    validated.write_text("validated", encoding="utf-8")
    calls: list[tuple[Path, bool, bool]] = []

    def validate(
        path: Path,
        require_exists: bool = False,
        *,
        probe_existing: bool = True,
    ) -> Path:
        calls.append((path, require_exists, probe_existing))
        return validated

    monkeypatch.setattr(
        note_import_discovery,
        "validate_path_simple",
        validate,
        raising=False,
    )

    discovery = discover_import_sources([selected], _discovery_bounds())

    assert calls == [(selected, False, False)]
    assert discovery.candidates[0].source.source_path == validated


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_posix_descriptor_cleanup_propagates_interruptions_after_closing_all(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    """Cleanup finishes every close but never converts a process interruption."""
    closed: list[int] = []

    def close(descriptor: int) -> None:
        closed.append(descriptor)
        if descriptor == 2:
            raise interruption()

    monkeypatch.setattr(note_import_discovery.os, "close", close)

    with pytest.raises(interruption):
        note_import_discovery._close_descriptors((1, 2))

    assert closed == [2, 1]


def test_directory_descriptors_are_opened_close_on_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every POSIX directory descriptor receives close-on-exec hardening."""
    close_on_exec = 0x400000
    monkeypatch.setattr(note_import_discovery.os, "O_CLOEXEC", close_on_exec)

    flags = note_import_discovery._directory_open_flags()

    assert flags & close_on_exec == close_on_exec


@pytest.mark.parametrize(
    "public_function",
    [
        note_import_discovery.discover_import_sources,
        note_import_discovery.read_discovered_source,
        note_import_parsers.parse_import_sources,
        note_import_planner.classify_import_batch,
        note_import_planner.analyze_root_collision,
        note_import_planner.resolve_root_collision,
        note_import_planner.confirm_uncertain_match,
        note_import_planner.apply_item_override,
    ],
)
def test_note_import_public_functions_use_google_style_docstrings(
    public_function: object,
) -> None:
    """Public import functions document inputs, outputs, and failure contracts."""
    documentation = inspect.getdoc(public_function)

    assert documentation is not None
    assert "Args:" in documentation
    assert "Returns:" in documentation
    assert "Raises:" in documentation
