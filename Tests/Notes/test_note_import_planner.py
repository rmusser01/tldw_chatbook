"""Contract tests for one-time Database Notes import planning."""

import os
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from tldw_chatbook.Notes import note_import_plan_models, note_import_planner
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
            note_import_planner.os,
            "fstat",
            lambda _descriptor: (_ for _ in ()).throw(OSError(private_error)),
        )
    else:
        real_close = os.close

        def close_then_fail(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError(private_error)

        monkeypatch.setattr(note_import_planner.os, "close", close_then_fail)

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
        note_import_planner.os,
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

    monkeypatch.setattr(note_import_planner.os, "close", close_then_fail)

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

    def injected_scandir(descriptor: int) -> os.ScandirIterator[str]:
        if operation == "scandir" and descriptor in child_descriptors:
            raise OSError(private_error)
        return real_scandir(descriptor)

    monkeypatch.setattr(note_import_planner.os, "open", tracked_open)
    monkeypatch.setattr(note_import_planner.os, "fstat", injected_fstat)
    monkeypatch.setattr(note_import_planner.os, "close", injected_close)
    monkeypatch.setattr(note_import_planner.os, "scandir", injected_scandir)

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

    def injected_scandir(descriptor: int) -> os.ScandirIterator[str]:
        if boundary == "root_scan" and descriptor in project_fds:
            raise error_type("unexpected discovery failure")
        return real_scandir(descriptor)

    monkeypatch.setattr(note_import_planner.os, "open", tracked_open)
    monkeypatch.setattr(note_import_planner.os, "fstat", injected_fstat)
    monkeypatch.setattr(note_import_planner.os, "close", tracked_close)
    monkeypatch.setattr(note_import_planner.os, "scandir", injected_scandir)

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

    monkeypatch.setattr(note_import_planner.os, "open", racing_open)
    monkeypatch.setattr(note_import_planner.os, "close", tracked_close)

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
    monkeypatch.delattr(note_import_planner.os, "O_NOFOLLOW")

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
    monkeypatch.setattr(note_import_planner.os, "open", guarded_open)

    discovery = discover_import_sources([root], _discovery_bounds())

    assert discovery.candidates == ()
    assert composed.stat_calls == 1
    assert decomposed.stat_calls == 1
    assert [failure.reason_code for failure in discovery.failures] == [
        "nested_unavailable",
        "nested_unavailable",
    ]
