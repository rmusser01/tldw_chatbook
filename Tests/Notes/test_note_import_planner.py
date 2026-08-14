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
    assert diagnostic.items[0].reason == "Ready to import."
    rendered = repr(diagnostic)
    assert "PRIVATE" not in rendered
    assert "/private/user" not in rendered
    assert "fingerprint" not in rendered
    assert "HASH-FINGERPRINT-SECRET" not in rendered
    with pytest.raises(FrozenInstanceError):
        diagnostic.item_count = 2  # type: ignore[misc]


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
