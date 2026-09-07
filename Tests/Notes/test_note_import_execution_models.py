"""Contracts for approved one-time Database Notes import execution models."""

import inspect
from dataclasses import FrozenInstanceError, asdict, replace
from itertools import repeat
from pathlib import Path

import pytest

from tldw_chatbook.Notes import note_import_execution_models, note_import_planner
from tldw_chatbook.Notes.note_import_execution_models import (
    MAX_IMPORT_REASON_CODE_LENGTH,
    ApprovedNoteImportPlan,
    ImportApprovalError,
    ImportEffectState,
    ImportExecutionDiagnostic,
    ImportExecutionProgress,
    ImportExecutionReceipt,
    ImportItemOutcome,
    ImportSessionState,
    approve_note_import_plan,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    MAX_IMPORT_ENTRIES,
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

_APPROVAL_ID = "00000000-0000-4000-8000-000000000001"
_SECRET_BODY = "Body secret that must never enter a public projection"
_SOURCE_PATH = Path("/private/user/Project/notes.json")
_PAYLOAD_DIGEST = "a" * 64
_SOURCE_DIGEST = "b" * 64
_HOSTILE_TEXT_SECRET = "hostile-secret-must-not-escape"


def _item(
    *,
    source_path: Path = _SOURCE_PATH,
    display_path: str = "Project/notes.json",
    payload: ParsedNotePayload | None = None,
    membership_segments: tuple[str, ...] = ("Imported Project", "Meetings"),
    selected_action: ImportAction = ImportAction.UPDATE_EXISTING,
    match: ImportMatch | None = None,
    replace_content: bool | None = None,
    add_membership: bool = True,
) -> ImportPreviewItem:
    if payload is None:
        payload = ParsedNotePayload(
            title="Secret title",
            content=_SECRET_BODY,
            keywords=("private-keyword",),
            template_name="Private template",
        )
    if match is None:
        match = ImportMatch(
            kind=ImportMatchKind.EXACT,
            note_id="note-id-secret",
            note_version=7,
        )
    if replace_content is None:
        replace_content = selected_action is ImportAction.UPDATE_EXISTING
    return ImportPreviewItem(
        item_id="item-1",
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path=display_path,
            source_path=source_path,
        ),
        payloads=(payload,),
        memberships=(
            ProposedFolderMembership(
                payload_index=0,
                folder_segments=membership_segments,
            ),
        ),
        classification=ImportClassification.CHANGED_REPEAT,
        reason="Ready after explicit review.",
        default_action=ImportAction.CREATE_NEW,
        selected_action=selected_action,
        allowed_actions=(
            ImportAction.SKIP,
            ImportAction.CREATE_NEW,
            ImportAction.UPDATE_EXISTING,
        ),
        match=match,
        replace_content=replace_content,
        add_membership=add_membership,
    )


def _plan(
    *,
    item: ImportPreviewItem | None = None,
    bounds: ImportBounds | None = None,
    proposed_folder_paths: tuple[tuple[str, ...], ...] = (
        ("Imported Project",),
        ("Imported Project", "Meetings"),
    ),
    root_collision: RootCollisionState | None = None,
) -> NoteImportPlan:
    if item is None:
        item = _item()
    if bounds is None:
        bounds = ImportBounds(
            max_files=50,
            max_file_bytes=1_000_000,
            max_total_bytes=5_000_000,
            max_depth=8,
            max_reason_length=240,
            max_entries=1_000,
            max_notes_per_file=100,
            max_keywords_per_note=50,
        )
    if root_collision is None:
        root_collision = RootCollisionState(
            proposed_label="Project",
            collides=True,
            choice=RootCollisionChoice.RENAMED_ROOT,
            resolved_label="Imported Project",
        )
    return NoteImportPlan(
        bounds=bounds,
        items=(item,),
        proposed_folder_paths=proposed_folder_paths,
        root_collision=root_collision,
    )


def _receipt(**overrides: object) -> ImportExecutionReceipt:
    values: dict[str, object] = {
        "approval_id": _APPROVAL_ID,
        "state": ImportSessionState.NEEDS_ATTENTION,
        "total": 4,
        "completed": 4,
        "imported": 1,
        "updated": 1,
        "skipped": 1,
        "failed": 1,
        "retryable": 1,
        "reason_code": "target_conflict",
        "_note_ids": ("note-id-secret",),
        "_folder_ids": ("folder-id-secret",),
        "_source_locator_digests": (_SOURCE_DIGEST,),
        "_payload_fingerprints": (_PAYLOAD_DIGEST,),
        "_raw_errors": ("raw failure at /private/user/Project/notes.json",),
    }
    values.update(overrides)
    return ImportExecutionReceipt(**values)  # type: ignore[arg-type]


def test_approval_rejects_an_unresolved_root_collision() -> None:
    unresolved = RootCollisionState(proposed_label="Project", collides=True)

    with pytest.raises(ImportApprovalError, match="resolved") as caught:
        approve_note_import_plan(_plan(root_collision=unresolved))

    assert _SECRET_BODY not in str(caught.value)
    assert str(_SOURCE_PATH) not in str(caught.value)


def test_approval_rejects_a_plan_that_exceeds_the_receipt_ledger_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        note_import_execution_models,
        "MAX_RECEIPT_LEDGER_ROWS",
        5,
        raising=False,
    )

    with pytest.raises(ImportApprovalError, match="receipt ledger") as caught:
        approve_note_import_plan(_plan(), approval_id=_APPROVAL_ID)

    assert _SECRET_BODY not in str(caught.value)
    assert str(_SOURCE_PATH) not in str(caught.value)


def test_approved_plan_is_opaque_and_bound_to_exact_effects() -> None:
    approved = approve_note_import_plan(_plan(), approval_id=_APPROVAL_ID)

    assert approved.approval_id == _APPROVAL_ID
    assert approved.plan is not None
    assert repr(approved) == "ApprovedNoteImportPlan(<private>)"
    assert _SECRET_BODY not in repr(approved)
    assert len(approved._private_plan_digest()) == 64
    assert set(approved._private_plan_digest()) <= set("0123456789abcdef")


def test_approved_plan_public_constructor_does_not_expose_a_digest() -> None:
    signature = str(inspect.signature(ApprovedNoteImportPlan))

    assert "digest" not in signature


def test_direct_construction_cannot_wrap_an_unresolved_plan() -> None:
    unresolved = _plan(
        root_collision=RootCollisionState(proposed_label="Project", collides=True)
    )
    digest = note_import_execution_models._private_plan_digest(unresolved)

    with pytest.raises(ImportApprovalError, match="approve_note_import_plan"):
        ApprovedNoteImportPlan(_APPROVAL_ID, unresolved, digest)  # type: ignore[call-arg]


def test_private_approved_plan_factory_revalidates_root_resolution() -> None:
    unresolved = _plan(
        root_collision=RootCollisionState(proposed_label="Project", collides=True)
    )
    factory = getattr(
        note_import_execution_models,
        "_create_approved_note_import_plan",
        None,
    )

    assert factory is not None
    with pytest.raises(ImportApprovalError, match="resolved") as caught:
        factory(unresolved, _APPROVAL_ID)

    assert _SECRET_BODY not in str(caught.value)
    assert str(_SOURCE_PATH) not in str(caught.value)


@pytest.mark.parametrize(
    "approval_id",
    [
        "not-a-uuid",
        "{00000000-0000-4000-8000-000000000001}",
        "00000000000040008000000000000001",
        "00000000-0000-4000-8000-00000000000G",
        "00000000-0000-4000-8000-000000000001-secret",
    ],
)
def test_approval_rejects_noncanonical_uuid_text_without_echoing_it(
    approval_id: str,
) -> None:
    with pytest.raises(ImportApprovalError) as caught:
        approve_note_import_plan(_plan(), approval_id=approval_id)

    assert approval_id not in str(caught.value)


def test_approval_requires_uuid_text_and_generates_a_canonical_uuid_by_default() -> (
    None
):
    with pytest.raises(ImportApprovalError, match="UUID text"):
        approve_note_import_plan(_plan(), approval_id=object())  # type: ignore[arg-type]

    generated = approve_note_import_plan(_plan())
    assert len(generated.approval_id) == 36
    assert generated.approval_id == generated.approval_id.casefold()


def test_approval_rejects_a_hostile_uuid_text_subclass_without_invoking_it() -> None:
    class HostileApprovalId(str):
        def replace(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError(_HOSTILE_TEXT_SECRET)

    approval_id = HostileApprovalId(_APPROVAL_ID)

    with pytest.raises(ImportApprovalError, match="UUID text") as caught:
        approve_note_import_plan(_plan(), approval_id=approval_id)

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert _APPROVAL_ID not in str(caught.value)
    assert caught.value.__context__ is None


def test_approval_sanitizes_unexpected_private_plan_digest_failures() -> None:
    concrete_path_type = type(Path("/"))

    class HostileSourcePath(concrete_path_type):  # type: ignore[misc,valid-type]
        def __str__(self) -> str:
            raise RuntimeError(_HOSTILE_TEXT_SECRET)

    source_path = HostileSourcePath("/private/user/Project/hostile.json")
    plan = _plan(item=_item(source_path=source_path))

    with pytest.raises(ImportApprovalError, match="validated safely") as caught:
        approve_note_import_plan(plan, approval_id=_APPROVAL_ID)

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert "/private/user" not in str(caught.value)
    assert _APPROVAL_ID not in str(caught.value)
    assert caught.value.__context__ is None


def test_approval_sanitizes_import_approval_errors_from_digest_construction() -> None:
    digest_origin_secret = "DIGEST_SECRET from raw ImportApprovalError"
    concrete_path_type = type(Path("/"))

    class HostileSourcePath(concrete_path_type):  # type: ignore[misc,valid-type]
        def __str__(self) -> str:
            raise ImportApprovalError(digest_origin_secret)

    source_path = HostileSourcePath("/private/user/Project/hostile.json")
    plan = _plan(item=_item(source_path=source_path))

    with pytest.raises(ImportApprovalError, match="validated safely") as caught:
        approve_note_import_plan(plan, approval_id=_APPROVAL_ID)

    assert digest_origin_secret not in str(caught.value)
    assert "raw ImportApprovalError" not in str(caught.value)
    assert "/private/user" not in str(caught.value)
    assert caught.value.__context__ is None


def test_approval_rejects_a_hostile_collision_subclass_before_attribute_dispatch() -> (
    None
):
    class HostileCollision(RootCollisionState):
        __slots__ = ("_hostile",)

        def __getattribute__(self, name: str) -> object:
            if name == "collides":
                try:
                    hostile = object.__getattribute__(self, "_hostile")
                except AttributeError:
                    hostile = False
                if hostile:
                    raise RuntimeError(_HOSTILE_TEXT_SECRET)
            return super().__getattribute__(name)

    collision = HostileCollision(proposed_label="Project", collides=False)
    object.__setattr__(collision, "_hostile", True)
    plan = _plan(root_collision=collision)

    with pytest.raises(ImportApprovalError, match="collision state") as caught:
        approve_note_import_plan(plan, approval_id=_APPROVAL_ID)

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert caught.value.__context__ is None


def test_plan_digest_is_deterministic_for_the_same_authority() -> None:
    first = approve_note_import_plan(_plan(), approval_id=_APPROVAL_ID)
    second = approve_note_import_plan(_plan(), approval_id=_APPROVAL_ID)

    assert first._private_plan_digest() == second._private_plan_digest()


def test_execution_payload_fingerprint_matches_the_planner_contract() -> None:
    payloads = (
        ParsedNotePayload(
            title="Cafe\u0301 notes",
            content="Re\u0301sume\u0301 body",
            keywords=("na\u0308ive", "東京"),
            template_name="Re\u0301union",
        ),
        ParsedNotePayload(
            title="Second",
            content="Emoji 📝",
            keywords=("two",),
            template_name=None,
        ),
    )

    assert note_import_execution_models._private_payload_fingerprint(
        payloads
    ) == note_import_planner._private_payload_fingerprint(payloads)


def _authority_variants() -> tuple[NoteImportPlan, ...]:
    base_item = _item()
    base_payload = base_item.payloads[0]
    base_match = base_item.match
    assert base_match is not None
    base_bounds = _plan().bounds
    return (
        _plan(item=replace(base_item, item_id="item-2")),
        _plan(
            item=replace(
                base_item,
                source=replace(
                    base_item.source,
                    kind=ImportSourceKind.SELECTED_FILE,
                ),
            )
        ),
        _plan(item=_item(source_path=Path("/private/user/Project/other.json"))),
        _plan(item=_item(display_path="Project/other.json")),
        _plan(item=_item(payload=replace(base_payload, title="Different title"))),
        _plan(item=_item(payload=replace(base_payload, content="Different body"))),
        _plan(
            item=_item(payload=replace(base_payload, keywords=("different-keyword",)))
        ),
        _plan(
            item=_item(
                payload=replace(base_payload, template_name="Different template")
            )
        ),
        _plan(
            item=_item(
                selected_action=ImportAction.CREATE_NEW,
                replace_content=False,
            )
        ),
        _plan(item=_item(replace_content=False, add_membership=True)),
        _plan(item=_item(replace_content=True, add_membership=False)),
        _plan(item=_item(match=replace(base_match, note_id="other-note-id"))),
        _plan(item=_item(match=replace(base_match, note_version=8))),
        _plan(
            item=replace(
                base_item,
                classification=ImportClassification.UNCERTAIN_MATCH,
                match=replace(base_match, kind=ImportMatchKind.USER_CONFIRMED),
            )
        ),
        _plan(item=_item(membership_segments=("Imported Project", "Different folder"))),
        _plan(
            proposed_folder_paths=(
                ("Imported Project",),
                ("Imported Project", "Different folder"),
            )
        ),
        _plan(bounds=replace(base_bounds, max_files=51)),
        _plan(bounds=replace(base_bounds, max_file_bytes=1_000_001)),
        _plan(bounds=replace(base_bounds, max_total_bytes=5_000_001)),
        _plan(bounds=replace(base_bounds, max_depth=9)),
        _plan(bounds=replace(base_bounds, max_reason_length=241)),
        _plan(bounds=replace(base_bounds, max_entries=1_001)),
        _plan(bounds=replace(base_bounds, max_notes_per_file=101)),
        _plan(bounds=replace(base_bounds, max_keywords_per_note=51)),
        _plan(
            root_collision=RootCollisionState(
                proposed_label="Project",
                collides=True,
                choice=RootCollisionChoice.UNIQUE_SIBLING,
                resolved_label="Imported Project",
            )
        ),
        _plan(
            root_collision=RootCollisionState(
                proposed_label="Different Project",
                collides=True,
                choice=RootCollisionChoice.RENAMED_ROOT,
                resolved_label="Imported Project",
            )
        ),
        _plan(
            proposed_folder_paths=(
                ("Different Imported",),
                ("Different Imported", "Meetings"),
            ),
            root_collision=RootCollisionState(
                proposed_label="Project",
                collides=True,
                choice=RootCollisionChoice.RENAMED_ROOT,
                resolved_label="Different Imported",
            ),
        ),
    )


@pytest.mark.parametrize("variant", _authority_variants())
def test_plan_digest_changes_for_each_authority_bearing_field(
    variant: NoteImportPlan,
) -> None:
    baseline = approve_note_import_plan(_plan(), approval_id=_APPROVAL_ID)
    changed = approve_note_import_plan(variant, approval_id=_APPROVAL_ID)

    assert baseline._private_plan_digest() != changed._private_plan_digest()


def test_plan_digest_is_sensitive_to_membership_payload_indexes() -> None:
    payloads = (
        ParsedNotePayload(title="First", content="First body"),
        ParsedNotePayload(title="Second", content="Second body"),
    )
    source = ImportSource(
        kind=ImportSourceKind.DIRECTORY_MEMBER,
        display_path="Project/two-notes.json",
        source_path=Path("/private/user/Project/two-notes.json"),
    )

    def item_with_indexes(first: int, second: int) -> ImportPreviewItem:
        return ImportPreviewItem(
            item_id="two-notes",
            source=source,
            payloads=payloads,
            memberships=(
                ProposedFolderMembership(
                    payload_index=first,
                    folder_segments=("Imported Project", "First"),
                ),
                ProposedFolderMembership(
                    payload_index=second,
                    folder_segments=("Imported Project", "Second"),
                ),
            ),
            classification=ImportClassification.NEW,
            reason="Ready to import.",
            default_action=ImportAction.CREATE_NEW,
            selected_action=ImportAction.CREATE_NEW,
            allowed_actions=(ImportAction.SKIP, ImportAction.CREATE_NEW),
            match=None,
            replace_content=False,
            add_membership=True,
        )

    first = approve_note_import_plan(
        _plan(item=item_with_indexes(0, 1)),
        approval_id=_APPROVAL_ID,
    )
    second = approve_note_import_plan(
        _plan(item=item_with_indexes(1, 0)),
        approval_id=_APPROVAL_ID,
    )

    assert first._private_plan_digest() != second._private_plan_digest()


def test_execution_projection_enums_have_stable_values() -> None:
    assert {state.value for state in ImportSessionState} == {
        "pending",
        "running",
        "cancelled",
        "completed",
        "needs_attention",
    }
    assert {outcome.value for outcome in ImportItemOutcome} == {
        "pending",
        "imported",
        "updated",
        "skipped",
        "failed",
    }
    assert {state.value for state in ImportEffectState} == {
        "pending",
        "applied",
        "failed",
    }


def test_public_execution_diagnostic_contains_counts_not_private_fields() -> None:
    receipt = _receipt()
    diagnostic = receipt.to_diagnostic()
    rendered = repr(diagnostic)
    serialized = asdict(diagnostic)

    assert diagnostic.imported == 1
    assert serialized == {
        "state": ImportSessionState.NEEDS_ATTENTION,
        "total": 4,
        "completed": 4,
        "imported": 1,
        "updated": 1,
        "skipped": 1,
        "failed": 1,
        "retryable": 1,
        "reason_code": "target_conflict",
    }
    forbidden = (
        _APPROVAL_ID,
        "note-id-secret",
        "folder-id-secret",
        "Project/notes.json",
        "/private/user",
        _SOURCE_DIGEST,
        _PAYLOAD_DIGEST,
        "raw failure",
        "_note_ids",
        "_folder_ids",
        "source",
        "fingerprint",
        "raw_errors",
    )
    for secret in forbidden:
        assert secret not in rendered
        assert secret not in str(serialized)


def test_receipt_copies_private_collections_and_hides_them_from_repr() -> None:
    note_ids = ["note-id-secret"]
    folder_ids = ["folder-id-secret"]
    raw_errors = ["raw private error"]
    receipt = _receipt(
        _note_ids=note_ids,
        _folder_ids=folder_ids,
        _raw_errors=raw_errors,
    )
    note_ids.append("later")
    folder_ids.append("later")
    raw_errors.append("later")

    assert receipt._note_ids == ("note-id-secret",)
    assert receipt._folder_ids == ("folder-id-secret",)
    assert receipt._raw_errors == ("raw private error",)
    assert "note-id-secret" not in repr(receipt)
    assert "folder-id-secret" not in repr(receipt)
    assert "raw private error" not in repr(receipt)


def test_receipt_hides_approval_id_from_repr() -> None:
    receipt = _receipt()

    assert _APPROVAL_ID not in repr(receipt)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("_note_ids", ("",)),
        ("_folder_ids", ("folder\x00id",)),
        ("_source_locator_digests", ("not-a-digest",)),
        ("_payload_fingerprints", ("A" * 64,)),
        ("_raw_errors", ("",)),
        ("_raw_errors", ("x" * 4097,)),
    ],
)
def test_receipt_rejects_invalid_private_values_without_echoing_them(
    field_name: str,
    value: tuple[str, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)) as caught:
        _receipt(**{field_name: value})

    if value[0]:
        assert value[0] not in str(caught.value)
    assert field_name not in str(caught.value)


def test_private_collection_iterator_errors_are_sanitized() -> None:
    class ExplodingPrivateValues:
        def __iter__(self) -> object:
            raise RuntimeError(_SECRET_BODY)

    with pytest.raises(ValueError, match="private collection") as caught:
        _receipt(_note_ids=ExplodingPrivateValues())

    assert _SECRET_BODY not in str(caught.value)
    assert "_note_ids" not in str(caught.value)


def test_private_collection_type_dispatch_and_iterator_errors_are_sanitized() -> None:
    class HostilePrivateValues:
        @property
        def __class__(self) -> type[object]:
            raise RuntimeError(_HOSTILE_TEXT_SECRET)

        def __iter__(self) -> object:
            raise RuntimeError(_HOSTILE_TEXT_SECRET)

    with pytest.raises(ValueError, match="private collection") as caught:
        _receipt(_note_ids=HostilePrivateValues())

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert "_note_ids" not in str(caught.value)
    assert caught.value.__context__ is None


def test_private_collection_ceiling_matches_the_planner_entry_limit() -> None:
    assert (
        note_import_execution_models.MAX_PRIVATE_IMPORT_COLLECTION_ITEMS
        == MAX_IMPORT_ENTRIES
    )


def test_private_collection_accepts_exactly_the_public_safety_ceiling() -> None:
    receipt = _receipt(_note_ids=repeat("note-id", MAX_IMPORT_ENTRIES))

    assert len(receipt._note_ids) == MAX_IMPORT_ENTRIES


def test_private_collection_rejects_one_item_over_the_public_safety_ceiling() -> None:
    with pytest.raises(ValueError, match="safety ceiling") as caught:
        _receipt(_note_ids=repeat("note-id", MAX_IMPORT_ENTRIES + 1))

    assert "_note_ids" not in str(caught.value)
    assert caught.value.__context__ is None


def test_private_collection_stops_an_infinite_iterator_at_ceiling_plus_one() -> None:
    class CountingInfinitePrivateValues:
        def __init__(self) -> None:
            self.pulls = 0

        def __iter__(self) -> "CountingInfinitePrivateValues":
            return self

        def __next__(self) -> str:
            self.pulls += 1
            if self.pulls > MAX_IMPORT_ENTRIES + 1:
                raise RuntimeError(_HOSTILE_TEXT_SECRET)
            return "note-id"

        def __repr__(self) -> str:
            return _HOSTILE_TEXT_SECRET

    values = CountingInfinitePrivateValues()

    with pytest.raises(ValueError, match="safety ceiling") as caught:
        _receipt(_note_ids=values)

    assert values.pulls == MAX_IMPORT_ENTRIES + 1
    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert "_note_ids" not in str(caught.value)
    assert caught.value.__context__ is None


def test_private_receipt_ids_require_exact_builtin_strings() -> None:
    class HostilePrivateId(str):
        def __repr__(self) -> str:
            return _HOSTILE_TEXT_SECRET

    with pytest.raises(ValueError) as caught:
        _receipt(_note_ids=(HostilePrivateId("safe-private-id"),))

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert "_note_ids" not in str(caught.value)
    assert caught.value.__context__ is None


def test_private_raw_errors_require_exact_builtin_strings_without_invocation() -> None:
    class HostileRawError(str):
        def __len__(self) -> int:
            raise RuntimeError(_HOSTILE_TEXT_SECRET)

    with pytest.raises(ValueError) as caught:
        _receipt(_raw_errors=(HostileRawError("safe-private-error"),))

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert "_raw_errors" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "reason_code",
    [
        "",
        "TargetConflict",
        "target conflict",
        "../private-path",
        "target-conflict",
        "café",
        "x" * (MAX_IMPORT_REASON_CODE_LENGTH + 1),
    ],
)
def test_reason_code_is_a_bounded_safe_machine_token_without_echo(
    reason_code: str,
) -> None:
    with pytest.raises(ValueError, match="reason_code") as caught:
        _receipt(reason_code=reason_code)

    if reason_code:
        assert reason_code not in str(caught.value)


def test_reason_code_rejects_coerced_values_and_accepts_none() -> None:
    with pytest.raises(TypeError, match="reason_code"):
        _receipt(reason_code=17)

    assert _receipt(reason_code=None).reason_code is None


@pytest.mark.parametrize(
    "projection_type",
    [
        ImportExecutionProgress,
        ImportExecutionDiagnostic,
        ImportExecutionReceipt,
    ],
)
def test_execution_projections_reject_hostile_reason_code_subclasses(
    projection_type: type[
        ImportExecutionProgress | ImportExecutionDiagnostic | ImportExecutionReceipt
    ],
) -> None:
    class HostileReasonCode(str):
        def __repr__(self) -> str:
            return _HOSTILE_TEXT_SECRET

    reason_code = HostileReasonCode("safe_token")
    values: dict[str, object] = {
        "state": ImportSessionState.COMPLETED,
        "total": 1,
        "completed": 1,
        "imported": 1,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "retryable": 0,
        "reason_code": reason_code,
    }
    if projection_type is ImportExecutionReceipt:
        values["approval_id"] = _APPROVAL_ID

    with pytest.raises(TypeError, match="reason_code") as caught:
        projection_type(**values)  # type: ignore[arg-type]

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"total": -1},
        {"completed": -1},
        {"imported": True},
        {"completed": 3},
        {"total": 3},
        {"retryable": 2},
        {"state": "completed"},
    ],
)
def test_execution_counts_and_state_fail_closed(overrides: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _receipt(**overrides)


@pytest.mark.parametrize(
    "projection_type",
    [
        ImportExecutionProgress,
        ImportExecutionDiagnostic,
        ImportExecutionReceipt,
    ],
)
@pytest.mark.parametrize(
    "invalid_counts",
    [
        {
            "state": ImportSessionState.PENDING,
            "total": 1,
            "completed": 1,
            "imported": 1,
        },
        {
            "state": ImportSessionState.COMPLETED,
            "total": 2,
            "completed": 1,
            "imported": 1,
        },
        {
            "state": ImportSessionState.COMPLETED,
            "total": 1,
            "completed": 1,
            "failed": 1,
            "retryable": 1,
        },
    ],
)
def test_execution_projections_reject_counts_that_contradict_their_state(
    projection_type: type[
        ImportExecutionProgress | ImportExecutionDiagnostic | ImportExecutionReceipt
    ],
    invalid_counts: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "state": ImportSessionState.RUNNING,
        "total": 1,
        "completed": 0,
        "imported": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "retryable": 0,
        "reason_code": None,
    }
    values.update(invalid_counts)
    if projection_type is ImportExecutionReceipt:
        values["approval_id"] = _APPROVAL_ID

    with pytest.raises(ValueError):
        projection_type(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "projection_type",
    [
        ImportExecutionProgress,
        ImportExecutionDiagnostic,
        ImportExecutionReceipt,
    ],
)
@pytest.mark.parametrize(
    "valid_counts",
    [
        {
            "state": ImportSessionState.COMPLETED,
            "total": 0,
            "completed": 0,
        },
        {
            "state": ImportSessionState.NEEDS_ATTENTION,
            "total": 2,
            "completed": 1,
            "failed": 1,
            "retryable": 1,
            "reason_code": "target_conflict",
        },
    ],
)
def test_execution_projections_accept_truthful_terminal_states(
    projection_type: type[
        ImportExecutionProgress | ImportExecutionDiagnostic | ImportExecutionReceipt
    ],
    valid_counts: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "state": ImportSessionState.RUNNING,
        "total": 0,
        "completed": 0,
        "imported": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "retryable": 0,
        "reason_code": None,
    }
    values.update(valid_counts)
    if projection_type is ImportExecutionReceipt:
        values["approval_id"] = _APPROVAL_ID

    projection = projection_type(**values)  # type: ignore[arg-type]

    assert projection.state is valid_counts["state"]
    assert projection.completed == valid_counts["completed"]


def test_execution_state_rejects_a_spoofed_import_session_state_class() -> None:
    class SpoofedSessionState:
        @property
        def __class__(self) -> type[ImportSessionState]:
            return ImportSessionState

        def __repr__(self) -> str:
            return _HOSTILE_TEXT_SECRET

    with pytest.raises(TypeError, match="ImportSessionState") as caught:
        ImportExecutionProgress(
            state=SpoofedSessionState(),  # type: ignore[arg-type]
            total=0,
            completed=0,
            imported=0,
            updated=0,
            skipped=0,
            failed=0,
            retryable=0,
        )

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert caught.value.__context__ is None


def test_execution_state_does_not_dispatch_a_raising_class_property() -> None:
    class RaisingSessionState:
        @property
        def __class__(self) -> type[object]:
            raise RuntimeError(_HOSTILE_TEXT_SECRET)

        def __repr__(self) -> str:
            return _HOSTILE_TEXT_SECRET

    with pytest.raises(TypeError, match="ImportSessionState") as caught:
        ImportExecutionProgress(
            state=RaisingSessionState(),  # type: ignore[arg-type]
            total=0,
            completed=0,
            imported=0,
            updated=0,
            skipped=0,
            failed=0,
            retryable=0,
        )

    assert _HOSTILE_TEXT_SECRET not in str(caught.value)
    assert caught.value.__context__ is None


def test_execution_projections_are_frozen() -> None:
    progress = ImportExecutionProgress(
        state=ImportSessionState.RUNNING,
        total=4,
        completed=1,
        imported=1,
        updated=0,
        skipped=0,
        failed=0,
        retryable=0,
        reason_code=None,
    )
    receipt = _receipt()
    diagnostic = receipt.to_diagnostic()
    approved = approve_note_import_plan(_plan(), approval_id=_APPROVAL_ID)

    for projection in (progress, receipt, diagnostic):
        with pytest.raises(FrozenInstanceError):
            projection.state = ImportSessionState.COMPLETED  # type: ignore[attr-defined,misc]
    with pytest.raises(FrozenInstanceError):
        approved.approval_id = "00000000-0000-4000-8000-000000000002"  # type: ignore[misc]


def test_diagnostic_is_independently_validated() -> None:
    diagnostic = ImportExecutionDiagnostic(
        state=ImportSessionState.COMPLETED,
        total=2,
        completed=2,
        imported=1,
        updated=1,
        skipped=0,
        failed=0,
        retryable=0,
        reason_code="completed",
    )

    assert diagnostic.completed == 2
    with pytest.raises(ValueError, match="completed"):
        replace(diagnostic, completed=1)


def test_approved_plan_requires_a_real_note_import_plan() -> None:
    with pytest.raises(ImportApprovalError, match="NoteImportPlan"):
        approve_note_import_plan(object())  # type: ignore[arg-type]

    with pytest.raises(ImportApprovalError, match="approve_note_import_plan"):
        ApprovedNoteImportPlan(
            approval_id=_APPROVAL_ID,
            plan=object(),  # type: ignore[arg-type]
            _ApprovedNoteImportPlan__plan_digest="c" * 64,
        )
