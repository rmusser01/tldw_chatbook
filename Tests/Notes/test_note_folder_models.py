"""Contract tests for normalized Database Note folder models."""

import unicodedata
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_chatbook.Notes.note_folder_models import (
    FolderCapabilityError,
    FolderPlacementId,
    FolderValidationError,
    NormalizedFolderName,
    NoteFolder,
    NoteFolderChildPage,
    NoteFolderMembership,
    NotePlacementPage,
    NotePlacementRecord,
    NoteTreeLocation,
    NoteTreeMutationContext,
    NoteTreePathStep,
    join_normalized_folder_path,
    normalize_folder_name,
)


def _folder(folder_id: str, parent_id: str | None = None) -> NoteFolder:
    return NoteFolder(
        folder_id=folder_id,
        parent_id=parent_id,
        name=folder_id,
        path=f"/{folder_id}",
        normalized_path=f"/{folder_id}",
        version=1,
        deleted=False,
    )


def _membership(
    membership_id: str, folder_id: str, note_id: str
) -> NoteFolderMembership:
    return NoteFolderMembership(
        membership_id=membership_id,
        folder_id=folder_id,
        note_id=note_id,
        ownership="manual",
        owner_id="user-1",
        owner_active=True,
        version=1,
    )


def test_child_page_preserves_exact_paging_metadata() -> None:
    page = NoteFolderChildPage(
        folders=(_folder("f1"),),
        total_folders=41,
        start_offset=20,
        previous_offset=0,
        next_offset=40,
    )

    assert page.total_folders == 41
    assert page.start_offset == 20
    assert page.previous_offset == 0
    assert page.next_offset == 40


def test_placement_page_preserves_duplicate_membership_identity_and_ancestors() -> None:
    page = NotePlacementPage(
        placements=(
            NotePlacementRecord(
                note={"id": "n1", "title": "One"},
                folder_id="f1",
                membership=_membership("m1", "f1", "n1"),
            ),
            NotePlacementRecord(
                note={"id": "n1", "title": "One"},
                folder_id="f1",
                membership=_membership("m2", "f1", "n1"),
            ),
        ),
        total_placements=41,
        start_offset=20,
        previous_offset=0,
        next_offset=40,
        ancestor_folders=(_folder("parent"), _folder("f1", "parent")),
    )

    assert [row.membership.membership_id for row in page.placements] == ["m1", "m2"]
    assert [folder.folder_id for folder in page.ancestor_folders] == ["parent", "f1"]


@pytest.mark.parametrize(
    ("page_type", "kwargs"),
    [
        (
            NoteFolderChildPage,
            {
                "folders": (),
                "total_folders": -1,
                "start_offset": 0,
                "previous_offset": None,
                "next_offset": None,
            },
        ),
        (
            NotePlacementPage,
            {
                "placements": (),
                "total_placements": 0,
                "start_offset": -1,
                "previous_offset": None,
                "next_offset": None,
            },
        ),
        (
            NotePlacementPage,
            {
                "placements": (),
                "total_placements": 0,
                "start_offset": 0,
                "previous_offset": -1,
                "next_offset": None,
            },
        ),
    ],
)
def test_page_envelopes_reject_negative_metadata(page_type: type, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        page_type(**kwargs)


@pytest.mark.parametrize(
    "page_factory",
    [
        lambda: NoteFolderChildPage((), 3, 20, 0, None),
        lambda: NotePlacementPage((), 3, 20, 0, None),
    ],
)
def test_page_envelopes_allow_empty_out_of_range_responses(page_factory) -> None:
    page = page_factory()
    assert page.start_offset == 20  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("page_type", "items_field", "items"),
    [
        (NoteFolderChildPage, "folders", (_folder("f1"),)),
        (
            NotePlacementPage,
            "placements",
            (NotePlacementRecord({"id": "n1"}, None, None),),
        ),
    ],
)
def test_page_envelopes_reject_nonempty_out_of_range_responses(
    page_type: type, items_field: str, items: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError):
        page_type(
            **{
                items_field: items,
                "total_folders" if items_field == "folders" else "total_placements": 3,
                "start_offset": 20,
                "previous_offset": 0,
                "next_offset": None,
            }
        )


@pytest.mark.parametrize(
    "page",
    [
        lambda: NoteFolderChildPage((_folder("f1"),), 3, 0, None, 4),
        lambda: NotePlacementPage(
            (NotePlacementRecord({"id": "n1"}, None, None),), 3, 0, None, 4
        ),
        lambda: NoteFolderChildPage((), 3, 1, 0, 2),
        lambda: NotePlacementPage((), 3, 1, 0, 2),
    ],
)
def test_page_envelopes_reject_impossible_cursor_and_empty_page_states(page) -> None:
    with pytest.raises(ValueError):
        page()


def test_tree_location_records_target_folder_path() -> None:
    location = NoteTreeLocation(
        placement_id=FolderPlacementId.folder("f2"),
        note_id=None,
        membership_id=None,
        path=(
            NoteTreePathStep(folder_id="f1", parent_id=None, containing_offset=3),
            NoteTreePathStep(folder_id="f2", parent_id="f1", containing_offset=7),
        ),
        placement_offset=None,
    )

    assert location.path[-1] == NoteTreePathStep("f2", "f1", 7)


def test_tree_location_enforces_folder_and_note_optional_field_contracts() -> None:
    with pytest.raises(ValueError):
        NoteTreeLocation(
            placement_id=FolderPlacementId.folder("f1"),
            note_id=None,
            membership_id="m1",
            path=(NoteTreePathStep("f1", None, 0),),
            placement_offset=None,
        )
    with pytest.raises(ValueError):
        NoteTreeLocation(
            placement_id=FolderPlacementId.unfiled("n1"),
            note_id="n1",
            membership_id="m1",
            path=(),
            placement_offset=0,
        )
    with pytest.raises(ValueError):
        NoteTreeLocation(
            placement_id=FolderPlacementId.note("f1", "n1", "m1"),
            note_id="n1",
            membership_id="m1",
            path=(NoteTreePathStep("f1", None, 0),),
            placement_offset=None,
        )


def test_tree_location_accepts_unfiled_and_exact_membership_note_placements() -> None:
    unfiled = NoteTreeLocation(
        placement_id=FolderPlacementId.unfiled("n1"),
        note_id="n1",
        membership_id=None,
        path=(),
        placement_offset=4,
    )
    filed = NoteTreeLocation(
        placement_id=FolderPlacementId.note("f1", "n1", "m1"),
        note_id="n1",
        membership_id="m1",
        path=(NoteTreePathStep("f1", None, 2),),
        placement_offset=4,
    )

    assert unfiled.membership_id is None
    assert filed.membership_id == "m1"


def test_tree_mutation_context_is_an_immutable_tuple_snapshot() -> None:
    context = NoteTreeMutationContext(
        folder_ids=("f1",),
        parent_ids=(None,),
        ancestor_ids=("root",),
        placement_parent_ids=("f1",),
    )

    with pytest.raises(FrozenInstanceError):
        context.folder_ids = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("  Résumé  ", NormalizedFolderName(display="Résumé", key="résumé")),
        ("CAFÉ", NormalizedFolderName(display="CAFÉ", key="café")),
        ("Straße", NormalizedFolderName(display="Straße", key="strasse")),
    ],
)
def test_normalize_folder_name_returns_trimmed_display_and_unicode_key(
    raw_name: str, expected: NormalizedFolderName
) -> None:
    """Normalization catches untrimmed names or incomplete Unicode matching."""
    assert normalize_folder_name(raw_name) == expected


@pytest.mark.parametrize("raw_name", ["", "   ", ".", "..", "a/b", "a\\b", "a\x00b"])
def test_normalize_folder_name_rejects_invalid_segments(raw_name: str) -> None:
    """Invalid segments cannot become ambiguous or unsafe path components."""
    with pytest.raises(FolderValidationError):
        normalize_folder_name(raw_name)


@pytest.mark.parametrize("raw_name", ["／", "＼", "．", "．．"])
def test_normalize_folder_name_rejects_compatibility_path_segments(
    raw_name: str,
) -> None:
    """NFKC compatibility characters cannot introduce path syntax."""
    with pytest.raises(FolderValidationError):
        normalize_folder_name(raw_name)


@pytest.mark.parametrize("raw_name", [None, 42, b"folder"])
def test_normalize_folder_name_rejects_non_strings(raw_name: object) -> None:
    """Non-text inputs must not silently gain a string representation."""
    with pytest.raises(FolderValidationError):
        normalize_folder_name(raw_name)  # type: ignore[arg-type]


def test_normalize_folder_name_enforces_display_length_limit() -> None:
    """The persisted display segment is limited to 255 characters."""
    assert normalize_folder_name("a" * 255).display == "a" * 255
    with pytest.raises(FolderValidationError):
        normalize_folder_name("a" * 256)


def test_normalize_folder_name_matches_composed_and_decomposed_accents() -> None:
    """Equivalent Unicode spellings must share one collision key."""
    assert normalize_folder_name("Café").key == normalize_folder_name("Cafe\u0301").key


def test_join_normalized_folder_path_preserves_single_root_separator() -> None:
    """Path joining catches malformed root and nested path construction."""
    assert join_normalized_folder_path("", "work") == "/work"
    assert join_normalized_folder_path("/work", "plans") == "/work/plans"


def test_folder_placement_ids_include_folder_context() -> None:
    """Repeated note placements must not collapse to one tree identity."""
    assert FolderPlacementId.folder("work") == "folder:work"
    assert FolderPlacementId.note("work", "note-1") == "note:work:note-1"
    assert FolderPlacementId.note("personal", "note-1") == "note:personal:note-1"
    assert FolderPlacementId.note("work", "note-1") != FolderPlacementId.note(
        "personal", "note-1"
    )
    assert FolderPlacementId.unfiled("note-1") == "unfiled:note-1"


def test_folder_placement_ids_escape_delimiters_in_opaque_ids() -> None:
    """Different opaque folder/note pairs cannot collapse to one tree identity."""
    assert FolderPlacementId.note("a:b", "c") != FolderPlacementId.note("a", "b:c")
    assert FolderPlacementId.note("a:b", "c") == "note:a%3Ab:c"
    assert FolderPlacementId.folder("a:b") == "folder:a%3Ab"
    assert FolderPlacementId.unfiled("a:b") == "unfiled:a%3Ab"


@pytest.mark.parametrize(
    ("parent_path", "child_key"),
    [
        ("relative", "child"),
        ("/work/../admin", "child"),
        ("/work//nested", "child"),
        ("/work", ""),
        ("/work", "."),
        ("/work", ".."),
        ("/work", "nested/child"),
    ],
)
def test_join_normalized_folder_path_rejects_ambiguous_inputs(
    parent_path: str, child_key: str
) -> None:
    """The public joiner fails closed on traversal and non-canonical inputs."""
    with pytest.raises(FolderValidationError):
        join_normalized_folder_path(parent_path, child_key)


def test_core_folder_records_are_immutable() -> None:
    """Callers cannot mutate folder or membership snapshots after a read."""
    folder = NoteFolder(
        folder_id="folder-1",
        parent_id=None,
        name="Work",
        path="/work",
        normalized_path="/work",
        version=1,
        deleted=False,
    )
    membership = NoteFolderMembership(
        membership_id="membership-1",
        folder_id="folder-1",
        note_id="note-1",
        ownership="manual",
        owner_id="user-1",
        owner_active=True,
        version=1,
    )

    with pytest.raises(FrozenInstanceError):
        folder.name = "Personal"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        membership.owner_active = False  # type: ignore[misc]


def test_capability_error_retains_machine_and_user_context() -> None:
    """Unsupported operations expose stable UI and programmatic error details."""
    error = FolderCapabilityError(
        reason_code="remote_mutation_unsupported",
        user_message="This source cannot rename folders.",
    )

    assert error.reason_code == "remote_mutation_unsupported"
    assert error.user_message == "This source cannot rename folders."
    assert str(error) == "This source cannot rename folders."


def _has_safe_normalized_key(value: str) -> bool:
    key = unicodedata.normalize("NFKC", value).casefold()
    return (
        bool(key)
        and key not in {".", ".."}
        and "/" not in key
        and "\\" not in key
        and "\x00" not in key
    )


_VALID_FOLDER_CORE = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cc", "Cs", "Zl", "Zp", "Zs"),
        blacklist_characters="\x00/\\",
    ),
    min_size=1,
    max_size=255,
).filter(_has_safe_normalized_key)
_VALID_FOLDER_DISPLAY = st.builds(
    lambda leading, core, trailing: f"{leading}{core}{trailing}",
    st.sampled_from(["", " ", "\t", "  \t"]),
    _VALID_FOLDER_CORE,
    st.sampled_from(["", " ", "\t", "\t  "]),
)
_VALID_NORMALIZED_SEGMENT = _VALID_FOLDER_CORE


@given(_VALID_FOLDER_DISPLAY)
def test_normalize_folder_name_is_idempotent_for_valid_display(raw_name: str) -> None:
    """A normalized display remains stable when normalized a second time."""
    normalized = normalize_folder_name(raw_name)
    assert normalize_folder_name(normalized.display) == normalized


@given(_VALID_FOLDER_DISPLAY)
def test_normalize_folder_name_key_is_nfkc_casefold(raw_name: str) -> None:
    """The collision key follows the specified Unicode normalization contract."""
    normalized = normalize_folder_name(raw_name)
    assert (
        normalized.key == unicodedata.normalize("NFKC", normalized.display).casefold()
    )


@given(st.lists(_VALID_NORMALIZED_SEGMENT, min_size=1, max_size=8))
def test_join_normalized_folder_path_keeps_valid_segments_unambiguous(
    segments: list[str],
) -> None:
    """Joining valid normalized segments cannot introduce unsafe path components."""
    path = ""
    for segment in segments:
        key = normalize_folder_name(segment).key
        path = join_normalized_folder_path(path, key)

    components = path.split("/")
    assert "//" not in path
    assert "." not in components
    assert ".." not in components
