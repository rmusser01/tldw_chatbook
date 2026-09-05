"""Chatbook 3.0 Canvas manifest contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from hashlib import sha256
from traceback import format_exception
from uuid import NAMESPACE_URL, uuid5

import pytest

from tldw_chatbook.Canvas.archive import (
    CANVAS_ARCHIVE_EXTENSION_VERSION,
    CanvasArchiveValidationError,
    canvas_revision_source_path,
)
from tldw_chatbook.Chatbooks.chatbook_models import (
    CanvasArchiveDocument,
    CanvasArchiveManifest,
    CanvasArchiveReopenHint,
    CanvasArchiveRevision,
    ChatbookManifest,
    ChatbookVersion,
    select_chatbook_version,
)

CANVAS_ID = "2c477c1f-b388-46f5-b643-ef987f38c99f"
REVISION_ID = "91109f30-83b8-4c7f-a785-16c86210f134"
CHILD_REVISION_ID = "ef672d59-b26d-4420-bece-954af890c08e"
SOURCE = "<!doctype html><title>café</title>"
SOURCE_BYTES = len(SOURCE.encode("utf-8"))
SOURCE_DIGEST = sha256(SOURCE.encode("utf-8")).hexdigest()


def _revision(
    *,
    revision_id: str = REVISION_ID,
    parent_revision_id: str | None = None,
    sequence: int = 1,
    title: str = "First title",
    runtime_profile: str = "canvas-v1",
    source_path: str | None = None,
) -> CanvasArchiveRevision:
    return CanvasArchiveRevision(
        revision_id=revision_id,
        parent_revision_id=parent_revision_id,
        sequence=sequence,
        title=title,
        runtime_profile=runtime_profile,
        source_path=source_path or canvas_revision_source_path(CANVAS_ID, revision_id),
        content_sha256=SOURCE_DIGEST,
        source_bytes=SOURCE_BYTES,
        actor_kind="assistant",
        origin_message_id="message-7",
        origin_turn_id="turn-3",
        created_at="2026-09-03T12:00:00+00:00",
        deleted_at=None,
    )


def _canvas_archive(
    *, revisions: tuple[CanvasArchiveRevision, ...] | None = None
) -> CanvasArchiveManifest:
    return CanvasArchiveManifest(
        extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
        total_source_bytes=SOURCE_BYTES * len(revisions or (_revision(),)),
        documents=(
            CanvasArchiveDocument(
                canvas_id=CANVAS_ID,
                conversation_id="conversation-4",
                created_at="2026-09-03T11:59:00+00:00",
                deleted_at=None,
                revisions=revisions or (_revision(),),
            ),
        ),
        reopen_hints=(
            CanvasArchiveReopenHint(
                conversation_id="conversation-4", canvas_id=CANVAS_ID
            ),
        ),
    )


def _document(
    index: int, *, conversation_id: str = "conversation-4"
) -> CanvasArchiveDocument:
    canvas_id = str(uuid5(NAMESPACE_URL, f"canvas-{index}"))
    revision_id = str(uuid5(NAMESPACE_URL, f"revision-{index}"))
    return CanvasArchiveDocument(
        canvas_id=canvas_id,
        conversation_id=conversation_id,
        created_at="2026-09-03T11:59:00+00:00",
        deleted_at=None,
        revisions=(
            CanvasArchiveRevision(
                **{
                    **_revision().to_dict(),
                    "revision_id": revision_id,
                    "source_path": canvas_revision_source_path(canvas_id, revision_id),
                }
            ),
        ),
    )


def test_canvas_v3_manifest_round_trips_all_revision_graph_metadata() -> None:
    child = _revision(
        revision_id=CHILD_REVISION_ID,
        parent_revision_id=REVISION_ID,
        sequence=2,
        title="Revisioned title",
    )
    original = ChatbookManifest(
        version=ChatbookVersion.V3,
        name="Canvas archive",
        description="A versioned Canvas graph",
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
        updated_at=datetime(2026, 9, 3, 1, tzinfo=UTC),
        canvas_archive=_canvas_archive(revisions=(_revision(), child)),
    )

    encoded = original.to_dict()
    restored = ChatbookManifest.from_dict(encoded)

    assert encoded["version"] == "3.0"
    assert encoded["canvas"]["extension_version"] == "1.0"
    assert encoded["canvas"]["total_source_bytes"] == SOURCE_BYTES * 2
    assert encoded["canvas"]["documents"][0]["canvas_id"] == CANVAS_ID
    assert encoded["canvas"]["documents"][0]["deleted_at"] is None
    assert encoded["canvas"]["documents"][0]["revisions"][1] == {
        "revision_id": CHILD_REVISION_ID,
        "parent_revision_id": REVISION_ID,
        "sequence": 2,
        "title": "Revisioned title",
        "runtime_profile": "canvas-v1",
        "source_path": f"canvas/{CANVAS_ID}/{CHILD_REVISION_ID}.html.txt",
        "content_sha256": SOURCE_DIGEST,
        "source_bytes": SOURCE_BYTES,
        "actor_kind": "assistant",
        "origin_message_id": "message-7",
        "origin_turn_id": "turn-3",
        "created_at": "2026-09-03T12:00:00+00:00",
        "deleted_at": None,
    }
    assert encoded["canvas"]["reopen_hints"] == [
        {"conversation_id": "conversation-4", "canvas_id": CANVAS_ID}
    ]
    assert restored.canvas_archive == original.canvas_archive


def test_canvas_manifest_records_are_immutable_and_source_free() -> None:
    revision = _revision()

    with pytest.raises(FrozenInstanceError):
        revision.title = "changed"  # type: ignore[misc]

    assert SOURCE not in repr(revision)
    assert SOURCE not in str(revision)
    assert "source" not in revision.to_dict()


@pytest.mark.parametrize(
    "source_path", [SOURCE, [SOURCE], "canvas/not-a-uuid/file.html.txt"]
)
def test_revision_alone_rejects_non_inert_source_path_shape(
    source_path: object,
) -> None:
    values = _revision().to_dict()
    values["source_path"] = source_path

    with pytest.raises(ValueError, match="invalid Canvas archive") as caught:
        CanvasArchiveRevision.from_dict(values)

    assert SOURCE not in str(caught.value)


@pytest.mark.parametrize(
    ("canvas_id", "revision_id"),
    [
        ("../escape", REVISION_ID),
        (CANVAS_ID, "/absolute"),
        (CANVAS_ID.upper(), REVISION_ID),
        (CANVAS_ID, "é"),
    ],
)
def test_canvas_source_path_rejects_unsafe_or_ambiguous_identifiers(
    canvas_id: str, revision_id: str
) -> None:
    with pytest.raises(ValueError, match="invalid Canvas archive"):
        canvas_revision_source_path(canvas_id, revision_id)


@pytest.mark.parametrize(
    "source_path",
    [
        f"canvas/{CANVAS_ID}/{REVISION_ID}.html",
        f"canvas/{CANVAS_ID}/../{REVISION_ID}.html.txt",
        f"/canvas/{CANVAS_ID}/{REVISION_ID}.html.txt",
        f"canvas\\{CANVAS_ID}\\{REVISION_ID}.html.txt",
        f"canvas/{CANVAS_ID}/{CHILD_REVISION_ID}.html.txt",
    ],
)
def test_revision_rejects_noncanonical_or_mismatched_inert_path(
    source_path: str,
) -> None:
    with pytest.raises(ValueError, match="invalid Canvas archive"):
        _canvas_archive(revisions=(_revision(source_path=source_path),))


@pytest.mark.parametrize(
    "changes",
    [
        {"content_sha256": "A" * 64},
        {"source_bytes": -1},
        {"source_bytes": 512 * 1024 + 1},
        {"runtime_profile": "../canvas-v1"},
        {"actor_kind": "system"},
        {"origin_message_id": ""},
        {"origin_turn_id": "x" * 257},
        {"sequence": 0},
        {"created_at": "not-a-timestamp"},
    ],
)
def test_revision_validation_fails_closed_without_echoing_source(
    changes: dict[str, object],
) -> None:
    values = _revision().to_dict()
    values.update(changes)

    with pytest.raises(ValueError, match="invalid Canvas archive") as caught:
        CanvasArchiveRevision.from_dict(values)

    assert SOURCE not in str(caught.value)


def test_validation_exception_chain_does_not_echo_malformed_metadata() -> None:
    raw_marker = "RAW_SOURCE_MARKER"
    values = _revision().to_dict()
    values["created_at"] = f"2026-{raw_marker}"

    with pytest.raises(CanvasArchiveValidationError) as caught:
        CanvasArchiveRevision.from_dict(values)

    assert raw_marker not in "".join(format_exception(caught.value))


def test_unknown_well_formed_runtime_profile_remains_inert_metadata() -> None:
    revision = _revision(runtime_profile="canvas-v9")

    restored = CanvasArchiveRevision.from_dict(revision.to_dict())

    assert restored.runtime_profile == "canvas-v9"
    assert restored.is_runtime_supported is False


def test_document_rejects_foreign_parent_and_noncanonical_sequence() -> None:
    with pytest.raises(ValueError, match="invalid Canvas archive"):
        CanvasArchiveDocument(
            canvas_id=CANVAS_ID,
            conversation_id="conversation-4",
            created_at="2026-09-03T11:59:00+00:00",
            deleted_at=None,
            revisions=({"sequence": 1},),  # type: ignore[arg-type]
        )

    foreign_parent = _revision(
        revision_id=CHILD_REVISION_ID,
        parent_revision_id="26695e80-bec2-41fa-b616-d41813c5bb74",
        sequence=2,
    )
    with pytest.raises(ValueError, match="invalid Canvas archive"):
        _canvas_archive(revisions=(_revision(), foreign_parent))

    skipped_sequence = _revision(
        revision_id=CHILD_REVISION_ID,
        parent_revision_id=REVISION_ID,
        sequence=3,
    )
    with pytest.raises(ValueError, match="invalid Canvas archive"):
        _canvas_archive(revisions=(_revision(), skipped_sequence))


def test_document_accepts_sibling_branches_and_deleted_history() -> None:
    sibling_a = _revision(
        revision_id=CHILD_REVISION_ID,
        parent_revision_id=REVISION_ID,
        sequence=2,
        title="Branch A",
    )
    sibling_b_id = "26695e80-bec2-41fa-b616-d41813c5bb74"
    sibling_b = CanvasArchiveRevision(
        **{
            **_revision().to_dict(),
            "revision_id": sibling_b_id,
            "parent_revision_id": REVISION_ID,
            "sequence": 3,
            "title": "Branch B",
            "source_path": canvas_revision_source_path(CANVAS_ID, sibling_b_id),
            "deleted_at": "2026-09-03T13:00:00+00:00",
        }
    )

    document = CanvasArchiveDocument(
        canvas_id=CANVAS_ID,
        conversation_id="conversation-4",
        created_at="2026-09-03T11:59:00+00:00",
        deleted_at="2026-09-03T14:00:00+00:00",
        revisions=(_revision(), sibling_b, sibling_a),
    )
    archive = CanvasArchiveManifest(
        extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
        total_source_bytes=SOURCE_BYTES * 3,
        documents=(document,),
    )

    assert [
        revision["revision_id"]
        for revision in archive.to_dict()["documents"][0]["revisions"]
    ] == [REVISION_ID, CHILD_REVISION_ID, sibling_b_id]


def test_manifest_rejects_duplicate_identity_and_invalid_reopen_hint() -> None:
    with pytest.raises(ValueError, match="invalid Canvas archive"):
        CanvasArchiveManifest(
            extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
            total_source_bytes=SOURCE_BYTES * 2,
            documents=(
                _canvas_archive().documents[0],
                _canvas_archive().documents[0],
            ),
        )

    with pytest.raises(ValueError, match="invalid Canvas archive"):
        CanvasArchiveManifest(
            extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
            total_source_bytes=SOURCE_BYTES,
            documents=_canvas_archive().documents,
            reopen_hints=(
                CanvasArchiveReopenHint(
                    conversation_id="another-conversation", canvas_id=CANVAS_ID
                ),
            ),
        )


def test_manifest_requires_exact_declared_aggregate_byte_count() -> None:
    with pytest.raises(ValueError, match="invalid Canvas archive"):
        CanvasArchiveManifest(
            extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
            total_source_bytes=SOURCE_BYTES + 1,
            documents=_canvas_archive().documents,
        )


def test_manifest_rejects_cross_document_identity_collision() -> None:
    first = _document(1)
    second_id = str(uuid5(NAMESPACE_URL, "canvas-2"))
    second = CanvasArchiveDocument(
        canvas_id=second_id,
        conversation_id="conversation-5",
        created_at="2026-09-03T11:59:00+00:00",
        deleted_at=None,
        revisions=(
            CanvasArchiveRevision(
                **{
                    **first.revisions[0].to_dict(),
                    "source_path": canvas_revision_source_path(
                        second_id, first.revisions[0].revision_id
                    ),
                }
            ),
        ),
    )

    with pytest.raises(ValueError, match="invalid Canvas archive"):
        CanvasArchiveManifest(
            extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
            total_source_bytes=SOURCE_BYTES * 2,
            documents=(first, second),
        )


def test_manifest_enforces_durable_canvas_count_per_conversation() -> None:
    documents = tuple(_document(index) for index in range(11))

    with pytest.raises(ValueError, match="invalid Canvas archive"):
        CanvasArchiveManifest(
            extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
            total_source_bytes=SOURCE_BYTES * len(documents),
            documents=documents,
        )


def test_manifest_serialization_is_deterministic() -> None:
    first_id = "08df9ec4-97d5-4ef2-b337-1ecdf67af094"
    first = CanvasArchiveDocument(
        canvas_id=first_id,
        conversation_id="conversation-4",
        created_at="2026-09-03T11:58:00+00:00",
        deleted_at="2026-09-03T13:00:00+00:00",
        revisions=(
            CanvasArchiveRevision(
                **{
                    **_revision().to_dict(),
                    "revision_id": "12bfbba4-a8fe-48dc-a92d-642ca42381d7",
                    "source_path": (
                        "canvas/08df9ec4-97d5-4ef2-b337-1ecdf67af094/"
                        "12bfbba4-a8fe-48dc-a92d-642ca42381d7.html.txt"
                    ),
                }
            ),
        ),
    )
    archive = CanvasArchiveManifest(
        extension_version=CANVAS_ARCHIVE_EXTENSION_VERSION,
        total_source_bytes=SOURCE_BYTES * 2,
        documents=(_canvas_archive().documents[0], first),
    )

    assert [item["canvas_id"] for item in archive.to_dict()["documents"]] == [
        first_id,
        CANVAS_ID,
    ]


def test_archive_version_selection_changes_only_for_canvas_records() -> None:
    assert select_chatbook_version(has_canvas_records=False) is ChatbookVersion.V2
    assert select_chatbook_version(has_canvas_records=True) is ChatbookVersion.V3


def test_v1_and_v2_manifest_serialization_remains_exactly_unchanged() -> None:
    for version in (ChatbookVersion.V1, ChatbookVersion.V2):
        manifest = ChatbookManifest(
            version=version,
            name="Compatibility",
            description="No Canvas",
            created_at=datetime(2026, 9, 3, tzinfo=UTC),
            updated_at=datetime(2026, 9, 3, 1, tzinfo=UTC),
        )
        encoded = manifest.to_dict()

        assert "canvas" not in encoded
        assert ChatbookManifest.from_dict(encoded).to_dict() == encoded


def test_v3_requires_canvas_records_and_old_versions_ignore_unknown_canvas_key() -> (
    None
):
    with pytest.raises(ValueError, match="Canvas records"):
        ChatbookManifest(
            version=ChatbookVersion.V3,
            name="Invalid",
            description="No Canvas",
        )

    legacy = ChatbookManifest(
        version=ChatbookVersion.V2,
        name="Legacy",
        description="Compatibility",
    ).to_dict()
    legacy["canvas"] = {"future": "ignored exactly as before"}
    assert ChatbookManifest.from_dict(legacy).canvas_archive is None


def test_raw_manifest_lists_are_bounded_before_child_record_parsing() -> None:
    document = _canvas_archive().documents[0].to_dict()
    document["revisions"] = [None] * 101
    with pytest.raises(CanvasArchiveValidationError) as revisions_error:
        CanvasArchiveDocument.from_dict(document)
    assert revisions_error.value.code == "too_many_revisions"

    extension = _canvas_archive().to_dict()
    extension["documents"] = [None] * 1_001
    with pytest.raises(CanvasArchiveValidationError) as documents_error:
        CanvasArchiveManifest.from_dict(extension)
    assert documents_error.value.code == "too_many_documents"

    extension = _canvas_archive().to_dict()
    extension["reopen_hints"] = [None] * 10_001
    with pytest.raises(CanvasArchiveValidationError) as hints_error:
        CanvasArchiveManifest.from_dict(extension)
    assert hints_error.value.code == "too_many_reopen_hints"
