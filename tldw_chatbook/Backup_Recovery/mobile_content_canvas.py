"""Canvas's declared source, graph, and owner provenance in private C1 plans."""

from tldw_chatbook.Canvas.archive import validate_runtime_profile
from tldw_chatbook.Canvas.limits import (
    CanvasRepositoryLimits,
    sha256_utf8,
    utf8_byte_length,
)
from tldw_chatbook.Canvas.repository import (
    CanvasImportBatch,
    CanvasImportDocument,
    CanvasImportRevision,
    CanvasRepositoryError,
    _validated_opaque_id,
    _validated_optional_timestamp,
    _validated_text,
    _validated_timestamp,
    _validated_uuid,
)

from .mobile_content_graph import _ContentBlocker, _UnprovedReference

_CANVAS_TABLES = ("canvas_documents", "canvas_revisions", "canvas_conversation_hints")


def _canvas(builder):
    limits = CanvasRepositoryLimits()
    documents = builder.by_id["canvas_documents"]
    revisions = builder.by_id["canvas_revisions"]

    def native(transform):
        def checked(row):
            try:
                transform(row)
            except CanvasRepositoryError:
                raise ValueError("native_canvas_invalid") from None

        return checked

    def document(row):
        _validated_uuid(row["id"], "canvas_id")
        _validated_timestamp(row["created_at"], "created_at")
        _validated_optional_timestamp(row["deleted_at"], "deleted_at")
        if row["deleted"] not in (0, 1) or bool(row["deleted"]) != (
            row["deleted_at"] is not None
        ):
            raise ValueError("invalid_canvas_deletion")
        owned = [r for r in revisions.values() if r["canvas_id"] == row["id"]]
        if not owned or len(owned) > limits.max_revisions_per_canvas:
            raise ValueError("invalid_canvas_revision_count")
        if sorted(r["sequence"] for r in owned) != list(range(1, len(owned) + 1)):
            raise ValueError("invalid_canvas_sequence")
        conversation_documents = [
            r
            for r in documents.values()
            if r["conversation_id"] == row["conversation_id"]
        ]
        if len(conversation_documents) > limits.max_canvases_per_conversation:
            raise ValueError("invalid_canvas_count")
        total = sum(
            utf8_byte_length(r["html"])
            for d in conversation_documents
            for r in revisions.values()
            if r["canvas_id"] == d["id"]
        )
        if total > limits.max_source_bytes_per_conversation:
            raise ValueError("invalid_canvas_source_capacity")
        mapped = builder.ref("canvas", row["id"])
        if mapped in {
            builder.identities.mapped("canvas_revision", r["id"])
            for r in revisions.values()
        }:
            raise _UnprovedReference("canvas_cross_identity", row["id"])
        builder.emit(
            "canvas_documents",
            row,
            row
            | {
                "id": mapped,
                "conversation_id": builder.ref("conversation", row["conversation_id"]),
            },
        )

    builder.handle("canvas_documents", native(document))

    def revision(row):
        _validated_uuid(row["id"], "revision_id")
        _validated_timestamp(row["created_at"], "created_at")
        _validated_optional_timestamp(row["deleted_at"], "deleted_at")
        _validated_text(row["title"], "title", limits.max_title_bytes)
        _validated_opaque_id(
            row["origin_turn_id"], "origin_turn_id", limits.max_origin_turn_id_bytes
        )
        _validated_opaque_id(row["origin_message_id"], "origin_message_id", 256)
        validate_runtime_profile(row["runtime_profile"])
        size = utf8_byte_length(row["html"])
        if (
            size > limits.max_source_bytes_per_revision
            or type(row["html_bytes"]) is not int
            or size != row["html_bytes"]
            or sha256_utf8(row["html"]) != row["content_sha256"]
        ):
            raise ValueError("invalid_canvas_source_digest")
        if row["actor_kind"] not in {"assistant", "user_rename", "user_import"}:
            raise ValueError("invalid_canvas_actor")
        document = documents.get(row["canvas_id"])
        if document is None:
            raise _UnprovedReference("canvas", row["canvas_id"])
        message = builder.by_id["messages"].get(row["origin_message_id"])
        if message is None or message["conversation_id"] != document["conversation_id"]:
            raise _UnprovedReference("canvas_origin_message", row["origin_message_id"])
        owned = [r for r in revisions.values() if r["canvas_id"] == row["canvas_id"]]
        if type(row["sequence"]) is not int or not 1 <= row["sequence"] <= len(owned):
            raise ValueError("invalid_canvas_sequence")
        if row["sequence"] == 1:
            if row["parent_revision_id"] is not None:
                raise ValueError("invalid_canvas_root")
        else:
            parent = revisions.get(row["parent_revision_id"])
            if (
                parent is None
                or parent["canvas_id"] != row["canvas_id"]
                or parent["sequence"] >= row["sequence"]
            ):
                raise _UnprovedReference("canvas_revision", row["parent_revision_id"])
        builder.emit(
            "canvas_revisions",
            row,
            row
            | {
                "id": builder.ref("canvas_revision", row["id"]),
                "canvas_id": builder.ref("canvas", row["canvas_id"]),
                "parent_revision_id": builder.ref(
                    "canvas_revision", row["parent_revision_id"], optional=True
                ),
                "origin_message_id": builder.ref("message", row["origin_message_id"]),
            },
        )

    builder.handle("canvas_revisions", native(revision))

    def hint(row):
        _validated_timestamp(row["updated_at"], "updated_at")
        document = documents.get(row["last_canvas_id"])
        if (
            document is None
            or document["deleted"]
            or document["conversation_id"] != row["conversation_id"]
        ):
            raise _UnprovedReference("canvas", row["last_canvas_id"])
        builder.emit(
            "canvas_conversation_hints",
            row,
            row
            | {
                "conversation_id": builder.ref("conversation", row["conversation_id"]),
                "last_canvas_id": builder.ref("canvas", row["last_canvas_id"]),
            },
        )

    builder.handle("canvas_conversation_hints", native(hint))

    # The actual native import dataclasses carry the bounded, proved graph.
    # Full row projections separately retain hint timestamps/deletion fields;
    # these batches are consumer inputs, never owning write authorization.
    projected = {
        table: {r.original: dict(r.values) for r in builder.rows if r.table == table}
        for table in _CANVAS_TABLES
    }
    batches = []
    for conversation in dict.fromkeys(d["conversation_id"] for d in documents.values()):
        originals = [
            d for d in documents.values() if d["conversation_id"] == conversation
        ]
        original_revisions = [
            r
            for r in revisions.values()
            if any(d["id"] == r["canvas_id"] for d in originals)
        ]
        hint_source = next(
            (
                r
                for r in builder.source["canvas_conversation_hints"]
                if r["conversation_id"] == conversation
            ),
            None,
        )
        if (
            any((d["id"],) not in projected["canvas_documents"] for d in originals)
            or any(
                (r["id"],) not in projected["canvas_revisions"]
                for r in original_revisions
            )
            or (
                hint_source is not None
                and (conversation,) not in projected["canvas_conversation_hints"]
            )
        ):
            builder.blockers.append(
                _ContentBlocker(
                    "canvas_documents",
                    (originals[0]["id"],),
                    "canvas_graph_incomplete",
                    "conversation",
                    conversation,
                )
            )
            continue
        docs = tuple(
            CanvasImportDocument(
                r["id"], r["conversation_id"], r["created_at"], r["deleted_at"]
            )
            for d in originals
            for r in (projected["canvas_documents"][(d["id"],)],)
        )
        revs = tuple(
            CanvasImportRevision(
                r["id"],
                r["canvas_id"],
                r["parent_revision_id"],
                r["sequence"],
                r["title"],
                r["runtime_profile"],
                r["html"],
                r["content_sha256"],
                r["html_bytes"],
                r["actor_kind"],
                r["origin_message_id"],
                r["origin_turn_id"],
                r["created_at"],
                r["deleted_at"],
            )
            for source in original_revisions
            for r in (projected["canvas_revisions"][(source["id"],)],)
        )
        reopen = (
            projected["canvas_conversation_hints"][(conversation,)]["last_canvas_id"]
            if hint_source
            else None
        )
        batches.append(
            CanvasImportBatch(
                builder.identities.mapped("conversation", conversation),
                docs,
                revs,
                reopen,
            )
        )
    return tuple(batches)
