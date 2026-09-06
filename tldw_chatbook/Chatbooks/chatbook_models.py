# chatbook_models.py
# Description: Data models for chatbook/knowledge pack structures
#
"""
Chatbook Models
---------------

Defines the data structures for chatbooks including manifest,
content organization, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..Canvas.archive import (
    CANVAS_ARCHIVE_EXTENSION_VERSION,
    CANVAS_ARCHIVE_SUPPORTED_RUNTIME_PROFILE,
    MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES,
    MAX_CANVAS_ARCHIVE_DOCUMENTS,
    MAX_CANVAS_ARCHIVE_MESSAGE_ID_BYTES,
    MAX_CANVAS_ARCHIVE_REOPEN_HINTS,
    MAX_CANVAS_ARCHIVE_REVISIONS,
    MAX_CANVAS_ARCHIVE_SOURCE_BYTES,
    MAX_CANVAS_ORIGIN_TURN_ID_BYTES,
    MAX_CANVASES_PER_CONVERSATION,
    MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION,
    MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
    MAX_REVISIONS_PER_CANVAS,
    CanvasArchiveValidationError,
    require_exact_fields,
    validate_actor_kind,
    validate_archive_uuid,
    validate_bounded_identifier,
    validate_digest,
    validate_inert_source_path_shape,
    validate_non_negative_int,
    validate_positive_int,
    validate_revision_source_path,
    validate_runtime_profile,
    validate_timestamp,
    validate_title,
)


class ChatbookVersion(Enum):
    """Chatbook format versions."""

    V1 = "1.0"
    V2 = "2.0"
    V3 = "3.0"


def select_chatbook_version(*, has_canvas_records: bool) -> ChatbookVersion:
    """Select 3.0 only when the export actually contains Canvas records."""

    if type(has_canvas_records) is not bool:
        raise TypeError("has_canvas_records must be a bool")
    return ChatbookVersion.V3 if has_canvas_records else ChatbookVersion.V2


@dataclass(frozen=True, slots=True)
class CanvasArchiveRevision:
    """Source-free manifest metadata for one immutable Canvas revision."""

    revision_id: str
    parent_revision_id: str | None
    sequence: int
    title: str
    runtime_profile: str
    source_path: str
    content_sha256: str
    source_bytes: int
    actor_kind: str
    origin_message_id: str
    origin_turn_id: str
    created_at: str
    deleted_at: str | None = None

    def __post_init__(self) -> None:
        validate_archive_uuid(self.revision_id, field_name="revision_id")
        if self.parent_revision_id is not None:
            validate_archive_uuid(
                self.parent_revision_id, field_name="parent_revision_id"
            )
        validate_positive_int(
            self.sequence, field_name="sequence", maximum=MAX_REVISIONS_PER_CANVAS
        )
        validate_title(self.title)
        validate_runtime_profile(self.runtime_profile)
        validate_inert_source_path_shape(self.source_path, revision_id=self.revision_id)
        validate_digest(self.content_sha256)
        validate_non_negative_int(
            self.source_bytes,
            field_name="source_bytes",
            maximum=MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
        )
        validate_actor_kind(self.actor_kind)
        validate_bounded_identifier(
            self.origin_message_id,
            field_name="origin_message_id",
            byte_limit=MAX_CANVAS_ARCHIVE_MESSAGE_ID_BYTES,
        )
        validate_bounded_identifier(
            self.origin_turn_id,
            field_name="origin_turn_id",
            byte_limit=MAX_CANVAS_ORIGIN_TURN_ID_BYTES,
        )
        validate_timestamp(self.created_at, field_name="created_at")
        validate_timestamp(self.deleted_at, field_name="deleted_at", optional=True)

    @property
    def is_runtime_supported(self) -> bool:
        """Whether this inert record names the V1 executable profile."""

        return self.runtime_profile == CANVAS_ARCHIVE_SUPPORTED_RUNTIME_PROFILE

    def validate_source_path(self, canvas_id: str) -> None:
        """Validate the path once the containing Canvas identity is known."""

        validate_revision_source_path(
            self.source_path,
            canvas_id=canvas_id,
            revision_id=self.revision_id,
        )

    def to_dict(self) -> dict:
        """Return the canonical source-free JSON record."""

        return {
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "sequence": self.sequence,
            "title": self.title,
            "runtime_profile": self.runtime_profile,
            "source_path": self.source_path,
            "content_sha256": self.content_sha256,
            "source_bytes": self.source_bytes,
            "actor_kind": self.actor_kind,
            "origin_message_id": self.origin_message_id,
            "origin_turn_id": self.origin_turn_id,
            "created_at": self.created_at,
            "deleted_at": self.deleted_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanvasArchiveRevision":
        """Parse one extension-1.0 revision without reading its source entry."""

        record = require_exact_fields(
            data,
            required=frozenset(
                {
                    "revision_id",
                    "parent_revision_id",
                    "sequence",
                    "title",
                    "runtime_profile",
                    "source_path",
                    "content_sha256",
                    "source_bytes",
                    "actor_kind",
                    "origin_message_id",
                    "origin_turn_id",
                    "created_at",
                    "deleted_at",
                }
            ),
        )
        return cls(**record)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CanvasArchiveDocument:
    """One stable Canvas identity and its complete revision metadata graph."""

    canvas_id: str
    conversation_id: str
    created_at: str
    deleted_at: str | None
    revisions: tuple[CanvasArchiveRevision, ...]

    def __post_init__(self) -> None:
        validate_archive_uuid(self.canvas_id, field_name="canvas_id")
        validate_bounded_identifier(
            self.conversation_id,
            field_name="conversation_id",
            byte_limit=MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES,
        )
        validate_timestamp(self.created_at, field_name="created_at")
        validate_timestamp(self.deleted_at, field_name="deleted_at", optional=True)
        if type(self.revisions) is not tuple or not self.revisions:
            raise CanvasArchiveValidationError("canvas_without_revisions")
        if len(self.revisions) > MAX_REVISIONS_PER_CANVAS:
            raise CanvasArchiveValidationError("too_many_revisions")
        if not all(
            isinstance(revision, CanvasArchiveRevision) for revision in self.revisions
        ):
            raise CanvasArchiveValidationError("invalid_revision")
        ordered = sorted(self.revisions, key=lambda revision: revision.sequence)
        if [revision.sequence for revision in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise CanvasArchiveValidationError("invalid_revision_sequence")
        revisions_by_id: dict[str, CanvasArchiveRevision] = {}
        for revision in ordered:
            if revision.revision_id in revisions_by_id:
                raise CanvasArchiveValidationError("duplicate_revision_id")
            revision.validate_source_path(self.canvas_id)
            if revision.sequence == 1:
                if revision.parent_revision_id is not None:
                    raise CanvasArchiveValidationError("invalid_root_parent")
            else:
                parent = revisions_by_id.get(revision.parent_revision_id or "")
                if parent is None or parent.sequence >= revision.sequence:
                    raise CanvasArchiveValidationError("invalid_parent")
            revisions_by_id[revision.revision_id] = revision

    def to_dict(self) -> dict:
        """Return this document with revisions in canonical sequence order."""

        return {
            "canvas_id": self.canvas_id,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "deleted_at": self.deleted_at,
            "revisions": [
                revision.to_dict()
                for revision in sorted(
                    self.revisions, key=lambda revision: revision.sequence
                )
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanvasArchiveDocument":
        """Parse one stable Canvas identity and revision graph."""

        record = require_exact_fields(
            data,
            required=frozenset(
                {
                    "canvas_id",
                    "conversation_id",
                    "created_at",
                    "deleted_at",
                    "revisions",
                }
            ),
        )
        revisions = record["revisions"]
        if not isinstance(revisions, list):
            raise CanvasArchiveValidationError("invalid_revisions")
        if len(revisions) > MAX_REVISIONS_PER_CANVAS:
            raise CanvasArchiveValidationError("too_many_revisions")
        return cls(
            canvas_id=record["canvas_id"],  # type: ignore[arg-type]
            conversation_id=record["conversation_id"],  # type: ignore[arg-type]
            created_at=record["created_at"],  # type: ignore[arg-type]
            deleted_at=record["deleted_at"],  # type: ignore[arg-type]
            revisions=tuple(
                CanvasArchiveRevision.from_dict(item) for item in revisions
            ),
        )


@dataclass(frozen=True, slots=True)
class CanvasArchiveReopenHint:
    """Conversation-local last-used Canvas identity; never synchronized."""

    conversation_id: str
    canvas_id: str

    def __post_init__(self) -> None:
        validate_bounded_identifier(
            self.conversation_id,
            field_name="conversation_id",
            byte_limit=MAX_CANVAS_ARCHIVE_CONVERSATION_ID_BYTES,
        )
        validate_archive_uuid(self.canvas_id, field_name="canvas_id")

    def to_dict(self) -> dict:
        """Return the canonical reopen-hint record."""

        return {"conversation_id": self.conversation_id, "canvas_id": self.canvas_id}

    @classmethod
    def from_dict(cls, data: dict) -> "CanvasArchiveReopenHint":
        """Parse a reopen hint without treating it as authority."""

        record = require_exact_fields(
            data, required=frozenset({"conversation_id", "canvas_id"})
        )
        return cls(
            conversation_id=record["conversation_id"],  # type: ignore[arg-type]
            canvas_id=record["canvas_id"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CanvasArchiveManifest:
    """Bounded Canvas extension metadata embedded in a Chatbook 3.0 manifest."""

    extension_version: str
    total_source_bytes: int
    documents: tuple[CanvasArchiveDocument, ...]
    reopen_hints: tuple[CanvasArchiveReopenHint, ...] = ()

    def __post_init__(self) -> None:
        if self.extension_version != CANVAS_ARCHIVE_EXTENSION_VERSION:
            raise CanvasArchiveValidationError("unsupported_extension_version")
        validate_non_negative_int(
            self.total_source_bytes,
            field_name="total_source_bytes",
            maximum=MAX_CANVAS_ARCHIVE_SOURCE_BYTES,
        )
        if type(self.documents) is not tuple or not self.documents:
            raise CanvasArchiveValidationError("missing_documents")
        if len(self.documents) > MAX_CANVAS_ARCHIVE_DOCUMENTS:
            raise CanvasArchiveValidationError("too_many_documents")
        if type(self.reopen_hints) is not tuple:
            raise CanvasArchiveValidationError("invalid_reopen_hints")
        if len(self.reopen_hints) > MAX_CANVAS_ARCHIVE_REOPEN_HINTS:
            raise CanvasArchiveValidationError("too_many_reopen_hints")

        documents_by_id: dict[str, CanvasArchiveDocument] = {}
        conversation_documents: set[tuple[str, str]] = set()
        conversation_canvas_counts: dict[str, int] = {}
        conversation_source_bytes: dict[str, int] = {}
        all_stable_ids: set[str] = set()
        revision_count = 0
        computed_source_bytes = 0
        for document in self.documents:
            if not isinstance(document, CanvasArchiveDocument):
                raise CanvasArchiveValidationError("invalid_document")
            if document.canvas_id in all_stable_ids:
                raise CanvasArchiveValidationError("duplicate_stable_id")
            all_stable_ids.add(document.canvas_id)
            documents_by_id[document.canvas_id] = document
            conversation_documents.add((document.conversation_id, document.canvas_id))
            conversation_canvas_counts[document.conversation_id] = (
                conversation_canvas_counts.get(document.conversation_id, 0) + 1
            )
            if (
                conversation_canvas_counts[document.conversation_id]
                > MAX_CANVASES_PER_CONVERSATION
            ):
                raise CanvasArchiveValidationError("conversation_canvas_limit")
            revision_count += len(document.revisions)
            document_source_bytes = sum(
                revision.source_bytes for revision in document.revisions
            )
            computed_source_bytes += document_source_bytes
            conversation_source_bytes[document.conversation_id] = (
                conversation_source_bytes.get(document.conversation_id, 0)
                + document_source_bytes
            )
            if (
                conversation_source_bytes[document.conversation_id]
                > MAX_DURABLE_SOURCE_BYTES_PER_CONVERSATION
            ):
                raise CanvasArchiveValidationError("conversation_source_byte_limit")
            for revision in document.revisions:
                if revision.revision_id in all_stable_ids:
                    raise CanvasArchiveValidationError("duplicate_stable_id")
                all_stable_ids.add(revision.revision_id)
        if revision_count > MAX_CANVAS_ARCHIVE_REVISIONS:
            raise CanvasArchiveValidationError("too_many_revisions")
        if computed_source_bytes != self.total_source_bytes:
            raise CanvasArchiveValidationError("source_byte_count_mismatch")

        hinted_conversations: set[str] = set()
        for hint in self.reopen_hints:
            if not isinstance(hint, CanvasArchiveReopenHint):
                raise CanvasArchiveValidationError("invalid_reopen_hint")
            if hint.conversation_id in hinted_conversations:
                raise CanvasArchiveValidationError("duplicate_reopen_hint")
            if (hint.conversation_id, hint.canvas_id) not in conversation_documents:
                raise CanvasArchiveValidationError("invalid_reopen_hint")
            if documents_by_id[hint.canvas_id].deleted_at is not None:
                raise CanvasArchiveValidationError("deleted_reopen_hint")
            hinted_conversations.add(hint.conversation_id)

    def to_dict(self) -> dict:
        """Return canonical deterministic Canvas extension JSON."""

        return {
            "extension_version": self.extension_version,
            "total_source_bytes": self.total_source_bytes,
            "documents": [
                document.to_dict()
                for document in sorted(
                    self.documents,
                    key=lambda document: (document.conversation_id, document.canvas_id),
                )
            ],
            "reopen_hints": [
                hint.to_dict()
                for hint in sorted(
                    self.reopen_hints,
                    key=lambda hint: (hint.conversation_id, hint.canvas_id),
                )
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanvasArchiveManifest":
        """Parse the complete known Canvas extension without opening entries."""

        record = require_exact_fields(
            data,
            required=frozenset(
                {
                    "extension_version",
                    "total_source_bytes",
                    "documents",
                    "reopen_hints",
                }
            ),
        )
        documents = record["documents"]
        reopen_hints = record["reopen_hints"]
        if not isinstance(documents, list) or not isinstance(reopen_hints, list):
            raise CanvasArchiveValidationError("invalid_extension_lists")
        if len(documents) > MAX_CANVAS_ARCHIVE_DOCUMENTS:
            raise CanvasArchiveValidationError("too_many_documents")
        if len(reopen_hints) > MAX_CANVAS_ARCHIVE_REOPEN_HINTS:
            raise CanvasArchiveValidationError("too_many_reopen_hints")
        return cls(
            extension_version=record["extension_version"],  # type: ignore[arg-type]
            total_source_bytes=record["total_source_bytes"],  # type: ignore[arg-type]
            documents=tuple(
                CanvasArchiveDocument.from_dict(item) for item in documents
            ),
            reopen_hints=tuple(
                CanvasArchiveReopenHint.from_dict(item) for item in reopen_hints
            ),
        )


class ContentType(Enum):
    """Types of content that can be included in a chatbook."""

    CONVERSATION = "conversation"
    NOTE = "note"
    CHARACTER = "character"
    MEDIA = "media"
    EMBEDDING = "embedding"
    PROMPT = "prompt"
    EVALUATION = "evaluation"
    # A user's kept briefing (ChaChaNotes `kept_briefings`, task-1780). Kept
    # scripts (`kept_scripts`) are NOT independently selectable -- they ride
    # with their parent kept briefing and are nested inside its exported
    # payload, mirroring how a conversation's messages are nested inside the
    # conversation's own JSON rather than being their own content type.
    KEPT_BRIEFING = "kept_briefing"


@dataclass
class ContentItem:
    """Individual content item in a chatbook."""

    id: str
    type: ContentType
    title: str
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str | None = None  # Relative path within chatbook

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentItem":
        """Create ContentItem from dictionary."""
        return cls(
            id=data["id"],
            type=ContentType(data["type"]),
            title=data["title"],
            description=data.get("description"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            file_path=data.get("file_path"),
        )


@dataclass
class Relationship:
    """Relationship between content items."""

    source_id: str
    target_id: str
    relationship_type: str  # e.g., "references", "parent_of", "requires"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "metadata": self.metadata,
        }


@dataclass
class ChatbookManifest:
    """Manifest file containing chatbook metadata and contents listing."""

    version: ChatbookVersion
    name: str
    description: str
    author: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Content summary
    content_items: list[ContentItem] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

    # Configuration
    include_media: bool = False
    include_embeddings: bool = False
    media_quality: str = "thumbnail"  # thumbnail, compressed, original

    # Statistics
    total_conversations: int = 0
    total_notes: int = 0
    total_characters: int = 0
    total_media_items: int = 0
    total_prompts: int = 0
    total_kept_briefings: int = 0
    total_size_bytes: int = 0

    # Metadata
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    language: str = "en"
    license: str | None = None
    canvas_archive: CanvasArchiveManifest | None = None

    def __post_init__(self) -> None:
        if self.version is ChatbookVersion.V3:
            if self.canvas_archive is None:
                raise ValueError("Chatbook 3.0 requires Canvas records")
        elif self.canvas_archive is not None:
            raise ValueError("Canvas records require Chatbook 3.0")

    def to_dict(self) -> dict:
        """Convert manifest to dictionary for JSON serialization."""
        result = {
            "version": self.version.value,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content_items": [item.to_dict() for item in self.content_items],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "include_media": self.include_media,
            "include_embeddings": self.include_embeddings,
            "media_quality": self.media_quality,
            "statistics": {
                "total_conversations": self.total_conversations,
                "total_notes": self.total_notes,
                "total_characters": self.total_characters,
                "total_media_items": self.total_media_items,
                "total_prompts": self.total_prompts,
                "total_kept_briefings": self.total_kept_briefings,
                "total_size_bytes": self.total_size_bytes,
            },
            "tags": self.tags,
            "categories": self.categories,
            "language": self.language,
            "license": self.license,
        }
        if self.version is ChatbookVersion.V3:
            assert self.canvas_archive is not None
            result["canvas"] = self.canvas_archive.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ChatbookManifest":
        """Create ChatbookManifest from dictionary."""
        version = ChatbookVersion(data["version"])
        canvas_archive = None
        if version is ChatbookVersion.V3:
            if "canvas" not in data:
                raise ValueError("Chatbook 3.0 requires Canvas records")
            canvas_archive = CanvasArchiveManifest.from_dict(data["canvas"])
        manifest = cls(
            version=version,
            name=data["name"],
            description=data["description"],
            author=data.get("author"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            canvas_archive=canvas_archive,
        )

        # Load content items
        manifest.content_items = [
            ContentItem.from_dict(item) for item in data.get("content_items", [])
        ]

        # Load relationships
        manifest.relationships = [
            Relationship(**rel) for rel in data.get("relationships", [])
        ]

        # Load configuration
        manifest.include_media = data.get("include_media", False)
        manifest.include_embeddings = data.get("include_embeddings", False)
        manifest.media_quality = data.get("media_quality", "thumbnail")

        # Load statistics
        stats = data.get("statistics", {})
        manifest.total_conversations = stats.get("total_conversations", 0)
        manifest.total_notes = stats.get("total_notes", 0)
        manifest.total_characters = stats.get("total_characters", 0)
        manifest.total_media_items = stats.get("total_media_items", 0)
        manifest.total_prompts = stats.get("total_prompts", 0)
        # Backward compat: bundles created before this content type existed
        # (task-1870) have no "total_kept_briefings" key at all -- default to
        # 0 rather than raising, same treatment every other statistic here
        # gets.
        manifest.total_kept_briefings = stats.get("total_kept_briefings", 0)
        manifest.total_size_bytes = stats.get("total_size_bytes", 0)

        # Load metadata
        manifest.tags = data.get("tags", [])
        manifest.categories = data.get("categories", [])
        manifest.language = data.get("language", "en")
        manifest.license = data.get("license")

        return manifest


@dataclass
class ChatbookContent:
    """Container for all chatbook content."""

    conversations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    characters: list[dict[str, Any]] = field(default_factory=list)
    media_items: list[dict[str, Any]] = field(default_factory=list)
    embeddings: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    kept_briefings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Chatbook:
    """Complete chatbook structure."""

    manifest: ChatbookManifest
    content: ChatbookContent
    base_path: Path | None = None

    def get_content_by_type(self, content_type: ContentType) -> list[ContentItem]:
        """Get all content items of a specific type."""
        return [
            item for item in self.manifest.content_items if item.type == content_type
        ]

    def get_content_by_id(self, content_id: str) -> ContentItem | None:
        """Get a specific content item by ID."""
        for item in self.manifest.content_items:
            if item.id == content_id:
                return item
        return None

    def get_related_content(self, content_id: str) -> list[ContentItem]:
        """Get all content items related to a specific item."""
        related_ids = set()

        # Find relationships where this item is source or target
        for rel in self.manifest.relationships:
            if rel.source_id == content_id:
                related_ids.add(rel.target_id)
            elif rel.target_id == content_id:
                related_ids.add(rel.source_id)

        # Return the actual content items
        return [item for item in self.manifest.content_items if item.id in related_ids]
