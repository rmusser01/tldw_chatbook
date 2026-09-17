"""Detached document values shared by workflow services."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Revision:
    """Immutable saved definition with portable identity and lineage.

    Attributes:
        workflow_id: Stable identity of the workflow.
        revision_id: Identity of this saved definition.
        parent_revision_ids: Revisions from which this definition descends.
        raw_json: Complete definition, including preserved opaque fields.
    """

    workflow_id: str
    revision_id: str
    parent_revision_ids: tuple[str, ...]
    raw_json: str


@dataclass(frozen=True)
class Draft:
    """Recoverable editor buffer attached to an exact saved base.

    Attributes:
        workflow_id: Workflow owning the buffer.
        base_revision_id: Saved revision on which editing began.
        generation: Monotonic edit generation for conflict detection.
        raw_text: Exact editor text, including incomplete or invalid JSON.
        last_valid_json: Last structurally valid projection for form rendering.
        error: Validation failure, or None when the buffer is valid.
    """

    workflow_id: str
    base_revision_id: str
    generation: int
    raw_text: str
    last_valid_json: str
    error: str | None


@dataclass(frozen=True)
class FieldEdit:
    """Exact JSON fragment intent, anchored to its original valid draft."""

    source: Draft
    pointer: str
    text: str


class DraftConflict(ValueError):
    """A draft write or save used an outdated generation."""


class DraftWriteFailed(RuntimeError):
    """The pending buffer remains owned in memory after a durable write failed."""


class InvalidDraft(ValueError):
    """A document cannot be saved as a structurally valid revision."""


class RevisionConflict(ValueError):
    """A revision does not match the requested workflow or current head."""


@dataclass(frozen=True)
class Issue:
    """Addressable authoring validation feedback.

    Attributes:
        pointer: JSON Pointer identifying the affected document location.
        code: Stable machine-readable issue category.
        message: Explanation shown to the author.
    """

    pointer: str
    code: str
    message: str
