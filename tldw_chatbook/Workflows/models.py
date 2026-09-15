"""Detached document values shared by workflow services."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Revision:
    workflow_id: str
    revision_id: str
    parent_revision_ids: tuple[str, ...]
    raw_json: str


@dataclass(frozen=True)
class Draft:
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
    pointer: str
    code: str
    message: str
