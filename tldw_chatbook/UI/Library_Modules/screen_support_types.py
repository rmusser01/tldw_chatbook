"""Library screen module-level support dataclasses and type aliases.

Moved verbatim out of ``tldw_chatbook/UI/Screens/library_screen.py`` by PR 0a
of the Library screen decomposition
(``.superpowers/sdd/2026-09-01-library-decomposition-foundation``; see
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``).
``library_screen.py`` re-exports every name here so its import surface is
unchanged; later decomposition tasks import directly from this module.
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Literal, TypeAlias

from textual.widget import Widget

from ...Library.library_ingest_jobs import ActiveIngestConsentScope
from ...Library.library_notes_state import LibraryNotesFocusIdentity


LibraryReaderDestination = Literal[
    "media",
    "collections",
    "conversations",
    "notes",
    "notes_files",
    "prompts",
    "skills",
]


@dataclasses.dataclass(frozen=True, slots=True)
class _LibraryIngestStartConsent:
    """Immutable identity of the submission a second Start may authorize."""

    fingerprint: str
    admission_scope: ActiveIngestConsentScope
    tooling_affected_count: int
    is_folder: bool
    request_fingerprint: str = ""
    consent_context_fingerprint: str = ""
    authoritative_refusal: bool = False
    candidate_changed: bool = False

    @property
    def active_job_ids(self) -> tuple[str, ...]:
        return self.admission_scope.active_job_ids

    @property
    def active_source_count(self) -> int:
        return self.admission_scope.active_source_count

    @property
    def owed(self) -> bool:
        return bool(
            self.active_job_ids or self.tooling_affected_count or self.candidate_changed
        )

    @property
    def allows_active_duplicate(self) -> bool:
        return bool(
            self.active_job_ids and self.admission_scope.active_job_ids_complete
        )


class LibraryEntryReconcileResult(Enum):
    """Outcome of projecting a source snapshot into the mounted Library."""

    APPLIED = "applied"
    ALREADY_CURRENT = "already-current"
    SUPERSEDED = "superseded"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class LibraryEntryFocusIdentity:
    """Portable semantic focus and scroll identity for an entry canvas."""

    widget_id: str = ""
    source_id: str = ""
    scroll_offset: tuple[int, int] | None = None


@dataclasses.dataclass(frozen=True)
class _LibraryEntryFocusCapture:
    """Focus intent carried across superseding strict entry syncs."""

    identity: LibraryEntryFocusIdentity
    outgoing_focus: Widget
    route_key: tuple[object, ...]
    notes_identity: LibraryNotesFocusIdentity | None = None


_LibraryMediaFinalFocusPolicy: TypeAlias = Literal["row", "control"]
_LibraryMediaSettlementOutcome: TypeAlias = Literal[
    "exact-settled",
    "exact-scroll-focus-fallback",
    "clamped-after-revision",
    "clamped-after-settlement-failure",
    "layout-settlement-failed",
]


@dataclasses.dataclass(frozen=True)
class _LibraryMediaReturnReceipt:
    """Transient normal-Media coordinates captured before leaving its list."""

    stable_id: str
    scroll_offset: tuple[int, int] | None
    content_signature: tuple[object, ...]
    layout_signature: tuple[object, ...]
    final_focus_policy: _LibraryMediaFinalFocusPolicy
    final_focus_identity: str | None


@dataclasses.dataclass(frozen=True)
class _LibraryMediaReturnSettlement:
    """Immutable authority for one current-owner Media return attempt."""

    request_id: int
    receipt: _LibraryMediaReturnReceipt
    final_focus_policy: _LibraryMediaFinalFocusPolicy
    final_focus_identity: str | None
    focus_intent_generation: int
    compose_generation: int
    media_lifecycle_generation: int
    presentation_epoch: int
    content_signature: tuple[object, ...]
    layout_signature: tuple[object, ...]
    route_identity: tuple[object, ...]
    media_view_identity: str
    shell_identity: int
    items_host_identity: int
    owner_identity: int
    exclusive_geometry_floor: int
    focus_anchor: Widget | None


@dataclasses.dataclass(frozen=True)
class _LibraryMediaSuccessfulFocusOwnership:
    """ABA-fenced focus ownership retained after one exact settlement."""

    request: _LibraryMediaReturnSettlement
    outer_generation: int
    selected_media_id: str | None
    target: Widget


@dataclasses.dataclass(frozen=True)
class _LibraryEmergencyReturnEligibility:
    """One immutable truth source for narrow-canvas recovery chrome."""

    visible: bool
    enabled: bool
    guarded: bool


@dataclasses.dataclass(frozen=True)
class _LibraryEmergencyRestoreReceipt:
    """Portable focus/scroll state owned by one interaction generation."""

    owner_id: str
    scroll_owner_id: str
    focus: LibraryEntryFocusIdentity
    route_key: tuple[object, ...]
    generation: int


@dataclasses.dataclass(frozen=True)
class _LibraryNotesRecomposeCapture:
    """Portable identity needed to rehydrate a replaced Notes canvas."""

    focus: LibraryNotesFocusIdentity
    recompose_generation: int
    scroll_generation: int
    focus_generation: int
    session_generation: int | None
    draft_revision: int | None
    preview: bool
    context: bool
    confirming_delete: bool


@dataclasses.dataclass(frozen=True)
class _LibraryNotesRestoreGuard:
    """Generations that keep deferred focus/scroll restoration current."""

    recompose_generation: int | None = None
    scroll_generation: int | None = None
    focus_generation: int | None = None


@dataclasses.dataclass(frozen=True)
class _LibraryNotesDeletedFolderReceipt:
    """One-session recovery authority for the last removed folder subtree."""

    folder_id: str
    name: str
    expected_version: int


class _ParakeetV2NoPendingReportError(RuntimeError):
    """Raised when confirmation has no retained preflight report."""
