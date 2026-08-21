"""Immutable, privacy-bounded presentation state for lasting Notes sync."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Literal

from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    validate_notes_sync_digest,
    validate_notes_sync_opaque_id,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    ManagedPlacementEffectKind,
    ReconciliationAttentionKind,
    ReconciliationPlan,
    ReconciliationSkipKind,
)

LastingSyncPhase = Literal[
    "choose",
    "configure",
    "checking",
    "review",
    "activating",
    "receipt",
    "roots",
]

_DIRECTIONS = frozenset({"bidirectional", "folder_to_notes", "notes_to_folder"})
_DESTINATIONS = frozenset({"local", "server"})
_PHASES = frozenset(
    {"choose", "configure", "checking", "review", "activating", "receipt", "roots"}
)
_ROOT_STATUSES = frozenset(
    {
        "active",
        "awaiting_cutover",
        "up_to_date",
        "changes_available",
        "paused",
        "offline",
        "passive",
        "needs_attention",
        "partial",
        "failed",
        "unsupported",
        "starting",
        "stopped",
        "stopping",
    }
)
_ROOT_NEXT_ACTIONS = frozenset(
    {
        "sync_now",
        "review_changes",
        "resume_sync",
        "reconnect_folder",
        "open_active_process",
        "review_settings",
        "resolve_cleanup",
        "wait",
        "none",
        "apply_reviewed",
        "finish_upgrade",
    }
)


@dataclass(frozen=True, slots=True, repr=False)
class LastingSyncSetup:
    """Fields shown on the explicit root-detail surface only."""

    display_name: str = ""
    folder: str = ""
    destination: str = "local"
    note_scope_id: str = "local-notes"
    direction: str = "bidirectional"
    server_available: bool = False
    server_disabled_reason: str = (
        "Unavailable - server sync-folder capability not installed"
    )
    validation_message: str = "Choose a display name, folder, and local destination."
    can_check: bool = False

    def __post_init__(self) -> None:
        if self.destination not in _DESTINATIONS:
            raise ValueError("unknown destination")
        if self.direction not in _DIRECTIONS:
            raise ValueError("unknown direction")
        if any(
            type(value) is not str or len(value) > 4096 or "\n" in value
            for value in (
                self.display_name,
                self.folder,
                self.note_scope_id,
                self.server_disabled_reason,
                self.validation_message,
            )
        ):
            raise ValueError("setup text must be bounded single-line text")
        if type(self.server_available) is not bool or type(self.can_check) is not bool:
            raise TypeError("setup flags must be booleans")

    def __repr__(self) -> str:
        return "LastingSyncSetup(<private root detail>)"


@dataclass(frozen=True, slots=True)
class LastingSyncReviewRow:
    """One path-free, bounded reviewed effect row."""

    item_id: str
    category: str
    effect: str
    choices: tuple[str, ...] = ()
    action_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.choices) is not tuple:
            raise TypeError("choices must be a tuple")
        validate_notes_sync_opaque_id(self.item_id, field_name="item_id")
        if self.action_id is not None:
            validate_notes_sync_opaque_id(self.action_id, field_name="action_id")
        if any(
            type(value) is not str or not value or len(value) > 160 or "\n" in value
            for value in self.choices
        ):
            raise ValueError("choices must be bounded display labels")
        if any(
            type(value) is not str or not value or len(value) > 240 or "\n" in value
            for value in (self.category, self.effect)
        ):
            raise ValueError("review labels must be bounded single-line text")


@dataclass(frozen=True, slots=True, repr=False)
class LastingSyncReview:
    """One paged mutation-free reconciliation projection."""

    root_id: str = ""
    observation_token: str = ""
    safe_count: int = 0
    attention_count: int = 0
    skip_count: int = 0
    managed_count: int = 0
    rows: tuple[LastingSyncReviewRow, ...] = ()
    page: int = 1
    page_count: int = 1
    stale: bool = False
    next_action: str = "Check changes"
    activation: bool = False

    def __post_init__(self) -> None:
        if self.root_id:
            validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        if self.observation_token:
            validate_notes_sync_digest(
                self.observation_token, field_name="observation_token"
            )
        if any(
            type(value) is not int or value < 0
            for value in (
                self.safe_count,
                self.attention_count,
                self.skip_count,
                self.managed_count,
            )
        ):
            raise ValueError("review counts must be non-negative exact integers")
        if type(self.rows) is not tuple or any(
            type(row) is not LastingSyncReviewRow for row in self.rows
        ):
            raise TypeError("rows must be a tuple of review rows")
        if len(self.rows) > 100:
            raise ValueError("review rows must be bounded to one page")
        if type(self.stale) is not bool or type(self.activation) is not bool:
            raise TypeError("review flags must be booleans")
        if (
            type(self.page) is not int
            or type(self.page_count) is not int
            or self.page < 1
            or self.page_count < 1
            or self.page > self.page_count
        ):
            raise ValueError("review page must be within page_count")
        if (
            type(self.next_action) is not str
            or not self.next_action
            or len(self.next_action) > 120
            or "\n" in self.next_action
        ):
            raise ValueError("next_action must be a bounded display label")

    def __repr__(self) -> str:
        return (
            "LastingSyncReview("
            f"safe={self.safe_count}, attention={self.attention_count}, "
            f"skipped={self.skip_count}, managed={self.managed_count}, "
            f"page={self.page}/{self.page_count}, stale={self.stale})"
        )


@dataclass(frozen=True, slots=True)
class LastingSyncRootRow:
    """Path-free root list row; paths belong only in a selected detail form."""

    root_id: str
    display_name: str
    status: str
    next_action: str
    status_label: str
    next_action_label: str
    action_id: str | None = None

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        if self.action_id is not None:
            validate_notes_sync_opaque_id(self.action_id, field_name="action_id")
        if self.status not in _ROOT_STATUSES:
            raise ValueError("unknown root status")
        if self.next_action not in _ROOT_NEXT_ACTIONS:
            raise ValueError("unknown root next action")
        if (
            not self.display_name.strip()
            or len(self.display_name) > 160
            or "/" in self.display_name
            or "\\" in self.display_name
        ):
            raise ValueError("display_name must be a bounded non-path label")
        if any(
            type(value) is not str or not value or len(value) > 160 or "\n" in value
            for value in (self.status_label, self.next_action_label)
        ):
            raise ValueError("root labels must be bounded single-line text")


@dataclass(frozen=True, slots=True)
class LibraryNotesLastingSyncSnapshot:
    """Complete immutable input for the two lasting-sync canvases."""

    phase: LastingSyncPhase
    lasting_available: bool
    setup: LastingSyncSetup
    review: LastingSyncReview
    roots: tuple[LastingSyncRootRow, ...]
    status_line: str
    receipt_line: str = ""
    root_page: int = 1
    root_page_count: int = 1

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError("unknown lasting-sync phase")
        if type(self.lasting_available) is not bool:
            raise TypeError("lasting_available must be a boolean")
        if (
            type(self.setup) is not LastingSyncSetup
            or type(self.review) is not LastingSyncReview
        ):
            raise TypeError("setup and review must be typed projections")
        if type(self.roots) is not tuple or any(
            type(root) is not LastingSyncRootRow for root in self.roots
        ):
            raise TypeError("roots must be a tuple of root rows")
        if len(self.roots) > 20:
            raise ValueError("roots must be bounded to one page")
        if (
            type(self.root_page) is not int
            or type(self.root_page_count) is not int
            or self.root_page < 1
            or self.root_page_count < 1
            or self.root_page > self.root_page_count
        ):
            raise ValueError("root page must be within root_page_count")
        if any(
            type(value) is not str or len(value) > 512 or "\n" in value
            for value in (self.status_line, self.receipt_line)
        ):
            raise ValueError("snapshot status text must be bounded single-line text")


def initial_lasting_sync_snapshot(
    *, lasting_available: bool = False
) -> LibraryNotesLastingSyncSnapshot:
    """Return the inert chooser snapshot used before atomic cutover."""

    if type(lasting_available) is not bool:
        raise TypeError("lasting_available must be a boolean")
    return LibraryNotesLastingSyncSnapshot(
        phase="choose",
        lasting_available=lasting_available,
        setup=LastingSyncSetup(),
        review=LastingSyncReview(),
        roots=(),
        status_line="Choose how files should relate to Library notes.",
    )


def set_setup_value(
    snapshot: LibraryNotesLastingSyncSnapshot,
    field: str,
    value: str,
) -> LibraryNotesLastingSyncSnapshot:
    """Set one known setup value and recompute mutation-free validation."""

    if field not in {
        "display_name",
        "folder",
        "destination",
        "note_scope_id",
        "direction",
    }:
        raise ValueError("unknown setup field")
    if type(value) is not str:
        raise TypeError("setup values must be strings")
    if field == "destination" and value not in _DESTINATIONS:
        raise ValueError("unknown destination")
    if field == "direction" and value not in _DIRECTIONS:
        raise ValueError("unknown direction")

    setup = replace(snapshot.setup, **{field: value})
    missing: list[str] = []
    if not setup.display_name.strip():
        missing.append("display name")
    if not setup.folder.strip():
        missing.append("folder")
    if not setup.note_scope_id.strip():
        missing.append("local Notes destination")
    if setup.destination == "server":
        message = "Choose a local Notes destination; server folder sync is unavailable."
        can_check = False
    elif missing:
        message = f"Choose {', '.join(missing)}."
        can_check = False
    elif not snapshot.lasting_available:
        message = "Lasting folder sync is unavailable until the reviewed cutover."
        can_check = False
    else:
        message = ""
        can_check = True
    return replace(
        snapshot,
        setup=replace(setup, validation_message=message, can_check=can_check),
    )


def build_reconciliation_review(
    plan: ReconciliationPlan,
    *,
    page: int = 1,
) -> LastingSyncReview:
    """Translate a public reconciliation plan into bounded, path-free rows."""

    if type(plan) is not ReconciliationPlan:
        raise TypeError("plan must be a ReconciliationPlan")
    rows: list[LastingSyncReviewRow] = []
    action_effects = {
        NotesSyncActionKind.CREATE_NOTE: "Create a Library note",
        NotesSyncActionKind.UPDATE_NOTE: "Update a Library note",
        NotesSyncActionKind.CREATE_FILE: "Create a folder file",
        NotesSyncActionKind.UPDATE_FILE: "Update a folder file",
        NotesSyncActionKind.MOVE_FILE: "Move a folder file",
        NotesSyncActionKind.NO_CHANGE: "No change",
    }
    for action in plan.safe_actions:
        rows.append(
            LastingSyncReviewRow(
                item_id=action.binding_id or action.action_id,
                category="safe",
                effect=action_effects.get(action.kind, "Review this change"),
                action_id=action.action_id,
            )
        )
    for attention in plan.attention:
        if attention.kind is ReconciliationAttentionKind.CONFLICT:
            effect = "Both file and note changed"
            choices = ("Keep file", "Keep note", "Keep both")
        elif attention.kind is ReconciliationAttentionKind.DELETION_REVIEW:
            effect = "One side was deleted"
            choices = (
                "Restore missing side",
                "Delete/archive counterpart",
                "Disconnect item",
            )
        else:
            effect = "Sync is paused for review"
            choices = ("Review settings", "Disconnect item")
        rows.append(
            LastingSyncReviewRow(
                item_id=attention.binding_id or attention.reason_code,
                category="attention",
                effect=effect,
                choices=choices,
            )
        )
    for group in plan.deletion_groups:
        rows.append(
            LastingSyncReviewRow(
                item_id=f"{len(group.items)}-deletions",
                category="attention",
                effect=f"Review {len(group.items)} deletions as a bounded group",
                choices=(
                    "Restore missing sides",
                    "Delete/archive counterparts",
                    "Disconnect items",
                ),
            )
        )
    skip_effects = {
        ReconciliationSkipKind.OFFLINE: "Folder is offline; reconnect it",
        ReconciliationSkipKind.UNSAFE_ROOT: "Unsafe folder was skipped",
        ReconciliationSkipKind.CAPABILITY: "Required capability is unavailable",
    }
    for skip in plan.skips:
        rows.append(
            LastingSyncReviewRow(
                item_id=skip.reason_code,
                category="skipped",
                effect=skip_effects[skip.kind],
            )
        )
    for effect in plan.managed_placement_effects:
        label = (
            "Preview explicit filesystem move"
            if effect.kind is ManagedPlacementEffectKind.FILE_MOVE
            else "Refresh file representation"
        )
        rows.append(
            LastingSyncReviewRow(
                item_id=effect.binding_id,
                category="managed placement",
                effect=label,
                choices=("Apply once", "Leave unchanged"),
            )
        )

    page_size = plan.page_size
    page_count = max(1, ceil(len(rows) / page_size))
    bounded_page = min(max(1, page), page_count)
    start = (bounded_page - 1) * page_size
    return LastingSyncReview(
        root_id=plan.root_id,
        observation_token=plan.observation_token,
        safe_count=len(plan.safe_actions),
        attention_count=len(plan.attention)
        + sum(len(group.items) for group in plan.deletion_groups),
        skip_count=len(plan.skips),
        managed_count=len(plan.managed_placement_effects),
        rows=tuple(rows[start : start + page_size]),
        page=bounded_page,
        page_count=page_count,
        next_action=(
            "Resolve attention"
            if plan.attention or plan.deletion_groups
            else "Apply reviewed"
        ),
    )


__all__ = [
    "LastingSyncReview",
    "LastingSyncReviewRow",
    "LastingSyncRootRow",
    "LastingSyncSetup",
    "LibraryNotesLastingSyncSnapshot",
    "build_reconciliation_review",
    "initial_lasting_sync_snapshot",
    "set_setup_value",
]
