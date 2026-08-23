"""Immutable, privacy-bounded presentation state for lasting Notes sync."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from math import ceil
from typing import Literal

from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictComparison,
    ConflictHistoryRow,
    ConflictReceipt,
    ConflictSelection,
    NotesSyncConflictChoice,
    eligible_conflict_reason,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    normalize_notes_sync_relative_path,
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
    "history",
    "roots",
]

_DIRECTIONS = frozenset({"bidirectional", "folder_to_notes", "notes_to_folder"})
_DESTINATIONS = frozenset({"local", "server"})
_PHASES = frozenset(
    {
        "choose",
        "configure",
        "checking",
        "review",
        "activating",
        "receipt",
        "history",
        "roots",
    }
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
        "review_migration",
        "resolve_cleanup",
        "wait",
        "none",
        "apply_reviewed",
        "finish_upgrade",
        "close_other_process_and_restart",
    }
)
LASTING_SYNC_HISTORY_PAGE_SIZE = 100
_SQLITE_INTEGER_MAX = 2**63 - 1


class LastingSyncApplyBlocker(StrEnum):
    """Typed reason the reviewed subset cannot be applied."""

    NONE = "none"
    NOTHING_SELECTED = "nothing_selected"
    STALE_REVIEW = "stale_review"
    ACTIVATION_REVIEW = "activation_review"
    DELETION_REVIEW = "deletion_review"
    MANAGED_PLACEMENT = "managed_placement"
    ROOT_OR_CAPABILITY = "root_or_capability"
    UNSUPPORTED_ATTENTION = "unsupported_attention"


_CHOICE_LABELS = {
    NotesSyncConflictChoice.KEEP_FILE: "Keep file",
    NotesSyncConflictChoice.KEEP_NOTE: "Keep note",
    NotesSyncConflictChoice.KEEP_BOTH: "Keep both",
    NotesSyncConflictChoice.SKIP: "Skip for now",
}


@dataclass(frozen=True, slots=True, repr=False)
class LastingSyncSetup:
    """Fields shown on the explicit root-detail surface only."""

    display_name: str = ""
    folder: str = ""
    destination: str = "local"
    note_scope_id: str = "local_note"
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


@dataclass(frozen=True, slots=True, repr=False)
class LastingSyncReviewRow:
    """One path-free, bounded reviewed effect row."""

    item_id: str
    category: str
    effect: str
    choices: tuple[str, ...] = ()
    action_id: str | None = None
    conflict_eligible: bool = False
    selected_choice: NotesSyncConflictChoice | None = None
    selected_label: str = ""
    conflict_title: str = ""
    conflict_relative_path: str = ""

    def __post_init__(self) -> None:
        if type(self.choices) is not tuple:
            raise TypeError("choices must be a tuple")
        validate_notes_sync_opaque_id(self.item_id, field_name="item_id")
        if self.action_id is not None:
            validate_notes_sync_opaque_id(self.action_id, field_name="action_id")
        if type(self.conflict_eligible) is not bool:
            raise TypeError("conflict_eligible must be a boolean")
        if (
            self.selected_choice is not None
            and type(self.selected_choice) is not NotesSyncConflictChoice
        ):
            raise TypeError("selected_choice must be a NotesSyncConflictChoice")
        if self.selected_choice is not None and not self.conflict_eligible:
            raise ValueError("only eligible conflicts may be selected")
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
        expected_label = (
            f"Selected: {_CHOICE_LABELS[self.selected_choice]}"
            if self.selected_choice is not None
            else ""
        )
        if self.selected_label != expected_label:
            raise ValueError("selected_label must match selected_choice")
        if self.conflict_title:
            if (
                not self.conflict_eligible
                or len(self.conflict_title) > 160
                or "\n" in self.conflict_title
                or "\r" in self.conflict_title
            ):
                raise ValueError("conflict_title must be a bounded conflict label")
        if self.conflict_relative_path:
            if not self.conflict_eligible:
                raise ValueError("only eligible conflicts may carry relative_path")
            object.__setattr__(
                self,
                "conflict_relative_path",
                normalize_notes_sync_relative_path(self.conflict_relative_path),
            )
        if bool(self.conflict_title) != bool(self.conflict_relative_path):
            raise ValueError(
                "conflict title and relative_path must be projected together"
            )

    def __repr__(self) -> str:
        """Keep private note labels out of logs and diagnostic projections."""

        return "LastingSyncReviewRow(<private>)"


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
    can_apply: bool = False
    apply_blocker: LastingSyncApplyBlocker = LastingSyncApplyBlocker.NOTHING_SELECTED

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
        if type(self.can_apply) is not bool:
            raise TypeError("can_apply must be a boolean")
        if type(self.apply_blocker) is not LastingSyncApplyBlocker:
            raise TypeError("apply_blocker must be a LastingSyncApplyBlocker")
        if self.can_apply != (self.apply_blocker is LastingSyncApplyBlocker.NONE):
            raise ValueError("can_apply must match apply_blocker")
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


def _validate_item_label(value: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 160
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("item_label must be bounded single-line text")


@dataclass(frozen=True, slots=True, repr=False)
class LastingSyncReceiptRow:
    """One fresh bounded at-action receipt."""

    operation_id: str
    item_label: str
    choice: NotesSyncConflictChoice
    state: str
    undo_available: bool
    undo_reason: str | None = None

    def __post_init__(self) -> None:
        ConflictReceipt(
            self.operation_id,
            self.choice,
            self.state,
            self.undo_available,
            self.undo_reason,
        )
        _validate_item_label(self.item_label)

    def __repr__(self) -> str:
        return "LastingSyncReceiptRow(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class LastingSyncHistoryRow:
    """One fresh bounded durable resolution-history row."""

    operation_id: str
    item_label: str
    choice: NotesSyncConflictChoice
    state: str
    completed_at: str | None
    updated_at: str
    undo_available: bool
    undo_reason: str | None = None

    def __post_init__(self) -> None:
        ConflictHistoryRow(
            self.operation_id,
            self.choice,
            self.state,
            self.completed_at,
            self.updated_at,
            self.undo_available,
            self.undo_reason,
        )
        _validate_item_label(self.item_label)

    def __repr__(self) -> str:
        return "LastingSyncHistoryRow(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class LastingSyncHistory:
    """One bounded newest-first durable history page."""

    root_id: str = ""
    rows: tuple[LastingSyncHistoryRow, ...] = ()
    page: int = 1
    has_next: bool = False
    unavailable: bool = False

    def __post_init__(self) -> None:
        if self.root_id:
            validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        if type(self.rows) is not tuple or any(
            type(row) is not LastingSyncHistoryRow for row in self.rows
        ):
            raise TypeError("history rows must be a tuple of history rows")
        if len(self.rows) > 100:
            raise ValueError("history rows must be bounded to one page")
        validate_lasting_sync_history_page(self.page)
        if type(self.has_next) is not bool or type(self.unavailable) is not bool:
            raise TypeError("history flags must be booleans")

    def __repr__(self) -> str:
        return (
            "LastingSyncHistory("
            f"rows={len(self.rows)}, page={self.page}, "
            f"has_next={self.has_next}, unavailable={self.unavailable})"
        )


def validate_lasting_sync_history_page(page: int) -> int:
    """Return a SQLite-safe page offset for one bounded history projection."""

    if type(page) is not int or page < 1:
        raise ValueError("history page must be a positive integer")
    largest_page = (_SQLITE_INTEGER_MAX // LASTING_SYNC_HISTORY_PAGE_SIZE) + 1
    if page > largest_page:
        raise ValueError("history page offset exceeds SQLite's integer range")
    return (page - 1) * LASTING_SYNC_HISTORY_PAGE_SIZE


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
    comparison: ConflictComparison | None = None
    receipts: tuple[LastingSyncReceiptRow, ...] = ()
    receipts_unavailable: bool = False
    history: LastingSyncHistory = LastingSyncHistory()
    root_page: int = 1
    root_page_count: int = 1
    history_available: bool = False
    conflict_focus_binding_id: str | None = None

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
        if (
            self.comparison is not None
            and type(self.comparison) is not ConflictComparison
        ):
            raise TypeError("comparison must be a ConflictComparison or None")
        if type(self.receipts) is not tuple or any(
            type(receipt) is not LastingSyncReceiptRow for receipt in self.receipts
        ):
            raise TypeError("receipts must be a tuple of receipt rows")
        if len(self.receipts) > 100:
            raise ValueError("receipts must be bounded to one page")
        if type(self.receipts_unavailable) is not bool:
            raise TypeError("receipts_unavailable must be a boolean")
        if type(self.history) is not LastingSyncHistory:
            raise TypeError("history must be a LastingSyncHistory")
        if type(self.history_available) is not bool:
            raise TypeError("history_available must be a boolean")
        if self.conflict_focus_binding_id is not None:
            validate_notes_sync_opaque_id(
                self.conflict_focus_binding_id,
                field_name="conflict_focus_binding_id",
            )
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

    setup = replace(snapshot.setup, **{field: value})  # type: ignore[arg-type]
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
    selections: tuple[ConflictSelection, ...] = (),
    stale: bool = False,
    activation: bool = False,
) -> LastingSyncReview:
    """Translate a public reconciliation plan into bounded, path-free rows."""

    if type(plan) is not ReconciliationPlan:
        raise TypeError("plan must be a ReconciliationPlan")
    if type(selections) is not tuple or any(
        type(selection) is not ConflictSelection for selection in selections
    ):
        raise TypeError("selections must be a tuple of ConflictSelection")
    selected_by_id = {
        selection.binding_id: selection.choice for selection in selections
    }
    if len(selected_by_id) != len(selections):
        raise ValueError("conflict selections must not contain duplicates")
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
    managed_binding_ids = frozenset(
        effect.binding_id for effect in plan.managed_placement_effects
    )
    for attention in plan.attention:
        conflict_eligible = False
        if attention.kind is ReconciliationAttentionKind.CONFLICT:
            effect = "Both file and note changed"
            choices: tuple[str, ...] = ("Keep file", "Keep note", "Keep both")
            if attention.binding_id is not None and eligible_conflict_reason(
                attention.reason_code,
                managed=attention.binding_id in managed_binding_ids,
            ):
                conflict_eligible = True
                effect = (
                    "Both file and note changed"
                    if attention.reason_code == "both_sides_changed"
                    else "This change is outside the root direction"
                )
                choices = (*choices, "Skip for now")
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
                conflict_eligible=conflict_eligible,
                selected_choice=(
                    selected_by_id.get(attention.binding_id)
                    if conflict_eligible
                    else None
                ),
                selected_label=(
                    f"Selected: {_CHOICE_LABELS[selected_by_id[attention.binding_id]]}"
                    if conflict_eligible and attention.binding_id in selected_by_id
                    else ""
                ),
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
    eligible_ids = {row.item_id for row in rows if row.conflict_eligible}
    if any(binding_id not in eligible_ids for binding_id in selected_by_id):
        raise ValueError("selection is not an eligible conflict")
    blocker = _apply_blocker(
        plan,
        selections=selections,
        stale=stale,
        activation=activation,
    )
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
        stale=stale,
        next_action=(
            "Resolve attention"
            if plan.attention or plan.deletion_groups
            else "Apply reviewed"
        ),
        activation=activation,
        can_apply=blocker is LastingSyncApplyBlocker.NONE,
        apply_blocker=blocker,
    )


def _apply_blocker(
    plan: ReconciliationPlan,
    *,
    selections: tuple[ConflictSelection, ...],
    stale: bool,
    activation: bool,
) -> LastingSyncApplyBlocker:
    if stale:
        return LastingSyncApplyBlocker.STALE_REVIEW
    if activation:
        return LastingSyncApplyBlocker.ACTIVATION_REVIEW
    if plan.deletion_groups or any(
        attention.kind is ReconciliationAttentionKind.DELETION_REVIEW
        for attention in plan.attention
    ):
        return LastingSyncApplyBlocker.DELETION_REVIEW
    if plan.managed_placement_effects:
        return LastingSyncApplyBlocker.MANAGED_PLACEMENT
    if plan.skips:
        return LastingSyncApplyBlocker.ROOT_OR_CAPABILITY
    if any(
        attention.kind is not ReconciliationAttentionKind.CONFLICT
        or attention.binding_id is None
        or not eligible_conflict_reason(attention.reason_code, managed=False)
        for attention in plan.attention
    ):
        return LastingSyncApplyBlocker.UNSUPPORTED_ATTENTION
    if not plan.safe_actions and not any(
        selection.choice is not NotesSyncConflictChoice.SKIP for selection in selections
    ):
        return LastingSyncApplyBlocker.NOTHING_SELECTED
    return LastingSyncApplyBlocker.NONE


__all__ = [
    "LASTING_SYNC_HISTORY_PAGE_SIZE",
    "LastingSyncApplyBlocker",
    "LastingSyncHistory",
    "LastingSyncHistoryRow",
    "LastingSyncReceiptRow",
    "LastingSyncReview",
    "LastingSyncReviewRow",
    "LastingSyncRootRow",
    "LastingSyncSetup",
    "LibraryNotesLastingSyncSnapshot",
    "build_reconciliation_review",
    "initial_lasting_sync_snapshot",
    "set_setup_value",
    "validate_lasting_sync_history_page",
]
