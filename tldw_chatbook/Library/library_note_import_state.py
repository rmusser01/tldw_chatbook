"""Pure, redacted presentation state for the reviewed Notes import workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import re

from tldw_chatbook.Notes.note_import_execution_models import (
    ApprovedNoteImportPlan,
    ImportExecutionProgress,
    ImportExecutionReceipt,
    ImportSessionState,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportMatchKind,
    ImportPreviewItem,
    NoteImportPlan,
    RootCollisionChoice,
    RootCollisionState,
)
from tldw_chatbook.Notes.note_import_planner import apply_item_override


MAX_IMPORT_REVIEW_PAGE_SIZE = 25


class NoteImportPhase(str, Enum):
    """Visible phases of one import-once session."""

    SELECT = "select"
    DESTINATION = "destination"
    CHECKING = "checking"
    REVIEW = "review"
    IMPORTING = "importing"
    RECEIPT = "receipt"


@dataclass(frozen=True, slots=True)
class NoteImportPage:
    """One bounded window over an immutable review plan."""

    items: tuple[ImportPreviewItem, ...] = field(default=(), repr=False)
    page_number: int = 1
    page_size: int = MAX_IMPORT_REVIEW_PAGE_SIZE
    total_items: int = 0
    page_count: int = 1
    has_previous: bool = False
    has_next: bool = False


@dataclass(frozen=True, slots=True)
class NoteImportWorkflowDiagnostic:
    """Count-only diagnostic projection safe for ordinary logs."""

    phase: NoteImportPhase
    selected_count: int
    review_item_count: int
    revision: int
    cancel_requested: bool


@dataclass(frozen=True, slots=True)
class NoteImportWorkflowSnapshot:
    """Immutable workflow authority; private material is hidden from repr."""

    phase: NoteImportPhase = NoteImportPhase.SELECT
    selected_paths: tuple[Path, ...] = field(default=(), repr=False)
    selection_is_folder: bool = False
    destination_segments: tuple[str, ...] = field(default=(), repr=False)
    destination_input: str = field(default="", repr=False)
    destination_error: str = ""
    plan: NoteImportPlan | None = field(default=None, repr=False)
    review_effects: tuple["NoteImportReviewEffect", ...] = field(default=(), repr=False)
    page: NoteImportPage = field(default_factory=NoteImportPage)
    approved_plan: ApprovedNoteImportPlan | None = field(default=None, repr=False)
    progress: ImportExecutionProgress | None = None
    receipt: ImportExecutionReceipt | None = field(default=None, repr=False)
    latest_receipt: ImportExecutionReceipt | None = field(default=None, repr=False)
    cancel_requested: bool = False
    decision_item_ids: frozenset[str] = frozenset()
    collision_rename_input: str = field(default="", repr=False)
    collision_rename_error: str = ""
    revision: int = 0

    @property
    def selected_count(self) -> int:
        return len(self.selected_paths)

    @property
    def requires_destination(self) -> bool:
        return bool(self.selected_paths) and not self.selection_is_folder

    @property
    def can_add_file(self) -> bool:
        return not self.selection_is_folder and self.phase in {
            NoteImportPhase.SELECT,
            NoteImportPhase.DESTINATION,
        }

    @property
    def can_check(self) -> bool:
        return (
            bool(self.selected_paths)
            and (self.selection_is_folder or bool(self.destination_segments))
            and not self.destination_error
            and self.phase in {NoteImportPhase.SELECT, NoteImportPhase.DESTINATION}
        )

    @property
    def approval_blocker(self) -> str:
        if self.phase is not NoteImportPhase.REVIEW or self.plan is None:
            return "Check the selection before importing."
        collision = self.plan.root_collision
        if collision is not None and collision.collides and collision.choice is None:
            return "Choose how to handle the folder name collision."
        if not self.plan.items:
            return "No importable items were found."
        return ""

    @property
    def can_approve(self) -> bool:
        return not self.approval_blocker

    @property
    def can_cancel(self) -> bool:
        return (
            self.phase in {NoteImportPhase.CHECKING, NoteImportPhase.IMPORTING}
            and not self.cancel_requested
        )

    @property
    def is_partial(self) -> bool:
        return bool(self.receipt and self.receipt.completed < self.receipt.total)

    @property
    def can_retry(self) -> bool:
        return bool(
            self.phase is NoteImportPhase.RECEIPT
            and self.receipt is not None
            and (
                self.receipt.retryable > 0
                or self.receipt.completed < self.receipt.total
            )
            and self.approved_plan is not None
        )

    @property
    def can_revisit_receipt(self) -> bool:
        return bool(
            self.latest_receipt is not None
            and (
                self.phase is NoteImportPhase.RECEIPT
                or (self.phase is NoteImportPhase.SELECT and not self.selected_paths)
            )
        )

    def to_diagnostic(self) -> NoteImportWorkflowDiagnostic:
        return NoteImportWorkflowDiagnostic(
            phase=self.phase,
            selected_count=self.selected_count,
            review_item_count=len(self.plan.items) if self.plan is not None else 0,
            revision=self.revision,
            cancel_requested=self.cancel_requested,
        )


@dataclass(frozen=True, slots=True)
class LibraryNoteImportItemSnapshot:
    """Path-safe render projection for one review item."""

    item_id: str
    name: str = field(repr=False)
    classification: str
    action: str
    reason: str = ""
    can_update: bool = False
    uncertain: bool = False
    confirmed: bool = False
    replace_content: bool = False
    add_membership: bool = False
    target_label: str = field(default="", repr=False)
    effect_summary: str = field(default="", repr=False)
    membership_summary: str = field(default="", repr=False)
    content_diff: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class NoteImportReviewEffect:
    """Bounded private comparison data for one immutable review item."""

    item_id: str
    target_title: str = field(default="", repr=False)
    target_version: int | None = None
    content_diff: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class LibraryNoteImportSnapshot:
    """Complete immutable input for the render-only import canvas."""

    phase: str
    selected_names: tuple[str, ...] = field(repr=False)
    selection_kind: str
    destination: str = field(repr=False)
    status_line: str
    preview_items: tuple[LibraryNoteImportItemSnapshot, ...]
    page: int
    page_count: int
    can_check: bool
    check_disabled_reason: str
    can_import: bool
    import_disabled_reason: str
    destination_error: str = ""
    collision_kind: str = ""
    collision_name: str = field(default="", repr=False)
    collision_choice: str = ""
    collision_reason: str = ""
    collision_rename_input: str = field(default="", repr=False)
    collision_rename_error: str = ""
    collision_rename_available: bool = False
    progress_completed: int = 0
    progress_total: int = 0
    progress_detail: str = ""
    receipt_line: str = ""
    receipt_detail: str = ""
    retryable_failures: int = 0
    retry_available: bool = False
    retry_label: str = ""
    can_cancel: bool = False


def _page(
    plan: NoteImportPlan | None, page_number: int, page_size: int
) -> NoteImportPage:
    size = min(max(int(page_size), 1), MAX_IMPORT_REVIEW_PAGE_SIZE)
    total = len(plan.items) if plan is not None else 0
    page_count = max(1, (total + size - 1) // size)
    number = min(max(int(page_number), 1), page_count)
    start = (number - 1) * size
    items = plan.items[start : start + size] if plan is not None else ()
    return NoteImportPage(
        items=items,
        page_number=number,
        page_size=size,
        total_items=total,
        page_count=page_count,
        has_previous=number > 1,
        has_next=number < page_count,
    )


def initial_note_import_snapshot(
    *,
    page_size: int = MAX_IMPORT_REVIEW_PAGE_SIZE,
    latest_receipt: ImportExecutionReceipt | None = None,
) -> NoteImportWorkflowSnapshot:
    """Return a fresh selection state while optionally retaining one receipt."""

    return NoteImportWorkflowSnapshot(
        page=_page(None, 1, page_size),
        latest_receipt=latest_receipt,
    )


def add_selected_file(
    state: NoteImportWorkflowSnapshot, path: Path
) -> NoteImportWorkflowSnapshot:
    if state.selection_is_folder:
        raise ValueError("Cannot add a file after selecting a folder.")
    if not isinstance(path, Path):
        raise TypeError("path must be a Path.")
    paths = (
        state.selected_paths
        if path in state.selected_paths
        else state.selected_paths + (path,)
    )
    return replace(
        state,
        phase=NoteImportPhase.DESTINATION,
        selected_paths=paths,
        plan=None,
        page=_page(None, 1, state.page.page_size),
        approved_plan=None,
        progress=None,
        receipt=None,
        cancel_requested=False,
        decision_item_ids=frozenset(),
        revision=state.revision + (paths != state.selected_paths),
    )


def select_folder(
    state: NoteImportWorkflowSnapshot, path: Path
) -> NoteImportWorkflowSnapshot:
    if state.selected_paths:
        raise ValueError("A folder must be selected without files.")
    if not isinstance(path, Path):
        raise TypeError("path must be a Path.")
    return replace(
        state,
        phase=NoteImportPhase.SELECT,
        selected_paths=(path,),
        selection_is_folder=True,
        destination_segments=(),
        destination_input="",
        destination_error="",
        revision=state.revision + 1,
    )


def set_destination_segments(
    state: NoteImportWorkflowSnapshot, segments: tuple[str, ...]
) -> NoteImportWorkflowSnapshot:
    copied = tuple(segments)
    for segment in copied:
        if (
            not isinstance(segment, str)
            or not segment.strip()
            or segment != segment.strip()
            or segment in {".", ".."}
            or "/" in segment
            or "\\" in segment
            or "\x00" in segment
        ):
            raise ValueError("destination contains an invalid folder segment.")
    changed = copied != state.destination_segments
    return replace(
        state,
        phase=NoteImportPhase.DESTINATION,
        destination_segments=copied,
        destination_input="/".join(copied),
        destination_error="",
        plan=None if changed else state.plan,
        page=_page(None, 1, state.page.page_size) if changed else state.page,
        approved_plan=None if changed else state.approved_plan,
        revision=state.revision + changed,
    )


def _folder_name_error(value: str) -> str:
    if not value:
        return "Enter a Notes destination."
    if value != value.strip():
        return "Remove leading or trailing spaces from each folder name."
    if value in {".", ".."}:
        return "Use a folder name other than '.' or '..'."
    if "\x00" in value:
        return "Remove the unsupported character from the folder name."
    return ""


def parse_note_import_destination(value: str) -> tuple[tuple[str, ...], str]:
    """Return safe destination segments and one inline correction message."""
    if not isinstance(value, str):
        raise TypeError("destination must be text.")
    if not value:
        return (), "Enter a Notes destination."
    parts = tuple(re.split(r"[/\\]", value))
    if any(not part for part in parts):
        return (), "Remove the empty folder between separators."
    for part in parts:
        error = _folder_name_error(part)
        if error:
            return (), error
    return parts, ""


def set_destination_input(
    state: NoteImportWorkflowSnapshot, value: str
) -> NoteImportWorkflowSnapshot:
    """Retain exact typed destination text with parsed, mutation-free authority."""
    segments, error = parse_note_import_destination(value)
    changed = (
        value != state.destination_input
        or segments != state.destination_segments
        or error != state.destination_error
    )
    return replace(
        state,
        phase=NoteImportPhase.DESTINATION,
        destination_input=value,
        destination_error=error,
        destination_segments=segments,
        plan=None if changed else state.plan,
        review_effects=() if changed else state.review_effects,
        page=_page(None, 1, state.page.page_size) if changed else state.page,
        approved_plan=None if changed else state.approved_plan,
        revision=state.revision + changed,
    )


def begin_checking(state: NoteImportWorkflowSnapshot) -> NoteImportWorkflowSnapshot:
    if (
        not state.can_check
        and state.phase is not NoteImportPhase.REVIEW
        and state.plan is None
    ):
        raise ValueError("The import selection is not ready to check.")
    return replace(
        state,
        phase=NoteImportPhase.CHECKING,
        plan=None,
        review_effects=(),
        page=_page(None, 1, state.page.page_size),
        approved_plan=None,
        progress=None,
        receipt=None,
        cancel_requested=False,
        decision_item_ids=frozenset(),
        collision_rename_input="",
        collision_rename_error="",
        revision=state.revision + 1,
    )


def show_review(
    state: NoteImportWorkflowSnapshot,
    plan: NoteImportPlan,
    *,
    review_effects: tuple[NoteImportReviewEffect, ...] = (),
) -> NoteImportWorkflowSnapshot:
    if state.phase is not NoteImportPhase.CHECKING:
        raise ValueError("Review may only follow checking.")
    if type(plan) is not NoteImportPlan:
        raise TypeError("plan must be a NoteImportPlan.")
    return replace(
        state,
        phase=NoteImportPhase.REVIEW,
        plan=plan,
        review_effects=tuple(review_effects),
        page=_page(plan, 1, state.page.page_size),
        approved_plan=None,
        cancel_requested=False,
        decision_item_ids=frozenset(),
        revision=state.revision + 1,
    )


def set_review_page(
    state: NoteImportWorkflowSnapshot, page_number: int
) -> NoteImportWorkflowSnapshot:
    if state.plan is None:
        raise ValueError("A review plan is required.")
    return replace(state, page=_page(state.plan, page_number, state.page.page_size))


def set_root_collision_resolution(
    state: NoteImportWorkflowSnapshot,
    choice: RootCollisionChoice,
    *,
    resolved_label: str | None = None,
) -> NoteImportWorkflowSnapshot:
    if state.plan is None or state.plan.root_collision is None:
        raise ValueError("A root collision is required.")
    collision = state.plan.root_collision
    updated_collision = RootCollisionState(
        proposed_label=collision.proposed_label,
        collides=collision.collides,
        choice=choice,
        resolved_label=resolved_label,
    )
    plan = replace(state.plan, root_collision=updated_collision)
    return replace(
        state,
        plan=plan,
        page=_page(plan, state.page.page_number, state.page.page_size),
        approved_plan=None,
        revision=state.revision + 1,
    )


def set_collision_rename(
    state: NoteImportWorkflowSnapshot,
    value: str,
    *,
    error: str,
) -> NoteImportWorkflowSnapshot:
    """Retain an exact rename proposal and clear stale rename approval."""
    if state.plan is None or state.plan.root_collision is None:
        raise ValueError("A root collision is required.")
    collision = state.plan.root_collision
    plan = state.plan
    if (
        collision.choice is RootCollisionChoice.RENAMED_ROOT
        and value != collision.resolved_label
    ):
        plan = replace(
            plan,
            root_collision=RootCollisionState(
                proposed_label=collision.proposed_label,
                collides=collision.collides,
            ),
        )
    changed = (
        value != state.collision_rename_input
        or error != state.collision_rename_error
        or plan is not state.plan
    )
    return replace(
        state,
        plan=plan,
        page=_page(plan, state.page.page_number, state.page.page_size),
        approved_plan=None if changed else state.approved_plan,
        collision_rename_input=value,
        collision_rename_error=error,
        revision=state.revision + changed,
    )


def set_item_decision(
    state: NoteImportWorkflowSnapshot,
    item_id: str,
    *,
    action: ImportAction,
    replace_content: bool,
    add_membership: bool,
) -> NoteImportWorkflowSnapshot:
    if state.plan is None:
        raise ValueError("A review plan is required.")
    item = next(
        (candidate for candidate in state.plan.items if candidate.item_id == item_id),
        None,
    )
    if (
        action is ImportAction.UPDATE_EXISTING
        and item is not None
        and item.match is not None
        and item.match.kind is ImportMatchKind.UNCERTAIN
    ):
        raise ValueError("The uncertain match must be confirmed before updating.")
    plan = apply_item_override(
        state.plan,
        item_id,
        action,
        replace_content=replace_content,
        add_membership=add_membership,
    )
    return replace(
        state,
        plan=plan,
        page=_page(plan, state.page.page_number, state.page.page_size),
        approved_plan=None,
        decision_item_ids=state.decision_item_ids | {item_id},
        revision=state.revision + 1,
    )


def set_approved_plan(
    state: NoteImportWorkflowSnapshot, approved: ApprovedNoteImportPlan
) -> NoteImportWorkflowSnapshot:
    if state.plan is None or approved.plan is not state.plan:
        raise ValueError("Approval must bind the exact current plan.")
    if not state.can_approve:
        raise ValueError(state.approval_blocker)
    return replace(state, approved_plan=approved)


def begin_importing(state: NoteImportWorkflowSnapshot) -> NoteImportWorkflowSnapshot:
    if (
        state.plan is None
        or state.approved_plan is None
        or state.approved_plan.plan is not state.plan
    ):
        raise ValueError("An exact approved plan is required before importing.")
    progress = ImportExecutionProgress(
        state=ImportSessionState.PENDING,
        total=len(state.plan.items),
        completed=0,
        imported=0,
        updated=0,
        skipped=0,
        failed=0,
        retryable=0,
    )
    return replace(
        state,
        phase=NoteImportPhase.IMPORTING,
        progress=progress,
        cancel_requested=False,
    )


def apply_import_progress(
    state: NoteImportWorkflowSnapshot, progress: ImportExecutionProgress
) -> NoteImportWorkflowSnapshot:
    if state.phase is not NoteImportPhase.IMPORTING:
        raise ValueError("Progress requires an active import.")
    previous = state.progress
    if previous is not None:
        admitting_first_progress = (
            previous.state is ImportSessionState.PENDING
            and previous.completed == 0
            and progress.state is not ImportSessionState.PENDING
        )
        if progress.total != previous.total and not admitting_first_progress:
            raise ValueError("Progress total cannot change.")
        counters = (
            "completed",
            "imported",
            "updated",
            "skipped",
            "failed",
            "retryable",
        )
        if any(getattr(progress, name) < getattr(previous, name) for name in counters):
            raise ValueError("Import progress cannot regress.")
    return replace(state, progress=progress)


def request_import_cancellation(
    state: NoteImportWorkflowSnapshot,
) -> NoteImportWorkflowSnapshot:
    if state.phase not in {NoteImportPhase.CHECKING, NoteImportPhase.IMPORTING}:
        raise ValueError("Cancellation requires active checking or importing.")
    if state.cancel_requested:
        return state
    return replace(state, cancel_requested=True)


def settle_import(
    state: NoteImportWorkflowSnapshot, receipt: ImportExecutionReceipt
) -> NoteImportWorkflowSnapshot:
    if state.phase is not NoteImportPhase.IMPORTING or state.approved_plan is None:
        raise ValueError("An active approved import is required.")
    if receipt.approval_id != state.approved_plan.approval_id:
        raise ValueError("Receipt approval does not match the active import.")
    return replace(
        state,
        phase=NoteImportPhase.RECEIPT,
        progress=None,
        receipt=receipt,
        latest_receipt=receipt,
        cancel_requested=False,
    )


def begin_retry(state: NoteImportWorkflowSnapshot) -> NoteImportWorkflowSnapshot:
    if not state.can_retry or state.approved_plan is None or state.plan is None:
        raise ValueError("The receipt has no retryable failures.")
    progress = ImportExecutionProgress(
        state=ImportSessionState.PENDING,
        total=state.receipt.total
        if state.receipt is not None
        else len(state.plan.items),
        completed=0,
        imported=0,
        updated=0,
        skipped=0,
        failed=0,
        retryable=0,
    )
    return replace(
        state,
        phase=NoteImportPhase.IMPORTING,
        progress=progress,
        cancel_requested=False,
    )


def revisit_latest_receipt(
    state: NoteImportWorkflowSnapshot,
) -> NoteImportWorkflowSnapshot:
    if state.latest_receipt is None:
        raise ValueError("There is no receipt to revisit.")
    return replace(
        state,
        phase=NoteImportPhase.RECEIPT,
        receipt=state.latest_receipt,
        progress=None,
        cancel_requested=False,
    )


def project_library_note_import_snapshot(
    state: NoteImportWorkflowSnapshot,
) -> LibraryNoteImportSnapshot:
    """Project private workflow authority into bounded, path-safe UI copy."""

    effects = {effect.item_id: effect for effect in state.review_effects}
    items = tuple(
        LibraryNoteImportItemSnapshot(
            item_id=item.item_id,
            name=item.source.display_path,
            classification=item.classification.value,
            action=item.selected_action.value,
            reason=item.reason,
            can_update=ImportAction.UPDATE_EXISTING in item.allowed_actions
            and item.match is not None
            and item.match.kind
            in {ImportMatchKind.EXACT, ImportMatchKind.USER_CONFIRMED},
            uncertain=bool(
                item.match is not None and item.match.kind is ImportMatchKind.UNCERTAIN
            ),
            confirmed=bool(
                item.match is not None
                and item.match.kind is ImportMatchKind.USER_CONFIRMED
            ),
            replace_content=item.replace_content,
            add_membership=item.add_membership,
            target_label=_target_label(effects.get(item.item_id)),
            effect_summary=_effect_summary(item),
            membership_summary=_membership_summary(item),
            content_diff=(
                effects[item.item_id].content_diff if item.item_id in effects else ""
            ),
        )
        for item in state.page.items
    )
    collision = state.plan.root_collision if state.plan is not None else None
    progress = state.progress
    receipt = state.receipt
    if state.phase is NoteImportPhase.CHECKING:
        status = f"◌ Checking {state.selected_count} selected source{'s' if state.selected_count != 1 else ''}…"
    elif state.phase is NoteImportPhase.REVIEW:
        status = f"Review {state.page.total_items} item{'s' if state.page.total_items != 1 else ''} before import."
    elif state.phase is NoteImportPhase.IMPORTING:
        status = (
            "Stopping after the current item…"
            if state.cancel_requested
            else "Importing notes…"
        )
    elif state.phase is NoteImportPhase.RECEIPT:
        status = _receipt_status(receipt)
    elif state.selected_count:
        status = f"{state.selected_count} {'folder' if state.selection_is_folder else 'file' + ('s' if state.selected_count != 1 else '')} selected."
    else:
        status = "Choose one or more files, or one folder."
    selection_kind = (
        "folder"
        if state.selection_is_folder
        else ("files" if state.selected_paths else "")
    )
    return LibraryNoteImportSnapshot(
        phase=state.phase.value,
        selected_names=tuple(path.name for path in state.selected_paths),
        selection_kind=selection_kind,
        destination=state.destination_input,
        destination_error=state.destination_error,
        status_line=status,
        preview_items=items,
        page=state.page.page_number,
        page_count=state.page.page_count,
        can_check=state.can_check,
        check_disabled_reason=(
            "Choose a source first."
            if not state.selected_paths
            else state.destination_error
            if state.destination_error
            else "Choose a Notes destination."
            if state.requires_destination and not state.destination_segments
            else ""
        ),
        can_import=state.can_approve,
        import_disabled_reason=state.approval_blocker,
        collision_kind="root" if collision and collision.collides else "",
        collision_name=(collision.resolved_label or collision.proposed_label)
        if collision
        else "",
        collision_choice=collision.choice.value
        if collision and collision.choice
        else "",
        collision_reason=(
            "Choose how to handle the existing folder."
            if collision and collision.collides and collision.choice is None
            else ""
        ),
        collision_rename_input=state.collision_rename_input,
        collision_rename_error=state.collision_rename_error,
        collision_rename_available=bool(
            state.collision_rename_input and not state.collision_rename_error
        ),
        progress_completed=progress.completed if progress else 0,
        progress_total=progress.total if progress else 0,
        progress_detail=(
            f"{progress.imported} imported · {progress.skipped} skipped · {progress.failed} failed"
            if progress
            else ""
        ),
        receipt_line=(
            f"{receipt.imported} imported · {receipt.updated} updated · {receipt.skipped} skipped · {receipt.failed} failed"
            if receipt
            else ""
        ),
        receipt_detail=_receipt_detail(receipt),
        retryable_failures=receipt.retryable if receipt else 0,
        retry_available=state.can_retry,
        retry_label=(
            f"Retry {receipt.retryable} "
            f"{'failure' if receipt.retryable == 1 else 'failures'}"
            if receipt and receipt.retryable
            else "Retry unfinished items"
            if receipt and receipt.completed < receipt.total
            else ""
        ),
        can_cancel=state.can_cancel,
    )


def _target_label(effect: NoteImportReviewEffect | None) -> str:
    if effect is None or not effect.target_title:
        return ""
    title = (
        effect.target_title
        if len(effect.target_title) <= 120
        else f"{effect.target_title[:119]}…"
    )
    version = (
        f" (version {effect.target_version})"
        if effect.target_version is not None
        else ""
    )
    return f"Existing note: {title}{version}."


def _membership_summary(item: ImportPreviewItem) -> str:
    if item.selected_action is not ImportAction.UPDATE_EXISTING:
        if item.selected_action is ImportAction.SKIP:
            return "Folder placement: no change."
        verb = "Create in"
    elif not item.add_membership:
        return "Folder placement: unchanged."
    else:
        verb = "Folder placement: add"
    paths = tuple(
        dict.fromkeys(" / ".join(entry.folder_segments) for entry in item.memberships)
    )
    shown = "; ".join(paths[:3])
    if len(paths) > 3:
        shown = f"{shown}; and {len(paths) - 3} more"
    return f"{verb} {shown}."


def _effect_summary(item: ImportPreviewItem) -> str:
    if item.selected_action is ImportAction.SKIP:
        return "Content: no change."
    if item.selected_action is ImportAction.CREATE_NEW:
        count = len(item.payloads)
        return f"Content: create {count} new {'note' if count == 1 else 'notes'}."
    return (
        "Content: replace existing content."
        if item.replace_content
        else "Content: keep existing content."
    )


def _receipt_status(receipt: ImportExecutionReceipt | None) -> str:
    if receipt is None:
        return "Import status unavailable."
    if receipt.state is ImportSessionState.COMPLETED:
        return "Import completed."
    if receipt.state is ImportSessionState.CANCELLED:
        return "Import cancelled after the current item."
    if receipt.state is ImportSessionState.NEEDS_ATTENTION:
        return "Import needs attention."
    return "Import status unavailable."


def _receipt_detail(receipt: ImportExecutionReceipt | None) -> str:
    if receipt is None:
        return ""
    if receipt.state is ImportSessionState.CANCELLED:
        return "Cancelled. Finished items were not rolled back."
    if receipt.state is ImportSessionState.NEEDS_ATTENTION:
        return "Some items failed. Completed changes were kept."
    return "All planned items settled."
