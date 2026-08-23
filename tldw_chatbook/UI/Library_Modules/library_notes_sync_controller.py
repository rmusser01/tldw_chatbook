"""Focused controller for inert lasting Notes sync presentation flows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Never, Protocol

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LASTING_SYNC_HISTORY_PAGE_SIZE,
    LastingSyncHistory,
    LastingSyncHistoryRow,
    LastingSyncReceiptRow,
    LastingSyncRootRow,
    LibraryNotesLastingSyncSnapshot,
    build_reconciliation_review,
    initial_lasting_sync_snapshot,
    set_setup_value,
    validate_lasting_sync_history_page,
)
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictApplyResult,
    ConflictComparison,
    ConflictSelection,
    NotesSyncConflictChoice,
)
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutionResult
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationPlan
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncControlResult,
    NotesSyncRootSetup,
    NotesSyncRuntimeSnapshot,
    RuntimeConflictHistoryRow,
    RuntimeConflictReceipt,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncDirection,
    NotesSyncOperationState,
)


class LastingSyncRuntimePort(Protocol):
    """Only the public runtime operations the Library UI is allowed to call.

    The UI plan's initial list omitted the TASK-19009 public manual-check alias
    and operation-specific cleanup method; both are included here so the root
    canvas uses the existing runtime instead of inventing parallel seams.
    """

    def snapshot(self) -> NotesSyncRuntimeSnapshot: ...

    async def check_root(self, root_id: str) -> ReconciliationPlan: ...

    async def review_setup(self, setup: NotesSyncRootSetup) -> ReconciliationPlan: ...

    async def abandon_setup(self, root_id: str) -> None: ...

    async def request_sync_now(self, root_id: str) -> ReconciliationPlan: ...

    async def apply_reviewed(
        self,
        root_id: str,
        observation_token: str,
        action_ids: tuple[str, ...],
        selections: tuple[ConflictSelection, ...] = (),
    ) -> ConflictApplyResult: ...

    async def compare_conflict(
        self, root_id: str, observation_token: str, binding_id: str
    ) -> ConflictComparison: ...

    async def active_conflict_receipts(
        self, root_id: str
    ) -> tuple[RuntimeConflictReceipt, ...]: ...

    def dismiss_conflict_receipt(self, root_id: str, operation_id: str) -> None: ...

    async def undo_resolution(
        self, root_id: str, operation_id: str
    ) -> NotesSyncExecutionResult: ...

    async def resolution_history(
        self,
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]: ...

    async def activate_root(
        self, root_id: str, authorization: object
    ) -> NotesSyncControlResult: ...

    async def pause_root(self, root_id: str) -> NotesSyncControlResult: ...

    async def resume_root(self, root_id: str) -> NotesSyncControlResult: ...

    async def retarget_root(
        self, root_id: str, target: str
    ) -> NotesSyncControlResult: ...

    async def disconnect_root(
        self, root_id: str, keep: bool
    ) -> NotesSyncControlResult: ...

    async def resolve_cleanup(self, root_id: str, operation_id: str) -> object: ...


class _ImportOncePort(Protocol):
    def begin_selection(self) -> None: ...


class InertLastingSyncRuntime:
    """Honest structural adapter when the app-owned runtime is unavailable."""

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        return NotesSyncRuntimeSnapshot("awaiting_cutover", "finish_upgrade")

    async def _blocked(self) -> Never:
        raise RuntimeError("notes_sync_cutover_not_admitted")

    async def check_root(self, root_id: str) -> ReconciliationPlan:
        return await self._blocked()

    async def review_setup(self, setup: NotesSyncRootSetup) -> ReconciliationPlan:
        return await self._blocked()

    async def abandon_setup(self, root_id: str) -> None:
        await self._blocked()

    async def request_sync_now(self, root_id: str) -> ReconciliationPlan:
        return await self._blocked()

    async def apply_reviewed(
        self,
        root_id: str,
        observation_token: str,
        action_ids: tuple[str, ...],
        selections: tuple[ConflictSelection, ...] = (),
    ) -> ConflictApplyResult:
        return await self._blocked()

    async def compare_conflict(
        self, root_id: str, observation_token: str, binding_id: str
    ) -> ConflictComparison:
        return await self._blocked()

    async def active_conflict_receipts(
        self, root_id: str
    ) -> tuple[RuntimeConflictReceipt, ...]:
        return await self._blocked()

    def dismiss_conflict_receipt(self, root_id: str, operation_id: str) -> None:
        raise RuntimeError("notes_sync_cutover_not_admitted")

    async def undo_resolution(
        self, root_id: str, operation_id: str
    ) -> NotesSyncExecutionResult:
        return await self._blocked()

    async def resolution_history(
        self,
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        return await self._blocked()

    async def activate_root(
        self, root_id: str, authorization: object
    ) -> NotesSyncControlResult:
        return await self._blocked()

    async def pause_root(self, root_id: str) -> NotesSyncControlResult:
        return await self._blocked()

    async def resume_root(self, root_id: str) -> NotesSyncControlResult:
        return await self._blocked()

    async def retarget_root(self, root_id: str, target: str) -> NotesSyncControlResult:
        return await self._blocked()

    async def disconnect_root(self, root_id: str, keep: bool) -> NotesSyncControlResult:
        return await self._blocked()

    async def resolve_cleanup(self, root_id: str, operation_id: str) -> object:
        return await self._blocked()


_STATUS_LABELS = {
    "up_to_date": "✓ Up to date",
    "changes_available": "◌ Changes available",
    "paused": "Ⅱ Paused",
    "offline": "⚠ Offline",
    "passive": "Ⅱ Open in another process",
    "needs_attention": "⚠ Needs attention",
    "partial": "⚠ Partial",
    "failed": "✕ Failed",
    "unsupported": "✕ Blocked",
    "starting": "◌ Starting",
}
_ACTION_LABELS = {
    "sync_now": "Check changes",
    "review_changes": "Review changes",
    "resume_sync": "Resume",
    "reconnect_folder": "Reconnect folder",
    "open_active_process": "Open active process",
    "review_settings": "Review settings",
    "review_migration": "Review migration",
    "resolve_cleanup": "Resolve recovery",
    "wait": "Wait",
    "none": "No action",
    "apply_reviewed": "Apply reviewed",
    "finish_upgrade": "Finish upgrade",
    "close_other_process_and_restart": "Close other process and restart",
}
_ROOT_PAGE_SIZE = 20
# TASK-21112: "not_configured" is the boot-deferred runtime — nothing is set
# up yet, but first-time setup must be offered (review_setup live-starts the
# machinery on demand), so it counts as available alongside "active".
_SETUP_READY_STATUSES = frozenset({"active", "not_configured"})
_CHOICES_BY_LABEL = {
    "Keep file": NotesSyncConflictChoice.KEEP_FILE,
    "Keep note": NotesSyncConflictChoice.KEEP_NOTE,
    "Keep both": NotesSyncConflictChoice.KEEP_BOTH,
    "Skip for now": NotesSyncConflictChoice.SKIP,
}


class LibraryNotesSyncController:
    """Own chooser/review tasks while storage and Textual remain late-bound."""

    def __init__(
        self,
        *,
        runtime: LastingSyncRuntimePort,
        import_controller: _ImportOncePort,
        publish_snapshot: Callable[[LibraryNotesLastingSyncSnapshot], None]
        | None = None,
    ) -> None:
        self._runtime = runtime
        self._import_controller = import_controller
        self._publish_snapshot = publish_snapshot
        self._review_plan: ReconciliationPlan | None = None
        self._selections: dict[tuple[str, str], NotesSyncConflictChoice] = {}
        self._comparison_generation = 0
        self._expanded_binding_id: str | None = None
        self._projection_root_id: str | None = None
        self._receipt_generation = 0
        self._history_generation = 0
        self._history_request_page = 1
        self._state = initial_lasting_sync_snapshot(
            lasting_available=runtime.snapshot().status in _SETUP_READY_STATUSES
        )
        self._all_roots: tuple[LastingSyncRootRow, ...] = ()
        self.refresh_roots()

    def _current_selections(self) -> tuple[ConflictSelection, ...]:
        plan = self._review_plan
        if plan is None:
            return ()
        return tuple(
            ConflictSelection(binding_id, choice)
            for (token, binding_id), choice in sorted(self._selections.items())
            if token == plan.observation_token
        )

    def _project_review(
        self,
        *,
        page: int | None = None,
        stale: bool | None = None,
        activation: bool | None = None,
    ) -> None:
        plan = self._review_plan
        if plan is None:
            return
        current = self._state.review
        self._state = replace(
            self._state,
            review=build_reconciliation_review(
                plan,
                page=current.page if page is None else page,
                selections=self._current_selections(),
                stale=current.stale if stale is None else stale,
                activation=current.activation if activation is None else activation,
            ),
        )

    def _clear_comparison(self) -> None:
        self._comparison_generation += 1
        self._expanded_binding_id = None
        if self._state.comparison is not None:
            self._state = replace(self._state, comparison=None)

    def _clear_ephemeral_review(self) -> None:
        self._selections.clear()
        self._clear_comparison()
        self._project_review()

    def invalidate_for_remount(self) -> None:
        """Release review-only state when the presenting canvas is replaced."""

        self._clear_ephemeral_review()
        self._invalidate_receipt_history(clear_root=True)

    def _invalidate_receipt_history(self, *, clear_root: bool = False) -> None:
        self._receipt_generation += 1
        self._history_generation += 1
        self._history_request_page = 1
        if clear_root:
            self._projection_root_id = None
        self._state = replace(
            self._state,
            receipts=(),
            history=LastingSyncHistory(),
        )

    def _switch_projection_root(self, root_id: str) -> None:
        if root_id == self._projection_root_id:
            return
        self._projection_root_id = root_id
        self._invalidate_receipt_history()

    def _start_receipt_request(self, root_id: str) -> int:
        self._switch_projection_root(root_id)
        self._receipt_generation += 1
        return self._receipt_generation

    def _receipt_is_current(self, root_id: str, generation: int) -> bool:
        return (
            root_id == self._projection_root_id
            and generation == self._receipt_generation
        )

    def _start_history_request(self, root_id: str, page: int) -> int:
        self._switch_projection_root(root_id)
        self._history_generation += 1
        self._history_request_page = page
        return self._history_generation

    def _history_is_current(self, root_id: str, page: int, generation: int) -> bool:
        return (
            root_id == self._projection_root_id
            and page == self._history_request_page
            and generation == self._history_generation
        )

    def _unavailable_copy(self) -> str:
        runtime = self._runtime.snapshot()
        if runtime.next_action == "close_other_process_and_restart":
            return (
                "Close the other Chatbook process and restart before activating "
                "folder sync"
            )
        if runtime.status == "failed":
            return "Lasting folder sync could not start. Review settings and restart."
        return "Lasting folder sync is unavailable until the reviewed cutover."

    @property
    def snapshot(self) -> LibraryNotesLastingSyncSnapshot:
        return self._state

    def _publish(self) -> None:
        if callable(self._publish_snapshot):
            self._publish_snapshot(self._state)

    def refresh_roots(self, *, publish: bool = True) -> None:
        """Refresh path-free root rows from the public runtime projection."""

        runtime = self._runtime.snapshot()
        available = runtime.status in _SETUP_READY_STATUSES
        self._all_roots = tuple(
            LastingSyncRootRow(
                root.root_id,
                "Sync folder (name unavailable before cutover)",
                root.status,
                root.next_action,
                _STATUS_LABELS.get(root.status, root.status.replace("_", " ").title()),
                _ACTION_LABELS.get(
                    root.next_action, root.next_action.replace("_", " ").title()
                ),
                root.action_id,
            )
            for root in runtime.roots
        )
        page_count = max(
            1, (len(self._all_roots) + _ROOT_PAGE_SIZE - 1) // _ROOT_PAGE_SIZE
        )
        page = min(self._state.root_page, page_count)
        start = (page - 1) * _ROOT_PAGE_SIZE
        self._state = replace(
            self._state,
            lasting_available=available,
            roots=self._all_roots[start : start + _ROOT_PAGE_SIZE],
            root_page=page,
            root_page_count=page_count,
        )
        if publish:
            self._publish()

    def set_root_page(self, page: int) -> None:
        """Show one bounded path-free page of roots."""

        if type(page) is not int:
            raise TypeError("page must be an integer")
        bounded = min(max(1, page), self._state.root_page_count)
        start = (bounded - 1) * _ROOT_PAGE_SIZE
        self._state = replace(
            self._state,
            phase="roots",
            roots=self._all_roots[start : start + _ROOT_PAGE_SIZE],
            root_page=bounded,
        )
        self._publish()

    def choose_relationship(self, relationship: str) -> str:
        """Choose one-time import or the gated lasting setup before any picker."""

        if relationship == "import_once":
            self._import_controller.begin_selection()
            return "import"
        if relationship != "keep_synced":
            raise ValueError("unknown relationship")
        available = self._runtime.snapshot().status in _SETUP_READY_STATUSES
        if available != self._state.lasting_available:
            self._state = replace(self._state, lasting_available=available)
        if not self._state.lasting_available:
            self._state = replace(
                self._state,
                status_line=self._unavailable_copy(),
            )
            self._publish()
            return "choose"
        self._invalidate_receipt_history(clear_root=True)
        self._state = replace(
            self._state,
            phase="configure",
            status_line="Configure a local lasting sync root.",
        )
        self._publish()
        return "configure"

    def set_setup(self, field: str, value: str) -> None:
        self._state = set_setup_value(self._state, field, value)
        self._publish()

    async def check_root(self, root_id: str) -> None:
        self._switch_projection_root(root_id)
        self._selections.clear()
        self._clear_comparison()
        self._state = replace(
            self._state, phase="checking", status_line="Checking changes…"
        )
        self._publish()
        try:
            plan = await self._runtime.check_root(root_id)
        except Exception:
            self._state = replace(
                self._state,
                phase="review",
                review=replace(self._state.review, next_action="Check again"),
                status_line="Check failed. Review root status, then Check again.",
            )
            self._publish()
            return
        self._switch_projection_root(plan.root_id)
        self._review_plan = plan
        self._state = replace(
            self._state,
            phase="review",
            review=build_reconciliation_review(plan),
            status_line="Review the mutation-free check before applying changes.",
        )
        self._publish()

    async def check_setup(self) -> None:
        """Run the same mutation-free check and label it as activation review."""
        setup = self._state.setup
        direction = NotesSyncDirection(setup.direction)
        self._selections.clear()
        self._clear_comparison()
        self._state = replace(
            self._state, phase="checking", status_line="Checking folder…"
        )
        self._publish()
        try:
            plan = await self._runtime.review_setup(
                NotesSyncRootSetup(
                    display_name=setup.display_name,
                    canonical_path=setup.folder,
                    note_scope_id=setup.note_scope_id,
                    direction=direction,
                )
            )
        except Exception:
            self._state = replace(
                self._state,
                phase="configure",
                status_line="Check failed. Review the folder and settings, then try again.",
            )
            self._publish()
            return
        self._switch_projection_root(plan.root_id)
        self._review_plan = plan
        self._state = replace(
            self._state,
            phase="review",
            review=build_reconciliation_review(plan, activation=True),
            status_line="Review setup effects before activating this root.",
        )
        self._publish()

    async def check_migration(self, root_id: str) -> None:
        """Build a current activation review for one paused migrated root."""

        await self.check_root(root_id)
        if self._state.phase == "review" and self._state.review.root_id == root_id:
            self._project_review(activation=True)
            self._state = replace(
                self._state,
                status_line="Review migration before activating this folder.",
            )
            self._publish()

    async def abandon_setup(self) -> None:
        """Release an unpersisted setup review when the user leaves it."""

        root_id = self._state.review.root_id
        if self._state.review.activation and root_id:
            try:
                await self._runtime.abandon_setup(root_id)
            except Exception:
                pass
        self._clear_ephemeral_review()
        self._invalidate_receipt_history(clear_root=True)
        self._review_plan = None

    def set_review_page(self, page: int) -> None:
        """Page the controller-private reviewed plan without copying its IDs."""

        if self._review_plan is None:
            raise RuntimeError("a current review is required")
        self._clear_comparison()
        self._project_review(page=page)
        if self._state.review.stale:
            self._state = replace(
                self._state,
                review=replace(self._state.review, next_action="Check again"),
            )
        self._state = replace(self._state, phase="review")
        self._publish()

    async def apply_reviewed(self) -> None:
        review = self._state.review
        plan = self._review_plan
        if plan is None or not review.root_id or not review.observation_token:
            raise RuntimeError("a current review is required")
        if not review.can_apply:
            raise ValueError(f"review_not_applicable:{review.apply_blocker.value}")
        action_ids = tuple(action.action_id for action in plan.safe_actions)
        selections = self._current_selections()
        receipt_generation = self._start_receipt_request(review.root_id)
        try:
            result = await self._runtime.apply_reviewed(
                review.root_id,
                review.observation_token,
                action_ids,
                selections,
            )
        except ValueError as error:
            if str(error) != "stale_review":
                raise
            if not self._receipt_is_current(review.root_id, receipt_generation):
                return
            self._selections.clear()
            self._clear_comparison()
            self._project_review(stale=True)
            self._state = replace(
                self._state,
                phase="review",
                review=replace(self._state.review, next_action="Check again"),
                status_line="The review is stale. Check again before applying.",
            )
            self._publish()
            return
        except Exception:
            if not self._receipt_is_current(review.root_id, receipt_generation):
                return
            self._selections.clear()
            self._clear_comparison()
            self._project_review(stale=True)
            self._state = replace(
                self._state,
                phase="review",
                review=replace(self._state.review, next_action="Check again"),
                status_line="Apply failed. Review root status, then Check again.",
            )
            self._publish()
            return
        if type(result) is not ConflictApplyResult:
            raise RuntimeError("invalid reviewed apply result")
        if not self._receipt_is_current(review.root_id, receipt_generation):
            return
        try:
            receipts = await self._read_receipts(review.root_id)
        except Exception:
            receipts = ()
        if not self._receipt_is_current(review.root_id, receipt_generation):
            return
        applied = result.safe_completed + result.conflicts_resolved
        if result.partial or result.needs_recovery or result.fresh_plan is None:
            self._selections.clear()
            self._clear_comparison()
            self.refresh_roots(publish=False)
            self._state = replace(
                self._state,
                phase="roots",
                receipts=receipts,
                status_line=f"{applied} applied · recovery needs attention.",
                receipt_line="",
            )
            self._publish()
            return

        self._review_plan = result.fresh_plan
        self._selections.clear()
        self._clear_comparison()
        self._state = replace(
            self._state,
            review=build_reconciliation_review(result.fresh_plan),
            receipts=receipts,
        )
        if result.attention_remains:
            count = result.unresolved_conflicts
            noun = "conflict remains" if count == 1 else "conflicts remain"
            self._state = replace(
                self._state,
                phase="review",
                status_line=f"{applied} applied · {count} {noun}.",
                receipt_line="",
            )
        else:
            self._state = replace(
                self._state,
                phase="receipt",
                status_line=f"{applied} applied · no conflicts remain.",
                receipt_line=f"{applied} applied · durable receipt recorded",
            )
        self.refresh_roots(publish=False)
        self._publish()

    async def sync_now(self, root_id: str) -> None:
        """Run the runtime's existing mutation-free manual reconciliation."""

        self._switch_projection_root(root_id)
        self._selections.clear()
        self._clear_comparison()
        self._state = replace(
            self._state, phase="checking", status_line="Checking changes…"
        )
        self._publish()
        try:
            plan = await self._runtime.request_sync_now(root_id)
        except Exception:
            self._state = replace(
                self._state,
                phase="roots",
                status_line="Manual check failed. Review root status, then try again.",
            )
            self._publish()
            return
        self._review_plan = plan
        self._state = replace(
            self._state,
            phase="review",
            review=build_reconciliation_review(plan),
            status_line="Manual check finished. Review exact effects.",
        )
        self._publish()

    async def resolve_cleanup(self, root_id: str, operation_id: str) -> None:
        """Forward one operation-specific recovery already exposed by the runtime."""

        self._switch_projection_root(root_id)
        try:
            await self._runtime.resolve_cleanup(root_id, operation_id)
        except Exception:
            self._state = replace(
                self._state,
                phase="roots",
                status_line="Recovery needs attention. Review root status, then try again.",
            )
            self._publish()
            return
        self._state = replace(
            self._state,
            phase="roots",
            status_line="Recovery reviewed. Check changes before the next mutation.",
        )
        self.refresh_roots()

    def stage_attention_choice(self, item_id: str, choice: str) -> None:
        """Stage one eligible conflict choice without calling the runtime."""

        row = next(
            (
                row
                for row in self._state.review.rows
                if row.item_id == item_id
                and row.conflict_eligible
                and choice in row.choices
            ),
            None,
        )
        typed_choice = _CHOICES_BY_LABEL.get(choice)
        if row is None or typed_choice is None:
            raise ValueError("unknown attention choice")
        token = self._state.review.observation_token
        if not token:
            raise RuntimeError("a current review is required")
        self._selections[(token, item_id)] = typed_choice
        self._project_review()
        self._state = replace(
            self._state,
            status_line="Choice staged. No changes yet.",
        )
        self._publish()

    async def show_conflict_comparison(self, binding_id: str) -> None:
        """Load one bounded comparison and publish only for current provenance."""

        review = self._state.review
        if not any(
            row.item_id == binding_id and row.conflict_eligible for row in review.rows
        ):
            raise ValueError("comparison binding is not on the current review page")
        self._clear_comparison()
        self._expanded_binding_id = binding_id
        generation = self._comparison_generation
        root_id = review.root_id
        token = review.observation_token
        try:
            comparison = await self._runtime.compare_conflict(
                root_id,
                token,
                binding_id,
            )
        except ValueError as error:
            if str(error) != "stale_review" or not self._comparison_is_current(
                generation, root_id, token, binding_id
            ):
                return
            self._selections.clear()
            self._clear_comparison()
            self._project_review(stale=True)
            self._state = replace(
                self._state,
                phase="review",
                review=replace(self._state.review, next_action="Check again"),
                status_line="The review is stale. Check again before applying.",
            )
            self._publish()
            return
        except Exception:
            if not self._comparison_is_current(generation, root_id, token, binding_id):
                return
            self._clear_comparison()
            self._state = replace(
                self._state,
                status_line="Comparison unavailable. Check again, then try once more.",
            )
            self._publish()
            return
        if not self._comparison_is_current(generation, root_id, token, binding_id):
            return
        self._state = replace(self._state, comparison=comparison)
        self._publish()

    def _comparison_is_current(
        self,
        generation: int,
        root_id: str,
        token: str,
        binding_id: str,
    ) -> bool:
        return (
            generation == self._comparison_generation
            and self._expanded_binding_id == binding_id
            and self._state.phase == "review"
            and self._state.review.root_id == root_id
            and self._state.review.observation_token == token
        )

    def return_to_conflict_choices(self) -> None:
        """Collapse the retained comparison without changing staged choices."""

        self._clear_comparison()
        self._publish()

    async def _read_receipts(self, root_id: str) -> tuple[LastingSyncReceiptRow, ...]:
        rows = await self._runtime.active_conflict_receipts(root_id)
        if type(rows) is not tuple or any(
            type(row) is not RuntimeConflictReceipt for row in rows
        ):
            raise RuntimeError("invalid conflict receipt projection")
        return tuple(
            LastingSyncReceiptRow(
                row.operation_id,
                row.item_label,
                row.choice,
                row.state,
                row.undo_available,
                row.undo_reason,
            )
            for row in rows
        )

    async def refresh_conflict_receipts(self, root_id: str) -> None:
        """Refresh current-runtime receipts from fresh bounded projections."""

        generation = self._start_receipt_request(root_id)
        try:
            receipts = await self._read_receipts(root_id)
        except Exception:
            if not self._receipt_is_current(root_id, generation):
                return
            self._state = replace(
                self._state,
                status_line="Resolution receipts are unavailable. Review history instead.",
            )
            self._publish()
            return
        if not self._receipt_is_current(root_id, generation):
            return
        self._state = replace(self._state, receipts=receipts)
        self._publish()

    async def dismiss_conflict_receipt(self, root_id: str, operation_id: str) -> None:
        """Dismiss one process-local receipt and refresh its bounded projection."""

        generation = self._start_receipt_request(root_id)
        try:
            self._runtime.dismiss_conflict_receipt(root_id, operation_id)
            receipts = await self._read_receipts(root_id)
        except Exception:
            if not self._receipt_is_current(root_id, generation):
                return
            self._state = replace(
                self._state,
                status_line="Receipt dismissal is unavailable. Try again.",
            )
            self._publish()
            return
        if not self._receipt_is_current(root_id, generation):
            return
        self._state = replace(self._state, receipts=receipts)
        self._publish()

    async def undo_conflict_resolution(self, root_id: str, operation_id: str) -> None:
        """Run one durable linked Undo and refresh the remaining receipts."""

        receipt_generation = self._start_receipt_request(root_id)
        history_page = (
            self._state.history.page
            if self._state.phase == "history" and self._state.history.root_id == root_id
            else None
        )
        history_generation = (
            self._start_history_request(root_id, history_page)
            if history_page is not None
            else None
        )
        try:
            result = await self._runtime.undo_resolution(root_id, operation_id)
            if type(result) is not NotesSyncExecutionResult:
                raise RuntimeError("invalid Undo result")
        except Exception:
            if not self._receipt_is_current(root_id, receipt_generation):
                return
            self._state = replace(
                self._state,
                status_line="Undo needs attention. Review Resolution history.",
            )
            self._publish()
            return
        if not self._receipt_is_current(root_id, receipt_generation):
            return
        if (
            result.state is not NotesSyncOperationState.COMPLETED
            or result.recovery_required
        ):
            self.refresh_roots(publish=False)
            self._state = replace(
                self._state,
                phase="roots",
                status_line="Undo recovery needs attention. Review Resolution history.",
            )
            self._publish()
            return
        try:
            receipts = await self._read_receipts(root_id)
        except Exception:
            if not self._receipt_is_current(root_id, receipt_generation):
                return
            self._state = replace(
                self._state,
                status_line="Undo finished, but its fresh projection is unavailable.",
            )
            self._publish()
            return
        if not self._receipt_is_current(root_id, receipt_generation):
            return
        history_is_current = history_page is not None and (
            history_generation is not None
            and self._history_is_current(root_id, history_page, history_generation)
        )
        if history_page is not None and not history_is_current:
            self.refresh_roots(publish=False)
            self._state = replace(self._state, receipts=receipts)
            self._publish()
            return
        try:
            history = (
                await self._read_history(root_id, history_page)
                if history_page is not None
                else None
            )
        except Exception:
            if not self._receipt_is_current(root_id, receipt_generation):
                return
            if history_page is not None and (
                history_generation is None
                or not self._history_is_current(
                    root_id, history_page, history_generation
                )
            ):
                self.refresh_roots(publish=False)
                self._state = replace(self._state, receipts=receipts)
                self._publish()
                return
            self._state = replace(
                self._state,
                receipts=receipts,
                status_line="Undo finished, but its fresh projection is unavailable.",
            )
            self._publish()
            return
        if not self._receipt_is_current(root_id, receipt_generation):
            return
        if history_page is not None and (
            history_generation is None
            or not self._history_is_current(root_id, history_page, history_generation)
        ):
            self.refresh_roots(publish=False)
            self._state = replace(self._state, receipts=receipts)
            self._publish()
            return
        self.refresh_roots(publish=False)
        self._state = replace(
            self._state,
            receipts=receipts,
            history=self._state.history if history is None else history,
            status_line="Undo finished. Check changes before applying again.",
        )
        self._publish()

    async def _read_history(self, root_id: str, page: int) -> LastingSyncHistory:
        offset = validate_lasting_sync_history_page(page)
        rows = await self._runtime.resolution_history(
            root_id,
            limit=LASTING_SYNC_HISTORY_PAGE_SIZE,
            offset=offset,
        )
        if type(rows) is not tuple or any(
            type(row) is not RuntimeConflictHistoryRow for row in rows
        ):
            raise RuntimeError("invalid resolution history projection")
        projected = tuple(
            LastingSyncHistoryRow(
                row.operation_id,
                row.item_label,
                row.choice,
                row.state,
                row.completed_at,
                row.updated_at,
                row.undo_available,
                row.undo_reason,
            )
            for row in rows
        )
        return LastingSyncHistory(
            root_id,
            projected,
            page,
            len(projected) == LASTING_SYNC_HISTORY_PAGE_SIZE,
        )

    async def show_resolution_history(self, root_id: str, *, page: int = 1) -> None:
        """Load one fresh bounded durable resolution-history page."""

        validate_lasting_sync_history_page(page)
        generation = self._start_history_request(root_id, page)
        self._clear_comparison()
        try:
            history = await self._read_history(root_id, page)
            status = "Resolution history loaded."
        except Exception:
            if not self._history_is_current(root_id, page, generation):
                return
            history = LastingSyncHistory(root_id, (), page, False, True)
            status = "Resolution history is unavailable. Try again."
        if not self._history_is_current(root_id, page, generation):
            return
        self._state = replace(
            self._state,
            phase="history",
            history=history,
            status_line=status,
        )
        self._publish()

    def stage_root_action(self, root_id: str, action: str) -> None:
        """Keep controls without a completed runtime seam explicit and inert."""

        known = {root.root_id for root in self._all_roots}
        if root_id not in known or action not in {
            "review",
            "recover",
            "retarget",
            "disconnect",
        }:
            raise ValueError("unknown root action")
        self._switch_projection_root(root_id)
        self._state = replace(
            self._state,
            phase="roots",
            status_line=(
                f"{action.title()} is unavailable in this release; "
                "no files or notes changed."
            ),
        )
        self._publish()

    async def activate_root(self, root_id: str) -> bool:
        self._switch_projection_root(root_id)
        if not self._state.lasting_available:
            self._state = replace(
                self._state,
                status_line="Lasting folder sync is unavailable until the reviewed cutover.",
            )
            self._publish()
            return False
        self._state = replace(
            self._state,
            phase="activating",
            status_line="Activating the reviewed sync root…",
        )
        self._publish()
        try:
            result = await self._runtime.activate_root(
                root_id, self._state.review.observation_token
            )
        except Exception:
            self._state = replace(
                self._state,
                phase="review",
                status_line="Activation failed. Review settings, then check again.",
            )
            self._publish()
            return False
        accepted = bool(getattr(result, "accepted", False))
        applied_count = getattr(result, "applied_count", 0)
        recovery = not accepted and getattr(result, "status", "") in {
            "failed",
            "partial",
            "needs_attention",
        }
        self._state = replace(
            self._state,
            phase="receipt" if accepted else "roots" if recovery else "review",
            status_line=(
                "Sync root activated."
                if accepted
                else "Activation needs attention. Open root recovery."
                if recovery
                else "Activation needs attention. Review settings, then check again."
            ),
            receipt_line=(
                f"{applied_count} applied · durable receipt recorded"
                if accepted
                else ""
            ),
        )
        self.refresh_roots()
        return accepted

    async def pause_root(self, root_id: str) -> None:
        self._switch_projection_root(root_id)
        await self._run_root_control(self._runtime.pause_root(root_id))

    async def resume_root(self, root_id: str) -> None:
        self._switch_projection_root(root_id)
        await self._run_root_control(self._runtime.resume_root(root_id))

    async def retarget_root(self, root_id: str, target: str) -> None:
        self._switch_projection_root(root_id)
        if not await self._run_root_control(
            self._runtime.retarget_root(root_id, target), refresh=False
        ):
            self._state = replace(
                self._state,
                status_line=(
                    "Retarget needs review and never infers deletions from the new folder."
                ),
            )
            self._publish()
            return
        self._state = replace(
            self._state,
            status_line="Retarget review never infers deletions from the new folder.",
        )
        self.refresh_roots()

    async def disconnect_root(
        self, root_id: str, *, keep_folder_organization: bool
    ) -> None:
        self._switch_projection_root(root_id)
        if not await self._run_root_control(
            self._runtime.disconnect_root(root_id, keep_folder_organization),
            refresh=False,
        ):
            self._state = replace(
                self._state,
                status_line=(
                    "Disconnect needs review; it never deletes files or notes. "
                    "Only this root's managed organization changes."
                ),
            )
            self._publish()
            return
        self._state = replace(
            self._state,
            status_line=(
                "Disconnect requested; it never deletes files or notes. "
                "Only this root's managed organization changes."
            ),
        )
        self.refresh_roots()

    async def _run_root_control(
        self,
        operation: Awaitable[NotesSyncControlResult],
        *,
        refresh: bool = True,
    ) -> bool:
        """Settle one control without exposing private exception text."""

        try:
            result = await operation
        except Exception:
            self._state = replace(
                self._state,
                phase="roots",
                status_line="Control failed. Review root status, then try its next action.",
            )
            self._publish()
            return False
        if getattr(result, "accepted", True) is False:
            self._state = replace(
                self._state,
                phase="roots",
                status_line="Action needs attention. Review settings, then Check changes.",
            )
            self.refresh_roots()
            return False
        if refresh:
            self.refresh_roots()
        return True


__all__ = [
    "InertLastingSyncRuntime",
    "LastingSyncRuntimePort",
    "LibraryNotesSyncController",
]
