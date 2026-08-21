"""Focused controller for inert lasting Notes sync presentation flows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Never, Protocol

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncRootRow,
    LibraryNotesLastingSyncSnapshot,
    build_reconciliation_review,
    initial_lasting_sync_snapshot,
    set_setup_value,
)
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationPlan
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncControlResult,
    NotesSyncRuntimeSnapshot,
)


class LastingSyncRuntimePort(Protocol):
    """Only the public runtime operations the Library UI is allowed to call.

    The UI plan's initial list omitted the TASK-19009 public manual-check alias
    and operation-specific cleanup method; both are included here so the root
    canvas uses the existing runtime instead of inventing parallel seams.
    """

    def snapshot(self) -> NotesSyncRuntimeSnapshot: ...

    async def check_root(self, root_id: str) -> ReconciliationPlan: ...

    async def request_sync_now(self, root_id: str) -> ReconciliationPlan: ...

    async def apply_reviewed(
        self, root_id: str, observation_token: str, action_ids: tuple[str, ...]
    ) -> tuple[object, ...]: ...

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

    async def request_sync_now(self, root_id: str) -> ReconciliationPlan:
        return await self._blocked()

    async def apply_reviewed(
        self, root_id: str, observation_token: str, action_ids: tuple[str, ...]
    ) -> tuple[object, ...]:
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
    "resolve_cleanup": "Resolve recovery",
    "wait": "Wait",
    "none": "No action",
    "apply_reviewed": "Apply reviewed",
    "finish_upgrade": "Finish upgrade",
}
_ROOT_PAGE_SIZE = 20


class LibraryNotesSyncController:
    """Own chooser/review tasks while storage and Textual remain late-bound."""

    def __init__(
        self,
        *,
        runtime: LastingSyncRuntimePort,
        import_controller: _ImportOncePort,
        lasting_available: bool,
        publish_snapshot: Callable[[LibraryNotesLastingSyncSnapshot], None]
        | None = None,
    ) -> None:
        if type(lasting_available) is not bool:
            raise TypeError("lasting_available must be a boolean")
        self._runtime = runtime
        self._import_controller = import_controller
        self._publish_snapshot = publish_snapshot
        self._review_plan: ReconciliationPlan | None = None
        self._state = initial_lasting_sync_snapshot(lasting_available=lasting_available)
        self._all_roots: tuple[LastingSyncRootRow, ...] = ()
        self.refresh_roots()

    @property
    def snapshot(self) -> LibraryNotesLastingSyncSnapshot:
        return self._state

    def _publish(self) -> None:
        if callable(self._publish_snapshot):
            self._publish_snapshot(self._state)

    def refresh_roots(self) -> None:
        """Refresh path-free root rows from the public runtime projection."""

        runtime = self._runtime.snapshot()
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
            roots=self._all_roots[start : start + _ROOT_PAGE_SIZE],
            root_page=page,
            root_page_count=page_count,
        )
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
        if not self._state.lasting_available:
            self._state = replace(
                self._state,
                status_line="Lasting folder sync is unavailable until the reviewed cutover.",
            )
            self._publish()
            return "choose"
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
        self._review_plan = plan
        self._state = replace(
            self._state,
            phase="review",
            review=build_reconciliation_review(plan),
            status_line="Review the mutation-free check before applying changes.",
        )
        self._publish()

    async def check_setup(self, root_id: str) -> None:
        """Run the same mutation-free check and label it as activation review."""

        await self.check_root(root_id)
        if self._state.phase == "review" and self._state.review.root_id:
            self._state = replace(
                self._state,
                review=replace(self._state.review, activation=True),
                status_line="Review setup effects before activating this root.",
            )
            self._publish()

    def set_review_page(self, page: int) -> None:
        """Page the controller-private reviewed plan without copying its IDs."""

        if self._review_plan is None:
            raise RuntimeError("a current review is required")
        current = self._state.review
        projected = build_reconciliation_review(self._review_plan, page=page)
        review = replace(
            projected,
            activation=current.activation,
            stale=current.stale,
            next_action=current.next_action if current.stale else projected.next_action,
        )
        self._state = replace(self._state, phase="review", review=review)
        self._publish()

    async def apply_reviewed(self) -> None:
        review = self._state.review
        plan = self._review_plan
        if plan is None or not review.root_id or not review.observation_token:
            raise RuntimeError("a current review is required")
        action_ids = tuple(action.action_id for action in plan.safe_actions)
        try:
            results = await self._runtime.apply_reviewed(
                review.root_id, review.observation_token, action_ids
            )
        except ValueError as error:
            if str(error) != "stale_review":
                raise
            self._state = replace(
                self._state,
                phase="review",
                review=replace(review, stale=True, next_action="Check again"),
                status_line="The review is stale. Check again before applying.",
            )
            self._publish()
            return
        except Exception:
            self._state = replace(
                self._state,
                phase="review",
                review=replace(review, stale=True, next_action="Check again"),
                status_line="Apply failed. Review root status, then Check again.",
            )
            self._publish()
            return
        self._state = replace(
            self._state,
            phase="receipt",
            status_line="Reviewed sync finished.",
            receipt_line=f"{len(results)} applied · durable receipt recorded",
        )
        self.refresh_roots()

    async def sync_now(self, root_id: str) -> None:
        """Run the runtime's existing mutation-free manual reconciliation."""

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
        """Acknowledge an attention choice without inventing an executor seam."""

        row = next(
            (
                row
                for row in self._state.review.rows
                if row.item_id == item_id and choice in row.choices
            ),
            None,
        )
        if row is None:
            raise ValueError("unknown attention choice")
        self._state = replace(
            self._state,
            status_line=(
                "Choice reviewed. Applying attention resolutions remains unavailable "
                "until the lasting-sync cutover."
            ),
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
        self._state = replace(
            self._state,
            phase="roots",
            status_line=(
                f"{action.title()} requires a reviewed flow that remains unavailable "
                "until the lasting-sync cutover; no files or notes changed."
            ),
        )
        self._publish()

    async def activate_root(self, root_id: str) -> bool:
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
            result = await self._runtime.activate_root(root_id, self._state.setup)
        except Exception:
            self._state = replace(
                self._state,
                phase="review",
                status_line="Activation failed. Review settings, then check again.",
            )
            self._publish()
            return False
        accepted = bool(getattr(result, "accepted", False))
        self._state = replace(
            self._state,
            phase="receipt" if accepted else "review",
            status_line=(
                "Sync root activated."
                if accepted
                else "Activation needs attention. Review settings, then check again."
            ),
        )
        self.refresh_roots()
        return accepted

    async def pause_root(self, root_id: str) -> None:
        await self._run_root_control(self._runtime.pause_root(root_id))

    async def resume_root(self, root_id: str) -> None:
        await self._run_root_control(self._runtime.resume_root(root_id))

    async def retarget_root(self, root_id: str, target: str) -> None:
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
