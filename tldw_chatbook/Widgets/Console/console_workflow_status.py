"""Selected workflow history, phase review and bounded Console status projection."""

from __future__ import annotations

import asyncio
from functools import partial
from uuid import uuid4

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Agents.workflow_models import (
    WorkflowExecutionRecovery,
    WorkflowQualityReview,
)
from tldw_chatbook.UI.Console_Modules.workflows import (
    WORKFLOW_VIEW_ERRORS,
    WorkflowViewReads,
    workflow_view_error,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


def status_label(state: str) -> str:
    return {
        "awaiting_phase_review": "Phase approval required",
        "awaiting_result_review": "Result review required",
        "paused": "Resume required",
        "awaiting_resume": "Resume required",
        "recovery_required": "Recovery required",
    }.get(state, state.replace("_", " ").capitalize())


class ConsoleWorkflowHistory(SafeModalDismissMixin, ModalScreen[None]):
    SAFE_MODAL_CONTENT = "#workflow-history"
    BINDINGS = (("escape", "request_safe_cancel", "Close"),)
    AUTO_FOCUS = "#workflow-close"

    def __init__(self, coordinator, conversation_id, open_workflow, new_workflow):
        super().__init__()
        self.coordinator = coordinator
        self.reads = WorkflowViewReads()
        self.conversation_id = conversation_id
        self.open_workflow = open_workflow
        self.new_workflow = new_workflow
        self._page_offset = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="workflow-history", classes="workflow-dialog"):
            yield Static("Workflow history", classes="workflow-title")
            yield Static("", id="workflow-page", markup=False)
            yield Select([], prompt="Choose a saved workflow", id="workflow-saved")
            with Horizontal(classes="workflow-actions"):
                yield Button("Previous", id="workflow-prev", disabled=True)
                yield Button("Next", id="workflow-next", disabled=True)
                yield Button("Open", id="workflow-open", disabled=True)
                yield Button("Retry", id="workflow-retry")
            with Horizontal(classes="workflow-actions"):
                yield Button("Close", id="workflow-close")
                yield Button("New workflow", id="workflow-new")

    def on_mount(self):
        self.query_one("#workflow-retry").display = False
        self.run_worker(self._load)

    async def on_unmount(self):
        await self.reads.drain()

    async def _load(self):
        try:
            page = self._page_offset
            try:
                service = self.coordinator.service
                rows = await self.reads.read(
                    service.call,
                    service.list_history,
                    self.conversation_id,
                    offset=page,
                )
            except WORKFLOW_VIEW_ERRORS as error:
                message = workflow_view_error("History", error)
                if (
                    not self.reads.closed
                    and self.query("#workflow-page")
                    and page == self._page_offset
                ):
                    self.query_one("#workflow-saved", Select).set_options([])
                    self.query_one("#workflow-open", Button).disabled = True
                    self.query_one("#workflow-next", Button).disabled = True
                    self.query_one("#workflow-prev", Button).disabled = page == 0
                    self.query_one("#workflow-retry").display = True
                    self.query_one("#workflow-page").update(
                        message + " Choose Retry to load this page."
                    )
                return
            if (
                self.reads.closed
                or not self.is_mounted
                or not self.query("#workflow-saved")
                or page != self._page_offset
            ):
                return
            self.query_one("#workflow-retry").display = False
            select = self.query_one("#workflow-saved", Select)
            select.set_options(
                [
                    (Text(f"{status_label(row['status'])} · {row['id']}"), row["id"])
                    for row in rows
                ]
            )
            if rows:
                select.value = rows[0]["id"]
            self.query_one("#workflow-open", Button).disabled = not rows
            self.query_one("#workflow-prev", Button).disabled = page == 0
            self.query_one("#workflow-next", Button).disabled = len(rows) < 50
            self.query_one("#workflow-page").update(
                f"Page {page // 50 + 1} · {len(rows)} saved"
                if rows
                else "No workflows on this page."
            )
        except NoMatches:
            return

    async def on_button_pressed(self, event: Button.Pressed):
        try:
            control = event.button.id
            if control in {"workflow-prev", "workflow-next"}:
                self._page_offset += -50 if control == "workflow-prev" else 50
                await self._load()
            elif control == "workflow-retry":
                await self._load()
            elif control == "workflow-close":
                await self.request_safe_cancel(source="button")
            elif control == "workflow-new" and self.dismiss_safe_once(None):
                self.new_workflow(self.conversation_id)
            elif control == "workflow-open":
                selected = self.query_one("#workflow-saved", Select).value
                if isinstance(selected, str) and self.dismiss_safe_once(None):
                    self.open_workflow(self.conversation_id, selected)
        except NoMatches:
            return


class ConsoleWorkflowStatus(SafeModalDismissMixin, ModalScreen[None]):
    """Present the exact chat/workflow and the decision prepared against its revision."""

    SAFE_MODAL_CONTENT = ".workflow-dialog"
    BINDINGS = (("escape", "request_safe_cancel", "Close"),)
    AUTO_FOCUS = "#workflow-close"

    def __init__(
        self,
        coordinator,
        conversation_id,
        workflow_id,
        *,
        open_changes,
        open_fleet,
        pending_approvals,
        open_approval,
    ):
        super().__init__()
        self.coordinator = coordinator
        self.reads = WorkflowViewReads()
        self.workflow_id, self.conversation_id = workflow_id, conversation_id
        self.open_changes, self.open_fleet = open_changes, open_fleet
        self.pending_approvals, self.open_approval = pending_approvals, open_approval
        self._tasks = ()
        self._action_notice = ""
        self.snapshot = self.decision = self.checkpoint = None
        self._unsubscribe = None
        self._reading = self._read_again = self._sending = False
        self._command_ids = {}
        self._result_page = 0
        self._visible_results = ()
        self._window_offset = 0
        self.windows = ()
        self._window_targets = {}

    def compose(self) -> ComposeResult:
        with Vertical(classes="workflow-dialog"):
            yield Static(
                "Workflow", id="workflow-title", classes="workflow-title", markup=False
            )
            yield Static("", id="workflow-state", markup=False)
            with VerticalScroll(classes="workflow-scroll"):
                yield Select([], prompt="Select a saved result", id="workflow-results")
                with Horizontal(id="workflow-result-pages", classes="workflow-actions"):
                    yield Button("Previous results", id="workflow-result-prev")
                    yield Button("Next results", id="workflow-result-next")
                yield Static("", id="workflow-evidence", markup=False)
                yield Static("", id="workflow-plan-review", markup=False)
                yield Static("", id="workflow-accounting", markup=False)
                yield Static("", id="workflow-capture", markup=False)
                with Horizontal(
                    id="workflow-capture-pages", classes="workflow-actions"
                ):
                    yield Button("Newer windows", id="workflow-window-newer")
                    yield Button("Older windows", id="workflow-window-older")
                yield Select(
                    [], prompt="Choose a Change Review window", id="workflow-windows"
                )
                yield Button("Open Changes", id="workflow-changes", disabled=True)
                yield Static("", id="workflow-tasks", markup=False)
                yield Select([], prompt="Choose a task to inspect", id="workflow-task")
                yield Button("Fleet", id="workflow-fleet")
                yield Button("Review pending tool decision", id="workflow-pending")
                yield Input(
                    placeholder="Instruction for a revised plan",
                    id="workflow-instruction",
                )
                with Horizontal(
                    id="workflow-history-actions", classes="workflow-actions"
                ):
                    yield Button("Review prior effects", id="workflow-recovery")
                    yield Button("Remove stored results", id="workflow-remove")
            yield Static("Loading workflow…", id="workflow-notice", markup=False)
            with Horizontal(classes="workflow-actions"):
                yield Button("Close", id="workflow-close")
                yield Button("Refresh", id="workflow-refresh")
                yield Button("Pause", id="workflow-pause", disabled=True)
                yield Button("Resume", id="workflow-resume", disabled=True)
                yield Button("Stop", id="workflow-stop", disabled=True)
            with Horizontal(classes="workflow-actions"):
                yield Button(
                    "Approve plan and start",
                    id="workflow-approve",
                    variant="primary",
                    disabled=True,
                )
                yield Button(
                    "Accept result",
                    id="workflow-accept",
                    variant="primary",
                    disabled=True,
                )
                yield Button(
                    "Revise remaining plan", id="workflow-revise", disabled=True
                )

    def on_mount(self):
        self._unsubscribe = self.coordinator.subscribe(self._published)
        self.call_after_refresh(self.request_refresh)

    async def on_unmount(self):
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        await self.reads.drain()

    def on_screen_resume(self):
        if self._unsubscribe is not None:
            self.call_later(self.request_refresh)

    def _published(self, view):
        if view.id == self.workflow_id and view.conversation_id == self.conversation_id:
            self.call_later(self.request_refresh)

    @property
    def _can_render(self):
        return (
            self._unsubscribe is not None
            and self.is_mounted
            and bool(self.query("#workflow-notice"))
        )

    def request_refresh(self):
        if not self._can_render:
            return
        self._read_again = True
        if not self._reading:
            self._reading = True
            try:
                self.run_worker(self._refresh)
            except RuntimeError as error:
                self._reading = False
                self._disable_commands()
                self.query_one("#workflow-notice").update(
                    workflow_view_error("Review", error) + ". Refresh to retry."
                )

    async def _refresh(self):
        try:
            service = self.coordinator.service
            try:
                while self._read_again and self._unsubscribe:
                    self._read_again = False
                    try:
                        full = await self.reads.read(
                            self.coordinator.get, self.workflow_id
                        )
                        if full.request.conversation_id != self.conversation_id:
                            raise PermissionError("Workflow belongs to another chat.")
                        view = await self.reads.read(
                            service.call, service.runtime_snapshot, self.workflow_id
                        )
                        if (view.control_revision, view.progress_cursor) != (
                            full.control_revision,
                            full.progress_cursor,
                        ):
                            raise ValueError("workflow_snapshot_superseded")
                        decision, repair_required = await self._review(full, view)
                        windows = await self.reads.read(
                            service.call,
                            service.change_windows,
                            self.workflow_id,
                            offset=self._window_offset,
                        )
                        active_seconds = await self.reads.read(
                            service.call, service.active_seconds, self.workflow_id
                        )
                        if repair_required:
                            current = await self.reads.read(
                                self.coordinator.get, self.workflow_id
                            )
                            if (
                                current.status,
                                current.control_revision,
                                current.progress_cursor,
                            ) != (
                                full.status,
                                full.control_revision,
                                full.progress_cursor,
                            ) or current.accounting.uncertain:
                                raise ValueError("workflow_snapshot_superseded")
                        if self._can_render:
                            self.active_seconds = active_seconds
                            self.windows = windows
                            self.snapshot, self.decision = full, decision
                            self._render_state(view, repair_required)
                    except WORKFLOW_VIEW_ERRORS as error:
                        message = workflow_view_error("Review", error)
                        if self._can_render:
                            self.decision = None
                            self._disable_commands()
                            self.query_one("#workflow-notice").update(
                                message + ". Refresh to read the current workflow."
                            )
            finally:
                self._reading = False
        except NoMatches:
            self.decision = None
            return

    async def _review(self, full, view):
        """Read the current decision or the native, drained repair disposition."""
        if (
            not full.plans
            or full.status
            not in {"awaiting_result_review", "awaiting_phase_review", "paused"}
            or full.accounting.uncertain
        ):
            return None, False
        service = self.coordinator.service
        phases = full.plans[-1].phases
        phase = next((p for p in phases if p.id == view.phase_id), phases[0])
        tasks = {task.task_id: task.status for task in view.tasks}
        states = [tasks.get(task.id) for task in phase.tasks]
        failed = view.phase_id == phase.id and "failed" in states
        if full.status == "paused" and not failed:
            return None, False
        operation = service.prepare_decision
        if full.status == "awaiting_result_review":
            operation = service.prepare_quality_review
        elif failed:
            operation = service.prepare_remaining_decision
        elif view.phase_id == phase.id:
            next_index = phases.index(phase) + 1
            if all(s == "completed" for s in states) and next_index < len(phases):
                phase = phases[next_index]
            elif "completed" in states:
                operation = service.prepare_remaining_decision
        try:
            decision = await self.reads.read(
                service.call, operation, self.workflow_id, phase_id=phase.id
            )
        except ValueError as error:
            if operation != service.prepare_remaining_decision or error.args != (
                "remaining_requires_repair",
            ):
                raise
            # Only this native disposition follows its original clean-checkpoint test.
            return None, True
        if decision.evidence is not None and decision.evidence.kind == "recovery":
            raise ValueError(f"Recovery required · {decision.evidence.reason}")
        if decision.control_revision != full.control_revision:
            raise ValueError("workflow_snapshot_superseded")
        return decision, False

    def _disable_commands(self):
        for name in (
            "approve",
            "accept",
            "revise",
            "resume",
            "pause",
            "stop",
            "recovery",
            "remove",
        ):
            self.query_one("#workflow-" + name, Button).disabled = True

    def _render_state(self, view, repair_required):
        full, decision = self.snapshot, self.decision
        self.query_one("#workflow-title").update(full.request.objective)
        label = "Plan repair required" if repair_required else status_label(full.status)
        heading = f"{label} · control revision {full.control_revision}"
        if decision is not None:
            heading = f"{label} · phase {decision.phase_id} · plan {decision.plan_revision} · control revision {decision.control_revision}"
        elif view.phase_id:
            heading += f" · last attempt {view.phase_id}"
        self.query_one("#workflow-state").update(heading)
        info = [
            f"Folder  {full.request.binding.locator} ({full.request.binding.access})",
            "Current uncommitted files are workflow input. Stop preserves edits.",
            "Targets  "
            + ", ".join(f"{t.provider}/{t.model}" for t in full.request.targets),
        ]
        if "\n" in full.request.objective or len(full.request.objective) > max(
            1, self.query_one("#workflow-title").size.width
        ):
            info.insert(0, full.request.objective)
        if full.plans:
            plan = full.plans[-1]
            info.append(
                f"Plan {len(full.plans)} · phases {len(plan.phases)} · outputs: "
                + ", ".join(plan.deliverables)
            )
            for phase in plan.phases:
                info.append(f"\nPhase {phase.id}")
                for task in phase.tasks:
                    info.extend(
                        [
                            f"{task.id} · {task.role} · {task.target.provider}/{task.target.model}",
                            task.instructions,
                            f"  Mode  {task.execution_mode} · Tools  "
                            + (", ".join(task.allowed_tools) or "none"),
                            "  Scope  "
                            + (
                                ", ".join(
                                    f"{f.access} {f.path}{'/**' if f.subtree else ''}"
                                    for f in task.footprint
                                )
                                or "no declared paths"
                            ),
                            "  Depends on  "
                            + (", ".join(task.dependencies) or "none")
                            + " · Inputs  "
                            + (", ".join(task.inputs) or "none")
                            + " · Outputs  "
                            + (", ".join(task.outputs) or "none")
                            + " · Checks  "
                            + (", ".join(task.expected_checks) or "none"),
                        ]
                    )
                    for check in task.verifiers:
                        info.append(
                            f"  Check {check.id} · {check.executor_tool_id} {check.verifier_path} "
                            f"{' '.join(check.arguments)} · expected exit {check.expected_exit_code}"
                        )
        self.query_one("#workflow-plan-review").update("\n".join(info))
        spent = full.accounting
        active, clock_reason = self.active_seconds
        active_label = (
            f"{active:.1f}s spent · {max(0, spent.limits.wall_seconds - active):.1f}s remaining"
            if active is not None
            else f"unavailable ({clock_reason.replace('_', ' ')})"
        )
        self.query_one("#workflow-accounting").update(
            "\nLimits · spent / remaining\n"
            + "   ".join(
                f"{k.replace('_', ' ')} {used}/{spent.available.get(k, 0)}"
                for k, used in spent.used.items()
            )
            + f"\nActive time  {active_label} (limit {spent.limits.wall_seconds:g}s) · output limit {spent.limits.output_tokens} tokens/call"
        )
        self.query_one("#workflow-tasks").update(
            "\n".join(f"{t.task_id} · {t.status}" for t in view.tasks)
        )
        tasks = tuple((t.task_id, t.run_id) for t in view.tasks if t.run_id)
        task_control = self.query_one("#workflow-task", Select)
        if tasks != self._tasks:
            self._tasks = tasks
            task_control.set_options([(Text(label), run_id) for label, run_id in tasks])
            if tasks:
                task_control.value = tasks[0][1]
        task_control.display = bool(tasks)
        self.query_one("#workflow-fleet").display = bool(tasks) or full.status in {
            "running",
            "pausing",
        }
        self.query_one("#workflow-pending").display = bool(
            self.pending_approvals(self.conversation_id)
        )
        self.query_one("#workflow-capture").update(
            "Phase Change Review\n"
            + (
                "\n".join(
                    f"{w['phase_id']} · window {w['ordinal']} · {w['state']} · baseline {w['baseline']} · ending {w['ending']}"
                    for w in self.windows
                )
                or "No phase capture recorded."
            )
            + "\nExisting capture consent and folder exclusions apply."
        )
        self.query_one("#workflow-capture-pages").display = (
            self._window_offset > 0 or len(self.windows) == 50
        )
        self.query_one("#workflow-window-newer").display = self._window_offset > 0
        self.query_one("#workflow-window-older").display = len(self.windows) == 50
        self._render_results()
        self._disable_commands()
        review = full.status in {"awaiting_phase_review", "awaiting_result_review"}
        disabled = (
            self._sending
            or full.accounting.uncertain
            or full.status == "recovery_required"
        )
        quality = decision is not None and decision.action == "complete"
        first = (
            full.plans
            and decision
            and decision.phase_id == full.plans[-1].phases[0].id
            and not full.results
        )
        approve = self.query_one("#workflow-approve", Button)
        approve.label = (
            "Approve plan and start"
            if first
            else "Approve remaining tasks"
            if decision and decision.reason == "remaining_ready"
            else "Approve next phase"
        )
        self.query_one("#workflow-accept", Button).label = (
            "Accept phase result"
            if quality and decision.phase_id != full.plans[-1].phases[-1].id
            else "Accept result"
        )
        allowed = {
            "approve": review and not quality and not repair_required,
            "accept": review and quality,
            "revise": review or repair_required,
            "resume": full.status == "paused",
            "pause": full.status in {"running", "planning"},
            "stop": full.status
            in {
                "planning",
                "running",
                "pausing",
                "paused",
                "awaiting_plan",
                "awaiting_phase_review",
                "awaiting_result_review",
            },
        }
        for name, applicable in allowed.items():
            btn = self.query_one("#workflow-" + name, Button)
            # Keep the same physical cells when a phase changes; a second click must never hit Pause or Accept.
            btn.visible = applicable
            btn.disabled = (
                disabled
                or not applicable
                or (
                    name in {"approve", "accept"}
                    and not (
                        decision
                        and isinstance(decision.evidence, WorkflowQualityReview)
                        and decision.evidence.ready
                    )
                )
            )
        self.query_one("#workflow-instruction").display = review or repair_required
        history_actions = {
            "recovery": full.status == "recovery_required"
            or (spent.uncertain and full.status in {"closed", "stopped", "completed"}),
            "remove": full.status in {"closed", "stopped", "completed"},
        }
        for name, applicable in history_actions.items():
            button = self.query_one("#workflow-" + name, Button)
            button.visible = applicable
            button.disabled = self._sending or not applicable
        self.query_one("#workflow-history-actions").display = any(
            history_actions.values()
        )
        reason = (
            "Review prior effects to close interrupted history. Active work and cleanup remain protected."
            if full.status == "recovery_required"
            else "Accounting is uncertain. Execution decisions are unavailable."
            if spent.uncertain
            else "The phase needs a revised plan. Enter an instruction to repair remaining work, or Stop."
            if repair_required
            else "Resume requests fresh review before any work restarts."
            if full.status == "paused"
            else decision.reason.replace("_", " ")
            if decision and not getattr(decision.evidence, "ready", False)
            else spent.pause_reason or ""
        )
        self.query_one("#workflow-notice").update(
            "\n".join(filter(None, (reason, self._action_notice)))
        )

    def _render_results(self):
        results = self.snapshot.results
        self._result_page = min(self._result_page, max(0, (len(results) - 1) // 50))
        start = self._result_page * 50
        page = results[start : start + 50]
        selector = self.query_one("#workflow-results", Select)
        if page != self._visible_results:
            self._visible_results = page
            self.checkpoint = None
            selector.set_options(
                [(Text(f"{r.status} · {r.run_id}"), r.attempt_id) for r in page]
            )
            selector.value = page[0].attempt_id if page else Select.NULL
        selector.display = bool(page)
        self.query_one("#workflow-result-pages").display = len(results) > 50
        self.query_one("#workflow-evidence").display = bool(page)
        for name, show in (("prev", start > 0), ("next", start + 50 < len(results))):
            self.query_one("#workflow-result-" + name).display = show
        targets = {ref: run_id for w in self.windows for run_id, ref in w["snapshots"]}
        if self.checkpoint:
            targets.update(
                {
                    ref: self.checkpoint.run_id
                    for ref in self.checkpoint.change_window_refs
                }
            )
        windows = self.query_one("#workflow-windows", Select)
        if targets != self._window_targets:
            self._window_targets = targets
            selected = windows.value
            windows.set_options(
                [(f"Whole-phase snapshot {ref}", ref) for ref in targets]
            )
            if targets:
                windows.value = selected if selected in targets else next(iter(targets))
        windows.display = bool(targets)
        changes = self.query_one("#workflow-changes", Button)
        changes.display = bool(targets)
        changes.disabled = not targets

    @on(Select.Changed, "#workflow-results")
    async def _select_result(self, event):
        try:
            result = next(
                (r for r in self._visible_results if r.attempt_id == event.value), None
            )
            self.checkpoint = None
            self.query_one("#workflow-changes", Button).disabled = True
            self.query_one("#workflow-evidence").update("")
            if not result:
                if self.snapshot is not None:
                    self._render_results()
                return
            service = self.coordinator.service
            try:
                evidence = await self.reads.read(
                    service.call,
                    service.read_checkpoint,
                    self.workflow_id,
                    result.attempt_id,
                )
            except WORKFLOW_VIEW_ERRORS as error:
                message = workflow_view_error("Checkpoint", error)
                if (
                    self._can_render
                    and self.query_one("#workflow-results", Select).value
                    == result.attempt_id
                ):
                    self.query_one("#workflow-evidence").update(
                        message + " Reselect the result or reopen detail to retry."
                    )
                return
            if not self._can_render:
                return
            selector = self.query_one("#workflow-results", Select)
            if selector.value != result.attempt_id:
                return
            self.checkpoint = evidence
            lines = [
                f"Result · {result.status}",
                result.summary,
                "Handoff",
                result.handoff,
                "Checks",
            ]
            lines.extend(f"{c.id} · {c.status} · {c.reason}" for c in result.checks)
            if not result.checks:
                lines.append("No checks recorded.")
            for record in evidence.verifiers:
                lines.extend(
                    [
                        f"{record.verifier_id} · exit {record.exit_code} · {record.shell_path}",
                        f"Invocation {record.invocation_id} · arguments {record.arguments_digest}",
                        f"Verifier {record.verifier_sha256} · shell manifest {record.shell_manifest}",
                        f"Inputs {record.input_manifest} · outputs {record.output_manifest}",
                        f"Complete {record.complete} · cleanup {record.cleanup_proven} · fresh {record.fresh}",
                        f"stdout\n{record.stdout}",
                        f"stderr\n{record.stderr}",
                    ]
                )
            lines.append(evidence.change_review_disclosure)
            if not evidence.change_window_refs:
                lines.append("No captured window for this checkpoint.")
            self.query_one("#workflow-evidence").update("\n".join(lines))
            self._render_results()
        except NoMatches:
            self.checkpoint = None
            return

    async def on_button_pressed(self, event: Button.Pressed):
        try:
            action = event.button.id.removeprefix("workflow-")
            if action == "close":
                await self.request_safe_cancel(source="button")
            elif action == "refresh":
                self._action_notice = ""
                self.request_refresh()
            elif action in {"result-prev", "result-next"}:
                self._result_page += 1 if action.endswith("next") else -1
                self._render_results()
            elif action in {"window-newer", "window-older"}:
                self._window_offset += 50 if action.endswith("older") else -50
                self.request_refresh()
            elif action == "changes":
                selected = self.query_one("#workflow-windows", Select).value
                if selected in self._window_targets:
                    self.open_changes(
                        self.snapshot, self._window_targets[selected], selected
                    )
            elif action == "fleet":
                selected = self.query_one("#workflow-task", Select).value
                self.open_fleet(
                    self,
                    self.conversation_id,
                    selected if isinstance(selected, str) else None,
                )
            elif action == "pending":
                self.open_approval(self, self.conversation_id)
            elif (
                not self._sending
                and self.snapshot is not None
                and not event.button.disabled
            ):
                full, decision = self.snapshot, self.decision
                instruction = self.query_one(
                    "#workflow-instruction", Input
                ).value.strip()
                if action == "revise" and not instruction:
                    self.query_one("#workflow-notice").update(
                        "Enter an instruction for the revised plan."
                    )
                    self.query_one("#workflow-instruction").focus()
                    return
                self._sending = True
                self._action_notice = ""
                self._disable_commands()
                try:
                    self.run_worker(
                        partial(self._act, action, full, decision, instruction)
                    )
                except RuntimeError as error:
                    self._sending = False
                    self._action_notice = workflow_view_error(
                        action.capitalize(), error
                    )
                    self.request_refresh()
        except NoMatches:
            self._sending = False
            return

    async def _act(self, action, full, decision, instruction):
        expected = full.control_revision
        if action in {"approve", "accept"}:
            key = (action, decision.canonical_json())
        else:
            key = (
                action,
                self.workflow_id,
                expected,
                instruction if action == "revise" else "",
            )
        command_id = self._command_ids.setdefault(key, str(uuid4()))
        try:
            if action in {"recovery", "remove"}:
                title, message, confirm = (
                    (
                        "Review prior workflow effects",
                        (
                            "Close this workflow after reviewing its prior file effects? Unknown charges stay recorded. "
                            "This does not resume work. Active work and cleanup must finish first."
                        ),
                        "Close reviewed workflow",
                    )
                    if action == "recovery"
                    else (
                        "Remove stored workflow results",
                        (
                            "Remove results, checkpoints and Change Review references? Plans, charges, agent-run history "
                            "and workspace files remain."
                        ),
                        "Remove results",
                    )
                )
                if (
                    not await self.app.push_screen_wait(
                        ConfirmationDialog(
                            title=title, message=message, confirm_label=confirm
                        )
                    )
                    or not self._can_render
                ):
                    return
            current = await self.reads.read(self.coordinator.get, self.workflow_id)
            if (current.control_revision, current.status) != (expected, full.status):
                raise ValueError("This review changed. Review the refreshed revision.")
            co = self.coordinator
            if action in {"approve", "accept"}:
                call = co.approve(decision, command_id=command_id)
            elif action == "revise":
                call = co.revise(
                    self.workflow_id,
                    expected_control_revision=expected,
                    instruction=instruction,
                    command_id=command_id,
                )
            elif action == "resume":
                call = co.resume(
                    self.workflow_id,
                    expected_control_revision=expected,
                    command_id=command_id,
                )
            elif action == "pause":
                call = co.pause(self.workflow_id, expected_control_revision=expected)
            elif action == "stop":
                call = co.stop(self.workflow_id, expected_control_revision=expected)
            elif action == "recovery":
                call = co.reconcile(
                    self.workflow_id,
                    expected_control_revision=expected,
                    decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
                    command_id=command_id,
                )
            elif action == "remove":
                call = co.remove_results(
                    self.workflow_id,
                    expected_control_revision=expected,
                    command_id=command_id,
                )
            else:
                return
            receipt = await asyncio.shield(call)
            label = {
                "approve": "Phase approval",
                "accept": "Quality acceptance",
                "pause": "Pause",
                "stop": "Stop",
                "resume": "Resume",
                "revise": "Revision",
                "recovery": "Recovery review",
                "remove": "Stored result removal",
            }[action]
            self._action_notice = (
                f"{label} received · revision {receipt.control_revision}."
            )
        except WORKFLOW_VIEW_ERRORS as error:
            self._action_notice = workflow_view_error(action.capitalize(), error)
            if self._can_render:
                for notice in self.query("#workflow-notice"):
                    notice.update(self._action_notice)
        finally:
            self._sending = False
            self.request_refresh()
