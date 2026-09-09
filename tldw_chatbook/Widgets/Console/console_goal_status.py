"""Explicit private selected-goal view; no scheduling, polling, or ledger ownership."""

from __future__ import annotations

import asyncio
from datetime import datetime

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static

from tldw_chatbook.Agents.goal_models import RecoveryResolution


class ConsoleGoalStatus(ModalScreen[None]):
    BINDINGS = (("escape", "close", "Close"),)
    DEFAULT_CSS = """
    ConsoleGoalStatus { align: center middle; }
    ConsoleGoalStatus > Vertical { width: 90; max-width: 100%; height: 95%; background: $surface; border: solid $primary; }
    ConsoleGoalStatus VerticalScroll { height: 1fr; padding: 0 1; }
    ConsoleGoalStatus Static { height: auto; }
    ConsoleGoalStatus Horizontal { height: 3; }
    ConsoleGoalStatus Button { min-width: 9; width: 1fr; }
    """

    def __init__(
        self, coordinator, goal_id: str, *, review_changes=None, app_instance=None
    ) -> None:
        super().__init__()
        self.coordinator, self.goal_id = coordinator, goal_id
        self.review_changes = review_changes
        self.app_instance = app_instance or coordinator.controller.app
        self.snapshot = None
        self._unsubscribe = None
        self._review_identity = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with VerticalScroll():
                yield Static("Goal runs", id="goal-state", markup=False)
                yield Static("", id="goal-accounting", markup=False)
                yield Static("", id="goal-reason", markup=False)
                yield Select([], prompt="Select saved iteration", id="goal-iteration")
                yield Static("", id="goal-report", markup=False)
                yield Static("", id="goal-evidence", markup=False)
                yield Static("", id="goal-error", markup=False)
            with Horizontal():
                yield Button("Retry setup", id="goal-retry-setup")
                yield Button("Pause", id="goal-pause")
                yield Button("Stop", id="goal-stop")
                yield Button("Resume", id="goal-resume")
                yield Button("Review", id="goal-review", variant="primary")
                yield Button("Changes", id="goal-changes")
            with Horizontal():
                yield Button("Accept result", id="goal-accept", variant="success")
                yield Button("Reject result", id="goal-reject")
                yield Button("Close uncertain", id="goal-recovery")
                yield Button("Remove payloads", id="goal-remove")
                yield Button("Close", id="goal-close")

    def on_mount(self) -> None:
        self._unsubscribe = self.coordinator.subscribe(self._changed)
        self.load_goal()

    def on_unmount(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()

    def _changed(self, goal_id: str) -> None:
        if self.is_mounted and goal_id == self.goal_id:
            self.load_goal()

    def action_close(self) -> None:
        self.dismiss(None)

    @work(exclusive=True, group="goal-read")
    async def load_goal(self) -> None:
        try:
            goal = await asyncio.to_thread(self.coordinator.service.get, self.goal_id)
        except Exception:  # noqa: BLE001 - private view failures must not crash the app
            if self.is_mounted:
                self.query_one("#goal-error", Static).update(
                    "Goal history is unavailable. Close and reopen to retry."
                )
            return
        if not self.is_mounted:
            return
        self.snapshot = goal
        if self._review_identity and self._review_identity[0] != goal.revision:
            self._review_identity = None
        running = self.coordinator.active_goal_id == goal.id
        self.query_one("#goal-state", Static).update(
            f"{goal.request.objective if goal.request else 'Removed goal'}\nGoal {goal.id}\n{goal.status.replace('_', ' ')}"
            + (" · executing" if running else "")
        )
        a = goal.accounting
        deadline = (
            datetime.fromtimestamp(a.deadline_at)
            .astimezone()
            .isoformat(timespec="seconds")
            if a.deadline_at
            else "starts at first acceptance"
        )
        self.query_one("#goal-accounting", Static).update(
            f"Iterations {goal.iteration_count}; calls {a.used['model_call']}; tokens used {a.used['tokens']}, reserved/unknown {a.reserved['tokens']}\nElapsed deadline: {deadline}. Pauses and waits count; Resume never refills."
        )
        reason = (goal.pause_reason or "").replace("_", " ")
        if goal.status == "recovery_required":
            reason += "\nExecution may have had effects. Close uncertain keeps unknown accounting and never authorizes replay or payload removal."
        self.query_one("#goal-reason", Static).update(
            reason or "Results and changes remain saved across navigation."
        )
        available = {
            "retry-setup": goal.status == "starting",
            "pause": goal.status == "ready",
            "stop": goal.status in {"ready", "paused", "pause_requested", "stopping"},
            "resume": goal.status in {"paused", "ready"} and not running,
            "review": goal.status == "awaiting_result_review",
            "changes": bool(goal.checkpoints) and self.review_changes is not None,
            "recovery": goal.status == "recovery_required",
            "remove": goal.status
            in {"paused", "stopped", "completed", "awaiting_result_review"}
            and not running,
            "accept": self._review_identity is not None
            and goal.status == "awaiting_result_review",
            "reject": self._review_identity is not None
            and goal.status == "awaiting_result_review",
        }
        for key, visible in available.items():
            self.query_one(f"#goal-{key}", Button).display = visible
        select = self.query_one("#goal-iteration", Select)
        old = select.value
        with select.prevent(Select.Changed):
            select.set_options(
                [(f"Iteration {c.ordinal} · {c.id}", c.id) for c in goal.checkpoints]
            )
            select.display = bool(goal.checkpoints)
            select.value = (
                old
                if old in {c.id for c in goal.checkpoints}
                else (goal.checkpoints[-1].id if goal.checkpoints else Select.NULL)
            )
        await self.show_iteration()

    @on(Select.Changed, "#goal-iteration")
    async def iteration_selected(self) -> None:
        self._review_identity = None
        self.query_one("#goal-accept", Button).display = False
        self.query_one("#goal-reject", Button).display = False
        self.query_one("#goal-error", Static).update("")
        await self.show_iteration()

    async def show_iteration(self) -> None:
        if self.snapshot is None:
            return
        checkpoint = self.selected_checkpoint()
        text = (
            self.snapshot.request.objective
            if self.snapshot.request
            else "Private payloads removed. Accounting is retained."
        )
        if checkpoint:
            text += f"\nCheckpoint {checkpoint.id}\nArtifact {checkpoint.artifact_digest}\n{checkpoint.report.summary}\n{checkpoint.report.candidate_draft}"
        self.query_one("#goal-report", Static).update(text)
        try:
            evidence = await asyncio.to_thread(
                self.coordinator.service.evidence, self.goal_id
            )
        except Exception:  # noqa: BLE001 - an evidence read must not close the app
            if self.is_mounted:
                self.query_one("#goal-error", Static).update(
                    "Evidence is unavailable. Reopen this goal to retry."
                )
            return
        if not self.is_mounted or checkpoint != self.selected_checkpoint():
            return
        selected = [e for e in evidence if checkpoint and e.attempt_id == checkpoint.id]
        self.query_one("#goal-evidence", Static).update(
            "\n\n".join(
                f"{e.id} · run {e.run_id} · {e.verifier_id}\n{e.reason}; passed={e.passed}; checked version {e.source_digest}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
                for e in selected
            )
        )

    def selected_checkpoint(self):
        selected = self.query_one("#goal-iteration", Select).value
        return (
            next((c for c in self.snapshot.checkpoints if c.id == selected), None)
            if self.snapshot
            else None
        )

    @on(Button.Pressed)
    async def control_pressed(self, event: Button.Pressed) -> None:
        action = (event.button.id or "").removeprefix("goal-")
        if action == "close":
            self.action_close()
            return
        if self.snapshot is None:
            return
        event.stop()
        service, goal = self.coordinator.service, self.snapshot
        try:
            if action == "retry-setup":
                await asyncio.shield(
                    self.coordinator.launch(
                        goal.request, goal.launch_id, app=self.app_instance
                    )
                )
            elif action in {"pause", "stop", "remove"}:
                method = (
                    service.remove_payloads
                    if action == "remove"
                    else getattr(service, action)
                )
                await asyncio.to_thread(method, goal.id)
            elif action == "resume":
                if goal.status == "paused":
                    await asyncio.to_thread(
                        service.resume, goal.id, expected_revision=goal.revision
                    )
                self.coordinator.start(goal.id)
            elif action == "review":
                checkpoint = self.selected_checkpoint()
                if checkpoint is None:
                    raise ValueError("Select an iteration to review.")
                decision = await asyncio.to_thread(
                    service.completion_check,
                    goal.id,
                    checkpoint_id=checkpoint.id,
                    artifact_digest=checkpoint.artifact_digest,
                )
                self._review_identity = (
                    goal.revision,
                    checkpoint.id,
                    checkpoint.artifact_digest,
                )
                self.query_one("#goal-error", Static).update(
                    f"Fresh checks: {decision.reason.replace('_', ' ')}\nAccept records quality for this exact artifact. It cannot resolve uncertain execution or supply missing proof."
                )
                self.query_one("#goal-error").scroll_visible()
            elif action in {"accept", "reject"} and self._review_identity:
                revision, checkpoint_id, artifact = self._review_identity
                await asyncio.to_thread(
                    service.review_result,
                    goal.id,
                    expected_revision=revision,
                    checkpoint_id=checkpoint_id,
                    artifact_digest=artifact,
                    accepted=action == "accept",
                )
                self._review_identity = None
            elif action == "recovery":
                await asyncio.to_thread(
                    service.resolve_recovery,
                    goal.id,
                    expected_revision=goal.revision,
                    resolution=RecoveryResolution.CLOSE_UNCERTAIN,
                )
            elif action == "changes":
                checkpoint = self.selected_checkpoint()
                if checkpoint:
                    self.review_changes(goal, checkpoint)
            self.coordinator.notify_goal_changed(goal.id)
        except Exception as exc:  # noqa: BLE001 - display a bounded save failure
            self.query_one("#goal-error", Static).update(
                str(exc)
                if isinstance(exc, (ValueError, RuntimeError))
                else "Goal action could not be saved. Reopen this goal to inspect its current state."
            )


class ConsoleGoalHistory(ModalScreen[None]):
    """Body-free history; private data is loaded only after explicit selection."""

    BINDINGS = (("escape", "close", "Close"),)
    DEFAULT_CSS = """
    ConsoleGoalHistory { align: center middle; }
    ConsoleGoalHistory > Vertical { width: 74; max-width: 100%; height: auto; max-height: 90%; background: $surface; border: solid $primary; padding: 1; }
    ConsoleGoalHistory Horizontal { height: 3; }
    ConsoleGoalHistory Button { width: 1fr; min-width: 10; }
    """

    def __init__(self, rows, *, open_goal, new_goal):
        super().__init__()
        self.rows, self.open_goal, self.new_goal = rows, open_goal, new_goal

    def compose(self):
        with Vertical():
            yield Static("Goal runs", markup=False)
            yield Select(
                [(f"{r.status.replace('_', ' ')} · {r.id}", r.id) for r in self.rows],
                prompt="Select a saved goal",
                id="goal-history",
            )
            if not self.rows:
                yield Static(
                    "No saved goals. Start a finite goal from this workspace.",
                    markup=False,
                )
            with Horizontal():
                yield Button("New goal", id="goal-history-new", variant="primary")
                yield Button(
                    "Open", id="goal-history-open", disabled=not bool(self.rows)
                )
                yield Button("Close", id="goal-history-close")

    def action_close(self):
        self.dismiss(None)

    @on(Button.Pressed)
    def pressed(self, event):
        if event.button.id == "goal-history-new":
            self.dismiss(None)
            self.new_goal()
        elif event.button.id == "goal-history-open":
            value = self.query_one("#goal-history", Select).value
            if value is not Select.NULL:
                self.dismiss(None)
                self.open_goal(value)
        else:
            self.dismiss(None)
