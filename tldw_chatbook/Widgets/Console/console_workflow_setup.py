"""Passive setup for the selected chat's native phased workflow."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from typing import ClassVar
from uuid import uuid4

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Label, Select, Static, TextArea

from tldw_chatbook.Agents.workflow_models import WorkflowSetup
from tldw_chatbook.UI.Console_Modules.workflows import (
    WORKFLOW_VIEW_ERRORS,
    workflow_view_error,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


class ConsoleWorkflowSetup(SafeModalDismissMixin, ModalScreen):
    SAFE_MODAL_CONTENT = ".workflow-dialog"
    BINDINGS = (("escape", "request_safe_cancel", "Close"),)
    AUTO_FOCUS = "#workflow-objective"

    def __init__(self, coordinator, conversation_id, planner, policy, choices):
        super().__init__()
        self.coordinator, self.conversation_id = coordinator, conversation_id
        self.planner, self.policy = planner, policy
        self.bindings, self.presets, self.capture = choices
        self.launch_id = str(uuid4())
        self._submitted = None
        self._sending = False
        self.targets = {"@chat": ()}
        for name, provider, model, fallbacks in self.presets:
            pair = (
                provider or planner[0],
                model or (planner[1] if not provider or provider == planner[0] else ""),
            )
            choices = tuple(dict.fromkeys((pair, *fallbacks)))
            if all(p and m for p, m in choices) and len(choices) <= 16:
                self.targets[name] = choices

    LIMIT_LABELS: ClassVar[dict[str, str]] = {
        "generations": "Planning / execution generations",
        "child_launches": "Task launches",
        "model_calls": "Model calls",
        "budget_tokens": "Total tokens",
        "output_tokens": "Output tokens per call",
        "wall_seconds": "Cumulative active seconds",
    }

    def compose(self) -> ComposeResult:
        with Vertical(classes="workflow-dialog"):
            yield Static("New workflow", classes="workflow-title")
            with VerticalScroll(id="workflow-setup-scroll", classes="workflow-scroll"):
                yield Label("Goal")
                yield TextArea(id="workflow-objective", classes="workflow-editor")
                yield Label("Working folder")
                yield Select(
                    [
                        (Text(binding.locator), binding.binding_id)
                        for binding, _ in self.bindings
                    ],
                    value=self.bindings[0][0].binding_id,
                    allow_blank=False,
                    id="workflow-binding",
                )
                yield Static(
                    "Existing uncommitted content is input. Planning is read-only; tasks require separate approval.",
                    classes="workflow-copy",
                    markup=False,
                )
                yield Static("", id="workflow-readiness", markup=False)
                yield Label("Execution targets")
                yield Select(
                    [
                        ("Current chat selection", "@chat"),
                        *[
                            (f"Captured preset · {name}", name)
                            for name in self.targets
                            if name != "@chat"
                        ],
                    ],
                    value="@chat",
                    allow_blank=False,
                    id="workflow-target",
                )
                yield Static("", id="workflow-targets", markup=False)
                limits = self.policy
                yield Static(
                    f"Limits\n{limits.generations} generations · {limits.child_launches} task launches · "
                    f"{limits.model_calls} model calls\n{limits.budget_tokens} tokens · "
                    f"{limits.output_tokens} output tokens/call · {limits.wall_seconds}s active time",
                    markup=False,
                )
                with Collapsible(title="Adjust run limits", collapsed=True):
                    for field in fields(limits):
                        yield Label(self.LIMIT_LABELS[field.name])
                        yield Input(
                            str(getattr(limits, field.name)),
                            type="integer",
                            id="workflow-limit-" + field.name.replace("_", "-"),
                        )
            yield Static("", id="workflow-notice", markup=False)
            with Horizontal(classes="workflow-actions"):
                yield Button("Close", id="workflow-close")
                yield Button("Prepare plan", id="workflow-plan", variant="primary")

    def on_mount(self):
        self._show_choices()
        self.call_after_refresh(
            self.query_one("#workflow-setup-scroll").scroll_home, animate=False
        )

    def on_select_changed(self, event: Select.Changed):
        self._show_choices()

    def on_text_area_changed(self, event):
        self._update_submit()

    def on_input_changed(self, event):
        self._update_submit()

    def _limits(self):
        values = {}
        for field in fields(self.policy):
            raw = self.query_one(
                "#workflow-limit-" + field.name.replace("_", "-"), Input
            ).value
            maximum = getattr(self.policy, field.name)
            if maximum == 0:
                return (
                    None,
                    f"{self.LIMIT_LABELS[field.name]} are disabled in Settings.",
                )
            if (
                not raw.isascii()
                or not raw.isdecimal()
                or len(raw) > 19
                or not 1 <= int(raw) <= maximum
            ):
                return (
                    None,
                    f"{self.LIMIT_LABELS[field.name]} must be between 1 and {maximum}.",
                )
            values[field.name] = int(raw)
        return type(self.policy)(**values), ""

    def _update_submit(self):
        if not self.query("#workflow-plan"):
            return
        _, reason = self._limits()
        self.query_one("#workflow-plan", Button).disabled = (
            bool(reason)
            or self._sending
            or self._safe_pending_close is not None
            or not self.query_one("#workflow-objective", TextArea).text.strip()
        )
        if not self._sending and self._submitted is None:
            self.query_one("#workflow-notice").update(reason)

    def _show_choices(self):
        if not self.query("#workflow-plan"):
            return
        selected = self.query_one("#workflow-binding", Select).value
        binding, excluded = next(
            ((b, n) for b, n in self.bindings if b.binding_id == selected),
            self.bindings[0],
        )
        status = self.capture
        state = "unavailable"
        if status is not None:
            state = f"capability {status.capability.state.value} · workspace consent {status.consent.state.value}"
            root = next(
                (r for r in status.roots if r.alias == binding.binding_id), None
            )
            if root:
                state += f" · {root.state.value}" + (
                    f" ({root.reason})" if root.reason else ""
                )
        self.query_one("#workflow-readiness").update(
            f"Change Review · {state}\n{excluded} folder exclusions · phase baseline not yet captured"
        )
        targets = self.targets.get(self.query_one("#workflow-target", Select).value, ())
        planner = (
            "/".join(self.planner)
            if self.planner[1]
            else f"{self.planner[0]}/configured model"
        )
        self.query_one("#workflow-targets").update(
            f"Planner  {planner}\nTask targets  "
            + (", ".join(f"{p}/{m}" for p, m in targets) or planner)
        )
        self._update_submit()

    async def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "workflow-close":
            await self.request_safe_cancel(source="button")
            return
        if event.button.id != "workflow-plan" or self._sending:
            return
        limits, reason = self._limits()
        if reason:
            self.query_one("#workflow-notice").update(reason)
            return
        try:
            binding = next(
                b
                for b, _ in self.bindings
                if b.binding_id == self.query_one("#workflow-binding", Select).value
            )
            setup = WorkflowSetup(
                conversation_id=self.conversation_id,
                objective=self.query_one("#workflow-objective", TextArea).text.strip(),
                binding=binding,
                policy=limits,
                planner=self.planner,
                targets=self.targets[self.query_one("#workflow-target", Select).value],
            )
            if self._submitted is not None and self._submitted != setup:
                self.launch_id = str(uuid4())
            self._submitted = setup
        except ValueError as error:
            self.query_one("#workflow-notice").update(
                workflow_view_error("Setup", error)
            )
            return
        self._sending = True
        event.button.disabled = True
        self.query_one("#workflow-notice").update("Preparing the plan…")
        self.run_worker(self._submit(setup, self._safe_mount_generation))

    async def _submit(self, setup, generation):
        try:
            snapshot = await asyncio.shield(
                self.coordinator.start(setup, launch_id=self.launch_id)
            )
            if self._safe_mount_generation == generation:
                self.dismiss_safe_once_when_on_top(snapshot)
        except WORKFLOW_VIEW_ERRORS as error:
            message = workflow_view_error("Prepare plan", error)
            if self.is_mounted and self._safe_mount_generation == generation:
                self.query_one("#workflow-notice").update(message)
        finally:
            if self._safe_mount_generation == generation:
                self._sending = False
                if self.is_mounted and not self._safe_dismiss_committed:
                    self._show_choices()
