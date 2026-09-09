"""Private immutable goal setup; provisioning and authority belong to the service."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from uuid import uuid4

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Select,
    SelectionList,
    Static,
    TextArea,
)

from tldw_chatbook.Agents.goal_models import GoalRequest, GoalSnapshot


class ConsoleGoalSetupModal(ModalScreen[GoalSnapshot | None]):
    """Edit the purpose of an already resolved authority selection."""

    BINDINGS = (("escape", "cancel", "Cancel"),)
    DEFAULT_CSS = """
    ConsoleGoalSetupModal { align: center middle; }
    ConsoleGoalSetupModal > Vertical { width: 76; max-width: 100%; height: 90%; background: $surface; border: solid $primary; }
    ConsoleGoalSetupModal VerticalScroll { height: 1fr; padding: 0 1; }
    ConsoleGoalSetupModal TextArea { height: 4; }
    ConsoleGoalSetupModal SelectionList { height: 4; }
    ConsoleGoalSetupModal Static, ConsoleGoalSetupModal Label { height: auto; }
    ConsoleGoalSetupModal Horizontal { height: 3; }
    ConsoleGoalSetupModal Button { min-width: 12; width: 1fr; }
    """

    def __init__(
        self,
        request: GoalRequest,
        *,
        start: Callable[[GoalRequest, str], Awaitable[GoalSnapshot]],
        bindings=(),
        tool_ids=(),
        configure=None,
    ) -> None:
        super().__init__()
        self.request, self._start = request, start
        self.bindings, self.tool_ids, self.configure = bindings, tool_ids, configure
        self.launch_id = uuid4().hex
        self._submitted: GoalRequest | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        p = self.request.policy
        with Vertical():
            with VerticalScroll():
                yield Label("Start a goal")
                yield Static(
                    f"{self.request.provider.provider} / {self.request.provider.model}",
                    markup=False,
                )
                yield Label("Objective")
                yield TextArea(self.request.objective, id="goal-objective")
                yield Label("Success criteria")
                yield TextArea(self.request.criteria, id="goal-criteria")
                if self.bindings:
                    yield Label("Project binding")
                    yield Select(
                        [
                            (b.locator + " (" + b.access + ")", b.binding_id)
                            for b in self.bindings
                        ],
                        value=self.request.binding.binding_id,
                        allow_blank=False,
                        id="goal-binding",
                    )
                    if any(b.access == "ro" for b in self.bindings):
                        yield Label("Additional read-only sources (optional)")
                        yield SelectionList(
                            *[
                                (b.locator, b.binding_id, False)
                                for b in self.bindings
                                if b.access == "ro"
                            ],
                            id="goal-sources",
                        )
                    yield Label("Tools (selection narrows existing permissions)")
                    yield SelectionList(
                        *[
                            (t, t, t in self.request.tool_scope.catalog_tools)
                            for t in self.tool_ids
                        ],
                        id="goal-tools",
                    )
                    yield Label("Trusted validation skill (optional)")
                    yield Input(placeholder="Skill name", id="goal-skill")
                    yield Label("Script path within the trusted skill")
                    yield Input(placeholder="scripts/check.py", id="goal-script")
                    yield Label(
                        "Arguments (JSON array; include the project path explicitly)"
                    )
                    yield Input(
                        "[]",
                        placeholder="Arguments as a JSON array; pass the project path explicitly",
                        id="goal-arguments",
                    )
                    yield Label(
                        "Checked input paths (JSON array, relative to the project)"
                    )
                    yield Input(
                        '["fixture.txt"]',
                        placeholder="Checked relative input paths as a JSON array",
                        id="goal-inputs",
                    )
                yield Static(
                    "Selected tools: "
                    + ", ".join(
                        self.request.tool_scope.catalog_tools
                        + self.request.tool_scope.runtime_tools
                    ),
                    markup=False,
                )
                yield Static(
                    "Selected checks: "
                    + (
                        "; ".join(
                            f"{v.id}: {v.verifier_path} {list(v.arguments)}; inputs {list(v.input_paths)}"
                            for v in self.request.verifiers
                        )
                        or "None — result review cannot supply missing objective proof."
                    ),
                    id="goal-selected-checks",
                    markup=False,
                )
                yield Static(
                    f"{p.iterations} iterations; {p.model_calls} model calls; {p.budget_tokens:,} budget tokens; {p.output_tokens:,} output tokens/call; {p.wall_seconds}s elapsed. Each iteration: {p.iteration_model_turns} turns, {p.iteration_steps} steps, {p.iteration_wall_seconds}s. Waits and pauses count after acceptance. Unknown usage stays reserved. Raising settings does not refill this goal.",
                    markup=False,
                )
                yield Static(
                    "Local file tools stay within the selected project. Trusted CLI scripts run with the configured executor’s actual authority; the scratch directory is not an OS sandbox. Existing trust and approval rules still apply.",
                    markup=False,
                )
                yield Checkbox(
                    "Require human result review", value=True, id="goal-human-review"
                )
                yield Static("", id="goal-setup-error", markup=False)
            with Horizontal():
                yield Button("Cancel", id="goal-cancel")
                yield Button(
                    "Review launch" if self.configure else "Start",
                    id="goal-start",
                    variant="primary",
                )

    def action_cancel(self) -> None:
        if not self._busy:
            self.dismiss(None)

    @on(Button.Pressed, "#goal-cancel")
    def cancel_pressed(self) -> None:
        self.action_cancel()

    @on(Button.Pressed, "#goal-start")
    def start_pressed(self) -> None:
        if not self._busy:
            self._busy = True
            self.query_one("#goal-start", Button).disabled = True
            self.launch()

    @work(exclusive=True)
    async def launch(self) -> None:
        try:
            if self._submitted is None:
                values = self.request.model_dump()
                values.update(
                    objective=self.query_one("#goal-objective", TextArea).text,
                    criteria=self.query_one("#goal-criteria", TextArea).text,
                    human_review_required=self.query_one(
                        "#goal-human-review", Checkbox
                    ).value,
                )
                if self.configure:
                    values["binding"] = next(
                        b
                        for b in self.bindings
                        if b.binding_id == self.query_one("#goal-binding", Select).value
                    ).model_dump()
                    source_ids = (
                        self.query_one("#goal-sources", SelectionList).selected
                        if self.query("#goal-sources")
                        else []
                    )
                    values["source_bindings"] = tuple(
                        b.model_dump()
                        for b in self.bindings
                        if b.binding_id in source_ids
                        and b.binding_id != values["binding"]["binding_id"]
                    )
                    values["tool_scope"]["catalog_tools"] = tuple(
                        self.query_one("#goal-tools", SelectionList).selected
                    )
                    arguments = json.loads(
                        self.query_one("#goal-arguments", Input).value
                    )
                    inputs = json.loads(self.query_one("#goal-inputs", Input).value)
                    if not isinstance(arguments, list) or not isinstance(inputs, list):
                        raise ValueError(
                            "Arguments and checked inputs must be JSON arrays."
                        )
                    self._submitted = await self.configure(
                        values,
                        self.query_one("#goal-skill", Input).value.strip(),
                        self.query_one("#goal-script", Input).value.strip(),
                        tuple(arguments),
                        tuple(inputs),
                    )
                else:
                    self._submitted = GoalRequest.model_validate(values)
                self._submitted.validate_verifier_invocations()
                if self.configure:
                    self.query_one("#goal-selected-checks", Static).update(
                        "Selected checks: "
                        + (
                            "; ".join(
                                f"{v.id}: {v.verifier_path} {list(v.arguments)}; inputs {list(v.input_paths)}"
                                for v in self._submitted.verifiers
                            )
                            or "None"
                        )
                    )
                    for widget in self.query(
                        "Input, TextArea, Select, SelectionList, Checkbox"
                    ):
                        widget.disabled = True
                    self.query_one("#goal-setup-error", Static).update(
                        "Launch selections are fixed. Start to run; Cancel to change the setup."
                    )
                    self.query_one("#goal-start", Button).label = "Start"
                    self.query_one("#goal-selected-checks").scroll_visible()
                    return
            result = await self._start(self._submitted, self.launch_id)
            self.dismiss(result)
        except Exception as exc:  # noqa: BLE001 - preserve a recoverable saved launch
            self.query_one("#goal-setup-error", Static).update(
                str(exc)
                if isinstance(exc, (ValueError, RuntimeError))
                else "Setup could not finish. Reopen Goal runs and retry the saved setup."
            )
        finally:
            self._busy = False
            if self.is_mounted:
                self.query_one("#goal-start", Button).disabled = False
