"""Launcher panel for skill evals. DB-free widget; the screen owns engines."""

from __future__ import annotations

from typing import Any, List

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Select, Static

from ...Evals.skill_eval.models import SkillEvalDepth
from ...Evals.skill_eval.runner import estimate_calls

_DEPTH_OPTIONS = [
    (f"quick — static only ({estimate_calls(SkillEvalDepth.QUICK)} calls)",
     SkillEvalDepth.QUICK),
    (f"standard — + judge ({estimate_calls(SkillEvalDepth.STANDARD)} calls)",
     SkillEvalDepth.STANDARD),
    (f"deep — + simulation ({estimate_calls(SkillEvalDepth.DEEP)} calls)",
     SkillEvalDepth.DEEP),
]


class SkillEvalPanel(Widget):
    """Subject summary, depth + model pickers, cost estimate, run/cancel."""

    DEFAULT_CSS = """
    SkillEvalPanel { padding: 1; }
    """

    class RunRequested(Message, namespace="skill_eval_panel"):
        def __init__(self, depth: SkillEvalDepth, generator_target_id: str,
                     judge_target_id: str) -> None:
            super().__init__()
            self.depth = depth
            self.generator_target_id = generator_target_id
            self.judge_target_id = judge_target_id

    class CancelRequested(Message, namespace="skill_eval_panel"):
        pass

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._depth: SkillEvalDepth = SkillEvalDepth.STANDARD
        self._targets: List[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("No subject selected.",
                         id="skill-eval-subject", markup=False)
            yield Select(_DEPTH_OPTIONS, id="skill-eval-depth",
                         value=self._depth)
            yield Select([], id="skill-eval-generator",
                         prompt="generator model")
            yield Select([], id="skill-eval-judge", prompt="judge model")
            yield Static("", id="skill-eval-estimate", markup=False)
            yield Button("Run", id="skill-eval-run")
            yield Button("Cancel", id="skill-eval-cancel")

    def on_mount(self) -> None:
        self._refresh_estimate()

    def set_subject(self, name: str, source: str) -> None:
        widget = self.query_one("#skill-eval-subject", Static)
        widget.update(f"Subject: {name} ({source})")

    def set_targets(self, rows: List[dict]) -> None:
        self._targets = list(rows)
        options = [
            (f"{r.get('name') or r['id']} "
             f"({r.get('provider')}/{r.get('model_id')})", r["id"])
            for r in rows
        ]
        self.query_one("#skill-eval-generator", Select).set_options(options)
        self.query_one("#skill-eval-judge", Select).set_options(options)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "skill-eval-depth":
            self._depth = event.value
            self._refresh_estimate()

    def _refresh_estimate(self) -> None:
        label = self.query_one("#skill-eval-estimate", Static)
        label.update(f"Estimated LLM calls: {estimate_calls(self._depth)}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-eval-run":
            gen = self.query_one("#skill-eval-generator", Select).value
            jud = self.query_one("#skill-eval-judge", Select).value
            # Select.NULL is a truthy sentinel (Textual 8), so a bare
            # `not gen` guard lets an unpicked model through as
            # str(Select.NULL) -- check identity against the sentinel too.
            if gen is Select.NULL or jud is Select.NULL or not gen or not jud:
                self.notify("Pick generator and judge models first.",
                            severity="warning")
                return
            self.post_message(self.RunRequested(self._depth, str(gen), str(jud)))
        elif event.button.id == "skill-eval-cancel":
            self.post_message(self.CancelRequested())
