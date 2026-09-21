"""Launcher panel for skill evals. DB-free widget; the screen owns engines."""

from __future__ import annotations

from typing import Any, List

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static

from ...Evals.skill_eval.models import SkillEvalDepth
from ...Evals.skill_eval.runner import estimate_calls, max_estimate_calls

_DEPTH_OPTIONS = [
    (f"quick — static only ({estimate_calls(SkillEvalDepth.QUICK)} calls)",
     SkillEvalDepth.QUICK),
    (f"standard — + judge ({estimate_calls(SkillEvalDepth.STANDARD)} calls)",
     SkillEvalDepth.STANDARD),
    (f"deep — + simulation ({estimate_calls(SkillEvalDepth.DEEP)} calls)",
     SkillEvalDepth.DEEP),
]


class SkillEvalPanel(Widget):
    """Subject picker + summary, depth + model pickers, cost estimate, run/cancel.

    The subject can be picked two ways (spec §10): a ``Select`` over the
    store's skills (fed via ``set_subjects``; labels carry the trust tier)
    or a free-text ``Input`` naming a skill directory path. The two controls
    follow a deliberate last-touched-wins rule, implemented simply and
    symmetrically: a store pick CLEARS the directory Input (and posts
    ``SubjectChanged(kind="store")``), while typing a non-empty path RESETS
    the store Select to its NULL sentinel (and posts ``SubjectChanged(
    kind="directory")``) -- so at most one control ever holds an effective
    choice, and emptying the Input resurrects nothing (re-pick explicitly).
    The message carries ``(subject_ref, subject_kind)`` from either
    control; the owning screen persists it onto the bench config.
    """

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

    class SubjectChanged(Message, namespace="skill_eval_panel"):
        """The user set the bench's subject (store pick or directory path)."""

        def __init__(self, subject_ref: str, subject_kind: str) -> None:
            super().__init__()
            self.subject_ref = subject_ref
            self.subject_kind = subject_kind

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._depth: SkillEvalDepth = SkillEvalDepth.STANDARD
        #: The EFFECTIVE subject: the last real choice from either subject
        #: control, or the ref the screen restored from a persisted bench.
        #: The Run guard (TASK-32885) reads this so a subjectless Run can
        #: never dispatch -- it used to become a real run that failed in
        #: the worker and left a failed row in the rail. A NULL picker pick
        #: never clears it: that matches what a run would actually use (the
        #: screen persists every SubjectChanged the moment it posts).
        self._subject_ref: str = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("No subject selected.",
                         id="skill-eval-subject", markup=False)
            yield Select([], id="skill-eval-subject-picker",
                         prompt="subject skill (store)")
            yield Input(id="skill-eval-subject-dir",
                        placeholder="or: skill directory path")
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

    #: The screen's remount fallback for a subjectless draft bench -- the
    #: display line must say something, but it is NOT a runnable subject.
    _NO_SUBJECT_DISPLAY = "(no subject set)"

    def set_subject(self, name: str, source: str) -> None:
        """Update the subject display line and the Run guard's subject.

        ``name`` is the screen's display string: a real subject ref, or the
        ``_NO_SUBJECT_DISPLAY`` sentinel for a subjectless draft. Only a
        real ref arms the guard's subject half.
        """
        if name and name != self._NO_SUBJECT_DISPLAY:
            self._subject_ref = name
        widget = self.query_one("#skill-eval-subject", Static)
        widget.update(f"Subject: {name} ({source})")

    def set_subjects(self, rows: List[dict]) -> None:
        """Feed the store-skill picker (screen-side ``store_skill_names``).

        Rows are store skill summaries (``name``, optional ``description``
        and ``trust_status``); each becomes one option labelled
        ``name (trust)`` keyed by the bare name, deduplicated so a store
        glitch (two same-named summaries) cannot produce two options whose
        values collide.
        """
        options: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            trust = str(row.get("trust_status") or "unknown")
            options.append((f"{name} ({trust})", name))
        self.query_one("#skill-eval-subject-picker", Select).set_options(options)

    def set_targets(self, rows: List[dict]) -> None:
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
        elif event.select.id == "skill-eval-subject-picker":
            # Select.NULL is a truthy sentinel (Textual 8) -- identity check,
            # the same discipline the Run guard below documents.
            if event.value is Select.NULL or not event.value:
                return
            # Last-touched wins: a store pick clears the directory path.
            self.query_one("#skill-eval-subject-dir", Input).value = ""
            self._subject_ref = str(event.value)
            self.post_message(self.SubjectChanged(str(event.value), "store"))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "skill-eval-subject-dir":
            return
        ref = event.value.strip()
        if not ref:
            # Emptying the Input is not a subject choice (and must not
            # resurrect a cleared store pick -- re-pick explicitly).
            return
        # Last-touched wins: a typed path resets the store pick.
        self.query_one("#skill-eval-subject-picker", Select).value = Select.NULL
        self._subject_ref = ref
        self.post_message(self.SubjectChanged(ref, "directory"))

    def _refresh_estimate(self) -> None:
        label = self.query_one("#skill-eval-estimate", Static)
        nominal = estimate_calls(self._depth)
        maximum = max_estimate_calls(self._depth)
        # Qodo F1: the judge layer retries each failed cell once, so the
        # worst case is the doubled judge budget -- shown as a parenthetical
        # next to the nominal count (quick needs no parenthetical: 0 max).
        if maximum > nominal:
            label.update(
                f"Estimated LLM calls: {nominal} "
                f"(max {maximum} with judge retries)")
        else:
            label.update(f"Estimated LLM calls: {nominal}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-eval-run":
            # TASK-32885: a missing subject used to dispatch a run that
            # failed in the worker (SubjectError) and left a failed row in
            # the rail -- guard it the same way the model pickers are
            # guarded, with a message that names the two ways to set one.
            if not self._subject_ref:
                self.notify(
                    "Pick a subject skill or type a directory path first.",
                    severity="warning",
                )
                return
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
