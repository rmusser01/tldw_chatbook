"""Launcher panel for skill evals. DB-free widget; the screen owns engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.validation import ValidationResult, Validator
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

#: What the directory-path input expects, shown inline beneath it.
_DIR_HINT_DEFAULT = "A folder containing SKILL.md"

#: The picker's neutral prompt (restored whenever the store holds skills).
_STORE_PROMPT = "subject skill (store)"

#: TASK-32883: with an empty store the picker swaps to a guidance prompt and
#: its overlay holds one guidance ROW carrying this sentinel value -- picking
#: it posts nothing (the handler rejects the sentinel), so a blank dead-end
#: overlay can never masquerade as a broken control or a selectable subject.
_EMPTY_STORE_SENTINEL = "__no_skills_in_store__"
_EMPTY_STORE_PROMPT = "No skills installed — type a path below"
_EMPTY_STORE_ROW = "No skills in your store — install one, or type a directory path below"

#: TASK-32884: with zero eval models the pickers must bootstrap, not blank.
#: The guidance row's sentinel is rejected by the change handler; the
#: message names the two real creation paths (bench editor "+ New target",
#: or the rail's "Create sample bench", which seeds a model in one step).
_NO_TARGETS_SENTINEL = "__no_eval_models__"
_NO_TARGETS_PROMPT = "no eval models configured"
_NO_TARGETS_ROW = (
    "No eval models — create one via '+ New target' in a bench editor, "
    "or run 'Create sample bench' once"
)
_NO_TARGETS_RUN_TOAST = (
    "No eval models configured — create one via '+ New target' in a bench "
    "editor, or run 'Create sample bench' once, then pick generator and "
    "judge here."
)


class _SkillDirectoryValidator(Validator):
    """Advisory validation for the directory-path subject (TASK-32883).

    Mirrors ``subject_from_directory``'s contract (absolute directory
    containing ``SKILL.md``) so feedback is immediate; the run's own
    resolver stays authoritative.
    """

    def validate(self, value: str) -> ValidationResult:
        stripped = value.strip()
        if not stripped:
            return self.success()
        path = Path(stripped)
        if path.is_absolute() and (path / "SKILL.md").is_file():
            return self.success()
        return self.failure("No SKILL.md found at that path")


class _SetSelect(Select):
    """A Select whose Enter key no-ops once a value is held (TASK-32889).

    Textual's stock Select binds Enter (with down/space/up) to opening
    the overlay -- so a keyboard user pressing Enter to "confirm" a set
    value silently re-opened the list with focus unchanged, the exact
    trap the HCI review's keyboard-only walkthrough hit repeatedly (B8).
    Space and the arrows keep opening; Enter opens only while nothing is
    chosen yet (the picking flow). The base class's combined binding is
    replaced wholesale because Textual matches whole key strings, not
    per-key subtraction.
    """

    _OPEN_KEYS = "down,space,up"

    BINDINGS = [
        *(b for b in Select.BINDINGS if b.key != "enter,down,space,up"),
        Binding(_OPEN_KEYS, "show_overlay", "Open", show=False),
        Binding("enter", "enter_opens_only_when_unset", "Open", show=False),
    ]

    def action_enter_opens_only_when_unset(self) -> None:
        if self.value is Select.NULL or not self.value:
            self.action_show_overlay()
        # Else: deliberate no-op -- the value stands.


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

    class CloseRequested(Message, namespace="skill_eval_panel"):
        """TASK-32889: the user asked to close the launch panel (Escape).

        Selection-level only -- the owning screen clears its selection;
        this never pops the Lab screen (that Escape remains deliberately
        unbound Lab-wide, per lab_frame's own contract).
        """

    class SubjectChanged(Message, namespace="skill_eval_panel"):
        """The user set the bench's subject (store pick or directory path)."""

        def __init__(self, subject_ref: str, subject_kind: str) -> None:
            super().__init__()
            self.subject_ref = subject_ref
            self.subject_kind = subject_kind

    class DepthChanged(Message, namespace="skill_eval_panel"):
        """TASK-32888: the depth pick, persisted by the screen so a launch
        in progress survives navigation away and back."""

        def __init__(self, depth: SkillEvalDepth) -> None:
            super().__init__()
            self.depth = depth

    class TargetsPicked(Message, namespace="skill_eval_panel"):
        """TASK-32888: one or both model picks changed, persisted by the
        screen (empty id = explicitly unset)."""

        def __init__(self, generator_target_id: str,
                     judge_target_id: str) -> None:
            super().__init__()
            self.generator_target_id = generator_target_id
            self.judge_target_id = judge_target_id

    #: TASK-32889: Escape closes the panel (widget-level binding -- the
    #: Lab-wide "Escape is a deliberate no-op" contract binds nothing at
    #: screen level, and widget bindings are the sanctioned finer grain).
    BINDINGS = [
        Binding("escape", "request_close", "Close panel", show=False),
    ]

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
            yield _SetSelect([], id="skill-eval-subject-picker",
                             prompt=_STORE_PROMPT)
            yield Input(id="skill-eval-subject-dir",
                        placeholder="or: skill directory path",
                        validators=[_SkillDirectoryValidator()])
            yield Static(_DIR_HINT_DEFAULT, id="skill-eval-subject-dir-hint",
                         markup=False)
            yield _SetSelect(_DEPTH_OPTIONS, id="skill-eval-depth",
                             value=self._depth)
            yield _SetSelect([], id="skill-eval-generator",
                             prompt="generator model")
            yield _SetSelect([], id="skill-eval-judge", prompt="judge model")
            yield Static("", id="skill-eval-estimate", markup=False)
            yield Button("Run", id="skill-eval-run")
            # TASK-32889: "Stop run", not "Cancel" -- the old label read as
            # "close this form" while the action only cancels an in-flight
            # run (a silent no-op when idle). It mounts DISABLED; the
            # screen's running-UI setters enable it for exactly the
            # in-flight window.
            yield Button("Stop run", id="skill-eval-cancel", disabled=True)

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
        picker = self.query_one("#skill-eval-subject-picker", Select)
        if not options:
            # TASK-32883: an empty store must not present a blank dead-end
            # overlay -- one guidance row (rejected by the change handler)
            # plus a prompt that names the alternative path input below.
            picker.set_options([(_EMPTY_STORE_ROW, _EMPTY_STORE_SENTINEL)])
            picker.prompt = _EMPTY_STORE_PROMPT
        else:
            picker.set_options(options)
            picker.prompt = _STORE_PROMPT

    def set_targets(self, rows: List[dict]) -> None:
        options = [
            (f"{r.get('name') or r['id']} "
             f"({r.get('provider')}/{r.get('model_id')})", r["id"])
            for r in rows
        ]
        for picker_id, prompt in (
            ("skill-eval-generator", "generator model"),
            ("skill-eval-judge", "judge model"),
        ):
            picker = self.query_one(f"#{picker_id}", Select)
            if options:
                picker.set_options(options)
                picker.prompt = prompt
            else:
                # TASK-32884: a blank picker is a dead end; the guidance
                # row is rejected on pick, the prompt names the state.
                picker.set_options([(_NO_TARGETS_ROW, _NO_TARGETS_SENTINEL)])
                picker.prompt = _NO_TARGETS_PROMPT

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "skill-eval-depth":
            self._depth = event.value
            self._refresh_estimate()
            self.post_message(self.DepthChanged(event.value))
        elif event.select.id in ("skill-eval-generator", "skill-eval-judge"):
            # TASK-32884: the no-models guidance row is not a target.
            if event.value == _NO_TARGETS_SENTINEL:
                event.select.value = Select.NULL
                self.notify(_NO_TARGETS_ROW, severity="information")
                return
            # TASK-32888: persist both picks in one message -- the screen
            # round-trips them exactly like the subject.
            generator = self.query_one("#skill-eval-generator", Select).value
            judge = self.query_one("#skill-eval-judge", Select).value

            def _id(value: object) -> str:
                if value is Select.NULL or not value:
                    return ""
                return str(value)

            self.post_message(
                self.TargetsPicked(_id(generator), _id(judge))
            )
        elif event.select.id == "skill-eval-subject-picker":
            # TASK-32883: the empty-store guidance row is not a subject.
            # Reject it, stay unset, and say why -- a silently stuck picker
            # reads as a broken control.
            if event.value == _EMPTY_STORE_SENTINEL:
                event.select.value = Select.NULL
                self.notify(_EMPTY_STORE_ROW, severity="information")
                return
            # Select.NULL is a truthy sentinel (Textual 8) -- identity check,
            # the same discipline the Run guard below documents.
            if event.value is Select.NULL or not event.value:
                return
            # Last-touched wins: a store pick clears the directory path.
            directory = self.query_one("#skill-eval-subject-dir", Input)
            directory.value = ""
            self._subject_ref = str(event.value)
            self.post_message(self.SubjectChanged(str(event.value), "store"))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "skill-eval-subject-dir":
            return
        ref = event.value.strip()
        hint = self.query_one("#skill-eval-subject-dir-hint", Static)
        if not ref:
            # Emptying the Input is not a subject choice (and must not
            # resurrect a cleared store pick -- re-pick explicitly).
            hint.update(_DIR_HINT_DEFAULT)
            return
        # TASK-32883: immediate inline feedback instead of a failed run
        # later -- the validator mirrors subject_from_directory's contract.
        if event.validation_result is not None and not event.validation_result.is_valid:
            hint.update("No SKILL.md found at that path")
        else:
            hint.update("")
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
                # TASK-32884: teach the way out when the pickers are empty
                # because no eval models exist at all -- "pick one first"
                # strands a user with nothing to pick from.
                generator = self.query_one("#skill-eval-generator", Select)
                has_real_options = any(
                    value is not Select.NULL and value != _NO_TARGETS_SENTINEL
                    for _label, value in generator._options
                )
                self.notify(
                    _NO_TARGETS_RUN_TOAST if not has_real_options
                    else "Pick generator and judge models first.",
                    severity="warning",
                )
                return
            self.post_message(self.RunRequested(self._depth, str(gen), str(jud)))
        elif event.button.id == "skill-eval-cancel":
            self.post_message(self.CancelRequested())

    def action_request_close(self) -> None:
        """Escape: ask the owning screen to close this panel (TASK-32889).

        Posts rather than acting -- the panel is DB-free and owns nothing
        about selection; the screen clears its selection, which never
        pops the Lab screen.
        """
        self.post_message(self.CloseRequested())
