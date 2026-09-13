# Console `ask_user` (PRD M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An agent in Console can call `ask_user(questions)` and get the user's multiple-choice answers back, through a card above the transcript that waits indefinitely by default, parks for background sessions, cancels with the run, and writes a transcript marker on resolve.

**Architecture:** `ask_user` is a `LocalToolSpec` registered only when the controller supplies an ask callback (the `todo_*` pattern), exempt from the Allow/Ask/Off permission layer, gated by `[tools] ask_user_enabled` (default ON). The callback is `ConsoleChatController.request_user_questions`, a fourth clone of the worktree-merge confirm round (round-keyed retained payload, park-or-mount, `use_human_input_wait`, cancel/deadline poll, `_remount_head` teardown) with a question-shaped decision and a `busy` fast path. The card is a new `ChatQuestionCard` mounted lazily into `ChatTaskCards` on the first pending payload, routed through `TaskResumeState.pending_question`.

**Tech Stack:** Python ≥3.11, Textual 8.2.8 (`RadioSet`, `SelectionList`, `Input` verified present), pytest + `pytest-asyncio`, no new dependencies.

**Spec:** `Docs/Development/Chatbook/Chatbook-Console-Agent-Interaction-PRD.md` (Feature A, A1–A7 and A9–A14; AC-A1–A5b, AC-A7–A13). A8 / AC-A6 (typed-answer interception) is **M3, not this plan**. Design rationale: `Docs/superpowers/specs/2026-08-19-console-user-interaction-design.md` §5 — where the two disagree (timeout), the PRD wins: **A7 default is `0` = no deadline** (ADR-067), not the spec's 240s.

## Global Constraints

- **No new module resident at UI-ready.** ADR-097's census (`Tests/Performance/test_ui_ready_module_census.py`) sits at the 972 cap on dev; +1 fails CI. `console_chat_controller`, `local_tool_provider`, `chat_task_cards`, `chat_screen`, `console_runtime`, `console_agent_bridge`, `chat_screen_state` are ALL resident at boot. Therefore: `Agents/ask_user_questions.py` is imported only inside functions that run at tool-build/call time; `Widgets/Chat_Widgets/chat_question_card.py` is imported only inside `ChatTaskCards._question_card()` and mounted on first use. The `Answered` message class lives in `chat_task_cards.py` so the screen's `@on` handler needs no card import.
- **No new `logger.*` call in production code.** The production diagnostic inventory is part of the REQUIRED "Derived artifacts" CI check; one new call fails it. Use `contextlib.suppress(Exception)` where the sibling rounds log at teardown.
- **CSS ships as `BUNDLED_CSS`, never `DEFAULT_CSS`** (`Tests/UI/test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist`). After editing any `BUNDLED_CSS`, run `python tldw_chatbook/css/build_css.py` and commit `tldw_chatbook/css/widget_defaults_self.tcss`.
- **Every new public method gets a Google-style docstring** (Qodo rule violation otherwise; cost M1 three findings).
- **Textual trap:** `is_mounted` is still `False` inside `on_mount`; guard repaint on the children existing (`query_one` + `NoMatches`), never on `is_mounted`.
- **Textual trap:** a method named `_render` on a widget shadows `Widget._render()`; never use that name.
- **Test interpreter:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider <files>` run from the worktree root (worktrees have no `.venv`). Run pytest BLOCKING (never in the background) with a long timeout.
- **Attribution:** every commit message ends with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_018q5PsHwn5kgHPmNwX9DKoo`.
- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/ask-user`, branch `feat/console-ask-user`, base `origin/dev` @ `50c9918935`. Check `git rev-parse --show-toplevel` before every commit — subagents have committed to the wrong tree before.
- `console_chat_controller.py` is ~24k lines. NEVER read it whole; use `grep -n` then `sed -n 'a,bp'` windows of ≤150 lines. Line numbers below are from `50c9918935` and drift; anchor on the quoted code.

---

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Agents/ask_user_questions.py` (new) | Pure: bounds (A1), validation, tool description (A13), JSON schema, result shapes (A6), exceptions. No I/O, no Textual. |
| `tldw_chatbook/Agents/local_tool_provider.py` | `LocalToolSpec.gate_exempt`; permission-layer exemption; `ASK_USER_GATE_KEY`; `AskUserCallback`; the `ask_user` spec + handler; `ask_user=` kwarg threaded to `_default_specs`. |
| `tldw_chatbook/Agents/builtin_tool_gate.py` | The hand-listed gate row (default ON). |
| `tldw_chatbook/Chat/console_agent_bridge.py` | `format_question_marker` + `append_question_marker` (A14). |
| `tldw_chatbook/Chat/console_chat_controller.py` | The question round: state, `request_user_questions`, `resolve_pending_question`, `pending_question_ids`, marshal/remount, revocation leg, timeout resolver, `_ask_user_wiring`. |
| `tldw_chatbook/Chat/console_runtime.py` | `set_pending_question` view-hook slot. |
| `tldw_chatbook/UI/Screens/chat_screen_state.py` | `TaskResumeState.pending_question` (live-only, dropped on restore like the skill fields). |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py` | `QuestionAnswered` message; lazy mount + routing of the question card; display gate. |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py` (new) | The card: sections, Other input, keyboard, submit, deadline copy, request-id round-trip, selection-preserving re-sync. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Setter, hook entry, `@on(ChatTaskCards.QuestionAnswered)` forwarder, inspector focus fallback. |
| `tldw_chatbook/config.py` | Documented `[console] ask_user_timeout_seconds` line. |
| `Docs/User_Guide/console/agent-runs-and-tools.md` | "Questions from the agent" section. |
| Tests | `Tests/Agents/test_ask_user_questions.py`, `Tests/Agents/test_ask_user_tool.py`, `Tests/Chat/test_console_ask_user_round.py`, `Tests/UI/test_chat_question_card.py`. |

---

### Task 1: Pure validation, description, schema, and result shapes

**Files:**
- Create: `tldw_chatbook/Agents/ask_user_questions.py`
- Test: `Tests/Agents/test_ask_user_questions.py`

**Interfaces:**
- Produces: `validate_questions(raw: object) -> list[dict[str, Any]]` (raises `AskUserValidationError`), `AskUserValidationError(ValueError)`, `AskUserBusyRefusal(ValueError)`, `ASK_USER_DESCRIPTION: str`, `ASK_USER_PARAMETERS: dict`, `ASK_USER_REFUSAL_COPY: str`, `busy_result() -> dict`, `unanswered_result(reason: str) -> dict`, `answered_result(answers: list[dict]) -> dict`, `empty_answers(questions) -> list[dict]`, constants `MAX_QUESTIONS=4, MIN_OPTIONS=2, MAX_OPTIONS=4, MAX_QUESTION_CHARS=500, MAX_HEADER_CHARS=12, MAX_LABEL_CHARS=100, MAX_DESCRIPTION_CHARS=300`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Agents/test_ask_user_questions.py
"""PRD Feature A1/A2/A6: bounds, validation, and result shapes for ask_user."""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.ask_user_questions import (
    ASK_USER_DESCRIPTION,
    ASK_USER_PARAMETERS,
    AskUserValidationError,
    answered_result,
    busy_result,
    empty_answers,
    unanswered_result,
    validate_questions,
)


def _q(**overrides):
    base = {
        "question": "Which database?",
        "header": "Database",
        "multiSelect": False,
        "options": [
            {"label": "Postgres", "description": "Managed, relational"},
            {"label": "SQLite", "description": "Embedded"},
        ],
    }
    base.update(overrides)
    return base


def test_valid_call_round_trips_cleaned_copies():
    out = validate_questions([_q()])
    assert out == [_q()]
    assert out[0] is not _q()  # defensive copy, not the caller's dict


def test_multiselect_defaults_false_and_description_optional():
    out = validate_questions([_q(multiSelect=True, options=[{"label": "a"}, {"label": "b"}])])
    assert out[0]["multiSelect"] is True
    assert out[0]["options"][0] == {"label": "a", "description": ""}
    out = validate_questions([{"question": "q", "header": "h", "options": [{"label": "a"}, {"label": "b"}]}])
    assert out[0]["multiSelect"] is False


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ([], "1 to 4"),
        ([_q()] * 5, "1 to 4"),
        ([_q(options=[{"label": "only"}])], "2 to 4"),
        ([_q(header="thirteen chars")], "12 characters"),
        ([_q(question="x" * 501)], "500 characters"),
        ([_q(question="   ")], "blank"),
        ([_q(multiSelect="yes")], "multiSelect"),
        ([_q(bogus=1)], "unknown keys"),
        ([_q(options=[{"label": "a"}, {"label": "A"}])], "repeats option label"),
        ([_q(options=[{"label": "a", "extra": 1}, {"label": "b"}])], "unknown keys"),
        ("not a list", "must be a list"),
        ([_q(question="\udcff bad")], "UTF-8"),
    ],
)
def test_rejections_name_the_problem(raw, fragment):
    with pytest.raises(AskUserValidationError) as excinfo:
        validate_questions(raw)
    assert fragment in str(excinfo.value)


def test_control_characters_flatten_and_newlines_collapse():
    out = validate_questions([_q(question="line one\nline\ttwo\x07")])
    assert out[0]["question"] == "line one line two"


def test_schema_never_declares_other_and_pins_the_bounds():
    items = ASK_USER_PARAMETERS["properties"]["questions"]["items"]
    assert ASK_USER_PARAMETERS["properties"]["questions"]["maxItems"] == 4
    assert items["properties"]["options"]["minItems"] == 2
    assert items["properties"]["options"]["maxItems"] == 4
    assert "other" not in str(ASK_USER_PARAMETERS).lower()
    assert items["additionalProperties"] is False


def test_description_spends_its_words_on_restraint():
    text = ASK_USER_DESCRIPTION.lower()
    assert "do not ask" in text
    assert "conventional default" in text
    assert "busy" in text


def test_result_shapes():
    assert busy_result()["answered"] is False and busy_result()["reason"] == "busy"
    assert "instruction" in busy_result()
    assert unanswered_result("timeout") == {"answered": False, "reason": "timeout"}
    answers = empty_answers(validate_questions([_q()]))
    assert answers == [
        {"question": "Which database?", "selected": [], "other_text": None, "unanswered": True}
    ]
    assert answered_result(answers) == {"answered": True, "answers": answers}
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Agents/test_ask_user_questions.py`
Expected: FAIL with `ModuleNotFoundError: tldw_chatbook.Agents.ask_user_questions`

- [ ] **Step 3: Write the module**

```python
# tldw_chatbook/Agents/ask_user_questions.py
"""Bounds, validation, and result shapes for the ``ask_user`` tool.

PRD Feature A (A1, A2, A6, A9, A13). Pure: no I/O, no Textual. The payload is
model-controlled text that goes straight to a card, so every bound in A1 is
enforced here before anything reaches a widget -- the same posture as
``SessionTodoStore``. Imported lazily by ``local_tool_provider`` so this
module never rides the boot path (ADR-097).
"""

from __future__ import annotations

import re
from typing import Any

MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4
MAX_QUESTION_CHARS = 500
MAX_HEADER_CHARS = 12
MAX_LABEL_CHARS = 100
MAX_DESCRIPTION_CHARS = 300

#: A9: the second consecutive ``busy`` in one run is refused outright.
MAX_CONSECUTIVE_BUSY = 2

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_NEWLINE_RE = re.compile(r"[\r\n\t]+")
_QUESTION_KEYS = frozenset({"question", "header", "multiSelect", "options"})
_OPTION_KEYS = frozenset({"label", "description"})

ASK_USER_DESCRIPTION = (
    "Ask the user up to 4 multiple-choice questions and wait for the answers. "
    "Use it ONLY for a decision that is genuinely the user's to make: a "
    "preference, a trade-off between valid designs, or something neither the "
    "code nor the conversation can tell you. Do not ask when a conventional "
    "default exists, when the answer is discoverable by reading the code or "
    "running a tool, when you can proceed and state your assumption, or to "
    "confirm a plan you already have. Batch related questions into ONE call "
    "instead of asking several times. Each question offers 2-4 options; the "
    "user can always type a free-text 'Other' answer instead. The result lists "
    "the selected labels per question; 'unanswered' marks questions the user "
    "skipped, and 'answered': false with a reason means no answer will come. "
    "If the reason is 'busy', another question is already waiting for the "
    "user: proceed without asking again this turn."
)

ASK_USER_REFUSAL_COPY = (
    "ask_user refused: it returned 'busy' twice in a row in this run. A "
    "question is already waiting for the user. Do not call ask_user again "
    "this turn; proceed without the answer."
)

_OPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "minLength": 1, "maxLength": MAX_LABEL_CHARS},
        "description": {"type": "string", "maxLength": MAX_DESCRIPTION_CHARS},
    },
    "required": ["label"],
    "additionalProperties": False,
}

ASK_USER_PARAMETERS = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_QUESTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QUESTION_CHARS,
                    },
                    "header": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_HEADER_CHARS,
                    },
                    "multiSelect": {"type": "boolean"},
                    "options": {
                        "type": "array",
                        "minItems": MIN_OPTIONS,
                        "maxItems": MAX_OPTIONS,
                        "items": _OPTION_SCHEMA,
                    },
                },
                "required": ["question", "header", "options"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


class AskUserValidationError(ValueError):
    """A rejected ``ask_user`` call; the message is the tool error the model sees."""


class AskUserBusyRefusal(ValueError):
    """A9: too many consecutive ``busy`` results in one run -- refused outright."""


def _clean_text(value: object, *, field: str, limit: int, required: bool = True) -> str:
    """Return ``value`` flattened for render, or raise with the field named.

    Args:
        value: The raw model-supplied value.
        field: Human-readable field path for the error message.
        limit: Maximum length AFTER cleaning.
        required: Whether a blank value is an error.

    Returns:
        The cleaned string: newlines/tabs collapsed to one space, other
        control characters removed, surrounding whitespace stripped.

    Raises:
        AskUserValidationError: Wrong type, invalid UTF-8, blank, or over limit.
    """
    if not isinstance(value, str):
        raise AskUserValidationError(f"{field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AskUserValidationError(f"{field} is not valid UTF-8") from exc
    cleaned = _CONTROL_RE.sub("", _NEWLINE_RE.sub(" ", value))
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    if required and not cleaned:
        raise AskUserValidationError(f"{field} must not be blank")
    if len(cleaned) > limit:
        raise AskUserValidationError(f"{field} exceeds {limit} characters")
    return cleaned


def validate_questions(raw: object) -> list[dict[str, Any]]:
    """Validate an ``ask_user`` call's ``questions`` and return cleaned copies.

    Args:
        raw: The call's ``questions`` value, straight from the model.

    Returns:
        1-4 question dicts, each ``{"question", "header", "multiSelect",
        "options": [{"label", "description"}]}`` with every string cleaned.

    Raises:
        AskUserValidationError: Any bound in PRD A1 violated. The message
            names the question/option index and the rule.
    """
    if not isinstance(raw, list):
        raise AskUserValidationError("questions must be a list")
    if not 1 <= len(raw) <= MAX_QUESTIONS:
        raise AskUserValidationError(f"questions must hold 1 to {MAX_QUESTIONS} items")
    questions: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise AskUserValidationError(f"question {index} must be an object")
        unknown = set(item) - _QUESTION_KEYS
        if unknown:
            raise AskUserValidationError(
                f"question {index} has unknown keys: {sorted(unknown)}"
            )
        for key in ("question", "header", "options"):
            if key not in item:
                raise AskUserValidationError(f"question {index} is missing {key}")
        multi = item.get("multiSelect", False)
        if not isinstance(multi, bool):
            raise AskUserValidationError(f"question {index}: multiSelect must be a boolean")
        options = item["options"]
        if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
            raise AskUserValidationError(
                f"question {index}: options must hold {MIN_OPTIONS} to {MAX_OPTIONS} items"
            )
        cleaned_options: list[dict[str, str]] = []
        seen: set[str] = set()
        for opt_index, option in enumerate(options, start=1):
            where = f"question {index} option {opt_index}"
            if not isinstance(option, dict):
                raise AskUserValidationError(f"{where} must be an object")
            unknown_option = set(option) - _OPTION_KEYS
            if unknown_option:
                raise AskUserValidationError(
                    f"{where} has unknown keys: {sorted(unknown_option)}"
                )
            label = _clean_text(option.get("label"), field=f"{where} label", limit=MAX_LABEL_CHARS)
            if label.casefold() in seen:
                raise AskUserValidationError(f"question {index} repeats option label {label!r}")
            seen.add(label.casefold())
            description = _clean_text(
                option.get("description", ""),
                field=f"{where} description",
                limit=MAX_DESCRIPTION_CHARS,
                required=False,
            )
            cleaned_options.append({"label": label, "description": description})
        questions.append(
            {
                "question": _clean_text(
                    item["question"], field=f"question {index} text", limit=MAX_QUESTION_CHARS
                ),
                "header": _clean_text(
                    item["header"], field=f"question {index} header", limit=MAX_HEADER_CHARS
                ),
                "multiSelect": multi,
                "options": cleaned_options,
            }
        )
    return questions


def busy_result() -> dict[str, Any]:
    """A9: the immediate result when a question is already live in the session."""
    return {
        "answered": False,
        "reason": "busy",
        "instruction": (
            "A question is already waiting for the user in this session. Do not "
            "retry ask_user now: proceed without the answer, or ask again in a "
            "later turn."
        ),
    }


def unanswered_result(reason: str) -> dict[str, Any]:
    """A6: the result for a round that ended without answers.

    Args:
        reason: ``"timeout"`` or ``"cancelled"``.
    """
    return {"answered": False, "reason": reason}


def answered_result(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """A6: wrap per-question answers into the tool result."""
    return {"answered": True, "answers": [dict(answer) for answer in answers]}


def empty_answers(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One ``unanswered`` entry per question -- the shape a blank submit yields."""
    return [
        {
            "question": str(question.get("question", "")),
            "selected": [],
            "other_text": None,
            "unanswered": True,
        }
        for question in questions
    ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Agents/test_ask_user_questions.py`
Expected: all PASS. If `test_control_characters_flatten_and_newlines_collapse` fails on double spaces, the `re.sub(r" {2,}", " ", ...)` line is what collapses `"line\ttwo"` → `"line two"`.

- [ ] **Step 5: Lint and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Agents/ask_user_questions.py Tests/Agents/test_ask_user_questions.py
git add tldw_chatbook/Agents/ask_user_questions.py Tests/Agents/test_ask_user_questions.py
git commit -m "feat(agents): ask_user bounds, validation, and result shapes (PRD A1/A2/A6/A13)"
```

---

### Task 2: Permission-layer exemption for a spec

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` — `LocalToolSpec` (the frozen dataclass at `class LocalToolSpec:`), `_verdict_for` (`def _verdict_for(self, name: str, args: dict, run_id: str) -> _LocalGateDecision:`), `_resolve_pending_gate` (`def _resolve_pending_gate(`).
- Test: `Tests/Agents/test_ask_user_tool.py` (created here, extended in Task 3)

**Interfaces:**
- Produces: `LocalToolSpec.gate_exempt: bool = False`. An exempt spec executes under permission state `"ask"` with no approval callback, and `_resolve_pending_gate` returns `(None, False)` for it, so batch review never lists it.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Agents/test_ask_user_tool.py
"""PRD Feature A: the ask_user LocalToolSpec, its gate, and its exemption."""

from __future__ import annotations

import json

import pytest

import tldw_chatbook.Agents.local_tool_provider as provider_module
from Tests.Agents.test_local_tool_provider import ASK, make_provider
from tldw_chatbook.Agents.local_tool_provider import (
    LocalApprovalEffect,
    LocalToolExposure,
    LocalToolSpec,
)


def _exempt_spec(name: str = "ping") -> LocalToolSpec:
    return LocalToolSpec(
        name=name,
        description="pong",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda args: "pong",
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(),
        gate_exempt=True,
    )


def _gated_spec(name: str = "gated") -> LocalToolSpec:
    return LocalToolSpec(
        name=name,
        description="gated",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda args: "ran",
        exposure=LocalToolExposure.CONSOLE_ONLY,
        approval_effects=(LocalApprovalEffect.MUTATES_LOCAL,),
    )


def test_gate_exempt_defaults_false():
    assert _gated_spec().gate_exempt is False


def test_exempt_spec_runs_under_ask_with_no_approval_callback(tmp_path):
    provider = make_provider(state=ASK, root=tmp_path, specs=[_exempt_spec(), _gated_spec()])
    result = provider.invoke("local:ping", {})
    assert result.ok is True and result.content == "pong"
    refused = provider.invoke("local:gated", {})
    assert refused.ok is False, "a non-exempt sibling still needs approval"


def test_exempt_spec_never_reaches_batch_review(tmp_path):
    provider = make_provider(state=ASK, root=tmp_path, specs=[_exempt_spec()])
    gate, resolve_failed = provider._resolve_pending_gate(
        "ping", {}, provider.hub_tool_for("ping")
    )
    assert gate is None and resolve_failed is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Agents/test_ask_user_tool.py`
Expected: FAIL with `TypeError: LocalToolSpec.__init__() got an unexpected keyword argument 'gate_exempt'`. If `make_provider(... root=tmp_path)` is not how that helper takes the root, open `Tests/Agents/test_local_tool_provider.py::make_provider` (line ~132) and match its kwarg name.

- [ ] **Step 3: Add the field and the two early returns**

In `LocalToolSpec`, after `approval_arguments: Callable[...] | None = None`:

```python
    #: PRD Feature A (A12): the spec is exempt from the Allow/Ask/Off
    #: permission layer -- it never raises an approval card and never
    #: appears in batch review. Reserved for tools that touch only the
    #: user's ATTENTION (``ask_user``): asking whether the agent may ask a
    #: question is two interruptions for one. Anything that touches data,
    #: disk, or network must leave this False.
    gate_exempt: bool = False
```

In `__post_init__`, after the `execution_policy` check:

```python
        if not isinstance(self.gate_exempt, bool):
            raise ValueError("LocalToolSpec gate_exempt must be a bool")
```

In `_verdict_for`, as the FIRST statements of the body (before `hub = self.hub_tool_for(name)`):

```python
        spec = self._specs.get(name)
        if spec is not None and spec.gate_exempt:
            return _LocalGateDecision(verdict="allow", approval_consumed=False)
```

In `_resolve_pending_gate`, as the FIRST statements of the body (before `try: state = self._resolve_state(hub)`):

```python
        exempt_spec = self._specs.get(name)
        if exempt_spec is not None and exempt_spec.gate_exempt:
            return None, False
```

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Agents/test_ask_user_tool.py Tests/Agents/test_local_tool_provider.py`
Expected: all PASS (the existing provider suite must stay green — the exemption is a no-op for every existing spec).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_ask_user_tool.py
git commit -m "feat(agents): LocalToolSpec.gate_exempt for attention-only tools (PRD A12)"
```

---

### Task 3: The `ask_user` spec, handler, and gate row

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` — near `WEB_DEEP_SEARCH_GATE_KEY = "web_deep_search_enabled"`; `LocalToolProvider.__init__` (`todo_store: SessionTodoStore | None = None,` kwarg block); the three `_default_specs(` call sites in `__init__`; `_default_specs` signature and the end of its body (after the `if todo_store is not None:` block that appends `todo_*`).
- Modify: `tldw_chatbook/Agents/builtin_tool_gate.py` — `all_tool_gates()` and `_gate_key_pairs()`.
- Test: `Tests/Agents/test_ask_user_tool.py` (extend)

**Interfaces:**
- Consumes: Task 1's `validate_questions`, `ASK_USER_DESCRIPTION`, `ASK_USER_PARAMETERS`; Task 2's `gate_exempt`.
- Produces: `ASK_USER_GATE_KEY = "ask_user_enabled"`; `AskUserCallback = Callable[[list[dict[str, Any]]], dict[str, Any]]`; `LocalToolProvider(..., ask_user: AskUserCallback | None = None)`; a `LocalToolSpec` named `"ask_user"` registered iff `ask_user is not None` AND `[tools] ask_user_enabled` (default True). The handler validates, calls `ask_user(questions)`, and returns the result as compact JSON text.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Agents/test_ask_user_tool.py`)

```python
from tldw_chatbook.Agents.ask_user_questions import AskUserBusyRefusal
from tldw_chatbook.Agents.builtin_tool_gate import _gate_key_pairs, all_tool_gates


def _names(provider) -> set[str]:
    return {spec.name for spec in provider.specs_for_exposure(LocalToolExposure.CONSOLE_ONLY)}


def test_ask_user_registered_only_when_a_callback_is_supplied(tmp_path):
    assert "ask_user" not in _names(make_provider(root=tmp_path))
    provider = make_provider(root=tmp_path, ask_user=lambda questions: {"answered": False, "reason": "cancelled"})
    assert "ask_user" in _names(provider)
    spec = next(s for s in provider.specs_for_exposure(LocalToolExposure.CONSOLE_ONLY) if s.name == "ask_user")
    assert spec.gate_exempt is True and spec.approval_effects == () and spec.tags == ()


def test_ask_user_absent_when_the_gate_is_off(tmp_path, monkeypatch):
    def fake_setting(section, key, default=None):
        if (section, key) == ("tools", "ask_user_enabled"):
            return False
        return default

    monkeypatch.setattr(provider_module, "get_cli_setting", fake_setting)
    provider = make_provider(root=tmp_path, ask_user=lambda questions: {})
    assert "ask_user" not in _names(provider)


def test_handler_validates_then_hands_cleaned_questions_to_the_callback(tmp_path):
    seen = []

    def _ask(questions):
        seen.append(questions)
        return {"answered": True, "answers": []}

    provider = make_provider(state=ASK, root=tmp_path, ask_user=_ask)
    raw = {"questions": [{"question": "Which?", "header": "Pick", "options": [{"label": "a"}, {"label": "b"}]}]}
    result = provider.invoke("local:ask_user", raw)
    assert result.ok is True, result.error
    assert json.loads(result.content) == {"answered": True, "answers": []}
    assert seen == [[{"question": "Which?", "header": "Pick", "multiSelect": False,
                      "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]}]]


def test_handler_rejects_bad_calls_with_an_actionable_error_and_never_calls_back(tmp_path):
    calls = []
    provider = make_provider(state=ASK, root=tmp_path, ask_user=lambda q: calls.append(q) or {})
    result = provider.invoke("local:ask_user", {"questions": [{"question": "q", "header": "h", "options": [{"label": "one"}]}]})
    assert result.ok is False and "2 to 4" in (result.error or "")
    assert calls == []


def test_busy_refusal_from_the_callback_is_a_tool_error(tmp_path):
    def _ask(questions):
        raise AskUserBusyRefusal("ask_user refused: busy twice")

    provider = make_provider(state=ASK, root=tmp_path, ask_user=_ask)
    result = provider.invoke("local:ask_user", {"questions": [{"question": "q", "header": "h", "options": [{"label": "a"}, {"label": "b"}]}]})
    assert result.ok is False and "busy twice" in (result.error or "")


def test_gate_row_is_enumerated_and_defaults_on():
    gate = next(g for g in all_tool_gates() if g.tool_name == "ask_user")
    assert (gate.section, gate.key, gate.group) == ("tools", "ask_user_enabled", "local")
    assert gate.enabled is True
    assert [(g.section, g.key) for g in all_tool_gates()] == _gate_key_pairs()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Agents/test_ask_user_tool.py`
Expected: the new tests FAIL (`unexpected keyword argument 'ask_user'`, `StopIteration` on the gate row).

- [ ] **Step 3: Implement in `local_tool_provider.py`**

Next to `WEB_DEEP_SEARCH_GATE_KEY`:

```python
#: PRD Feature A (A12): `[tools] ask_user_enabled`, default ON -- a deliberate
#: exception to the off-by-default gates. Every other gate is off because the
#: tool touches data, disk, or network; this one touches only the user's
#: attention, and a tool whose purpose is to initiate contact cannot be
#: discovered while invisible. Hand-listed in `all_tool_gates()` like
#: `web_deep_search`.
ASK_USER_GATE_KEY = "ask_user_enabled"
ASK_USER_DEFAULT_ENABLED = True

#: The controller's blocking ask: cleaned questions in, PRD A6 result dict out.
AskUserCallback = Callable[[list[dict[str, Any]]], dict[str, Any]]
```

The handler factory (place it after `_make_todo_list_handler` / before `_default_specs`):

```python
def _make_ask_user_handler(ask_user: AskUserCallback) -> Callable[[dict], str]:
    """Build ``ask_user`` for one Console session's ask callback.

    Validation (PRD A1/A2) runs BEFORE the callback so a rejected call never
    mounts anything; the callback's own ``AskUserBusyRefusal`` (A9)
    propagates as a tool error through the provider's handler boundary.

    Args:
        ask_user: The controller's ``request_user_questions`` bound to the
            run's session.

    Returns:
        The spec handler: ``args`` dict in, compact JSON text out.
    """
    from .ask_user_questions import validate_questions

    def _handler(args: dict) -> str:
        values = _exact_task_args(args, allowed={"questions"}, required={"questions"})
        questions = validate_questions(values["questions"])
        return _todo_json(ask_user(questions))

    return _handler
```

`_default_specs` signature gains `ask_user: AskUserCallback | None = None,` after `on_todo_change`. At the END of `_default_specs`'s body (after the `if todo_store is not None:` block, before `return specs`):

```python
    if ask_user is not None and coerce_bool_setting(
        get_cli_setting("tools", ASK_USER_GATE_KEY, ASK_USER_DEFAULT_ENABLED),
        ASK_USER_DEFAULT_ENABLED,
    ):
        from .ask_user_questions import ASK_USER_DESCRIPTION, ASK_USER_PARAMETERS

        specs.append(
            LocalToolSpec(
                name="ask_user",
                description=ASK_USER_DESCRIPTION,
                parameters=ASK_USER_PARAMETERS,
                handler=_make_ask_user_handler(ask_user),
                exposure=LocalToolExposure.CONSOLE_ONLY,
                approval_effects=(),
                tags=(),
                gate_exempt=True,
            )
        )
```

`LocalToolProvider.__init__`: add `ask_user: AskUserCallback | None = None,` after `on_todo_change: TodoChangeCallback | None = None,`; pass `ask_user=ask_user,` to ALL THREE `_default_specs(` calls (grep `_default_specs(` inside `__init__` — the `elif self._admitted_roots is None`, `elif not self._admitted_roots`, and the per-authority loop).

- [ ] **Step 4: Implement the gate row in `builtin_tool_gate.py`**

In `all_tool_gates()`, after the `web_deep_search` `gates.append(...)` and before `return gates`:

```python
    from .local_tool_provider import ASK_USER_DEFAULT_ENABLED, ASK_USER_GATE_KEY

    gates.append(
        ToolGate(
            section="tools",
            key=ASK_USER_GATE_KEY,
            tool_name="ask_user",
            description=_ASK_USER_DESCRIPTION,
            enabled=coerce_bool_setting(
                get_cli_setting("tools", ASK_USER_GATE_KEY, ASK_USER_DEFAULT_ENABLED),
                ASK_USER_DEFAULT_ENABLED,
            ),
            group="local",
        )
    )
```

Module-level constant beside `_WEB_DEEP_SEARCH_DESCRIPTION`:

```python
_ASK_USER_DESCRIPTION = (
    "Lets an agent ask you up to four multiple-choice questions on a card above "
    "the Console transcript. On by default: it touches only your attention, "
    "never your data. Turn it off to remove the tool from every agent."
)
```

In `_gate_key_pairs()`, after `pairs.append(("tools", WEB_DEEP_SEARCH_GATE_KEY))`:

```python
    from .local_tool_provider import ASK_USER_GATE_KEY

    pairs.append(("tools", ASK_USER_GATE_KEY))
```

(Move the import to the existing `from .local_tool_provider import WEB_DEEP_SEARCH_GATE_KEY` line in each function: `from .local_tool_provider import ASK_USER_GATE_KEY, WEB_DEEP_SEARCH_GATE_KEY`.)

- [ ] **Step 5: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Agents/test_ask_user_tool.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_builtin_tool_gate.py`
Expected: all PASS. `test_builtin_tool_gate.py` has a count/parity test at line ~637 comparing `all_tool_gates()` to `_gate_key_pairs()` — both were extended, so it stays green. If a test in `test_builtin_tool_gate.py` pins the exact NUMBER of local-group rows, update that literal and say so in the commit.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/builtin_tool_gate.py Tests/Agents/test_ask_user_tool.py
git commit -m "feat(agents): register ask_user when the Console supplies an ask callback (PRD A1/A9/A12/A13)"
```

---

### Task 4: Transcript marker (A14)

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` — after `def format_todo_marker(` (module-level, ~line 1500) and after `def append_todo_marker(` (method, ~line 8069).
- Test: `Tests/Chat/test_console_ask_user_round.py` (created here, extended in Task 5)

**Interfaces:**
- Produces: `format_question_marker(asked_by: str, questions: list[dict[str, Any]], result: dict[str, Any]) -> str`; `ConsoleAgentBridge.append_question_marker(session_id: str, text: str) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_ask_user_round.py
"""PRD Feature A: the question round (A5-A7, A9-A11, A14) and its marker."""

from __future__ import annotations

from tldw_chatbook.Chat.console_agent_bridge import format_question_marker


def _questions():
    return [
        {"question": "Which DB?", "header": "DB", "multiSelect": False,
         "options": [{"label": "Postgres", "description": ""}, {"label": "SQLite", "description": ""}]},
        {"question": "Regions?", "header": "Region", "multiSelect": True,
         "options": [{"label": "eu", "description": ""}, {"label": "us", "description": ""}]},
    ]


def test_marker_lists_each_question_with_its_answer():
    result = {"answered": True, "answers": [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": [], "other_text": "apac only", "unanswered": False},
    ]}
    text = format_question_marker("agent", _questions(), result)
    assert text.splitlines() == [
        "? Questions from the agent (2):",
        "  Which DB? → Postgres",
        "  Regions? → other: apac only",
    ]


def test_marker_names_a_sub_agent_and_flattens_control_text():
    result = {"answered": True, "answers": [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": [], "other_text": None, "unanswered": True},
    ]}
    questions = _questions()
    questions[0]["question"] = "Which\nDB?\x07"
    text = format_question_marker("sub-agent", questions, result)
    assert text.splitlines()[0] == "? Questions from a sub-agent (2):"
    assert text.splitlines()[1] == "  Which DB? → Postgres"
    assert text.splitlines()[2] == "  Regions? → (unanswered)"


def test_marker_for_timeout_and_cancel_stamps_every_question():
    text = format_question_marker("agent", _questions(), {"answered": False, "reason": "timeout"})
    assert text.splitlines()[1:] == ["  Which DB? → (timed out)", "  Regions? → (timed out)"]
    text = format_question_marker("agent", _questions(), {"answered": False, "reason": "cancelled"})
    assert text.splitlines()[2] == "  Regions? → (cancelled)"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Chat/test_console_ask_user_round.py`
Expected: FAIL with `ImportError: cannot import name 'format_question_marker'`

- [ ] **Step 3: Implement**

After `format_todo_marker` in `console_agent_bridge.py`:

```python
def format_question_marker(
    asked_by: str, questions: list[dict[str, Any]], result: dict[str, Any]
) -> str:
    """Render the transcript record of a resolved ``ask_user`` round (PRD A14).

    Same conventions as ``format_todo_marker``: display-only, one header line
    then one line per question, every label passed through
    ``_sanitize_task_marker_label`` (newlines and terminal controls
    flattened, truncated) because the text is model-controlled.

    Args:
        asked_by: ``"agent"`` or ``"sub-agent"``.
        questions: The validated questions the round showed.
        result: The PRD A6 result dict the tool returned.

    Returns:
        The marker text.
    """
    who = "a sub-agent" if asked_by == "sub-agent" else "the agent"
    lines = [f"? Questions from {who} ({len(questions)}):"]
    answers = result.get("answers") if result.get("answered") else None
    reason = str(result.get("reason") or "cancelled")
    stamp = {"timeout": "(timed out)", "cancelled": "(cancelled)"}.get(reason, "(cancelled)")
    for index, question in enumerate(questions):
        label = _sanitize_task_marker_label(str(question.get("question") or ""))
        if answers is None:
            outcome = stamp
        else:
            answer = answers[index] if index < len(answers) else {}
            selected = [str(item) for item in (answer.get("selected") or [])]
            other = answer.get("other_text")
            parts = []
            if selected:
                parts.append(", ".join(selected))
            if other:
                parts.append(f"other: {other}")
            outcome = "; ".join(parts) if parts else "(unanswered)"
            outcome = _sanitize_task_marker_label(outcome)
        lines.append(f"  {label} → {outcome}")
    return "\n".join(lines)
```

After `append_todo_marker` in `ConsoleAgentBridge`:

```python
    def append_question_marker(self, session_id: str, text: str) -> None:
        """Surface a resolved ``ask_user`` round in the transcript (PRD A14).

        Same seam as ``append_todo_marker``: called on the agent worker
        thread, in-memory append with ``persist=False``; written on resolve
        only, so a question pending when the app is killed leaves nothing.

        Args:
            session_id: The round's owning session.
            text: ``format_question_marker``'s output.
        """
        self._append_marker(
            session_id,
            text,
            activity_presentation=ConsoleActivityPresentation(
                "feedback", "Question answered", "done"
            ),
        )
```

(`"feedback"` is in `_CONSOLE_ACTIVITY_KINDS`; the third positional is the same `"done"` state `append_todo_marker` passes.)

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Chat/test_console_ask_user_round.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_ask_user_round.py
git commit -m "feat(console): transcript marker for a resolved ask_user round (PRD A14)"
```

---

### Task 5: The question round in the controller

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`:
  - imports: ensure `import contextlib` exists (grep `^import contextlib`); `current_run_actor` is already imported (`from tldw_chatbook.Agents.run_context import current_run_actor, current_run_id`).
  - module constants: after `_DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS = 0.0`.
  - `__init__`: after `self._parked_worktree_merge_payloads: dict[str, dict[str, Any]] = {}`.
  - new methods after `def resolve_pending_skill_script(` / `pending_skill_script_ids` block (before `def request_worktree_merge_confirm(`).
  - `_todo_wiring` neighbourhood: new `_ask_user_wiring` after `def _todo_wiring(`; `LocalToolProvider(` construction: add `**self._ask_user_wiring(session_id),` right after `**self._todo_wiring(session_id),`.
  - the three activation sites: after each `self._remount_parked_worktree_merge(<id>)` line (`session.id` in `new_session`, `session_id` in `switch_session`, `new_active_id` in `close_session`) add `self._remount_parked_question(<id>)`.
  - `revoke_approval_rounds_for_run`: add the question leg.
- Test: `Tests/Chat/test_console_ask_user_round.py` (extend)

**Interfaces:**
- Consumes: Task 1 (`AskUserBusyRefusal`, `ASK_USER_REFUSAL_COPY`, `MAX_CONSECUTIVE_BUSY`, `busy_result`, `unanswered_result`, `answered_result`, `empty_answers`), Task 4 (`format_question_marker`, `append_question_marker`).
- Produces: attributes `set_pending_question: Callable[[dict | None], None] | None`, `ask_user_timeout_seconds: Callable[[], float] | None`, `_pending_question_rounds`, `_pending_question_lock`, `_parked_question_payloads`, `_question_bounces: dict[str, int]`; methods `request_user_questions(questions, *, session_id=None) -> dict`, `resolve_pending_question(answers, request_id=None) -> None`, `pending_question_ids() -> list[str]`, `_marshal_pending_question(payload) -> None`, `_remount_parked_question(session_id) -> None`, `_revoke_question_rounds(run_id) -> list[tuple[str, str | None]]`, `_resolve_ask_user_timeout_seconds() -> float`, `_ask_user_wiring(session_id) -> dict[str, Any]`.
- Payload handed to the UI: `{"questions": [...], "asked_by": "agent"|"sub-agent", "timeout_seconds": float, "request_id": str, "session_id": str, "deadline_monotonic": float | None}`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_ask_user_round.py`)

```python
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_skill_script_confirm import _FakeApp, _wait_until
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Agents.ask_user_questions import AskUserBusyRefusal
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController


@pytest.fixture
def make_controller():
    made = []

    def _make():
        store = persisted_console_store()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        controller.app = _FakeApp()
        controller.pending_question_payloads = []
        controller.set_pending_question = controller.pending_question_payloads.append
        made.append(controller)
        return controller

    yield _make
    for controller in made:
        controller.begin_shutdown()


def _start(controller, questions, *, session_id=None, run_id=""):
    box = {}

    def worker():
        with use_run_id(run_id):
            try:
                box["result"] = controller.request_user_questions(questions, session_id=session_id)
            except Exception as exc:  # noqa: BLE001 - the test reads it
                box["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    return thread, box


def test_no_ui_returns_cancelled_immediately(make_controller):
    controller = make_controller()
    controller.set_pending_question = None
    assert controller.request_user_questions(_questions()) == {"answered": False, "reason": "cancelled"}


def test_answer_round_trip_and_marker(make_controller):
    controller = make_controller()
    markers = []
    controller._agent_bridge = SimpleNamespace(
        append_question_marker=lambda sid, text: markers.append((sid, text))
    )
    thread, box = _start(controller, _questions())
    _wait_until(lambda: bool(controller.pending_question_ids()))
    payload = controller.pending_question_payloads[-1]
    assert payload["questions"] == _questions() and payload["asked_by"] == "agent"
    assert payload["timeout_seconds"] == 0.0 and payload["deadline_monotonic"] is None
    answers = [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": ["eu"], "other_text": None, "unanswered": False},
    ]
    controller.resolve_pending_question(answers, request_id=payload["request_id"])
    thread.join(timeout=5)
    assert box["result"] == {"answered": True, "answers": answers}
    assert controller.pending_question_payloads[-1] is None, "teardown clears the card"
    assert markers and "Which DB? → Postgres" in markers[0][1]


def test_resolve_with_a_stale_or_missing_id_is_dropped(make_controller):
    controller = make_controller()
    thread, box = _start(controller, _questions())
    _wait_until(lambda: bool(controller.pending_question_ids()))
    controller.resolve_pending_question([], request_id=None)
    controller.resolve_pending_question([], request_id="not-this-round")
    assert controller.pending_question_ids(), "still armed"
    controller.begin_shutdown()
    thread.join(timeout=5)
    assert box["result"]["answered"] is False


def test_timeout_auto_continues_with_a_deadline_on_the_card(make_controller):
    controller = make_controller()
    controller.ask_user_timeout_seconds = lambda: 1.0
    thread, box = _start(controller, _questions())
    _wait_until(lambda: bool(controller.pending_question_payloads))
    assert controller.pending_question_payloads[0]["timeout_seconds"] == 1.0
    assert controller.pending_question_payloads[0]["deadline_monotonic"] is not None
    thread.join(timeout=5)
    assert box["result"] == {"answered": False, "reason": "timeout"}


def test_timeout_reads_console_config_when_no_seam(make_controller, monkeypatch):
    import tldw_chatbook.Chat.console_chat_controller as module

    monkeypatch.setattr(
        module, "get_cli_setting",
        lambda section, key, default=None: 7 if (section, key) == ("console", "ask_user_timeout_seconds") else default,
    )
    controller = make_controller()
    assert controller._resolve_ask_user_timeout_seconds() == 7.0
    monkeypatch.setattr(module, "get_cli_setting", lambda s, k, d=None: "garbage")
    assert controller._resolve_ask_user_timeout_seconds() == 0.0


def test_second_ask_in_the_same_session_is_busy_and_the_third_is_refused(make_controller):
    controller = make_controller()
    session = controller.new_session(title="s")
    thread, box = _start(controller, _questions(), session_id=session.id, run_id="run-1")
    _wait_until(lambda: bool(controller.pending_question_ids()))
    with use_run_id("run-1"):
        first = controller.request_user_questions(_questions(), session_id=session.id)
        assert first["answered"] is False and first["reason"] == "busy"
        with pytest.raises(AskUserBusyRefusal):
            controller.request_user_questions(_questions(), session_id=session.id)
    controller.resolve_pending_question([], request_id=controller.pending_question_ids()[0])
    thread.join(timeout=5)
    assert box["result"]["answered"] is True


def test_a_parked_background_round_mounts_on_switch(make_controller):
    controller = make_controller()
    first = controller.new_session(title="first")
    second = controller.new_session(title="second")
    parked = []
    controller.park_pending_approval = parked.append
    thread, box = _start(controller, _questions(), session_id=first.id)
    _wait_until(lambda: bool(controller.pending_question_ids()))
    assert parked == [first.id]
    assert controller.pending_question_payloads[-1] is None or controller.pending_question_payloads == [None]
    controller.switch_session(first.id)
    assert controller.pending_question_payloads[-1]["session_id"] == first.id
    controller.switch_session(second.id)
    assert controller.pending_question_payloads[-1] is None
    controller.resolve_pending_question([], request_id=controller.pending_question_ids()[0])
    thread.join(timeout=5)
    assert box["result"]["answered"] is True


def test_revoking_the_run_returns_cancelled(make_controller):
    controller = make_controller()
    session = controller.new_session(title="s")
    thread, box = _start(controller, _questions(), session_id=session.id, run_id="run-9")
    _wait_until(lambda: bool(controller.pending_question_ids()))
    assert controller.revoke_approval_rounds_for_run("run-9") == 1
    thread.join(timeout=5)
    assert box["result"] == {"answered": False, "reason": "cancelled"}
    assert controller.pending_question_ids() == []


def test_wiring_registers_the_callback_only_with_a_view(make_controller):
    controller = make_controller()
    session = controller.new_session(title="s")
    wiring = controller._ask_user_wiring(session.id)
    assert set(wiring) == {"ask_user"}
    controller.set_pending_question = None
    assert controller._ask_user_wiring(session.id) == {}
    assert controller._ask_user_wiring(None) == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Chat/test_console_ask_user_round.py`
Expected: the new tests FAIL with `AttributeError: 'ConsoleChatController' object has no attribute 'request_user_questions'`.

- [ ] **Step 3: Implement the state and constants**

After `_DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS = 0.0`:

```python
#: PRD A7: `[console] ask_user_timeout_seconds`. 0 = no deadline (ADR-067);
#: a positive value auto-continues the run with `answered: false`.
_DEFAULT_ASK_USER_TIMEOUT_SECONDS = 0.0
```

In `__init__`, after `self._parked_worktree_merge_payloads: dict[str, dict[str, Any]] = {}`:

```python
        #: PRD Feature A: the ask_user question round -- same shape as the
        #: skill-script confirm (round-keyed registry + retained payloads).
        self.set_pending_question: Callable[[dict | None], None] | None = None
        self.ask_user_timeout_seconds: Callable[[], float] | None = None
        self._pending_question_rounds: dict[str, dict[str, Any]] = {}
        self._pending_question_lock = threading.Lock()
        self._parked_question_payloads: dict[str, dict[str, Any]] = {}
        #: A9: consecutive `busy` results per run id; reset on any real round.
        self._question_bounces: dict[str, int] = {}
```

- [ ] **Step 4: Implement the round** (insert before `def request_worktree_merge_confirm(`)

```python
    def _resolve_ask_user_timeout_seconds(self) -> float:
        """PRD A7: the question deadline -- injected seam, else config, else 0.

        Returns:
            Seconds before an unanswered question auto-continues; ``0.0``
            (the default) means no deadline. Never negative.
        """
        if self.ask_user_timeout_seconds is not None:
            try:
                return max(0.0, float(self.ask_user_timeout_seconds()))
            except Exception:  # noqa: BLE001 -- fail open to the documented default
                pass
        try:
            return max(
                0.0,
                float(
                    get_cli_setting(
                        "console", "ask_user_timeout_seconds", _DEFAULT_ASK_USER_TIMEOUT_SECONDS
                    )
                ),
            )
        except (TypeError, ValueError):
            return _DEFAULT_ASK_USER_TIMEOUT_SECONDS

    def request_user_questions(
        self, questions: list[dict[str, Any]], *, session_id: str | None = None
    ) -> dict[str, Any]:
        """WORKER THREAD: show ``questions`` on a card and wait for the answers.

        PRD Feature A (A5-A7, A9-A11, A14). Clones
        ``request_worktree_merge_confirm``'s round machinery -- fresh
        request id, park-or-mount under the TASK-910 contract, poll under
        ``use_human_input_wait`` so the owning run's tool clock pauses,
        cancel/deadline checks -- with a question-shaped decision. Two
        differences: a second call while this session already has a live
        round returns ``busy`` at once (A9: depth is expressed by batching
        questions, never by queueing rounds), and every outcome is recorded
        in the transcript on resolve (A14).

        Args:
            questions: Validated questions (``ask_user_questions.
                validate_questions`` output).
            session_id: The run's OWNING session; ``None`` never parks.

        Returns:
            ``{"answered": True, "answers": [...]}`` or ``{"answered":
            False, "reason": "timeout" | "cancelled" | "busy"}``.

        Raises:
            AskUserBusyRefusal: ``MAX_CONSECUTIVE_BUSY`` consecutive busy
                results in one run (A9's retry-loop ceiling).
        """
        from tldw_chatbook.Agents.ask_user_questions import (
            ASK_USER_REFUSAL_COPY,
            MAX_CONSECUTIVE_BUSY,
            AskUserBusyRefusal,
            answered_result,
            busy_result,
            empty_answers,
            unanswered_result,
        )
        from tldw_chatbook.Chat.console_agent_bridge import format_question_marker

        if self.app is None or self.set_pending_question is None:
            return unanswered_result("cancelled")
        owning_session_id = (
            session_id if session_id is not None else (self.store.active_session_id or "")
        )
        owning_run_id = current_run_id()
        with self._pending_question_lock:
            live = any(
                state.get("session_id") == owning_session_id
                for state in self._pending_question_rounds.values()
            )
            if live:
                bounces = self._question_bounces.get(owning_run_id, 0) + 1
                self._question_bounces[owning_run_id] = bounces
            else:
                bounces = 0
                self._question_bounces.pop(owning_run_id, None)
        if live:
            if bounces >= MAX_CONSECUTIVE_BUSY:
                raise AskUserBusyRefusal(ASK_USER_REFUSAL_COPY)
            return busy_result()
        event = threading.Event()
        decision: dict[str, Any] = {}
        request_id = str(uuid4())
        round_cancel_event = self._bind_round_cancel_signal(session_id)
        visit_cancel_event = self._bind_visit_cancel_signal()
        round_state: dict[str, Any] = {
            "event": event,
            "decision": decision,
            "session_id": owning_session_id,
            "run_id": owning_run_id,
            "revoked": False,
        }
        with self._pending_question_lock:
            self._pending_question_rounds[request_id] = round_state
        timeout_seconds = self._resolve_ask_user_timeout_seconds()
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        actor = current_run_actor()
        asked_by = "sub-agent" if actor is not None and actor.kind == "subagent" else "agent"
        card_payload: dict[str, Any] = {
            "questions": [dict(question) for question in questions],
            "asked_by": asked_by,
            "timeout_seconds": timeout_seconds,
            "request_id": request_id,
            "session_id": owning_session_id,
            "deadline_monotonic": deadline,
        }
        is_parked = session_id is not None and session_id != (
            self.store.active_session_id or ""
        )
        is_head = True
        if session_id is not None:
            self.add_pending_round(session_id, request_id)
            is_head = self._park_round_payload(
                self._parked_question_payloads, request_id, card_payload
            )
        try:
            if is_parked:
                if self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            elif is_head:
                self._marshal_pending_question(card_payload)
            outcome = "answered"
            with use_human_input_wait(owning_run_id):
                while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                    if self._is_session_cancelled(
                        session_id,
                        cancel_event=round_cancel_event,
                        visit_event=visit_cancel_event,
                    ):
                        outcome = "cancelled"
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        outcome = "timeout"
                        break
            if round_state["revoked"]:
                outcome = "cancelled"
            if outcome == "answered":
                answers = decision.get("answers")
                result = answered_result(answers if answers else empty_answers(questions))
            else:
                result = unanswered_result(outcome)
            bridge = self._agent_bridge
            if bridge is not None:
                with contextlib.suppress(Exception):
                    bridge.append_question_marker(
                        owning_session_id,
                        format_question_marker(asked_by, questions, result),
                    )
            return result
        finally:
            with self._pending_question_lock:
                self._pending_question_rounds.pop(request_id, None)
            self._unpark_round_payload(self._parked_question_payloads, request_id)
            if session_id is not None:
                self.discard_pending_round(session_id, request_id)
            with contextlib.suppress(Exception):
                self._remount_head(
                    self._parked_question_payloads,
                    self.set_pending_question,
                    owning_session_id if session_id is not None else None,
                )

    def resolve_pending_question(
        self, answers: list[dict[str, Any]], request_id: str | None = None
    ) -> None:
        """UI THREAD: hand the card's answers to the waiting worker thread.

        Strict ``request_id`` match, exactly like
        ``resolve_pending_skill_script``: a resolve with no id, or an id
        from any round but the armed one, is silently dropped.

        Args:
            answers: One PRD A6 answer dict per question, in order.
            request_id: The armed round's id as echoed back by the card.
        """
        if request_id is None:
            return
        with self._pending_question_lock:
            round_state = self._pending_question_rounds.get(request_id)
        if round_state is None:
            return
        round_state["decision"]["answers"] = [dict(answer) for answer in answers]
        round_state["event"].set()

    def pending_question_ids(self) -> list[str]:
        """Return the request ids of every armed question round, arm order.

        Returns:
            The armed round ids; empty when none is pending.
        """
        with self._pending_question_lock:
            return list(self._pending_question_rounds)

    def _marshal_pending_question(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a question payload to the UI thread.

        Args:
            payload: The card payload to show, or None to hide the card.
        """
        if self.app is not None and self.set_pending_question is not None:
            self.app.call_from_thread(self.set_pending_question, payload)

    def _remount_parked_question(self, session_id: str) -> None:
        """UI THREAD: re-derive the question card for the session now viewed.

        Called from ``switch_session``/``new_session``/``close_session``
        beside the other card re-derives (PRD A10).

        Args:
            session_id: The session being activated/viewed.
        """
        if self.set_pending_question is None:
            return
        self.set_pending_question(
            self._head_round_payload(self._parked_question_payloads, session_id)
        )

    def _revoke_question_rounds(self, run_id: str) -> list[tuple[str, str | None]]:
        """Fail this run's question rounds closed as ``cancelled`` (PRD A10).

        Registry work only, under ``_pending_question_lock``; the caller
        (``revoke_approval_rounds_for_run``) does the badge and card work.

        Args:
            run_id: The cancelled/abandoned run.

        Returns:
            ``(request_id, session_id)`` per revoked round.
        """
        revoked: list[tuple[str, str | None]] = []
        with self._pending_question_lock:
            for request_id, state in list(self._pending_question_rounds.items()):
                if state.get("run_id") != run_id:
                    continue
                state["revoked"] = True
                self._pending_question_rounds.pop(request_id, None)
                revoked.append((request_id, state.get("session_id") or None))
                event = state.get("event")
                if event is not None:
                    event.set()
        return revoked
```

- [ ] **Step 5: Wire the revocation leg**

In `revoke_approval_rounds_for_run`, after `script_revoked = self._revoke_skill_script_rounds(run_id)` add `question_revoked = self._revoke_question_rounds(run_id)`; after the `for request_id, session_id in script_revoked:` loop add:

```python
        for request_id, session_id in question_revoked:
            if session_id is not None:
                self.discard_pending_round(session_id, request_id)
            with contextlib.suppress(Exception):
                self._remount_head(
                    self._parked_question_payloads,
                    self.set_pending_question,
                    session_id,
                )
```

and change `total = len(revoked) + len(script_revoked)` to `total = len(revoked) + len(script_revoked) + len(question_revoked)`.

- [ ] **Step 6: Wire activation and provider composition**

After each of the three `self._remount_parked_worktree_merge(...)` calls (`session.id`, `session_id`, `new_active_id`), add the matching `self._remount_parked_question(...)`.

After `def _todo_wiring(` (its `return {...}`), add:

```python
    def _ask_user_wiring(self, session_id: str | None) -> dict[str, Any]:
        """The ``ask_user`` kwarg for ``LocalToolProvider`` (PRD A10/A12).

        Empty -- so the tool is never registered -- when there is no
        session context or no view to show a card on (headless runs).

        Args:
            session_id: THIS run's owning session id.

        Returns:
            ``{"ask_user": callback}`` or ``{}``.
        """
        if session_id is None or self.app is None or self.set_pending_question is None:
            return {}

        def _ask(questions: list[dict[str, Any]]) -> dict[str, Any]:
            return self.request_user_questions(questions, session_id=session_id)

        return {"ask_user": _ask}
```

In `_compose_local_provider`'s `LocalToolProvider(` call, right after `**self._todo_wiring(session_id),` add `**self._ask_user_wiring(session_id),`.

- [ ] **Step 7: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Chat/test_console_ask_user_round.py Tests/Chat/test_console_skill_script_confirm.py Tests/Chat/test_skill_script_concurrent_confirms.py`
Expected: all PASS except failures already red on dev (see Task 8's baseline step; on 2026-09-04 `test_console_skill_script_confirm.py` had 2 pre-existing failures in some runs). If `test_a_parked_background_round_mounts_on_switch` fails on the first assertion, check whether `park_pending_approval` is a plain attribute on the controller (grep `self.park_pending_approval` in `__init__`); it is one of the `CONSOLE_VIEW_HOOK_SLOTS`, default `None`.

- [ ] **Step 8: Lint and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --diff tldw_chatbook/Chat/console_chat_controller.py | grep -c "request_user_questions"   # expect 0 (no new lint in the new code)
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_ask_user_round.py
git commit -m "feat(console): ask_user question round -- busy, park, cancel, timeout, marker (PRD A5-A7/A9-A11/A14)"
```

---

### Task 6: Resume state, the card, and its lazy mount in the card slot

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py` — `TaskResumeState`.
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`.
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py`.
- Regenerate: `tldw_chatbook/css/widget_defaults_self.tcss` (via `python tldw_chatbook/css/build_css.py`).
- Test: `Tests/UI/test_chat_question_card.py`

**Interfaces:**
- Produces: `TaskResumeState.pending_question: dict[str, Any] | None = None`, `has_pending_question() -> bool`, serialized in `to_dict`, dropped by `from_dict`; `ChatTaskCards.QuestionAnswered(Message)` with `.answers: list[dict]`, `.request_id: str | None`; `ChatTaskCards._question_card(create: bool) -> ChatQuestionCard | None`; `ChatQuestionCard.set_questions(payload: dict | None)`, `collect_answers() -> list[dict]`, `format_question_deadline(timeout_seconds) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_chat_question_card.py
"""PRD Feature A: the question card, its state plumbing, and the card slot."""

from __future__ import annotations

import pytest
from textual import on
from textual.app import ComposeResult
from textual.widgets import Input, RadioButton, RadioSet, SelectionList, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


def _payload(n_questions: int = 2, n_options: int = 2, *, request_id: str = "round-1", timeout: float = 0.0):
    return {
        "request_id": request_id,
        "session_id": "s1",
        "timeout_seconds": timeout,
        "deadline_monotonic": None,
        "asked_by": "agent",
        "questions": [
            {
                "question": f"Question {q}?",
                "header": f"Q{q}",
                "multiSelect": q % 2 == 1,
                "options": [
                    {"label": f"opt{q}{o}", "description": f"desc {o}"} for o in range(n_options)
                ],
            }
            for q in range(n_questions)
        ],
    }


# --- state ---------------------------------------------------------------


def test_state_carries_and_serializes_a_pending_question():
    state = TaskResumeState(pending_question=_payload())
    assert state.has_pending_question() is True
    assert state.to_dict()["pending_question"]["request_id"] == "round-1"


def test_restored_state_drops_the_pending_question_so_no_dead_card_appears():
    restored = TaskResumeState.from_dict(TaskResumeState(pending_question=_payload()).to_dict())
    assert restored.pending_question is None


# --- card under the real CSS ---------------------------------------------


class _Harness(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.answered = []

    def compose(self) -> ComposeResult:
        yield ChatTaskCards(id="chat-task-cards")

    @on(ChatTaskCards.QuestionAnswered)
    def _record(self, event) -> None:
        self.answered.append((event.request_id, event.answers))


async def _mount(app, pilot, payload):
    cards = app.query_one(ChatTaskCards)
    cards.sync_state(TaskResumeState(pending_question=payload))
    await pilot.pause()
    await pilot.pause()
    return cards.query_one("#chat-question-card")


@pytest.mark.asyncio
async def test_card_is_absent_until_a_question_arrives_then_renders_every_section():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        cards = app.query_one(ChatTaskCards)
        assert not list(cards.query("#chat-question-card")), "lazy: nothing mounted at boot"
        card = await _mount(app, pilot, _payload())
        assert cards.display is True and card.display is True
        sections = list(card.query(".question-section"))
        assert len(sections) == 2
        assert "2 questions" in str(card.query_one("#question-title", Static).render())
        assert isinstance(sections[0].query_one(".question-options"), RadioSet)
        assert isinstance(sections[1].query_one(".question-options"), SelectionList)
        assert all(section.query(Input) for section in sections), "Other input on every question"
        assert str(card.query_one("#question-deadline", Static).render()) == ""


@pytest.mark.asyncio
async def test_submit_returns_selections_other_text_and_unanswered_with_the_request_id():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(3))
        sections = list(card.query(".question-section"))
        list(sections[0].query(RadioButton))[1].value = True
        sections[1].query_one(SelectionList).select(0)
        sections[1].query_one(SelectionList).select(1)
        sections[2].query_one(Input).value = "something else"
        await pilot.pause()
        await pilot.click("#question-submit")
        await pilot.pause()
        assert app.answered == [
            ("round-1", [
                {"question": "Question 0?", "selected": ["opt01"], "other_text": None, "unanswered": False},
                {"question": "Question 1?", "selected": ["opt10", "opt11"], "other_text": None, "unanswered": False},
                {"question": "Question 2?", "selected": [], "other_text": "something else", "unanswered": False},
            ]),
        ]
        assert card.display is False


@pytest.mark.asyncio
async def test_partial_submit_marks_the_skipped_question_unanswered():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(2))
        list(card.query(".question-section"))[0].query(RadioButton)[0].value = True
        await pilot.pause()
        await pilot.click("#question-submit")
        await pilot.pause()
        (_, answers), = app.answered
        assert answers[1] == {"question": "Question 1?", "selected": [], "other_text": None, "unanswered": True}


@pytest.mark.asyncio
async def test_number_keys_select_within_the_focused_question():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1, 3))
        picker = card.query_one(RadioSet)
        picker.focus()
        await pilot.pause()
        await pilot.press("3")
        await pilot.pause()
        assert picker.pressed_index == 2
        other = card.query_one(Input)
        other.focus()
        await pilot.press("2")
        await pilot.pause()
        assert other.value == "2", "digits typed into Other are text, not selections"
        assert picker.pressed_index == 2


@pytest.mark.asyncio
async def test_enter_submits_from_anywhere_in_the_card():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1))
        card.query_one(RadioSet).focus()
        await pilot.press("2")
        await pilot.press("enter")
        await pilot.pause()
        assert app.answered and app.answered[0][1][0]["selected"] == ["opt01"]


@pytest.mark.asyncio
async def test_resync_of_the_same_round_keeps_the_users_selection():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1))
        list(card.query(RadioButton))[1].value = True
        await pilot.pause()
        app.query_one(ChatTaskCards).sync_state(TaskResumeState(pending_question=_payload(1, timeout=30)))
        await pilot.pause()
        assert card.query_one(RadioSet).pressed_index == 1, "same request_id: no rebuild"
        assert "Auto-continues in 0:30" == str(card.query_one("#question-deadline", Static).render())
        app.query_one(ChatTaskCards).sync_state(TaskResumeState(pending_question=_payload(1, request_id="round-2")))
        await pilot.pause()
        await pilot.pause()
        assert card.query_one(RadioSet).pressed_index == -1, "new round: fresh sections"


@pytest.mark.asyncio
async def test_clearing_hides_the_card_and_the_slot():
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(1))
        cards = app.query_one(ChatTaskCards)
        cards.sync_state(TaskResumeState())
        await pilot.pause()
        assert card.display is False and cards.display is False


@pytest.mark.asyncio
async def test_four_by_four_card_stays_bounded_under_bundled_css():
    """AC-A13: 4 questions x 4 described options must not eat the transcript."""
    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        card = await _mount(app, pilot, _payload(4, 4))
        await pilot.pause()
        assert card.region.height <= 24, card.region
        sections = card.query_one("#question-sections")
        assert sections.region.height <= 15, sections.region
        submit = card.query_one("#question-submit")
        assert submit.region.height > 0 and submit.region.y < 40, "Submit stays reachable"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/UI/test_chat_question_card.py`
Expected: FAIL with `TypeError: TaskResumeState.__init__() got an unexpected keyword argument 'pending_question'` and `AttributeError: type object 'ChatTaskCards' has no attribute 'QuestionAnswered'`.

- [ ] **Step 3: `TaskResumeState`**

After `pending_skill_script: dict[str, Any] | None = None`:

```python
    # PRD Feature A: the live ask_user round's card payload. Same
    # architecture as the two skill-confirm fields above -- fully live
    # within one screen instance, never repopulated by `from_dict` (the
    # round is a worker thread blocked on the OLD controller's Event).
    pending_question: dict[str, Any] | None = None
```

After `has_pending_skill_script`:

```python
    def has_pending_question(self) -> bool:
        """Return whether an ask_user question card should be shown."""
        return bool(self.pending_question)
```

In `to_dict`, after `"pending_skill_script": self.pending_skill_script,` add `"pending_question": self.pending_question,`. In `from_dict`'s `return cls(` call, after `pending_skill_script=None,` add `pending_question=None,`.

- [ ] **Step 4: `ChatTaskCards`**

Add imports: `from textual.css.query import NoMatches` and `from textual.message import Message`. Inside the class, before `__init__`:

```python
    class QuestionAnswered(Message):
        """The user submitted the ask_user card (PRD Feature A).

        Defined here, not on the card, so ``ChatScreen``'s ``@on`` handler
        needs no import of the lazily-loaded card module (ADR-097).
        """

        def __init__(self, answers: list[dict[str, Any]], request_id: str | None) -> None:
            """Carry the answers and the round id they resolve.

            Args:
                answers: One PRD A6 answer dict per question, in order.
                request_id: The pending round's id, echoed back unchanged.
            """
            super().__init__()
            self.answers = answers
            self.request_id = request_id
```

New method after `compose`:

```python
    def _question_card(self, *, create: bool):
        """Return the question card, mounting it on first use.

        The card module is NOT imported at boot: ``ChatTaskCards`` composes
        during startup and ADR-097's UI-ready module census sits at its cap.
        The first pending question mounts the card after the skill-script
        card, so nothing about it loads until an agent actually asks.

        Args:
            create: Mount the card when it is absent.

        Returns:
            The ``ChatQuestionCard``, or None when absent and not created.
        """
        try:
            return self.query_one("#chat-question-card")
        except NoMatches:
            if not create:
                return None
            from tldw_chatbook.Widgets.Chat_Widgets.chat_question_card import (
                ChatQuestionCard,
            )

            card = ChatQuestionCard(id="chat-question-card")
            self.mount(card, after=self.query_one(SkillScriptConfirmCard))
            return card
```

In `sync_state`, after `script_card.set_script(task_state.pending_skill_script)`:

```python
        question_card = self._question_card(create=bool(task_state.pending_question))
        if question_card is not None:
            question_card.set_questions(task_state.pending_question)
```

and in the `self.display = (` expression add `or task_state.has_pending_question()` after `or task_state.has_pending_skill_script()`. Update the class docstring's first line to mention questions.

- [ ] **Step 5: The card**

```python
# tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py
"""The ``ask_user`` question card (PRD Feature A: A2-A5, A7, A11).

Mounted lazily by ``ChatTaskCards`` on the first pending question, in the
same slot above the transcript where approvals appear. One section per
question -- header chip, question text, options, an always-present "Other"
input -- inside a bounded, scrolling container so four described questions
cannot push the transcript off screen.

The round-trip contract: ``set_questions(payload)`` stores
``payload["request_id"]`` and ``QuestionAnswered`` echoes it back; the
controller strict-matches it, so a stale submit is dropped, never
misapplied to a newer round.
"""

from __future__ import annotations

from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Input, RadioButton, RadioSet, SelectionList, Static
from textual.widgets.selection_list import Selection

from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

_NUMBER_KEYS = {"1": 0, "2": 1, "3": 2, "4": 3}


def format_question_deadline(timeout_seconds: float | None) -> str:
    """Return the countdown copy for an armed question deadline (PRD A7).

    Mirrors ``chat_approval_card.format_approval_deadline``: say nothing
    rather than invent a number.

    Args:
        timeout_seconds: Remaining seconds, or None/0 when no deadline.

    Returns:
        ``"Auto-continues in M:SS"`` or ``""``.
    """
    try:
        total = int(timeout_seconds or 0)
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    return f"Auto-continues in {total // 60}:{total % 60:02d}"


class ChatQuestionCard(Container):
    """Multiple-choice questions from the agent, answered in place."""

    BINDINGS = [
        Binding("enter", "submit_answers", "Submit answers", show=False, priority=True),
    ]

    BUNDLED_CSS = """
    ChatQuestionCard {
        height: auto;
        max-height: 24;
        border: round $accent;
        padding: 0 1;
    }
    ChatQuestionCard > #question-title {
        height: 1;
        text-style: bold;
    }
    ChatQuestionCard > #question-deadline {
        height: auto;
        color: $text-muted;
    }
    ChatQuestionCard > #question-sections {
        height: auto;
        max-height: 15;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    ChatQuestionCard .question-section {
        height: auto;
        margin-bottom: 1;
    }
    ChatQuestionCard .question-header {
        height: 1;
        color: $accent;
        text-style: bold;
    }
    ChatQuestionCard .question-text {
        height: auto;
    }
    ChatQuestionCard .question-options {
        height: auto;
        border: none;
        padding: 0;
    }
    ChatQuestionCard .question-other {
        height: 3;
    }
    ChatQuestionCard > #question-actions {
        height: 3;
        align-horizontal: right;
    }
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Start hidden with no round; ``set_questions`` shows it."""
        super().__init__(*args, **kwargs)
        self.display = False
        self._payload: dict[str, Any] | None = None
        self._request_id: str | None = None
        self._rendered_request_id: str | None = None
        self._questions: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        """Yield the title, deadline line, scrolling sections, and Submit."""
        yield Static("", id="question-title", markup=False)
        yield Static("", id="question-deadline", markup=False)
        yield VerticalScroll(id="question-sections")
        yield Horizontal(
            Button("Submit", id="question-submit", variant="primary"),
            id="question-actions",
        )

    def on_mount(self) -> None:
        """Paint a payload that arrived before the children existed."""
        if self._payload:
            self._paint()

    def set_questions(self, payload: dict[str, Any] | None) -> None:
        """Show the card for ``payload``, or hide it when None.

        A payload carrying the SAME ``request_id`` as the one on screen only
        refreshes the deadline copy -- ``ChatTaskCards.sync_state`` re-syncs
        every card on any task-state change, and rebuilding the sections
        would wipe the user's half-made selections.

        Args:
            payload: ``{"questions", "asked_by", "timeout_seconds",
                "request_id", ...}`` from the controller, or None.
        """
        if not payload:
            self.display = False
            self._payload = None
            self._request_id = None
            self._rendered_request_id = None
            self._questions = []
            try:
                self.query_one("#question-sections", VerticalScroll).remove_children()
            except NoMatches:
                pass
            return
        self._payload = dict(payload)
        self._request_id = payload.get("request_id")
        self._questions = [dict(q) for q in (payload.get("questions") or [])]
        self.display = True
        self._paint()

    def collect_answers(self) -> list[dict[str, Any]]:
        """Read every section into PRD A6 answer dicts, in question order.

        Returns:
            One ``{"question", "selected", "other_text", "unanswered"}`` per
            question; a question with neither a selection nor Other text is
            ``unanswered`` (A5: partial submission is allowed).
        """
        answers: list[dict[str, Any]] = []
        sections = list(self.query(".question-section"))
        for index, question in enumerate(self._questions):
            options = question.get("options") or []
            selected: list[str] = []
            other: str | None = None
            if index < len(sections):
                section = sections[index]
                if question.get("multiSelect"):
                    picker = section.query_one(SelectionList)
                    selected = [
                        str(options[i]["label"]) for i in picker.selected if 0 <= i < len(options)
                    ]
                else:
                    radio = section.query_one(RadioSet)
                    if 0 <= radio.pressed_index < len(options):
                        selected = [str(options[radio.pressed_index]["label"])]
                other = section.query_one(Input).value.strip() or None
            answers.append(
                {
                    "question": str(question.get("question", "")),
                    "selected": selected,
                    "other_text": other,
                    "unanswered": not selected and other is None,
                }
            )
        return answers

    def action_submit_answers(self) -> None:
        """Submit whatever is answered (Enter anywhere in the card, A4/A5)."""
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Submit on the Submit button.

        Args:
            event: The button press; only ``#question-submit`` is consumed.
        """
        if event.button.id != "question-submit":
            return
        event.stop()
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter inside an Other box submits the whole card.

        Args:
            event: The Input's submit event.
        """
        event.stop()
        self._submit()

    def on_key(self, event: events.Key) -> None:
        """``1``-``4`` pick an option in the focused question's picker (A4).

        Digits typed into an Other input are text and are left alone.

        Args:
            event: The key event.
        """
        index = _NUMBER_KEYS.get(event.key)
        if index is None:
            return
        focused = self.app.focused
        if not isinstance(focused, (RadioSet, SelectionList)) or self not in focused.ancestors:
            return
        event.stop()
        if isinstance(focused, RadioSet):
            buttons = list(focused.query(RadioButton))
            if index < len(buttons):
                buttons[index].value = True
        else:
            if index < focused.option_count:
                focused.toggle(index)

    def _submit(self) -> None:
        answers = self.collect_answers()
        request_id = self._request_id
        self.display = False
        self.post_message(ChatTaskCards.QuestionAnswered(answers, request_id))

    def _paint(self) -> None:
        try:
            title = self.query_one("#question-title", Static)
            deadline = self.query_one("#question-deadline", Static)
            sections = self.query_one("#question-sections", VerticalScroll)
        except NoMatches:
            return
        payload = self._payload or {}
        who = "A sub-agent" if payload.get("asked_by") == "sub-agent" else "The agent"
        count = len(self._questions)
        title.update(f"{who} has {count} question{'s' if count != 1 else ''} for you:")
        deadline.update(format_question_deadline(payload.get("timeout_seconds")))
        if self._rendered_request_id == self._request_id:
            return
        self._rendered_request_id = self._request_id
        sections.remove_children()
        key = (self._request_id or "none")[:8]
        sections.mount_all(
            self._build_section(key, index, question)
            for index, question in enumerate(self._questions)
        )

    @staticmethod
    def _option_prompt(option: dict[str, Any]) -> str:
        label = str(option.get("label", ""))
        description = str(option.get("description") or "")
        return f"{label} — {description}" if description else label

    def _build_section(self, key: str, index: int, question: dict[str, Any]) -> Vertical:
        options = question.get("options") or []
        picker: RadioSet | SelectionList
        if question.get("multiSelect"):
            picker = SelectionList(
                *[Selection(self._option_prompt(option), i) for i, option in enumerate(options)],
                classes="question-options",
            )
        else:
            picker = RadioSet(
                *[RadioButton(self._option_prompt(option)) for option in options],
                classes="question-options",
            )
        return Vertical(
            Static(str(question.get("header", "")), classes="question-header", markup=False),
            Static(str(question.get("question", "")), classes="question-text", markup=False),
            picker,
            Input(placeholder="Other…", classes="question-other"),
            id=f"question-{key}-{index}",
            classes="question-section",
        )
```

Notes for the implementer: `RadioButton.value = True` is the programmatic way to press a radio (it drives `RadioSet.pressed_index`); `SelectionList.select(value)`/`toggle(value)` take the option VALUE (we use the option index as the value); `RadioSet.pressed_index` is `-1` when nothing is pressed; `Widget.ancestors` excludes the widget itself. Section ids carry the request id prefix so a rebuild for a new round never collides with the previous round's not-yet-pruned children (`DuplicateIds`).

- [ ] **Step 6: Regenerate the widget-defaults sheet and run**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py | tail -3
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py | tail -2
git status --short tldw_chatbook/css   # expect only widget_defaults_self.tcss modified
```

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/UI/test_chat_question_card.py Tests/UI/test_chat_task_cards_sync.py Tests/UI/test_skill_script_confirm_card.py Tests/UI/test_widget_css_consolidation.py`
Expected: all PASS except `test_class_level_css_stays_within_the_allowlist` (baseline-red on dev: six pre-existing `DEFAULT_CSS` declarations in files this plan does not touch). If `test_enter_submits_from_anywhere_in_the_card` fails because Enter toggled the radio instead, the card's `priority=True` binding is not winning — check that `BINDINGS` is on the card class (an ancestor of the focused picker) and that the key name is `"enter"`. If `test_four_by_four_card_stays_bounded_under_bundled_css` fails on height, `SelectionList`/`RadioSet` default heights are the culprit: add `max-height: 5;` to `.question-options`.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py tldw_chatbook/css/widget_defaults_self.tcss Tests/UI/test_chat_question_card.py
git commit -m "feat(console): the ask_user question card, mounted lazily in the task-card slot (PRD A2-A5/A7/A11)"
```

---

### Task 7: Screen and runtime wiring

**Files:**
- Modify: `tldw_chatbook/Chat/console_runtime.py` — `CONSOLE_VIEW_HOOK_SLOTS`, after the `"set_task_panel"` slot.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — `console_view_hooks()` dict (after `"set_task_panel": self._set_console_task_panel,`); new methods next to `_set_console_task_panel`; new `@on` handler next to `@on(SkillScriptConfirmCard.ScriptDecided)`; `handle_console_inspector_review_approval`.
- Test: `Tests/UI/test_chat_question_card.py` (extend)

**Interfaces:**
- Consumes: Task 5's `set_pending_question` attribute and `resolve_pending_question`; Task 6's `ChatTaskCards.QuestionAnswered`, `TaskResumeState.pending_question`.
- Produces: `ChatScreen._set_console_pending_question(payload: dict | None) -> None`, `ChatScreen.handle_console_question_answered(event) -> None`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_chat_question_card.py`)

```python
from unittest.mock import Mock

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_chat_screen_forwards_answers_to_the_controller_with_request_id(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    controller = Mock()
    screen._console_chat_controller = controller
    answers = [{"question": "q", "selected": ["a"], "other_text": None, "unanswered": False}]
    screen.handle_console_question_answered(ChatTaskCards.QuestionAnswered(answers, "round-7"))
    controller.resolve_pending_question.assert_called_once_with(answers, request_id="round-7")


def test_chat_screen_question_handler_tolerates_no_controller(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    screen._console_chat_controller = None
    screen.handle_console_question_answered(ChatTaskCards.QuestionAnswered([], "round-7"))


def test_chat_screen_setter_replaces_only_the_pending_question(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    screen._task_resume_state = TaskResumeState(summary="keep me", pending_skill_script={"skill_name": "x"})
    screen.sync_task_resume_state = Mock()
    screen._set_console_pending_question(_payload())
    state = screen._task_resume_state
    assert state.pending_question["request_id"] == "round-1"
    assert state.summary == "keep me" and state.pending_skill_script == {"skill_name": "x"}
    screen.sync_task_resume_state.assert_called_once()


def test_the_hook_slot_is_declared_for_the_new_setter():
    from tldw_chatbook.Chat.console_runtime import CONSOLE_VIEW_HOOK_SLOTS

    slot = next(s for s in CONSOLE_VIEW_HOOK_SLOTS if s.name == "set_pending_question")
    assert slot.target == "controller" and slot.viewless_default is None and slot.why
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/UI/test_chat_question_card.py -k "chat_screen or hook_slot"`
Expected: FAIL with `AttributeError: 'ChatScreen' object has no attribute 'handle_console_question_answered'` and `StopIteration`.

- [ ] **Step 3: Runtime slot** (`console_runtime.py`, after the `"set_task_panel"` `ConsoleViewHookSlot(...)`)

```python
    ConsoleViewHookSlot(
        "set_pending_question",
        "controller",
        why="`request_user_questions` returns `{answered: False, reason: "
        "'cancelled'}` when it is None, and `_ask_user_wiring` registers no "
        "tool at all without it -- a viewless run cannot be asked, which is "
        "PRD A10's headless posture.",
    ),
```

- [ ] **Step 4: Screen**

Hooks dict, after `"set_task_panel": self._set_console_task_panel,`:

```python
            # PRD Feature A: the ask_user question card.
            "set_pending_question": self._set_console_pending_question,
```

Next to `_set_console_task_panel` (ensure `from dataclasses import replace` is imported at the top of `chat_screen.py`; grep `^from dataclasses import` and extend it if `replace` is missing):

```python
    def _set_console_pending_question(self, payload: dict[str, Any] | None) -> None:
        """PRD Feature A: replace only the pending question in the task state.

        UI-thread target of the controller's ``set_pending_question`` hook;
        ``ChatTaskCards.sync_state`` mounts/updates/hides the card from it.

        Args:
            payload: The round's card payload, or None to hide the card.
        """
        self.set_task_resume_state(
            replace(self._task_resume_state, pending_question=payload)
        )
```

Next to `@on(SkillScriptConfirmCard.ScriptDecided)`:

```python
    @on(ChatTaskCards.QuestionAnswered)
    def handle_console_question_answered(self, event: Any) -> None:
        """Forward the card's answers to the controller's armed round.

        Args:
            event: ``ChatTaskCards.QuestionAnswered`` carrying ``answers``
                and the round's ``request_id``.
        """
        event.stop()
        controller = self._console_chat_controller
        if controller is not None:
            controller.resolve_pending_question(event.answers, request_id=event.request_id)
```

In `handle_console_inspector_review_approval`, replace the first guard:

```python
        if self._console_pending_approval_count() <= 0:
            self.app_instance.notify(
                CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning"
            )
            return
```

with:

```python
        if self._console_pending_approval_count() <= 0:
            # PRD A4: the chip's focus action is how keyboard-only users
            # reach a question card that deliberately never steals focus.
            question = next(
                (c for c in self.query("#chat-question-card") if c.display), None
            )
            if question is not None:
                with contextlib.suppress(Exception):
                    question.scroll_visible(animate=False)
                target = next(iter(question.query("RadioSet, SelectionList, Input")), None)
                if target is not None:
                    target.focus()
                return
            self.app_instance.notify(
                CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning"
            )
            return
```

(`contextlib` — grep `^import contextlib` in `chat_screen.py`; add it if absent.)

- [ ] **Step 5: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/UI/test_chat_question_card.py Tests/Chat/test_console_viewless_hooks.py::test_every_slot_names_a_real_attribute_on_the_target_it_declares Tests/UI/test_console_runtime_ownership.py`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_chat_question_card.py
git commit -m "feat(console): wire the question card through the screen and the view-hook slots (PRD A4/A10)"
```

---

### Task 8: Config doc, User Guide, PRD fix, baselines, preflight, PR

**Files:**
- Modify: `tldw_chatbook/config.py` — beside `# approval_timeout_seconds = 0  # Console approval-card auto-deny ceiling: ...` (~line 5033).
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md` — before `### Approvals — tools ask before they run`, after the "Task panel" section.
- Modify: `Docs/Development/Chatbook/Chatbook-Console-Agent-Interaction-PRD.md` — the one parenthetical citing the engine's 300s `RunBudget` default.

- [ ] **Step 1: Config doc line**

After the `# approval_timeout_seconds = 0 ...` line add:

```python
# ask_user_timeout_seconds = 0  # Console ask_user auto-continue: 0 (default) waits for an answer indefinitely; e.g. 120 continues the run without an answer after 120s
```

- [ ] **Step 2: User Guide section**

```markdown
### Questions from the agent

An agent can stop and ask you up to four multiple-choice questions. The
questions appear on a card above the transcript, in the same slot as
approvals, and the run waits until you answer:

```
The agent has 2 questions for you:
Database   Which database should the migration target?
  ( ) Postgres — managed, relational
  (•) SQLite — embedded
  Other…
Region     Which regions? (pick any)
  [x] eu
  [ ] us
  Other…
                                                    [ Submit ]
```

- Pick an option per question, or type your own answer in **Other…** — it
  is always there, whatever the agent offered. `1`–`4` pick an option in the
  question you are on, `Tab` moves between questions, `Enter` submits.
- You can submit with questions left blank; the agent sees them as
  unanswered and decides what to do.
- The card never grabs focus from something you are typing. If you need to
  reach it from the keyboard, the inspector's **Review approval** action
  focuses the question card when no approval is pending.
- By default the question waits as long as it takes. To make an unanswered
  question expire instead, set `ask_user_timeout_seconds` under `[console]`
  in your config; the card then shows *Auto-continues in m:ss* and the run
  carries on without an answer when it reaches zero.
- A question for a tab you are not looking at lights that tab's badge and
  shows a toast; visit the tab to answer it. Stopping the run clears the card.
- Every answered round leaves one line per question in the transcript, so
  the exchange stays in the record.

The tool is on by default. To remove it from every agent, set
`ask_user_enabled = false` under `[tools]`.
```

- [ ] **Step 3: PRD parenthetical**

`grep -n "300" Docs/Development/Chatbook/Chatbook-Console-Agent-Interaction-PRD.md` — the parenthetical citing the engine's 300s `RunBudget` where Console's effective ceiling is 3600s: change it to say `DEFAULT_CONSOLE_MAX_TOOL_CALL_SECONDS = 3600` (`console_agent_bridge.py`) is the Console ceiling and 300s is the engine default. Also tick nothing in the PRD's AC list (it is a requirements doc; the PR body carries the AC mapping).

- [ ] **Step 4: Measure the baseline on clean dev BEFORE trusting any failure count**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && git fetch -q origin dev && git worktree add --detach -q .claude/worktrees/devbase-askuser origin/dev
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/devbase-askuser
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Chat/test_console_skill_script_confirm.py Tests/Chat/test_skill_script_concurrent_confirms.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_console_headless_approval.py Tests/Chat/test_console_viewless_hooks.py Tests/UI/test_console_runtime_ownership.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_builtin_tool_gate.py Tests/UI/test_chat_task_cards_sync.py Tests/UI/test_skill_script_confirm_card.py Tests/UI/test_console_task_panel.py 2>&1 | grep -E "^FAILED|passed|failed" > /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/8da1c869-9cb9-4431-a1df-f9cc16dbbb63/scratchpad/askuser-devbase.txt
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && git worktree remove --force .claude/worktrees/devbase-askuser
```

- [ ] **Step 5: Run the same battery plus the new suites on the branch and diff**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/ask-user
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Chat/test_console_skill_script_confirm.py Tests/Chat/test_skill_script_concurrent_confirms.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_console_headless_approval.py Tests/Chat/test_console_viewless_hooks.py Tests/UI/test_console_runtime_ownership.py Tests/Agents/test_local_tool_provider.py Tests/Agents/test_builtin_tool_gate.py Tests/UI/test_chat_task_cards_sync.py Tests/UI/test_skill_script_confirm_card.py Tests/UI/test_console_task_panel.py Tests/Agents/test_ask_user_questions.py Tests/Agents/test_ask_user_tool.py Tests/Chat/test_console_ask_user_round.py Tests/UI/test_chat_question_card.py 2>&1 | grep -E "^FAILED|passed|failed" > /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/8da1c869-9cb9-4431-a1df-f9cc16dbbb63/scratchpad/askuser-branch.txt
diff <(grep ^FAILED /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/8da1c869-9cb9-4431-a1df-f9cc16dbbb63/scratchpad/askuser-devbase.txt | sort) <(grep ^FAILED /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/8da1c869-9cb9-4431-a1df-f9cc16dbbb63/scratchpad/askuser-branch.txt | sort)
```

Expected: the diff is EMPTY (no new failures; every `FAILED` on the branch also fails on clean dev). Any `>` line is a regression to fix before the PR.

- [ ] **Step 6: Boot ratchets and preflight**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/Performance/test_ui_latency_guardrails.py Tests/Utils/test_ui_responsiveness_stall_persist.py 2>&1 | grep -E "ui-ready-census|boot-css|^FAILED|passed|failed"
./scripts/preflight.sh 2>&1 | tail -3
```

Expected: census count unchanged from dev (966 locally on macOS; the `+`/`-` drift lines must not list `ask_user_questions`, `chat_question_card`), CSS-bytes ratchet green, `preflight: all derived-artifact checks passed.` If the census lists a new module, something imports it at module level — find it with `grep -rn "ask_user_questions\|chat_question_card" tldw_chatbook --include="*.py"` and move that import inside the function that needs it.

- [ ] **Step 7: Commit, push, PR**

```bash
git add tldw_chatbook/config.py Docs/User_Guide/console/agent-runs-and-tools.md Docs/Development/Chatbook/Chatbook-Console-Agent-Interaction-PRD.md
git commit -m "docs(console): ask_user timeout setting, User Guide section, PRD ceiling correction"
git push -u origin feat/console-ask-user
gh pr create --base dev --head feat/console-ask-user --title "feat(console): ask_user -- an agent can ask the user multiple-choice questions (PRD M2)" --body-file <body>
```

PR body: summary; the AC-A1..A5b, A7..A13 mapping to tests (AC-A6/A8 explicitly deferred to M3); the boot-path decisions (lazy import, lazy mount, message class in the resident module); the baseline diff evidence from Step 5; `./scripts/preflight.sh` green; the census count. End with `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and `https://claude.ai/code/session_018q5PsHwn5kgHPmNwX9DKoo`.

Then the standing merge recipe: wait for Qodo (its summary comment appears first; inline findings on the review), fix every finding, reply on each thread naming the fix and the pinning test, rebase onto `origin/dev`, push, and merge within seconds of the required "Derived artifacts reproduce from their sources" check passing on the exact head (dev is `strict: true` and moves ~6 commits per 15 minutes; auto-merge is disabled; never `--admin`).

---

## Self-Review

**Spec coverage (PRD Feature A minus A8):**
- A1 tool shape + strict validation → Task 1, Task 3 (schema + handler). A2 "Other" injected by the card, absent from the schema → Task 1 (test), Task 6. A3 card in the task-card slot, bounded height with internal scroll → Task 6 (`#question-sections max-height: 15`, geometry test). A4 keyboard: `1`–`4`, Tab (Textual focus chain), Enter, Esc (existing binding), chip focus → Task 6 (`on_key`, priority Enter binding), Task 7 (inspector fallback). A5 partial submission → Task 6 (`unanswered`). A6 result shape → Task 1, Task 5. A7 timeout setting default 0, auto-continue, deadline copy → Task 5 (`_resolve_ask_user_timeout_seconds`), Task 6 (`format_question_deadline`), Task 8 (config doc). A9 busy + two-bounce refusal → Task 5. A10 park/badge/toast via `park_pending_approval`, cancel → `cancelled`, headless → not registered → Task 5 (`_ask_user_wiring`, revocation leg), Task 7 (slot `why`). A11 sub-agents may ask, card names the asker → Task 5 (`asked_by` from `current_run_actor`), Task 6 (title copy). *Limitation, stated:* runs carry no display label on dev, so the card says "A sub-agent" rather than a name. A12 gate default ON, exempt from the permission layer → Task 2, Task 3. A13 restraint description → Task 1 (the internal-prompts registry registration the spec mentioned is NOT done; the description lives on the spec — say so in the PR). A14 transcript marker → Task 4, Task 5.
- AC-A1 → `test_answer_round_trip_and_marker` + `test_submit_returns_selections...`. AC-A2 → `test_rejections_name_the_problem`, `test_handler_rejects_bad_calls...`. AC-A3 → `test_card_is_absent_until...` (Other on every question) + `test_submit_returns...` (other_text). AC-A4 → `test_partial_submit...`. AC-A5 → `test_answer_round_trip_and_marker` (`timeout_seconds == 0`, no deadline, no countdown) — the "still alive after 5s with `max_tool_call_seconds` lowered to 2" leg rides `use_human_input_wait`, already pinned by the skill-script suite for the same context manager. AC-A5b → `test_timeout_auto_continues...` + `test_resync...` (countdown copy). AC-A7 → `test_second_ask_in_the_same_session_is_busy...`. AC-A8 → `test_a_parked_background_round_mounts_on_switch`. AC-A9 → `test_revoking_the_run_returns_cancelled`. AC-A10 → not pinned by a dedicated test; the question round has its own registry, lock, payload map and setter, so an approval and a question never share state — note it in the PR as covered by construction. AC-A11 → `test_ask_user_absent_when_the_gate_is_off`, `test_wiring_registers_the_callback_only_with_a_view`. AC-A12 → `test_marker_*`. AC-A13 → `test_four_by_four_card_stays_bounded_under_bundled_css`.

**Placeholder scan:** none; every step carries code or an exact command.

**Type consistency:** `AskUserCallback` (Task 3) = `Callable[[list[dict[str, Any]]], dict[str, Any]]`, matched by `_ask_user_wiring._ask` (Task 5). `resolve_pending_question(answers, request_id=...)` (Task 5) is what `handle_console_question_answered` calls (Task 7) with `event.answers`, `event.request_id` from `ChatTaskCards.QuestionAnswered(answers, request_id)` (Task 6). The card payload keys produced in Task 5 (`questions`, `asked_by`, `timeout_seconds`, `request_id`, `session_id`, `deadline_monotonic`) are exactly what `set_questions`/`_paint` read in Task 6 and what `_head_round_payload` snapshots (`timeout_seconds` recomputed from `deadline_monotonic`). `format_question_marker(asked_by, questions, result)` (Task 4) is called with those three in Task 5.
