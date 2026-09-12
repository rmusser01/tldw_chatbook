# Console `ask_user` typed answers (PRD M3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A message sent from the Console composer while a question card is mounted answers the question instead of starting a turn (PRD A8 / AC-A6), with slash commands and staged context exempt.

**Architecture:** One screen-side interception in `_send_console_message_from_visible_action`, placed after command handling and immediately before `_dispatch_console_draft_send`. It reads the mounted `ChatQuestionCard`'s current selections, fills `other_text` on every still-unanswered question with the typed text, resolves the round through the controller's existing `resolve_pending_question`, and returns without dispatching. No controller or card changes.

**Tech Stack:** Python ≥3.11, Textual 8.2.8, pytest. No new dependencies.

**Spec:** `Docs/Development/Chatbook/Chatbook-Console-Agent-Interaction-PRD.md` Feature A, A8 and AC-A6. Requires M2 (PR #2379) merged: `ChatQuestionCard.collect_answers`, `ConsoleChatController.resolve_pending_question`, `TaskResumeState.pending_question`.

## Global Constraints

- Same as the M2 plan: no new module resident at UI-ready; no new `logger.*` calls; docstrings on new public methods; `.venv` interpreter from the worktree root; blocking pytest; attribution trailers on every commit.
- The PRD, not the design spec, is the authority: A8 does NOT ask for a transcript user row (the design spec did). The A14 marker already records `other: <text>` per question on resolve, so no user row is appended -- a `persist=False` USER row would be a new shape for the transcript reconciler to misread.
- `ChatScreen(mock_chat_host)` construction is broken on dev (real-path check); drive screen methods as unbound functions on `SimpleNamespace` stubs, the pattern `Tests/UI/test_console_command_composer.py` already uses for `_send_console_message_from_visible_action`.
- Worktree: create `.claude/worktrees/ask-user-m3` on branch `feat/console-ask-user-typed-answers` from `origin/dev` AFTER #2379 merges. Check `git rev-parse --show-toplevel` before every commit, and `cd` into the worktree in every patch command.

---

### Task 1: The interception

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — `_send_console_message_from_visible_action` (its final statement `return await self._dispatch_console_draft_send(draft, stash=stash)`), plus one new method beside `_set_console_pending_question`.
- Test: `Tests/UI/test_console_ask_user_typed_answers.py`

**Interfaces:**
- Consumes: `ChatQuestionCard.collect_answers() -> list[dict]`, `ChatQuestionCard._request_id`, `ChatQuestionCard.set_questions(None)`, `ConsoleChatController.resolve_pending_question(answers, request_id=...)`, `self._console_pending_image_attachment()`, `self._retrieval._pending_launch()` (the staged Library-evidence launch, `None` when nothing is staged), `self._clear_console_composer_draft()`.
- Produces: `ChatScreen._answer_pending_question_with_draft(draft: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_console_ask_user_typed_answers.py
"""PRD A8 / AC-A6: a composer send answers a mounted question instead of starting a turn."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _Card:
    def __init__(self, answers, request_id="round-1", display=True):
        self.display = display
        self._request_id = request_id
        self._answers = answers
        self.cleared = False

    def collect_answers(self):
        return [dict(a) for a in self._answers]

    def set_questions(self, payload):
        assert payload is None
        self.cleared = True


def _answers():
    return [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": [], "other_text": None, "unanswered": True},
    ]


def _screen(card, *, image=None, launch=None, controller=None):
    cleared = []
    return SimpleNamespace(
        _console_chat_controller=controller if controller is not None else Mock(),
        query=lambda selector: [card] if card is not None else [],
        _console_pending_image_attachment=lambda: image,
        _retrieval=SimpleNamespace(_pending_launch=lambda: launch),
        _clear_console_composer_draft=lambda: cleared.append(True),
        _cleared=cleared,
    )


def test_typed_text_fills_every_unanswered_question_and_resolves_the_round():
    card = _Card(_answers())
    screen = _screen(card)
    assert ChatScreen._answer_pending_question_with_draft(screen, "  apac only  ") is True
    screen._console_chat_controller.resolve_pending_question.assert_called_once()
    answers, kwargs = screen._console_chat_controller.resolve_pending_question.call_args
    assert kwargs == {"request_id": "round-1"}
    assert answers[0] == [
        {"question": "Which DB?", "selected": ["Postgres"], "other_text": None, "unanswered": False},
        {"question": "Regions?", "selected": [], "other_text": "apac only", "unanswered": False},
    ]
    assert card.cleared is True and screen._cleared == [True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"card": None},
        {"card": _Card(_answers(), display=False)},
        {"card": _Card(_answers(), request_id=None)},
        {"card": _Card(_answers()), "image": object()},
        {"card": _Card(_answers()), "launch": object()},
        {"card": _Card(_answers()), "controller": None},
    ],
    ids=["no-card", "hidden-card", "no-round-id", "staged-attachment", "staged-rag", "no-controller"],
)
def test_no_interception_without_a_live_card_or_with_staged_context(kwargs):
    controller = kwargs.pop("controller", Mock())
    card = kwargs.pop("card")
    screen = _screen(card, controller=controller, **kwargs)
    if controller is None:
        screen._console_chat_controller = None
    assert ChatScreen._answer_pending_question_with_draft(screen, "text") is False
    if controller is not None:
        controller.resolve_pending_question.assert_not_called()
    assert screen._cleared == []


def test_blank_draft_never_intercepts():
    screen = _screen(_Card(_answers()))
    assert ChatScreen._answer_pending_question_with_draft(screen, "   ") is False
    screen._console_chat_controller.resolve_pending_question.assert_not_called()


@pytest.mark.asyncio
async def test_visible_send_resolves_the_question_and_does_not_dispatch():
    """End to end through the real send action: a plain message with a live
    card answers it; the draft is never queued as a turn."""
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
    from tldw_chatbook.UI.Console_Modules.command_registry import CommandParse, KIND_NOT_COMMAND

    composer = ConsoleComposerBar()
    composer.insert_text("use apac only")
    dispatched = []

    async def dispatch(draft, *, stash=None):
        dispatched.append(draft)
        return True

    card = _Card(_answers())
    controller = Mock()
    screen = SimpleNamespace(
        _console_pending_send_stash=None,
        _raw_cli=SimpleNamespace(start_user_command=Mock()),
        _console_composer_or_none=lambda: composer,
        query_one=lambda *_a, **_k: composer,
        query=lambda selector: [card],
        _console_pending_image_attachment=lambda: None,
        _focus_console_composer_if_needed=lambda **_k: None,
        _dismiss_console_guidance=lambda: None,
        _console_command_registry=SimpleNamespace(parse=lambda draft: CommandParse(kind=KIND_NOT_COMMAND)),
        _console_unknown_send_armed=None,
        _dispatch_console_draft_send=dispatch,
        _console_chat_controller=controller,
        _retrieval=SimpleNamespace(_pending_launch=lambda: None),
        _clear_console_composer_draft=lambda: composer.clear_draft(),
        _answer_pending_question_with_draft=lambda draft: ChatScreen._answer_pending_question_with_draft(screen, draft),
    )
    assert await ChatScreen._send_console_message_from_visible_action(screen) is False
    assert dispatched == []
    controller.resolve_pending_question.assert_called_once()
```

Check the import paths in the last test before running: `grep -rn "^class CommandParse\|^KIND_NOT_COMMAND" tldw_chatbook/UI/Console_Modules/` and `grep -n "def clear_draft\|def insert_text" tldw_chatbook/Widgets/Console/console_composer_bar.py`; use the real names. If `clear_draft` does not exist, substitute the composer method `_clear_console_composer_draft` calls (grep its body at `def _clear_console_composer_draft`).

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/UI/test_console_ask_user_typed_answers.py`
Expected: FAIL with `AttributeError: type object 'ChatScreen' has no attribute '_answer_pending_question_with_draft'`.

- [ ] **Step 3: Implement**

New method beside `_set_console_pending_question`:

```python
    def _answer_pending_question_with_draft(self, draft: str) -> bool:
        """PRD A8: let a composer send answer the mounted question card.

        Only a MOUNTED card on the viewed session intercepts; a parked or
        background round never touches the composer. The typed text becomes
        ``other_text`` for every question still unanswered on the card, the
        card's existing selections ride along, and the round resolves
        through the controller's strict request-id match. Staged context
        (a pending image attachment or a staged Library-evidence launch)
        refuses interception: carrying it into a tool result is meaningless
        and discarding it silently would destroy work the user staged.

        Args:
            draft: The composer text as sent.

        Returns:
            True when the draft answered the question and must NOT also be
            dispatched as a turn; False to send normally.
        """
        text = draft.strip()
        if not text:
            return False
        controller = self._console_chat_controller
        if controller is None:
            return False
        card = next((c for c in self.query("#chat-question-card") if c.display), None)
        if card is None:
            return False
        request_id = getattr(card, "_request_id", None)
        if not request_id:
            return False
        if self._console_pending_image_attachment() is not None:
            return False
        if self._retrieval._pending_launch() is not None:
            return False
        answers = card.collect_answers()
        for answer in answers:
            if answer.get("unanswered") or (
                not answer.get("selected") and answer.get("other_text") is None
            ):
                answer["other_text"] = text
                answer["unanswered"] = False
        card.set_questions(None)
        self._clear_console_composer_draft()
        controller.resolve_pending_question(answers, request_id=request_id)
        return True
```

In `_send_console_message_from_visible_action`, replace the final statement:

```python
        return await self._dispatch_console_draft_send(draft, stash=stash)
```

with:

```python
        if self._answer_pending_question_with_draft(draft):
            return False
        return await self._dispatch_console_draft_send(draft, stash=stash)
```

(The slash-command and unknown-command branches above it already `return False` before this point, so a `/`-command never reaches the interception -- A8's first exemption falls out of the existing order.)

- [ ] **Step 4: Run to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:cacheprovider Tests/UI/test_console_ask_user_typed_answers.py Tests/UI/test_console_command_composer.py`
Expected: all PASS (the command-composer suite's stubs lack `_answer_pending_question_with_draft`; if any of its send-path tests now raise `AttributeError`, add `_answer_pending_question_with_draft=lambda draft: False` to THAT test's stub -- do not weaken the interception with a `getattr` default, the real screen always has the method).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_ask_user_typed_answers.py
git commit -m "feat(console): a composer send answers the mounted ask_user card (PRD A8)"
```

---

### Task 2: User Guide, baseline, PR

**Files:**
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md` — the "Questions from the agent" section.

- [ ] **Step 1: Add the typed-answer bullet** after the "You can submit with questions left blank" bullet:

```markdown
- You can also just type your answer and send it. While a question card is
  up, a plain message answers the questions you have not picked an option
  for (it becomes their **Other…** text) and is not sent to the agent as a
  new message. Two exceptions: a `/` command runs as usual, and a message
  with a staged attachment or staged Library evidence is sent as a normal
  turn, leaving the question up for you to answer on the card.
```

- [ ] **Step 2: Baseline and battery** — the M2 plan's Task 8 Steps 4-6 verbatim (detached `origin/dev` worktree, same file list plus `Tests/UI/test_console_ask_user_typed_answers.py` and `Tests/UI/test_console_command_composer.py` on the branch), `./scripts/preflight.sh`, and the census (`Tests/Performance/test_ui_ready_module_census.py`, expect no `+` line for this change).

- [ ] **Step 3: Commit, push, PR**

```bash
git add Docs/User_Guide/console/agent-runs-and-tools.md
git commit -m "docs(console): typed answers to an agent's question"
git push -u origin feat/console-ask-user-typed-answers
gh pr create --base dev --head feat/console-ask-user-typed-answers --title "feat(console): typed answers resolve a mounted ask_user question (PRD M3)" --body-file <body>
```

PR body: A8 mapping to the tests (AC-A6's three legs: plain send answers; `/` command dispatches normally; staged attachment dispatches normally -- name the test ids), baseline diff evidence, preflight, census. Then the standing merge recipe (Qodo → fix/answer → rebase → merge on the required check; Qodo re-reviews IN PLACE after a push, so merge by hand once the summary comment's `updated_at` is past the push and checks are green).

---

## Self-Review

**Spec coverage:** A8 sentence by sentence -- "A message sent while a question is mounted answers it" (Task 1 method + hook), "the text becomes `other_text` for every unanswered question" (the fill loop; already-answered questions keep their selections), "the round resolves" (`resolve_pending_question`), "the message is not also sent as a turn" (`return False` before dispatch, pinned by `test_visible_send_resolves_the_question_and_does_not_dispatch`), "slash commands dispatch normally and leave the question pending" (existing branch order, exercised by the command-composer suite), "a send with staged attachments or RAG evidence goes out as a normal turn" (`staged-attachment`, `staged-rag` cases). AC-A6 is those three legs. Only a mounted card intercepts (the `display` check); a parked round never has a mounted card on the viewed session.

**Placeholder scan:** none.

**Type consistency:** `collect_answers()` returns the M2 answer dicts (`question`, `selected`, `other_text`, `unanswered`) and `resolve_pending_question(answers, request_id=...)` takes exactly that list -- both from M2's plan, unchanged here.
