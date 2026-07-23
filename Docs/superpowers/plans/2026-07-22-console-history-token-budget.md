# Console History Token Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound the native Console conversation history by real tokens against the model window before dispatch — dropping oldest whole turns while always preserving the system prompt and the current turn, and telling the user when it trimmed.

**Architecture:** A new pure module `console_history_budget.py` does the multimodal-aware token counting and the whole-turn trimming (unit-tested with an injected deterministic counter). `console_chat_controller.py` calls it once at the single dispatch choke point (`_stream_assistant_response`, before the agent/direct branch — both branches use the passed `provider_messages`), and appends a display-only SYSTEM trim note when history was dropped.

**Spec:** `Docs/superpowers/specs/2026-07-22-console-history-token-budget-design.md` — read it first.

**Tech Stack:** Python ≥3.11, existing `tldw_chatbook/Utils/token_counter.py` seam (`get_model_token_limit`, `count_tokens_messages`), pytest + pytest-asyncio.

## Global Constraints

- Work in a **git worktree off `origin/dev`** (Task 1). The main checkout has another session's uncommitted work — never touch it.
- pytest runs from the **worktree's own venv**; verify `import tldw_chatbook` resolves to the worktree before trusting results. **tiktoken is NOT installed** — the real counter falls back to word-split estimation, so trim-logic tests inject a deterministic `count_fn` and never assert real-tokenizer numbers.
- Consume the existing `token_counter` seam (`get_model_token_limit`, `count_tokens_messages`); do NOT reimplement window tables or tokenizers (tasks 320/321 sharpen them behind the seam).
- Trim at the single choke point `_stream_assistant_response` (before the `_agent_runtime_enabled` branch) so the direct AND agent/tool paths are bounded.
- The trim note is a display-only `ConsoleMessageRole.SYSTEM` row (filtered from `_provider_message_payloads`); the store is append-only, so it renders with the exchange.
- Preserve: the contiguous leading `role=="system"` prefix, and the current turn (everything from the last `role=="user"` message to the end).
- `budget = window − response_reservation − max(512, window // 50)`; `response_reservation = self.max_tokens or DEFAULT_RESPONSE_RESERVATION (1024)`; `per_image_tokens = 1024`.
- Commit after each task; messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known dev-tip baseline: ~scheduling + shell/snapshot failures pre-exist, and `Tests/Chat` has 6 `mocker`-fixture errors (pytest-mock absent in a fresh venv). Judge regressions against that, not zero.

## File Structure

- `tldw_chatbook/Chat/console_history_budget.py` — NEW, pure: `BoundResult`, `DEFAULT_RESPONSE_RESERVATION`, `count_console_messages_tokens`, `bound_messages_to_window`, `_group_turns`. Single responsibility: count + trim. Imports only `token_counter`.
- `tldw_chatbook/Chat/console_chat_controller.py` — MODIFY: add the import; trim at the top of `_stream_assistant_response`; add `_append_history_trimmed_note`.
- `Tests/Chat/test_console_history_budget.py` — NEW: unit tests for the pure module.
- `Tests/Chat/test_console_chat_controller.py` — MODIFY: add the wiring tests.

---

### Task 1: Worktree, environment, docs, backlog task

**Files:**
- Create: worktree at `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-history-budget` on branch `feat/console-history-token-budget`
- Copy in: the spec + this plan from `chore/harness-review-tasks-320-334`

**Interfaces:**
- Produces: a green-baseline worktree every later task runs inside.

- [ ] **Step 1: Create the worktree off origin/dev**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git fetch origin
git worktree add ../tldw_chatbook-history-budget -b feat/console-history-token-budget origin/dev
cd ../tldw_chatbook-history-budget
```

- [ ] **Step 2: Bring the spec and plan onto the branch**

```bash
git checkout chore/harness-review-tasks-320-334 -- \
  "Docs/superpowers/specs/2026-07-22-console-history-token-budget-design.md" \
  "Docs/superpowers/plans/2026-07-22-console-history-token-budget.md"
git add Docs/superpowers
git commit -m "docs: import Console history token-budget spec and plan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Create the worktree venv (takes a few minutes)**

```bash
python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -c "import tldw_chatbook, pathlib; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

Expected: the printed path starts with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-history-budget/`. If it points at the main checkout, STOP.

- [ ] **Step 4: Baseline the controller test file**

```bash
.venv/bin/pytest Tests/Chat/test_console_chat_controller.py -q -p no:cacheprovider 2>&1 | tail -4
```

Expected: pass (record any pre-existing failures verbatim as the baseline).

- [ ] **Step 5: Create the backlog task record**

TASK-322 already exists in `backlog/tasks/`. Move it to In Progress and attach the plan:

```bash
backlog task edit 322 -s "In Progress" --plan "Docs/superpowers/plans/2026-07-22-console-history-token-budget.md"
git add backlog && git commit -m "docs(backlog): TASK-322 in progress, attach plan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Pure token-budget module

**Files:**
- Create: `tldw_chatbook/Chat/console_history_budget.py`
- Test: `Tests/Chat/test_console_history_budget.py`

**Interfaces:**
- Consumes: `tldw_chatbook.Utils.token_counter.get_model_token_limit`, `count_tokens_messages`.
- Produces (Task 3 relies on these exact names):
  - `DEFAULT_RESPONSE_RESERVATION = 1024`
  - `@dataclass(frozen=True) class BoundResult: messages: list[dict]; dropped_count: int`
  - `count_console_messages_tokens(messages: list[dict], model: str, *, per_image_tokens: int = 1024) -> int`
  - `bound_messages_to_window(messages: list[dict], *, model: str, provider: str, response_reservation: int, per_image_tokens: int = 1024, window: int | None = None, count_fn: Callable[[list[dict], str], int] | None = None) -> BoundResult`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_console_history_budget.py`:

```python
"""Unit tests for the Console history token-budget trimmer."""

from tldw_chatbook.Chat.console_history_budget import (
    BoundResult,
    DEFAULT_RESPONSE_RESERVATION,
    bound_messages_to_window,
    count_console_messages_tokens,
)


def _words(*texts: str) -> int:
    return sum(len(t.split()) for t in texts)


# A deterministic counter: 1 token per whitespace word in every string text
# part, plus 10 tokens per image part. tiktoken-independent.
def _wordcount(messages, model):  # noqa: ARG001
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text", "").split())
                else:
                    total += 10
        else:
            total += len(str(content).split())
    return total


def _msg(role, text):
    return {"role": role, "content": text}


def test_fits_under_budget_drops_nothing():
    msgs = [_msg("system", "sys"), _msg("user", "hi there"), _msg("assistant", "hello back")]
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=1000, count_fn=_wordcount
    )
    assert result.dropped_count == 0
    assert result.messages == msgs


def test_over_budget_drops_oldest_whole_turns_keeps_system_and_current():
    # window 20, reservation 0, margin max(512, 0) -> 512 makes budget negative;
    # use a big window and a big reservation instead to get a small positive budget.
    msgs = [
        _msg("system", "you are helpful"),          # 3 words
        _msg("user", "old question one two"),        # turn A (4)
        _msg("assistant", "old answer one two"),     # turn A (4)
        _msg("user", "mid question three four"),     # turn B (4)
        _msg("assistant", "mid answer three four"),  # turn B (4)
        _msg("user", "current question five"),       # current turn (3)
    ]
    # budget: window 1000 - reservation 980 - max(512, 20)=512 -> negative? No:
    # choose window 1000, reservation 0, but shrink via a tiny window instead:
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0,
        window=525, count_fn=_wordcount,  # budget = 525 - 0 - 512 = 13
    )
    # keep must fit in 13 tokens: system(3) + current(3) = 6; +turn B(8)=14 > 13 -> drop B too;
    # +turn A also dropped. So only system + current survive.
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user"]
    assert result.messages[0]["content"] == "you are helpful"
    assert result.messages[-1]["content"] == "current question five"
    # dropped the 4 middle messages (turns A + B)
    assert result.dropped_count == 4


def test_keeps_one_turn_when_it_fits():
    msgs = [
        _msg("system", "sys one"),                   # 2
        _msg("user", "old one two three four"),      # turn A (5)
        _msg("assistant", "old ans"),                # turn A (2)
        _msg("user", "current q"),                   # current (2)
    ]
    # window 521, reservation 0, margin 512 -> budget 9.
    # system(2)+current(2)=4; +turnA(7)=11 > 9 -> drop turn A. Result 4 <= 9.
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=521, count_fn=_wordcount
    )
    assert [m["role"] for m in result.messages] == ["system", "user"]
    assert result.dropped_count == 2


def test_degenerate_system_plus_current_over_budget_kept_anyway():
    msgs = [_msg("system", "a b c d e"), _msg("user", "f g h i j")]  # 5 + 5 = 10
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=515, count_fn=_wordcount
    )  # budget = 515 - 512 = 3 < 10, but nothing droppable
    assert result.messages == msgs
    assert result.dropped_count == 0


def test_window_override_takes_precedence_over_lookup():
    msgs = [_msg("user", "one two three four five six")]
    # No system, single user turn = current turn -> never dropped regardless.
    result = bound_messages_to_window(
        msgs, model="gpt-4", provider="openai", response_reservation=0, window=1, count_fn=_wordcount
    )
    assert result.dropped_count == 0
    assert result.messages == msgs


def test_leading_assistant_orphan_is_its_own_droppable_unit():
    msgs = [
        _msg("system", "s"),
        _msg("assistant", "orphan a b c d e f g h"),  # 9-word leading orphan
        _msg("user", "cur"),
    ]
    result = bound_messages_to_window(
        msgs, model="m", provider="p", response_reservation=0, window=515, count_fn=_wordcount
    )  # budget 3: system(1)+current(1)=2 fits; orphan(9) can't be added -> dropped
    assert [m["role"] for m in result.messages] == ["system", "user"]
    assert result.dropped_count == 1


def test_multimodal_content_counted_without_error_and_images_cost():
    # Real counter (no injected count_fn). Verifies list content doesn't crash
    # and each image adds per_image_tokens.
    text_only = [{"role": "user", "content": "hello world"}]
    with_image = [
        {"role": "user", "content": [
            {"type": "text", "text": "hello world"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]}
    ]
    base = count_console_messages_tokens(text_only, "gpt-4")
    withimg = count_console_messages_tokens(with_image, "gpt-4", per_image_tokens=1024)
    assert withimg >= base + 1024


def test_default_response_reservation_value():
    assert DEFAULT_RESPONSE_RESERVATION == 1024
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/pytest Tests/Chat/test_console_history_budget.py -q -p no:cacheprovider
```

Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Chat.console_history_budget`.

- [ ] **Step 3: Implement the module**

Create `tldw_chatbook/Chat/console_history_budget.py`:

```python
"""Bound Console conversation history by real tokens before dispatch.

Pure counting + whole-turn trimming, consumed by ConsoleChatController at the
dispatch choke point. Depends only on the token_counter seam (get_model_token_
limit / count_tokens_messages), which tasks 320/321 sharpen later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tldw_chatbook.Utils.token_counter import (
    count_tokens_messages,
    get_model_token_limit,
)

DEFAULT_RESPONSE_RESERVATION = 1024
DEFAULT_PER_IMAGE_TOKENS = 1024
_MIN_SAFETY_MARGIN = 512


@dataclass(frozen=True)
class BoundResult:
    """Result of trimming a provider message list to the model window."""

    messages: list[dict[str, Any]]
    dropped_count: int


def count_console_messages_tokens(
    messages: list[dict[str, Any]],
    model: str,
    *,
    per_image_tokens: int = DEFAULT_PER_IMAGE_TOKENS,
) -> int:
    """Token count for Console provider payloads, multimodal-aware.

    ``count_tokens_messages`` assumes string ``content`` and crashes on the
    Console's vision payloads (``content`` is a list of ``{type:text}`` /
    ``{type:image_url}`` parts). This flattens each list ``content`` to its
    concatenated text before delegating to ``count_tokens_messages`` (so
    text counting stays byte-identical, and 320/321 flow through), then adds
    ``per_image_tokens`` per image part.

    Args:
        messages: Provider payload dicts (``role``/``content``).
        model: Model name for the underlying tokenizer.
        per_image_tokens: Flat token estimate charged per image part.

    Returns:
        Estimated total prompt tokens.
    """
    flattened: list[dict[str, Any]] = []
    image_count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            texts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            image_count += sum(
                1
                for part in content
                if isinstance(part, dict) and part.get("type") != "text"
            )
            flattened.append(
                {**message, "content": " ".join(t for t in texts if t)}
            )
        else:
            flattened.append(message)
    return count_tokens_messages(flattened, model) + per_image_tokens * image_count


def _group_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group middle history into whole turns (a user + its following rows).

    Any rows before the first user message (e.g. a leading orphan assistant)
    form their own first group. Dropping a whole group never splits a
    user/assistant pair — nor a tool_call/tool_result pair, were tool rows
    ever present in the payload.
    """
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def bound_messages_to_window(
    messages: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    response_reservation: int,
    per_image_tokens: int = DEFAULT_PER_IMAGE_TOKENS,
    window: int | None = None,
    count_fn: Callable[[list[dict[str, Any]], str], int] | None = None,
) -> BoundResult:
    """Drop oldest whole turns until the payload fits the model window.

    Always preserves the leading system prefix and the current turn (from the
    last user message to the end). Returns the trimmed list and how many
    history messages were removed.

    Args:
        messages: Full provider payload, post dictionaries/skills.
        model: Model name (tokenizer + window lookup).
        provider: Provider name (window lookup fallback).
        response_reservation: Tokens reserved for the reply.
        per_image_tokens: Per-image token estimate.
        window: Explicit context window; ``None`` uses the token_counter lookup.
        count_fn: Injectable counter ``(messages, model) -> int``; ``None``
            uses ``count_console_messages_tokens``.

    Returns:
        ``BoundResult(messages, dropped_count)``.
    """
    counter = count_fn or (
        lambda msgs, mdl: count_console_messages_tokens(
            msgs, mdl, per_image_tokens=per_image_tokens
        )
    )
    win = window if window is not None else get_model_token_limit(model, provider)
    budget = win - response_reservation - max(_MIN_SAFETY_MARGIN, win // 50)

    # System prefix = contiguous leading system rows.
    sys_end = 0
    while sys_end < len(messages) and messages[sys_end].get("role") == "system":
        sys_end += 1
    system_prefix = messages[:sys_end]
    rest = messages[sys_end:]

    # Current turn = from the last user message to the end.
    last_user = None
    for index in range(len(rest) - 1, -1, -1):
        if rest[index].get("role") == "user":
            last_user = index
            break
    if last_user is None:
        # No user turn to anchor on -- nothing safe to trim.
        return BoundResult(messages, 0)

    current_turn = rest[last_user:]
    kept_turns = _group_turns(rest[:last_user])

    def assemble() -> list[dict[str, Any]]:
        return (
            system_prefix
            + [m for turn in kept_turns for m in turn]
            + current_turn
        )

    dropped = 0
    assembled = assemble()
    while kept_turns and counter(assembled, model) > budget:
        removed = kept_turns.pop(0)
        dropped += len(removed)
        assembled = assemble()

    return BoundResult(assembled, dropped)
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/pytest Tests/Chat/test_console_history_budget.py -q -p no:cacheprovider
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_history_budget.py Tests/Chat/test_console_history_budget.py
git commit -m "feat(console): pure token-budget trimmer for conversation history

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Wire the trim into the dispatch choke point

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (import ~line 14 block; `_stream_assistant_response` ~line 1826; add `_append_history_trimmed_note` near `_append_failure_system_row` ~line 1804)
- Modify: `Tests/Chat/test_console_chat_controller.py` (append wiring tests)

**Interfaces:**
- Consumes: `bound_messages_to_window`, `DEFAULT_RESPONSE_RESERVATION` (Task 2).
- Produces: no new public symbols; the send path now trims + notifies.

- [ ] **Step 1: Write the failing wiring tests**

Append to `Tests/Chat/test_console_chat_controller.py`. It already defines `RecordingStreamingGateway` (captures `messages_seen`), `StreamingGateway`, `ConsoleChatStore`, `ConsoleChatController`, `ConsoleProviderSelection`, `ConsoleMessageRole` and imports pytest — reuse them; add `from tldw_chatbook.Chat import console_history_budget` at the top. The send entry point is `await controller.submit_draft(<draft>)`, which sends on the **active** session (`controller.new_session(...)` creates + activates one). `ConsoleProviderSelection` uses `explicit_model=`/`configured_model=` (there is no `model=` field); its `max_tokens` is the response reservation.

```python
@pytest.mark.asyncio
async def test_send_trims_history_and_appends_note(monkeypatch):
    # Force a tiny window so a short history trims.
    monkeypatch.setattr(
        console_history_budget, "get_model_token_limit", lambda model, provider: 520
    )
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.update_provider_selection(
        ConsoleProviderSelection(
            provider="llama_cpp",
            explicit_model="test-model",
            configured_model="test-model",
            max_tokens=0,
        )
    )
    session = controller.new_session(title="Chat 1")  # creates + activates
    # Seed an over-budget history before the current turn.
    for i in range(6):
        store.append_message(session.id, role=ConsoleMessageRole.USER, content=f"old user {i} aa bb cc dd")
        store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content=f"old asst {i} aa bb cc dd")

    await controller.submit_draft("current question here")

    # The gateway saw a trimmed list (fewer than the full seeded history + turn).
    assert gateway.messages_seen is not None
    assert len(gateway.messages_seen) < 13
    # The latest user turn survived.
    assert any(
        m.get("role") == "user" and "current question here" in str(m.get("content", ""))
        for m in gateway.messages_seen
    )
    # A display-only SYSTEM trim note was appended to the transcript.
    rows = store.messages_for_session(session.id)
    assert any(
        r.role == ConsoleMessageRole.SYSTEM and "trimmed" in r.content.lower()
        for r in rows
    )


@pytest.mark.asyncio
async def test_send_that_fits_does_not_trim_or_note(monkeypatch):
    monkeypatch.setattr(
        console_history_budget, "get_model_token_limit", lambda model, provider: 100000
    )
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.update_provider_selection(
        ConsoleProviderSelection(
            provider="llama_cpp", explicit_model="test-model", configured_model="test-model"
        )
    )
    session = controller.new_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="one small turn")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="ok")

    await controller.submit_draft("next question")

    assert gateway.messages_seen is not None
    rows = store.messages_for_session(session.id)
    assert not any(
        r.role == ConsoleMessageRole.SYSTEM and "trimmed" in r.content.lower()
        for r in rows
    )
```

Note: if `submit_draft` dispatches on a different session than the seeded one (it calls `store.ensure_session(...)` on the active workspace), assert instead that `gateway.messages_seen` includes the seeded `"old user"` content to confirm the seeded history reached dispatch; adjust the session setup so the seeded rows are on the session `submit_draft` uses (activate it first, seed second).

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/pytest Tests/Chat/test_console_chat_controller.py -q -p no:cacheprovider \
  -k "trims_history or fits_does_not_trim"
```

Expected: FAIL — the trim note isn't appended / the gateway sees the full history (no trim wired yet).

- [ ] **Step 3: Add the import**

In `tldw_chatbook/Chat/console_chat_controller.py`, near the other `from tldw_chatbook.Chat...` imports (~line 14):

```python
from tldw_chatbook.Chat.console_history_budget import (
    DEFAULT_RESPONSE_RESERVATION,
    bound_messages_to_window,
)
```

- [ ] **Step 4: Add the trim note helper**

Add immediately after `_append_failure_system_row` (~line 1816):

```python
    def _append_history_trimmed_note(self, session_id: str, dropped: int) -> None:
        """Append a transcript-only system row noting history was trimmed."""
        try:
            self.store.append_message(
                session_id,
                role=ConsoleMessageRole.SYSTEM,
                content=(
                    "Earlier messages were trimmed to fit the model's context "
                    f"window ({dropped} dropped)."
                ),
            )
        except KeyError:
            # Session vanished mid-send; the dispatched payload was still bounded.
            pass
```

- [ ] **Step 5: Trim at the top of `_stream_assistant_response`**

In `_stream_assistant_response`, insert immediately after the `def` signature and BEFORE the `if self._agent_runtime_enabled and self._agent_bridge is not None:` branch (~line 1826):

```python
        bound = bound_messages_to_window(
            provider_messages,
            model=self.model or self.configured_model or "",
            provider=self.provider,
            response_reservation=self.max_tokens or DEFAULT_RESPONSE_RESERVATION,
        )
        provider_messages = bound.messages
        if bound.dropped_count:
            self._append_history_trimmed_note(
                self.store.session_id_for_message(assistant_message_id),
                bound.dropped_count,
            )
```

(The reassigned `provider_messages` flows into both the agent branch — which copies it — and the direct stream below.)

- [ ] **Step 6: Run the wiring tests + the file suite**

```bash
.venv/bin/pytest Tests/Chat/test_console_chat_controller.py -q -p no:cacheprovider 2>&1 | tail -6
```

Expected: the two new tests pass and the rest of the file is unchanged from the Task-1 baseline. If a pre-existing controller test now sees a trimmed payload or an extra SYSTEM row, it means that test seeds an over-window history under the real (untrimmed) assumption — inspect: if it asserts on the exact dispatched message count, update it to the trimmed expectation (never disable the trim); list any such change in the report.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): bound history by tokens at dispatch + trim note

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Full sweep, close-out, finish branch

**Files:**
- Modify: TASK-322 backlog file (AC checkboxes + implementation notes)

**Interfaces:**
- Consumes: everything prior.

- [ ] **Step 1: Targeted + broad sweep**

```bash
.venv/bin/pytest Tests/Chat/test_console_history_budget.py Tests/Chat/test_console_chat_controller.py -q -p no:cacheprovider 2>&1 | tail -6
.venv/bin/pytest Tests/Chat -q -p no:cacheprovider 2>&1 | tail -8
```

Classify every `Tests/Chat` failure: (a) documented baseline (the 6 `mocker`-fixture errors from missing pytest-mock; any anthropic-tools/etc. pre-existing failure), or (b) NOVEL. For any suspected-novel failure in a file this branch did not touch, reproduce it at `origin/dev` in a throwaway worktree before calling it a regression. Any novel regression must be fixed before proceeding.

- [ ] **Step 2: Close out TASK-322**

Check off all five ACs (`backlog task edit 322 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5`), add Implementation Notes (the pure module + counting adapter, the single-choke-point wiring covering agent+direct, the display-only note, the window-override seam for 325, intra-loop growth deferred to 326), set status Done (`backlog task edit 322 -s Done`), then commit the backlog change.

- [ ] **Step 3: Finish the branch**

Invoke `superpowers:finishing-a-development-branch`. PR targets `dev`. Re-verify TASK-322 has no ID collision at PR time (dev mints IDs concurrently). CI is intentionally cancelled in this repo — verify locally, don't block on CI.

---

## Self-Review Notes (already applied)

- Spec coverage: pure multimodal-aware counter + whole-turn trimmer with injectable counter (Task 2, AC#1/#2/#5); single-choke-point wiring covering direct + agent paths (Task 3, AC#1/#4); display-only SYSTEM trim note (Task 3, AC#3); boundary/preservation/degenerate/multimodal/window-override tests (Task 2) + trimming-vs-not controller tests (Task 3, AC#5). No gaps.
- Placeholders: none — full code for the module, the wiring, and every test; the two "confirm the exact name" notes point to real grep-checks, not deferred work.
- Type consistency: `bound_messages_to_window(..., window=None, count_fn=None) -> BoundResult`, `count_console_messages_tokens(messages, model, *, per_image_tokens=1024)`, `DEFAULT_RESPONSE_RESERVATION`, and `_append_history_trimmed_note(session_id, dropped)` are used identically in the module (Task 2) and at the call site (Task 3).
