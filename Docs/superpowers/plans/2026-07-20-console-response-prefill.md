# Console Response Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/prefill` in the native Console lets the user supply the opening of the assistant's reply; it is sent as a trailing `{"role": "assistant"}` provider message and the model continues from it.

**Architecture:** A pure parsing/validation module (`console_prefill.py`) + a `/prefill` slash command feed per-session state (one-shot on `ConsoleChatSession`, pinned on `ConsoleSessionSettings`). `_stream_assistant_response` gains a `prefill` parameter that bypasses the agent-runtime gate, appends the trailing assistant payload message, and seeds the display by pushing the prefill as the first stream chunk. Pinned persists in `conversations.metadata` via a merge-safe key write.

**Tech Stack:** Python ≥3.11, Textual, pytest (venv-only — activate `.venv`), SQLite (ChaChaNotes).

**Spec:** `Docs/superpowers/specs/2026-07-20-console-response-prefill-design.md` — read it before starting. Commit it with Task 1.

## Global Constraints

- Native Console only. Do NOT touch the legacy Chat window (`Chat_Window_Enhanced.py`, `chat_events.py`).
- No schema migration. `conversations.metadata` (JSON TEXT, v20) is shared with `active_dictionaries` — every metadata write must preserve sibling keys.
- Prefill text is stripped (`.strip()`) at arm time; empty-after-strip is an inline error; max 4,000 chars.
- Metadata key: `pinned_response_prefill`. Command name: `prefill`. Handler id: `prefill`.
- One-shot applies to `submit_draft` only and wins over pinned; consumed only on `complete`/`stopped` terminal outcomes; retained on blocked/failed. Pinned applies to submit/retry/regenerate; `continue_from_message` NEVER gets prefill.
- A prefilled send bypasses the agent-runtime gate (direct provider stream; tools/MCP skipped that turn).
- The display seed must never set the `emitted_content` flag in `_stream_assistant_response` (zero provider tokens must still fail with "Provider stream ended without content").
- User-visible text containing prefill content must be markup-escaped (`rich.markup.escape` via the codebase's existing `escape_markup` usage in chat_screen).
- Work in a git worktree off `origin/dev` (see `superpowers:using-git-worktrees`); this repo's checkout is shared by concurrent sessions.
- Run tests with the project venv: `source .venv/bin/activate && python -m pytest ...` from the worktree root.

---

### Task 1: Pure prefill module (`console_prefill.py`)

**Files:**
- Create: `tldw_chatbook/Chat/console_prefill.py`
- Test: `Tests/Chat/test_console_prefill.py`
- Also: `git add Docs/superpowers/specs/2026-07-20-console-response-prefill-design.md Docs/superpowers/plans/2026-07-20-console-response-prefill.md` in this task's commit.

**Interfaces:**
- Consumes: nothing (stdlib only — `dataclasses`, `json`, `typing`).
- Produces (used by Tasks 4, 6):
  - `PREFILL_MAX_CHARS: int = 4000`
  - `PINNED_PREFILL_METADATA_KEY: str = "pinned_response_prefill"`
  - `ACTION_STATUS/ACTION_CLEAR/ACTION_PIN/ACTION_ONE_SHOT/ACTION_ERROR: str` constants
  - `PrefillCommandAction(kind: str, text: str = "", error: str = "")` frozen dataclass
  - `parse_prefill_args(args: str) -> PrefillCommandAction`
  - `describe_prefill_preview(text: str, max_chars: int = 60) -> str`
  - `pinned_prefill_from_conversation_metadata(raw_metadata: object) -> str | None`

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_console_prefill.py`:

```python
"""Tests for the pure /prefill command parsing + metadata helpers."""

import json

from tldw_chatbook.Chat.console_prefill import (
    ACTION_CLEAR,
    ACTION_ERROR,
    ACTION_ONE_SHOT,
    ACTION_PIN,
    ACTION_STATUS,
    PINNED_PREFILL_METADATA_KEY,
    PREFILL_MAX_CHARS,
    PrefillCommandAction,
    describe_prefill_preview,
    parse_prefill_args,
    pinned_prefill_from_conversation_metadata,
)


class TestParsePrefillArgs:
    def test_bare_args_is_status(self):
        assert parse_prefill_args("") == PrefillCommandAction(kind=ACTION_STATUS)
        assert parse_prefill_args("   ") == PrefillCommandAction(kind=ACTION_STATUS)

    def test_clear_matches_only_as_entire_args(self):
        assert parse_prefill_args("clear").kind == ACTION_CLEAR
        assert parse_prefill_args("  CLEAR  ").kind == ACTION_CLEAR
        # 'clear' with trailing text is a one-shot whose text starts with 'clear'
        result = parse_prefill_args("clear the table, then")
        assert result == PrefillCommandAction(
            kind=ACTION_ONE_SHOT, text="clear the table, then"
        )

    def test_pin_requires_trailing_text(self):
        result = parse_prefill_args("pin *She pauses*")
        assert result == PrefillCommandAction(kind=ACTION_PIN, text="*She pauses*")
        bare_pin = parse_prefill_args("pin")
        assert bare_pin.kind == ACTION_ERROR
        assert "pin" in bare_pin.error

    def test_pin_word_prefix_is_one_shot(self):
        result = parse_prefill_args("pinch of salt")
        assert result == PrefillCommandAction(kind=ACTION_ONE_SHOT, text="pinch of salt")

    def test_plain_text_is_one_shot_stripped(self):
        result = parse_prefill_args("  {\"answer\":  ")
        assert result == PrefillCommandAction(kind=ACTION_ONE_SHOT, text="{\"answer\":")

    def test_over_length_is_error(self):
        result = parse_prefill_args("x" * (PREFILL_MAX_CHARS + 1))
        assert result.kind == ACTION_ERROR
        assert str(PREFILL_MAX_CHARS) in result.error

    def test_pin_over_length_is_error(self):
        result = parse_prefill_args("pin " + "x" * (PREFILL_MAX_CHARS + 1))
        assert result.kind == ACTION_ERROR

    def test_max_length_exactly_is_accepted(self):
        result = parse_prefill_args("x" * PREFILL_MAX_CHARS)
        assert result.kind == ACTION_ONE_SHOT
        assert len(result.text) == PREFILL_MAX_CHARS


class TestDescribePrefillPreview:
    def test_short_text_verbatim(self):
        assert describe_prefill_preview("Sure thing:") == "Sure thing:"

    def test_long_text_truncated_with_ellipsis(self):
        text = "a" * 100
        preview = describe_prefill_preview(text)
        assert len(preview) == 60
        assert preview.endswith("…")

    def test_newlines_flattened_to_spaces(self):
        assert describe_prefill_preview("line one\nline two") == "line one line two"


class TestPinnedPrefillFromConversationMetadata:
    def test_reads_key_from_json(self):
        raw = json.dumps({PINNED_PREFILL_METADATA_KEY: "*She pauses*"})
        assert pinned_prefill_from_conversation_metadata(raw) == "*She pauses*"

    def test_none_metadata_returns_none(self):
        assert pinned_prefill_from_conversation_metadata(None) is None

    def test_invalid_json_returns_none(self):
        assert pinned_prefill_from_conversation_metadata("{not json") is None

    def test_non_dict_json_returns_none(self):
        assert pinned_prefill_from_conversation_metadata("[1, 2]") is None

    def test_missing_key_returns_none(self):
        assert pinned_prefill_from_conversation_metadata("{}") is None

    def test_non_string_value_returns_none(self):
        raw = json.dumps({PINNED_PREFILL_METADATA_KEY: 42})
        assert pinned_prefill_from_conversation_metadata(raw) is None

    def test_blank_string_value_returns_none(self):
        raw = json.dumps({PINNED_PREFILL_METADATA_KEY: "   "})
        assert pinned_prefill_from_conversation_metadata(raw) is None

    def test_sibling_keys_ignored(self):
        raw = json.dumps(
            {"active_dictionaries": [1, 2], PINNED_PREFILL_METADATA_KEY: "Voice:"}
        )
        assert pinned_prefill_from_conversation_metadata(raw) == "Voice:"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_console_prefill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_prefill'`

- [ ] **Step 3: Write the module**

Create `tldw_chatbook/Chat/console_prefill.py`:

```python
"""Pure parsing/validation for the native Console ``/prefill`` command.

No dependency on Textual, the running app, or any I/O — mirrors
``console_command_grammar.py``. The screen layer owns all UI wiring; the
controller/store layers own arming state and payload assembly. This module
only classifies the ``/prefill`` args string and reads the pinned-prefill
key out of a raw conversation ``metadata`` JSON string.

Normalization: prefill text is ``.strip()``-ed at parse time. The spec
requires right-trimming (Anthropic rejects a trailing-whitespace assistant
turn); leading whitespace is stripped too for predictability, matching how
``/system`` name args are treated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PREFILL_MAX_CHARS = 4000
"""Maximum accepted prefill length; longer input is rejected at arm time."""

PINNED_PREFILL_METADATA_KEY = "pinned_response_prefill"
"""Key inside ``conversations.metadata`` JSON holding the pinned prefill."""

ACTION_STATUS = "status"
ACTION_CLEAR = "clear"
ACTION_PIN = "pin"
ACTION_ONE_SHOT = "one-shot"
ACTION_ERROR = "error"

_PIN_SUBCOMMAND = "pin"
_CLEAR_SUBCOMMAND = "clear"
_PREVIEW_MAX_CHARS = 60


@dataclass(frozen=True)
class PrefillCommandAction:
    """One classified ``/prefill`` invocation.

    Args:
        kind: One of the ``ACTION_*`` constants.
        text: Normalized prefill text for ``pin``/``one-shot`` kinds.
        error: Human-readable message for the ``error`` kind.
    """

    kind: str
    text: str = ""
    error: str = ""


def _validated(kind: str, text: str) -> PrefillCommandAction:
    if not text:
        return PrefillCommandAction(
            kind=ACTION_ERROR, error="Prefill text is empty."
        )
    if len(text) > PREFILL_MAX_CHARS:
        return PrefillCommandAction(
            kind=ACTION_ERROR,
            error=f"Prefill is too long ({len(text)} chars; max {PREFILL_MAX_CHARS}).",
        )
    return PrefillCommandAction(kind=kind, text=text)


def parse_prefill_args(args: str) -> PrefillCommandAction:
    """Classify the args string of one ``/prefill`` invocation.

    ``clear`` matches only as the entire (stripped) args; ``pin`` matches
    only as the first token with trailing text. A one-shot whose text
    literally starts with ``pin `` or equals ``clear`` therefore cannot be
    expressed — documented spec limitation.
    """
    stripped = args.strip()
    if not stripped:
        return PrefillCommandAction(kind=ACTION_STATUS)
    if stripped.lower() == _CLEAR_SUBCOMMAND:
        return PrefillCommandAction(kind=ACTION_CLEAR)
    first, _, remainder = stripped.partition(" ")
    if first.lower() == _PIN_SUBCOMMAND:
        pin_text = remainder.strip()
        if not pin_text:
            return PrefillCommandAction(
                kind=ACTION_ERROR, error="Usage: /prefill pin <text>."
            )
        return _validated(ACTION_PIN, pin_text)
    return _validated(ACTION_ONE_SHOT, stripped)


def describe_prefill_preview(text: str, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    """Return a single-line preview of ``text``, truncated with an ellipsis."""
    flattened = " ".join(text.split())
    if len(flattened) <= max_chars:
        return flattened
    return flattened[: max_chars - 1] + "…"


def pinned_prefill_from_conversation_metadata(raw_metadata: object) -> str | None:
    """Read the pinned prefill out of a raw conversation ``metadata`` value.

    Mirrors ``local_chat_dictionary_service._active_dictionaries``'s guarded
    parse (the ``json.loads(None)`` crash class): any missing/invalid shape
    yields ``None`` rather than raising.
    """
    try:
        meta = json.loads(raw_metadata or "{}")
    except (TypeError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    value = meta.get(PINNED_PREFILL_METADATA_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Chat/test_console_prefill.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_prefill.py Tests/Chat/test_console_prefill.py \
  Docs/superpowers/specs/2026-07-20-console-response-prefill-design.md \
  Docs/superpowers/plans/2026-07-20-console-response-prefill.md
git commit -m "feat(console): add pure /prefill parsing + metadata helpers"
```

---

### Task 2: Register `/prefill` in the command grammar

**Files:**
- Modify: `tldw_chatbook/Chat/console_command_grammar.py` (constants near line 36-44; `default_console_registry()` near line 162)
- Modify: `Tests/UI/test_console_command_composer.py:50-57` (the `UNKNOWN_NOPE_HINT` / `UNKNOWN_NADA_HINT` constants hardcode the available-command list — they break when `/prefill` registers)
- Test: `Tests/Chat/test_console_command_grammar.py`

**Interfaces:**
- Consumes: existing `ConsoleCommand`, `ConsoleCommandRegistry`.
- Produces (used by Task 6): `PREFILL_COMMAND_NAME = "prefill"`, `PREFILL_COMMAND_ARGUMENT_HINT = "[pin|clear] [text]"`, `PREFILL_COMMAND_HANDLER_ID = "prefill"`; `/prefill` registered in `default_console_registry()`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chat/test_console_command_grammar.py` (mirror the existing `/skills` test style; import `default_console_registry`, `CommandParse` as the file already does):

```python
def test_default_console_registry_includes_prefill():
    registry = default_console_registry()
    assert "prefill" in registry.available_names()


def test_prefill_parses_with_args():
    registry = default_console_registry()
    assert registry.parse("/prefill pin *She pauses*") == CommandParse(
        "command", "prefill", "pin *She pauses*"
    )
    assert registry.parse("/prefill") == CommandParse("command", "prefill", "")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Chat/test_console_command_grammar.py -v -k prefill`
Expected: FAIL — `'prefill' not in available_names()` (and the parse returns `unknown`).

- [ ] **Step 3: Implement registration**

In `tldw_chatbook/Chat/console_command_grammar.py`, after the `SKILLS_COMMAND_*` constants:

```python
PREFILL_COMMAND_NAME = "prefill"
PREFILL_COMMAND_ARGUMENT_HINT = "[pin|clear] [text]"
PREFILL_COMMAND_HANDLER_ID = "prefill"
```

In `default_console_registry()`, after the skills `registry.register(...)` block:

```python
    registry.register(
        ConsoleCommand(
            name=PREFILL_COMMAND_NAME,
            argument_hint=PREFILL_COMMAND_ARGUMENT_HINT,
            handler_id=PREFILL_COMMAND_HANDLER_ID,
        )
    )
```

Update the module docstring's command list if it enumerates commands, and the `default_console_registry` docstring ("built-in ``/prompt`` and ``/system``…") to mention `/prefill`.

- [ ] **Step 4: Fix the unknown-command hint constants**

In `Tests/UI/test_console_command_composer.py`, update both constants (the hint is derived from `available_names()`, so it now includes `/prefill`):

```python
UNKNOWN_NOPE_HINT = (
    "Unknown command /nope — available: /prompt, /system, /skills, /prefill. "
    "Press Enter again to send as text."
)
UNKNOWN_NADA_HINT = (
    "Unknown command /nada — available: /prompt, /system, /skills, /prefill. "
    "Press Enter again to send as text."
)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest Tests/Chat/test_console_command_grammar.py -v && python -m pytest Tests/UI/test_console_command_composer.py -v`
Expected: grammar tests all PASS. Composer tests pass at their pre-existing baseline (this repo has known pre-existing UI-test failures; the requirement is: no NEW failures vs. running them before this change — if unsure, `git stash && pytest ... && git stash pop` to compare).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_command_grammar.py \
  Tests/Chat/test_console_command_grammar.py Tests/UI/test_console_command_composer.py
git commit -m "feat(console): register /prefill slash command"
```

---

### Task 3: State homes — settings field, session field, store accessors

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py` (`ConsoleSessionSettings` fields end near line 176)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`ConsoleChatSession` dataclass ~line 123-134; accessors near `session_draft` ~line 315-323)
- Test: `Tests/Chat/test_console_session_settings.py`, `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: existing dataclasses.
- Produces (used by Tasks 4, 5, 6):
  - `ConsoleSessionSettings.pinned_prefill: str | None = None`
  - `ConsoleChatSession.one_shot_prefill: str | None = None`
  - `ConsoleChatStore.session_one_shot_prefill(session_id: str) -> str | None`
  - `ConsoleChatStore.set_session_one_shot_prefill(session_id: str, prefill: str | None) -> ConsoleChatSession`

- [ ] **Step 1: Write the failing tests**

In `Tests/Chat/test_console_session_settings.py` add (match the file's existing import of `ConsoleSessionSettings`; construct with the same minimal args existing tests use — `provider` is the only required field):

```python
def test_pinned_prefill_defaults_none_and_replaces():
    from dataclasses import replace

    settings = ConsoleSessionSettings(provider="llama_cpp")
    assert settings.pinned_prefill is None
    pinned = replace(settings, pinned_prefill="*She pauses*")
    assert pinned.pinned_prefill == "*She pauses*"
    assert settings.pinned_prefill is None
```

In `Tests/Chat/test_console_chat_store.py` add:

```python
def test_one_shot_prefill_accessors_round_trip():
    store = ConsoleChatStore()
    session = store.create_session(title="Chat 1")
    assert store.session_one_shot_prefill(session.id) is None
    store.set_session_one_shot_prefill(session.id, "Sure thing:")
    assert store.session_one_shot_prefill(session.id) == "Sure thing:"
    store.set_session_one_shot_prefill(session.id, None)
    assert store.session_one_shot_prefill(session.id) is None


def test_one_shot_prefill_is_per_session():
    store = ConsoleChatStore()
    session_a = store.create_session(title="A")
    session_b = store.create_session(title="B")
    store.set_session_one_shot_prefill(session_a.id, "only A")
    assert store.session_one_shot_prefill(session_b.id) is None
```

(Note: check how existing store tests call `create_session` — if it requires `workspace_id` or activation, copy the file's existing minimal-session construction exactly.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_chat_store.py -v -k prefill`
Expected: FAIL — `TypeError`/`AttributeError` (unknown field / missing methods).

- [ ] **Step 3: Implement**

In `console_session_settings.py`, add as the LAST field of `ConsoleSessionSettings` (after `source: str = "derived"`):

```python
    pinned_prefill: str | None = None
```

In `console_chat_store.py`, add to `ConsoleChatSession` after `pending_attachments`:

```python
    one_shot_prefill: str | None = None
```

Add accessors right after `set_session_draft` (mirroring the draft accessors):

```python
    def session_one_shot_prefill(self, session_id: str) -> str | None:
        """Return the armed one-shot response prefill for a session, if any."""
        return self._session_or_raise(session_id).one_shot_prefill

    def set_session_one_shot_prefill(
        self, session_id: str, prefill: str | None
    ) -> ConsoleChatSession:
        """Arm (or clear, with ``None``) the one-shot response prefill."""
        session = self._session_or_raise(session_id)
        session.one_shot_prefill = prefill
        return session
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_chat_store.py -v`
Expected: all PASS (new and pre-existing).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_session_settings.py tldw_chatbook/Chat/console_chat_store.py \
  Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): add prefill state homes (session settings + store)"
```

---

### Task 4: Pinned persistence — merge-safe metadata write + store write-through

**Files:**
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` (new method after `update_conversation_system_prompt`, ~line 222)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (Protocol `ConsoleChatPersistence` ~line 45-109; new `set_session_pinned_prefill` after `set_session_system_prompt` ~line 950; flush in `persist_session_if_needed` ~line 859-877)
- Test: `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: `PINNED_PREFILL_METADATA_KEY` from Task 1; `ConsoleSessionSettings.pinned_prefill` from Task 3.
- Produces (used by Tasks 5, 6):
  - `ChatPersistenceService.update_conversation_pinned_prefill(*, conversation_id: str, pinned_prefill: str | None) -> bool`
  - `ConsoleChatStore.set_session_pinned_prefill(session_id: str, prefill: str | None) -> tuple[ConsoleChatSession, bool]` (returns `(session, persisted)` exactly like `set_session_system_prompt`)
  - `persist_session_if_needed` flushes a pinned prefill when it first creates the conversation.

- [ ] **Step 1: Write the failing tests**

In `Tests/Chat/test_console_chat_store.py`:

(a) Extend the file's existing `FakePersistence` class (~line 444-510) with a recorder, following its `update_conversation_system_prompt` → `updated_system_prompts` pattern:

```python
    # inside FakePersistence.__init__:
    self.updated_pinned_prefills = []

    # new method on FakePersistence:
    def update_conversation_pinned_prefill(self, *, conversation_id, pinned_prefill):
        self.updated_pinned_prefills.append((conversation_id, pinned_prefill))
        return True
```

(b) New tests (copy session/persistence construction from the file's existing `set_session_system_prompt` tests; `ConsoleSessionSettings` import already exists in the file or add it):

```python
def test_set_session_pinned_prefill_updates_memory_and_writes_through():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(provider="llama_cpp")
    session.persisted_conversation_id = "conv-1"

    updated, persisted = store.set_session_pinned_prefill(session.id, "Voice:")
    assert persisted is True
    assert updated.settings.pinned_prefill == "Voice:"
    assert persistence.updated_pinned_prefills == [("conv-1", "Voice:")]

    updated, persisted = store.set_session_pinned_prefill(session.id, None)
    assert updated.settings.pinned_prefill is None
    assert persistence.updated_pinned_prefills[-1] == ("conv-1", None)


def test_set_session_pinned_prefill_blank_normalizes_to_none():
    store = ConsoleChatStore()
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(provider="llama_cpp")
    updated, persisted = store.set_session_pinned_prefill(session.id, "   ")
    assert updated.settings.pinned_prefill is None
    assert persisted is True  # no durable write needed


def test_set_session_pinned_prefill_persistence_failure_keeps_memory():
    class ExplodingPersistence(FakePersistence):
        def update_conversation_pinned_prefill(self, **kwargs):
            raise RuntimeError("db locked")

    persistence = ExplodingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(provider="llama_cpp")
    session.persisted_conversation_id = "conv-1"
    updated, persisted = store.set_session_pinned_prefill(session.id, "Voice:")
    assert persisted is False
    assert updated.settings.pinned_prefill == "Voice:"


def test_persist_session_if_needed_flushes_pinned_prefill():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Chat 1")
    session.settings = ConsoleSessionSettings(
        provider="llama_cpp", pinned_prefill="Voice:"
    )
    store.persist_session_if_needed(session.id)
    assert persistence.updated_pinned_prefills == [("conv-1", "Voice:")]
```

(c) Real-DB merge-safety test — append to the file's real-DB section (mirror the existing `CharactersRAGDB` construction at ~line 947-977 for imports and setup):

```python
def test_update_conversation_pinned_prefill_preserves_sibling_metadata(tmp_path):
    import json

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    service = ChatPersistenceService(db)
    conversation_id = service.create_conversation(
        assistant_kind="generic", assistant_id="console", conversation_title="T"
    )
    # Pre-seed a sibling key the dictionary-attach feature owns.
    record = db.get_conversation_by_id(conversation_id)
    db.update_conversation(
        conversation_id,
        {"metadata": json.dumps({"active_dictionaries": [1, 2]})},
        expected_version=record["version"],
    )

    assert service.update_conversation_pinned_prefill(
        conversation_id=conversation_id, pinned_prefill="Voice:"
    )
    meta = json.loads(db.get_conversation_by_id(conversation_id)["metadata"])
    assert meta["active_dictionaries"] == [1, 2]
    assert meta["pinned_response_prefill"] == "Voice:"

    assert service.update_conversation_pinned_prefill(
        conversation_id=conversation_id, pinned_prefill=None
    )
    meta = json.loads(db.get_conversation_by_id(conversation_id)["metadata"])
    assert meta["active_dictionaries"] == [1, 2]
    assert "pinned_response_prefill" not in meta

    assert not service.update_conversation_pinned_prefill(
        conversation_id="missing-conv", pinned_prefill="x"
    )
```

(If the real-DB tests in the file also build the workspace registry, copy that setup verbatim; `create_conversation` may require `workspace_id`/`scope_type` — match the neighboring test exactly.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Chat/test_console_chat_store.py -v -k "pinned_prefill or prefill"`
Expected: FAIL — missing methods.

- [ ] **Step 3: Implement `ChatPersistenceService.update_conversation_pinned_prefill`**

In `chat_persistence_service.py`, after `update_conversation_system_prompt` (add `import json` and the key import at the top: `from tldw_chatbook.Chat.console_prefill import PINNED_PREFILL_METADATA_KEY`):

```python
    def update_conversation_pinned_prefill(
        self,
        *,
        conversation_id: str,
        pinned_prefill: str | None,
    ) -> bool:
        """Set or clear the pinned response prefill in conversation metadata.

        Merge-safe: re-parses the current ``metadata`` JSON and rewrites only
        its own key, preserving siblings such as ``active_dictionaries``
        (mirrors ``LocalChatDictionaryService._write_active_dictionaries``).
        Optimistic-lock conflicts (``ConflictError``) propagate to the caller.

        Returns:
            True when the write happened; False when the conversation does
            not exist.
        """
        record = self.db.get_conversation_by_id(str(conversation_id))
        if record is None:
            return False
        try:
            meta = json.loads(record.get("metadata") or "{}")
        except (TypeError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        if pinned_prefill:
            meta[PINNED_PREFILL_METADATA_KEY] = pinned_prefill
        else:
            meta.pop(PINNED_PREFILL_METADATA_KEY, None)
        self.db.update_conversation(
            str(conversation_id),
            {"metadata": json.dumps(meta)},
            expected_version=record["version"],
        )
        return True
```

- [ ] **Step 4: Implement the store side**

In `console_chat_store.py`:

(a) Add to the `ConsoleChatPersistence` Protocol (after the `update_conversation_system_prompt` entry, matching its style):

```python
    def update_conversation_pinned_prefill(
        self,
        *,
        conversation_id: str,
        pinned_prefill: str | None,
    ) -> bool:
        """Set or clear the pinned response prefill on a conversation."""
        ...
```

(b) Add `set_session_pinned_prefill` right after `set_session_system_prompt` — copy its structure exactly (in-memory `replace(...)`, getattr-guarded write-through, catch-log-flag; the getattr guard keeps old fakes/persistence objects without the new method working):

```python
    def set_session_pinned_prefill(
        self, session_id: str, prefill: str | None
    ) -> tuple[ConsoleChatSession, bool]:
        """Set or clear a session's pinned response prefill.

        Mirrors ``set_session_system_prompt``: updates the in-memory
        settings snapshot and, when the session already owns a persisted
        conversation, writes through to conversation metadata. A durable
        write failure is caught and logged; the in-memory value is kept and
        the honest ``persisted`` flag is returned.
        """
        session = self._session_or_raise(session_id)
        normalized = prefill if isinstance(prefill, str) and prefill.strip() else None
        if session.settings is not None:
            session.settings = replace(session.settings, pinned_prefill=normalized)
        persisted = True
        if (
            session.persisted_conversation_id is not None
            and self.persistence is not None
        ):
            update_pinned = getattr(
                self.persistence, "update_conversation_pinned_prefill", None
            )
            if callable(update_pinned):
                try:
                    update_pinned(
                        conversation_id=session.persisted_conversation_id,
                        pinned_prefill=normalized,
                    )
                except Exception:
                    persisted = False
                    logger.bind(
                        session_id=session_id,
                        conversation_id=session.persisted_conversation_id,
                    ).exception(
                        "Failed to persist Console pinned prefill; "
                        "in-memory session keeps the applied value."
                    )
        return session, persisted
```

(c) In `persist_session_if_needed`, after `session.persisted_conversation_id = self.persistence.create_conversation(...)` and before `return`:

```python
        pinned_prefill = (
            session.settings.pinned_prefill if session.settings is not None else None
        )
        if pinned_prefill:
            update_pinned = getattr(
                self.persistence, "update_conversation_pinned_prefill", None
            )
            if callable(update_pinned):
                try:
                    update_pinned(
                        conversation_id=session.persisted_conversation_id,
                        pinned_prefill=pinned_prefill,
                    )
                except Exception:
                    logger.bind(
                        session_id=session_id,
                        conversation_id=session.persisted_conversation_id,
                    ).exception("Failed to flush pinned prefill on first persist.")
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest Tests/Chat/test_console_chat_store.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py \
  Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): merge-safe pinned-prefill persistence in conversation metadata"
```

---

### Task 5: Controller send path — resolve, bypass, payload, seed, consume

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`:
  - `submit_draft` (prefill resolution after `_apply_chat_dictionaries`, ~line 431)
  - `retry_message` / `regenerate_message` (pinned resolution before their `_stream_assistant_response` calls, ~lines 1069-1074 / 1170-1178)
  - `_stream_assistant_response` (~line 1817): new params, gate condition, payload append, mode-specific seeding, one-shot consumption
  - new private helpers `_pinned_prefill_for_session`, `_resolve_submit_prefill`, `_consume_one_shot_prefill`
- Test: `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Consumes: `session_one_shot_prefill` / `set_session_one_shot_prefill` / `session_settings` (Task 3); `ConsoleSessionSettings.pinned_prefill` (Task 3).
- Produces (used by Task 6): fully wired send behavior; no new public API beyond the two store-driven states — the screen arms state, the controller consumes it.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chat/test_console_chat_controller.py` (reuse the file's existing `StreamingGateway`, `RecordingStreamingGateway`, `BlockedGateway`, `FailingBeforeChunkGateway`, `EmptyStreamingGateway` fakes and its `ConsoleChatStore`/`ConsoleChatController` construction style; `ConsoleSessionSettings` import exists in the file):

```python
def _arm_session(store):
    """Create+activate a session with settings; return it."""
    session = store.ensure_session(
        workspace_id=store.workspace_context.active_workspace_id
    )
    if session.settings is None:
        session.settings = ConsoleSessionSettings(provider="llama_cpp")
    return session


@pytest.mark.asyncio
async def test_submit_with_one_shot_prefill_appends_trailing_assistant_and_seeds():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "Sure thing:")

    result = await controller.submit_draft("hello")
    assert result.submitted
    assert gateway.messages_seen[-1] == {
        "role": "assistant",
        "content": "Sure thing:",
    }
    assert gateway.messages_seen[-2]["role"] == "user"
    messages = store.messages_for_session(session.id)
    assert messages[-1].content == "Sure thing:ok"  # seed + RecordingStreamingGateway's "ok"
    assert messages[-1].status == "complete"
    # one-shot consumed on complete
    assert store.session_one_shot_prefill(session.id) is None


@pytest.mark.asyncio
async def test_submit_with_pinned_prefill_applies_and_survives():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "Voice:")

    await controller.submit_draft("hello")
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "Voice:"}
    # pinned survives the send
    assert store.session_settings(session.id).pinned_prefill == "Voice:"


@pytest.mark.asyncio
async def test_one_shot_wins_over_pinned_then_pinned_resumes():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "PINNED")
    store.set_session_one_shot_prefill(session.id, "ONESHOT")

    await controller.submit_draft("first")
    assert gateway.messages_seen[-1]["content"] == "ONESHOT"
    await controller.submit_draft("second")
    assert gateway.messages_seen[-1]["content"] == "PINNED"


@pytest.mark.asyncio
async def test_blocked_send_retains_one_shot():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "KEEP")
    await controller.submit_draft("hello")
    assert store.session_one_shot_prefill(session.id) == "KEEP"


@pytest.mark.asyncio
async def test_failed_send_retains_one_shot_and_shows_prefill():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingBeforeChunkGateway()
    )
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "KEEP")
    await controller.submit_draft("hello")
    assert store.session_one_shot_prefill(session.id) == "KEEP"
    messages = store.messages_for_session(session.id)
    assert messages[-1].status == "failed"
    assert messages[-1].content == "KEEP"  # seed materialized, no provider tokens


@pytest.mark.asyncio
async def test_zero_token_stream_fails_with_prefill_only_content():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=EmptyStreamingGateway()
    )
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "PRE")
    await controller.submit_draft("hello")
    messages = store.messages_for_session(session.id)
    assert messages[-1].status == "failed"
    assert messages[-1].content == "PRE"
    assert store.session_one_shot_prefill(session.id) == "PRE"


@pytest.mark.asyncio
async def test_stop_mid_stream_consumes_one_shot():
    store = ConsoleChatStore()

    class StopAfterFirstChunkGateway(StreamingGateway):
        def __init__(self):
            self.controller = None

        async def stream_chat(self, resolution, messages):
            yield "partial"
            self.controller._stop_requested = True
            yield "never-shown"

    gateway = StopAfterFirstChunkGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    gateway.controller = controller
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "PRE")
    await controller.submit_draft("hello")
    messages = store.messages_for_session(session.id)
    assert messages[-1].status == "stopped"
    assert messages[-1].content.startswith("PRE")
    assert store.session_one_shot_prefill(session.id) is None


@pytest.mark.asyncio
async def test_retry_zero_tokens_leaves_failed_content_untouched():
    """A pinned-prefill retry that yields no tokens must not seed: the lazy
    prepare_message_retry never runs, so the original failed content (the
    seed from the first attempt) stays exactly as it was."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingBeforeChunkGateway()
    )
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "PINNED")
    await controller.submit_draft("hello")
    failed = store.messages_for_session(session.id)[-1]
    assert failed.status == "failed"
    assert failed.content == "PINNED"  # seed from the failed first attempt

    controller.provider_gateway = EmptyStreamingGateway()
    await controller.retry_message(failed.id)
    after = store.get_message(failed.id)
    assert after.status == "failed"
    assert after.content == "PINNED"  # untouched — no double-seed, no wipe


@pytest.mark.asyncio
async def test_retry_applies_pinned_but_not_one_shot():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingBeforeChunkGateway()
    )
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "PINNED")
    await controller.submit_draft("hello")
    failed = store.messages_for_session(session.id)[-1]
    assert failed.status == "failed"

    gateway = RecordingStreamingGateway()
    controller.provider_gateway = gateway
    result = await controller.retry_message(failed.id)
    assert result.submitted
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "PINNED"}
    retried = store.get_message(failed.id)
    assert retried.status == "complete"
    assert retried.content == "PINNEDok"


@pytest.mark.asyncio
async def test_regenerate_applies_pinned_into_variant():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    await controller.submit_draft("hello")
    original = store.messages_for_session(session.id)[-1]
    store.set_session_pinned_prefill(session.id, "PINNED")

    await controller.regenerate_message(original.id)
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "PINNED"}
    regenerated = store.get_message(original.id)
    assert regenerated.content == "PINNEDok"


@pytest.mark.asyncio
async def test_continue_never_gets_prefill():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    await controller.submit_draft("hello")
    assistant = store.messages_for_session(session.id)[-1]
    store.set_session_pinned_prefill(session.id, "PINNED")
    store.set_session_one_shot_prefill(session.id, "ONESHOT")

    await controller.continue_from_message(assistant.id)
    # continue keeps its synthetic USER instruction; nothing assistant-trailing
    assert gateway.messages_seen[-1]["role"] == "user"
    # one-shot untouched (continue is not a normal send)
    assert store.session_one_shot_prefill(session.id) == "ONESHOT"


@pytest.mark.asyncio
async def test_prefilled_send_bypasses_agent_loop():
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome

    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return RunOutcome(status=RUN_DONE, steps=[], final_text="agent says")

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    session = _arm_session(store)

    # Control: without prefill the agent path handles the send.
    await controller.submit_draft("no prefill")
    assert len(bridge_calls) == 1
    assert gateway.messages_seen is None

    # With prefill armed the direct provider path handles it.
    store.set_session_one_shot_prefill(session.id, "PRE")
    await controller.submit_draft("with prefill")
    assert len(bridge_calls) == 1  # unchanged
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "PRE"}
```

Adjust the control-test expectations to the real agent-path plumbing if `_run_agent_reply` requires more bridge surface than `run_reply` (e.g. `_agent_conversation_id` needs a persisted store, or `RunOutcome` needs more fields) — the file's existing agent-path tests around lines 1671-1776 and 2182-2218 show the working patterns; mirror them. The assertion that matters is: **prefill ⇒ `gateway.messages_seen` is populated and the bridge is NOT called; no prefill ⇒ bridge called.** If making the no-prefill control leg work requires disproportionate fake-bridge scaffolding, keep only the prefill leg plus a direct unit assertion that the gate condition includes `prefill is None`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/Chat/test_console_chat_controller.py -v -k prefill`
Expected: FAIL — payloads lack the trailing assistant message; contents lack the seed; one-shot never consumed.

- [ ] **Step 3: Implement controller helpers**

In `console_chat_controller.py`, near the other small private helpers (e.g. after `_ensure_user_continuation_instruction`):

```python
    def _pinned_prefill_for_session(self, session_id: str) -> str | None:
        """Return the session's pinned response prefill, if any."""
        settings = self.store.session_settings(session_id)
        pinned = getattr(settings, "pinned_prefill", None) if settings else None
        return pinned or None

    def _resolve_submit_prefill(self, session_id: str) -> tuple[str | None, bool]:
        """Return ``(prefill, from_one_shot)`` for a normal send.

        One-shot wins over pinned for the send it is armed for; pinned
        resumes afterward (the one-shot is only cleared on a complete or
        stopped outcome — see ``_consume_one_shot_prefill``).
        """
        one_shot = self.store.session_one_shot_prefill(session_id)
        if one_shot:
            return one_shot, True
        return self._pinned_prefill_for_session(session_id), False

    def _consume_one_shot_prefill(
        self, assistant_message_id: str, used_one_shot: bool
    ) -> None:
        """Clear the armed one-shot after a send that used it terminated
        ``complete`` or ``stopped``. Blocked and failed sends never call
        this, so retry reproduces the original intent (spec §2)."""
        if not used_one_shot:
            return
        try:
            session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return
        self.store.set_session_one_shot_prefill(session_id, None)
```

- [ ] **Step 4: Thread `prefill` through `_stream_assistant_response`**

Modify the signature (line ~1817):

```python
    async def _stream_assistant_response(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, str]],
        assistant_message_id: str,
        prepare_retry: bool = False,
        variant_mode: bool = False,
        prefill: str | None = None,
        prefill_from_one_shot: bool = False,
    ) -> ConsoleSubmitResult:
```

Change the agent gate (first statement) to bypass when a prefill applies (spec §2: a prefilled send is a raw model continuation — the agent loop is skipped for that turn):

```python
        if (
            self._agent_runtime_enabled
            and self._agent_bridge is not None
            and prefill is None
        ):
            return await self._run_agent_reply(...)  # unchanged call
```

After the gate, append the trailing assistant payload message (a copy, so callers' lists are not mutated) — insert right after the gate, before `self._active_assistant_message_id = assistant_message_id`:

```python
        if prefill:
            provider_messages = [
                *provider_messages,
                {
                    "role": ConsoleMessageRole.ASSISTANT.value,
                    "content": prefill,
                },
            ]
```

Seed the display for normal/variant modes — insert right after the existing `if variant_mode: self.store.begin_variant_stream(assistant_message_id)` block and before `self._set_run_state(... STREAMING ...)`. Retry mode must NOT seed here (a failed message rejects chunks until `prepare_message_retry` runs, and the lazy prepare preserves original content on a zero-token retry):

```python
        if prefill and not prepare_retry:
            try:
                self.store.append_stream_chunk(assistant_message_id, prefill)
            except KeyError:
                return self._session_closed_result()
```

Seed retry mode at the lazy-prepare point — the existing block inside the chunk loop becomes:

```python
                if prepare_retry and not retry_prepared:
                    self.store.prepare_message_retry(assistant_message_id)
                    retry_prepared = True
                    if prefill:
                        try:
                            self.store.append_stream_chunk(
                                assistant_message_id, prefill
                            )
                        except KeyError:
                            return self._session_closed_result()
```

(Do NOT touch `emitted_content` in any seed path — it must keep reflecting provider chunks only.)

Consume the one-shot at the terminal outcomes that used it. Add `self._consume_one_shot_prefill(assistant_message_id, prefill_from_one_shot)` immediately BEFORE each of these four `return` statements in the direct path (all in the current function body):
1. the mid-loop stop: `return ConsoleSubmitResult(True, True, stopped.content)` (after the first `_mark_stream_stopped` inside the chunk loop)
2. the post-loop stop: same pattern after the loop
3. the completion return: `return ConsoleSubmitResult(True, True, completed.content)`
4. the `asyncio.CancelledError` stop: the `return ConsoleSubmitResult(True, True, stopped.content)` inside that handler

Do NOT add it to the failed/blocked/exception exits.

- [ ] **Step 5: Wire the three send sites**

`submit_draft` — after `provider_messages = await self._apply_chat_dictionaries(provider_messages, session.id)` and before `self._notify_submission_accepted()`:

```python
        prefill, prefill_from_one_shot = self._resolve_submit_prefill(session.id)
```

…and extend its `_stream_assistant_response` call:

```python
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=assistant.id,
            prefill=prefill,
            prefill_from_one_shot=prefill_from_one_shot,
        )
```

`retry_message` — before its `_stream_assistant_response` call add `prefill = self._pinned_prefill_for_session(session_id)` and pass `prefill=prefill` (no `prefill_from_one_shot`):

```python
        prefill = self._pinned_prefill_for_session(session_id)
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            prepare_retry=True,
            prefill=prefill,
        )
```

`regenerate_message` — same pattern:

```python
        prefill = self._pinned_prefill_for_session(session_id)
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            variant_mode=True,
            prefill=prefill,
        )
```

`continue_from_message` — unchanged (no prefill argument).

- [ ] **Step 6: Run the new tests, then the full controller + store suites**

Run: `python -m pytest Tests/Chat/test_console_chat_controller.py -v -k prefill`
Expected: all new tests PASS.
Run: `python -m pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py -v`
Expected: no regressions (every pre-existing test still passes — these suites are green at baseline).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): apply response prefill on the direct send path with agent-loop bypass"
```

---

### Task 6: Screen wiring — `/prefill` handler, resume load, inspector rows

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`:
  - imports (grammar constants + console_prefill helpers)
  - `_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` (~line 8975) and the `dispatch_map` in `_dispatch_console_command` (~line 9007)
  - new handler `_console_command_prefill` (place after `_console_command_apply_system`, ~line 9276)
  - `_console_session_settings_for_resume` (~line 3225)
  - `_selected_console_conversation_inspector_rows` (~line 5950)
- Test: `Tests/UI/test_console_command_composer.py`

**Interfaces:**
- Consumes: Task 1 (`parse_prefill_args`, `describe_prefill_preview`, `pinned_prefill_from_conversation_metadata`, `ACTION_*`), Task 2 (`PREFILL_COMMAND_NAME`, `PREFILL_COMMAND_HANDLER_ID`), Task 3/4 store accessors (`session_one_shot_prefill`, `set_session_one_shot_prefill`, `set_session_pinned_prefill`, `session_settings`).
- Produces: complete user-facing feature.

- [ ] **Step 1: Write the failing screen test**

Add to `Tests/UI/test_console_command_composer.py`, using the file's exact harness pattern (`_build_test_app` + `_configure_native_ready_console` + `ConsoleHarness`, drive via `composer.load_draft(...)` + `#console-send-message` press, assert via `_system_message_contents(console)` — same as `test_console_unknown_command_first_enter_renders_hint_and_does_not_send` at line ~80):

```python
@pytest.mark.asyncio
async def test_console_prefill_command_arms_one_shot_and_confirms():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prefill Sure thing:")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Prefill armed for next send")

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert store.session_one_shot_prefill(session_id) == "Sure thing:"
        assert composer.draft_text() == ""  # handled command clears its draft


@pytest.mark.asyncio
async def test_console_prefill_pin_and_clear_round_trip():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = console.query_one("#console-send-message", Button)
        store = console._ensure_console_chat_store()

        composer.load_draft("/prefill pin Voice:")
        send_button.press()
        await _wait_for_text(console, pilot, "Prefill pinned")
        session_id = store.active_session_id
        assert store.session_settings(session_id).pinned_prefill == "Voice:"

        composer.load_draft("/prefill clear")
        send_button.press()
        await _wait_for_text(console, pilot, "Prefill cleared")
        assert store.session_settings(session_id).pinned_prefill is None
        assert store.session_one_shot_prefill(session_id) is None
        assert composer.draft_text() == ""
```

(If `store.active_session_id` is `None` before the first command lands, the handler's `ensure_session` creates it — read `session_id` AFTER the first send, as written above.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest Tests/UI/test_console_command_composer.py -v -k prefill`
Expected: FAIL — `/prefill` dispatches to no handler (nothing armed, no system message).

- [ ] **Step 3: Implement dispatch wiring + handler**

In `chat_screen.py`:

(a) Extend the grammar import (the file already imports `PROMPT_COMMAND_NAME` etc. from `console_command_grammar`) with `PREFILL_COMMAND_NAME, PREFILL_COMMAND_HANDLER_ID`, and add a new import:

```python
from tldw_chatbook.Chat.console_prefill import (
    ACTION_CLEAR,
    ACTION_ERROR,
    ACTION_ONE_SHOT,
    ACTION_PIN,
    ACTION_STATUS,
    describe_prefill_preview,
    parse_prefill_args,
    pinned_prefill_from_conversation_metadata,
)
```

(b) `_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` — add `PREFILL_COMMAND_NAME: PREFILL_COMMAND_HANDLER_ID,`.

(c) `dispatch_map` in `_dispatch_console_command` — add `PREFILL_COMMAND_HANDLER_ID: self._console_command_prefill,`.

(d) The handler (after `_console_command_apply_system`; use the codebase's existing markup-escape import in this file — check how existing system messages escape user text and reuse that exact call, referred to below as `escape_markup`):

```python
    async def _console_command_prefill(self, parse: CommandParse) -> None:
        """Arm, pin, clear, or report the Console response prefill (`/prefill`).

        One-shot (`/prefill <text>`) applies to the next normal send only
        and wins over pinned; `/prefill pin <text>` applies to every
        submit/retry/regenerate until cleared and write-throughs to
        conversation metadata when the session is persisted. Errors leave
        the draft in place for correction (mirrors `/system`'s
        no-system-part behavior); handled outcomes clear it.
        """
        action = parse_prefill_args(parse.args)
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id
        )
        if action.kind == ACTION_ERROR:
            await self._append_native_console_system_message(
                escape_markup(action.error)
            )
            return
        if action.kind == ACTION_STATUS:
            one_shot = store.session_one_shot_prefill(session.id)
            settings = store.session_settings(session.id)
            pinned = getattr(settings, "pinned_prefill", None) if settings else None
            lines = []
            if one_shot:
                lines.append(
                    "Prefill (next send only): "
                    f"'{escape_markup(describe_prefill_preview(one_shot))}'"
                )
            if pinned:
                lines.append(
                    f"Prefill (pinned): '{escape_markup(describe_prefill_preview(pinned))}'"
                )
            if not lines:
                lines.append("No prefill armed.")
            await self._append_native_console_system_message("\n".join(lines))
            self._clear_console_composer_draft()
            return
        if action.kind == ACTION_CLEAR:
            store.set_session_one_shot_prefill(session.id, None)
            _session, persisted = store.set_session_pinned_prefill(session.id, None)
            copy = "Prefill cleared."
            if not persisted:
                copy += " (Warning: saved conversation not updated.)"
            await self._append_native_console_system_message(copy)
            self._clear_console_composer_draft()
            return
        preview = escape_markup(describe_prefill_preview(action.text))
        if action.kind == ACTION_PIN:
            _session, persisted = store.set_session_pinned_prefill(
                session.id, action.text
            )
            copy = (
                f"Prefill pinned: '{preview}'. Applies to every send, retry, and "
                "regenerate until /prefill clear. The reply continues directly "
                "from the last character; tool calling is skipped on prefilled sends."
            )
            if not persisted:
                copy += " (Warning: saved conversation not updated.)"
            await self._append_native_console_system_message(copy)
            self._clear_console_composer_draft()
            return
        # ACTION_ONE_SHOT
        store.set_session_one_shot_prefill(session.id, action.text)
        await self._append_native_console_system_message(
            f"Prefill armed for next send: '{preview}'. The reply continues "
            "directly from the last character; tool calling is skipped on "
            "prefilled sends."
        )
        self._clear_console_composer_draft()
```

After writing it, read `_apply_console_session_system_prompt` (`chat_screen.py:9326`) and append the same inspector/session-state refresh tail it uses after mutating settings (whatever call it makes to re-render the inspector — reuse it verbatim after the pin/clear/one-shot mutations). If it makes no such call (the inspector refreshes on its own cadence), add nothing.

- [ ] **Step 4: Resume load + inspector rows**

(a) `_console_session_settings_for_resume` — change the return to also restore the pinned prefill:

```python
        pinned_prefill = pinned_prefill_from_conversation_metadata(
            conversation.get("metadata")
        )
        return replace(
            settings, system_prompt=system_prompt, pinned_prefill=pinned_prefill
        )
```

(b) `_selected_console_conversation_inspector_rows` — before the final `return (...)`, build optional prefill rows from the in-memory session (NO DB I/O — this function already holds `active_session`) and include them in the returned tuple:

```python
        prefill_rows: list[ConsoleDisplayRow] = []
        one_shot = active_session.one_shot_prefill
        if one_shot:
            prefill_rows.append(
                ConsoleDisplayRow(
                    "Prefill (next send only)", describe_prefill_preview(one_shot)
                )
            )
        session_settings = active_session.settings
        pinned = (
            session_settings.pinned_prefill if session_settings is not None else None
        )
        if pinned:
            prefill_rows.append(
                ConsoleDisplayRow("Prefill (pinned)", describe_prefill_preview(pinned))
            )
```

…and append `*prefill_rows` to the returned tuple of rows.

- [ ] **Step 5: Run tests**

Run: `python -m pytest Tests/UI/test_console_command_composer.py -v -k prefill`
Expected: new tests PASS.
Run: `python -m pytest Tests/UI/test_console_command_composer.py Tests/Chat/ -v`
Expected: no NEW failures vs. baseline (see Task 2 Step 5 note about the pre-existing UI-failure baseline).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_command_composer.py
git commit -m "feat(console): /prefill command, resume restore, and what's-in-play rows"
```

---

### Task 7: Full verification + live smoke

**Files:**
- Modify (only if fixes needed): files from Tasks 1-6.
- Modify: `Docs/superpowers/specs/2026-07-20-console-response-prefill-design.md` — record the §5 verification-item outcomes (Gemini / Responses-API) in a short "Verification results" appendix.

**Interfaces:** none new — this is the spec's §5/§7 verification pass.

- [ ] **Step 1: Full test suites**

Run: `python -m pytest Tests/Chat/ -v` — expected: all pass.
Run: `python -m pytest Tests/UI/test_console_command_composer.py Tests/UI/test_console_native_chat_flow.py -v` — expected: no new failures vs. baseline.

- [ ] **Step 2: Live smoke against the local llama-server**

Launch the app (`python3 -m tldw_chatbook.app` — run from the worktree root; note the `python -m` cwd trap: make sure the worktree, not the main checkout, is on `sys.path`) with the local llama-server running on `:9099`. In the Console:
1. `/prefill Sure thing, here is the JSON: {` then send "give me a 2-key sample object" → reply must visibly start with the prefill text; inspector shows nothing armed afterward (one-shot consumed).
2. `/prefill pin *speaking as the ship's AI*` → send twice → both replies start with the pin; "What's in play" shows the pinned row; regenerate the last reply → variant also starts with the pin.
3. `/prefill clear` → next send has no prefill.
4. `/prefill` bare → status line correct at each stage.
5. Save/persist the conversation (send while persistence is on), quit, relaunch, resume the conversation → pinned prefill still applies (metadata round-trip).
6. Stop a prefilled response mid-stream → message keeps prefill + partial, status stopped.

- [ ] **Step 3: Spec §5 verification items (provider compatibility)**

With whatever providers are configured locally (skip gracefully if no key, but record which were checked):
- Anthropic: prefilled send → confirm literal continuation and no API error (right-trim guard).
- Gemini (`chat_with_google`): prefilled send → confirm the trailing `model`-role turn is accepted. If the handler/API rejects it, implement the spec §5 incompatible-provider guard IN THIS BRANCH before finishing: in `_stream_assistant_response`, when `resolution.provider` is in the rejecting set, drop the prefill from BOTH the payload and the display seed and post an inline system row ("Prefill skipped: <provider> does not accept a prefilled response.") — the transcript must never show text the provider did not receive. Add a controller test for the guard. Do not silently ship a provider that errors.
- OpenAI with Responses API enabled (`use_responses_api`): same check for the `input`-shaped branch.
Record all three outcomes in the spec appendix.

- [ ] **Step 4: Run the repo verify flow**

Use the `verify` skill (drive the real app flow, not just tests) if not already covered by Step 2. Self-review the full diff (`git diff dev...HEAD`).

- [ ] **Step 5: Commit + finish**

```bash
git add -A && git commit -m "test(console): prefill verification results + spec appendix"
```

Then follow `superpowers:finishing-a-development-branch` (PR against `dev`; CI is intentionally cancelled by another build in this repo — verify locally, don't block on CI).
