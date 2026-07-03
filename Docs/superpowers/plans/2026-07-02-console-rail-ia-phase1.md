# Console Rail IA (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the Console left rail into four collapsible sections (Session / Context / Model / Details), auto-title conversations from the first user message, and show relative-age labels on conversation rows.

**Architecture:** Pure-state modules (`console_chat_models`, `console_rail_state`, `conversation_browser_state`, `console_session_settings`) gain small pure functions/fields first, each unit-tested without Textual. New widgets (`ConsoleRailSectionHeader`, `ConsoleWorkspaceDetailsTray`) stay single-purpose. `chat_screen.py` changes are orchestration only: compose the four sections, dispatch toggle buttons by id (existing idiom), persist via the existing `console.rail_state` config path.

**Tech Stack:** Python 3.11+, Textual, pytest + pytest-asyncio (Textual pilot), existing `_build_test_app`/`ConsoleHarness` UI harness.

**Spec:** `Docs/superpowers/specs/2026-07-02-console-dual-audience-ux-design.md` (§1 Rail IA, §8 Phase 1). Two spec items — recent-first ordering and workspace scoping — already exist in `conversation_browser_state.py` (`_sort_normal_rows`, `default_collapsed = workspace_id != active_workspace_id`); they get regression tests only (Task 3).

## Global Constraints

- Run tests with the venv interpreter and isolated home so real config is never touched:
  `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q <target> --tb=short`
- The `timeout` shell command is not available in this environment.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is **generated**. Edit only `tldw_chatbook/css/components/_agentic_terminal.tcss`, then run `./build_css.sh` and commit both files.
- ~33 pre-existing UI test failures are the known baseline; judge regressions against files you touch, not the global count. CI checks may be cancelled intentionally — verify locally.
- Section titles are exactly `Session`, `Context`, `Model`, `Details`. First-run defaults: Session/Context/Model open, Details collapsed.
- Section toggles keep the existing `-`/`+` glyphs (matching the conversation-browser toggles). The `▸`/`▾` glyph pass is Phase 4, not this plan.
- Preserve existing widget ids `#console-staged-context-tray`, `#console-workspace-context`, `#console-left-rail`, `#console-left-rail-body` — tests and sync methods query them.
- Auto-title truncation limit: 30 characters including the `...` suffix.
- Console screen changes require screenshot QA and explicit user approval before merge (Task 11).
- Commit after each task. End commit messages with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: Auto-title pure helpers

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py` (near `DEFAULT_CONSOLE_SESSION_TITLE`, line 36)
- Test: `Tests/Chat/test_console_chat_models.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `is_default_console_session_title(title: str) -> bool`, `derive_console_session_title(draft: str, *, max_length: int = 30) -> str`, constant `CONSOLE_AUTO_TITLE_MAX_LENGTH = 30`. Task 2 imports all three from `tldw_chatbook.Chat.console_chat_models`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_chat_models.py`:

```python
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_AUTO_TITLE_MAX_LENGTH,
    derive_console_session_title,
    is_default_console_session_title,
)


def test_is_default_console_session_title_matches_chat_number_pattern():
    assert is_default_console_session_title("Chat 1")
    assert is_default_console_session_title("  Chat 42  ")
    assert not is_default_console_session_title("API refactor plan")
    assert not is_default_console_session_title("Chat")
    assert not is_default_console_session_title("Chat one")
    assert not is_default_console_session_title("")


def test_derive_console_session_title_collapses_whitespace():
    assert derive_console_session_title("  fix   the\nlogin  bug ") == "fix the login bug"


def test_derive_console_session_title_truncates_long_drafts():
    draft = "please review the workspace registry service for thread safety"
    title = derive_console_session_title(draft)
    assert len(title) <= CONSOLE_AUTO_TITLE_MAX_LENGTH
    assert title.endswith("...")
    assert title == "please review the workspace..."


def test_derive_console_session_title_empty_draft_returns_empty():
    assert derive_console_session_title("   \n  ") == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_chat_models.py -k "auto_title or default_console_session_title or derive_console" --tb=short`

Expected: FAIL with `ImportError: cannot import name 'derive_console_session_title'`.

- [ ] **Step 3: Implement the helpers**

In `tldw_chatbook/Chat/console_chat_models.py`, add `import re` to the imports, then below `DEFAULT_CONSOLE_SESSION_TITLE = "Chat 1"` add:

```python
CONSOLE_AUTO_TITLE_MAX_LENGTH = 30
_DEFAULT_CONSOLE_SESSION_TITLE_RE = re.compile(r"^Chat \d+$")


def is_default_console_session_title(title: str) -> bool:
    """Return whether a session title is an auto-numbered default like ``Chat 3``."""
    return bool(_DEFAULT_CONSOLE_SESSION_TITLE_RE.match(str(title or "").strip()))


def derive_console_session_title(
    draft: str,
    *,
    max_length: int = CONSOLE_AUTO_TITLE_MAX_LENGTH,
) -> str:
    """Derive a conversation title from the first user message.

    Args:
        draft: Validated composer draft text.
        max_length: Maximum title length including the ellipsis suffix.

    Returns:
        A collapsed, truncated title, or an empty string for blank drafts.
    """
    collapsed = " ".join(str(draft or "").split())
    if not collapsed:
        return ""
    if len(collapsed) <= max_length:
        return collapsed
    return f"{collapsed[: max_length - 3].rstrip()}..."
```

- [ ] **Step 4: Run tests to verify they pass**

Same command as Step 2. Expected: PASS (4 new tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py Tests/Chat/test_console_chat_models.py
git commit -m "feat(console): add auto-title derivation helpers"
```

---

### Task 2: Auto-title hook in submit_draft

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`submit_draft`, lines 96–134)
- Test: `Tests/Chat/test_console_chat_controller.py`

**Interfaces:**
- Consumes: Task 1 helpers; `ConsoleChatStore.rename_session(session_id, title)` (exists, `console_chat_store.py:192`).
- Produces: behavior only — an accepted first send renames a default-titled, not-yet-persisted session before the user message is appended, so `persist_session_if_needed` creates the conversation with the derived title.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_chat_controller.py` (self-contained; no fixtures needed):

```python
from types import SimpleNamespace

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class _AutoTitleReadyGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(ready=True, visible_copy="")

    async def stream_chat(self, resolution, messages):
        yield "ok"


def _auto_title_controller() -> ConsoleChatController:
    return ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=_AutoTitleReadyGateway(),
    )


@pytest.mark.asyncio
async def test_submit_draft_auto_titles_default_session_from_first_message():
    controller = _auto_title_controller()
    session = controller.new_session()
    assert session.title == "Chat 1"

    await controller.submit_draft("fix the login bug in the auth flow")

    assert controller.store.sessions()[0].title == "fix the login bug in the au..."


@pytest.mark.asyncio
async def test_submit_draft_preserves_user_renamed_session_title():
    controller = _auto_title_controller()
    session = controller.new_session()
    controller.store.rename_session(session.id, "My research thread")

    await controller.submit_draft("hello there")

    assert controller.store.sessions()[0].title == "My research thread"


@pytest.mark.asyncio
async def test_submit_draft_does_not_retitle_after_first_send():
    controller = _auto_title_controller()
    controller.new_session()

    await controller.submit_draft("first message decides the title")
    first_title = controller.store.sessions()[0].title
    await controller.submit_draft("second message must not retitle")

    assert controller.store.sessions()[0].title == first_title
```

(The expected literal is deterministic: the draft collapses to 34 chars, so the title is the first 27 chars — `"fix the login bug in the au"` — plus `"..."`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_chat_controller.py -k auto_title --tb=short`

Expected: FAIL — title remains `"Chat 1"`.

- [ ] **Step 3: Implement the hook**

In `console_chat_controller.py`, extend the models import (line 9–14) with `derive_console_session_title` and `is_default_console_session_title`, and add `ConsoleChatSession` to the existing `console_chat_store` import (line 15). In `submit_draft`, immediately before `self.store.append_message(session.id, role=ConsoleMessageRole.USER, ...)` (line 117), insert:

```python
        self._maybe_auto_title_session(session, clean_draft)
```

Add the private method after `new_session`:

```python
    def _maybe_auto_title_session(self, session: ConsoleChatSession, draft: str) -> None:
        """Title a default-named session from its first accepted message."""
        if session.persisted_conversation_id is not None:
            return
        if not is_default_console_session_title(session.title):
            return
        derived = derive_console_session_title(draft)
        if derived:
            self.store.rename_session(session.id, derived)
```

(`session` in `submit_draft` comes from `ensure_session`, which returns the live store object, so reading `title`/`persisted_conversation_id` directly is current state; the rename still goes through the store's public `rename_session` for validation.)

- [ ] **Step 4: Run tests to verify they pass**

Same command as Step 2, plus the full controller file:
`env HOME=... .venv/bin/python -m pytest -q Tests/Chat/test_console_chat_controller.py --tb=short`

Expected: new tests PASS, no regressions in the file.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): auto-title sessions from first accepted message"
```

---

### Task 3: Relative-age labels in the conversation browser builder

**Files:**
- Modify: `tldw_chatbook/Workspaces/conversation_browser_state.py`
- Test: `Tests/Workspaces/test_console_conversation_browser_state.py`

**Interfaces:**
- Consumes: existing `_normalize_input_row` (line 309), `build_console_conversation_browser_state` (line 165).
- Produces: `format_console_relative_age(value: str, *, now: datetime) -> str`; `build_console_conversation_browser_state` gains keyword `now: datetime | None = None` and auto-fills `updated_label` from `updated_sort` when the input row's label is empty. Task 4 relies on this: any row with a parseable `updated_sort` gets a label with no caller changes.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Workspaces/test_console_conversation_browser_state.py` (reuse the file's existing input-row factory if one exists; otherwise construct `ConsoleConversationBrowserInputRow` directly):

```python
from datetime import datetime, timezone

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
    format_console_relative_age,
)

_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


def test_format_console_relative_age_buckets():
    assert format_console_relative_age("2026-07-02T11:59:40+00:00", now=_NOW) == "now"
    assert format_console_relative_age("2026-07-02T11:58:00+00:00", now=_NOW) == "2m"
    assert format_console_relative_age("2026-07-02T10:59:00+00:00", now=_NOW) == "1h"
    assert format_console_relative_age("2026-06-29T12:00:00+00:00", now=_NOW) == "3d"
    assert format_console_relative_age("2026-06-10T12:00:00+00:00", now=_NOW) == "3w"
    assert format_console_relative_age("2024-06-10T12:00:00+00:00", now=_NOW) == "2y"


def test_format_console_relative_age_tolerates_bad_input():
    assert format_console_relative_age("", now=_NOW) == ""
    assert format_console_relative_age("not a timestamp", now=_NOW) == ""
    # SQLite space-separated naive timestamps are treated as UTC.
    assert format_console_relative_age("2026-07-02 11:58:00", now=_NOW) == "2m"
    # Future timestamps clamp to "now".
    assert format_console_relative_age("2026-07-02T13:00:00+00:00", now=_NOW) == "now"


def _input_row(**overrides):
    defaults = dict(
        row_key="conv-1",
        conversation_id="conv-1",
        native_session_id=None,
        title="Example",
        scope_type="workspace",
        workspace_id="ws-1",
        workspace_label="Workspace 1",
        updated_sort="2026-07-02T11:58:00+00:00",
    )
    defaults.update(overrides)
    return ConsoleConversationBrowserInputRow(**defaults)


def test_builder_fills_updated_label_from_updated_sort():
    state = build_console_conversation_browser_state(
        rows=[_input_row()],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    row = workspaces.groups[0].rows[0]
    assert row.updated_label == "2m"


def test_builder_keeps_caller_supplied_updated_label():
    state = build_console_conversation_browser_state(
        rows=[_input_row(updated_label="today")],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    assert workspaces.groups[0].rows[0].updated_label == "today"


def test_non_active_workspace_groups_default_collapsed_regression():
    state = build_console_conversation_browser_state(
        rows=[
            _input_row(),
            _input_row(row_key="conv-2", conversation_id="conv-2", workspace_id="ws-2", workspace_label="Workspace 2"),
        ],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    by_id = {group.group_id: group for group in workspaces.groups}
    assert not by_id["workspace:ws-1"].collapsed
    assert by_id["workspace:ws-2"].collapsed


def test_rows_sorted_recent_first_regression():
    state = build_console_conversation_browser_state(
        rows=[
            _input_row(row_key="old", conversation_id="old", title="Old", updated_sort="2026-06-01T00:00:00+00:00"),
            _input_row(row_key="new", conversation_id="new", title="New", updated_sort="2026-07-01T00:00:00+00:00"),
        ],
        active_workspace_id="ws-1",
        now=_NOW,
    )
    workspaces = next(s for s in state.sections if s.section_id == "workspaces")
    titles = [row.title for row in workspaces.groups[0].rows]
    assert titles == ["New", "Old"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Workspaces/test_console_conversation_browser_state.py -k "relative_age or updated_label or default_collapsed_regression or recent_first" --tb=short`

Expected: FAIL with `ImportError: cannot import name 'format_console_relative_age'`. (The two regression tests may pass immediately — that is fine; they lock in existing behavior.)

- [ ] **Step 3: Implement**

In `conversation_browser_state.py`, add to imports: `from datetime import datetime, timezone`. Add near the top-level helpers:

```python
def _parse_browser_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_console_relative_age(value: str, *, now: datetime) -> str:
    """Return a compact age label such as ``2m``, ``1h``, ``3d`` for a timestamp.

    Args:
        value: ISO-8601-ish timestamp text; naive values are treated as UTC.
        now: Reference time for age calculation.

    Returns:
        Compact age label, or an empty string when the value is unparseable.
    """
    parsed = _parse_browser_timestamp(value)
    if parsed is None:
        return ""
    total_seconds = max(0.0, (now - parsed).total_seconds())
    minutes = int(total_seconds // 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 7:
        return f"{days}d"
    weeks = days // 7
    if days < 365:
        return f"{weeks}w"
    return f"{days // 365}y"
```

In `build_console_conversation_browser_state` (line 165), add parameter `now: datetime | None = None` (after `group_row_limit`), and at the top of the body:

```python
    reference_now = now or datetime.now(timezone.utc)
```

Change the `prepared_rows` line (line 203) to pass it through:

```python
    prepared_rows = tuple(_normalize_input_row(row, now=reference_now) for row in rows)
```

Change `_normalize_input_row` (line 309) signature to `def _normalize_input_row(row: ConsoleConversationBrowserInputRow, *, now: datetime) -> ConsoleConversationBrowserInputRow:` and replace its `updated_label=` line with:

```python
        updated_label=(
            str(row.updated_label or "")
            or format_console_relative_age(str(row.updated_sort or ""), now=now)
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

`env HOME=... .venv/bin/python -m pytest -q Tests/Workspaces/test_console_conversation_browser_state.py --tb=short`

Expected: PASS including all pre-existing tests (the `now` parameter is optional, so existing callers are unaffected).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/conversation_browser_state.py Tests/Workspaces/test_console_conversation_browser_state.py
git commit -m "feat(console): fill conversation row age labels from updated_sort"
```

---

### Task 4: Activity timestamps for native sessions

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`ConsoleChatSession` line 67–76, `append_message` line 307)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_native_console_browser_rows`, line 2105–2142)
- Test: `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: Task 3 (labels auto-derive from `updated_sort`).
- Produces: `ConsoleChatSession.updated_at: str` (ISO-8601 UTC), refreshed by `append_message`. `_native_console_browser_rows` sets `updated_sort=session.updated_at` so native rows get age labels for free.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_chat_store.py`:

```python
from datetime import datetime

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def test_create_session_records_updated_at():
    store = ConsoleChatStore()
    session = store.create_session()
    parsed = datetime.fromisoformat(session.updated_at)
    assert parsed.tzinfo is not None


def test_append_message_touches_session_updated_at():
    store = ConsoleChatStore()
    session = store.create_session()
    original = session.updated_at
    store._sessions[session.id].updated_at = "2020-01-01T00:00:00+00:00"

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")

    touched = store._sessions[session.id].updated_at
    assert touched != "2020-01-01T00:00:00+00:00"
    assert datetime.fromisoformat(touched) >= datetime.fromisoformat(original)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_chat_store.py -k updated_at --tb=short`

Expected: FAIL with `AttributeError: ... no attribute 'updated_at'`.

- [ ] **Step 3: Implement**

In `console_chat_store.py`, add to imports: `from datetime import datetime, timezone`. Add above the dataclass:

```python
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
```

Add to `ConsoleChatSession` (after `draft: str = ""`):

```python
    updated_at: str = field(default_factory=_utc_now_iso)
```

In `append_message` (line 307), after `self._messages_by_session[session_id].append(message)` add:

```python
        self._sessions[session_id].updated_at = _utc_now_iso()
```

In `chat_screen.py` `_native_console_browser_rows` (line 2139), replace `updated_sort="",` with:

```python
                updated_sort=str(session.updated_at or ""),
```

- [ ] **Step 4: Run tests to verify they pass**

`env HOME=... .venv/bin/python -m pytest -q Tests/Chat/test_console_chat_store.py --tb=short`

Expected: PASS, no regressions (existing `restore_state`/`restore_persisted_session` tests still pass — `replace(session)` copies the new field automatically).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): track native session activity for age labels"
```

---

### Task 5: Section open/collapse preferences

**Files:**
- Modify: `tldw_chatbook/Chat/console_rail_state.py`
- Test: `Tests/Chat/test_console_rail_state.py`

**Interfaces:**
- Consumes: existing `ConsoleRailPreferences`, `coerce_console_rail_preferences` (line 170), `serialize_console_rail_preferences` (line 189), `ConsoleRailState` (line 79), `build_console_rail_state` (line 355).
- Produces: constants `CONSOLE_RAIL_SECTION_IDS = ("session", "context", "model", "details")`; `ConsoleRailPreferences` fields `session_open=True, context_open=True, model_open=True, details_open=False`; matching fields on `ConsoleRailState`, populated by `build_console_rail_state`. Task 9 reads `rail_state.session_open` etc. at compose time and writes prefs via `dataclasses.replace`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_rail_state.py`:

```python
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_RAIL_SECTION_IDS,
    ConsoleRailPreferences,
    build_console_rail_preference_key,
    build_console_rail_state,
    coerce_console_rail_preferences,
    serialize_console_rail_preferences,
)


def test_console_rail_section_defaults():
    prefs = ConsoleRailPreferences()
    assert CONSOLE_RAIL_SECTION_IDS == ("session", "context", "model", "details")
    assert prefs.session_open is True
    assert prefs.context_open is True
    assert prefs.model_open is True
    assert prefs.details_open is False


def test_coerce_console_rail_preferences_reads_section_fields():
    coerced = coerce_console_rail_preferences(
        {"left_open": True, "details_open": "true", "model_open": "off"}
    )
    assert coerced.details_open is True
    assert coerced.model_open is False
    assert coerced.session_open is True  # missing key -> default


def test_serialize_console_rail_preferences_round_trips_sections():
    prefs = ConsoleRailPreferences(details_open=True, context_open=False)
    serialized = serialize_console_rail_preferences(prefs)
    assert serialized["details_open"] is True
    assert serialized["context_open"] is False
    assert coerce_console_rail_preferences(serialized) == prefs


def test_build_console_rail_state_carries_section_flags():
    key = build_console_rail_preference_key(workspace_id="ws", session_id="s")
    state = build_console_rail_state(
        preference_key=key,
        stored_preferences={"details_open": True, "session_open": False},
    )
    assert state.details_open is True
    assert state.session_open is False
    assert state.context_open is True
    assert state.model_open is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py -k section --tb=short`

Expected: FAIL with `ImportError: cannot import name 'CONSOLE_RAIL_SECTION_IDS'`.

- [ ] **Step 3: Implement**

In `console_rail_state.py`:

Below `CONSOLE_RAIL_RIGHT_DEFAULT_OPEN` (line 11) add:

```python
CONSOLE_RAIL_SECTION_IDS = ("session", "context", "model", "details")
```

Extend `ConsoleRailPreferences` (line 61):

```python
@dataclass(frozen=True)
class ConsoleRailPreferences:
    """Persisted user preferences for Console side rail openness."""

    left_open: bool = CONSOLE_RAIL_LEFT_DEFAULT_OPEN
    right_open: bool = CONSOLE_RAIL_RIGHT_DEFAULT_OPEN
    session_open: bool = True
    context_open: bool = True
    model_open: bool = True
    details_open: bool = False
```

Extend `ConsoleRailState` (line 79) with the same four fields and defaults, after `right_forced_collapsed: bool = False`:

```python
    session_open: bool = True
    context_open: bool = True
    model_open: bool = True
    details_open: bool = False
```

Extend `coerce_console_rail_preferences` (line 170) return:

```python
    return ConsoleRailPreferences(
        left_open=_coerce_bool(raw.get("left_open"), defaults.left_open),
        right_open=_coerce_bool(raw.get("right_open"), defaults.right_open),
        session_open=_coerce_bool(raw.get("session_open"), defaults.session_open),
        context_open=_coerce_bool(raw.get("context_open"), defaults.context_open),
        model_open=_coerce_bool(raw.get("model_open"), defaults.model_open),
        details_open=_coerce_bool(raw.get("details_open"), defaults.details_open),
    )
```

Extend `serialize_console_rail_preferences` (line 189):

```python
    return {
        "left_open": bool(preferences.left_open),
        "right_open": bool(preferences.right_open),
        "session_open": bool(preferences.session_open),
        "context_open": bool(preferences.context_open),
        "model_open": bool(preferences.model_open),
        "details_open": bool(preferences.details_open),
    }
```

In `build_console_rail_state` (line 355), add to the returned `ConsoleRailState`:

```python
        session_open=preferences.session_open,
        context_open=preferences.context_open,
        model_open=preferences.model_open,
        details_open=preferences.details_open,
```

Note: `_coerce_bool` already treats `"off"` as False via `_FALSE_STRINGS`.

- [ ] **Step 4: Run tests to verify they pass**

`env HOME=... .venv/bin/python -m pytest -q Tests/Chat/test_console_rail_state.py --tb=short`

Expected: PASS including all pre-existing tests (new fields have defaults; stored dicts without them coerce to defaults).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_rail_state.py Tests/Chat/test_console_rail_state.py
git commit -m "feat(console): persist rail section open/collapse preferences"
```

---

### Task 6: ConsoleRailSectionHeader widget

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_rail_section.py`
- Test: `Tests/UI/test_console_rail_sections.py` (new file)

**Interfaces:**
- Consumes: nothing project-specific.
- Produces: `ConsoleRailSectionHeader(title, *, section_id, open, **kwargs)` composing a title `Static` (`#console-rail-section-title-{section_id}`) and toggle `Button` (`#console-rail-section-toggle-{section_id}`), plus `sync_open(open: bool)` and constant `CONSOLE_RAIL_SECTION_TOGGLE_PREFIX = "console-rail-section-toggle-"`. Task 9 composes four of these and dispatches `Button.Pressed` by the id prefix.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_console_rail_sections.py`:

```python
"""Console rail section header widget contracts."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)


class _HeaderApp(App):
    def compose(self):
        yield ConsoleRailSectionHeader(
            "Details",
            section_id="details",
            open=False,
            id="header-under-test",
        )


@pytest.mark.asyncio
async def test_rail_section_header_renders_title_and_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        title = app.query_one("#console-rail-section-title-details", Static)
        assert str(getattr(title.renderable, "plain", title.renderable)) == "Details"
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "+"
        assert toggle.tooltip == "Expand Details"


@pytest.mark.asyncio
async def test_rail_section_header_sync_open_flips_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        header = app.query_one("#header-under-test", ConsoleRailSectionHeader)
        header.sync_open(True)
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "-"
        assert toggle.tooltip == "Collapse Details"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_rail_sections.py --tb=short`

Expected: FAIL with `ModuleNotFoundError` for `console_rail_section`.

- [ ] **Step 3: Implement the widget**

Create `tldw_chatbook/Widgets/Console/console_rail_section.py`:

```python
"""Collapsible Console left-rail section header chrome."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

CONSOLE_RAIL_SECTION_TOGGLE_PREFIX = "console-rail-section-toggle-"


class ConsoleRailSectionHeader(Horizontal):
    """One-line rail section header with a collapse/expand toggle.

    Attributes:
        title: User-facing section title.
        section_id: Stable section id used in child widget ids.
        open: Whether the associated section body is currently visible.
    """

    def __init__(
        self,
        title: str,
        *,
        section_id: str,
        open: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="console-rail-section-header", **kwargs)
        self.title = title
        self.section_id = section_id
        self.open = open
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1

    def compose(self) -> ComposeResult:
        title = Static(
            self.title,
            id=f"console-rail-section-title-{self.section_id}",
            classes="console-rail-section-title",
            markup=False,
        )
        title.styles.width = "1fr"
        yield title
        toggle = Button(
            self._toggle_label(),
            id=f"{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            classes="console-workspace-action console-rail-section-toggle",
            compact=True,
        )
        toggle.tooltip = self._toggle_tooltip()
        toggle.styles.width = 3
        toggle.styles.min_width = 3
        toggle.styles.max_width = 3
        yield toggle

    def sync_open(self, open: bool) -> None:
        """Refresh the toggle affordance after the section body visibility changes."""
        self.open = open
        toggle = self.query_one(
            f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            Button,
        )
        toggle.label = self._toggle_label()
        toggle.tooltip = self._toggle_tooltip()

    def _toggle_label(self) -> str:
        return "-" if self.open else "+"

    def _toggle_tooltip(self) -> str:
        return f"Collapse {self.title}" if self.open else f"Expand {self.title}"
```

- [ ] **Step 4: Run test to verify it passes**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_rail_section.py Tests/UI/test_console_rail_sections.py
git commit -m "feat(console): add rail section header widget"
```

---

### Task 7: Extract ConsoleWorkspaceDetailsTray

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_workspace_details.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Test: `Tests/UI/test_console_rail_sections.py` (extend)

**Interfaces:**
- Consumes: `ConsoleWorkspaceContextState` (`tldw_chatbook/Workspaces/display_state.py:185`), `ConsoleWorkspaceStatusPair` (imported from `console_workspace_context`).
- Produces: `ConsoleWorkspaceDetailsTray(state, **kwargs)` — a `Vertical` rendering exactly the rows `_compose_status_rows` renders today (Storage/Sync/File tools/Server handoff/Handoff list/ACP handoff), with `sync_state(state)` that recomposes. `ConsoleWorkspaceContextTray` no longer renders status rows and gains `show_heading: bool = True`. All existing status-row widget ids (`#console-workspace-authority-label`, `#console-workspace-sync-label`, `#console-workspace-runtime-label`, `#console-workspace-server-readiness-label`, `#console-workspace-handoff-title`, `#console-workspace-handoff-rows`, `#console-workspace-acp-handoff-detail`, `#console-workspace-acp-handoff-audit`, …) are preserved inside the new tray.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_rail_sections.py`:

```python
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


def _workspace_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Default",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Runtime: none, file tools disabled",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )


class _DetailsApp(App):
    def compose(self):
        yield ConsoleWorkspaceDetailsTray(_workspace_state(), id="details-tray")


@pytest.mark.asyncio
async def test_details_tray_renders_status_and_handoff_rows():
    app = _DetailsApp()
    async with app.run_test(size=(60, 30)):
        assert app.query_one("#console-workspace-authority-label")
        assert app.query_one("#console-workspace-sync-label")
        assert app.query_one("#console-workspace-runtime-label")
        assert app.query_one("#console-workspace-server-readiness-label")
        assert app.query_one("#console-workspace-handoff-title")
        assert app.query_one("#console-workspace-acp-handoff-audit")


class _ContextTrayApp(App):
    def compose(self):
        yield ConsoleWorkspaceContextTray(
            _workspace_state(),
            show_heading=False,
            id="context-tray",
        )


@pytest.mark.asyncio
async def test_context_tray_without_heading_omits_status_rows():
    app = _ContextTrayApp()
    async with app.run_test(size=(60, 30)):
        assert not list(app.query("#console-workspace-context-title"))
        assert not list(app.query("#console-workspace-authority-label"))
        assert not list(app.query("#console-workspace-handoff-title"))
        assert app.query_one("#console-workspace-selected-conversation")
```

(If `ConsoleWorkspaceContextState` requires more constructor arguments than shown, copy the minimal-state factory from `Tests/Workspaces/test_workspace_display_state.py` instead of `_workspace_state()` above — keep the assertions identical.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_rail_sections.py -k "details_tray or without_heading" --tb=short`

Expected: FAIL with `ModuleNotFoundError` for `console_workspace_details`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Widgets/Console/console_workspace_details.py`. **Move** (do not copy) these members out of `ConsoleWorkspaceContextTray` in `console_workspace_context.py`: `_compose_status_rows` (line 869), `_status_pair` (line 298), `_split_status_row` (line 277), `_friendly_status_label` (line 974), `_friendly_detail_copy` (line 1011). The new module:

```python
"""Console workspace plumbing details tray (Storage, Sync, Handoff)."""

from __future__ import annotations

import re
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceStatusPair,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState
```

This is a mechanical **move** with exact sources — no new logic. Into the class `ConsoleWorkspaceDetailsTray(Vertical)`:

1. `__init__(self, state: ConsoleWorkspaceContextState, **kwargs: Any)` — calls `super().__init__(**kwargs)`, then sets `self.state = state`, `self.styles.height = "auto"`, `self.styles.min_height = 0`.
2. `sync_state(self, state)` — sets `self.state = state` then `self.refresh(recompose=True)`.
3. `compose()` — paste the body of `ConsoleWorkspaceContextTray._compose_status_rows` (console_workspace_context.py lines 869–936) verbatim.
4. Paste these members from `ConsoleWorkspaceContextTray` unchanged: `_status_pair` (lines 298–326), `_split_status_row` (lines 277–296), `_friendly_status_label` (lines 974–1009), `_friendly_detail_copy` (lines 1011–1023), and the `_static` staticmethod (lines 262–274).
5. Move the module-level `_AUTHORITY_LABELS` dict (console_workspace_context.py lines 38–48) into the new module.

Then in `console_workspace_context.py`: delete the moved members, delete `yield from self._compose_status_rows()` from `compose()` (line 497), and remove the `re` import and `_AUTHORITY_LABELS` dict if nothing else in the file references them (keep `_static` there — the context tray still uses it).

Add the heading flag to `ConsoleWorkspaceContextTray.__init__`:

```python
    def __init__(
        self,
        state: ConsoleWorkspaceContextState,
        *,
        show_heading: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.show_heading = show_heading
```

and guard the first yield in `compose()`:

```python
        if self.show_heading:
            yield self._static(
                self.state.heading,
                id="console-workspace-context-title",
                classes="destination-section",
            )
```

- [ ] **Step 4: Run tests to verify they pass, then check for fallout**

Run the new tests, then the existing suites that exercise the tray:

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_rail_sections.py --tb=short
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_internals_decomposition.py --tb=short
```

Expected: new tests PASS. Any pre-existing test that asserts status rows render inside `#console-workspace-context` will fail here — that is expected fallout resolved in Task 9 when the details tray is mounted in the rail; if a failing test only checks widget existence on screen (not parentage), defer it to Task 9 and note it in the commit message. Do not delete assertions.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_details.py tldw_chatbook/Widgets/Console/console_workspace_context.py Tests/UI/test_console_rail_sections.py
git commit -m "refactor(console): extract workspace details tray from context tray"
```

---

### Task 8: Model section line builder

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py` (below `ConsoleSettingsSummaryState`, line 194)
- Test: `Tests/Chat/test_console_session_settings.py`

**Interfaces:**
- Consumes: `ConsoleSettingsSummaryState` (fields `provider_row`, `model_row`, `context_row`, `sampling_row`, `transport_row` — preformatted `"Label: value"` strings).
- Produces: `build_console_model_section_lines(summary: ConsoleSettingsSummaryState) -> tuple[str, str]` — line 1 `"{provider} / {model}"`, line 2 compact `"T 0.60 · 0 / 8,192 tokens · Streaming: off"`. Task 9 renders these in the Model section and re-renders them on settings sync.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_session_settings.py`:

```python
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSettingsSummaryState,
    build_console_model_section_lines,
)


def test_model_section_lines_compact_summary():
    summary = ConsoleSettingsSummaryState(
        model_row="Model: gpt-4o (Missing key)",
        context_row="Context: 0 / 8,192 tokens; 4,096 response tokens",
        sampling_row="Sampling: T 0.60, P 0.95, min_p 0.05",
        identity_row="Persona: General",
        provider_row="Provider: openai",
        transport_row="Streaming: off",
    )
    line1, line2 = build_console_model_section_lines(summary)
    assert line1 == "openai / gpt-4o (Missing key)"
    assert line2 == "T 0.60 · 0 / 8,192 tokens · Streaming: off"


def test_model_section_lines_tolerate_missing_rows():
    summary = ConsoleSettingsSummaryState(
        model_row="",
        context_row="",
        sampling_row="",
        identity_row="",
    )
    line1, line2 = build_console_model_section_lines(summary)
    assert line1 == "not selected / no model"
    assert line2 == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_session_settings.py -k model_section_lines --tb=short`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

In `console_session_settings.py`, add below `ConsoleSettingsSummaryState`:

```python
def _summary_row_value(row: str) -> str:
    text = str(row or "").strip()
    _label, separator, value = text.partition(":")
    return value.strip() if separator else text


def build_console_model_section_lines(
    summary: ConsoleSettingsSummaryState,
) -> tuple[str, str]:
    """Build the two compact Model rail-section lines from summary rows.

    Args:
        summary: Preformatted Console settings summary rows.

    Returns:
        Tuple of ``(provider/model line, sampling·context·streaming line)``.
    """
    provider = _summary_row_value(summary.provider_row) or "not selected"
    model = _summary_row_value(summary.model_row) or "no model"
    sampling = _summary_row_value(summary.sampling_row).partition(",")[0].strip()
    context = _summary_row_value(summary.context_row).partition(";")[0].strip()
    transport = str(summary.transport_row or "").strip()
    detail_parts = [part for part in (sampling, context, transport) if part]
    return f"{provider} / {model}", " · ".join(detail_parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_session_settings.py Tests/Chat/test_console_session_settings.py
git commit -m "feat(console): add compact model section line builder"
```

---

### Task 9: Restructure the left rail into four sections

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — compose block (lines 4579–4612), `_set_console_rail_preference` (line 3442), `on_button_pressed` dispatch (near line 7402), `_sync_console_workspace_context` (line 3482), the settings-summary sync method containing line 1160.
- Test: `Tests/UI/test_console_persistent_rails.py`

**Interfaces:**
- Consumes: Tasks 5–8 (`rail_state.session_open`…, `ConsoleRailSectionHeader`, `CONSOLE_RAIL_SECTION_TOGGLE_PREFIX`, `ConsoleWorkspaceDetailsTray`, `build_console_model_section_lines`), `CONSOLE_RAIL_SECTION_IDS`.
- Produces: rail body ids `#console-rail-section-header-{session,context,model,details}` and `#console-rail-section-body-{...}`; `#console-workspace-details`; `#console-model-section-line1`, `#console-model-section-line2`, `#console-model-section-configure`. Toggling persists via `console.rail_state`.

- [ ] **Step 1: Write the failing pilot tests**

Append to `Tests/UI/test_console_persistent_rails.py` (reusing its `_build_test_app`, `ConsoleHarness`, `_wait_for_selector`, `_is_displayed` helpers):

```python
@pytest.mark.asyncio
async def test_console_left_rail_renders_four_sections_with_details_collapsed():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-details")

        for section_id in ("session", "context", "model", "details"):
            assert _is_displayed(
                console.query_one(f"#console-rail-section-header-{section_id}")
            )
        assert _is_displayed(console.query_one("#console-rail-section-body-session"))
        assert _is_displayed(console.query_one("#console-rail-section-body-context"))
        assert _is_displayed(console.query_one("#console-rail-section-body-model"))
        assert not _is_displayed(console.query_one("#console-rail-section-body-details"))
        # Session content: workspace context tray without duplicate heading.
        assert _is_displayed(console.query_one("#console-workspace-context"))
        _assert_selector_hidden_or_absent(console, "#console-workspace-context-title")
        # Details content exists but is hidden.
        assert list(console.query("#console-workspace-details"))
        # Model section content.
        assert _is_displayed(console.query_one("#console-model-section-line1"))
        assert _is_displayed(console.query_one("#console-model-section-configure"))


@pytest.mark.asyncio
async def test_console_details_toggle_expands_and_persists():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-details")

        await pilot.click("#console-rail-section-toggle-details")
        await pilot.pause(0.1)
        assert _is_displayed(console.query_one("#console-rail-section-body-details"))
        assert _is_displayed(console.query_one("#console-workspace-authority-label"))

    rail_state_config = app.app_config.get("console", {}).get("rail_state", {})
    assert any(
        isinstance(value, dict) and value.get("details_open") is True
        for value in rail_state_config.values()
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py -k "four_sections or details_toggle" --tb=short`

Expected: FAIL — `#console-rail-section-header-details` never appears.

- [ ] **Step 3: Rewrite the left-rail body compose**

Add imports in `chat_screen.py`: `ConsoleRailSectionHeader` and `CONSOLE_RAIL_SECTION_TOGGLE_PREFIX` from `tldw_chatbook.Widgets.Console.console_rail_section`; `ConsoleWorkspaceDetailsTray` from `tldw_chatbook.Widgets.Console.console_workspace_details`; `build_console_model_section_lines` from `tldw_chatbook.Chat.console_session_settings`; `CONSOLE_RAIL_SECTION_IDS` from `tldw_chatbook.Chat.console_rail_state`; `replace as dataclass_replace` from `dataclasses`. Confirm `Mapping` is importable in this module (add `from collections.abc import Mapping` if the file does not already import it); `NoMatches`/`QueryError` are already used in the sync methods.

Replace the body of the `with VerticalScroll(id="console-left-rail-body", ...):` block (lines 4579–4612) with:

```python
                    with VerticalScroll(
                        id="console-left-rail-body",
                        classes="console-left-rail-body",
                    ):
                        # Section 1: Session (workspace + conversations).
                        yield ConsoleRailSectionHeader(
                            "Session",
                            section_id="session",
                            open=rail_state.session_open,
                            id="console-rail-section-header-session",
                        )
                        session_body = Vertical(
                            id="console-rail-section-body-session",
                            classes="console-rail-section-body",
                        )
                        session_body.styles.height = "auto"
                        if not rail_state.session_open:
                            session_body.styles.display = "none"
                        with session_body:
                            workspace_context_tray = ConsoleWorkspaceContextTray(
                                workspace_context_state,
                                show_heading=False,
                                id="console-workspace-context",
                                classes="console-left-rail-section",
                            )
                            workspace_context_tray.styles.width = "100%"
                            workspace_context_tray.styles.min_width = 0
                            yield self._frame_console_region(
                                workspace_context_tray,
                                variant=self._workspace_context_frame_variant(
                                    workspace_context_state
                                ),
                            )

                        # Section 2: Context (staged sources).
                        yield ConsoleRailSectionHeader(
                            "Context",
                            section_id="context",
                            open=rail_state.context_open,
                            id="console-rail-section-header-context",
                        )
                        context_body = Vertical(
                            id="console-rail-section-body-context",
                            classes="console-rail-section-body",
                        )
                        context_body.styles.height = "auto"
                        if not rail_state.context_open:
                            context_body.styles.display = "none"
                        with context_body:
                            staged_context_tray = ConsoleStagedContextTray(
                                staged_context_state,
                                id="console-staged-context-tray",
                                classes="console-left-rail-section",
                            )
                            staged_context_tray.styles.width = "100%"
                            staged_context_tray.styles.min_width = 0
                            staged_context_tray.styles.height = "auto"
                            staged_context_tray.styles.min_height = (
                                3 if staged_context_state.is_empty else 4
                            )
                            staged_context_tray.styles.max_height = (
                                6 if staged_context_state.is_empty else 10
                            )
                            yield self._frame_console_region(
                                staged_context_tray,
                                variant=self._staged_context_frame_variant(
                                    staged_context_state
                                ),
                            )

                        # Section 3: Model (compact settings summary).
                        yield ConsoleRailSectionHeader(
                            "Model",
                            section_id="model",
                            open=rail_state.model_open,
                            id="console-rail-section-header-model",
                        )
                        model_body = Vertical(
                            id="console-rail-section-body-model",
                            classes="console-rail-section-body",
                        )
                        model_body.styles.height = "auto"
                        if not rail_state.model_open:
                            model_body.styles.display = "none"
                        with model_body:
                            model_line1, model_line2 = build_console_model_section_lines(
                                self._build_console_settings_summary_state()
                            )
                            line1 = Static(
                                model_line1,
                                id="console-model-section-line1",
                                classes="console-model-section-line",
                                markup=False,
                            )
                            yield line1
                            line2 = Static(
                                model_line2,
                                id="console-model-section-line2",
                                classes="console-model-section-line",
                                markup=False,
                            )
                            yield line2
                            configure = Button(
                                "Configure",
                                id="console-model-section-configure",
                                classes="console-workspace-action",
                                compact=True,
                            )
                            configure.tooltip = "Configure Console session settings"
                            yield configure

                        # Section 4: Details (storage, sync, handoff plumbing).
                        yield ConsoleRailSectionHeader(
                            "Details",
                            section_id="details",
                            open=rail_state.details_open,
                            id="console-rail-section-header-details",
                        )
                        details_body = Vertical(
                            id="console-rail-section-body-details",
                            classes="console-rail-section-body",
                        )
                        details_body.styles.height = "auto"
                        if not rail_state.details_open:
                            details_body.styles.display = "none"
                        with details_body:
                            details_tray = ConsoleWorkspaceDetailsTray(
                                workspace_context_state,
                                id="console-workspace-details",
                                classes="console-left-rail-section",
                            )
                            details_tray.styles.width = "100%"
                            details_tray.styles.min_width = 0
                            yield details_tray
```

Note the section order change: Session now precedes Context (spec §1). If `ConsoleWorkspaceContextTray._fit_height_to_content` fights the new nesting (it stretches to the scroll parent's height), change its `parent_region` clamp to skip stretching when the tray is inside a `.console-rail-section-body` container — check visually in Step 6 and via the pilot tests.

- [ ] **Step 4: Wire the toggle handler and persistence**

Extend `_set_console_rail_preference` (line 3442) with a `section_updates` keyword and `dataclasses.replace`:

```python
    def _set_console_rail_preference(
        self,
        *,
        left_open: bool | None = None,
        right_open: bool | None = None,
        section_updates: Mapping[str, bool] | None = None,
        notify_on_failure: bool = True,
    ) -> ConsoleRailState:
```

and replace the `next_preferences = ConsoleRailPreferences(...)` block (lines 3464–3467) with:

```python
        changes: dict[str, bool] = {}
        if left_open is not None:
            changes["left_open"] = bool(left_open)
        if right_open is not None:
            changes["right_open"] = bool(right_open)
        for section_id, section_open in (section_updates or {}).items():
            if section_id in CONSOLE_RAIL_SECTION_IDS:
                changes[f"{section_id}_open"] = bool(section_open)
        next_preferences = dataclass_replace(current, **changes)
```

Add the dispatch branch in `on_button_pressed` (next to the `console-conversation-browser-section-toggle-` branch, line 7409):

```python
        if button_id and button_id.startswith(CONSOLE_RAIL_SECTION_TOGGLE_PREFIX):
            event.stop()
            self._toggle_console_rail_section(
                button_id.removeprefix(CONSOLE_RAIL_SECTION_TOGGLE_PREFIX)
            )
            return
```

Add the toggle method near `_set_console_rail_preference`:

```python
    def _toggle_console_rail_section(self, section_id: str) -> None:
        """Flip one left-rail section open state, then sync body and header."""
        if section_id not in CONSOLE_RAIL_SECTION_IDS:
            return
        rail_state = self._current_console_rail_state()
        next_open = not getattr(rail_state, f"{section_id}_open")
        self._set_console_rail_preference(
            section_updates={section_id: next_open},
            notify_on_failure=False,
        )
        try:
            body = self.query_one(f"#console-rail-section-body-{section_id}")
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                ConsoleRailSectionHeader,
            )
        except NoMatches:
            return
        body.styles.display = "block" if next_open else "none"
        header.sync_open(next_open)
```

(`_current_console_rail_state` exists — it is the method ending at line 3440; `NoMatches` is already imported in this file — verify, otherwise import from `textual.css.query`.)

- [ ] **Step 5: Wire state syncs**

In `_sync_console_workspace_context` (line 3482), after `workspace_context.sync_state(state)` (line 3492) and before the `call_after_refresh` block, add:

```python
            try:
                details_tray = self.query_one(
                    "#console-workspace-details", ConsoleWorkspaceDetailsTray
                )
            except (NoMatches, QueryError):
                pass
            else:
                details_tray.sync_state(state)
```

Rewrite `_sync_console_settings_summary` (line 1157) to reuse one summary state for both surfaces:

```python
    def _sync_console_settings_summary(self) -> None:
        """Refresh the mounted Console settings summary surfaces if present."""
        summary_state = self._build_console_settings_summary_state()
        try:
            summary = self.query_one("#console-settings-summary", ConsoleSettingsSummary)
        except (NoMatches, QueryError):
            pass
        else:
            summary.sync_state(summary_state)
        model_line1, model_line2 = build_console_model_section_lines(summary_state)
        try:
            self.query_one("#console-model-section-line1", Static).update(model_line1)
            self.query_one("#console-model-section-line2", Static).update(model_line2)
        except (NoMatches, QueryError):
            pass
```

Route the Configure button: in `on_button_pressed`, next to the `console-settings-open` branch (line 7402), add:

```python
        if button_id == "console-model-section-configure":
            await self.on_console_settings_open(event)
            return
```

- [ ] **Step 6: Run the new tests, then the touched suites**

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py -k "four_sections or details_toggle" --tb=short
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_rail_sections.py --tb=short
```

Expected: new tests PASS. Update any pre-existing tests that asserted the old rail order (staged tray first) or status rows directly under `#console-workspace-context` — point them at the new section ids; carry over any Task 7 deferred failures here and make them green. Preserve assertion intent; do not delete coverage.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): restructure left rail into Session/Context/Model/Details sections"
```

---

### Task 10: Section CSS and generated stylesheet

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` (via `./build_css.sh`)
- Test: `Tests/UI/test_console_persistent_rails.py`

**Interfaces:**
- Consumes: class names from Tasks 6/9: `.console-rail-section-header`, `.console-rail-section-title`, `.console-rail-section-toggle`, `.console-rail-section-body`, `.console-model-section-line`.
- Produces: styled selectors present in both the component file and the generated stylesheet.

- [ ] **Step 1: Write the failing stylesheet test**

Append to `Tests/UI/test_console_persistent_rails.py`, following `test_generated_console_stylesheet_includes_rail_rules` (line 216) which reads the CSS files from disk:

```python
def test_generated_console_stylesheet_includes_rail_section_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        ".console-rail-section-header",
        ".console-rail-section-title",
        ".console-rail-section-toggle",
        ".console-rail-section-body",
        ".console-model-section-line",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/UI/test_console_persistent_rails.py -k rail_section_rules --tb=short`

Expected: FAIL — selectors absent.

- [ ] **Step 3: Add the CSS and rebuild**

In `_agentic_terminal.tcss`, next to the existing `.console-rail-header` rules (around line 994), add:

```css
.console-rail-section-header {
    height: 1;
    margin-top: 1;
}

.console-rail-section-title {
    width: 1fr;
    text-style: bold;
}

.console-rail-section-toggle {
    width: 3;
    min-width: 3;
}

.console-rail-section-body {
    height: auto;
    min-height: 0;
}

.console-model-section-line {
    height: 1;
    color: $text-muted;
}
```

Match the file's existing token usage — if sibling rules use a different muted color token than `$text-muted` (check `.console-workspace-empty-copy`), use that token instead. Then run `./build_css.sh`.

- [ ] **Step 4: Run test to verify it passes**

Same command as Step 2. Expected: PASS. Also rerun the Task 9 pilot tests to confirm layout is unaffected.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_persistent_rails.py
git commit -m "style(console): add rail section styles"
```

---

### Task 11: Verification, screenshot QA, and approval gate

**Files:**
- Create: screenshots under `Docs/superpowers/qa/console-rail-ia-2026-07/`
- Modify: `Docs/superpowers/specs/2026-07-02-console-dual-audience-ux-design.md` (no content change; only if QA reveals a needed deviation, document it)

- [ ] **Step 1: Run the affected test set**

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q \
  Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_rail_state.py \
  Tests/Chat/test_console_session_settings.py \
  Tests/Workspaces/test_console_conversation_browser_state.py \
  Tests/UI/test_console_rail_sections.py Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_internals_decomposition.py --tb=short
```

Expected: PASS (modulo the documented pre-existing baseline failures in files not touched here).

- [ ] **Step 2: Live screenshot QA**

Use the textual-serve live-capture workflow (real app CSS, not harness CSS — see `Docs/superpowers/qa/` precedents). Capture at minimum:
1. Fresh Console — four sections visible, Details collapsed, Model shows two compact lines.
2. Details expanded via toggle.
3. A conversation list after two sends in differently-named sessions — auto-titles and age labels visible.
4. Relaunch after (2) — Details stays expanded (persistence).

Save to `Docs/superpowers/qa/console-rail-ia-2026-07/` with descriptive names.

- [ ] **Step 3: User approval gate**

Present the screenshots to the user for explicit approval before any merge (standing project rule: every Console screen change needs explicit user approval). Do not merge without it.

- [ ] **Step 4: Commit QA artifacts**

```bash
git add Docs/superpowers/qa/console-rail-ia-2026-07/
git commit -m "docs(console): rail IA phase 1 QA evidence"
```
