# Console Keyboard Layer (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Console power users full keyboard reach: a Ctrl+K fuzzy session switcher, a Ctrl+M model popover, Escape-to-composer, direct message-action keys, tab-strip bindings, pane-contextual footer hints, and a Console command-palette provider.

**Architecture:** Two new `ModalScreen` widgets (switcher, popover) follow the existing Console modal idiom (`ModalScreen[T]` + `push_screen(modal, callback)`), applying results exclusively through existing chat_screen seams (`switch_session` path, `_resume_console_workspace_conversation`, `_replace_active_console_session_settings`, rename modal). A new pure module builds switcher results from the existing browser input rows. `chat_screen.py` gains only bindings, open/apply orchestration, contextual footer registration, and the palette provider is a posting-style `textual.command.Provider` scoped to the Console screen.

**Tech Stack:** Python 3.11+, Textual (CommandPalette/Provider, ModalScreen), pytest + pytest-asyncio pilot, existing `_build_test_app`/`ConsoleHarness` harness.

**Spec:** `Docs/superpowers/specs/2026-07-02-console-dual-audience-ux-design.md` §3 (Keyboard layer) + §6 (posting patterns). Phases 1–2 merged (PRs #576/#577).

## Already exists — do NOT rebuild

- Transcript keyboard selection is DONE: `ConsoleTranscript` binds `down,j`/`up,k`/`enter`/`escape` with `selected_message_id`, `select_message()`, `focus_action()` (console_transcript.py:320-470). Only the direct `c`/`e`/`r` shortcuts are missing (Task 5).
- Footer plumbing is DONE: `AppFooterStatus.set_workbench_shortcuts(*, source: str, shortcuts: tuple[tuple[str, str], ...])` and `chat_screen._register_console_footer_shortcuts()` (chat_screen.py:746) already registers a static `CONSOLE_WORKBENCH_SHORTCUTS`. Task 6 only makes the tuple contextual.
- F6/Shift+F6 pane cycling, Tab/Shift+Tab trapping under the setup modal, and the `_console_setup_modal_blocking()` guard all exist.

## Global Constraints

- Run tests: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q <target> --tb=short`. The `timeout` shell command is unavailable.
- Keys: `ctrl+k` switcher, `alt+m` popover (SUBSTITUTED 2026-07-05: `ctrl+m` is byte-identical to Enter on non-Kitty terminals per Textual KEY_ALIASES — an accidental-send hazard; Alt+M keeps the mnemonic and encodes distinctly), `ctrl+t` new tab, `alt+1`..`alt+9` tab jump, `escape` to composer (screen-level, non-priority), `c`/`e`/`r` on the focused transcript. The ACTIONS are fixed; if a key proves dead in a terminal, substitute and document — never silently drop the action.
- **Documented deviation from spec §3:** no Ctrl+Enter "open in new tab" in the switcher — resuming a persisted conversation already opens as a new tab (existing `restore_persisted_session` semantics), and native rows ARE tabs; Enter activates, full stop. F2-rename in the switcher is kept via a rename result chained to the existing rename modal.
- **Every new screen-level binding/action must no-op while `_console_setup_modal_blocking()`** (except palette/tab-bar behavior, which stays app-level).
- New modals use inline `DEFAULT_CSS` like the existing Console modals (no generated-stylesheet work in this phase).
- Copy: switcher title `Switch Session`, input placeholder `Search conversations…`; popover title `Model`, full-settings button label `Full settings…`.
- Existing seams are the ONLY apply paths: native activation mirrors the session-tab click (set workspace → `controller.switch_session` → `_sync_native_console_chat_ui` → focus composer); persisted rows go through `_resume_console_workspace_conversation(conversation_id, target_scope_type=…, target_workspace_id=…)`; settings apply via `_replace_active_console_session_settings(settings)`; rename via `_open_console_session_rename_modal(session_id)`.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`, `git add .`, or `git commit -a`.** Never touch `.claude/settings.local.json`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Console screen changes require live screenshot QA + explicit user approval before merge (Task 8).
- Branch: `claude/console-keyboard-phase3` from current `dev`.

## File Structure

- Create `tldw_chatbook/Chat/console_switcher_state.py` — pure switcher results (no Textual).
- Create `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py` — Ctrl+K modal.
- Create `tldw_chatbook/Widgets/Console/console_model_popover.py` — Ctrl+M modal.
- Modify `tldw_chatbook/Widgets/Console/console_transcript.py` — c/e/r bindings.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py` — bindings, open/apply wiring, contextual footer, tab-jump helper.
- Modify `tldw_chatbook/app.py` — register `ConsoleCommandProvider`.
- Create `tldw_chatbook/UI/console_command_provider.py` — palette provider.
- Tests: new `Tests/Chat/test_console_switcher_state.py`; extend `Tests/UI/test_console_rail_sections.py` (widget tests), `Tests/UI/test_console_internals_decomposition.py` (pilot), `Tests/UI/test_console_native_chat_flow.py` (switch/resume flows).

---

### Task 1: Pure switcher results module

**Files:**
- Create: `tldw_chatbook/Chat/console_switcher_state.py`
- Test: `Tests/Chat/test_console_switcher_state.py` (new)

**Interfaces:**
- Consumes: `ConsoleConversationBrowserInputRow` and `ReverseKey` from `tldw_chatbook/Workspaces/conversation_browser_state.py`.
- Produces (Tasks 2–3 rely on):
  - `@dataclass(frozen=True) ConsoleSwitcherEntry(row_key: str, title: str, subtitle: str, native_session_id: str | None, conversation_id: str | None, scope_type: str, workspace_id: str | None, is_active: bool)`
  - `build_console_switcher_entries(rows: Iterable[ConsoleConversationBrowserInputRow], *, query: str = "", limit: int = 20) -> tuple[ConsoleSwitcherEntry, ...]` — dedup by `row_key` (first wins), case-insensitive substring match of every whitespace-separated query token against `title + workspace_label + status`, recent-first (`selected` row first, then `ReverseKey(updated_sort)`, then title casefold), capped at `limit`. `subtitle` = `" - ".join(part for part in (workspace_label, status, updated_label) if part)`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_console_switcher_state.py`:

```python
"""Pure Console session-switcher result contracts."""

from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleSwitcherEntry,
    build_console_switcher_entries,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


def _row(**overrides) -> ConsoleConversationBrowserInputRow:
    defaults = dict(
        row_key="conv-1",
        conversation_id="conv-1",
        native_session_id=None,
        title="API refactor plan",
        scope_type="workspace",
        workspace_id="ws-1",
        workspace_label="Workspace 1",
        status="workspace-thread",
        updated_label="2m",
        updated_sort="2026-07-04T11:58:00+00:00",
    )
    defaults.update(overrides)
    return ConsoleConversationBrowserInputRow(**defaults)


def test_entries_are_recent_first_with_active_pinned():
    rows = [
        _row(row_key="old", conversation_id="old", title="Old chat",
             updated_sort="2026-06-01T00:00:00+00:00"),
        _row(row_key="new", conversation_id="new", title="New chat",
             updated_sort="2026-07-04T00:00:00+00:00"),
        _row(row_key="active", conversation_id="active", title="Active chat",
             selected=True, updated_sort="2026-05-01T00:00:00+00:00"),
    ]
    titles = [entry.title for entry in build_console_switcher_entries(rows)]
    assert titles == ["Active chat", "New chat", "Old chat"]
    assert build_console_switcher_entries(rows)[0].is_active is True


def test_query_tokens_all_must_match_case_insensitive():
    rows = [
        _row(row_key="a", conversation_id="a", title="Groq testing"),
        _row(row_key="b", conversation_id="b", title="API refactor plan"),
    ]
    hits = build_console_switcher_entries(rows, query="groq test")
    assert [e.title for e in hits] == ["Groq testing"]
    assert build_console_switcher_entries(rows, query="REFACTOR")[0].title == "API refactor plan"
    # Token can match workspace label or status, not just title.
    assert [e.title for e in build_console_switcher_entries(rows, query="workspace 1 api")] == [
        "API refactor plan"
    ]


def test_entries_dedupe_by_row_key_and_cap_at_limit():
    rows = [_row(row_key="dup", conversation_id="dup", title="First wins"),
            _row(row_key="dup", conversation_id="dup", title="Second loses")]
    hits = build_console_switcher_entries(rows)
    assert len(hits) == 1 and hits[0].title == "First wins"
    many = [
        _row(row_key=f"k{i}", conversation_id=f"k{i}", title=f"Chat {i}",
             updated_sort=f"2026-07-04T{i:02d}:00:00+00:00")
        for i in range(30)
    ]
    assert len(build_console_switcher_entries(many, limit=20)) == 20


def test_subtitle_joins_available_parts():
    entry = build_console_switcher_entries([_row()])[0]
    assert entry.subtitle == "Workspace 1 - workspace-thread - 2m"
    bare = build_console_switcher_entries(
        [_row(row_key="x", workspace_label="", status="", updated_label="")]
    )[0]
    assert bare.subtitle == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Chat/test_console_switcher_state.py --tb=short`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Chat/console_switcher_state.py`:

```python
"""Pure result contracts for the Console session switcher (Ctrl+K)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    ReverseKey,
)

CONSOLE_SWITCHER_RESULT_LIMIT = 20


@dataclass(frozen=True)
class ConsoleSwitcherEntry:
    """One selectable result in the Console session switcher."""

    row_key: str
    title: str
    subtitle: str
    native_session_id: str | None
    conversation_id: str | None
    scope_type: str
    workspace_id: str | None
    is_active: bool


def _matches(row: ConsoleConversationBrowserInputRow, tokens: list[str]) -> bool:
    haystack = " ".join((row.title, row.workspace_label, row.status)).lower()
    return all(token in haystack for token in tokens)


def build_console_switcher_entries(
    rows: Iterable[ConsoleConversationBrowserInputRow],
    *,
    query: str = "",
    limit: int = CONSOLE_SWITCHER_RESULT_LIMIT,
) -> tuple[ConsoleSwitcherEntry, ...]:
    """Build deduped, recent-first switcher results for a query.

    Args:
        rows: Browser input rows from the chat screen row builders.
        query: Whitespace-separated tokens; every token must match the row's
            title, workspace label, or status (case-insensitive substring).
        limit: Maximum number of entries returned.

    Returns:
        Up to ``limit`` entries, active row first, then most recent.
    """
    tokens = [token for token in query.lower().split() if token]
    seen: set[str] = set()
    deduped: list[ConsoleConversationBrowserInputRow] = []
    for row in rows:
        key = str(row.row_key or "")
        if not key or key in seen:
            continue
        seen.add(key)
        if tokens and not _matches(row, tokens):
            continue
        deduped.append(row)

    deduped.sort(
        key=lambda row: (
            not row.selected,
            ReverseKey(str(row.updated_sort or "")),
            row.title.casefold(),
            row.row_key,
        )
    )
    entries = []
    for row in deduped[: max(1, int(limit))]:
        subtitle = " - ".join(
            part
            for part in (row.workspace_label, row.status, row.updated_label)
            if str(part or "").strip()
        )
        entries.append(
            ConsoleSwitcherEntry(
                row_key=str(row.row_key),
                title=str(row.title or "Untitled conversation"),
                subtitle=subtitle,
                native_session_id=row.native_session_id,
                conversation_id=row.conversation_id,
                scope_type=str(row.scope_type or ""),
                workspace_id=row.workspace_id,
                is_active=bool(row.selected),
            )
        )
    return tuple(entries)
```

- [ ] **Step 4: Run tests to verify they pass**

Same command. Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_switcher_state.py Tests/Chat/test_console_switcher_state.py
git commit -m "feat(console): pure session-switcher result builder"
```

---

### Task 2: ConsoleSessionSwitcherModal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py`
- Test: `Tests/UI/test_console_rail_sections.py` (append)

**Interfaces:**
- Consumes: Task 1 (`ConsoleSwitcherEntry`, `build_console_switcher_entries`).
- Produces (Task 3 relies on):
  - `@dataclass(frozen=True) ConsoleSwitcherChoice(kind: str, entry: ConsoleSwitcherEntry)` with `kind` `"activate" | "rename"` (module-level, same file).
  - `ConsoleSessionSwitcherModal(ModalScreen[ConsoleSwitcherChoice | None])` with `__init__(self, *, rows: tuple[ConsoleConversationBrowserInputRow, ...]) -> None` — the modal re-filters locally as the query changes (no async round-trips; rows are gathered once by the opener).
  - Behavior: an `Input` (`#console-switcher-query`, placeholder `Search conversations…`, focused on mount) above a `Vertical` list (`#console-switcher-results`) of result `Button`s (`id=f"console-switcher-result-{index}"`, classes `console-switcher-result`, active entry gets class `console-switcher-result-active`; label = title on line 1 + subtitle on line 2 like the rail rows). `Input.Changed` rebuilds the list via `build_console_switcher_entries(rows, query=…)`. Enter in the input activates the FIRST result; clicking/pressing a result button dismisses `ConsoleSwitcherChoice("activate", entry)`. `f2` (modal BINDINGS) dismisses `ConsoleSwitcherChoice("rename", first_or_focused_entry)` only when that entry has a `native_session_id`. `escape` dismisses `None`. Empty results show `Static` `#console-switcher-empty` `"No matches."`.

- [ ] **Step 1: Write the failing widget tests**

Append to `Tests/UI/test_console_rail_sections.py` (reuse its `App`/`pytest` imports):

```python
from tldw_chatbook.Chat.console_switcher_state import ConsoleSwitcherEntry
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


def _switcher_rows() -> tuple[ConsoleConversationBrowserInputRow, ...]:
    def row(key, title, native=None, **kw):
        return ConsoleConversationBrowserInputRow(
            row_key=key, conversation_id=None if native else key,
            native_session_id=native, title=title, scope_type="workspace",
            workspace_id="ws-1", workspace_label="Workspace 1",
            updated_sort="2026-07-04T10:00:00+00:00", **kw,
        )
    return (
        row("native-1", "Groq testing", native="sess-1", selected=True),
        row("conv-2", "API refactor plan"),
        row("conv-3", "Tides explainer"),
    )


class _SwitcherApp(App):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    async def on_mount(self) -> None:
        def _capture(choice):
            self.result = choice
        await self.push_screen(
            ConsoleSessionSwitcherModal(rows=_switcher_rows()), callback=_capture
        )


@pytest.mark.asyncio
async def test_switcher_lists_recent_first_and_filters_on_typing():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        first = app.screen.query_one("#console-switcher-result-0", Button)
        assert "Groq testing" in str(first.label)
        await pilot.click("#console-switcher-query")
        await pilot.press(*"refactor")
        await pilot.pause()
        first = app.screen.query_one("#console-switcher-result-0", Button)
        assert "API refactor plan" in str(first.label)
        assert not list(app.screen.query("#console-switcher-result-1"))


@pytest.mark.asyncio
async def test_switcher_enter_activates_first_result():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-query")
        await pilot.press(*"tides")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSwitcherChoice)
        assert app.result.kind == "activate"
        assert app.result.entry.title == "Tides explainer"


@pytest.mark.asyncio
async def test_switcher_f2_requests_rename_for_native_entry():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSwitcherChoice)
        assert app.result.kind == "rename"
        assert app.result.entry.native_session_id == "sess-1"


@pytest.mark.asyncio
async def test_switcher_escape_dismisses_none_and_empty_query_shows_no_matches():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-query")
        await pilot.press(*"zzzz")
        await pilot.pause()
        assert list(app.screen.query("#console-switcher-empty"))
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_rail_sections.py -k switcher --tb=short`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the modal**

Create `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py` following the `ConsoleWorkspaceSwitcherModal`/`ConsoleRenameSessionModal` idiom (ModalScreen, inline DEFAULT_CSS, `@on` handlers):

```python
"""Console fuzzy session switcher modal (Ctrl+K)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleSwitcherEntry,
    build_console_switcher_entries,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


@dataclass(frozen=True)
class ConsoleSwitcherChoice:
    """Result returned by the session switcher modal."""

    kind: str
    entry: ConsoleSwitcherEntry


class ConsoleSessionSwitcherModal(ModalScreen["ConsoleSwitcherChoice | None"]):
    """Fuzzy-find and activate a Console session or persisted conversation."""

    DEFAULT_CSS = """
    ConsoleSessionSwitcherModal {
        align: center middle;
    }

    #console-switcher-modal {
        width: 72;
        height: auto;
        max-height: 30;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-switcher-results {
        height: auto;
        max-height: 20;
        margin: 1 0 0 0;
    }

    .console-switcher-result {
        width: 100%;
        height: 2;
        min-height: 2;
        margin: 0;
    }
    """

    BINDINGS = [
        ("escape", "dismiss_switcher", "Cancel"),
        ("f2", "rename_entry", "Rename"),
    ]

    def __init__(
        self,
        *,
        rows: tuple[ConsoleConversationBrowserInputRow, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rows = rows
        self._entries: tuple[ConsoleSwitcherEntry, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="console-switcher-modal"):
            yield Static("Switch Session", classes="console-modal-header")
            yield Input(
                placeholder="Search conversations…",
                id="console-switcher-query",
            )
            yield Vertical(id="console-switcher-results")

    def on_mount(self) -> None:
        self.query_one("#console-switcher-query", Input).focus()
        self._refresh_results("")

    def _refresh_results(self, query: str) -> None:
        self._entries = build_console_switcher_entries(self._rows, query=query)
        results = self.query_one("#console-switcher-results", Vertical)
        results.remove_children()
        if not self._entries:
            results.mount(
                Static("No matches.", id="console-switcher-empty", markup=False)
            )
            return
        for index, entry in enumerate(self._entries):
            label = entry.title if not entry.subtitle else f"{entry.title}\n  {entry.subtitle}"
            button = Button(
                label,
                id=f"console-switcher-result-{index}",
                classes="console-switcher-result",
                compact=True,
            )
            button.set_class(entry.is_active, "console-switcher-result-active")
            button.tooltip = f"Switch to {entry.title}"
            results.mount(button)

    @on(Input.Changed, "#console-switcher-query")
    def _query_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._refresh_results(event.value)

    @on(Input.Submitted, "#console-switcher-query")
    def _query_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._entries:
            self.dismiss(ConsoleSwitcherChoice("activate", self._entries[0]))

    @on(Button.Pressed, ".console-switcher-result")
    def _result_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if 0 <= index < len(self._entries):
            self.dismiss(ConsoleSwitcherChoice("activate", self._entries[index]))

    def action_dismiss_switcher(self) -> None:
        self.dismiss(None)

    def action_rename_entry(self) -> None:
        for entry in self._entries:
            if entry.native_session_id:
                self.dismiss(ConsoleSwitcherChoice("rename", entry))
                return
```

- [ ] **Step 4: Run tests to verify they pass**

Same command. Expected: PASS (4 tests). If `results.mount(...)` inside a sync handler races the refresh in tests, await `pilot.pause()` in tests is already present; if mounting requires it, switch `_refresh_results` mounting to `results.mount_all([...])` — keep behavior identical.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_session_switcher_modal.py Tests/UI/test_console_rail_sections.py
git commit -m "feat(console): fuzzy session switcher modal"
```

---

### Task 3: Ctrl+K wiring in ChatScreen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (BINDINGS block ~line 335; new methods near `_open_console_session_rename_modal` ~line 766)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2; existing `_native_console_browser_rows()` (chat_screen.py:2179), `_membership_console_browser_rows()` (:2218), `_sync_persisted_console_browser_rows(query="")` (:2410); activation seams: `_set_active_workspace_for_console_session(session_id)` (:1587 area), `controller.switch_session`, `_sync_native_console_chat_ui()`, `_focus_console_composer_if_needed(force=True)`, `_resume_console_workspace_conversation(conversation_id, target_scope_type=…, target_workspace_id=…)` (:1819), `_open_console_session_rename_modal(session_id)` (:766), `_console_setup_modal_blocking()`.
- Produces: `Binding("ctrl+k", "open_console_session_switcher", "Switch session", show=False)`; `def action_open_console_session_switcher(self) -> None` (guarded, gathers rows, pushes modal); `def _apply_console_switcher_choice(self, choice) -> None` callback.

- [ ] **Step 1: Write the failing pilot tests**

Append to `Tests/UI/test_console_native_chat_flow.py` (reuse `_build_test_app`, `ConsoleHarness`, `_wait_for_selector`, `_configure_native_ready_console`):

```python
@pytest.mark.asyncio
async def test_ctrl_k_opens_session_switcher_and_activates_native_session():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        # Create a second native session so there is something to switch to.
        await pilot.click("#console-new-chat-tab")
        await pilot.pause(0.2)
        store = console._console_chat_store
        first_session = store.sessions()[0]
        assert store.active_session_id != first_session.id

        await pilot.press("ctrl+k")
        await pilot.pause(0.2)
        assert app.screen.__class__.__name__ == "ConsoleSessionSwitcherModal"
        query = app.screen.query_one("#console-switcher-query")
        assert app.focused is query
        # First entry is the ACTIVE session; pick the other one by typing its title.
        await pilot.press(*first_session.title.split()[0].lower())
        await pilot.pause(0.2)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert store.active_session_id == first_session.id


@pytest.mark.asyncio
async def test_ctrl_k_is_inert_while_setup_modal_blocks():
    app = _build_test_app()  # blocked: no provider ready
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        await pilot.press("ctrl+k")
        await pilot.pause(0.2)
        assert app.screen.__class__.__name__ != "ConsoleSessionSwitcherModal"


@pytest.mark.asyncio
async def test_switcher_rename_choice_chains_to_rename_modal():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.press("ctrl+k")
        await pilot.pause(0.2)
        await pilot.press("f2")
        await pilot.pause(0.3)
        assert app.screen.__class__.__name__ == "ConsoleRenameSessionModal"
        await pilot.press("escape")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py -k "ctrl_k or switcher_rename" --tb=short`

Expected: FAIL — ctrl+k unbound.

- [ ] **Step 3: Implement**

In `chat_screen.py`:

1. Imports: `ConsoleSessionSwitcherModal`, `ConsoleSwitcherChoice` from the new widget module.
2. Add to `BINDINGS`: `Binding("ctrl+k", "open_console_session_switcher", "Switch session", show=False),`
3. Add methods (near the rename-modal opener):

```python
    def action_open_console_session_switcher(self) -> None:
        """Open the Ctrl+K fuzzy session switcher."""
        if self._console_setup_modal_blocking():
            return
        rows = [
            *self._native_console_browser_rows(),
            *self._membership_console_browser_rows(),
        ]
        persisted_rows, _total, _error = self._sync_persisted_console_browser_rows()
        rows.extend(persisted_rows)
        self.app.push_screen(
            ConsoleSessionSwitcherModal(rows=tuple(rows)),
            callback=self._apply_console_switcher_choice,
        )

    def _apply_console_switcher_choice(
        self, choice: ConsoleSwitcherChoice | None
    ) -> None:
        """Apply a switcher selection through the existing activation seams."""
        if choice is None:
            return
        entry = choice.entry
        if choice.kind == "rename" and entry.native_session_id:
            self._open_console_session_rename_modal(entry.native_session_id)
            return
        if choice.kind != "activate":
            return
        if entry.native_session_id:
            controller = self._ensure_console_chat_controller()
            self._set_active_workspace_for_console_session(entry.native_session_id)
            controller.switch_session(entry.native_session_id)

            async def _finish_native_switch() -> None:
                await self._sync_native_console_chat_ui()
                self._focus_console_composer_if_needed(force=True)

            self.run_worker(_finish_native_switch(), exclusive=False)
            return
        if entry.conversation_id:
            self.run_worker(
                self._resume_console_workspace_conversation(
                    entry.conversation_id,
                    target_scope_type=entry.scope_type or None,
                    target_workspace_id=entry.workspace_id,
                ),
                exclusive=False,
            )
```

Mirror the exact session-tab click sequence (chat_screen.py:7833-7843) — if that block wraps the switch differently (e.g., no worker), match IT rather than this sketch, and note the deviation in your report.

- [ ] **Step 4: Run tests to verify they pass, plus neighbors**

```bash
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py -k "ctrl_k or switcher_rename" --tb=short
env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_native_chat_flow.py --tb=no
```

Expected: new tests PASS; file otherwise green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): ctrl+k session switcher wiring"
```

---

### Task 4: ConsoleModelPopover + Ctrl+M wiring

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_model_popover.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (binding + open/apply)
- Test: `Tests/UI/test_console_rail_sections.py` (widget), `Tests/UI/test_console_internals_decomposition.py` (pilot)

**Interfaces:**
- Consumes: `ConsoleSessionSettings`, `build_console_provider_options(providers_models)`, `build_console_model_options(provider, providers_models, current_model)` from `tldw_chatbook/Chat/console_session_settings.py`; `get_cli_providers_and_models()` from `tldw_chatbook.config` (the settings modal's option source — verify by reading `console_settings_modal.py`'s constructor call site in chat_screen `_open_console_settings` (:548) and mirror EXACTLY how it obtains `providers_models`); apply path `_replace_active_console_session_settings(settings)` (:1169); full settings `_open_console_settings()` (:548).
- Produces:
  - Module-level sentinel `CONSOLE_POPOVER_OPEN_FULL_SETTINGS = "open-full-settings"`.
  - `ConsoleModelPopover(ModalScreen["ConsoleSessionSettings | str | None"])` with `__init__(self, *, settings: ConsoleSessionSettings, providers_models: Mapping[str, Sequence[str]], **kwargs: Any)`.
  - Layout: title `Model`; provider `Select` (`#console-popover-provider`, options from `build_console_provider_options`, value = current provider); model `Select` (`#console-popover-model`, options from `build_console_model_options`, rebuilt on provider change); temperature `Input` (`#console-popover-temperature`, current value or blank); streaming `Button` toggle (`#console-popover-streaming`, label `Streaming: on|off`, flips on press); `Apply` (`#console-popover-apply`, primary), `Full settings…` (`#console-popover-full-settings`), Escape cancels (None).
  - Apply dismisses `dataclasses.replace(self._settings, provider=…, model=…, temperature=…, streaming=…)` — ONLY those four fields change; temperature parse failure keeps the original value. `Full settings…` dismisses the sentinel string.
  - chat_screen: `Binding("alt+m", "open_console_model_popover", "Model", show=False)`; `action_open_console_model_popover` (guarded by `_console_setup_modal_blocking()` — the popover is a power-user shortcut; first-run goes through the modal's own action) pushing the popover with the ACTIVE session settings (`store.session_settings(active_id)` or the same default-settings source `_open_console_settings` uses — mirror it); callback: sentinel → `self.run_worker(self._open_console_settings(), exclusive=False)`; `ConsoleSessionSettings` → `_replace_active_console_session_settings(result)`.

- [ ] **Step 1: Write failing widget tests** (append to `Tests/UI/test_console_rail_sections.py`)

```python
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Widgets.Console.console_model_popover import (
    CONSOLE_POPOVER_OPEN_FULL_SETTINGS,
    ConsoleModelPopover,
)

_POPOVER_PROVIDERS = {"llama_cpp": ["model-a", "model-b"], "openai": ["gpt-4o"]}


class _PopoverApp(App):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    async def on_mount(self) -> None:
        settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
        def _capture(result):
            self.result = result
        await self.push_screen(
            ConsoleModelPopover(
                settings=settings, providers_models=_POPOVER_PROVIDERS
            ),
            callback=_capture,
        )


@pytest.mark.asyncio
async def test_popover_apply_returns_replaced_settings():
    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        model_select = app.screen.query_one("#console-popover-model")
        model_select.value = "model-b"
        await pilot.click("#console-popover-streaming")
        await pilot.pause()
        await pilot.click("#console-popover-apply")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSessionSettings)
        assert app.result.model == "model-b"
        assert app.result.provider == "llama_cpp"
        # ConsoleSessionSettings defaults streaming True; one toggle flips it.
        assert app.result.streaming is False


@pytest.mark.asyncio
async def test_popover_full_settings_returns_sentinel_and_escape_cancels():
    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-popover-full-settings")
        await pilot.pause()
        assert app.result == CONSOLE_POPOVER_OPEN_FULL_SETTINGS
    app2 = _PopoverApp()
    async with app2.run_test(size=(90, 30)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
        assert app2.result is None
```

Note: if `ConsoleSessionSettings(provider=…, model=…)` requires more constructor arguments, build it via `build_default_console_session_settings({}, provider="llama_cpp", model="model-a")` instead — check the dataclass first; the assertion on `streaming` must then match that object's starting value flipped once.

- [ ] **Step 2: Run to verify failure** — `-k popover`, expect `ModuleNotFoundError`.

- [ ] **Step 3: Implement the popover** following the settings modal's option-building conventions:

```python
"""Console quick model popover (Ctrl+M)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_model_options,
    build_console_provider_options,
)

CONSOLE_POPOVER_OPEN_FULL_SETTINGS = "open-full-settings"


class ConsoleModelPopover(ModalScreen["ConsoleSessionSettings | str | None"]):
    """Quick provider/model/temperature/streaming switcher for the session."""

    DEFAULT_CSS = """
    ConsoleModelPopover {
        align: center middle;
    }

    #console-model-popover {
        width: 60;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-popover-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }
    """

    BINDINGS = [("escape", "dismiss_popover", "Cancel")]

    def __init__(
        self,
        *,
        settings: ConsoleSessionSettings,
        providers_models: Mapping[str, Sequence[str]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._providers_models = providers_models
        self._streaming = bool(settings.streaming)

    def compose(self) -> ComposeResult:
        provider_options = [
            (option.label, option.value)
            for option in build_console_provider_options(self._providers_models)
        ]
        model_options = [
            (option.label, option.value)
            for option in build_console_model_options(
                self._settings.provider, self._providers_models, self._settings.model
            )
        ]
        with Vertical(id="console-model-popover"):
            yield Static("Model", classes="console-modal-header")
            yield Select(
                provider_options,
                value=self._settings.provider,
                id="console-popover-provider",
            )
            yield Select(
                model_options,
                value=self._settings.model if self._settings.model else Select.BLANK,
                id="console-popover-model",
                allow_blank=True,
            )
            yield Input(
                value="" if self._settings.temperature is None else str(self._settings.temperature),
                placeholder="Temperature",
                id="console-popover-temperature",
            )
            yield Button(
                f"Streaming: {'on' if self._streaming else 'off'}",
                id="console-popover-streaming",
                compact=True,
            )
            with Horizontal(id="console-popover-actions"):
                yield Button("Full settings…", id="console-popover-full-settings", compact=True)
                yield Button("Apply", id="console-popover-apply", variant="primary", compact=True)

    @on(Select.Changed, "#console-popover-provider")
    def _provider_changed(self, event: Select.Changed) -> None:
        event.stop()
        provider = str(event.value)
        options = [
            (option.label, option.value)
            for option in build_console_model_options(
                provider, self._providers_models, None
            )
        ]
        model_select = self.query_one("#console-popover-model", Select)
        model_select.set_options(options)

    @on(Button.Pressed, "#console-popover-streaming")
    def _toggle_streaming(self, event: Button.Pressed) -> None:
        event.stop()
        self._streaming = not self._streaming
        event.button.label = f"Streaming: {'on' if self._streaming else 'off'}"

    @on(Button.Pressed, "#console-popover-full-settings")
    def _full_settings(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(CONSOLE_POPOVER_OPEN_FULL_SETTINGS)

    @on(Button.Pressed, "#console-popover-apply")
    def _apply(self, event: Button.Pressed) -> None:
        event.stop()
        provider_value = self.query_one("#console-popover-provider", Select).value
        model_value = self.query_one("#console-popover-model", Select).value
        temperature_text = self.query_one("#console-popover-temperature", Input).value.strip()
        temperature = self._settings.temperature
        if temperature_text:
            try:
                temperature = float(temperature_text)
            except ValueError:
                pass
        self.dismiss(
            replace(
                self._settings,
                provider=str(provider_value),
                model=None if model_value in (None, Select.BLANK) else str(model_value),
                temperature=temperature,
                streaming=self._streaming,
            )
        )

    def action_dismiss_popover(self) -> None:
        self.dismiss(None)
```

Adjust field names against the real `ConsoleSessionSettings` dataclass (provider/model/temperature/streaming exist per Phase 1–2 usage; if `replace()` rejects a field, the dataclass is not frozen-compatible — read it and use its actual copy idiom). If `ConsoleSessionSettings` is not a dataclass, mirror how `console_settings_modal.py` constructs its result.

Then wire chat_screen: binding `ctrl+m` → `action_open_console_model_popover` (guarded), gather `providers_models` and current settings EXACTLY as `_open_console_settings` does (read chat_screen.py:548-566 and reuse its sources), push with callback:

```python
    def _apply_console_model_popover_result(
        self, result: "ConsoleSessionSettings | str | None"
    ) -> None:
        if result is None:
            return
        if result == CONSOLE_POPOVER_OPEN_FULL_SETTINGS:
            self.run_worker(self._open_console_settings(), exclusive=False)
            return
        self._replace_active_console_session_settings(result)
```

- [ ] **Step 4: Pilot test + run**

Append to `Tests/UI/test_console_internals_decomposition.py`:

```python
@pytest.mark.asyncio
async def test_alt_m_opens_model_popover_and_apply_updates_session_settings():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.press("alt+m")
        await pilot.pause(0.2)
        assert app.screen.__class__.__name__ == "ConsoleModelPopover"
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert app.screen.__class__.__name__ != "ConsoleModelPopover"
```

Run: the two `-k popover` widget tests + this pilot + full `Tests/UI/test_console_rail_sections.py`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_model_popover.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): ctrl+m quick model popover"
```

---

### Task 5: Direct message-action keys on the transcript

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (`ConsoleTranscript.BINDINGS`, ~line 324)
- Test: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Consumes: existing selection model (`selected_message_id`, BINDINGS at console_transcript.py:324-329) and the action-button focus/dispatch machinery (`focus_action(message_id, action_id)` :411; buttons `console-message-action-{action_id}-{message_id}`; screen dispatcher `handle_console_message_action` chat_screen.py:6066).
- Produces: transcript BINDINGS additions `("c", "invoke_selected_action('copy')", "Copy")`, `("e", "invoke_selected_action('edit')", "Edit")`, `("r", "invoke_selected_action('regenerate')", "Regenerate")` — implemented as `def action_invoke_selected_action(self, action_id: str) -> None` which, when a message is selected AND the corresponding action button exists for it, presses that button (`self.query_one(f"#console-message-action-{action_id}-{self.selected_message_id}", Button).press()`); silently no-ops when nothing is selected or the action is unavailable for that message.

- [ ] **Step 1: Failing pilot test** (append to `Tests/UI/test_console_native_chat_flow.py`; reuse the send-harness from `test_console_send_refreshes_workspace_conversation_rail_after_persistence` to produce a transcript with an assistant message):

```python
@pytest.mark.asyncio
async def test_transcript_c_key_copies_selected_message():
    # Arrange exactly as the existing send test does (copy its setup verbatim),
    # ending with an assistant reply visible in the transcript.
    ...
    transcript = console.query_one("#console-transcript", ConsoleTranscript)
    transcript.focus()
    await pilot.press("down")   # select first message
    await pilot.pause(0.1)
    assert transcript.selected_message_id is not None
    await pilot.press("c")
    await pilot.pause(0.3)
    # The copy action surfaces its confirmation the same way the mouse path
    # does — assert on the same signal the existing copy-action test uses
    # (grep for the existing message-copy test and reuse its assertion).
```

Replace the `...` with the named test's arrange block and the final assertion with whatever the existing mouse-driven copy test asserts (grep `copy` in this file — reuse its exact post-action assertion so intent matches the mouse path). If the transcript widget id differs from `#console-transcript`, use the id from `_wait_for_selector` calls in neighboring tests.

- [ ] **Step 2: Run to verify failure** — `-k c_key`, expect selection works but `c` does nothing (assertion fails).

- [ ] **Step 3: Implement** in `console_transcript.py`: extend BINDINGS:

```python
        ("c", "invoke_selected_action('copy')", "Copy"),
        ("e", "invoke_selected_action('edit')", "Edit"),
        ("r", "invoke_selected_action('regenerate')", "Regenerate"),
```

and add:

```python
    def action_invoke_selected_action(self, action_id: str) -> None:
        """Press the selected message's action button for ``action_id``."""
        message_id = self.selected_message_id
        if not message_id:
            return
        selector = f"#console-message-action-{action_id}-{message_id}"
        try:
            button = self.query_one(selector, Button)
        except NoMatches:
            return
        button.press()
```

(Import `NoMatches` if not present; if action buttons only mount after `enter` (confirm_selection), call `self.action_confirm_selection()` first when the button is missing, then retry the query once — check `_action_row` mounting behavior at :703 and match reality.)

- [ ] **Step 4: Run** the new test + the transcript-related tests in the file (`-k "transcript or message_action"`). Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): direct copy/edit/regenerate keys on transcript selection"
```

---

### Task 6: Escape-to-composer, Ctrl+T, Alt+1..9, contextual footer hints

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`

**Interfaces:**
- Consumes: `_focus_console_composer_if_needed(force=True)`; `_create_native_console_session_from_active_context()` (:795); the session-tab activation sequence (:7833-7843); `_register_console_footer_shortcuts` (:746) + `AppFooterStatus.set_workbench_shortcuts(*, source, shortcuts: tuple[tuple[str, str], ...])`; the existing `CONSOLE_WORKBENCH_SHORTCUTS` constant (grep its definition); `_console_setup_modal_blocking()`.
- Produces:
  - BINDINGS: `Binding("escape", "focus_console_composer_home", "Composer", show=False)` (NOT priority — widget-level escapes like transcript clear-selection and modal dismiss keep winning); `Binding("ctrl+t", "new_console_tab", "New tab", show=False)`; `Binding("alt+1", "jump_console_tab(1)", show=False)` … through `alt+9` (nine bindings).
  - `def action_focus_console_composer_home(self) -> None` — guard blocking, then `_focus_console_composer_if_needed(force=True)`.
  - `def action_new_console_tab(self) -> None` — guard blocking, then `self.run_worker(self._create_native_console_session_from_active_context(), exclusive=False)`.
  - `def action_jump_console_tab(self, number: int) -> None` — guard blocking; sessions = `store.sessions()`; if `1 <= number <= len(sessions)`, activate `sessions[number-1].id` via a new shared helper `_activate_native_console_session(session_id)` extracted from the tab-click block (:7833-7843) so the tab click, the switcher (Task 3), and alt-jump share one path (refactor the tab-click handler and Task 3's callback to call it).
  - `def build_console_footer_shortcuts(pane: str) -> tuple[tuple[str, str], ...]` — MODULE-LEVEL pure function in chat_screen.py: `pane` in `{"composer", "transcript", "rail", "blocked"}` returning at most 5 pairs: composer → `(("Ctrl+K", "Switch"), ("Alt+M", "Model"), ("Ctrl+T", "New tab"), ("F6", "Panes"), ("Ctrl+P", "Palette"))`; transcript → `(("↑/↓", "Select"), ("C", "Copy"), ("E", "Edit"), ("R", "Regen"), ("Esc", "Composer"))`; rail → `(("Enter", "Open"), ("Ctrl+K", "Switch"), ("Esc", "Composer"), ("F6", "Panes"))`; blocked → `(("Enter", "Configure"), ("Ctrl+P", "Palette"))`.
  - `_register_console_footer_shortcuts` gains an optional `pane: str = "composer"` parameter and passes `build_console_footer_shortcuts(pane)`; a screen-level `on_descendant_focus` hook (extend the existing one if present — grep `def on_descendant_focus` in chat_screen.py) maps the focused widget to a pane (composer bar → composer; transcript → transcript; left rail/handles → rail; setup modal → blocked) and re-registers.

- [ ] **Step 1: Failing tests** (append to `Tests/UI/test_console_internals_decomposition.py`):

```python
def test_build_console_footer_shortcuts_per_pane():
    from tldw_chatbook.UI.Screens.chat_screen import build_console_footer_shortcuts
    composer = build_console_footer_shortcuts("composer")
    assert ("Ctrl+K", "Switch") in composer and len(composer) <= 5
    transcript = build_console_footer_shortcuts("transcript")
    assert ("C", "Copy") in transcript and ("Esc", "Composer") in transcript
    assert build_console_footer_shortcuts("blocked") == (
        ("Enter", "Configure"), ("Ctrl+P", "Palette"),
    )


@pytest.mark.asyncio
async def test_escape_returns_focus_to_composer_and_ctrl_t_opens_tab():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        rail = console.query_one("#console-left-rail")
        rail.focus()
        await pilot.pause(0.1)
        await pilot.press("escape")
        await pilot.pause(0.2)
        composer = console.query_one("#console-native-composer")
        assert app.focused is composer or composer in getattr(app.focused, "ancestors", [])
        store = console._console_chat_store
        before = len(store.sessions())
        await pilot.press("ctrl+t")
        await pilot.pause(0.4)
        assert len(store.sessions()) == before + 1


@pytest.mark.asyncio
async def test_alt_digit_jumps_to_tab():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.press("ctrl+t")
        await pilot.pause(0.4)
        store = console._console_chat_store
        first = store.sessions()[0]
        assert store.active_session_id != first.id
        await pilot.press("alt+1")
        await pilot.pause(0.4)
        assert store.active_session_id == first.id
```

- [ ] **Step 2: Run to verify failure** — `-k "footer_shortcuts or escape_returns or alt_digit"`.

- [ ] **Step 3: Implement** per the Produces block. The `_activate_native_console_session(session_id)` extraction: move the body of the tab-click activation branch (set workspace → switch → sync worker → focus composer) into the helper; the click handler, `action_jump_console_tab`, and Task 3's `_apply_console_switcher_choice` all call it (update Task 3's code if it landed with its own inline copy).

- [ ] **Step 4: Run** the three new tests + `Tests/UI/test_console_internals_decomposition.py` in full. Expected: PASS (no baseline failures in this file).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): escape-home, tab bindings, contextual footer hints"
```

---

### Task 7: Console command-palette provider

**Files:**
- Create: `tldw_chatbook/UI/console_command_provider.py`
- Modify: `tldw_chatbook/app.py` (COMMANDS registration, ~line 1134-1143)
- Test: `Tests/UI/test_console_internals_decomposition.py`

**Interfaces:**
- Consumes: Textual `from textual.command import Hit, Hits, Provider` (idiom: app.py:413-479 `ThemeProvider`); ChatScreen actions from Tasks 3/4/6 (`action_open_console_session_switcher`, `action_open_console_model_popover`, `action_new_console_tab`, `action_focus_console_composer_home`) plus existing `_open_console_settings`.
- Produces: `ConsoleCommandProvider(Provider)` — posting-style context-aware: commands are yielded ONLY when the active screen is the Console (`type(self.screen).__name__ == "ChatScreen"` or isinstance check via deferred import). Command list (label, help):
  - `Console: Switch session…` → `screen.action_open_console_session_switcher()`
  - `Console: Change model…` → `screen.action_open_console_model_popover()`
  - `Console: New chat tab` → `screen.action_new_console_tab()`
  - `Console: Focus composer` → `screen.action_focus_console_composer_home()`
  - `Console: Session settings…` → `screen.run_worker(screen._open_console_settings(), exclusive=False)`
  Implement both `async def discover()` (all commands) and `async def search(query)` (matcher-scored), mirroring ThemeProvider's structure exactly. Register in `App.COMMANDS` union in app.py.

- [ ] **Step 1: Failing test** (append to `Tests/UI/test_console_internals_decomposition.py`):

```python
@pytest.mark.asyncio
async def test_console_command_provider_lists_commands_only_on_console():
    from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        provider = ConsoleCommandProvider(screen=console, match_style=None)
        hits = [hit async for hit in provider.search("switch session")]
        assert hits, "expected Console commands on the Console screen"

        class _FakeScreen:  # not a ChatScreen
            pass
        other = ConsoleCommandProvider(screen=_FakeScreen(), match_style=None)
        other_hits = [hit async for hit in other.search("switch session")]
        assert other_hits == []
```

(If `Provider.__init__` takes different arguments, mirror how existing provider tests construct them — grep Tests/ for `ThemeProvider(` or `Provider(`; if none exist, constructing with `(screen, match_style)` per Textual's signature is correct for the installed version — verify against `textual.command.Provider.__init__`.)

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `tldw_chatbook/UI/console_command_provider.py` mirroring `ThemeProvider` (app.py:413-479) with the five commands above, guarding every yield on the screen being the Console:

```python
"""Console-scoped command palette provider (posting-style)."""

from __future__ import annotations

from textual.command import Hit, Hits, Provider


class ConsoleCommandProvider(Provider):
    """Yield Console actions only while the Console screen is active."""

    def _console_screen(self):
        screen = self.screen
        if type(screen).__name__ != "ChatScreen":
            return None
        return screen

    def _commands(self, screen) -> tuple[tuple[str, object, str], ...]:
        return (
            ("Console: Switch session…", screen.action_open_console_session_switcher,
             "Fuzzy-find and activate a conversation (Ctrl+K)"),
            ("Console: Change model…", screen.action_open_console_model_popover,
             "Quick provider/model/temperature switch (Alt+M)"),
            ("Console: New chat tab", screen.action_new_console_tab,
             "Open a new Console chat tab (Ctrl+T)"),
            ("Console: Focus composer", screen.action_focus_console_composer_home,
             "Return focus to the composer (Esc)"),
            ("Console: Session settings…",
             lambda: screen.run_worker(screen._open_console_settings(), exclusive=False),
             "Open the full session settings modal"),
        )

    async def discover(self) -> Hits:
        screen = self._console_screen()
        if screen is None:
            return
        for label, callback, help_text in self._commands(screen):
            yield Hit(1.0, label, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        screen = self._console_screen()
        if screen is None:
            return
        matcher = self.matcher(query)
        for label, callback, help_text in self._commands(screen):
            score = matcher.match(label)
            if score > 0:
                yield Hit(score, matcher.highlight(label), callback, help=help_text)
```

Match the exact Hit construction used by ThemeProvider in app.py (positional/keyword shape varies by Textual version — copy its form). Register in app.py COMMANDS: add `ConsoleCommandProvider` to the union with a lazy import near the other providers.

- [ ] **Step 4: Run** the new test + `env HOME=... .venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py --tb=no` (all pass) + a quick app-level import smoke: `.venv/bin/python -c "import tldw_chatbook.app"`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/console_command_provider.py tldw_chatbook/app.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): console command palette provider"
```

---

### Task 8: Verification, screenshot QA, approval gate

**Files:**
- Create: screenshots under `Docs/superpowers/qa/console-keyboard-2026-07/`

- [ ] **Step 1: Full affected test run**

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q \
  Tests/Chat/test_console_switcher_state.py Tests/UI/test_console_rail_sections.py \
  Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_session_settings.py --tb=short
```

Expected: ALL PASS except the one documented pre-existing dev failure (`test_console_left_rail_prioritizes_attach_and_active_conversation`).

- [ ] **Step 2: Live screenshot QA**

Proven recipe (bundled chromium, `.intro-dialog` wait, route-abort external, fresh/seeded HOMEs, kill stale app processes first; driver `/private/tmp/tldw-console-rail-ia-cdp-20260702/cap.py` — copy it if missing). Use the ready-seeded config (llama_cpp) HOME. Capture:
1. Ctrl+K switcher open with a query typed and filtered results.
2. Alt+M popover open.
3. Footer hints with composer focused vs transcript focused (two frames showing different hint sets).
4. Command palette (Ctrl+P) filtered to `Console:` commands.

- [ ] **Step 3: User approval gate** — present captures; no merge without explicit approval.

- [ ] **Step 4: Commit QA artifacts**

```bash
git add Docs/superpowers/qa/console-keyboard-2026-07/
git commit -m "docs(console): keyboard layer QA evidence"
```
