# Console Left Sidebar Usability & Legibility Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the Console left rail so the Session, Context, Model, and Details sections are visually scannable, add workspace switch/create affordances, and improve model/source readability without adding icons or new backend flows.

**Architecture:** The change is primarily a CSS and widget-composition pass on existing Console rail components. A shared workspace-identity helper is moved from the Library screen to the workspace registry service so the Console can create local workspaces. All visual changes stay within the existing four-section rail structure and persistence model.

**Tech Stack:** Python 3.11+, Textual ≥3.3.0, pytest, SQLite workspace registry, existing tldw_chatbook CSS bundle.

**Spec:** `Docs/superpowers/specs/2026-07-18-console-left-sidebar-usability-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backlog/decisions/017-console-left-rail-usability.md` | ADR recording the left-rail visual-language and shared-workspace-helper decisions. |
| `tldw_chatbook/Workspaces/registry_service.py` | Shared `next_local_workspace_identity()` helper; workspace CRUD. |
| `tldw_chatbook/UI/Screens/library_screen.py` | Adopt shared helper for existing Library create-workspace flow. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Compose Console rail sections; wire `[New]` workspace handler; refresh model rows. |
| `tldw_chatbook/Widgets/Console/console_rail_section.py` | Remove inline header height constraints. |
| `tldw_chatbook/Widgets/Console/console_workspace_context.py` | New workspace row + `[New]` button; 12-column status labels; conversation-browser header styling. |
| `tldw_chatbook/Widgets/Console/console_staged_context.py` | Context tray header with count badge; two-line source rows; remove dead `Attach` button. |
| `tldw_chatbook/css/components/_agentic_terminal.tcss` | Rail header, body, label, toggle, and source-status styles. |
| `Tests/UI/test_console_*.py` | Updated selectors and new coverage. |

---

### Task 1: Create the ADR

**Files:**
- Create: `backlog/decisions/017-console-left-rail-usability.md`

- [ ] **Step 1: Write the ADR**

Use the repository's ADR template. Record:
- Decision: Adopt a text-only, bordered-section left-rail visual language for the Console screen.
- Decision: Move local-workspace identity generation into `Workspaces/registry_service.py` for cross-screen reuse.
- Consequences: Affects future Console rail features; both Library and Console screens depend on the shared helper.

- [ ] **Step 2: Commit**

```bash
git add backlog/decisions/017-console-left-rail-usability.md
git commit -m "docs(adr): console left-rail usability redesign"
```

---

### Task 2: Move workspace identity helper to shared service

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:2161-2177` (helper definition)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:8876-8894` (caller)
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Test: `Tests/Workspaces/test_workspace_registry_service.py` (or create a focused test if none exists)

- [ ] **Step 1: Write a failing test for the shared helper**

```python
def test_next_local_workspace_identity_generates_unique_ids(tmp_path: Path):
    """The shared helper produces a unique workspace id and name each call."""
    from tldw_chatbook.Workspaces.registry_service import next_local_workspace_identity
    from Tests.Workspaces.test_workspace_registry_service import build_test_registry

    registry_service = build_test_registry(tmp_path)
    id1, name1 = next_local_workspace_identity(registry_service)
    id2, name2 = next_local_workspace_identity(registry_service)
    assert id1 != id2
    assert name1 != name2
    assert id1.startswith("workspace-local-")
    assert "Workspace" in name1
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
cd /worktree/path && PYTHONPATH=. python -m pytest Tests/Workspaces/test_workspace_registry_service.py::test_next_local_workspace_identity_generates_unique_ids -v
```

Expected: `ImportError: cannot import name 'next_local_workspace_identity'`

- [ ] **Step 3: Move the helper into registry_service.py**

Copy `_next_local_workspace_identity` logic from `library_screen.py` into `Workspaces/registry_service.py` as `next_local_workspace_identity(registry_service)`. Keep the same id/name format.

- [ ] **Step 4: Update LibraryScreen to use the shared helper**

Replace the body of `library_screen.py:_next_local_workspace_identity()` with a call to the shared helper, or delete the method and update `create_local_workspace()` to import/call `next_local_workspace_identity`.

- [ ] **Step 5: Run the test and confirm it passes**

```bash
PYTHONPATH=. python -m pytest Tests/Workspaces/test_workspace_registry_service.py::test_next_local_workspace_identity_generates_unique_ids -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/UI/Screens/library_screen.py Tests/Workspaces/test_workspace_registry_service.py
git commit -m "refactor(workspaces): share next_local_workspace_identity between Library and Console"
```

---

### Task 3: Restyle rail section headers

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_rail_section.py:33-39`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:1774-1792`
- Test: `Tests/UI/test_console_rail_sections.py` (create if missing, or add to `Tests/UI/test_console_persistent_rails.py`)

- [ ] **Step 1: Write a failing test for header height**

```python
def test_section_header_allows_border_height():
    from tldw_chatbook.Widgets.Console.console_rail_section import ConsoleRailSectionHeader
    header = ConsoleRailSectionHeader("Session", section_id="session", open=True)
    # Inline height constraints should be gone so CSS can set min-height 2.
    assert header.styles.height is None or header.styles.height.value != 1
    assert header.styles.max_height is None
```

- [ ] **Step 2: Remove inline height constraints**

Delete:
```python
self.styles.height = 1
self.styles.min_height = 1
self.styles.max_height = 1
```
from `ConsoleRailSectionHeader.__init__`.

- [ ] **Step 3: Update CSS**

Replace the `.console-rail-section-header` block with:
```css
.console-rail-section-header {
    height: auto;
    min-height: 2;
    border-top: solid $ds-column-line;
    content-align: center middle;
}

.console-rail-section-title {
    width: 1fr;
    text-style: bold;
    color: $ds-text-primary;
}

.console-rail-section-toggle {
    width: 3;
    min-width: 3;
    text-style: bold;
}

.console-rail-section-toggle:focus {
    background: $ds-action-focus 30%;
    text-style: underline bold;
}
```

- [ ] **Step 4: Run the test and snapshot/visual smoke**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_rail_sections.py::test_section_header_allows_border_height -v
PYTHONPATH=. python -m pytest Tests/UI/test_console_persistent_rails.py -v -k header
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_rail_section.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_rail_sections.py
git commit -m "feat(console): restyle rail section headers"
```

---

### Task 4: Restyle rail section bodies

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:1789-1792`
- Test: existing UI tests

- [ ] **Step 1: Update `.console-rail-section-body` CSS**

```css
.console-rail-section-body {
    height: auto;
    min-height: 0;
    padding: 0 1 1 1;
}
```

(No bottom border; the next header's top border provides the divider.)

- [ ] **Step 2: Run targeted UI tests**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_workspace_context_rail.py -q
```

Expected: PASS (may need selector updates later)

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss
git commit -m "style(console): restyle rail section bodies"
```

---

### Task 5: Update Session workspace row and add [New] button

**Files:**
- Modify: `tldw_chatbook/Workspaces/display_state.py:184-216` (add `workspace_name`, `scope_label`, `new_workspace_enabled` to `ConsoleWorkspaceContextState`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (builder that produces `ConsoleWorkspaceContextState`)
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py:88-112` (status label width)
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py:434-470` (compose Session tray)
- Test: `Tests/UI/test_console_workspace_context_rail.py` (create or update)

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_session_tray_shows_workspace_scope_and_new_button():
    from textual.widgets import Button, Static
    from textual.app import App
    from tldw_chatbook.Widgets.Console.console_workspace_context import ConsoleWorkspaceContextTray
    from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState

    class TestApp(App):
        def compose(self):
            yield ConsoleWorkspaceContextTray(
                ConsoleWorkspaceContextState(
                    heading="Session",
                    workspace_label="demo",
                    workspace_name="demo",
                    authority_label="local",
                    sync_label="local",
                    runtime_label="local",
                    conversation_rows=(),
                    conversation_empty_copy="",
                    change_workspace_enabled=True,
                    change_workspace_recovery="",
                    new_conversation_enabled=True,
                    new_conversation_recovery="",
                    recovery_copy="",
                    scope_label="conv-1",
                    new_workspace_enabled=True,
                )
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleWorkspaceContextTray)
        workspace_label = tray.query_one("#console-active-workspace .console-workspace-status-label", Static)
        scope_label = tray.query_one("#console-active-scope .console-workspace-status-label", Static)
        assert "Workspace" in workspace_label.renderable.plain
        assert "Scope" in scope_label.renderable.plain
        assert tray.query_one("#console-new-workspace", Button)

@pytest.mark.asyncio
async def test_status_label_width_is_twelve():
    from textual.app import App
    from tldw_chatbook.Widgets.Console.console_workspace_context import ConsoleWorkspaceStatusPair

    class TestApp(App):
        def compose(self):
            yield ConsoleWorkspaceStatusPair("Workspace", "demo", label_id="l", value_id="v")

    app = TestApp()
    async with app.run_test():
        label = app.query_one(".console-workspace-status-label")
        assert label.styles.width.value == 12
```

- [ ] **Step 2: Change status label width to 12**

In `ConsoleWorkspaceStatusPair.compose()`, set `label_widget.styles.width = 12` (and `min_width = 12`).

- [ ] **Step 3: Compose the new workspace row**

First, extend `ConsoleWorkspaceContextState` with defaults:
- `workspace_name: str = ""`
- `scope_label: str = ""`
- `new_workspace_enabled: bool = False`

Update the builder in `ChatScreen` (`_build_console_workspace_context_state`) to populate these fields:
- `workspace_name` from the active workspace record name (or the existing label).
- `scope_label` from the current conversation id/label.
- `new_workspace_enabled` from `getattr(app_instance, "workspace_registry_service", None) is not None`.

Then update `ConsoleWorkspaceContextTray.compose()` to yield:
1. `#console-active-workspace` `ConsoleWorkspaceStatusPair` with label `Workspace` and value `self.state.workspace_name`.
2. A new `Horizontal` containing `#console-change-workspace` and `#console-new-workspace` Buttons, placed on the line below the workspace value and aligned under the value column. Achieve the alignment by setting `margin-left: 12` on the button `Horizontal` (matching the 12-column label width) or by yielding a 12-column placeholder `Static` followed by the buttons.
3. `#console-active-scope` `ConsoleWorkspaceStatusPair` with label `Scope` and value `self.state.scope_label` (or an empty fallback).
4. `#console-workspace-recovery` Static if present.
5. Conversation browser as before.

Set `[New].disabled = not self.state.new_workspace_enabled`. Set `[Switch].disabled = not self.state.change_workspace_enabled`.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_workspace_context_rail.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_context.py Tests/UI/test_console_workspace_context_rail.py
git commit -m "feat(console): add Switch/New workspace actions to Session rail"
```

---

### Task 6: Wire [New] workspace handler in ChatScreen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:990-1045` (near existing workspace switch handler)
- Test: `Tests/UI/test_console_new_workspace.py` (create)

- [ ] **Step 1: Write a failing test**

```python
@pytest.mark.asyncio
async def test_console_new_workspace_creates_and_activates(tmp_path: Path):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from textual.widgets import Button

    class TestApp(App):
        def compose(self):
            yield ChatScreen(self)

    app = TestApp()
    registry_service = build_test_registry(tmp_path)
    app.workspace_registry_service = registry_service
    async with app.run_test() as pilot:
        chat = app.query_one(ChatScreen)
        before = len(registry_service.list_workspaces())
        chat.query_one("#console-new-workspace", Button).press()
        await pilot.pause(0.2)
        assert len(registry_service.list_workspaces()) == before + 1
        active = registry_service.get_active_workspace()
        assert active is not None
```

- [ ] **Step 2: Add the handler**

Add `@on(Button.Pressed, "#console-new-workspace")` handler in `ChatScreen`:
- Get `workspace_registry_service` from `app_instance`.
- If missing, notify and return.
- Call `next_local_workspace_identity(registry_service)`.
- Call `registry_service.create_workspace(workspace_id=..., name=..., description="Local workspace created from Console.")`.
- On success: `registry_service.set_active_workspace(workspace_id)`, then run the same post-switch flow as the existing switch handler (`_sync_console_chat_core_state`, `_activate_console_session_for_workspace`, `_sync_console_workspace_context`, `run_worker(self._sync_native_console_chat_ui(), exclusive=True)`).
- On `WorkspaceRegistryServiceError` or any unexpected exception: notify "Workspace could not be created."

- [ ] **Step 3: Run the test**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_new_workspace.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_new_workspace.py
git commit -m "feat(console): implement New workspace button in left rail"
```

---

### Task 7: Style conversation-browser sub-section header

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py:601-612` (browser header)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (add `.console-workspace-conversations-header`)
- Test: existing UI tests

- [ ] **Step 1: Update the browser header composition**

There are two compose paths that yield `#console-workspace-conversations-header`: the grouped browser path (`_compose_conversation_browser`) and the legacy path (`_compose_legacy_conversation_section`). Update both so the `Horizontal` has `classes="console-rail-header console-workspace-conversations-header"` and the title Static uses `classes="console-rail-section-title"`.

Do not change `_CONVERSATION_BROWSER_HEADER_HEIGHT`; it controls inner section/group header heights, not the outer browser header. The outer header's two-cell height is handled solely by the new CSS.

- [ ] **Step 2: Add CSS**

```css
.console-workspace-conversations-header {
    height: auto;
    min-height: 2;
    border-top: solid $ds-column-line;
    content-align: center middle;
}
```

- [ ] **Step 3: Run targeted tests**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_workspace_context_rail.py -v -k conversation
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/css/components/_agentic_terminal.tcss
git commit -m "style(console): style conversation browser header like rail sections"
```

---

### Task 8: Redesign Context staged-sources tray

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_staged_context.py:32-62`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/UI/test_console_staged_context.py` (create or update)

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_staged_context_renders_source_count():
    from textual.app import App
    from tldw_chatbook.Widgets.Console.console_staged_context import ConsoleStagedContextTray
    from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState, ConsoleDisplayRow

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(
                    heading="Context",
                    summary="",
                    rows=(ConsoleDisplayRow("Source", "readme.md", status="ready"),),
                )
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        assert tray.query_one("#console-staged-context-count")

@pytest.mark.asyncio
async def test_staged_context_omits_attach_button():
    from textual.app import App
    from tldw_chatbook.Widgets.Console.console_staged_context import ConsoleStagedContextTray
    from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState

    class TestApp(App):
        def compose(self):
            yield ConsoleStagedContextTray(
                ConsoleStagedContextState(heading="Context", summary="", rows=())
            )

    app = TestApp()
    async with app.run_test():
        tray = app.query_one(ConsoleStagedContextTray)
        assert not list(tray.query("#console-staged-context-attach"))
```

- [ ] **Step 2: Update compose**

Replace the flat row loop with:
1. A `Horizontal` header containing a `Static("Sources", id="console-staged-context-title")` and `#console-staged-context-count` badge. Do not use `self.state.heading` for the header label.
2. For each row: a `Vertical` with `.console-staged-source-name` (row.value) and `.console-staged-source-status` (normalized status). Normalize status as: `ready`/`available`/`attached` → `ready`; `retrieving`/`running`/`stale` → `running`; `blocked`/`missing`/`unavailable` → `blocked`; anything else → `muted`.
3. If empty: `#console-staged-context-empty` Static with "No sources attached. Stage sources from Library." and remove the `Attach` Button.
4. Recovery Static if present.

- [ ] **Step 3: Add CSS**

```css
#console-staged-context-title {
    width: 1fr;
    text-style: bold;
}

#console-staged-context-count {
    width: auto;
    text-align: right;
    color: $ds-text-muted;
}

.console-staged-source-name {
    text-wrap: wrap;
    max-height: 2;
    overflow: hidden;
}

.console-staged-source-status {
    height: 1;
    color: $ds-text-muted;
}

.console-staged-source-status.ready { color: $ds-status-ready; }
.console-staged-source-status.running { color: $ds-status-running; }
.console-staged-source-status.blocked { color: $ds-status-blocked; }

#console-staged-context-empty {
    color: $ds-text-muted;
}
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_staged_context.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_staged_context.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_staged_context.py
git commit -m "feat(console): redesign staged context tray with status colors"
```

---

### Task 9: Redesign Model section rows

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:5309-5341` (compose)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:1566-1580` (sync method)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/UI/test_console_model_section.py` (create)

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_model_section_renders_four_rows():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class TestApp(App):
        def compose(self):
            yield ChatScreen(self)

    app = TestApp()
    async with app.run_test():
        chat = app.query_one(ChatScreen)
        assert chat.query_one("#console-model-section-provider")
        assert chat.query_one("#console-model-section-model")
        assert chat.query_one("#console-model-section-temperature")
        assert chat.query_one("#console-model-section-max-tokens")

@pytest.mark.asyncio
async def test_model_sync_updates_rows():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class TestApp(App):
        def compose(self):
            yield ChatScreen(self)

    app = TestApp()
    async with app.run_test():
        chat = app.query_one(ChatScreen)
        chat._sync_console_settings_summary()
        provider = chat.query_one("#console-model-section-provider .console-model-section-value", Static)
        assert provider.renderable.plain.strip()
```

- [ ] **Step 2: Replace compose in ChatScreen**

Instead of `line1`/`line2` statics, yield:
- `#console-model-section-provider` (label `Provider`, value from `summary_state.provider_row`)
- `#console-model-section-model` (label `Model`, value from `summary_state.model_row`)
- `#console-model-section-temperature` (label `Temperature`, value parsed from `summary_state.sampling_row` via regex `T ([\d.]+)`)
- `#console-model-section-max-tokens` (label `Max tokens`, value parsed from `summary_state.sampling_row` via regex `max_tokens (\d+)`)
- `#console-model-section-recovery` (only when `summary_state.readiness_label` is non-empty and not `"Ready"`; displays the readiness label)
- `#console-model-section-configure` Button

Each row is a `Horizontal` with a 12-column muted label Static (`classes="console-model-section-label"`) and a value Static (`classes="console-model-section-value"`). Display `"—"` when a parsed value is missing.

- [ ] **Step 3: Update sync method**

In `_sync_console_settings_summary()`:
- Query the four new row value widgets.
- Update Provider and Model values from `summary_state.provider_row` and `summary_state.model_row`.
- Extract Temperature and Max tokens from `summary_state.sampling_row` using regex; update those values (or `"—"`).
- Show/hide `#console-model-section-recovery` based on `summary_state.readiness_label` (visible when non-empty and not `"Ready"`).

- [ ] **Step 4: Add CSS**

```css
.console-model-section-line {
    height: auto;
    min-height: 1;
}

.console-model-section-label {
    width: 12;
    min-width: 12;
    color: $ds-text-muted;
}

.console-model-section-value {
    width: 1fr;
    min-width: 10;
    color: $ds-text-primary;
}

#console-model-section-provider .console-model-section-value,
#console-model-section-model .console-model-section-value {
    text-wrap: wrap;
    max-height: 3;
    overflow: hidden;
}

#console-model-section-temperature .console-model-section-value,
#console-model-section-max-tokens .console-model-section-value {
    text-wrap: nowrap;
}

#console-model-section-recovery {
    color: $ds-status-blocked;
}
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_model_section.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_model_section.py
git commit -m "feat(console): split model settings into labeled rows"
```

---

### Task 10: Rail title rename and Details styling

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:5207-5213` (rail title Static)
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_details.py` (if row classes need styling)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/UI/test_console_rail_title.py` (create)

- [ ] **Step 1: Write a failing test**

Create `Tests/UI/test_console_rail_title.py`:

```python
@pytest.mark.asyncio
async def test_console_rail_title_reads_console_context():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class TestApp(App):
        def compose(self):
            yield ChatScreen(self)

    app = TestApp()
    async with app.run_test():
        title = app.query_one("#console-context-rail-title", Static)
        assert "Console context" in title.renderable.plain
```

- [ ] **Step 2: Rename rail title**

Change `Static("Session & Context", ...)` to `Static("Console context", ...)`. Update the collapse button tooltip from `"Collapse Session & Context rail"` to `"Collapse Console context rail"`.

- [ ] **Step 3: Add Details CSS**

`ConsoleWorkspaceDetailsTray` already renders `ConsoleWorkspaceStatusPair` widgets, which use `.console-workspace-status-label` and `.console-workspace-status-value`. The 12-column label width is set in Task 5. Add color styling:

```css
.console-workspace-status-label {
    color: $ds-text-muted;
}
.console-workspace-status-value {
    color: $ds-text-primary;
}
.console-workspace-empty-copy {
    color: $ds-text-muted;
}
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_rail_title.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_workspace_details.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_rail_title.py
git commit -m "feat(console): rename rail title and style Details rows"
```

---

### Task 11: Full UI gate and regression fixes

**Files:**
- All modified above
- Tests: `Tests/UI`

- [ ] **Step 1: Run targeted Console tests**

```bash
PYTHONPATH=. python -m pytest Tests/UI/test_console_ Tests/UI/test_master_shell_navigation.py -q
```

- [ ] **Step 2: Fix any failing selectors/counts**

Common breaks and likely affected test files:
- `Tests/UI/test_console_rail_sections.py`, `Tests/UI/test_console_workspace_context_rail.py`, `Tests/UI/test_console_staged_context.py`, `Tests/UI/test_console_model_section.py`, `Tests/UI/test_console_new_workspace.py`, `Tests/UI/test_console_rail_title.py`, and `Tests/UI/test_console_persistent_rails.py`.
- Tests counting `#console-rail-section-header-*` will still pass (IDs unchanged).
- Tests querying `#console-active-workspace` copy may need to assert the new row format.
- Tests querying `#console-model-section-line1` must be updated to the new IDs.
- Tests expecting `#console-staged-context-attach` must be removed/updated.
- Add new classes (`.console-staged-source-status`, `.console-model-section-label`, `.console-model-section-value`, `.console-workspace-conversations-header`) to `Tests/UI/test_console_persistent_rails.py` if it guards generated stylesheets.

- [ ] **Step 3: Run full UI gate**

```bash
PYTHONPATH=. python -m pytest Tests/UI -q
```

Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add Tests/UI
git commit -m "test(console): update UI tests for left-rail redesign"
```

---

## Execution Notes

- Work should be done in a dedicated worktree (e.g. `.worktrees/console-sidebar-usability`) branched from `dev`.
- Rebase onto latest `dev` before opening a PR.
- Each task commits independently so the diff stays reviewable.
- The ADR must be created before implementation code changes begin.
- If any CSS class is added, add it to `Tests/UI/test_console_persistent_rails.py` if that test guards generated stylesheets.

## Implementation Notes

- All planned tasks completed; the full `Tests/UI` gate passes (Set A, Set B, Set C) plus targeted Console smoke tests after the final rebase.
- Code review fixes applied before merge:
  - `#console-model-section-recovery` is now always composed (hidden via `display: none`) and `_sync_console_settings_summary` only toggles visibility/updates text, avoiding a full-screen `recompose=True` on the Console polling path.
  - `change_workspace_recovery` copy is rendered below the Switch/New button row when switching is disabled, restoring the explanation that the disabled `[Switch]` button previously lost.
  - New rail CSS classes were added to the generated-stylesheet guard in `Tests/UI/test_console_persistent_rails.py`.
- Out-of-scope but gate-blocking fix: removed the stable `id` from rule-row separators in `tldw_chatbook/Widgets/Console/console_transcript.py` to prevent `DuplicateIds` errors during transcript recomposes that blocked the UI gate. This change is recorded here for changelog/traceability purposes.
