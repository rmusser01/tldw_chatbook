# Chat-First UX Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `tldw_chatbook` into a chat-first agentic console with one unified shell, Chat as the primary programming/control surface, and supporting Library/Study/Characters/Settings destinations that follow a single screen contract.

**Architecture:** This plan performs a shell-first migration. It introduces a persistent app shell, replaces the current many-route tab model with a smaller destination IA, migrates coding workflows into Chat, and wraps the existing feature windows behind new destination screens instead of rewriting every subsystem at once. The plan preserves working surfaces where possible, adds thin adapter widgets for aggregation, and moves progressively from shell/routing to Chat, Library, Study, handoffs, and safety/resume behavior.

**Tech Stack:** Python 3.11, Textual screens/widgets/reactive/message system, pytest UI tests, existing `tldw_chatbook` screen wrappers and widget modules, git worktrees

---

## File Structure

- `tldw_chatbook/UI/Navigation/app_shell.py`
  Persistent shell container that owns the global header, top-level navigation, and screen body region.
- `tldw_chatbook/Widgets/app_header.py`
  Product header with workspace selector, scope indicators, and quick-switch entry point.
- `tldw_chatbook/UI/Navigation/navigation_messages.py`
  Shared Textual messages for navigation, workspace changes, and cross-surface handoffs.
- `tldw_chatbook/state/workspace_state.py`
  Workspace/global/mixed-scope state and label helpers used by the shell and destination screens.
- `tldw_chatbook/UI/Navigation/main_navigation.py`
  Top-level navigation reduced to `Chat`, `Library`, `Study`, `Characters`, `Models & Tools`, `Automation / Feeds`, and `Settings`.
- `tldw_chatbook/UI/Navigation/base_app_screen.py`
  Base screen contract. Must stop rendering its own top navigation once `AppShell` exists.
- `tldw_chatbook/app.py`
  Root app composition and screen switching. Must mount exactly one shell and stop composing legacy nav widgets by default.
- `tldw_chatbook/navigation/screen_registry.py`
  Canonical destination registry with new top-level routes plus legacy aliases.
- `tldw_chatbook/Constants.py`
  Transition constants and aliases. Keep legacy names only as compatibility aliases, not primary IA labels.
- `tldw_chatbook/UI/Screens/library_screen.py`
  New top-level Library destination with section switcher and adapters for Notes, Media, Search, Ingest, and Chatbooks.
- `tldw_chatbook/Widgets/library_sections.py`
  Thin adapter widgets that host existing notes/media/search/ingest/chatbooks content inside Library.
- `tldw_chatbook/Widgets/Note_Widgets/notes_workspace.py`
  Reusable notes workspace widget extracted from `NotesScreen.compose_content()` so Notes can be embedded in Library without nesting `Screen` objects.
- `tldw_chatbook/Widgets/library_action_bar.py`
  Selection-aware action bar for `Use in Chat`, preview, and packaging actions.
- `tldw_chatbook/UI/Screens/characters_screen.py`
  Top-level Characters destination that wraps the CCP surface with a destination header and section switcher.
- `tldw_chatbook/UI/Screens/models_tools_screen.py`
  Top-level Models & Tools destination with sections for Models, Speech, Evals, and Tools.
- `tldw_chatbook/Widgets/models_tools_sections.py`
  Adapter widgets for current LLM/STTS/Evals/Tools surfaces.
- `tldw_chatbook/UI/Screens/automation_feeds_screen.py`
  Top-level Automation / Feeds destination that wraps the current subscription surface.
- `tldw_chatbook/UI/Screens/settings_screen.py`
  Top-level Settings destination with sections for app settings, customization, logs, and stats.
- `tldw_chatbook/Widgets/section_switcher.py`
  Reusable local section switcher for destination-internal navigation.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_context_bar.py`
  Workspace, repo, branch, persona, and execution-scope chips with quick actions.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`
  Inline task/progress/result/diff cards for the Chat thread.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`
  Inline approval request presentation for risky or privileged actions.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_resume_panel.py`
  Recent task history and resume affordances.
- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
  Main Chat surface. Must become conversation-first and absorb coding/runtime controls as contextual panels.
- `tldw_chatbook/UI/Screens/chat_screen.py`
  Chat-level state, destination header context, handoff entry points, and task continuity wiring.
- `tldw_chatbook/UI/Screens/chat_screen_state.py`
  Chat session/task persistence updates for approvals, diffs, and resume state.
- `tldw_chatbook/UI/Coding_Window.py`
  Transitional source of coding tools to migrate into Chat sub-panels.
- `tldw_chatbook/UI/Screens/coding_screen.py`
  Transitional redirect/alias only; should not remain in primary navigation once Chat reaches parity.
- `tldw_chatbook/Widgets/Study/study_dashboard.py`
  Default Study landing dashboard with due work, recents, and resume actions.
- `tldw_chatbook/Widgets/Study/quiz_session_widget.py`
  Focused quiz entry/session surface with explicit scope and result state.
- `tldw_chatbook/UI/Study_Window.py`
  Existing Study container to split so dashboard and quiz flow can be added without further bloating the file.
- `tldw_chatbook/UI/Screens/study_screen.py`
  Top-level Study wrapper and handoff target.
- `Tests/UI/test_app_shell.py`
  New tests for single-shell composition and workspace header behavior.
- `Tests/UI/test_destination_shells.py`
  New tests for top-level IA and section switchers.
- `Tests/UI/test_library_screen.py`
  New tests for Library default section, switching, selection, and `Use in Chat`.
- `Tests/UI/test_chat_agentic_console.py`
  New tests for Chat context bar, at-rest layout, and coding-panel migration.
- `Tests/UI/test_chat_first_handoffs.py`
  New tests for `Library -> Chat`, `Study -> Chat`, and `Characters -> Chat`.
- `Tests/UI/test_study_dashboard.py`
  New tests for Study dashboard and quiz entry flow.
- `Tests/UI/test_chat_approvals_and_resume.py`
  New tests for approval cards and task continuity.
- `Docs/Development/chat-first-shell-migration.md`
  Implementation notes, migration decisions, and legacy-route cleanup notes.
- `Docs/Development/navigation-architecture-analysis.md`
  Update to reflect the new shell and destination model rather than the old screen migration story.

### Implementation Constraints

- Use a dedicated git worktree before editing code. Do not implement this plan in the current dirty workspace.
- Do not rewrite existing Notes/Media/Search/Chatbooks internals in the same change that creates the new top-level destination shells. Wrap first, refine second.
- Do not try to mount `Screen` objects inside other `Screen` objects. Extract reusable section widgets or host the underlying `Window`/`Widget` classes instead.
- Keep old route names as aliases during migration, but remove them from primary navigation immediately once the new shell is live.
- Preserve current working tests where possible. When a test only validates obsolete IA, replace it with a test that validates the new destination model.

### Task 1: Create The Unified App Shell

**Files:**
- Create: `tldw_chatbook/UI/Navigation/app_shell.py`
- Create: `tldw_chatbook/Widgets/app_header.py`
- Create: `tldw_chatbook/UI/Navigation/navigation_messages.py`
- Create: `tldw_chatbook/state/workspace_state.py`
- Create: `Tests/UI/test_app_shell.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Navigation/base_app_screen.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/state/navigation_state.py`

- [ ] **Step 1: Write the failing shell tests**

```python
@pytest.mark.asyncio
async def test_app_shell_renders_one_header_and_one_navigation(mock_config):
    app = TldwCli()
    async with app.run_test() as pilot:
        assert pilot.app.query("#app-header").count() == 1
        assert pilot.app.query("MainNavigationBar").count() == 1
        assert pilot.app.query("#screen-content").count() == 1

@pytest.mark.asyncio
async def test_base_screen_no_longer_renders_its_own_navigation():
    class ScreenHost(App):
        def compose(self):
            yield ChatScreen(Mock())

    app = ScreenHost()
    async with app.run_test() as pilot:
        screen = app.query_one(ChatScreen)
        assert screen.query("MainNavigationBar").count() == 0
```

- [ ] **Step 2: Run the shell tests to verify they fail**

Run:

```bash
pytest Tests/UI/test_app_shell.py -q
```

Expected: FAIL because `AppShell`, `AppHeader`, and the single-shell composition do not exist yet.

- [ ] **Step 3: Write the minimal shell implementation**

```python
# tldw_chatbook/state/workspace_state.py
@dataclass
class WorkspaceState:
    current_workspace: str = "default"
    scope_label: str = "Workspace"
    is_global_override: bool = False

    def visible_scope_text(self) -> str:
        return "Global" if self.is_global_override else self.current_workspace
```

```python
# tldw_chatbook/UI/Navigation/app_shell.py
class AppShell(Container):
    def compose(self) -> ComposeResult:
        yield AppHeader(id="app-header")
        yield MainNavigationBar(id="app-main-navigation")
        yield Container(id="app-shell-body")
```

```python
# tldw_chatbook/UI/Navigation/base_app_screen.py
def compose(self) -> ComposeResult:
    with Container(id="screen-content"):
        yield from self.compose_content()
```

```python
# tldw_chatbook/app.py
def _create_main_ui_widgets(self) -> List[Widget]:
    return [AppShell(id="app-shell"), AppFooterStatus(id="app-footer-status")]
```

- [ ] **Step 4: Run the shell and navigation regression tests**

Run:

```bash
pytest Tests/UI/test_app_shell.py -q
```

Expected: PASS, proving the shell composes exactly one header and one top navigation bar.

- [ ] **Step 5: Commit the shell foundation**

```bash
git add tldw_chatbook/UI/Navigation/app_shell.py tldw_chatbook/Widgets/app_header.py tldw_chatbook/UI/Navigation/navigation_messages.py tldw_chatbook/state/workspace_state.py tldw_chatbook/app.py tldw_chatbook/UI/Navigation/base_app_screen.py tldw_chatbook/UI/Navigation/main_navigation.py tldw_chatbook/state/navigation_state.py Tests/UI/test_app_shell.py
git commit -m "feat: add unified app shell foundation"
```

### Task 2: Replace The Top-Level IA With Destination Shells

**Files:**
- Create: `tldw_chatbook/UI/Screens/library_screen.py`
- Create: `tldw_chatbook/UI/Screens/characters_screen.py`
- Create: `tldw_chatbook/UI/Screens/models_tools_screen.py`
- Create: `tldw_chatbook/UI/Screens/automation_feeds_screen.py`
- Create: `tldw_chatbook/UI/Screens/settings_screen.py`
- Create: `tldw_chatbook/Widgets/section_switcher.py`
- Create: `Tests/UI/test_destination_shells.py`
- Modify: `tldw_chatbook/navigation/screen_registry.py`
- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/UI/Tab_Links.py`
- Modify: `tldw_chatbook/UI/Tab_Bar.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/UI/test_tab_links_navigation.py`

- [ ] **Step 1: Write the failing destination-shell tests**

```python
@pytest.mark.asyncio
async def test_main_navigation_shows_new_top_level_destinations(mock_config):
    app = TldwCli()
    async with app.run_test() as pilot:
        labels = [str(button.label) for button in app.query(".nav-button")]
        assert labels == [
            "Chat", "Library", "Study", "Characters",
            "Models & Tools", "Automation / Feeds", "Settings"
        ]

@pytest.mark.asyncio
async def test_legacy_aliases_resolve_to_new_destinations():
    registry = ScreenRegistry()
    assert registry.get_screen_class("notes").__name__ == "LibraryScreen"
    assert registry.get_screen_class("coding").__name__ == "ChatScreen"
```

- [ ] **Step 2: Run the destination tests to verify they fail**

Run:

```bash
pytest Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py -q
```

Expected: FAIL because the new destination screens and alias registry are not implemented.

- [ ] **Step 3: Implement destination wrappers and route aliases**

```python
# tldw_chatbook/UI/Screens/library_screen.py
class LibraryScreen(BaseAppScreen):
    current_section = reactive("notes")

    def compose_content(self) -> ComposeResult:
        yield SectionSwitcher(["notes", "media", "search", "ingest", "chatbooks"], id="library-switcher")
        yield ContentSwitcher(initial="notes", id="library-content")
```

```python
# tldw_chatbook/navigation/screen_registry.py
self._screens.update({
    "chat": ChatScreen,
    "library": LibraryScreen,
    "study": StudyScreen,
    "characters": CharactersScreen,
    "models_tools": ModelsToolsScreen,
    "automation_feeds": AutomationFeedsScreen,
    "settings": SettingsScreen,
})
self._aliases.update({
    "coding": "chat",
    "notes": "library",
    "media": "library",
    "search": "library",
    "ingest": "library",
    "chatbooks": "library",
    "ccp": "characters",
    "conversation": "characters",
    "llm": "models_tools",
    "stts": "models_tools",
    "evals": "models_tools",
    "subscription": "automation_feeds",
    "subscriptions": "automation_feeds",
    "tools_settings": "settings",
    "customize": "settings",
    "logs": "settings",
    "stats": "settings",
})
```

- [ ] **Step 4: Run the destination and alias regression tests**

Run:

```bash
pytest Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py Tests/UI/test_tab_links_navigation.py -q
```

Expected: PASS with legacy aliases still accepted, but only the new destination set shown in primary navigation.

- [ ] **Step 5: Commit the new top-level IA**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/characters_screen.py tldw_chatbook/UI/Screens/models_tools_screen.py tldw_chatbook/UI/Screens/automation_feeds_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Widgets/section_switcher.py tldw_chatbook/navigation/screen_registry.py tldw_chatbook/Constants.py tldw_chatbook/UI/Navigation/main_navigation.py tldw_chatbook/UI/Tab_Links.py tldw_chatbook/UI/Tab_Bar.py Tests/UI/test_destination_shells.py Tests/UI/test_screen_navigation.py Tests/UI/test_tab_links_navigation.py
git commit -m "feat: replace legacy tab IA with destination shells"
```

### Task 3: Build The Library Destination Around Existing Surfaces

**Files:**
- Create: `tldw_chatbook/Widgets/library_sections.py`
- Create: `tldw_chatbook/Widgets/library_action_bar.py`
- Create: `tldw_chatbook/Widgets/Note_Widgets/notes_workspace.py`
- Create: `Tests/UI/test_library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tldw_chatbook/UI/Screens/media_screen.py`
- Modify: `tldw_chatbook/UI/Screens/media_ingest_screen.py`
- Modify: `tldw_chatbook/UI/Screens/search_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chatbooks_screen.py`

- [ ] **Step 1: Write the failing Library tests**

```python
@pytest.mark.asyncio
async def test_library_defaults_to_notes_section(mock_config):
    app = TldwCli()
    async with app.run_test() as pilot:
        app.post_message(NavigateToScreen("library"))
        await pilot.pause()
        assert app.screen.current_section == "notes"

@pytest.mark.asyncio
async def test_library_use_in_chat_action_only_appears_on_selection(mock_config):
    app = TldwCli()
    async with app.run_test() as pilot:
        app.post_message(NavigateToScreen("library"))
        await pilot.pause()
        action_bar = app.screen.query_one("#library-action-bar")
        assert action_bar.display is False
```

- [ ] **Step 2: Run the Library tests to verify they fail**

Run:

```bash
pytest Tests/UI/test_library_screen.py -q
```

Expected: FAIL because the Library content switcher and action bar do not exist yet.

- [ ] **Step 3: Wrap existing Notes/Media/Search/Ingest/Chatbooks surfaces inside Library**

```python
# tldw_chatbook/Widgets/library_sections.py
class NotesLibrarySection(Widget):
    def compose(self) -> ComposeResult:
        yield NotesWorkspace(self.app, id="library-notes-workspace")

class MediaLibrarySection(Widget):
    def compose(self) -> ComposeResult:
        yield MediaWindow(self.app, classes="window")
```

```python
# tldw_chatbook/Widgets/library_action_bar.py
class LibraryActionBar(Horizontal):
    def watch_selection_count(self, count: int) -> None:
        self.display = count > 0
```

Use thin adapters first. Do not rewrite Notes/Media/Search internals in this task.

If `NotesScreen` cannot be cleanly reused, extract its current `compose_content()` body into a reusable `NotesWorkspace` widget first, then mount that widget in both `NotesScreen` and `LibraryScreen`.

- [ ] **Step 4: Run focused Library and supporting-surface regressions**

Run:

```bash
pytest Tests/UI/test_library_screen.py Tests/UI/test_notes_screen.py Tests/UI/test_ingestion_ui_redesigned.py Tests/UI/test_media_window_v88_textual.py Tests/UI/test_search_rag_window.py -q
```

Expected: PASS, with Library switching sections without regressing the wrapped content surfaces.

- [ ] **Step 5: Commit the Library destination**

```bash
git add tldw_chatbook/Widgets/library_sections.py tldw_chatbook/Widgets/library_action_bar.py tldw_chatbook/Widgets/Note_Widgets/notes_workspace.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/notes_screen.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/search_screen.py tldw_chatbook/UI/Screens/chatbooks_screen.py Tests/UI/test_library_screen.py
git commit -m "feat: build unified library destination"
```

### Task 4: Make Chat The Default Agentic Console

**Files:**
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_context_bar.py`
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`
- Create: `Tests/UI/test_chat_agentic_console.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Modify: `tldw_chatbook/UI/Coding_Window.py`
- Modify: `tldw_chatbook/UI/Screens/coding_screen.py`
- Modify: `Tests/UI/test_chat_window_enhanced.py`

- [ ] **Step 1: Write the failing Chat-console tests**

```python
@pytest.mark.asyncio
async def test_chat_shows_workspace_repo_and_persona_context(mock_config):
    app = TldwCli()
    async with app.run_test() as pilot:
        bar = app.screen.query_one("ChatContextBar")
        assert bar.query_one("#chat-workspace-chip")
        assert bar.query_one("#chat-scope-chip")

@pytest.mark.asyncio
async def test_coding_alias_opens_chat_not_a_second_primary_surface(mock_config):
    app = TldwCli()
    async with app.run_test() as pilot:
        app.post_message(NavigateToScreen("coding"))
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
```

- [ ] **Step 2: Run the Chat-console tests to verify they fail**

Run:

```bash
pytest Tests/UI/test_chat_agentic_console.py -q
```

Expected: FAIL because Chat does not yet expose the context bar or absorb the coding route.

- [ ] **Step 3: Implement the Chat context bar and coding-panel migration**

```python
# tldw_chatbook/Widgets/Chat_Widgets/chat_context_bar.py
class ChatContextBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield Static(id="chat-workspace-chip")
        yield Static(id="chat-repo-chip")
        yield Static(id="chat-branch-chip")
        yield Static(id="chat-persona-chip")
```

```python
# tldw_chatbook/UI/Screens/coding_screen.py
class CodingScreen(BaseAppScreen):
    async def on_mount(self) -> None:
        self.app.post_message(NavigateToScreen("chat"))
        self.app.post_message(OpenChatPanel(panel_id="coding-tools"))
```

```python
# tldw_chatbook/UI/Chat_Window_Enhanced.py
def compose(self) -> ComposeResult:
    yield ChatContextBar(id="chat-context-bar")
    yield ChatTaskCardStack(id="chat-task-cards")
    ...
```

- [ ] **Step 4: Run the Chat regressions**

Run:

```bash
pytest Tests/UI/test_chat_agentic_console.py Tests/UI/test_chat_window_enhanced.py Tests/UI/test_send_stop_button.py -q
```

Expected: PASS, with Chat remaining conversation-first and Coding treated as a legacy alias into Chat tooling.

- [ ] **Step 5: Commit the Chat-console migration**

```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_context_bar.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/UI/Chat_Window_Enhanced.py tldw_chatbook/UI/Coding_Window.py tldw_chatbook/UI/Screens/coding_screen.py Tests/UI/test_chat_agentic_console.py Tests/UI/test_chat_window_enhanced.py
git commit -m "feat: make chat the primary agentic console"
```

### Task 5: Add Workspace Labels And Cross-Surface Handoffs

**Files:**
- Create: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `tldw_chatbook/state/workspace_state.py`
- Modify: `tldw_chatbook/UI/Navigation/navigation_messages.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
- Modify: `tldw_chatbook/Widgets/library_action_bar.py`

- [ ] **Step 1: Write the failing handoff tests**

```python
@pytest.mark.asyncio
async def test_library_item_can_be_sent_to_current_chat(mock_config):
    ...
    assert chat_screen.pending_handoff["source"] == "library"

@pytest.mark.asyncio
async def test_persona_handoff_updates_chat_scope_chip(mock_config):
    ...
    assert context_bar.query_one("#chat-persona-chip").renderable == "Research Persona"
```

- [ ] **Step 2: Run the handoff tests to verify they fail**

Run:

```bash
pytest Tests/UI/test_chat_first_handoffs.py -q
```

Expected: FAIL because the handoff messages and state transitions are not implemented.

- [ ] **Step 3: Implement explicit handoff messages and visible confirmation**

```python
# tldw_chatbook/UI/Navigation/navigation_messages.py
class UseInChat(Message):
    def __init__(self, source: str, payload: dict, target_session_id: str | None = None):
        super().__init__()
        self.source = source
        self.payload = payload
        self.target_session_id = target_session_id
```

```python
# tldw_chatbook/UI/Screens/chat_screen.py
@on(UseInChat)
def handle_use_in_chat(self, message: UseInChat) -> None:
    self.chat_state.pending_handoff = {
        "source": message.source,
        "payload": message.payload,
    }
```

Every handoff must show what context moved into Chat and preserve the active session when possible.

- [ ] **Step 4: Run handoff and scope regressions**

Run:

```bash
pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_library_screen.py Tests/UI/test_chat_agentic_console.py -q
```

Expected: PASS, with explicit scope labeling and visible handoff confirmation in Chat.

- [ ] **Step 5: Commit the handoff layer**

```bash
git add tldw_chatbook/state/workspace_state.py tldw_chatbook/UI/Navigation/navigation_messages.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/study_screen.py tldw_chatbook/UI/Screens/ccp_screen.py tldw_chatbook/Widgets/library_action_bar.py Tests/UI/test_chat_first_handoffs.py
git commit -m "feat: add chat-first context handoffs"
```

### Task 6: Add The Study Dashboard And Quiz Entry Flow

**Files:**
- Create: `tldw_chatbook/Widgets/Study/__init__.py`
- Create: `tldw_chatbook/Widgets/Study/study_dashboard.py`
- Create: `tldw_chatbook/Widgets/Study/quiz_session_widget.py`
- Create: `Tests/UI/test_study_dashboard.py`
- Modify: `tldw_chatbook/UI/Study_Window.py`
- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
- Modify: `Tests/UI/test_study_screen.py`

- [ ] **Step 1: Write the failing Study tests**

```python
@pytest.mark.asyncio
async def test_study_defaults_to_dashboard(mock_config):
    app = TldwCli()
    async with app.run_test() as pilot:
        app.post_message(NavigateToScreen("study"))
        await pilot.pause()
        assert app.screen.query_one("#study-dashboard")

@pytest.mark.asyncio
async def test_quiz_section_has_focused_entry_flow(mock_config):
    ...
    assert study_window.query_one("#quiz-session")
```

- [ ] **Step 2: Run the Study tests to verify they fail**

Run:

```bash
pytest Tests/UI/test_study_dashboard.py Tests/UI/test_study_screen.py -q
```

Expected: FAIL because Study still opens the old multi-tool sidebar surface without a dashboard or quiz flow.

- [ ] **Step 3: Extract the dashboard and quiz shell**

```python
# tldw_chatbook/Widgets/Study/study_dashboard.py
class StudyDashboard(Widget):
    def compose(self) -> ComposeResult:
        yield Static("Due today", id="study-due-today")
        yield Static("Recent decks", id="study-recent-decks")
        yield Button("Resume last session", id="study-resume-last")
```

```python
# tldw_chatbook/Widgets/Study/quiz_session_widget.py
class QuizSessionWidget(Widget):
    def compose(self) -> ComposeResult:
        yield Static(id="quiz-scope-summary")
        yield Button("Start quiz", id="quiz-start")
        yield Static(id="quiz-results")
```

- [ ] **Step 4: Run the Study regressions**

Run:

```bash
pytest Tests/UI/test_study_dashboard.py Tests/UI/test_study_screen.py Tests/UI/test_screen_navigation.py -q
```

Expected: PASS, with Study entering on the dashboard and exposing a coherent quiz entry/session path even if advanced quiz authoring remains deferred.

- [ ] **Step 5: Commit the Study modernization**

```bash
git add tldw_chatbook/Widgets/Study/__init__.py tldw_chatbook/Widgets/Study/study_dashboard.py tldw_chatbook/Widgets/Study/quiz_session_widget.py tldw_chatbook/UI/Study_Window.py tldw_chatbook/UI/Screens/study_screen.py Tests/UI/test_study_dashboard.py Tests/UI/test_study_screen.py
git commit -m "feat: add study dashboard and quiz flow"
```

### Task 7: Add Approvals And Task Continuity To Chat

**Files:**
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_resume_panel.py`
- Create: `Tests/UI/test_chat_approvals_and_resume.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`

- [ ] **Step 1: Write the failing approval/resume tests**

```python
@pytest.mark.asyncio
async def test_chat_renders_inline_approval_card_for_privileged_action(mock_config):
    ...
    assert chat.query_one("ChatApprovalCard")

@pytest.mark.asyncio
async def test_chat_resume_panel_shows_last_task_summary(mock_config):
    ...
    assert resume_panel.query_one("#resume-next-action")
```

- [ ] **Step 2: Run the approval/resume tests to verify they fail**

Run:

```bash
pytest Tests/UI/test_chat_approvals_and_resume.py -q
```

Expected: FAIL because approval cards and task resume state are not implemented yet.

- [ ] **Step 3: Persist task state and render inline approval/resume widgets**

```python
# tldw_chatbook/UI/Screens/chat_screen_state.py
@dataclass
class TaskResumeState:
    summary: str = ""
    last_step: str = ""
    pending_approval: dict | None = None
    diff_summary: str = ""
    next_action: str = ""
```

```python
# tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py
class ChatApprovalCard(Widget):
    def compose(self) -> ComposeResult:
        yield Static(id="approval-summary")
        yield Button("Allow once", id="approval-allow-once")
        yield Button("Deny", id="approval-deny")
        yield Button("Review details", id="approval-details")
```

- [ ] **Step 4: Run the Chat approval/resume regressions**

Run:

```bash
pytest Tests/UI/test_chat_approvals_and_resume.py Tests/UI/test_chat_agentic_console.py Tests/UI/test_chat_first_handoffs.py -q
```

Expected: PASS, with approvals and resume state visible inline in Chat instead of being hidden in logs.

- [ ] **Step 5: Commit the safety and continuity layer**

```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py tldw_chatbook/Widgets/Chat_Widgets/chat_resume_panel.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Chat_Window_Enhanced.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py Tests/UI/test_chat_approvals_and_resume.py
git commit -m "feat: add chat approvals and task resume"
```

### Task 8: Update Docs And Run Focused Regression Coverage

**Files:**
- Create: `Docs/Development/chat-first-shell-migration.md`
- Modify: `Docs/Development/navigation-architecture-analysis.md`
- Modify: `README.md`
- Test: `Tests/UI/test_app_shell.py`
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_library_screen.py`
- Test: `Tests/UI/test_chat_agentic_console.py`
- Test: `Tests/UI/test_chat_first_handoffs.py`
- Test: `Tests/UI/test_study_dashboard.py`
- Test: `Tests/UI/test_chat_approvals_and_resume.py`
- Test: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Document the shell migration and legacy-route policy**

Write `Docs/Development/chat-first-shell-migration.md` with these sections:

```md
# Chat-First Shell Migration
## New Top-Level IA
## Legacy Route Aliases
## Coding Destination Deprecation
## Workspace Scope Rules
## Testing Strategy
```

- [ ] **Step 2: Update navigation docs and README entry points**

Make sure `Docs/Development/navigation-architecture-analysis.md` and `README.md` describe the new destination IA and Chat-first product model instead of the old `chat/media/notes/coding/...` top-level list.

- [ ] **Step 3: Run the focused UI regression suite**

Run:

```bash
pytest Tests/UI/test_app_shell.py Tests/UI/test_destination_shells.py Tests/UI/test_library_screen.py Tests/UI/test_chat_agentic_console.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_study_dashboard.py Tests/UI/test_chat_approvals_and_resume.py Tests/UI/test_screen_navigation.py -q
```

Expected: PASS, with all new shell/destination/chat-first tests green.

- [ ] **Step 4: Run the supporting regression suite for wrapped surfaces**

Run:

```bash
pytest Tests/UI/test_notes_screen.py Tests/UI/test_ingestion_ui_redesigned.py Tests/UI/test_search_rag_window.py Tests/UI/test_media_window_v88_textual.py Tests/UI/test_chat_window_enhanced.py -q
```

Expected: PASS, confirming that the new top-level shell and wrappers did not break the existing embedded feature surfaces.

- [ ] **Step 5: Commit docs and regression updates**

```bash
git add Docs/Development/chat-first-shell-migration.md Docs/Development/navigation-architecture-analysis.md README.md
git commit -m "docs: document chat-first shell migration"
```

## Manual Review Checklist

- [ ] Chat is the default landing destination
- [ ] Only one top-level navigation system is visible
- [ ] `Coding` is not shown as a primary destination
- [ ] Library, Study, Characters, Models & Tools, Automation / Feeds, and Settings each show a local section switcher when applicable
- [ ] Every workspace/global scope transition is visibly labeled
- [ ] `Use in Chat` / `Use persona in current chat` handoffs preserve session context
- [ ] Chat reads as conversation-first with secondary panels collapsed by default
- [ ] Approvals, failures, progress, and resume state appear inline in Chat

## Execution Notes

- Start implementation in a dedicated worktree, not in the current dirty workspace.
- Keep commits aligned with the task boundaries above.
- Do not skip the regression commands after each task. The shell migration touches routing and composition code that will create broad regressions if left unchecked.
