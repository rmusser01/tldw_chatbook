# Chat-First Shell And Label Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the shared chat-first shell refinement: clearer global navigation labels, a combined Chat shell bar that exposes active session context, and tab-safe context syncing without expanding into a full workspace or study redesign.

**Architecture:** Keep route IDs stable and limit the UI change to the shared shell. Express the new IA through clustered ordering and copy changes in the top nav, then introduce one combined Chat shell bar that merges active-session context with the existing compact model controls. Drive the shell bar from the same saved session contract used for restore, and add explicit tab-container events so active-session context stays correct in both single-session and tabbed chat modes.

**Tech Stack:** Python 3.11+, Textual, existing `ChatSessionData` / `TabState` models, pytest

---

## Scope Check

This plan intentionally implements only the shared shell slice from the approved spec:

- navigation label and cluster cleanup
- combined Chat shell bar
- active-session context visibility and sync
- focused regression coverage

Approved Workspace, Flashcards, and Quiz layout ideas are follow-on verticals and should not be pulled into this implementation batch.

## File Map

- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
  Responsibility: Change user-facing labels and clustered ordering while preserving existing route IDs and the single-row shell.
- Modify: `Tests/UI/test_screen_navigation.py`
  Responsibility: Prove the new navigation copy and clustered ordering without breaking routed screen IDs.

- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py`
  Responsibility: Render backend, scope, assistant identity, and session title in one compact shell widget, accept both `TabState` and `ChatSessionData`, and host or wrap the existing compact model controls without regressing their behavior.
- Modify: `tldw_chatbook/Widgets/compact_model_bar.py`
  Responsibility: Support composition inside the combined shell bar without losing current provider/model/temperature sync behavior or existing widget/query seams that other chat code depends on.
- Create: `Tests/UI/test_chat_shell_bar.py`
  Responsibility: Verify fallback labels, formatting rules, and narrow-shell rendering expectations for the new shell bar in isolation.

- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
  Responsibility: Replace the standalone compact model bar mount point with the combined shell bar and preserve task-surface ordering.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  Responsibility: Derive shell-bar context from restored state and live active sessions, then sync the shell bar after mount, restore, and tab changes.
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
  Responsibility: Add narrowly scoped helpers for shell-bar display resolution if needed, without changing the persisted session contract.
- Modify: `Tests/UI/test_chat_approvals_and_resume.py`
  Responsibility: Verify the shell stack order and that restored/workspace-scoped sessions render explicit context immediately.
- Modify: `Tests/UI/test_chat_screen_state.py`
  Responsibility: Verify shell-context fallback resolution stays aligned with saved tab/session state assumptions.

- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py`
  Responsibility: Publish active-session changes on create, reuse, switch, and close so the shell bar can stay live in tabbed chat mode.
- Modify: `Tests/UI/test_chat_approvals_and_resume.py`
  Responsibility: Extend the chat harness to cover tab-driven shell-bar updates without regressing inline approval and resume behavior.

- Modify: `Docs/Development/chat-first-shell-migration.md`
  Responsibility: Record that the shell now uses a combined context-and-controls bar and that `Coding` is visually demoted but still routable.
- Modify: `Docs/Development/navigation-architecture-analysis.md`
  Responsibility: Record the implemented clustered-nav wording and shell-bar behavior so follow-on verticals inherit the right model.

## Task 1: Lock The Navigation Copy And Cluster Order

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write the failing navigation tests**

```python
@pytest.mark.asyncio
async def test_main_navigation_uses_library_copy_and_demotes_coding():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="chat")

    async with TestApp().run_test() as pilot:
        nav = pilot.app.query_one(MainNavigationBar)
        assert nav.query_one("#nav-ccp", Button).label == "Library"

        labels = [button.label for button in nav.query(".nav-button")]
        assert labels[:2] == ["Chat", "Chatbooks"]
        assert labels[-1] == "Coding"
```

- [ ] **Step 2: Run the focused navigation tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_screen_navigation.py -q`
Expected: FAIL because `ccp` still renders as `Conv/Char` and `Coding` still sits in the primary workspace cluster.

- [ ] **Step 3: Implement the minimal navigation copy and ordering change**

```python
NAV_GROUPS = [
    ("Work", [("chat", "Chat"), ("chatbooks", "Chatbooks")]),
    ("Content", [("notes", "Notes"), ("media", "Media"), ("ingest", "Ingest"), ("search", "Search"), ("subscriptions", "Subscriptions")]),
    ("Library", [("ccp", "Library"), ("study", "Study")]),
    ("AI", [("llm", "LLM"), ("stts", "S/TT/S"), ("evals", "Evals")]),
    ("System", [("tools_settings", "Settings"), ("customize", "Customize"), ("logs", "Logs"), ("stats", "Stats"), ("coding", "Coding")]),
]
```

- [ ] **Step 4: Re-run the focused navigation tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_screen_navigation.py -q`
Expected: PASS, with stable `#nav-<route>` IDs and updated user-facing labels.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Navigation/main_navigation.py Tests/UI/test_screen_navigation.py
git commit -m "feat: update chat-first navigation labels"
```

## Task 2: Build The Combined Chat Shell Bar With Explicit Fallbacks And Control Passthrough

**Files:**
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py`
- Modify: `tldw_chatbook/Widgets/compact_model_bar.py`
- Create: `Tests/UI/test_chat_shell_bar.py`

- [ ] **Step 1: Write the failing shell-bar widget tests**

```python
def test_shell_bar_renders_explicit_fallback_labels():
    context = ChatShellContext.from_session_data(None)

    assert context.backend_label == "Local"
    assert context.scope_label == "Global"
    assert context.assistant_label == "Assistant: General"
    assert context.session_label == "Session: New chat"


def test_shell_bar_accepts_tab_state_and_session_data():
    tab_state = TabState(
        tab_id="tab-1",
        title="Scoped Session",
        runtime_backend="server",
        assistant_kind="persona",
        assistant_id="study.coach",
        scope_type="workspace",
        workspace_id="workspace-42",
    )
    session = ChatSessionData(
        tab_id="tab-1",
        title="Scoped Session",
        runtime_backend="server",
        assistant_kind="persona",
        assistant_id="study.coach",
        scope_type="workspace",
        workspace_id="workspace-42",
    )
    assert ChatShellContext.from_tab_state(tab_state).assistant_label == "Persona: study.coach"
    assert ChatShellContext.from_session_data(session).assistant_label == "Persona: study.coach"


def test_shell_bar_prefers_live_resolved_labels_over_raw_ids():
    session = ChatSessionData(
        tab_id="tab-2",
        title="Scoped Session",
        runtime_backend="server",
        assistant_kind="persona",
        assistant_id="study.coach",
        scope_type="workspace",
        workspace_id="workspace-42",
    )
    resolver = ChatShellLabelResolver(
        workspace_name="Research Methods",
        persona_label="Study Coach",
    )
    context = ChatShellContext.from_session_data(session, resolver=resolver)

    assert context.scope_label == "Workspace: Research Methods"
    assert context.assistant_label == "Persona: Study Coach"


def test_shell_bar_truncation_keeps_backend_scope_and_assistant_before_title():
    context = ChatShellContext(
        backend_label="Server",
        scope_label="Workspace: Research Methods",
        assistant_label="Persona: Study Coach",
        session_label="Session: Very long generated study session title",
    )

    text = " | ".join(context.prioritized_segments(max_width=56))

    assert "Server" in text
    assert "Workspace:" in text
    assert "Persona:" in text
    assert "Very long generated study session title" not in text
```

```python
@pytest.mark.asyncio
async def test_shell_bar_preserves_embedded_compact_control_ids_and_sync(...):
    async with app.run_test() as pilot:
        shell_bar = pilot.app.query_one("#chat-shell-bar", ChatShellBar)
        shell_bar.sync_compact_controls(provider="openai", model="gpt-4.1", temperature="0.2")

        assert pilot.app.query_one("#compact-api-provider", Select).value == "openai"
        assert pilot.app.query_one("#compact-api-model", Select).value == "gpt-4.1"
        assert pilot.app.query_one("#compact-temperature", Input).value == "0.2"
```

```python
@pytest.mark.asyncio
async def test_shell_bar_controls_are_keyboard_reachable_without_focus_trap(...):
    async with app.run_test() as pilot:
        await pilot.press("tab")
        assert pilot.app.focused.id in {
            "compact-api-provider",
            "compact-api-model",
            "compact-temperature",
            "compact-sidebar-toggle",
        }
```

- [ ] **Step 2: Run the focused shell-bar tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_chat_shell_bar.py -q`
Expected: FAIL because the shell-bar widget and display-resolution helper do not exist yet.

- [ ] **Step 3: Implement the minimal shell-bar widget and context formatter**

```python
@dataclass
class ChatShellLabelResolver:
    workspace_name: Optional[str] = None
    persona_label: Optional[str] = None
    character_label: Optional[str] = None


@dataclass
class ChatShellContext:
    backend_label: str
    scope_label: str
    assistant_label: str
    session_label: str

    @classmethod
    def from_tab_state(cls, tab_state, resolver=None):
        return cls.from_session_data(tab_state, resolver=resolver)

    @classmethod
    def from_session_data(cls, session_data, resolver=None):
        backend = "Server" if getattr(session_data, "runtime_backend", "local") == "server" else "Local"
        scope_type = getattr(session_data, "scope_type", None) or "global"
        workspace_id = getattr(session_data, "workspace_id", None)
        workspace_name = getattr(resolver, "workspace_name", None) if resolver else None
        scope = f"Workspace: {workspace_name or workspace_id}" if scope_type == "workspace" and (workspace_name or workspace_id) else "Global"

        if getattr(session_data, "assistant_kind", None) == "character":
            character_name = getattr(resolver, "character_label", None) if resolver else None
            character_name = character_name or getattr(session_data, "character_name", None) or getattr(session_data, "character_id", None)
            assistant = f"Character: {character_name}" if character_name else "Assistant: General"
        elif getattr(session_data, "assistant_kind", None) == "persona":
            persona_label = getattr(resolver, "persona_label", None) if resolver else None
            persona_value = persona_label or getattr(session_data, "assistant_id", None)
            assistant = f"Persona: {persona_value}" if persona_value else "Assistant: General"
        else:
            assistant = "Assistant: General"

        title = getattr(session_data, "title", None) or "New chat"
        return cls(backend, scope, assistant, f"Session: {title}")

    def prioritized_segments(self, max_width: int) -> list[str]:
        primary = [self.backend_label, self.scope_label, self.assistant_label]
        session = self.session_label
        while len(" | ".join(primary + [session])) > max_width and len(session) > len("Session:"):
            session = session[:-1]
        if len(" | ".join(primary + [session])) <= max_width:
            return primary + [session]
        return primary
```

- [ ] **Step 4: Re-run the focused shell-bar tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_chat_shell_bar.py -q`
Expected: PASS, with a dedicated widget that can render explicit fallback metadata without depending on sidebars.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py tldw_chatbook/Widgets/compact_model_bar.py Tests/UI/test_chat_shell_bar.py
git commit -m "feat: add combined chat shell bar"
```

## Task 3: Mount The Shell Bar In Chat And Sync It From Restored State

**Files:**
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `Tests/UI/test_chat_approvals_and_resume.py`
- Modify: `Tests/UI/test_chat_screen_state.py`

- [ ] **Step 1: Write the failing chat-shell integration tests**

```python
@pytest.mark.asyncio
async def test_chat_window_mounts_shell_bar_between_task_surface_and_chat_content(...):
    async with app.run_test() as pilot:
        main_content = pilot.app.query_one("#chat-main-content", Container)
        task_surface = pilot.app.query_one("#chat-task-surface", Container)
        shell_bar = pilot.app.query_one("#chat-shell-bar", Container)
        chat_log = pilot.app.query_one("#chat-log", VerticalScroll)

        children = list(main_content.children)
        assert children.index(task_surface) < children.index(shell_bar) < children.index(chat_log)


def test_chat_screen_syncs_workspace_scoped_restore_into_shell_bar(mock_chat_host):
    screen = ChatScreen(mock_chat_host)
    shell_bar = Mock()
    screen.chat_window = Mock()
    screen.chat_window.get_shell_bar.return_value = shell_bar
    screen.chat_state.tabs = [
        TabState(
            tab_id="default",
            title="Scoped Session",
            runtime_backend="server",
            scope_type="workspace",
            workspace_id="workspace-42",
            assistant_kind="persona",
            assistant_id="study.coach",
            is_active=True,
        )
    ]
    screen.chat_state.active_tab_id = "default"

    screen.sync_shell_bar_from_state()

    shell_bar.sync_from_session.assert_called_once()
```

- [ ] **Step 2: Run the focused chat integration tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_chat_approvals_and_resume.py Tests/UI/test_chat_screen_state.py -q`
Expected: FAIL because Chat still mounts `CompactModelBar` directly, there is no combined shell-bar mount point, and `ChatScreen` has no shell-bar sync seam.

- [ ] **Step 3: Implement the minimal chat mount and restore syncing**

```python
with Container(id="chat-main-content"):
    yield ChatTaskCards(id="chat-task-surface")
    yield ChatShellBar(self.app_instance, id="chat-shell-bar", compact_bar_id="compact-model-bar")
```

```python
def sync_shell_bar_from_state(self) -> None:
    shell_bar = self.chat_window.get_shell_bar() if self.chat_window else None
    if shell_bar is None:
        return
    active_tab = self.chat_state.get_active_tab()
    shell_bar.sync_from_tab_state(active_tab)
```

Preserve the existing compact-bar query seam during this task.
If other chat code still queries `#compact-model-bar`, keep that ID alive inside the combined shell bar or update the callers in the same task.

- [ ] **Step 4: Re-run the focused chat integration tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_chat_approvals_and_resume.py Tests/UI/test_chat_screen_state.py -q`
Expected: PASS, with the shell bar mounted in the right place and restored state visible immediately.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Chat_Window_Enhanced.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/chat_screen_state.py Tests/UI/test_chat_approvals_and_resume.py Tests/UI/test_chat_screen_state.py
git commit -m "feat: sync chat shell bar from restored state"
```

## Task 4: Keep The Shell Bar Correct In Tabbed Chat

**Files:**
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_chat_approvals_and_resume.py`

- [ ] **Step 1: Write the failing tab-sync tests**

```python
@pytest.mark.asyncio
async def test_tab_switch_updates_shell_bar_when_tabs_enabled(...):
    async with app.run_test() as pilot:
        tab_container = pilot.app.query_one(ChatTabContainer)
        shell_bar = pilot.app.query_one("#chat-shell-bar", ChatShellBar)

        first_id = tab_container.active_session_id
        second_id = await tab_container.create_new_tab(
            session_data=ChatSessionData(
                tab_id="ignored",
                title="Persona Session",
                runtime_backend="server",
                assistant_kind="persona",
                assistant_id="study.coach",
                scope_type="workspace",
                workspace_id="workspace-42",
            )
        )
        await tab_container.switch_to_tab_async(second_id)

        assert "Persona: study.coach" in shell_bar.render_to_text()
        assert "Workspace: workspace-42" in shell_bar.render_to_text()
        assert second_id != first_id


@pytest.mark.asyncio
async def test_tab_reuse_and_close_republish_shell_context(...):
    async with app.run_test() as pilot:
        tab_container = pilot.app.query_one(ChatTabContainer)
        original_id = tab_container.active_session_id

        reused_id = await tab_container.create_new_tab(
            session_data=ChatSessionData(
                tab_id="ignored",
                title="Existing Conversation",
                conversation_id="conv-1",
                runtime_backend="server",
            )
        )
        assert reused_id == original_id or reused_id in tab_container.sessions

        await tab_container.close_tab(tab_container.active_session_id)

        shell_bar = pilot.app.query_one("#chat-shell-bar", ChatShellBar)
        assert shell_bar is not None
```

- [ ] **Step 2: Run the tabbed-chat tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_chat_approvals_and_resume.py -q`
Expected: FAIL because the tab container does not publish active-session changes consistently and the shell bar cannot respond to create, reuse, switch, and close lifecycle updates.

- [ ] **Step 3: Implement explicit active-session change notifications**

```python
class ActiveSessionChanged(Message):
    def __init__(self, session_data: Optional[ChatSessionData]):
        super().__init__()
        self.session_data = session_data

def _publish_active_session(self) -> None:
    session = self.get_active_session()
    payload = session.session_data if session is not None else None
    self.post_message(self.ActiveSessionChanged(payload))
```

```python
async def switch_to_tab_async(self, tab_id: str) -> None:
    ...
    self.active_session_id = tab_id
    ...
    self._publish_active_session()
```

Also publish the active session after:

- `create_new_tab()` returns an existing reusable session
- a newly created tab becomes active
- `close_tab()` selects the next active tab or clears the active session

- [ ] **Step 4: Re-run the tabbed-chat tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_chat_approvals_and_resume.py -q`
Expected: PASS, with tab create, reuse, switch, and close all driving shell-bar context updates.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_chat_approvals_and_resume.py
git commit -m "feat: keep chat shell bar in sync across tabs"
```

## Task 5: Update Migration Docs And Run The Shell Regression Sweep

**Files:**
- Modify: `Docs/Development/chat-first-shell-migration.md`
- Modify: `Docs/Development/navigation-architecture-analysis.md`
- Modify: `docs/superpowers/specs/2026-04-21-chat-first-shell-label-cleanup-design.md`

- [ ] **Step 1: Update the docs to match the shipped shell**

```md
- The Chat shell now uses a combined shell bar for context plus quick model/runtime controls.
- `Coding` remains routable for compatibility but is visually demoted from the primary work cluster.
- Active chat context is now visible in both restored and tabbed sessions.
```

- [ ] **Step 2: Run the focused shell regression sweep**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI/test_screen_navigation.py Tests/UI/test_chat_shell_bar.py Tests/UI/test_chat_approvals_and_resume.py Tests/UI/test_chat_screen_state.py -q`
Expected: PASS with no navigation regressions and explicit shell-context coverage.

- [ ] **Step 3: Run the broader UI regression sweep**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest Tests/UI -x -q`
Expected: PASS, or only pre-existing skips/warnings unrelated to the shell slice.

- [ ] **Step 4: Commit**

```bash
git add Docs/Development/chat-first-shell-migration.md Docs/Development/navigation-architecture-analysis.md docs/superpowers/specs/2026-04-21-chat-first-shell-label-cleanup-design.md
git commit -m "docs: record chat-first shell implementation"
```

## Deferred Follow-On Verticals

Do not pull these into this implementation plan. They are the next UX slices after the shared shell lands cleanly:

- Workspace hub layout: promote workspace artifacts, pinned context, and launch actions into a terminal-first hub
- Flashcards practice layout: split review mode from deck-management mode
- Quiz attempt layout: split attempt mode from quiz authoring and add direct recovery into Chat
- Cross-module shell consistency: reuse the shell-context pattern in Study and future workspace-scoped shells
