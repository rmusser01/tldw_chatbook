# Gate 1.5 Console Internals Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gate 1 Console shell's embedded legacy `ChatWindowEnhanced` presentation with Console-native components while preserving existing chat behavior through explicit compatibility seams.

**Architecture:** Treat `ChatScreen` as the Console orchestrator and move display state into pure `tldw_chatbook/Chat/` contracts before moving UI into focused `tldw_chatbook/Widgets/Console/` widgets. Reuse existing chat services, `ChatSession`, `ChatTabContainer`, event handlers, and Chatbook artifact seams where safe; keep `ChatWindowEnhanced` as a backward-compatible direct widget, not as the Console's full embedded UI. Each implementation PR must preserve current route IDs and existing chat regressions while shrinking the legacy surface area.

**Tech Stack:** Python 3.12, Textual, pytest, Backlog.md, existing `ChatScreen`, `ChatSession`, `ChatTabContainer`, `CompactModelBar`, `ChatShellBar`, `ConsoleLiveWorkLaunch`, and Chatbook artifact services.

---

## Source Of Truth

- Binding design gate: `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
- Binding layout contract: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Current Gate 1 evidence: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md`
- Current roadmap: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Parent backlog gate: `TASK-10.6`
- Child execution slices: `TASK-10.6.1` through `TASK-10.6.5`

Gate 1.5 is required because Gate 1 intentionally wrapped the existing chat surface for compatibility. That wrapper is not the end state: the Console must become one coherent agentic workbench with native staged context, transcript, composer, provider/model controls, RAG/source state, tools, approvals, artifact controls, and recoverable runtime/provider states.

## Scope

Included:

- Pure Console display-state contracts for provider/model readiness, source/RAG context, tool readiness, approvals, artifacts, live-work provenance, and recovery copy.
- Console-owned widgets for the control bar, staged context tray, transcript/event region, composer/action row, and run inspector.
- Migration away from mounting the full `ChatWindowEnhanced` chrome inside `#console-transcript-region`.
- Compatibility guardrails for basic chat, tab/session state, provider/model selection, streaming/non-streaming fallback, source handoffs, Chatbook artifact save/reopen, tool-call visibility, and persona/character attachment.
- QA evidence and roadmap/backlog tracking for Gate 1.5 closeout.

Excluded:

- Full Library-native Search/RAG implementation. That is Gate 1.6.
- MCP server/tool tree redesign, ACP runtime UI, Personas/Skills destination rewrites, or broad Gate 2 screen adaptation.
- Removing `ChatWindowEnhanced` from the codebase. It can remain as a compatibility widget for legacy/direct usage until replacement is proven.
- Changing route IDs, command palette routes, or existing navigation ownership.

## File Structure

### Create

- `tldw_chatbook/Chat/console_display_state.py`
  - Pure dataclasses and builders for Console chrome labels, staged context rows, inspector rows, and blocked/recovery states.
- `tldw_chatbook/Widgets/Console/__init__.py`
  - Package export seam for Console-native widgets.
- `tldw_chatbook/Widgets/Console/console_control_bar.py`
  - Provider/model/persona/source/RAG status controls visible outside the transcript region.
- `tldw_chatbook/Widgets/Console/console_staged_context.py`
  - Staged context tray for handoff/live-work/search evidence provenance.
- `tldw_chatbook/Widgets/Console/console_run_inspector.py`
  - Run/tool/approval/artifact/recovery inspector.
- `tldw_chatbook/Widgets/Console/console_session_surface.py`
  - Native transcript/session host that reuses `ChatTabContainer` and/or `ChatSession` without the full `ChatWindowEnhanced` chrome.
- `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  - Native composer/action row or compatibility adapter around the active session input/send/stop controls.
- `Tests/UI/test_console_internals_decomposition.py`
  - Gate 1.5 mounted UI regressions and compatibility tests.
- `Tests/Chat/test_console_display_state.py`
  - Pure unit tests for Console display-state contracts.
- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-5-console-internals-decomposition.md`
  - QA evidence for the final closeout task.

### Modify

- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Replace inline Console shell rendering with Console-native widgets and orchestration.
- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
  - Add only compatibility seams needed to avoid duplicate chrome, or leave it untouched if Console can reuse lower-level widgets directly.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py`
  - Only if needed: expose active-session and session creation seams without changing behavior.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_session.py`
  - Only if needed: expose input/send/stop/artifact action hooks for the native Console composer.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py`
  - Only if needed: move context-label logic into the Console control bar while preserving existing tests.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add source TCSS for Console-native regions. Do not edit generated CSS directly.
- `tldw_chatbook/css/core/_variables.tcss`
  - Add named sizing variables if fixed Console dimensions are required.
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate with `tldw_chatbook/css/build_css.py` after source TCSS changes.
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Track Gate 1.5 task hierarchy and verification state.
- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
  - Link Gate 1.5 QA evidence during closeout.
- `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
  - Add parent progress notes only during closeout.
- `backlog/tasks/task-10.6*.md`
  - Update task status, plans, AC checkboxes, and implementation notes as each slice completes.

### Read Before Editing

- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- `tldw_chatbook/Chat/console_live_work.py`
- `tldw_chatbook/Chat/chat_handoff_models.py`
- `tldw_chatbook/Chat/chat_models.py`
- `tldw_chatbook/Widgets/Chat_Widgets/chat_session.py`
- `tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py`
- `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py`
- `tldw_chatbook/Widgets/compact_model_bar.py`
- `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
- `Tests/UI/test_chat_first_handoffs.py`
- `Tests/UI/test_chat_shell_bar.py`
- `Tests/UI/test_chat_tab_container.py`
- `Tests/UI/test_chat_window_enhanced.py`
- `Tests/UI/test_console_live_work_handoffs.py`

Use the repo virtualenv from the repository root, or activate the virtualenv and use `PY=python`:

```bash
PY=.venv/bin/python
```

## Risk Controls

- Do not remove `ChatWindowEnhanced` until focused compatibility tests cover the replacement path.
- Do not duplicate provider/model state in two independent places. The Console-native control bar must synchronize through existing handlers or a single adapter seam.
- Do not move Library Search/RAG behavior into Console. Console can invoke/use RAG, but Gate 1.6 owns the Library-native retrieval workflow.
- Do not edit `tldw_chatbook/css/tldw_cli_modular.tcss` manually. Edit source TCSS, run `build_css.py`, then commit the regenerated bundle.
- Avoid exact paragraph assertions in UI tests. Assert selectors, enabled/disabled state, target payloads, and key durable labels.
- Keep every slice green before continuing. If a compatibility test exposes legacy coupling, add an adapter seam rather than widening `ChatScreen`.

---

### Task 1: Gate 1.5.1 Console Display-State Contracts

Backlog: `TASK-10.6.1`

**Files:**
- Create: `Tests/Chat/test_console_display_state.py`
- Create: `Tests/UI/test_console_internals_decomposition.py`
- Create: `tldw_chatbook/Chat/console_display_state.py`
- Modify: `tldw_chatbook/Chat/console_live_work.py` only if existing launch/readiness state should be adapted into display rows
- Test: `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`

- [ ] **Step 1: Run current Console baseline**

Run:

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py --tb=short
```

Expected: pass with existing warnings only. If this fails, stop and investigate before adding Gate 1.5 tests.

- [ ] **Step 2: Add pure display-state red tests**

Create `Tests/Chat/test_console_display_state.py`:

```python
from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleInspectorState,
    ConsoleStagedContextState,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch


def test_console_control_state_exposes_provider_model_and_context_labels():
    state = ConsoleControlState.from_values(
        provider="OpenAI",
        model="gpt-5.5",
        persona="Researcher",
        rag_enabled=True,
        staged_source_count=3,
        tool_count=4,
        approval_count=1,
    )

    assert state.provider_label == "Provider: OpenAI"
    assert state.model_label == "Model: gpt-5.5"
    assert state.persona_label == "Persona: Researcher"
    assert state.rag_label == "RAG: on"
    assert state.sources_label == "Sources: 3 staged"
    assert state.tools_label == "Tools: 4 ready"
    assert state.approvals_label == "Approvals: 1 pending"


def test_console_staged_context_state_preserves_live_work_payload_provenance():
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Transformer notes",
        status="ready",
        recovery="Review citations before sending.",
        payload={"source_id": "note-1", "citation_count": 2},
    )

    state = ConsoleStagedContextState.from_live_work(launch)

    assert state.heading == "Staged Context"
    assert "Transformer notes" in state.summary
    assert any(row.label == "source_id" and row.value == "note-1" for row in state.rows)
    assert state.recovery == "Review citations before sending."


def test_console_inspector_state_combines_readiness_artifact_and_recovery_rows():
    state = ConsoleInspectorState.from_values(
        live_work_title="Daily papers",
        provider_ready=False,
        provider_recovery="Configure a provider before sending.",
        rag_status="missing index",
        artifact_status="save available after response",
        approval_count=0,
    )

    text = state.to_plain_text()
    assert "Daily papers" in text
    assert "Provider: blocked" in text
    assert "Configure a provider before sending." in text
    assert "RAG: missing index" in text
    assert "Artifacts: save available after response" in text
```

Expected before implementation: import failure for `tldw_chatbook.Chat.console_display_state`.

- [ ] **Step 3: Add mounted legacy-chrome guardrail red test**

Create `Tests/UI/test_console_internals_decomposition.py` with helpers imported from the Gate 1 test:

```python
import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)


@pytest.mark.asyncio
async def test_console_gate15_does_not_mount_full_legacy_chat_window_chrome():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert console.query_one("#console-control-bar")
        assert console.query_one("#console-session-surface")
        assert console.query_one("#console-native-composer")

        assert not console.query("#chat-enhanced-sidebar")
        assert not console.query("#toggle-chat-left-sidebar")
        assert not console.query("#chat-main-content")


@pytest.mark.asyncio
async def test_console_gate15_keeps_existing_chat_send_control_reachable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        text = _visible_text(console)
        assert "Send" in text
        send_controls = [
            button
            for button in console.query(Button)
            if (button.id or "").startswith("send-stop-chat")
            or button.has_class("console-send-button")
        ]
        assert send_controls
```

Expected before implementation: fail because current Console still mounts the full `ChatWindowEnhanced` chrome and does not expose the new native selectors.

- [ ] **Step 4: Implement minimal pure display-state contracts**

Create `tldw_chatbook/Chat/console_display_state.py` with frozen dataclasses and no Textual imports:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch


def _clean(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


@dataclass(frozen=True)
class ConsoleDisplayRow:
    label: str
    value: str
    status: str = "ready"
    recovery: str = ""

    @property
    def text(self) -> str:
        suffix = f" - {self.recovery}" if self.recovery else ""
        return f"{self.label}: {self.value}{suffix}"


@dataclass(frozen=True)
class ConsoleControlState:
    provider_label: str
    model_label: str
    persona_label: str
    rag_label: str
    sources_label: str
    tools_label: str
    approvals_label: str

    @classmethod
    def from_values(
        cls,
        *,
        provider: Any = None,
        model: Any = None,
        persona: Any = None,
        rag_enabled: bool = False,
        staged_source_count: int = 0,
        tool_count: int = 0,
        approval_count: int = 0,
    ) -> "ConsoleControlState":
        return cls(
            provider_label=f"Provider: {_clean(provider, 'not selected')}",
            model_label=f"Model: {_clean(model, 'not selected')}",
            persona_label=f"Persona: {_clean(persona, 'Default')}",
            rag_label=f"RAG: {'on' if rag_enabled else 'off'}",
            sources_label=f"Sources: {staged_source_count} staged",
            tools_label=f"Tools: {tool_count} ready",
            approvals_label=f"Approvals: {approval_count} pending",
        )
```

Add `ConsoleStagedContextState` and `ConsoleInspectorState` in the same file. Keep all functions pure and easy to test.

- [ ] **Step 5: Run Task 1 tests green**

Run:

```bash
$PY -m pytest -q Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py::test_console_gate15_does_not_mount_full_legacy_chat_window_chrome Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions --tb=short
```

Expected after full Task 1 implementation: pure display-state tests pass. The mounted legacy-chrome guardrail may remain marked `xfail(strict=True)` only if Task 1 deliberately records the red state for Task 2; do not mark the backlog AC done until either it is red in TDD evidence or the minimal native selectors exist.

- [ ] **Step 6: Update task and commit**

Update `backlog/tasks/task-10.6.1 - Gate-1.5.1-Console-native-display-state-contracts.md` with the implementation plan, checked ACs that are truly satisfied, and implementation notes.

```bash
git add Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py tldw_chatbook/Chat/console_display_state.py "backlog/tasks/task-10.6.1 - Gate-1.5.1-Console-native-display-state-contracts.md"
git commit -m "Add Console native display-state contracts"
```

---

### Task 2: Gate 1.5.2 Native Controls And Staged Context

Backlog: `TASK-10.6.2`

**Files:**
- Create: `tldw_chatbook/Widgets/Console/__init__.py`
- Create: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Create: `tldw_chatbook/Widgets/Console/console_staged_context.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_chat_shell_bar.py`
- Test: `Tests/UI/test_chat_first_handoffs.py`

- [ ] **Step 1: Add failing control/context mounted test**

Extend `Tests/UI/test_console_internals_decomposition.py`:

```python
@pytest.mark.asyncio
async def test_console_native_control_bar_and_staged_context_reflect_pending_handoff():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "Library Search/RAG",
        "title": "Transformer notes",
        "status": "ready",
        "recovery": "Review citations before sending.",
        "payload": {"source_id": "note-1", "citation_count": 2},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-bar")

        text = _visible_text(console)
        assert "Provider:" in text
        assert "Model:" in text
        assert "Persona:" in text
        assert "RAG:" in text
        assert "Sources: 1 staged" in text
        assert "Transformer notes" in text
        assert "citation_count: 2" in text
        assert "Review citations before sending." in text
```

Expected before implementation: fails because `#console-control-bar` is missing or context is not native.

- [ ] **Step 2: Create `ConsoleControlBar`**

Implement `tldw_chatbook/Widgets/Console/console_control_bar.py`:

```python
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


class ConsoleControlBar(Horizontal):
    def __init__(self, state: ConsoleControlState, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        yield Static(self.state.provider_label, id="console-provider-label")
        yield Static(self.state.model_label, id="console-model-label")
        yield Static(self.state.persona_label, id="console-persona-label")
        yield Static(self.state.rag_label, id="console-rag-label")
        yield Static(self.state.sources_label, id="console-sources-label")
        yield Static(self.state.tools_label, id="console-tools-label")
        yield Static(self.state.approvals_label, id="console-approvals-label")
        yield CompactModelBar(self.app_instance, id="console-compact-model-bar")
```

Use `id="console-control-bar"` when mounted from `ChatScreen`.

- [ ] **Step 3: Create `ConsoleStagedContextTray`**

Implement `tldw_chatbook/Widgets/Console/console_staged_context.py`:

```python
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState


class ConsoleStagedContextTray(Vertical):
    def __init__(self, state: ConsoleStagedContextState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(self.state.heading, id="console-staged-context-title", classes="destination-section")
        yield Static(self.state.summary, id="console-staged-context-summary")
        for index, row in enumerate(self.state.rows):
            yield Static(row.text, id=f"console-staged-context-row-{index}")
        if self.state.recovery:
            yield Static(self.state.recovery, id="console-staged-context-recovery", classes="destination-recovery")
```

- [ ] **Step 4: Wire controls and staged context into `ChatScreen`**

In `ChatScreen.compose_content()`:

- Build `pending_launch = self._consume_pending_console_launch()`.
- Build `ConsoleControlState` from current provider/model defaults plus staged count.
- Build `ConsoleStagedContextState.from_live_work(pending_launch)` or `.empty()`.
- Mount `ConsoleControlBar(..., id="console-control-bar", classes="ds-panel")` in place of the plain `#console-mode-bar` copy.
- Mount `ConsoleStagedContextTray(..., id="console-staged-context-tray", classes="console-region")`.

Preserve `#console-mode-bar` as either the same control bar container or a compatibility alias if existing tests require it.

- [ ] **Step 5: Add TCSS source rules and regenerate CSS**

Edit only source TCSS:

```css
#console-control-bar {
    height: auto;
    min-height: 3;
}

.console-staged-context-row,
.console-control-label {
    width: auto;
    min-width: 0;
}
```

Then run:

```bash
$PY tldw_chatbook/css/build_css.py
```

Expected: generated bundle timestamp updates. If the build warns about pre-existing missing optional feature TCSS, record the warning in implementation notes only if the build exits 0.

- [ ] **Step 6: Run focused verification**

Run:

```bash
$PY -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_chat_shell_bar.py Tests/UI/test_chat_first_handoffs.py --tb=short
git diff --check
```

Expected: all selected tests pass with known warnings only.

- [ ] **Step 7: Update task and commit**

```bash
git add tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/Widgets/Console/console_control_bar.py tldw_chatbook/Widgets/Console/console_staged_context.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_internals_decomposition.py "backlog/tasks/task-10.6.2 - Gate-1.5.2-Console-native-controls-and-staged-context.md"
git commit -m "Add Console native controls and staged context"
```

---

### Task 3: Gate 1.5.3 Native Transcript And Composer Surface

Backlog: `TASK-10.6.3`

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Create: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Optional Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py`
- Optional Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_session.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_chat_tab_container.py`
- Test: `Tests/UI/test_chat_first_handoffs.py`
- Test: `Tests/UI/test_chat_window_enhanced.py`

- [ ] **Step 1: Add failing native session/composer test**

Extend `Tests/UI/test_console_internals_decomposition.py`:

```python
@pytest.mark.asyncio
async def test_console_native_session_surface_uses_chat_sessions_without_full_chat_window():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-session-surface")

        assert console.query_one("#console-session-surface")
        assert console.query_one("#console-native-composer")
        assert console.query("#chat-sessions-container") or console.query(".chat-session")
        assert not console.query("#chat-window")
        assert not console.query("#chat-main-content")
        assert not console.query("#chat-enhanced-sidebar")
```

Expected before implementation: fails because the full `#chat-window` is still mounted.

- [ ] **Step 2: Create `ConsoleSessionSurface`**

Implement `tldw_chatbook/Widgets/Console/console_session_surface.py`:

```python
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


class ConsoleSessionSurface(Vertical):
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.tab_container: ChatTabContainer | None = None

    def compose(self) -> ComposeResult:
        yield Static("Transcript / Event Stream", id="console-transcript-title", classes="destination-section")
        yield ChatTaskCards(id="console-task-surface")
        self.tab_container = ChatTabContainer(self.app_instance, id="console-chat-tabs")
        self.tab_container.enhanced_mode = True
        yield self.tab_container
```

If always using `ChatTabContainer` creates unacceptable behavior drift when tabs are disabled, add a `ConsoleSingleSessionSurface` adapter instead of changing user-facing behavior silently.

- [ ] **Step 3: Create `ConsoleComposerBar`**

Start as a visible adapter that points at the active session composer rather than duplicating send behavior:

```python
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static


class ConsoleComposerBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield Static("Composer", id="console-composer-title", classes="destination-section")
        yield Static("Use the active session input. Send/stop remains wired through chat handlers.", id="console-composer-status")
        yield Button("Save Chatbook", id="console-save-chatbook", classes="destination-action-button")
```

Only wire `Save Chatbook` if an existing artifact save target is available. Otherwise give it a disabled state and explicit recovery.

- [ ] **Step 4: Replace `ChatWindowEnhanced` in `ChatScreen`**

Update `ChatScreen`:

- Remove `yield self._ensure_chat_window()` from `compose_content()`.
- Mount `ConsoleSessionSurface(..., id="console-session-surface", classes="console-region")` inside `#console-transcript-region`.
- Mount `ConsoleComposerBar(id="console-native-composer", classes="console-region ds-panel")`.
- Update `_get_tab_container()` to first query `#console-chat-tabs`, then fall back to the legacy `self.chat_window` path for compatibility.
- Update `on_mount()`, `save_state()`, `restore_state()`, and `_perform_state_restoration()` checks so they do not require `self.chat_window` when the native surface is present.

- [ ] **Step 5: Preserve handoff and state behavior**

Run and fix only regressions caused by the migration:

```bash
$PY -m pytest -q Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_tab_container.py Tests/UI/test_chat_screen_state.py --tb=short
```

Expected: pass. If handoff creation depends on `ChatWindowEnhanced`, move the dependency behind `_get_tab_container()` instead of remounting the legacy widget.

- [ ] **Step 6: Run focused Gate 1.5 verification**

Run:

```bash
$PY -m pytest -q Tests/UI/test_console_internals_decomposition.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions Tests/UI/test_chat_window_enhanced.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_tab_container.py --tb=short
git diff --check
```

Expected: all selected tests pass with known warnings only. `ChatWindowEnhanced` direct tests should still pass because the widget remains available for legacy/direct harness usage.

- [ ] **Step 7: Update task and commit**

```bash
git add tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Chat_Widgets/chat_tab_container.py tldw_chatbook/Widgets/Chat_Widgets/chat_session.py Tests/UI/test_console_internals_decomposition.py "backlog/tasks/task-10.6.3 - Gate-1.5.3-Console-native-transcript-and-composer-surface.md"
git commit -m "Move Console transcript and composer to native surface"
```

---

### Task 4: Gate 1.5.4 Run Inspector, Approvals, Tools, RAG, And Artifacts

Backlog: `TASK-10.6.4`

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/Chat/test_console_display_state.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/UI/test_chat_approvals_and_resume.py`

- [ ] **Step 1: Add failing inspector state tests**

Extend `Tests/Chat/test_console_display_state.py`:

```python
def test_console_inspector_rows_include_tools_approvals_rag_and_artifacts():
    state = ConsoleInspectorState.from_values(
        live_work_title="Grounded answer",
        provider_ready=True,
        rag_status="2 cited chunks",
        tool_status="calculator ready",
        artifact_status="Chatbook save available",
        approval_count=1,
        approval_recovery="Review requested file write.",
    )

    text = state.to_plain_text()
    assert "Tools: calculator ready" in text
    assert "Approvals: 1 pending" in text
    assert "Review requested file write." in text
    assert "RAG: 2 cited chunks" in text
    assert "Artifacts: Chatbook save available" in text
```

- [ ] **Step 2: Add mounted blocked/recovery tests**

Extend `Tests/UI/test_console_internals_decomposition.py` with scenarios for:

- provider blocked.
- RAG/source index unavailable.
- approval pending.
- Chatbook save unavailable before assistant response.

Assert durable labels and disabled reasons, not exact paragraphs.

- [ ] **Step 3: Create `ConsoleRunInspector`**

Implement `tldw_chatbook/Widgets/Console/console_run_inspector.py`:

```python
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import ConsoleInspectorState


class ConsoleRunInspector(Vertical):
    def __init__(self, state: ConsoleInspectorState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static("Run Inspector", id="console-run-inspector-title", classes="destination-section")
        for index, row in enumerate(self.state.rows):
            yield Static(row.text, id=f"console-run-inspector-row-{index}")
        yield Button(
            "Review Approval",
            id="console-review-approval",
            disabled=not self.state.has_pending_approval,
            classes="destination-action-button",
        )
        yield Button(
            "Save Chatbook",
            id="console-inspector-save-chatbook",
            disabled=not self.state.can_save_chatbook,
            classes="destination-action-button",
        )
```

- [ ] **Step 4: Wire inspector state from current Console sources**

In `ChatScreen`, build `ConsoleInspectorState` from:

- pending `ConsoleLiveWorkLaunch`.
- provider readiness from `Chat.provider_readiness`.
- staged context count from pending handoff/live work.
- initial tool/artifact/approval placeholders from existing app/session state where available.

If a value is not currently available, show an honest unavailable row with owner/next action rather than a fake ready state.

- [ ] **Step 5: Wire action handlers conservatively**

Add handlers in `ChatScreen` only when an existing app method or widget action is available:

- `#console-review-approval`: route to existing approval/resume surface if mounted; otherwise notify with recovery.
- `#console-inspector-save-chatbook`: route to existing Chatbook save action only when an assistant response target exists; otherwise disabled with reason.

Do not create a new persistence path in this slice.

- [ ] **Step 6: Run focused verification**

Run:

```bash
$PY -m pytest -q Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_approvals_and_resume.py --tb=short
git diff --check
```

Expected: all selected tests pass with known warnings only.

- [ ] **Step 7: Update task and commit**

```bash
git add tldw_chatbook/Widgets/Console/console_run_inspector.py tldw_chatbook/Chat/console_display_state.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py "backlog/tasks/task-10.6.4 - Gate-1.5.4-Console-run-inspector-approvals-tools-and-artifacts.md"
git commit -m "Add Console native run inspector"
```

---

### Task 5: Gate 1.5.5 QA Closeout And Tracking

Backlog: `TASK-10.6.5`

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-5-console-internals-decomposition.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
- Modify: `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- Modify: `backlog/tasks/task-10.6 - Product-Maturity-Phase-3.6-Gate-1.5-Console-Internals-Decomposition.md`
- Modify: `backlog/tasks/task-10.6.5 - Gate-1.5.5-Console-internals-QA-closeout.md`
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Add evidence tracking test**

Add to `Tests/UI/test_console_internals_decomposition.py`:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE15_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/"
    "2026-05-07-gate-1-5-console-internals-decomposition.md"
)
ROADMAP = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
TASK_10_6 = Path(
    "backlog/tasks/task-10.6 - Product-Maturity-Phase-3.6-Gate-1.5-Console-Internals-Decomposition.md"
)


def _repo_text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_gate15_console_internals_evidence_is_tracked():
    evidence = _repo_text(GATE15_EVIDENCE)
    roadmap = _repo_text(ROADMAP)
    readme = _repo_text(PHASE_3_README)
    task = _repo_text(TASK_10_6)

    for heading in ("## Scope", "## Walkthrough", "## Verification", "## Residual Risk", "## Exit Decision"):
        assert heading in evidence
    for selector in ("#console-control-bar", "#console-session-surface", "#console-native-composer", "#console-run-inspector"):
        assert selector in evidence
    assert GATE15_EVIDENCE.name in readme
    assert GATE15_EVIDENCE.name in roadmap
    assert "Gate 1.5" in roadmap
    assert "TASK-10.6" in roadmap
    assert "status: Done" in task
    assert "## Implementation Notes" in task
```

Expected before docs/task updates: fail because evidence is missing and parent task is not done.

- [ ] **Step 2: Run full focused verification**

Run:

```bash
$PY -m pytest -q Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py Tests/UI/test_chat_tab_container.py Tests/UI/test_chat_window_enhanced.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_approvals_and_resume.py --tb=short
```

Expected: pass with known warnings only.

- [ ] **Step 3: Perform manual QA walkthrough**

Run the app in a clean or controlled config state:

```bash
$PY -m tldw_chatbook.app
```

Walk through:

- Open Console from Home.
- Confirm provider/model controls are visible outside the transcript.
- Confirm staged context tray is visible before sending.
- Confirm the transcript/event region does not contain the full legacy sidebar chrome.
- Confirm composer send/stop path is visible and keyboard reachable.
- Confirm missing provider/model state gives cause, impact, and recovery.
- Confirm a Library/Search/RAG or live-work handoff appears in staged context with provenance.
- Confirm artifact/Chatbook save control is either available with target or disabled with reason.

Record what was verified and what remains uncertain in the QA evidence doc.

- [ ] **Step 4: Add QA evidence and update tracking**

Create `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-5-console-internals-decomposition.md` with:

- `## Scope`
- `## Walkthrough`
- `## Functional Result`
- `## Verification`
- `## Defects`
- `## Residual Risk`
- `## Exit Decision`

Update:

- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
- `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- `backlog/tasks/task-10.6 - Product-Maturity-Phase-3.6-Gate-1.5-Console-Internals-Decomposition.md`
- `backlog/tasks/task-10.6.5 - Gate-1.5.5-Console-internals-QA-closeout.md`

- [ ] **Step 5: Mark ACs only after evidence is true**

Check each completed AC in:

- `TASK-10.6`
- `TASK-10.6.1`
- `TASK-10.6.2`
- `TASK-10.6.3`
- `TASK-10.6.4`
- `TASK-10.6.5`

Set `TASK-10.6` and `TASK-10.6.5` to `Done` only if the QA evidence proves the Console is usable, not merely renderable.

- [ ] **Step 6: Run evidence test green**

Run:

```bash
$PY -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_gate15_console_internals_evidence_is_tracked --tb=short
git diff --check
```

Expected: pass; no whitespace errors.

- [ ] **Step 7: Commit closeout**

```bash
git add Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-5-console-internals-decomposition.md Docs/superpowers/qa/product-maturity/phase-3/README.md Docs/superpowers/trackers/product-maturity-roadmap.md Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md "backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md" "backlog/tasks/task-10.6 - Product-Maturity-Phase-3.6-Gate-1.5-Console-Internals-Decomposition.md" "backlog/tasks/task-10.6.5 - Gate-1.5.5-Console-internals-QA-closeout.md" Tests/UI/test_console_internals_decomposition.py
git commit -m "Record Gate 1.5 Console internals verification"
```

---

### Task 6: Final PR Verification

**Files:**
- Read: all modified files
- Verify: focused test suite and diff hygiene

- [ ] **Step 1: Run full Gate 1.5 focused suite**

Run:

```bash
$PY -m pytest -q Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_chat_shell_bar.py Tests/UI/test_chat_tab_container.py Tests/UI/test_chat_window_enhanced.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_approvals_and_resume.py --tb=short
```

Expected: all selected tests pass with known warnings only.

- [ ] **Step 2: Run diff hygiene**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; branch has only intended committed changes.

- [ ] **Step 3: Self-review against the binding specs**

Re-read:

```bash
sed -n '1,240p' Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
rg -n "### Console|Gate 1.5|Library: Search/RAG" Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md
```

Confirm:

- Console no longer embeds the full legacy `ChatWindowEnhanced` chrome inside the workbench.
- Console keeps route id `chat`.
- Library-native Search/RAG remains deferred to Gate 1.6.
- Existing chat handoffs, tab/session state, provider/model controls, live-work launches, and Chatbook artifact paths remain compatible.
- Blocked states include cause, impact, owner, and next action.

- [ ] **Step 4: Prepare PR summary**

Use:

```markdown
## Summary
- Added Console-native display-state contracts and widgets for control bar, staged context, transcript/session, composer, and run inspector.
- Removed the full embedded ChatWindowEnhanced chrome from the Console workbench while preserving direct compatibility tests.
- Recorded Gate 1.5 QA evidence and roadmap/backlog tracking.

## Verification
- `$PY -m pytest -q Tests/Chat/test_console_display_state.py Tests/UI/test_console_internals_decomposition.py ... --tb=short`
- `git diff --check`
```

- [ ] **Step 5: Commit any final fixes**

If final verification required fixes:

```bash
git add <changed-files>
git commit -m "Stabilize Gate 1.5 Console internals"
```

Expected: branch is ready for PR against `dev`.
