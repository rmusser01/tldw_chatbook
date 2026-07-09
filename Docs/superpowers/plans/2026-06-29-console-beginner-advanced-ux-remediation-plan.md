# Console Beginner and Advanced UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Console screen so beginners get direct setup recovery and advanced regular users get one dense, canonical workflow surface without duplicate controls.

**Architecture:** Keep the existing Console frame: global nav, top workflow controls, recovery/status, context rail, transcript, inspector, composer, footer. Consolidate duplicated Workbench `ModeStrip`/`CommandStrip` content into the Console-owned control strip, leaving shared Workbench seams mounted hidden only where compatibility tests still need selectors. Put beginner recovery and advanced operational state into display-state builders first, then render those contracts in focused widgets.

**Tech Stack:** Python 3.11+, Textual widgets/TCSS, pytest async UI harness, Textual `export_screenshot()`, Backlog.md task `TASK-144`.

---

## ADR Check

ADR required: no

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`

Reason: this plan implements the existing Workbench UI System decision. It does not introduce a new storage model, runtime boundary, or navigation architecture. It corrects Console screen composition and state exposure inside the accepted Workbench frame.

## Scope

In scope:

- Remove duplicated visible provider/action rows.
- Make blocked first-run setup directly actionable.
- Make transcript empty state useful for beginner activation.
- Make inspector useful in blocked/ready states, or visually lighter when it has no useful detail.
- Show disabled-action reasons near the disabled controls.
- Preserve advanced keyboard-first density and visible core actions.
- Regenerate SVG and PNG visual evidence.

Out of scope:

- Redesigning global navigation across all screens.
- Migrating non-Console destinations.
- Changing provider configuration persistence.
- Replacing Textual navigation or the command palette.
- Full provider streaming soak with a live external endpoint.

## Current Findings Mapped to Fixes

```text
Finding                                      Plan fix
------------------------------------------   ------------------------------------------
Duplicate state/action rows                  Canonical Console workflow strip
Beginner setup recovery is indirect           Actionable recovery callout
Empty transcript dominates first run          Activation-oriented empty transcript panel
Inspector consumes space without value        Blocked/ready inspector summaries and collapse rules
Disabled actions lack nearby reason           Inline disabled reason row/chips
Left rail is spacious for sparse state        Compact rail empty-state copy and spacing
Advanced users need quick scan                One run recipe row plus source/tool/approval groups
```

## Files and Responsibilities

- `backlog/tasks/task-144 - Address-Console-beginner-and-advanced-workflow-UX-findings.md`
  Track acceptance criteria, implementation plan link, and implementation notes.

- `tldw_chatbook/Widgets/Console/console_workbench_state.py`
  Owns Console route-level actions, recovery copy, recovery action labels, and compatibility Workbench state.

- `tldw_chatbook/Chat/console_display_state.py`
  Owns pure display-state contracts for control-strip labels, inspector rows, disabled reasons, and empty/recovery summaries.

- `tldw_chatbook/Widgets/Console/console_control_bar.py`
  Becomes the one visible top workflow strip. It renders state chips and canonical top-level actions from Workbench state. It must not duplicate with `ModeStrip`/`CommandStrip`.

- `tldw_chatbook/UI/Screens/chat_screen.py`
  Wires the consolidated strip, hides compatibility Workbench strips, routes recovery actions, and passes display state to transcript/inspector/composer.

- `tldw_chatbook/Widgets/Console/console_transcript.py`
  Renders activation-oriented empty state content and buttons when there are no messages.

- `tldw_chatbook/Widgets/Console/console_run_inspector.py`
  Renders blocked/ready inspector summaries, advanced run recipe, source/tool/approval/artifact groups, and collapse/lightweight states.

- `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  Shows disabled reasons close to Send/Save/Stop and keeps setup recovery aligned with the top recovery state.

- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  Source styles for dense strip, recovery, inspector, transcript empty state, rail spacing, and disabled reasons.

- `tldw_chatbook/css/tldw_cli_modular.tcss`
  Generated CSS. Rebuild with `PATH=.venv/bin:$PATH python3 tldw_chatbook/css/build_css.py`.

- `Tests/UI/test_console_workbench_contract.py`
  Contract tests for single canonical strip, direct recovery actions, inspector usefulness, disabled reasons, and empty transcript actions.

- `Tests/UI/test_workbench_visual_snapshots.py`
  Visual smoke assertions for current screenshots.

- `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`
  QA notes and visual evidence paths.

- `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/`
  Regenerated normal, compact, focus, and command-palette SVG/PNG evidence.

---

### Task 1: Normalize TASK-144 and Add State-Contract Tests

**Files:**
- Modify: `backlog/tasks/task-144 - Address-Console-beginner-and-advanced-workflow-UX-findings.md`
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workbench_state.py`
- Modify: `tldw_chatbook/Chat/console_display_state.py`

- [ ] **Step 1: Split TASK-144 acceptance criteria**

Update the task file so each acceptance criterion is its own checkbox:

```markdown
- [ ] Console has one canonical visible state/action control strip.
- [ ] Blocked setup state exposes direct recovery actions.
- [ ] Transcript empty state provides useful workflow launch actions.
- [ ] Inspector shows actionable blocked/run/source/tool/approval/artifact state or collapses when not useful.
- [ ] Disabled actions expose nearby reasons.
- [ ] Beginner and advanced workflow visual evidence is captured and reviewed.
- [ ] Targeted Console Workbench tests and visual snapshot checks pass.
```

- [ ] **Step 2: Add failing state tests for direct setup recovery**

Add tests in `Tests/UI/test_console_workbench_contract.py`:

```python
def test_console_workbench_recovery_names_specific_setup_action():
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="Provider setup needed: choose a model",
        provider_action_label="Choose model",
        can_send=False,
    )

    assert state.recovery is not None
    assert state.recovery.action is not None
    assert state.recovery.action.label == "Choose model"
    assert state.recovery.action.primary is True
    assert "Send is blocked" in state.recovery.body
```

Expected current failure: the action label is still generic in some mounted states, or the test exposes where `ChatScreen` must pass a specific label.

- [ ] **Step 3: Add failing display-state tests for disabled reasons**

If a new display-state helper is introduced, test it directly:

```python
def test_console_disabled_reason_copy_prefers_setup_blocker():
    reason = build_console_disabled_reason(
        action_id="send",
        has_draft=False,
        send_blocked=True,
        setup_blocked_reason="Provider setup needed: choose a model",
    )

    assert reason == "Send disabled: choose a model"
```

If avoiding a new helper, add the mounted composer test in Task 5 instead.

- [ ] **Step 4: Run tests to verify red**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_workbench_recovery_names_specific_setup_action -q
```

Expected: fail for the missing/specific recovery contract before implementation.

- [ ] **Step 5: Implement minimal state-contract additions**

Update `tldw_chatbook/Widgets/Console/console_workbench_state.py` so recovery labels can be specific and remain primary:

```python
action=WorkbenchAction(
    id="provider-recovery",
    label=provider_action_label,
    tooltip=provider_action_label,
    primary=True,
)
```

Add pure helpers in `tldw_chatbook/Chat/console_display_state.py` only if the tests need shared disabled-reason copy. Keep helpers small and string-only.

- [ ] **Step 6: Verify green**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_workbench_recovery_names_specific_setup_action -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add "backlog/tasks/task-144 - Address-Console-beginner-and-advanced-workflow-UX-findings.md" Tests/UI/test_console_workbench_contract.py tldw_chatbook/Widgets/Console/console_workbench_state.py tldw_chatbook/Chat/console_display_state.py
git commit -m "Define Console UX recovery state contracts"
```

---

### Task 2: Build One Canonical Console Workflow Strip

**Files:**
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add failing mounted test for no duplicate visible strip**

Add this test:

```python
@pytest.mark.asyncio
async def test_console_has_one_canonical_visible_state_action_strip():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert not _is_displayed(console.query_one("#console-workbench-mode-strip"))
        assert not _is_displayed(console.query_one("#console-workbench-command-strip"))
        control_bar = console.query_one("#console-control-bar")
        assert _is_displayed(control_bar)

        visible_text = " ".join(
            _widget_text(child)
            for child in control_bar.walk_children()
            if _is_displayed(child)
        )
        assert visible_text.count("Provider:") == 1
        assert visible_text.count("Model:") == 1
        assert visible_text.count("Settings") == 1
        assert visible_text.count("Library RAG") == 1
```

- [ ] **Step 2: Run test to verify red**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_has_one_canonical_visible_state_action_strip -q
```

Expected: fail because `#console-workbench-mode-strip` and `#console-workbench-command-strip` are currently visible.

- [ ] **Step 3: Extend `ConsoleControlBar` to accept top actions**

Change `ConsoleControlBar.__init__`:

```python
def __init__(
    self,
    state: ConsoleControlState,
    app_instance: Any,
    *,
    actions: tuple[WorkbenchAction, ...] = (),
    on_sidebar_toggle_requested: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> None:
    ...
    self.actions = actions
```

Import `WorkbenchAction` from `tldw_chatbook.UI.Workbench.workbench_state`.

Render action buttons from `self.actions`, excluding composer-owned `send` and `stop`:

```python
TOP_ACTION_IDS = {
    "new-tab",
    "settings",
    "attach-context",
    "run-library-rag",
    "save-chatbook",
    "help",
}
```

- [ ] **Step 4: Preserve action routing**

Use the existing `_workbench_action_id` attribute and `WorkbenchActionRequested` message. Disabled top actions should remain visible only when their reason is useful; otherwise keep them subdued with tooltip.

- [ ] **Step 5: Hide shared Workbench strips as compatibility seams**

In `tldw_chatbook/UI/Screens/chat_screen.py`, replace visible composition of `ModeStrip` and `CommandStrip` with `_hidden_console_workbench_widget(...)` while keeping sync calls tolerant:

```python
yield self._hidden_console_workbench_widget(
    ModeStrip(..., id="console-workbench-mode-strip", ...)
)
yield self._hidden_console_workbench_widget(
    CommandStrip(..., id="console-workbench-command-strip", ...)
)
```

Pass actions to `ConsoleControlBar`:

```python
ConsoleControlBar(
    control_state,
    self.app_instance,
    actions=workbench_state.actions,
    ...
)
```

- [ ] **Step 6: Update CSS**

In `_agentic_terminal.tcss`, keep the hidden strips zero-height for Console only:

```css
#console-workbench-mode-strip,
#console-workbench-command-strip {
    display: none;
    height: 0;
    min-height: 0;
    max-height: 0;
    padding: 0;
    margin: 0;
    border: none;
}
```

- [ ] **Step 7: Rebuild CSS**

Run:

```bash
PATH=.venv/bin:$PATH python3 tldw_chatbook/css/build_css.py
```

Expected: exit 0. Existing warning for missing `features/_evaluation_v2.tcss` is acceptable.

- [ ] **Step 8: Verify green**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_has_one_canonical_visible_state_action_strip -q
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add Tests/UI/test_console_workbench_contract.py tldw_chatbook/Widgets/Console/console_control_bar.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Consolidate Console workflow controls"
```

---

### Task 3: Make Blocked Setup Recovery Directly Actionable

**Files:**
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workbench_state.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add failing mounted recovery-action test**

Add:

```python
@pytest.mark.asyncio
async def test_console_blocked_setup_recovery_has_primary_choose_model_action():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        recovery = console.query_one("#workbench-recovery-callout")
        assert _is_displayed(recovery)
        action = console.query_one("#workbench-recovery-action")
        assert _is_displayed(action)
        assert str(action.label) == "Choose model"
        assert action.disabled is False
```

- [ ] **Step 2: Run test to verify red**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_blocked_setup_recovery_has_primary_choose_model_action -q
```

Expected: fail if mounted recovery still says `Open Settings`.

- [ ] **Step 3: Add specific recovery label derivation**

In `ChatScreen`, add a helper near `_build_console_workbench_state`:

```python
@staticmethod
def _console_provider_recovery_action_label(blocker_copy: str) -> str:
    text = blocker_copy.lower()
    if "model" in text:
        return "Choose model"
    if "api key" in text:
        return "Add API key"
    if "endpoint" in text:
        return "Configure endpoint"
    return "Open Settings"
```

Pass the helper output into `build_console_workbench_state(...)`.

- [ ] **Step 4: Route recovery action to the right existing handler**

Keep `provider-recovery` routed through the existing settings/open-provider path in `WorkbenchActionRequested` handling. Do not create a new modal if settings already supports provider/model configuration.

- [ ] **Step 5: Style recovery action as the primary next step**

Ensure `#workbench-recovery-action` has enough width and primary contrast, without increasing the callout height beyond the existing 4 rows.

- [ ] **Step 6: Verify green and no regressions**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_blocked_setup_recovery_has_primary_choose_model_action Tests/UI/test_console_workbench_contract.py::test_console_composer_keeps_primary_actions_and_setup_recovery_visible -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add Tests/UI/test_console_workbench_contract.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_workbench_state.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Make Console setup recovery actionable"
```

---

### Task 4: Replace Passive Empty Transcript With Activation Panel

**Files:**
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add failing empty-state action test**

Add:

```python
@pytest.mark.asyncio
async def test_console_empty_transcript_exposes_beginner_activation_actions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        empty_panel = console.query_one("#console-transcript-empty-state")
        assert _is_displayed(empty_panel)
        assert "Choose model" in _widget_text(empty_panel)
        assert _is_displayed(console.query_one("#console-empty-choose-model"))
        assert _is_displayed(console.query_one("#console-empty-attach-context"))
        assert _is_displayed(console.query_one("#console-empty-run-library-rag"))
```

- [ ] **Step 2: Run test to verify red**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_empty_transcript_exposes_beginner_activation_actions -q
```

Expected: fail because the current empty transcript is plain static copy.

- [ ] **Step 3: Add empty panel widgets**

In `console_transcript.py`, update empty row rendering to use a dedicated `Vertical` container when no messages exist:

```python
class ConsoleTranscriptEmptyAction(Button):
    pass
```

Render:

```python
with Vertical(id="console-transcript-empty-state", classes="console-transcript-empty-state"):
    yield Static("Start Console", id="console-empty-title", ...)
    yield Static(self.empty_state_copy, id="console-empty-body", ...)
    yield Button("Choose model", id="console-empty-choose-model", ...)
    yield Button("Attach context", id="console-empty-attach-context", ...)
    yield Button("Run Library RAG", id="console-empty-run-library-rag", ...)
```

If nested `yield` from helper methods is cleaner, keep it inside `ConsoleTranscript._build_row_widget()`.

- [ ] **Step 4: Route empty actions**

On empty action press, post `WorkbenchActionRequested` with IDs:

- `provider-recovery` for choose model
- `attach-context`
- `run-library-rag`

Import `WorkbenchActionRequested` from `tldw_chatbook.UI.Workbench.workbench_widgets`.

- [ ] **Step 5: Make copy setup-aware**

From `ChatScreen._sync_console_transcript_guidance()`, pass blocked setup copy so empty body says:

```text
Choose a model to enable Send. Then type in Composer or attach context.
```

For ready state:

```text
Type in Composer, attach sources, or run Library RAG before sending.
```

- [ ] **Step 6: Style the panel without card clutter**

Use one unframed panel inside the transcript:

```css
.console-transcript-empty-state {
    width: 100%;
    height: auto;
    padding: 1 2;
    background: transparent;
}
```

Buttons should be one-line, aligned left, and visually consistent with top strip actions.

- [ ] **Step 7: Verify green**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_empty_transcript_exposes_beginner_activation_actions -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add Tests/UI/test_console_workbench_contract.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Add Console transcript activation empty state"
```

---

### Task 5: Make Inspector Useful for Blocked and Advanced States

**Files:**
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Modify: `tldw_chatbook/Widgets/Console/console_run_inspector.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add failing blocked-inspector test**

Add:

```python
@pytest.mark.asyncio
async def test_console_blocked_inspector_explains_impact_and_next_action():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        inspector = console.query_one("#console-run-inspector-state")
        visible_text = " ".join(
            _widget_text(child)
            for child in inspector.walk_children()
            if _is_displayed(child)
        )
        assert "Blocked" in visible_text
        assert "Send is blocked" in visible_text
        assert "Choose model" in visible_text
```

- [ ] **Step 2: Add failing advanced-inspector test**

Add:

```python
@pytest.mark.asyncio
async def test_console_ready_inspector_shows_run_recipe_and_operational_groups():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        inspector = console.query_one("#console-run-inspector-state")
        text = " ".join(
            _widget_text(child)
            for child in inspector.walk_children()
            if _is_displayed(child)
        )
        assert "Run recipe" in text
        assert "Sources" in text
        assert "Tools" in text
        assert "Approvals" in text
        assert "Artifacts" in text
```

- [ ] **Step 3: Run tests to verify red**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_blocked_inspector_explains_impact_and_next_action Tests/UI/test_console_workbench_contract.py::test_console_ready_inspector_shows_run_recipe_and_operational_groups -q
```

Expected: fail because the inspector currently shows sparse setup state.

- [ ] **Step 4: Extend inspector rows in display state**

In `console_display_state.py`, add rows or normalize existing rows for:

- `Run recipe`
- `Blocked impact`
- `Next action`
- `Sources`
- `Tools`
- `Approvals`
- `Artifacts`

Keep labels stable so tests can query IDs.

- [ ] **Step 5: Update inspector grouping**

In `console_run_inspector.py`, make the first visible group always useful:

```python
(
    "Run",
    "console-inspector-run-heading",
    ("Run recipe", "Live work", "Setup", "Blocked impact", "Next action", "Provider"),
)
```

Then show `Source Readiness`, `Tools`, `Approvals`, `Artifacts`, and finally secondary conversation/session groups.

- [ ] **Step 6: Collapse or lighten low-value inspector content**

If no selected message and no live work exist, keep only:

```text
Status: Ready
Run recipe: provider / model / sources / tools / approvals
```

Do not render a large blank right rail with only `setup`.

- [ ] **Step 7: Verify green**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_blocked_inspector_explains_impact_and_next_action Tests/UI/test_console_workbench_contract.py::test_console_ready_inspector_shows_run_recipe_and_operational_groups -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add Tests/UI/test_console_workbench_contract.py tldw_chatbook/Chat/console_display_state.py tldw_chatbook/Widgets/Console/console_run_inspector.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Make Console inspector action-first"
```

---

### Task 6: Show Disabled Action Reasons Near Controls

**Files:**
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add failing composer disabled-reason test**

Add:

```python
@pytest.mark.asyncio
async def test_console_composer_shows_send_disabled_reason_near_send():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        reason = console.query_one("#console-send-disabled-reason")
        assert _is_displayed(reason)
        assert "Send disabled" in _widget_text(reason)
        assert "choose model" in _widget_text(reason).lower()
```

- [ ] **Step 2: Run test to verify red**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_composer_shows_send_disabled_reason_near_send -q
```

Expected: fail because reason copy is currently only placeholder/recovery copy, not a named disabled reason near Send.

- [ ] **Step 3: Add composer reason static**

In `ConsoleComposerBar.compose()`, add:

```python
yield Static("", id="console-send-disabled-reason", classes="console-send-disabled-reason")
```

Place it near the action row, not above the whole composer.

- [ ] **Step 4: Sync reason state**

In `sync_action_state()`, update and show reason when send is not ready:

```python
if send_blocked and setup_blocked_reason:
    reason.update("Send disabled: choose a model")
    reason.styles.display = "block"
elif not has_draft:
    reason.update("Send disabled: type a message")
    reason.styles.display = "block"
else:
    reason.update("")
    reason.styles.display = "none"
```

Use the exact setup reason helper from Task 1 if created.

- [ ] **Step 5: Add optional top-strip disabled reason for Save Chatbook**

If `save-chatbook` remains visible but disabled in the top strip, render a short reason chip:

```text
Save unavailable: no artifact
```

Only show it on focus or in compact form if always-visible copy crowds the strip.

- [ ] **Step 6: Verify green**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_composer_shows_send_disabled_reason_near_send -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add Tests/UI/test_console_workbench_contract.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/Widgets/Console/console_control_bar.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Show Console disabled action reasons"
```

---

### Task 7: Tighten Left Rail Density and First-Run Priority

**Files:**
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add failing left-rail density test**

Add a structural test that prevents sparse first-run rail layout from hiding the primary attach action:

```python
@pytest.mark.asyncio
async def test_console_left_rail_prioritizes_attach_and_active_conversation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        rail = console.query_one("#console-left-rail")
        visible_text = " ".join(
            _widget_text(child)
            for child in rail.walk_children()
            if _is_displayed(child)
        )
        assert visible_text.index("Attach") < visible_text.index("Conversations")
        assert "No staged work" in visible_text
        assert "Chat 1" in visible_text
```

- [ ] **Step 2: Run test to verify current behavior**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_left_rail_prioritizes_attach_and_active_conversation -q
```

Expected: may already pass. If it passes, keep it as a regression and continue with CSS density changes under existing visual tests.

- [ ] **Step 3: Reduce rail vertical dead space**

In `_agentic_terminal.tcss`, adjust rail sections:

```css
.console-left-rail-section {
    margin: 0 0 1 0;
    padding: 0 1;
}
```

Keep section separation readable, but avoid large blank blocks around empty staged context and conversations.

- [ ] **Step 4: Improve copy**

In `ChatScreen` rail composition, prefer:

```text
Staged Context
No sources attached.
```

over less actionable copy when the attach button is the next action.

- [ ] **Step 5: Verify**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_console_workbench_contract.py::test_console_left_rail_prioritizes_attach_and_active_conversation Tests/UI/test_workbench_visual_snapshots.py::test_console_workbench_normal_and_compact_snapshots -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add Tests/UI/test_console_workbench_contract.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Tighten Console context rail density"
```

---

### Task 8: Regenerate Visual Evidence and Update QA

**Files:**
- Modify: `Tests/UI/test_workbench_visual_snapshots.py`
- Modify: `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`
- Modify: `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/*.svg`
- Modify: `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/*.png`

- [ ] **Step 1: Strengthen visual snapshot assertions**

Update `_assert_console_density_evidence()` in `Tests/UI/test_workbench_visual_snapshots.py` to assert:

```python
assert normalized_svg.count("Provider:") == 1
assert "Choose model" in normalized_svg
assert "Send disabled" in normalized_svg or "Setup required" in normalized_svg
assert "Run recipe" in normalized_svg
```

Adjust counts if normal/focus screenshots intentionally include hidden compatibility text in SVG export. The assertion should count visible rendered text only after `export_screenshot(simplify=True)`.

- [ ] **Step 2: Run visual tests before regenerating artifacts**

Run:

```bash
PATH=.venv/bin:$PATH pytest Tests/UI/test_workbench_visual_snapshots.py -q
```

Expected: pass or expose final visual assertion mismatch to correct before artifact refresh.

- [ ] **Step 3: Regenerate SVG artifacts**

Use the existing Textual harness script pattern:

```bash
PATH=.venv/bin:$PATH python3 - <<'PY'
from pathlib import Path
from unittest.mock import patch
import asyncio

from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_workbench_visual_snapshots import _open_console, _test_cli_setting
from textual.widgets import Button

OUT = Path("Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual")
OUT.mkdir(parents=True, exist_ok=True)

async def capture() -> None:
    captures = (
        ("normal", "normal", None),
        ("compact", "compact", None),
        ("command-palette", "normal", "palette"),
        ("focus-state", "normal", "focus"),
    )
    for name, density, mode in captures:
        app = _build_test_app()
        app.app_config = getattr(app, "app_config", {}) or {}
        app.app_config.setdefault("appearance", {})["ui_density"] = density
        with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
            async with app.run_test(size=(140, 42)) as pilot:
                await _open_console(app, pilot)
                if mode == "palette":
                    await pilot.press("ctrl+p")
                    await pilot.pause()
                elif mode == "focus":
                    action = app.screen.query_one("#console-control-settings", Button)
                    action.focus()
                    await pilot.pause()
                svg = app.export_screenshot(
                    title=f"Console Workbench {name.replace('-', ' ')}",
                    simplify=True,
                )
                (OUT / f"console-workbench-{name}.svg").write_text(svg, encoding="utf-8")

asyncio.run(capture())
PY
```

- [ ] **Step 4: Normalize generated SVG whitespace**

Run:

```bash
perl -pi -e 's/[ \t]+$//' Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/console-workbench-*.svg
```

- [ ] **Step 5: Render PNG evidence**

Run with escalation if Chromium is blocked by sandbox:

```bash
node -e "import('playwright').then(async ({ chromium }) => { const path = await import('node:path'); const { pathToFileURL } = await import('node:url'); const base = path.resolve('Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual'); const names = ['console-workbench-normal','console-workbench-focus-state','console-workbench-compact','console-workbench-command-palette']; const browser = await chromium.launch({ headless: true }); for (const name of names) { const page = await browser.newPage({ viewport: { width: 1800, height: 1150 }, deviceScaleFactor: 1 }); page.setDefaultTimeout(120000); const svg = path.join(base, name + '.svg'); await page.goto(pathToFileURL(svg).href, { waitUntil: 'load', timeout: 120000 }); await page.waitForTimeout(250); await page.screenshot({ path: path.join(base, name + '.png'), fullPage: false, timeout: 120000, animations: 'disabled' }); await page.close(); } await browser.close(); }).catch((err) => { console.error(err); process.exit(1); })"
```

- [ ] **Step 6: Update QA notes**

Update `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md` with:

- New TASK-144 summary.
- Beginner workflow evidence.
- Advanced workflow evidence.
- PNG paths and dimensions.
- Known residual risks.

- [ ] **Step 7: Verify artifacts**

Run:

```bash
wc -c Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/*.svg Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/*.png
sips -g pixelWidth -g pixelHeight Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/*.png
rg -n "Provider:|Model:|Choose&#160;model|Run&#160;recipe|Send&#160;disabled|Library&#160;RAG" Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/*.svg
rg -n "Traceback|Unhandled exception|Unable to mount|Internal Error|<.* object at 0x" Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual/*.svg
git diff --check
```

Expected:

- SVG and PNG files are non-empty.
- PNGs are 1800x1150 unless the renderer changes.
- Positive `rg` finds expected labels.
- Error-marker `rg` returns no matches.
- `git diff --check` exits 0.

- [ ] **Step 8: Commit**

```bash
git add Tests/UI/test_workbench_visual_snapshots.py Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/visual
git commit -m "Refresh Console UX visual evidence"
```

---

### Task 9: Final Verification and Task Closeout

**Files:**
- Modify: `backlog/tasks/task-144 - Address-Console-beginner-and-advanced-workflow-UX-findings.md`

- [ ] **Step 1: Run targeted suite**

Run:

```bash
PATH=.venv/bin:$PATH pytest \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_workbench_visual_snapshots.py \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_workbench_focus_help.py \
  Tests/UI/test_ui_responsiveness.py \
  Tests/UI/test_ui_responsiveness_artifacts.py \
  Tests/UI/test_destination_visual_parity_correction.py::test_console_first_start_shows_left_rail_main_and_right_handle \
  -q
```

Expected: pass with only existing dependency warnings.

- [ ] **Step 2: Run selected legacy decomposition seams**

Run:

```bash
PATH=.venv/bin:$PATH pytest \
  Tests/UI/test_console_internals_decomposition.py::test_console_control_bar_renders_readable_summary_line \
  Tests/UI/test_console_internals_decomposition.py::test_console_native_control_bar_and_staged_context_reflect_pending_handoff \
  Tests/UI/test_console_internals_decomposition.py::test_console_native_control_bar_uses_existing_compact_model_sync_seam \
  Tests/UI/test_console_internals_decomposition.py::test_console_control_labels_refresh_after_compact_control_sync \
  Tests/UI/test_console_internals_decomposition.py::test_console_run_inspector_shows_blocked_provider_and_missing_rag_source \
  Tests/UI/test_console_internals_decomposition.py::test_console_run_inspector_exposes_pending_approval_and_chatbook_artifact_actions \
  -q
```

Expected: pass.

- [ ] **Step 3: Run CSS build**

Run:

```bash
PATH=.venv/bin:$PATH python3 tldw_chatbook/css/build_css.py
```

Expected: exit 0. Existing `features/_evaluation_v2.tcss` missing-module warning is acceptable.

- [ ] **Step 4: Run route-switch soak**

Run:

```bash
PATH=.venv/bin:$PATH python3 Tests/UI/run_workbench_soak.py --output Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts --route-switches 6 --idle-seconds 10
```

Expected:

```text
route switches: 6, failures: 0, focus failures: 0, workers before: 0, workers after: 0
```

- [ ] **Step 5: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: exit 0.

- [ ] **Step 6: Update TASK-144 notes and mark Done**

Update acceptance criteria to `[x]` and add implementation notes summarizing:

- One canonical workflow strip.
- Direct setup recovery.
- Activation empty state.
- Useful inspector.
- Disabled reasons.
- Visual proof and verification commands.

Then run:

```bash
backlog task edit 144 -s Done
```

- [ ] **Step 7: Commit closeout**

```bash
git add "backlog/tasks/task-144 - Address-Console-beginner-and-advanced-workflow-UX-findings.md" Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts
git commit -m "Close Console UX remediation task"
```

---

## Execution Order

Recommended order:

1. Task 1: state contracts.
2. Task 2: single canonical strip.
3. Task 3: direct blocked recovery.
4. Task 4: transcript activation empty state.
5. Task 5: useful inspector.
6. Task 6: disabled reasons.
7. Task 7: rail density.
8. Task 8: visual evidence.
9. Task 9: final verification and closeout.

This order avoids visual churn: first remove duplicated controls, then improve recovery and empty states, then tune secondary regions.

## Risk Controls

- Keep compatibility widgets mounted hidden until legacy tests are migrated.
- Do not route beginner recovery through the command palette.
- Do not remove global navigation border in this plan.
- Do not make inspector a modal.
- Do not add new persistence or provider configuration flows.
- Treat screenshots as required evidence, not optional polish.

## Final Acceptance Checklist

- [ ] Beginner can identify and activate `Choose model` without guessing.
- [ ] Advanced user sees one scannable run recipe/control strip.
- [ ] No duplicated provider/model/RAG/source/tool/approval rows are visible.
- [ ] Empty transcript gives actionable starts, not passive instructions.
- [ ] Inspector justifies its right-rail footprint.
- [ ] Disabled Send/Save/Stop states explain why near the controls.
- [ ] Normal, compact, focus, and command-palette PNGs render and are committed.
- [ ] Targeted tests, CSS build, route-switch soak, artifact scan, and diff hygiene pass.
