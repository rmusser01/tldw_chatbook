# Conversation Settings Connection-First Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompose the full Console modal around a clear connection setup path while preserving fast, keyboard-first access to advanced per-conversation controls.

**Architecture:** Keep `ConsoleSettingsModal` as the single full-modal owner, add one focused searchable provider picker, and derive field visibility from existing provider execution/capability seams. Textual `Collapsible` sections own progressive disclosure; TASK-30010 retains disclosure/focus state and TASK-30011 supplies typed blocker/evidence state.

**Tech Stack:** Python 3.11, Textual 8.x (`Collapsible`, `OptionList`, `Select`), existing Console provider/capability services, TCSS, pytest/Textual Pilot.

**Spec:** `Docs/superpowers/specs/2026-09-02-console-conversation-settings-ready-to-send-design.md`

## Global Constraints

- Name the surface `Conversation settings`; credentials remain editable only in F9 Settings > Providers & Models.
- Connection fields precede advanced tuning in composition and keyboard order.
- Unknown capability support stays available under Advanced with neutral copy; only authoritative `unsupported` evidence removes a control from layout and focus traversal.
- There is exactly one primary completion action at a time.
- Plain language distinguishes conversation-only application from defaults for future conversations.
- Edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, then run the CSS builder; never hand-edit the generated bundle.
- Only targeted tests run unless the user separately approves a full sweep.

---

### Task 1: Add a searchable grouped provider picker

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_provider_picker.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Create: `Tests/Widgets/test_console_provider_picker.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: `ConsoleProviderPicker(provider_options: Sequence[ConsoleSettingsOption], current_provider: str | None)`
- Produces: `ConsoleProviderPicker.ProviderSelected(provider: str)`
- Produces: provider groups `Cloud`, `Local`, `Custom`, and `Other`
- Consumes: `build_console_provider_options()` and `provider_display_name()`

- [ ] **Step 1: Write red keyboard/search/group tests**

Cover case-insensitive filtering by display name and config key, group ordering, disabled group headings, Up/Down/Enter selection, Escape restoration, no-match copy, markup escaping, and current-value preservation.

```python
await pilot.click("#console-settings-provider-picker-input")
await pilot.press("l", "l", "a", "m", "a")
assert picker.visible_provider_ids() == ("llama_cpp", "local_llamacpp")
await pilot.press("down", "enter")
assert selected.provider == "llama_cpp"
```

- [ ] **Step 2: Verify the component tests fail**

Run: `pytest Tests/Widgets/test_console_provider_picker.py Tests/UI/test_console_session_settings.py -k 'provider_picker or searchable_provider' -q`

Expected: FAIL because `ConsoleProviderPicker` is absent and the modal still uses `Select`.

- [ ] **Step 3: Implement the bounded picker**

Mirror the proven input/result ownership in `ModelSearchPicker` without catalog loading or a generic base class. Build groups from existing provider keys and URL-based/local identity helpers; treat unclassified supported keys as `Other`. Store only the option tuple and current provider, cap visible results at 30, escape every rendered label, and never accept arbitrary typed provider IDs.

- [ ] **Step 4: Replace the modal provider Select**

Mount the picker at `#console-settings-provider-picker`, preserve per-provider model/Base URL drafts before switching, refresh readiness and field visibility after `ProviderSelected`, and include nested input/result IDs in TASK-30010's draft/focus allowlists.

- [ ] **Step 5: Verify provider selection behavior**

Run: `pytest Tests/Widgets/test_console_provider_picker.py Tests/UI/test_console_session_settings.py -k 'provider or draft' -q`

Expected: PASS.

- [ ] **Step 6: Commit the provider picker**

```bash
git add tldw_chatbook/Widgets/Console/console_provider_picker.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Widgets/test_console_provider_picker.py Tests/UI/test_console_session_settings.py
git commit -m "feat: add searchable Conversation provider picker"
```

### Task 2: Centralize generation-control availability

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_support.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Test: `Tests/Chat/test_console_provider_support.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: `ConsoleGenerationControl` values `reasoning_effort`, `reasoning_summary`, `verbosity`, `thinking_effort`, `thinking_budget_tokens`
- Produces: `ConsoleControlSupport` values `supported`, `unsupported`, `unknown`
- Produces: `console_generation_control_support(provider: str, model: str | None, control: ConsoleGenerationControl) -> ConsoleControlSupport`

- [ ] **Step 1: Write red parameterized support tests**

Use exact provider/model combinations already covered by `model_capabilities.py` and local execution-key behavior. Assert known no-effect pairs are `unsupported`, known implemented pairs are `supported`, and custom/unknown models remain `unknown`.

- [ ] **Step 2: Verify the support tests fail**

Run: `pytest Tests/Chat/test_console_provider_support.py -k 'generation_control_support' -q`

Expected: FAIL because the typed support query does not exist.

- [ ] **Step 3: Implement a pure adapter over existing facts**

Resolve the provider through `resolve_console_provider_identity()`, reuse current local thinking payload rules and `model_capabilities.py` predicates, and return `unknown` whenever existing sources cannot prove support either way. Move modal-local no-effect/provider sets into this adapter; do not add a second registry.

- [ ] **Step 4: Render support without silent data loss**

Hide `unsupported` rows with `display = False`. Leave `unknown` rows mounted under Advanced and render `Support not verified for this model.` Never hide a control only because its draft is blank, and never silently coerce a retained value.

- [ ] **Step 5: Verify capability and focus behavior**

Run: `pytest Tests/Chat/test_console_provider_support.py Tests/UI/test_console_session_settings.py -k 'support or provider_specific or focus_order' -q`

Expected: PASS.

- [ ] **Step 6: Commit capability-driven disclosure**

```bash
git add tldw_chatbook/Chat/console_provider_support.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Chat/test_console_provider_support.py Tests/UI/test_console_session_settings.py
git commit -m "feat: derive Conversation controls from provider support"
```

### Task 3: Recompose the modal around Connection and Advanced

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: `console-settings-connection`, `console-settings-readiness-panel`, `console-settings-connection-actions`
- Produces: `console-settings-generation-advanced`, `console-settings-identity-advanced`, `console-settings-request-estimate`
- Consumes: TASK-30010 draft disclosure state and TASK-30011 readiness

- [ ] **Step 1: Write red composition and disclosure tests**

Assert title and DOM order, first focus target, default disclosure for new/blocked and configured states, explicit targeted opening, snapshot restoration, hidden-control traversal, and exact zero/one/many-model wording.

```python
ids = [widget.id for widget in modal.query(".console-settings-modal-section")]
assert ids.index("console-settings-connection") < ids.index("console-settings-generation-advanced")
assert str(modal.query_one(".console-modal-header", Static).renderable) == "Conversation settings"
```

- [ ] **Step 2: Verify layout tests fail**

Run: `pytest Tests/UI/test_console_session_settings.py -k 'connection_first or disclosure or modal_title or model_count_copy' -q`

Expected: FAIL on the current flat section order and title.

- [ ] **Step 3: Compose the connection-first hierarchy**

In order, render provider, conditional credential recovery, conditional Base URL, model/custom ID, discover/verify affordances, primary readiness, independent evidence rows, and one recovery action. Put sampling, generation controls, identity, and request estimate inside separate `Collapsible` widgets. Keep Context and memory as a peer view, with destructive/immediate actions in a visibly secondary region.

- [ ] **Step 4: Replace free-form enumerations with Selects**

Use existing accepted-value tables for reasoning effort, reasoning summary, verbosity, and thinking effort. Retain numeric Inputs for token budgets and sampling. Map `Select.NULL` to `None`; reject invalid restored values inline rather than coercing them.

- [ ] **Step 5: Restore disclosure deliberately**

Default Advanced generation closed when operability is `not_ready` or no prior snapshot exists; open it when a navigation target names an advanced control; otherwise restore the session snapshot. Collapsed descendants must not be focusable.

- [ ] **Step 6: Verify hierarchy and constrained inputs**

Run: `pytest Tests/UI/test_console_session_settings.py -k 'connection or advanced or context or select or focus' -q`

Expected: PASS.

- [ ] **Step 7: Commit the connection-first composition**

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/UI/test_console_session_settings.py
git commit -m "feat: recompose Conversation settings connection first"
```

### Task 4: Make completion primacy and save scope explicit

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_summary.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Generate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_session_settings.py`
- Test: `Tests/UI/test_console_rail_sections.py`

**Interfaces:**
- Produces: primary `Use for this conversation`
- Produces: secondary `Save as provider defaults` / `Save as generation defaults`
- Produces: persistent `#console-settings-primary-disabled-reason`

- [ ] **Step 1: Write red action-hierarchy and copy tests**

Cover ready, missing credential, invalid endpoint, missing model, run-active, context-operation-active, and defaults-save-capable states. Assert exactly one visible enabled button has `variant="primary"`, and every disabled completion button has adjacent persistent reason copy.

- [ ] **Step 2: Verify action tests fail**

Run: `pytest Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_sections.py -k 'primary_action or save_scope or disabled_reason or provider_default_copy' -q`

Expected: FAIL on current `Save` / `Save model defaults` ambiguity.

- [ ] **Step 3: Implement scope-specific actions**

Rename the session action `Use for this conversation`. Render defaults actions only for changed fields they actually persist, with adjacent `Used by future conversations for <provider>.` copy. Keep credential navigation, discovery, reset, compaction, and cancel secondary. Use TASK-30011 blocker codes for disabled reasons.

- [ ] **Step 4: Correct cross-entry copy**

Render `Provider default` only for genuine inheritance, `Not estimated` when no context estimate exists, singular `1 model`, and shared `provider_display_name()` labels in modal and rail.

- [ ] **Step 5: Build CSS and verify**

Run: `python3 tldw_chatbook/css/build_css.py`

Run: `pytest Tests/Chat/test_console_provider_support.py Tests/Widgets/test_console_provider_picker.py Tests/Widgets/test_model_search_picker.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_sections.py -q`

Expected: PASS.

- [ ] **Step 6: Verify diff and commit**

Run: `git diff --check`

Expected: no output.

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/Widgets/Console/console_settings_summary.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_sections.py
git commit -m "fix: clarify Conversation settings completion scope"
```
