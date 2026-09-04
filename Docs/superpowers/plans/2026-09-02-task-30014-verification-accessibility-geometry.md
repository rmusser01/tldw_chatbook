# Conversation Settings Verification Accessibility and Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Conversation settings with honest live verification, deterministic keyboard/accessibility behavior, and usable compact-through-wide terminal layouts.

**Architecture:** Keep probe availability in the provider endpoint domain and inject ChatScreen-owned connection tests that reuse `settings_endpoint_probe.py`; the reusable modal never imports a Settings screen module. Reuse the Console gateway's `AuxiliaryCompletionRequest` for an explicitly confirmed, one-token paid generation check. Both operations publish through TASK-30011's exact-identity evidence store. The modal owns focus/announcements and toggles a compact class from measured width; visual verification uses isolated Textual apps and rendered frames.

**Tech Stack:** Python 3.11, Textual 8.x workers/Pilot, existing endpoint probe and Console provider gateway, TCSS, pytest, Textual screenshot/export APIs.

**Spec:** `Docs/superpowers/specs/2026-09-02-console-conversation-settings-ready-to-send-design.md`

## Global Constraints

- Connection tests never generate tokens or imply generation success.
- Generation tests run only after a fresh explicit confirmation for that request; confirmation is never remembered.
- Generation tests bypass transcript/history/tools and cap requested output at one token through `AuxiliaryCompletionRequest`.
- Network results are exact-identity/generation fenced, cancellable, timeout-bounded, and sanitized. A generation test uses a 15-second request timeout, zero retries, and a 20-second outer coroutine deadline.
- Accessibility state is visible in text and exposed through the best Textual-supported semantics; color is supplementary.
- The approved modal accelerator is `Ctrl+Enter`; it shadows neither repo globals nor terminal-convention bindings.
- Only targeted tests run unless the user separately approves a full sweep.

---

### Task 1: Unify model discovery with meaningful connection probes

**Files:**
- Modify: `tldw_chatbook/Chat/provider_endpoint_contract.py`
- Modify: `tldw_chatbook/UI/Screens/settings_endpoint_probe.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Test: `Tests/Chat/test_provider_endpoint_contract.py`
- Test: `Tests/UI/test_settings_endpoint_probe.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: `ConnectionProbeAvailability` values `models_route`, `unavailable`
- Produces: `connection_probe_availability(provider: str, endpoint: str | None) -> ConnectionProbeAvailability`
- Produces: injected `ConnectionTester = Callable[[ProviderDraftIdentity], Awaitable[ProviderProbeResult]]`
- Consumes: `probe_settings_endpoint(..., purpose=SettingsEndpointProbePurpose.CHAT_CATALOG)`

- [ ] **Step 1: Write red availability and result tests**

Cover URL-based local/custom providers, cloud providers, missing/invalid URLs, reachable lists, listing-unavailable responses, timeouts, refusal, unauthorized/forbidden, cancellation, identity edits during a request, and absence of a duplicate Discover/Test action pair.

- [ ] **Step 2: Verify tests fail**

Run: `pytest Tests/Chat/test_provider_endpoint_contract.py Tests/UI/test_settings_endpoint_probe.py Tests/UI/test_console_session_settings.py -k 'connection_probe_availability or test_connection' -q`

Expected: FAIL because the modal has discovery but no typed connection-probe availability.

- [ ] **Step 3: Implement a pure availability adapter**

Implement availability in `provider_endpoint_contract.py`. Return `models_route` only when `resolve_provider_endpoint()` and existing provider contracts produce a valid bounded models-route probe; otherwise return `unavailable`. Do not create another provider list.

- [ ] **Step 4: Add Test connection with honest fallback copy**

For `models_route`, rename/reuse the existing discovery action as `Test connection & list models`; do not mount a second action for the same `/models` request. Otherwise mount no test button and show `No non-billable live connection check is available for this provider.` `ChatScreen` injects a tester that calls `probe_settings_endpoint()` with `CHAT_CATALOG`; the modal dispatches it through an exclusive worker, feeds its model IDs into TASK-30013 provenance handling, captures the exact identity, cancels on close/edit, and publishes only if identity still matches.

- [ ] **Step 5: Verify supported/unsupported behavior**

Run: `pytest Tests/Chat/test_provider_endpoint_contract.py Tests/UI/test_settings_endpoint_probe.py Tests/UI/test_console_session_settings.py -k 'connection or probe or stale' -q`

Expected: PASS.

- [ ] **Step 6: Commit connection verification**

```bash
git add tldw_chatbook/Chat/provider_endpoint_contract.py tldw_chatbook/UI/Screens/settings_endpoint_probe.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Chat/test_provider_endpoint_contract.py Tests/UI/test_settings_endpoint_probe.py Tests/UI/test_console_session_settings.py
git commit -m "feat: add honest Conversation connection tests"
```

### Task 2: Add an explicitly authorized paid generation test

**Files:**
- Modify: `tldw_chatbook/Chat/provider_test_evidence.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Test: `Tests/Chat/test_provider_test_evidence.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: `ConsoleGenerationTestRequest(settings: ConsoleSessionSettings, identity: ProviderDraftIdentity)` with settings excluded from repr
- Produces: `ConsoleGenerationTestAvailability` values `supported`, `unsupported`
- Produces: `console_generation_test_availability(provider: str) -> ConsoleGenerationTestAvailability`, derived from `resolve_console_provider_identity()` and the existing handler catalog
- Produces: injected `GenerationTester = Callable[[ConsoleGenerationTestRequest], Awaitable[ProviderGenerationProbeResult]]`
- Produces: `ChatScreen._build_console_provider_selection_for_settings(session_id: str, settings: ConsoleSessionSettings) -> ConsoleProviderSelection`
- Consumes: `ConsoleProviderGateway.resolve_for_send()`, `AuxiliaryCompletionRequest`, and `complete_auxiliary()`
- Produces: confirmation actions `Run 1-token test` and `Cancel`

- [ ] **Step 1: Write red consent, billing-bound, and evidence tests**

Assert no call on initial button press, no remembered consent, exactly one call after confirmation, `max_output_tokens == 1`, non-streaming/no tools/no transcript persistence, 15-second request timeout, zero retries, 20-second outer deadline, cancellability, stale rejection, sanitized failures, supported/unsupported provider projection, and independent generation evidence.

```python
await pilot.click("#console-settings-test-generation")
assert gateway.calls == []
assert confirmation.display
await pilot.click("#console-settings-confirm-generation")
assert gateway.requests[0].max_output_tokens == 1
assert runtime.store.messages(session_id) == original_messages
```

- [ ] **Step 2: Verify consent tests fail**

Run: `pytest Tests/Chat/test_provider_test_evidence.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'generation_test or one_token or explicit_confirmation' -q`

Expected: FAIL because the modal has no generation-test flow.

- [ ] **Step 3: Implement the ChatScreen-owned tester**

Factor the current `_build_console_provider_selection_uncached()` projection so the tester can build a selection from the validated modal draft without mutating the session. Resolve that selection through `ConsoleProviderGateway.resolve_for_send()`, then replace request policy with `request_timeout=15.0`, `request_retries=0`, and `request_retry_delay=0.0`. Build `AuxiliaryCompletionRequest` with fixed message `Reply with one short token.`, no response format, and `max_output_tokens=1`; call `complete_auxiliary()` inside `asyncio.timeout(20.0)`. Convert the result/failure to bounded `ProviderGenerationProbeResult` codes without returning response text, raw exception, usage payload, endpoint, or credential.

- [ ] **Step 4: Add per-request confirmation and cancellation**

The first button reveals `This sends one paid generation request and may incur provider charges.` Confirm starts one exclusive worker and immediately resets consent. While running, expose `Cancel test`; cancellation/close invalidates the generation token and retains only any prior exact-identity evidence.

- [ ] **Step 5: Publish exact evidence only**

Show `Test generation` only when `console_generation_test_availability()` is `supported`; otherwise show fixed `Generation test unavailable for this provider.` copy. Set generation `succeeded` only for a supported completion and never display its text. Set `failed` with a bounded category for a current-identity failure. Provider, endpoint, model, or relevant generation-setting edits change the facet to `changed_since_test`.

- [ ] **Step 6: Verify paid test behavior**

Run: `pytest Tests/Chat/test_provider_test_evidence.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'generation or evidence or cancel' -q`

Expected: PASS.

- [ ] **Step 7: Commit explicit generation verification**

```bash
git add tldw_chatbook/Chat/provider_test_evidence.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Chat/test_provider_test_evidence.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py
git commit -m "feat: add confirmed one-token generation test"
```

### Task 3: Harden keyboard, focus, and accessibility semantics

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_provider_picker.py`
- Modify: `tldw_chatbook/Widgets/model_search_picker.py`
- Test: `Tests/UI/test_console_session_settings.py`
- Test: `Tests/Widgets/test_console_provider_picker.py`
- Test: `Tests/Widgets/test_model_search_picker.py`

**Interfaces:**
- Produces: `Binding("ctrl+enter", "activate_primary", "Apply", show=True)`
- Produces: deterministic logical focus order beginning with provider and ending with completion/cancel
- Produces: one status announcement per settled operation

- [ ] **Step 1: Write red keyboard/accessibility tests**

Cover initial focus, forward/reverse traversal, collapsed controls absent from traversal, picker Escape behavior, selected-state text, visible labels, tooltip/description copy, non-color-only readiness, `Ctrl+Enter`, disabled-primary no-op, and exactly one announcement per connection/generation result.

- [ ] **Step 2: Verify accessibility tests fail**

Run: `pytest Tests/UI/test_console_session_settings.py Tests/Widgets/test_console_provider_picker.py Tests/Widgets/test_model_search_picker.py -k 'keyboard or accessible or announcement or ctrl_enter or focus_order' -q`

Expected: FAIL on missing binding/semantics and focus ambiguity.

- [ ] **Step 3: Implement explicit labels and descriptions**

Give every interactive control a preceding visible `Label` or `Static` in the same field row, a stable widget `name` where useful to Textual tooling, and bounded keyboard-reachable tooltip/help text for added context. Selected/readiness states include literal `Selected`, `Ready to send`, `Not ready`, `Not tested`, or `Failed`; do not rely on icon/color alone. Do not invent unsupported ARIA attributes.

- [ ] **Step 4: Implement deterministic focus and activation**

After compose/restore, focus the requested valid control or provider fallback. When a control hides/disables, move focus to its section heading or recovery action. `action_activate_primary()` invokes only the visible enabled primary; otherwise focus `#console-settings-primary-disabled-reason` without dismissing.

- [ ] **Step 5: Announce transitions once**

Centralize terminal worker announcements in `_announce_verification_result()`; inline statuses remain persistent, while `notify(..., markup=False)` fires once for a current terminal result and never for stale/cancelled outcomes.

- [ ] **Step 6: Verify keyboard behavior**

Run: `pytest Tests/UI/test_console_session_settings.py Tests/Widgets/test_console_provider_picker.py Tests/Widgets/test_model_search_picker.py -k 'focus or keyboard or accessibility or announcement or picker' -q`

Expected: PASS.

- [ ] **Step 7: Commit accessibility behavior**

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/Widgets/Console/console_provider_picker.py tldw_chatbook/Widgets/model_search_picker.py Tests/UI/test_console_session_settings.py Tests/Widgets/test_console_provider_picker.py Tests/Widgets/test_model_search_picker.py
git commit -m "fix: harden Conversation settings keyboard accessibility"
```

### Task 4: Verify compact, normal, and wide geometry

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Generate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_console_settings_geometry.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: modal class `-conversation-settings-compact` below 100 columns
- Produces: size matrix `(80, 24)`, `(100, 30)`, `(160, 40)`

- [ ] **Step 1: Write red geometry and reachability tests**

At every size, assert no negative-width region, no horizontal overflow, Connection/footer actions reachable via scrolling/focus, full action labels, wrapped status text, and vertically stacked compact actions.

```python
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 40)])
async def test_conversation_settings_has_no_horizontal_overflow(size):
    async with app.run_test(size=size) as pilot:
        body = app.screen.query_one("#console-settings-body")
        assert body.virtual_size.width <= body.size.width
```

- [ ] **Step 2: Verify geometry tests fail**

Run: `pytest Tests/UI/test_console_settings_geometry.py -q`

Expected: FAIL on the missing size-aware layout and fixed-width rows.

- [ ] **Step 3: Toggle compact layout from measured width**

In `on_resize`, set `-conversation-settings-compact` from modal/container width. In source TCSS, make width fluid with a bounded maximum, let labels/controls shrink, stack Connection/footer actions in compact mode, and preserve one vertical scroll owner. Do not shorten labels to fit.

- [ ] **Step 4: Rebuild generated CSS**

Run: `python3 tldw_chatbook/css/build_css.py`

Expected: generated bundle changes and contains compact selectors exactly once.

- [ ] **Step 5: Export and inspect rendered frames**

Use the geometry harness and Textual screenshot/export API to capture blocked first-time and ready power-user states at all sizes in a temporary directory. Inspect header, Connection, readiness, disclosure, and completion hierarchy for clipping/detached footer.

- [ ] **Step 6: Verify geometry and commit**

Run: `pytest Tests/UI/test_console_settings_geometry.py Tests/UI/test_console_session_settings.py -q`

Expected: PASS.

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_settings_geometry.py Tests/UI/test_console_session_settings.py
git commit -m "fix: make Conversation settings responsive"
```

### Task 5: Run isolated journey UAT and evidence gates

**Files:**
- Modify: `backlog/tasks/task-30014 - Harden-Conversation-Settings-verification-accessibility-and-geometry.md`
- Optional, only for a new reproduced trap: `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run the complete focused automated slice**

```bash
pytest Tests/State/test_conversation_settings_navigation.py Tests/State/test_pending_handoff_store.py Tests/Chat/test_console_provider_support.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_provider_setup_persistence.py Tests/Chat/test_provider_test_evidence.py Tests/Widgets/test_console_provider_picker.py Tests/Widgets/test_model_search_picker.py Tests/UI/test_settings_endpoint_probe.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_local_server_discovery_card.py Tests/UI/test_console_local_server_probe_isolation.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_settings_geometry.py -q
```

Expected: PASS.

- [ ] **Step 2: Run static/diff checks**

Run: `python3 -m compileall -q tldw_chatbook/Chat tldw_chatbook/UI/Navigation tldw_chatbook/UI/Screens tldw_chatbook/Widgets`

Expected: exit 0.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 3: Start with an isolated profile**

Create a temporary home/config root with `mktemp -d`, launch `python3 -m tldw_chatbook.app` with the profile-specific environment variables documented in `backlog/docs/lessons-live-verification.md`, and use only fake cloud credentials plus a disposable local HTTP fixture. Never launch against the developer's real config.

- [ ] **Step 4: Walk the cloud first-time journey**

Verify missing-key readiness, exact credential deep-link, dirty-Settings conflict handling, save/return, raw draft/focus restoration, and truthful evidence. Confirm no prompt, prefill, Base URL, or key appears in handoff/log output.

- [ ] **Step 5: Walk the local-hosting journey**

Verify refused endpoint, correction, zero/one/many fixture models, provenance groups, unverified confirmation invalidation, endpoint write failure, successful persist-before-apply, and `Ready to send` with generation `Not tested`.

- [ ] **Step 6: Walk the power-user/accessibility journey**

Verify keyboard search, restored Advanced disclosure, constrained controls, `Ctrl+Enter`, disabled-primary reason, Tab/Shift+Tab at all sizes, paid-test confirmation/cancellation against a fake gateway, and one announcement per result.

- [ ] **Step 7: Record evidence and close task hygiene**

Add concise implementation notes to TASK-30010 through TASK-30014, check each AC only when evidence exists, record exact commands/frame locations, link existing ADRs, and move each task to Done only after its own Definition of Done. Add a lesson only for a newly observed reproducible incident.

- [ ] **Step 8: Commit verification notes**

```bash
git add "backlog/tasks/task-30010 - Add-safe-Conversation-Settings-credential-return-contract.md" "backlog/tasks/task-30011 - Separate-Conversation-Settings-operability-from-verification-evidence.md" "backlog/tasks/task-30012 - Recompose-Conversation-Settings-around-connection-first-disclosure.md" "backlog/tasks/task-30013 - Make-local-endpoint-saves-atomic-and-model-provenance-visible.md" "backlog/tasks/task-30014 - Harden-Conversation-Settings-verification-accessibility-and-geometry.md"
git commit -m "docs: record Conversation settings verification evidence"
```
