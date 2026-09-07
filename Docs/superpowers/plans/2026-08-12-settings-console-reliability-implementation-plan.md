# Settings and Console Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make provider Settings understandable and truthful, apply saved defaults to a chosen conversation only on explicit request, and make Console sends lossless and non-duplicating across refusal, failure, cancellation, and retry.

**Architecture:** Extract pure Settings presentation models from the large screen, retain one save path through the provider foundation, and carry explicit apply requests through a revisioned typed handoff. Refactor Console submission so all preflight checks precede the single durable user-turn commit; every accepted attempt then terminates as completed, failed, or cancelled without restoring submitted text.

**Tech Stack:** Python 3.11+, Textual, dataclasses, existing Console store/controller, pending handoff store, pytest, pytest-asyncio.

## Global Constraints

- Complete the provider connection foundation and first-run provider handoff plans first.
- Settings saves defaults for new conversations; it never silently changes the active Console session.
- Apply handoffs contain session/provider/model/profile identities and revisions, never endpoint or credential values.
- Preflight refusal creates no message row and preserves the current draft plus staged attachments.
- Acceptance creates exactly one durable user turn and clears only the submitted draft snapshot/staging.
- Failure and cancellation retain the accepted user turn and add one terminal assistant attempt.
- Unsupported provider fields remain stored and are not silently cleared.
- Use `Tests/...` as the canonical test path spelling.

---

## File Structure

- Create `tldw_chatbook/UI/Screens/settings_provider_view_model.py`: task-oriented overview, provider grouping/search results, generation groups, and context-capacity presentation.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: render the view model, save semantics, and targeted apply action.
- Modify `tldw_chatbook/Widgets/Console/console_settings_modal.py`: compact grouped settings presentation.
- Read the existing `config.get_runtime_config_snapshot().generation` as the process-local configuration revision after successful saves.
- Modify `tldw_chatbook/UI/Navigation/pending_handoff_store.py`: add typed settings-apply intent.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: stage and consume the exact apply intent.
- Modify `tldw_chatbook/Chat/console_chat_controller.py`: move the user commit after preflight and normalize terminal attempts.
- Modify `tldw_chatbook/Chat/console_chat_store.py` only if a dedicated cancelled transition is absent.
- Modify focused tests named in each task.

### Task 1: Extract task-oriented provider Settings presentation

**Files:**
- Create: `tldw_chatbook/UI/Screens/settings_provider_view_model.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:10406-10750`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Create: `Tests/UI/test_settings_provider_view_model.py`

**Interfaces:**
- Produces `SettingsOverviewPresentation` ordered as configuration, last test, storage/privacy, and sync.
- Produces `ProviderPickerGroup` and `ProviderPickerOption` preserving unknown IDs.
- Produces `build_provider_picker_groups(catalog, saved_provider, query)`.

- [ ] **Step 1: Write failing overview-order and provider-search tests**

```python
def test_overview_leads_with_user_tasks():
    view = build_settings_overview(_configured_snapshot())
    assert [row.key for row in view.primary_rows] == [
        "configuration", "last_connection_test", "storage_privacy", "sync"
    ]
    assert "handoff" not in " ".join(row.label.lower() for row in view.primary_rows)


def test_provider_picker_preserves_saved_unknown_and_manual_entry():
    groups = build_provider_picker_groups(_catalog("openai"), "my_proxy", "proxy")
    options = [option for group in groups for option in group.options]
    assert any(option.provider_id == "my_proxy" and option.saved_unknown for option in options)
    assert any(option.action == "enter_provider_id" for option in options)
```

Add search tests for display name/provider ID, stable grouping, empty search, and disabled headings.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/UI/test_settings_provider_view_model.py Tests/UI/test_settings_configuration_hub.py -k "overview or provider_picker or search or unknown" -v`

Expected: FAIL because Settings currently renders system internals first and uses a long flat provider selector.

- [ ] **Step 3: Implement immutable presentation records**

```python
@dataclass(frozen=True, slots=True)
class ProviderPickerOption:
    provider_id: str | None
    label: str
    search_text: str
    saved_unknown: bool = False
    action: Literal["select", "enter_provider_id"] = "select"

@dataclass(frozen=True, slots=True)
class ProviderPickerGroup:
    group_id: str
    label: str
    options: tuple[ProviderPickerOption, ...]
```

Normalize only for comparison; preserve the saved unknown provider text for display and editing. Keep runtime ownership, server binding, and handoff rows in an Advanced/Diagnostics disclosure.

- [ ] **Step 4: Render the overview and searchable grouped picker**

Add a search input plus grouped `OptionList`; headings are disabled. Include an **Enter provider ID** action that opens the existing manual provider input path. Assert endpoint/API key edits survive filter changes.

Run: `.venv/bin/python -m pytest Tests/UI/test_settings_provider_view_model.py Tests/UI/test_settings_configuration_hub.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the Settings overview and picker**

```bash
git add tldw_chatbook/UI/Screens/settings_provider_view_model.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_provider_view_model.py Tests/UI/test_settings_configuration_hub.py
git commit -m "feat: organize provider settings by user task"
```

### Task 2: Group generation controls and make context capacity honest

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_provider_view_model.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:8536-8910,10504-10850`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py:1034-1150`
- Modify: `tldw_chatbook/Chat/console_session_settings.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_console_context_controls.py`
- Modify: `Tests/Chat/test_console_context_policy.py`

**Interfaces:**
- Produces `GenerationControlGroup` for Common, Reasoning, Advanced sampling, and provider-specific controls.
- Produces `ContextCapacityPresentation(state, headline, action, details)`.
- Produces `build_context_capacity_presentation(estimate)` with an explicit unknown state.

- [ ] **Step 1: Write failing grouping and unknown-capacity tests**

```python
def test_unknown_context_inputs_do_not_present_fallback_as_capacity():
    view = build_context_capacity_presentation(_estimate(window=None, source=None))
    assert view.state == "unavailable"
    assert view.headline == "Capacity unavailable"
    assert "8,001" not in repr(view)


def test_generation_groups_show_only_supported_fields():
    groups = build_generation_groups(_capabilities(reasoning=True, min_p=False))
    assert [group.label for group in groups] == ["Common", "Reasoning", "Advanced sampling"]
    assert "reasoning_effort" in _field_ids(groups)
    assert "min_p" not in _field_ids(groups)
```

Add tests that hidden unsupported values remain in saved data, group reset affects only the selected provider/model profile, and details expose source precedence, provider cap, response reserve, safety margin, and mandatory input.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_context_controls.py Tests/Chat/test_console_context_policy.py Tests/UI/test_settings_configuration_hub.py -k "capacity or generation_group or scoped_reset or unsupported" -v`

Expected: FAIL because unknown capacity currently receives fallback-looking numeric copy and controls are presented as one long scroll.

- [ ] **Step 3: Implement capability groups without data loss**

Build display groups from the existing provider capability metadata. Filter only the rendered controls; leave the draft/persisted mapping untouched for unsupported fields. Reset receives exact `(provider_key, model_id, group_id)` and deletes or restores only fields declared in that group.

- [ ] **Step 4: Implement one context source precedence**

```python
@dataclass(frozen=True, slots=True)
class ContextCapacityPresentation:
    state: Literal["available", "unavailable"]
    headline: str
    action: str | None
    details: tuple[tuple[str, str], ...]
```

Use selected model profile, provider-reported limit, and explicit user override in the precedence already encoded by `build_console_context_estimate`; remove presentation-layer numeric fallback. Unknown input produces **Capacity unavailable** and an action to set/discover the model context window.

- [ ] **Step 5: Verify compact Settings and Console modal layouts**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_context_controls.py Tests/Chat/test_console_context_policy.py Tests/UI/test_settings_configuration_hub.py -v`

Expected: PASS with group headings, local help text, no horizontal overflow, and no nested cards at narrow and standard viewports.

- [ ] **Step 6: Commit grouped generation and context presentation**

```bash
git add tldw_chatbook/UI/Screens/settings_provider_view_model.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/Chat/console_session_settings.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_context_controls.py Tests/Chat/test_console_context_policy.py
git commit -m "fix: clarify generation and context settings"
```

### Task 3: Add revisioned Apply to current conversation

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:17475-17740`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:4020-4145`
- Modify: `Tests/State/test_pending_handoff_store.py`
- Create: `Tests/UI/test_settings_apply_current_conversation.py`

**Interfaces:**
- Produces `ConsoleSettingsApplyIntent(session_id, provider, model, profile_id, config_revision)`.
- Adds `HandoffChannel.CONSOLE_SETTINGS_APPLY`.

- [ ] **Step 1: Write failing save/apply/race tests**

```python
def test_apply_intent_is_secret_and_endpoint_free():
    intent = ConsoleSettingsApplyIntent("session-7", "custom", "model-a", "default", 4)
    assert set(asdict(intent)) == {"session_id", "provider", "model", "profile_id", "config_revision"}


async def test_console_refuses_apply_when_session_changed(app):
    intent = _apply_intent(session_id="session-old", config_revision=4)
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_SETTINGS_APPLY, intent)
    app.console_store.activate_session("session-new")
    assert not app.chat_screen.consume_pending_console_settings_apply()
    assert app.console_store.active_session.settings.provider == "openai"
```

Also test revision mismatch, non-idle session, save failure, successful exact apply, and later user edits. Assert the button is disabled before save and after draft edits.

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/UI/test_settings_apply_current_conversation.py -v`

Expected: FAIL because only provider-only intent exists and there is no save revision.

- [ ] **Step 3: Capture the existing runtime configuration generation**

```python
@dataclass(frozen=True, slots=True)
class ConsoleSettingsApplyIntent:
    session_id: str
    provider: str
    model: str
    profile_id: str
    config_revision: int
```

After a fully applied atomic save, read `get_runtime_config_snapshot().generation` and copy it into the intent as `config_revision`. Console reads a fresh runtime snapshot when consuming the claim. The generation is process-local, increases after every published configuration mutation, and therefore fails closed after unrelated concurrent writes without adding a second revision owner.

- [ ] **Step 4: Implement explicit Settings and Console actions**

After save, report **Saved for new conversations** and enable **Apply to current conversation** only for the session captured when Settings opened, while idle. Stage the exact intent. Console rechecks active session ID, idle state, runtime config generation, and profile identity before replacing `ConsoleSessionSettings`; refusal acknowledges the stale intent without mutation.

- [ ] **Step 5: Run handoff regressions**

Run: `.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/UI/test_settings_apply_current_conversation.py Tests/UI/test_console_session_settings.py Tests/UI/test_settings_provider_switch_atomic.py -v`

Expected: PASS.

- [ ] **Step 6: Commit targeted Settings apply**

```bash
git add tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py Tests/State/test_pending_handoff_store.py Tests/UI/test_settings_apply_current_conversation.py Tests/UI/test_console_session_settings.py Tests/UI/test_settings_provider_switch_atomic.py
git commit -m "feat: apply saved provider defaults explicitly"
```

### Task 4: Move every Console preflight check before history mutation

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:2291-2645`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/UI/test_console_send_draft_snapshot.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Produces internal `ConsoleSubmissionSnapshot(text, attachments, session_id, draft_revision)`.
- Produces internal `ConsolePreflightOutcome(accepted, reason, provider, request_context)`.
- Preserves public `ConsoleSubmitResult` and UI acceptance callback behavior.

- [ ] **Step 1: Write failing zero-history refusal tests**

```python
async def test_missing_provider_refusal_preserves_draft_and_creates_no_history(controller):
    controller.composer.set_text("hello once")
    controller.store.stage_attachment("session-1", _attachment("image-a"))
    result = await controller.submit_draft("session-1")
    assert not result.accepted
    assert controller.store.messages_for_session("session-1") == []
    assert controller.composer.text == "hello once"
    assert [item.id for item in controller.store.pending_attachments("session-1")] == ["image-a"]
```

Repeat for missing model, invalid substitution, RAG preflight failure, unsupported attachment, and context refusal. Assert no auto-title change for every refused submission.

- [ ] **Step 2: Run focused tests and observe optimistic user rows**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_send_draft_snapshot.py -k "preflight or refusal or preserves_draft or no_history" -v`

Expected: FAIL because `submit_draft` currently appends an optimistic user row before provider resolution and marks it failed when blocked.

- [ ] **Step 3: Capture a snapshot, then preflight without mutation**

```python
@dataclass(frozen=True, slots=True)
class ConsoleSubmissionSnapshot:
    session_id: str
    text: str
    attachments: tuple[MessageAttachment, ...]
    draft_revision: int
```

Resolve provider/model, substitutions, staged RAG, attachment capability, and context admission using the snapshot. Return a refused `ConsoleSubmitResult` directly on failure. Do not title, append, persist, clear, or mark any message during preflight.

- [ ] **Step 4: Commit the user turn once after acceptance**

After preflight accepts, derive the title, append/persist one user row with the snapshot attachments, clear only matching staged attachments, and invoke `_on_console_submission_accepted` with the snapshot revision so later composer text is retained.

- [ ] **Step 5: Run refusal and acceptance tests**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_send_draft_snapshot.py Tests/UI/test_console_native_chat_flow.py -k "submit or preflight or accepted or attachment" -v`

Expected: PASS with zero rows for refusal and one durable user row for acceptance.

- [ ] **Step 6: Commit the preflight transaction fix**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_send_draft_snapshot.py Tests/UI/test_console_native_chat_flow.py
git commit -m "fix: keep refused Console sends out of history"
```

### Task 5: Normalize failure, cancellation, and retry ownership

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:2620-2750,5220-5450`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:3500-3640` if required for cancellation.
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/UI/test_console_send_draft_snapshot.py`

**Interfaces:**
- Accepted failure creates one assistant row with `status="failed"` and retry linkage to the accepted user message.
- Accepted cancellation creates one assistant row with `status="cancelled"` or the existing canonical stopped/cancelled status, distinct from failed.
- Retry consumes the stored user turn and attachments and creates no user row.

- [ ] **Step 1: Write failing row-count and late-composer tests**

```python
async def test_accepted_failure_retry_never_duplicates_user_turn(controller):
    controller.gateway.fail_next(RuntimeError("bounded failure"))
    first = await controller.submit_text("session-1", "hello once")
    assert first.accepted
    assert _roles(controller, "session-1") == ["user", "assistant"]
    failed = controller.store.messages_for_session("session-1")[-1]
    controller.gateway.respond_next("recovered")
    await controller.retry_failed_queue_turn(failed.id)
    assert _roles(controller, "session-1").count("user") == 1


async def test_late_failure_does_not_restore_over_new_composer_text(controller):
    pending = asyncio.create_task(controller.submit_text("session-1", "submitted"))
    await controller.gateway.wait_until_started()
    controller.composer.set_text("typed later")
    controller.gateway.fail(RuntimeError("bounded failure"))
    await pending
    assert controller.composer.text == "typed later"
```

Add attachment identity assertions across retry and a cancellation test requiring one user plus one cancelled assistant attempt.

- [ ] **Step 2: Run the terminal-attempt tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_send_draft_snapshot.py -k "accepted_failure or retry_never or cancellation or late_failure" -v`

Expected: at least one test fails under the current optimistic-row/retry behavior.

- [ ] **Step 3: Bind terminal attempts to the accepted user row**

Pass the accepted user message ID and immutable attachments into the request attempt. On exception, create or transition exactly one assistant attempt to failed. On `CancelledError`, preserve the user turn and transition the assistant attempt to the canonical cancelled state; do not route cancellation through generic failure copy.

- [ ] **Step 4: Make retry assistant-only**

`retry_failed_queue_turn` resolves the failed assistant's linked user message, reads its persisted content and attachments, and dispatches a new assistant attempt. It must not call `submit_draft`, append a user message, clear staging, or change composer text.

- [ ] **Step 5: Run the full Console reliability group**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_send_draft_snapshot.py Tests/UI/test_console_send_disabled_state.py -v`

Expected: PASS.

- [ ] **Step 6: Commit terminal attempt ownership**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_send_draft_snapshot.py Tests/UI/test_console_send_disabled_state.py
git commit -m "fix: make Console failure and retry nonduplicating"
```

### Task 6: Run the Settings and Console slice gate

**Files:**
- Verify only; modify failures only when caused by this plan.

- [ ] **Step 1: Run focused lint**

Run: `.venv/bin/python -m ruff check tldw_chatbook/UI/Screens/settings_provider_view_model.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py`

Expected: PASS.

- [ ] **Step 2: Run Settings regressions**

Run: `.venv/bin/python -m pytest Tests/UI/test_settings_provider_view_model.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_provider_switch_atomic.py Tests/UI/test_settings_provider_test_draft.py Tests/UI/test_settings_apply_current_conversation.py Tests/UI/test_console_context_controls.py Tests/State/test_pending_handoff_store.py -v`

Expected: PASS.

- [ ] **Step 3: Run Console regressions**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_context_policy.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_send_draft_snapshot.py Tests/UI/test_console_send_disabled_state.py Tests/UI/test_console_session_settings.py -v`

Expected: PASS.

- [ ] **Step 4: Commit gate-only corrections when needed**

```bash
git add tldw_chatbook Tests
git commit -m "test: close Settings and Console regressions"
```

Skip this commit when the gate requires no corrections.
