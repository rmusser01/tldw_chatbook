# Conversation Settings Return Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users configure a missing credential in canonical Settings and return to the exact originating Conversation settings draft without leaking content or overwriting unrelated work.

**Architecture:** Console stores a versioned raw modal snapshot in its existing `ScreenStateStore` payload; a new typed `PendingHandoffStore` channel carries only session identity, the Console-settings revision, logical focus, and its own revision token. Settings and Console parse allowlisted navigation targets, and Settings guards any restored provider draft before switching providers.

**Tech Stack:** Python 3.11, Textual 8.x, dataclasses, existing `ScreenStateStore`, existing `PendingHandoffStore`, pytest/Textual Pilot.

**Spec:** `Docs/superpowers/specs/2026-09-02-console-conversation-settings-ready-to-send-design.md`

## Global Constraints

- Credentials remain editable only in F9 Settings > Providers & Models.
- Return handoffs/navigation context never contain API keys, prompts, prefills, transcripts, raw Base URLs, or arbitrary result text.
- Suspended draft content is process-memory only and owned by the Console screen snapshot.
- Handoffs remain typed, single-slot, last-write-wins, claim/acknowledge/release, and application-thread-affine under ADR-033.
- Existing leave-Console run guards remain authoritative.
- No module-level token cache, root application-state object, credential field, provider registry, or disk persistence is added.
- Only targeted tests run unless the user separately approves a full sweep.

---

### Task 1: Add an exact Console-settings revision

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Test: `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Produces: `ConsoleChatSession.settings_revision: int`
- Produces: `ConsoleChatStore.session_settings_revision(session_id: str) -> int`
- Produces: private `ConsoleChatStore._bump_settings_revision(session_id: str) -> None`

- [ ] **Step 1: Write failing revision tests**

Add tests proving the revision starts at zero, advances only when modal-owned state actually changes, covers settings/context/name/system/prefill changes, and does not advance for message streaming or a no-op assignment.

```python
def test_console_settings_revision_tracks_only_settings_owned_changes(store):
    session = store.create_session(settings=_settings())
    assert store.session_settings_revision(session.id) == 0
    store.replace_session_settings(session.id, replace(_settings(), temperature=0.2))
    assert store.session_settings_revision(session.id) == 1
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    assert store.session_settings_revision(session.id) == 1
```

- [ ] **Step 2: Verify the tests fail for the missing API**

Run: `pytest Tests/Chat/test_console_chat_store.py -k 'settings_revision' -q`

Expected: FAIL because `session_settings_revision` and `settings_revision` do not exist.

- [ ] **Step 3: Implement the narrow revision bus**

Add `settings_revision: int = 0` to `ConsoleChatSession`; bump it after successful in-memory changes in `replace_session_settings`, `set_session_context_policy_overrides`, `set_session_user_display_name_override`, `set_session_system_prompt`, and `set_session_pinned_prefill`. Preserve zero on construction/restore and do not reuse payload or identity revisions.

```python
def session_settings_revision(self, session_id: str) -> int:
    """Return the process-local revision of modal-owned session settings."""
    return self._session_or_raise(session_id).settings_revision

def _bump_settings_revision(self, session_id: str) -> None:
    self._session_or_raise(session_id).settings_revision += 1
```

- [ ] **Step 4: Verify focused store behavior**

Run: `pytest Tests/Chat/test_console_chat_store.py -k 'settings_revision or replace_session_settings or context_policy or system_prompt or pinned_prefill or user_display_name' -q`

Expected: PASS.

- [ ] **Step 5: Commit the revision seam**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat: add Console settings revision fence"
```

### Task 2: Add typed return and navigation contracts

**Files:**
- Create: `tldw_chatbook/UI/Navigation/conversation_settings_navigation.py`
- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Modify: `tldw_chatbook/UI/Navigation/__init__.py`
- Test: `Tests/State/test_pending_handoff_store.py`
- Create: `Tests/State/test_conversation_settings_navigation.py`

**Interfaces:**
- Produces: `ConversationSettingsReturnIntent(session_id: str, settings_revision: int, active_view: Literal["model", "context"], focus_control_id: str | None)`
- Produces: `ProviderSettingsNavigationTarget.from_context(context: Mapping[str, object]) -> ProviderSettingsNavigationTarget | None`
- Produces: `ConsoleSettingsReturnTarget.from_context(context: Mapping[str, object]) -> ConsoleSettingsReturnTarget | None`
- Produces: `ConversationSettingsReturnOutcome` values `credential_saved`, `provider_settings_saved`, `without_saving`
- Produces: `HandoffChannel.CONVERSATION_SETTINGS_RETURN`

- [ ] **Step 1: Write strict parser and handoff tests**

Cover valid round trips, unknown-key rejection, invalid provider/field/outcome/revision rejection, structural detachment, replacement, exact claim acknowledgement, stale acknowledgement, release, and explicit clear.

```python
def test_provider_settings_target_rejects_unknown_context_key():
    assert ProviderSettingsNavigationTarget.from_context({
        "category": "providers-models",
        "provider": "openai",
        "field": "api_key",
        "return_revision": 4,
        "unexpected": "value",
    }) is None
```

- [ ] **Step 2: Verify the contract tests fail**

Run: `pytest Tests/State/test_pending_handoff_store.py Tests/State/test_conversation_settings_navigation.py -q`

Expected: FAIL because the new types/channel/module are absent.

- [ ] **Step 3: Implement exact immutable contracts**

Use frozen slotted dataclasses, explicit safe-character/length validation, a positive integer for the handoff revision, a bounded control-ID allowlist, and exact context key sets. Add the channel to `_detached_value()` and `_copy_value()` without a generic deepcopy fallback.

```python
class ConversationSettingsReturnOutcome(StrEnum):
    CREDENTIAL_SAVED = "credential_saved"
    PROVIDER_SETTINGS_SAVED = "provider_settings_saved"
    WITHOUT_SAVING = "without_saving"
```

- [ ] **Step 4: Verify state contracts**

Run: `pytest Tests/State/test_pending_handoff_store.py Tests/State/test_conversation_settings_navigation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the typed contracts**

```bash
git add tldw_chatbook/UI/Navigation/conversation_settings_navigation.py tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Navigation/__init__.py Tests/State/test_pending_handoff_store.py Tests/State/test_conversation_settings_navigation.py
git commit -m "feat: define Conversation settings return handoff"
```

### Task 3: Suspend and restore the exact modal draft

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/UI/test_console_session_settings.py`
- Test: `Tests/State/test_screen_state_store.py`
- Test: `Tests/UI/test_screen_navigation.py`

**Interfaces:**
- Produces: `ConsoleSettingsDraftSnapshot.to_mapping() -> dict[str, object]`
- Produces: `ConsoleSettingsDraftSnapshot.from_mapping(source: Mapping[str, object]) -> ConsoleSettingsDraftSnapshot | None`
- Produces: `ConsoleSettingsCredentialRequest(snapshot: ConsoleSettingsDraftSnapshot, provider: str, model: str | None)` as a new modal result variant
- Consumes: `ConversationSettingsReturnIntent` and `ConsoleChatStore.session_settings_revision()`

- [ ] **Step 1: Write red tests for raw draft fidelity and privacy separation**

Exercise temporarily invalid sampling text, per-provider model/Base URL drafts, context overrides, user name, system prompt, pinned prefill, active view, scroll anchor, focused control, and disclosure state. Assert prompt/prefill/raw endpoint appear only inside `native_console_state["suspended_conversation_settings"]`, never in the handoff or Settings route context.

```python
assert restored.raw_values["console-settings-temperature"] == "0.7.2"
assert restored.settings.system_prompt == "private system text"
assert "private system text" not in repr(return_intent)
assert "127.0.0.1" not in repr(settings_navigation_context)
```

- [ ] **Step 2: Verify snapshot/return tests fail**

Run: `pytest Tests/UI/test_console_session_settings.py -k 'credential_request or suspended_draft or return_restore' -q`

Expected: FAIL because the snapshot and credential-request result do not exist.

- [ ] **Step 3: Implement versioned snapshot capture and rehydration**

Add an explicit allowlist of modal widget IDs and primitive value types; serialize `ConsoleSessionSettings` fields, context overrides via `to_dict()`, provider draft maps, disclosure flags, and logical focus. `from_mapping()` must return `None` on malformed structure and copy every nested mapping/sequence it retains.

Preserve both the origin system prompt and pinned prefill when a restored draft is subsequently saved; neither field may be dropped merely because it is not edited by this modal.

```python
@dataclass(frozen=True, slots=True)
class ConsoleSettingsCredentialRequest:
    snapshot: ConsoleSettingsDraftSnapshot = field(repr=False)
    provider: str
    model: str | None
```

- [ ] **Step 4: Route credential requests through ChatScreen with guarded settlement**

Handle the modal result variant in `_open_console_settings`: retain the suspended snapshot on `ChatScreen`, stage the typed return handoff, and post the typed Settings context. Extend the existing `NavigateToScreen` path with one optional, single-settlement completion callback so the source can discard the exact pending handoff and reopen/retain its draft when flush, confirmation, transition admission, startup, overlay dismissal, or target construction rejects the route. The app handler owns settlement on every terminal path; ordinary navigation remains unchanged. Add the snapshot key to `_serialize_native_console_state()` / `_restore_native_console_state()` with absent-key backward compatibility. Do not apply the draft as session settings during suspension or bypass any leave-Console guard.

- [ ] **Step 5: Verify screen snapshot and modal suites**

Run: `pytest Tests/State/test_screen_state_store.py Tests/UI/test_console_session_settings.py Tests/UI/test_screen_navigation.py -k 'snapshot or credential or return or settings or navigation_completion' -q`

Expected: PASS.

- [ ] **Step 6: Commit exact draft suspension**

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Navigation/main_navigation.py tldw_chatbook/app.py Tests/UI/test_console_session_settings.py Tests/State/test_screen_state_store.py Tests/UI/test_screen_navigation.py
git commit -m "feat: suspend Conversation settings across credential setup"
```

### Task 4: Guard Settings drafts and complete the return

**Round-1 review scope amendment:** The exact return retry needs a value-free
`PendingHandoffStore.exact_revision_status()` query to distinguish an exact
claim still held in flight by an outgoing Console screen from a consumed or
superseded revision. This adds
`tldw_chatbook/UI/Navigation/pending_handoff_store.py` and its existing focused
store test file to Task 4 scope. It exposes no handoff payload, changes no
persistence schema or owner, and requires no new ADR; ADR-033 remains the
governing application-session ownership decision.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Consumes: `ProviderSettingsNavigationTarget`, `ConsoleSettingsReturnTarget`, `ConversationSettingsReturnOutcome`
- Produces: Settings continuation actions `settings-provider-return`, `settings-provider-stay`, `settings-provider-return-without-save`
- Produces: conflict actions `settings-provider-conflict-review`, `settings-provider-conflict-discard`, `settings-provider-conflict-return`

- [ ] **Step 1: Write failing dirty-draft and return journey tests**

Cover clean deep-link, same-provider dirty disclosure, different-provider Review/Discard/Return, save failure, credential-only save, broader save, return without saving, Stay abandonment, stale revision, deleted session, temporary session, superseded token, repeated return, focus fallback, and an env-var name whose process value is absent.

```python
assert provider_input.value == "anthropic"
assert api_key_input.has_focus
assert "Endpoint" in str(existing_changes_summary.render())
assert store.session_settings(origin_id) == origin_settings
```

- [ ] **Step 2: Verify journey tests fail**

Run: `pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -k 'conversation_settings_return or provider_navigation_conflict' -q`

Expected: FAIL because Settings does not yet retain or resolve the typed continuation.

- [ ] **Step 3: Implement the Settings conflict and continuation regions**

Parse Providers & Models context only through `ProviderSettingsNavigationTarget`. For the same provider, preserve the draft and list dirty field display names. For a different provider, stage the target and show the three explicit conflict actions. Show continuation actions after fully applied save; show Return without saving behind the existing discard guard while dirty; clear the handoff on Stay.

- [ ] **Step 4: Implement Console claim/restore settlement**

Add `ChatScreen.apply_navigation_context()` for `ConsoleSettingsReturnTarget`. Claim only the exact revision, compare session ID and `settings_revision`, reopen the modal with its snapshot, restore logical focus if present, refresh cached provider config, and acknowledge only after restoration or terminal rejection. Release transient mount failures.

- [ ] **Step 5: Add mutation-aware and env-var recovery copy**

Return only an enum in navigation context. Map it to fixed screen-owned copy and re-run canonical readiness; if the selected credential source is an absent environment variable, display the export/relaunch recovery and remain blocked.

- [ ] **Step 6: Verify the complete round trip**

Run: `pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_session_settings.py -k 'conversation_settings or provider_navigation or credential or screen_state' -q`

Expected: PASS.

- [ ] **Step 7: Commit the guarded return flow**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat: return safely from provider credential settings"
```

### Task 5: Run security and ownership gates

**Files:**
- Modify: `Tests/State/test_pending_handoff_store.py`
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`

**Interfaces:**
- Verifies all interfaces produced by Tasks 1–4.

- [ ] **Step 1: Add sentinel and ownership regression assertions**

Use a fake credential only in the masked Settings field and assert it is absent from rendered output, handoff repr/value copies, navigation context, Console snapshots, and captured logs. Separately prove prompt/prefill survive only through the private screen snapshot.

- [ ] **Step 2: Run the focused security/ownership slice**

Run: `pytest Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/State/test_conversation_settings_navigation.py Tests/Chat/test_console_chat_store.py Tests/UI/test_console_session_settings.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_native_chat_flow.py -q`

Expected: PASS.

- [ ] **Step 3: Run static checks for forbidden parallel state**

Run: `rg -n "conversation_settings.*(cache|registry)|return_intent" tldw_chatbook`

Expected: no module-level cache/registry and no legacy untyped return-intent implementation.

- [ ] **Step 4: Verify the diff and commit test hardening**

Run: `git diff --check`

Expected: no output.

```bash
git add Tests/State/test_pending_handoff_store.py Tests/UI/test_console_session_settings.py Tests/UI/test_settings_configuration_hub.py
git commit -m "test: harden Conversation settings return privacy"
```
