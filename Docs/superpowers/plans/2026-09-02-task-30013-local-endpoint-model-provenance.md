# Local Endpoint and Model Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local model setup trustworthy by persisting required endpoints before use, labeling model provenance, and fencing every discovery or unverified-model decision to the exact draft identity.

**Architecture:** Reuse `provider_setup_persistence.py` for atomic provider endpoint/model mutations and `canonical_connection_identity()` for endpoint identity. Extend the existing source-aware `ResolvedProviderModelOption` projection with display provenance, teach `ModelSearchPicker` to render those typed groups, and bind unsaved-endpoint discovery/evidence to a monotonically increasing modal draft generation.

**Tech Stack:** Python 3.11, immutable dataclasses/StrEnum, existing provider setup/evidence/discovery services, Textual 8.x, pytest/Textual Pilot.

**Spec:** `Docs/superpowers/specs/2026-09-02-console-conversation-settings-ready-to-send-design.md`

## Global Constraints

- No new provider registry, model cache, or persistence format is introduced.
- A required unsaved endpoint is never used implicitly when the execution path cannot pin a conversation-only Base URL.
- Default persistence must complete before session mutation or modal dismissal.
- Model discovery proves only that a bounded models route responded; it never proves chat generation.
- Provider/endpoint/model comparisons use canonical helpers, never display strings.
- Raw endpoints are excluded from status text, logs, handoffs, and reprs; display uses the existing safe endpoint formatter.
- Only targeted tests run unless the user separately approves a full sweep.

---

### Task 1: Project typed model provenance

**Files:**
- Modify: `tldw_chatbook/UI/Screens/provider_model_resolution.py`
- Test: `Tests/UI/test_provider_model_resolution.py`
- Test: `Tests/Provider/test_provider_model_resolution.py`

**Interfaces:**
- Produces: `ConsoleModelProvenance` values `served_now`, `current_catalog`, `saved_fallback`, `custom_unverified`
- Extends: `ResolvedProviderModelOption` with defaulted `provenance: ConsoleModelProvenance` and `verified_for_connection: bool = False`
- Consumes: existing `MergedModelEntry.source`, cloud catalog provider identity, and exact modal discovery identity
- Preserves: `resolve_provider_model_options()` ordering, authority, and merge-cap behavior

- [ ] **Step 1: Write red provenance/precedence tests**

Cover existing merge ordering/caps, current cloud catalog versus saved fallback, a current-unlisted value, stable provenance assignment, and absence of false `served_now` claims from provider-only cached entries.

```python
options = await resolve_provider_model_options(
    {"OpenAI": ["saved-old"]},
    _FakeScope(_entries("OpenAI", ["catalog-current"])),
    provider="OpenAI",
)
assert [(item.model_id, item.provenance) for item in options] == [
    ("catalog-current", ConsoleModelProvenance.CURRENT_CATALOG),
]
assert options[0].verified_for_connection is False
```

- [ ] **Step 2: Verify tests fail for missing provenance**

Run: `pytest Tests/UI/test_provider_model_resolution.py Tests/Provider/test_provider_model_resolution.py -k 'provenance' -q`

Expected: FAIL because resolved model options expose source but no display provenance.

- [ ] **Step 3: Implement one pure precedence projection**

Map `runtime_discovered` / `persisted_discovered` entries from ADR-020 cloud providers to `current_catalog`, `saved` to `saved_fallback`, and `current_unlisted` to `custom_unverified`. Do not assign `served_now` inside the provider-only catalog resolver because `MergedModelEntry` lacks the unsaved draft endpoint identity. Keep `verified_for_connection=False` for every scope-derived option.

- [ ] **Step 4: Preserve legacy callers**

Give the new fields defaults and leave existing label/source/capability/persisted behavior unchanged. The modal may create a `served_now` option only from a successful TASK-30013 exact-identity manual probe; stale results never enter the picker.

- [ ] **Step 5: Verify projection compatibility**

Run: `pytest Tests/UI/test_provider_model_resolution.py Tests/Provider/test_provider_model_resolution.py -q`

Expected: PASS.

- [ ] **Step 6: Commit model provenance**

```bash
git add tldw_chatbook/UI/Screens/provider_model_resolution.py Tests/UI/test_provider_model_resolution.py Tests/Provider/test_provider_model_resolution.py
git commit -m "feat: project Console model provenance"
```

### Task 2: Render provenance in the searchable model picker

**Files:**
- Modify: `tldw_chatbook/Widgets/model_search_picker.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Test: `Tests/Widgets/test_model_search_picker.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Extends: `ModelSearchPicker.set_provenance_options(provider: str, options: Sequence[ResolvedProviderModelOption]) -> None`
- Produces: disabled headings `Served now`, `Current catalog`, `Saved fallback`, `Custom / unverified`
- Produces: selection identity by model ID, not rendered row index

- [ ] **Step 1: Write red grouped-result tests**

Assert headings are visible and nonselectable, selected IDs remain correct with headings interleaved, filtering retains only non-empty groups, markup-like IDs render literally, and old untyped picker callers remain unchanged.

- [ ] **Step 2: Verify picker tests fail**

Run: `pytest Tests/Widgets/test_model_search_picker.py -k 'provenance or grouped' -q`

Expected: FAIL because the picker stores a flat string list.

- [ ] **Step 3: Implement typed result rows**

Keep `_matches` as `ResolvedProviderModelOption` values when provenance is supplied. Render `Option(group_label, disabled=True)` before each group and retain an explicit `option_id -> model_id` mapping so headings cannot shift selection. Keep existing catalog loading/discovery APIs as compatibility inputs until the modal switches.

- [ ] **Step 4: Wire the modal to one merged projection**

Use the options already returned by `resolve_provider_model_options()`, then merge the current exact-identity modal probe as `ResolvedProviderModelOption(provenance="served_now", verified_for_connection=True)`. Render adjacent persistent provenance copy: `Served by this endpoint now`, `Current provider catalog`, `Saved fallback`, or `Custom model ID; generation not verified.`

- [ ] **Step 5: Verify picker and modal output**

Run: `pytest Tests/Widgets/test_model_search_picker.py Tests/UI/test_console_session_settings.py -k 'model or provenance' -q`

Expected: PASS.

- [ ] **Step 6: Commit visible provenance**

```bash
git add tldw_chatbook/Widgets/model_search_picker.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Widgets/test_model_search_picker.py Tests/UI/test_console_session_settings.py
git commit -m "feat: show model provenance in Conversation settings"
```

### Task 3: Fence discovery to the exact draft identity

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Test: `Tests/UI/test_console_local_server_discovery_card.py`
- Test: `Tests/UI/test_console_local_server_probe_isolation.py`
- Test: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces: `ConsoleModelDiscoveryIdentity(provider_key: str, connection_identity: tuple[str, str], draft_generation: int)`
- Produces: `ConsoleUnverifiedModelDecision(identity: ConsoleModelDiscoveryIdentity, model_id: str)`
- Consumes: `canonical_connection_identity()` and existing evidence identity rules

- [ ] **Step 1: Write red stale-result and confirmation tests**

Cover provider switch during probe, endpoint edit during probe, two rapid discoveries, zero/one/many results, return to a prior provider draft, custom model absent from a successful listing, exact confirmation, and invalidation on provider/endpoint/model edit.

- [ ] **Step 2: Verify isolation tests fail**

Run: `pytest Tests/UI/test_console_local_server_discovery_card.py Tests/UI/test_console_local_server_probe_isolation.py Tests/UI/test_console_session_settings.py -k 'stale or discovery_identity or unverified_model' -q`

Expected: FAIL because discovery is currently keyed by provider only.

- [ ] **Step 3: Add monotonic draft generation**

Increment the modal generation whenever provider, canonical endpoint, or selected model changes. Capture provider key, connection identity, and generation before dispatch. Apply results only when all three still match; otherwise discard without clearing newer status/evidence.

- [ ] **Step 4: Add exact unverified-model confirmation**

When a successful current-identity listing omits the selected model, block primary completion and show secondary `Keep unverified model`. Store only the exact identity/model tuple. Invalidate on every identity generation change and never serialize it as verification evidence.

- [ ] **Step 5: Make discovery copy exact**

Before request: `List models from this endpoint; this does not test generation.` On success use `No models reported`, `1 model listed`, or `<n> models listed`; on failure use bounded category copy and safe endpoint display. Never say connected, verified, or passed.

- [ ] **Step 6: Verify isolation and copy**

Run: `pytest Tests/Chat/test_provider_test_evidence.py Tests/UI/test_console_local_server_discovery_card.py Tests/UI/test_console_local_server_probe_isolation.py Tests/UI/test_console_session_settings.py -q`

Expected: PASS.

- [ ] **Step 7: Commit identity-fenced discovery**

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/UI/test_console_local_server_discovery_card.py Tests/UI/test_console_local_server_probe_isolation.py Tests/UI/test_console_session_settings.py
git commit -m "fix: fence model discovery to the current endpoint"
```

### Task 4: Persist required endpoints before applying the session

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Chat/test_provider_setup_persistence.py`
- Test: `Tests/UI/test_console_session_settings.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Produces: `ConsoleSettingsPersistAndApply(settings, provider: str, model: str, endpoint: str, expected_settings_revision: int)` with `settings` and `endpoint` excluded from repr
- Consumes: `build_provider_setup_mutation()`, `bind_provider_setup_precondition()`, `persist_provider_setup()`, and `provider_setup_draft_identity()`
- Produces: primary `Save endpoint & use model`

- [ ] **Step 1: Write red ordering/failure/conflict tests**

Assert no session mutation before successful persistence, writer failure retains modal/draft, identity conflict retains modal, successful persistence refreshes config before applying, duplicate activation is idempotent, and paths without request-pinned endpoints never offer conversation-only endpoint use.

```python
assert runtime.store.session_settings(session_id) == original
assert modal.is_mounted
assert str(modal.query_one("#console-settings-error", Static).renderable) == (
    "Could not save the endpoint. Nothing was applied to this conversation."
)
```

- [ ] **Step 2: Verify persistence journey tests fail**

Run: `pytest Tests/Chat/test_provider_setup_persistence.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py -k 'persist_and_apply or endpoint_save_order or endpoint_conflict' -q`

Expected: FAIL because the modal's defaults writer is not a typed provider-setup transaction.

- [ ] **Step 3: Build the typed mutation without writing**

The modal validates provider/model/endpoint and returns only those non-credential draft values in `ConsoleSettingsPersistAndApply`; it neither receives credentials nor calls config writers. `ChatScreen` captures expected settings revision, resolves the canonical credential source/revision from current config without exposing it to the modal, creates and binds the existing provider setup mutation against that config snapshot, and runs `persist_provider_setup()` off the application thread.

- [ ] **Step 4: Settle persistence before application**

On `ConfigMutationResult.fully_applied=True`, reload canonical config, re-resolve readiness, verify session/revision, then call `_apply_console_settings_result()` and close. On failure, conflict, deletion, or revision mismatch, leave the session untouched, retain/reopen the same draft, and render fixed recovery copy.

- [ ] **Step 5: Gate endpoint scope honestly**

Use existing request-pinning behavior to expose `Use endpoint for this conversation` only where `ConsoleProviderResolution.base_url` reaches the real adapter. Otherwise only show `Save endpoint & use model`, with adjacent `Updates this provider for future conversations.`

- [ ] **Step 6: Verify atomic behavior**

Run: `pytest Tests/Chat/test_provider_setup_persistence.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py -k 'endpoint or provider_setup or persist_and_apply' -q`

Expected: PASS.

- [ ] **Step 7: Run the focused local setup slice**

Run: `pytest Tests/Chat/test_console_session_settings.py Tests/Chat/test_provider_setup_persistence.py Tests/Chat/test_provider_test_evidence.py Tests/Widgets/test_model_search_picker.py Tests/UI/test_console_local_server_discovery_card.py Tests/UI/test_console_local_server_probe_isolation.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py -q`

Expected: PASS.

- [ ] **Step 8: Verify diff and commit**

Run: `git diff --check`

Expected: no output.

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_provider_setup_persistence.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat: save local endpoints before applying Conversation settings"
```
