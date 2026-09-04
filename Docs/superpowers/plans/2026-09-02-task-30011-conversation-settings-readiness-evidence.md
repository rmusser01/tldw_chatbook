# Conversation Settings Readiness Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate whether Chatbook can attempt a send from the configuration and network evidence actually observed for the selected provider/model.

**Architecture:** Extend the existing `ProviderReadinessSnapshot`/`ProviderTestEvidenceStore` facets and project them into an immutable Console status model. Pure precedence chooses one blocker/recovery action; Console and Settings own their display copy and consume the same sanitized facts.

**Tech Stack:** Python 3.11, dataclasses, Literal enums, existing provider-readiness/evidence services, Textual 8.x, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-02-console-conversation-settings-ready-to-send-design.md`

## Global Constraints

- `Ready to send` means Chatbook found no local blocker and permits an attempt; it never guarantees provider acceptance or generation.
- Configuration, credential, endpoint, model, and generation evidence remain independent.
- Screens own user-facing prose; domain/service results expose typed reason codes and bounded sanitized facts.
- Existing provider identity, credential, endpoint, catalog, and evidence owners are reused.
- Raw credentials, headers, response bodies, exception prose, and credential-bearing URLs are never exposed.
- Only targeted tests run unless the user separately approves a full sweep.

---

### Task 1: Extend evidence facets without breaking Settings

**Files:**
- Modify: `tldw_chatbook/Chat/provider_test_evidence.py`
- Create: `Tests/Chat/test_provider_test_evidence.py`

**Interfaces:**
- Produces: `CredentialFacet = Literal["not_required", "missing", "present_unverified", "authenticated"]`
- Produces: `GenerationFacet = Literal["not_tested", "testing", "succeeded", "failed", "changed_since_test"]`
- Produces: `GenerationFailureCategory = Literal["authentication", "rate_limit", "bad_request", "timeout", "connection_error", "provider_error"]`
- Produces: `ProviderGenerationProbeResult(generation: Literal["succeeded", "failed"], category: GenerationFailureCategory | None)`
- Extends: `ProviderReadinessSnapshot` with defaulted `credential` and `generation` fields
- Extends: `ProviderTestEvidence` with defaulted credential/generation evidence and `generation_category` while preserving existing callers
- Extends: `ProviderTestEvidenceStore` with exact-identity `begin_generation()` / `settle_generation()` tokens independent of endpoint-probe tokens

- [ ] **Step 1: Write invariant and backward-compatibility tests**

```python
def test_ready_configuration_can_retain_unverified_credential_evidence():
    snapshot = ProviderReadinessSnapshot(
        configuration="configured",
        credential="present_unverified",
        endpoint="not_tested",
        model="unconfirmed",
        generation="not_tested",
    )
    assert snapshot.credential == "present_unverified"
```

Also reject authenticated credentials with incomplete configuration, a succeeded generation with a generation failure category, and a failed generation without a bounded category. A successful generation may coexist with `endpoint="not_tested"` because the live completion itself is separate evidence; it upgrades credential evidence to `authenticated` when a credential was required.

- [ ] **Step 2: Verify tests fail for missing facets**

Run: `pytest Tests/Chat/test_provider_test_evidence.py -q`

Expected: FAIL on the new constructor fields/assertions.

- [ ] **Step 3: Implement defaulted, validated facets**

Keep current positional parameters stable. Add fields after existing defaults, validate exact literals, and preserve current begin/settle/rebase behavior for endpoint/model-only evidence. Give generation its own opaque single-use token so a connection probe and a generation probe cannot settle each other's operation; both require the same current `ProviderDraftIdentity` and draft generation.

- [ ] **Step 4: Verify evidence-store compatibility**

Run: `pytest Tests/Chat/test_provider_test_evidence.py Tests/UI/test_settings_provider_test_draft.py -q`

Expected: PASS.

- [ ] **Step 5: Commit evidence facets**

```bash
git add tldw_chatbook/Chat/provider_test_evidence.py Tests/Chat/test_provider_test_evidence.py
git commit -m "feat: separate provider verification evidence facets"
```

### Task 2: Define the Console operability projection

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py`
- Test: `Tests/Chat/test_console_session_settings.py`

**Interfaces:**
- Produces: `ConsoleOperability = Literal["ready_to_send", "not_ready"]`
- Produces: `ConsoleSettingsBlockerCode` for provider, credential, endpoint syntax/persistence/reachability, model, unsupported provider, and active-run blockers
- Produces: `ConsoleSettingsRecoveryAction` enum-like literal values
- Extends: `ConsoleSettingsReadiness` with `operability`, `blocker`, `recovery_action`, `provider_display_name`, and evidence facets while retaining `label`, `detail`, and `native_send_supported` compatibility properties

- [ ] **Step 1: Write the precedence table as parameterized red tests**

```python
@pytest.mark.parametrize(
    ("settings", "expected_blocker"),
    [
        (_settings(provider=""), "provider_missing"),
        (_settings(provider="openai", model=None), "credential_missing"),
        (_settings(provider="llama_cpp", base_url="bad"), "endpoint_invalid"),
    ],
)
def test_console_readiness_selects_one_highest_priority_blocker(settings, expected_blocker):
    assert build_console_settings_readiness(settings, app_config={}).blocker == expected_blocker
```

- [ ] **Step 2: Verify precedence tests fail**

Run: `pytest Tests/Chat/test_console_session_settings.py -k 'operability or blocker or verification' -q`

Expected: FAIL because the projection lacks the new fields.

- [ ] **Step 3: Implement one pure projection**

Resolve in this order: provider identity → supported execution → endpoint syntax → required endpoint persistence → credential presence → model presence → known endpoint failure → ready. Attach evidence independently and use `provider_display_name()` for display identity. Do not parse display prose to infer state.

- [ ] **Step 4: Verify compatibility and warning behavior**

Run: `pytest Tests/Chat/test_console_session_settings.py Tests/Chat/test_provider_readiness.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Console operability**

```bash
git add tldw_chatbook/Chat/console_session_settings.py Tests/Chat/test_console_session_settings.py
git commit -m "feat: project Console operability separately from evidence"
```

### Task 3: Render one honest status across Console surfaces

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_summary.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_session_settings.py`
- Test: `Tests/UI/test_console_rail_sections.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Consumes: extended `ConsoleSettingsReadiness`
- Produces: fixed screen-owned copy maps from blocker/evidence codes to labels, details, and primary recovery labels

- [ ] **Step 1: Write cross-surface consistency tests**

For each canonical state, assert modal, rail summary, blocked setup card, and send gate agree on operability while preserving evidence qualifiers such as `credential not verified`, `endpoint unreachable`, and `generation not tested`.

- [ ] **Step 2: Verify UI tests fail on prose-derived state**

Run: `pytest Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_native_chat_flow.py -k 'operability or verification_evidence or readiness_copy' -q`

Expected: FAIL until all surfaces consume typed fields.

- [ ] **Step 3: Replace prose parsing with typed rendering**

Remove `_is_ready_readiness_detail()` and any label/detail substring tests. Render the primary line from `operability`, evidence rows from facets, and one recovery action from `recovery_action`. Preserve detailed fixed copy in the screen/widget layer.

- [ ] **Step 4: Verify Console consistency**

Run: `pytest Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_native_chat_flow.py -k 'readiness or blocked or provider' -q`

Expected: PASS.

- [ ] **Step 5: Commit cross-surface rendering**

```bash
git add tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/Widgets/Console/console_settings_summary.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat: render honest Console readiness evidence"
```

### Task 4: Align Settings test outcomes and return copy

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_provider_test_draft.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

**Interfaces:**
- Consumes: `ProviderReadinessSnapshot`, `ProviderTestEvidence`, and typed return outcomes from TASK-30010
- Produces: separate local configuration verdict and live endpoint/model evidence rows

- [ ] **Step 1: Write tests for non-overclaiming Settings results**

Assert a cloud local-only check never says passed/verified, a reachable model endpoint does not claim generation, a refused endpoint remains failed despite valid fields, stale evidence is marked stale, and save-result return copy reflects credential-only versus broader mutations.

- [ ] **Step 2: Verify the new result tests fail**

Run: `pytest Tests/UI/test_settings_provider_test_draft.py Tests/UI/test_settings_configuration_hub.py -k 'evidence or overclaim or return_copy' -q`

Expected: FAIL on current composite `Provider test passed` copy.

- [ ] **Step 3: Render Settings from facets and fixed copy**

Keep `_provider_readiness_test_report()` local-only, label it `Configuration check`, and append live evidence only when the evidence store contains an exact identity match. Preserve existing save-lease rebasing and redaction.

- [ ] **Step 4: Run the complete focused readiness slice**

Run: `pytest Tests/Chat/test_provider_test_evidence.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_session_settings.py Tests/UI/test_settings_provider_test_draft.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_native_chat_flow.py -q`

Expected: PASS.

- [ ] **Step 5: Verify diff and commit**

Run: `git diff --check`

Expected: no output.

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_provider_test_draft.py Tests/UI/test_settings_configuration_hub.py
git commit -m "fix: keep provider readiness claims evidence-scoped"
```
