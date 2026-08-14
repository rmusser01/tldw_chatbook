# First-Run Provider Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a new user configure either llama.cpp or a custom OpenAI-compatible API from the wizard, discover or enter a model, understand the saved setup, and enter Console with the intended provider/model active.

**Architecture:** Keep pure commit and eligibility decisions in `first_run_setup_state.py`, screen orchestration in `FirstRunSetupWizard.py`, and active-session ownership in the Console store. First-run drafts consume the shared provider endpoint and persistence contracts; a typed, secret-free handoff either refreshes an untouched default session or creates a new session.

**Tech Stack:** Python 3.11+, Textual, httpx mock transports, pytest, pytest-asyncio, existing Console session store and pending handoff infrastructure.

## Global Constraints

- Complete the provider connection foundation plan before this plan.
- Coordinate with `Docs/superpowers/plans/2026-08-12-first-run-reliability-implementation-plan.md` before editing `FirstRunSetupWizard.py` or `_wizards.tcss`.
- Its keyboard-safe `OptionList`, required-step failure handling, selected-provider-only discovery, pinned footer, and responsive checks are required inputs to this plan; its crash-recovery draft is independent and must not be duplicated here.
- Never persist an API key in setup recovery state, handoffs, model cache keys, logs, or notifications.
- Detection never overwrites typed endpoint input without explicit selection.
- Placeholder/status rows are disabled and can never become model IDs.
- Existing user-owned Console sessions are never silently changed.
- Use `Tests/...` as the canonical test path spelling.

---

## File Structure

- Modify `tldw_chatbook/UI/Wizards/first_run_setup_state.py`: provider draft, model discovery key, summary actions, atomic commit, and untouched-session eligibility.
- Modify `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`: endpoint/auth controls, discovery results, model behavior, pinned navigation, summary hierarchy, and finish handoff.
- Modify `tldw_chatbook/css/features/_wizards.tcss`: stable endpoint/auth/discovery layout and fixed footer.
- Modify `tldw_chatbook/UI/Navigation/pending_handoff_store.py`: add a revisioned first-chat intent.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: consume the first-chat intent against the exact session/default revision.
- Modify focused wizard, handoff, and Console tests.

### Task 1: Establish the shared-wizard integration gate

**Files:**
- Verify: `Docs/superpowers/plans/2026-08-12-first-run-reliability-implementation-plan.md`
- Verify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Verify: `tldw_chatbook/css/features/_wizards.tcss`

- [ ] **Step 1: Confirm the reliability prerequisites are present**

Run: `rg -n "OptionList|setup-step-retry|setup-step-manual|setup-footer|selected_provider" tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/css/features/_wizards.tcss`

Expected: matches for an `OptionList` provider control, explicit required-step recovery actions, selected-provider discovery, and a pinned setup footer. If any are absent, execute Tasks 1, 3, 4, and 5 from the reliability plan, then rerun this check.

- [ ] **Step 2: Run the prerequisite wizard tests**

Run: `.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_first_run_wizard_live_contract.py -k "provider_down or required_provider or selected_provider or pinned or compact" -v`

Expected: PASS.

- [ ] **Step 3: Record the prerequisite commit in the implementation log**

Run: `git log -1 --oneline -- tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`

Expected: the implementer records the commit ID in the task notes before continuing; no code commit is created for this verification-only task.

- [ ] **Step 4: Commit prerequisite integration corrections when needed**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/css/features/_wizards.tcss Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_first_run_wizard_live_contract.py
git commit -m "test: integrate first-run reliability prerequisites"
```

Skip this commit when the prerequisite plan's own commits pass the integration gate without corrections.

### Task 2: Add pure first-run provider draft and commit contracts

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py`
- Modify: `Tests/Wizards/test_first_run_setup_state.py`
- Modify: `Tests/Wizards/test_first_run_setup_integration.py`

**Interfaces:**
- Produces `ProviderCredentialDraft(source, value, revision)` whose value is excluded from repr/comparison and never serialized.
- Produces `FirstRunProviderDraft(provider, endpoint, credential)`.
- Produces `FirstRunModelDiscoveryKey(provider_key, connection_identity, credential_source, credential_revision)`.
- Produces `build_first_run_provider_commit(provider_draft, model_id, app_config) -> ProviderSetupMutation`.
- Produces `FirstRunSummaryAction = Literal["start_chatting", "review_provider", "explore_home", "review_settings"]`.

- [ ] **Step 1: Write failing draft and atomic commit tests**

```python
def test_llama_full_chat_url_commit_persists_legacy_root_and_defaults():
    draft = FirstRunProviderDraft(
        provider="llama_cpp",
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        credential=ProviderCredentialDraft("none", "", 0),
    )
    mutation = build_first_run_provider_commit(draft, "local-model", {})
    assert mutation.section_values["api_settings.llama_cpp"]["api_url"] == "http://127.0.0.1:8080"
    assert mutation.section_values["chat_defaults"] == {"provider": "llama_cpp", "model": "local-model"}
    assert mutation.section_values["provider_setup.confirmed"] == {"llama_cpp": True}


def test_discovery_key_separates_same_provider_different_endpoints():
    first = build_first_run_model_discovery_key(_draft(endpoint="http://127.0.0.1:8080"))
    second = build_first_run_model_discovery_key(_draft(endpoint="http://127.0.0.1:8081"))
    assert first != second
    assert "secret" not in repr(first).lower()


def test_provider_credential_value_is_memory_only_and_repr_safe():
    credential = ProviderCredentialDraft("draft", "test-secret", 3)
    assert "test-secret" not in repr(credential)
    with pytest.raises(TypeError):
        build_setup_draft_commit(_draft(credential=credential))
```

Add state-table tests proving complete setup selects `start_chatting`, incomplete setup selects `review_provider`, and the secondary/tertiary actions remain `explore_home` and `review_settings` exactly once.

- [ ] **Step 2: Run the state tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_state.py Tests/Wizards/test_first_run_setup_integration.py -k "provider_draft or discovery_key or full_chat_url or summary_action or atomic" -v`

Expected: FAIL because provider and model commits are currently separate and the first-run connection draft is not typed.

- [ ] **Step 3: Implement the draft and delegate persistence construction**

```python
@dataclass(frozen=True, slots=True)
class ProviderCredentialDraft:
    source: Literal["none", "draft", "environment"]
    value: str = field(repr=False, compare=False)
    revision: int = 0


@dataclass(frozen=True, slots=True)
class FirstRunProviderDraft:
    provider: str
    endpoint: str
    credential: ProviderCredentialDraft = field(repr=False)


@dataclass(frozen=True, slots=True)
class FirstRunModelDiscoveryKey:
    provider_key: str
    connection_identity: tuple[str, str]
    credential_source: str
    credential_revision: int
```

Resolve the endpoint through `resolve_provider_endpoint`, reject drafts with errors, and delegate the section/deletion mapping to `build_provider_setup_mutation`. The wizard container owns `ProviderCredentialDraft` in memory between Provider and Model; only the persistence builder may read its value. Model discovery receives the value for the request but constructs its cache/evidence identity from source and revision only. Setup recovery serialization, logs, notifications, test evidence, and handoffs reject the credential record.

- [ ] **Step 4: Replace split wizard commits with one final mutation**

Provider **Continue** stages the connection draft in the wizard container without a config write. Model **Continue** supplies the selected/manual model to `SetupWizardContainer.commit_config`, which performs one `apply_settings_mutation_to_cli_config` call for provider settings, `chat_defaults`, and confirmation metadata. It mirrors the complete result into `app_config` only when `fully_applied` is true. Exiting before Model states that the staged provider step was not saved; later steps may accurately describe provider/model as committed.

Run: `.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_state.py Tests/Wizards/test_first_run_setup_integration.py -v`

Expected: PASS, including failure tests where no partial in-memory update occurs.

- [ ] **Step 5: Commit the first-run state boundary**

```bash
git add tldw_chatbook/UI/Wizards/first_run_setup_state.py tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_setup_state.py Tests/Wizards/test_first_run_setup_integration.py
git commit -m "fix: commit first-run provider setup atomically"
```

### Task 3: Make manual endpoint and optional authentication first-class

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:255-735`
- Modify: `tldw_chatbook/css/features/_wizards.tcss`
- Modify: `Tests/Wizards/test_first_run_setup_wizard.py`
- Modify: `Tests/UI/test_product_maturity_phase1_first_run.py`

**Interfaces:**
- Provider step exposes `#setup-provider-endpoint`, `#setup-provider-effective-chat`, `#setup-provider-auth-toggle`, `#setup-provider-api-key`, `#setup-provider-detect`, and `#setup-provider-test`.
- Detection results use disabled heading/status options and selectable endpoint candidates.

- [ ] **Step 1: Write failing control and non-overwrite tests**

```python
async def test_llama_provider_shows_manual_endpoint_and_optional_auth():
    step = ProviderStep(discover=_no_servers, probe=_passing_probe)
    async with step.run_test() as pilot:
        await _select_provider(pilot, "llama_cpp")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        assert endpoint.display
        assert step.query_one("#setup-provider-auth-toggle", Collapsible).title == "Authentication (optional)"


async def test_detection_does_not_replace_typed_endpoint_until_selected():
    step = ProviderStep(discover=_two_servers, probe=_passing_probe)
    async with step.run_test() as pilot:
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = "http://127.0.0.1:9999/v1/chat/completions"
        await pilot.click("#setup-provider-detect")
        await pilot.pause()
        assert endpoint.value == "http://127.0.0.1:9999/v1/chat/completions"
        assert step.query(".setup-discovered-endpoint").results_count == 2
```

Add `test_custom_provider_credentials_are_optional`, `test_effective_chat_url_is_safe`, `test_endpoint_rejects_userinfo_query_and_fragment`, and `test_connection_test_receives_exact_provider_draft`. Each test asserts the named behavior and that no raw credential appears in rendered text or captured probe metadata.

- [ ] **Step 2: Run focused tests and confirm the controls are absent**

Run: `.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_product_maturity_phase1_first_run.py -k "manual_endpoint or optional_auth or detection_does_not_replace or effective_chat" -v`

Expected: FAIL.

- [ ] **Step 3: Implement provider-specific progressive disclosure**

Render Endpoint immediately for URL-based providers. Render Authentication collapsed for llama.cpp and custom OpenAI-compatible providers with copy stating credentials are optional. Update the effective chat/models display from `ProviderEndpointResolution`; show bounded warnings inline. Detection publishes every `DiscoveredLocalServer` as a selectable result and writes the endpoint only from its selection event.

- [ ] **Step 4: Add stable layout rules and run viewport tests**

Give endpoint rows fixed labels, wrapping values, minimum control heights, and keep discovery results inside the scrollable body while navigation stays in the footer. Run:

`.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_product_maturity_phase1_first_run.py -k "provider or endpoint or compact or browser" -v`

Expected: PASS at the narrow supported size, 120x40, and 177x45.

- [ ] **Step 5: Commit first-run connection controls**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/css/features/_wizards.tcss Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_product_maturity_phase1_first_run.py
git commit -m "feat: add manual first-run provider endpoints"
```

### Task 4: Scope model discovery to the exact draft connection

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:854-1035`
- Modify: `Tests/Wizards/test_first_run_setup_wizard.py`
- Modify: `Tests/UI/test_first_run_wizard_live_contract.py`

**Interfaces:**
- Model step accepts `provider_draft: FirstRunProviderDraft`.
- Discovery cache is keyed by `FirstRunModelDiscoveryKey`.
- Manual model input remains enabled for listing-unavailable and connection-failed states.

- [ ] **Step 1: Write failing exact-connection and placeholder tests**

```python
async def test_model_step_probes_the_provider_step_draft():
    seen = []
    async def discover(*, provider, endpoint, credential_source):
        seen.append((provider, endpoint, credential_source))
        return _models("draft-model")
    step = ModelStep(provider_draft=_draft(endpoint="http://127.0.0.1:8222/v1"), discover=discover)
    async with step.run_test():
        await step.wait_for_discovery()
    assert seen == [("llama_cpp", "http://127.0.0.1:8222/v1", "none")]


async def test_listing_unavailable_status_cannot_be_selected_as_model():
    step = ModelStep(provider_draft=_draft(), discover=_models_404)
    async with step.run_test():
        await step.wait_for_discovery()
        assert step.manual_model_input.display
        assert step.selected_model_id is None
```

Also test cache separation by endpoint and credential revision, late result fencing, and manual model entry after timeout/404.

- [ ] **Step 2: Run tests and confirm catalog-only discovery fails**

Run: `.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_first_run_wizard_live_contract.py -k "provider_step_draft or listing_unavailable or discovery_cache or late_discovery" -v`

Expected: FAIL because `_load_models` currently resolves with `staged_settings=None`.

- [ ] **Step 3: Pass the exact draft and fence discovery generations**

Replace ambient configuration lookup with the typed draft. Maintain `dict[FirstRunModelDiscoveryKey, tuple[str, ...]]`; increment a generation whenever provider, endpoint, or credential revision changes; discard settled results whose generation or key no longer matches.

- [ ] **Step 4: Render honest manual fallback states**

For `model_listing_unavailable`, show “Model listing unavailable; enter the model ID used by this endpoint.” For connection failures, show the bounded category and Retry. Status rows are disabled `OptionList` options with `model_id=None`.

Run: `.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_first_run_wizard_live_contract.py -k "model or discovery or manual" -v`

Expected: PASS.

- [ ] **Step 5: Commit exact-draft model discovery**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_first_run_wizard_live_contract.py
git commit -m "fix: scope first-run model discovery to draft"
```

### Task 5: Add unambiguous summary actions and targeted first-chat handoff

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:3457-3525,3845-4050`
- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:4020-4145`
- Modify: `Tests/State/test_pending_handoff_store.py`
- Modify: `Tests/Wizards/test_first_run_setup_wizard.py`
- Modify: `Tests/UI/test_console_session_settings.py`

**Interfaces:**
- Produces `ConsoleFirstChatIntent(session_id, provider, model, config_revision)`.
- Produces `is_untouched_default_session(session, messages, draft, staged_attachments) -> bool`.
- Adds `HandoffChannel.CONSOLE_FIRST_CHAT`.

- [ ] **Step 1: Write failing eligibility, race, and hierarchy tests**

```python
def test_untouched_default_session_requires_no_user_owned_state():
    assert is_untouched_default_session(_default_session(), (), "", ())
    assert not is_untouched_default_session(_default_session(), (), "draft", ())
    assert not is_untouched_default_session(_user_settings_session(), (), "", ())


def test_first_chat_intent_contains_no_endpoint_or_credential():
    intent = ConsoleFirstChatIntent("session-1", "llama_cpp", "model-a", 17)
    assert set(asdict(intent)) == {"session_id", "provider", "model", "config_revision"}
```

Add Pilot assertions for complete and incomplete summary primary actions, unique Home/Settings actions, no Finish/Finish later buttons, and refusal when the target session or config revision changed.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_console_session_settings.py -k "first_chat or untouched_default or summary_primary or finish_later" -v`

Expected: FAIL.

- [ ] **Step 3: Implement the secret-free first-chat intent**

```python
@dataclass(frozen=True, slots=True)
class ConsoleFirstChatIntent:
    session_id: str
    provider: str
    model: str
    config_revision: int
```

Read `get_runtime_config_snapshot().generation` after the successful setup mutation and use it as `config_revision`. On **Start chatting**, reuse the active session only when the pure eligibility check passes. Otherwise create a new Console session from the latest defaults and stage its exact ID. Console acknowledges the intent only when session ID and runtime config generation still match; otherwise it notifies without mutation.

- [ ] **Step 4: Replace summary and welcome action hierarchy**

Use **Start chatting** or **Review provider setup** as the sole primary action based on completeness, then **Explore Home**, then **Review settings**. Welcome uses **Skip setup**; mid-flow exit uses **Exit setup** and existing committed-step recovery copy. Suppress redundant toasts until the wizard closes.

- [ ] **Step 5: Run handoff and first-run UI regressions**

Run: `.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_first_run_wizard_live_contract.py Tests/UI/test_console_session_settings.py Tests/UI/test_product_maturity_phase1_first_run.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the first-chat handoff**

```bash
git add tldw_chatbook/UI/Wizards/first_run_setup_state.py tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/chat_screen.py Tests/State/test_pending_handoff_store.py Tests/Wizards/test_first_run_setup_wizard.py Tests/UI/test_first_run_wizard_live_contract.py Tests/UI/test_console_session_settings.py Tests/UI/test_product_maturity_phase1_first_run.py
git commit -m "feat: hand first-run defaults to Console safely"
```

### Task 6: Run the first-run slice gate

**Files:**
- Verify only; modify failures only when introduced by this plan.

- [ ] **Step 1: Run focused lint**

Run: `.venv/bin/python -m ruff check tldw_chatbook/UI/Wizards/first_run_setup_state.py tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/chat_screen.py`

Expected: PASS.

- [ ] **Step 2: Run the complete first-run and handoff suite**

Run: `.venv/bin/python -m pytest Tests/Wizards/test_first_run_setup_state.py Tests/Wizards/test_first_run_setup_wizard.py Tests/Wizards/test_first_run_setup_integration.py Tests/UI/test_first_run_wizard_live_contract.py Tests/UI/test_product_maturity_phase1_first_run.py Tests/State/test_pending_handoff_store.py Tests/UI/test_console_session_settings.py -v`

Expected: PASS.

- [ ] **Step 3: Run source-level secret and placeholder checks**

Run: `rg -n "api_key|credential|Loading|Unavailable|No models" tldw_chatbook/UI/Wizards/first_run_setup_state.py tldw_chatbook/UI/Navigation/pending_handoff_store.py Tests/State/test_pending_handoff_store.py`

Expected: no raw credential value is present in a dataclass, cache key, handoff, persisted recovery draft, or test assertion; status strings are not assigned as model IDs.

- [ ] **Step 4: Commit gate-only corrections when needed**

```bash
git add tldw_chatbook/UI/Wizards tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/features/_wizards.tcss Tests/Wizards Tests/UI Tests/State/test_pending_handoff_store.py
git commit -m "test: close first-run provider regressions"
```

Skip this commit when the gate requires no corrections.
