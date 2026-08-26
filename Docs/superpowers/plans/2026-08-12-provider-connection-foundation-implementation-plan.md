# Provider Connection Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give first-run setup, Settings, model discovery, and Console execution one compatibility-preserving endpoint, readiness, evidence, and persistence contract.

**Architecture:** Add pure typed contracts under `tldw_chatbook/Chat`, retain provider-specific persisted shapes through adapters, and route existing discovery and Settings probes through those contracts. Persist provider settings, the default provider/model pair, and explicit setup provenance with the existing atomic configuration mutation API.

**Tech Stack:** Python 3.11+, dataclasses, `urllib.parse`, httpx, Textual state models, pytest, TOML-backed configuration.

## Global Constraints

- Baseline is `origin/dev` at `5414d811b8720c1c32c5813f96925a82c60c5f72`.
- Do not introduce named connections or migrate existing configuration.
- Preserve proxy prefixes and provider-specific endpoint keys.
- Never store secrets, secret hashes, query strings, fragments, or response bodies in evidence or user-visible diagnostics.
- Scheme-less input is valid only for `localhost` and `127.0.0.1`.
- A models-route result is not a chat-completion test.
- Evidence is process-local and latest-generation-wins.
- Use `Tests/...` as the canonical test path spelling.

---

## File Structure

- Create `tldw_chatbook/Chat/provider_endpoint_contract.py`: endpoint parsing, route derivation, safe display, and provider persistence adapters.
- Create `tldw_chatbook/Chat/provider_test_evidence.py`: readiness facets, semantic draft identity, evidence, and verdict computation.
- Create `tldw_chatbook/Chat/provider_setup_persistence.py`: provider/model key mapping, provenance, and atomic commit construction.
- Modify `tldw_chatbook/Chat/local_server_discovery.py`: consume resolved models URLs rather than append route strings.
- Modify `tldw_chatbook/Chat/console_provider_endpoints.py`: delegate effective endpoint and display behavior to the shared contract.
- Modify `tldw_chatbook/Chat/provider_readiness.py`: expose structured readiness while preserving compatibility properties needed by existing callers.
- Modify `tldw_chatbook/UI/Screens/settings_endpoint_probe.py`: return bounded structured model-listing outcomes.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: consume the shared draft identity and atomic provider commit.
- Modify `tldw_chatbook/config.py`: no new writer; use `apply_settings_mutation_to_cli_config` as the sole mutation boundary.
- Add or modify the focused tests named in each task.

### Task 1: Define and test endpoint interpretation

**Files:**
- Create: `tldw_chatbook/Chat/provider_endpoint_contract.py`
- Create: `Tests/Chat/test_provider_endpoint_contract.py`

**Interfaces:**
- Produces `EndpointForm = Literal["origin", "api_base", "chat_url", "models_url", "legacy_local"]`.
- Produces `ProviderEndpointResolution` with `provider_key`, `normalized_input`, `persisted_endpoint`, `chat_url`, `models_url`, safe displays, form, warnings, and errors.
- Produces `resolve_provider_endpoint(provider: str, value: object) -> ProviderEndpointResolution`.
- Produces `canonical_connection_identity(provider: str, value: object) -> tuple[str, str] | None`.

- [ ] **Step 1: Write the endpoint table tests**

```python
@pytest.mark.parametrize(
    ("provider", "entered", "persisted", "chat", "models", "form"),
    [
        ("custom", "http://127.0.0.1:9000", "http://127.0.0.1:9000/v1/chat/completions", "http://127.0.0.1:9000/v1/chat/completions", "http://127.0.0.1:9000/v1/models", "origin"),
        ("custom", "http://127.0.0.1:9000/v1", "http://127.0.0.1:9000/v1/chat/completions", "http://127.0.0.1:9000/v1/chat/completions", "http://127.0.0.1:9000/v1/models", "api_base"),
        ("custom", "https://example.test/proxy/v1/chat/completions", "https://example.test/proxy/v1/chat/completions", "https://example.test/proxy/v1/chat/completions", "https://example.test/proxy/v1/models", "chat_url"),
        ("custom", "https://example.test/proxy/v1/models", "https://example.test/proxy/v1/chat/completions", "https://example.test/proxy/v1/chat/completions", "https://example.test/proxy/v1/models", "models_url"),
        ("llama_cpp", "http://127.0.0.1:8080/v1/chat/completions", "http://127.0.0.1:8080", "http://127.0.0.1:8080/v1/chat/completions", "http://127.0.0.1:8080/v1/models", "chat_url"),
        ("llama_cpp", "http://127.0.0.1:8080/completion", "http://127.0.0.1:8080", "http://127.0.0.1:8080/v1/chat/completions", "http://127.0.0.1:8080/v1/models", "legacy_local"),
    ],
)
def test_resolve_provider_endpoint_table(provider, entered, persisted, chat, models, form):
    result = resolve_provider_endpoint(provider, entered)
    assert result.errors == ()
    assert (result.persisted_endpoint, result.chat_url, result.models_url, result.form) == (persisted, chat, models, form)
```

Add rejection tests for remote scheme-less hosts, userinfo, query strings, fragments, encoded suffix delimiters, repeated `/v1` suffixes, and unsupported schemes. Assert remote explicit HTTP returns one warning rather than an error.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/Chat/test_provider_endpoint_contract.py -v`

Expected: FAIL because the contract module does not exist.

- [ ] **Step 3: Implement strict suffix parsing and provider adapters**

```python
@dataclass(frozen=True, slots=True)
class ProviderEndpointResolution:
    provider_key: str
    normalized_input: str
    persisted_endpoint: str | None
    chat_url: str | None
    models_url: str | None
    persisted_display: str
    chat_display: str
    models_display: str
    form: EndpointForm | None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


_OPENAI_SUFFIXES = (
    ("/v1/chat/completions", "chat_url"),
    ("/v1/models", "models_url"),
    ("/v1", "api_base"),
)
```

Parse with `urlsplit`, reject userinfo/query/fragment before normalization, identify exactly one terminal suffix, and preserve every path segment before it. For a generic OpenAI-compatible provider, persist `chat_url`; for `llama_cpp`/`local_llamacpp`, persist the root before the API suffix. Use `safe_endpoint_display` logic that reconstructs only scheme, hostname, port, and path.

- [ ] **Step 4: Prove semantic equivalence and safety**

Add assertions that whitespace/trailing slash/base/full-chat forms produce the same `canonical_connection_identity`, proxy prefixes remain distinct, and every invalid result has `persisted_endpoint is None`. Run:

`.venv/bin/python -m pytest Tests/Chat/test_provider_endpoint_contract.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the endpoint contract**

```bash
git add tldw_chatbook/Chat/provider_endpoint_contract.py Tests/Chat/test_provider_endpoint_contract.py
git commit -m "feat: define shared provider endpoint contract"
```

### Task 2: Route discovery and Settings probes through derived URLs

**Files:**
- Modify: `tldw_chatbook/Chat/local_server_discovery.py:105-138,385-497`
- Modify: `tldw_chatbook/Chat/console_provider_endpoints.py`
- Modify: `tldw_chatbook/UI/Screens/settings_endpoint_probe.py`
- Modify: `Tests/Chat/test_local_server_discovery.py`
- Create: `Tests/UI/test_settings_endpoint_probe.py`

**Interfaces:**
- `probe_models_endpoint` consumes `ProviderEndpointResolution.models_url`.
- `SettingsEndpointProbeOutcome` exposes `state`, `category`, `model_ids`, and `summary`.
- `normalize_probe_base_url` remains as a compatibility wrapper returning the resolved persistence root for old local callers.

- [ ] **Step 1: Write failing route and outcome tests**

```python
async def test_settings_probe_full_chat_url_uses_sibling_models_route():
    seen = []
    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/proxy/v1/chat/completions",
            provider="custom",
            http_client=client,
        )
    assert seen == ["https://example.test/proxy/v1/models"]
    assert outcome.state == "model_listing_unavailable"
    assert outcome.summary == "Model listing unavailable; chat endpoint not tested"
```

Also cover `200` with model IDs, `401`, `403`, timeout, refused, malformed JSON, and Ollama `/api/tags` fallback. Assert no summary contains the entered URL or exception text.

- [ ] **Step 2: Run focused tests and observe the doubled suffix regression**

Run: `.venv/bin/python -m pytest Tests/Chat/test_local_server_discovery.py Tests/UI/test_settings_endpoint_probe.py -v`

Expected: FAIL because the current helpers append `/v1/models` to a chat-completions URL and classify every non-2xx response as unreachable.

- [ ] **Step 3: Implement structured model-listing outcomes**

```python
EndpointProbeState = Literal["reachable", "unreachable", "model_listing_unavailable"]

@dataclass(frozen=True, slots=True)
class SettingsEndpointProbeOutcome:
    state: EndpointProbeState
    summary: str
    category: str | None = None
    model_ids: tuple[str, ...] = ()

    @property
    def reachable(self) -> bool:
        return self.state == "reachable"
```

Use only bounded categories: `timeout`, `connection_refused`, `unauthorized`, `forbidden`, `http_status`, `invalid_payload`, and `connection_error`. Disable redirects on every probe so a validated destination cannot redirect the request elsewhere. Classify `404` on the derived models route as `model_listing_unavailable`; do not claim chat reachability.

- [ ] **Step 4: Replace compatibility helper internals and run regressions**

Delegate `effective_provider_endpoint`, `safe_endpoint_display`, and generic comparison normalization to the shared contract while preserving their signatures. Run:

`.venv/bin/python -m pytest Tests/Chat/test_provider_endpoint_contract.py Tests/Chat/test_local_server_discovery.py Tests/UI/test_settings_endpoint_probe.py Tests/UI/test_console_local_server_discovery_card.py -v`

Expected: PASS.

- [ ] **Step 5: Commit probe integration**

```bash
git add tldw_chatbook/Chat/local_server_discovery.py tldw_chatbook/Chat/console_provider_endpoints.py tldw_chatbook/UI/Screens/settings_endpoint_probe.py Tests/Chat/test_local_server_discovery.py Tests/UI/test_settings_endpoint_probe.py Tests/UI/test_console_local_server_discovery_card.py
git commit -m "fix: derive provider discovery routes safely"
```

### Task 3: Model structured readiness and process-local test evidence

**Files:**
- Create: `tldw_chatbook/Chat/provider_test_evidence.py`
- Modify: `tldw_chatbook/Chat/provider_readiness.py`
- Modify: `Tests/Chat/test_provider_readiness.py`
- Modify: `Tests/UI/test_settings_provider_test_draft.py`

**Interfaces:**
- Produces `ProviderReadinessSnapshot` with configuration, endpoint, and model facets.
- Produces `ProviderDraftIdentity` and `ProviderTestEvidence` without secret values or hashes.
- Produces `ProviderTestEvidenceStore.begin`, `.settle`, `.evidence_for`, and `.invalidate`.
- Produces `provider_readiness_verdict(snapshot) -> ProviderReadinessVerdict`.

- [ ] **Step 1: Write facet, identity, and race tests**

```python
def test_models_404_never_becomes_verified_or_connection_failed():
    snapshot = ProviderReadinessSnapshot(
        configuration="configured",
        endpoint="model_listing_unavailable",
        model="unconfirmed",
    )
    verdict = provider_readiness_verdict(snapshot)
    assert verdict.code == "model_listing_unavailable"
    assert "chat" in verdict.detail.lower()


def test_late_test_result_cannot_attach_to_newer_draft():
    store = ProviderTestEvidenceStore()
    first = store.begin(_identity(endpoint="http://127.0.0.1:8001", draft_generation=1))
    store.begin(_identity(endpoint="http://127.0.0.1:8002", draft_generation=2))
    assert not store.settle(first, _probe(model_ids=("model-a",)))
```

Assert evidence contains credential source kind and in-memory credential revision, but its dataclass fields and `repr` contain no credential value or digest field.

- [ ] **Step 2: Run the readiness tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/Chat/test_provider_readiness.py Tests/UI/test_settings_provider_test_draft.py -v`

Expected: FAIL because readiness is boolean-oriented and the Settings test state is tied to raw widget comparisons.

- [ ] **Step 3: Implement the pure readiness and evidence records**

```python
ConfigurationFacet = Literal["incomplete", "configured"]
EndpointFacet = Literal["not_tested", "testing", "reachable", "unreachable", "model_listing_unavailable"]
ModelFacet = Literal["missing", "confirmed", "unconfirmed"]

@dataclass(frozen=True, slots=True)
class ProviderDraftIdentity:
    provider_key: str
    connection_identity: tuple[str, str]
    credential_source: Literal["none", "stored", "environment", "draft"]
    credential_revision: int
    draft_generation: int

@dataclass(frozen=True, slots=True)
class ProviderTestEvidence:
    identity: ProviderDraftIdentity
    endpoint: EndpointFacet
    model_ids: tuple[str, ...]
    category: str | None = None
```

Keep compatibility properties such as `ready` and `reason` on `ProviderReadiness`, but derive them from the facets. Request-time readiness must call the existing environment credential resolver each time.

- [ ] **Step 4: Add semantic-save evidence tests**

Test that saving an equivalent base/full URL preserves evidence only after `ConfigMutationResult.fully_applied` is true; changing provider, canonical connection identity, or credential source invalidates it; selecting a model returned by the evidence does not. Run:

`.venv/bin/python -m pytest Tests/Chat/test_provider_readiness.py Tests/UI/test_settings_provider_test_draft.py -v`

Expected: PASS with no assertion that a models `404` proves or disproves chat.

- [ ] **Step 5: Commit readiness and evidence**

```bash
git add tldw_chatbook/Chat/provider_test_evidence.py tldw_chatbook/Chat/provider_readiness.py Tests/Chat/test_provider_readiness.py Tests/UI/test_settings_provider_test_draft.py
git commit -m "feat: separate provider readiness evidence"
```

### Task 4: Add provider/model ownership accessors and atomic setup commits

**Files:**
- Create: `tldw_chatbook/Chat/provider_setup_persistence.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:8167-8795,17475-17740`
- Modify: `Tests/UI/test_settings_provider_switch_atomic.py`
- Create: `Tests/Chat/test_provider_setup_persistence.py`
- Modify: `Tests/test_config_console_defaults.py`

**Interfaces:**
- Produces `provider_endpoint_key(provider)`, `provider_model_key(provider)`, and `provider_credential_keys(provider)`.
- Produces `resolve_remembered_provider_model(app_config, provider) -> str | None`.
- Produces `build_provider_setup_mutation(draft, app_config) -> ProviderSetupMutation`.
- Uses `[provider_setup.confirmed].<provider_key> = true` as secret-free provenance.

- [ ] **Step 1: Write precedence, repair, and atomicity tests**

```python
def test_matching_chat_default_wins_over_legacy_provider_model():
    config = {
        "chat_defaults": {"provider": "custom", "model": "new-model"},
        "api_settings": {"custom": {"model": "old-model"}},
    }
    assert resolve_remembered_provider_model(config, "custom") == "new-model"


def test_setup_mutation_updates_pair_and_confirmation_together():
    mutation = build_provider_setup_mutation(_draft(provider="llama_cpp", model="qwen"), {})
    assert mutation.section_values["chat_defaults"] == {"provider": "llama_cpp", "model": "qwen"}
    assert mutation.section_values["provider_setup.confirmed"] == {"llama_cpp": True}
    assert mutation.section_values["api_settings.llama_cpp"]["api_url"] == "http://127.0.0.1:8080"
```

Add a failure test proving in-memory Settings values do not change when the atomic writer returns `ConfigMutationResult(False, False, "before_replace")` or `ConfigMutationResult(True, False, "cache_reload")`.

Add `test_config_without_confirmation_uses_legacy_readiness_heuristic` to prove existing installations remain readable without `[provider_setup.confirmed]`, and `test_template_endpoint_without_user_acceptance_is_not_explicitly_configured` to cover a fresh profile.

- [ ] **Step 2: Run focused tests and confirm split writes fail**

Run: `.venv/bin/python -m pytest Tests/Chat/test_provider_setup_persistence.py Tests/UI/test_settings_provider_switch_atomic.py Tests/test_config_console_defaults.py -v`

Expected: FAIL because the accessors and one-shot setup mutation do not exist and Settings currently saves defaults and provider values separately.

- [ ] **Step 3: Implement explicit key mappings and mutation construction**

```python
@dataclass(frozen=True, slots=True)
class ProviderSetupMutation:
    section_values: Mapping[str, Mapping[str, object]]
    delete_keys: Mapping[str, tuple[str, ...]]
    semantic_identity: ProviderDraftIdentity


def persist_provider_setup(mutation: ProviderSetupMutation) -> ConfigMutationResult:
    return apply_settings_mutation_to_cli_config(
        mutation.section_values,
        delete_keys=mutation.delete_keys,
    )
```

The provider key table must encode established keys already read by the application, including `api_url` for llama.cpp and the current custom OpenAI-compatible endpoint/model keys. Clearing an endpoint adds the provider confirmation key to `delete_keys`; unrelated provider confirmations remain untouched.

- [ ] **Step 4: Replace Settings split writes and verify explicit repair only**

Build one mutation from the validated draft, call the atomic writer once, refresh in-memory state only when `fully_applied`, and preserve exact semantic test evidence. Add assertions that unrelated Settings saves do not repair provider model values. Run:

`.venv/bin/python -m pytest Tests/Chat/test_provider_setup_persistence.py Tests/UI/test_settings_provider_switch_atomic.py Tests/UI/test_settings_provider_test_draft.py Tests/test_config_console_defaults.py -v`

Expected: PASS.

- [ ] **Step 5: Commit atomic provider persistence**

```bash
git add tldw_chatbook/Chat/provider_setup_persistence.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Chat/test_provider_setup_persistence.py Tests/UI/test_settings_provider_switch_atomic.py Tests/UI/test_settings_provider_test_draft.py Tests/test_config_console_defaults.py
git commit -m "fix: save provider defaults atomically"
```

### Task 5: Run the foundation gate

**Files:**
- Verify only; modify failing files only when the failure is caused by this plan.

- [ ] **Step 1: Run formatting and static checks on touched modules**

Run: `.venv/bin/python -m ruff check tldw_chatbook/Chat/provider_endpoint_contract.py tldw_chatbook/Chat/provider_test_evidence.py tldw_chatbook/Chat/provider_setup_persistence.py tldw_chatbook/Chat/local_server_discovery.py tldw_chatbook/Chat/console_provider_endpoints.py tldw_chatbook/Chat/provider_readiness.py tldw_chatbook/UI/Screens/settings_endpoint_probe.py`

Expected: PASS.

- [ ] **Step 2: Run the provider foundation suite**

Run: `.venv/bin/python -m pytest Tests/Chat/test_provider_endpoint_contract.py Tests/Chat/test_local_server_discovery.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_provider_setup_persistence.py Tests/UI/test_settings_endpoint_probe.py Tests/UI/test_settings_provider_test_draft.py Tests/UI/test_settings_provider_switch_atomic.py Tests/UI/test_console_local_server_discovery_card.py Tests/test_config_console_defaults.py -v`

Expected: PASS.

- [ ] **Step 3: Inspect the final diff for forbidden data and duplicate normalizers**

Run: `rg -n "api_key|credential.*hash|/v1/models|/v1/chat/completions" tldw_chatbook/Chat/provider_test_evidence.py tldw_chatbook/Chat/local_server_discovery.py tldw_chatbook/UI/Screens/settings_endpoint_probe.py`

Expected: evidence contains only credential source/revision fields; route literals are centralized in the endpoint contract except Ollama's provider-specific `/api/tags` fallback.

- [ ] **Step 4: Commit gate-only corrections when needed**

```bash
git add tldw_chatbook Tests
git commit -m "test: close provider foundation regressions"
```

Skip this commit when the gate requires no corrections.
