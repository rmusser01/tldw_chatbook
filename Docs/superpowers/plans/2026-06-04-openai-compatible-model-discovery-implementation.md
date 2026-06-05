# OpenAI-Compatible Model Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users manually discover models from configured OpenAI-compatible provider endpoints, use discovered models in Settings and Console immediately, and explicitly persist selected discovered models into the existing `[providers]` model list.

**Architecture:** Add a local model discovery substrate inside `LLM_Provider_Catalog` that reuses existing Console provider identity/readiness rules, stores discovery results in an app-lifetime cache, and writes persisted model IDs back to the exact existing top-level `[providers]` key. Settings owns the discover/persist workflow; Console consumes the merged saved plus discovered model list without creating a parallel provider registry.

**Tech Stack:** Python 3.11+, Textual, `httpx`, dataclasses/Pydantic-style validation patterns already used in the repo, TOML config via `tldw_chatbook/config.py`, pytest mounted UI tests.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-04-openai-compatible-model-discovery-prd-design.md`
- ADR workflow: `backlog/decisions/README.md`
- Existing provider catalog: `tldw_chatbook/LLM_Provider_Catalog/`
- Existing Console provider identity: `tldw_chatbook/Chat/console_provider_support.py`
- Existing provider readiness: `tldw_chatbook/Chat/provider_readiness.py`
- Existing llama.cpp URL normalization: `tldw_chatbook/Chat/console_session_settings.py`
- Existing Settings hub: `tldw_chatbook/UI/Screens/settings_screen.py`
- Existing Console model selection paths: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Screens/provider_model_resolution.py`

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/002-openai-compatible-model-discovery.md`

Reason: This feature defines durable provider/config/catalog ownership boundaries, model ID persistence rules, endpoint discovery behavior, and local-vs-server credential scope. Future contributors are likely to ask whether model discovery belongs in Settings, the provider catalog, Console, or server sync; the ADR should make that boundary explicit.

## File Structure

- Create: `backlog/decisions/002-openai-compatible-model-discovery.md`
  - Records the local v1 discovery boundary, exact `[providers]` persistence rule, runtime-cache-first behavior, and server/keyring exclusions.
- Modify: `backlog/tasks/`
  - Create or update a Backlog task for this implementation before writing code. Link the ADR and this plan.
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_contracts.py`
  - Defines immutable discovery result/value objects, status enums, error categories, and safe metadata shape.
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py`
  - Resolves a requested provider to an exact top-level `[providers]` list key using existing Console identity normalization, and detects missing or ambiguous matches.
- Create: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`
  - Builds `/v1/models` URLs from configured endpoints, validates OpenAI-compatible eligibility, performs manual HTTP discovery, normalizes provider responses, and redacts sensitive data.
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_cache.py`
  - Stores discovered models for the app session, keyed by provider list key and endpoint fingerprint.
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_merge.py`
  - Merges saved and discovered model IDs for selectors, computes `known|inferred|unknown` capability status, and preserves saved model order.
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_persistence.py`
  - Persists selected discovered raw model IDs to the exact top-level `[providers].<provider_list_key>` list, deduping without deleting existing saved models.
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py`
  - Wires local discovery, cache, merged model listing, and persistence helper calls into the existing local provider catalog surface.
- Modify: `tldw_chatbook/LLM_Provider_Catalog/llm_provider_catalog_scope_service.py`
  - Exposes local-only discovery through the existing scope service and policy action boundary.
- Modify: `tldw_chatbook/LLM_Provider_Catalog/__init__.py`
  - Exports only the public discovery contracts needed by Settings and tests.
- Modify: `tldw_chatbook/config.py`
  - Add or reuse config-write helper logic for exact top-level `[providers]` updates, preserving key spelling/casing.
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
  - Adds Settings Providers & Models discovery/persist controls, status banners, warning copy, and recovery states.
- Modify: `tldw_chatbook/UI/Screens/provider_model_resolution.py`
  - Ensures selector model lists include discovered models with warning metadata when available.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  - Ensures Console uses the same merged saved+discovered provider model list and displays unknown capability warnings consistently.
- Test: `Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py`
- Test: `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`
- Test: `Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`
- Test: `Tests/UI/test_console_session_settings.py`

## Implementation Rules

- Use `superpowers:test-driven-development` for every task that changes behavior.
- Use `superpowers:verification-before-completion` before any final status, commit, PR, or "done" claim.
- Do not create a parallel provider registry.
- Do not change configured provider names or model IDs when persisting.
- Do not write discovered models to config automatically after discovery.
- Do not support server keyring credentials or tldw_server catalog sync in v1.
- Do not query native Ollama `/api/tags` or native koboldcpp `/api/v1/generate` in v1.
- Do not infer `known` capabilities solely because an endpoint reports `vision=false`.
- Do not use local-machine absolute paths in docs or verification commands.

---

### Task 1: Backlog Task and ADR Setup

**Files:**
- Create: `backlog/decisions/002-openai-compatible-model-discovery.md`
- Modify: `backlog/tasks/task-78 - OpenAI-Compatible-Model-Discovery.md`
- Modify: `Docs/superpowers/plans/2026-06-04-openai-compatible-model-discovery-implementation.md` if plan deviations are discovered

- [ ] **Step 1: Read existing ADR and task conventions**

Run:

```bash
sed -n '1,220p' backlog/decisions/README.md
sed -n '1,220p' backlog/decisions/001-adopt-backlog-decisions-as-canonical-adrs.md
backlog task list --plain
```

Expected: You understand the ADR format and confirm `TASK-78` is the implementation task.

- [ ] **Step 2: Verify the Backlog task and move it to In Progress**

Run:

```bash
backlog task 78 --plain
backlog task edit 78 -s "In Progress"
```

Expected: `backlog/tasks/task-78 - OpenAI-Compatible-Model-Discovery.md` exists and the task is in progress before code changes begin.

- [ ] **Step 3: Add the ADR**

Create `backlog/decisions/002-openai-compatible-model-discovery.md` with:

```markdown
# ADR 002: OpenAI-Compatible Model Discovery

## Status

Accepted

## Context

Users need to discover models exposed by configured OpenAI-compatible endpoints without waiting for app or config updates. Existing Chatbook model selection is backed by the top-level `[providers]` config and Console readiness/execution provider identity logic.

## Decision

Chatbook v1 model discovery is local, manual, and scoped to existing configured OpenAI-compatible providers. Discovery results are stored in a runtime cache first. Users must explicitly persist selected raw model IDs into the exact existing top-level `[providers].<provider-list-key>` entry. Settings owns discovery and persistence. Console consumes the same merged saved and discovered model list.

Provider identity and persistence must reuse Console provider normalization. If multiple top-level `[providers]` keys normalize to the same provider identity, persistence is refused until the user resolves the ambiguity.

## Consequences

Discovery does not create a new provider registry and does not auto-save endpoint results. Unknown discovered models remain usable with capability warnings. Native provider-specific discovery paths, server keyring credentials, and tldw_server catalog sync are deferred to later ADRs.
```

Expected: ADR is explicit about ownership and deferrals.

- [ ] **Step 4: Add task implementation plan**

Run:

```bash
backlog task edit 78 --plan "ADR required: yes
ADR path: backlog/decisions/002-openai-compatible-model-discovery.md
Reason: Defines provider/config/catalog boundaries and persistence policy.

1. Add discovery contracts and provider key resolution tests.
2. Add OpenAI-compatible endpoint discovery parsing and safe error handling.
3. Add runtime cache merge and persistence helpers.
4. Wire the local provider catalog and scope service.
5. Add Settings discover and save workflow.
6. Add Console merged model consumption and warnings.
7. Run focused tests and manual UI QA."
```

Expected: Task plan includes the ADR check.

- [ ] **Step 5: Verify and commit setup**

Run:

```bash
git diff --check
git status --short
```

Expected: Only the ADR, `backlog/tasks/task-78 - OpenAI-Compatible-Model-Discovery.md`, and any plan adjustment are changed.

Commit:

```bash
git add backlog/decisions/002-openai-compatible-model-discovery.md "backlog/tasks/task-78 - OpenAI-Compatible-Model-Discovery.md" Docs/superpowers/plans/2026-06-04-openai-compatible-model-discovery-implementation.md
git commit -m "Document model discovery implementation boundary"
```

---

### Task 2: Discovery Contracts and Provider List Key Resolution

**Files:**
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_contracts.py`
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py`
- Test: `Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py`

- [ ] **Step 1: Write failing provider list key resolution tests**

Create `Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py` with tests covering:

```python
def test_resolves_exact_top_level_provider_key_for_openrouter():
    providers = {"OpenRouter": ["openrouter/auto"], "OpenAI": ["gpt-4.1"]}
    result = resolve_provider_list_key("openrouter", providers)
    assert result.status == "resolved"
    assert result.provider_list_key == "OpenRouter"


def test_preserves_custom_2_key_spelling():
    providers = {"Custom_2": ["existing-model"]}
    result = resolve_provider_list_key("custom_2", providers)
    assert result.status == "resolved"
    assert result.provider_list_key == "Custom_2"


def test_ambiguous_normalized_provider_keys_refuse_persistence():
    providers = {"Custom": ["a"], "custom": ["b"]}
    result = resolve_provider_list_key("custom", providers)
    assert result.status == "ambiguous"
    assert sorted(result.matches) == ["Custom", "custom"]


def test_missing_provider_key_reports_missing():
    result = resolve_provider_list_key("openrouter", {"OpenAI": ["gpt-4.1"]})
    assert result.status == "missing"
    assert result.provider_list_key is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py --tb=short
```

Expected: FAIL because modules/functions do not exist.

- [ ] **Step 3: Implement contract dataclasses**

Create `tldw_chatbook/LLM_Provider_Catalog/model_discovery_contracts.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DiscoverySource = Literal["saved", "runtime_discovered", "persisted_discovered"]
CapabilityStatus = Literal["known", "inferred", "unknown"]
ProviderKeyResolutionStatus = Literal["resolved", "missing", "ambiguous"]
DiscoveryErrorKind = Literal[
    "unsupported_endpoint",
    "missing_endpoint",
    "missing_credentials",
    "ambiguous_provider_key",
    "request_failed",
    "invalid_response",
]


@dataclass(frozen=True)
class ProviderModelListKeyResolution:
    requested_provider: str
    normalized_provider: str
    provider_list_key: str | None
    status: ProviderKeyResolutionStatus
    matches: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveredModel:
    provider: str
    provider_list_key: str
    model_id: str
    display_name: str
    source: DiscoverySource
    endpoint_fingerprint: str
    discovered_at: str
    metadata_raw_safe: dict[str, Any] = field(default_factory=dict)
    capability_status: CapabilityStatus = "unknown"
    persisted: bool = False


@dataclass(frozen=True)
class ModelDiscoveryError:
    kind: DiscoveryErrorKind
    message: str
    recovery_hint: str


@dataclass(frozen=True)
class ModelDiscoveryResult:
    provider: str
    provider_list_key: str | None
    endpoint_fingerprint: str | None
    status: Literal["success", "unsupported", "error"]
    models: tuple[DiscoveredModel, ...] = ()
    error: ModelDiscoveryError | None = None
    policy_action: str = "llm.catalog.models.discover.local"


@dataclass(frozen=True)
class MergedModelEntry:
    provider: str
    provider_list_key: str
    model_id: str
    display_name: str
    source: DiscoverySource
    capability_status: CapabilityStatus
    persisted: bool


@dataclass(frozen=True)
class PersistenceResult:
    provider: str
    provider_list_key: str | None
    status: Literal["saved", "missing_provider_key", "ambiguous_provider_key", "error"]
    saved_model_ids: tuple[str, ...] = ()
    message: str = ""
```

- [ ] **Step 4: Implement provider list key resolver**

Create `tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py` and reuse:

```python
from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity
from tldw_chatbook.Chat.provider_readiness import provider_config_key
```

Implementation requirements:

- Normalize the requested provider through `resolve_console_provider_identity(provider).readiness_key`.
- Compare against every exact key in the top-level `[providers]` dict using `provider_config_key(existing_key)`.
- Return `resolved` only for exactly one match.
- Return `missing` for zero matches.
- Return `ambiguous` for more than one match and include every exact matching key.
- Never synthesize a new provider list key.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/model_discovery_contracts.py tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py
git commit -m "Add model discovery provider key resolution"
```

---

### Task 3: OpenAI-Compatible Endpoint Discovery Client

**Files:**
- Create: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`
- Test: `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`

- [ ] **Step 1: Write failing URL and eligibility tests**

Create tests for:

```python
def test_chat_completions_url_maps_to_models_url():
    assert build_models_url("https://api.example.test/v1/chat/completions", "custom") == "https://api.example.test/v1/models"


def test_llamacpp_completion_url_maps_to_v1_models():
    assert build_models_url("http://127.0.0.1:9099/completion", "llama_cpp") == "http://127.0.0.1:9099/v1/models"


def test_native_kobold_generate_is_not_eligible():
    assert supports_openai_compatible_model_discovery("koboldcpp", "http://127.0.0.1:5001/api/v1/generate") is False


def test_kobold_with_explicit_v1_endpoint_is_eligible():
    assert supports_openai_compatible_model_discovery("koboldcpp", "http://127.0.0.1:5001/v1") is True


def test_ollama_native_tags_is_not_eligible_in_v1():
    assert supports_openai_compatible_model_discovery("ollama", "http://127.0.0.1:11434/api/tags") is False
```

- [ ] **Step 2: Write failing response normalization tests**

Cover:

```python
def test_normalizes_openai_models_response_to_raw_model_ids():
    payload = {"data": [{"id": "gpt-4.1"}, {"id": "gpt-4.1-mini"}]}
    models = normalize_models_response(payload, provider="OpenAI", provider_list_key="OpenAI", endpoint_fingerprint="abc", now_iso="2026-06-04T12:00:00Z")
    assert [model.model_id for model in models] == ["gpt-4.1", "gpt-4.1-mini"]
    assert models[0].source == "runtime_discovered"


def test_response_metadata_does_not_include_sensitive_headers():
    payload = {"data": [{"id": "model-a", "owned_by": "org"}]}
    models = normalize_models_response(payload, provider="Custom", provider_list_key="Custom", endpoint_fingerprint="abc", now_iso="2026-06-04T12:00:00Z")
    assert "authorization" not in {key.lower() for key in models[0].metadata_raw_safe}
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py --tb=short
```

Expected: FAIL because implementation does not exist.

- [ ] **Step 4: Implement endpoint helpers**

In `openai_compatible_model_discovery.py`, implement:

```python
def supports_openai_compatible_model_discovery(provider_identity: str, normalized_endpoint: str | None) -> bool: ...
def build_models_url(endpoint: str, provider_identity: str) -> str: ...
def fingerprint_endpoint(endpoint: str) -> str: ...
def normalize_models_response(...) -> tuple[DiscoveredModel, ...]: ...
```

Implementation requirements:

- Eligible by endpoint shape and provider identity, not provider name alone.
- Support `/models`, `/v1/models`, `/v1`, `/v1/chat/completions`, `/chat/completions`, `/completion`, and `/completions`.
- Reuse `normalize_llamacpp_base_url()` for llama.cpp-like `/completion` and `/completions` handling.
- Reject native `koboldcpp` `/api/v1/generate` unless configured endpoint is explicitly OpenAI-compatible.
- Reject native `ollama` `/api/tags` in v1.
- Fingerprint endpoints without exposing query params, credentials, or headers.
- Return raw endpoint model IDs.
- Deduplicate duplicate model IDs while preserving endpoint order.

- [ ] **Step 5: Implement async manual discovery**

Add:

```python
async def discover_openai_compatible_models(
    *,
    provider: str,
    provider_list_key: str,
    endpoint: str,
    api_key: str | None,
    timeout_seconds: float = 10.0,
) -> ModelDiscoveryResult:
    ...
```

Requirements:

- Use `httpx.AsyncClient`.
- Add `Authorization: Bearer <api_key>` only when a key is present.
- Do not log or store the API key.
- Return typed errors for unsupported endpoint, request failure, and invalid response.
- Do not persist any models.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py --tb=short
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py
git commit -m "Add OpenAI-compatible endpoint model discovery"
```

---

### Task 4: Runtime Cache, Merge, Capability Status, and Persistence

**Files:**
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_cache.py`
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_merge.py`
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_persistence.py`
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py`

- [ ] **Step 1: Write failing cache tests**

Cover:

```python
def test_cache_lists_models_by_provider_and_endpoint_fingerprint():
    cache = ModelDiscoveryCache()
    cache.replace("OpenRouter", "fp1", (model("openrouter/auto"),))
    assert [m.model_id for m in cache.list("OpenRouter", "fp1")] == ["openrouter/auto"]


def test_cache_clear_removes_only_requested_provider():
    cache = ModelDiscoveryCache()
    cache.replace("OpenAI", "fp1", (model("gpt-4.1"),))
    cache.replace("OpenRouter", "fp2", (model("openrouter/auto"),))
    cache.clear("OpenAI")
    assert cache.list("OpenAI", "fp1") == ()
    assert cache.list("OpenRouter", "fp2")
```

- [ ] **Step 2: Write failing merge and capability tests**

Cover:

```python
def test_merge_preserves_saved_order_then_adds_discovered_models():
    merged = merge_saved_and_discovered_models(
        saved_model_ids=["gpt-4.1", "gpt-4.1-mini"],
        discovered_models=(model("gpt-4.1-mini"), model("gpt-4.1-nano")),
    )
    assert [entry.model_id for entry in merged] == ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
    assert merged[-1].source == "runtime_discovered"


def test_vision_false_does_not_make_capabilities_known():
    status = resolve_discovered_model_capability_status("OpenAI", "new-model", {"vision": False})
    assert status == "unknown"
```

- [ ] **Step 3: Write failing persistence tests**

Use a temp config file or monkeypatch the settings loader/writer. Cover:

```python
def test_persist_appends_raw_model_ids_to_exact_provider_key():
    providers = {"OpenRouter": ["existing"]}
    updated = append_models_to_provider_list(providers, "OpenRouter", ["new-model", "existing"])
    assert updated["OpenRouter"] == ["existing", "new-model"]


def test_persistence_refuses_ambiguous_provider_key():
    providers = {"Custom": ["a"], "custom": ["b"]}
    result = persist_discovered_models_to_settings(
        providers_config=providers,
        requested_provider="custom",
        model_ids=["new-model"],
    )
    assert result.status == "ambiguous_provider_key"
```

- [ ] **Step 4: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py --tb=short
```

Expected: FAIL because implementation does not exist.

- [ ] **Step 5: Implement cache and merge helpers**

Requirements:

- Cache is app-lifetime only.
- Cache key includes `provider_list_key` and `endpoint_fingerprint`.
- Merge saved IDs first, then discovered IDs, deduped by raw model ID.
- Mark saved entries as `source="saved"` and persisted discovered entries as `persisted=True`.
- Unknown capability status remains visible for discovered models unless the existing capabilities map proves the model is known.

- [ ] **Step 6: Implement persistence helper**

Requirements:

- Update only top-level `[providers].<provider_list_key>`.
- Preserve exact key spelling/casing.
- Append new raw model IDs in selected order.
- Deduplicate without deleting existing saved entries.
- Refuse `missing` and `ambiguous` provider key states.
- Return a typed result with a user-safe message for Settings.
- Refresh `app.providers_models` after a successful save in the caller.

- [ ] **Step 7: Run focused tests**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py --tb=short
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/model_discovery_cache.py tldw_chatbook/LLM_Provider_Catalog/model_discovery_merge.py tldw_chatbook/LLM_Provider_Catalog/model_discovery_persistence.py tldw_chatbook/config.py Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py
git commit -m "Add model discovery cache merge and persistence"
```

---

### Task 5: Local Provider Catalog and Scope Service Wiring

**Files:**
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/llm_provider_catalog_scope_service.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/__init__.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`
- Modify: `tldw_chatbook/runtime_policy/registry.py`
- Test: `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`
- Test: `Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py`
- Test: `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`
- Test: `Tests/RuntimePolicy/test_runtime_policy_core.py`

- [x] **Step 1: Write failing local catalog tests**

Add tests for:

```python
async def test_local_catalog_discovers_models_for_configured_openai_compatible_provider():
    service = LocalLLMProviderCatalogService(...)
    result = await service.discover_models(provider="Custom", staged_settings=None)
    assert result.status == "success"
    assert result.models[0].model_id == "test-model"


def test_local_catalog_lists_merged_saved_and_runtime_discovered_models():
    service = LocalLLMProviderCatalogService(...)
    service.discovery_cache.replace("Custom", "fp", (model("new-model"),))
    merged = service.list_models(provider="Custom", include_discovered=True)
    assert "new-model" in [entry.model_id for entry in merged]
```

- [x] **Step 2: Write failing scope policy tests**

Add tests for:

```python
async def test_scope_service_allows_local_model_discovery_policy_action():
    scope = LLMProviderCatalogScopeService(local_service=local, server_service=server)
    result = await scope.discover_models(mode="local", provider="Custom")
    assert result.policy_action == "llm.catalog.models.discover.local"


async def test_scope_service_does_not_call_server_discovery_in_v1():
    result = await scope.discover_models(mode="server", provider="Custom")
    assert result.status == "unsupported"
```

- [x] **Step 3: Run focused tests and verify failure**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py --tb=short
```

Expected: FAIL for missing service methods.

Evidence: Initial focused run failed with missing `settings_loader` constructor support and missing `LLMProviderCatalogScopeService.discover_models`, confirming the tests covered absent behavior before implementation.

- [x] **Step 4: Implement local service methods**

Add public methods matching the PRD:

```python
async def discover_models(self, *, provider: str, staged_settings: dict | None = None) -> ModelDiscoveryResult: ...
def list_discovered_models(self, *, provider: str | None = None) -> tuple[DiscoveredModel, ...]: ...
def clear_discovered_models(self, *, provider: str | None = None) -> None: ...
def merge_saved_and_discovered_models(self, *, provider: str) -> tuple[MergedModelEntry, ...]: ...
def persist_discovered_models_to_settings(self, *, provider: str, model_ids: list[str]) -> PersistenceResult: ...
```

Requirements:

- Endpoint precedence:
  1. Staged unsaved Settings values if passed
  2. `[api_settings.<provider>]` keys: `api_base_url`, `api_base`, `base_url`, `api_url`, `endpoint`
  3. Provider defaults already used by Console readiness/execution
  4. Missing endpoint error
- Credential precedence:
  1. Staged Settings credential source
  2. provider-specific `[api_settings.<provider>]`
  3. existing environment-variable/config `api_key` readiness resolution
- Local v1 keyring-backed server credentials remain unsupported.

- [x] **Step 5: Implement scope service method**

Requirements:

- Use policy action `llm.catalog.models.discover.local`.
- `mode="local"` calls the local service.
- `mode="server"` returns unsupported in v1 with safe copy, not a silent no-op.
- Also route `list_discovered_models`, `clear_discovered_models`, `merge_saved_and_discovered_models`, and `persist_discovered_models_to_settings` through the scope service so Settings/Console do not need to bypass the source-aware seam.
- Register `llm.catalog.models.discover.*` and `llm.catalog.models.persist.*` with runtime policy.
- Reject placeholder API keys, preserve injected empty environments, fail closed on duplicate normalized `api_settings` blocks, and filter discovered model cache reads to the current endpoint fingerprint.

- [x] **Step 6: Run focused tests**

Run:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py --tb=short
```

Expected: PASS.

Evidence:

```bash
python -m pytest -q Tests/LLM_Provider_Catalog Tests/RuntimePolicy/test_runtime_policy_core.py::test_policy_engine_knows_local_model_discovery_actions Tests/RuntimePolicy/test_runtime_policy_core.py::test_policy_engine_knows_server_model_discovery_actions_but_blocks_in_local_mode --tb=short
# 92 passed in 7.19s

git diff --check
# clean
```

Note: Full `Tests/RuntimePolicy/test_runtime_policy_core.py::test_runtime_policy_registry_contains_full_audited_rows` still reports a pre-existing `kanban_boards_tasks` registry/fixture mismatch that is present on `origin/dev`; it is not introduced by Task 5.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py tldw_chatbook/LLM_Provider_Catalog/llm_provider_catalog_scope_service.py tldw_chatbook/LLM_Provider_Catalog/__init__.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py
git commit -m "Wire local provider catalog model discovery"
```

---

### Task 6: Settings Providers & Models Discovery Workflow

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

- [x] **Step 1: Write failing mounted Settings tests**

Add mounted tests covering:

```python
async def test_settings_provider_models_shows_discover_models_action_for_eligible_provider():
    async with app.run_test() as pilot:
        await open_settings_category(pilot, "Providers & Models")
        assert pilot.app.query_one("#settings-discover-provider-models").disabled is False


async def test_settings_discovery_success_shows_runtime_models_and_save_action():
    async with app.run_test() as pilot:
        await open_settings_category(pilot, "Providers & Models")
        await pilot.click("#settings-discover-provider-models")
        await wait_for_text(pilot, "Discovered 2 models")
        assert "runtime discovered" in pilot.app.query_one("#settings-provider-models-detail").renderable.plain


async def test_settings_discovery_ambiguous_provider_key_shows_recovery_copy():
    async with app.run_test() as pilot:
        await open_settings_category(pilot, "Providers & Models")
        await pilot.click("#settings-discover-provider-models")
        await wait_for_text(pilot, "Multiple provider entries match")
```

Reuse existing mounted Settings helpers where available. If no helper exists for a selector in this file, add a small local helper in the test module rather than embedding fixed `pilot.pause(...)` delays.

- [x] **Step 2: Run Settings tests and verify failure**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: FAIL for missing controls/states.

- [x] **Step 3: Add Settings UI state**

In `settings_screen.py`, add state fields for:

- selected provider for discovery
- discovery status: idle/loading/success/error
- discovered model rows
- selected discovered model IDs for save
- last safe recovery message

Use existing Settings hub panel styling. Do not create a new top-level route.

- [x] **Step 4: Add Settings controls and copy**

Add controls with stable IDs:

- `#settings-discover-provider-models`
- `#settings-save-discovered-provider-models`
- `#settings-clear-discovered-provider-models`
- `#settings-model-discovery-status`
- `#settings-discovered-models-list`

Copy requirements:

- Eligible idle: `Discover models from configured endpoint`
- Unknown capability warning: `Capabilities unknown until saved or verified; text chat is assumed.`
- Ambiguous provider key: `Multiple provider entries match this provider. Rename or remove duplicates before saving discovered models.`
- Unsupported native endpoint: `This endpoint is not OpenAI-compatible for v1 discovery. Configure a /v1 endpoint to discover models.`

- [x] **Step 5: Wire discover and save actions**

Requirements:

- Discover calls the local provider catalog service.
- Save persists only explicitly selected discovered model IDs.
- Successful save refreshes `app.providers_models`.
- Discovery results remain visible even before save.
- Errors never show secrets or full authorization headers.

- [x] **Step 6: Run focused Settings tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: PASS or only unrelated pre-existing failures documented with exact failing tests.

- [x] **Step 7: Capture actual UI screenshot for approval**

Run the app through the established textual-web/CDP path used by this project, navigate to Settings > Providers & Models, trigger a stubbed or safe discovery state, and capture an actual browser screenshot.

Expected: The user can see the discover status, discovered model list, warning copy, and save control in the real rendered UI before the PR proceeds.

Evidence:

- Focused discovery regressions passed: `3 passed, 1 warning`.
- Full Settings mounted suite passed: `161 passed, 1 warning`.
- CDP idle screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/settings/model-discovery-providers-models-cdp-2026-06-04.png`
- CDP recovery screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/settings/model-discovery-providers-models-recovery-cdp-2026-06-04.png`
- User approval: passed by follow-up instruction to continue after actual screenshot review.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py
git commit -m "Add Settings model discovery workflow"
```

---

### Task 7: Console Merged Model Consumption and Unknown Capability Warnings

**Files:**
- Modify: `tldw_chatbook/UI/Screens/provider_model_resolution.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_session_settings.py`

- [x] **Step 1: Write failing Console model list tests**

Add tests for:

```python
def test_console_model_resolution_includes_runtime_discovered_models():
    app = fake_app_with_discovered_model("Custom", "new-model")
    models = resolve_provider_model_options(app, provider="Custom")
    assert "new-model" in [option.model_id for option in models]


def test_console_unknown_discovered_model_has_warning_copy():
    app = fake_app_with_discovered_model("Custom", "new-model", capability_status="unknown")
    state = build_console_session_settings_state(app, provider="Custom", model="new-model")
    assert "Capabilities unknown" in state.warning
```

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q Tests/UI/test_console_session_settings.py --tb=short
```

Expected: FAIL for missing discovered model integration.

Actual: Added `test_console_model_resolution_includes_runtime_discovered_models` and `test_console_settings_modal_can_select_runtime_discovered_model_with_warning`. The saved-order regression failed before the resolver fix with `['gpt-5', 'gpt-4.1'] != ['gpt-4.1', 'gpt-5']`.

- [x] **Step 3: Add merged model resolver**

In `provider_model_resolution.py`, add or extend a resolver that:

- reads saved models from `app.providers_models`
- asks the local provider catalog/cache for runtime discovered models when available
- keeps saved order first
- marks runtime-only entries with warning metadata
- returns raw endpoint model IDs for selection

- [x] **Step 4: Wire Console session/settings state**

In `chat_screen.py`, ensure:

- model dropdown/options use the merged resolver
- selecting a discovered model updates the same provider/model state used by send
- unknown capability warning appears in the Console settings/inspector area without blocking send
- blocked send feedback still reflects provider readiness, not catalog knownness

- [x] **Step 5: Run focused Console tests**

Run:

```bash
python -m pytest -q Tests/UI/test_console_session_settings.py --tb=short
```

Expected: PASS or only unrelated pre-existing failures documented with exact failing tests.

Actual: `python -m pytest -q Tests/UI/test_console_session_settings.py --tb=short` passed with `59 passed, 1 warning`.

- [x] **Step 6: Capture actual Console screenshot for approval**

Run the app through textual-web/CDP and capture Console with a runtime discovered model selected.

Expected: The rendered Console shows the selected discovered model, unknown capability warning, and no contradictory provider readiness state.

Actual: Captured actual textual-web/CDP screenshot at `Docs/superpowers/qa/product-maturity/screen-qa/console/model-discovery-runtime-console-cdp-2026-06-04.png`. The rendered Console shows `Model: gpt-5 (Capabilities unknown)` and `Credential: ready`.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/provider_model_resolution.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_session_settings.py
git commit -m "Use discovered models in Console selection"
```

---

### Task 8: End-to-End Regression Sweep and Documentation Closeout

**Files:**
- Modify: `backlog/tasks/task-78 - OpenAI-Compatible-Model-Discovery.md`
- Modify: `Docs/superpowers/specs/2026-06-04-openai-compatible-model-discovery-prd-design.md` only if implementation reveals a necessary contract clarification
- Modify: `Docs/superpowers/plans/2026-06-04-openai-compatible-model-discovery-implementation.md` only if implementation deviates

- [x] **Step 1: Run focused provider discovery tests**

Run:

```bash
python -m pytest -q \
  Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py \
  Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py \
  Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py \
  Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py \
  Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py \
  --tb=short
```

Expected: PASS.

Actual: `python -m pytest -q Tests/LLM_Provider_Catalog/test_model_discovery_provider_identity.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py --tb=short` passed with `80 passed`.

- [x] **Step 2: Run focused UI tests**

Run:

```bash
python -m pytest -q \
  Tests/UI/test_settings_configuration_hub.py \
  Tests/UI/test_console_session_settings.py \
  --tb=short
```

Expected: PASS or known unrelated failures documented with issue references.

Actual: `python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py --tb=short` passed with `220 passed, 1 warning`.

- [x] **Step 3: Run config and diff hygiene checks**

Run:

```bash
git diff --check
python -m pytest -q Tests/LLM_Provider_Catalog Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py --tb=short
```

Expected: PASS.

Actual: `git diff --check` returned no output. `python -m pytest -q Tests/LLM_Provider_Catalog Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py --tb=short` passed with `308 passed, 8 warnings`.

- [x] **Step 4: Manual QA with a local OpenAI-compatible endpoint**

If a local llama.cpp-compatible endpoint is running on `127.0.0.1:9099`, configure a compatible endpoint such as `http://127.0.0.1:9099/v1` or `http://127.0.0.1:9099/v1/chat/completions` and manually verify:

- Settings discovers models from `/v1/models`.
- A discovered model appears without auto-saving.
- Saving selected models appends raw model IDs to the existing provider list.
- Console can select a runtime-only discovered model.
- Console can send with that selected model if provider readiness is satisfied.
- Unsupported native endpoints show recovery copy.

Do not block the PR on local endpoint availability; document if unavailable.

Actual: `curl -sf --max-time 3 http://127.0.0.1:9099/v1/models` exited `7`; no local OpenAI-compatible endpoint was reachable, so manual endpoint QA is documented as unavailable and non-blocking.

- [x] **Step 5: Update Backlog implementation notes**

Run:

```bash
backlog task edit 78 -s Done --notes "Implemented OpenAI-compatible model discovery as a local manual workflow. Added provider list key resolution, endpoint discovery, runtime cache, saved/discovered merge, explicit Settings persistence, Console consumption, focused tests, and ADR 002. Deferred native provider-specific discovery, server keyring credentials, and tldw_server catalog sync per the PRD."
```

Before marking Done, ensure all task acceptance criteria are checked in the task file.

- [x] **Step 6: Final commit**

```bash
git add "backlog/tasks/task-78 - OpenAI-Compatible-Model-Discovery.md" Docs/superpowers/specs/2026-06-04-openai-compatible-model-discovery-prd-design.md Docs/superpowers/plans/2026-06-04-openai-compatible-model-discovery-implementation.md
git commit -m "Close out model discovery implementation plan"
```

- [x] **Step 7: Prepare PR**

Run:

```bash
git status --short
git log --oneline origin/dev..HEAD
```

Expected: Clean worktree and a readable stack of small commits.

PR title:

```text
Add OpenAI-compatible model discovery
```

PR body must include:

- ADR path
- What was implemented
- What was explicitly deferred
- Focused test commands and results
- Manual QA screenshots or note that endpoint/UI QA was unavailable
