# Model Catalog Auto-Refresh Implementation Plan

> Executed on branch feature/model-catalog-auto-refresh; ADR renumbered to 019 and task to 299 during execution (main gained 014-018).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically refresh model lists for OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, and Mistral on app startup, surfacing them in the chat model selector (capped merge + type-to-filter search picker) with an opt-in per-provider write-through to `config.toml`.

**Architecture:** Extend the existing `tldw_chatbook/LLM_Provider_Catalog/` package (ADR-002 pipeline: discover → cache → merge → persist) with six-provider coverage, a disk-backed TTL cache, and a startup refresher; amend ADR-002 via ADR-019 for opt-in write-through. All failures degrade to cached/saved models; startup is never blocked.

**Tech Stack:** Python 3.11+, Textual 8.x, httpx (async), loguru, pytest (mocked `httpx.MockTransport`, real filesystem via `tmp_path`).

**Spec:** `Docs/superpowers/specs/2026-07-17-model-catalog-auto-refresh-design.md` (read first — it is the source of truth for behavior).

**Worktree:** Execute in a dedicated git worktree (use superpowers:using-git-worktrees). Do not commit to the user's current branch directly.

**Verified facts (2026-07-17):** all six `/models` routes exist (OpenRouter 200 unauthenticated; others 401 pre-auth). Anthropic pagination params are `after_id`/`before_id`, `limit` 1–1000 (per official anthropic-sdk-python source). Existing user configs pick up new `CONFIG_TOML_CONTENT` defaults via in-memory deep merge (`config.py:2753`, `config.py:697-699`) — no config migration needed.

---

### Task 0: Governance — backlog task + ADR-019

This repo uses Backlog.md (see AGENTS.md). ADR required: **yes** — this feature amends ADR-002's "explicit user persistence only" decision.

**Files:**
- Create: `backlog/decisions/019-automatic-model-catalog-refresh.md`
- Create: `backlog/tasks/task-299 - Auto-refresh model catalogs for cloud providers.md` (via CLI)

- [ ] **Step 1: Create the backlog task**

```bash
backlog task create "Auto-refresh model catalogs for cloud providers" \
  -d "Refresh model lists for OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, and Mistral on app startup via the LLM_Provider_Catalog discovery pipeline. Disk-backed TTL cache feeds selectors (capped merge + search picker); per-provider opt-in write-through appends new models to [providers]. Amends ADR-002; see ADR-019 and Docs/superpowers/specs/2026-07-17-model-catalog-auto-refresh-design.md." \
  --ac "Startup auto-refresh fetches stale providers in a background worker without blocking app start,Fetched small catalogs (<=50) merge into chat selector and survive restarts,Large catalogs stay saved-list-only in dropdown and are searchable via picker,Write-through is append-only with baseline guard for oversized first fetch,Global toggle off / opt-out / no key / fresh cache each skip fetching,Failures degrade to cached models with at most one notification,Anthropic uses x-api-key + pagination; Z.AI ships config defaults,ADR-019 linked; tests pass"
```

Note the returned task id (`<N>`).

- [ ] **Step 2: Write ADR-019** following `backlog/decisions/000-template.md`:

```markdown
# ADR-019: Automatic model catalog refresh for cloud providers

Status: Accepted
Date: 2026-07-17
Related Task: [backlog/tasks/task-299 - Auto-refresh model catalogs for cloud providers.md](../tasks/task-299%20-%20Auto-refresh%20model%20catalogs%20for%20cloud%20providers.md)
Supersedes: N/A (amends ADR-002)

## Decision

Cloud-provider model lists (OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, Mistral)
auto-refresh on app startup through the ADR-002 discovery pipeline: fetched models
persist to a disk-backed TTL cache and merge into selectors (capped; oversized
catalogs reachable via a search picker). A per-provider opt-in write-through
appends new model IDs to `[providers]` in config.toml (append-only; oversized
first fetch establishes a baseline without appending).

## Context

ADR-002 kept discovery manual and persistence explicit-only. Users want fresh
model lists without manual steps. ADR-002's runtime-cache-first and scoped-design
constraints still hold; this amends only the "no auto-save" consequence, and only
as an opt-in. OpenRouter's catalog is public (no key required).

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Silent full rewrite of `[providers]` | Clobbers hand-curated lists; contradicts ADR-002's core stance |
| Standalone ModelRefreshService parallel to the catalog | Duplicates fetch/merge/persist; ADR-002 forbids a parallel registry |
| Bundled static model list updated with releases | Recreates the manual-update problem upstream |

## Consequences

- Startup performs bounded background network I/O (per-provider, 10s timeout, stale-after 24h default); failures degrade silently to cached/saved models.
- Write-through is append-only and never removes models; users prune `[providers]` themselves.
- `model_catalog_cache.json` under the user data dir stores model IDs + timestamps only (no credentials).
- Manual Discover/Save/Clear flows from ADR-002 remain unchanged.

## Links

- Spec: Docs/superpowers/specs/2026-07-17-model-catalog-auto-refresh-design.md
- Plan: Docs/superpowers/plans/2026-07-17-model-catalog-auto-refresh.md
- Amends: backlog/decisions/002-openai-compatible-model-discovery.md
```

- [ ] **Step 3: Link plan + ADR from the task**

```bash
backlog task edit 299 --plan "Implementation plan: Docs/superpowers/plans/2026-07-17-model-catalog-auto-refresh.md\nADR: backlog/decisions/019-automatic-model-catalog-refresh.md (amends ADR-002)"
backlog task edit <N> -s "In Progress"
```

- [ ] **Step 4: Commit**

```bash
git add backlog/decisions/019-automatic-model-catalog-refresh.md backlog/tasks/
git commit -m "docs: ADR-019 + backlog task for model catalog auto-refresh"
```

---

### Task 1: Discovery client — auth profiles, path policy, Anthropic pagination

**Files:**
- Modify: `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`
- Test: `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`

- [ ] **Step 1: Write failing tests** (append to the test file; it already imports the module and uses `httpx.MockTransport`-style fakes — check its tail for the client-injection pattern before writing):

```python
def test_openrouter_api_v1_path_maps_to_models():
    assert supports_openai_compatible_model_discovery(
        "openrouter", "https://openrouter.ai/api/v1"
    ) is True
    assert (
        build_models_url("https://openrouter.ai/api/v1", "openrouter")
        == "https://openrouter.ai/api/v1/models"
    )


def test_zai_paas_v4_path_maps_to_models():
    assert supports_openai_compatible_model_discovery(
        "zai", "https://api.z.ai/api/paas/v4"
    ) is True
    assert (
        build_models_url("https://api.z.ai/api/paas/v4", "zai")
        == "https://api.z.ai/api/paas/v4/models"
    )


def test_anthropic_uses_x_api_key_headers():
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        build_discovery_auth_headers,
    )

    headers = build_discovery_auth_headers("anthropic", "sk-ant-test")
    assert headers == {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}
    assert build_discovery_auth_headers("openai", "sk-test") == {
        "Authorization": "Bearer sk-test"
    }
    assert build_discovery_auth_headers("anthropic", None) == {}


@pytest.mark.asyncio
async def test_anthropic_paginates_with_after_id():
    requests: list[dict] = []
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(dict(request.url.params))
        seen_headers.update({k.lower(): v for k, v in request.headers.items()})
        page = len(requests)
        payload = (
            {"data": [{"id": f"claude-{page}"}], "has_more": True, "last_id": f"claude-{page}"}
            if page == 1
            else {"data": [{"id": "claude-2"}], "has_more": False, "last_id": "claude-2"}
        )
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await discover_openai_compatible_models(
            provider="anthropic",
            provider_list_key="Anthropic",
            endpoint="https://api.anthropic.com/v1",
            api_key="sk-ant-test",
            client=client,
        )
    assert result.status == "success"
    assert [m.model_id for m in result.models] == ["claude-1", "claude-2"]
    assert requests[0] == {"limit": "1000"}
    assert requests[1] == {"limit": "1000", "after_id": "claude-1"}
    # Anthropic auth headers, not Bearer:
    assert seen_headers["x-api-key"] == "sk-ant-test"
    assert seen_headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in seen_headers


@pytest.mark.asyncio
async def test_openai_does_not_paginate():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert "limit" not in dict(request.url.params)
        return httpx.Response(200, json={"data": [{"id": "gpt-x"}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await discover_openai_compatible_models(
            provider="openai",
            provider_list_key="OpenAI",
            endpoint="https://api.openai.com/v1",
            api_key="sk-test",
            client=client,
        )
    assert result.status == "success"
    assert calls == 1
```

(The header assertions above verify the anthropic auth profile end-to-end: `x-api-key` + `anthropic-version`, and no `Authorization` Bearer header.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py -x -q`
Expected: FAIL (`build_discovery_auth_headers` doesn't exist; `/api/v1` and `/api/paas/v4` unsupported)

- [ ] **Step 3: Implement** in `openai_compatible_model_discovery.py`:

a) Extend the explicit-path frozenset (line 45-53):

```python
_EXPLICIT_OPENAI_COMPATIBLE_ENDPOINT_PATHS = frozenset(
    {
        "/models",
        "/v1",
        "/v1/models",
        "/completion",
        "/completions",
        "/api/v1",
        "/api/paas/v4",
    }
)
```

b) Extend `_models_path_for_endpoint_path` (after the `"/v1"` branch, line ~166):

```python
    if normalized_path in {"/api/v1", "/api/paas/v4"}:
        return f"{normalized_path}/models"
```

c) Add auth profiles + pagination constants (after `_COMPACT_SENSITIVE_METADATA_KEY_SUFFIXES`, ~line 99):

```python
_ANTHROPIC_PROVIDER_KEY = "anthropic"
_ANTHROPIC_VERSION_HEADER = "2023-06-01"
_ANTHROPIC_MODELS_PAGE_LIMIT = 1000
_ANTHROPIC_MAX_MODEL_PAGES = 10


def build_discovery_auth_headers(provider_identity: str, api_key: str | None) -> dict[str, str]:
    """Return provider-appropriate auth headers for a models request."""
    if not api_key:
        return {}
    if _normalized_provider_identity(provider_identity) == _ANTHROPIC_PROVIDER_KEY:
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION_HEADER,
        }
    return {"Authorization": f"Bearer {api_key}"}
```

d) In `discover_openai_compatible_models` (line 354+): replace `headers = {"Authorization": f"Bearer {api_key}"} if api_key else None` (line 392) with:

```python
    headers = build_discovery_auth_headers(provider, api_key) or None
    paginate = _normalized_provider_identity(provider) == _ANTHROPIC_PROVIDER_KEY
```

e) Replace the single-GET `_request_payload` inner function with a paginating version that returns a list of payloads (same error results as today, constructed via the existing `_discovery_error` calls — keep messages identical). **Catch `httpx.HTTPStatusError` before `httpx.HTTPError`** (it is a subclass) so 401/403 map to the existing-but-unused `missing_credentials` error kind — the refresher treats that as a quiet not-ready skip instead of a per-launch failure notification:

```python
    async def _request_payloads(
        active_client: httpx.AsyncClient,
    ) -> tuple[list[Mapping[str, Any]] | None, ModelDiscoveryResult | None]:
        payloads: list[Mapping[str, Any]] = []
        params: dict[str, Any] | None = (
            {"limit": _ANTHROPIC_MODELS_PAGE_LIMIT} if paginate else None
        )
        for _page in range(_ANTHROPIC_MAX_MODEL_PAGES if paginate else 1):
            try:
                response = await active_client.get(models_url, headers=headers, params=params)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {401, 403}:
                    return None, ModelDiscoveryResult(
                        provider=provider,
                        provider_list_key=provider_list_key,
                        endpoint_fingerprint=endpoint_fingerprint,
                        status="error",
                        error=_discovery_error(
                            "missing_credentials",
                            "The models endpoint rejected the configured credentials.",
                            "Check the API key configured for this provider.",
                        ),
                    )
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "request_failed",
                        "Model discovery request failed.",
                        "Check the endpoint URL, server availability, and credentials.",
                    ),
                )
            except httpx.HTTPError:
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "request_failed",
                        "Model discovery request failed.",
                        "Check the endpoint URL, server availability, and credentials.",
                    ),
                )
            try:
                payload = response.json()
            except ValueError:
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "invalid_response",
                        "The models endpoint did not return valid JSON.",
                        "Use an endpoint that returns a JSON object with a data array of model IDs.",
                    ),
                )
            if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "invalid_response",
                        "The models endpoint did not return a valid OpenAI-compatible response.",
                        "Use an endpoint that returns a JSON object with a data array of model IDs.",
                    ),
                )
            payloads.append(payload)
            if not paginate:
                break
            last_id = payload.get("last_id")
            if bool(payload.get("has_more")) and isinstance(last_id, str) and last_id:
                params = {"limit": _ANTHROPIC_MODELS_PAGE_LIMIT, "after_id": last_id}
                continue
            break
        return payloads, None
```

f) Update the call site (lines 427-480): call `_request_payloads`, then combine pages before normalizing:

```python
    try:
        if client is not None:
            payloads, request_error = await _request_payloads(client)
        else:
            async with httpx.AsyncClient(timeout=timeout_seconds) as active_client:
                payloads, request_error = await _request_payloads(active_client)
    except httpx.HTTPError:
        return ModelDiscoveryResult(...)  # unchanged request_failed result
    if request_error is not None:
        return request_error
    if not payloads:
        return ModelDiscoveryResult(...)  # unchanged invalid_response result

    combined_data: list[Any] = []
    for payload in payloads:
        combined_data.extend(payload["data"])

    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        models = normalize_models_response(
            {"data": combined_data},
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint or "",
            now_iso=now_iso,
        )
    except ValueError:
        return ModelDiscoveryResult(...)  # unchanged invalid_response result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py -q`
Expected: PASS (including all pre-existing tests — the existing test at line 61-66 asserting bare anthropic base URLs are ineligible must still pass: we changed only explicit paths)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py
git commit -m "feat: cloud provider auth profiles, endpoint paths, Anthropic pagination in model discovery"
```

---

### Task 2: Default endpoints + Z.AI/`[model_catalog]` config defaults

**Files:**
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py:44-47`
- Modify: `tldw_chatbook/config.py` (`CONFIG_TOML_CONTENT`: `[providers]` ~line 1626, `[api_settings]` ~line 1777, new `[model_catalog]` section ~line 1644)
- Test: `Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py`, `Tests/test_config_model_catalog_defaults.py` (new)

- [ ] **Step 1: Write failing tests**

`Tests/test_config_model_catalog_defaults.py` (new):

```python
import tomllib

from tldw_chatbook.config import CONFIG_TOML_CONTENT


def test_zai_provider_and_settings_defaults_exist():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert isinstance(parsed["providers"].get("ZAI"), list)
    zai_settings = parsed["api_settings"]["zai"]
    assert zai_settings["api_key_env_var"] == "ZAI_API_KEY"
    assert zai_settings["api_base_url"] == "https://api.z.ai/api/paas/v4"


def test_model_catalog_defaults_exist():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    section = parsed["model_catalog"]
    assert section["auto_refresh_enabled"] is True
    assert section["stale_after_hours"] == 24
    assert section["auto_refresh_disabled"] == []
    assert section["write_to_config"] == []
```

In `test_local_llm_provider_catalog_service.py`, add a test that discovery resolves each new default endpoint with no configured `api_base_url` (follow the file's existing fixture pattern for constructing the service with a fake `settings_loader`):

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "list_key", "expected_url"),
    [
        ("anthropic", "Anthropic", "https://api.anthropic.com/v1/models"),
        ("mistralai", "MistralAI", "https://api.mistral.ai/v1/models"),
        ("moonshot", "Moonshot", "https://api.moonshot.ai/v1/models"),
        ("zai", "ZAI", "https://api.z.ai/api/paas/v4/models"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1/models"),
    ],
)
async def test_cloud_provider_default_endpoints_resolve(provider, list_key, expected_url):
    seen_urls: list[str] = []

    async def fake_client(**kwargs):
        seen_urls.append(build_models_url(kwargs["endpoint"], kwargs["provider"]))
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint="fp",
            status="success",
            models=(),
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {list_key: ["placeholder-model"]},
        settings_loader=lambda: {},
        discovery_client=fake_client,
        environ={},
    )
    result = await service.discover_models(provider=list_key)
    assert result.status == "success"
    assert seen_urls == [expected_url]
```

(`discover_models` does not gate on an API key, and the fake client succeeds regardless — no env vars needed in this test.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/test_config_model_catalog_defaults.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py -x -q`
Expected: FAIL (no ZAI/model_catalog defaults; endpoints unresolved)

- [ ] **Step 3: Implement**

a) `local_llm_provider_catalog_service.py` line 44-47:

```python
_DEFAULT_OPENAI_COMPATIBLE_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "mistralai": "https://api.mistral.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "zai": "https://api.z.ai/api/paas/v4",
}
```

b) `config.py` `CONFIG_TOML_CONTENT`, after the `OpenRouter = [...]` line (~1626), before `# Local Providers`:

```toml
ZAI = ["glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v", "glm-4-32b-0414-128k"]
```

c) After the `[providers]` block ends (after `local_mlx_lm = ["None"]`, ~line 1643), before `[api_settings]`:

```toml
[model_catalog]
# Automatic model-list refresh for cloud providers (ADR-019).
auto_refresh_enabled = true
stale_after_hours = 24 # 0 = refetch every launch
auto_refresh_disabled = [] # exact [providers] keys to opt out, e.g. ["ZAI"]
write_to_config = [] # exact [providers] keys whose new models append to this file
```

d) After the `[api_settings.moonshot]` block (~line 1777), before `# --- Local Providers ---`:

```toml
    [api_settings.zai] # Matches key in [providers]
    api_key_env_var = "ZAI_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "glm-4.5"
    temperature = 0.7
    top_p = 0.95
    max_tokens = 4096
    api_base_url = "https://api.z.ai/api/paas/v4"
    timeout = 90
    retries = 3
    retry_delay = 5
    streaming = false
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/test_config_model_catalog_defaults.py Tests/LLM_Provider_Catalog/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py tldw_chatbook/config.py Tests/test_config_model_catalog_defaults.py Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py
git commit -m "feat: default discovery endpoints + Z.AI/model_catalog config defaults"
```

---

### Task 3: Disk-backed TTL cache (`ModelCatalogDiskStore`)

**Files:**
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py`
- Test: `Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py`

- [ ] **Step 1: Write failing tests**

```python
from datetime import UTC, datetime, timedelta

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)


def _store(tmp_path):
    return ModelCatalogDiskStore(tmp_path / "model_catalog_cache.json")


def test_round_trip_into_memory_cache(tmp_path):
    store = _store(tmp_path)
    store.record("OpenAI", "https://api.openai.com/v1", ["gpt-a", "gpt-b"],
                 fetched_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC))
    store.save()

    cache = ModelDiscoveryCache()
    reloaded = _store(tmp_path)
    reloaded.load_into(cache)
    models = cache.list("OpenAI", "https://api.openai.com/v1")
    assert [m.model_id for m in models] == ["gpt-a", "gpt-b"]
    assert all(m.source == "runtime_discovered" for m in models)
    assert reloaded.fetched_at("OpenAI", "https://api.openai.com/v1") == datetime(
        2026, 7, 17, 12, 0, tzinfo=UTC
    )


def test_missing_and_corrupt_files_load_empty(tmp_path):
    cache = ModelDiscoveryCache()
    _store(tmp_path).load_into(cache)  # missing
    (tmp_path / "model_catalog_cache.json").write_text("{not json", encoding="utf-8")
    _store(tmp_path).load_into(cache)  # corrupt
    assert cache.list() == ()


def test_staleness_boundaries(tmp_path):
    store = _store(tmp_path)
    now = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
    store.record("OpenAI", "fp", ["gpt-a"], fetched_at=now - timedelta(hours=23, minutes=59))
    assert store.is_stale("OpenAI", "fp", stale_after_hours=24, now=now) is False
    store.record("OpenAI", "fp", ["gpt-a"], fetched_at=now - timedelta(hours=24, minutes=1))
    assert store.is_stale("OpenAI", "fp", stale_after_hours=24, now=now) is True
    assert store.is_stale("OpenAI", "fp", stale_after_hours=0, now=now) is True
    assert store.is_stale("Nobody", "fp", stale_after_hours=24, now=now) is True


def test_prune_drops_unconfigured_providers(tmp_path):
    store = _store(tmp_path)
    store.record("OpenAI", "fp1", ["gpt-a"])
    store.record("Ghost", "fp2", ["ghost-model"])
    store.prune({"OpenAI"})
    assert store.fetched_at("Ghost", "fp2") is None
    assert store.fetched_at("OpenAI", "fp1") is not None


def test_disk_store_holds_no_credentials(tmp_path):
    store = _store(tmp_path)
    store.record("OpenAI", "fp", ["gpt-a"])
    store.save()
    raw = (tmp_path / "model_catalog_cache.json").read_text(encoding="utf-8")
    assert "api_key" not in raw and "Authorization" not in raw and "x-api-key" not in raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py -x -q`
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Implement** `model_discovery_disk_cache.py`:

```python
"""Disk-backed store for discovered model snapshots with fetch timestamps."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import DiscoveredModel

CACHE_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp as timezone-aware UTC; None when invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class ModelCatalogDiskStore:
    """JSON store mirroring ModelDiscoveryCache entries plus fetched_at.

    Stores model IDs and timestamps only — never credentials or headers.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._model_ids: dict[tuple[str, str], tuple[str, ...]] = {}
        self._fetched_at: dict[tuple[str, str], datetime] = {}

    def fetched_at(self, provider_list_key: str, endpoint_fingerprint: str) -> datetime | None:
        return self._fetched_at.get((str(provider_list_key), str(endpoint_fingerprint)))

    def is_stale(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
        *,
        stale_after_hours: float,
        now: datetime | None = None,
    ) -> bool:
        """Return True when the entry is missing or older than the threshold.

        A threshold of 0 (or less) means always-stale: refetch every launch.
        """
        if stale_after_hours <= 0:
            return True
        fetched = self.fetched_at(provider_list_key, endpoint_fingerprint)
        if fetched is None:
            return True
        current = now or _utc_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return (current - fetched).total_seconds() >= stale_after_hours * 3600

    def record(
        self,
        provider_list_key: str,
        endpoint_fingerprint: str,
        model_ids,
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        key = (str(provider_list_key), str(endpoint_fingerprint))
        self._model_ids[key] = tuple(str(model_id) for model_id in model_ids)
        stamp = fetched_at or _utc_now()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        self._fetched_at[key] = stamp

    def prune(self, keep_provider_list_keys: set[str]) -> None:
        """Drop entries for providers no longer configured."""
        for key in tuple(self._fetched_at):
            if key[0] not in keep_provider_list_keys:
                self._fetched_at.pop(key, None)
                self._model_ids.pop(key, None)

    def load_into(self, cache: ModelDiscoveryCache) -> None:
        """Populate the in-memory cache from disk; missing/corrupt loads empty."""
        self._model_ids.clear()
        self._fetched_at.clear()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable model catalog cache {self.path}: {exc}")
            return
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            return
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            provider_list_key = str(entry.get("provider_list_key") or "").strip()
            endpoint_fingerprint = str(entry.get("endpoint_fingerprint") or "").strip()
            fetched = _parse_timestamp(entry.get("fetched_at"))
            raw_ids = entry.get("models")
            if not provider_list_key or not endpoint_fingerprint or fetched is None:
                continue
            if not isinstance(raw_ids, list):
                continue
            model_ids = tuple(
                model_id.strip()
                for model_id in raw_ids
                if isinstance(model_id, str) and model_id.strip()
            )
            discovered_at = fetched.isoformat().replace("+00:00", "Z")
            cache.replace(
                provider_list_key,
                endpoint_fingerprint,
                tuple(
                    DiscoveredModel(
                        provider=provider_list_key,
                        provider_list_key=provider_list_key,
                        model_id=model_id,
                        display_name=model_id,
                        source="runtime_discovered",
                        endpoint_fingerprint=endpoint_fingerprint,
                        discovered_at=discovered_at,
                    )
                    for model_id in model_ids
                ),
            )
            key = (provider_list_key, endpoint_fingerprint)
            self._model_ids[key] = model_ids
            self._fetched_at[key] = fetched

    def save(self) -> None:
        """Atomically write the store (temp file + rename)."""
        entries: dict[str, dict] = {}
        for key, model_ids in self._model_ids.items():
            fetched = self._fetched_at.get(key)
            if fetched is None:
                continue
            entries[f"{key[0]}|{key[1]}"] = {
                "provider_list_key": key[0],
                "endpoint_fingerprint": key[1],
                "fetched_at": fetched.isoformat().replace("+00:00", "Z"),
                "models": list(model_ids),
            }
        payload = {"version": CACHE_VERSION, "entries": entries}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/model_discovery_disk_cache.py Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
git commit -m "feat: disk-backed TTL store for discovered model catalogs"
```

---

### Task 4: `[model_catalog]` settings parsing + merge cap constant

**Files:**
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py`
- Test: `Tests/LLM_Provider_Catalog/test_model_catalog_settings.py`

- [ ] **Step 1: Write failing tests**

```python
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
    SELECTOR_MERGE_CAP,
    ModelCatalogSettings,
    load_model_catalog_settings,
)


def test_defaults_when_section_missing():
    settings = load_model_catalog_settings({})
    assert settings.auto_refresh_enabled is True
    assert settings.stale_after_hours == 24.0
    assert settings.auto_refresh_disabled == frozenset()
    assert settings.write_to_config == frozenset()


def test_full_section_parsed_and_normalized():
    settings = load_model_catalog_settings(
        {
            "model_catalog": {
                "auto_refresh_enabled": False,
                "stale_after_hours": 12,
                "auto_refresh_disabled": ["ZAI"],
                "write_to_config": ["OpenRouter", "MistralAI"],
            }
        }
    )
    assert settings.auto_refresh_enabled is False
    assert settings.stale_after_hours == 12.0
    assert settings.auto_refresh_disabled == frozenset({"zai"})
    assert settings.write_to_config == frozenset({"openrouter", "mistralai"})


def test_garbage_values_fall_back_safely():
    settings = load_model_catalog_settings(
        {"model_catalog": {"stale_after_hours": "banana", "auto_refresh_disabled": "ZAI"}}
    )
    assert settings.stale_after_hours == 24.0
    assert settings.auto_refresh_disabled == frozenset()


def test_zero_stale_hours_is_allowed():
    settings = load_model_catalog_settings({"model_catalog": {"stale_after_hours": 0}})
    assert settings.stale_after_hours == 0.0


def test_six_providers_and_cap():
    assert set(AUTO_REFRESH_PROVIDER_LIST_KEYS) == {
        "OpenAI", "Anthropic", "MistralAI", "Moonshot", "OpenRouter", "ZAI",
    }
    assert SELECTOR_MERGE_CAP == 50
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: Implement** `model_catalog_settings.py`:

```python
"""Settings contract for automatic model catalog refresh (ADR-019)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.provider_readiness import provider_config_key

AUTO_REFRESH_PROVIDER_LIST_KEYS: tuple[str, ...] = (
    "OpenAI",
    "Anthropic",
    "MistralAI",
    "Moonshot",
    "OpenRouter",
    "ZAI",
)

SELECTOR_MERGE_CAP = 50
DEFAULT_STALE_AFTER_HOURS = 24.0


@dataclass(frozen=True)
class ModelCatalogSettings:
    """Parsed [model_catalog] config; provider sets hold normalized keys."""

    auto_refresh_enabled: bool = True
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS
    auto_refresh_disabled: frozenset[str] = frozenset()
    write_to_config: frozenset[str] = frozenset()


def _normalized_key_set(value: object) -> frozenset[str]:
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(
        provider_config_key(str(entry))
        for entry in value
        if isinstance(entry, str) and entry.strip()
    )


def load_model_catalog_settings(settings: Mapping[str, Any] | None) -> ModelCatalogSettings:
    section = (settings or {}).get("model_catalog", {})
    if not isinstance(section, Mapping):
        return ModelCatalogSettings()
    enabled = section.get("auto_refresh_enabled", True)
    try:
        stale_after_hours = max(0.0, float(section.get("stale_after_hours", DEFAULT_STALE_AFTER_HOURS)))
    except (TypeError, ValueError):
        stale_after_hours = DEFAULT_STALE_AFTER_HOURS
    return ModelCatalogSettings(
        auto_refresh_enabled=enabled if isinstance(enabled, bool) else True,
        stale_after_hours=stale_after_hours,
        auto_refresh_disabled=_normalized_key_set(section.get("auto_refresh_disabled")),
        write_to_config=_normalized_key_set(section.get("write_to_config")),
    )
```

- [ ] **Step 4: Run to verify PASS**

Run: `python3 -m pytest Tests/LLM_Provider_Catalog/test_model_catalog_settings.py -q`

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py Tests/LLM_Provider_Catalog/test_model_catalog_settings.py
git commit -m "feat: model_catalog settings parsing and selector merge cap"
```

---

### Task 5: Refresh report types + startup refresher

**Files:**
- Create: `tldw_chatbook/LLM_Provider_Catalog/model_auto_refresh.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py` (add method)
- Test: `Tests/LLM_Provider_Catalog/test_model_auto_refresh.py`

- [ ] **Step 1: Write failing tests** covering: skip-when-disabled / not-ready / fresh; refresh-when-stale; diff computed against pre-update cache; baseline guard (empty previous + catalog > cap → nothing saved); small-catalog backfill; write-through append-only with fresh settings re-read; OpenRouter refreshes without a key; consolidated notification text. Skeleton (expand per case; use fakes for `discovery_client`, `settings_loader`, `provider_catalog_loader`):

```python
import pytest

from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
    format_refresh_notification,
)
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    ModelCatalogSettings,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    ModelDiscoveryResult,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    fingerprint_endpoint,
)


def _discovered(list_key: str, *ids: str) -> tuple[DiscoveredModel, ...]:
    return tuple(
        DiscoveredModel(
            provider=list_key, provider_list_key=list_key, model_id=m,
            display_name=m, source="runtime_discovered",
            endpoint_fingerprint="fp", discovered_at="2026-07-17T00:00:00Z",
        )
        for m in ids
    )


def _service(models_by_provider, saved_calls, **overrides):
    async def fake_client(**kwargs):
        models = models_by_provider.get(kwargs["provider_list_key"], ())
        # Return the REAL endpoint fingerprint: the refresher computes previous_ids
        # under self._current_endpoint_fingerprint(), so a hardcoded fingerprint
        # would silently mismatch and make previous_ids always empty.
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
            models=models,
        )

    def fake_save(section_values):
        saved_calls.append(section_values)
        return True

    # settings_loader must include the providers mapping: the write-through append
    # base comes from a FRESH settings read, not the catalog loader's snapshot.
    default_settings = {
        "providers": {
            k: ["saved-1"]
            for k in ("OpenAI", "Anthropic", "MistralAI", "Moonshot", "OpenRouter", "ZAI")
        }
    }

    return LocalLLMProviderCatalogService(
        provider_catalog_loader=overrides.get(
            "catalog_loader",
            lambda: {k: ["saved-1"] for k in ("OpenAI", "Anthropic", "MistralAI", "Moonshot", "OpenRouter", "ZAI")},
        ),
        settings_loader=overrides.get("settings_loader", lambda: default_settings),
        discovery_client=fake_client,
        save_discovered_models_callback=fake_save,
        environ=overrides.get("environ", {"OPENAI_API_KEY": "sk", "ANTHROPIC_API_KEY": "sk",
                                          "MISTRAL_API_KEY": "sk", "MOONSHOT_API_KEY": "sk",
                                          "ZAI_API_KEY": "sk"}),  # no OPENROUTER key: public catalog
    )


@pytest.mark.asyncio
async def test_openrouter_refreshes_without_api_key(tmp_path):
    saved_calls = []
    service = _service({"OpenRouter": _discovered("OpenRouter", "a/b")}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(),
        disk_store=store,
        provider_list_keys=("OpenRouter",),
    )
    assert report.outcomes[0].status == "refreshed"


@pytest.mark.asyncio
async def test_baseline_guard_suppresses_oversized_first_write(tmp_path):
    big = _discovered("OpenRouter", *[f"vendor/m{i}" for i in range(60)])
    saved_calls = []
    service = _service({"OpenRouter": big}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(write_to_config=frozenset({"openrouter"})),
        disk_store=store,
        provider_list_keys=("OpenRouter",),
    )
    assert report.outcomes[0].status == "baseline"
    assert saved_calls == []  # nothing written on oversized first fetch


@pytest.mark.asyncio
async def test_small_catalog_backfills_on_first_write(tmp_path):
    saved_calls = []
    service = _service({"OpenAI": _discovered("OpenAI", "saved-1", "new-1", "new-2")}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(write_to_config=frozenset({"openai"})),
        disk_store=store,
        provider_list_keys=("OpenAI",),
    )
    assert report.outcomes[0].status == "refreshed"
    assert saved_calls == [{"providers": {"OpenAI": ["saved-1", "new-1", "new-2"]}}]


@pytest.mark.asyncio
async def test_second_fetch_appends_only_new_since_baseline(tmp_path):
    saved_calls = []
    service = _service({"OpenRouter": _discovered("OpenRouter", *[f"v/m{i}" for i in range(60)])}, saved_calls)
    store = ModelCatalogDiskStore(tmp_path / "cache.json")
    settings = ModelCatalogSettings(write_to_config=frozenset({"openrouter"}))
    await service.refresh_stale_configured_providers(
        catalog_settings=settings, disk_store=store, provider_list_keys=("OpenRouter",))
    # second fetch adds one model
    service2 = _service(
        {"OpenRouter": _discovered("OpenRouter", *[f"v/m{i}" for i in range(60)], "v/new")},
        saved_calls,
    )
    service2.discovery_cache = service.discovery_cache  # share prior cache state
    report = await service2.refresh_stale_configured_providers(
        catalog_settings=settings, disk_store=store, provider_list_keys=("OpenRouter",),
        force=True,
    )
    assert report.outcomes[0].saved_model_ids == ("v/new",)


def test_notification_none_when_nothing_happened():
    from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
        ProviderRefreshOutcome, RefreshReport,
    )
    report = RefreshReport((
        ProviderRefreshOutcome(provider_list_key="OpenAI", status="skipped_fresh"),
    ))
    assert format_refresh_notification(report) is None


def test_notification_reports_cached_and_failed():
    from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
        ProviderRefreshOutcome, RefreshReport,
    )
    report = RefreshReport((
        ProviderRefreshOutcome(provider_list_key="OpenAI", status="refreshed", new_model_ids=("gpt-x",)),
        ProviderRefreshOutcome(provider_list_key="ZAI", status="failed", error_kind="request_failed"),
    ))
    message = format_refresh_notification(report)
    assert "OpenAI" in message and "cached" in message
    assert "ZAI" in message and "using cached list" in message
```

(Write the remaining skip-condition tests — disabled / not-ready / fresh / auth-failure — following the same fake-service pattern; the `force=True` flag bypasses the staleness check for tests. For the auth-failure case, have the fake `discovery_client` return a `ModelDiscoveryResult` with `status="error"` and `error.kind == "missing_credentials"` and assert the outcome is `skipped_not_ready`, not `failed`.)

- [ ] **Step 2: Run to verify failure** (module doesn't exist)

- [ ] **Step 3: Implement**

a) Create `tldw_chatbook/LLM_Provider_Catalog/model_auto_refresh.py`:

```python
"""Automatic startup refresh for cloud provider model catalogs (ADR-019)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RefreshStatus = Literal[
    "refreshed",
    "baseline",
    "skipped_disabled",
    "skipped_not_ready",
    "skipped_fresh",
    "failed",
]


@dataclass(frozen=True)
class ProviderRefreshOutcome:
    """Result of one provider's auto-refresh attempt."""

    provider_list_key: str
    status: RefreshStatus
    new_model_ids: tuple[str, ...] = ()
    saved_model_ids: tuple[str, ...] = ()
    error_kind: str | None = None
    write_failed: bool = False  # cache updated but config write-through failed


@dataclass(frozen=True)
class RefreshReport:
    """Aggregated result of one startup auto-refresh pass."""

    outcomes: tuple[ProviderRefreshOutcome, ...] = ()


def format_refresh_notification(report: RefreshReport) -> str | None:
    """Build one consolidated user notification, or None when nothing changed."""
    parts: list[str] = []
    failures: list[str] = []
    write_failures: list[str] = []
    for outcome in report.outcomes:
        if outcome.status == "refreshed" and outcome.saved_model_ids:
            parts.append(f"{outcome.provider_list_key}: {len(outcome.saved_model_ids)} new saved")
        elif outcome.status == "refreshed" and outcome.new_model_ids:
            parts.append(f"{outcome.provider_list_key}: {len(outcome.new_model_ids)} new cached")
        elif outcome.status == "baseline":
            parts.append(f"{outcome.provider_list_key}: catalog cached")
        if outcome.write_failed:
            write_failures.append(outcome.provider_list_key)
        if outcome.status == "failed":
            failures.append(outcome.provider_list_key)
    if write_failures:
        parts.append(
            f"config save failed for {', '.join(write_failures)} (models cached instead)"
        )
    if not parts and not failures:
        return None
    message = f"Model lists updated — {', '.join(parts)}" if parts else ""
    if failures:
        message += ("; " if message else "") + (
            f"refresh failed for {', '.join(failures)} (using cached list)"
        )
    return message or None
```

b) Add to `local_llm_provider_catalog_service.py` (imports at top; method after `discover_models`):

```python
# new imports
from loguru import logger
from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
    ProviderRefreshOutcome,
    RefreshReport,
)
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
    SELECTOR_MERGE_CAP,
    ModelCatalogSettings,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)
```

```python
    async def refresh_stale_configured_providers(
        self,
        *,
        catalog_settings: ModelCatalogSettings,
        disk_store: ModelCatalogDiskStore,
        provider_list_keys: Sequence[str] = AUTO_REFRESH_PROVIDER_LIST_KEYS,
        merge_cap: int = SELECTOR_MERGE_CAP,
        force: bool = False,
        on_config_saved: Callable[[], None] | None = None,
    ) -> RefreshReport:
        """Refresh stale cloud-provider catalogs; optionally append new models to config.

        Never raises for per-provider failures; each becomes a "failed" outcome.
        Write-through is append-only, computed against the pre-fetch cache entry,
        with a baseline guard for oversized first fetches (ADR-019).
        """
        self._enforce("llm.catalog.models.discover.local")
        outcomes: list[ProviderRefreshOutcome] = []
        catalog = self._catalog()
        saved_settings = self._settings()

        for requested_key in provider_list_keys:
            resolution = resolve_provider_list_key(requested_key, catalog)
            if resolution.status != "resolved" or resolution.provider_list_key is None:
                outcomes.append(ProviderRefreshOutcome(
                    provider_list_key=requested_key, status="skipped_not_ready"))
                continue
            provider_key = resolution.normalized_provider
            list_key = resolution.provider_list_key

            if provider_key in catalog_settings.auto_refresh_disabled:
                outcomes.append(ProviderRefreshOutcome(
                    provider_list_key=list_key, status="skipped_disabled"))
                continue

            api_key = self._resolve_api_key(
                provider=list_key,
                provider_key=provider_key,
                saved_settings=saved_settings,
                staged_settings=None,
            )
            # OpenRouter's catalog is public; everything else needs credentials.
            if api_key is None and provider_key != "openrouter":
                outcomes.append(ProviderRefreshOutcome(
                    provider_list_key=list_key, status="skipped_not_ready"))
                continue

            fingerprint = self._current_endpoint_fingerprint(provider_key=provider_key)
            if fingerprint is None:
                outcomes.append(ProviderRefreshOutcome(
                    provider_list_key=list_key, status="skipped_not_ready"))
                continue

            if not force and not disk_store.is_stale(
                list_key, fingerprint,
                stale_after_hours=catalog_settings.stale_after_hours,
            ):
                outcomes.append(ProviderRefreshOutcome(
                    provider_list_key=list_key, status="skipped_fresh"))
                continue

            # Snapshot the pre-fetch cache entry for the diff BEFORE discover_models
            # replaces it (also empty after fingerprint change/corruption — the
            # baseline guard below keeps that safe).
            previous_ids = {
                model.model_id
                for model in self.discovery_cache.list(list_key, fingerprint)
            }
            saved_ids = set(catalog.get(list_key, []))

            result = await self.discover_models(provider=list_key)
            if result.status != "success":
                if result.error and result.error.kind == "missing_credentials":
                    # Bad/rejected key: quiet not-ready skip, no per-launch nagging.
                    outcomes.append(ProviderRefreshOutcome(
                        provider_list_key=list_key, status="skipped_not_ready"))
                    continue
                outcomes.append(ProviderRefreshOutcome(
                    provider_list_key=list_key,
                    status="failed",
                    error_kind=result.error.kind if result.error else "request_failed",
                ))
                continue

            fresh_ids = [model.model_id for model in result.models]
            new_ids = tuple(
                model_id for model_id in fresh_ids
                if model_id not in previous_ids and model_id not in saved_ids
            )

            saved_to_config: tuple[str, ...] = ()
            write_failed = False
            status = "refreshed"
            if provider_key in catalog_settings.write_to_config:
                if not previous_ids and len(fresh_ids) > merge_cap:
                    # Oversized first fetch: establish baseline, append nothing so
                    # e.g. OpenRouter's 300+ catalog is never dumped into config.
                    status = "baseline"
                elif new_ids:
                    self._enforce("llm.catalog.models.persist.local")
                    # Re-read current settings so the append base is the latest file
                    # state, not the startup snapshot.
                    fresh_providers = self._providers_from_settings(self._settings())
                    persist_result = persist_discovered_models_to_settings(
                        providers_config=fresh_providers,
                        requested_provider=list_key,
                        model_ids=new_ids,
                        save_callback=self.save_discovered_models_callback,
                    )
                    if persist_result.status == "saved" and persist_result.saved_model_ids:
                        saved_to_config = persist_result.saved_model_ids
                        if on_config_saved is not None:
                            on_config_saved()
                    elif persist_result.status == "error":
                        # Cache is already updated; config untouched. Log + surface
                        # in the consolidated notification (spec error handling).
                        write_failed = True
                        logger.warning(
                            f"Model catalog write-through failed for {list_key}: "
                            f"{persist_result.message}"
                        )

            disk_store.record(list_key, fingerprint, fresh_ids)
            outcomes.append(ProviderRefreshOutcome(
                provider_list_key=list_key,
                status=status,
                new_model_ids=new_ids,
                saved_model_ids=saved_to_config,
                write_failed=write_failed,
            ))

        disk_store.prune(set(catalog) | set(provider_list_keys))
        disk_store.save()
        return RefreshReport(outcomes=tuple(outcomes))

    @staticmethod
    def _providers_from_settings(settings: Mapping[str, Any]) -> dict[str, list[str]]:
        providers = settings.get("providers", {}) if isinstance(settings, Mapping) else {}
        if not isinstance(providers, Mapping):
            return {}
        return {
            str(provider): [str(m) for m in models if isinstance(m, str)]
            for provider, models in providers.items()
            if isinstance(provider, str) and isinstance(models, list)
        }
```

(`persist_discovered_models_to_settings` — the module-level function — is already imported in this file at line 22-25.)

- [ ] **Step 4: Run to verify PASS**

Run: `python3 -m pytest Tests/LLM_Provider_Catalog/ -q`
Expected: PASS (all existing + new)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/model_auto_refresh.py tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py Tests/LLM_Provider_Catalog/test_model_auto_refresh.py
git commit -m "feat: startup auto-refresh with TTL skips, diff write-through, baseline guard"
```

---

### Task 6: App wiring — disk-cache load, startup worker, notification

**Files:**
- Modify: `tldw_chatbook/app.py` (service wiring ~line 3751-3756, `on_mount` ~line 5450)

Note: the discovery client is async, so the refresh runs as an **async** worker on the Textual event loop — `self.notify()` is then safe to call directly (the spec's `call_from_thread` note assumed a thread worker; async is cleaner here and equally non-blocking since httpx is async).

- [ ] **Step 1: Write failing test** — `Tests/LLM_Provider_Catalog/test_app_model_catalog_wiring.py`:

a) **Routing test (catches the Textual App→Screen message-direction trap):** with a minimal Textual test app whose screen stack contains a stub screen exposing `handle_model_catalog_refreshed`, assert `forward_model_catalog_refreshed` calls it and returns `True`; with a plain app, returns `False`:

```python
import pytest
from textual.app import App
from textual.screen import Screen

from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
    ModelCatalogRefreshed,
    forward_model_catalog_refreshed,
)


@pytest.mark.asyncio
async def test_forward_reaches_mounted_screen_handler():
    calls = []

    class StubScreen(Screen):
        async def handle_model_catalog_refreshed(self, event):
            calls.append(set(event.providers))

    class StubApp(App):
        def on_mount(self):
            self.push_screen(StubScreen())

    app = StubApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        event = ModelCatalogRefreshed(providers={"OpenAI"})
        assert await forward_model_catalog_refreshed(app, event) is True
    assert calls == [{"OpenAI"}]


@pytest.mark.asyncio
async def test_forward_returns_false_without_handler():
    app = App()
    async with app.run_test():
        event = ModelCatalogRefreshed(providers={"OpenAI"})
        assert await forward_model_catalog_refreshed(app, event) is False
```

b) **Refresh-loop test:** construct the app's refresh coroutine against a stub app double (or run the real app via `Tests/textual_test_harness.py` if a pattern exists — check that file first) asserting: returns early when `auto_refresh_enabled=False` (no discovery calls); `notify` called exactly once when the report has refreshed/failed outcomes; `on_config_saved` triggers `_init_providers_models`; `ModelCatalogRefreshed` posted when any outcome is refreshed/baseline.

- [ ] **Step 2: Implement**

a) After the `LocalLLMProviderCatalogService` creation (`app.py:3751-3756`), add:

```python
        # ADR-019: load the disk-backed model catalog cache before selectors build.
        try:
            from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
                ModelCatalogDiskStore,
            )
            self.model_catalog_disk_store = ModelCatalogDiskStore(
                get_user_data_dir() / "model_catalog_cache.json"
            )
            self.model_catalog_disk_store.load_into(
                self.local_llm_provider_catalog_service.discovery_cache
            )
        except Exception:
            logger.opt(exception=True).error("Failed to load model catalog disk cache")
            self.model_catalog_disk_store = None
```

b) Add the refresh coroutine method on `TldwCli`:

```python
    async def _refresh_model_catalogs(self) -> None:
        """ADR-019 startup auto-refresh; never blocks or crashes startup."""
        try:
            from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
                format_refresh_notification,
            )
            from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
                load_model_catalog_settings,
            )
            if self.model_catalog_disk_store is None:
                return
            catalog_settings = load_model_catalog_settings(load_settings())
            if not catalog_settings.auto_refresh_enabled:
                return
            report = await self.local_llm_provider_catalog_service.refresh_stale_configured_providers(
                catalog_settings=catalog_settings,
                disk_store=self.model_catalog_disk_store,
                on_config_saved=self._init_providers_models,
            )
            refreshed = {
                outcome.provider_list_key
                for outcome in report.outcomes
                if outcome.status in {"refreshed", "baseline"}
            }
            if refreshed:
                from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
                    ModelCatalogRefreshed,
                )
                self.post_message(ModelCatalogRefreshed(providers=refreshed))
            message = format_refresh_notification(report)
            if message:
                self.notify(message, severity="information")
        except Exception:
            logger.opt(exception=True).error("Model catalog auto-refresh failed")
```

(`load_settings` is already imported in app.py — verify; it is used at startup.)

c) In `on_mount` (`app.py:5450`), after the existing mount body, kick off the worker:

```python
        self.run_worker(
            self._refresh_model_catalogs(),
            exclusive=True,
            group="model-catalog-refresh",
        )
```

d) Add the `ModelCatalogRefreshed` message **and a forwarder** to `model_auto_refresh.py` (also add `Any` to the existing `typing` import there):

```python
from typing import Any, Literal  # Literal already imported in Task 5
from textual.message import Message


class ModelCatalogRefreshed(Message):
    """Posted when startup refresh updated one or more provider catalogs."""

    def __init__(self, *, providers: set[str]) -> None:
        super().__init__()
        self.providers = frozenset(providers)


async def forward_model_catalog_refreshed(app: Any, event: ModelCatalogRefreshed) -> bool:
    """Forward the event to a mounted screen exposing a refresh handler.

    Textual messages bubble UP only: App.post_message() never reaches a Screen's
    @on handler (verified against Textual 8.2.7). The App-level handler calls this
    to reach the chat screen via duck typing; returns False when no mounted screen
    handles it (e.g. chat tab never opened — options build fresh on mount anyway).
    """
    for screen in getattr(app, "screen_stack", ()):  # pragma: no branch
        handler = getattr(screen, "handle_model_catalog_refreshed", None)
        if callable(handler):
            await handler(event)
            return True
    return False
```

e) Add the App-level handler on `TldwCli` (Textual delivers App-posted messages to App handlers only):

```python
    @on(ModelCatalogRefreshed)
    async def on_model_catalog_refreshed(self, event: ModelCatalogRefreshed) -> None:
        from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
            forward_model_catalog_refreshed,
        )
        await forward_model_catalog_refreshed(self, event)
```

(Import `ModelCatalogRefreshed` at app.py module level alongside the other catalog imports; `@on` needs the class at class-definition time.)

- [ ] **Step 3: Run tests**

Run: `python3 -m pytest Tests/LLM_Provider_Catalog/ -q` + `python3 -c "from tldw_chatbook.app import TldwCli"` (import sanity)
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/LLM_Provider_Catalog/model_auto_refresh.py Tests/LLM_Provider_Catalog/test_app_model_catalog_wiring.py
git commit -m "feat: wire startup model catalog auto-refresh into app lifecycle"
```

---

### Task 7: Selector merge cap + chat screen integration + selection preservation

**Files:**
- Modify: `tldw_chatbook/UI/Screens/provider_model_resolution.py` (cap in `resolve_provider_model_options`, line 136-162)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (provider-change handler, lines 426-479; new refresh handler; options-rebuild helper)
- Modify: `tldw_chatbook/Widgets/settings_sidebar.py` (transient restore in compose, lines 135-146)
- Test: `Tests/UI/test_provider_model_resolution.py` (new), `Tests/UI/test_chat_screen_model_options.py` (new, if harness allows; otherwise cover logic in resolution tests)

- [ ] **Step 1: Write failing tests** (`Tests/UI/test_provider_model_resolution.py`; use a stub app object with `providers_models` and a fake `llm_provider_catalog_scope_service`):

```python
import pytest

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    resolve_provider_model_options,
)


class _FakeScope:
    def __init__(self, entries):
        self._entries = entries

    async def merge_saved_and_discovered_models(self, *, mode, provider):
        return self._entries


class _FakeApp:
    def __init__(self, providers_models, entries):
        self.providers_models = providers_models
        self.llm_provider_catalog_scope_service = _FakeScope(entries)


def _entries(provider, ids, source="runtime_discovered"):
    return tuple(
        MergedModelEntry(
            provider=provider, provider_list_key=provider, model_id=m,
            display_name=m, source=source, capability_status="unknown", persisted=False,
        )
        for m in ids
    )


@pytest.mark.asyncio
async def test_small_discovered_catalog_merges_with_label():
    app = _FakeApp({"OpenAI": ["saved-1"]}, _entries("OpenAI", ["new-1"]))
    options = await resolve_provider_model_options(app, provider="OpenAI")
    assert [o.model_id for o in options] == ["saved-1", "new-1"]
    assert "runtime discovered" in options[1].label


@pytest.mark.asyncio
async def test_oversized_catalog_stays_saved_only():
    app = _FakeApp({"OpenRouter": ["saved-1"]},
                   _entries("OpenRouter", [f"v/m{i}" for i in range(60)]))
    options = await resolve_provider_model_options(app, provider="OpenRouter")
    assert [o.model_id for o in options] == ["saved-1"]


@pytest.mark.asyncio
async def test_uncapped_returns_full_catalog_for_picker():
    app = _FakeApp({"OpenRouter": ["saved-1"]},
                   _entries("OpenRouter", [f"v/m{i}" for i in range(60)]))
    options = await resolve_provider_model_options(app, provider="OpenRouter", merge_cap=None)
    assert len(options) == 61


@pytest.mark.asyncio
async def test_current_model_inserted_as_transient_when_missing():
    app = _FakeApp({"OpenAI": ["saved-1"]}, ())
    options = await resolve_provider_model_options(
        app, provider="OpenAI", current_model="picked-elsewhere")
    assert options[0].model_id == "picked-elsewhere"
```

(`test_current_model_inserted_as_transient_when_missing` documents existing behavior at lines 159-161 — the picker restore path relies on it.)

- [ ] **Step 2: Run to verify failure** (`merge_cap` param doesn't exist)

- [ ] **Step 3: Implement**

a) `provider_model_resolution.py`: add `merge_cap` param (import `SELECTOR_MERGE_CAP` from `LLM_Provider_Catalog.model_catalog_settings`):

```python
async def resolve_provider_model_options(
    app_instance: Any,
    *,
    provider: str,
    current_model: str | None = None,
    merge_cap: int | None = SELECTOR_MERGE_CAP,
) -> list[ResolvedProviderModelOption]:
    """Return saved and runtime-discovered model selector options for a provider.

    Discovered entries merge only when the provider's total discovered catalog is
    at or below ``merge_cap`` (ADR-019); pass ``merge_cap=None`` for the uncapped
    list (search picker). Oversized catalogs stay saved-list-only in dropdowns.
    """
```

and in the body, replace the merged-entry loop (lines 152-157) with:

```python
    merged_entries = await _merged_model_entries_from_scope(app_instance, provider=provider_key)
    discovered_count = sum(1 for entry in merged_entries if str(entry.source) == "runtime_discovered")
    include_discovered = merge_cap is None or discovered_count <= merge_cap
    for entry in merged_entries:
        if str(entry.source) == "runtime_discovered" and not include_discovered:
            continue
        option = _option_from_entry(entry)
        if option.model_id and option.model_id not in seen_model_ids:
            options.append(option)
            seen_model_ids.add(option.model_id)
```

b) `chat_screen.py`: replace the config-only model rebuild in `handle_provider_change` (lines 431-457) with a shared helper on the screen:

```python
    async def _rebuild_chat_model_options(
        self,
        provider: str,
        *,
        current_model: str | None = None,
        select_first: bool = False,
    ) -> None:
        """Rebuild #chat-api-model options (saved + capped discovered) preserving selection."""
        from tldw_chatbook.UI.Screens.provider_model_resolution import (
            resolve_provider_model_options,
        )
        if not self.chat_window:
            return
        try:
            model_select = self.chat_window.query_one("#chat-api-model", Select)
        except Exception:
            return
        options = await resolve_provider_model_options(
            self.app, provider=provider, current_model=current_model,
        )
        select_options = [(option.label, option.model_id) for option in options]
        model_select.set_options(select_options)  # resets selection — re-apply below
        if select_first and options:
            model_select.value = options[0].model_id
        elif current_model and any(o.model_id == current_model for o in options):
            model_select.value = current_model
        model_select.prompt = "Select Model..." if options else "No models available"
```

In `handle_provider_change` (chat_screen.py:426-479), replace the **entire outer `try:` body** (lines 431-476 — from `from tldw_chatbook.config import get_cli_providers_and_models` through the `else: logger.error("chat_window is None")` line, keeping the final `except Exception` at 478-479 and the `logger.info` at 429) with this complete body, which preserves the guard structure and the kept compact-bar/orientation calls:

```python
            new_provider = str(event.value)
            await self._rebuild_chat_model_options(new_provider, select_first=True)

            # Find the model select widget within the chat window
            if self.chat_window:
                try:
                    model_select = self.chat_window.query_one("#chat-api-model", Select)
                    selected_model = None if _is_empty_select_value(model_select.value) else str(model_select.value)
                    self._sync_compact_shell_controls(
                        provider=new_provider,
                        model=selected_model,
                    )
                except Exception as e:
                    logger.error(f"Could not find model select widget: {e}")

                # Sync to compact model bar
                try:
                    from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar
                    compact_bar = self.chat_window.query_one("#compact-model-bar", CompactModelBar)
                    compact_bar.sync_from_sidebar(provider=new_provider)
                except Exception:
                    logger.debug("Compact bar not found for provider sync")
                self.chat_window.refresh_first_run_orientation(new_provider)
            else:
                logger.error("chat_window is None")
```

(`_rebuild_chat_model_options` already no-ops when `chat_window` is None, so calling it before the guard is safe.)

c) `chat_screen.py`: add the refresh handler **as a plain public method** (the App forwards the event via `forward_model_catalog_refreshed` — a `@on` handler here would never fire for an App-posted message; imports `provider_config_key`):

```python
    async def handle_model_catalog_refreshed(self, event) -> None:
        """Re-merge options when startup refresh updated the active provider."""
        if not self.chat_window:
            return
        try:
            provider_select = self.chat_window.query_one("#chat-api-provider", Select)
            model_select = self.chat_window.query_one("#chat-api-model", Select)
        except Exception:
            return
        provider = str(provider_select.value or "").strip()
        if not provider:
            return
        if provider_config_key(provider) not in {
            provider_config_key(list_key) for list_key in event.providers
        }:
            return
        current = None if _is_empty_select_value(model_select.value) else str(model_select.value)
        await self._rebuild_chat_model_options(provider, current_model=current)
```

d) `settings_sidebar.py` compose (lines 135-139): keep a previously-picked model as a transient option instead of falling back:

```python
            initial_models = providers_models.get(default_provider, [])
            model_options = [(model, model) for model in initial_models]
            current_model_value = (
                default_model if default_model else (initial_models[0] if initial_models else Select.BLANK)
            )
            if default_model and default_model not in initial_models:
                # ADR-019: a model picked outside the saved list (e.g. via search
                # picker) restores as a transient option instead of falling back.
                model_options = [(default_model, default_model)] + model_options
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/UI/test_provider_model_resolution.py Tests/LLM_Provider_Catalog/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/provider_model_resolution.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/settings_sidebar.py Tests/UI/test_provider_model_resolution.py
git commit -m "feat: capped selector merge, merged options in chat screen, selection preservation"
```

---

### Task 8: Search picker widget

**Files:**
- Create: `tldw_chatbook/Widgets/model_search_picker.py`
- Modify: `tldw_chatbook/Widgets/settings_sidebar.py` (compose picker under `#chat-api-model` when `id_prefix == "chat"`, after line 146)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (ModelSelected handler)
- Test: `Tests/Widgets/test_model_search_picker.py`

- [ ] **Step 1: Write failing tests** (use `Tests/textual_test_utils.py` / `textual_test_harness.py` patterns — check them for the established way to run a widget in a test app):

Core logic cases: substring filter matches provider prefixes (`anthropic/` filters OpenRouter IDs); empty query hides results; selection posts `ModelSelected` with the right ID (verify via `option_index` mapping, not Option ids — model IDs contain `/` and `:` which are invalid in Textual DOM ids).

- [ ] **Step 2: Run to verify failure**, then **Step 3: Implement** `Widgets/model_search_picker.py`:

```python
"""Type-to-filter model picker over the full (uncapped) provider catalog."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, OptionList, Select
from textual.widgets._option_list import Option


class ModelSearchPicker(Widget):
    """Substring search across saved + discovered models for the active provider.

    OpenRouter model IDs embed the upstream provider prefix
    (``anthropic/claude-3.7-sonnet``), so provider search works naturally (ADR-019).
    """

    MAX_RESULTS = 20

    class ModelSelected(Message):
        """Posted when the user picks a model from the search results."""

        def __init__(self, model_id: str) -> None:
            super().__init__()
            self.model_id = model_id

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._matches: list[str] = []

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search all models…", id="model-search-picker-input")
        yield OptionList(id="model-search-picker-results")

    def on_mount(self) -> None:
        self.query_one("#model-search-picker-results", OptionList).display = False

    def _current_provider(self) -> str | None:
        try:
            provider_select = self.app.query_one("#chat-api-provider", Select)
        except Exception:
            return None
        value = str(provider_select.value or "").strip()
        return value or None

    @on(Input.Changed, "#model-search-picker-input")
    async def _handle_query(self, event: Input.Changed) -> None:
        results = self.query_one("#model-search-picker-results", OptionList)
        query = event.value.strip().lower()
        self._matches = []
        results.clear_options()
        if not query:
            results.display = False
            return
        provider = self._current_provider()
        if not provider:
            results.display = False
            return
        from tldw_chatbook.UI.Screens.provider_model_resolution import (
            resolve_provider_model_options,
        )
        options = await resolve_provider_model_options(
            self.app, provider=provider, merge_cap=None,
        )
        self._matches = [
            option.model_id for option in options if query in option.model_id.lower()
        ][: self.MAX_RESULTS]
        for model_id in self._matches:
            results.add_option(Option(model_id))
        results.display = bool(self._matches)

    @on(OptionList.OptionSelected, "#model-search-picker-results")
    def _handle_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        if index is None or not (0 <= index < len(self._matches)):
            return
        model_id = self._matches[index]
        self.query_one("#model-search-picker-input", Input).value = ""
        self.post_message(self.ModelSelected(model_id))
```

In `settings_sidebar.py` after the model `Select` (line 146), only for chat:

```python
            if id_prefix == "chat":
                from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker
                yield ModelSearchPicker(id="model-search-picker")
```

In `chat_screen.py` (add `from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker` to the module imports — `@on` needs the message class at class-definition time, same as `ModelCatalogRefreshed` in Task 6):

```python
    @on(ModelSearchPicker.ModelSelected)
    async def handle_model_search_selected(self, event: ModelSearchPicker.ModelSelected) -> None:
        """Insert a picked model as a transient option and select it (ADR-019)."""
        model_id = event.model_id.strip()
        if not model_id or not self.chat_window:
            return
        try:
            provider_select = self.chat_window.query_one("#chat-api-provider", Select)
            model_select = self.chat_window.query_one("#chat-api-model", Select)
        except Exception:
            return
        provider = str(provider_select.value or "").strip()
        # current_model=model_id makes the rebuild insert it as a transient option
        # when it is not in the merged options (Select rejects out-of-options values).
        await self._rebuild_chat_model_options(provider, current_model=model_id)
        model_select.value = model_id
        # Persist so the pick restores on next launch (sidebar re-inserts it as a
        # transient option when it is absent from the saved list).
        from tldw_chatbook.config import save_settings_to_cli_config
        save_settings_to_cli_config({"chat_defaults": {"model": model_id}})
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/Widgets/test_model_search_picker.py Tests/UI/ -q -k "model" `
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/model_search_picker.py tldw_chatbook/Widgets/settings_sidebar.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Widgets/test_model_search_picker.py
git commit -m "feat: type-to-filter model search picker with transient option insertion"
```

---

### Task 9: Settings UI toggles

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (Model discovery subsection, after the `SelectionList` at line 5419-5424; handlers near 4429-4570)
- Test: `Tests/UI/test_settings_model_catalog_toggles.py` (new)

- [ ] **Step 1: Write failing tests**: toggles render for all six providers; changing a toggle persists `[model_catalog]` via `save_settings_to_cli_config` (patch/spy it); saved values load back into widget state on mount.

- [ ] **Step 2: Run to verify failure**, then **Step 3: Implement** — first add `Checkbox` to the `textual.widgets` import in `settings_screen.py` (line 22-31 currently imports Button/Collapsible/Input/Rule/Select/SelectionList/Static/TextArea) and import `AUTO_REFRESH_PROVIDER_LIST_KEYS` from `tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings`. Then, in the compose block after the discovery `SelectionList` (line 5424):

```python
            yield Static("Automatic refresh (ADR-019)", classes="destination-section")
            yield Checkbox(
                "Auto-refresh model lists on startup",
                id="settings-model-catalog-auto-refresh",
            )
            with Horizontal(classes="settings-input-row"):
                yield Static("Refresh after (hours):", classes="settings-status-row")
                yield Input(
                    "24",
                    id="settings-model-catalog-stale-hours",
                    type="integer",
                    tooltip="0 = refetch every launch.",
                )
            for _provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                _pid = _provider.lower()
                with Horizontal(classes="settings-input-row"):
                    yield Checkbox(
                        f"{_provider}: auto-refresh",
                        id=f"settings-mc-auto-{_pid}",
                    )
                    yield Checkbox(
                        "save to config",
                        id=f"settings-mc-write-{_pid}",
                        tooltip=(
                            "Append newly discovered models to config.toml — "
                            "large catalogs like OpenRouter only add newly released "
                            "models after a first baseline."
                        ),
                    )
```

Add handlers (near the discovery handlers at 4429-4570; ensure `provider_config_key`, `load_settings`, and `load_model_catalog_settings` are imported — the first two already are in this file, add the third):

- On mount of this subsection (follow the screen's existing pattern for initializing control values): read `load_model_catalog_settings(load_settings())` and set the master checkbox, hours input, and per-provider checkboxes (compare normalized keys: `provider_config_key(provider) in settings.auto_refresh_disabled` → auto-refresh unchecked; `in settings.write_to_config` → write checked).
- `@on(Checkbox.Changed)` / `@on(Input.Changed)` for these ids → collect current widget states and persist immediately:

```python
save_settings_to_cli_config({
    "model_catalog": {
        "auto_refresh_enabled": master.value,
        "stale_after_hours": int(hours.value or "24"),
        "auto_refresh_disabled": [p for p in AUTO_REFRESH_PROVIDER_LIST_KEYS
                                  if not auto_boxes[p].value],
        "write_to_config": [p for p in AUTO_REFRESH_PROVIDER_LIST_KEYS
                            if write_boxes[p].value],
    }
})
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/UI/test_settings_model_catalog_toggles.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_model_catalog_toggles.py
git commit -m "feat: settings toggles for model catalog auto-refresh and write-through"
```

---

### Task 10: Live smoke check + full validation + backlog DoD

- [ ] **Step 1: Live smoke check** (manual, one time — real keys from the user's env, never logged):

```bash
python3 - <<'EOF'
import asyncio, os
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    discover_openai_compatible_models,
)

CASES = [
    ("openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ("anthropic", "Anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    ("mistralai", "MistralAI", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    ("moonshot", "Moonshot", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    ("zai", "ZAI", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
    ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", None),  # public
]

async def main():
    for provider, list_key, endpoint, env_var in CASES:
        result = await discover_openai_compatible_models(
            provider=provider, provider_list_key=list_key, endpoint=endpoint,
            api_key=os.environ.get(env_var) if env_var else None,
        )
        print(f"{list_key}: {result.status}, {len(result.models)} models")

asyncio.run(main())
EOF
```

Expected: `success` for every provider with a configured key (skip any whose key you don't have — note it); Z.AI returns an OpenAI-shaped list (confirming the assumption flagged in the spec); Anthropic returns > 20 models (pagination working).

- [ ] **Step 2: Full test suite + lint**

```bash
python3 -m pytest Tests/LLM_Provider_Catalog/ Tests/UI/ Tests/Widgets/test_model_search_picker.py Tests/test_config_model_catalog_defaults.py -q
python3 -m pytest Tests/ -q -x --timeout=600  # or the repo's standard quick suite; check Tests/README.md
```

- [ ] **Step 3: Update documentation** — AGENTS.md "Special Systems" (Tool Calling section area): add a short "Model Catalog Auto-Refresh (ADR-019)" entry pointing to the ADR/spec.

- [ ] **Step 4: Complete the backlog task** — mark every AC `- [x]`, add `## Implementation Notes` (approach, files, trade-offs), then:

```bash
backlog task edit <N> -s Done --notes "<summary>"
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "docs: AGENTS.md entry + backlog closeout for model catalog auto-refresh"
```

---

## Notes for the executor

- **DRY/YAGNI**: reuse `persist_discovered_models_to_settings`, `resolve_provider_list_key`, `provider_config_key`, `validate_url`, `save_settings_to_cli_config` — do not re-implement them.
- **Never log API keys**; the disk store holds model IDs + timestamps only.
- The spec's `call_from_thread` notification note is superseded by the async-worker approach in Task 6 (documented there).
- If `Tests/textual_test_harness.py` offers an established app-running pattern, prefer it for Tasks 6-9 tests over hand-rolled stubs.
