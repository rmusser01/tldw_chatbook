# Generic Hosted Provider Engine — Phase 2 (Inference-Cloud Presets + Long-Tail Engine Swap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Together, Fireworks, Cerebras, and Perplexity as engine preset records (no per-provider modules), and move the ADR-146 custom-endpoint `openai_compatible` family onto the strict engine with keyless support and a documented tolerant profile.

**Architecture:** Phase 1 (branch `feat/provider-engine-phase1`, ADR-179) built `provider_registry.py` (identity + preset records) and `LLM_Calls/hosted_provider_engine.py` (strict factory on `hosted_chat`). Phase 2 adds four curated records (data only), two engine extensions the long tail needs (`bearer_optional` auth and tolerant top-level extras), and swaps `family_execution_key("openai_compatible")` from the legacy `custom` slot to a new engine-driven `custom-hosted` execution key — leaving the two named `custom-openai-api` slots untouched.

**Tech Stack:** Python ≥3.12, `requests` via `hosted_chat`. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md` (Phase 2 section + "Response variance and strictness") — read before starting; this plan argues from it.

## Global Constraints

- Registry module `tldw_chatbook/provider_registry.py` stays stdlib-only (no `tldw_chatbook.*` imports).
- Fail-closed strictness: required-shape validation (choices, message, tool calls, usage) is NEVER relaxed; tolerant extras are top-level/event keys only, shape-checked then dropped, and ONLY for the long-tail family record. Curated records keep `tolerant_response_extras=False` (parity-pinned).
- Curated cloud presets hard-require API keys; only the `custom-hosted` family record uses `bearer_optional` auth (parity-pinned both directions).
- `moonshot.py`/`zai.py` and the two named `custom-openai-api` slots are NOT migrated or behavior-changed.
- **xAI/Grok is excluded** — no record, no handler, no docs (ADR-179).
- API keys env-first, masked `api_settings` otherwise, never logged (ADR-012). No shipped `model` key for any preset whose models come from discovery (Phase 1 blank-model lesson: a present-but-blank model is rejected by the resolver; omit the key entirely).
- Tests: `.venv/bin/python -m pytest` from the worktree root, targeted files only. Commit after every green step, conventional commits with ADR-179 references.
- Subagent-dispatch note: if dispatch limits are exhausted, tasks may run inline per the Phase 1 precedent (record in ledger).
- Live probes are env-gated, paid, skip-clean; they do NOT block task Done (Phase 1 precedent — recorded pending, like the Databricks gate).

## Provider facts (verified 2026-09-24; sources: provider docs + [Perplexity docs](https://docs.perplexity.ai))

| Provider | key | Base URL | Env var | Discovery route | Notes |
|---|---|---|---|---|---|
| Together | `together` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | `models` | tools yes |
| Fireworks | `fireworks` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | `models` | tools yes; R1-family models return `reasoning_content` → disposition `proprietary` |
| Cerebras | `cerebras` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | `models` | tools yes |
| Perplexity | `perplexity` | `https://api.perplexity.ai` (NO `/v1` on chat) | `PERPLEXITY_API_KEY` | `v1/models` | responses carry top-level `citations` → `response_allowances={"citations"}` (verified at live probe); Chat-Completions surface used — the Agent API transition is spec open item O-4 |

Plan decision (spec-consistent refinement): the spec's `auth_scheme="none"` for the long tail is implemented as `"bearer_optional"` — send `Authorization` only when a credential resolved, since ADR-146 entries MAY carry keys and must keep sending them. A literal no-auth scheme exists for nothing in Phase 2; `bearer_optional` covers keyless AND keyed entries with one semantic.

Deferred minors from Phase 1 folded into Task 1 (per ledger triage): the unguarded `else` in the catalog service's engine branch, and the `resolve_hosted_engine_request` alias parity pin.

---

### Task 1: Backlog task + engine hardening (deferred minors)

**Files:**
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py` (`_resolve_hosted_provider` engine branch)
- Modify: `Tests/LLM_Calls/test_hosted_provider_engine_handler.py` (alias parity test)
- Create: backlog task via CLI

**Interfaces:**
- Produces: hardened engine branch (`engine_driven` guard); pinned alias parity. No new interfaces.

- [ ] **Step 1: Create the Phase-2 backlog task**

```bash
backlog task create "Generic hosted provider engine Phase 2: inference-cloud presets + long-tail engine swap" \
  -d "Implement Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md Phase 2" \
  -a @Robert -s "In Progress" \
  --ac "Together/Fireworks/Cerebras/Perplexity selectable+usable as presets,preset-cost test proves no per-provider module,custom-ep openai_compatible family executes via engine,keyless entries keep working,tolerant profile scoped to long tail only,curated presets still hard-require keys,docs updated"
backlog task edit <id> --plan "Plan: Docs/superpowers/plans/2026-09-24-generic-hosted-provider-engine-phase2.md"
```

- [ ] **Step 2: Write the failing tests**

In `Tests/LLM_Calls/test_hosted_provider_engine_handler.py` add:

```python
def test_resolve_hosted_engine_request_alias_matches_private_signature():
    """resolve_hosted_engine_request forwards the full private surface."""
    import inspect
    from tldw_chatbook.LLM_Calls import hosted_provider_engine as engine

    public = inspect.signature(engine.resolve_hosted_engine_request).parameters
    private = inspect.signature(engine.resolve_hosted_request).parameters
    assert list(public) == list(private)
```

In a new `Tests/LLM_Provider_Catalog/test_engine_branch_guard.py`:

```python
import pytest

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired  # only if the service raises this; otherwise skip this import


def test_engine_branch_rejects_non_engine_strict_hosted_keys(monkeypatch):
    """A strict-hosted key that is NOT engine-driven must fail loudly,
    not be silently handed to resolve_hosted_engine_request."""
    from tldw_chatbook.LLM_Provider_Catalog import local_llm_provider_catalog_service as service
    from tldw_chatbook import provider_registry

    record = provider_registry.DATABRICKS
    fake = dataclasses_replaced = __import__("dataclasses").classes.replace(
        record, engine_driven=False, key="fake-strict-hosted"
    )
    registry_copy = dict(provider_registry.RECORDS_BY_KEY)
    registry_copy["fake-strict-hosted"] = fake
    monkeypatch.setattr(provider_registry, "RECORDS_BY_KEY", registry_copy)
    # _resolve_hosted_provider is module-internal; call through it with the fake key
    with pytest.raises(Exception) as excinfo:
        service._resolve_hosted_provider(
            "fake-strict-hosted",
            app_config={"api_settings": {"fake-strict-hosted": {}}},
            environ={},
        )
    assert "fake-strict-hosted" in str(excinfo.value)
```

(If `_resolve_hosted_provider`'s real signature differs — read it first at ~L256 of the service module — adapt the call to the actual parameters; the assertion is only that a non-engine strict-hosted key raises rather than reaching `resolve_hosted_engine_request`.)

- [ ] **Step 3: Run to verify RED**

Run: `.venv/bin/python -m pytest Tests/LLM_Provider_Catalog/test_engine_branch_guard.py Tests/LLM_Calls/test_hosted_provider_engine_handler.py -q`
Expected: new tests FAIL (alias parity mismatch is possible but unlikely — the branch-guard test fails because the unguarded `else` calls the engine resolver and raises a resolution error not naming the key, or succeeds).

- [ ] **Step 4: Implement**

- Alias: make `resolve_hosted_engine_request` forward via `*args, **kwargs` to `resolve_hosted_request` OR keep the explicit signature (if the parity test already passes, keep explicit and the test pins it — either state is green; do not churn).
- Guard: in `_resolve_hosted_provider`'s engine branch, before calling `resolve_hosted_engine_request`, check `record.engine_driven`; if not, `raise` an error naming the key (follow the module's existing error style for unsupported strict-hosted keys).

- [ ] **Step 5: GREEN + regression**

Run: `.venv/bin/python -m pytest Tests/LLM_Provider_Catalog/ Tests/LLM_Calls/test_hosted_provider_engine_handler.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py Tests/LLM_Provider_Catalog/test_engine_branch_guard.py Tests/LLM_Calls/test_hosted_provider_engine_handler.py
git commit -m "fix: guard catalog engine branch on engine_driven; pin resolver alias parity (ADR-179)"
```

---

### Task 2: Engine `bearer_optional` auth

**Files:**
- Modify: `tldw_chatbook/provider_registry.py` (`ProviderRecord.auth_scheme` docstring gains the literal)
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py` (resolution + transport construction)
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py` (header construction, ~L520-565)
- Test: `Tests/LLM_Calls/test_hosted_provider_engine_auth.py` (new)

**Interfaces:**
- Produces: `auth_scheme` values `{"bearer", "bearer_optional", "api_key_header"}` (Phase 3 adds `api_key_header`); `resolve_hosted_request` raises `ChatConfigurationError` on a missing key ONLY when `record.auth_scheme == "bearer"`; `HostedHTTPTransportConfig` accepts an empty `api_key` when `auth_scheme == "bearer_optional"`; `owned_json_post` omits the `Authorization` header entirely when the key is empty (never sends `Bearer ` with nothing).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Calls/test_hosted_provider_engine_auth.py
import dataclasses

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.LLM_Calls.hosted_provider_engine import resolve_hosted_request
from tldw_chatbook.provider_registry import DATABRICKS

_OPTIONAL = dataclasses.replace(
    DATABRICKS,
    key="optional-auth",
    display_name="Optional Auth",
    api_key_env_var=None,
    api_key_env_candidates=(),
    auth_scheme="bearer_optional",
)


def test_bearer_preset_still_requires_a_key():
    with pytest.raises(ChatConfigurationError):
        resolve_hosted_request(
            DATABRICKS,
            explicit_base_url="https://dbc-1.cloud.databricks.com",
            app_config={"api_settings": {"databricks": {}}},
            environ={},
        )


def test_bearer_optional_resolves_without_a_key():
    resolution = resolve_hosted_request(
        _OPTIONAL,
        explicit_base_url="https://anywhere.example/v1",
        app_config={"api_settings": {"optional-auth": {}}},
        environ={},
    )
    assert resolution.api_key == ""


def test_bearer_optional_still_uses_a_resolved_key():
    resolution = resolve_hosted_request(
        _OPTIONAL,
        explicit_api_key="stored-key",
        explicit_base_url="https://anywhere.example/v1",
    )
    assert resolution.api_key == "stored-key"


def test_transport_omits_authorization_header_when_key_empty(monkeypatch):
    """No `Bearer ` header with an empty key — keyless local servers reject it."""
    import tldw_chatbook.LLM_Calls.hosted_chat as hosted_chat

    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"id": "r", "object": "chat.completion", "created": 1,
                    "model": "m",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": "ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        return _FakeResponse()

    monkeypatch.setattr(hosted_chat.requests.Session, "post", fake_post)
    hosted_chat.owned_json_post(
        config=hosted_chat.HostedHTTPTransportConfig(
            provider="optional-auth",
            base_url="https://anywhere.example/v1",
            api_key="",
            timeout=10.0,
            retries=0,
            retry_delay=0.0,
            auth_scheme="bearer_optional",
        ),
        route="chat/completions",
        payload={"model": "m", "messages": [], "stream": False},
        streaming=False,
    )
    assert "Authorization" not in captured["headers"]
```

- [ ] **Step 2: RED** — Run: `.venv/bin/python -m pytest Tests/LLM_Calls/test_hosted_provider_engine_auth.py -q` — Expected: FAIL (no `auth_scheme` on transport config / resolution still requires keys).

- [ ] **Step 3: Implement**

1. `HostedHTTPTransportConfig` gains `auth_scheme: str = "bearer"`. Validation at `owned_json_post` (~L517-534): when `auth_scheme == "bearer"`, keep the current non-empty-key requirement; when `"bearer_optional"`, allow an empty string. Header construction (~L561): `Authorization: Bearer …` only when `config.api_key` is non-empty (all schemes — never send an empty Bearer).
2. Engine resolution (`_resolve_api_key`): required only when `record.auth_scheme == "bearer"`; for `bearer_optional` return `""` when nothing resolves (keep the explicit/settings/env precedence when something does).
3. Engine factory: pass `auth_scheme=record.auth_scheme` into `HostedHTTPTransportConfig`.
4. Registry: extend the `auth_scheme` field docstring/comment with the new literal.

- [ ] **Step 4: GREEN + regression**

Run: `.venv/bin/python -m pytest Tests/LLM_Calls/test_hosted_provider_engine_auth.py Tests/LLM_Calls/test_hosted_chat_allowances.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_zai.py Tests/LLM_Calls/test_moonshot.py -q`
Expected: PASS (default `bearer` keeps existing behavior byte-identical).

- [ ] **Step 5: Commit** — `feat: engine bearer_optional auth for keyed-or-keyless endpoints (ADR-179)`

---

### Task 3: Tolerant top-level extras (long-tail profile)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py` (top-level + event-level checks from Phase 1 Task 3)
- Modify: `tldw_chatbook/provider_registry.py` (`ProviderRecord.tolerant_response_extras: bool = False`)
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py` (feed the flag through)
- Test: `Tests/LLM_Calls/test_hosted_chat_allowances.py` (extend)

**Interfaces:**
- Produces: `normalize_hosted_chat_response(..., tolerant_top_level_extras: bool = False)` and the same kwarg on `HostedChatStream`/`hosted_chat_request`; engine passes `record.tolerant_response_extras`. When True, unknown top-level/event keys are shape-checked and dropped instead of rejected; choice/delta/message/tool/usage checks are untouched.

- [ ] **Step 1: Failing tests** — extend `Tests/LLM_Calls/test_hosted_chat_allowances.py`:

```python
def test_tolerant_profile_ignores_any_top_level_extra():
    turn = normalize_hosted_chat_response(
        _ok_response(service_tier="x", citations=[ "https://a" ], anything_else={"k": 1}),
        finish_policy=_Policy(),
        tolerant_top_level_extras=True,
    )
    assert turn.text == "hi"


def test_tolerant_profile_still_fails_closed_on_required_shapes():
    bad = _ok_response()
    bad["choices"] = []
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            bad, finish_policy=_Policy(), tolerant_top_level_extras=True
        )


def test_tolerant_profile_does_not_extend_to_choice_or_delta_keys():
    event_like = _ok_response()
    event_like["choices"][0]["logprobs"] = None
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            event_like, finish_policy=_Policy(), tolerant_top_level_extras=True
        )


def test_engine_record_flag_feeds_through():
    import dataclasses
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
        normalize_hosted_provider_response,
    )
    from tldw_chatbook.provider_registry import DATABRICKS

    tolerant = dataclasses.replace(DATABRICKS, tolerant_response_extras=True)
    turn = normalize_hosted_provider_response(
        tolerant, _ok_response(citations=["https://a"])
    )
    assert turn.text == "hi"
    # curated record stays strict:
    with pytest.raises(Exception):
        normalize_hosted_provider_response(
            DATABRICKS, _ok_response(citations=["https://a"])
        )
```

(Reuse the file's existing `_ok_response`/`_Policy` helpers — extend `_ok_response(**extra)` if it lacks that shape.)

- [ ] **Step 2: RED** — Expected: FAIL on the missing kwarg.

- [ ] **Step 3: Implement** — mirror the Phase 1 allowances mechanics: where the checks compute `set(response) - KNOWN - allowed_extra_keys`, a True flag replaces the reject-on-remainder with drop-all-remainder for the TOP-LEVEL and EVENT-level checks only. Engine wrapper passes the record flag. Add the record field default `False`.

- [ ] **Step 4: GREEN + regression**

Run: `.venv/bin/python -m pytest Tests/LLM_Calls/test_hosted_chat_allowances.py Tests/LLM_Calls/test_hosted_provider_engine_policy.py Tests/LLM_Calls/test_hosted_chat.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat: tolerant top-level response profile scoped to long-tail records (ADR-179)`

---

### Task 4: Four inference-cloud presets + preset-cost test

**Files:**
- Modify: `tldw_chatbook/provider_registry.py` (four records + ALL_RECORDS)
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (four dispatch entries)
- Modify: `tldw_chatbook/config.py` (`[providers]` empty seeds + four `[api_settings.*]` tables)
- Modify: `Chat/provider_readiness.py`, `Chat/console_provider_support.py`, `Agents/native_tools.py`, `LLM_Provider_Catalog/model_catalog_settings.py` (+ `openai_compatible_model_discovery.py` only if a preset's default URL fails the discovery gate — check `/inference/v1` for fireworks)
- Test: `Tests/test_provider_registry.py` (extend), `Tests/Chat/test_dispatch_registry_parity.py` (extend), new `Tests/LLM_Calls/test_inference_cloud_presets.py`

**Interfaces:**
- Consumes: `build_hosted_chat_handler`, `ENGINE_PROVIDER_PARAM_MAP`, registry parity-test patterns (Phase 1).
- Produces: registry keys `together`, `fireworks`, `cerebras`, `perplexity` (config keys `Together`, `Fireworks`, `Cerebras`, `Perplexity`), dispatch-registered and audited via the registry derivation.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Calls/test_inference_cloud_presets.py
"""Preset cheapness: four inference clouds are records + one line each."""
import pytest

from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

_PRESETS = ("together", "fireworks", "cerebras", "perplexity")


@pytest.mark.parametrize("key", _PRESETS)
def test_preset_registered_and_record_shaped(key):
    record = RECORDS_BY_KEY[key]
    assert record.engine_driven is True
    assert record.auth_scheme == "bearer"
    assert record.tolerant_response_extras is False
    assert record.auto_refresh is True
    assert record.native_tools is True
    assert record.base_url_suffix is None  # full default URLs shipped
    assert callable(API_CALL_HANDLERS[key])


def test_no_provider_specific_python_modules_ship_for_presets():
    """The Phase 2 promise: presets are data, not modules."""
    from pathlib import Path
    llm_calls = Path("tldw_chatbook/LLM_Calls")
    for key in _PRESETS:
        assert not list(llm_calls.glob(f"{key}*.py")), key


def test_preset_default_urls_and_env_vars():
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY
    expected = {
        "together": ("https://api.together.xyz/v1", ("TOGETHER_API_KEY",)),
        "fireworks": ("https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",)),
        "cerebras": ("https://api.cerebras.ai/v1", ("CEREBRAS_API_KEY",)),
        "perplexity": ("https://api.perplexity.ai", ("PERPLEXITY_API_KEY",)),
    }
    for key, (url, envs) in expected.items():
        record = RECORDS_BY_KEY[key]
        assert record.default_base_url == url, key
        assert record.api_key_env_candidates == envs, key


def test_perplexity_discovery_route_and_allowances():
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY
    record = RECORDS_BY_KEY["perplexity"]
    assert record.discovery_route == "v1/models"  # chat base has no /v1
    assert "citations" in record.response_allowances


def test_fireworks_proprietary_reasoning_disposition():
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY
    assert RECORDS_BY_KEY["fireworks"].reasoning_disposition == "proprietary"
```

(Fix the deliberate import wart in the header — import only `RECORDS_BY_KEY`.) Also extend `Tests/test_provider_registry.py`'s parity tests: the audited-set/cloud/auto-refresh/native-tools comparisons must now include the four keys (no exclusions), and the env-var/base-URL tomllib test picks them up automatically once config tables ship.

- [ ] **Step 2: RED** — Expected: FAIL (`RECORDS_BY_KEY` misses the keys).

- [ ] **Step 3: Implement**

1. Four records in `provider_registry.py` (follow `DATABRICKS`'s shape; `settings_defaults` WITHOUT a `model` key — models come from discovery; timeout 90, retries 3, retry_delay 5.0, streaming True, api_key_env_var set):

```python
TOGETHER = ProviderRecord(
    key="together", config_key="Together", display_name="Together",
    classification=_CLOUD,
    api_key_env_var="TOGETHER_API_KEY",
    api_key_env_candidates=("TOGETHER_API_KEY",),
    default_base_url="https://api.together.xyz/v1",
    native_tools=True, auto_refresh=True,
    settings_defaults={"api_key_env_var": "TOGETHER_API_KEY", "streaming": True,
                       "timeout": 90, "retries": 3, "retry_delay": 5.0},
    engine_driven=True,
)
# fireworks: same shape, reasoning_disposition="proprietary"
# cerebras: same shape
# perplexity: same shape + discovery_route="v1/models",
#   response_allowances=frozenset({"citations"}),
#   base_url_suffix=None (default base already correct; chat route appended by hosted_chat)
```

2. `ALL_RECORDS` += the four; dispatch entries in `Chat_Functions.py` (`"together": build_hosted_chat_handler(TOGETHER)`, …) — the audited set and cloud classification derive automatically; verify parity tests.
3. `config.py`: `"Together" = []` etc. in `[providers]` (empty — discovery fills; same rationale as Databricks) and four `[api_settings.<key>]` tables mirroring `settings_defaults` + `api_base_url = "<default>"` (these providers HAVE fixed defaults, unlike Databricks) — NO `model` key.
4. Readiness/display/native-tools/catalog literal sets: add the four keys each (auto-refresh uses config-key spellings: "Together", "Fireworks", "Cerebras", "Perplexity").
5. Discovery gate: run `supports_openai_compatible_model_discovery` mentally per preset default URL; if `/inference/v1` (fireworks) is not an eligible explicit path, add it to `_EXPLICIT_OPENAI_COMPATIBLE_ENDPOINT_PATHS` following the `/openai/v1` precedent from Phase 1 Task 12 (+ its `_models_path_for_endpoint_path` branch if needed) with a test.

- [ ] **Step 4: GREEN + regression**

Run: `.venv/bin/python -m pytest Tests/LLM_Calls/test_inference_cloud_presets.py Tests/test_provider_registry.py Tests/Chat/test_dispatch_registry_parity.py Tests/test_config_databricks.py Tests/test_config_model_catalog_defaults.py Tests/LLM_Provider_Catalog/ -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat: together/fireworks/cerebras/perplexity presets — records, not modules (ADR-179)`

---

### Task 5: Custom-endpoint engine swap (`custom-hosted`)

**Files:**
- Modify: `tldw_chatbook/provider_registry.py` (`CUSTOM_HOSTED` record)
- Modify: `tldw_chatbook/Chat/custom_endpoint_registry.py` (`family_execution_key` ~L453-464)
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (dispatch entry)
- Modify: `tldw_chatbook/Chat/provider_continuation.py` (Literal + `_PAIRINGS`: `("custom-hosted", "chat_completions")`)
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (only where the resolution for custom-ep entries picks the execution key — read `family_execution_key` usages first)
- Test: new `Tests/Chat/test_custom_endpoint_engine_swap.py`; extend `Tests/Chat/test_custom_endpoint_registry.py` expectations

**Interfaces:**
- Produces: `family_execution_key("openai_compatible") == "custom-hosted"`; `API_CALL_HANDLERS["custom-hosted"] = build_hosted_chat_handler(CUSTOM_HOSTED)`; `"custom-hosted": ENGINE_PROVIDER_PARAM_MAP` in `PROVIDER_PARAM_MAP`. The `custom-hosted` record: `key="custom-hosted"`, `config_key="Custom-hosted"` (not in any `[providers]` table — execution-only), `classification="local"`, `engine_driven=True`, `default_base_url=None` (per-entry explicit URLs), `auth_scheme="bearer_optional"`, `tolerant_response_extras=True`, `native_tools=True`, `auto_refresh=False`, no env candidates, `settings_defaults={"streaming": True, "timeout": 120, "retries": 1, "retry_delay": 1.0}` (matching the legacy `chat_with_custom_openai` defaults at `LLM_API_Calls_Local.py` ~L2102-2104).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_custom_endpoint_engine_swap.py
"""openai_compatible registry entries execute through the strict engine."""
import pytest


def test_family_execution_key_routes_openai_compatible_to_engine():
    from tldw_chatbook.Chat.custom_endpoint_registry import family_execution_key
    assert family_execution_key("openai_compatible") == "custom-hosted"
    assert family_execution_key("llama_cpp") == "llama_cpp"
    assert family_execution_key("ollama") == "ollama"


def test_custom_hosted_record_shape():
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY
    record = RECORDS_BY_KEY["custom-hosted"]
    assert record.auth_scheme == "bearer_optional"
    assert record.tolerant_response_extras is True
    assert record.default_base_url is None
    assert record.auto_refresh is False


def test_dispatch_registration_and_param_map():
    from tldw_chatbook.Chat.Chat_Functions import (
        API_CALL_HANDLERS, ENGINE_PROVIDER_PARAM_MAP, PROVIDER_PARAM_MAP,
    )
    assert callable(API_CALL_HANDLERS["custom-hosted"])
    assert PROVIDER_PARAM_MAP["custom-hosted"] is ENGINE_PROVIDER_PARAM_MAP


def test_keyless_and_keyed_entries_both_execute(monkeypatch):
    """One canned engine transport; keyless sends no Authorization, keyed does."""
    import tldw_chatbook.LLM_Calls.hosted_provider_engine as engine
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    captured = []

    class _Turn:
        text = "ok"; tool_calls = (); assistant_message = {"role": "assistant", "content": "ok"}
        finish_reason = "stop"; reasoning_content = None
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    def fake_owned_json_post(*, config, route, payload, streaming):
        captured.append(config)
        if streaming:
            class _S:
                def __iter__(self_inner): return iter(())
                def close(self_inner): pass
                terminal_turn = _Turn()
            return _S()
        return {"id": "r", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": _Turn.usage}

    monkeypatch.setattr(engine, "owned_json_post", fake_owned_json_post)
    handler = engine.build_hosted_chat_handler(RECORDS_BY_KEY["custom-hosted"])
    messages = [{"role": "user", "content": "hi"}]

    handler(input_data=messages, api_base_url="http://127.0.0.1:9/v1",
            model="local-model", streaming=False)          # keyless
    handler(input_data=messages, api_base_url="http://127.0.0.1:9/v1",
            model="local-model", api_key="entry-key", streaming=False)  # keyed

    assert captured[0].api_key == "" and captured[0].auth_scheme == "bearer_optional"
    assert captured[1].api_key == "entry-key"


def test_named_custom_slots_untouched():
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
        chat_with_custom_openai, chat_with_custom_openai_2,
    )
    assert API_CALL_HANDLERS["custom-openai-api"] is chat_with_custom_openai
    assert API_CALL_HANDLERS["custom-openai-api-2"] is chat_with_custom_openai_2
```

Also update `Tests/Chat/test_custom_endpoint_registry.py`: any assertion expecting `family_execution_key("openai_compatible") == "custom"` flips to `"custom-hosted"` (grep the file first; keep every other assertion intact).

- [ ] **Step 2: RED** — Expected: FAIL on the execution-key change.

- [ ] **Step 3: Implement**

1. Registry record `CUSTOM_HOSTED` per the Interfaces block; add to `ALL_RECORDS` (audited set and local classification derive; the registry's config-table parity test skips records without `[api_settings]` tables — verify).
2. `family_execution_key`: `"custom-hosted" if family == "openai_compatible" else family`.
3. Dispatch + param map entries.
4. `provider_continuation.py`: extend Literal + `_PAIRINGS`.
5. Gateway: grep `family_execution_key(` call sites — the resolution builder for custom-ep entries flows the returned key into `execution_key`; nothing else should change (per-call `base_url`/`model`/`api_key` overrides already ride the resolution → `chat_api_call` explicit kwargs → engine resolver honors them). If any site hardcodes `"custom"`, route it through `family_execution_key`.
6. Tolerant + optional-auth flags flow from the record (Tasks 2-3).

- [ ] **Step 4: GREEN + regression**

Run: `.venv/bin/python -m pytest Tests/Chat/test_custom_endpoint_engine_swap.py Tests/Chat/test_custom_endpoint_registry.py Tests/Chat/test_dispatch_registry_parity.py Tests/test_provider_registry.py -q`
Then the Console suites that exercise custom-ep end to end (grep `custom-ep:` in Tests/Chat — run those files; their failure sets must equal any documented pre-existing baselines).

- [ ] **Step 5: Commit** — `feat: custom-endpoint openai_compatible family executes via the strict engine (ADR-179)`

---

### Task 6: Live probe file (env-gated, skip-clean)

**Files:**
- Create: `Tests/Chat/test_live_inference_cloud_api.py`

**Interfaces:**
- Consumes: the Phase 1 `test_live_databricks_api.py` subprocess pattern (isolated profile, structural-metadata-only stdout).

- [ ] **Step 1: Write the probes**

One module, four providers, each double-gated on `<PROVIDER>_API_KEY` + an explicit opt-in flag (`TLDW_LIVE_TOGETHER=1` etc., following `test_live_moonshot_zai_api.py`'s gate style), with optional `TLDW_LIVE_<P>_MODEL` overrides. The child (one shared child script, provider-parameterized like the moonshot/zai one) per provider: `GET {base}/models` (Perplexity: `{base}/v1/models`) asserting ≥1 id; one non-streaming chat via `build_hosted_chat_handler(RECORDS_BY_KEY[key])` asserting text+usage; one streamed round-trip asserting terminal usage; envelope key-name capture printed as JSON. Structural tests (gate matrix, profile isolation ordering) run always.

- [ ] **Step 2: Verify skip-clean**

Run: `.venv/bin/python -m pytest Tests/Chat/test_live_inference_cloud_api.py -v`
Expected: structural tests PASS, paid probes SKIPPED with opt-in copy.

- [ ] **Step 3: Commit** — `test: inference-cloud live probes (env-gated, skip-clean; ADR-179)`

(If you have any of the four keys — Together and Cerebras have free tiers — run that provider's probe once and record the envelope keys; Perplexity's `citations` allowance gets confirmed or corrected from its output. Not a Done blocker; record pending like the Databricks gate.)

---

### Task 7: Docs + verification battery + close-out

**Files:**
- Modify: `Docs/User_Guide/settings.md`, `Docs/User_Guide/console.md`, `README.md`
- Modify: backlog task notes

- [ ] **Step 1: Docs** — Settings guide: add a combined "Inference clouds (Together, Fireworks, Cerebras, Perplexity)" subsection after the Databricks one — table of the four (default base, env var), discovery-first model lists (no shipped models), Perplexity note (citations tolerated; discovery at `/v1/models`), Fireworks note (private reasoning on R1-family → not shown in transcripts, zai-style). Console guide: one short subsection (ordinary streaming path; standard tool loop; discovery reuses chat credential). README: append the four env vars to the API-keys list. Custom-endpoints section in settings.md: note the family now runs the strict engine (unknown-but-safe response fields ignored; malformed core shapes still fail closed; keyless local servers keep working). No xAI anywhere.
- [ ] **Step 2: Full targeted battery**

```bash
.venv/bin/python -m pytest Tests/test_provider_registry.py Tests/LLM_Calls/test_inference_cloud_presets.py Tests/LLM_Calls/test_hosted_provider_engine_auth.py Tests/LLM_Calls/test_hosted_chat_allowances.py Tests/LLM_Calls/test_hosted_provider_engine_handler.py Tests/Chat/test_dispatch_registry_parity.py Tests/Chat/test_custom_endpoint_engine_swap.py Tests/Chat/test_custom_endpoint_registry.py Tests/Chat/test_databricks_console_surfaces.py Tests/test_config_databricks.py Tests/LLM_Provider_Catalog/ Tests/Chat/test_sensitive_llm_logging.py Tests/Chat/test_chat_unit_mocked_APIs.py -q
```
Expected: all green (document any pre-existing deviations in the task notes with stash-verified evidence, Phase 1 style).

- [ ] **Step 3: Backlog close** — check the ACs that are demonstrably met; live-probe ACs recorded pending; `backlog task edit <id> --notes "..."`.

- [ ] **Step 4: Commit** — `docs: inference-cloud presets + strict custom endpoints (ADR-179 phase 2)`

---

## Self-Review (completed)

**Spec coverage:** Phase 2 AC 1 (four presets, records-only, preset-cost test) → Task 4; AC 2 (engine swap + identity/cached-models/credential-precedence preserved) → Task 5 (entries keep `custom-ep:<slug>` identity, cached `models`, env→stored precedence — only the execution key changes); AC 3 (keyless) → Tasks 2+5; AC 4 (tolerant profile) → Task 3; docs → Task 7. The spec's "evidence-gated swap with parity tests against the old path" → Task 5 Step 4's custom-ep suite runs + the canned-behavior tests in the new file.

**Type consistency:** `auth_scheme` literal set and `tolerant_response_extras` names are used identically in Tasks 2/3/4/5; `custom-hosted` key spelling consistent across registry/dispatch/continuation/tests; `discovery_route="v1/models"` matches the Phase 1 field.

**Placeholders:** Task 1's branch-guard test and Task 6's child script carry explicit "read the real signature / follow the named pattern" instructions with named sources — no unnamed work. Task 4's fireworks/cerebras records are written as "same shape" comments next to a complete TOGETHER record — the deltas are enumerated in the test file (`test_preset_default_urls_and_env_vars` pins every value).

**Known deferrals:** live probes pending credentials (Phase 1 precedent); Perplexity Agent-API transition recorded as spec open item O-4; Phase 1 deferred minors beyond Task 1's two remain drive-by.
