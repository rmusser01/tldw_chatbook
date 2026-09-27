# Generic Hosted Provider Engine — Phase 1 (Registry + Engine + Databricks) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the provider registry (single source of truth for provider identity), the generic strict adapter engine on `hosted_chat`, and ship Databricks (AI Gateway / external models) as the first engine-driven provider, end-to-end.

**Architecture:** A new stdlib-only leaf module `tldw_chatbook/provider_registry.py` holds one frozen record per provider (identity fields for all providers; preset fields for engine-driven ones). A new `LLM_Calls/hosted_provider_engine.py` builds `chat_with_*` handlers from preset-variant records by parameterizing the boilerplate currently duplicated in `zai.py`/`moonshot.py`. Databricks is a record plus one dispatch registration line. Existing providers are not migrated (ADR-063 evidence-gated policy).

**Tech Stack:** Python ≥3.12, `requests` (existing, via `hosted_chat`), stdlib `dataclasses`. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md` — read it before starting; this plan argues from it.

## Global Constraints

- Registry module `tldw_chatbook/provider_registry.py` is stdlib-only: **no `tldw_chatbook.*` imports** (config.py consumes it; import cycles are the failure this rule prevents).
- Fail-closed strictness: unknown response/stream fields fail closed unless in the record's `response_allowances`; required-shape validation (choices, message, tool calls, usage) is **never** relaxed.
- `moonshot.py` and `zai.py` are NOT migrated or modified (beyond import-surface additions explicitly listed below).
- **xAI/Grok is excluded** — never add a record, preset, or handler for it (ADR-179 records this).
- API keys: env var first, else masked `api_settings` value; never prefilled, never logged (ADR-012).
- Run tests from the repo root with the project venv active. **Targeted runs only** unless the user asks for a full sweep.
- ADR-179 must exist before any code task starts (Task 1).
- Commit after every green step. Conventional commits (`feat:`, `test:`, `docs:`, `refactor:`).
- Design-token rules (ADR-150) do not apply — Phase 1 adds no styled UI; settings copy only.

---

### Task 1: ADR-179 and Phase-1 backlog task

**Files:**
- Create: `backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md`
- Reference: `Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md`

**Interfaces:**
- Produces: ADR-179 referenced by every later task's docs and the backlog task.

- [ ] **Step 1: Write ADR-179**

Use the repo's existing ADR format (read `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md` for structure). Content requirements — status `Accepted` (2026-09-23), context, decision (verbatim essentials):

```markdown
# ADR-179: Generic hosted provider engine and preset registry

## Status
Accepted (2026-09-23)

## Context
Adding a first-class hosted provider costs a ~1,000-line strict adapter plus
edits to ~20 scattered literal tables (Moonshot/ZAI: 39 files, +9,185/-1,052).
Goal: Hermes-style provider breadth without losing the strict hosted wire
boundary of ADR-062/063.

## Decision
1. `tldw_chatbook/provider_registry.py` (stdlib-only leaf) is the single
   source of provider identity; the scattered literal tables
   (`_cloud_provider_keys`, readiness key sets, display names, endpoint maps,
   `NATIVE_TOOLS_PROVIDERS`, `AUTO_REFRESH_PROVIDER_LIST_KEYS`,
   `SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS`, continuation pairings) become
   registry-derived, each guarded by a parity test.
2. Engine-driven providers are preset records consumed by
   `LLM_Calls/hosted_provider_engine.py::build_hosted_chat_handler`,
   which delegates transport to `hosted_chat` (ADR-062 boundary unchanged).
3. Response strictness stays fail-closed; per-preset `response_allowances`
   (recorded from live-probe envelopes) are the only tolerated extras.
   The long-tail tolerant profile (Phase 2) is scoped to user-registered
   custom endpoints only.
4. moonshot.py and zai.py are not migrated (ADR-063 evidence-gated policy).
5. xAI/Grok support is deliberately excluded (maintainer decision).
6. Bedrock is in scope via its OpenAI-compatible endpoints + Bearer API keys
   (AWS, Dec 2025); native Converse wire and SigV4 stay out (fallback only).

## Consequences
Adding an OpenAI-compatible hosted provider becomes one registry record +
one dispatch registration line + tests. Sensitive-request audit and
cloud/local classification become registry-coverage tests instead of
hand-typed literals. Related: ADR-002, ADR-012, ADR-020, ADR-062, ADR-063,
ADR-146.
```

- [ ] **Step 2: Create the Phase-1 backlog task and link everything**

```bash
backlog task create "Generic hosted provider engine Phase 1: registry + engine + Databricks" \
  -d "Implement Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md Phase 1" \
  -a @Robert -s "In Progress" \
  --ac "Databricks usable in Console (streaming + non-streaming) via AI Gateway (live-verified),engine contract tests parameterized over records green,registry coverage parity tests green,no behavior change for existing providers,ADR-179 linked,docs updated"
backlog task edit <id> --plan "Plan: Docs/superpowers/plans/2026-09-23-generic-hosted-provider-engine-phase1.md"
```

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md
git commit -m "docs: add ADR-179 generic hosted provider engine and preset registry"
```

---

### Task 2: Provider registry module with parity tests

**Files:**
- Create: `tldw_chatbook/provider_registry.py`
- Test: `Tests/test_provider_registry.py`

**Interfaces:**
- Produces: `ProviderRecord` (frozen dataclass), module-level records tuple
  `ALL_RECORDS`, lookups `RECORDS_BY_KEY: dict[str, ProviderRecord]`,
  derived sets `CLOUD_PROVIDER_CONFIG_KEYS: tuple[str, ...]`,
  `LOCAL_PROVIDER_CONFIG_KEYS`, `ENGINE_RECORDS: tuple[ProviderRecord, ...]`,
  `AUTO_REFRESH_KEYS: frozenset[str]`, `NATIVE_TOOLS_KEYS: frozenset[str]`,
  `AUDITED_ENDPOINT_KEYS: frozenset[str]` (== `frozenset(API_CALL_HANDLERS)`
  equivalents incl. aliases), and `DATABRICKS: ProviderRecord`.
- Consumes: nothing from `tldw_chatbook` (stdlib only).

- [ ] **Step 1: Write the failing parity tests**

The tests compare registry data with today's literal tables — transcribe
values into records until green. This is the safety net that makes the
refactor honest (and guards the `e080a2fb92` class of bug).

```python
# Tests/test_provider_registry.py
"""Registry parity tests: registry data must equal today's literal tables."""
from tldw_chatbook.provider_registry import (
    ALL_RECORDS,
    AUDITED_ENDPOINT_KEYS,
    CLOUD_PROVIDER_CONFIG_KEYS,
    DATABRICKS,
    ENGINE_RECORDS,
    RECORDS_BY_KEY,
)
from tldw_chatbook.Chat.Chat_Functions import (
    API_CALL_HANDLERS,
    SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS,
)


def test_records_unique_and_complete():
    keys = [record.key for record in ALL_RECORDS]
    assert len(keys) == len(set(keys))
    # every dispatch key is a registry key or an alias of one
    assert set(API_CALL_HANDLERS) - set(RECORDS_BY_KEY) <= set()


def test_audited_set_matches_handlers():
    # Until Task 8 flips the derivation, the literal set lacks "databricks"
    # (the registry already includes it). Compare excluding it.
    assert SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS == AUDITED_ENDPOINT_KEYS - {"databricks"}


def test_cloud_classification_matches_config():
    # On dev, _cloud_provider_keys is a module-level LIST (config.py ~L9843),
    # not a callable. Same exclusion as the audited-set test until Task 9
    # flips the derivation.
    from tldw_chatbook.config import _cloud_provider_keys
    assert tuple(sorted(_cloud_provider_keys)) == tuple(
        sorted(set(CLOUD_PROVIDER_CONFIG_KEYS) - {"Databricks"})
    )


def test_databricks_preset_shape():
    record = DATABRICKS
    assert record.key == "databricks"
    assert record.classification == "cloud"
    assert record.engine_driven is True
    assert record.auth_scheme == "bearer"
    assert record.api_key_env_candidates == ("DATABRICKS_TOKEN",)
    assert record.base_url_suffix == "/openai/v1"
    assert record.default_base_url is None  # workspace host is per-account
    assert record.reasoning_disposition == "ignored"
    assert "databricks" in {r.key for r in ENGINE_RECORDS}
```

Note on the two exclusions: Tasks 8 and 9 flip the derivations and tighten
these tests to full equality (`test_audited_set_is_registry_derived_and_covers_all_handlers`,
`test_databricks_classified_cloud`). Same for the `AUTO_REFRESH_KEYS` and
`NATIVE_TOOLS_KEYS` parity tests added in Step 3: until Tasks 11–12 add
`databricks` to those literal sets, compare excluding it (use
`AUTO_REFRESH_KEYS - {"databricks"}` etc.), then drop the exclusion in the
task that flips the literal.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/test_provider_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.provider_registry'`.

- [ ] **Step 3: Write the registry module**

```python
"""Single source of truth for provider identity and hosted preset data.

Leaf module: stdlib imports ONLY, so ``config.py`` can consume it without
import cycles (ADR-179). Identity fields cover every provider; preset
fields (``engine_driven``) are consumed by
``LLM_Calls.hosted_provider_engine``. Behavior never lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

_CLOUD = "cloud"
_LOCAL = "local"


@dataclass(frozen=True)
class ProviderRecord:
    """Identity (all providers) + preset data (engine-driven providers)."""

    key: str
    config_key: str
    display_name: str
    classification: str
    api_key_env_var: str | None = None
    api_key_env_candidates: tuple[str, ...] = ()
    default_base_url: str | None = None
    native_tools: bool = False
    reasoning_effort: bool = False
    auto_refresh: bool = False
    settings_defaults: Mapping[str, object] = field(default_factory=dict)
    pricing_seeds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    # --- preset fields (engine-driven providers only) ---
    engine_driven: bool = False
    base_url_suffix: str | None = None
    finish_terminal: frozenset[str] = frozenset({"stop", "tool_calls", "length"})
    finish_provider_errors: frozenset[str] = frozenset()
    payload_flags: frozenset[str] = frozenset(
        {"temperature", "top_p", "max_tokens", "stop", "response_format", "seed", "n", "user"}
    )
    reasoning_effort_key: str | None = None
    extra_body_fields: Mapping[str, object] = field(default_factory=dict)
    response_allowances: frozenset[str] = frozenset()
    reasoning_disposition: str = "ignored"
    auth_scheme: str = "bearer"  # Phase 1: bearer only; see spec §3
    continuation_protocol: str | None = "chat_completions"
    discovery_route: str = "models"


# --- Databricks (first engine-driven preset; AI Gateway external models) ---
DATABRICKS = ProviderRecord(
    key="databricks",
    config_key="Databricks",
    display_name="Databricks",
    classification=_CLOUD,
    api_key_env_var="DATABRICKS_TOKEN",
    api_key_env_candidates=("DATABRICKS_TOKEN",),
    default_base_url=None,  # workspace host is per-account; user-configured
    native_tools=True,      # confirmed (or honestly disabled) at live gate (Task 14)
    reasoning_effort=False, # gateway model support varies; enable per-model later
    auto_refresh=True,
    settings_defaults={
        "api_key_env_var": "DATABRICKS_TOKEN",
        "model": "",
        "streaming": True,
        "timeout": 90,
        "retries": 3,
        "retry_delay": 5.0,
    },
    pricing_seeds={},       # gateway pricing is workspace/model-config dependent
    engine_driven=True,
    base_url_suffix="/openai/v1",
    reasoning_disposition="ignored",
    auth_scheme="bearer",
)

# --- existing cloud providers (opaque identity records) ---
# TRANSCRIPTION STEP: read config.py region containing the [providers] table
# (search for `"OpenAI"` around line 4041) and the [api_settings.*] tables
# (lines ~4093-4510). Transcribe each provider's api_key_env_var and default
# api_base_url EXACTLY from those tables. The records below carry the values
# to verify; fix them from config.py until the parity tests pass.
OPENAI = ProviderRecord(
    key="openai", config_key="OpenAI", display_name="OpenAI", classification=_CLOUD,
    api_key_env_var="OPENAI_API_KEY",
    api_key_env_candidates=("OPENAI_API_KEY",),
    default_base_url="https://api.openai.com/v1",
    native_tools=True, reasoning_effort=True, auto_refresh=True,
)
# ... transcribe the same way for: anthropic, cohere, groq, openrouter,
# deepseek, mistral (config_key "MistralAI"), google, huggingface,
# moonshot, zai, qwencloud — copy each provider's env var name and default
# base URL from config.py's [api_settings.<key>] table.

# --- local providers (opaque identity records; classification=_LOCAL) ---
# Transcribe the LOCAL provider config keys from config.py's
# _cloud_provider_keys complement (search `_cloud_provider_keys` in
# config.py): llama_cpp/LlamaCPP, koboldcpp, oobabooga, tabbyapi, vllm,
# ollama, aphrodite, local-llm, custom-openai-api, custom-openai-api-2,
# mlx_lm, local_llamacpp, local_llamafile, local_vllm, local_ollama,
# local_mlx_lm — display names from Chat/console_provider_support.py
# `_PROVIDER_DISPLAY_NAMES` (line ~79).

ALL_RECORDS: tuple[ProviderRecord, ...] = (
    OPENAI, ANTHROPIC, COHERE, GROQ, OPENROUTER, DEEPSEEK, MISTRAL, GOOGLE,
    HUGGINGFACE, MOONSHOT, ZAI, QWENCLOUD, DATABRICKS,
    # ...local records...
)

RECORDS_BY_KEY: dict[str, ProviderRecord] = {record.key: record for record in ALL_RECORDS}
#: Dispatch-key aliases (legacy spellings) mapped onto canonical records.
ALIASES: dict[str, str] = {
    "mistralai": "mistral",
    "local_llamacpp": "llama_cpp",
    "local_llamafile": "llama_cpp",
    "local_vllm": "vllm",
    "local_ollama": "ollama",
    "local_mlx_lm": "mlx_lm",
}

CLOUD_PROVIDER_CONFIG_KEYS: tuple[str, ...] = tuple(
    record.config_key for record in ALL_RECORDS if record.classification == _CLOUD
)
LOCAL_PROVIDER_CONFIG_KEYS: tuple[str, ...] = tuple(
    record.config_key for record in ALL_RECORDS if record.classification == _LOCAL
)
ENGINE_RECORDS: tuple[ProviderRecord, ...] = tuple(
    record for record in ALL_RECORDS if record.engine_driven
)
AUTO_REFRESH_KEYS: frozenset[str] = frozenset(
    record.key for record in ALL_RECORDS if record.auto_refresh
)
NATIVE_TOOLS_KEYS: frozenset[str] = frozenset(
    record.key for record in ALL_RECORDS if record.native_tools
)
#: Every dispatch key (canonical + aliases) — the sensitive-audit universe.
AUDITED_ENDPOINT_KEYS: frozenset[str] = frozenset(RECORDS_BY_KEY) | frozenset(ALIASES)
```

Fill in every record named in `ALL_RECORDS` (no ellipses in the real file);
transcription against `config.py` and
`Chat/console_provider_support.py::_PROVIDER_DISPLAY_NAMES` is part of this
task. `auto_refresh` flags must match `LLM_Provider_Catalog/
model_catalog_settings.py::AUTO_REFRESH_PROVIDER_LIST_KEYS` (line ~13);
`native_tools` flags must match `Agents/native_tools.py::
NATIVE_TOOLS_PROVIDERS` (line ~31) — add two more parity tests asserting
those equalities (same style as `test_cloud_classification_matches_config`)
before moving on. Databricks is NOT yet added to those two literal sets —
that happens in Tasks 11–12; until then the parity tests must compare
*excluding* `databricks` (use `AUTO_REFRESH_KEYS - {"databricks"}`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/test_provider_registry.py -v`
Expected: PASS (transcribe until green; wrong env vars/URLs surface here).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/provider_registry.py Tests/test_provider_registry.py
git commit -m "feat: add provider registry leaf module with parity tests (ADR-179)"
```

---

### Task 3: `hosted_chat` tolerated-extra-keys validation

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py` (`normalize_hosted_chat_response` ~L386, `HostedChatStream.__init__` ~L120, `hosted_chat_request` ~L459)
- Test: `Tests/LLM_Calls/test_hosted_chat_allowances.py`

**Interfaces:**
- Produces: `normalize_hosted_chat_response(response, *, finish_policy, allowed_extra_keys: frozenset[str] = frozenset())`,
  `HostedChatStream(records, *, finish_policy, allowed_extra_keys: frozenset[str] = frozenset())`,
  `hosted_chat_request(..., allowed_extra_keys: frozenset[str] = frozenset())`.
- Existing callers (zai, moonshot) pass no new argument → behavior unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Calls/test_hosted_chat_allowances.py
import pytest

from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    normalize_hosted_chat_response,
)


class _Policy:
    reasoning_disposition = "ignored"

    def validate_finish(self, *, finish_reason, has_text, has_calls):
        assert finish_reason == "stop"
        return finish_reason

    def validate_reasoning_content(self, value):
        return None


def _ok_response(**extra):
    body = {
        "id": "r1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    body.update(extra)
    return body


def test_unknown_top_level_key_fails_closed_without_allowance():
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(_ok_response(service_tier="default"), finish_policy=_Policy())


def test_allowlisted_extra_key_is_ignored():
    turn = normalize_hosted_chat_response(
        _ok_response(service_tier="default"),
        finish_policy=_Policy(),
        allowed_extra_keys=frozenset({"service_tier"}),
    )
    assert turn.text == "hi"


def test_allowance_never_relaxes_required_shapes():
    bad = _ok_response()
    bad["choices"] = []  # empty choices is a required-shape violation
    with pytest.raises(HostedChatProtocolError):
        normalize_hosted_chat_response(
            bad, finish_policy=_Policy(), allowed_extra_keys=frozenset({"service_tier"})
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/LLM_Calls/test_hosted_chat_allowances.py -v`
Expected: FAIL — `normalize_hosted_chat_response() got an unexpected keyword argument 'allowed_extra_keys'`.

- [ ] **Step 3: Implement**

In `normalize_hosted_chat_response`: add the keyword parameter; change the
top-level rejection at ~L394 from

```python
if set(response) - { "id", "object", "created", "model", "system_fingerprint", "choices", "usage" }:
```

to subtract `allowed_extra_keys` from that set difference before rejecting
(allowlisted keys are simply dropped — they already passed
`_json_shape_is_safe`). In `HostedChatStream.__init__`: accept and store
`allowed_extra_keys`; in `_consume_event` apply the same subtraction to the
event-level check (~L210). In `hosted_chat_request`: accept and forward the
parameter to both paths. Do not touch any other validation branch.

- [ ] **Step 4: Run tests to verify they pass, plus strict-tier regression**

```bash
pytest Tests/LLM_Calls/test_hosted_chat_allowances.py Tests/LLM_Calls/test_zai.py Tests/LLM_Calls/test_moonshot.py -v
```
Expected: PASS (zai/moonshot unchanged — they don't pass the new kwarg).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/hosted_chat.py Tests/LLM_Calls/test_hosted_chat_allowances.py
git commit -m "feat: hosted_chat tolerated-extra-keys response allowances (ADR-179)"
```

---

### Task 4: Engine — request resolution

**Files:**
- Create: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py`
- Test: `Tests/LLM_Calls/test_hosted_provider_engine_resolution.py`

**Interfaces:**
- Consumes: `ProviderRecord` from Task 2; `config.provider_settings_for_key`,
  `config.resolve_provider_api_key`, `config.get_runtime_config_snapshot`;
  `hosted_chat.normalize_hosted_chat_base_url`.
- Produces:

```python
@dataclass(frozen=True)
class HostedProviderResolution:
    provider: str
    model: str
    api_key: str          # repr=False, compare=False
    base_url: str
    timeout: float
    retries: int
    retry_delay: float
    streaming: bool

def resolve_hosted_request(
    record: ProviderRecord, *,
    explicit_api_key=None, explicit_base_url=None, explicit_model=None,
    explicit_timeout=None, explicit_retries=None, explicit_retry_delay=None,
    app_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> HostedProviderResolution
```

- [ ] **Step 1: Write the failing tests**

Port the resolution tests from `Tests/LLM_Calls/test_zai.py` (find its
`resolve_zai_request` tests; reuse their config-fixture style), replacing
provider specifics with the Databricks record:

```python
# Tests/LLM_Calls/test_hosted_provider_engine_resolution.py
import pytest

from tldw_chatbook.LLM_Calls.hosted_provider_engine import resolve_hosted_request
from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.provider_registry import DATABRICKS


def _config(api_settings):
    return {"api_settings": api_settings}


def test_explicit_args_win_over_settings_and_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-key")
    resolution = resolve_hosted_request(
        DATABRICKS,
        explicit_api_key="explicit-key",
        explicit_base_url="https://dbc-1.cloud.databricks.com",
        explicit_model="gpt-4o",
        app_config=_config({"databricks": {"api_key": "stored-key", "model": "claude-4-sonnet"}}),
        environ={"DATABRICKS_TOKEN": "env-key"},
    )
    assert resolution.api_key == "explicit-key"
    assert resolution.model == "gpt-4o"
    assert resolution.base_url == "https://dbc-1.cloud.databricks.com/openai/v1"


def test_bare_workspace_host_gets_suffix_and_full_url_is_kept():
    bare = resolve_hosted_request(
        DATABRICKS, explicit_api_key="k", explicit_base_url="https://dbc-1.cloud.databricks.com",
    )
    full = resolve_hosted_request(
        DATABRICKS, explicit_api_key="k",
        explicit_base_url="https://dbc-1.cloud.databricks.com/openai/v1",
    )
    assert bare.base_url == full.base_url == "https://dbc-1.cloud.databricks.com/openai/v1"


def test_terminal_chat_completions_paste_is_rejected():
    with pytest.raises(ChatConfigurationError):
        resolve_hosted_request(
            DATABRICKS, explicit_api_key="k",
            explicit_base_url="https://dbc-1.cloud.databricks.com/openai/v1/chat/completions",
        )


def test_missing_key_and_missing_url_are_actionable():
    with pytest.raises(ChatConfigurationError) as missing_key:
        resolve_hosted_request(DATABRICKS, app_config=_config({"databricks": {}}), environ={})
    assert "Databricks" in str(missing_key.value)
    with pytest.raises(ChatConfigurationError) as missing_url:
        resolve_hosted_request(
            DATABRICKS, explicit_api_key="k", app_config=_config({"databricks": {}}), environ={},
        )
    assert "Databricks" in str(missing_url.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_resolution.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name`.

- [ ] **Step 3: Implement**

Create `hosted_provider_engine.py` with the `HostedProviderResolution`
dataclass and `resolve_hosted_request`, ported from
`LLM_Calls/zai.py::resolve_zai_request` (L189–236) and its helpers
(`_resolve_api_key` L566, `_resolve_base_url` L591, `_positive_number`,
`_nonnegative_number`, `_nonnegative_integer`, `_resolve_streaming`,
`_resolve_string` L554) parameterized by the record:

- `_resolve_api_key`: env candidates are
  `record.api_key_env_candidates`; settings key is
  `api_settings.<record.key>`; every error message is
  `f"{record.display_name} ..."` (replaces the hardcoded "Z.ai" strings).
- `_resolve_base_url`: when no explicit URL and `record.default_base_url is
  None`, raise `ChatConfigurationError(provider=record.key, message=f"{record.display_name} workspace base URL is required.")`.
  Then: if the URL's path is empty or `/` **and** `record.base_url_suffix`
  is set, append the suffix; otherwise validate as-is. Both branches finish
  with `normalize_hosted_chat_base_url(candidate, default=...)` which
  already rejects terminal `/chat/completions` suffixes and other unsafe
  shapes — map its `ValueError` to `ChatConfigurationError` with display-name
  copy, exactly as zai does.
- Numeric/streaming defaults come from `record.settings_defaults`
  (`timeout`, `retries`, `retry_delay`, `streaming`) instead of hardcoded
  literals.
- Provider id in errors: `record.key`; display prefix: `record.display_name`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_resolution.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/hosted_provider_engine.py Tests/LLM_Calls/test_hosted_provider_engine_resolution.py
git commit -m "feat: hosted provider engine request resolution (ADR-179)"
```

---

### Task 5: Engine — payload builder

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py`
- Test: `Tests/LLM_Calls/test_hosted_provider_engine_payload.py`

**Interfaces:**
- Consumes: `HostedProviderResolution` (Task 4),
  `Chat.provider_continuation.ProviderContinuationCheckpoint`.
- Produces:

```python
def build_hosted_chat_payload(
    record: ProviderRecord, *, resolution: HostedProviderResolution,
    messages_payload: Sequence[Mapping[str, Any]], system_message=None,
    streaming=None, tools=None, tool_choice=None, reasoning_effort=None,
    provider_continuations=(), temperature=None, top_p=None, max_tokens=None,
    stop=None, response_format=None, seed=None, n=None, user=None, **_generic,
) -> dict[str, Any]
```

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Calls/test_hosted_provider_engine_payload.py
import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution, build_hosted_chat_payload,
)
from tldw_chatbook.provider_registry import DATABRICKS


def _resolution(streaming=True):
    return HostedProviderResolution(
        provider="databricks", model="gpt-4o", api_key="k",
        base_url="https://dbc-1.cloud.databricks.com/openai/v1",
        timeout=90.0, retries=3, retry_delay=5.0, streaming=streaming,
    )


def test_databricks_payload_snapshot():
    payload = build_hosted_chat_payload(
        DATABRICKS, resolution=_resolution(),
        messages_payload=[{"role": "user", "content": "hi"}],
        system_message="be brief", temperature=0.2, max_tokens=512,
    )
    assert payload == {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 512,
    }


def test_tools_and_choice_pass_through_openai_shape():
    tools = [{"type": "function", "function": {
        "name": "get_time", "description": "time", "parameters": {"type": "object"}}}]
    payload = build_hosted_chat_payload(
        DATABRICKS, resolution=_resolution(streaming=False),
        messages_payload=[{"role": "user", "content": "hi"}],
        tools=tools, tool_choice="auto",
    )
    assert payload["tools"] == tools and payload["tool_choice"] == "auto"
    assert payload["stream"] is False


def test_message_and_tool_validation_is_strict():
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            DATABRICKS, resolution=_resolution(),
            messages_payload=[{"role": "wizard", "content": "hi"}],
        )
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            DATABRICKS, resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "1bad", "description": "d",
                                                      "parameters": {"type": "object"}}}],
        )


def test_flag_off_fields_are_dropped():
    # reasoning_effort is flag-off for Databricks Phase 1
    with pytest.raises(ChatBadRequestError):
        build_hosted_chat_payload(
            DATABRICKS, resolution=_resolution(),
            messages_payload=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_payload.py -v`
Expected: FAIL — `cannot import name 'build_hosted_chat_payload'`.

- [ ] **Step 3: Implement**

Port from `zai.py`: `_normalize_messages` (L643), `_normalize_call_batch`
(L709), `_normalize_tools` (L748), `_normalize_tool_choice` (L781),
`_apply_continuations` (L792), `_find_owner` (L830), `_validate_sampler`
(L851), `_positive_integer` (L863), `_normalize_stop` (L869),
`_normalize_response_format` (L882), `_bounded_identifier` (L899),
`_json_shape_is_bounded` (L910) — copy them into the engine with every
"Z.ai" string replaced by `record.display_name` (keep the strict bodies
verbatim otherwise). Then `build_hosted_chat_payload`, modeled on
`build_zai_chat_payload` (L239) with these deltas:

- Base payload: `{"model", "messages", "stream"}` plus, per
  `record.payload_flags`: `temperature` → `temperature`, `top_p` → `top_p`,
  `max_tokens` → `max_tokens`, `stop`, `seed`, `n`, `user`,
  `response_format`; each still validated by the ported helpers.
- `reasoning_effort`: only when `record.reasoning_effort` is on; when off
  and a value is supplied, raise `ChatBadRequestError(f"{record.display_name} reasoning effort is unsupported.")`.
- `record.extra_body_fields`: merged last (validated bounded JSON scalars).
- No provider-invented fields (zai's `thinking`, `request_id`, `user_id`
  quirks do not exist here — that is the point of the engine).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_payload.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/hosted_provider_engine.py Tests/LLM_Calls/test_hosted_provider_engine_payload.py
git commit -m "feat: hosted provider engine strict payload builder (ADR-179)"
```

---

### Task 6: Engine — finish policy, response/stream wrappers

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py`
- Test: `Tests/LLM_Calls/test_hosted_provider_engine_policy.py`

**Interfaces:**
- Produces:

```python
class HostedPresetFinishPolicy:  # implements HostedChatFinishPolicy
    def __init__(self, record: ProviderRecord) -> None: ...
    reasoning_disposition: ReasoningDisposition  # from record
    def validate_finish(self, *, finish_reason, has_text, has_calls) -> str: ...
    def validate_reasoning_content(self, value) -> str | None: ...

class HostedProviderStream(Iterator[dict[str, Any]]):  # mirrors ZAIStream
    def __init__(self, stream: HostedChatStream, *, record, resolution=None,
                 provider_continuations=()) -> None: ...
    terminal_turn: HostedChatTurn
    provider_continuation: ProviderContinuationCheckpoint | None
    def close(self) -> None: ...

class HostedProviderResponse(dict[str, Any]):  # mirrors ZAIResponse
    terminal_turn: HostedChatTurn
    provider_continuation: ProviderContinuationCheckpoint | None

def normalize_hosted_provider_response(record, response) -> HostedChatTurn
```

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Calls/test_hosted_provider_engine_policy.py
import pytest

from tldw_chatbook.LLM_Calls.hosted_provider_engine import HostedPresetFinishPolicy
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.provider_registry import DATABRICKS


def test_allowed_terminals_pass_and_inconsistent_state_fails():
    policy = HostedPresetFinishPolicy(DATABRICKS)
    assert policy.validate_finish(finish_reason="stop", has_text=True, has_calls=False) == "stop"
    assert policy.validate_finish(finish_reason="tool_calls", has_text=False, has_calls=True) == "tool_calls"
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(finish_reason="tool_calls", has_text=True, has_calls=False)
    with pytest.raises(HostedChatProtocolError):
        policy.validate_finish(finish_reason="content_filter", has_text=True, has_calls=False)


def test_reasoning_disposition_ignored_drops_reasoning():
    policy = HostedPresetFinishPolicy(DATABRICKS)
    assert DATABRICKS.reasoning_disposition == "ignored"
    assert policy.validate_reasoning_content("thinking...") is None
```

Also port the stream-wrapper tests from `Tests/LLM_Calls/test_zai.py` that
cover `ZAIStream` visible-chunk behavior (reasoning stripped, empty content
frames kept non-diagnostic) and `ZAIResponse` terminal metadata — same
assertions against `HostedProviderStream`/`HostedProviderResponse` built
from the Databricks record.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_policy.py -v`
Expected: FAIL — import errors.

- [ ] **Step 3: Implement**

- `HostedPresetFinishPolicy`: port `ZAIFinishPolicy` (zai.py L73) with the
  literal sets replaced by `record.finish_terminal` /
  `record.finish_provider_errors` (provider-error reasons raise
  `ChatProviderError(provider=record.key, status_code=502)`),
  `reasoning_disposition = record.reasoning_disposition`, and
  `validate_reasoning_content` implementing the disposition:
  `"ignored"` → return `None`; `"displayable"`/`"proprietary"` → validate
  string-or-None and return it (mirrors zai).
- `normalize_hosted_provider_response`: port `normalize_zai_response`
  (zai.py L317) — deep-copy, no Databricks-specific coercion (keep the
  dict-arguments→string tool coercion: some gateways return dict
  arguments), then `normalize_hosted_chat_response(..., finish_policy=
  HostedPresetFinishPolicy(record), allowed_extra_keys=
  record.response_allowances)`; map `HostedChatProtocolError` to
  `ChatProviderError(provider=record.key, status_code=502)` with
  display-name message.
- `HostedProviderStream`/`HostedProviderResponse`: port `ZAIStream`
  (zai.py L112) / `ZAIResponse` (L163) replacing "Z.ai" with
  `record.display_name`; strip `reasoning_content` only when disposition is
  not `"displayable"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_policy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/hosted_provider_engine.py Tests/LLM_Calls/test_hosted_provider_engine_policy.py
git commit -m "feat: hosted provider engine finish policy and response wrappers (ADR-179)"
```

---

### Task 7: Engine — continuations and the handler factory

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py`
- Test: `Tests/LLM_Calls/test_hosted_provider_engine_handler.py`

**Interfaces:**
- Consumes: Tasks 4–6 outputs; `Chat.provider_continuation`
  (`ProviderContinuationCheckpoint`, `ContinuationRound`, `ContinuationCall`,
  `ContinuationRestoreTarget`, `dump_provider_continuation_json`,
  `parse_provider_continuation_json`, `validate_continuation_restore`).
- Produces:

```python
def build_hosted_chat_handler(record: ProviderRecord) -> Callable[..., dict[str, Any] | HostedProviderStream]
```

  The returned callable has the `chat_with_zai` signature (zai.py L368–391):
  `(input_data, model=None, api_key=None, system_message=None, temp=None,
  maxp=None, streaming=False, max_tokens=None, tools=None,
  custom_prompt_arg=None, api_base_url=None, tool_choice=None, stop=None,
  response_format=None, user=None, reasoning_effort=None,
  provider_continuations=(), request_timeout=None, request_retries=None,
  request_retry_delay=None)`.
  Also `resolve_hosted_engine_request(record, ...)` — thin public alias of
  `resolve_hosted_request` for the catalog service (Task 12).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Calls/test_hosted_provider_engine_handler.py
from tldw_chatbook.LLM_Calls.hosted_provider_engine import build_hosted_chat_handler
from tldw_chatbook.provider_registry import DATABRICKS


def test_factory_signature_accepts_chat_api_call_kwargs(monkeypatch):
    handler = build_hosted_chat_handler(DATABRICKS)
    import inspect
    params = set(inspect.signature(handler).parameters)
    expected = {
        "input_data", "model", "api_key", "system_message", "temp", "maxp",
        "streaming", "max_tokens", "tools", "custom_prompt_arg",
        "api_base_url", "tool_choice", "stop", "response_format", "user",
        "reasoning_effort", "provider_continuations", "request_timeout",
        "request_retries", "request_retry_delay",
    }
    assert expected <= params


def test_continuation_roundtrip_builds_checkpoint(monkeypatch):
    # Monkeypatch owned_json_post (engine module symbol) to return a canned
    # tool-call response; assert handler returns HostedProviderResponse whose
    # provider_continuation.checkpoint fields are (provider="databricks",
    # protocol="chat_completions", state="active", rounds[0].calls[0].name == tool name).
    ...
```

(Write `test_continuation_roundtrip_builds_checkpoint` fully: port the
equivalent zai continuation test from `Tests/LLM_Calls/test_zai.py` /
`Tests/Chat/test_kimi_zai_native_tools.py`, monkeypatching the engine's
`owned_json_post` symbol with a canned non-streaming response containing
one tool call.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_handler.py -v`
Expected: FAIL — `cannot import name 'build_hosted_chat_handler'`.

- [ ] **Step 3: Implement**

- Continuation candidate: port `_zai_continuation_candidate` (zai.py L490)
  parameterized by `record.key`/`record.continuation_protocol`; skip
  entirely when `record.continuation_protocol is None`.
- `build_hosted_chat_handler(record)`: closure assembling resolve →
  payload → `owned_json_post(config=HostedHTTPTransportConfig(provider=
  record.key, ...), route="chat/completions", payload, streaming)` →
  stream/`normalize_hosted_provider_response` → response object, mirroring
  `chat_with_zai` (zai.py L368–455) minus zai-specific quirks. When
  constructing `HostedChatStream` for the streaming path, pass
  `allowed_extra_keys=record.response_allowances` (Task 3 parameter) — the
  non-streaming path already gets allowances via
  `normalize_hosted_provider_response`. Wrap the
  body with the same metrics counters used by the
  `chat_with_zai`/`chat_with_moonshot` wrappers in
  `LLM_API_Calls.py` (read L5466–5618; replicate the exact
  `log_counter`/`log_histogram` call shapes, substituting
  `provider=record.key`) — but WITHOUT the LLM_API_Calls wrapper module:
  the counters live inside the factory closure, once, for all engine
  providers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/LLM_Calls/test_hosted_provider_engine_handler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/hosted_provider_engine.py Tests/LLM_Calls/test_hosted_provider_engine_handler.py
git commit -m "feat: hosted provider engine continuations and handler factory (ADR-179)"
```

---

### Task 8: Dispatch registration for Databricks

**Files:**
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (imports L54–80,
  `API_CALL_HANDLERS` L120–151, `SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS`
  L190–222, `PROVIDER_PARAM_MAP` L232+)
- Test: `Tests/Chat/test_dispatch_registry_parity.py`

**Interfaces:**
- Consumes: `build_hosted_chat_handler` (Task 7), `provider_registry.AUDITED_ENDPOINT_KEYS` (Task 2).
- Produces: `API_CALL_HANDLERS["databricks"]`,
  `ENGINE_PROVIDER_PARAM_MAP` (module constant in Chat_Functions.py).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_dispatch_registry_parity.py
from tldw_chatbook.Chat.Chat_Functions import (
    API_CALL_HANDLERS,
    SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS,
)
from tldw_chatbook.provider_registry import AUDITED_ENDPOINT_KEYS


def test_databricks_registered_and_audited():
    assert "databricks" in API_CALL_HANDLERS
    assert callable(API_CALL_HANDLERS["databricks"])
    assert "databricks" in SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS


def test_audited_set_is_registry_derived_and_covers_all_handlers():
    assert SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS == AUDITED_ENDPOINT_KEYS
    assert set(API_CALL_HANDLERS) <= SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_dispatch_registry_parity.py -v`
Expected: FAIL — `databricks` not registered.

- [ ] **Step 3: Implement**

In `Chat_Functions.py`:

1. Import: `from tldw_chatbook.LLM_Calls.hosted_provider_engine import build_hosted_chat_handler` and
   `from tldw_chatbook.provider_registry import AUDITED_ENDPOINT_KEYS, DATABRICKS`.
2. Add to `API_CALL_HANDLERS`: `"databricks": build_hosted_chat_handler(DATABRICKS),`
3. Replace the literal `SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS` frozenset
   body with `frozenset(AUDITED_ENDPOINT_KEYS)` — keep the explanatory
   comment block above it (L186–189), amending it: *"kept explicit via the
   provider registry (ADR-179); the parity test forces every newly
   registered chat handler through the sensitive auxiliary-request audit."*
4. Add `ENGINE_PROVIDER_PARAM_MAP`: copy the `"zai"` entry of
   `PROVIDER_PARAM_MAP` (L822) verbatim into a new module-level constant,
   then reference it from both `"zai"`'s entry (unchanged value) and a new
   `"databricks"` entry (`PROVIDER_PARAM_MAP["databricks"] =
   ENGINE_PROVIDER_PARAM_MAP` — same object). Verify every mapped target
   name exists in the factory handler signature from Task 7.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/Chat/test_dispatch_registry_parity.py Tests/Chat/test_sensitive_llm_logging.py Tests/Chat/test_chat_unit_mocked_APIs.py -v
```
Expected: PASS (existing suites confirm no dispatch regression).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/Chat_Functions.py Tests/Chat/test_dispatch_registry_parity.py
git commit -m "feat: register databricks via engine; derive audited endpoint set from registry"
```

---

### Task 9: config.py — Databricks tables and cloud classification

**Files:**
- Modify: `tldw_chatbook/config.py` (`[providers]`-backing defaults — search for `API_MODELS_BY_PROVIDER` ~L3468 and the `[api_settings]` default tables, `[api_settings.databricks]` added among them; `_cloud_provider_keys` **list** ~L9843)
- Test: `Tests/test_config_databricks.py`

**Interfaces:**
- Consumes: `provider_registry.CLOUD_PROVIDER_CONFIG_KEYS`, `DATABRICKS.settings_defaults`.
- Produces: default config containing `[providers].Databricks = []` (empty — spec: workspace-dependent availability) and `[api_settings.databricks]` with the record's `settings_defaults`; `_cloud_provider_keys()` returns registry-derived tuple including `"Databricks"`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/test_config_databricks.py
from tldw_chatbook.config import load_default_config_values  # or the module's default-table accessor used by existing config tests —
# read Tests/test_config_model_catalog_defaults.py first and use the SAME accessor.


def test_providers_table_has_empty_databricks_entry():
    providers = <default config>["providers"]
    assert providers.get("Databricks") == []


def test_api_settings_databricks_defaults():
    settings = <default config>["api_settings"]["databricks"]
    assert settings["api_key_env_var"] == "DATABRICKS_TOKEN"
    assert settings["streaming"] is True
    assert "api_base_url" not in settings or settings["api_base_url"] in (None, "")


def test_databricks_classified_cloud():
    from tldw_chatbook.config import _cloud_provider_keys
    assert "Databricks" in _cloud_provider_keys
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/test_config_databricks.py -v`
Expected: FAIL — no Databricks entries.

- [ ] **Step 3: Implement**

1. In the default `[providers]` dict: add `"Databricks": [],` next to the
   other cloud providers (comment: model availability is
   workspace-dependent; fills via discovery or manual seeding).
2. Add the `[api_settings.databricks]` default table using
   `DATABRICKS.settings_defaults` values (no `api_base_url` key).
3. `_cloud_provider_keys`: replace the hand-typed **list** (config.py
   ~L9843) with `CLOUD_PROVIDER_CONFIG_KEYS` from the registry (the Task 2
   parity test already proves equality with the old list plus `Databricks`;
   update that test to include `databricks` now — remove the exclusion).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/test_config_databricks.py Tests/test_provider_registry.py Tests/test_config_model_catalog_defaults.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py Tests/test_config_databricks.py
git commit -m "feat: databricks config tables; cloud classification from provider registry"
```

---

### Task 10: Readiness, endpoints, display names, setup persistence

**Files:**
- Modify: `tldw_chatbook/Chat/provider_readiness.py` (`PROVIDERS_REQUIRING_API_KEY_KEYS` ~L55–71)
- Modify: `tldw_chatbook/Chat/console_provider_endpoints.py` (`_BUILTIN_PROVIDER_ENDPOINTS` ~L43–58)
- Modify: `tldw_chatbook/Chat/console_provider_support.py` (`_PROVIDER_DISPLAY_NAMES` ~L79–100)
- Modify: `tldw_chatbook/Chat/provider_setup_persistence.py` (canonical key set + display aliases ~L126–182)
- Test: extend `Tests/Chat/test_provider_readiness.py`; create `Tests/Chat/test_databricks_console_surfaces.py`

**Interfaces:**
- Produces: Databricks present in readiness key requirements (key **and** base URL required), endpoint map documentation entry, display name "Databricks", setup-persistence canonical key.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_databricks_console_surfaces.py
def test_readiness_requires_key_and_base_url():
    from tldw_chatbook.Chat.provider_readiness import provider_readiness
    # follow the existing test pattern in Tests/Chat/test_provider_readiness.py:
    # with no DATABRICKS_TOKEN env and no api_settings.databricks.api_key,
    # readiness for "databricks" is blocked with copy naming the API key;
    # with a key but no api_base_url, blocked with copy naming the workspace URL.


def test_display_name_and_endpoint_doc_entries():
    from tldw_chatbook.Chat.console_provider_support import _PROVIDER_DISPLAY_NAMES
    from tldw_chatbook.Chat.console_provider_endpoints import _BUILTIN_PROVIDER_ENDPOINTS
    assert _PROVIDER_DISPLAY_NAMES.get("databricks") == "Databricks"
    # no shipped default endpoint URL (per-account); when the map requires a
    # value use the documented suffix rule:
    assert _BUILTIN_PROVIDER_ENDPOINTS.get("databricks", "<workspace-host>/openai/v1").endswith("/openai/v1")


def test_setup_persistence_accepts_databricks():
    from tldw_chatbook.Chat.provider_setup_persistence import <canonical key set accessor>  # read the module's set name at L126
    assert "databricks" in <canonical key set accessor>()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_databricks_console_surfaces.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add `"databricks"` to each surface, following each file's existing entry
style. Readiness detail (spec data flow): Databricks requires key **and**
`api_base_url` — if the readiness module supports per-provider
required-fields beyond the API key (read its credential-resolution branch
~L409–421 for the strict-hosted pattern moonshot/zai use), add Databricks
to the strict-hosted branch; otherwise extend the module with a minimal
required-base-url set `PROVIDERS_REQUIRING_BASE_URL_KEYS = frozenset({"databricks"})`
and surface actionable copy in its blocked-send message.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/Chat/test_databricks_console_surfaces.py Tests/Chat/test_provider_readiness.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/provider_readiness.py tldw_chatbook/Chat/console_provider_endpoints.py tldw_chatbook/Chat/console_provider_support.py tldw_chatbook/Chat/provider_setup_persistence.py Tests/Chat/test_databricks_console_surfaces.py
git commit -m "feat: databricks readiness, endpoints, display name, setup persistence"
```

---

### Task 11: Continuation registration and generic gateway finish-policy entry

**Files:**
- Modify: `tldw_chatbook/Chat/provider_continuation.py` (`ContinuationProvider` Literal L26, `_PAIRINGS` L66–73)
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (`_HOSTED_THINKING_FINISH_POLICIES` L236–239)
- Test: `Tests/Chat/test_databricks_continuation_registration.py`

**Interfaces:**
- Produces: `ContinuationProvider` gains `"databricks"`; `_PAIRINGS` includes `("databricks", "chat_completions")`; gateway policy map resolves `databricks` → the engine's preset policy (generic entry, not a class per provider).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_databricks_continuation_registration.py
def test_continuation_provider_and_pairings_cover_databricks():
    from tldw_chatbook.Chat.provider_continuation import _PAIRINGS, ContinuationProvider
    import typing
    assert "databricks" in typing.get_args(ContinuationProvider)
    assert ("databricks", "chat_completions") in _PAIRINGS


def test_gateway_finish_policy_resolves_engine_presets():
    from tldw_chatbook.Chat.console_provider_gateway import resolve_finish_policy
    assert resolve_finish_policy("databricks") is not None  # engine preset policy
    assert resolve_finish_policy("zai") is not None          # existing behavior
    assert resolve_finish_policy("openai") is None           # non-hosted untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_databricks_continuation_registration.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

1. `provider_continuation.py`: extend the Literal with `"databricks"`; add
   the pairing. (`_PAIRINGS` derivation-from-registry is desirable but the
   Literal is static — a one-line mechanical edit per provider is the
   accepted residual (spec, residual touchpoints).)
2. `console_provider_gateway.py`: introduce a module-level
   `resolve_finish_policy(key)` wrapping the existing lookup site (~L433
   `policy = _HOSTED_THINKING_FINISH_POLICIES.get(key)`): return the map
   hit, else fall back for engine-driven providers — look up the record in
   `provider_registry.RECORDS_BY_KEY` and return
   `HostedPresetFinishPolicy(record)` when `record.engine_driven`, else
   `None`. Cache per key in a module-level dict (policies are stateless).
   The existing call sites use `resolve_finish_policy` instead of the bare
   `.get`. Import cycle note: gateway already imports from `LLM_Calls`
   (L169–170), so importing the engine follows the existing pattern.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/Chat/test_databricks_continuation_registration.py Tests/Chat/test_kimi_zai_native_tools.py -v
```
Expected: PASS (zai/moonshot continuation behavior untouched).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/provider_continuation.py tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_databricks_continuation_registration.py
git commit -m "feat: databricks continuation registration; generic engine finish-policy gateway entry"
```

---

### Task 12: Native tools and model catalog auto-refresh

**Files:**
- Modify: `tldw_chatbook/Agents/native_tools.py` (`NATIVE_TOOLS_PROVIDERS` L31–64)
- Modify: `tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py` (`AUTO_REFRESH_PROVIDER_LIST_KEYS` L13–21)
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py` (`_STRICT_HOSTED_PROVIDER_KEYS` ~L74, `_resolve_hosted_provider` ~L256–276)
- Test: `Tests/LLM_Provider_Catalog/test_databricks_catalog_refresh.py`

**Interfaces:**
- Consumes: `resolve_hosted_engine_request` (Task 7).
- Produces: `"databricks"` in `NATIVE_TOOLS_PROVIDERS` and `AUTO_REFRESH_PROVIDER_LIST_KEYS`; catalog service resolves Databricks endpoint+key through the engine resolver, then discovers models at `{base_url}/{record.discovery_route}`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Provider_Catalog/test_databricks_catalog_refresh.py
def test_auto_refresh_list_includes_databricks():
    from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import AUTO_REFRESH_PROVIDER_LIST_KEYS
    assert "databricks" in AUTO_REFRESH_PROVIDER_LIST_KEYS


def test_native_tools_includes_databricks():
    from tldw_chatbook.Agents.native_tools import NATIVE_TOOLS_PROVIDERS
    assert "databricks" in NATIVE_TOOLS_PROVIDERS


def test_catalog_service_builds_models_url_from_workspace_host():
    # Follow the existing strict-hosted test pattern in
    # Tests/LLM_Provider_Catalog/ for moonshot/zai: with api_settings.databricks
    # api_base_url=https://dbc-1.cloud.databricks.com and a key, the service's
    # endpoint resolution yields https://dbc-1.cloud.databricks.com/openai/v1/models
    ...
```

(Write `test_catalog_service_builds_models_url_from_workspace_host` fully
by porting the moonshot/zai equivalent from the existing
`Tests/LLM_Provider_Catalog/` suite — same fixtures, provider swapped.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/LLM_Provider_Catalog/test_databricks_catalog_refresh.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

1. Add `"databricks"` to both literal sets (update the Task 2 parity-test
   exclusions — they can now compare without removing `databricks`).
2. In `local_llm_provider_catalog_service.py::_resolve_hosted_provider`,
   add a branch for engine-driven providers: when the resolved record is
   engine-driven (`provider_registry.RECORDS_BY_KEY[key].engine_driven`),
   call `resolve_hosted_engine_request(record, app_config=...)` and reuse
   its `base_url`/`api_key` exactly as the moonshot/zai branches reuse
   their resolvers. The discovery URL is
   `{base_url}/{record.discovery_route}` via the existing
   `build_models_url` + `discover_openai_compatible_models` path.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/LLM_Provider_Catalog/ -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/native_tools.py tldw_chatbook/LLM_Provider_Catalog/ Tests/LLM_Provider_Catalog/test_databricks_catalog_refresh.py
git commit -m "feat: databricks native tools + model catalog auto-refresh via engine resolver"
```

---

### Task 13: Docs — User Guide, README, settings copy

**Files:**
- Modify: `Docs/User_Guide/console.md`, `Docs/User_Guide/settings.md`, `README.md`
- Modify: the Phase-1 backlog task (notes)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Document Databricks setup**

In `Docs/User_Guide/settings.md` (provider settings section) and
`Docs/User_Guide/console.md` (provider selection), add a Databricks
subsection following the Moonshot/Z.ai entries' structure (grep those docs
for "Z.ai" and mirror): what the provider is (AI Gateway external models),
`DATABRICKS_TOKEN` env var, `api_base_url` = workspace host or full
`…/openai/v1` URL, model seeding via discovery or manual `[providers]`
entry. In `README.md`, add Databricks to the provider list (and do NOT add
xAI — excluded per ADR-179).

- [ ] **Step 2: Update the backlog task notes**

```bash
backlog task edit <id> --notes "Phase 1 implementation complete pending live gate (Task 14)"
```

- [ ] **Step 3: Commit**

```bash
git add Docs/User_Guide/console.md Docs/User_Guide/settings.md README.md
git commit -m "docs: databricks provider setup guides (ADR-179 phase 1)"
```

---

### Task 14: Live verification gate (Databricks workspace)

**Files:**
- Create: `Tests/LLM_Calls/test_live_databricks_api.py`
- Modify: `tldw_chatbook/provider_registry.py` (only if `response_allowances` must be recorded)

**Interfaces:**
- Consumes: a real Databricks workspace (env: `DATABRICKS_TOKEN`,
  `DATABRICKS_HOST` = e.g. `https://dbc-XXXX.cloud.databricks.com`, and
  `DATABRICKS_TEST_MODEL` = a gateway external-model id).

- [ ] **Step 1: Write the live tests**

Port the env-gated live pattern (module-level `pytest.mark.skipif` on
missing env vars — no `test_live_*` file exists on dev; the env-gate
convention below is self-contained):
`pytest.mark.skipif(not os.environ.get("DATABRICKS_TOKEN") or
not os.environ.get("DATABRICKS_HOST"), reason="live credentials not set")`.
Four tests:

1. `test_models_listing_settles_o1` — GET
   `{host}/openai/v1/models` (engine discovery route). If it 404s, GET
   `{host}/api/2.0/serving-endpoints` and record which shape works. Either
   way, assert at least one model id is discovered, print the route used.
   Write the finding into the backlog task notes (O-1 settlement).
2. `test_chat_roundtrip` — non-streaming send via
   `build_hosted_chat_handler(DATABRICKS)`, assert text and usage.
3. `test_streamed_roundtrip` — streaming send, consume `HostedProviderStream`
   fully, assert terminal turn usage present.
4. `test_wire_envelope_allowances` — capture the raw top-level response
   keys (print them); if any key outside the `hosted_chat` allowlist
   appears, add it to `DATABRICKS.response_allowances` and re-run Tasks
   3/6 unit tests. No silent relaxations — the printed envelope is the
   evidence.

- [ ] **Step 2: Run with credentials**

Run: `pytest Tests/LLM_Calls/test_live_databricks_api.py -v -s`
Expected: PASS with real workspace. If tool calling through a tool-capable
gateway model fails or is unavailable, flip `DATABRICKS.native_tools` to
`False` (honest flag — spec AC allows it) and note it.

- [ ] **Step 3: If no credentials are available**

Do NOT mark the phase Done. Record in the backlog task:
"live gate pending — needs DATABRICKS_TOKEN/DATABRICKS_HOST". This task is
the Phase 1 Done blocker (spec AC; `backlog/docs/lessons-live-verification.md`).

- [ ] **Step 4: Commit**

```bash
git add Tests/LLM_Calls/test_live_databricks_api.py tldw_chatbook/provider_registry.py
git commit -m "test: databricks live verification gate; record response allowances (O-1)"
```

- [ ] **Step 5: Close out the backlog task**

After all ACs are demonstrably met (live gate included):

```bash
backlog task edit <id> -s Done --notes "Phase 1 complete: registry + engine + databricks live-verified; O-1 settled: <route used>"
```

---

## Self-Review (completed)

**Spec coverage:** registry (Tasks 2, 8–12), engine (4–7), allowances (3),
Databricks end-to-end (8–13), live gate + O-1 (14), ADR (1). Settings-UI
copy: Task 13 covers docs; the settings screen derives providers from the
handler catalog automatically (verified in exploration — `supported_console_provider_catalog`),
so no settings_screen.py edit is required in Phase 1 beyond what display-name
maps provide (Task 10). Pricing: Databricks ships no seeds (workspace-dependent),
matching the spec's data-flow note.

**Type consistency:** `HostedProviderResolution` fields identical in Tasks
4/5/7; `build_hosted_chat_handler(record)` used by Task 8 matches Task 7's
produced signature; `resolve_hosted_engine_request` named in Task 7 and
consumed in Task 12; `AUDITED_ENDPOINT_KEYS` named in Task 2, consumed in
Task 8.

**Known deferred items (Phase 2+, not gaps):** auth_scheme `"none"` /
`"api_key_header"`, long-tail tolerant profile, choice-level response
allowances (Azure), preset-cost test.
