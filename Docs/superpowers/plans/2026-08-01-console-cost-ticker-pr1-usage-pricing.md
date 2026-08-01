# Console Cost Ticker PR1 — Usage Capture + Pricing Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture real per-message token usage (streaming + non-streaming) through the Console provider path, persist it per message, and add a seeded, config-overridable pricing catalog — the foundation PRs 2 (caching) and 3 (ticker UI) stand on.

**Architecture:** Providers emit usage as extra OpenAI-style SSE chunks (empty `choices`, populated `usage`) that older code ignores harmlessly; the gateway records the raw usage payload onto the existing `ConsoleProviderStreamSignals` out-of-band object (chunk contract unchanged); the controller converts it to a `ProviderUsage` (four disjoint buckets) and attaches it to the assistant message; the store persists it as JSON in a new local-only `usage_json` column; a `PricingCatalog` module (copying the `model_capabilities.py` pattern) turns buckets into dollars at read time.

**Tech Stack:** Python ≥3.11, dataclasses, sqlite3 (ChaChaNotes schema v28→v29), pytest + pytest-asyncio, unittest.mock.

**Spec:** `Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md` (approved). This plan implements the **PR1** section only.

## Global Constraints

- Worktree: `/private/tmp/tldw-cost-ticker` (off `origin/dev`). Create branch `feat/console-cost-usage-foundation` from `origin/dev` before Task 1. The `docs/console-cost-ticker-spec` branch is docs-only — do not build on it.
- Setup once: `cd /private/tmp/tldw-cost-ticker && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`. **Run pytest only from this venv** (house rule).
- **Never store dollars** — only tokens + provider + model (+ partial flag). Cost is computed at read time.
- `ProviderUsage` buckets are **disjoint**: `uncached_input`, `cache_read`, `cache_write`, `output`.
- The new DB column is **local-only**: it must NOT appear in any `messages_sync_*` trigger payload (precedent: v19/v24/v25/v26 local-only migrations).
- Usage is **captured if present, never required** — absent usage must never fail a send or a load.
- `git stash` is forbidden (repo-wide stack shared across worktrees).
- Do not hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss` (generated; not needed in PR1 anyway).
- Schema version numbers below assume v28 is current — **re-verify `_CURRENT_SCHEMA_VERSION` at implementation and at merge**; concurrent sessions bump it routinely. If it moved, renumber the migration accordingly everywhere it appears.
- All file:line references below were verified against `origin/dev` @ `2166a6775`; treat them as anchors, not gospel — re-locate by symbol name if drifted.

---

### Task 1: `ProviderUsage` model + payload adapters

**Files:**
- Create: `tldw_chatbook/Chat/provider_usage.py`
- Test: `Tests/Chat/test_provider_usage.py`

**Interfaces:**
- Consumes: nothing (leaf module; stdlib only).
- Produces:
  - `ProviderUsage` frozen dataclass: fields `uncached_input: int`, `cache_read: int`, `cache_write: int`, `output: int`, `provider: str`, `model: str`, `partial: bool` (all with defaults `0`/`""`/`False`).
  - `ProviderUsage.from_provider_payload(payload: Mapping[str, Any] | None, *, provider: str, model: str, partial: bool = False) -> ProviderUsage | None` (classmethod).
  - `ProviderUsage.to_json(self) -> str` and `ProviderUsage.from_json(raw: str | None) -> ProviderUsage | None` (classmethod).
  - `total_tokens` property (`sum of the four buckets`).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_provider_usage.py
"""ProviderUsage: disjoint-bucket normalization of provider usage payloads.

Spec: Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md (PR1).
Buckets are DISJOINT: uncached_input excludes cached tokens on every
provider, so cross-provider cost math is well-defined.
"""

from tldw_chatbook.Chat.provider_usage import ProviderUsage


def test_anthropic_native_payload_maps_directly():
    payload = {
        "input_tokens": 3571,
        "output_tokens": 727,
        "cache_read_input_tokens": 6656,
        "cache_creation_input_tokens": 1024,
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="anthropic", model="claude-sonnet-4-6"
    )
    assert usage == ProviderUsage(
        uncached_input=3571,
        cache_read=6656,
        cache_write=1024,
        output=727,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )


def test_openai_chat_payload_subtracts_cached_from_prompt():
    # OpenAI prompt_tokens INCLUDES cached tokens — naive mapping double-counts.
    payload = {
        "prompt_tokens": 2000,
        "completion_tokens": 150,
        "total_tokens": 2150,
        "prompt_tokens_details": {"cached_tokens": 1536},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-4o"
    )
    assert usage.uncached_input == 464
    assert usage.cache_read == 1536
    assert usage.cache_write == 0
    assert usage.output == 150


def test_openai_chat_payload_without_details_has_zero_cache():
    payload = {"prompt_tokens": 100, "completion_tokens": 20}
    usage = ProviderUsage.from_provider_payload(
        payload, provider="groq", model="llama-3.3-70b-versatile"
    )
    assert usage.uncached_input == 100
    assert usage.cache_read == 0
    assert usage.output == 20


def test_openai_responses_payload_detected_before_anthropic_shape():
    # Responses API uses input_tokens like Anthropic — input_tokens_details
    # disambiguates and must be checked FIRST.
    payload = {
        "input_tokens": 1200,
        "output_tokens": 90,
        "total_tokens": 1290,
        "input_tokens_details": {"cached_tokens": 1024},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-5-mini"
    )
    assert usage.uncached_input == 176
    assert usage.cache_read == 1024
    assert usage.cache_write == 0
    assert usage.output == 90


def test_unrecognized_payload_returns_none():
    assert (
        ProviderUsage.from_provider_payload(
            {"tokens": 5}, provider="x", model="y"
        )
        is None
    )
    assert ProviderUsage.from_provider_payload(None, provider="x", model="y") is None
    assert ProviderUsage.from_provider_payload("nope", provider="x", model="y") is None


def test_negative_and_noninteger_values_clamp_to_zero():
    payload = {"prompt_tokens": "not-a-number", "completion_tokens": -5}
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-4o"
    )
    assert usage.uncached_input == 0
    assert usage.output == 0


def test_cached_larger_than_prompt_clamps_uncached_to_zero():
    payload = {
        "prompt_tokens": 100,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 150},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-4o"
    )
    assert usage.uncached_input == 0
    assert usage.cache_read == 150


def test_json_round_trip_preserves_all_fields():
    original = ProviderUsage(
        uncached_input=1,
        cache_read=2,
        cache_write=3,
        output=4,
        provider="anthropic",
        model="claude-sonnet-4-6",
        partial=True,
    )
    assert ProviderUsage.from_json(original.to_json()) == original


def test_from_json_rejects_garbage():
    assert ProviderUsage.from_json(None) is None
    assert ProviderUsage.from_json("") is None
    assert ProviderUsage.from_json("{not json") is None
    assert ProviderUsage.from_json('"a string"') is None


def test_total_tokens_sums_buckets():
    usage = ProviderUsage(uncached_input=1, cache_read=2, cache_write=3, output=4)
    assert usage.total_tokens == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_provider_usage.py -v`
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Chat.provider_usage`

- [ ] **Step 3: Implement the module**

```python
# tldw_chatbook/Chat/provider_usage.py
"""Normalized per-message token usage for the Console cost ticker.

Buckets are DISJOINT (spec: 2026-08-01-console-cost-ticker-design.md):
Anthropic's input_tokens already excludes cached tokens; OpenAI's
prompt_tokens includes them, so the adapters subtract. Dollars are never
stored — pricing is applied at read time by the pricing catalog.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _as_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    uncached_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    provider: str = ""
    model: str = ""
    partial: bool = False

    @property
    def total_tokens(self) -> int:
        return self.uncached_input + self.cache_read + self.cache_write + self.output

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | None) -> "ProviderUsage | None":
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return cls(
            uncached_input=_as_count(data.get("uncached_input")),
            cache_read=_as_count(data.get("cache_read")),
            cache_write=_as_count(data.get("cache_write")),
            output=_as_count(data.get("output")),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            partial=bool(data.get("partial")),
        )

    @classmethod
    def from_provider_payload(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        provider: str,
        model: str,
        partial: bool = False,
    ) -> "ProviderUsage | None":
        if not isinstance(payload, Mapping):
            return None
        common = {"provider": provider, "model": model, "partial": partial}
        # OpenAI Responses API: has input_tokens like Anthropic — the
        # input_tokens_details key disambiguates, so check it FIRST.
        if isinstance(payload.get("input_tokens_details"), Mapping):
            total_input = _as_count(payload.get("input_tokens"))
            cached = _as_count(payload["input_tokens_details"].get("cached_tokens"))
            return cls(
                uncached_input=max(total_input - cached, 0),
                cache_read=cached,
                output=_as_count(payload.get("output_tokens")),
                **common,
            )
        # OpenAI chat-completions shape: prompt_tokens INCLUDES cached tokens.
        if "prompt_tokens" in payload:
            total_input = _as_count(payload.get("prompt_tokens"))
            details = payload.get("prompt_tokens_details")
            cached = (
                _as_count(details.get("cached_tokens"))
                if isinstance(details, Mapping)
                else 0
            )
            return cls(
                uncached_input=max(total_input - cached, 0),
                cache_read=cached,
                output=_as_count(payload.get("completion_tokens")),
                **common,
            )
        # Anthropic-native shape: buckets are already disjoint.
        if "input_tokens" in payload:
            return cls(
                uncached_input=_as_count(payload.get("input_tokens")),
                cache_read=_as_count(payload.get("cache_read_input_tokens")),
                cache_write=_as_count(payload.get("cache_creation_input_tokens")),
                output=_as_count(payload.get("output_tokens")),
                **common,
            )
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_provider_usage.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/provider_usage.py Tests/Chat/test_provider_usage.py
git commit -m "feat(console): add ProviderUsage disjoint-bucket usage model"
```

---

### Task 2: Pricing catalog module

**Files:**
- Create: `tldw_chatbook/LLM_Calls/pricing_catalog.py`
- Create: `Tests/LLM_Calls/test_pricing_catalog.py` (create the directory; add `__init__.py` only if sibling dirs like `Tests/Chat/` have one — check with `ls Tests/Chat/__init__.py`)

**Interfaces:**
- Consumes: `ProviderUsage` from Task 1; `load_cli_config_and_ensure_existence` from `tldw_chatbook.config` (lazy import inside `__init__`, mirroring `model_capabilities.py:152`).
- Produces:
  - `ModelPricing` frozen dataclass: `input_per_mtok: float`, `output_per_mtok: float`, `cache_read_per_mtok: float | None`, `cache_write_per_mtok: float | None`, `as_of: str` (ISO date).
  - `CostBreakdown` frozen dataclass: `input_cost: float`, `cache_read_cost: float`, `cache_write_cost: float`, `output_cost: float`, `total: float`, `as_of: str`.
  - `PricingCatalog` class: `get_pricing(provider: str, model: str) -> ModelPricing | None`, `cost_for_usage(usage: ProviderUsage) -> CostBreakdown | None`.
  - Module functions: `get_pricing_catalog() -> PricingCatalog` (lazy global singleton), `reload_pricing_catalog() -> None` (resets global to None — mirrors `model_capabilities.py:335-379`).
  - Config override section: top-level `[pricing]` table with `models` (direct `"provider:model"` keys), `patterns` (per-provider regex list), read exactly the way `model_capabilities.py:143-172` reads `[model_capabilities]`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/LLM_Calls/test_pricing_catalog.py
"""PricingCatalog: seeded rates -> config overrides -> pattern fallback.

Rates are dollars per MILLION tokens. Unknown model => None (the UI shows
tokens instead of a fabricated price). Local providers => $0.00 pricing.
"""

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import (
    CostBreakdown,
    ModelPricing,
    PricingCatalog,
)


def _catalog(config=None):
    # Passing config={} skips the config-file read, same convention as
    # ModelCapabilities(config=...) in tldw_chatbook/model_capabilities.py.
    return PricingCatalog(config=config if config is not None else {})


def test_seeded_anthropic_sonnet_rates():
    pricing = _catalog().get_pricing("anthropic", "claude-sonnet-4-6")
    assert pricing is not None
    assert pricing.input_per_mtok == 3.00
    assert pricing.output_per_mtok == 15.00
    assert pricing.cache_read_per_mtok == 0.30
    assert pricing.cache_write_per_mtok == 3.75


def test_pattern_fallback_covers_unlisted_family_member():
    # An unlisted claude-sonnet-* variant should resolve via pattern.
    pricing = _catalog().get_pricing("anthropic", "claude-sonnet-4-5-20250929")
    assert pricing is not None
    assert pricing.input_per_mtok == 3.00


def test_unknown_model_returns_none():
    assert _catalog().get_pricing("anthropic", "totally-unknown-model") is None
    assert _catalog().get_pricing("no-such-provider", "x") is None


def test_local_provider_returns_zero_pricing():
    pricing = _catalog().get_pricing("llama_cpp", "any-gguf-model")
    assert pricing is not None
    assert pricing.input_per_mtok == 0.0
    assert pricing.output_per_mtok == 0.0


def test_config_override_beats_seed():
    config = {
        "models": {
            "anthropic:claude-sonnet-4-6": {
                "input_per_mtok": 1.0,
                "output_per_mtok": 2.0,
                "cache_read_per_mtok": 0.1,
                "cache_write_per_mtok": 1.25,
                "as_of": "2026-09-01",
            }
        }
    }
    pricing = _catalog(config).get_pricing("anthropic", "claude-sonnet-4-6")
    assert pricing.input_per_mtok == 1.0
    assert pricing.as_of == "2026-09-01"


def test_cost_for_usage_multiplies_disjoint_buckets():
    usage = ProviderUsage(
        uncached_input=1_000_000,
        cache_read=1_000_000,
        cache_write=1_000_000,
        output=1_000_000,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    cost = _catalog().cost_for_usage(usage)
    assert isinstance(cost, CostBreakdown)
    assert cost.input_cost == 3.00
    assert cost.cache_read_cost == 0.30
    assert cost.cache_write_cost == 3.75
    assert cost.output_cost == 15.00
    assert cost.total == 22.05


def test_cost_for_usage_unknown_model_returns_none():
    usage = ProviderUsage(uncached_input=10, provider="anthropic", model="unknown")
    assert _catalog().cost_for_usage(usage) is None


def test_cache_buckets_with_null_rates_cost_zero():
    # Providers without a cache-write concept have cache_write_per_mtok=None;
    # tokens landing in that bucket must cost 0, not crash.
    config = {
        "models": {
            "openai:gpt-test": {
                "input_per_mtok": 2.0,
                "output_per_mtok": 8.0,
                "cache_read_per_mtok": 1.0,
                "cache_write_per_mtok": None,
                "as_of": "2026-08-01",
            }
        }
    }
    usage = ProviderUsage(
        uncached_input=0, cache_write=1_000_000, provider="openai", model="gpt-test"
    )
    cost = _catalog(config).cost_for_usage(usage)
    assert cost.cache_write_cost == 0.0


def test_every_seeded_entry_has_as_of_date():
    catalog = _catalog()
    for key, entry in catalog.direct_mappings.items():
        assert entry.get("as_of"), f"seed entry {key} missing as_of"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/LLM_Calls/test_pricing_catalog.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the module**

Structure copies `tldw_chatbook/model_capabilities.py` exactly (plain dicts for seeds, compiled per-provider patterns, lazy global). Key implementation points:

```python
# tldw_chatbook/LLM_Calls/pricing_catalog.py
"""Per-model pricing (dollars per million tokens) for the cost ticker.

Resolution order: [pricing].models config override -> seeded direct map ->
[pricing].patterns config override -> seeded pattern fallback -> local
provider zero-rate -> None. None means "no pricing data" and the UI must
show token counts instead of fabricating a dollar figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tldw_chatbook.Chat.provider_usage import ProviderUsage

_SEED_AS_OF = "2026-08-01"

# Providers that run locally: always $0.00.
LOCAL_PROVIDERS = frozenset(
    {
        "llama_cpp",
        "llamacpp",
        "ollama",
        "vllm",
        "koboldcpp",
        "kobold",
        "oobabooga",
        "tabbyapi",
        "aphrodite",
        "local-llm",
        "mlx-lm",
        "onnx",
        "transformers",
    }
)

_ZERO = {
    "input_per_mtok": 0.0,
    "output_per_mtok": 0.0,
    "cache_read_per_mtok": 0.0,
    "cache_write_per_mtok": 0.0,
    "as_of": _SEED_AS_OF,
}

def _entry(inp, out, cr=None, cw=None, as_of=_SEED_AS_OF):
    return {
        "input_per_mtok": inp,
        "output_per_mtok": out,
        "cache_read_per_mtok": cr,
        "cache_write_per_mtok": cw,
        "as_of": as_of,
    }

# Anthropic: cache read = 0.1x input, cache write = 1.25x input (5-min TTL).
DEFAULT_MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    "anthropic:claude-opus-4-1": _entry(15.00, 75.00, 1.50, 18.75),
    "anthropic:claude-sonnet-4-6": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-sonnet-4-5": _entry(3.00, 15.00, 0.30, 3.75),
    "anthropic:claude-haiku-4-5": _entry(1.00, 5.00, 0.10, 1.25),
    # ... full seed table: see Step 4 (verification) before finalizing
    # OpenAI / Google / Mistral / Cohere / Groq / DeepSeek entries.
    "openai:gpt-4o": _entry(2.50, 10.00, 1.25, None),
    "openai:gpt-4o-mini": _entry(0.15, 0.60, 0.075, None),
    "openai:gpt-4.1": _entry(2.00, 8.00, 0.50, None),
    "openai:gpt-4.1-mini": _entry(0.40, 1.60, 0.10, None),
    "google:gemini-2.5-pro": _entry(1.25, 10.00, None, None),
    "google:gemini-2.5-flash": _entry(0.30, 2.50, None, None),
    "mistral:mistral-large-latest": _entry(2.00, 6.00, None, None),
    "mistral:mistral-small-latest": _entry(0.10, 0.30, None, None),
    "cohere:command-r-plus": _entry(2.50, 10.00, None, None),
    "cohere:command-r": _entry(0.15, 0.60, None, None),
    "groq:llama-3.3-70b-versatile": _entry(0.59, 0.79, None, None),
    "deepseek:deepseek-chat": _entry(0.27, 1.10, 0.07, None),
}

DEFAULT_PRICING_PATTERNS: Dict[str, List[Dict[str, Any]]] = {
    "anthropic": [
        {"pattern": r"^claude-opus", **_entry(15.00, 75.00, 1.50, 18.75)},
        {"pattern": r"^claude-sonnet", **_entry(3.00, 15.00, 0.30, 3.75)},
        {"pattern": r"^claude-haiku", **_entry(1.00, 5.00, 0.10, 1.25)},
    ],
    "openai": [
        {"pattern": r"^gpt-4o-mini", **_entry(0.15, 0.60, 0.075, None)},
        {"pattern": r"^gpt-4o", **_entry(2.50, 10.00, 1.25, None)},
    ],
}
```

Then `ModelPricing` / `CostBreakdown` frozen dataclasses; `PricingCatalog.__init__(config=None)` loading the top-level `pricing` config section (lazy `load_cli_config_and_ensure_existence` import when `config is None`, exactly like `model_capabilities.py:149-155`); `direct_mappings` = seed dict merged with `config["models"]` overrides (override wins); `pattern_configs` likewise; `get_pricing(provider, model)` lowercases both, checks direct key `f"{provider}:{model}"`, then patterns for that provider (first match), then `LOCAL_PROVIDERS` zero-rate, else `None`, returning a `ModelPricing`; `cost_for_usage(usage)` calls `get_pricing(usage.provider, usage.model)` and multiplies each bucket by `rate/1_000_000` treating `None` cache rates as `0.0`, `round(x, 6)` each; module-level `_global_catalog` + `get_pricing_catalog()` + `reload_pricing_catalog()`.

- [ ] **Step 4: Verify seeded rates against official pricing pages**

For each provider in the seed table, WebSearch/WebFetch its official pricing page (Anthropic rates above are already verified: Opus 4.1 $15/$75, Sonnet 4.6 & 4.5 $3/$15, Haiku 4.5 $1/$5, cache read 0.1×, cache write 1.25×). Verify and correct **every** OpenAI/Google/Mistral/Cohere/Groq/DeepSeek number (drafts above may be stale), extend the table to the current model lineup of each provider, and set `_SEED_AS_OF` to the verification date. Cross-check the seeded provider keys against the provider names the Console actually uses (`grep -o '"[a-z_]*":' tldw_chatbook/Chat/Chat_Functions.py | sort -u` around `PROVIDER_PARAM_MAP` / `API_CALL_HANDLERS`) — keys must match those exact strings, and extend `LOCAL_PROVIDERS` with any local handler names found there that are missing.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest Tests/LLM_Calls/test_pricing_catalog.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/LLM_Calls/pricing_catalog.py Tests/LLM_Calls/
git commit -m "feat(pricing): seeded config-overridable per-model pricing catalog"
```

---

### Task 3: Anthropic streaming usage emission

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (`chat_with_anthropic` stream_generator, ~:1285-1431)
- Test: `Tests/Chat/test_anthropic_streaming_usage.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (wire contract Task 5 depends on): the streaming generator yields, in addition to today's chunks, SSE lines of the shape
  `data: {"id": ..., "object": "chat.completion.chunk", "model": ..., "choices": [], "usage": {<anthropic-native keys>}}\n\n`
  — one after `message_start` (input + cache buckets) and one after the event loop ends (adds cumulative `output_tokens` from the last `message_delta`). Consumers that ignore empty-`choices` chunks are unaffected. Non-streaming already returns `"usage"` (`LLM_API_Calls.py:1490`) — no change there.

- [ ] **Step 1: Write the failing test**

Mirror the mocking pattern documented at the top of `Tests/Chat/test_anthropic_native_tools.py` (patch `requests.Session.post`, drive via `chat_api_call`):

```python
# Tests/Chat/test_anthropic_streaming_usage.py
"""chat_with_anthropic streaming must surface usage as empty-choices SSE
chunks (message_start -> input/cache buckets; end of stream -> output)."""

import json
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}".encode("utf-8")


ANTHROPIC_STREAM_LINES = [
    _sse(
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "usage": {
                    "input_tokens": 3571,
                    "cache_read_input_tokens": 6656,
                    "cache_creation_input_tokens": 1024,
                },
            },
        }
    ),
    _sse(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
    ),
    _sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 727},
        }
    ),
    _sse({"type": "message_stop"}),
]


def _usage_chunks(raw_chunks):
    found = []
    for raw in raw_chunks:
        body = raw.removeprefix("data:").strip()
        if not body or body == "[DONE]":
            continue
        payload = json.loads(body)
        if payload.get("usage") is not None:
            assert payload.get("choices") == [], "usage chunks carry no choices"
            found.append(payload["usage"])
    return found


@patch("requests.Session.post")
def test_streaming_emits_input_then_output_usage_chunks(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = iter(ANTHROPIC_STREAM_LINES)
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    generator = chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="claude-sonnet-4-6",
        streaming=True,
    )
    chunks = list(generator)

    usages = _usage_chunks(chunks)
    assert len(usages) == 2
    assert usages[0]["input_tokens"] == 3571
    assert usages[0]["cache_read_input_tokens"] == 6656
    assert usages[0]["cache_creation_input_tokens"] == 1024
    assert usages[1]["output_tokens"] == 727
    # Text chunks still flow, and [DONE] still terminates.
    assert any('"content": "Hello"' in c for c in chunks)
    assert chunks[-1].strip() == "data: [DONE]"


@patch("requests.Session.post")
def test_streaming_without_usage_events_emits_no_usage_chunk(mock_post):
    lines = [
        _sse(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            }
        ),
        _sse({"type": "message_stop"}),
    ]
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = iter(lines)
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    generator = chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="claude-sonnet-4-6",
        streaming=True,
    )
    assert _usage_chunks(list(generator)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/Chat/test_anthropic_streaming_usage.py -v`
Expected: FAIL — `len(usages) == 2` assertion (no usage chunks emitted today)

- [ ] **Step 3: Implement usage emission in `stream_generator`**

Inside `chat_with_anthropic`'s `stream_generator()` (`LLM_API_Calls.py:1285`):

1. Before the `for line_bytes ...` loop, add state and a helper:

```python
                usage_accumulator: dict = {}

                def _usage_sse_chunk() -> str:
                    sse_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": current_model,
                        "choices": [],
                        "usage": dict(usage_accumulator),
                    }
                    return f"data: {json.dumps(sse_chunk)}\n\n"
```

2. Add a `message_start` branch (alongside the existing `content_block_start` branch at ~:1322) that captures and immediately emits input-side usage, so an aborted stream still surfaces what the API already billed:

```python
                                if anthropic_event.get("type") == "message_start":
                                    start_usage = (
                                        anthropic_event.get("message") or {}
                                    ).get("usage")
                                    if isinstance(start_usage, dict) and start_usage:
                                        usage_accumulator.update(start_usage)
                                        yield _usage_sse_chunk()
                                    continue
```

3. In the existing `message_delta` branch (`:1361-1365`), replace the commented-out line with real capture (no yield here — output_tokens is cumulative and re-sent per delta):

```python
                                    delta_usage = anthropic_event.get("usage")
                                    if isinstance(delta_usage, dict):
                                        usage_accumulator.update(delta_usage)
```

4. After the `for` loop completes (still inside `try`, before the `finally` that yields `[DONE]`), emit the final merged usage if output arrived:

```python
                    if "output_tokens" in usage_accumulator:
                        yield _usage_sse_chunk()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_anthropic_streaming_usage.py Tests/Chat/test_anthropic_native_tools.py -v`
Expected: new tests PASS; existing native-tools streaming tests still PASS (regression check)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_anthropic_streaming_usage.py
git commit -m "feat(anthropic): emit usage as empty-choices SSE chunks in streaming"
```

---

### Task 4: OpenAI `stream_options` + Responses usage passthrough

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (`chat_with_openai` payload ~:601-607 and stream_generator ~:704-757; `_responses_stream_to_chat_sse` `response.completed` branch ~:317-325)
- Test: `Tests/Chat/test_openai_streaming_usage.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: OpenAI chat-completions streaming requests carry `"stream_options": {"include_usage": true}` (native `chat_with_openai` only — **no other provider function is touched**, per spec); OpenAI's own final usage chunk (`choices: []`, `usage: {...}`) already passes through verbatim (`:718-721`). The Responses path re-emits `event["response"]["usage"]` on its `response.completed` chunk. Degrade rule: a 400 whose body names `stream_options` triggers one retry without the parameter.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_openai_streaming_usage.py
"""chat_with_openai: stream_options opt-in + graceful 400 fallback +
Responses-API usage passthrough."""

import json
from unittest.mock import Mock, patch

import requests

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import _responses_stream_to_chat_sse


def _streaming_ok_response(lines):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(lines)
    response.close = Mock()
    return response


@patch("requests.Session.post")
def test_streaming_payload_includes_stream_options(mock_post):
    mock_post.return_value = _streaming_ok_response(
        ['data: {"choices": [{"delta": {"content": "hi"}}]}', "data: [DONE]"]
    )
    generator = chat_api_call(
        "openai",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    list(generator)
    sent_payload = mock_post.call_args[1]["json"]
    assert sent_payload["stream_options"] == {"include_usage": True}


@patch("requests.Session.post")
def test_non_streaming_payload_omits_stream_options(mock_post):
    ok = Mock()
    ok.status_code = 200
    ok.raise_for_status = Mock()
    ok.json.return_value = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    mock_post.return_value = ok
    chat_api_call(
        "openai",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=False,
    )
    assert "stream_options" not in mock_post.call_args[1]["json"]


@patch("requests.Session.post")
def test_400_naming_stream_options_retries_without_it(mock_post):
    bad = Mock()
    bad.status_code = 400
    bad.text = '{"error": {"message": "Unknown parameter: stream_options"}}'
    bad.raise_for_status.side_effect = requests.exceptions.HTTPError(response=bad)
    ok = _streaming_ok_response(
        ['data: {"choices": [{"delta": {"content": "hi"}}]}', "data: [DONE]"]
    )
    mock_post.side_effect = [bad, ok]

    generator = chat_api_call(
        "openai",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    chunks = list(generator)

    assert mock_post.call_count == 2
    retry_payload = mock_post.call_args_list[1][1]["json"]
    assert "stream_options" not in retry_payload
    assert any("hi" in c for c in chunks)


def test_responses_completed_event_carries_usage_through():
    lines = [
        'data: {"type": "response.output_text.delta", "delta": "hi"}',
        (
            'data: {"type": "response.completed", "response": {"usage": '
            '{"input_tokens": 1200, "output_tokens": 90, '
            '"input_tokens_details": {"cached_tokens": 1024}}}}'
        ),
    ]
    response = _streaming_ok_response(lines)
    chunks = list(_responses_stream_to_chat_sse(response, model="gpt-5-mini"))

    completed = [
        json.loads(c.removeprefix("data:").strip())
        for c in chunks
        if c.strip() not in ("data: [DONE]",) and '"usage"' in c
    ]
    assert len(completed) == 1
    assert completed[0]["usage"]["input_tokens"] == 1200
    assert completed[0]["usage"]["input_tokens_details"]["cached_tokens"] == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_openai_streaming_usage.py -v`
Expected: FAIL on `stream_options` assertion, retry-count assertion, and usage-passthrough assertion

- [ ] **Step 3: Implement**

1. Payload (`chat_with_openai`, after the `payload` dict is assembled at ~:601-608):

```python
    if final_streaming and not use_responses_api:
        payload["stream_options"] = {"include_usage": True}
```

2. Degrade rule inside `stream_generator()` (~:709-712) — replace the single `session.post(...)` + `raise_for_status()` with:

```python
                    response = session.post(
                        api_url, headers=headers, json=payload, stream=True, timeout=180
                    )
                    if (
                        response.status_code == 400
                        and "stream_options" in payload
                        and "stream_options" in (response.text or "")
                    ):
                        logger.warning(
                            "OpenAI: endpoint rejected stream_options; retrying without usage reporting."
                        )
                        retry_payload = {
                            k: v for k, v in payload.items() if k != "stream_options"
                        }
                        response = session.post(
                            api_url,
                            headers=headers,
                            json=retry_payload,
                            stream=True,
                            timeout=180,
                        )
                    response.raise_for_status()
```

3. Responses passthrough — in `_responses_stream_to_chat_sse`'s `response.completed` branch (~:317-325), before building `chunk`:

```python
            elif event_type == "response.completed":
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                completed_usage = (event.get("response") or {}).get("usage")
                if isinstance(completed_usage, dict):
                    chunk["usage"] = completed_usage
                yield f"data: {json.dumps(chunk)}\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_openai_streaming_usage.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_openai_streaming_usage.py
git commit -m "feat(openai): opt into stream usage reporting with 400 fallback"
```

---

### Task 5: Gateway records usage payload onto stream signals

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (`ConsoleProviderStreamSignals` :64-86; `normalize_provider_response` :1242-1300; `_content_from_provider_item` :1522; `_content_from_sse_data` :1537)
- Test: append to `Tests/Chat/test_console_provider_gateway.py`

**Interfaces:**
- Consumes: usage-bearing chunks from Tasks 3-4 (mappings or SSE strings whose payload has a `"usage"` mapping).
- Produces (Task 6 depends on): `ConsoleProviderStreamSignals.usage_payload: dict[str, Any] | None = None` and `record_usage_payload(self, payload: Mapping[str, Any]) -> None` which **merges** (`{**old, **new}`) so Anthropic's two-chunk emission (input first, output later) accumulates. `stream_chat`'s yield contract is **unchanged** (`str | ProviderToolCalls`).

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_provider_gateway.py`, following the existing fixture idiom at `:1010-1032`:

```python
@pytest.mark.asyncio
async def test_stream_chat_records_usage_payload_from_sse_chunk() -> None:
    usage_line = (
        'data: {"object": "chat.completion.chunk", "choices": [], '
        '"usage": {"prompt_tokens": 100, "completion_tokens": 20}}'
    )

    def fake_chat_api_call(**_kwargs):
        yield 'data: {"choices": [{"delta": {"content": "hi"}}]}'
        yield usage_line

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = ConsoleProviderStreamSignals()

    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]

    assert chunks == ["hi"]  # usage chunk yields no text
    assert signals.usage_payload == {"prompt_tokens": 100, "completion_tokens": 20}


@pytest.mark.asyncio
async def test_stream_chat_merges_split_usage_payloads() -> None:
    # Anthropic emits input-side usage at message_start and output at end.
    def fake_chat_api_call(**_kwargs):
        yield (
            'data: {"choices": [], "usage": {"input_tokens": 3571, '
            '"cache_read_input_tokens": 6656}}'
        )
        yield 'data: {"choices": [{"delta": {"content": "hi"}}]}'
        yield 'data: {"choices": [], "usage": {"output_tokens": 727}}'

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key": "k"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model="claude-sonnet-4-6")
    )
    signals = ConsoleProviderStreamSignals()
    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]
    assert signals.usage_payload == {
        "input_tokens": 3571,
        "cache_read_input_tokens": 6656,
        "output_tokens": 727,
    }


@pytest.mark.asyncio
async def test_non_streaming_mapping_response_records_usage() -> None:
    def fake_chat_api_call(**_kwargs):
        return {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = ConsoleProviderStreamSignals()
    chunks = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]
    assert chunks == ["hello"]
    assert signals.usage_payload == {"prompt_tokens": 10, "completion_tokens": 2}


@pytest.mark.asyncio
async def test_stream_without_usage_leaves_signals_none() -> None:
    def fake_chat_api_call(**_kwargs):
        yield "plain text"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
    )
    signals = ConsoleProviderStreamSignals()
    _ = [
        chunk
        async for chunk in gateway.stream_chat(
            resolution, [{"role": "user", "content": "hi"}], signals=signals
        )
    ]
    assert signals.usage_payload is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -k usage -v`
Expected: FAIL — `ConsoleProviderStreamSignals` has no `usage_payload`

- [ ] **Step 3: Implement**

1. `ConsoleProviderStreamSignals` (`:64`): add field + merge-recorder:

```python
    usage_payload: dict[str, Any] | None = None

    def record_usage_payload(self, payload: Mapping[str, Any]) -> None:
        """Merge a provider usage payload (Anthropic splits input/output)."""
        merged = dict(self.usage_payload or {})
        merged.update(payload)
        self.usage_payload = merged
```

2. Add a module-level extractor next to `_content_from_provider_mapping`:

```python
def _maybe_record_usage(
    payload: Mapping[str, Any],
    signals: "ConsoleProviderStreamSignals | None",
) -> None:
    if signals is None:
        return
    usage = payload.get("usage")
    if isinstance(usage, Mapping) and usage:
        signals.record_usage_payload(usage)
```

3. Thread `signals` down: give `_content_from_provider_item` and `_content_from_sse_data` a keyword-only `signals: "ConsoleProviderStreamSignals | None" = None` parameter. In `_content_from_sse_data`, call `_maybe_record_usage(payload, signals)` right after the `isinstance(payload, Mapping)` check (`:1547`). In `_content_from_provider_item`, pass `signals` through to `_content_from_sse_data` and call `_maybe_record_usage(item, signals)` in the `isinstance(item, Mapping)` branch (`:1532`).

4. In `normalize_provider_response` (`:1263` and the iteration loop `:1275`), pass `signals=signals` to every `_content_from_provider_item(...)` call.

- [ ] **Step 4: Run tests to verify they pass (plus gateway regression)**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -v`
Expected: new tests PASS; the kwargs-pinning test `test_stream_chat_generic_non_streaming_yields_completion_once` (`:950-1008`) still PASSES — `_chat_api_kwargs` was not touched.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat(console): gateway records provider usage onto stream signals"
```

---

### Task 6: Controller attaches usage to the assistant message

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py` (`ConsoleChatMessage` :396-426)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (new `set_message_usage`)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_run_direct_provider_reply` :5815-6020)
- Modify: `Tests/Chat/conftest.py:37` and every `stream_chat` stub in `Tests/Chat/test_console_chat_controller.py` (`StreamingGateway :52`, `RecordingStreamingGateway :71`, `CapturingGateway :80`, `FailingStreamingGateway :101`, `FailingBeforeChunkGateway :107`, `EmptyStreamingGateway :114`)
- Test: append to `Tests/Chat/test_console_chat_controller.py` and `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: `ConsoleProviderStreamSignals.usage_payload` (Task 5), `ProviderUsage.from_provider_payload` (Task 1).
- Produces (Tasks 8-9 depend on):
  - `ConsoleChatMessage.usage: "ProviderUsage | None" = None` (new field, after `citation_presentation`).
  - `ConsoleChatStore.set_message_usage(self, message_id: str, usage: ProviderUsage) -> ConsoleChatMessage` — sets the field in-store only (no persistence side effect; the terminal mark flushes).
  - Controller behavior: on stream completion (success **and** stopped), usage from signals is converted with `provider=resolution.provider, model=resolution.model` (`partial=True` on the stopped path) and attached **before** `mark_message_complete` / `finalize_variant_stream` / `_mark_stream_stopped`.

- [ ] **Step 1: Write the failing tests**

Store test (append to `Tests/Chat/test_console_chat_store.py`):

```python
def test_set_message_usage_sets_field_without_persist_call():
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="hi"
    )
    usage = ProviderUsage(uncached_input=10, output=5, provider="openai", model="gpt-4o")

    updated = store.set_message_usage(message.id, usage)

    assert updated.usage == usage
    assert store.get_message(message.id).usage == usage


def test_set_message_usage_unknown_id_raises_keyerror():
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store = ConsoleChatStore()
    store.ensure_session(title="Chat 1")
    with pytest.raises(KeyError):
        store.set_message_usage("missing", ProviderUsage())
```

Controller test (append to `Tests/Chat/test_console_chat_controller.py`; reuse the local `StreamingGateway` idiom):

```python
class UsageEmittingGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages, **kwargs):
        signals = kwargs.get("signals")
        for chunk in ("hel", "lo"):
            yield chunk
        if signals is not None:
            signals.record_usage_payload(
                {"prompt_tokens": 100, "completion_tokens": 20}
            )


@pytest.mark.asyncio
async def test_completed_message_carries_normalized_usage():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=UsageEmittingGateway()
    )
    session = store.ensure_session(title="Chat 1")

    result = await controller.submit_draft("hi")
    assert result.accepted

    messages = store.messages_for_session(session.id)
    assistant = messages[-1]
    assert assistant.status == "complete"
    assert assistant.usage is not None
    assert assistant.usage.uncached_input == 100
    assert assistant.usage.output == 20
    assert assistant.usage.partial is False
    assert assistant.usage.provider  # attributed from resolution
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_chat_store.py -k usage -v` and `pytest Tests/Chat/test_console_chat_controller.py -k usage -v`
Expected: FAIL — no `set_message_usage`, no `usage` field

- [ ] **Step 3: Implement model + store**

1. `console_chat_models.py` — add to `ConsoleChatMessage` (after `citation_presentation`, `:426`), with a `TYPE_CHECKING` import for `ProviderUsage`:

```python
    # Normalized token usage for THIS generation (None for user rows, legacy
    # rows, and providers that reported nothing). Persisted as usage_json.
    usage: "ProviderUsage | None" = None
```

2. `console_chat_store.py` — new method next to `mark_message_complete` (`:1890`):

```python
    def set_message_usage(
        self, message_id: str, usage: ProviderUsage
    ) -> ConsoleChatMessage:
        """Attach normalized usage; the terminal mark persists it."""
        message = self._message_or_raise(message_id)
        message.usage = usage
        return self._snapshot(message)
```

- [ ] **Step 4: Implement controller attach + always-pass signals**

In `_run_direct_provider_reply` (`:5815`):

1. Ensure signals always exist — replace the `stream_signals is None` branching at `:5876-5886` with:

```python
            if stream_signals is None:
                stream_signals = ConsoleProviderStreamSignals()
            provider_stream = self.provider_gateway.stream_chat(
                resolution,
                provider_messages,
                signals=stream_signals,
            )
```

2. Add a private helper on the controller:

```python
    def _attach_stream_usage(
        self,
        assistant_message_id: str,
        stream_signals: ConsoleProviderStreamSignals | None,
        resolution: Any,
        *,
        partial: bool,
    ) -> None:
        """Best-effort: absent usage must never fail a send (spec PR1)."""
        payload = getattr(stream_signals, "usage_payload", None)
        if not payload:
            return
        usage = ProviderUsage.from_provider_payload(
            payload,
            provider=str(getattr(resolution, "provider", "") or ""),
            model=str(getattr(resolution, "model", "") or ""),
            partial=partial,
        )
        if usage is None:
            return
        try:
            self.store.set_message_usage(assistant_message_id, usage)
        except KeyError:
            pass
```

3. Call it `partial=False` immediately **before** the success finalization block (`:5969`), and `partial=True` immediately before each `_mark_stream_stopped(...)` call inside this method (`:5892`, `:5925`, `:5989`). Do **not** attach on the failed/empty paths (spec: failed sends produce no usage row).

4. Update the test stubs so the new `signals=` keyword doesn't break them: in `Tests/Chat/conftest.py:37` and each stub listed in **Files**, change `async def stream_chat(self, resolution, messages):` to `async def stream_chat(self, resolution, messages, **kwargs):` (mechanical; grep `def stream_chat` in `Tests/Chat/`).

- [ ] **Step 5: Verify the variant path**

Read `finalize_variant_stream` (`console_chat_store.py:~1596`) and the regenerate flow. Confirm the usage attach before `finalize_variant_stream` lands the usage on the message the variant flush persists. Add one test in `Tests/Chat/test_console_variant_stream.py` asserting a regenerated variant's completed message carries the new generation's usage. If variant persistence writes a separate DB row per variant (expected — `variant_of` FK), each flush picks up the then-current `message.usage`, which is the correct per-generation attribution; document whatever is found in the test's docstring.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_stop_reliability.py -v`
Expected: all PASS (stop-reliability guards the stopped-path change)

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py \
        tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/
git commit -m "feat(console): attach normalized stream usage to assistant messages"
```

---

### Task 7: DB migration v28→v29 — `usage_json` column

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_message_usage.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (`_CURRENT_SCHEMA_VERSION` :165; `migration_steps` :4307-4332; new `_migrate_from_v28_to_v29`; `add_message` INSERT :7753; `update_message` whitelist :8364; SELECT column lists at :6870-6874, :6913-6918, :6950-6953, :7825, :8223)
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py` (`normalize_message_row` :150-191)
- Test: `Tests/DB/test_chachanotes_message_usage_migration.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure schema work; JSON string in/out).
- Produces (Tasks 8-9 depend on): nullable `messages.usage_json TEXT` column; `add_message` accepts `msg_data["usage_json"]`; `update_message(..., usage_json=...)` whitelisted; every conversation-tree SELECT returns `usage_json`; `normalize_message_row` includes `"usage_json"` in its dict.

- [ ] **Step 1: Write the failing migration test**

Copy the seeding idiom from `Tests/DB/test_chachanotes_character_authority_migration.py:36-45` (monkeypatch `_CURRENT_SCHEMA_VERSION` to build the old-version DB, then reopen at the new version):

```python
# Tests/DB/test_chachanotes_message_usage_migration.py
"""v28 -> v29: local-only messages.usage_json column (cost ticker PR1).

Local-only means: the column must NOT appear in any messages_sync_* trigger
payload — same precedent as v24/v25/v26 local tables.
"""

from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        ("chachanotes",),  # match the constant used by the sibling migration tests
    ).fetchone()
    return int(row[0])


def _message_columns(connection) -> set[str]:
    return {
        row[1] for row in connection.execute("PRAGMA table_info(messages)").fetchall()
    }


def _seed_v28_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as v28_patch:
        v28_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 28)
        db = CharactersRAGDB(path, client_id="migration-seed")
        connection = db.get_connection()
        assert _version(connection) == 28
        assert "usage_json" not in _message_columns(connection)
        db.close_connection()


def test_migration_adds_usage_json_and_bumps_version(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v28_database(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()
    assert _version(connection) == 29
    assert "usage_json" in _message_columns(connection)
    db.close_connection()


def test_usage_json_excluded_from_sync_triggers(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v28_database(db_path, monkeypatch)
    db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()
    triggers = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_sync%'"
    ).fetchall()
    assert triggers, "expected messages sync triggers to exist"
    for (sql,) in triggers:
        assert "usage_json" not in (sql or "")
    db.close_connection()


def test_add_and_update_message_round_trip_usage_json(tmp_path):
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="usage-test")
    conv_id = db.add_conversation({"title": "t"})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "hi",
            "usage_json": '{"uncached_input": 10}',
        }
    )
    row = db.get_message_by_id(msg_id)
    assert row["usage_json"] == '{"uncached_input": 10}'

    db.update_message(msg_id, usage_json='{"uncached_input": 99}')
    assert db.get_message_by_id(msg_id)["usage_json"] == '{"uncached_input": 99}'
```

Before running: open a sibling migration test (e.g. `Tests/DB/test_chachanotes_character_authority_migration.py`) and align the helper details (`schema_name` constant, `add_conversation` argument shape, connection accessors) with what those tests actually use — copy their exact idioms.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/DB/test_chachanotes_message_usage_migration.py -v`
Expected: FAIL — version still 28 / no `usage_json` column

- [ ] **Step 3: Implement the migration**

1. SQL file `tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_message_usage.sql`:

```sql
-- ChaChaNotes v28 -> v29: local-only per-message usage (cost ticker PR1).
-- DDL only. NOTE: no trigger DDL — usage_json is LOCAL-ONLY and must never
-- reach sync_log (same rule as v19/v24/v25/v26 local-only migrations).

ALTER TABLE messages ADD COLUMN usage_json TEXT DEFAULT NULL;
```

2. In `ChaChaNotes_DB.py`: bump `_CURRENT_SCHEMA_VERSION = 29` (`:165`, update the trailing comment); register `28: self._migrate_from_v28_to_v29` in `migration_steps` (`:4331`); add the runner following the embedded-script pattern of `_migrate_from_v24_to_v25` (`:3949-3968`) — version-guard `!= 28` raising `SchemaError`, execute the ALTER, then `UPDATE db_schema_version SET version = 29 WHERE schema_name = ? AND version = 28` with a `rowcount != 1` `SchemaError` check (copy `_update_character_authority_schema_version` `:4107-4121` shape), and a comment `-- Keep this runner SQL aligned with tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_message_usage.sql`.

3. Plumb the column:
   - `add_message` INSERT (`:7753`): add `usage_json` to the column list and a `?` placeholder; bind `msg_data.get("usage_json")`.
   - `update_message` whitelist (`:8364-8371`): append `"usage_json"`.
   - Add `m.usage_json` to the SELECT lists in `get_root_messages_for_conversation` (`:6913`), `get_messages_for_conversation_by_parent_ids` (`:6950`), `get_messages_for_conversation` (`:8223`), `get_latest_message_for_conversation` (`:6870`), and `get_message_by_id` (`:7825`).
   - `chat_conversation_service.py` `normalize_message_row` (`:173-191`): add `"usage_json": row_value("usage_json")` using the same row-access idiom the function already uses for `feedback`.

- [ ] **Step 4: Run tests to verify they pass (plus DB regression)**

Run: `pytest Tests/DB/test_chachanotes_message_usage_migration.py Tests/ChaChaNotesDB/ -v`
Expected: new tests PASS; existing ChaChaNotes suite PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/ tldw_chatbook/Chat/chat_conversation_service.py Tests/DB/test_chachanotes_message_usage_migration.py
git commit -m "feat(db): v29 migration adds local-only messages.usage_json"
```

---

### Task 8: Persistence write path (store → adapter → DB)

**Files:**
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` (`create_message` :556-570; `update_message_content` :378-390)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`ConsoleChatPersistence` Protocol :64-113; `_persist_new_message` :2567-2586; `_persist_existing_message` :2738-2758)
- Test: append to `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: `usage_json` column plumbing (Task 7), `ConsoleChatMessage.usage` (Task 6).
- Produces: `ChatPersistenceService.create_message(..., usage_json: Optional[str] = None)` and `update_message_content(..., usage_json: Optional[str] = None)`; the store passes `usage_json=message.usage.to_json()` **only when** the adapter declares the kwarg (via the existing `_persistence_accepts_kwarg` probe, `:2510`) — narrow test fakes without the kwarg keep working untouched.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_chat_store.py`, next to the existing `FakePersistence` (`:573`) / `RecordingPersistence` (`:1792`) tests:

```python
def test_terminal_flush_passes_usage_json_to_accepting_persistence():
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    class UsagePersistence(RecordingPersistence):  # RecordingPersistence at :1792
        def __init__(self):
            super().__init__()
            self.update_usage_values = []

        def update_message_content(self, *, usage_json=None, **kwargs):
            self.update_usage_values.append(usage_json)
            return super().update_message_content(**kwargs)

    persistence = UsagePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hello")
    store.set_message_usage(
        message.id,
        ProviderUsage(uncached_input=10, output=2, provider="openai", model="gpt-4o"),
    )

    store.mark_message_complete(message.id)

    assert persistence.update_usage_values
    stored = persistence.update_usage_values[-1]
    assert stored is not None and '"uncached_input": 10' in stored


def test_narrow_persistence_without_usage_kwarg_still_works():
    # FakePersistence (:596/:622) declares keyword-only params and no
    # **kwargs — the _persistence_accepts_kwarg probe must skip usage_json.
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    store.append_stream_chunk(message.id, "hello")
    store.set_message_usage(message.id, ProviderUsage(uncached_input=1))

    completed = store.mark_message_complete(message.id)  # must not raise
    assert completed.status == "complete"
```

(Align constructor/fixture details with how the surrounding tests in that file build `ConsoleChatStore(persistence=...)` and sessions — copy the local idiom.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_chat_store.py -k usage_json -v`
Expected: first test FAILS (no `usage_json` forwarded); second may pass trivially — keep it as the regression guard.

- [ ] **Step 3: Implement**

1. `chat_persistence_service.py`: add `usage_json: Optional[str] = None` to both `create_message` (include in `message_payload` at `:686-695` as `"usage_json": usage_json`) and `update_message_content` (pass `usage_json=usage_json` to `self.db.update_message(...)` at each of the three call sites `:508/:538/:549` — only when not `None`, to avoid overwriting an existing value with NULL on content-only updates).
2. `console_chat_store.py`:
   - Extend the `ConsoleChatPersistence` Protocol method signatures with `usage_json: str | None = None`.
   - `_persist_existing_message` (`:2749`): after building `update_kwargs`, add:

```python
        if message.usage is not None and self._persistence_accepts_kwarg(
            self.persistence.update_message_content, "usage_json"
        ):
            update_kwargs["usage_json"] = message.usage.to_json()
```

   - `_persist_new_message` (`:2586`): same guarded pattern for `create_kwargs` against `self.persistence.create_message`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_chat_store.py -v`
Expected: all PASS (including every pre-existing FakePersistence/RecordingPersistence test)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): persist message usage_json through the adapter seam"
```

---

### Task 9: Hydration — reopened conversations restore usage

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_console_messages_from_conversation_tree` :7447-7508)
- Test: append to `Tests/UI/test_console_resume_active_path.py` (or the sibling resume test file whose fixtures already build a conversation tree — inspect both and pick the one with a tree-dict fixture)

**Interfaces:**
- Consumes: `normalize_message_row`'s `"usage_json"` key (Task 7), `ProviderUsage.from_json` (Task 1).
- Produces: restored `ConsoleChatMessage.usage` populated from the DB, so PR3's cost total is real for reopened conversations.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_resume_active_path.py`, reusing its real-DB `_resume_into_store(db, conversation_id)` helper (the file's own idiom — real `CharactersRAGDB` behind the real service chain, so this test also end-to-end-covers Task 7's SELECT plumbing):

```python
def test_resume_restores_usage_from_usage_json():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(
            id="usage-conv-1",
            title="Usage",
            scope_type="global",
            state="in-progress",
        )
        u1 = db.add_message(
            {
                "id": "m-usage-u1",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "u1",
                "timestamp": "2026-01-01T00:00:00.000000+00:00",
            }
        )
        db.add_message(
            {
                "id": "m-usage-a1",
                "conversation_id": conversation_id,
                "parent_message_id": u1,
                "sender": "assistant",
                "role": "assistant",
                "content": "a1",
                "timestamp": "2026-01-01T00:00:01.000000+00:00",
                "usage_json": (
                    '{"uncached_input": 10, "cache_read": 0, "cache_write": 0,'
                    ' "output": 5, "provider": "openai", "model": "gpt-4o",'
                    ' "partial": false}'
                ),
            }
        )

        store, session = _resume_into_store(db, conversation_id)

        assistant = store.messages_for_session(session.id)[-1]
        assert assistant.content == "a1"
        assert assistant.usage is not None
        assert assistant.usage.uncached_input == 10
        assert assistant.usage.output == 5
        assert assistant.usage.provider == "openai"
    finally:
        db.close_connection()


def test_resume_tolerates_null_and_garbage_usage_json():
    # Legacy rows (NULL) and corrupt JSON must load with usage=None, never raise.
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(
            id="usage-conv-2",
            title="UsageLegacy",
            scope_type="global",
            state="in-progress",
        )
        u1 = db.add_message(
            {
                "id": "m-legacy-u1",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "u1",
                "timestamp": "2026-01-01T00:00:00.000000+00:00",
            }
        )
        db.add_message(
            {
                "id": "m-legacy-a1",
                "conversation_id": conversation_id,
                "parent_message_id": u1,
                "sender": "assistant",
                "role": "assistant",
                "content": "a1",
                "timestamp": "2026-01-01T00:00:01.000000+00:00",
                "usage_json": "{broken",
            }
        )

        store, session = _resume_into_store(db, conversation_id)

        assert all(
            m.usage is None for m in store.messages_for_session(session.id)
        )
    finally:
        db.close_connection()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_resume_active_path.py -k usage -v`
Expected: FAIL — `usage` is None for the populated node

- [ ] **Step 3: Implement**

In `_console_messages_from_conversation_tree`'s `_walk` (`chat_screen.py:7470`), next to the `image_mime_type` extraction:

```python
            usage = ProviderUsage.from_json(node.get("usage_json"))
```

and pass `usage=usage` in the `ConsoleChatMessage(...)` construction (`:7496-7507`). Import `ProviderUsage` at the top of `chat_screen.py` alongside the other `tldw_chatbook.Chat.*` imports. (`from_json` already returns `None` for null/garbage — no extra guard needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_resume_active_path.py Tests/UI/test_console_native_chat_flow.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/
git commit -m "feat(console): restore per-message usage when reopening conversations"
```

---

### Task 10: Full-suite gate + PR prep

**Files:**
- No new code. Test/verify only.

- [ ] **Step 1: Run the touched-module suites**

Run: `pytest Tests/Chat/ Tests/DB/ Tests/ChaChaNotesDB/ Tests/UI/test_console_resume_active_path.py Tests/UI/test_console_native_chat_flow.py Tests/LLM_Calls/ -x -q`
Expected: all PASS

- [ ] **Step 2: Run the full suite**

Run: `pytest -n 8 -q`
Expected: green (matches the CI core/ui shards). Investigate any failure before proceeding — do not hand-wave "pre-existing" without checking the same test on a clean `origin/dev` checkout.

- [ ] **Step 3: Re-verify the schema version claim**

Run: `git fetch origin dev && git show origin/dev:tldw_chatbook/DB/ChaChaNotes_DB.py | grep "_CURRENT_SCHEMA_VERSION ="`
Expected: still `28` upstream. If another session shipped 29 meanwhile, renumber this migration to the next free version (constant, registry key, runner name, SQL filename + header, and test) before pushing.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/console-cost-usage-foundation
gh pr create --base dev --title "feat(console): usage capture + pricing foundation (cost ticker PR1)" --body "$(cat <<'EOF'
PR1 of the console cost-ticker program (spec: Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md).

- Providers emit usage (Anthropic streaming via empty-choices SSE chunks; OpenAI via stream_options opt-in with 400 fallback; Responses API passthrough)
- Gateway records raw usage payloads onto ConsoleProviderStreamSignals (chunk contract unchanged)
- Controller normalizes into disjoint-bucket ProviderUsage and attaches to assistant messages (partial=true on stopped streams)
- Schema v29: local-only messages.usage_json (excluded from sync triggers) + hydration on conversation reopen
- New config-overridable pricing catalog seeded for cloud providers (never fabricates prices; local providers $0)

UI-invisible by design — PR2 (caching) and PR3 (ticker chip) build on this.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Spec-coverage checklist (PR1 section)

| Spec requirement | Task |
|---|---|
| Anthropic streaming usage from `message_start`/`message_delta` | 3 |
| OpenAI `stream_options` per-provider opt-in + 4xx degrade | 4 |
| Non-streaming usage passthrough | 5 (gateway records mapping usage; providers already return it) |
| Gateway exposes usage on run completion, chunk contract unchanged | 5 |
| Controller attaches usage to persisted assistant message | 6 |
| Disjoint buckets + per-provider adapters, unit-tested | 1 |
| Aborted stream → partial usage record | 3 (early input-side chunk) + 6 (`partial=True` on stopped) |
| Variants carry their own usage | 6 Step 5 |
| Ephemeral sessions: in-store field, inherits persistence behavior | 6 + 8 |
| Migration: nullable usage column, tokens-not-dollars, version re-verify | 7 + 10 Step 3 |
| Hydration read-back for reopened conversations | 9 |
| Pricing catalog: seeds → config overrides → patterns → $0 local → None | 2 |
| `as_of` date on every entry | 2 |
| Migration + hydration round-trip on real in-memory SQLite | 7 + 9 |
