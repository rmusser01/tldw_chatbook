# Generic Hosted Provider Engine — Phase 2 (Inference-Cloud Presets + Long-Tail Engine Swap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Together, Fireworks, and Cerebras as engine preset records (allowances derived from captured real-server fixtures), and move the ADR-146 custom-endpoint `openai_compatible` family onto the strict engine — swapped at the gateway's identity-resolution site — with keyless support, a fixture-gated tolerant profile, and zero parameter regression for local-server users.

**Architecture:** Phase 1 (branch `feat/provider-engine-phase1`, ADR-179) built `provider_registry.py` and `LLM_Calls/hosted_provider_engine.py` on `hosted_chat`. Phase 2 lands in evidence-first order: **capture real-server fixtures first** (Task 2), widen the engine's tolerance to what the fixtures prove real servers send (Tasks 3-4), then land presets whose allowances come from those fixtures (Task 5), and finally swap the custom-endpoint family at the gateway identity site — `family_execution_key()` and every consumer of its `"custom"` value stay untouched (Task 6).

**Tech Stack:** Python ≥3.12, `requests` via `hosted_chat`. No new dependencies. Local servers for fixtures: llama-server, Ollama (both runnable locally); vLLM where a runtime exists.

**Spec:** `Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md` (Summary item 3 with the Perplexity ruling, "Response variance and strictness", Phase 2 ACs — all amended 2026-09-24). Read before starting.

## Global Constraints

- Registry module `tldw_chatbook/provider_registry.py` stays stdlib-only (no `tldw_chatbook.*` imports).
- **Evidence-gated, in order**: no widening, allowance, or swap lands before the fixtures that justify it exist (Task 2 precedes Tasks 4-6; each consumes the fixtures Task 2 captured).
- Fail-closed strictness: required-shape validation (choices present, message role, tool-call shape, usage shape) is NEVER relaxed on any profile; the tolerant profile's extra acceptance is exactly: shape-safe unknown top-level/event keys, **null-valued** unknown choice/message keys, and empty-text `stop`/`length` finishes. Curated records stay fully strict outside their `response_allowances` (parity-pinned).
- Curated cloud presets hard-require API keys; only the `custom-hosted` family record uses `bearer_optional` auth (parity-pinned both directions).
- `moonshot.py`/`zai.py`, the two named `custom-openai-api` slots, `family_execution_key()`, and every consumer of its `"custom"` return value are NOT changed. The swap happens at the gateway identity site only.
- **Excluded providers**: xAI/Grok (ADR-179), and **Perplexity in Phase 2** — Sonar Chat Completions sunsets 2026-09-27; the Agent API is a different wire (tracked as spec open item O-4).
- No shipped `model` key for discovery-first presets (Phase 1 blank-model lesson: a present-but-blank model is rejected; omit the key).
- API keys env-first, masked `api_settings` otherwise, never logged (ADR-012).
- Tests: `.venv/bin/python -m pytest` from the worktree root, targeted files only. Commit after every green step, conventional commits with ADR-179 references. `backlog task create --ac` takes one criterion per flag — repeat the flag, never comma-join.
- Live probes are env-gated, paid, skip-clean; they do not block task Done (Phase 1 precedent) — but **fixture capture from free tiers and local servers is a hard prerequisite** (Task 2), not deferrable.
- Pre-work (before Task 1): once PR #2824 merges, rebase `feat/provider-engine-phase1` onto `origin/dev` (the duplicated baseline-fix commit drops out automatically as an already-applied patch). If executing before the merge, proceed on the current branch and rebase before any PR.

## Provider facts (verified 2026-09-24)

| Provider | key | Base URL | Env var | Discovery | Notes |
|---|---|---|---|---|---|
| Together | `together` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | `models` | free tier exists; **memory to verify by fixture**: top-level `prompt`, choice-level `logprobs` |
| Fireworks | `fireworks` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | `models` | R1-family returns `reasoning_content` → disposition `proprietary` |
| Cerebras | `cerebras` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | `models` | free tier exists; **memory to verify by fixture**: top-level `time_info` |
| ~~Perplexity~~ | — | — | — | — | **dropped**: Sonar Chat Completions sunsets 2026-09-27; Agent API rejects Chat-Completions params; no client-side function calling. Spec O-4. |

The "memory to verify" rows are exactly what Task 2's fixtures settle — allowances are written from captures, never from this table's memory claims.

---

### Task 1: Backlog task + registry-derived strict-hosted key list

**Files:**
- Modify: `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py` (`_STRICT_HOSTED_PROVIDER_KEYS`, ~L85, consumed at ~L207/389/556)
- Modify: `Tests/LLM_Provider_Catalog/test_engine_branch_guard.py` (replaced — see below)
- Create: backlog task via CLI

**Interfaces:**
- Produces: `_STRICT_HOSTED_PROVIDER_KEYS` derived as `frozenset({"moonshot", "zai"}) | {key for key, record in RECORDS_BY_KEY.items() if record.engine_driven}` (import `RECORDS_BY_KEY` from `provider_registry` the way the service already imports it — patch the service's own imported name in tests). No engine-branch guard is needed once the set is derived: a key cannot reach the engine branch unless its record is engine-driven. The Phase-1 deferred "unguarded else" minor is dissolved by this derivation, not patched.

- [ ] **Step 1: Create the Phase-2 backlog task** — one `--ac` flag per criterion (the CLI does not split commas):

```bash
backlog task create "Generic hosted provider engine Phase 2: inference-cloud presets + long-tail engine swap" \
  -d "Implement Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md Phase 2 (as amended 2026-09-24)" \
  -a @Robert -s "In Progress" \
  --ac "Real-server fixtures captured before swap/presets" \
  --ac "Together/Fireworks/Cerebras selectable+usable with fixture-derived allowances" \
  --ac "Preset-cost test proves no per-provider module" \
  --ac "custom-ep openai_compatible family executes via engine with identity/sessions unchanged" \
  --ac "Keyless entries keep working and curated presets still hard-require keys" \
  --ac "No parameter regression for local-server users (custom param surface + api_settings.custom timeouts)" \
  --ac "Tolerant profile scoped to long tail and gated by fixtures" \
  --ac "Docs updated"
backlog task edit <id> --plan "Plan: Docs/superpowers/plans/2026-09-24-generic-hosted-provider-engine-phase2.md"
```

- [ ] **Step 2: Write the failing test** — replace `Tests/LLM_Provider_Catalog/test_engine_branch_guard.py` with:

```python
"""The strict-hosted key list derives from the provider registry."""


def test_strict_hosted_keys_derive_from_registry():
    from tldw_chatbook.LLM_Provider_Catalog import (
        local_llm_provider_catalog_service as service,
    )
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    expected = {"moonshot", "zai"} | {
        key for key, record in RECORDS_BY_KEY.items() if record.engine_driven
    }
    assert service._STRICT_HOSTED_PROVIDER_KEYS == expected
    # every engine-driven key is admitted; nothing else is
    assert "databricks" in service._STRICT_HOSTED_PROVIDER_KEYS
    assert "openai" not in service._STRICT_HOSTED_PROVIDER_KEYS
```

Also add the alias-parity pin to `Tests/LLM_Calls/test_hosted_provider_engine_handler.py` (presented as a **pin that may already be green** — both functions carry the same explicit signature today; do not claim a RED step for it):

```python
def test_resolve_hosted_engine_request_alias_matches_private_signature():
    """Public alias forwards the full private surface (pinned, not RED-first)."""
    import inspect
    from tldw_chatbook.LLM_Calls import hosted_provider_engine as engine

    assert list(inspect.signature(engine.resolve_hosted_engine_request).parameters) == (
        list(inspect.signature(engine.resolve_hosted_request).parameters)
    )
```

- [ ] **Step 3: RED** — Run: `.venv/bin/python -m pytest Tests/LLM_Provider_Catalog/test_engine_branch_guard.py -q` — Expected: FAIL (today's set is the hard-coded `{"moonshot","zai","databricks"}`; derived adds nothing new until Task 5's records land — the test fails only if the derivation is wrong, and starts passing once the literal is replaced).

- [ ] **Step 4: Implement** — replace the hard-coded frozenset at ~L85 with the derivation (module import time; `provider_registry` is a stdlib-only leaf, no cycle). Keep the module's existing import style for `RECORDS_BY_KEY`.

- [ ] **Step 5: GREEN + regression** — `.venv/bin/python -m pytest Tests/LLM_Provider_Catalog/ Tests/LLM_Calls/test_hosted_provider_engine_handler.py -q` — PASS.

- [ ] **Step 6: Commit** — `refactor: derive strict-hosted key list from provider registry (ADR-179)`

---

### Task 2: Capture real-server fixtures (the evidence gate)

**Files:**
- Create: `Tests/fixtures/longtail/` (fixture JSONs + `CAPTURE.md`)
- Create: `Tests/fixtures/longtail/capture_local.py` (capture script)
- Create: `Tests/fixtures/longtail/capture_cloud.py` (env-gated cloud capture)
- Test: `Tests/LLM_Calls/test_longtail_fixture_characterization.py`

**Interfaces:**
- Produces: `Tests/fixtures/longtail/<server>.json` per captured server — `{"server": "...", "base_url": "...", "chat_response": <full non-streaming body>, "stream_events": [<full SSE data payloads in order>], "captured_at": "...", "capture_cmd": "..."}`; and the characterization test's per-server **rejection inventory** (the exact unknown keys the current strict parser rejects), which Tasks 4-5 must drive to empty.

- [ ] **Step 1: Write the capture scripts**

`capture_local.py` — for each server the environment can run, start it (subprocess, temp model where needed), issue one tiny chat + one streamed chat against its OpenAI-compatible endpoint, record the FULL raw bodies (never normalized), stop it:
  - **llama-server**: `llama-server -m <tiny.gguf> --port 8901 --host 127.0.0.1` (model path via `--model` arg or env `LLAMA_MODEL`; skip with a printed note if no binary/model).
  - **Ollama**: `ollama serve` on a free port + `OLLAMA_HOST`, `ollama pull qwen2.5:0.5b` first (skip if no binary).
  - **vLLM**: only if `docker` and ~2GB RAM are available: `docker run --rm -p 8903:8000 vllm/vllm-openai Qwen/Qwen2.5-0.5B` (CPU works, slowly; skip with a note otherwise).
  - LM Studio is a GUI app — not scripted; llama-server's wire is its family's representative. Record the skip in `CAPTURE.md`.

`capture_cloud.py` — env-gated (skip-clean): `TOGETHER_API_KEY` / `CEREBRAS_API_KEY` / `FIREWORKS_API_KEY` present → one minimal chat + stream per provider against the base URLs in the table above, full bodies recorded. Free tiers make Together/Cerebras costless; Fireworks records only if a key exists.

- [ ] **Step 2: Run the captures and commit the fixtures**

Run each script; commit whatever was captured (llama.cpp + Ollama at minimum are expected; cloud fixtures as keys allow). `CAPTURE.md` lists: date, versions, which servers were skipped and why, and the exact replay command. Fixtures are data, not secrets — redact nothing (they contain no credentials by construction).

- [ ] **Step 3: Write the characterization test**

```python
# Tests/LLM_Calls/test_longtail_fixture_characterization.py
"""Replay captured real-server fixtures through the CURRENT strict parser.

Phase 2's evidence gate: this test documents, per server, exactly which
unknown keys today's closed allowlists reject (top-level, choice, message).
Tasks 4-5 consume this inventory; when they land, the same fixtures must
parse clean under each server's profile. The inventory is asserted, not
printed — a fixture that starts failing differently is a spec event.
"""
import json
from pathlib import Path

import pytest

from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    normalize_hosted_chat_response,
)

_FIXTURES = sorted((Path(__file__).parent.parent / "fixtures" / "longtail").glob("*.json"))
_SERVERS = [p.stem for p in _FIXTURES]


class _Policy:
    reasoning_disposition = "ignored"

    def validate_finish(self, *, finish_reason, has_text, has_calls):
        return finish_reason  # permissive: we only inventory KEY rejections here

    def validate_reasoning_content(self, value):
        return None


@pytest.mark.parametrize("server", _SERVERS)
def test_current_strict_parser_rejection_inventory(server):
    fixture = json.loads(next(p for p in _FIXTURES if p.stem == server).read_text())
    body = fixture["chat_response"]
    with pytest.raises(HostedChatProtocolError) as excinfo:
        normalize_hosted_chat_response(body, finish_policy=_Policy())
    message = str(excinfo.value)
    # The inventory: which level's unknown keys the fixture tripped on.
    # Update these per-server expectations to match REALITY when the fixture
    # lands (e.g. llama-server: choice logprobs; together: prompt + choice
    # logprobs; cerebras: time_info) — the assertion pins the evidence.
    assert "malformed" in message
```

(Write the per-server expected-failure assertions from what the fixtures actually show — run once with `pytest -s` and a temporary print to enumerate, then pin each server's exact rejected keys as explicit `assert` lines. The point is a committed, reviewable inventory.)

- [ ] **Step 4: GREEN (as an inventory), commit** — `test+fixtures: capture real long-tail server envelopes; pin strict-parser rejection inventory (ADR-179)`

---

### Task 3: Engine `bearer_optional` auth

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py` (`_resolve_api_key` + transport construction)
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py` (validation + header construction, ~L517-565)
- Modify: `tldw_chatbook/provider_registry.py` (`auth_scheme` comment gains the literal)
- Test: `Tests/LLM_Calls/test_hosted_provider_engine_auth.py` (new)

**Interfaces:**
- Produces: `auth_scheme ∈ {"bearer", "bearer_optional", "api_key_header"}`; resolution raises on a missing key ONLY for `"bearer"`; for `"bearer_optional"` the resolution succeeds with `api_key == ""` when nothing resolves; an **explicit `""` counts as "no key"** (never an "explicit API key is invalid" error); `HostedHTTPTransportConfig(api_key="", auth_scheme="bearer_optional")` is valid and `owned_json_post` omits the `Authorization` header entirely when the key is empty. Two `_resolve_api_key` behavior changes make `bearer_optional` reachable at all: **skip the env-var-name validity check when no env-var name is set** (today a `None` name raises "api_key_env_var is invalid" before candidates are ever tried), and treat an explicit empty string as no-key rather than invalid.

- [ ] **Step 1: Failing tests** — the Phase-1-drafted file, corrected per review. **Reuse the transport fakes already proven in `Tests/LLM_Calls/test_hosted_chat.py`** (its fake `Session.post`/response helpers handle `self`, headers, and the close semantics `owned_json_post` expects — read that file first and import/adapt rather than hand-rolling a new `_FakeResponse`); the fake `post` signature must be `def fake_post(self, url, **kwargs)`:

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


def test_bearer_optional_resolves_without_a_key_and_without_an_env_name():
    resolution = resolve_hosted_request(
        _OPTIONAL,
        explicit_base_url="https://anywhere.example/v1",
        app_config={"api_settings": {"optional-auth": {}}},
        environ={},
    )
    assert resolution.api_key == ""


def test_bearer_optional_treats_explicit_empty_string_as_no_key():
    resolution = resolve_hosted_request(
        _OPTIONAL,
        explicit_api_key="",
        explicit_base_url="https://anywhere.example/v1",
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
    # Adapt the Session/response fake pattern from Tests/LLM_Calls/test_hosted_chat.py
    # (fake post signature is (self, url, **kwargs)); capture headers; assert
    # "Authorization" not in captured headers for an empty-key bearer_optional
    # config, and present for a non-empty key.
    ...
```

(Write the last test fully against the chosen fake — the assertions are: empty key → no `Authorization` header; non-empty key → `Authorization: Bearer <key>`.)

- [ ] **Step 2: RED** — `.venv/bin/python -m pytest Tests/LLM_Calls/test_hosted_provider_engine_auth.py -q` — FAIL (`bearer_optional` unknown; `_OPTIONAL` resolution raises on the None env name).

- [ ] **Step 3: Implement** — per the Interfaces block: `_resolve_api_key` env-name-check skip + explicit-empty-as-no-key + required-only-for-`"bearer"`; `HostedHTTPTransportConfig.auth_scheme` field with the validation split; conditional header build (never an empty `Bearer`); engine factory passes `record.auth_scheme` through.

- [ ] **Step 4: GREEN + regression** — `.venv/bin/python -m pytest Tests/LLM_Calls/test_hosted_provider_engine_auth.py Tests/LLM_Calls/test_hosted_chat_allowances.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_zai.py Tests/LLM_Calls/test_moonshot.py -q` — PASS (default `bearer` byte-identical).

- [ ] **Step 5: Commit** — `feat: engine bearer_optional auth for keyed-or-keyless endpoints (ADR-179)`

---

### Task 4: Widen tolerance — levels, null-valued extras, empty-text finishes

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py` (choice/message/delta checks + finish-policy-adjacent handling stays in the engine)
- Modify: `tldw_chatbook/provider_registry.py` (`tolerant_response_extras: bool = False`)
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py` (allowance plumb-through at all levels; `HostedPresetFinishPolicy` tolerant branch)
- Test: extend `Tests/LLM_Calls/test_hosted_chat_allowances.py`; flip the Task 2 characterization inventory

**Interfaces:**
- Produces:
  - `allowed_extra_keys` (curated allowances) now subtracts from **all four closed allowlists**: top-level response keys, stream-event keys, choice-level keys, and message/delta-level keys. An allowlisted value must be `None` or shape-safe; it is dropped.
  - `tolerant_response_extras=True` (long-tail record only): any shape-safe unknown top-level/event key dropped; any unknown **null-valued** choice/message/delta key dropped; unknown **non-null** choice/message keys still fail closed; and the engine's finish policy accepts `stop`/`length` with empty text and no tool calls (empty reply, the legacy behavior) — `tool_calls`-finish without calls, or `stop` with calls, still fail.
- Consumes: Task 2's fixture inventory — the exact keys it pinned (expected: choice `logprobs`, vLLM `stop_reason`, message `refusal`, Together top-level `prompt` + choice `logprobs`, Cerebras `time_info`).

- [ ] **Step 1: Failing tests** — keep the matrix minimal per review (default-False pins + one CUSTOM-tailored pin + the fixture flip):

```python
def test_curated_allowances_apply_at_choice_and_message_levels():
    # allowlisted choice key "logprobs" (null) and message key "refusal" (null)
    # are dropped and the turn parses; the SAME body WITHOUT the allowance
    # still fails closed. One non-streaming case + one stream-event case.
    ...


def test_tolerant_profile_accepts_null_valued_choice_and_message_extras():
    # tolerant flag + logprobs/stop_reason/refusal nulls -> parses.
    # tolerant flag + choice key with a NON-null unknown value -> still raises.
    ...


def test_tolerant_finish_policy_allows_empty_text_stop_and_length():
    # engine finish policy on a tolerant record: finish stop/length, no text,
    # no calls -> empty-text turn (legacy behavior). stop-with-calls and
    # tool_calls-without-calls still raise. Strict records unchanged
    # (existing Phase 1 tests already pin that).
    ...
```

Then **flip the Task 2 characterization test**: each fixture's body must now parse under the profile it will ship with (long-tail fixtures under the tolerant flag; cloud fixtures under their preset allowances — where a fixture needs an allowance that doesn't exist yet, that value goes into the Task 5 record and the flip lands with Task 5; sequence: flip long-tail fixtures here, cloud fixtures in Task 5's step).

- [ ] **Step 2: RED → implement → GREEN** — mirror the Phase 1 allowances mechanics at the two additional levels; null-check before dropping; engine finish-policy branch keyed on `record.tolerant_response_extras`; engine wrappers pass `tolerant_top_level_extras=record.tolerant_response_extras` (and the record flag) through.

- [ ] **Step 3: Regression** — `.venv/bin/python -m pytest Tests/LLM_Calls/test_hosted_chat_allowances.py Tests/LLM_Calls/test_hosted_provider_engine_policy.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_hosted_provider_engine_handler.py -q` — PASS. Also update the Phase 1 test whose name/pins said tolerance never extends to choice/delta keys — the tolerant-profile scoping is now explicitly null-only at those levels; rename and re-pin it (curated-allowance-free strict records still reject `logprobs: null`).

- [ ] **Step 4: Commit** — `feat: level-scoped allowances + fixture-gated long-tail tolerant profile (ADR-179)`

---

### Task 5: Three inference-cloud presets (allowances from fixtures) + preset-cost test

**Files:**
- Modify: `tldw_chatbook/provider_registry.py` (three records + ALL_RECORDS)
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (three dispatch entries)
- Modify: `tldw_chatbook/config.py` (`[providers]` empty seeds + three `[api_settings.*]` tables)
- Modify: `Chat/provider_readiness.py`, `Chat/console_provider_support.py`, `Agents/native_tools.py`, `LLM_Provider_Catalog/model_catalog_settings.py`, `LLM_Provider_Catalog/openai_compatible_model_discovery.py` (only if `/inference/v1` fails the discovery gate — follow the `/openai/v1` precedent)
- Test: `Tests/LLM_Calls/test_inference_cloud_presets.py` (new), parity extensions, Task 2 cloud-fixture flips

**Interfaces:**
- Produces: registry keys `together`, `fireworks`, `cerebras` (config keys `Together`, `Fireworks`, `Cerebras`), dispatch-registered; each record's `response_allowances` is **exactly the unknown-key set its captured fixture tripped** (Task 2's inventory), no more.

- [ ] **Step 1: Failing tests**

```python
# Tests/LLM_Calls/test_inference_cloud_presets.py
"""Preset cheapness: three inference clouds are records + one line each."""
from pathlib import Path

import pytest

from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

_PRESETS = ("together", "fireworks", "cerebras")
_LLM_CALLS = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"


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
    for key in _PRESETS:
        assert not list(_LLM_CALLS.glob(f"{key}*.py")), key


def test_preset_default_urls_and_env_vars():
    expected = {
        "together": ("https://api.together.xyz/v1", ("TOGETHER_API_KEY",)),
        "fireworks": ("https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",)),
        "cerebras": ("https://api.cerebras.ai/v1", ("CEREBRAS_API_KEY",)),
    }
    for key, (url, envs) in expected.items():
        record = RECORDS_BY_KEY[key]
        assert record.default_base_url == url, key
        assert record.api_key_env_candidates == envs, key


def test_preset_allowances_match_captured_fixtures():
    """Allowances are the fixture inventory, not memory. For each preset with
    a captured fixture, response_allowances == the unknown-key set the Task 2
    inventory pinned for it (write the exact expected frozensets from the
    fixtures — e.g. together: frozenset({"prompt"}) plus choice-level
    {"logprobs"} if the allowance type splits by level; match whatever shape
    Task 4 landed)."""
    ...


def test_fireworks_proprietary_reasoning_disposition():
    assert RECORDS_BY_KEY["fireworks"].reasoning_disposition == "proprietary"


def test_captured_cloud_fixtures_parse_under_preset_allowances():
    """Flip of the Task 2 cloud-side inventory: each captured cloud fixture
    body parses through the engine under its own record."""
    ...
```

Also extend `Tests/test_provider_registry.py` parity tests to include the three keys with **no exclusions**.

- [ ] **Step 2: RED → implement** — three records (TOGETHER full code shown; fireworks = same + `reasoning_disposition="proprietary"` + fixture allowances; cerebras = same + fixture allowances):

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
    response_allowances=<from fixture inventory>,  # e.g. frozenset({"prompt"})
)
```

`ALL_RECORDS` += three; dispatch entries `"together": build_hosted_chat_handler(TOGETHER)` etc.; config `[providers]` empty seeds + `[api_settings.<key>]` tables with `api_base_url` + settings_defaults (NO `model`); readiness/display/native-tools literals; catalog auto-refresh config-key entries. `_STRICT_HOSTED_PROVIDER_KEYS` picks them up automatically (Task 1 derivation).

- [ ] **Step 3: GREEN + regression** — `.venv/bin/python -m pytest Tests/LLM_Calls/test_inference_cloud_presets.py Tests/test_provider_registry.py Tests/Chat/test_dispatch_registry_parity.py Tests/test_config_databricks.py Tests/test_config_model_catalog_defaults.py Tests/LLM_Provider_Catalog/ Tests/LLM_Calls/test_longtail_fixture_characterization.py -q` — PASS.

- [ ] **Step 4: Commit** — `feat: together/fireworks/cerebras presets with fixture-derived allowances (ADR-179)`

---

### Task 6: Custom-endpoint engine swap — at the gateway identity site

**Files:**
- Modify: `tldw_chatbook/provider_registry.py` (`CUSTOM_HOSTED` record)
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (~L3968 identity site; context-window lookups ~L3734/3784; `_thinking_stream_capability` set)
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (dispatch + `CUSTOM_PROVIDER_PARAM_MAP`)
- Modify: `tldw_chatbook/LLM_Calls/hosted_provider_engine.py` (closure param surface for the custom family)
- Modify: `tldw_chatbook/Chat/provider_continuation.py` (Literal + `_PAIRINGS`)
- Modify: `Agents/native_tools.py` (`"custom-hosted"` added)
- Test: `Tests/Chat/test_custom_endpoint_engine_swap.py` (new); extend `Tests/Chat/test_custom_endpoint_registry.py` expectations only where they asserted execution internals

**Interfaces:**
- Produces:
  - `family_execution_key()` **unchanged** (still returns `"custom"` for `openai_compatible`). At the gateway's identity-resolution site (~L3968), when the resolved entry's family is `openai_compatible`, the executor slot is set via `dataclasses.replace(identity, execution_key="custom-hosted")` — readiness key, display identity, saved-session identity, and `canonical_connection_identity` all keep the `"custom"`-based values they have today.
  - `API_CALL_HANDLERS["custom-hosted"] = build_hosted_chat_handler(CUSTOM_HOSTED)`; `PROVIDER_PARAM_MAP["custom-hosted"] = CUSTOM_PROVIDER_PARAM_MAP` (a copy of the legacy custom map at `Chat_Functions.py:596` — the FULL surface: `minp, topk, seed, n, presence_penalty, frequency_penalty, logit_bias, logprobs, top_logprobs, thinking_budget_tokens` beyond the engine map).
  - Engine closure accepts those kwargs, gated by `payload_flags` (records without the flags keep the Phase 1 raise-on-supplied behavior; `CUSTOM_HOSTED.payload_flags` includes them, with validators ported to strict form: bounded floats for penalties/min_p/top_k, shape-checked `logit_bias` dict, `logprobs` bool + `top_logprobs` bounded int, `thinking_budget_tokens` positive int, `seed`/`n` ints).
  - `CUSTOM_HOSTED`: `key="custom-hosted"`, `config_key="Custom-hosted"` (execution-only; no `[providers]`/`[api_settings]` table), `classification="local"`, `engine_driven=True`, `default_base_url=None`, `auth_scheme="bearer_optional"`, `tolerant_response_extras=True`, `native_tools=True`, `auto_refresh=False`, no env candidates. **Timeout/retry defaults come from the legacy slot**: resolution reads `api_settings.custom`'s `api_timeout`/`api_retries`/`api_retry_delay`/`streaming` when present (fallback 120/1/1/True) so users who tuned those settings keep them — pin with a test.
  - Continuation: `ContinuationProvider` Literal + `_PAIRINGS` gain `"custom-hosted"`/`("custom-hosted", "chat_completions")`.
  - Context-window lookups (~L3734/3784): verify what they do for the new key; alias `custom-hosted` to the same entry `"custom"` resolves to (additive, no behavior change for existing keys).

- [ ] **Step 1: Failing tests**

```python
# Tests/Chat/test_custom_endpoint_engine_swap.py
"""openai_compatible registry entries execute through the strict engine,
swapped at the gateway identity site only."""
import dataclasses

import pytest


def test_family_execution_key_unchanged():
    from tldw_chatbook.Chat.custom_endpoint_registry import family_execution_key
    assert family_execution_key("openai_compatible") == "custom"   # UNCHANGED
    assert family_execution_key("llama_cpp") == "llama_cpp"
    assert family_execution_key("ollama") == "ollama"


def test_gateway_swaps_execution_key_only_for_openai_compatible():
    # Build (or mock) the gateway identity resolution for a custom-ep entry
    # with family openai_compatible: assert resolved.execution_key ==
    # "custom-hosted" AND resolved.readiness_key / selected_provider /
    # canonical identity equal what they were before the swap (pin the
    # pre-swap values by constructing the same entry against the legacy
    # execution path or by asserting the concrete expected strings).
    ...


def test_saved_session_with_custom_ep_slug_still_ready_and_identity_stable():
    """The regression that would have caught the family_execution_key bug:
    a session saved with provider="custom-ep:<slug>" still resolves ready,
    and its canonical connection identity is byte-identical before/after
    the swap (construct the identity for the same entry via the un-swapped
    code path constants and compare)."""
    ...


def test_custom_hosted_record_shape():
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY
    record = RECORDS_BY_KEY["custom-hosted"]
    assert record.auth_scheme == "bearer_optional"
    assert record.tolerant_response_extras is True
    assert record.default_base_url is None
    assert record.auto_refresh is False


def test_custom_param_surface_is_the_legacy_one():
    from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP
    legacy = PROVIDER_PARAM_MAP["custom-openai-api"]
    ours = PROVIDER_PARAM_MAP["custom-hosted"]
    assert set(ours) == set(legacy)  # no parameter silently dropped
    for key, value in legacy.items():
        assert ours.get(key) == value


def test_engine_closure_accepts_custom_family_params():
    # build_hosted_chat_handler(CUSTOM_HOSTED) closure: inspect.signature
    # contains minp, topk, seed, n, presence_penalty, frequency_penalty,
    # logit_bias, logprobs, top_logprobs, thinking_budget_tokens; a canned
    # transport (reuse the Task 3 / test_hosted_chat fake; NON-streaming
    # only — no unused streaming branch in the fake) round-trips one call
    # with those kwargs and the captured payload contains each field.
    ...


def test_custom_timeouts_come_from_legacy_settings():
    # resolution with api_settings.custom = {api_timeout: 45, api_retries: 2,
    # api_retry_delay: 3, streaming: false} -> transport config carries
    # 45/2/3/False; absent table -> record defaults 120/1/1/True.
    ...


def test_named_custom_slots_untouched():
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
        chat_with_custom_openai, chat_with_custom_openai_2,
    )
    assert API_CALL_HANDLERS["custom-openai-api"] is chat_with_custom_openai
    assert API_CALL_HANDLERS["custom-openai-api-2"] is chat_with_custom_openai_2
```

- [ ] **Step 2: RED → implement** — registry record; gateway `replace(identity, execution_key=...)` at the openai_compatible branch of the ~L3968 site; capability sets keyed on execution key (`native_tools`, `_thinking_stream_capability`, continuation Literal/pairings, context-window alias); dispatch + CUSTOM map; engine closure extension + validators.

- [ ] **Step 3: GREEN + regression** — `.venv/bin/python -m pytest Tests/Chat/test_custom_endpoint_engine_swap.py Tests/Chat/test_custom_endpoint_registry.py Tests/Chat/test_dispatch_registry_parity.py Tests/test_provider_registry.py -q`, then every Console suite that greps to `custom-ep:` — failure sets must equal documented pre-existing baselines (stash-verify anything new).

- [ ] **Step 4: Commit** — `feat: custom-endpoint openai_compatible family executes via the strict engine, swapped at the gateway identity site (ADR-179)`

---

### Task 7: Live probes, docs, battery, close-out

**Files:**
- Create: `Tests/Chat/test_live_inference_cloud_api.py`
- Modify: `Docs/User_Guide/settings.md`, `Docs/User_Guide/console.md`, `README.md`, backlog task

- [ ] **Step 1: Live probes** — one module, three providers (Together/Fireworks/Cerebras), double-gated (`TLDW_LIVE_TOGETHER=1` + `TOGETHER_API_KEY`, etc.), Phase 1's `test_live_databricks_api.py` subprocess pattern (isolated profile; structural-metadata-only stdout: models count, model id, response key names). Structural gate/isolation tests run always; paid probes skip-clean. If Task 2 lacked a key for a provider, its first run here records the envelope and reconciles `response_allowances` (amend the record if reality differs — never silent).
- [ ] **Step 2: Docs** — Settings guide subsection "Inference clouds (Together, Fireworks, Cerebras)" (table: base/env var; discovery-first model lists; Fireworks private-reasoning note). Console guide: short subsection. README: three env vars. Custom-endpoints section: the family now runs the strict engine (shape-safe unknown fields ignored; malformed core shapes fail closed; keyless servers keep working; tuned `api_settings.custom` timeouts still apply). No xAI, no Perplexity.
- [ ] **Step 3: Full targeted battery** — every suite named in Tasks 1-6 plus `Tests/Chat/test_sensitive_llm_logging.py Tests/Chat/test_chat_unit_mocked_APIs.py`; all green (document any pre-existing deviations with stash-verified evidence).
- [ ] **Step 4: Backlog close + commit** — check demonstrable ACs; probes recorded pending where keys are absent. `docs: inference-cloud presets + strict custom endpoints (ADR-179 phase 2)`

---

## Self-Review (completed)

**Review-blocker coverage:** (1) swap moved to the gateway identity site with `family_execution_key` pinned unchanged + saved-session regression + context-window aliasing; (2) tolerance widened to null-valued choice/message keys + empty-text finishes + fixture gate; (3) fixture capture precedes presets and allowances derive from it; (4) Perplexity dropped with verified facts (sunset 2026-09-27; no client-side tools — recorded in the spec); (5) `CUSTOM_PROVIDER_PARAM_MAP` = the legacy surface + closure extension + `api_settings.custom` timeout sourcing. Task 1: registry-derived key list, guard and its broken test deleted, alias pin reframed, per-criterion `--ac`. Test-code bugs fixed: fakes reused from `test_hosted_chat.py` with `self`-aware signatures, `Path(__file__)` anchoring, no dead streaming branch, no comma-joined `--ac`.

**Order matches the reviewer's:** rebase note in Global Constraints → fixtures (Task 2) → auth + widening (Tasks 3-4) → presets from fixtures (Task 5) → gateway-site swap (Task 6).

**Type consistency:** `bearer_optional`, `tolerant_response_extras`, `CUSTOM_PROVIDER_PARAM_MAP`, `custom-hosted` spellings used identically across tasks; `_STRICT_HOSTED_PROVIDER_KEYS` derivation defined in Task 1, consumed implicitly by Task 5 records.
