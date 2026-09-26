# Generic Hosted Provider Engine — Phase 2 (Inference-Cloud Presets + Long-Tail Engine Swap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Together, Fireworks, and Cerebras as engine preset records (level-keyed allowances derived from captured real-server fixtures), and move the ADR-146 custom-endpoint `openai_compatible` family onto the strict engine — swapped at the gateway's identity-resolution site, with base-URL/credential forwarding, reasoning parity, full settings fallbacks, and a config kill switch.

**Architecture:** Phase 1 (branch `feat/provider-engine-phase1`, ADR-179) built `provider_registry.py` and `LLM_Calls/hosted_provider_engine.py` on `hosted_chat`. Phase 2 lands evidence-first: **capture fixtures first** (Task 2), widen tolerance to what fixtures prove (Tasks 3-4), land presets with fixture-derived allowances (Task 5), then swap the custom family at the gateway identity site — every gateway surface keyed on the custom family moves to one shared `CUSTOM_OPENAI_EXECUTION_KEYS` constant so the new key inherits base-URL forwarding, credential forwarding, thinking/reasoning paths, and trace finals (Task 6).

**Tech Stack:** Python ≥3.12, `requests` via `hosted_chat`. Local fixture servers: llama-server, Ollama (vLLM expected-skip on macOS).

**Spec:** `Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md` (Summary item 3 Perplexity ruling, "Response variance and strictness" with level-keyed allowances, Phase 2 ACs — amended 2026-09-24). Read before starting.

## Global Constraints

- Registry module stays stdlib-only (no `tldw_chatbook.*` imports).
- **Evidence-gated, in order**: fixtures (Task 2) precede widening (Task 4), presets (Task 5), and the swap (Task 6). Each consumes the fixture inventory.
- Fail-closed strictness: required-shape validation never relaxes on any profile. Tolerance is exactly: shape-safe unknown top-level/event keys, null-valued unknown choice/message keys, the two allowlisted non-null choice keys (`logprobs` object-or-null, `stop_reason` scalar), and empty-text `stop`/`length` finishes. Curated records stay strict outside their **level-keyed** allowances; moonshot/zai behavior pinned byte-identical under the level split.
- `moonshot.py`/`zai.py`, the named `custom-openai-api` handlers, `family_execution_key()`, and every identity/readiness consumer of `"custom"` are unchanged. The swap is at the gateway identity site; every execution-key-keyed surface the new key must flow through moves to `CUSTOM_OPENAI_EXECUTION_KEYS`.
- **Excluded**: xAI/Grok (ADR-179); Perplexity in Phase 2 (Sonar Chat Completions sunsets 2026-09-27; spec O-4).
- No shipped `model` key for discovery-first presets; API keys env-first, masked otherwise, never logged (ADR-012).
- Tests: `.venv/bin/python -m pytest` from the worktree root, targeted only. `backlog task create --ac` takes one criterion per flag. Live probes env-gated/skip-clean and don't block Done; **fixture capture is a hard prerequisite**.
- Pre-work: once PR #2824 merges, rebase `feat/provider-engine-phase1` onto `origin/dev` (duplicate baseline commit drops automatically).

## Provider facts (verified 2026-09-24)

| Provider | key | Base URL | Env var | Notes |
|---|---|---|---|---|
| Together | `together` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | free tier; memory-to-verify: top-level `prompt`, choice `logprobs` |
| Fireworks | `fireworks` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | R1-family `reasoning_content` → `proprietary` |
| Cerebras | `cerebras` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | free tier; memory-to-verify: top-level `time_info` |
| ~~Perplexity~~ | — | — | — | dropped (sunset 2026-09-27; spec O-4) |

---

### Task 1: Backlog task + registry pins (no RED theater)

**Files:**
- Create: backlog task via CLI
- Modify: `Tests/LLM_Calls/test_hosted_provider_engine_handler.py`, `Tests/LLM_Provider_Catalog/test_engine_branch_guard.py`

**Interfaces:** none new. Both tests below are **pins** — both are expected green the moment they land (the derived set equals today's hard-coded `{"moonshot","zai","databricks"}` because Databricks is the only engine-driven key until Task 5; the alias already mirrors the private signature). They exist to make Task 5's additions automatic and unremarkable.

- [ ] **Step 1: Create the backlog task** (one `--ac` per flag):

```bash
backlog task create "Generic hosted provider engine Phase 2: inference-cloud presets + long-tail engine swap" \
  -d "Implement Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md Phase 2 (as amended 2026-09-24)" \
  -a @Robert -s "In Progress" \
  --ac "Real-server fixtures (plain+tools+streamed rounds; cloud /models) captured before swap/presets" \
  --ac "Together/Fireworks/Cerebras selectable+usable with fixture-derived level-keyed allowances" \
  --ac "Preset-cost test proves no per-provider module; moonshot/zai allowances byte-identical" \
  --ac "custom-ep family executes via engine: gateway-site swap, shared CUSTOM_OPENAI_EXECUTION_KEYS, base-URL+credential forwarding, saved sessions unchanged" \
  --ac "Reasoning parity and api_settings.custom fallbacks carry over; kill switch ships" \
  --ac "Keyless entries work; curated presets hard-require keys" \
  --ac "Tolerant profile fixture-gated and scoped" \
  --ac "Docs updated"
backlog task edit <id> --plan "Plan: Docs/superpowers/plans/2026-09-24-generic-hosted-provider-engine-phase2.md"
```

- [ ] **Step 2: Add the two pins**

In `Tests/LLM_Calls/test_hosted_provider_engine_handler.py`:

```python
def test_resolve_hosted_engine_request_alias_matches_private_signature():
    """PIN (green on arrival): public alias forwards the full private surface."""
    import inspect
    from tldw_chatbook.LLM_Calls import hosted_provider_engine as engine

    assert list(inspect.signature(engine.resolve_hosted_engine_request).parameters) == (
        list(inspect.signature(engine.resolve_hosted_request).parameters)
    )
```

Replace `Tests/LLM_Provider_Catalog/test_engine_branch_guard.py` with the derivation pin (and apply the derivation itself — replacing the hard-coded `_STRICT_HOSTED_PROVIDER_KEYS` at `local_llm_provider_catalog_service.py:~85` with `frozenset({"moonshot", "zai"}) | {key for key, record in RECORDS_BY_KEY.items() if record.engine_driven}`, importing `RECORDS_BY_KEY` in the service's existing style):

```python
"""PIN: the strict-hosted key list derives from the provider registry."""


def test_strict_hosted_keys_derive_from_registry():
    from tldw_chatbook.LLM_Provider_Catalog import (
        local_llm_provider_catalog_service as service,
    )
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    expected = {"moonshot", "zai"} | {
        key for key, record in RECORDS_BY_KEY.items() if record.engine_driven
    }
    assert service._STRICT_HOSTED_PROVIDER_KEYS == expected
    assert "databricks" in service._STRICT_HOSTED_PROVIDER_KEYS
    assert "openai" not in service._STRICT_HOSTED_PROVIDER_KEYS
```

- [ ] **Step 3: Verify green + regression** — `.venv/bin/python -m pytest Tests/LLM_Provider_Catalog/ Tests/LLM_Calls/test_hosted_provider_engine_handler.py -q` — PASS (this is the pin's point: the derivation changes nothing today; Task 5's records extend it for free).

- [ ] **Step 4: Commit** — `refactor: derive strict-hosted key list from registry; pin alias parity (ADR-179)`

---

### Task 2: Capture real-server fixtures (the evidence gate)

**Files:**
- Create: `Tests/fixtures/longtail/` (fixtures + `CAPTURE.md` + `capture_local.py` + `capture_cloud.py`)
- Test: `Tests/LLM_Calls/test_longtail_fixture_characterization.py`

**Interfaces:**
- Produces, per server: `{"server", "base_url", "chat_response", "tool_call_response", "stream_events", "models_response", "captured_at", "capture_cmd"}` — full raw bodies, **never normalized**; `capture_cmd` is a sanitized template (the credential appears as the literal placeholder `<key>`; the scripts must never write the real key or an `Authorization` header into any stored field — the request code builds headers at call time only).
- Produces the **inventory test**: per server, the sets of unknown keys at each closed level (top, choice, message; and event/choice/delta for streams), **computed by set difference against the allowlists** — never parsed from exception text (the exceptions don't name keys), with an empty set being a valid, expected outcome (Ollama may parse clean).

- [ ] **Step 1: Capture scripts** — three rounds per server, not one:
  1. plain chat (tiny prompt);
  2. **tool-call round**: one function tool + `tool_choice` (captures tool-call shape and finish consistency — the strictest checks);
  3. **streamed** chat (full ordered SSE `data` payloads).
  Plus, for cloud captures only: **`GET /models`** (records each preset's discovery route working — Fireworks' `/inference/v1` included).

  `capture_local.py`: llama-server (`llama-server -m <tiny.gguf> --port 8901`), Ollama (`ollama serve` + pull a 0.5b model) — skip-with-note per server when binaries/models are absent. vLLM: **expected skip on macOS** (`vllm/vllm-openui` is a CUDA image); attempt only if a CUDA runtime demonstrably exists, otherwise record the skip — and `CAPTURE.md` then states explicitly that any vLLM-specific key (`stop_reason`) is **unverified memory, not evidence** until a Linux capture lands. LM Studio (GUI) unscripted; llama-server represents the family.

  `capture_cloud.py`: env-gated (`TOGETHER_API_KEY`/`CEREBRAS_API_KEY`/`FIREWORKS_API_KEY`), same three rounds + models GET per provider.

- [ ] **Step 2: Run captures, commit fixtures + CAPTURE.md** (date, versions, skips-with-reasons, replay command; the vLLM memory-vs-evidence note when skipped).

- [ ] **Step 3: Inventory test** — computes per-server inventories by difference and asserts them as pinned expected data:

```python
# Tests/LLM_Calls/test_longtail_fixture_characterization.py
"""Per-server inventory of unknown keys at each closed allowlist level,
computed by set difference from captured fixtures. Empty sets are valid.
Tasks 4/6 must drive every long-tail inventory to acceptance under the
custom profile, and cloud inventories into preset allowances — the SAME
fixtures replayed through BOTH parsers (non-streaming body AND the
ordered stream events) once the widening lands."""
```

Implementation shape: for each fixture, `set(body) - KNOWN_TOP`, `set(choice) - KNOWN_CHOICE`, `set(message) - KNOWN_MESSAGE` per choice; for streams, the same at event/choice/delta levels across `stream_events`; expected values pinned as literal sets per server (write them from the first run — the pinned sets ARE the committed evidence). Include a replay helper that will be extended in Task 4 to assert acceptance.

- [ ] **Step 4: Commit** — `test+fixtures: capture real long-tail envelopes (plain/tools/streamed + models); pin unknown-key inventory (ADR-179)`

---

### Task 3: Engine `bearer_optional` auth

(Unchanged from the prior revision — summarized; write the full TDD cycle at execution.)

**Files:** `hosted_provider_engine.py` (`_resolve_api_key`, transport), `hosted_chat.py` (~L517-565), `provider_registry.py` (comment), new `Tests/LLM_Calls/test_hosted_provider_engine_auth.py`.

**Binding interface:** `auth_scheme ∈ {"bearer", "bearer_optional", "api_key_header"}`; missing key raises only for `"bearer"`; `"bearer_optional"` resolves with `api_key == ""` when nothing resolves; **explicit `""` is "no key", not invalid**; **skip the env-var-name validity check when no env-var name is set** (today a `None` name raises before candidates are tried — that bug is why the task's `_OPTIONAL` record would fail without it); `owned_json_post` omits `Authorization` entirely for an empty key (never an empty `Bearer`). Tests reuse the proven `Session`/response fakes from `Tests/LLM_Calls/test_hosted_chat.py` (fake `post(self, url, **kwargs)` — read that file first). Default `bearer` byte-identical (zai/moonshot/hosted_chat regression set stays green). RED → implement → GREEN → commit `feat: engine bearer_optional auth…`.

---

### Task 4: Level-keyed allowances + fixture-gated tolerant profile

**Files:**
- Modify: `hosted_chat.py` (choice/message checks; stream event/choice/delta), `provider_registry.py` (**`choice_allowances: frozenset[str] = frozenset()`, `message_allowances: frozenset[str] = frozenset()`** — `response_allowances` keeps its Phase 1 top/event meaning, unchanged), `hosted_provider_engine.py` (plumb-through; tolerant finish branch)
- Test: extend `Tests/LLM_Calls/test_hosted_chat_allowances.py`; flip Task 2's replay

**Interfaces:**
- Curated allowances are **level-scoped by field**: `response_allowances` subtracts at top/event only (moonshot/zai semantics **pinned byte-identical** — their existing sets provably cannot widen levels); `choice_allowances`/`message_allowances` subtract at their levels, values validated as null / scalar / shape-safe mapping and dropped.
- `tolerant_response_extras=True` (custom family only): shape-safe unknown top/event keys dropped; null-valued unknown choice/message keys dropped; unknown non-null choice/message keys fail closed **unless in that record's choice/message allowances**; engine finish policy accepts `stop`/`length` with empty text and no calls (legacy empty reply) — `stop`-with-calls and `tool_calls`-without-calls still fail.
- **CUSTOM_HOSTED's own allowances ship here** (they're the two fixture-known non-null cases): `choice_allowances=frozenset({"logprobs", "stop_reason"})` — `logprobs` arrives as an object when the caller requested logprobs (Task 6 forwards that parameter), `stop_reason` as the matched stop string/token id.

**Tests (minimal matrix per prior review + the flips):**
1. moonshot/zai byte-identity: a body with an unknown CHOICE key fails for them even though their top-level allowances exist (level split proof).
2. Curated `choice_allowances` accept `logprobs` object-or-null / `stop_reason` scalar; without the allowance, both fail closed (one non-streaming + one stream replay).
3. Tolerant: null-valued unknowns accepted; non-null unknown NOT in allowances rejected; empty-text stop/length → empty turn; stop-with-calls still raises.
4. **Flip Task 2**: every long-tail fixture body AND stream replay parses under the custom profile (tolerant + its allowances); cloud fixtures parse under allowances where Task 5 records them (that flip lands with Task 5).

RED → implement → GREEN (incl. re-pinning the Phase 1 test that rejected `logprobs: null` at choice level — now scoped to non-allowlisted, non-tolerant records) → commit `feat: level-keyed allowances + fixture-gated long-tail tolerant profile (ADR-179)`.

---

### Task 5: Three inference-cloud presets (allowances from the fixture inventory)

**Files:** `provider_registry.py` (three records + ALL_RECORDS), `Chat_Functions.py` (three dispatch entries), `config.py` (empty `[providers]` seeds + three `[api_settings.*]` tables with `api_base_url`, no `model`), `provider_readiness.py`, `console_provider_support.py`, `native_tools.py`, `model_catalog_settings.py` (config-key spellings), `openai_compatible_model_discovery.py` (only if `/inference/v1` fails the gate — `/openai/v1` precedent).

**Interfaces:** keys `together`/`fireworks`/`cerebras`; each record's `response_allowances`/`choice_allowances`/`message_allowances` are **exactly the level-matched unknown-key sets its captured fixture inventory pinned** (Together's expected-from-memory `prompt`+choice `logprobs` become real only if the fixture shows them); `_STRICT_HOSTED_PROVIDER_KEYS` extends automatically (Task 1 pin).

**Tests:** `Tests/LLM_Calls/test_inference_cloud_presets.py` — registration/record shape (bearer, non-tolerant, auto_refresh, native_tools, no suffix); preset-cost (`Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"` anchor, no `<key>*.py`); default URLs/env vars; **allowances == fixture inventory** (literal expected sets per provider); fireworks `proprietary` disposition; **cloud fixtures parse under each record** (the Task 2 cloud flip). Parity extensions in `Tests/test_provider_registry.py` with no exclusions. RED → implement → GREEN + registry/config/catalog regression → commit `feat: together/fireworks/cerebras presets with fixture-derived allowances (ADR-179)`.

---

### Task 6: Custom-endpoint engine swap — gateway site + the full custom-family surface

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` — the identity site (~L3968), `_CUSTOM_CREDENTIAL_DECISION_PROVIDERS` (:231, used :5202/6818/6902), the base-URL forwarding sets (:6785-6819), `_finish_policy_for` (pin only)
- Modify: `tldw_chatbook/Chat/console_provider_support.py` — `_CUSTOM_OPENAI_THINKING_KEYS` (:118, inside `_LOCAL_REASONING_EXECUTION_KEYS`)
- Modify: `tldw_chatbook/Chat/console_trace_final_values.py` (:764/772)
- Audit-and-decide (listed below): `console_session_settings.py:98`, `provider_endpoint_contract.py:48`, `provider_setup_persistence.py:150`, `model_discovery_provider_identity.py:18`, `console_provider_picker.py:218`, `native_tools.py:44`
- Modify: `provider_registry.py` (`CUSTOM_HOSTED`), `Chat_Functions.py` (dispatch + `CUSTOM_PROVIDER_PARAM_MAP`), `hosted_provider_engine.py` (closure surface + ADR-066 reasoning composition + settings-section defaults), `provider_continuation.py` (Literal + `_PAIRINGS`), `config.py` (`[console] custom_endpoints_use_engine = true`)
- Test: `Tests/Chat/test_custom_endpoint_engine_swap.py` + new literal-grep test

**Binding decisions (from review):**

1. **One shared constant**: define `CUSTOM_OPENAI_EXECUTION_KEYS = frozenset({"custom-openai-api", "custom-openai-api-2", "custom-hosted"})` in `console_provider_support.py` (or the lowest module the sites can import — not the registry, which stays stdlib-only). Replace each literal set at the sites above with it. Per-site decisions:
   - gateway:231 credential-decision providers → **include** (without it `api_key_resolved` is never sent).
   - gateway:6785-6819 base-URL forwarding (the local-reasoning branch, moonshot/zai, and the explicit custom/mistral/vllm set) → **include custom-hosted** in the custom-family membership — this is Blocker 1: without it every custom-endpoint send dies with "base URL is required" (`CUSTOM_HOSTED` has no settings table and `default_base_url=None`; the per-entry URL lives only in the gateway resolution).
   - support:118 thinking keys (→ `_LOCAL_REASONING_EXECUTION_KEYS`) → **include** (preserves the local-reasoning path and `chat_template_kwargs` replay).
   - trace:764/772 → **include** (trace final values keep the base URL).
   - The six audit sites → check each: include where keyed on execution key and the custom family belongs; leave where keyed on identity (which still says `custom`/`custom-openai-api` via `family_execution_key`). Record the per-site decision in the task report.
2. **Literal-grep test**: scan `tldw_chatbook/` for the bare strings `"custom-openai-api"` / `"custom-openai-api-2"` outside the constant's defining module (and tests) — fail on any hit, so new sites can't slip past the constant.
3. **Reasoning parity (concrete bug)**: identity resolution still reports reasoning effort supported for custom-ep (support.py:320 → `custom`/`custom-openai-api`), so the engine must honor it exactly as legacy did — `CUSTOM_HOSTED` gains the reasoning flags, and the engine closure ports the legacy ADR-066 composition from `chat_with_custom_openai` (`reasoning_effort` verbatim where consumed; the llama-family `reasoning_budget_tokens` translation). Pin: a custom-ep send with a reasoning level produces the legacy-composed payload fields.
4. **Kill switch**: `[console] custom_endpoints_use_engine` (default `true`). The identity site reads it: `false` → leave `execution_key` as `"custom"` (the untouched legacy path — its base-URL/credential forwarding still works because those keys are unchanged); `true` → `dataclasses.replace(identity, execution_key="custom-hosted")`. Pin both directions.
5. **Full settings fallbacks, not just timeouts**: the legacy handler reads, per call from `api_settings.custom` (verify the exact key names in `chat_with_custom_openai`'s `cfg.get` calls — including the timeout spellings): `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens` (default **4096**), `seed`, `stop`, `response_format`, `streaming` (default **False**), timeouts/retries. Mechanism: record field `defaults_settings_section: str | None = None`; `CUSTOM_HOSTED` sets `"custom"`; engine resolution precedence becomes **explicit kwargs > that section > `settings_defaults`** (record defaults mirror legacy: `streaming=False`, `max_tokens=4096`). Pin: section-tuned values flow; absent section → record defaults; streaming defaults False.
6. `CUSTOM_PROVIDER_PARAM_MAP` = the legacy custom map surface (`Chat_Functions.py:596`): `set(ours) == set(legacy)`, values equal; engine closure accepts `minp, topk, seed, n, presence_penalty, frequency_penalty, logit_bias, logprobs, top_logprobs, thinking_budget_tokens` + reasoning composition, gated by `payload_flags` (records without flags keep the raise-on-supplied behavior), strict validators.
7. `_finish_policy_for` (gateway ~276) auto-attaches a `HostedPresetFinishPolicy` to every engine-driven key — custom-hosted inherits the hosted thinking round-trip machinery. **Pin** that this is the case (a test asserting the resolver returns the engine policy for `custom-hosted`) — intended, now guarded.
8. **Dropped from the prior revision**: the context-window alias (~3734/3784) — those lookups flow through `family_execution_key()`, which never returns `custom-hosted`; nothing to do.
9. `family_execution_key()` pinned unchanged (`== "custom"` for `openai_compatible`); saved-session regression (provider=`custom-ep:<slug>` ready, canonical identity byte-identical); named slots untouched; continuation Literal + `_PAIRINGS` gain `custom-hosted`; keyless/keyed canned-transport test (reuse Task 3's fake; **non-streaming only** — no dead streaming branch).

**Cycle:** failing tests (all of the above as concrete assertions) → implement → GREEN + custom-ep Console suites at documented baselines → commit `feat: custom-endpoint family via the strict engine — gateway-site swap, shared custom keys, parity + kill switch (ADR-179)`.

---

### Task 7: Live probes, docs, battery, close-out

- **Probes**: `Tests/Chat/test_live_inference_cloud_api.py` — three providers, double-gated, Phase 1's subprocess pattern; models count/id + response key names as the only stdout. First runs reconcile each record's allowances against reality (amend, never silent).
- **Docs**: Settings subsection "Inference clouds (Together, Fireworks, Cerebras)" (base/env table; discovery-first; Fireworks private-reasoning note). Console guide short subsection. README env vars. Custom-endpoints section: strict engine swap (tolerant profile semantics; keyless unchanged; **`api_settings.custom` fallbacks preserved including `streaming=false` default**; `[console] custom_endpoints_use_engine` kill switch documented). No xAI, no Perplexity.
- **Battery**: every suite named in Tasks 1-6 + `Tests/Chat/test_sensitive_llm_logging.py Tests/Chat/test_chat_unit_mocked_APIs.py`; green or stash-verified pre-existing.
- **Backlog close + commit** `docs: inference-cloud presets + strict custom endpoints (ADR-179 phase 2)`.

---

## Self-Review (completed)

**Review coverage:** Blocker 1 → base-URL forwarding via the shared constant (Task 6.1, with the failure mode named). Blocker 2 → full site table with per-site decisions + literal-grep test + reasoning-parity fix (6.1-6.3) + kill switch (6.4). Gap 3 → logprobs/stop_reason as explicit non-null choice allowances on CUSTOM_HOSTED (Task 4) — spec amended to match. Gap 4 → allowances level-keyed by field; moonshot/zai byte-identity pinned (Task 4.1); decided, not deferred. Gap 5 → inventory by set difference, empty allowed, streams replayed (Task 2.3). Gap 6 → tool rounds + models GET + sanitized capture_cmd + vLLM expected-skip/memory labeling (Task 2.1-2.2). Gap 7 → full `api_settings.custom` fallback port incl. `streaming=False` and `max_tokens=4096`, key names verified against the handler (Task 6.5). Minors → Task 1 reframed as pins; context-window alias dropped; `_finish_policy_for` auto-attach pinned (6.7).

**Order unchanged and correct:** fixtures → auth/tolerance → presets → swap, with Task 6 now carrying the site table, the forwarding fix, and the kill switch — the plan's riskiest task is its most specified one.
