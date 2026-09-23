# Generic Hosted Provider Engine and Preset Registry — Design

- **Date:** 2026-09-23
- **Status:** Draft (pending user review)
- **Scope:** Cloud LLM provider support expansion, starting with Databricks, moving
  towards Hermes-style provider breadth (any OpenAI-compatible endpoint usable)
  without adopting Hermes's zero-strictness model.
- **Governance:** requires new ADR-179 (see [ADR](#adr)).

```text
ADR required: yes
ADR path: backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md
Reason: new provider/runtime boundary (preset-driven adapter form), changes how
providers register across config/Console/settings surfaces, and records the
deliberate xAI exclusion and the Bedrock OpenAI-compat-first scope (native
Converse wire deferred as fallback).
```

## Summary

Today a first-class hosted provider costs a ~1,000-line strict adapter plus
edits to roughly twenty scattered literal tables (the Moonshot/ZAI addition
touched 39 files, +9,185/−1,052). That cost curve cannot reach Hermes-level
breadth. This design introduces:

1. **A provider identity registry** — one stdlib-only leaf module that becomes
   the single source of truth for the literal tables that classify, name, and
   wire providers across config, readiness, Console, settings, and the model
   catalog.
2. **A generic strict adapter engine** — parameterized by **preset records**
   (data, not code), absorbing the boilerplate currently duplicated between
   `zai.py` and `moonshot.py`, riding the existing `hosted_chat` wire boundary.
3. **Curated presets** — Databricks first; Together, Fireworks, Cerebras,
   Perplexity next; Azure OpenAI, Gemini's OpenAI-compatible layer, and AWS
   Bedrock (via Bedrock's own OpenAI-compatible endpoints, verified
   December 2025) after two small transport extensions.

Adding a curated provider drops from "20-file ritual" to "one registry record
+ one preset record + one registration line + tests + docs".

## Goals

- Databricks (AI Gateway / external models flavor) works as a first-class
  provider: Settings surface, readiness, Console chat with streaming and
  native tool calling, model auto-refresh, pricing, docs.
- Adding subsequent OpenAI-compatible hosted providers is a data change
  measured in tens of lines, not an adapter measured in hundreds.
- The long tail: any OpenAI-compatible endpoint is usable — via presets for
  curated names, via the evolved ADR-146 custom-endpoint registry for
  everything else.
- Strictness on the OpenAI Chat-Completions contract is preserved (fail-closed
  shape validation, redacted errors, bounded retries/SSE ownership from
  `hosted_chat`).

## Non-goals

- **xAI/Grok support — deliberately excluded** (maintainer decision). No
  preset, no adapter; recorded in ADR-179 so the omission is visibly
  intentional.
- Migrating `moonshot.py`/`zai.py` (or the legacy `LLM_API_Calls.py`
  providers) onto the engine — ADR-063's evidence-gated migration policy
  stands; migration is a possible follow-up task, not this design.
- Bedrock's **native wire** (Converse/InvokeModel) and **SigV4/IAM auth**:
  Bedrock is supported via its OpenAI-compatible endpoints + long-term
  Bearer API keys (AWS, December 2025) — a preset, not a Converse adapter.
  The native wire remains a documented fallback only if a required model
  isn't exposed through the OpenAI-compatible surface.
- Full enterprise identity auth (SigV4, Entra, service-account ADC).
  Key-based auth only; seams left open (Phase 4 candidates).
- Anthropic-wire or Google-native-wire providers; those remain hand-written
  adapters.

## Background and motivation

- Adding one provider today: ~1,000-line adapter (`LLM_Calls/zai.py`,
  `moonshot.py` are structurally near-identical — resolver helpers, message
  and tool normalization, continuation checkpointing are copy-paste), plus
  registration across `Chat/Chat_Functions.py` (`API_CALL_HANDLERS`,
  `SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS`, `PROVIDER_PARAM_MAP`),
  `config.py` (`[providers]`, `[api_settings.X]`, `_cloud_provider_keys`),
  `Chat/provider_readiness.py`, `Chat/console_provider_{support,endpoints}.py`,
  `Chat/console_session_settings.py`, `Chat/console_settings_defaults.py`,
  `Chat/provider_setup_persistence.py`, `Chat/provider_continuation.py`,
  `Agents/native_tools.py`, `model_capabilities.py`,
  `LLM_Calls/pricing_catalog.py`, `UI/Screens/settings_screen.py`,
  `LLM_Provider_Catalog/` (two files), tests, and docs.
- The scattered literals have already bitten: Moonshot/ZAI were silently
  misclassified as local providers until `e080a2fb92`.
- The loose generic tier (`chat_with_custom_openai`,
  `LLM_API_Calls_Local.py:1959`) shows decay typical of undisciplined shared
  paths: "Ollama" error strings in the custom provider, a historical
  crash-on-every-call bug, exactly two config slots.
- Hermes (Nous Research) achieves breadth by accepting *any* OpenAI-compatible
  endpoint with no per-provider code and no contract strictness. We want the
  breadth without giving up the strict wire boundary this repo built
  (ADR-062/063).

## Dialogue decisions (constraints from the maintainer)

1. Wishlist beyond Databricks: Together, Fireworks, Cerebras, Perplexity;
   Azure OpenAI, AWS Bedrock, GCP Vertex; plus a generic long-tail path.
2. **No xAI/Grok.**
3. Databricks flavor in use: **external models via AI Gateway** — plain
   OpenAI-compatible at `{workspace-host}/openai/v1`, Bearer PAT auth.
4. Enterprise clouds: **key-based auth now**, structured so identity-based
   auth can be added per provider later without rework.
5. **AWS Bedrock is in scope** (added at spec review): Bedrock exposes
   native OpenAI-compatible Chat Completions/Responses endpoints with
   long-term Bearer API keys (AWS announcement, December 2025) — so Bedrock
   is a preset on the engine, not a Converse strict adapter.

## Architecture

Three pieces. Dependency direction: `provider_registry` is a leaf (stdlib
only) importable by `config.py`; preset records are data in that same leaf
module, consumed by the engine and `config.py` alike; the engine lives in
`LLM_Calls/` beside `hosted_chat.py`.

### 1. `tldw_chatbook/provider_registry.py` — provider registry (new, leaf)

One stdlib-only module (no internal imports — `config.py` already imports 30
internal modules, so it must consume registry data without cycle risk) holding
one frozen dataclass record per provider. Built-in hand-written providers get
opaque identity records; engine-driven providers get a **preset-variant
record** carrying identity *and* wire data, so there is exactly one record
(not two overlapping ones) per provider:

Identity fields (every provider):

| Field | Purpose |
| --- | --- |
| `key` | Canonical provider id (`"databricks"`) used in dispatch and config |
| `config_key` | Display-cased `[providers]`/`[api_settings]` spelling (`"Databricks"`) |
| `display_name` | User-facing label (replaces `_PROVIDER_DISPLAY_NAMES` literals) |
| `classification` | `"cloud"` \| `"local"` — feeds `_cloud_provider_keys` |
| `api_key_env_var` + candidates | Feeds readiness key requirements (`DATABRICKS_TOKEN`, …) |
| `default_base_url` | May be `None` when per-account (Databricks workspace host) |
| `native_tools` | Feeds `Agents/native_tools.py::NATIVE_TOOLS_PROVIDERS` |
| `reasoning_effort` | Feeds reasoning-effort support sets in Console/settings |
| `auto_refresh` | Membership in `AUTO_REFRESH_PROVIDER_LIST_KEYS` |
| `settings_defaults` | Emits the `[api_settings.<key>]` default table (model, temperature, timeout, streaming, …) |
| `pricing_seeds` | Optional per-model $/Mtok entries for `pricing_catalog.py` — only for model ids unambiguously served by one provider; shared ids (e.g. Gemini via two providers) are omitted, not guessed |

Preset fields (engine-driven providers only — see §2):

| Field | Purpose |
| --- | --- |
| `base_url_rule` | Default URL + normalization (see data flow: append suffix only to a path-empty URL; never rewrite other paths) |
| `finish_reasons` | Allowed terminals; provider-terminal-error reasons → `ChatProviderError(502)` |
| `payload_flags` | temperature / top_p / stop / response_format (incl. `json_schema`) / `reasoning_effort` key name / extra body fields |
| `response_allowances` | Extra top-level/event keys tolerated (and ignored) during response/stream validation — the code-verified mechanism for real-world variance (`service_tier`, etc.); empty by default |
| `reasoning_disposition` | `"displayable" \| "proprietary" \| "ignored"` (per `HostedChatFinishPolicy`) |
| `auth_scheme` | `"bearer"` (Phase 1–2) \| `"api_key_header"` (Phase 3, Azure) \| `"none"` (long-tail family only, Phase 2) |
| `discovery` | Model-list route shape (`"openai_models"` default, or a custom path — Databricks O-1) |

Consumer sites stop hand-maintaining literals and import derived
frozensets/mappings (e.g. `CLOUD_PROVIDER_KEYS`,
`PROVIDERS_REQUIRING_API_KEYS`, display-name maps, builtin endpoint map). The
derived collections stay explicit at import time so the existing parity-test
regime keeps working; new parity tests assert **registry coverage**:

- `set(API_CALL_HANDLERS) ⊆ registry keys ∪ documented legacy aliases`
- `SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS ⊇ set(API_CALL_HANDLERS)` (the
  sensitive-request audit property survives — this is a hard requirement)
- `_cloud_provider_keys` ≡ registry cloud classification

`Chat_Functions.py`'s comment demanding explicit tables (L186–189) is honored
in spirit: the audit is still enforced by a parity test, now against the
registry instead of against hand-typed literals.

### 2. `LLM_Calls/hosted_provider_engine.py` (new)

`build_hosted_chat_handler(record) -> chat_with_*` returns a handler with the
same signature and param surface the dispatch expects (mirrors
`chat_with_zai`'s signature so `PROVIDER_PARAM_MAP` entries stay uniform —
engine providers share one param-map constant, exactly as `zai`/`moonshot`
each carry one today). The engine reads preset-variant records from the leaf
registry (§1) — behavior lives here, data lives in the leaf, and `config.py`
never imports from `LLM_Calls/`.

The engine absorbs, once, the boilerplate duplicated today: config/env
resolution (`_resolve_api_key`, `_resolve_base_url`, numeric/streaming
resolvers), message/tool/response-format normalization (lifted from
`zai.py`'s OpenAI-contract normalizers), finish-policy construction from
data, response/stream wrapping, and chat-completions continuation
checkpoints (`protocol="chat_completions"`, parameterized provider/model/
base URL). Error messages are prefixed with the preset display name. The
engine emits the LLM call/latency metric counters **centrally** — the thin
per-provider metric wrappers in `LLM_API_Calls.py` (the
`chat_with_moonshot`/`chat_with_zai` wrapper pattern) are not replicated per
preset.

**Escape hatch rule:** if a provider proves genuinely quirky, the answer is a
zai-style hand-written strict adapter — never a bespoke hack flag on a
preset. The engine stays honest by refusing to absorb unbounded weirdness.

### 3. `hosted_chat.py` extensions (small, existing file)

- `HostedHTTPTransportConfig` gains `auth_scheme` (default `"bearer"`;
  `"api_key_header"` sends `api-key: <key>` instead of `Authorization` —
  Azure in Phase 3). The hardcoded Bearer moves to scheme-selected
  construction at the `session.post` call site (`hosted_chat.py:561`).
- `"none"` auth (Phase 2, long-tail family only): today the transport
  rejects an empty API key outright (`hosted_chat.py:520-521`), but the
  ADR-146 registry legitimately includes keyless local servers. The engine
  passes `auth_scheme="none"` with an empty key; curated cloud presets keep
  `auth_scheme` requiring a key, and keyless stays forbidden for them.
- Response validation gains a **tolerated-extra-keys** input (fed from the
  preset's `response_allowances`): unknown keys outside a preset's allowance
  still fail closed; unknown keys inside it are validated as ignorable
  (shape-checked, then dropped). Required-shape validation (choices,
  message, tool calls, usage) is never relaxed.
- Phase 3 only: preset-controlled route/query shaping for Azure
  (`/openai/deployments/{model}/chat/completions?api-version=…`). The seam:
  `owned_json_post`'s `route` parameter becomes a fully-formed relative path
  plus optional query parameters supplied by the caller. Base-URL validation
  still rejects query strings in *configured* URLs; transport-level query
  params are constructed, never configured.

## Provider identity integration (surfaces the registry replaces)

Per the empirical new-provider checklist, the literal tables that become
registry-derived: `config.py` `[providers]` seeds, `[api_settings.X]`
default tables (emitted from each record's `settings_defaults`),
`_cloud_provider_keys`; `Chat/Chat_Functions.py` audit + param-map
constants; `Chat/provider_readiness.py` key sets; `Chat/console_provider_endpoints.py` `_BUILTIN_PROVIDER_ENDPOINTS` (Databricks
entry documents the `/openai/v1` suffix); `console_provider_support.py`
display names + reasoning-effort sets; `provider_setup_persistence.py`
canonical keys/aliases; `Agents/native_tools.py` `NATIVE_TOOLS_PROVIDERS`;
`Chat/provider_continuation.py` — the `ContinuationProvider` Literal
(`provider_continuation.py:26`) and the `_PAIRINGS` (provider, protocol)
frozenset both gain registry-driven members (the Literal itself stays a
one-line static edit, type-checked; `_PAIRINGS` derives from the registry);
`Chat/console_provider_gateway.py` `_HOSTED_THINKING_FINISH_POLICIES`
(`console_provider_gateway.py:236`) gains one generic entry covering every
engine provider (a shared finish-policy instance built from the preset) —
**built once in Phase 1**, not per provider; `LLM_Provider_Catalog/
model_catalog_settings.py` `AUTO_REFRESH_PROVIDER_LIST_KEYS`;
`local_llm_provider_catalog_service.py` strict-hosted resolution branch
(engine presets expose a `resolve_request`-shaped entry point exactly like
`resolve_zai_request`); `LLM_Calls/pricing_catalog.py` seeds.
`UI/Screens/settings_screen.py` provider pickers already derive from the
handler catalog — only display-name and reasoning-effort copy change.

### Residual per-provider touchpoints (honest accounting)

Some Console-machinery sites are behavioral branches, not key sets, and do
not become data-driven. Phase 1's job includes genericizing each **once** so
later presets don't touch them, but per-phase verification is still required:
`console_session_settings.py` and `console_settings_defaults.py` capability
predicates, `console_trace_final_values.py` reasoning/trace branches, and
the agent-bridge surfaces (`Agents/agent_service.py`,
`console_agent_bridge.py`). The Phase acceptance criterion for every later
preset is that these need **zero** provider-specific edits; where one
cannot be genericized, the deviation is recorded in that phase's plan
before implementation.

## Data flow (Databricks, Phase 1)

1. **Setup:** Settings → Databricks section (registry-generated): API key env
   var `DATABRICKS_TOKEN` (or masked stored key per ADR-012), `api_base_url`
   = workspace host or full `…/openai/v1` URL, default model, streaming.
   Readiness: requires key **and** base URL (no shipped default — workspace
   hosts are per-account); blocked-send recovery flow guides both.
2. **Send:** `chat_api_call` → `API_CALL_HANDLERS["databricks"]` (engine-built)
   → resolve from `[api_settings.databricks]`/env → normalize base URL:
   append `/openai/v1` **only** when the configured URL's path is empty or
   `/` (a pasted bare workspace host); any other path is validated as-is, and
   a terminal `/chat/completions` paste is rejected with actionable copy →
   payload per preset flags → `owned_json_post(route="chat/completions",
   Bearer PAT)` → strict normalization with preset finish policy →
   OpenAI-shaped response dict with terminal metadata + continuation
   checkpoints (identical consumer shape to `chat_with_zai` output).
3. **Model auto-refresh:** catalog service resolves Databricks via the
   engine's `resolve_request` entry point (strict-hosted branch), discovers
   models (see open item O-1), consent-gated write-through to `[providers]`
   per ADR-020. No shipped model seed — gateway model availability is
   workspace-dependent, so `[providers].Databricks` starts empty and fills
   via discovery or manual seeding.
4. **Cost ticker:** preset `pricing_seeds` where prices are known; unknown
   models are simply omitted (pricing is already optional).

## Error handling

- `Chat_Deps` taxonomy unchanged: 401/403 → `ChatAuthenticationError`
  ("check the API key"), 429 → `ChatRateLimitError`, other 4xx →
  `ChatBadRequestError`, transport/protocol → `ChatProviderError` with preset
  display name; retries honor `Retry-After` with exponential backoff
  (existing `hosted_chat` behavior).
- Missing key/base URL → `ChatConfigurationError` with the provider display
  name; readiness and blocked-send recovery already pattern-match these.
- Finish-policy data violations and malformed OpenAI-contract responses fail
  closed as 502 `ChatProviderError` (strict tier semantics preserved).

### Response variance and strictness (the load-bearing trade-off)

`hosted_chat` rejects unknown response/stream-event fields
(`normalize_hosted_chat_response`, `HostedChatStream._consume_event`). The
hand-written adapters survive this because each provider was live-verified;
real "OpenAI-compatible" servers do add benign fields (`service_tier`,
Azure's `prompt_filter_results`, vLLM/Together extras). Rules:

- **Curated presets:** unknown fields fail closed *unless* listed in the
  preset's `response_allowances`. Allowances are recorded from a live-probe
  envelope capture during the provider's phase — never guessed.
- **Long tail (Phase 2 custom-ep family):** a deliberately *tolerant*
  profile — unknown-but-shape-safe top-level/event keys are ignored;
  required-shape validation (choices, message, tool calls, usage) is never
  relaxed. This is a documented weakening, scoped only to user-registered
  endpoints, recorded in ADR-179.
- Allowances and the tolerant profile never bypass required-shape
  validation, output bounds, or redaction.

## Credentials and security (ADR-012 boundary intact)

- Keys resolve env-var-first (`DATABRICKS_TOKEN`), else masked
  `api_settings` value stored via Settings; never prefilled, never in
  diagnostics/logs (engine inherits `hosted_chat` redaction; preset records
  and registry records contain no secrets; `repr=False` patterns preserved).
- Engine-built handlers enter `SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS` through
  registry derivation — the sensitive auxiliary-request audit applies to
  them exactly as to hand-written handlers (parity-tested).

## Phases

### Phase 1 — Engine + Databricks (the proving milestone)

Build: `provider_registry.py` (including the Databricks preset-variant
record); `hosted_provider_engine.py`; registry-derived
tables rolled out consumer-by-consumer (each consumer swap is independently
testable); Databricks wired end-to-end.

Acceptance criteria:

- [ ] Databricks selectable and usable in the Console (non-streaming and
      streaming) with a real workspace via AI Gateway (live-verified per
      `backlog/docs/lessons-live-verification.md`: models list, one chat
      round-trip, one streamed round-trip with usage capture).
- [ ] Wire envelope captured and encoded: the live probe records the actual
      response/stream envelopes, and any fields beyond the `hosted_chat`
      allowlists are encoded in the Databricks `response_allowances` — no
      silent relaxations.
- [ ] Continuation registration: `ContinuationProvider`/`_PAIRINGS` cover
      `databricks`, and the gateway's finish-policy map has the single
      generic engine entry (no per-provider gateway code).
- [ ] Native tool calling works (tool round-trip with continuation
      checkpoint round-trip), or — if a gateway model lacks tool support —
      the capability flag is honestly off and Console hides tool affordances.
- [ ] Readiness blocks sends with actionable copy until key + workspace URL
      exist; blocked-send recovery works.
- [ ] Model auto-refresh discovers gateway models (O-1 settled by live
      probe) with consent gating per ADR-020.
- [ ] Registry coverage parity tests green (handlers ⊆ audited set; cloud
      classification correct — no repeat of `e080a2fb92`).
- [ ] Engine contract tests parameterized over presets pass; Databricks
      payload/finish-reason snapshots recorded.
- [ ] No behavior change for existing providers (targeted suites:
      `test_chat_unit_mocked_APIs.py`, `test_sensitive_llm_logging.py`,
      `test_provider_readiness.py`, catalog/config defaults tests).
- [ ] ADR-179 written and linked; User Guide (console/settings) + README
      provider list updated.

### Phase 2 — Preset cheapness + long tail

- [ ] Together, Fireworks, Cerebras, Perplexity presets: each is a record +
      tests + docs; a "preset cost" test asserts no provider-specific Python
      module is needed for a flag-clean OpenAI-compatible provider.
- [ ] ADR-146 `openai_compatible` family executes via the engine (strict
      tier) instead of `chat_with_custom_openai`; registry entries keep
      `custom-ep:<slug>` identity, cached models, and credential precedence
      (env → stored). Evidence-gated swap with parity tests against the old
      path.
- [ ] Keyless endpoints keep working: a `custom-ep` entry with no credential
      executes via `auth_scheme="none"`, and curated cloud presets still
      hard-require keys (parity-tested).
- [ ] The long-tail tolerant profile (see *Response variance and
      strictness*) is implemented and covered by tests: unknown-but-shape-safe
      fields ignored, required shapes still fail closed.

### Phase 3 — Enterprise, key-based

- [ ] `auth_scheme="api_key_header"` in `hosted_chat` transport.
- [ ] Azure OpenAI preset: deployment-shaped URLs (`/openai/deployments/
      {model}/chat/completions?api-version=…`), model discovery via
      deployment listing, key-based auth.
- [ ] Gemini OpenAI-compatible layer preset (OpenAI-compat endpoint with API
      key), distinct from the existing native Google adapter.
- [ ] AWS Bedrock preset on the engine: OpenAI-compatible endpoint with a
      region-configured base URL (per-account like Databricks — no shipped
      default), Bearer API-key auth, model discovery per O-3, response
      allowances from a live probe. Readiness copy notes Bedrock API-key
      expiry (long-term but time-limited keys) as a likely 401 cause. If a
      required model turns out not to be exposed via the OpenAI-compatible
      endpoints, the native-Converse-fallback decision escalates to a spec
      addendum before implementation.

### Phase 4 — deferred (separate future specs/tasks)

Bedrock-native Converse wire (fallback only, see Phase 3); enterprise
identity auth (SigV4/Entra/ADC) plugged at the `auth_scheme` seam; possible
evidence-gated migration of `moonshot.py`/`zai.py` onto the engine.

## Testing strategy

- **Engine (once, parameterized over presets):** resolution precedence
  (explicit > api_settings > env), base-URL normalization rules per preset
  (including bare-host Databricks paste and the only-when-path-empty append
  rule), payload building per flags (snapshot fixtures), finish-policy
  tables, strict response/stream validation (malformed cases fail closed),
  response-allowance behavior (allowlisted extras ignored, non-allowlisted
  extras fail closed, required shapes never relaxed), continuation
  checkpoint round-trip, keyless auth handling, redaction of keys in
  errors/logs.
- **Per preset:** record validity; payload snapshot; readiness resolution;
  registry membership; pricing seeds load.
- **Parity:** registry coverage vs `API_CALL_HANDLERS`, audited set,
  `_cloud_provider_keys`, display names, endpoints map, auto-refresh list.
- **Live:** Databricks live gate in Phase 1 (pattern:
  `Tests/LLM_Calls/test_live_moonshot_zai_api.py`); each later preset gets at
  least one live probe before its task closes.
- Per repo policy, targeted suites only unless a full sweep is requested.

## Open items (resolved during Phase 1, none blocking the design)

- **O-1 — Databricks model listing:** whether the AI Gateway exposes models
  at `/openai/v1/models` (standard discovery) or only via
  `GET /api/2.0/serving-endpoints` (custom discovery shape). Live probe
  decides; the preset's discovery field supports both, and manual
  `[providers]` seeding works regardless.
- **O-2 — Azure exact URL shaping:** finalized in the Phase 3 plan; the seam
  (constructed route + query params in the transport call, never in
  configured base URLs) is fixed by this design.
- **O-3 — Bedrock OpenAI-compatible specifics:** exact endpoint URL pattern
  (region placeholder), model-ID format (inference-profile prefixes such as
  `us.*`), whether a `/v1/models`-style listing exists or discovery needs
  the native ListFoundationModels shape, response-envelope allowances, and
  API-key lifetime handling in readiness copy. Pinned from AWS docs + live
  probe in the Phase 3 plan; manual `[providers]` seeding works regardless.

## Alternatives considered

- **A — adapter per provider** (status quo): maximum per-provider control,
  but the ~20-site registration ritual and ~1,000-line adapters per provider
  make Hermes-level breadth unreachable; duplication between `zai.py`/
  `moonshot.py` already demonstrates the decay.
- **C — registry-only (literal Hermes):** every provider a user-registered
  custom endpoint; cheapest code, but pushes setup burden to users, loses
  baked defaults/capabilities/pricing, and conflicts with the app's
  first-class provider UX (Settings sections, readiness, catalog
  auto-refresh).
- **B (chosen) — engine + presets:** Hermes's breadth via the generic path,
  this repo's strictness via the `hosted_chat` boundary, first-class identity
  via the registry; bespoke adapters remain available where the wire truly
  differs (Bedrock).
