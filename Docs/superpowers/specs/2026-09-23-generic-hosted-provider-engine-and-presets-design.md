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
deliberate xAI exclusion and Bedrock deferral.
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
   Perplexity next; Azure OpenAI and Gemini's OpenAI-compatible layer after
   two small transport extensions; Bedrock deliberately deferred (non-OpenAI
   Converse wire).

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
- AWS Bedrock (Converse API is not OpenAI-shaped → needs a true strict
  adapter) and full enterprise identity auth (SigV4, Entra, service-account
  ADC). Key-based auth only; seams left open (Phase 4 candidates).
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

## Architecture

Three pieces. Dependency direction: `provider_registry` is a leaf (stdlib
only) importable by `config.py`; the engine lives in `LLM_Calls/` beside
`hosted_chat.py`; presets are data consumed by the engine and registered by
callers.

### 1. `tldw_chatbook/provider_registry.py` — identity catalog (new)

One frozen dataclass record per provider (built-in hand-written providers get
opaque records too — the registry covers *identity*, not implementation):

| Field | Purpose |
| --- | --- |
| `key` | Canonical provider id (`"databricks"`) used in dispatch and config |
| `display_name` | User-facing label (replaces `_PROVIDER_DISPLAY_NAMES` literals) |
| `classification` | `"cloud"` \| `"local"` — feeds `_cloud_provider_keys` |
| `api_key_env_var` + candidates | Feeds readiness key requirements (`DATABRICKS_TOKEN`, …) |
| `default_base_url` | May be `None` when per-account (Databricks workspace host) |
| `native_tools` | Feeds `Agents/native_tools.py::NATIVE_TOOLS_PROVIDERS` |
| `reasoning_effort` | Feeds reasoning-effort support sets in Console/settings |
| `auto_refresh` | Membership in `AUTO_REFRESH_PROVIDER_LIST_KEYS` |
| `pricing_seeds` | Optional per-model $/Mtok entries for `pricing_catalog.py` |

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

### 2. `LLM_Calls/hosted_provider_engine.py` + `hosted_provider_presets.py` (new)

`build_hosted_chat_handler(preset) -> chat_with_*` returns a handler with the
same signature and param surface the dispatch expects (mirrors
`chat_with_zai`'s signature so `PROVIDER_PARAM_MAP` entries stay uniform —
engine providers share one param-map constant).

`HostedProviderPreset` holds everything provider-specific as data:

| Field | Examples |
| --- | --- |
| `key`, `display_name` | `"databricks"`, `"Databricks"` |
| `base_url_rule` | Default URL + normalization (append `/openai/v1` when the user pasted a bare Databricks workspace host; reject terminal `/chat/completions` paths — reuse `normalize_hosted_chat_base_url`) |
| `api_key_env_candidates` | `("DATABRICKS_TOKEN",)` |
| `finish_reasons` | allowed terminals `{stop, tool_calls, length}`; provider-terminal-error reasons → `ChatProviderError(502)` |
| `payload_flags` | temperature / top_p / stop / response_format (incl. `json_schema`) / `reasoning_effort` key name / extra body fields |
| `reasoning_disposition` | `"displayable" \| "proprietary" \| "ignored"` (per `HostedChatFinishPolicy`) |
| `auth_scheme` | `"bearer"` (Phase 1–2) \| `"api_key_header"` (Phase 3, Azure) |
| `capability_hints` | native tools, streaming, reasoning effort (mirror registry flags; registry is authoritative) |

The engine absorbs, once, the boilerplate duplicated today: config/env
resolution (`_resolve_api_key`, `_resolve_base_url`, numeric/streaming
resolvers), message/tool/response-format normalization (lifted from
`zai.py`'s OpenAI-contract normalizers), finish-policy construction from
data, response/stream wrapping, and chat-completions continuation
checkpoints (`protocol="chat_completions"`, parameterized provider/model/
base URL). Error messages are prefixed with the preset display name.

**Escape hatch rule:** if a provider proves genuinely quirky, the answer is a
zai-style hand-written strict adapter — never a bespoke hack flag on a
preset. The engine stays honest by refusing to absorb unbounded weirdness.

### 3. `hosted_chat.py` extensions (small, existing file)

- `HostedHTTPTransportConfig` gains `auth_scheme` (default `"bearer"`;
  `"api_key_header"` sends `api-key: <key>` instead of `Authorization` —
  Azure in Phase 3). The hardcoded Bearer moves to scheme-selected
  construction at the `session.post` call site (`hosted_chat.py:561`).
- Phase 3 only: preset-controlled route/query shaping for Azure
  (`/openai/deployments/{model}/chat/completions?api-version=…`). The seam:
  `owned_json_post`'s `route` parameter becomes a fully-formed relative path
  plus optional query parameters supplied by the caller. Base-URL validation
  still rejects query strings in *configured* URLs; transport-level query
  params are constructed, never configured.

## Provider identity integration (surfaces the registry replaces)

Per the empirical new-provider checklist, the literal tables that become
registry-derived: `config.py` `[providers]` seeds, `[api_settings.X]`
default tables (a per-preset defaults builder emits these),
`_cloud_provider_keys`; `Chat/Chat_Functions.py` audit + param-map
constants; `Chat/provider_readiness.py` key sets; `Chat/console_provider_endpoints.py` `_BUILTIN_PROVIDER_ENDPOINTS` (Databricks
entry documents the `/openai/v1` suffix); `console_provider_support.py`
display names + reasoning-effort sets; `provider_setup_persistence.py`
canonical keys/aliases; `Agents/native_tools.py` `NATIVE_TOOLS_PROVIDERS`;
`LLM_Provider_Catalog/model_catalog_settings.py`
`AUTO_REFRESH_PROVIDER_LIST_KEYS`; `local_llm_provider_catalog_service.py`
strict-hosted resolution branch (engine presets expose a
`resolve_request`-shaped entry point exactly like `resolve_zai_request`);
`LLM_Calls/pricing_catalog.py` seeds. `UI/Screens/settings_screen.py`
provider pickers already derive from the handler catalog — only display-name
and reasoning-effort copy change.

## Data flow (Databricks, Phase 1)

1. **Setup:** Settings → Databricks section (registry-generated): API key env
   var `DATABRICKS_TOKEN` (or masked stored key per ADR-012), `api_base_url`
   = workspace host or full `…/openai/v1` URL, default model, streaming.
   Readiness: requires key **and** base URL (no shipped default — workspace
   hosts are per-account); blocked-send recovery flow guides both.
2. **Send:** `chat_api_call` → `API_CALL_HANDLERS["databricks"]` (engine-built)
   → resolve from `[api_settings.databricks]`/env → normalize base URL
   (append `/openai/v1` to a bare host) → payload per preset flags →
   `owned_json_post(route="chat/completions", Bearer PAT)` → strict
   normalization with preset finish policy → OpenAI-shaped response dict with
   terminal metadata + continuation checkpoints (identical consumer shape to
   `chat_with_zai` output).
3. **Model auto-refresh:** catalog service resolves Databricks via the
   engine's `resolve_request` entry point (strict-hosted branch), discovers
   models (see open item O-1), consent-gated write-through to `[providers]`
   per ADR-020. Manual `[providers].Databricks` seeding works regardless.
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

Build: `provider_registry.py`; `hosted_provider_engine.py` +
`hosted_provider_presets.py` with `DATABRICKS` preset; registry-derived
tables rolled out consumer-by-consumer (each consumer swap is independently
testable); Databricks wired end-to-end.

Acceptance criteria:

- [ ] Databricks selectable and usable in the Console (non-streaming and
      streaming) with a real workspace via AI Gateway (live-verified per
      `backlog/docs/lessons-live-verification.md`: models list, one chat
      round-trip, one streamed round-trip with usage capture).
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

### Phase 3 — Enterprise, key-based

- [ ] `auth_scheme="api_key_header"` in `hosted_chat` transport.
- [ ] Azure OpenAI preset: deployment-shaped URLs (`/openai/deployments/
      {model}/chat/completions?api-version=…`), model discovery via
      deployment listing, key-based auth.
- [ ] Gemini OpenAI-compatible layer preset (OpenAI-compat endpoint with API
      key), distinct from the existing native Google adapter.

### Phase 4 — deferred (separate future specs/tasks)

Bedrock Converse strict adapter; enterprise identity auth (SigV4/Entra/ADC)
plugged at the `auth_scheme` seam; possible evidence-gated migration of
`moonshot.py`/`zai.py` onto the engine.

## Testing strategy

- **Engine (once, parameterized over presets):** resolution precedence
  (explicit > api_settings > env), base-URL normalization rules per preset
  (including bare-host Databricks paste), payload building per flags
  (snapshot fixtures), finish-policy tables, strict response/stream
  validation (malformed cases fail closed), continuation checkpoint
  round-trip, redaction of keys in errors/logs.
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
