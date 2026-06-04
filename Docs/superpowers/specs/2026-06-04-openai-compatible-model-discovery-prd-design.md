# OpenAI-Compatible Model Discovery PRD

Date: 2026-06-04
Status: Draft for review
Target branch: `dev`

## Purpose

Allow users to query an existing configured OpenAI-compatible API endpoint and discover available models, so they can use newly released or endpoint-hosted models without waiting for Chatbook app updates or static config updates.

The first version focuses on safe, explicit, provider-scoped discovery. It expands user choice without silently rewriting configuration or making network calls during normal navigation.

## Current Code Findings

Relevant files:

- `tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py`
- `tldw_chatbook/LLM_Provider_Catalog/server_llm_provider_catalog_service.py`
- `tldw_chatbook/LLM_Provider_Catalog/llm_provider_catalog_scope_service.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Screens/provider_model_resolution.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/config.py`
- `tldw_chatbook/model_capabilities.py`
- `Tests/LLM_Provider_Catalog/`
- `Tests/UI/test_settings_configuration_hub.py`
- `Tests/UI/test_console_session_settings.py`

Observed state:

- Chatbook already has an `LLM_Provider_Catalog` seam with local and server catalog services.
- `LocalLLMProviderCatalogService` exposes model lists from static local `[providers]` config through `get_cli_providers_and_models()`.
- `ServerLLMProviderCatalogService` can observe server-backed LLM provider/model catalog APIs through `TLDWAPIClient`.
- `LLMProviderCatalogScopeService` already preserves source-aware routing, policy action IDs, and local/server separation.
- Settings owns global provider/model defaults and provider+model profile persistence.
- Console consumes provider/model state through shared effective provider/model resolution and active session settings.
- Current selectable model lists are static unless the user manually edits config.
- Console has provider execution parity work that requires provider/model labels, Settings, and send readiness to agree.

Relevant existing design constraints:

- Settings must remain the global configuration hub for Providers & Models.
- Console remains the primary agentic control surface and owns active chat/run state.
- The app must not expose raw API keys or secret-bearing URLs in UI, tests, screenshots, or logs.
- Provider/runtime boundaries and cross-module service contracts require ADR hygiene before implementation.

## Product Goal

Users can manually discover the model list for a configured OpenAI-compatible provider endpoint, select discovered models immediately, and explicitly save discovered models when they want them to persist.

Success means:

- A user can use a newly available model from an existing endpoint in Console without editing TOML manually.
- Settings clearly distinguishes saved models from session-discovered models.
- Discovery failure is recoverable and does not damage existing model configuration.
- Unknown models remain usable with conservative capability assumptions.

## V1 Scope

In scope:

- Existing configured providers only.
- OpenAI-compatible model-list endpoints, primarily `/v1/models` style APIs.
- Manual discovery from `Settings -> Providers & Models`.
- Runtime discovery cache for current app session.
- Explicit persistence into saved provider model configuration.
- Console consumption of saved plus runtime-discovered model options.
- Conservative unknown-capability handling.
- Safe error categorization and secret redaction.

Out of scope:

- Automatic background refresh.
- Discovery on app startup, Settings mount, or Console mount.
- Ad hoc provider creation from arbitrary URLs.
- Broad provider-specific adapters for non-OpenAI-compatible APIs.
- Remote `tldw_server` catalog-first workflows.
- Pricing, tokenizer, context-window, or capability database updates from third-party endpoints.
- Full model capability inference beyond existing rules and safe endpoint metadata.

## User Workflow

Primary workflow:

1. User opens `Settings -> Providers & Models`.
2. User selects or reviews an existing configured provider.
3. If the provider has an OpenAI-compatible endpoint or base URL, Settings shows `Discover models`.
4. User clicks `Discover models`.
5. Chatbook queries the provider endpoint's model-list route using the configured credential source.
6. Settings shows discovery status: `checking`, `found N models`, `failed`, or `no models returned`.
7. Discovered models appear in the model picker with a `Discovered` marker.
8. User can select a discovered model immediately for Console defaults or active Console session use.
9. User can explicitly persist discovered models by saving provider settings.
10. Console model pickers merge saved models and runtime-discovered models for the active provider.

Required UX behavior:

- No network call occurs just by opening Settings or Console.
- Discovered models are visibly distinguishable from saved config models.
- Unknown models are selectable, but labeled with conservative capability assumptions.
- Discovery failures show actionable recovery copy without exposing secrets.
- If a selected discovered model is not saved, the UI states that it is session-scoped.

## Data Model

Discovery produces normalized model records rather than exposing raw endpoint payloads directly.

Recommended record shape:

```text
provider: string
provider_list_key: string
model_id: string
display_name: string
source: saved | discovered
endpoint_fingerprint: string
discovered_at: ISO timestamp
metadata_raw_safe: mapping
capability_status: known | inferred | unknown
persisted: boolean
```

Model identifier rule:

- `model_id` is the raw provider model string returned by the endpoint, for example `gpt-4.1`, `openai/gpt-oss-20b`, or `Qwen3-30B-A3B`.
- UI selectors and `[providers].<provider>` persistence use this raw `model_id`.
- Catalog `record_id` values may remain provider-prefixed, for example `local:llm_model:<provider>/<model_id>`, but those record IDs must never be written as selectable model values.

Runtime cache:

- Holds discovered models for the current app session.
- Is keyed by provider plus endpoint fingerprint, not provider name alone.
- Deduplicates discovered models against saved `[providers]` models.
- Invalidates when endpoint/base URL or credential source changes.
- Does not store raw API keys.

Persistence:

- Saving provider settings may promote discovered model IDs into the saved provider model list.
- Persisted models become ordinary saved config models on the next launch.
- Selecting a discovered model as a provider/model default should either persist that model or warn that the default will not survive restart.
- Runtime cache and persisted config remain separate in code and UI.

Endpoint fingerprint:

- Must be stable for the normalized provider endpoint.
- Must not include raw API keys, bearer tokens, request headers, or full query strings.
- Should change when the endpoint/base URL or credential source identity changes.

Provider identity rule:

- Discovery must use the same provider identity normalization as Console readiness and send execution.
- Implementation must route display/config inputs through the existing provider identity helpers from Console provider parity work, including `provider_config_key()` and the Console provider identity resolver.
- Cache keys, endpoint lookup, discovery eligibility, persistence, Settings display, and Console model selection must use one resolved provider identity record instead of independently normalizing provider names.
- The resolved provider identity must include a `provider_list_key` or `saved_models_key`: the exact top-level `[providers]` key that will be read and written for saved model options.
- `provider_list_key` must be resolved by normalized match against the existing `[providers]` keys before appending discovered models. If no existing key matches, Settings must require an explicit provider-save flow rather than silently creating an alias entry.
- If multiple existing `[providers]` keys normalize to the same provider identity, persistence must refuse to write and show a cleanup/selection recovery path. It must not guess which alias to update.
- Persistence must preserve the exact existing `[providers]` key spelling and casing, for example `OpenRouter`, `Custom_2`, `vLLM`, and `MistralAI`.
- Alias-sensitive providers such as `Custom`, `custom-openai-api`, `Custom_2`, `custom_2`, `OpenRouter`, `openrouter`, `vLLM`, `vllm`, `local-llm`, `local_llm`, `MistralAI`, and `mistralai` must have explicit regression coverage.

## Capability Handling

Capability resolution order:

1. Existing `model_capabilities` exact or pattern rules.
2. Safe endpoint metadata, if the response includes model type or modality hints.
3. Conservative unknown fallback.

Unknown fallback:

- Treat the model as text-chat only.
- Do not assume vision support.
- Do not assume tool support.
- Do not assume tokenizer, pricing, or context-window metadata.

User-facing copy:

```text
Some discovered models have unknown capabilities. Chatbook will treat them as text-chat models until configured otherwise.
```

Unknown capability status must not block selection or send by itself. Provider readiness, endpoint reachability, credential status, and execution support remain the blockers.

Implementation must add a status-bearing capability resolver rather than overloading existing boolean capability results.

Recommended shape:

```text
resolve_discovered_model_capability_status(provider, model_id, endpoint_metadata) -> {
  status: known | inferred | unknown
  capabilities: mapping
  reason: string
}
```

Rules:

- `known` means an exact or pattern rule in `model_capabilities.py` matched.
- `inferred` means endpoint metadata safely identified type/modality.
- `unknown` means no matching rule or safe metadata was available.
- A returned capability such as `vision = false` must not by itself imply `known`; otherwise unknown text models would be mislabeled as known.

## Architecture

Use the existing provider catalog boundary. Do not create a parallel model registry.

Core components:

- `LLM_Provider_Catalog`: add a local endpoint discovery service for OpenAI-compatible model-list endpoints.
- `LocalLLMProviderCatalogService`: expose saved plus runtime-discovered models for local Chatbook providers.
- `LLMProviderCatalogScopeService`: keep source-aware routing and add policy-gated discovery/persistence actions.
- `SettingsScreen`: add discovery trigger, status/result display, discovered markers, persistence affordance, and failure recovery copy.
- `ChatScreen` and Console settings modal: consume merged saved-plus-discovered model lists for the selected provider.
- `model_capabilities`: resolve known, inferred, or unknown capability status for discovered records.

Proposed flow:

```text
Settings action
  -> LLMProviderCatalogScopeService.discover_models(mode="local", provider)
  -> LocalEndpointModelDiscoveryService
  -> configured endpoint + credential source
  -> normalized discovered model records
  -> runtime discovery cache
  -> Settings/Console provider-model option builders
```

Policy/action IDs:

```text
llm.catalog.models.discover.local
```

Persistence is a Settings config-write action, not provider catalog mutation. The local catalog seam may expose discovery observation and runtime cache reads, but it must not own local provider configuration editing. Settings owns writes to saved configuration and should continue to use the existing Settings save/revert validation path.

The local discovery path observes locally configured provider endpoints. The server catalog path remains separate and continues to represent active-server state.

## Runtime API Contracts

Implementation should introduce small, testable discovery/cache contracts instead of embedding discovery state inside `SettingsScreen`.

Recommended service contracts:

```text
discover_models(
  provider_identity,
  *,
  endpoint_override: str | None = None,
  credential_source_override: str | None = None,
  request_timeout_seconds: float = 20.0,
) -> ModelDiscoveryResult

list_discovered_models(provider_identity) -> tuple[DiscoveredModelRecord, ...]

clear_discovered_models(provider_identity, *, reason: str) -> None

merge_saved_and_discovered_models(
  provider_identity,
  saved_models: Sequence[str],
) -> MergedModelOptions

persist_discovered_models_to_settings(
  provider_identity,
  model_ids: Sequence[str],
  settings_writer,
) -> PersistDiscoveryResult
```

Recommended ownership:

- Runtime cache owner: a local discovery/cache service owned by `LLM_Provider_Catalog` or a focused sibling module, injected into Settings and Console through `app`.
- Persistence owner: Settings config adapter/save path.
- Console consumer: `ChatScreen._providers_models()` or its successor should merge `app.providers_models` with runtime-discovered models.
- Settings consumer: Providers & Models should read the same merged options as Console.

Refresh behavior:

- After a successful discovery, Settings updates its model picker from the merged model options.
- Console sees the same discovered models without restart.
- After explicit persistence, Settings updates `[providers].<provider_list_key>`, updates `app.providers_models`, and keeps the runtime cache entry marked `persisted=True` or removes it as duplicate.
- If Settings reverts provider endpoint/credential changes, discovery results for stale endpoint fingerprints are hidden.

## OpenAI-Compatible Endpoint Contract

V1 supports endpoints that behave like:

```text
GET <base_url>/models
GET <base_url>/v1/models
```

Discovery endpoint eligibility is based on both resolved provider identity and endpoint shape. A provider name alone is not enough, because some providers have native default endpoints that are not OpenAI-compatible model-list endpoints.

The implementation should normalize common OpenAI-compatible base URL shapes:

- If the configured endpoint already ends in `/v1`, query `/v1/models`.
- If the configured endpoint already ends in `/models`, query that exact endpoint.
- If the configured endpoint is a chat completions URL, derive the corresponding `/models` URL only when safe and unambiguous.
- If the endpoint is a known direct llama.cpp-style `/completion` or `/completions` URL, derive the origin using the existing llama.cpp normalization behavior, then query `<origin>/v1/models`.
- Otherwise append `/v1/models` only for providers whose identity and endpoint contract are explicitly OpenAI-compatible.

Native endpoint shapes that are not safely OpenAI-compatible must be unsupported in v1. Examples:

- `koboldcpp` default `/api/v1/generate` is not eligible for OpenAI-compatible discovery unless the user configures a separate OpenAI-compatible `/v1` or `/v1/chat/completions` endpoint.
- Native Anthropic, Gemini, Cohere, Hugging Face search/router, and Ollama `/api/tags` model discovery are not part of this v1.

Endpoint config key precedence:

1. Staged unsaved Settings values for the active Providers & Models category, if the user triggered discovery from Settings with unsaved endpoint edits.
2. Provider-specific `[api_settings.<provider>]` keys in this order: `api_base_url`, `api_base`, `base_url`, `api_url`, `endpoint`.
3. Provider defaults already used by Console provider execution/readiness.
4. No endpoint found.

Credential source precedence:

1. Staged unsaved Settings credential source, if discovery is triggered from Settings and the user has edited it.
2. Provider-specific credential source in `[api_settings.<provider>]`.
3. Existing environment-variable and config `api_key` resolution already used by local provider readiness.

Keyring-backed server credentials are out of scope for this local v1 discovery flow. If a future implementation adds local keyring provider credentials, it must define a separate credential-store contract, ownership boundary, and redaction tests.

Discovery triggered from Console should use persisted provider settings only until Console has an explicit endpoint-edit workflow. Discovery triggered from Settings may use staged Settings values, but must label results as tied to unsaved endpoint changes and must invalidate them if the user reverts.

Safe endpoint display:

- Display helpers must strip query strings, userinfo, fragments, and authorization-bearing parameters before showing endpoint text.
- Logs and error messages must use the same safe endpoint display helper.

Accepted response shapes should include the OpenAI style:

```json
{
  "object": "list",
  "data": [
    {"id": "model-name", "object": "model"}
  ]
}
```

The normalizer may also accept compatible list/dict variants if they can be mapped without ambiguity. Unsupported shapes become `Invalid response`, not a crash.

## Discoverable Provider Eligibility

V1 discovery is not enabled for every provider with an endpoint string. A provider is discoverable only when both conditions are true:

1. The resolved provider identity is allowed for OpenAI-compatible discovery.
2. The normalized endpoint shape is OpenAI-compatible or safely derivable to an OpenAI-compatible `/models` endpoint.

Eligible provider identity groups:

- `openai`
- `openrouter`
- `custom` / `custom-openai-api`
- `custom_2` / `custom-openai-api-2`
- `llama_cpp` / `local_llamacpp`
- `local_llamafile`
- `vllm` / `local_vllm`
- `tabbyapi`
- `aphrodite`
- Other provider identities only if their existing provider execution contract is explicitly OpenAI-compatible.

Conditionally eligible identity groups:

- `koboldcpp` only when its configured endpoint is OpenAI-compatible, not when using the default native `/api/v1/generate` endpoint.
- `ollama` / `local_ollama` only when using the configured OpenAI-compatible `/v1` endpoint, not native `/api/tags` in this v1.

Non-goal examples for v1:

- Anthropic native API model discovery.
- Gemini native API model discovery.
- Cohere native API model discovery.
- Hugging Face router/model search discovery outside OpenAI-compatible `/models`.
- Native Kobold `/api/v1/generate` discovery.
- Ollama native `/api/tags` discovery.

Implementation should prefer an explicit provider capability flag or mapping such as:

```text
supports_openai_compatible_model_discovery(provider_identity, normalized_endpoint) -> bool
```

This flag must be tested against alias-sensitive provider identities and native-endpoint false positives.

## Settings UX

Providers & Models should add a discovery area near the provider/model controls:

```text
Provider: OpenAI-compatible custom
Endpoint: http://127.0.0.1:9099/v1
Credential: env OPENAI_API_KEY present

[Discover models] [Save discovered models]
Status: Found 3 models · Last checked this session

Model:
  saved-model
  discovered-model-a   Discovered
  discovered-model-b   Discovered · Unknown capabilities
```

Required controls:

- `Discover models`: starts manual discovery.
- `Save discovered models`: persists currently discovered models for this provider.
- Existing save/revert controls: continue to own provider default/profile persistence.

Required status details:

- Last discovery state.
- Count of discovered models.
- Count of new models after dedupe.
- Whether selected model is saved or session-discovered.
- Whether unknown capabilities were detected.

## Console UX

Console should use merged provider model options for the selected provider:

```text
Model options = saved provider models + runtime discovered provider models
```

Rules:

- Saved models appear normally.
- Discovered models show a compact `Discovered` marker where the UI supports it.
- If Console is using a discovered-but-unsaved model, Settings/Console recovery copy should offer to save it rather than forcing the user to edit TOML.
- Discovered models should not contradict Console readiness. The provider/model labels in the control bar, run inspector, and send preflight must resolve to the same model.

## Discovery States And UX Copy

Discovery states:

| State | Copy |
| --- | --- |
| Idle | `Model discovery has not run for this provider.` |
| Checking | `Checking configured endpoint for available models...` |
| Success | `Found N models from <provider>.` |
| No models | `Endpoint responded, but no models were returned.` |
| Unreachable | `Could not reach endpoint. Check the provider base URL.` |
| Unauthorized | `Endpoint rejected the configured credential. Check API key source.` |
| Invalid response | `Endpoint responded, but the model list format was not recognized.` |
| Timeout | `Discovery timed out. Check network/server status and retry.` |
| Unsupported | `This provider is not configured as OpenAI-compatible for discovery.` |

Recovery rules:

- Never display raw API keys, bearer tokens, request headers, or secret-bearing URLs.
- Show endpoint host/path safely, but redact query strings.
- Discovery failure must not clear existing saved models.
- Discovery failure must not clear previous discovered models unless the endpoint fingerprint changed.
- Persistence failure must leave runtime-discovered models usable in the current session.

## Configuration And Persistence Rules

Persisted discovered models update the top-level `[providers]` model list consumed by `get_cli_providers_and_models()`.

Persistence must:

- Preserve existing saved model order.
- Append new discovered model IDs after existing models unless the user reorders them in a future feature.
- Deduplicate exact model IDs.
- Avoid deleting saved models that were not returned by discovery.
- Validate provider identity and endpoint fingerprint before writing.
- Resolve and write the exact existing `[providers].<provider_list_key>` entry by normalized match; do not create duplicate alias keys.
- Refuse persistence with clear recovery copy when no exact saved-model provider key can be resolved.
- Refuse persistence with clear recovery copy when multiple saved-model provider keys normalize to the same identity. Example: `[providers].OpenRouter` and `[providers].openrouter` both exist.
- Keep secrets out of persisted discovery metadata.
- Update `app.providers_models` after save so Console and Settings reflect the persisted list without restart.
- Write raw model IDs only, not provider-prefixed catalog `record_id` values.

The design intentionally avoids automatic config mutation. A successful discovery only changes runtime catalog state until the user saves.

## Security And Privacy

Discovery uses existing provider credential resolution and redaction rules.

Required safeguards:

- Do not print raw API keys, bearer tokens, request headers, or auth-bearing URLs.
- Do not store raw endpoint responses if they contain unsafe fields.
- Categorize endpoint errors before display.
- Keep request/response diagnostic detail available in logs only after redaction.
- Avoid automatic network calls from passive navigation.
- Respect runtime policy decisions before dispatching endpoint discovery.

## Acceptance Criteria

Functional:

- [ ] User can manually discover models for a configured OpenAI-compatible provider.
- [ ] Discovery uses the provider's configured endpoint and credential source.
- [ ] Discovered models appear in Settings model selection with a `Discovered` marker.
- [ ] Console can select and use discovered models during the same app session.
- [ ] Saved provider models and discovered runtime models are deduplicated.
- [ ] User can explicitly persist discovered models into provider settings.
- [ ] Unknown discovered models remain usable with conservative capability warnings.
- [ ] Discovery failure does not remove saved models or break current Console selection.

Safety:

- [ ] No raw API keys, bearer tokens, request headers, or secret-bearing URLs are displayed or logged.
- [ ] Unauthorized, unreachable, timeout, empty response, and invalid response states are distinguishable.
- [ ] Endpoint or credential changes invalidate stale discovery results for that provider.
- [ ] No network discovery runs automatically on app startup, Settings mount, or Console mount.

## Validation Plan

Automated tests:

- Unit tests for OpenAI-compatible endpoint URL normalization.
- Unit tests for model-list response normalization.
- Unit tests for discovery error categorization.
- Unit tests for runtime cache dedupe and endpoint-fingerprint invalidation.
- Unit tests for provider-list key resolution, including no match, exact single match, and duplicate normalized matches.
- Unit tests for unknown capability classification.
- Settings mounted tests for idle, checking, success, failure, unknown capabilities, and persist states.
- Console mounted tests proving discovered models are selectable and provider/model labels remain consistent.
- Regression tests proving saved model lists survive failed discovery.
- Redaction tests proving keys, bearer tokens, headers, and secret-bearing URLs are absent from UI strings and logs.

Manual/CDP QA:

- Discover models from a local OpenAI-compatible endpoint.
- Select a discovered model in Settings and verify Console sees it.
- Save discovered models, restart, and verify the model remains available.
- Trigger unreachable, unauthorized, invalid response, empty model list, and timeout states.
- Capture Settings and Console screenshots showing discovered markers and unknown-capability warnings.

## ADR Check

ADR required: yes for implementation.

ADR path: `backlog/decisions/NNN-openai-compatible-model-discovery.md` or an existing provider-catalog/runtime-boundary ADR if one already covers this exact decision.

Reason: implementation changes provider/runtime boundaries, source-aware catalog contracts, runtime cache ownership, and config persistence rules.

This PRD does not create the ADR because it is a product requirements document. The implementation plan must either create the ADR before code changes or link an existing canonical ADR that covers the same architectural decision.

## Follow-Up Opportunities

Future work after v1:

- Stale discovery prompts without automatic refresh.
- Ad hoc endpoint discovery and provider creation.
- Server catalog integration as a separate source-aware workflow.
- Provider-specific discovery adapters for non-OpenAI-compatible APIs.
- Capability profile authoring for discovered models.
- Model pricing/context-window/tokenizer metadata management.
- User-controlled model ordering and hiding.
