# ADR-147: Agent provider routing — preset routing, gated spawn overrides, child-owned params

Status: Accepted
Date: 2026-09-11
Related Task: [TASK-32477](../tasks/task-32477%20-%20Agent-provider-routing-preset-routing-gated-spawn-overrides.md)
Related Spec: [Agent provider routing design](../../Docs/superpowers/specs/2026-09-11-agent-provider-routing-design.md)
Amends: [ADR-146](146-console-custom-endpoint-registry.md) (registry entries gain an optional `params` table)
Follow-up: TASK-32508 (preset `fallback_models` chains — deferred)

## Context

A Console agent run resolves one provider+model at send time and everything it spawns inherits that identity: `AgentDefinition.model` could only swap the model on the *same* provider endpoint. Users running a strong cloud model alongside a cheap/local one (e.g. Kimi API as architect, a local qwen3.8-27B as implementer) could not have a master agent delegate onto the other backend. ADR-146's registry made endpoints first-class for Console sessions but said "sampling and generation settings are never copied; they remain governed by the existing per-provider defaults chain" — leaving registry endpoints without any endpoint-owned param store. Two further facts shaped the decision: the Agents runtime never plumbs session sampling params into runs at all (`AgentService.call_model` sends `self.chat_call(...)` with only `tools`/continuation kwargs, so params fall back to `chat_api_call`'s internal config resolution), and fleet continuation re-resolves the agent definition by name, so editing a preset between finish and resume would silently move a child mid-conversation. The pi-subagents project's models doc was consulted as prior art; its precedence chain (per-run override → role override → default → parent) converges with the one chosen here.

## Decision

- **Routing resolves through one pure resolver** (`Agents/agent_routing.py`), invoked by `AgentService.spawn()` before fleet reservation. Resolution order, each level filling only blanks left above: (1) ad-hoc spawn args — honored only when `[agents] spawn_override_enabled` and matching `spawn_override_allowlist`; (2) the named preset's `provider`/`model`; (3) `[agents] subagent_default_provider/model`; (4) inherit the parent's provider+model. A `RoutingError` is returned as the spawn tool's error result (`SpawnAdmissionRefusal`): no silent cross-provider fallback, no fleet slot consumed.
- **Ad-hoc args carry identity only.** `spawn_subagent` gains optional `provider`/`model` string args — never URLs (endpoints come only from config/registry, blocking context exfiltration to an attacker endpoint) and never params. Allowlist entries are a bare provider id (`llama_cpp`, `custom-ep:qwen-local`) or a `provider/model-glob` form (`llama_cpp/qwen3.8-*`; fnmatch, case-insensitive). A final-provider guard applies whenever any ad-hoc arg is present: the resolved `provider/model` must match the allowlist or the provider must equal the parent's, so a model-only arg cannot ride a paid default the user never allowlisted. The args are omitted from the tool schema entirely when the override flag is off.
- **`AgentDefinition` gains `provider` and `params`.** `model` without `provider` keeps the legacy same-endpoint semantics; existing presets are unchanged. `definition_fingerprint` extends to the new fields (they shape what ran).
- **Children never inherit parent sampling/API params.** A child's params are built fresh for its resolved provider+model through a six-layer stack, highest first: preset `params` → `[api_settings.<provider>].model_defaults.<model>` → `[console.provider_defaults.<provider>]` → registry entry `params` → `[chat_defaults]` → `[api_settings.<provider>]` scalars → function fallbacks. Uniform rule, including plain spawns — a deliberate behavior change from today's implicit `chat_api_call` config fallback: child params become explicit, deterministic, and visible in the run log. The known-param set and validators live in a new pure `Chat/sampling_params.py` shared by presets, registry entries, and the resolver; transport-level keys (`streaming`) are excluded.
- **Registry entries gain an optional `params` table** (`[custom_endpoints.<slug>.params]`), amending ADR-146's "template scope is limited to family/endpoint/model list": endpoint-owned tuning is more specific than global chat_defaults and slots above that layer, below Console per-provider saved defaults (delivered via an optional `extra_sources` parameter on `build_default_console_session_settings`, empty by default so session defaults stay byte-identical). Entries without `params` behave exactly as before.
- **The resolved target is persisted on the run row** (`resolved_provider`, `resolved_model`, `resolved_base_url`, `resolved_params_json`, schema v16). Continuation/resume reuses the snapshot; only legacy NULL rows re-resolve live. Editing a preset, endpoint, or config default after a spawn never moves an already-spawned child.

## Alternatives

- **Named route registry** (a separate route entity presets reference) was rejected: it duplicates `api_settings`/registry knowledge and adds a config surface the scenario doesn't need.
- **Policy-file rules engine** was rejected as speculative; the allowlist expresses the one rule users asked for and can grow into policy later.
- **Preset `fallback_models` chains** (pi-subagents-style retryable-failure fallback, pre-tool-activity only) were deferred to TASK-32508, not rejected: the chain is user-authored so it does not violate the no-silent-fallback rule, but its interactions with budgeting, continuation, and admission are a design surface of their own.
- **A thinking-level ceiling** (`subagent_max_thinking`, pi-subagents `maxThinking`) was rejected: params here are all user-authored, so there is little to guard against.
- **Provider-scoped role override matrices** (pi-subagents `agentOverridesByProvider`) were rejected: preset + default layers express the same intent without a matrix.
- **Inheriting parent params for same-provider children only** was rejected in favor of the uniform rule: a split rule is harder to reason about, and the runtime never delivered session params to runs anyway.
- **Validating preset providers against the registry at authoring time** was rejected: entries can be deleted between authoring and spawn, so registry existence is a spawn-time `RoutingError`, not a validation error.
- **Ad-hoc params in spawn args** was rejected: model-generated params on a paid provider are a cost/injection surface; params come only from user-authored stores.

## Consequences

- New `[agents]` keys ship commented-out with defaults owned in `Agents/agent_routing.py`: `subagent_default_provider`, `subagent_default_model`, `spawn_override_enabled` (false), `spawn_override_allowlist` ([]).
- Tool shaping, capability filtering, and pricing attribution need no new code: `provider_supports_native_tools` runs per run, the gateway consults `model_capabilities` per send, and `_budget_weighted_tokens` keys on each run's own provider/model (local children account as unpriced-local).
- Per-child rail summaries show the resolved target; routing failures name the failing level (override / preset / default).
- Stale allowlist slugs (deleted registry entries) are flagged in Settings UI, never silently dropped; a "Test routing" dry-run action reports each preset/default target's resolved provider/model/params and readiness.
