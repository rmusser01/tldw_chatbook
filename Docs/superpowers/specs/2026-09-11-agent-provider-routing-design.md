# Agent Provider Routing — Design Spec

Date: 2026-09-11
Status: Approved design (pre-plan)
Governance: ADR required (provider/runtime boundary + cross-module interface);
the ADR also **amends ADR-146** (registry entries gain a `params` table —
precedent: ADR-020 amends ADR-002). Backlog task to be created at plan time.

## Problem

A Console agent run resolves exactly one `ConsoleProviderSelection` at send
time (`build_console_provider_selection_from_settings`,
`Chat/console_chat_controller.py:288`), and everything the run spawns inherits
it. `AgentDefinition.model` (`Agents/agent_models.py:375`) can only swap the
model *on the same provider endpoint*. There is no way for a master/control
agent to run on one backend (e.g. Kimi API) while delegating implementation
work to a child on another (e.g. a local qwen3.8-27B served via llama.cpp,
ollama, or a `custom-ep:` registry entry from ADR-146 / PR #2617).

## Locked decisions (from brainstorming)

1. **Mechanism: presets primary, ad-hoc args as gated override.** Named agent
   presets carry routing; `spawn_subagent` gains optional ad-hoc args honored
   only under a user-controlled flag.
2. **Default: explicit sub-agent default setting.** A new
   `[agents] subagent_default_provider/model`; when unset, children inherit
   the parent's provider+model (today's behavior).
3. **Override scope: flag + allowlist.** Ad-hoc args require
   `[agents] spawn_override_enabled = true` AND the requested provider must be
   in `[agents] spawn_override_allowlist`. Presets are user-authored and
   therefore unrestricted by the allowlist.
4. **Children never inherit the parent's sampling/API params.** A child's
   params are built fresh for its resolved provider+model — uniform rule,
   including the plain-inherit path (behavior change, see "Sampling / API
   params").
5. **Param ownership: endpoint defaults + preset overrides (D).** Registry
   entries gain an optional `params` table (endpoint-owned tuning);
   `AgentDefinition` presets gain optional `params` applied above everything
   else (role-owned tuning).

## Goals

- A master agent can spawn children onto a different provider+model than its
  own, via named presets or (when enabled) ad-hoc args.
- The user owns the routing table and the params: presets, endpoint entries,
  the sub-agent default, the override flag, and the allowlist are all
  user-configured.
- Cross-provider routing reuses the existing readiness/identity/execution
  seams, including `custom-ep:` registry providers.
- Routing decisions are loud, persisted, and visible — never silent fallbacks.

## Non-goals

- Ad-hoc sampling params in spawn tool args (params come only from
  user-authored stores: presets, registry entries, config — never from
  model-generated args).
- Named route registry as a separate config entity (rejected approach 2).
- Policy-file rules engine (rejected approach 3).
- Per-provider budget accounting; unpriced local models keep the existing
  "unpriced keeps previous accounting" behavior
  (`Agents/agent_service.py:_budget_weighted_tokens`). Per-child pricing
  attribution needs no work: that function already keys on each run's own
  provider/model, so a local child is accounted as unpriced-local while a
  cloud master bills at cloud rates.
- Any change to how the top-level (session) provider+model is chosen.

## Architecture

New **pure** module `Agents/agent_routing.py` (preserving the rule that
`agent_service.py` is the only impure Agents module). It owns:

- Config-dataclass defaults for the new `[agents]` keys (mirroring how
  `DEFAULT_MAX_LIVE_SUBAGENTS` etc. live in `agent_service.py`), so the
  shipped `config.py` template keeps the keys commented out.
- The resolver:

```
resolve_spawn_target(app_config, parent_selection, preset, override_args, agents_config)
    → SpawnTarget | RoutingError
```

`SpawnTarget` carries `provider`, `model`, `base_url` (already resolved,
including registry lookup for `custom-ep:` ids), the fully resolved sampling
params (see below), plus a `source` tag (`override` / `preset` / `default` /
`inherit`) for logging and persistence.

### Resolution order (provider + model)

Each level fills only blanks left by the levels above:

1. **Ad-hoc spawn args** (`provider`, `model` from the spawn tool call).
   Honored only when `spawn_override_enabled` is true; when `provider` is
   given it must appear in `spawn_override_allowlist`. A model-only ad-hoc
   arg swaps the model on whatever provider levels 2–4 resolve.
2. **Preset routing fields** — the spawned `AgentDefinition`'s `provider`
   and `model`.
3. **Sub-agent default** — `[agents] subagent_default_provider` /
   `subagent_default_model`.
4. **Inherit parent** — the parent's provider+model (NOT its params; see
   below).

The winner is validated through the existing seams:
`resolve_console_provider_identities` (`Chat/console_provider_support.py`)
for display/readiness/execution keys (custom-ep aware), then readiness
(missing credential / no reachable endpoint → `RoutingError`), producing a
full `ConsoleProviderSelection` for the child. Tool shaping needs no change:
`provider_supports_native_tools` is already evaluated per run inside
`_make_call_model` (`agent_service.py:1314,1565`). Unknown local models fall
back to default token-window handling, same as any unlisted model today.

`AgentService.spawn()` (`agent_service.py:3115`) calls the resolver **before**
reserving fleet capacity; a `RoutingError` is returned to the master as the
spawn tool's error result and no fleet slot is consumed.

### Sampling / API params

A child **never** inherits sampling or API params (temperature, top_p, min_p,
top_k, max_tokens, seed, presence/frequency penalty, reasoning/verbosity/
thinking knobs, streaming) from the parent's `ConsoleProviderSelection`. Its
params are built fresh for its resolved provider+model through the same
layering a brand-new Console session to that provider would get
(`build_default_console_session_settings`,
`Chat/console_session_settings.py:509`), extended with the two new sources.
Precedence, highest first:

1. **Preset `params`** — only when the spawn names a preset that sets them.
2. **Per-model profile** — `[api_settings.<provider>].model_defaults.<model>`.
3. **Console saved defaults** — `[console.provider_defaults.<provider>]`.
4. **Registry entry `params`** — `[custom_endpoints.<slug>.params]`, projected
   into the provider-settings layer (custom-ep targets only; slots above the
   global chat_defaults because endpoint-authored tuning is more specific
   than global defaults).
5. **Global `[chat_defaults]`**.
6. **Raw `[api_settings.<provider>]` scalars**, then function fallbacks
   (0.7 / 0.95 / …).

This is a **uniform rule**: even a plain same-provider child gets fresh
provider-default params rather than the parent's session-tweaked values.
**Behavior change:** today a plain spawn inherits the session's live sampling
params; after this change it gets the provider's configured defaults. Called
out in the ADR and the user guide.

The merged params pass through the same capability filtering as a fresh send
(`model_capabilities`), so provider-specific knobs (e.g. a Kimi
`reasoning_effort`) never leak onto a model that doesn't support them; a
dropped knob is noted in the run log. Oversized `max_tokens` for a small
local model is handled by the existing `repair_request_fits_model_window`
path. Ad-hoc spawn args carry **no** params (see Security policy).

## Data model changes

### `AgentDefinition` (`Agents/agent_models.py`)

- New field `provider: str = ""`. Semantics of the existing `model` field
  widen from "same endpoint only" to "model on the preset's provider".
  `model` set **without** `provider` keeps today's exact same-endpoint
  behavior — existing presets are unaffected.
- New field `params`: optional mapping of sampling-param name → value.
  Validated against the known param-name set (typo guard) with per-key type
  checks; unknown keys are validation errors.
- No `base_url` field: built-in providers resolve URLs from `api_settings`;
  additional local endpoints are `custom-ep:` registry entries (their
  entries carry URL + credentials, ADR-146). Presets name the slug.
- `validate_agent_definition` additionally checks `provider` is a known
  built-in provider id or matches the `custom-ep:<slug>` form. Registry
  existence is **not** checked at validation time (entries can be deleted
  between authoring and spawn); that surfaces as a spawn-time
  `RoutingError` instead.
- The definition dict serialization (`agent_models.py:417`) gains `provider`
  and `params`.

### Registry entries (`Chat/custom_endpoint_registry.py`, amends ADR-146)

- `CustomEndpointEntry` gains `params`: an optional mapping loaded from a
  `[custom_endpoints.<slug>.params]` TOML sub-table; validated with the same
  known-name/type rules as preset params. Secret-handling unchanged
  (`api_key` stays repr-excluded; params hold no secrets by validation).
- `build_entry_mutation` / `load_custom_endpoints` round-trip the table;
  entries without it behave exactly as today.

### Config `[agents]` (template at `config.py:3046`)

New keys, shipped commented-out with defaults owned by
`Agents/agent_routing.py`:

- `subagent_default_provider` (string, empty = unset → inherit)
- `subagent_default_model` (string, empty = provider's configured/default model)
- `spawn_override_enabled` (bool, default false)
- `spawn_override_allowlist` (list of provider ids, may include
  `custom-ep:<slug>` entries; default empty)

### Database (`DB/AgentRuns_DB.py`, `_CURRENT_SCHEMA_VERSION` 15 → 16)

One version step adding:

- `agent_definitions.provider TEXT NOT NULL DEFAULT ''`
- `agent_definitions.params_json TEXT NOT NULL DEFAULT '{}'`
- `agent_runs` resolved-target snapshot columns (nullable):
  `resolved_provider`, `resolved_model`, `resolved_base_url`,
  `resolved_params_json` — the full merged param set, so a resumed child is
  stable even if the preset, the registry entry, or config defaults were
  edited after the spawn.

### Resume / continuation semantics

Today fleet continuation re-resolves the agent definition *by name*, so
editing a preset between finish and resume would silently move the child to
a different provider mid-conversation. Fix: the resolved target — provider,
model, base_url, and merged params — is persisted on the run row at spawn;
continuation/resume reuses the persisted snapshot. Live re-resolution is the
fallback only for legacy rows with NULL snapshot columns. (Note the intended
consequence: editing a built-in provider's URL or an endpoint's params does
not move an already-spawned child — snapshot semantics.)

## Spawn tool contract & master visibility

- `spawn_subagent` gains optional string args `provider` and `model` — and
  nothing else; no params, no URLs.
- The args are **omitted from the tool schema entirely** when
  `spawn_override_enabled` is false — the model cannot attempt what is not
  advertised. Schema construction (identity-path schema referenced at
  `agent_service.py:1170`) becomes config-dependent.
- When enabled, the schema description enumerates the allowlisted provider
  ids with their configured models (from `api_settings` and registry cached
  `CustomEndpointEntry.models`), so the master chooses real targets instead
  of guessing.
- Named-agent enumeration (existing) gains each preset's routing in its
  description, e.g. "runs on custom-ep:qwen-local / qwen3.8-27b".

## Security policy

- Spawn args are model-generated and therefore untrusted: `provider` accepts
  only a known built-in provider id or a `custom-ep:<slug>` matching the
  registry slug pattern. **Ad-hoc args never carry URLs or params** —
  endpoints and sampling come only from user-controlled config/registry/
  presets, so a prompt-injected master can neither point a child at an
  arbitrary endpoint to exfiltrate context nor crank params (e.g. max
  reasoning budget) on a paid provider.
- The allowlist is the cost/injection guard for ad-hoc routing; presets are
  trusted because the user authored them.
- No credential material flows through the resolver's outputs beyond what
  the existing gateway already handles; registry entries' `api_key` is
  repr-excluded upstream (ADR-146).

## Error handling

`RoutingError` variants, always surfaced as the spawn tool's error result so
the master can pick another target or ask the user:

- `override_disabled` — ad-hoc args present while the flag is off
  (defense-in-depth; the args are also absent from the schema).
- `provider_not_allowlisted` — ad-hoc `provider` not in the allowlist.
- `unknown_provider` — id matches no built-in provider.
- `unknown_endpoint_slug` — `custom-ep:<slug>` not in the registry.
- `provider_not_ready` — missing credential or no reachable endpoint.
- `no_model_resolved` — fires only when routing selected a provider other
  than plain inherit (levels 1–3) and every level left model blank and the
  target provider has no configured/default model. The inherit path keeps
  today's behavior exactly (a blank model means the endpoint's server-side
  default, as now) and never raises this error.

Each error names the failing level (override / preset / default). There is
**no** automatic cross-provider fallback. The rail summary shows each
child's resolved target (e.g. "qwen-local · qwen3.8-27b").

## Settings UI

`Widgets/settings_agents_panel.py` gains:

- Per-preset `provider` / `model` pickers, fed by the same provider-option
  builder as Console settings so `custom-ep:` entries appear with their
  `display_name`, plus per-preset `params` editing (the known sampling keys).
- Sub-agent default provider/model pickers.
- `spawn_override_enabled` toggle and an allowlist multi-select.
- Stale allowlist entries (deleted registry slugs) are flagged in the UI,
  not silently dropped from config.

The endpoint editor (`Widgets/Console/console_endpoint_template_modal.py`)
gains an optional `params` section for the same known sampling keys.

## Testing

- **Unit:** pure resolver matrix — every resolution level, blank-filling,
  each `RoutingError` variant, custom-ep slug resolution, model-only ad-hoc
  args.
- **Unit (params):** the six-layer precedence stack, incl. preset-over-entry
  and entry-over-chat_defaults ordering; unknown/mistyped param keys rejected
  for both presets and registry entries; capability filtering drops
  unsupported knobs with a run-log note.
- **Integration:** spawn through `AgentService` with routed preset, with
  sub-agent default, and with override on/off; assert the child's
  `ConsoleProviderSelection` (provider, model, base_url, **and merged
  params**), the persisted snapshot columns, and that refusals return an
  error result without consuming a fleet slot.
- **Migration tests** for schema v16: column defaults, legacy rows unchanged.
- **Backward-compat regression:** existing preset with `model` but no
  `provider` keeps same-endpoint override behavior; registry entries without
  `params` load unchanged.
- **Settings panel/modal tests:** picker populations include custom-ep
  display names; stale allowlist slug flagging; params editing round-trips.
- **Live verification** (per `backlog/docs/lessons-live-verification.md`):
  real local endpoint registered as `custom-ep:` with its own `params`,
  cheap cloud model as master, one two-step delegation exercised end to end,
  confirming in the run log that the child's request carried the endpoint's
  params, not the master's.

## Expected touch list

- New: `Agents/agent_routing.py`, spec/plan docs, ADR.
- Modified: `Agents/agent_models.py`, `Agents/agent_service.py` (spawn hook,
  schema gating), `DB/AgentRuns_DB.py` (v16 migration), `config.py`
  (template comments), `Chat/custom_endpoint_registry.py` (entry params),
  `Widgets/settings_agents_panel.py`,
  `Widgets/Console/console_endpoint_template_modal.py` (params editing),
  `Chat/console_agent_bridge.py` (rail summary target display).
- Tests: new `Tests/Agents/test_agent_routing.py` plus integration,
  migration, registry, and settings-panel coverage.

## Rollout

1. Create ADR in `backlog/decisions/` (amending ADR-146 for entry `params`)
   and Backlog task; link both ways.
2. Implement per the writing-plans output (pure resolver + params stack
   first, then DB migration, then spawn integration, then UI).
3. Update `Docs/User_Guide/console/agent-runs-and-tools.md` with the new
   `[agents]` keys, preset routing/params fields, and the plain-spawn
   params behavior change.
