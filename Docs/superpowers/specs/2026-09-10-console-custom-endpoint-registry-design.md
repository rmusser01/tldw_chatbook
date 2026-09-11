# Console Custom Endpoint Registry — Design

- **Date:** 2026-09-10
- **Status:** Draft — pending user review
- **Origin:** 2026-09-10 Conversation Settings modal UX review (issue P0-4) and follow-up decision "(c) both, sequenced"
- **Related:** `Docs/superpowers/specs/2026-09-02-console-conversation-settings-ready-to-send-design.md` (TASK-30012 modal recomposition), `backlog/tasks/task-32307` (landed P0 stopgaps)
- **ADR required:** Yes. Storage schema, provider-boundary mapping, and credential handling are ADR-worthy decisions; the ADR is authored before implementation begins and linked from the Backlog task and implementation plan.

## Problem

Chatbook can only address two user-defined OpenAI-compatible endpoints: the
hardcoded `custom` and `custom_2` provider slots. Users who run several local
or remote servers (a localhost llama.cpp, a GPU box on the LAN, a rented
OpenAI-compatible endpoint) cannot name, create, or clone endpoint entries.
The 2026-09-10 UX review documented the resulting friction: picking a custom
provider and typing a new base URL in Conversation Settings dead-ends against
the "endpoint is not saved" send block, and the only durable homes for a new
URL are the two anonymous slots or hand-editing `config.toml`.

The landed stopgap (task-32307) corrected the recovery copy to name the actual
in-modal unblock ("Save model defaults"). This design removes the underlying
limitation: a named, user-extensible endpoint registry.

## Goals

1. Users can create an endpoint entry from any existing provider or endpoint
   as a template, give it a display name, and select it as a provider in the
   Console settings modal in one session, without leaving the modal or editing
   `config.toml`.
2. Registry entries are durable config: selecting one never triggers the
   "endpoint is not saved" wall, and sessions using them survive restart.
3. Templates carry the family's endpoint semantics (URL normalization, wire
   family, model discovery behavior) so a "llama.cpp GPU box" entry behaves
   exactly like the built-in llama.cpp provider, pointed at another origin.
4. Existing `custom` and `custom_2` config keeps working unchanged.
5. Secrets never appear in UI copy or logs (existing `safe_endpoint_display`
   discipline extends to registry surfaces).

## Non-goals

- Redesigning the settings modal's layout or save flow — that is the
  TASK-30012 connection-first recomposition. This design supplies the data
  model and resolution seams that the recomposed modal (and the current modal,
  until then) consume.
- Multi-account or multi-key management per cloud provider.
- Automatic background scanning for remote endpoints (localhost auto-discovery
  already exists and is extended only to the extent of the 9099 candidate
  landed in task-32307).

## Approaches considered

**A. Slot multiplication** — lazily materialize `custom_3 … custom_N` config
sections. Rejected: perpetuates anonymous slots, hardcodes each new key into
execution-key sets (`CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS`, endpoint
persist keys, alias maps), and config sprawl grows linearly with endpoints.

**C. Settings-only endpoint manager** — manage endpoints exclusively in F9
Settings; the modal only picks. Rejected as the sole surface: it reintroduces
the context switch the review flagged. Retained as half of the chosen approach:
creation happens in the modal (where the need arises), full management
(rename, delete, review) lives in F9 Settings.

**B. Named endpoint registry (chosen)** — a config-owned table of named
endpoint entries, each mapped at resolution time onto an existing execution
family. Entries render as first-class provider options; because they are
persisted at creation, the unsaved-endpoint block cannot apply to them.

## Data model

A new top-level config table, separate from `api_settings` so provider-key
iteration (readiness, catalog, gateway) is untouched by registry plumbing:

```toml
[custom_endpoints.llama-gpu-box]
display_name = "GPU box llama.cpp"
family = "llama_cpp"            # "llama_cpp" | "openai_compatible" | "ollama"
base_url = "http://192.168.1.5:8080"
api_key = ""                    # optional; mirrors existing provider semantics
api_key_env = "GPU_BOX_KEY"     # optional env-var name; wins over api_key
models = ["qwen3.8-27b"]        # optional cached discovery results
created_from = "llama_cpp"      # template provenance, informational
```

- **Slug** (`llama-gpu-box`): lowercase `[a-z0-9-]`, derived from the display
  name at creation with a uniquifying suffix; stable identity for references.
- **`family`** selects the execution path: `llama_cpp` rides the direct llama
  path (`uses_direct_llama_path`, session endpoint permitted); `ollama` adds
  the `/api/tags` discovery fallback; `openai_compatible` behaves as `custom`
  does today. No new wire families are introduced.
- **Endpoint values** are interpreted by the existing
  `provider_endpoint_contract` (`resolve_provider_endpoint`,
  `canonical_connection_identity`), including form detection
  (origin / api_base / chat_url / models_url) and the remote-HTTP warning.
- **Credentials** mirror existing provider settings precedence
  (env reference → stored key → none) and reuse the same at-rest handling;
  display paths use `safe_endpoint_display` only.

Config access goes through a new pure module (`Chat/custom_endpoint_registry.py`)
with load/validate/mutate helpers; writes go through
`save_settings_to_cli_config` with the same off-thread discipline the modal's
Save-as-default already uses. Validation rejects blank display names, invalid
slugs, unknown families, and endpoints that fail `validate_url` after
family-appropriate normalization (`normalize_llamacpp_base_url` for
`llama_cpp`).

## Provider identity and resolution

- A registry entry's provider id is `custom-ep:<slug>` (e.g.
  `custom-ep:llama-gpu-box`). `provider_config_key` and
  `resolve_console_provider_identity` learn this prefix: it resolves to the
  entry's family execution key with an endpoint pin.
- `build_console_provider_options` appends registry entries after the
  built-in Custom & legacy group, as their own visible run labeled with
  `display_name` (order: creation time, then display name).
- Session settings persist `provider = "custom-ep:<slug>"`;
  `ConsoleSessionSettings.base_url` resolves from the entry at build time and
  is re-resolved on send by the gateway's existing
  `effective_provider_endpoint(provider_key, selected_endpoint, settings)`
  seam, so a renamed/edited entry's URL flows into in-flight sessions on their
  next send (matching how configured endpoints behave today).
- Readiness for an entry is the family's readiness (API-key rules included);
  URL validity is checked against the entry, not a config alias.
- `console_session_endpoint_survives_restart` returns `True` for
  `custom-ep:` providers by construction (the entry is the persisted
  fallback), which retires the session-only toast for them.

## Creation from template (modal flow)

Inside Conversation Settings' Base URL area (current modal now; the
TASK-30012 recomposed connection panel later):

1. **New endpoint from template…** opens a picker listing: every provider
   option (built-ins included), existing registry entries (as "duplicate"),
   and a bare "OpenAI-compatible (blank)" starter.
2. The form pre-fills from the template: `family` (template's execution
   family; "blank" starts as `openai_compatible`), `base_url` (template's
   configured endpoint or the family default), optional `models` list. The
   user edits the name and URL. Sampling and generation settings are **not**
   copied — they remain governed by the existing per-provider defaults chain.
3. **Create** validates, writes the entry via the registry module, switches
   the modal's provider to the new `custom-ep:<slug>` value, and runs
   **Discover models** against the new URL (user-initiated probe, same
   timeout and honesty rules as the existing button; failure is reported, it
   does not undo creation).
4. Cancel at any point leaves config untouched.

Duplicate (template = existing entry) copies `display_name` (suffixed
"(copy)"), `family`, `base_url`, credential references, and `models`.

## Management (F9 Settings)

A "Custom endpoints" section under Providers & Models lists entries with
family, safe endpoint display, and model count; actions are Rename (display
name only — slugs are immutable), Edit URL/credentials (validated as above;
existing sessions re-resolve on next send), and Delete. Delete is blocked with
an actionable message while any known conversation references the entry;
the recovery path is "detach to conversation-only endpoint" (drops the
reference, keeps the session's current URL as a session-only base_url) or
switching the conversation's provider. The two built-in slots (`custom`,
`custom_2`) remain listed as before, each gaining a one-way
"Convert to named endpoint" action that creates a registry entry from the
slot's config; the slot itself is left intact.

## Error handling

- Registry load failures (malformed TOML section) disable the registry
  surface for that boot with one warning naming the offending slug; built-in
  providers are unaffected.
- Creation/edit validation errors surface inline in the modal form (same
  error-banner discipline as the current modal), never as partial writes.
- Discovery failures never roll back a created entry; they only report.

## Testing

- Pure unit tests: slug derivation/uniqueness, family resolution of
  `custom-ep:` ids through `resolve_console_provider_identity`, endpoint
  identity/normalization per family, restart-survival `True`, option
  ordering, validation rejections, load-failure containment.
- Mutation tests through the registry module backed by a temp config file
  (atomic rewrite, precedence env over stored key).
- Textual Pilot tests: template picker flow, create-and-switch, discover
  after create, convert-from-slot, delete guard with a referencing session.
- Regression: `custom` / `custom_2` behavior and all existing gateway
  resolution tests unchanged.

## Migration and rollout

No migration is performed. `custom`/`custom_2` keep working; the convert
action is optional and one-way. The `custom_endpoints` table is absent for
existing users until they create an entry.

## Decisions

| Decision | Choice |
| --- | --- |
| Endpoint == provider instance or URL alias? | Provider instance in the picker (own id, own models); alias at execution (family path + pinned endpoint) |
| Registry storage | Top-level `[custom_endpoints.<slug>]`, outside `api_settings` |
| Secret storage | Mirror existing provider semantics; `api_key_env` reference preferred, stored `api_key` supported |
| Template scope | family + endpoint + model list; sampling stays global |
| Built-in slots | Kept; optional one-way convert to named entry |
| Slug changes | Not supported (rename changes display name only) |
