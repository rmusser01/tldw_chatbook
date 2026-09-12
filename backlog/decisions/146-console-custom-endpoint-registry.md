# ADR-146: Console custom endpoint registry

Status: Accepted
Date: 2026-09-10
Related Task: [TASK-32308](../tasks/task-32308%20-%20Console-custom-endpoint-registry-named-endpoints-from-templates.md)
Related Spec: [Console custom endpoint registry design](../../Docs/superpowers/specs/2026-09-10-console-custom-endpoint-registry-design.md)

## Context

Chatbook could address only two user-defined OpenAI-compatible endpoints: the hardcoded `custom` and `custom_2` provider slots. The 2026-09-10 Conversation Settings UX review (issue P0-4) documented the resulting dead end — picking a custom provider and typing a new base URL trips the "endpoint is not saved" send block, and the only durable homes for a new URL are the two anonymous slots or hand-editing `config.toml`. Users running several local or remote servers (a localhost llama.cpp, a LAN GPU box, a rented OpenAI-compatible endpoint) cannot name, create, or clone endpoint entries. Task-32307 landed recovery-copy stopgaps; this decision records the registry that removes the underlying limitation.

## Decision

- Registry entries persist in a new top-level config table, `[custom_endpoints.<slug>]`, outside `api_settings`, so provider-key iteration (readiness, catalog, gateway) is untouched by registry plumbing. Each entry carries `display_name`, `family` (`llama_cpp` | `openai_compatible` | `ollama`), `base_url`, optional credentials, an optional cached `models` list, and informational `created_from`. Slugs are lowercase `[a-z0-9-]`, 1-64 characters (`^[a-z0-9-]{1,64}$`), derived from the display name at creation with a uniquifying suffix, and immutable: rename changes the display name only. Config access goes through a pure registry module with validation (blank display names, invalid slugs, unknown families, `validate_url` after family-appropriate normalization) and writes through `save_settings_to_cli_config` off-thread.
- A registry entry is a provider instance in the picker and an alias at execution. Its provider id is `custom-ep:<slug>`; `provider_config_key` and `resolve_console_provider_identity` resolve that prefix onto the entry's family execution key with the endpoint pinned. The session `base_url` is the endpoint carrier: `ConsoleSessionSettings.base_url` resolves from the entry at build time and is re-resolved on send through the gateway's existing `effective_provider_endpoint(provider_key, selected_endpoint, settings)` seam, so a renamed or edited entry's URL flows into in-flight sessions on their next send. No new wire families are introduced; endpoint values are interpreted by the existing `provider_endpoint_contract`, including form detection and the remote-HTTP warning.
- Credentials mirror existing provider semantics with `api_key_env` taking precedence over the stored `api_key` (env reference, then stored key, then none) and the same at-rest handling; display paths use `safe_endpoint_display` only, so secrets never appear in UI copy or logs.
- Template scope is limited to the family, the endpoint, and the model list. Sampling and generation settings are never copied; they remain governed by the existing per-provider defaults chain.
- The built-in `custom` and `custom_2` slots are retained and keep working unchanged. Each gains an optional one-way "Convert to named endpoint" action that creates a registry entry from the slot's config and leaves the slot itself intact. No migration is performed; the `custom_endpoints` table is absent until a user creates an entry.
- At seams without `app_config` access, an unresolvable `custom-ep:` prefix falls back to the family `custom` execution key, matching how other unresolvable provider ids degrade.

## Alternatives

- Slot multiplication (lazily materializing `custom_3 … custom_N` sections) was rejected: it perpetuates anonymous slots, hardcodes each new key into execution-key sets, endpoint persist keys, and alias maps, and config sprawl grows linearly with endpoint count.
- A settings-only endpoint manager (F9 exclusive, modal only picks) was rejected as the sole surface because it reintroduces the context switch the review flagged; it is retained as half of the chosen approach — creation happens in the modal where the need arises, full management lives in F9 Settings.
- Storing entries under `api_settings` was rejected: every provider-key loop (readiness, catalog, gateway) would have to learn and filter registry slugs, coupling unrelated surfaces to registry plumbing.

## Consequences

- Entries are persisted at creation, so selecting one can never trigger the unsaved-endpoint block: `console_session_endpoint_survives_restart` is true for `custom-ep:` providers by construction, retiring the session-only toast for them.
- Registry entries render as first-class provider options — own id, own model list — labeled with `display_name`, appended after the built-in Custom & legacy group, ordered by creation time then display name.
- Readiness for an entry is the family's readiness (API-key rules included); URL validity is checked against the entry, not a config alias.
- F9 Settings gains a Custom endpoints section with rename (display name only), edit URL/credentials (existing sessions re-resolve on next send), and delete guarded while any known conversation references the entry, with detach-to-conversation-only-endpoint as the recovery path.
- Registry load failures (malformed TOML section) disable the registry surface for that boot with one warning naming the offending slug; built-in providers are unaffected. Discovery failures report but never roll back a created entry.
