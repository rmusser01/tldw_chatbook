# ADR-020: Automatic model catalog refresh for cloud providers

Status: Accepted (amended 2026-08-14: consent-gated startup)
Date: 2026-07-17
Related Tasks:
- [backlog/tasks/task-301 - Auto-refresh-model-catalogs-for-cloud-providers.md](../tasks/task-301%20-%20Auto-refresh-model-catalogs-for-cloud-providers.md)
- [backlog/tasks/task-3600 - Console-model-dropdown-offers-retired-Anthropic-models-while-the-catalog-cache-holds-the-current-set.md](../tasks/task-3600%20-%20Console-model-dropdown-offers-retired-Anthropic-models-while-the-catalog-cache-holds-the-current-set.md)
Supersedes: N/A (amends ADR-002)

## Decision

Cloud-provider model lists (OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, Mistral)
auto-refresh on app startup through the ADR-002 discovery pipeline: fetched models
persist to a disk-backed TTL cache and merge into selectors (capped; oversized
catalogs reachable via a search picker). A per-provider opt-in write-through
appends new model IDs to `[providers]` in config.toml (append-only; oversized
first fetch establishes a baseline without appending).

For those auto-refreshed cloud providers, an endpoint-scoped snapshot is
authoritative for new selector choices while it is present in the runtime
cache. Saved IDs confirmed by the snapshot remain selectable; saved-only IDs
do not remain in the ordinary selector merely because append-only persistence
retained them. The active session model is preserved even when it is absent
from the snapshot so opening settings never silently changes a running
conversation. When no endpoint snapshot is available, selectors fall back to
the saved list. This authority rule does not change ADR-002's additive merge
for manually discovered local providers.

## Context

ADR-002 kept discovery manual and persistence explicit-only. Users want fresh
model lists without manual steps. ADR-002's runtime-cache-first and scoped-design
constraints still hold; this amends only the "no auto-save" consequence, and only
as an opt-in. OpenRouter's catalog is public (no key required).

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Silent full rewrite of `[providers]` | Clobbers hand-curated lists; contradicts ADR-002's core stance |
| Standalone ModelRefreshService parallel to the catalog | Duplicates fetch/merge/persist; ADR-002 forbids a parallel registry |
| Bundled static model list updated with releases | Recreates the manual-update problem upstream |
| Treat append-only `[providers]` entries as permanently selectable | Makes a historical persistence log override the current endpoint catalog and keeps retired models in the Console picker |
| Remove a current session model when it disappears from the snapshot | Silently mutates conversation state and prevents unrelated settings edits; the current value must remain visible until the user chooses another model |

## Consequences

- Startup performs bounded background network I/O (per-provider, 10s timeout, stale-after 24h default); failures degrade to cached/saved models and are surfaced via one consolidated end-of-refresh notification.
- Write-through is append-only and never removes models; users prune `[providers]` themselves.
- Append-only persistence is durable configuration history, not permanent selector authority. For auto-refreshed cloud providers, a cached endpoint snapshot filters new choices while the current session value remains preserved.
- `model_catalog_cache.json` under the user data dir stores model IDs + timestamps only (no credentials).
- Manual Discover/Save/Clear flows from ADR-002 remain unchanged.

## Amendment (2026-08-14): confirm-first startup consent

The startup refresh is no longer silent-by-default. A new persisted setting,
`[model_catalog] refresh_consent_recorded` (default `false`), gates the refresh:
until the user answers a one-time dialog ("Check model lists online?"), no
provider endpoints are contacted. Allowing records consent (`true`) and runs the
refresh that session and thereafter; declining persists
`auto_refresh_enabled = false` alongside the recorded consent so the question is
never asked twice. Only an explicit boolean `true` counts as consent — garbage
values fall back to not-consented (privacy-safe). `_refresh_model_catalogs`
itself re-checks consent, so no code path can refresh unconsented. Recording
consent via the Settings toggle was considered and rejected: the toggle is saved
on unrelated edits in the same category (e.g. stale-hours keystrokes), so it is
not an unambiguous confirmation.

## Links

- Spec: Docs/superpowers/specs/2026-07-17-model-catalog-auto-refresh-design.md
- Plan: Docs/superpowers/plans/2026-07-17-model-catalog-auto-refresh.md
- Amends: backlog/decisions/002-openai-compatible-model-discovery.md
