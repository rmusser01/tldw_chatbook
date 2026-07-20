# ADR-020: Automatic model catalog refresh for cloud providers

Status: Accepted
Date: 2026-07-17
Related Task: [backlog/tasks/task-301 - Auto-refresh-model-catalogs-for-cloud-providers.md](../tasks/task-301%20-%20Auto-refresh-model-catalogs-for-cloud-providers.md)
Supersedes: N/A (amends ADR-002)

## Decision

Cloud-provider model lists (OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, Mistral)
auto-refresh on app startup through the ADR-002 discovery pipeline: fetched models
persist to a disk-backed TTL cache and merge into selectors (capped; oversized
catalogs reachable via a search picker). A per-provider opt-in write-through
appends new model IDs to `[providers]` in config.toml (append-only; oversized
first fetch establishes a baseline without appending).

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

## Consequences

- Startup performs bounded background network I/O (per-provider, 10s timeout, stale-after 24h default); failures degrade to cached/saved models and are surfaced via one consolidated end-of-refresh notification.
- Write-through is append-only and never removes models; users prune `[providers]` themselves.
- `model_catalog_cache.json` under the user data dir stores model IDs + timestamps only (no credentials).
- Manual Discover/Save/Clear flows from ADR-002 remain unchanged.

## Links

- Spec: Docs/superpowers/specs/2026-07-17-model-catalog-auto-refresh-design.md
- Plan: Docs/superpowers/plans/2026-07-17-model-catalog-auto-refresh.md
- Amends: backlog/decisions/002-openai-compatible-model-discovery.md
