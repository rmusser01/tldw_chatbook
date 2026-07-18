---
id: TASK-226
title: Auto-refresh model catalogs for cloud providers
status: In Progress
assignee: []
created_date: '2026-07-18 05:43'
updated_date: '2026-07-18 05:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh model lists for OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, and Mistral on app startup via the LLM_Provider_Catalog discovery pipeline. Disk-backed TTL cache feeds selectors (capped merge + search picker); per-provider opt-in write-through appends new models to [providers]. Amends ADR-002; see ADR-014 and Docs/superpowers/specs/2026-07-17-model-catalog-auto-refresh-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Startup auto-refresh fetches stale providers in a background worker without blocking app start
- [ ] #2 Fetched small catalogs (<=50) merge into chat selector and survive restarts
- [ ] #3 Large catalogs stay saved-list-only in dropdown and are searchable via picker
- [ ] #4 Write-through is append-only with baseline guard for oversized first fetch
- [ ] #5 Global toggle off / opt-out / no key / fresh cache each skip fetching
- [ ] #6 Failures degrade to cached models with at most one notification
- [ ] #7 Anthropic uses x-api-key + pagination; Z.AI ships config defaults
- [ ] #8 ADR-014 linked; tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-07-17-model-catalog-auto-refresh.md
ADR: backlog/decisions/014-automatic-model-catalog-refresh.md (amends ADR-002)
<!-- SECTION:PLAN:END -->
