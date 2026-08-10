---
id: TASK-3600
title: Console model dropdown offers retired Anthropic models while the catalog cache
  holds the current set
status: In Progress
assignee:
- '@codex'
created_date: 2026-08-07 22:06
labels:
- console
- models
dependencies: []
updated_date: 2026-08-10 21:46
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during RAG-port P0's live walkthrough (task-11 report): the Console model dropdown lists models the Anthropic API now 404s (claude-3-haiku-20240307, claude-3-5-haiku-20241022) while the app's own model_catalog_cache.json for that endpoint holds the current set; sending on a listed retired model yields a bare "provider returned HTTP 400".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dropdown reflects the current catalog for the configured endpoint
- [x] #2 Sending on a retired model yields an actionable error naming the model, not a bare HTTP 400
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-020 to define endpoint-authoritative selector reconciliation while preserving the current session model and offline fallback.
2. Add a catalog snapshot/resolution contract that distinguishes endpoint-confirmed, saved-only, and current-not-listed models without treating absence as definitive retirement.
3. Make Console selectors prefer the current endpoint snapshot and retain saved models only when no endpoint snapshot is available.
4. Improve provider rejection copy so it names the selected model and provider or endpoint.
5. Add mutation-resistant resolver, selector, and error-copy regression tests; run targeted tests and Ruff.

ADR required: yes
ADR path: backlog/decisions/020-automatic-model-catalog-refresh.md
Reason: This changes saved-versus-live selector authority and the service contract governed by ADR-020.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented endpoint-authoritative Console model selection under amended ADR-020. The catalog merge now preserves endpoint-confirmed provenance, distinguishes empty snapshots from missing snapshots, filters saved-only cloud history, preserves an unlisted active session model, and keeps local/manual catalogs additive. Provider HTTP 400 copy now names the provider and selected model and directs users back to the model picker without exposing raw exception details. Added resolver, cache/service, modal handoff, picker, provenance, and gateway regressions, including a mutation check that failed when the saved-history filter was disabled and passed after restoration. Isolated verification: 85 catalog/resolver/picker tests, 144 gateway tests, and 49 model-focused Console modal tests passed; scoped Ruff, fatal-error Ruff for the legacy screen module, and `git diff --check` passed. Repository architecture checks remain red on baseline issues outside this change: `chat_screen.py` is 19,743 lines against a 17,727-line ratchet while this patch removes three lines, and the blocking-I/O suite reports eight stale entries for retired Chatbooks screens. Task remains In Progress until repository-wide verification is green.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
