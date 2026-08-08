---
id: TASK-3600
title: >-
  Console model dropdown offers retired Anthropic models while the catalog cache
  holds the current set
status: To Do
assignee: []
created_date: '2026-08-07 22:06'
labels:
  - console
  - models
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during RAG-port P0's live walkthrough (task-11 report): the Console model dropdown lists models the Anthropic API now 404s (claude-3-haiku-20240307, claude-3-5-haiku-20241022) while the app's own model_catalog_cache.json for that endpoint holds the current set; sending on a listed retired model yields a bare "provider returned HTTP 400".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dropdown reflects the current catalog for the configured endpoint
- [ ] #2 Sending on a retired model yields an actionable error naming the model, not a bare HTTP 400
<!-- AC:END -->
