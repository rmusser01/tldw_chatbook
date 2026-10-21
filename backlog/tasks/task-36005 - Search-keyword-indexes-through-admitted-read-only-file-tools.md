---
id: TASK-36005
title: Search keyword indexes through admitted read-only file tools
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-21 14:32'
labels: []
dependencies:
  - TASK-36004
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Accelerate project discovery while keeping current files and the requesting run authoritative.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Only compatible readable local bindings advertise fs_search_index; strict bounded inputs return at most10 live hits and16KiB output with workspace-file provenance.
- [ ] #2 Run exclusion floors and current exclusions filter names and counts before ranking; current identity/hash and final cross-process authority acceptance prevent stale disclosures.
- [ ] #3 Index Allowed cannot bypass fs_read Blocked or Ask, existing path/instruction/review/budget guards, or a removed binding.
- [ ] #4 Saved Library choices/scopes and ordinary file tools remain unchanged; keyword lookup needs no embeddings/network and cached candidate p95 is measured against500ms.
<!-- AC:END -->
