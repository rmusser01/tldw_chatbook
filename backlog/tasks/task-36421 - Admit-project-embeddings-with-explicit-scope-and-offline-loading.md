---
id: TASK-36421
title: Admit project embeddings with explicit scope and offline loading
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-23 03:27'
labels: []
dependencies:
  - TASK-36420
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow project embeddings to use their own admitted folder lifetime while preserving Library defaults and honest optional-feature availability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Actual embedding constructor, activation and native worker entrances select the exact code-issued project scope before effects, refuse forged/stale/wrong-target or dead transfers, and never infer Library source owners.
- [ ] #2 Accepted async/native work retains the actual live parent and physical retirement; default Library scope behavior remains compatible.
- [ ] #3 Project local embedding loads use a complete existing local snapshot with offline-only loading; a missing model or optional dependency reports setup required without download or keyword failure.
- [ ] #4 Actual embedding factory success/retry/failure and propagated errors expose no raw credential-bearing endpoint or request secrets; only secret-free configuration/destination references are persisted.
- [ ] #5 Model, configuration, artifact and destination compatibility distinguishes same-dimension changes and never treats Library policy as project consent.
<!-- AC:END -->
