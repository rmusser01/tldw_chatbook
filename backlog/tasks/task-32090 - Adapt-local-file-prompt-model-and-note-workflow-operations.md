---
id: TASK-32090
title: Adapt local file prompt model and note workflow operations
status: To Do
assignee: []
created_date: '2026-09-08 20:57'
labels:
  - workflows
  - adapters
dependencies:
  - TASK-32088
  - TASK-32089
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the five operation subsets needed for a real local file-to-note workflow through captured Chatbook services, with pinned server-shaped inputs and results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The supported media_ingest plaintext, prompt dotted-template, local llm, wait_for_human, and notes create subsets have explicit config, output, and error fixtures pinned to server dev.
- [ ] #2 Production adapters use captured local services and deny unsupported operations, implicit provider fallback, test-mode simulated results, and unapproved file or note effects.
- [ ] #3 A note write has a stable effect identity and cannot create a second note on an uncertain retry; adapter payloads do not enter ordinary diagnostics.
- [ ] #4 The captured local model request preserves endpoint, keyless decision, timeout and retry policy, enforces bounded response reads, and refuses redirects or environment-proxy retargeting.
<!-- AC:END -->
