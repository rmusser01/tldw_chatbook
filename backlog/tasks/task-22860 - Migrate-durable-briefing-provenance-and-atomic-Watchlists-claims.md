---
id: TASK-22860
title: Migrate durable briefing provenance and atomic Watchlists claims
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:16'
labels:
  - watchlists
  - briefings
  - database
  - migration
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-watchlists-agent-boundary-and-provenance.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve ordered briefing evidence independently of mutable source and item rows, and enforce database-backed single-active claims for source checks and briefing generation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A versioned, transactional Subscriptions migration stores ordered briefing-item snapshots including item/source identity, featured/cited positions, sanitized URLs, dates, and a provenance format version.
- [ ] #2 Existing junction rows migrate as `legacy_best_effort` without inventing selection or citation order; newly completed briefings write `ordered_snapshot` provenance before publishing `complete`.
- [ ] #3 Completed briefing provenance remains readable and in original order after referenced source/item edits or deletion, while nullable live links may still expose current supplemental state.
- [ ] #4 Partial unique indexes enforce at most one queued/running source-check receipt per source and one generating briefing per collection across threads/processes.
- [ ] #5 Migration reconciliation deterministically keeps the newest active receipt and terminalizes older duplicates with fixed, non-sensitive recovery state.
- [ ] #6 Owner-level accept/transition APIs resolve uniqueness races to the winning durable receipt and release claims on every terminal transition.
- [ ] #7 Migration rollback, legacy upgrade, deletion survival, duplicate reconciliation, idempotent reopen, and the complete Subscriptions migration suite pass against temporary databases only.
<!-- AC:END -->
