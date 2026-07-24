---
id: TASK-401.7
title: Persist bounded citation source observations
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 11:08'
labels:
  - rag
  - citations
  - resolvers
  - database
dependencies:
  - TASK-401.4
  - TASK-401.3
  - TASK-401.5
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Store the latest current-source availability, permission, content, location, and capability observation without mutating sealed history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Observations are keyed by trace, prompt set, evidence ordinal, opaque snapshot reference, and resolver and replace only the latest bounded value.
- [ ] #2 Availability, permission, content state, location state, capability, and observed time remain distinct.
- [ ] #3 Observation writes cannot modify completeness at seal, submitted snapshots, or historical locators.
- [ ] #4 Stale, revoked, offline, authentication-required, ambiguous, and unavailable observation tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing strict observation-state and contradictory/boundary contract tests.
2. Add failing repository keying, stale-write, nonce, rerun prompt-set isolation, and no-history-accumulation tests.
3. Implement immutable bounded CitationSourceObservation plus compare-and-replace upsert/read without touching sealed trace, historical locator, prompt, completeness, or snapshot data.
4. Run focused observation/repository/lifecycle plus adjacent persistence/migration/benchmark regressions, lint, diff, and both independent review gates.
5. Complete acceptance criteria and implementation notes only after approval.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This task implements ADR-024’s separate mutable current-source observation contract over the existing v25 observation table without changing sealed provenance.
<!-- SECTION:PLAN:END -->
