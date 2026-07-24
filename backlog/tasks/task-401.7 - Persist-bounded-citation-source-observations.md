---
id: TASK-401.7
title: Persist bounded citation source observations
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 12:54'
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
- [x] #1 Observations are keyed by trace, prompt set, evidence ordinal, opaque snapshot reference, and resolver and replace only the latest bounded value.
- [x] #2 Availability, permission, content state, location state, capability, and observed time remain distinct.
- [x] #3 Observation writes cannot modify completeness at seal, submitted snapshots, or historical locators.
- [x] #4 Stale, revoked, offline, authentication-required, ambiguous, and unavailable observation tests pass.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded mutable citation source observations separately from sealed provenance. Added strict independent-axis validation, exact keyed compare-and-replace persistence, current authorization filtering, trusted resolver binding across purge, explicit tombstone-timestamp revocation observations, and transactionally serialized read/revocation policy checks.

Files: tldw_chatbook/Chat/citation_source_locators.py, tldw_chatbook/Chat/citation_trace_repository.py, tldw_chatbook/Chat/citation_payload_lifecycle.py, and Tests/Chat/test_citation_source_observations.py. No lifecycle fixture edits.

Commits: ddc8ae890 (initial observation contract and persistence), b2c09189f (independent axes), 25a307552 (current policy hardening), and 0cbdc2dc7 (trusted revocation persistence and race safety).

Verification: 49 focused observation tests passed; 422 adjacent persistence, migration, adapter, and benchmark tests passed; Ruff check, Ruff format check, and git diff --check passed. Spec review and final quality review approved the implementation. Separately, the lifecycle suite produced 44 passes and 6 pre-existing fixed-clock GC failures; the failing lifecycle fixtures were not edited.

ADR required: yes. Existing ADR-024 applies; no new ADR was created because this implementation follows its separate mutable current-source observation contract without changing schema or ownership boundaries.
<!-- SECTION:NOTES:END -->
