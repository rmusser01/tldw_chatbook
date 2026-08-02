---
id: TASK-1822
title: Refresh stale production diagnostic inventory on dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-02 02:15'
updated_date: '2026-08-02 02:15'
labels:
  - testing
  - baseline
  - security
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the repository architecture gate by reviewing current production diagnostic ownership and persistent-sink topology against ADR-029, then recording the accepted current baseline without changing runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every production diagnostic owner changed since the reviewed baseline is inspected for metadata-only safety under ADR-029.
- [ ] #2 Persistent sink topology is verified unchanged or any change is explicitly reviewed and documented.
- [ ] #3 The checked production diagnostic inventory exactly matches current dev source.
- [ ] #4 The focused diagnostic-inventory architecture tests pass.
- [ ] #5 No production runtime behavior changes are introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: This refresh applies the existing metadata-only diagnostic boundary and records current ownership; it makes no new architecture, sink, storage, or security decision.

1. Reproduce the stale-inventory failure and generate a deterministic current inventory for comparison.
2. Review every changed owner entry and the corresponding source diagnostics against ADR-029; confirm persistent-sink topology is unchanged.
3. Replace only the reviewed inventory artifact.
4. Run the focused architecture tests, inventory checker, JSON validation, and diff hygiene.
5. Record verification and review results, check all acceptance criteria, and close the task.
<!-- SECTION:PLAN:END -->
