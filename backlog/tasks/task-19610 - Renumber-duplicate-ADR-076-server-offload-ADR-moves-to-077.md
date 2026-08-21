---
id: TASK-19610
title: 'Renumber duplicate ADR-076: server-offload ADR moves to 077 (later claimant)'
status: To Do
assignee: []
created_date: '2026-08-21 11:30'
updated_date: '2026-08-21 11:30'
labels:
  - adr
  - backlog-hygiene
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dev carries two ADRs numbered 076: `076-library-lifecycle-progressive-disclosure.md` and `076-server-offloaded-scheduled-agent-tasks.md`. The trace (2026-08-21) establishes provenance and the mover per the repo's add-commit rule:

- **Library-lifecycle 076** landed first: commit `1c567f3ae`, 2026-08-20 **14:24**.
- **Server-offload 076** landed second: commit `222379065`, 2026-08-20 **18:44** — the PR #1832 merge-window renumber (072→076) checked that 072–075 were taken but missed that 076 had been claimed ~4 hours earlier.

**Decision per the keeper rule** (later claimant moves, regardless of reference count): the **server-offload ADR renumbers 076→077** (verified free: nothing above 076 on dev). The library ADR keeps 076 — it is earlier and carries ~13 referencing files (tasks 19022–19025, three superpowers plans, more), but provenance is the rule, not weight.

Scope of the renumber (grep both forms — per the ADR-collision lesson in this same PR):
1. Rename `backlog/decisions/076-server-offloaded-scheduled-agent-tasks.md` → `077-server-offloaded-scheduled-agent-tasks.md`
2. Its `# ADR-076:` header → `# ADR-077:`
3. The README index row (currently links the 076 file)
4. TASK-18940's plan-section references (path + number; it already carries a renumber note from 072→076 — update to record the full chain 072→076→077)
5. Any other `076-server-offloaded` slug references (verify with `git grep`; scope `ADR-076` greps carefully — bare `ADR-076` also matches the library ADR's references, which must NOT be touched)
6. Also index the library ADR in the README while there — it is currently unindexed (how the collision stayed invisible)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exactly one ADR numbered 076 remains on dev (library-lifecycle); `backlog/decisions/` contains no duplicate numbers (verified with a sorted listing assertion or equivalent check)
- [ ] #2 The server-offload ADR exists as `077-server-offloaded-scheduled-agent-tasks.md` with a matching `# ADR-077:` header, and its README row links the new filename
- [ ] #3 Every reference to the server-offload ADR resolves to 077 (TASK-18940 plan section records the renumber chain 072→076→077); no library-ADR reference was modified (before/after grep diff scoped to `076-server-offloaded`/`ADR-077`-after contexts)
- [ ] #4 The library-lifecycle ADR gains a README index row (it is currently unindexed)
- [ ] #5 The renumber commit message states the provenance (add-commit ids and timestamps) so the board records why the later claimant moved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: hygiene renumber of an existing Proposed ADR; the ADR's content and status are unchanged — only its number, filename, and references move.

1. `git mv` the ADR file; update its header
2. Update the README row (server-offload → 077; add the library row)
3. Update TASK-18940's plan references with the chain note
4. Scoped greps to confirm no stragglers and no library-side edits
5. Commit with the provenance in the message
<!-- SECTION:PLAN:END -->
