---
id: TASK-19005
title: Plan mutation-free lasting Notes reconciliation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:42'
updated_date: '2026-08-21 03:07'
labels:
  - notes
  - sync
  - filesystem
dependencies:
  - TASK-19004
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Classify local folder and Library observations into safe operations, attention items, skips, identity-proven moves, and deletion candidates without mutating either authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root admission and observation reject unsafe overlap, symlink traversal, aliases, hard links, unsupported encodings, unstable identities, and representation or metadata loss.
- [x] #2 Supported UTF-8 BOM, newline, final-newline, mode, and identity profiles round-trip without silent normalization.
- [x] #3 A pure deterministic planner applies the ADR direction matrix, classifies identity-proven moves before missing-side deletion candidates, and never chooses a conflict or deletion winner.
- [x] #4 Out-of-direction changes, both-side changes, filesystem moves implied by note changes, ambiguous identity, and capability loss become explicit attention or skip actions.
- [x] #5 Repeated planning is idempotent, creates no root, binding, receipt, recovery, note, folder, file, or configuration mutation, and rejects stale reviewed observations before apply.
- [x] #6 Deletion-burst grouping uses measured representative-tree evidence rather than a speculative production threshold.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin RED tests for byte representation, identity, containment, and platform capability without mutating either authority.
2. Extend PinnedSyncRoot with the minimum descriptor-verified byte observation, guarded replacement, and same-root move primitives; preserve legacy wrappers.
3. Write the full direction and missing-side matrix, then implement a pure deterministic reconciliation planner over frozen models only.
4. Add a deterministic representative-tree benchmark and use measured results for bounded paging and deletion grouping.
5. Run the focused task gate, benchmark, static checks, independent review, documentation, and task hygiene.

ADR required: no new ADR
ADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: ADR-059 and ADR-073 already define reconciliation authority, direction behavior, identity safety, representation preservation, deletion review, and filesystem capability boundaries; this task implements the mutation-free planner and low-level primitives without activating sync.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added descriptor-pinned, bounded byte observation plus platform-gated atomic no-clobber/exchange primitives. POSIX preserves supported representation, mode, and xattrs; unsafe links, ACLs, ownership, flags, encodings, metadata, identities, and commit uncertainty fail closed. Windows remains native-handle, read-only observation only.
- Added a frozen, validated, redacted, deterministic reconciliation planner covering all directions, changes, moves, missing sides, offline/capability/duplicate cases, deletion review, and stale version/capability-generation fencing. Planning performs no I/O or mutation and never selects a conflict or deletion winner.
- Measured deterministic representative trees at 99/100/101/1,000/5,000/10,000 items. On the final verification run, 100 items planned in 3.777 ms mixed / 2.751 ms deletion at about 207/186 KiB; 10,000 planned in 374.876/266.923 ms at about 20.6/18.6 MiB. `RECONCILIATION_PAGE_SIZE=100` bounds each future review payload; `DELETION_GROUP_THRESHOLD=100` keeps 99 deletions itemized and groups 100+ into one lossless typed root preview. The benchmark supports off-loop planning at scale; it does not claim page size changes planner runtime.
- Commit: `e01a7e150`. Final task gate: 96 passed, 1 dependency warning. Benchmark boundary/scale assertions, Ruff, formatting, and diff checks passed. Two independent reviews reported Ready with no findings; 20 targeted race/rollback/metadata/privacy probes also passed.
- ADR check: no new ADR; implementation follows ADR-059, ADR-073, and the existing ADR-055 destructive-action boundary.
<!-- SECTION:NOTES:END -->
