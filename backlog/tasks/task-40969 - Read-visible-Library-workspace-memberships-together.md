---
id: TASK-40969
title: Read visible Library workspace memberships together
status: Done
created_date: 2026-11-17 23:43
assignee:
- '@codex'
labels:
- library
- performance
- workspaces
updated_date: 2026-11-17 23:57
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first populated Library frame waits for a separately admitted workspace membership read for every visible source. On the sealed profile, 41 reads occupied 676 ms of a 701 ms reconciliation. Reduce this delay while keeping the exact global visibility and workspace handoff decisions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library projects the same source identities, membership ordering, labels, and active-workspace eligibility for linked, cross-workspace, unassigned, and id-less rows.
- [x] #2 The normal projection uses one admitted lookup for the bounded visible page and preserves the per-item recovery behavior when the grouped lookup fails or is unavailable.
- [x] #3 Repeated paired private macOS PTY visits use the same sealed fixture and full observation window and show a noticeable improvement in first populated paint. Warm navigation and ordinary resource retirement are verified.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the existing populated Library reconciliation and record the exact membership calls and native timings.
2. Pin equivalent source identity, ordering, visibility, handoff eligibility, and recovery with real SQLite tests. Count admitted membership queries on the normal Library projection.
3. Read bounded batches of exact (item_type, item_id) pairs through the existing WorkspaceDB connection and registry seam. Decode with the existing membership model. Keep the single-item path for registry adapters and grouped-read failures.
4. Re-run the mounted Library and workspace regressions. Compare three alternating isolated macOS PTY pairs on the sealed corpus with the same cold and warm observation windows, and verify ordinary quit and zero owned descriptors. Retain all timings and misses.
ADR required: yes, covered by existing ADR-353
ADR path: backlog/decisions/353-sqlite-store-kernel.md
Reason: The grouped read uses the kernel’s admitted connection and the same scope and ownership. ADR-225 continues to govern app-owned handle retirement. The optimization changes query granularity within the existing workspace registry and preserves the eligibility contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The Library Workspace projection groups exact visible item pairs through the existing registry connection. Each SELECT is bounded to 200 pairs and preserves the original membership decoding and ordering. The per-item path handles unavailable adapters and a failed grouped read. Source identity, workspace labels, eligibility, and native ownership continue to use the existing contracts in ADR-353 and ADR-225.

An initial trace spent 676 ms on 41 membership calls inside a 701 ms reconciliation. Three alternating macOS PTY pairs used the sealed 10,000-chat profile and kept both 13-second observation windows. Every candidate used one grouped membership call before the populated frame. Cold median fell from 2,496 to 1,823 ms (27%), with savings of 525, 788 and 1,066 ms in the pairs. Warm median was 155 to 151 ms. The worst cold loop gap fell from 1,248 to 640 ms, and the count of gaps above 50 ms was 23 to 26. All six quit normally with zero owned SQLite descriptors and zero profile guard violations. See the [paired results](../../Docs/QA/console-library-responsiveness/README.md#library-pty-results).

All 98 targeted Workspace tests passed, and both import budgets and the two Ruff gates passed. The ordinary mounted Library modules produced the same 17 passes and identical 91 pre-mount source-selection refusals on the baseline and candidate. Bootstrap-profile controls passed 23 cases. The remaining landing-sync-count and conversation-focus assertions both reproduced on the exact baseline. Reports retain all outcomes. The scoped task closes on the grouped-read verification and paired timing evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The populated Library frame painted 27% sooner at the median across three paired PTY visits. The displayed workspace decisions matched the original path and all six processes retired their owned SQLite descriptors.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
