---
id: TASK-554
title: Resolve citation epic TASK-401 identity collision
status: Done
assignee: []
created_date: '2026-07-24 21:33'
updated_date: '2026-07-24 21:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore unambiguous Backlog identity by preserving the older completed response-prefill task as TASK-401 and moving the citation-provenance epic plus its child hierarchy to a collision-free root identifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The completed response-prefill task remains TASK-401 and its historical code and test references remain unchanged
- [x] #2 The citation-provenance epic, child identifiers, parent links, dependencies, ADR, and implementation-plan references consistently use TASK-553
- [x] #3 TASK-553 and TASK-554 are unclaimed by fetched remote refs, committed local refs, and other active worktrees
- [x] #4 Every current active Backlog task has exactly one canonical frontmatter identifier and no local identifier is duplicated
- [x] #5 Focused Backlog inspection, repository task-ID harness, and diff validation pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the duplicate TASK-401 failure and identify which files and references belong to the older response-prefill task versus the newer citation-provenance hierarchy.
2. Preserve the older completed TASK-401 identity and mechanically renumber the citation epic and its twelve completed foundation children to TASK-553 and TASK-553.1 through TASK-553.12, chosen after checking fetched remote refs, committed local refs, and active worktrees.
3. Update only citation-owned parent links, dependencies, ADR-024, and the approved citation foundation plan; leave response-prefill code and test references on TASK-401.
4. Verify Backlog resolution, exact parent/dependency closure, repository-wide active-task ID uniqueness, the full product-maturity harness, and diff hygiene.
5. Request independent review before documenting and closing the remediation.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This repair changes Backlog identifiers and documentation references only; ADR-024’s citation architecture and runtime contracts remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Preserved the older completed response-prefill work as TASK-401 and mechanically renumbered the citation-provenance epic plus all twelve children to TASK-553 and TASK-553.1 through TASK-553.12. Filenames, frontmatter IDs, parent links, dependency links, ADR-024, and the approved foundation plan now agree.
- Restored explicit TASK-553 parent links for maintenance children .11 and .12 after the root-ID ambiguity was removed, and corrected .12’s historical note to distinguish the removed .11 link from .12’s standalone creation. No completed implementation content changed.
- The first candidate TASK-552 was rejected during independent review because an active File Notes worktree already owns TASK-552/.1. TASK-553 and remediation TASK-554 were then verified absent from committed local and remote refs and from every other active worktree before allocation.
- Verification: Backlog resolves TASK-401, TASK-553 with all 12 subtasks, and TASK-554; a YAML hierarchy check confirms every child parent and dependency target; the complete product-maturity task-ID harness passes 2/2; stale-reference, trailing-whitespace, final-newline, and `git diff --check` checks pass.
- Independent review verified the cross-worktree allocation, mechanical history mapping, response-prefill preservation, hierarchy closure, documentation references, and task hygiene, approving closeout with no remaining findings.
- ADR required: no; ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md; reason: identifier and documentation repair only, with ADR-024’s architecture and runtime contracts unchanged.
<!-- SECTION:NOTES:END -->
