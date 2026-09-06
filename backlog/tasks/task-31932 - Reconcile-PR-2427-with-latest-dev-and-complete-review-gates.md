---
id: TASK-31932
title: Reconcile PR 2427 with latest dev and complete review gates
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 17:14'
updated_date: '2026-09-06 17:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve the reviewed test repairs and newly landed dev behavior while making PR 2427 eligible for normal reviewed integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The PR contains latest dev with all reviewed behavior preserved and no unresolved rebase conflicts.
- [x] #2 Review-created Backlog collisions are renumbered with upstream identities and historical evidence paths preserved.
- [ ] #3 Affected complete-file tests and derived artifact checks pass without weakening contracts or raising screen size limits.
- [ ] #4 Qodo findings and required checks on the final revision are handled before normal merge.
- [x] #5 Newly landed character-navigation test fixtures finalize only their own resources, with all seven cases passing and no retained SQLite descriptors under native attribution.
- [ ] #6 New dev boot-worker warning probes retain their owned Loguru sinks for the mounted observation window, and newly attributed worker/smoke fixtures finalize exact owned resources.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the pushed head and isolated worktree, rebase onto fetched dev, and reconcile overlapping owners without losing upstream behavior.
2. Census all task buckets and refs/worktrees; renumber only review-owned collisions, preserve upstream identities and evidence paths, and verify the exact mapping.
3. Run complete targeted files for rebased runtime contracts and architecture inventories. Reconcile stale owner/constructor assertions and tighten slack budgets to measured counts. Keep genuine size failures open; obtain design approval before further owner extraction.
4. Review diagnostic statement deltas before regenerating the manifest. Run CSS, profile-path, schema, index, Backlog, affected tests, and scoped static checks.
5. Publish with an exact force-with-lease, request normal review, address verified Qodo findings, and merge without bypass only after final-head review and required checks succeed.
6. Apply the existing exact-owner real-app fixture adapter to the three new character-navigation test constructors, preserving all behavioral assertions. Verify the complete file with native descriptor attribution and shared cleanup fault controls. This is test-only reuse of established lifecycle APIs, with no new ADR or ownership design required.
7. Extend the already reviewed Loguru capture-lifetime repair (TASK-31901) to newly landed boot-worker parameter cases, keeping all three state transitions and the unknown-worker positive control. Attribute and close only exact worker/smoke fixture-owned resources through existing test lifecycle adapters, then run both complete files and shared cleanup controls with native attribution.

ADR required: no for the rebase and test/derived-artifact reconciliation.
ADR path: N/A; existing DESIGN.md section 7 governs retained controller ownership.
Reason: these steps preserve established storage, runtime, and UI contracts. Any new architectural change requires a separately approved design.

The proposed Canvas/citation ownership paydown is pending user approval under the brainstorming skill; it is not implemented.
<!-- SECTION:PLAN:END -->
