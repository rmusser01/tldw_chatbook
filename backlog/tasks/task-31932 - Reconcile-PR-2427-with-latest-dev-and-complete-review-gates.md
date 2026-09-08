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
- [ ] #1 The PR contains latest dev with all reviewed behavior preserved and no unresolved rebase conflicts.
- [x] #2 Review-created Backlog collisions are renumbered with upstream identities and historical evidence paths preserved.
- [ ] #3 Affected complete-file tests and derived artifact checks pass without weakening contracts or raising screen size limits.
- [ ] #4 Qodo findings and required checks on the final revision are handled before normal merge.
- [x] #5 Newly landed character-navigation test fixtures finalize only their own resources, with all seven cases passing and no retained SQLite descriptors under native attribution.
- [x] #6 New dev boot-worker warning probes retain their owned Loguru sinks for the mounted observation window, and newly attributed worker/smoke fixtures finalize exact owned resources.
- [x] #7 Incremental agent-step persistence tests finalize their owned database and write real run logs under their test-owned root without retaining a process-global workspace database.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the pushed head and isolated worktree, rebase onto fetched dev, and reconcile overlapping owners without losing upstream behavior.
2. Census all task buckets and refs/worktrees; renumber only review-owned collisions, preserve upstream identities and evidence paths, and verify the exact mapping.
3. Run complete targeted files for rebased runtime contracts and architecture inventories. Reconcile stale owner/constructor assertions and tighten slack budgets to measured counts. Keep genuine size failures open; obtain design approval before further owner extraction.
4. Review diagnostic statement deltas before regenerating the manifest. Run CSS, profile-path, schema, index, Backlog, affected tests, and scoped static checks.
5. Publish with an exact force-with-lease, request normal review, address verified Qodo findings, and merge without bypass only after final-head review and required checks succeed.
6. Apply the existing exact-owner real-app fixture adapter to the three new character-navigation test constructors, preserving all behavioral assertions. Verify the complete file with native descriptor attribution and shared cleanup fault controls. This is test-only reuse of established lifecycle APIs, with no new ADR or ownership design required.
7. Extend the already reviewed Loguru capture-lifetime repair (TASK-32014) to newly landed boot-worker parameter cases, keeping all three state transitions and the unknown-worker positive control. Attribute and close only exact worker/smoke fixture-owned resources through existing test lifecycle adapters, then run both complete files and shared cleanup controls with native attribution.

ADR required: no for the rebase and test/derived-artifact reconciliation.
ADR path: N/A; existing DESIGN.md section 7 governs retained controller ownership.
Reason: these steps preserve established storage, runtime, and UI contracts. Any new architectural change requires a separately approved design.

8. User-approved 2026-09-07: move private Canvas actions and citation discovery helpers into the existing ConsoleMessageController; preserve screen event, UI, and worker hooks, current dev behavior, and unchanged size limits. Verify all affected complete files and callback-ownership contracts.
9. User-approved 2026-09-07: restore useful settings failure diagnostics using fixed operation/phase, exception type, and validated opaque identifiers only. Exclude raw exception text, drafts, credentials, and provider/model labels. Add fault-injection privacy and usefulness assertions before implementing; regenerate the diagnostic inventory only after reviewing statement changes.
10. Finalize the incremental agent-step fixture with its existing database close API and supply RunLogWriter with the test-owned database directory. This preserves real logging while avoiding unrelated process-global workspace lookup. Verify the complete file with native descriptor attribution. No new ADR: test-only ownership using established constructors and lifecycle methods.
11. The second dev rebase removed the previously reviewed Save-as-Note owner repair and regression. Restore its test first, verify the wrong-owner failure, and retain the established notes_user_id contract. This is a routine regression repair under the existing notes ownership boundary, not a new storage design; unrelated upstream feature removals remain outside this repair.
12. Repair the reproduced Canvas gateway partial-body read: collect at most the existing byte limit plus one through EOF before JSON decoding; preserve oversized/malformed request refusal and cancellation. Add deterministic split-stream/boundary controls, run the complete gateway and browser files, and obtain independent review. Existing ADR-121 governs the unchanged bounded request/confirmation boundary; no new ADR required for this transport bug fix.

The isolated recovery worktree is `.worktrees/pr2427-review-recovery`. Rebase onto fetched dev `3090013cfea4dbf6133ac43d024656e2eb3a2a56` completed locally at `4a74c5d7e02552a5351d59df7647ea8811526bab`; publication remains open (AC #1). The three review-only collisions were renumbered to TASK-32013/32014/32015, with 3,589 task records passing the identity guard. Agent persistence has 34 complete-file passes and no retained SQLite descriptors under the native observer. Verification and final-head review remain open; see the dated reconciliation report for exact evidence and remaining failures.
<!-- SECTION:PLAN:END -->
