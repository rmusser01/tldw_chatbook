---
id: TASK-31816
title: Close newly attributed Console controller and hydration fixture handles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 05:39'
updated_date: '2026-09-06 05:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close fixture-owned SQLite handles observed after passing controller and hydration tests in the remaining inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every positively attributed database in the two affected files is closed through its real owner after controller work drains; no foreign registry is closed.
- [x] #2 The complete two-file selection passes with zero retained test SQLite descriptors under the native post-finalizer probe and unchanged behavior assertions.
- [x] #3 Scoped lint, formatting, review and evidence are complete with no production, global conftest, GC or resource-threshold changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only resource ownership repair using existing close and quiescence APIs. 1. Preserve native attribution from the 479-test Chat inventory selection: 14 retaining cases in test_console_chat_controller and 14 in test_console_conversation_hydration. 2. Reuse the explicitly imported Console resource fixture only for owned ChaChaNotes/controller handles; explicitly finalize distinct real WorkspaceDB/AgentRunsDB instances where demonstrated. 3. Assert zero registered ChaChaNotes handles and retain all behavioral assertions; do not drain cached foreign registries. 4. Run both complete files with the native probe, static checks and independent review, then document exact results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused exact Console owner tracking in the controller and hydration files. Added a standard-library ExitStack for explicitly registered auxiliary database closes after controller shutdown and ChaChaNotes quiescence; local hydration teardown disposes only its exact app runtimes. No global on_unmount or foreign registry cleanup. Baseline 479-test split retained handles in 28 cases; imports removed ChaChaNotes retention but left 8 auxiliary cases. Final controller/hydration/control run: 323 passed, zero retained descriptors. Root verification of every importing file: 563 passed in 185.03s, no retained SQLite descriptors or FD-growth warning, three dependency warnings. Nine auxiliary-order RED controls failed before callback closure and pass now, including errors/cancellation. Full scoped lint, changed-region format, diff checks and independent review pass. ADR required: no; existing ownership and lifecycle APIs unchanged. Checkpoint records evidence.
<!-- SECTION:NOTES:END -->

Owner-order refinement: allow the explicitly imported resource fixture to yield
a standard-library ExitStack for exact test-owned auxiliary database callbacks.
Close that stack after controller shutdown and ChaChaNotes quiescence, preserving
errors and cancellation while attempting remaining resources. Extend the focused
fixture regression to prove auxiliary cleanup occurs only after controller attempts
and still runs on failure. Hydration app fixtures must dispose only their own
Console runtime and close the exact Workspace/Subscriptions/Evals/Library DBs;
do not invoke broad app.on_unmount subprocess/logging teardown on unmounted apps.
