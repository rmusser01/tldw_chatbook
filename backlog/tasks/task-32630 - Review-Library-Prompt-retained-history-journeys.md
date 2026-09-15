---
id: TASK-32630
title: Review Library Prompt retained-history journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 15:24'
updated_date: '2026-09-15 16:03'
labels:
  - library
  - prompts
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through the saved Prompt History controls, checking keyboard access, retained previews, restore confirmation, and recovery without changing retained-history semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 History opens from More actions with readable keyboard focus at wide and compact sizes in dark and light themes.
- [x] #2 Version selection, bounded older-page loading, and read-only previews remain usable without replacing live editor fields or moving focus outside History.
- [x] #3 Cancel and confirmed restore return to readable controls; restore preserves the retained snapshot and creates the expected current version.
- [x] #4 Load and restore failures, dirty-state gating, and unavailable snapshots retain truthful feedback and a usable recovery path.
- [x] #5 Targeted checks and isolated native verification document the results, with no full repository sweep.
- [x] #6 The user guide and workflow audit describe the verified History flow and any remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/049-local-prompt-retained-version-history.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Review and repair existing History focus, visibility and recovery using the current retained-history, modal and design-token contracts; no storage or service-boundary redesign.

1. Use production CSS and real SQLite to exercise More actions to History, lazy loading, selection/preview and older-page navigation at 170x48 and 80x24 in both themes. Confirm live editor fields and keyboard focus survive local History updates.
2. Exercise actual confirmation Cancel and Restore, unchanged retained rows/new current version, no_change, load and restore failure recovery and dirty-state gating. Reproduce any observed defect before implementing its smallest repair; preserve UUID/scope/request and optimistic concurrency guards.
3. Run the affected History/controller/state/DB checks and applicable token/bundle governance only. Verify the visible journey in an isolated native app profile, with actual terminal sizes, readable controls and fresh normal-exit/persistence evidence.
4. Update the user guide, workflow audit and task notes, review the final diff, and commit locally. Integration into dev and a full repository sweep remain outside this pass.

Allocation: fresh origin fetch; reachable object paths and 43 live worktrees had maximum 32629. Content references across 311 refs did not claim 32630; CLI allocated TASK-32630 with the matching title.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
More actions → History now closes its menu, reveals Info from Basic and focuses the visible disclosure. History preserves focus through selection, paging and recovery, while its editor fields remain mounted. Four-row token-backed version controls keep both label lines readable. Restore begins after modal return; Cancel/retry return to Restore, and successful adoption returns to the updated History title without overriding newer focus.

Verification: 226 targeted checks passed (76 UI, 124 controller/state/database/normalization, 26 token/bundle/wiring). No new Ruff diagnostics; focused files and the native runner pass formatting checks. The private native app passed 170×48 dark and 80×24 light, including 12 UI-created versions, paging, literal preview, Cancel and v1→v13 restore. All six captures were rendered and inspected. Normal Ctrl+Q returned exit 0; read-only SQLite confirms both active v13 records and all retained versions. The owned terminal session was closed. No full sweep was run.

Core files: library_prompts_controller.py, prompt_history_region.py, library_prompts_canvas.py, library_screen.py, _library_panels.tcss and its generated screen bundle; new keyboard journeys, one existing test readiness correction, the user guide, workflow audit and QA record. The region keeps its post-recompose focus callback local to avoid an eager Widgets.Library import cycle; contracts remain unchanged. Added the observed focus/paint/modal-return lesson.

ADR required: no; applies existing backlog/decisions/049-local-prompt-retained-version-history.md, 086-library-adaptive-reader-shell.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md. Evidence: Docs/superpowers/qa/2026-09-15-prompt-history/README.md. The existing transient success toast can cover lower rows while the focused title remains readable. Native failure injection, a full repository sweep and integration into dev were not part of this pass.
<!-- SECTION:NOTES:END -->
