---
id: TASK-31002
title: Models mount only the active provider view
status: Done
assignee: []
created_date: '2026-09-03 12:42'
updated_date: '2026-09-03 13:17'
labels:
  - lab
  - performance
dependencies: []
references:
  - >-
    backlog/tasks/task-2900 -
    Lab-Models-defer-heavy-provider-views-past-first-paint.md
  - Docs/superpowers/specs/2026-07-26-lab-destination-console-frame-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eliminate the apparent freeze when opening Lab > Models by making the llama.cpp view visible without waiting for unrelated provider panes and by avoiding eager construction of inactive model-management views.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Models paints an interactive llama.cpp view without waiting for unrelated provider views to mount.
- [x] #2 Inactive provider pane bodies do not mount before first selection; after first selection each mounted pane is cached so switching preserves unsaved inputs, focus, search/results, hydration, and worker ownership behavior.
- [x] #3 A mounted responsiveness regression proves Models navigation does not breach the repository's 250 ms event-loop-stall threshold under the production CSS bundle.
- [x] #4 Targeted Models, GGUF source, Lab frame, and view-switch tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted regressions proving llama.cpp is active before inactive pane bodies exist and first-selected panes remain cached.
2. Run the new tests against the current implementation and confirm failures identify the eager-mount/delayed-activation behavior.
3. Refactor LLMManagementWindow to compose lightweight pane shells, populate llama.cpp immediately, and lazily populate/cache each requested view through a widget-owned mount worker while retaining screen-owned install and process state.
4. Re-run focused Models and GGUF tests, add a production-CSS responsiveness assertion, and compare event-loop timing with the baseline investigation.
5. Update the task notes and any user documentation affected by the lifecycle change.

ADR required: no
ADR path: N/A
Reason: This is a performance bug fix inside the existing Models presentation boundary, adopting the established one-active-view Textual pattern without changing storage, provider/runtime contracts, security, or user-visible information architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced batch mounting with stable provider shells: llama.cpp composes initially, while every other provider body mounts on first selection and remains cached for subsequent switches.
- Preserved screen-owned install hydration, remote install context, GGUF inventory state, process controls, focus restoration, and Ollama probing across the new lazy lifecycle. Added a guard for progress updates that race nested widget composition.
- Addressed post-PR review findings by retaining server bodies across failed mounts, deferring exact GGUF handoff commits until lazy runtime controls exist, and replaying exact Installed-row navigation after that pane's first mount. Added focused scheduler guard and retry coverage plus Google-style public API documentation.
- Reviewed and regenerated the production diagnostic inventory after the lazy-mount exception message changed. The sole added argument is the internal allowlisted provider view key; it contains no user content, secret, path, or URL.
- Added red/green coverage for initial mount shape, per-view first selection, cached input identity/state, Installed service construction, and event-loop responsiveness under the production CSS bundle.
- Verification on current `origin/dev`: 294 targeted cases passed across deferred Models views, all GGUF modes, the complete Labs adoption file, applicable Installed-model cases, and the legacy downloader architecture guard. Ruff, `py_compile`, and `git diff --check` passed. The unrelated pre-existing `textual-light` disabled-button contrast case remains at 1.96:1; the full repository suite was not run without explicit opt-in.
- ADR required: no. The change remains within the existing Models UI presentation boundary and introduces no durable architectural contract. No user-facing documentation change was required.
<!-- SECTION:NOTES:END -->
