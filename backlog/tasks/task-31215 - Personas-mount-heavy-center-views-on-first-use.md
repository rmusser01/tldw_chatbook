---
id: TASK-31215
title: Personas mount heavy center views on first use
status: Done
assignee: []
created_date: '2026-09-04 00:23'
updated_date: '2026-09-04 02:50'
labels:
  - roleplay
  - performance
dependencies: []
references:
  - >-
    backlog/tasks/task-2725 - Roleplay screen switch takes 2s where every other
    screen takes under 1s.md
  - backlog/tasks/task-31002 - Models-mount-only-the-active-provider-view.md
  - backlog/decisions/115-personas-demand-mounted-center-views.md
  - >-
    Docs/superpowers/specs/2026-09-03-personas-demand-mounted-center-views-design.md
  - Docs/superpowers/plans/2026-09-03-personas-demand-mounted-center-views.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining Roleplay navigation stall by keeping the four heavy inactive center views out of the post-first-paint load path and mounting each only when its workflow is first activated, without losing restore, selection, editor, or Console-handoff state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Personas reaches a usable initial Characters surface without mounting any inactive heavy center view.
- [x] #2 First use mounts only the requested heavy view and revisiting reuses the same widget state.
- [x] #3 Restore and deep-link intents replay after the required view is ready without applying stale state to another view.
- [x] #4 Transient mount failures remain retryable and leaving the screen prevents stale callbacks from mutating detached UI.
- [x] #5 Targeted Personas lifecycle and four-mode workflow tests pass, and a production-CSS responsiveness regression stays under the 250 ms event-loop-stall threshold.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red-first mounted tests proving real initial load leaves all four heavy roots absent and stable slots preserve document order.
2. Implement screen-owned first-use construction, mount, cache, hydration, retry, and lifecycle-generation behavior.
3. Add red-first workflow tests and route character/persona create/edit plus dictionary/lore select/create/edit/restore through the exact view admission boundary.
4. Add concurrency, transient-failure, teardown, and production-CSS 250 ms heartbeat regressions.
5. Run focused Personas and architecture verification, scoped static checks, self-review, and close the task with measured evidence.

ADR required: yes
ADR path: backlog/decisions/115-personas-demand-mounted-center-views.md
Reason: the change defines the long-lived lifecycle and restore/admission contract shared by the Personas screen and its four authoring widgets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented screen-owned, first-use mounting for the character editor, persona editor, dictionary manager, and lore manager. Stable lightweight slots preserve document order while per-view locks, cached instances, retryable mount failures, and lifecycle-generation checks protect concurrent activation and teardown. Character/persona creation and editing, actor packs, dictionary/lore workflows, restore intents, runtime-source hydration, and character TTS presentation now cross the exact view admission boundary before mutating widget state. A natural-height slot replaced the planned auto-height wrapper after real pointer testing exposed painted but non-hit-testable controls. Verification: 378 Personas Workbench tests, 246 other affected Personas tests, and 6 architecture checks passed (630 targeted tests total); the production-CSS heartbeat regression enforces a sub-250 ms event-loop stall; compileall, Ruff lint/format checks, and git diff whitespace checks passed. The full repository suite was not run, per the repository targeted-test policy.

PR #2364 review follow-up: verified Qodo's concurrency finding with a red-first regression that pauses after DOM mount but before hydration. Replaced queryability-based reuse with an explicit ready-view cache published only after hydration and cleared on unmount, so concurrent callers remain behind the per-view lock and a failed hydration is retried safely. Added the required on_unmount lifecycle docstring. Post-fix verification: 247 affected workflow/lifecycle tests and 378 Workbench tests passed (625 total); compileall, Ruff lint/format, and git diff whitespace checks passed.

CI follow-up: the Derived Artifacts gate correctly detected the intentional new bounded mount-failure warning in personas_screen.py. The statement-level diagnostic audit confirmed the call interpolates only the internal allowlisted view key and contains no user content, secret, path, or URL. Regenerated Docs/security/production-diagnostic-inventory.json; the inventory checker passes with 568 owners, 1,337 TASK-492 calls, 7,568 TASK-494 calls, and 10 sink files.
<!-- SECTION:NOTES:END -->
