---
id: TASK-32603
title: Review Library Prompt save and browse continuity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 06:15'
updated_date: '2026-09-15 06:46'
labels:
  - library
  - prompts
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through Prompt creation, saved editor actions and return to Items. Resolve whether the previously captured Loading pane is transient or a product defect, with evidence from production styles and isolated local storage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-save browse settlement and saved editor continuity are verified with real local persistence; reproduced defects in this path are repaired.
- [x] #2 Basic and Advanced reader controls are reviewed at wide and compact terminal sizes, including keyboard reachability and truthful loading or empty states.
- [x] #3 Targeted verification and bounded native evidence are recorded with explicit limitations and the Library audit is updated.
- [x] #4 Basic, Advanced and Info content uses the existing editor scroll owner; focused text and controls are painted rather than clipped behind sibling content.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: bounded verification and routine continuity repairs implement existing reader, focus and design-token contracts without changing ownership or product structure.

1. Reproduce new Prompt save with real SQLite and production CSS, waiting for Items state separately from source counts. Capture Basic/Advanced at 170x48 and 80x24 in both themes.
2. Trace any reproduced defect to its owning state or layout boundary; add a failing behavior regression before a minimal repair.
3. Verify saved actions, editor retention, resize and return/reopen against persisted content with targeted tests and a private native app journey.
4. Record findings, limitations and verification in the Library audit; self-review and commit locally.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed the real Prompt create/save/mode/Back/reopen journey under existing ADR-086/150/161. New prompt now starts its resident Items request, and first save refreshes the exact browse scope as well as counts. Natural-height sections prevent Basic/Advanced/Info content being clipped under metadata. Browse updates synchronize an empty work pane after Back/Discard while retaining an active editor. Saved block markers and provenance adopt persisted state without replacing text controls.

Added a real SQLite, production-CSS matrix across empty/populated stores, both themes and 170x48/80x24. Assertions cover actual painted text, focus and widget retention, exact persistence, clean block state and keyboard return/reopen. Three conflict tests now wait through subtree replacement for displayed/enabled controls; persistence and history assertions remain. Updated the user guide and Library audit. Evidence: Docs/superpowers/qa/2026-09-14-prompt-continuity/README.md.

Verification: 8 new matrix cases, 88 affected Prompts cases, 63 reader/browse/resize-budget cases, 31 governance cases, plus three overlapping conflict readiness cases passed. Changed methods formatted, new test lint/format clean, no additional baseline Ruff diagnostics. No full repository sweep. Native UI journeys passed with private local databases at both measured sizes; final wide exit 0 was verified. Compact final shutdown is unqualified: its older exit file was stale and the owned terminal was closed while its final process remained active. Both owned sessions are removed. Existing startup diagnostics remain recorded.

Deviation: cross-size focus loss was reproduced and is separately tracked as TASK-32602; the stable-size matrix does not claim resize focus retention. A trial local scroll callback did not address later shared focus restoration and was removed. The CLI initially assigned 32601, already owned by a Workflows task on another ref; scanning 312 refs and live worktrees established 32603 as the next free ID. No new ADR or boundary change; integration into dev remains pending.

Final post-edit verification: the complete Prompt reader file plus the new continuity file passed together (28 cases). The final affected Prompts selection passed all 88 cases; the three repaired conflict waits also passed separately. Self-review and diff whitespace checks completed.
<!-- SECTION:NOTES:END -->
