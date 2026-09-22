---
id: TASK-32913
title: Keep MCP tool results current and raw responses readable on merged dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-22 16:50'
updated_date: '2026-09-22 17:15'
labels:
  - mcp
  - ui
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need the Test Tool outcome to describe their latest attempt, and raw responses to remain inspectable in compact terminals. Resume the saved bounded PR2719 repair on the current merged runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Raw response label and body remain keyboard reachable and readable at compact and wide sizes in both themes.
- [x] #2 Local argument validation replaces every previous result detail without execution, while preserving the argument draft and exact displayed permission context.
- [x] #3 Corrected execution restores current output; close/reopen and stale-result identity guards retain their behavior.
- [x] #4 Targeted checks and private real-stdio native evidence qualify the repair without raising CSS budgets or changing permission/runtime authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Resume saved PR2719 from verified PR2722 merged dev ed2a579062. PR2770/TASK32882 already supplies the scrolling prerequisite; retain unrelated edits and the current permission/execution pipeline.

ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
Reason: bounded repair of existing result replacement and token-backed disclosure sizing; no new authority, runtime, storage or service contract.

1. Preserve PR2722 closeout and read saved PR2719/task32833 requirements. The old task32833 collides with the landed Workspace exclusions task, so record this fresh task as its current continuation; keep historical evidence labeled.
2. Bring the saved regression tests onto merged dev first and reproduce stale output/disclosure clipping. Integrate only saved product fix and source TCSS rules into their current owners; rebuild generated CSS, never hand-edit it. Preserve current admission/profile/result guards.
3. Verify success -> invalid arguments -> corrected success, no execution on invalid input, exact context, draft retention, closed/stale panels, raw keyboard access and both theme/size cells. Run targeted tests, static/design/source guards without a full sweep or raised budgets.
4. Reuse current native private-profile/stdio harness infrastructure, with shared CLI validation, checkout provenance, network guard, real painted input/actions, private lifecycle and source hashes. Qualify four theme/size journeys, inspect captures and retain raw evidence with normalized exports.
5. Obtain independent review, update existing PR2719 against dev and present a fresh concrete gallery. Keep this task In Progress until its own owner visual approval, current-head CI, accumulated review and final dev/conflict/actual-merge verification. Do not import stale PR2718 implementation or expand to other screens.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented full result replacement after argument validation failure and token-backed compact raw disclosure sizing. Reuses existing server/tool/profile guards and preserves drafts. Eight regressions reproduced on merged dev and pass with the repair; 281 related inspector/Workbench cases, 43 layout/governance cases and 21 runner-admission cases pass. The dimension ratchet still flags the same twelve inherited declarations; no allowances changed. Inspector harness now retains its collection-time config source via the existing bootstrap_profile marker after an isolated run exposed recovery admission errors. Independent review found no product issues; its native cleanup finding was fixed. Four real-stdio native theme/size journeys and private lifecycle checks pass. Requalifying incoming dev startup changes before publication; owner visual approval, CI/review and merge remain pending. Existing ADRs 150, 161 and 031 apply; no new ADR.

Conflict-free rebase onto dev9e33252708 is qualified: 17 post-rebase cases pass (eight repeated regressions plus nine incoming cases), all nine artifact guards pass again, and sixteen new native captures preserve the inspector appearance. Final native lifecycle is clean with fixture deletion/teardown postconditions verified. Total 362 distinct targeted passes; one inherited dimension-ratchet failure remains unchanged. Independent integration review is clear. Evidence and limitations: Docs/superpowers/qa/2026-09-18-mcp-inspector-results/current-dev/README.md. Remains In Progress for owner visual approval and final PR gates.
<!-- SECTION:NOTES:END -->
