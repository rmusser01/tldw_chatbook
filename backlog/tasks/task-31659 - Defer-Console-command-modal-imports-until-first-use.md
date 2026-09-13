---
id: TASK-31659
title: Defer Console command modal imports until first use
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:38'
updated_date: '2026-09-05 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep command-only modal modules off the Console boot path while preserving first-use commands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Style and rewind commands retain behavior
- [x] #2 Command-only modal imports are deferred without changing boot budgets
- [x] #3 Video capacity prompt retains first-use behavior and stays off the boot import path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: Implements existing first-use import discipline without changing runtime interfaces.
1. Confirm modal imports are used only by command methods.
2. Move imports to first-use methods and remove obsolete screen aliases.
3. Run focused command tests and report boot census to root.
4. Defer the single video-capacity modal import at its capacity-exceeded use site; add a fresh-process import guard and run video-capacity behavior tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved style-picker and rewind imports into their four command methods, retaining choice/row annotations under TYPE_CHECKING. The final census also identified the video-capacity prompt as first-use work; its single import now lives at the capacity-exceeded prompt call. Fresh-process guard was RED for the capacity modal before that edit and GREEN afterward. No controller construction, budget or runtime policy changed; existing ADR-097 applies. Final combined modal/import/census run: 147 passed in 109.48s, including full style, rewind-modal and video-capacity files. Warm UI-ready census is 972/972 after this deferral and the separately tracked vLLM target deferral. Root separately repaired three baseline style fixtures; no failures excluded from this final run. Ruff lint/format and diff checks passed; self-review preserved all first-use validation and modal behavior.
<!-- SECTION:NOTES:END -->
