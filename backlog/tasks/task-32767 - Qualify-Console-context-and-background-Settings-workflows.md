---
id: TASK-32767
title: Qualify Console context and background Settings workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 03:08'
updated_date: '2026-09-18 04:02'
labels:
  - design-system
  - settings
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users must be able to read, edit and save Console context and background defaults, recover rejected or failed saves, and see saved effects in the live Console. Infinite frame rates currently raise during configuration normalization, and a context percentage label exceeds the fixed label column.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Invalid non-finite background frame rates load the documented default without preventing Settings or Console startup; valid bounded rates keep their existing meaning.
- [x] #2 Context and background labels, values, keyboard focus and actions remain readable in compact/wide layouts and both themes.
- [x] #3 Context ratio and frame-rate validation preserve invalid drafts; navigation, Revert, failed-save recovery and real saved configuration agree with effective policy and background settings.
- [x] #4 Native private-profile journeys demonstrate live effect activation/deactivation and context defaults without changing transcript identity or making provider requests; targeted checks and clean lifecycle evidence are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md, backlog/decisions/150-design-token-system-and-design-language.md, backlog/decisions/161-component-pattern-library.md (existing). Reason: preserve the existing context policy and background effect contracts while correcting validation/paint and qualifying existing workflows. Read TASK-75 and its approved June 2 background spec. 1. Reproduce non-finite rate loading and context label clipping; inspect original targeted assertions and keep private config ownership intact. 2. Apply minimal existing-contract fixes and add mounted keyboard save/validation/navigation/Revert/recovery checks across sizes/themes. 3. Verify actual policy/default persistence and live background application, using targeted original regressions and native private terminal evidence. 4. Record independent review, verification and remaining bounds in the workstream ledger. Allocation scan: max32766 across273 refs/30 worktrees; CLI probe offered32767.
Native run002 demonstrated that a cached Console keeps the prior effect after a successful Settings save. Add a failing mounted save/return regression, preserve transcript identity, publish changed backgrounds through the existing appearance refresh and reconcile the current background on Console resume without restarting unchanged animations.
The delayed-save regression additionally proved disk/cache success after Settings was popped but no runtime publication: its detached worker could no longer resolve self.app. Retain the host before the blocking write and verify both completion timings with seeded user/assistant transcript text. This preserves the existing save/publication contract.
Populated native captures exposed an additional paint gap: Textual8 transcript blank strips occlude the sibling effect even with transparent CSS. Keep the approved dedicated effect renderer/timer, compose it in a lower docked child layer of the transcript viewport, preserve it through transcript recomposition, and verify compositor output, message/selection/scroll stability, effect disable and native visibility. ADR required: no; this repairs the approved June2 presentation-layer contract without changing message or persistence ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the bounded Context and Background Effects Settings review. Fixed non-finite FPS loading, the clipped percentage label, live publication after save and after Settings removal, and actual effect paint beneath message rows. Preserved the dedicated renderer through recomposition, local layer ordering, selection clicks, scrollback and timer cleanup. Added private-profile keyboard/save/recovery and compositor/lifecycle regressions; updated Settings documentation and the completion ledger. 68 distinct targeted cases pass, plus two previously counted budget integration cases; scoped static checks and backlog/diagnostic guards pass. Four native size/theme cells and 12 rendered, inspected captures qualify actual visible particles and unchanged synthetic messages, with clean exit and private-profile integrity. The final attachment-only timer guard follows native capture and has explicit targeted evidence. Independent review found no remaining blocker. No full suite or provider calls. Existing ADR-052/150/161 and the approved June2 presentation-layer contract apply; no new ADR required. Evidence and limitations: Docs/superpowers/qa/2026-09-17-settings-context-effects/README.md.
<!-- SECTION:NOTES:END -->
