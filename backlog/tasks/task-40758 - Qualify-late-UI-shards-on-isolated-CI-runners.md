---
id: TASK-40758
title: Qualify late UI shards on isolated CI runners
status: In Progress
created_date: 2026-11-13 23:46
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The full self-hosted PR gate reaches later UI shards but exposes a collapsed-handle focus loss and fixtures that assume asynchronous initialization or filesystem timestamp advancement has already occurred. Preserve the original behavioral assertions while making the required tests reliable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Activation focus restoration retains a user's reachable collapsed Inspector handle, and its painted focus cue remains visible at both existing terminal sizes.
- [x] #2 Workflows loading-state geometry is checked after its actual workbench mounts, including deliberately delayed initialization; missing panes still fail within the existing wait budget.
- [x] #3 CSS staleness fixtures explicitly establish the timestamp ordering claimed by their scenarios, and still detect changed content when generated sheets are newer.
- [ ] #4 Focused regressions, independent review, and the affected normal self-hosted PR lanes pass without dropping selectors, coverage, isolation, or raising deadlines.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Existing focus-restoration bug fix and test fixture preconditions; no new runtime or CI boundary.
1. Read the three failed shard logs and reproduce exact failures; trace focus ownership, Workflows initialization and CSS manifest mtimes.
2. Check the actual focused control with the existing ancestor-visibility helper so a visible collapsed handle is retained and hidden controls are still repaired; pin deterministic focus restoration before changing production. Wait for the actual Workflows workbench and explicitly order edited-source/generated-sheet timestamps; reproduce these fixtures with delayed startup/coarse timestamps.
3. Run focused tests and failure controls; review independently, retain all existing assertions and lane topology, then qualify in the normal PR gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The activation restore now checks the visible focused control rather than only its mapped logical pane. This preserves a collapsed Inspector handle while keeping terminal and collapsed-composer behavior, ancestor visibility and fallback unchanged. The existing painted-focus test invokes the restore explicitly before its original paint/losslessness assertions. Workflows loading-state tests wait for the actual workbench with the existing two-second selector helper. CSS fixtures explicitly order timestamps; the builder integration captures its next timestamp before a write can reset the synthetic future timestamp. No CI topology, selectors, test assertions, isolation, deadlines or architecture budgets changed. Updated Console architecture and recorded the observed traps in lessons-testing-evidence. ADR-359 remains applicable; no new ADR required.

Evidence: run43519057842 failed UI22/24/25 (logs /tmp/tldw-ci-seventh-ui{22,24,25}-failure.log). A focus-owner probe found the composer owned both captures in failing cases; explicit activation restore failed both original terminal sizes before the correction (/tmp/tldw-ci-focus-red.log). A controlled 250ms Workflows initializer delay and coarse-clock source writes reproduced three fixture failures; corrected fixtures and focus checks passed23 cases11.33s. Independent review caught a second timestamp collision after the integration fixture refreshed its manifest and four lines over the existing ChatScreen limit; both were corrected before push. Updated fixed-coarse-clock probe plus both architecture guards passed34 cases12.15s (/tmp/tldw-ci-late-reviewed-green.log); ChatScreen remains24568lines/742methods. Negative controls disabling manifest hashing or preventing panes from mounting both failed as expected (/tmp/tldw-ci-late-negative.log).

Broader affected verification passed63 cases; seven adjacent keyboard cases stopped in app_factory before touched code because that older module lacks its required bootstrap-profile declaration. Rechecking that module with the documented collection-time profile via a temporary plugin passed all16 cases23.35s; no repository profile/marker was changed. Logs /tmp/tldw-ci-late-ui-focused.log and /tmp/tldw-ci-keyboard-profile.log. F-rule lint, whitespace and backlog validators pass. Differential Ruff comparison found no new findings (existing33 in ChatScreen and9 in destination parity); changed lines conform to the formatter without reformatting existing large-file debt. Independent follow-up review found no blockers. Loaded normal PR qualification remains pending, so task stays In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
