---
id: TASK-32190
title: Stop repeated failed tool calls with changing arguments
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 19:53'
updated_date: '2026-09-09 20:04'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bound agent runs that keep retrying an unavailable tool with different queries or parameters, while allowing successful tool work and recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three consecutive ordinary failures of the same tool stop the run with an actionable message even when arguments differ.
- [x] #2 Successful tool results and switching tools reset the streak; existing exact-call loop and permission handling remain intact.
- [x] #3 The final failed tool result is recorded, native tool histories remain coherent, and per-run/child state is isolated.
- [x] #4 Focused runtime tests cover stop, reset, native batches and independent runs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/078-structured-agent-tool-outcome-provenance.md
Reason: extend existing per-run repeated-call guard using authoritative ordinary failed outcomes; no provider/storage/permission contract or new configuration.
1. Write failing pure-loop cases for changing arguments and ordinary failed outcomes.
2. Add the smallest per-run same-tool failure streak at the existing settled-result boundary; stop at the existing repeated-call threshold with actionable copy.
3. Preserve complete result/continuation state and test successful recovery, tool switches, independent runs and native batches.
4. Run targeted runtime tests and scoped lint, document and review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added two per-run streak variables and a settled-result guard using the existing repeated-call threshold. Three consecutive ordinary failures from the same tool stop with actionable copy even when arguments differ. Successful, blocked and other nonfailure outcomes reset; review-refused continuation paths reset too. Results and continuation Finished events precede termination; native partial batches retain the existing coherent history boundary. Updated runtime regressions and current Settings/user-guide wording.

RED: five missing-feature failures, seven controls already passing. GREEN: 190 targeted runtime/provider-continuation/review/fleet/search integration tests; independent review passed 16 controls. Combined touched search/runtime/provider/local-server run passed 637 with three skips and an unrelated pre-existing filesystem-read ledger failure. Settings suite: 27 passed, one original save-click failure reproduced with original Settings module. Scoped formatting and test lint pass; existing runtime F821 FallbackRuntime and formatting debt are unchanged, no new diagnostics. Existing ADR078; no settings, dependencies or permission boundary added.
<!-- SECTION:NOTES:END -->
