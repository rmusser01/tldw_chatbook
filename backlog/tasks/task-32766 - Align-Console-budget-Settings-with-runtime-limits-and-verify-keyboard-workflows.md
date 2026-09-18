---
id: TASK-32766
title: >-
  Align Console budget Settings with runtime limits and verify keyboard
  workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 02:40'
updated_date: '2026-09-18 03:05'
labels:
  - design-system
  - settings
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users must be able to edit, save and revisit the five Console agent-run budgets without a successful Settings save silently turning into a different runtime budget. Current Settings accepts step values above the existing 199999 runtime cap; the resolver substitutes 25000. Qualify the mounted keyboard, persistence and recovery paths in both compact and wide layouts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Steps outside the runtime-supported range cannot be saved from Settings; the entered value remains editable with readable feedback, and loading a legacy over-limit value agrees with the runtime fallback.
- [x] #2 Every accepted budget value reaches the private saved configuration and the next real runtime budget resolution; valid maximum steps and documented zero/unlimited semantics are preserved.
- [x] #3 Keyboard navigation, visible labels and focus, invalid-save refusal, navigation retention, Revert, save-failure recovery and retry work at compact and wide widths in both themes.
- [x] #4 Original budget tests execute under an owned process-lifetime profile without relaxing production recovery ownership, and targeted regressions plus native visual/lifecycle evidence are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md (existing trace ownership), backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md. Reason: apply the existing runtime/config step cap in the existing Settings boundary; no schema, runtime limit, provider or UX architecture change. 1. Preserve original assertions while making source-bound budget tests use the established private-profile runner; add a mounted failing over-limit save test and legacy-value parity check. 2. Reuse MAX_CONSOLE_AGENT_MAX_STEPS for Settings load/validation/help; keep rejected drafts and all other ranges. 3. Exercise real private saves and next-run resolution, maximum/zero values, keyboard paint/navigation/Revert, injected failure/retry and resize/theme behavior. 4. Run targeted regressions and governance/static checks, review changes, capture native dark/light compact/wide journeys, record lifecycle and update the completion ledger. Starting baseline: merged dev e89f28d751; ID allocation found max32765 across316refs/30worktrees, CLI probe offered32766.
The wide painted-label regression also clips the seconds units. Shorten the two time labels within the existing 24-column form pattern; preserve units and existing help. Assert the rendered save receipt because the existing presenter normalizes transient staged state after successful persistence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Settings now reuses the existing 199999-step config cap for load/validation/help, retaining invalid drafts and displaying the runtime default for legacy over-limit values. Shortened both time labels to preserve seconds units in the existing form. Original assertion sets are preserved under the established private-profile helper; new keyboard journeys verify real persistence and runtime agreement, Revert, failure/retry, maximum/zero values and resize/theme paint. Updated the user guide and completion ledger. Existing ADR-080/150/161 apply; no new ADR. Verification: 34 budget UI cases plus 87 runtime/governance cases pass; scoped lint/format and backlog/diagnostic guards pass, with unchanged 116 legacy Settings lint findings. Independent review is resolved. Four native dark/light compact/wide cells and 12 inspected captures passed with normal exit, 11 healthy private databases and unchanged default fingerprints. QA: Docs/superpowers/qa/2026-09-17-settings-agent-budget/README.md. An initial native capture timing failure is retained and not counted; final runner waits for Toast settlement. No provider request or full-suite run. Broader component review remains active.
<!-- SECTION:NOTES:END -->
