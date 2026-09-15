---
id: TASK-32628
title: Review Library Prompt More actions and deletion recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 07:39'
updated_date: '2026-09-15 08:14'
labels:
  - library
  - prompts
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through the saved Prompt action menu and its recovery transitions. Verify the controls users can reach in the real layout, and repair any gaps between the visible action flow and the existing persistence contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All six More actions controls are reachable and readable by keyboard in wide and compact production layouts; Escape returns to the opener.
- [x] #2 Duplicate creates a detached unsaved copy with the original content while leaving the saved source unchanged.
- [x] #3 Delete cancellation and failure preserve the saved editor; confirmed deletion offers usable Undo and Dismiss, and Undo restores the row and counts through the existing service.
- [x] #4 Targeted tests and an isolated native journey document the action and recovery results without a full repository sweep.
- [x] #5 A failed deletion shows a readable error while retaining the live fields and access to the Delete action for retry.
- [x] #6 Saved Prompts created through New prompt support the same confirmed deletion and recovery as Prompts opened from Browse.
- [x] #7 Replacing a Prompt work pane during recovery safely ignores its pending resize callback.
- [x] #8 Neighboring reader retry and compact focus checks wait for completed UI transitions without weakening their identity, content, read-only or paint assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/055-library-destructive-action-reversibility-rule.md; backlog/decisions/060-atomic-local-prompt-batch-mutations.md; backlog/decisions/150-design-token-system-and-design-language.md (existing)
Reason: Verify and repair the existing saved-Prompt action and recovery contracts without changing persistence or long-lived UX structure.

1. Exercise the six-action disclosure with keyboard navigation and production CSS at 170x48 and 80x24 in both themes.
2. Verify duplicate detachment, delete cancel/failure, confirm, Undo and Dismiss against isolated real SQLite; fix observed control reachability or transition gaps using existing tokens and service seams.
3. Run affected targeted tests and governance checks, then a disposable native terminal journey with fresh completion evidence.
4. Record findings, update guidance where behavior was repaired, self-review and commit locally.

Task hygiene: CLI proposed 32604, which is below the 32627 maximum across 310 refs and live worktrees; reassigned this newly created review to free ID 32628 before implementation.

Observed defect: the single-item delete error updates a Static below the scrolling editor viewport while More actions stays open. Reveal this existing status for delete failures only; preserve field identity, action focus, and ordinary save behavior.

Native diagnosis: saved Create-route Prompts fail the Browse-only confirmation and mutation-owner checks. Reuse the existing active-editor predicate for single-item deletion while keeping bulk selection Browse-only. Recovery test diagnosis: Textual is_mounted remains true on a detached work pane; additionally require attachment before reading its screen in the deferred resize callback.

Neighboring verification follow-up: a gated Select compose reproduces reader-test shutdown before the replacement save menu mounts. Wait for its mounted state before finishing the existing retry assertion. The compact continuity test samples a focused field before scrolling completes; wait for the focused field to be painted before its unchanged paint assertion (animator-idle alone did not guarantee a completed frame). Keep production behavior unchanged and verify both repairs with targeted reruns.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired single-item Prompt deletion recovery in the existing design and service contracts: failed deletion reveals its status while retaining live fields; confirmed deletion and mutation settlement now cover saved Create-route editors as well as Browse; detached work panes ignore deferred resize callbacks. Existing item/version, token, in-flight and bulk-selection gates remain.

Added 11 real-SQLite/production-CSS action journeys. Repaired two neighboring test readiness checks without weakening assertions: a controlled delayed Select compose reproduced teardown before mount, and measured scrolling explained early Advanced paint sampling. Final selections pass 127 distinct targeted checks (11 + 49 + 36 + 31), with overlapping reruns excluded. Changed methods/tests are formatted; no new Ruff diagnostics were added. Legacy controller/screen/reader-test diagnostic counts remain 16/205/1.

The isolated native TldwCli/LinuxDriver journey passes at 170x48 dark and 80x24 light, including Create-route save, six action controls, Duplicate, Cancel, failure/retry, Undo and Dismiss. Fresh normal compact quit returns exit 0 and the shell. Read-only persisted checks retain the source at version 1 and leave copies deleted at version 4. Existing startup/quit notices remain documented.

Updated the Prompt user guide, workflow audit and testing lessons. Evidence, screenshots and reproducible probes: Docs/superpowers/qa/2026-09-15-prompt-actions/README.md. Self-review confirms no storage/schema, token, dependency or permission-boundary changes. Existing ADRs 055, 060, 086 and 150 apply (linked in the plan); no new ADR is required. Full action flows for Export, Copy, History, Collections and Use in Console remain in the broader review; integration into dev remains pending.
<!-- SECTION:NOTES:END -->
