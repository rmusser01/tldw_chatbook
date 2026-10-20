---
id: TASK-35870
title: Keep Chunking Lab shortcut hints honest during busy and editing states
status: In Progress
created_date: 2026-10-20 19:25
labels:
- chunking-lab
- textual-patterns
- ux
priority: low
documentation:
- Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md
assignee:
- '@codex'
updated_date: 2026-10-20 21:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chunking Lab footer and F1 help must describe the actions currently available while preview runs, guarded recovery, leaving, or text editing changes eligibility. Disabled controls must not coexist with misleading action hints.

### Evidence and bounded scope

chunking_lab_screen.py:658-668 disables mutation buttons for guarded/leaving state and Run while busy. on_descendant_focus():731-739 advertises r/p/s whenever focus is non-text and a coordinator exists, independently of those states. Executing the actual extracted method produced identical idle and busy-plus-guarded hints. Text-focus check_action():686-698 suppresses printable actions, while generic app.py F1 enumerates raw bindings. Busy-state timing in a full mounted Lab remains acceptance verification.

Book basis: Chapter 8 printed pp.154-158 / PDF pp.163-167 and printed pp.161-162 / PDF pp.170-171.

### Existing ownership and decisions

TASK-31645 is the completed Lab authoring owner, including truthful keyboard/footer AC4. TASK-31428 is its stale pre-renumber counterpart, not an unimplemented second Lab: Docs/Chunking_Lab_Verification.md:216-226 records 31428 -> 31645. This task is only the newly verified state/help gap. TASK-35678 supplies a Settings-specific precedent, not a dependency for a global framework.

Related tasks: [TASK-31645](./task-31645%20-%20Chunking-Lab-Library-screen-and-recoverable-authoring-flow.md); [TASK-31428](./task-31428%20-%20Chunking-Lab-Library-screen-and-recoverable-authoring-flow.md); [TASK-35678](./task-35678%20-%20Share-Settings-action-availability-across-shortcuts-buttons-and-help.md).

ADR required: no new ADR expected.
ADR paths: [031-tui-keybinding-and-footer-hint-conventions](../decisions/031-tui-keybinding-and-footer-hint-conventions.md); [118-chunking-lab-local-execution-and-recovery](../decisions/118-chunking-lab-local-execution-and-recovery.md); [152-roleplay-mode-keys-single-letters-nav-layer-reserved](../decisions/152-roleplay-mode-keys-single-letters-nav-layer-reserved.md).
Reason: No new ADR is expected: retain the existing Lab route, coordinator, recovery and worker contracts, and existing keyboard conventions.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Footer hints update on idle/busy, guarded, leaving, candidate readiness and focus transitions even when focus itself does not change, and advertise only executable actions in that context.
- [ ] #2 F1 presents the same action-support facts with honest scope requirements for text editors and narrow layouts, rather than blindly listing raw printable bindings.
- [ ] #3 Button, keyboard and direct-action paths preserve current execution/recovery validation and useful refusal guidance; printable r/p/s still type in text controls and safe Escape behavior is unchanged.
- [ ] #4 Mounted Lab tests cover busy and recovery-guard transitions, editor focus, candidate availability, and compact/wide F1/footer behavior without relaxing runtime or recovery limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; backlog/decisions/152-roleplay-mode-keys-single-letters-nav-layer-reserved.md
Reason: Correct current local action/help projection, preserving the Lab coordinator, recovery and worker contracts.

- Add actual mounted Lab tests for focus-stable idle -> busy -> idle, guarded/recovery/leaving and candidate readiness changes, plus Input/TextArea/Select focus and compact/wide F1. Use existing deterministic local coordinator/runner fixtures, not provider/network calls.
- Record expected misleading hints/help failures on original code.
- Share a small surface-local action-support projection between controls/footer/help, refresh on state and focus changes, and preserve useful refusal paths and printable typing.
- Run new tests and affected Lab screen/recovery/results tests plus relevant static/import guards; record exact evidence.
- Update docs, self-review and commit owned files; leave completed TASK-31645 and stale pre-renumber TASK-31428 unchanged.

Detailed plan: Docs/superpowers/plans/2026-10-20-textual-book-followup-fixes.md (Task 6).
<!-- SECTION:PLAN:END -->
