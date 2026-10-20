---
id: TASK-35676
title: 'Keep Console Behavior form refreshes from staging user edits'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19'
labels:
  - settings
  - textual-patterns
  - correctness
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Priority 1 of the Textual-book follow-ups: qualify and correct the bounded form-event problem before structural refactoring. Source review at checkout 473c32b4ed found that SettingsScreen's Console Behavior synchronization assigns the paste-collapse Checkbox and threshold Input values under temporary _syncing_console_paste_toggle / _syncing_console_threshold flags (settings_screen.py around 35424-35441). Their Changed handlers (around 28802 and 28961) check those flags before staging edits. Messages may arrive after the flag resets; neighboring controls already use Textual's prevent(...) context.

This is a source-supported timing concern, not a reproduced user-facing failure. Restrict this task to the two composer-paste controls and their loading, refresh, Save and Revert paths. Establish the mounted behavior first, correct only the demonstrated gap, and record a no-change disposition if current behavior already meets the criteria. Do not expand this into a repository-wide suppression rewrite.

Preserve mount-time Input/Select event handling and existing provider/RAG/image/video suppression queues: suppressing a synchronous assignment does not retroactively suppress an event already queued or one produced by a later mount. No change to draft ownership, validation, persistence, or instant-apply controls.

Book basis: Michael Driscoll, Creating TUI Applications with Textual and Python, 2025-07-25 edition, chapter 7 pp. 144-145. Use the repository's pinned Textual version and existing prevention idioms, not unverified sample code.

ADR required: no new ADR.
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md.
Reason: bounded event-origin correctness fix within the existing staged settings contract.

Deduplication: TASK-15740/TASK-15673 cover the completed provider repopulation fixes; TASK-23191 covers the completed Video Generation initial-dirty fix; TASK-16480 contains related deferred-event history. TASK-32761 owns prior Console/Storage workflow qualification. TASK-35017 owns Console Behavior page order/disclosures, which this task must not redesign.

### Related work and decisions

- [TASK-15740](../tasks/task-15740%20-%20Pressing-Save-in-Providers-Models-marks-untouched-fields-dirty-and-refuses-the-save.md): existing owner or precedent described above.
- [TASK-15673](../tasks/task-15673%20-%20Settings-navigation-preselect-applies-the-provider-then-silently-drops-the-model.md): existing owner or precedent described above.
- [TASK-23191](../tasks/task-23191%20-%20Video-Generation-settings-show-State-Unsaved-changes-on-a-fresh-profile-with-no-edits.md): existing owner or precedent described above.
- [TASK-16480](../tasks/task-16480%20-%20Six-new-dev-red-rows-from-the-mid-August-churn.md): existing owner or precedent described above.
- [TASK-32761](../tasks/task-32761%20-%20Qualify-Console-Behavior-and-Storage-Settings-workflows.md): existing owner or precedent described above.
- [TASK-35017](../tasks/task-35017%20-%20Open-Console-Behavior-on-chat-defaults-and-fold-agent-capture-and-replay-sections-into-Advanced.md): existing owner or precedent described above.
- [033-settings-commit-models-three-honestly-labeled](../decisions/033-settings-commit-models-three-honestly-labeled.md)
- [150-design-token-system-and-design-language](../decisions/150-design-token-system-and-design-language.md)
- [161-component-pattern-library](../decisions/161-component-pattern-library.md)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A mounted regression distinguishes an actual value-changing programmatic refresh from a user edit for both paste controls, waits for queued Changed messages, and records the pre-change result.
- [ ] #2 Opening, repopulating, saving and confirmed reverting the paste controls leave no phantom dirty keys or newly staged result message once queued events settle; unrelated staged values retain their existing semantics.
- [ ] #3 Real user toggle/threshold edits still stage the intended values, retain existing invalid-value feedback, save through the existing writer, and revert through the existing confirmation flow.
- [ ] #4 Initial mount and category leave/re-entry do not swallow the first genuine edit, including one equal to an earlier programmatic value; existing suppression queues outside this section are unchanged.
- [ ] #5 Only affected Settings tests run under a private profile, with a real keyboard Save/Revert journey for any behavioral fix; no real user configuration is modified. Evidence distinguishes reproduced defects from source-only concerns.
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/161-component-pattern-library.md
Reason: bounded changes under the existing Settings and approved screen-decomposition contracts; persistence and service ownership stay with their existing owners.
1. Mount the original Console Behavior controls and force value-changing refreshes; record queued-event staging before fixing.
2. Prevent only the two paste-control programmatic Changed messages, preserving actual edits and mount semantics.
3. Verify affected load/save/revert/category-return paths with private-profile targeted tests.
<!-- SECTION:PLAN:END -->
