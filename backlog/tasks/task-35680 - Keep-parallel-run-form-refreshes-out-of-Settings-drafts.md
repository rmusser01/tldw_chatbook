---
id: TASK-35680
title: Keep parallel-run form refreshes out of Settings drafts
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19'
labels:
  - settings
  - textual-patterns
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First follow-up from the Textual-book reanalysis after the paste-control work: prevent the separate Max parallel agent runs control from contributing an untouched value to an existing Console Behavior draft during programmatic refresh.

A mounted private-profile diagnostic on PR #3176's product source reproduced this independently of the paste controls. Starting with saved max_parallel_runs=1 and paste_collapse_threshold=50, stage threshold=120, change the backing max_parallel_runs to 2, call the real Console Behavior refresh, and wait for queued messages. The draft unexpectedly acquires max_parallel_runs: 2 alongside the intended threshold edit. The equality assertion fails with one extra item; the probe finished in 44.09 seconds. This proves draft pollution under that controlled refresh, not a production data-loss or corruption report.

The remaining screen synchronization flag resets before Input.Changed is handled. TASK-35676 deliberately fixed only the two paste controls; it does not own this additional control. Keep the follow-up bounded to parallel-run event origin, real editing, validation and the existing staged Save/Revert contract. Agent budget, tool-result display and side-chat flags have a similar source shape but are not reproduced findings or additional scope here.

Book basis: Michael Driscoll, Creating TUI Applications with Textual and Python, 2025-07-25 edition, chapter 7 pp. 144-145 (preventing programmatic messages). The project uses Textual 8.2.8; follow its existing prevention idioms and preserve mount/re-entry behavior.

ADR required: no new ADR expected.
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md.
Reason: a bounded event-origin correction within the existing draft, validation and persistence boundary.

### Related work

- [TASK-35676](./task-35676%20-%20Keep-Console-Behavior-form-refreshes-from-staging-user-edits.md): completed paste-only event-origin fix; use its regression pattern without reopening its scope.
- [TASK-32761](./task-32761%20-%20Qualify-Console-Behavior-and-Storage-Settings-workflows.md): existing Console Behavior workflow qualification.
- [TASK-18600](./task-18600%20-%20Expose-Console-agent-run-budget-limits-in-Settings-and-raise-defaults-for-long-running-sessions.md): existing budget-control work; no budget policy change is requested.
- [ADR-033](../decisions/033-settings-commit-models-three-honestly-labeled.md) and [ADR-031](../decisions/031-tui-keybinding-and-footer-hint-conventions.md).
- [PR #3176](https://github.com/rmusser01/tldw_chatbook/pull/3176) and [follow-up analysis](../../Docs/superpowers/qa/2026-10-19-textual-book-followup.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A mounted regression reproduces the extra max_parallel_runs draft value on the pre-fix source with an unrelated staged field and records the queued-event failure.
- [x] #2 Opening, value-changing refreshes, Save, confirmed Revert and category re-entry do not stage untouched parallel-run values or report a new edit after queued messages settle.
- [x] #3 Genuine parallel-run typing, including the first edit after refresh/re-entry, retains validation feedback, saves through the existing writer and reverts through the existing confirmation flow; unrelated staged values remain intact.
- [x] #4 Targeted private-profile regressions and a real keyboard Save/Revert journey qualify the correction, and an appropriate regression enters the existing PR test gate without weakening safety or performance guards.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
Reason: correct one control's event origin within the existing draft, validation, writer and keyboard contracts; no ownership or policy change.

1. Promote the prior diagnostic to a permanent mounted regression and observe the extra draft value on the current PR source, including untouched result feedback and a subsequent genuine edit.
2. Replace only parallel-run programmatic event suppression with the existing Textual prevent(Input.Changed) idiom, removing its temporary flag. Ignore initialized/equal or stale queued values at the handler while retaining the next genuine edit and unchanged-value editing state.
3. Qualify clean mount, validation, keyboard Save, confirmed Revert and category return under private profiles, plus the affected paste regression; add bounded coverage to the existing PR gate and tighten the measured Settings size ratchet.
4. Run a real private-profile keyboard journey, review the final diff, update task/evidence and PR #3176, then inspect current-head CI and external feedback. Address verified findings; document any external review-service limitation without bypassing it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the parallel-run sync flag with Textual prevent(Input.Changed). The handler now ignores stale queued values and values already represented by the current draft/config, covering initial mount and category return without suppressing the next genuine edit. Validation, screen-owned drafts and the configuration writer remain unchanged; no other control family was modified.

The permanent origin regression failed on the original PR source with extra max_parallel_runs:2. A separate narrowed re-entry regression also failed specifically on the untouched parallel-run key. Final three cases pass (159.41s), including compact/wide real-writer invalid-0/Save-12/confirmed-Revert journeys and field-search/category-return checks. Existing direct-call max-parallel save plus two Settings size checks pass (58.14s); the two neighboring paste-origin cases also pass. New test lint/format, generated CSS, worker contract, Backlog and PR census guards pass. Settings' 13 inherited Ruff findings are unchanged; the screen shrank to 35,804 lines and its pin was tightened. Three regression entries were appended to the existing UI gate (336 total), preserving prior entries.

Full private-profile keyboard verification used F6 to reach the action pane, s to save 12, then r and Tab/Enter to confirm discarding a later edit. The control and private TOML returned to 12 with a clean UI and zero unhandled exceptions. Independent product/test review and final test-setup recheck found no actionable issues. Detailed original failures, test-setup corrections and limits are in [qualification](../../Docs/superpowers/qa/2026-10-19-task-35680-qualification.md).

ADR required: no new ADR; [ADR-033](../decisions/033-settings-commit-models-three-honestly-labeled.md) and [ADR-031](../decisions/031-tui-keybinding-and-footer-hint-conventions.md) continue to govern. Modified the screen handler/sync, permanent regression file, existing save test's private-profile marker, size ratchet and UI census. Updated PR #3176 and follow-up analysis. The unavailable Backlog CLI required direct Markdown updates verified by both repository Backlog guards. No full suite ran. Remote CI/review results are reported separately; Qodo reported exhausted workspace credits rather than code findings.
<!-- SECTION:NOTES:END -->
