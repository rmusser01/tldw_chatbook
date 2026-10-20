---
id: TASK-35680
title: Keep parallel-run form refreshes out of Settings drafts
status: To Do
assignee: []
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
- [ ] #1 A mounted regression reproduces the extra max_parallel_runs draft value on the pre-fix source with an unrelated staged field and records the queued-event failure.
- [ ] #2 Opening, value-changing refreshes, Save, confirmed Revert and category re-entry do not stage untouched parallel-run values or report a new edit after queued messages settle.
- [ ] #3 Genuine parallel-run typing, including the first edit after refresh/re-entry, retains validation feedback, saves through the existing writer and reverts through the existing confirmation flow; unrelated staged values remain intact.
- [ ] #4 Targeted private-profile regressions and a real keyboard Save/Revert journey qualify the correction, and an appropriate regression enters the existing PR test gate without weakening safety or performance guards.
<!-- AC:END -->
