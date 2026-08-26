---
id: TASK-2705
title: 'Console /rewind: cancelling the menu leaves "/rewind" in the composer'
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31'
updated_date: '2026-08-20 07:57'
labels:
  - console
  - polish
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dismissing the Rewind menu with Esc (or "Never mind") leaves the literal
`/rewind` text sitting in the composer draft, so the next thing the user
types continues after it (e.g. `/rewind/bogus` → unknown-command hint). The
other slash commands that clear the draft on successful dispatch behave the
same way on cancel, but `/rewind` is the one whose menu is routinely
opened-and-cancelled while browsing. Observed live on dev @ ff435772c
(G1 user-guide session, 2026-07-31); documented as a quirk in
`Docs/User_Guide/console/branching-and-rewind.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After an argument-free `/rewind` successfully opens the Rewind menu,
      Esc or "Never mind" leaves the invocation consumed while preserving
      text typed after the Enter keypress (empty if there was no later text).
- [x] #2 If `/rewind` cannot open because there are no prompts, its captured
      invocation is restored and the existing warning is shown.
- [x] #3 A modal-launch failure or a changed/replaced composer never loses or
      clears the user's current draft.
- [x] #4 Choosing "Restore to here" still replaces the draft with the restored
      prompt text (existing behavior preserved).
- [x] #5 The User Guide quirk note is updated/removed to match.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: this is a localized command-draft cleanup bug fix within the existing
Console/composer contracts; it changes no storage, service boundary,
dependency, security policy, or long-lived architecture.

1. Add mounted RED tests for keyboard and visible-Send cancellation, Restore,
   no-row refusal, non-empty arguments, launch failure, and stale composers.
2. Add the narrow argument-free `/rewind` branch at the existing command-send
   boundary and return an opened/refused result from the rewind handler.
3. Guard visible-Send cleanup by composer identity, edit serial, generation,
   and dispatched text; restore keyboard stashes on refusal or exception.
4. Remove the resolved User Guide workaround and run the bounded rewind/send/
   safe-dismissal test and static-analysis matrix.
5. Complete independent review, task evidence, acceptance criteria, and Done
   status before branch integration.

Detailed executable plan:
`Docs/superpowers/plans/2026-08-19-task-2705-rewind-cancel-draft.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation and task-specific review are complete. The cumulative
independent specification and quality reviews both approved the change with no
P0-P2 findings; this closeout preserves their evidence and the final status.

- Commits `3fae3d9a4`, `9a2c58d9a`, and `9bceb7d5d` add the narrow successful
  argument-free `/rewind` consumption path, clarify mounted product-path
  coverage, and protect refused/failed dispatches plus changed or replaced
  composers with stash rollback and identity/revision guards.
- Task 1 RED was `2 failed, 3 passed, 12 deselected`: only keyboard and visible
  Send consumption failed. Focused GREEN was `5 passed, 12 deselected`; the
  adjacent rewind/modal run was `29 passed`. A minor test-quality correction
  then passed `5 passed, 12 deselected`; test Ruff and diff checks were clean,
  and specification/quality re-review approved with no remaining P0-P2.
- Task 2 RED was `4 failed, 1 passed`: keyboard launch lost its stash and the
  identity, edit/retype, and generation cases stale-cleared; visible launch was
  the passing control. Focused GREEN was `5 passed, 17 deselected, 2 warnings`;
  adjacent rewind/modal was `34 passed, 2 warnings`; send-snapshot was
  `13 passed, 2 warnings`. Specification review ran 47 focused tests and found
  no P0-P2; quality review found no issues. Interim test Ruff, `py_compile`,
  and diff checks were clean.
- The one bounded final matrix produced `182 passed, 2 failed, 3 warnings in
  291.51s (0:04:51)`. Both failures are pre-existing modal-inventory baseline
  failures in `test_console_modal_dismissal.py`: the runtime AST resolver tries
  to import `tldw_chatbook.UI.Widgets.Console` and raises
  `ModuleNotFoundError`. Running only those two node IDs at base `6d1c89cce`
  reproduced `2 failed, 2 warnings in 3.93s`. Final warnings were
  `RequestsDependencyWarning: urllib3 (2.6.3) or chardet (6.0.0dev0)/`
  `charset_normalizer (3.4.4) doesn't match a supported version!`,
  `DeprecationWarning: 'audioop' is deprecated and slated for removal in`
  ` Python 3.13`, and `UserWarning: pkg_resources is deprecated as an API.`
  ` See https://setuptools.pypa.io/en/latest/pkg_resources.html. The`
  ` pkg_resources package is slated for removal as early as 2025-11-30.`
  ` Refrain from using this package or pin to Setuptools<81.`
- Targeted static evidence: compileall and `git diff --check` passed. Ruff on
  the touched production/test files reported only the pre-existing unused
  `inspect` import in `chat_screen.py`; MyPy reported the same 42 whole-file
  errors present at base `6d1c89cce`; and Ruff format reported the test file
  would be reformatted at both HEAD and that base. No unrelated baseline debt
  was changed.
- Files changed by the implementation are
  `tldw_chatbook/UI/Screens/chat_screen.py` and
  `Tests/UI/test_console_rewind_restore.py`; closeout preparation removes the
  resolved workaround from
  `Docs/User_Guide/console/branching-and-rewind.md` and records evidence in this
  task and its executable plan. ADR required: no (`ADR path: N/A`). No new
  general lesson emerged, so no lessons document changed.
<!-- SECTION:NOTES:END -->
