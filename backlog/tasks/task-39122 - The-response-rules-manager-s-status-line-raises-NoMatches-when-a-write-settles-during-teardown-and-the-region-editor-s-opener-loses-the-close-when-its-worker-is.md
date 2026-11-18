---
id: TASK-39122
title: "The response-rules manager's status line raises NoMatches when a write settles during teardown, and the region editor's opener loses the close when its worker is cancelled before mount"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: '2026-11-07 05:58'
labels:
- follow-up
- ui
- console
- personas
priority: low
references:
- tldw_chatbook/Widgets/Console/response_rules_modal.py
- tldw_chatbook/UI/Persona_Modules/portrait_regions.py
- Tests/UI/test_response_rules_modal.py
dependencies:
- TASK-38979
- TASK-38980
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-modals, which lands TASK-38979 and TASK-38980. Line numbers are dev's.

**The status line.**
- `ResponseRulesModal._status` (`Widgets/Console/response_rules_modal.py:193-195`) guards its `query_one("#rr-status")` on `is_mounted`. Textual 8.2.8 sets that flag True on mount and never resets it.
- So a write that settles while the dialog's children are being removed raises `NoMatches`. `_buttons` (`:446-447`) already guards by querying, with the comment "Owned workers can finish while Textual removes the dialog's children."
- It was seen in a round-11 run: `test_inherited_exclusion_masks_and_exact_promotion_preview` raised `WorkerFailed: … NoMatches("No nodes match '#rr-status' …")` from `_action` after `await self._reload(...)`. That path is unchanged from dev.

**The region editor's opener.**
- `edit_regions`' `finally` (`UI/Persona_Modules/portrait_regions.py:127-131`) retires the editor only `if modal.is_mounted`, and clears the controller's reference either way.
- If the worker is cancelled after the push but before mount, the `finally` skips the retire. A probe of that shape leaves the editor on the stack once it mounts, and a later `discard()` no longer finds it.
- On fu/r11-modals `PortraitRegionEditor.retire` handles a pre-mount call, but this `finally` never makes one.
- This path was not reproduced through the real Personas controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A response-rules action that settles while the manager's children are being removed raises nothing; a deterministic test is red on dev
- [x] #2 When the regions worker is cancelled after the push and before mount, the editor still closes and the controller's `close()` returns; a test that holds the mount is red on dev
<!-- AC:END -->

## Implementation Plan

1. `ResponseRulesModal._status`: guard by querying `#rr-status` (as `_buttons` already does) instead of the sticky `is_mounted`, so a write that settles while the dialog's children are removed repaints nothing and raises nothing.
2. `edit_regions`' `finally`: call `modal.retire()` unconditionally; since TASK-38979 `retire()` queues itself until mount when called before it, so a worker cancelled between push and mount still closes the editor.
3. Tests: a `disable` case in `test_manager_detached_subtree_callbacks_do_not_read_widgets` (hold `set_binding`, remove `#response-rules-dialog`, release); a `cancel` case in `test_retiring_the_editor_before_it_mounts_still_closes_it` (hold `on_mount`, cancel the regions worker, release, then `close()`). Show both red on the base first.

## Implementation Notes

Both fixes are one-line guards at the seam every caller routes through.

- **Status line.** `ResponseRulesModal._status` now updates whatever `#rr-status` a query finds (none once Textual has removed the dialog's children), the idiom `_buttons` already used. The sticky `is_mounted` guard is gone. `_action`'s `except Exception` branch also reports through `_status`, so the `test` path's `_show_examples` miss is covered by the same guard.
- **Region editor opener.** `edit_regions`' `finally` calls `modal.retire()` unconditionally. Since TASK-38979, `retire()` queues itself with `call_later` until the editor mounts, so a worker cancelled between push and mount still closes the editor. On the normal path the editor is mounted and `retire()` does what it did before (its dismiss is once-only).

**Tests (red on base, green on head).**
- `Tests/UI/test_response_rules_modal.py::test_manager_detached_subtree_callbacks_do_not_read_widgets[disable]`: holds `set_binding` in `run_db_call`, removes `#response-rules-dialog` (the status line sits outside `#rr-body`), releases. Before the fix: `NoMatches: No nodes match '#rr-status' on ResponseRulesModal()`. It runs in the console-p0-gate step, which lists the node without a parameter.
- `Tests/UI/test_portrait_region_editor.py::test_retiring_the_editor_before_it_mounts_still_closes_it[cancel]`: holds the editor's `on_mount`, cancels the regions worker, releases, then awaits `controller.close()`. Before the fix: `assert PortraitRegionEditor() not in [Screen, PersonasScreen, PortraitRegionEditor()]`. Appended at the census tail.

**Runs (load 44-64).** Both files: 61 passed at `-n 4`. Covering files (console response rules, end-to-end rules, Personas portrait performance, modal dismissal, quit-in-flight modals and hooks, rule write contention): 273 passed, 1 failed. The failure, `test_console_response_rules.py::test_active_run_copy_takes_priority_over_an_earlier_rule_verdict` (`'Waiting for reply…' == 'Streaming response.'`), fails the same way alone on base 8b762b52fa.

**Lane cost** (serial, the steps' flags, 3 runs, load 43-58): `[disable]` 5.2-8.1 s, `[cancel]` 5.8-8.2 s. The console-p0-gate step's 55 targets ran 269 passed in 1021 s at load 45-54.

Files: `tldw_chatbook/Widgets/Console/response_rules_modal.py`, `tldw_chatbook/UI/Persona_Modules/portrait_regions.py`, the two test files, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
