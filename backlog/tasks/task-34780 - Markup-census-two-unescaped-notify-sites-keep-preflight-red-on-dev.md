---
id: TASK-34780
title: 'Markup census: two unescaped notify sites keep preflight red on dev'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-10 17:49'
updated_date: '2026-10-10 18:15'
labels:
  - ui
  - markup
  - preflight
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
dev's preflight is red: the TASK-1513 markup-interpolation census reports two notify sites that interpolate runtime text into a markup-parsed toast without markup=False -- the Console message Delete/Undo flow and the Chatbook SmartContentTree load-failure path. Exception text carrying Rich markup metacharacters (e.g. '[/b]') would raise MarkupError when the toast renders, or render mangled. Both sites pre-date the census commit; its baseline was generated against an older base and never re-run after rebase, so dev was red from the moment it landed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every notify in the Console message Delete/Undo flow that shows exception text or an interpolated runtime value (Delete refusal, Undo refusal, Undo success) renders it literally (markup=False), per the TASK-1513 convention
- [x] #2 SmartContentTree's content-load failure toast (worker and legacy async paths) renders the exception text literally (markup=False)
- [x] #3 A regression test per site shows a bracket-containing message (e.g. '[/b]') reaches a real toast verbatim without raising; it fails on dev and passes with the fix
- [x] #4 The markup-interpolation census check and ./scripts/preflight.sh pass, with no census row added or grown to hide either site (the census only shrinks)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the TASK-1513 convention (CLAUDE.md + check_markup_interpolation.py): notify sites that interpolate runtime text pass markup=False.
2. Write failing regression pins in Tests/UI/test_markup_interpolation_widget_pins.py driving the real code with a "[/b]" message and a real toast (run_test(notifications=True)).
3. Pass markup=False on the Delete-refusal, Undo-refusal and Undo-success toasts in message_delete; route both SmartContentTree load-failure paths through one markup=False _report_load_failure.
4. Shrink the census (only the two now-stale rows drop); run the census check, preflight, and the touched modules' test files against an origin/dev baseline.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed both census-flagged notify sites the way TASK-1513 prescribes for notify (markup=False, no allowlisting), plus the sibling exception-text toasts in the same flows.

Why dev went red: neither site was introduced after the census. The census commit (96ddb10496, TASK-1513) pinned a baseline generated on an older base and was red on its own tree after rebase: TASK-33628.5 (060919ae72, merged in #3031) had moved the "Restored N messages" toast from `_offer_receipt.settle` into `_delete.undo`, and the B18 perf change (f7a47c0609, merged in #3043) had copied SmartContentTree's pinned `load_all_content` toast into a new `_report_load_failure`. The census keys on qualname, so both read as new sites. CI did not catch it: derived-artifacts.yml does not run check_markup_interpolation.py (TASK-1513's owner note); only preflight and the non-UI pytest job's `test_guard_is_green_against_the_committed_census` do.

Changes:
- `UI/Console_Modules/message_delete.py`: markup=False on the Delete refusal (`str(exc)` of a storage ValueError), the Undo refusal (`ConsoleDeleteUndoError`, which can carry a storage ValueError's text) and the Undo success toast.
- `UI/Widgets/SmartContentTree.py`: `_report_load_failure` passes markup=False; the legacy `load_all_content` path now calls it instead of keeping a second copy of the toast.
- `scripts/markup_interpolation_census.tsv`: regenerated with --write after the fix; the only change is two rows dropped (`message_delete _offer_receipt.settle`, `SmartContentTree.load_all_content`), both now 0. Nothing added or grown.
- `Tests/UI/test_markup_interpolation_widget_pins.py`: four pins that drive the real code with "disk said [/b] no" under `run_test(notifications=True)`. Without that flag run_test leaves out the ToastRack and the Toast never renders, so the markup parse is never exercised. Each pin asserts the notification is verbatim with markup=False and that the mounted Toast renders it.

Evidence (venv 3.12, worktree on origin/dev a1dffa5fcb):
- Pins with the production diff reversed: 4 failed / 9 passed, and the toast render raises `MarkupError: closing tag '[/b]' does not match any open tag`. With the fix: 13 passed.
- `check_markup_interpolation.py`: FAIL (2 new) on dev, OK after. `./scripts/preflight.sh`: exit 0, all checks passed.
- Touched test files compared with dev by test id. Tests/Scripts/test_check_markup_interpolation.py: 1 failed (test_guard_is_green_against_the_committed_census) on dev, 16 passed after. test_smart_content_tree_perf: 5 passed on both. test_console_delete_off_loop: 14 passed on both. test_smart_content_tree_tooltips: setup ERROR (RecoveryRequired raw_participant_not_installed) on both, environmental. test_console_message_delete_outcomes / _undo: some ChatScreen tests time out before the delete is armed ("Timed out waiting for #console-message-action-more-..."), on both sides, and which ids fail changes between runs (dev 1+3 reds, after 1+1). The only after-side red not red on dev, test_undo_is_refused_and_stays_on_offer_while_a_dispatch_is_pending, passes when run on its own (1 passed).
<!-- SECTION:NOTES:END -->
