---
id: TASK-16815
title: Three more dev-red rows switcher pair and unmount repair
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-16 14:55'
labels:
  - test-health
dependencies: []
---
## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-08-16 while closing PR #1720 (companion to TASK-16480's six, of
which none overlap): three more tests are red on untouched dev (verified on
dev tip 8f1671f3a and attributed commit-by-commit).

- `Tests/UI/test_console_native_chat_flow.py::test_ctrl_k_opens_session_switcher_and_activates_native_session`
- `Tests/UI/test_console_native_chat_flow.py::test_switcher_rename_choice_chains_to_rename_modal`
- `Tests/UI/test_console_session_settings.py::test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume`

Attribution (bisect/parent-check evidence, 2026-08-16):

1+2. ONE root cause: `ChatScreen.action_open_console_session_switcher`
raises `AttributeError: 'ChatScreen' object has no attribute
'_current_console_conversation_id'` (`tldw_chatbook/UI/Screens/chat_screen.py:3003`)
-- the method exists only on the session controller
(`UI/Console_Modules/session.py:1999`). Introduced by 520b1ec12 (browser
consolidation, PR #1661): its diff adds the unqualified call while removing
the correct `self._session._current_console_conversation_id()` forms.
Verified: test passes at 520b1ec12^, fails at 520b1ec12. The rename-chain
test fails downstream of the same error (the switcher modal never opens).
USER-FACING IMPACT: Ctrl+K switcher is dead, and the same missing-method
call in the `/research` handler (`chat_screen.py:16887`, from e1f3a4424,
deep-research PR #1670 lineage) breaks `/research <question>` at runtime.

3. First bad commit 0a79c4d1c ("wip(console): runtime lifetime + teardown
split -- CHECKPOINT, 4 tests red", task-15860 arc, merged via PR #1680).
Observable split on dev tip: the unmount-time repair SCHEDULING still works
(`app._console_roleplay_repair_generation == 1` asserted and passes) but the
resume-side CONSUMPTION never happens
(`_console_roleplay_repair_consumed_generation` never set; the durable
roleplay writes never flip to "Cecelia"). The checkpoint's own message says
NOT reviewed / NOT complete; this red rode through the merge unreconciled.

ADR required: no
ADR path: N/A
Reason: test-health attribution and repair of existing behavior; no new
architectural decision.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan (the how)

1. Red tests are the three dev-red rows themselves (verified red on dev tip 8f1671f3a)
2. Tests 1+2: add the missing ChatScreen delegation seam `_current_console_conversation_id` -> `self._session._current_console_conversation_id()` (one-line delegator, matching the screen's existing controller-delegation pattern); this fixes the switcher pair AND the `/research` call site at once
3. Test 3: trace the resume-side repair consumption drop in the task-15860 teardown split (instrument `_consume_pending_console_roleplay_repair` and the unmount ordering), fix at the dispatch point, do not loosen the test
4. Run the three red tests + their whole suites (native chat flow, console session settings) + lint
5. Task closeout, PR

ADR required: no
ADR path: N/A
Reason: restoring pre-existing behavior across a refactor seam; no new decision

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fix restores a `ChatScreen` delegation seam for `_current_console_conversation_id` (or corrects the two call sites to `self._session._current_console_conversation_id()`); Ctrl+K switcher and rename-chain tests pass on dev
- [x] #2 `/research <question>` no longer raises the missing-method AttributeError (its call site shares the fix; covered by a test or the existing research suites)
- [x] #3 The unmount-timeout test's repair chain consumes on resume again (`_console_roleplay_repair_consumed_generation` set, durable roleplay writes updated); traced to the specific teardown/lifetime change that dropped the dispatch and fixed there rather than by loosening the test
- [x] #4 The three modules' relevant suites pass whole on dev
- [x] #5 Findings recorded against TASK-16480's inventory so the dev-red ledger stays one place
<!-- AC:END -->

## Implementation Notes

- Tests 1+2 (switcher pair): added the missing one-line delegation seam
  `ChatScreen._current_console_conversation_id` -> session controller
  (`UI/Screens/chat_screen.py`). Fixes Ctrl+K (broken by 520b1ec12/PR #1661)
  and `/research <question>` (broken by e1f3a4424/PR #1670 lineage) with one
  seam; both switcher tests green, research command test green.
- Test 3 (unmount repair): traced to a stale fixture, not a product break.
  task-15860 made the Console runtime/store app-owned, so the test's two
  co-mounted ChatScreens share ONE store; the per-screen persistence pair
  aliased and the repair write bound to the hung double (stack dump: writer
  blocked in HungPersistence via persist_roleplay_projection_plan). Reworked
  the fixture to one shared double whose FIRST system write hangs and later
  writes record -- every original assertion preserved (teardown bound,
  repair generation/consumed, durable identity flip, gc collectibility,
  clean loop). Whole suites: 501 passed (was 498+3 red), decomposition 141.
- Modified files: `tldw_chatbook/UI/Screens/chat_screen.py`,
  `Tests/UI/test_console_session_settings.py`, TASK-16480 ledger note.
- Deviation from plan: AC #3 anticipated a product-side dispatch fix; the
  stack-level trace showed the dispatch/repair machinery works as designed
  once the fixture matches the app-owned store. Documented rather than
  forcing a product change.
