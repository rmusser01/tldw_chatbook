---
id: TASK-38950
title: "Every return to the Console restyles the whole screen, and a vertical breakpoint flip restyles it again"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-07 00:34
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Navigation/base_app_screen.py
- Tests/Performance/run_console_mount_profile.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-33273 (To Do) says "Warm resume restyles all 533 nodes", but none of its acceptance criteria covers it. TASK-36511.5 (Done, no code change) measured it and concluded it needs a design change.

- In Textual 8.2.8, `Screen._on_screen_resume` restyles every widget on the screen (`update_node_styles(animate=False)`) when `refresh_styles` is set (`textual/screen.py:1465-1484`). `ScreenResume.refresh_styles` defaults to True (`textual/events.py:919`), and `switch_screen` and `push_screen` post it with that default (`textual/app.py:2616`, `:2622`, `:2956`).
- `Screen._on_resize` calls `update_classes(breakpoints)` (`textual/screen.py:1510-1545`). When a `BaseAppScreen.VERTICAL_BREAKPOINTS` class flips (`tldw_chatbook/UI/Navigation/base_app_screen.py:66`), the whole screen restyles again.
- TASK-36511.5 measured:
  - in the send harness, 538-554 of 555 nodes restyled per resume, for 55-58 ms of CPU at load 9;
  - with the sampler on the compact gates, a 285 ms CPU median per resume, with the breakpoint restyle in 27% of the resume samples;
  - with the warm-visit profiler, a 205 ms median to full ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A return to the Console with no style input changed while it was away restyles fewer nodes than today's 538-554 of 555 in the send harness, with per-resume counts recorded before and after; or, if that is impossible without re-implementing Textual's _on_screen_resume, the notes say why, backed by a measured prototype
- [x] #2 Any reduction keeps everything else the resume handler does: removing the suspended class, bumping stack_updates, refreshing notifications, setting auto-focus, and re-laying out after a size change. A theme or CSS change made while the Console was away still shows on return, and a test for that is red under a variant that always skips the restyle
- [x] #3 A shell-header-compact/shell-header-normal breakpoint flip restyles only widgets whose rules depend on those classes, counted, or the notes say why not
- [x] #4 The warm-visit profiler's full-ready median and the first-mount and boot-budget tests do not regress
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the base first: a scratch probe in the send harness (Console installed, switched to a placeholder screen and back 8 times at 160x45) counts the Console nodes `Stylesheet.apply` restyles per return, and the CPU of each return.
2. Resume: Textual's dispatch runs ChatScreen's own `ScreenResume` handlers before `Screen._on_screen_resume`, which restyles only while `event.refresh_styles` is set. ChatScreen's handler clears that flag when no style input moved while the Console was away: the parsed rules are the very list its last full restyle used, and the App node's classes and pseudo-classes are the ones it used and the ones it left with. A covered screen's own class changes restyle at once, so only these can go stale. Nothing of Textual's is patched, and its handler still does everything else.
3. Breakpoints: ChatScreen sets an empty `VERTICAL_BREAKPOINTS`, so `Screen._on_resize` skips its whole-screen `update_classes`. Its `on_resize` applies `BaseAppScreen.VERTICAL_BREAKPOINTS` itself and restyles only the nodes the rules naming those classes target (the names Textual's `rules_map` matches nodes by), with their subtrees.
4. Tests in `Tests/UI/test_console_projection_restyles.py` (a whole-file census entry):
   - a return with no style change: restyled count, computed-style fidelity against a full restyle, the stack update, and a size change made while away;
   - a theme changed while away shows on return, red under an always-skip variant;
   - a header breakpoint flip restyles exactly the nodes its rules style, and keeps fidelity, red under the base behaviour.
5. Re-measure the probe, run the warm-visit profiler on base and branch, the first-mount and boot-budget tests (`bootbudget.sh`), the covering tests and the size ratchets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Nothing of Textual's is patched or re-implemented. Both helpers live in `UI/Console_Modules/frame.py`.
- **Resume.** Textual's dispatch runs ChatScreen's own `ScreenResume` handler (`_prepare_console_resume`, which was `_arm_console_arrival_focus`) before `Screen._on_screen_resume`. That handler restyles only while `event.refresh_styles` is set, and `skip_unchanged_resume_restyle` clears the flag when no style input moved while the Console was away:
  - the parsed rules are the very list its last full restyle used (`stylesheet.rules` is a new list after every parse);
  - the App node's classes and pseudo-classes are the ones that restyle used and the ones `on_screen_suspend` recorded as the Console left (`note_suspend_style_inputs`).
  - A suspended screen's own class and pseudo-class changes restyle at once, because Textual's `App.update_styles` handles non-App nodes immediately. Only those two inputs can go stale.
  - Every `pop_screen` posts `refresh_styles=False` (Textual 8.2.8 `app.py:3112`), and Textual then restyles nothing. On such a resume the handler compares the App's classes and pseudo-classes with the suspend record. If they moved while the Console was covered, its nodes may hold the old ones, so the handler drops its restyle record and the next return restyles in full. The first version ignored pops entirely, and review r11c-rev1 found that gap.
  - Everything else Textual's handler does still runs.
- **Breakpoints (AC#3).** ChatScreen sets `VERTICAL_BREAKPOINTS = []`, so `Screen._on_resize` skips its whole-screen `update_classes`.
  - `ChatScreen.on_resize` applies `BaseAppScreen.VERTICAL_BREAKPOINTS` with `apply_scoped_breakpoints`. It picks the same class as `Screen._get_breakpoint_classes` and sets it with `update=False`.
  - It then restyles the nodes whose selector names meet the `selector_names` of the rules that name a breakpoint class, with their subtrees. Those names are the key Textual's own `rules_map` matches nodes by, so no other node can be styled by those rules.
  - Today the matching rules are `.shell-header-compact .workbench-header` and `... .workbench-header-subtitle`.
  - The scan over all 5,843 rules took 139 ms at load 86 (r11c-rev1), so its result is cached on the screen. The cache key is the identity of `stylesheet.rules` and the set of breakpoint classes.
  - `BaseAppScreen`'s breakpoint comment now names ChatScreen's scoped override.

**Measured (AC#1).**
- Send harness at 160x45, Console installed and switched to a placeholder screen and back 8 times, through run.sh (scratch probe, not committed), base and branch interleaved at load 33-36.
  - Base: 546 of 546 Console nodes restyled on every return, 94-162 ms of CPU per return (switch plus one pause).
  - Branch: 0 of 546 on every return, 47-54 ms of CPU. The first return took 196 ms.
- Warm-visit profiler (`Tests/Performance/run_console_mount_profile.py`, 8 Library -> Console visits per round, 170x48), 4 interleaved rounds at load 32-78:

| round | base full-ready median | branch full-ready median | base first-interactive median | branch first-interactive median |
|---|---|---|---|---|
| 1 | 897 ms | 483 ms | 384 ms | 25 ms |
| 2 | 873 ms | 384 ms | 293 ms | 26 ms |
| 3 | 568 ms | 175 ms | 225 ms | 16 ms |
| 4 | 282 ms | 169 ms | 85 ms | 18 ms |

**Tests** (in `Tests/UI/test_console_projection_restyles.py`, a whole-file census entry).
- `test_a_return_restyles_the_console_only_when_a_style_input_moved` (AC#1, AC#2) mounts the Console once and drives four returns:
  - a warm-up return;
  - a measured return, where the Console must restyle under a tenth of its nodes (0 of 555 measured), every computed style must equal a forced full restyle, and `stack_updates` must move;
  - a return after a terminal resize made while away, which must lay the Console out at the new size;
  - a return after a theme change made while away, where the test checks that the change styles something in the Console and that the return shows it.
  - Red with the skip turned off (the base behaviour): `a return with no style change restyled 555 of 555 nodes`.
  - Red under a variant that always clears the flag: `the Console came back with styles from before the theme change`.
  - The warm-up return is there because a widget type first mounted after the Console's first resume adds its `DEFAULT_CSS`, and the reparse makes a new rules list. Under load that made the first return restyle in full in 2 of 3 runs, which is the conservative outcome. Found from a probe that logged the inputs and each `Stylesheet.parse` caller (`_register` -> `apply` -> `rules`).
- `test_a_header_breakpoint_flip_restyles_only_what_its_rules_style` (AC#3) dispatches a Resize from 30 to 24 rows straight to the screen. Both heights are under 34, so `-console-compact` does not flip as well.
  - The nodes restyled must be exactly the nodes the breakpoint rules style (11 of 555), the styles must match a full restyle, and a real resize back must restore `shell-header-normal`.
  - Red with Textual's breakpoints restored (the base behaviour): `the flip restyled 555 of 555 nodes; its rules style 11`.
- **Review r11c-rev1 fixes.** A theme switch moves the rules and the App classes together, so a mutant dropping either check passed the first suite. Each check now has a case of its own, and every mutant below was killed by a temporary edit of `frame.py` and run through run.sh:
  - In the first test, a return after an `add_source` rule targeting `#console-shell` was added while away. M1 drops the rules-identity check and fails with `a rule added while away did not show on return`.
  - New test `test_an_app_class_that_moved_while_the_console_was_covered_shows_on_return`, using a harness-local `App.-resume-probe #console-shell { tint }` rule:
    - The App class is set while the Console is current and cleared while it is away. M2 drops the suspend comparison and fails with `an App class cleared while away still styles the Console`.
    - The App class is set while current, then cleared under a pushed screen that pops. Before the pop fix (M3: the drop removed) it was red with `an App class cleared under a popped screen still styles the Console`.
  - The breakpoint test now also adds a rule while the Console is current and flips again. A scan cache that ignores the rules (M4) fails with `a rule added to the sheet was not scanned`.
  - It then applies another breakpoint class over the same sheet. A scan cache that ignores the class set (M5) fails with `a new breakpoint class was not scanned`. The first version of that check passed M5 by accident: its rule targeted `#console-shell`, whose subtree holds every node, so the check now targets `#console-transcript-region`.
- **Review r11c-rev2.** Nothing pinned that a pop with no App change keeps the skip. The covered-screen test now starts by pushing and popping a screen with nothing changed, then returns: under a tenth of the nodes may restyle. M3b drops the record on every pop and fails with `a pop with no App change cost the next return 555 of 555 nodes`.
- After the review fixes, the changed test files at `-n 4` gave 50 passed (load about 110). Preflight passed, and `Tests/CI` gave 1215 passed and 1 skipped.
- AC#2's other resume duties are untouched by construction, because only the flag changes. The suspended-screen class is `""` in this app, and the notifications and auto-focus are not asserted. `stack_updates` and the size-change relayout are asserted.
- 4 repeats of the new tests, serial, at load 50-67: all passed.

**No regression (AC#4).**
- `bash $FU/bootbudget.sh fu-r11-console`: 32 passed.
  - It warned the UI-ready census at 1038/1038 with a snapshot drift of +1. That drift is not from this branch. After the machine's reboot, 3 recounts each on the base and on the branch read 1038, with the same module added against the snapshot, `tldw_chatbook.DB.character_conversation_search`. Before the reboot, one recount each had read 1037.
  - The preimport payload read 426388/426388 LOC on both base and branch.
- Warm-visit profiler: improved, as in the table above.
- First-mount files, `-n 4`: `test_console_first_send_preimport.py`, `test_boot_budget_ratchet_messages.py` and `test_console_mount_profile.py` passed.
- `test_console_first_chat_first_mount.py` is load-flaky on both arms, at `-n 4` and load 34-87:
  - Head failed 6 nodes over 3 of its 9 whole-file runs. Base failed 1 node in 1 of its 4 runs.
  - The base failure, `test_held_shared_defaults_are_written_before_the_workspace_record`, is one of head's.
  - The 4 nodes head failed then passed 24 of 24 in 6 repeats on head at load 59-83, and the last 6 whole-file runs on head were clean.
  - The failure messages were not captured, so the cause is not shown here.

**Covering.** 18 files (resume, suspend, navigation, arrival focus, compact floor, visual parity, long-chat bounds, runtime ownership, edge focus restyles and others), `-n 4`, `-p no:randomly`, head against base with the same command.
- Head: 166 failed, 260 passed. Base: 163 failed, 260 passed.
- Nearly all of the failures are the local `RecoveryRequired` trap (TASK-38111), for example all 7 compact-floor tests on both.
- The 4 `test_console_runtime_ownership.py` nodes that failed only on head passed 8 of 8 on head and on base when rerun. The one base-only failure is a `test_console_workbench_contract.py` node.
- `Tests/CI` at `-n 4`: 1215 passed, 1 skipped. `Tests/Architecture/test_screen_size_ratchet.py` and `test_module_size_ratchet.py`: 205 passed, with `chat_screen.py` at 24614 lines against its 24616 ceiling. The handler docstring was trimmed to keep it under, so no waiver was used.

**Not changed.** `#console-shell`'s `-console-compact` class (`_adapt_console_shell_to_height`) still restyles about 517 nodes when the height crosses 34 rows. It is the same shape, outside this task. Proposed as a follow-up.

**Files:** `tldw_chatbook/UI/Console_Modules/frame.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Navigation/base_app_screen.py` (comment), `Tests/UI/test_console_projection_restyles.py`, `backlog/docs/lessons-textual.md`.
<!-- SECTION:NOTES:END -->
