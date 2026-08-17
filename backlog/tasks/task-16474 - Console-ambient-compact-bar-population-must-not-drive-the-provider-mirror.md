---
id: TASK-16474
title: Console ambient compact bar population must not drive the provider mirror
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-15 15:10'
labels: []
dependencies: []
---
## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report (2026-08-15): the Console Provider status chip "changes to a seemingly random one". `_console_control_provider` is the in-memory mirror that OUTRANKS `chat_defaults.provider` when fresh session defaults are derived (`UI/Screens/provider_model_resolution.py`, console_provider before chat_defaults), and `on_console_compact_provider_changed` (`UI/Screens/chat_screen.py`, `#compact-api-provider` Select.Changed) cannot distinguish a user change from programmatic population. The mount/population burst therefore set the mirror at every (re)compose — so a mirror the user set by applying settings was silently reverted to the `chat_defaults`-derived value whenever the bar remounted, and the next new session or stale-default refresh derived the wrong provider. `CompactModelBar` also computed `available_providers[0]` (first `[providers]` key in file order) as a fallback default (`Widgets/compact_model_bar.py` compose/on_mount).

Implementation correction (2026-08-15, this task's own red-test run): the bar inside `ConsoleControlBar` is LIVE UI, not dead — despite the `console-hidden-control` class it renders in the expanded header (a visible-text diff at 160x48 showed "OpenAI ▼ gpt-5.6-terra ▼ Temp:" and removing it broke `test_console_native_missing_key_blocks_before_clearing_generic_draft` plus the header layout). The fix is event suppression on programmatic population, NOT bar removal.

ADR required: no — restoring the documented mirror contract ("Mirror native compact provider changes", i.e. user changes); no interface or boundary change.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan (the how)

1. Suppress Select.Changed during programmatic population in `CompactModelBar`: wrap the on_mount value sets, the sync_from_sidebar value/option sets, with `prevent(Select.Changed)` (red test: `test_compact_provider_mirror_untouched_by_mount_population`, which also exercises the sidebar reverse-sync)
2. Drop the `available_providers[0]` arbitrary default in compose/on_mount; leave the provider select on its prompt when chat_defaults.provider is missing/unresolvable (red test: `test_compact_bar_selects_no_arbitrary_first_provider`)
3. Keep the bar mounted in `ConsoleControlBar` (live UI — see the description correction)
4. Run the Console suites (session settings, internals decomposition, native chat flow)

ADR required: no
ADR path: N/A
Reason: restores the documented mirror contract (user changes only); the bar and its placement are unchanged

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Console mount with zero user interaction, `_console_control_provider` / `_console_control_model` remain unset; only a genuine user selection on a visible control writes them, and a later programmatic repopulation/recompose (mount burst, sidebar reverse-sync) never writes or reverts them
- [x] #2 The compact bar stays mounted and functional (it is live UI in the expanded header); its programmatic population simply no longer emits user-intent Select.Changed events
- [x] #3 When `chat_defaults.provider` is missing or unresolvable, the compact bar never surfaces an arbitrary first `[providers]` key as if selected (provider select shows its prompt / stays blank until the user chooses)
- [x] #4 Regression tests added red in `Tests/UI/test_console_provider_persistence_regressions.py` pass: `test_compact_provider_mirror_untouched_by_mount_population`, `test_compact_bar_selects_no_arbitrary_first_provider`
- [x] #5 The Console suites (`Tests/UI/test_console_session_settings.py`, `test_console_internals_decomposition.py`, `test_console_native_chat_flow.py`) stay green (modulo pre-existing dev reds: the unmount-timeout and ctrl-k switcher tests); in particular the task-177 refresh journey still derives from `chat_defaults` when no user selection exists
<!-- AC:END -->
## Implementation Notes

- `CompactModelBar` (`Widgets/compact_model_bar.py`): all programmatic population (on_mount value sets, sync_from_sidebar value/option sets) wrapped in `prevent(Select.Changed)`; provider select `allow_blank=False -> True` because Textual's Select force-picks `options[0]` at mount/set_options when blank is disallowed (that internal auto-pick fires Changed outside any suppression and was the actual ambient mirror writer); blank-sentinel guard added to `handle_compact_provider_change`; the arbitrary `available_providers[0]` fallback removed; on_mount requests one coalesced control-bar sync to keep the rail/inspector refresh cadence the ambient events used to provide.
- `_sync_compact_shell_controls` (`UI/Screens/chat_screen.py`) no longer assigns the mirrors directly (test-only seam; programmatic sync is population).
- COURSE CORRECTION: the original plan removed the CompactModelBar from `ConsoleControlBar` as "dead UI" -- wrong. It renders live in the expanded header (visible-text diff at 160x48); removing it broke the header layout and `test_console_native_missing_key_blocks_before_clearing_generic_draft`. The bar stays; suppression is the fix. That test was also re-pinned: it scraped "missing API key" from COLLAPSED inspector rows whose display stamps depend on rail-cascade timing (pre-existing race); it now waits on the composer's deterministic "add an API key to continue" copy, same contract.
- Modified files: `Widgets/compact_model_bar.py`, `UI/Screens/chat_screen.py`, `Tests/UI/test_console_provider_persistence_regressions.py`, `Tests/UI/test_console_native_chat_flow.py` (test re-pin). Both regression tests verified red on dev state, green after.
