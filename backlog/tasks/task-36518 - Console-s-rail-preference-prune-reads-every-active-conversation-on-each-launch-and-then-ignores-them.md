---
id: TASK-36518
title: Console's rail-preference prune reads every active conversation on each launch and then ignores them
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- console
- performance
- cleanup
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_rail_state.py
- Tests/Performance/test_boot_worker_census.py
- Tests/Chat/test_console_rail_state_prune.py
dependencies:
- TASK-36385
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while checking TASK-36385's list of startup config writes (PR #3225), on dev 5b5920b8e2.

After each launch's first Console sync (chat_screen.py:17684), `_dispatch_console_rail_preference_prune` (chat_screen.py:11870-11890) starts the `_prune_console_rail_preferences` worker. The boot worker census lists it (test_boot_worker_census.py:150). Whenever `[console.rail_state]` holds any key, which is true once a user has toggled a rail section, the worker pages through every active conversation, 1,000 rows at a time, under an owned connection (chat_screen.py:11922-11930). It builds a set of live ids from them.

`collect_prunable_console_rail_keys` then discards that set (`del live_scope_ids`, console_rail_state.py:316). Since TASK-718, prunability depends only on the key's shape: it drops the legacy per-conversation and per-session keys, and current code never writes those keys (`build_console_rail_preference_key`, :250-292). So the scan is dead work on every launch, and the write it guards is a one-time legacy cleanup.

TASK-34406 (In Progress) moves `_prune_console_rail_preferences` into its own `console-rail-prune` worker group and changes the boot-census row this task's AC#3 touches. Whichever lands second rebases onto the other.

The other startup writers on TASK-36385's list were checked too, and each writes once by design:
- `_stamp_new_profile_library_lifecycle` writes only on a profile this run created.
- `_persist_env_key_notice_flag` writes only while the notice flag is unset (`env_keys_that_silenced_first_run`, first_run_setup_state.py:982).
- The Persona Buddy migration writes only for a legacy `PersonaBuddySelection`, which a successful migration replaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a profile whose rail state holds no legacy per-conversation or per-session key, a launch that opens Console reads no conversation rows for the prune and starts no prune worker
- [ ] #2 On a profile that holds such keys, exactly those keys are deleted, once, and the current layout keys are kept
- [ ] #3 The boot worker census and the prune tests pass, updated where the worker no longer starts
<!-- AC:END -->
