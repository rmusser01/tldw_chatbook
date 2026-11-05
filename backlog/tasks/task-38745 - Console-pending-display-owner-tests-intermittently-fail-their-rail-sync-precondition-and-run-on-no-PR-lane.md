---
id: TASK-38745
title: "Console pending-display owner tests intermittently fail their rail-sync precondition and run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-04 06:34
labels:
- follow-up
- tests
- console
- flaky
priority: medium
references:
- Tests/UI/test_console_pending_display_owner_controls.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`Tests/UI/test_console_pending_display_owner_controls.py` has two tests, both `@private_profile_test`:
- `test_pending_fragment_current_owner_and_saved_generation_remain_display_only` (:48);
- `test_pending_fragment_declines_unknown_owner_custom_reader_and_malformed_metadata` (:84).

Both start with the shared `_prepared` helper (:30). It asserts `await projection.warm()` (:33), then `console._sync_console_rail_and_controls() is not False` (:34).

In large `-n 4` runs during round 6, both nodes failed now and then at :34: the warm-up had returned True, and the rail sync still returned False. TASK-38650's notes also record a dev failure in this file. On this base, a single `-n 4` run of this file together with two other files failed the first node at :34 (`assert False is not False`). The load average was about 8 on 18 CPUs.

`_sync_console_rail_and_controls` (`UI/Screens/chat_screen.py:20945`) returns the result of `run_console_presentation_sync`, and False means that sync deferred. So the precondition depends on timing that a successful warm-up does not settle. The helper does not say which deferral it expects to have cleared.

The file runs on no PR lane. It has no entry in `scripts/ui_pr_gate_census.txt` and is not a target in any workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes name, from evidence, which deferral makes the rail sync return False after a successful warm-up
- [ ] #2 `_prepared` waits for the state its later asserts depend on, or, if the deferral is a product bug, the product stops deferring there; the failure is reproduced before the change, under load or with a temporary source edit
- [ ] #3 Both nodes pass 20 of 20 `-n 4` runs of the file, with the load average recorded in the notes
- [ ] #4 The file runs on a PR lane (a census entry or a workflow target), and its measured time in that lane is recorded in the notes
<!-- AC:END -->
