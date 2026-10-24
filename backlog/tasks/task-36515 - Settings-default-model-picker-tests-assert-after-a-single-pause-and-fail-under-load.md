---
id: TASK-36515
title: Settings default-model picker tests assert after a single pause and fail under load
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
- flaky
- settings
priority: medium
references:
- Tests/UI/test_settings_default_model_picker.py
- Tests/UI/test_settings_connect_rows.py
dependencies:
- TASK-36380
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36380's review round 1 (PR #3224, 2026-10-23). `Tests/UI/test_settings_default_model_picker.py` is listed whole in `scripts/ui_pr_gate_census.txt`, so its flakes fail the required UI Fast Lane. Several of its tests change an Input or press keys, wait one `pilot.pause()`, and then assert state that the change reaches through queued messages (Input.Changed, the picker's adapter, the category draft). One pause is enough only when the screen's queue drains in one idle window.

The sites on dev 5b5920b8e2:
- `test_choosing_a_discovered_model_replaces_a_set_default_and_saves_with_s`: pause at :263, asserts at :265-271.
- `test_revert_puts_the_saved_default_back_in_the_picker`: pause at :364, asserts at :365-366.
- `test_an_unlisted_model_id_is_entered_through_custom_id_and_validated`: pauses at :395, :397, :400 and :403, asserts at :398-409; and `field.value = ...` then one pause at :411-416.
- `test_an_invalid_custom_id_is_rolled_back_when_the_field_is_left`: :438-444.
- `test_a_custom_id_the_save_cannot_write_is_refused_as_it_is_typed`: :569-585.
- `test_an_id_the_save_refuses_is_named_as_the_cause_not_config_access`: pause at :616, asserts at :617-618.

Evidence from TASK-36380's runs (`-n 4`, load average 30-38 on 18 CPUs): `test_an_id_the_save_refuses_is_named_as_the_cause_not_config_access[saved-model]` failed 1 of 4 runs at that PR's head, and on dev `test_an_invalid_custom_id_is_rolled_back_when_the_field_is_left[focus-moves-away]` failed 1 of 3. Three concurrent `-n 4` runs of the whole file on dev at load average 22 passed (31 of 31 each), so the failures need heavier load. The file already imports `_until` for the waits that were fixed earlier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every assertion in the file that follows an edit, key press or focus change waits for the asserted state with a bounded poll, or the task notes record why that state is set synchronously
- [ ] #2 Each rewritten test passes 20 of 20 serial single-node runs, and a negative control (a delayed handler for the state it waits on) fails the old form and passes the new one
<!-- AC:END -->
