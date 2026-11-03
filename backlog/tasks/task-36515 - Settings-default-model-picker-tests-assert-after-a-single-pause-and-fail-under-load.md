---
id: TASK-36515
title: Settings default-model picker tests assert after a single pause and fail under load
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: 2026-11-03 10:09
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
- [x] #1 Every assertion in the file that follows an edit, key press or focus change waits for the asserted state with a bounded poll, or the task notes record why that state is set synchronously
- [x] #2 Each rewritten test passes 20 of 20 serial single-node runs, and a negative control (a delayed handler for the state it waits on) fails the old form and passes the new one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Walk every assertion in `Tests/UI/test_settings_default_model_picker.py` that follows an Input edit, a key press, a focus change or a mouse press, and name the state it reads and the message chain that sets it.
2. Replace each single `pilot.pause()` (or `_settle`) before such an assertion with the file's existing `_until` on the asserted state itself, with a predicate that cannot raise; keep the assertions for their messages. Where two steps end in the same state, put a distinguishable step between them so the wait can still fail.
3. Record in the notes any assertion left on a fixed window (absence checks, the modelled press duration) and why.
4. Negative control: a temporary source edit that defers the handlers the waits depend on (screen staging, picker focus render, Custom ID echo); run the old file (temporary copy) and the new file under it; revert.
5. 20 serial single-node runs of each rewritten node, the whole file with `-n 4`, preflight; then notes, ticks, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Every single `pilot.pause()` (or `_settle`) before an assertion that follows an Input edit, a key press, a focus change or a mouse press in `Tests/UI/test_settings_default_model_picker.py` is now the file's existing `_until` (from `test_settings_connect_rows`, 100 x 0.05 s) on the asserted state itself, all of its conjuncts; the assertions stay for their messages. No new helper. Predicates cannot raise: `.get("model")` instead of `["model"]`, and `paints_the_head` returns False on an empty region. The six sites the description names are converted, and so is every other such site in the file: the open-list render after focus (`one_row` x2, `changing_provider`, `discovered_models_stay` x2, `click_on_a_listed_model` x2), the search landing's focus, the typed filter's rows and pick, the click's pick, the revert's confirm dialog and result, the save results after `s`, and the rebuilt picker.

**Two waits needed a distinguishable step, or they could not fail.**
- `a_custom_id_the_save_cannot_write`: the second unwritable id followed the first, and both leave the same state (no model, empty adapter, the same status), so a wait for it would end on the first id's state. Each id now starts from the longest writable id, held and staged, so each wait sees its own id's refusal.
- `a_discovered_model_becomes_the_saved_default`: `press(*served, "enter", "escape", "s")` let `s` land before the pick was staged; the save then had nothing to save, and the "unsaved" poll ended at once on a clean category. The pick is staged first now, then Esc, then `s`. Under the control below the old form failed "save did not finish".
- `koboldcpp_root_and_prefix_root`: after `field.value = prefix`, `_press_discover`'s wait for a listing ended at once on the root's listing if the endpoint's `Input.Changed` (which stages the endpoint and clears the listing) had not landed; a late one then reset the status to the idle copy. The test now waits for the endpoint to be staged and the listing cleared, and for the status widget's own text. This test failed 1 of 32 in the first base run of this task (`-n 4`, load 23): `'No models discovered yet. ...' == 'Endpoint path ...'`. That message has a second cause, found in the review round below.

**Left on a fixed window, on purpose (absence checks a poll cannot wait for).**
- `click_on_a_listed_model_picks_it[True]`: `pause(0.12)` is the modelled press duration; the list must stay open across it, past the field's 0.05 s blur timer. Focus moving to the list is polled first.
- `click_below_an_open_list...` (4 nodes): after the release, `pause(0.1)` is the window in which a wrong control would be activated; `toggled <= {pressed.id}` and `toggled == set()` assert an absence. Focus on the pressed title and the collapse are polled first; the held case's "another control is under the pointer" is polled.

**Set synchronously, so no wait of its own.** The saves (`s`) run synchronously in the binding's action: the mutation, `app.providers_models`, the draft pop and `_provider_save_result` all land in one call, so after the wait for the cleared category (or the refusal text) the mutation list, the lists and the retained draft are read without another poll. `app.providers_models == before` after the pick, and `mutations == []` after a refusal, are absences that the same handler leaves as they were. The post-exit file and request assertions read what the waited-for save and listing already wrote.

**Negative control (temporary source edits, reverted; patches kept in the scratchpad).**
- A: `handle_model_value_changed` (adapter -> draft staging), `handle_provider_endpoint_changed`, the picker's focus re-render and the picker's Custom ID `ModelValueChanged` post each deferred 0.4 s with `set_timer`.
- B (on top of A): the search landing's focus, the temperature field's staging and the roll-back toast deferred 0.4 s.
- Old form (a temporary copy of the file at base) and new form run together with `-n 4`: under A, old 17 failed of 32 and new 32 of 32 passed (load ~27); under A+B, old 19 failed and new 32 of 32 passed (load ~24). The old failures are on the waited states: `'gpt-4o' == 'o3-mini-new'`, `'my-private-mod' == 'my-private-model'`, `'new-chat default' == 'edited *'`, `'gpt-4.1' == ''`, `'o3-mini-new' is not in list`, the search landing's focus on the category button, the koboldcpp status and requests, "save did not finish". Old forms not failed by either control: the 8 synchronous `custom_id_and_the_combined_save` cases (no UI), the 4 `click_below_an_open_list` nodes, whose converted waits read Textual's own focus-on-press and layout, which no product handler of ours sets, and `a_discovered_model_becomes_the_saved_default` under A+B (it failed under A).

**Runs.** Each of the 24 mounted nodes (every async test in the file; each has at least one converted wait) ran 20 times, one pytest process per run with that node alone and no xdist: 480 of 480 passed. Four such loops ran side by side, and other agents shared the 18 CPUs: 1-minute load 15-34, median 24.6. Two runs (the `held-*` nodes' 20th) never started because the scratchpad's `run.sh` had been deleted (it was restored with its earlier content) and were run again. The whole file with `-n 4` passed 32 of 32 three times (load 25-28). Before the change, the first whole-file `-n 4` run of this task failed `koboldcpp_root_and_prefix_root` 1 of 32 (load 23); see above. `ruff check` reports the same 4 RUF059 findings as base; the file is `ruff format`ted, as it was at base.

**Review round (coordinator, at 000fd58dc5).**
- The in-loop wait for the longest id in `a_custom_id_the_save_cannot_write` now asserts `holds_longest()` after it, so a timeout cannot let the next refusal be checked on stale state.
- The koboldcpp test then failed once in a two-file `-n 4` run (load 9), with the same message as the base failure above.
  - A probe at the third press showed the cause: the Discover button still carried Textual's 0.2 s `-active` press effect from the second Discover. `Button.action_press` ignores Enter while that class is set, so no discovery ran (no "running" reset after the press) and the status stayed idle. The waits this task added made the steps faster, so the press landed inside the window more often.
  - `_press_discover` and the third press now wait for `-active` to clear before Enter. This trap is already in `lessons-live-verification.md` (TASK-32739, the same Discover button).
  - Control: the button's `active_effect_duration` stretched to 1.0 s in temporary test copies failed the pre-fix form 4 of 4 and passed the fixed form 4 of 4 (load 13-17). Unstretched, the fixed form passed 24 of 24 runs, 4 at a time (load ~13).

**Files.** `Tests/UI/test_settings_default_model_picker.py`. Census untouched (per-node rows already exist; no test was added, removed or re-parametrized; `Tests/CI/test_ui_pr_gate_census_checker.py` 83 passed).
<!-- SECTION:NOTES:END -->
