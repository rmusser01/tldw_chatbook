---
id: TASK-39120
title: "Four Console UI test files are red on dev and run on no PR lane: some behind the RecoveryRequired wall, some on drifted harnesses, and some with assertions that may be real regressions"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 07:31
labels:
- follow-up
- tests
- console
- dev-red
priority: low
references:
- Tests/UI/test_console_store_continuity.py
- Tests/UI/test_console_fleet_survivor_tick.py
- Tests/UI/test_console_inspector_section.py
- Tests/UI/app_factory.py
dependencies:
- TASK-38111
- TASK-39044
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. Line numbers are dev's.

**The runs.** On dev at `-n 4`:
- plain: 16 failed, 37 passed (load 30-54);
- under a `bootstrap_profile` marker plugin: 6 failed, 13 passed (load 28-35).

RR below is `RecoveryRequired: raw_source_selection_changed`, raised at `Tests/UI/app_factory.py:130`.

**`Tests/UI/test_console_store_continuity.py`: 5 of 6 red.**

| Node | Plain | Under the marker |
|---|---|---|
| `manual_stream` | RR | real red: `:255`, the view shows the "Setup complete …" empty state instead of the reply |
| `reattach_render_failure` | RR | `TimeoutError` at the 2 s wait at `:335`; cause undetermined |
| `transcript_payload` | RR | real red: `:624`, the active leaf does not point at the last persisted row |
| `rapid_route` | RR | passes |
| `a_wake` | real red: `:546`, "the wake notice is in the store but was never RENDERED on return" | same |

**`Tests/UI/test_console_fleet_survivor_tick.py`.**
- All 7 red nodes are RR only, and pass under the marker; TASK-32470's entries for this file are stale.
- `test_survivor_tick_observes_new_published_usage_without_a_new_timer` failed once under the marker at `:223` (`_console_fleet_survivor_timer is None`).

**`Tests/UI/test_console_inspector_section.py`.**
- The two summary tests are real reds: `:199` (`assert 120 == 60`) and `:222` (`assert 123 == 60`).
- The header renders 6 rows high despite its `h-1` class. `_SectionHarness` (`:85`) appears not to load the app stylesheet bundle that the utility classes moved into in ae9093a714, which suggests drift.
- A third node is red too, but TASK-39044 fixes it on round-11 branch fu/r11-modals.

**Also.** `test_legacy_completion_intake_repaints_the_pause_without_recovery` is RR when run plain and passes under the marker.

**Lanes.** None of these files is in the census or `derived-artifacts.yml`. They run only in `test.yml` (push to main).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The RR-only nodes pass under their plain command (with TASK-38111's fix, or this task's own); the plain-run table above is the before evidence
- [x] #2 Each real red is classified as test drift or a product defect, with the direction recorded: the store-continuity render, active-leaf, wake-render and timeout nodes, and the two summary tests; any product defect gets a test that is red on dev
- [x] #3 Each file runs on a PR lane with measured times (never the fast PR contract), or the notes record why it does not
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run the four files plain and under a scratch `bootstrap_profile` plugin on base 8b762b52fa, and tag each red RR, drift or product.
2. Do not fix RecoveryRequired (brief: TASK-39141's stream owns it). List the RR nodes instead.
3. For each real red, find the commit or mechanism, and record which side should change.
4. Fix the test drift in the test files and check the fixes under the temporary marker. Record any product defect with its red test.
5. Put each file that runs clean under its plain command on a PR lane (UI census tail). Record why the others stay off.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Classified every red on base 8b762b52fa (2026-11-07). The fixable test drift is fixed. RecoveryRequired is listed for TASK-39141's stream and not fixed (brief: only that stream fixes RR). One product defect is recorded with its red test.

**Runs on base, `-n 4`.**
- The four files plain: 27 failed, 38 passed (load 39-47). That is the before evidence for AC#1. The 12 `test_console_automatic_work_pause.py` nodes are the "fourth file", which the filing only named by one node.
- Under a scratch `bootstrap_profile` plugin, store continuity + fleet survivor + work pause: 6 failed, 21 passed (load 50-60).

**RR only: fails plain with `RecoveryRequired: raw_source_selection_changed` at `Tests/UI/app_factory.py:130`, passes under the marker. Not fixed; for TASK-39141's stream.** 23 nodes:
- `test_console_automatic_work_pause.py`: all 12. That is `test_pause_copy_is_painted_with_no_live_children_and_survives_view_clear` ×7, `test_run_history_and_manual_send_leave_old_paused_chain_unchanged`, `test_mount_seed_discovers_saved_results_without_badges`, `test_both_concurrent_wake_sessions_are_recognized_as_active`, `test_legacy_completion_intake_repaints_the_pause_without_recovery` and `test_unreadable_history_does_not_claim_unverified_results_are_saved`.
- `test_console_fleet_survivor_tick.py`: 7. That is `test_the_tick_stops_itself_with_one_final_settle_paint`, `test_unseen_mark_reaches_the_sidebar_browser_rows`, `test_an_idle_console_never_gains_a_survivor_timer`, `test_mount_claim_switches_to_the_settled_conversations_session`, `test_unseen_mark_paints_the_tab_badge_and_viewing_clears_it`, `test_turn_outcome_glyphs_outrank_the_unseen_badge` and `test_survivor_elapsed_advances_with_no_other_interaction`.
- `test_console_store_continuity.py`: 4 nodes are RR when run plain: `test_manual_stream_completes_once_across_real_navigation`, `test_reattach_render_failure_retries_before_decisions_and_view_timers`, `test_transcript_payload_db_and_active_leaf_all_agree` and `test_rapid_route_switching_leaves_console_interactive`. The first three also have the real reds below.

**Real reds (AC#2).**
- `test_console_inspector_section.py`, two summary tests (`120 == 60`, `123 == 60`). **Test drift** from ae9093a714 (2026-09-14), which moved the utility classes (`h-1`) into the app bundle. `_SectionHarness` did not load it, so the header laid out 6 rows tall. Fix: it loads `APP_STYLESHEETS`, as TASK-39044 did for `_ScrolledSectionHarness`. The file passes 38 of 38 at `-n 4` and 38 of 38 serially (50 s at load 148-161).
- Store continuity, render on return (`manual_stream`, red: "Setup complete …" instead of the reply) and wake render (`a_wake`, red at `:546`). **Test drift** from fed0b7257a (TASK-31520, 2026-09-05), which made the Console route reusable. `_seed_console` pushed its own `ChatScreen(app)`. The route then returned the app's installed ChatScreen, whose attach reconcile never ran (a probe read `reconciled=False, running=False` for 18 s, `_messages` empty) because the runtime's view stayed on the pushed screen across the suspend. `_run_headless_wake_turn` also asserted `chat2 is not chat, "screens are never cached"`. Fix: `_seed_console` routes to the app's Console, and the helper asserts the reuse. Both pass under the marker (the wake test plain too: it is `@private_profile_test`).
- Store continuity, active leaf (`transcript_payload`, red at the leaf check). **Product defect.** After a headless wake turn, the durable active leaf stays on the wake notice (SYSTEM row) and never advances to the wake reply that is chained under it. Rows: user → assistant → notice → reply; the leaf is the notice. A 10 s poll after return never saw it move, at base and with the drift fixed. A resume would end at the notice and drop the reply. Its red test is this node (under the marker; RR plain). Not fixed here: proposed as a follow-up below.
- Store continuity, `reattach_render_failure` (TimeoutError at the 2 s wait on base; `[] == [True]` with the seed routed). **Test drift**, same commit. The test needs a *fresh* view to re-attach after navigation, but the reusable route now resumes the same screen: no view start, no `_restore_collapsible_states` timer. To keep its intent it needs a path that still builds a fresh Console (the route cache dropped on a runtime-identity flip). That is a redesign, proposed as a follow-up and not done here.
- `test_survivor_tick_observes_new_published_usage_without_a_new_timer` (private profile, so it runs plain) failed on base at `_console_fleet_survivor_timer is None` after a fixed `pilot.pause(0.5)`. Under the marker at load 50-60, `test_the_tick_stops_itself_…` and `test_survivor_elapsed_…` failed their timing preconditions. **Test timing** (fixed pauses), load-sensitive. Not fixed; follow-up.

**Lanes (AC#3).** `test_console_inspector_section.py` joins the UI census tail as one whole-file entry: 38 tests, 50 s serial at load about 150 (39 s of test phases). The other three files stay off the lanes. They fail under their plain command on RecoveryRequired (and store continuity still has the product red and the redesign), and a PR lane runs them plain.

**Files.** `Tests/UI/test_console_inspector_section.py`, `Tests/UI/test_console_store_continuity.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md` (the reused-route lesson).

AC#1 stays unticked: the RR nodes still fail plain, by the brief's ruling (TASK-39141's stream). After this change, store continuity plain gives 4 failed (all RR), 2 passed; before it was 5 failed (4 RR plus the wake render), 1 passed.

**Proposed follow-ups (not done here).**
- Product: after a headless wake turn, move the durable active leaf to the wake reply, not leave it on the notice. Red test: `test_transcript_payload_db_and_active_leaf_all_agree`, run under a `bootstrap_profile` marker until the RR fix lands.
- Test redesign: retarget `test_reattach_render_failure_retries_before_decisions_and_view_timers` at a path that still builds a fresh Console view (the reusable-route cache dropped on a runtime-identity flip).
- Test timing: `test_console_fleet_survivor_tick.py` waits on fixed pauses (`pilot.pause(0.5)` before `_console_fleet_survivor_timer is not None`, and the stop-edge preconditions). Replace them with bounded waits on the timer and the poll state.
- RR: the 23 nodes above, for TASK-39141's stream.
<!-- SECTION:NOTES:END -->
