---
id: TASK-39120
title: "Four Console UI test files are red on dev and run on no PR lane: some behind the RecoveryRequired wall, some on drifted harnesses, and some with assertions that may be real regressions"
status: To Do
created_date: 2026-11-07 02:03
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
- [ ] #2 Each real red is classified as test drift or a product defect, with the direction recorded: the store-continuity render, active-leaf, wake-render and timeout nodes, and the two summary tests; any product defect gets a test that is red on dev
- [ ] #3 Each file runs on a PR lane with measured times (never the fast PR contract), or the notes record why it does not
<!-- AC:END -->
