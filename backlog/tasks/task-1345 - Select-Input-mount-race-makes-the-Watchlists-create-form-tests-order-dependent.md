---
id: TASK-1345
title: Select/Input mount race makes the Watchlists create-form tests order-dependent
status: In Progress
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_watchlists_source_create_form.py` passes 3/3 in isolation but fails when run after
`Tests/UI/test_watchlists_content_pane.py`. Proven pre-existing rather than caused by Phase D:
with **all** of Phase D's new tests deselected, the create-form tests still fail 3/3 in that order.

Symptoms are a `Select`/`Input` mount race — `NoMatches` on `SelectCurrent`, and a truncated value
(`'orning' == 'Morning'`) indicating the input was read while still mounting.

The failures are intermittent across runs, so a green CI run is not evidence the race is gone.

**Corrected 2026-07-30 (TASK-1343):** the race is **not confined to a named test**. Three
consecutive runs of `Tests/UI/ -k watchlist` produced three different failing sets: it moved among
three tests in `test_watchlists_source_create_form.py` and surfaced once in
`test_watchlists_source_frequency_control.py`. Both files pass in isolation (15/15 and 19/19,
reproduced). Only the two tree-chevron failures are constant.

Consequence for anyone reading a test run: **do not quote a fixed test name as the expected
baseline** for this race. Doing so generates false regression reports when it moves, and false
all-clear when it lands somewhere unlisted. Characterise it by file and by ordering instead.

**Root cause established 2026-07-29 (TASK-1362 Task 5):** `Widget.focus()` only *schedules* focus via
`app.call_later`; any `reactive(recompose=True)` assignment landing in that gap (e.g. `_load_sources`
assigning `sources`) remounts the form, so the callback fires on a detached widget and focus is
**silently dropped** — no error, no retry. The noise-selectors branch raised the frequency under the
`test_watchlists_content_pane.py -> test_watchlists_source_create_form.py` ordering from rare to
~8-in-17. Three narrow mitigations reduced but did not eliminate it; none were shipped, deliberately —
a shrunk race is a hidden race. The durable fix is a policy for the recompose/focus interaction
(TASK-1035 lineage): either focus-restoration after recompose (the `_build_detail_pane` seeding
pattern generalised) or a focus API that survives remount.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The root cause of the mount race is identified and stated, not worked around with a sleep
- [ ] #2 The create-form tests pass regardless of the order the UI suite runs in, demonstrated by running them immediately after the content-pane suite
- [ ] #3 A deliberately re-introduced form of the race fails the tests, proving they discriminate it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Root cause (confirmed in `sources_pane.py:recompose` :638-652): `_pending_create_focus` is READ
and CLEARED before `.focus()` (which only SCHEDULES focus via `call_later`) has landed. A second
`recompose=True` assignment (`sources` from `_load_sources`) firing in that gap remounts the field
— the scheduled callback fires on a detached widget and is dropped — and since the intent was
already cleared, the interleaving recompose recovers nothing (`_focused_create_field_id()` returns
None because focus never landed). Intent lost.
1. Durable fix: make the create-focus intent STICKY until focus is CONFIRMED on a mounted target.
   `recompose` re-applies `_pending_create_focus` without clearing it; a `call_after_refresh`
   confirmation clears it only once `screen.focused.id == target`. Whichever recompose is LAST in a
   burst wins; nothing eagerly discards the intent. Case-2 (user-moved focus, external rebuild)
   still uses `_focused_create_field_id()` and must NOT be yanked back to field 0 — the confirm-clear
   is what prevents that.
2. Deterministic test (AC#2/#3): FORCE the interleave — open the form then assign `sources` in the
   same pump so both recomposes queue, and assert field 0 is focused after settle; run it
   immediately after the content-pane suite. AC#3: reverting to eager-clear reds it.
<!-- SECTION:PLAN:END -->
