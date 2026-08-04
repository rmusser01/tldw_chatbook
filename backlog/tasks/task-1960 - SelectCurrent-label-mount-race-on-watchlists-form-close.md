---
id: TASK-1960
title: SelectCurrent #label mount race on the Watchlists Sources form-close recompose
status: In Progress
assignee: []
created_date: '2026-08-02 17:20'
labels:
  - watchlists
  - textual
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split off task-1345 (the "Select/Input mount race" half of that task's original title). Task-1345's
confirmed root cause and fix (sticky-until-confirmed `_pending_create_focus`) resolved the
**focus-drop** symptom completely. This task is the **other, separate** symptom named in
task-1345's description — `NoMatches` on `SelectCurrent` — which task-1345's fix does not touch.

`Tests/UI/test_watchlists_source_create_form.py::test_a_source_can_be_created_end_to_end_through_the_form`
fails intermittently **in isolation** (zero ordering involved): reproduced 2/2 in isolation on the
unmodified `dev` baseline, and confirmed to still fail 2/2 in isolation with task-1345's sticky-focus
fix applied — proving the two symptoms have independent root causes despite sharing a task history.

**Root cause, found with `TEXTUAL=debug`** (this env var makes Textual print *every* captured
exception for a test, not just the first one pytest shows by default — essential here, since the
default view hid that all 3 toolbar filter `Select`s fail the same way in a single run):

```
Select._on_mount -> _init_selected_option -> self.value = hint -> _watch_value
  -> select_current.update(prompt) -> SelectCurrent.query_one("#label", Static)
  -> NoMatches: No nodes match '#label' on SelectCurrent(...)
```

`Select._watch_value` already guards the case where `SelectCurrent` itself isn't mounted yet
(`except NoMatches: pass`) — but not the narrower case where `SelectCurrent` **is** mounted (so
`self.query_one(SelectCurrent)` succeeds) while `SelectCurrent`'s *own* child (`#label`, a `Static`
its `compose()` yields) has not finished mounting. `Select._on_mount` assumes it has.

It happens specifically on the recompose that **closes** the create-source form after a successful
submit — never on the *opening* recompose, which mounts the same 3 toolbar `Select` filters
(`sources-type-select`, `sources-status-filter`, `sources-active-filter`) without incident in the
same test. At the moment the close-recompose runs, `WatchlistsCollectionsScreen._create_source` has
a worker chain concurrently active (`_refresh_overview_data`, `_load_sources`, `_load_tree_data`) —
`handle_create_source_requested`'s own comment already documents "`_create_source` ... can ...
trigger a full-screen recompose fast enough to win [a] race", i.e. this general hazard class
(concurrent recomposes racing async worker chains) is already known, if informally worked around,
elsewhere on this same screen.

Per Textual's own mounting code (`message_pump.py:_pre_process`, `widget.py:AwaitMount.__await__`),
a widget's `Mount` event is *supposed* to be strictly ordered after its own `Compose` event (which
recursively mounts, and awaits, its children) — structurally this should make the crash impossible.
That it reproduces anyway points to a genuine asyncio task-scheduling interaction under concurrent
load that has not been fully explained, not a simple ordering bug this task's author fully
understands yet.

### What was tried and explicitly rejected (both measured, neither shipped)

1. Swapping `_finish_create_submit`'s scheduling from `self.call_later(...)` to
   `self.call_after_refresh(...)` in `SourcesPane._submit_create_form` — still failed ~4/5 runs.
2. Running the same close (`_finish_create_submit`) from a freshly spawned worker task
   (`self.run_worker(...)` instead of `call_later`) — this ACTUALLY reduced the failure rate
   substantially: 15/15 clean in plain isolation, but repeated testing of the exact scenario AC#2
   cares about (`Tests/UI/test_watchlists_content_pane.py` immediately followed by this test) still
   showed 2/8 failures (~25%, down from ~100% before). Per this project's own established rule for
   this exact task ("a shrunk race is a hidden race" — see task-1345's history, where three earlier
   narrow mitigations were measured and deliberately not shipped for the same reason), this was
   reverted rather than shipped. **Also introduced a real regression while it was in place**:
   `Tests/Watchlists/test_watchlists_sources_pane.py::test_sources_pane_new_source_form_posts_request`
   (a bare-`SourcesPane` harness with no real screen) depends on the pane closing its OWN form
   after submit regardless of any listener; an alternate version of this experiment that moved the
   close into `WatchlistsCollectionsScreen._create_source` instead broke that contract entirely
   (the form never closed at all in that harness). Any future fix must preserve
   "`SourcesPane` closes its own form after submit, independent of whether anything is listening
   for `CreateSourceRequested`" as an invariant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The mechanism behind the `SelectCurrent`/`#label` mount race on the form-close recompose is understood well enough to fix at the mechanism, not just measured to reduce its frequency
- [ ] #2 `test_a_source_can_be_created_end_to_end_through_the_form` passes deterministically (10/10) both in isolation and immediately after `Tests/UI/test_watchlists_content_pane.py`, with no sleep or bounded-retry involved in the fix
- [ ] #3 `Tests/Watchlists/test_watchlists_sources_pane.py::test_sources_pane_new_source_form_posts_request` (and the rest of that file) stays green — the fix must not depend on `WatchlistsCollectionsScreen` being present
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Not started. Candidate directions, cheapest/least risky first:
1. Re-examine why `AwaitMount`'s wait on `_mounted_event` (which should make `SelectCurrent`'s own
   `#label` fully mounted before `Select._on_mount` runs) doesn't hold under this specific
   concurrent load — this task's author was not able to fully explain the empirical failure against
   the structural guarantee in the time available; a `textual` maintainer / upstream issue search
   may already know this shape of bug.
2. Stop tearing down and rebuilding the toolbar's 3 filter `Select`s on every `show_create_form`
   toggle at all -- they have nothing to do with the create form. This means removing
   `recompose=True` from `show_create_form` and hand-managing the create form's own subtree
   (mount/remove it directly in `watch_show_create_form`/a dedicated method) instead of relying on
   `SourcesPane.recompose()`'s current "tear down everything, remount everything" behavior. This is
   the structurally "right" fix but is a real architecture change: it would require reworking
   task-1345's sticky-focus mechanism (built around `recompose()` firing on `show_create_form`
   changes) and re-validating every existing geometry/tab-order test in
   `test_watchlists_source_create_form.py`, since they currently all rely on a full pane recompose.
   Do this as a deliberate, reviewed step, not a quick patch.
3. Whatever the fix, re-run the two experiments already tried (this task's Description) to confirm
   they are actually subsumed/no longer necessary, and add a deterministic (non-flaky) regression
   test alongside the existing intermittent one if practical.
<!-- SECTION:PLAN:END -->
