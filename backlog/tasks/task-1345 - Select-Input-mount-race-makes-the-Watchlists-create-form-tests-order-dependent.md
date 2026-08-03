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
- [x] #1 The root cause of the mount race is identified and stated, not worked around with a sleep
- [x] #2 The create-form tests pass regardless of the order the UI suite runs in, demonstrated by running them immediately after the content-pane suite -- **narrowed 2026-08-02: scoped to the focus-drop race this task's confirmed root cause covers.** See split note below.
- [x] #3 A deliberately re-introduced form of the race fails the tests, proving they discriminate it
<!-- AC:END -->

**Split 2026-08-02.** This task's title and original description named two symptoms sharing one
history: focus silently dropped (`_pending_create_focus`), and `NoMatches` on `SelectCurrent`. Only
the first has a confirmed, understood root cause (TASK-1362 Task 5, above) and is what this task's
Implementation Plan and fix actually address. AC#2 is fully met for that scope: every test that
depends on the focus-restoration mechanism passes, in either order, across repeated runs of the
content-pane -> create-form pair and repeated isolated runs (see Implementation Notes for counts).

The second symptom (`test_a_source_can_be_created_end_to_end_through_the_form`, `NoMatches` on
`SelectCurrent`) does **not** share this task's confirmed root cause -- it reproduces identically
whether the focus fix is applied or not, and a bounded follow-up attempt at fixing it directly
(below) reduced but did not eliminate it, which this task's own history already established as not
shippable ("a shrunk race is a hidden race"). Split out to **task-1960**
(`backlog/tasks/task-1960 - SelectCurrent-label-mount-race-on-watchlists-form-close.md`), which
carries the `TEXTUAL=debug` diagnosis and both rejected mitigations forward so the next attempt does
not have to re-derive them.

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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** `SourcesPane.recompose()` (`tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`) no
longer clears `_pending_create_focus` before `.focus()` has had a chance to land. It now computes
`restore = self._pending_create_focus or self._focused_create_field_id()`, leaves
`_pending_create_focus` armed across the method, and — if the form is still open and a target was
resolved — calls `.focus()` and schedules a new `_confirm_create_focus(target)` via
`call_after_refresh`. That confirmation clears the intent **only** once
`self.screen.focused.id == target`; if focus hasn't landed yet it re-schedules itself against the
next refresh (a wait on real, observable state — not a sleep or a bounded retry count). If the form
closed while a recompose was in flight, the intent is dropped immediately (it's for a form that no
longer exists). Whichever recompose is LAST in a burst therefore wins, because each one re-applies
`.focus()` against whatever is currently mounted, and the intent survives until one of them is
actually confirmed. Case 2 (user has tabbed elsewhere, then an unrelated rebuild happens) is
unaffected: once confirmed, `_pending_create_focus` is `None` again, so later recomposes fall
through to `_focused_create_field_id()`, which reports the user's current focus rather than
resurrecting the stale opening intent.

**Tests added** (`Tests/UI/test_watchlists_source_create_form.py`):
- `test_a_sources_reload_interleaving_the_open_does_not_lose_focus` — the AC#2/#3 discriminator.
  Forces the interleave deterministically by calling the internal `_check_recompose()` seam
  directly, twice, back to back with no intervening `pilot.pause()`: `pane.show_create_form = True`
  then `await pane._check_recompose()` (runs the opening recompose to completion; its `.focus()`
  call has only *scheduled* the change via `app.call_later`, not landed it, since nothing has
  yielded to the app's own queue yet), then `pane.sources = [...]` and `await
  pane._check_recompose()` again (the interleaving reload, forced to run before that scheduled
  focus lands). Confirmed both directions: fails with `screen.focused is None` when `recompose` is
  reverted to eager-clear (mutation test), passes reliably (5/5 manual repeats) with the fix.
  An earlier version of this test drove the interleave by monkeypatching `Widget.focus` to trigger
  the `sources` reassignment from inside the scheduling call — that version passed even against the
  unfixed eager-clear code, because `app.call_later`'s `set_focus` callback apparently tends to run
  before a `call_next`-queued recompose gets a turn, so by the time the second recompose read
  focus-restoration state, focus had already genuinely landed and the (unrelated) case-2 fallback
  papered over the bug. The direct `_check_recompose()` approach avoids depending on that ordering.
- `test_an_external_rebuild_does_not_yank_focus_back_to_the_first_field` — case-2 regression guard.
  No pre-existing test covered this specific scenario for the create form (searched thoroughly);
  passes on both the fixed and the reverted code, as expected for a "must still work" guard rather
  than a discriminator.

**Known, separate, pre-existing issue — NOT fixed here (AC#2 partial).**
`test_a_source_can_be_created_end_to_end_through_the_form` fails intermittently **in isolation**
(no ordering involved), on the unmodified `dev` baseline and unchanged by this fix: reproduced
2/2 in isolation before this change, and confirmed to still fail 2/2 in isolation with the sticky
fix applied. Root-caused with `TEXTUAL=debug` (prints all captured exceptions, not just the
first): `Select._on_mount` → `_init_selected_option` → `self.value = hint` → `_watch_value` →
`select_current.update(prompt)` → `SelectCurrent.query_one("#label", Static)` raises `NoMatches` —
i.e. a **`Select` widget's own internal mount race**, unrelated to `_pending_create_focus`. It
happens specifically on the recompose that *closes* the form after a successful submit (never on
the *opening* recompose, which mounts the same 3 toolbar `Select` filters without incident), while
`WatchlistsCollectionsScreen._create_source` has a worker chain concurrently active
(`_refresh_overview_data`, `_load_sources`, `_load_tree_data` — the screen's own
`handle_create_source_requested` comment already documents "`_create_source` ... can ... trigger a
full-screen recompose fast enough to win the race", i.e. this general hazard class is known
elsewhere in this screen). Two things were tried and did **not** fix it, ruling out the obvious
narrow explanations: (1) swapping `_finish_create_submit`'s `self.call_later(...)` for
`self.call_after_refresh(...)` (still failed ~4/5 runs); (2) the sticky-focus fix itself (no
effect, confirmed above). This points to genuine asyncio task-scheduling nondeterminism in
Textual's own Mount-event ordering for nested compound widgets (`Select` → `SelectCurrent` →
`#label`) under concurrent load, not anything `SourcesPane.recompose()` controls. A real fix likely
means not tearing down/rebuilding the toolbar's filter `Select`s on every `show_create_form`
toggle at all (a `compose()`/reactive-scoping change), which is a materially different, larger
change than this task's confirmed root cause and plan — left open rather than attempted here.
Recommend a follow-up task specifically for the `Select`/toolbar-churn half of this task's
original title.

**Verification:** isolated file run 16 passed / 1 failed (the known issue above); the
content-pane → create-form ordered pair run 3× back to back: 56 passed/1 failed, 56/1, 55/2 (the
known issue, 1 or 2 of its 2 size-parametrized cases, every time — no focus-related test failed in
any of the 3 runs); `Tests/UI/ -k watchlist` (full file, unfiltered by name): 261 passed, 1 failed
(the known issue), 7076 deselected, in ~588s — no tree-chevron failures observed this run. Mutation
test on the AC#2/#3 discriminator: reverting `recompose()` to eager-clear reds it with
`screen.focused is None`, confirming it actually discriminates the fixed mechanism.

**Modified files:**
- `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py` — sticky-until-confirmed `recompose()` /
  new `_confirm_create_focus()`.
- `Tests/UI/test_watchlists_source_create_form.py` — two new tests (interleave discriminator,
  case-2 regression guard).

**2026-08-02 addendum — bounded follow-up attempt on the `Select` race, reverted.** Asked to take
one bounded attempt at the `SelectCurrent` symptom too (it's named in this task's own description).
Root-caused precisely with `TEXTUAL=debug`: `Select._on_mount` -> `_init_selected_option` ->
`self.value = hint` -> `_watch_value` -> `select_current.update(prompt)` ->
`SelectCurrent.query_one("#label", Static)` raises `NoMatches` — `SelectCurrent` itself is mounted
(so `Select._watch_value`'s existing `except NoMatches: pass` guard doesn't catch it), but its own
child `#label` hasn't finished mounting yet. Two things tried:
1. `_finish_create_submit`'s scheduling swapped `call_later` -> `call_after_refresh` (already
   documented above) — insufficient.
2. Running `_finish_create_submit` via `self.run_worker(...)` instead of `call_later` (a genuinely
   different asyncio task, not just a later point on the same queue) — this **measurably helped**:
   15/15 clean in plain isolation (up from 2/2 failing), but the content-pane -> create-form ordered
   scenario AC#2 actually cares about still showed 2/8 failures (~25%, down from ~100% before). Per
   this task's own established rule ("a shrunk race is a hidden race" — three earlier focus-race
   mitigations were rejected on exactly this basis), reverted rather than shipped. This variant also
   has a real regression risk if implemented differently: an earlier version that moved the close
   into `WatchlistsCollectionsScreen._create_source` instead of the pane's own worker broke
   `Tests/Watchlists/test_watchlists_sources_pane.py::test_sources_pane_new_source_form_posts_request`,
   because that bare-`SourcesPane` harness (no real screen) depends on the pane closing its own form
   independent of any listener — caught and reverted before committing.

Both files (`sources_pane.py`, `watchlists_collections_screen.py`) were confirmed back to exactly
their committed state (`git diff` clean) after reverting. Full diagnosis, both rejected mitigations,
and next-step candidates carried forward to **task-1960** rather than re-derived from scratch.
<!-- SECTION:NOTES:END -->

## Qodo fix wave (2026-08-02)

Qodo flagged two real defects in the sticky mechanism: (1) `_confirm_create_focus` rescheduled via
`call_after_refresh` forever when focus never reached the exact `target`, with the intent stuck
armed; (2) `recompose` computed `restore = pending or focused_field`, so the stale intent WON over
the user's current field — a rebuild during the confirm window could yank them off a field they had
Tabbed to. Fixed by: FIX A — `recompose` now prefers the user's current in-form field
(`_focused_create_field_id() or _pending_create_focus`), so the intent only wins during the genuine
mid-burst drop (focus None); FIX B — `_confirm_create_focus` clears the intent once focus lands on
ANY real widget (in-form sibling or outside the form), and its remaining reschedule (focus still
None) is bounded by `_CREATE_FOCUS_CONFIRM_MAX_ATTEMPTS = 20`. During self-review the two clear
branches (in-form / outside-form) were found behaviorally identical (both clear) and collapsed into
one load-bearing branch, so the discriminating test is not backstopped by a redundant branch. Three
new tests (yank-to-stale-target discriminator for FIX A; give-up-after-max and clear-on-non-target
for FIX B), all mutation-verified; ordered-pair focus run 3×3 clean.
