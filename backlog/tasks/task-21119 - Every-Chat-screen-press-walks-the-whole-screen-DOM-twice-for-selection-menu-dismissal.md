---
id: TASK-21119
title: >-
  Every Chat-screen press walks the whole screen DOM twice for selection-menu
  dismissal
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 18:06'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21119).

`chat_screen.py:18939-18990`: `_dismiss_console_selection_menus_outside_transcript` runs
`self.query(ConsoleTranscript)` and `self.query(ConsoleSelectionMenu)` - two full-screen DOM
traversals - and is invoked on BOTH on_mouse_down and on_click of the same physical press
(~4 traversals per click) on the largest-DOM screen in the app. A direct contributor to the
click-lag symptom on every click.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dismissal early-returns via a mounted-menu flag/registry (at most one menu is ever mounted) and a cached transcript reference - no full-screen queries when nothing is mounted
- [x] #2 Selection-menu dismissal behavior is unchanged (covered by existing selection tests)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the real per-press cost with a counter probe over the production Console pilot (instrument screen.query, count ConsoleTranscript/ConsoleSelectionMenu walks per physical press) -- red-first.
2. Add constructor-registered candidate registries (WeakSet) for ConsoleSelectionMenu and ConsoleTranscript; re-derive attachment from the live DOM at read time so the registry can over-report but never miss a mounted node.
3. Add SelectionManager.is_idle + ConsoleTranscript.has_pending_selection_ui so the screen handler can prove its per-transcript cleanup is a no-op (keyboard-selection state has no menu but must still clear).
4. Rewrite _dismiss_console_selection_menus_outside_transcript to gate on the registries and early-return before any DOM work; keep the ancestor guard and the removal semantics identical.
5. Route ConsoleTranscript._attached_selection_menus through the same registry (it made a third full-screen walk on every in-transcript press).
6. Control arms: menu mounted on the screen still dismissed; selection-without-menu still cleared; in-transcript press still left alone. A/B every red against the base.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`ChatScreen._dismiss_console_selection_menus_outside_transcript` now gets both
collections from constructor-registered `WeakSet` registries on the two widget
classes and returns before touching the DOM when there is nothing to dismiss.
`ConsoleTranscript._attached_selection_menus` was routed through the same
registry (it was making a third full-screen walk on every in-transcript press).

### Measured cost (counter probe over the production Console pilot)

`Tests/UI/test_console_selection_dismissal_perf.py` shadows `screen.query` and
counts `ConsoleTranscript`/`ConsoleSelectionMenu` walks per PHYSICAL press with
no menu mounted (the always-case). On base `ae817fefe`:

| press | handler invocations | full-screen walks | after |
|---|---|---|---|
| composer (`#console-native-composer`) | 1 (the composer stops the Click) | 3 — `{transcript: 1, menu: 2}` | **0** |
| rail (`#console-rail-system-line`) | 2 (MouseDown + Click) | 6 — `{transcript: 2, menu: 4}` | **0** |

The finding's "~4 per press" undercounted: each invocation costs THREE walks,
not two, because `transcript._remove_selection_menu()` ran its own
`screen.query(ConsoleSelectionMenu)`. The probe test was red-first on base
(`{'transcript': 1, 'menu': 2}` vs the asserted 0).

### Why the mechanism cannot go stale

Registration happens in `__init__` — synchronous, and strictly before the
widget can be attached. That matters: Textual delivers `Mount` through the
widget's own message pump (`App._register` → `post_message(events.Mount())`)
while `_register_child` attaches the node synchronously, so an `on_mount`-based
registry would be blind during that window and dismissal would silently no-op
for a menu that is already in the DOM.

The registry is only a CANDIDATE set; liveness is never read from it.
`selection_menus_on_screen` / `console_transcripts_on_screen` re-derive
attachment from the live DOM (`widget.parent is not None` then
`widget.screen is screen`, the same scope the old `screen.query(...)` had).
So the two failure directions are asymmetric by construction:

- over-report (built-but-never-mounted, or removed before `Unmount` ran) →
  filtered by the DOM check; worst case we pay the walk the fix removed.
- under-report → would need a mounted widget that never ran `__init__`.

`_on_unmount` discards only to keep the candidate set small, and the weak
references expire on their own if that hook is skipped during teardown.

### Invalidation census

Nothing in these paths needs to maintain state; each is listed with the reason
it is already covered.

Menu mount sites: `ConsoleTranscript._text_selected` → `self.screen.mount(...)`
(the one production site — mouse drag release and keyboard `Enter` both land
here); test harnesses mounting on a transcript, on the app, on the screen, or
yielding one from `compose()`. All construct the widget → all registered.

Menu removal paths: `_remove_selection_menu()` (Escape, click-outside, every
action handler, the row-body press guard, the two reconciliation guards);
`action_dismiss` / `_on_click` self-removal; the screen dismissal loop;
`_text_selected`'s awaited pre-remount removal; framework teardown (parent
recompose, prune, screen pop, app exit). Each ends with the node detached
(`_parent = None`) or `_pruning` — both already handled exactly as before
(`_pruning` menus are still skipped so removal stays single-shot).

Transcript mount sites: `ConsoleTranscriptSurface.__init__` builds the
production transcript (rebuilt whenever `ConsoleSessionSurface` recomposes);
test harnesses construct their own. Unmount: session-surface recompose, screen
recompose/pop, app exit. Covered by the same DOM re-derivation — pinned by
`test_a_recomposed_transcript_replaces_the_old_one_with_no_bookkeeping`, which
recomposes the session surface, asserts the transcript object really was
swapped, and asserts the helper returns the NEW one only.

### Behaviour preservation

The gate is deliberately wider than "a menu is mounted". The old pass also
cancelled the selection manager and cleared the highlighted row on every
transcript, and keyboard-selection mode (`s`) arms exactly that state with NO
menu — a menu-only early return would have left the reverse-video strip
painted after a click on the composer. `SelectionManager.is_idle` is exactly
the post-`cancel()` field state (including the one-shot `just_finished` /
`release_click_pending` click tokens), and `ConsoleTranscript.
has_pending_selection_ui` adds the origin-row handle, so the skipped
transcripts are provable no-ops rather than assumed ones.

Deviation from AC #1's wording: no single cached transcript reference and no
"at most one menu" assumption. Both helpers return LISTS with the same screen
scope as the queries they replace — a cached reference is the failure mode
TASK-21116 hit twice, and relying on the singleton invariant would have made a
harness that mounts two menus silently leak one.

### Tests

- New: `Tests/UI/test_console_selection_dismissal_perf.py` (5) — the counter
  probe plus four control arms: screen-mounted menu still dismissed, active
  selection with no menu still cleared, in-transcript press still left alone,
  recompose swap picked up with no bookkeeping.
- Selection suites (10 files): 209 passed, 4 failed — all 4 fail identically on
  the base commit (`test_menu_open_row_body_click_dismisses_menu_and_toggles`,
  `test_menu_anchor_derives_from_row_region_and_stays_in_transcript`, two in
  `test_console_transcript_selection_contract.py`).
- Every test file importing the three changed modules (46 files, `-n 6`):
  77 failed / 1078 passed with the fix and 77 failed / 1078 passed on base,
  with IDENTICAL failure lists. The only delta was the error count inside the
  flaky `test_console_realtime_wiring.py` (35 vs 34); run alone that file gives
  8 errors with the fix and 10 on base — nondeterministic, not ours.
- Console click/shell suites (6 files): 5 failed / 226 passed on base,
  5 failed / 231 passed with the fix (+5 = the new probe tests), same failures.
- `--collect-only` sweep: 56876 collected, 5 pre-existing collection errors
  (base: 56871 / 6, the extra being this task's new file, absent there).
- `./scripts/preflight.sh`: all derived-artifact checks pass; the diagnostic
  inventory reports no drift.

### Review round 1 (merge-ready, three Minors taken)

The reviewer reproduced the numbers in both directions and ran 22 differential
liveness checks (registry vs `screen.query`) with 0 mismatches, including
`_pruning` before/after the prune message, nested-in-container mid-prune, and a
`ModalScreen` pushed above in both directions. Mutation kills confirmed the
mechanism carries load: dropping menu registration → +14 failures, transcript
registration → +2, a menu-only gate → +2. Three fixes landed:

- **Screen scope was unpinned** (the one that mattered). Relaxing
  `menu.screen is screen` to `menu.screen is not None` — the line that makes
  the registry equivalent to `screen.query(...)` rather than "any attached menu
  in the app" — passed all 221 tests in the 11 selection suites, mine included.
  New arm `test_the_registries_are_scoped_to_one_screen_not_the_whole_app`
  pushes a `ModalScreen`, mounts a menu on it, and asserts the Console sees
  neither the modal's menu nor lends it its transcript, plus the behavioural
  form (the Console's dismissal pass leaves the foreign menu mounted).
  Verified to kill the mutant on BOTH helpers: menu side fails on
  `selection_menus_on_screen(console) == []`, transcript side on
  `console_transcripts_on_screen(modal) == []`.
- **Statement order regressed the drag hot path.** The gate ran ahead of the
  ancestor guard, so an in-transcript press scanned both registries before
  returning (base: zero work). The ancestor walk is first again, and
  `test_press_inside_the_transcript_still_leaves_the_menu_alone` now counts the
  helper calls and asserts `{menus: 0, transcripts: 0}` — verified live by
  re-inserting a pre-walk scan (fails with `{'menus': 1}`).
- **`_LIVE_TRANSCRIPTS` claimed a symmetry it did not have**: the menu discards
  on unmount, the transcript never did. Added `ConsoleTranscript._on_unmount`
  (no `super()` needed — Textual dispatches `_on_unmount` from every class in
  the MRO). The recompose test now asserts the corpse leaves the candidate set
  and the replacement is in it; stubbing the discard to `pass` fails that test,
  so the hook is not dead code.

Known constraint, by report (not fixed): the registries' "cannot under-report"
guarantee rests on every instance passing through `__init__`. The repo does use
`X.__new__(X)` test doubles (`test_ui_responsiveness.py:441`,
`test_console_transcript_window_reconcile.py:305`); one of those applied to
these two classes AND mounted would be invisible to the gate. No such double
exists today for either class. The evidence doc's "twice per press" count at
`Docs/Design/2026-08-22-holistic-perf-review.md:265` is left for close-out.

Post-fix counts: 11 selection suites 215 passed / 4 failed (the same four that
fail on base); the 46-file related sweep 77 failed / 1078 passed with a failure
list identical to base (errors 32, inside the flaky realtime-wiring band that
gave 34 on base); `preflight.sh` and the diagnostic inventory clean.

Modified: `tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/Widgets/Console/console_selection_menu.py`,
`tldw_chatbook/Widgets/Console/console_transcript.py`,
`tldw_chatbook/Widgets/Console/console_selection.py`.
Added: `Tests/UI/test_console_selection_dismissal_perf.py`.
<!-- SECTION:NOTES:END -->
