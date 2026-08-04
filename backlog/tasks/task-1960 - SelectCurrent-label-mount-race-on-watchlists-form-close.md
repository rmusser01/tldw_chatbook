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
- [x] #1 The mechanism behind the `SelectCurrent`/`#label` mount race on the form-close recompose is understood well enough to fix at the mechanism, not just measured to reduce its frequency
- [x] #2 `test_a_source_can_be_created_end_to_end_through_the_form` passes deterministically (10/10) both in isolation and immediately after `Tests/UI/test_watchlists_content_pane.py`, with no sleep or bounded-retry involved in the fix
- [x] #3 `Tests/Watchlists/test_watchlists_sources_pane.py::test_sources_pane_new_source_form_posts_request` (and the rest of that file) stays green — the fix must not depend on `WatchlistsCollectionsScreen` being present
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed at the mechanism. `PruneSafeSelect` (new,
`tldw_chatbook/Widgets/prune_safe_select.py`) replaces stock `Select` at the 12 call sites the
Watchlists screen's own recompose can mount.

### AC#1 — the confirmed mechanism

Confirmed empirically by instrumenting `App._prune`, `Widget.mount`, `SelectCurrent.update`,
`WatchlistsCollectionsScreen.recompose/refresh` and `SourcesPane.recompose` in a throwaway scaffold
and capturing a real failure. The crash is **not** an ordering bug in Textual's Compose-before-Mount
guarantee; it is an escape hatch out of it, formed by two individually reasonable behaviours:

1. `Widget.mount` (`textual/widget.py:1451`) opens with
   `if self._closing or self._pruning: return AwaitMount(self, [])`. A widget caught by a prune
   mounts **nothing**, silently, and returns an already-satisfied awaitable — no exception, no log.
2. `MessagePump._pre_process` (`textual/message_pump.py:598-613`) dispatches `Compose` then `Mount`, and
   its `finally` (lines 609-612) sets `_mounted_event` and `_is_mounted = True` **unconditionally** —
   including when the `Compose` dispatch mounted nothing because of (1).

`App._prune` (`textual/app.py:4381-4395`; the assignment itself is 4395) stamps `_pruning = True`
over a `walk_children` snapshot. So a `SelectCurrent` that has been *registered* into the DOM but
whose own `Compose` has not yet run gets stamped, its `mount([#label, ▼, ▲])` becomes a no-op, and
it still reports `is_mounted=True` with zero children. Its parent `Select`'s `AwaitMount` unblocks,
`Select._on_mount` runs, and
`_init_selected_option → self.value = hint → _watch_value → SelectCurrent.update → query_one("#label")`
raises. Upstream guards the outer `query_one(SelectCurrent)` but neither the `#label` lookup one
level below it (`_select.py:256`) nor the `query_one(SelectOverlay)` inside its own `else:` branch
(`_select.py:613`) — a third unguarded shape, found by the whole-branch review and covered by this
fix for free, since the whole watcher is no-op'd.

Captured state at the crash, all three toolbar filter Selects in one run:

```
FATAL SelectCurrent#label missing is_mounted=True pruning=True children=0
      parent=Select(id=sources-type-select) parent_pruning=True
      screen_recompose_depth=1 pane_recompose_depth=0
```

immediately preceded (0.3 ms) by three
`MOUNT-SUPPRESSED SelectCurrent pruning=True is_mounted=False lost=[NonSelectableStatic(id=label), ...]`.

**Two parts of this task's own prior hypothesis were REFUTED by the instrumentation:**

- *`overview_data` is the concurrent destroyer* — **false** on this path. Logging every
  `refresh(recompose=True)` with its caller showed the post-submit screen recompose is requested by
  `_apply_local_wc_snapshot:983` and `_load_tree_data:912`, never by `overview_data`: with the
  controller mocked, `_refresh_overview_data` assigns an **equal** dict, so the reactive never fires.
  Removing `recompose=True` from `overview_data` would have changed nothing here. (The three
  requests also coalesce into a single recompose, so removing any one of them cannot remove the
  recompose at all.)
- *The interruption happens on the pane's form-close recompose* — the crash is one step later. The
  pane's close-recompose is itself killed first (`MOUNT-SUPPRESSED SourcesPane
  lost=[Vertical#sources-toolbar, DataTable#sources-table]`, a separate silent defect: the pane's
  recompose mounts nothing at all). The Selects that actually crash are the ones the **screen's**
  recompose mounts into its freshly built `SourcesPane`, caught by a later prune — in the test, the
  `run_test` teardown; live, the equivalent is navigating away mid-recompose.

Because the destroying prune can be any of several actors, and because the screen recompose rebuilds
the whole `SourcesPane` (3 more Selects) regardless, neither candidate structural fix is sufficient:
de-recomposing `overview_data` (direction 1) does not touch the actual requesters, and hand-managing
the create-form subtree (direction 2) removes the pane's cascade but not the screen's. Both would
have been frequency reductions — exactly what this task refused. The guard is the only change that
holds regardless of which task wins the race.

### The fix

`PruneSafeSelect` overrides the two mount-time methods that reach into children and makes them
no-ops while `_pruning`/`_closing`:

- `_watch_value` — the confirmed crash site.
- `_setup_options_renderables` — the sibling shape (`query_one(SelectOverlay)`, wholly unguarded
  upstream), reachable when the prune catches the `Select` one level higher.

Both are plain-method/watcher overrides, so the subclass replaces the base cleanly: Textual resolves
watchers by `getattr(obj, "_watch_value")` (`reactive.py:390`), not by MRO fan-out. Overriding
`_on_mount` would **not** have worked — `_get_dispatch_methods` yields the handler from *every* class
in the MRO, so the base would still run.

This is deliberately not a catch-and-carry-on. `_pruning` means Textual has already committed to
removing the widget, so it is never painted again and there is no stale-placeholder failure mode
(the concern that made a naive `except NoMatches` unacceptable). Any other route to a half-composed
`Select` still raises loudly.

CSS is unaffected: Textual type selectors match every CSS-inheriting base class
(`DOMNode._css_bases`), so `Select { ... }` still applies — pinned by a test, and confirmed by the
20 geometry/tab-order tests in `test_watchlists_source_create_form.py` staying green.

### Verification

- AC#2: **10/10 in isolation** and **12/12 immediately after `Tests/UI/test_watchlists_content_pane.py`**
  (one invocation, that order). Baseline on the same machine before the fix: 3 failures in the first
  10 parametrized runs of the AC#2 order. No sleep and no bounded retry anywhere in the fix.
  What that test detects is the **crash**: a `NoMatches` out of `_on_mount` reaches
  `App._handle_exception` and fails the run, which is exactly what the pre-fix baseline measured.
  Its "the form closed" assertion used to prove nothing else — see the review wave below.
- The race itself still occurs after the fix and is simply absorbed — instrumented runs still show
  3 `MOUNT-SUPPRESSED SelectCurrent` per run with `fatal_count=0`. Both rejected experiments are
  subsumed: neither `_finish_create_submit`'s scheduling nor its worker/`call_later` shape is touched.
- AC#3: `Tests/Watchlists/test_watchlists_sources_pane.py` 27/27 (26 existing + 1 new). The pane
  still closes its own form with no screen present; the fix is entirely inside the widget.
- `Tests/Watchlists/` 384 passed; `Tests/UI/test_watchlists_*` 172 passed; `--collect-only Tests/UI
  Tests/Watchlists Tests/Widgets` 8574 collected, no errors.
- Mutation-verified (each reverted individually → RED → restored): the `_watch_value` guard, the
  `_setup_options_renderables` guard, the `or self._closing` clause of *each* guard, the `_value`
  sync, and one `PruneSafeSelect(` call site reverted to `Select(`.

### Files

- Added `tldw_chatbook/Widgets/prune_safe_select.py`, `Tests/Widgets/test_prune_safe_select.py`
  (9 tests, including two controls that fail if upstream Textual ever fixes the escape hatch — the
  signal that this class can be retired).
- `PruneSafeSelect` adopted at **all 15** `Select` construction sites in the Watchlists feature:
  `sources_pane.py` (5), `artifacts_pane.py` (3), `rules_pane.py` (2), `items_pane.py` (1),
  `watchlists_collections_screen.py` (1), `briefing_preset_modal.py` (2),
  `kept_briefings_modal.py` (1). The last three were added in the review wave (below): they sit in
  `ModalScreen`s rather than in the screen's own recompose tree, but both modals
  `refresh(recompose=True)` from background workers and can be dismissed mid-recompose — dismiss
  pops the screen, which prunes it — so they carried the identical crash chain. Zero stock `Select`
  now remains anywhere under `UI/Watchlists_Modules/` or in `watchlists_collections_screen.py`.
- `Tests/Watchlists/test_watchlists_sources_pane.py`: structural pin, since a reverted call site
  would otherwise only be caught by the intermittent failure this task was filed against.
- Deleted `Tests/UI/test_zz_scaffold_1960.py` — this task's own earlier instrumentation scaffold,
  accidentally committed to `dev` in `749fa2e88` (a task-2060 commit) and never used by any test run.

### Known-but-not-fixed, found on the way

`SourcesPane`'s form-close recompose can silently mount nothing when the screen's recompose prunes it
mid-flight (`MOUNT-SUPPRESSED SourcesPane lost=[sources-toolbar, sources-table]`). It is invisible
today only because the screen recompose immediately rebuilds the pane wholesale. The underlying
cause is the same Textual escape hatch, and the real remedy is the standing recommendation from
TASK-1541 — stop full-screen-recomposing this screen from background loaders
(`_apply_local_wc_snapshot:983`, `_load_tree_data:912`) and patch in place instead. That is a much
larger change than this task, and it is now the *only* thing left that this task's crash depended on.

### Review wave (whole-branch adversarial review: 0 Critical, 2 Important, 4 Minor)

The review confirmed the mechanism line-by-line against Textual 8.2.8 and found the guard sound
rather than a shrunk race — `App._prune` is synchronous and `invoke_watcher` calls a 1-parameter
sync watcher inline, so there is no `await` between `if self._pruning or self._closing` and
`query_one("#label")` for a stamp to land in. It also established that `SelectCurrent._pruning`
implies `Select._pruning` structurally (the child is reachable in the prune walk only *through* the
parent), which is what makes a guard on the parent correct. All six findings are fixed:

- **Important 1** — three stock `Select`s with the identical shape survived in
  `kept_briefings_modal.py` and `briefing_preset_modal.py`. Converted; the Files list above now
  states the class scope as "all 15 sites in the feature" rather than naming one pane.
- **Important 2** — AC#2's `assert not pane.query("#sources-create-form")` was **vacuous**: by the
  time it ran, the captured `pane` was `_pruning=True, children=0, is_running=False`, and for a
  measured 0.14-0.32s there was no `SourcesPane` on screen at all. Repaired to wait for the settled
  pane (`_settled_sources_pane`) and assert positively that it remounted its table and carries no
  form, plus that no form survives anywhere else on the screen. Proven non-vacuous by mutation: with
  the screen's `_source_create_form_open = False` neutered, the **old** assertion stayed GREEN and
  the new one goes RED.
- **Minor 1** — the guard returned before `self._value = value`, leaving `value`/`_value` divergent.
  `Select._watch_value`'s first statement is the only non-DOM work it does, so the guard now keeps
  the shadow in sync and drops only the child lookups. The residual revival-repaint case (a future
  Textual that un-prunes nodes would show a placeholder over a real value) is **not** closed by that
  sync — the reactive still sees no change on the way back — and is documented explicitly in
  `_watch_value`'s docstring as an invariant of Textual, not of this class.
- **Minor 2** — added two tests exercising `_closing` with `_pruning` false, one per guard; deleting
  `or self._closing` from either now goes RED. (Hand-setting `_closing` deadlocks `run_test`
  teardown, because `_close_messages` early-returns when the flag is already set and so never
  enqueues its exit sentinel — the tests restore the flag across the subtree in a `finally`.)
- **Minor 3** — dropped the harness docstring's "byte-for-byte" overclaim; it is a reconstruction of
  the captured DOM state, not a replay, and now says so.
- **Minor 4** — corrected the Textual line citations above (`app.py` 4394→4381-4395, assignment at
  4395; `message_pump.py` 588→598-613, `finally` at 609-612) and recorded the third unguarded
  lookup the review found, `query_one(SelectOverlay)` at `_select.py:613`.

### Qodo wave (PR #1315): private Textual API coupling

One finding: `PruneSafeSelect` reads `_pruning`/`_closing` and overrides two private `Select`
methods, while `pyproject.toml` allowed any `textual>=8.0.0,<9`. The insidious mode is a *rename*:
the overrides silently stop being overrides — Python defines two methods nothing ever calls, nothing
raises, and TASK-1960's race returns as unattributable flakiness.

- **Accepted, fail-fast check.** `_require_internals()` raises
  `PruneSafeSelectCompatibilityError` (a `RuntimeError`) naming the missing internal and telling the
  reader to re-verify the prune-time mount no-op mechanism against their Textual version. Called at
  **module import** for `_setup_options_renderables`/`_watch_value` on stock `Select`, and in
  **`__init__`** (after `super().__init__`) for the `_pruning`/`_closing` instance flags. Cost: one
  import-time check plus two `hasattr` calls per widget.
- **Accepted, floor tightened.** `textual>=8.0.0,<9` → `>=8.2.8,<9`: the declared range now starts
  at the version the mechanism was actually verified against.
- **Declined: `getattr(self, "_pruning", False)`.** Defaulting a missing flag to `False` makes the
  guard silently inert — precisely the hidden-race class this task exists to refuse. A loud stop is
  strictly better than a guard that quietly does nothing, and the fail-fast check covers the same
  risk with a signal.

Four tests pin it, including two wiring tests (the checks are *called*, not merely defined): the
`__init__` half via a spy over a real construction, and the import half by re-executing the module's
own source in a throwaway namespace against a stand-in `Select` with one method renamed away — so
neither call can be deleted without a red test.
<!-- SECTION:NOTES:END -->
