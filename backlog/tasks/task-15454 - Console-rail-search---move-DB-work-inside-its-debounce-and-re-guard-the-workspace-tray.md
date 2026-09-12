---
id: TASK-15454
title: 'Console rail search: move DB work inside its debounce and re-guard the workspace tray'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified first-hand: the `#console-workspace-conversation-search` handler (`chat_screen.py:1843-1907`) debounces only the FTS worker; synchronously per keystroke, on the event loop, it invalidates the persisted-rows TTL cache, reads workspace labels twice (including a possible write transaction via `ensure_default_workspace`), runs one `list_workspace_conversations` SELECT per workspace, reads starred ids at least twice, and then calls `_sync_console_workspace_context()` — which recomposes the workspace context tray unconditionally across up to 3 tray instances (~180-450 widgets), because the tray's equality guard was deliberately reverted (`Widgets/Console/console_workspace_context.py:546-560`, pinned by a test) after a full-equality guard caused a click-targeting regression.

Fix direction: move everything between `:1858-1893` inside the debounced timer callback; then design a NARROWER structural-key guard for the tray that avoids the historical regression (the reverted guard failed on full equality — a structural key over row identity/order can skip no-op recomposes without recreating the click-targeting bug). Stability constraints: keep the existing pinning test, and add a regression test for the click-targeting case that forced the revert before re-introducing any guard. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Zero SQLite queries and zero tray recompose on the keystroke path before the debounce fires (evidence)
- [x] #2 Tray recomposes only on structural change, with the historical click-targeting regression covered by a test
- [x] #3 Search results and rail behavior unchanged (existing surface green)
- [x] #4 `_console_composer_or_none()` no longer walks the whole DOM twice per keystroke, and a cached reference can never be returned once it is detached (folded in from task-15452's review: it is 61% of the residual per-keystroke cost after 15452)
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Read the handler, the tray, the reverted-guard history and the two tests the
   revert names; reproduce the historical click-targeting regression first-hand by
   temporarily applying the naive full-equality guard and watching those two tests
   go red (evidence, not folklore).
2. Keystroke path: leave only pure state bookkeeping (query, tokens, timer swap)
   in `Input.Changed`; the TTL-cache invalidation, the row derivation and the
   tray sync all move behind the existing 0.2 s timer. The debounced callback
   re-checks the cancellation token/query before doing any work. Keep the
   already-cached rows visually consistent with the newest query via the pure
   in-memory filter (no DB) so a tick-driven sync inside the debounce window
   cannot paint rows that contradict the search box.
3. Tray guard: skip `refresh(recompose=True)` only when the incoming state is
   value-equal AND the mounted DOM is provably the DOM that state produced --
   a structural row signature (ordered row ids + row keys, the things that
   determine click targets) recorded by `compose()` itself and compared against
   the same signature read back out of the live DOM, plus "no recompose already
   pending" and "mounted". Any mismatch (including the fresh-tray / superseded-
   rows desync the revert was about) recomposes exactly as today.
4. Memoize `_console_composer_or_none()` on a class-level attribute, revalidating
   `is_attached`/`is_mounted`/id/parent-screen on every hit and falling back to a
   fresh query otherwise.
5. Tests born red where feasible: a keystroke-path no-DB/no-recompose test, tray
   guard tests including a regression test for the historical click-targeting
   case, and composer-memo tests including a detached-widget mutation control.
6. Measure the per-keystroke residual before/after with an isolated probe;
   run the rail/search/tray/workspace suites plus the task-15452 gate suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Three changes, all measured. Plan followed; the one deviation is recorded
under "The revert no longer reproduces" below.

**1. The keystroke path is now bookkeeping only.** `Input.Changed` keeps the
query/token mirrors and the timer swap, and defers everything else to a new
`ChatScreen._start_console_conversation_browser_search(query, token)` that the
0.2 s timer arms: the TASK-251 TTL invalidation, the native/membership row
derivation, `_sync_console_workspace_context()`, and the `run_worker` kick.
The debounced callback re-asserts the token/query contract before it does
anything, so a superseded timer can never search for replaced text. One
non-bookkeeping line stays in the handler: `_filter_console_browser_rows_for_
query` over the rows already in memory (a pure filter, no service, no DB), so
a poll tick landing inside the debounce window cannot paint rows that
contradict the search box. Backspacing to empty is now debounced like any
other keystroke — it used to clear synchronously, and that clear ran the same
full derivation chain. The "Clear" BUTTON path is untouched and still
immediate.

**2. The tray guard is evidence-based, not equality-based.**
`ConsoleWorkspaceContextTray.compose()` now records the ordered `(row id, row
key)` pairs it builds — published only if the generator completes — and
`_can_skip_recompose` skips only when: the rail has pushed at least one state
into this instance, that signature exists, no recompose is latched, the tray
is mounted with children, the state is value-equal, AND the rows read back
out of the LIVE DOM still match the recorded signature. Row id + row key is
the identity Console click routing dispatches on, so a match means every
click target is present, in place, and pointing where it did. This is the
direct answer to the revert: the reverted guard asked "does the tray remember
this state", which is not the same question as "is the tray showing it".

**The revert no longer reproduces.** Before designing anything, the naive
`if state == self.state: return` guard was applied to today's dev and the
suites the revert named were run. Both witness tests
(`..._selection_keeps_query_active`, `..._invalidates_pending_worker`) PASS,
and across `test_console_native_chat_flow.py` (309) +
`test_console_rail_sections.py` the only failures were the two tick-gating
pins and one pre-existing failure. The July regression has been dissolved by
later work (most likely TASK-1900's non-echoing search input and TASK-1191's
fit-pass rework). That is a reason to re-guard, not to guard loosely, so the
DOM check stands anyway — and it is mutation-tested both ways: replacing
`_can_skip_recompose` with `return state == self.state` reds the two safety
tests, and with `return False` reds the two skip tests.

**3. `_console_composer_or_none()` is memoized** on a class-level
`_console_composer_ref` (the `__new__()` fixture convention), revalidated on
every hit via `is_mounted` AND `self in cached.ancestors_with_self` rather
than invalidated from teardown hooks — so a detached node can never be
returned even if a future teardown path forgets the memo exists.

**Measured** (isolated probe, scratch HOME/XDG/TLDW_CONFIG_PATH, headless
Pilot, 12-keystroke burst; `scratchpad/probe_15454.py`, fast M-series Mac,
small seeded workspace set — a real workspace/conversation set scales the
"before" column, not the "after"):

| per 12-keystroke burst | before (dev 7cfe8df4e) | after |
|---|---|---|
| handler wall cost, per keystroke | 0.558 ms | **0.009 ms** |
| registry calls during the burst | 252 | **0** |
| tray recomposes during the burst | 36 | **0** |
| workspace-context syncs during the burst | 12 | **0** |
| registry calls, burst + settle | 293 | **62** |
| tray recomposes, burst + settle | 42 | **3** |
| `_console_composer_or_none()` | 1182.6 µs | **0.40 µs** |

The after-settle numbers are higher than before-settle precisely because the
work moved there; the totals are what improved.

**Tests.** New: `Tests/UI/test_console_rail_search_debounce.py` (8) and
`Tests/UI/test_console_workspace_tray_recompose_guard.py` (6). Six of the
eight debounce tests were confirmed born-red against HEAD content; the other
two are deliberate controls. Updated with in-test comments explaining why:
`test_console_tick_gating.py::..._tray_sync_state_always_recomposes` (the pin
this task was required to revisit — now asserts the evidence-based contract
plus a desync control), `..._fresh_tray_still_synced_mid_run` (clears the
marker on all three projections, as a real full-screen recompose does),
`test_console_native_chat_flow.py::..._search_worker_uses_dedicated_group`
(source inspection followed the `run_worker` into the debounced callback),
`test_console_workspace_controller.py::..._empty_query_clears_state_*`.

**Review round 1 — DOM evidence was vacuous for the row-less projections.**
The DOM-evidence clause only compared grouped-browser rows, and
`#console-session-context` / `#console-workspaces-context` build none — their
row signature was `()` on both sides and matched anything, so for those two
trays the guard degenerated toward the reverted full-equality shape. Closed
both ways the review offered: `compose` now also records the ordered ids of
the FIXED controls those projections build (`_record_composed_node`, a
one-expression wrapper at each site) and the guard requires that half to match
too; and a pin test asserts every id-carrying node mounted in either
projection appears in that signature, with `ConsoleWorkspaceStatusPair`'s own
subtree as the single deliberate exemption. The pin discriminates: a mutation
that adds a dynamic per-row Button to the Sessions projection without
recording it reds the pin **while `_can_skip_recompose` still returns True** —
both facts asserted in the same test, so the pin is demonstrably what catches
the hole. Two further tests cover the new half: a pruned Switch button now
forces a heal (it did not before), and an out-of-band mount (the screen's
`#console-new-workspace-conversation` alias, which lands directly in the
Conversations tray) is deliberately NOT treated as drift — requiring exact DOM
equality would make that tray permanently unskippable.

**Why the User Guide "Verified against" stamp was not bumped.** The page's
conversation-browser table row was updated to state the new debounce, but the
stamp (`dev @ ff435772c — 2026-07-31`) was left alone on purpose: bumping it
asserts the WHOLE page was re-verified against a running app, and only this one
behaviour was. Claiming otherwise would make the stamp worthless for the next
reader.

**Modified/added:** `tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/Widgets/Console/console_workspace_context.py`,
`Docs/User_Guide/console/sessions-tabs-workspaces.md`, the four test files
above, plus the two new ones.

**Known pre-existing failures, verified at HEAD content, not caused here:**
`test_console_rail_sections.py::test_popover_apply_returns_replaced_settings`
and the three in `test_console_rail_width_budget.py`.
<!-- SECTION:NOTES:END -->
