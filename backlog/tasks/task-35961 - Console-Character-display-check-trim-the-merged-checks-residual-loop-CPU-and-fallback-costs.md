---
id: TASK-35961
title: >-
  Console Character display check: trim the merged check's residual loop-CPU and
  fallback costs
status: Done
assignee:
  - '@claude'
created_date: '2026-10-21 01:32'
updated_date: '2026-10-27 20:39'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.6 merged the Console Character display check's scope probe and reload into one finite callback, one handle and one ADR-125 helper. A few costs went up and were documented there. An unchanged check, the common one (about once a second during a reply), spends about 0.5-0.9 ms more UI-loop CPU than the original probe, because it qualifies the stock readers (_stock_character_refresh_readers) on the loop for every check. A first scope read that does not settle costs one merged helper plus the whole original probe-then-refresh route. A refresh dispatch customized during the read takes 5 finite callbacks against 4, and a final pair that keeps failing loads the groups 4 times against 3. Only the loop CPU is on the steady-state cadence; the rest are race, error or customization paths. A cheaper early exit for an unchanged check, and a fallback that reuses what the merged read already learned, would remove them without weakening reader provenance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unchanged stock display check costs no more UI-loop CPU than the original probe route did, shown by a paired in-process measurement
- [x] #2 A first scope read that does not settle starts no more helper processes than the original probe-then-refresh route
- [x] #3 A customized, replaced or borrowed reader is still refused (the existing Character source- and entry-control tests stay green)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure (paired, in-process, same host): loop-thread CPU of an unchanged stock display check on the merged route vs the original pre-15.6 probe route (pre-15.6 module loaded beside the current one), and attribute it (qualification passes per check on the loop).
2. Tests first (red on dev): an unchanged settled check does no full reader qualification on the loop (count _character_reader_current calls on the loop thread <= the original route's); a first scope read that does not settle starts no more helpers than the original route needs after it (merged read reused as the probe).
3. Settled checks: build the readers on the loop with only the checks the loop needs (structure, the readers it calls, captured callables), keep full reader provenance on the worker before every read; a refusal at the worker's entry opens no handle and takes the original route. The first pair's on-loop midpoint verifies the readers it runs; after it, the changed path keeps every original loop check.
4. Reuse an unsettled/failed merged first read as the probe's outcome in both callers instead of probing again.
5. Re-run the Character source/entry-control and one-helper suites (AC#3), paired loop-CPU measurement, live spot-check with pf2probe.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** A Console Character display check whose display shows a settled fingerprint (the common one, about once a second during a reply) no longer qualifies the whole stock reader set on the UI loop. It qualifies on the loop what runs there, and the worker qualifies every reader before and after each read, as before. A first scope read that does not settle, or fails, is now the probe's outcome: the check reloads without probing the scope again on a handle of its own.

**Measured first (dev 3be493413a).** In process, paired: the pre-15.6 module (`715a0d7b49^1`) loaded beside the current one, both driving the same kind of settled display over their own real `CharactersRAGDB`, interleaved, 150 unchanged checks per arm, loop-thread CPU by `time.thread_time()`, GC paused while sampling.
- Paired median, merged minus original probe route: run-time check +0.30 and +0.33 ms; idle check +0.34 and +0.45 ms (two runs).
- Attribution, counted with `sys.setprofile` on the loop thread (Python 3.12's cProfile records every thread, see the lesson): 252 single-reader qualifications (`_character_reader_current`) per unchanged check against the probe route's 94. They came from one more presentation check (`refresh_if_scope_changed`'s gate), the selector's two passes (`_stock_character_refresh_readers` plus its own `require_current`), and two whole `require_current` calls in the pair's on-loop midpoint. Timed alone: presentation check 43 us, selector 109 us, `require_current` 54 us.

**Change** (`UI/Console_Modules/character_context.py`).
- A settled check selects its readers with `_settled_character_refresh_readers`: the owner's structure (as `_stock_character_presentation` checks it), the two identity readers the selection calls, `_await_database_work` (added after review: the check hands it the finite callback next, on the loop), and the four captured bound callbacks' identities. No whole-set pass on the loop. The construction is shared with the full selector (`_character_reader_sources`, `_build_character_refresh_readers`); the full selector's checks and order are unchanged and it still serves unsettled checks, `refresh()` and the known-changed test.
- The finite callback qualifies every reader in the worker before its first read, as it did. For a settled check, a reader changed after the selection is refused there before anything is read (no handle opens) and the check takes the original route.
- The on-loop midpoint of each pair in the first scope read (every attempt, up to `_SCOPE_CAPTURE_ATTEMPTS` = 3) runs `require_loop_readers` (the two identity readers and the captured accessors) through a new `_loop_check` argument of the pair; the worker's checks around each read are unchanged, and the pairs after that read (the changed path's fence read) and the changed path's post-pair owner check keep the full checks.
- The run-time gate before `_check_recent_batch` qualifies that method alone for a settled check (`_stock_character_check`); an unsettled check keeps the whole-set gate.
- `_character_reader_current` runs the same checks in the same order with plain loops instead of a generator per record.
- AC#2: `_check_recent_batch` returns `_CharacterScopeProbed` for a first read that did not settle, or a callback that failed (in that read or after it), for a current owner, and both callers take the original route's branch for that probe outcome (`invalidate_scope` + reload, or the read-error reload) instead of calling `_capture_scope` again.

**After.** Loop qualifications per unchanged check: 96 (probe route 94). Paired median, branch minus original probe route, three runs of 150 checks per arm at host load 28-32: run-time -0.058, -0.074, -0.052 ms; idle -0.009, -0.017, -0.046 ms.

**Live spot-check** (2026-10-27; real app, Anthropic claude-haiku-4-5 through the review proxy, isolated profile, null keyring, 160x45, caffeinate held; dev `bd21-base` a2b7104d1e vs this branch, one launch each, pf2 kit plus a `sys.monitoring` hook on the exact code objects, nothing wrapped). Unchanged checks that read the scope: dev median 1.12 ms loop CPU (p90 1.38, n=21), branch 0.79 ms (p90 1.07, n=25). Evidence: `tools/evidence/bd5/P35/`.

**Tests.** New `Tests/Backup_Recovery/test_character_settled_check_loop_cost.py` (8 cases), all red on dev for the intended reason:
- unchanged settled check, active/idle: dev runs the selector once and `require_current` 3 times on the loop, 136 single-reader checks outside the entry's presentation checks;
- unsettled first read, active/idle: dev 3 helpers / 6 pairs / 3 callbacks (expected 2 / 5 / 2);
- reader changed after selection, 2 methods x active/idle: dev ends the check after the refused callback (1 callback, no read); now it takes the original probe (2 callbacks, 1 pair, 1 helper) and runs neither body.
Mutation checks: (a) skipping the worker's entry check for a settled check makes the changed body run (refusal test red); (b) passing the full `require_current` to the first midpoint again turns the loop-cost test red.

**Not addressed (outside the ACs).** The two remaining fallback costs from the description: a refresh dispatch customized during the read (5 finite callbacks against 4) and a final pair that keeps failing (groups loaded 4 times against 3). Filed as TASK-35961.1. The new Backup_Recovery file, like 15.6's, runs only in the core suite, not a PR lane.

**Review follow-ups (2026-10-27, pre-merge review of the first cut; no blocking findings).**
- Fixed: a settled check's selection now also qualifies `_await_database_work` on the loop (`_CHARACTER_SETTLED_SELECTION_READERS`). The display entry's whole-set checks run before ambient accessors, and the first cut handed `read_current` to a dispatch changed in between. New test: the dispatch changed as `_check_recent_batch` starts never receives the merged callback (red on the first cut, green on dev and now).
- Pinned: the settled midpoint's `require_loop_readers` had no failing test when made a no-op. New test: an identity reader changed as a settled pair's midpoint starts never runs its changed body on the loop (4 cases; red only under that mutation, by design green on dev, which checks the whole set there).
- Pinned: AC#2's failed-read branch. New test: a first scope read that fails reloads through 2 finite callbacks and 2 helpers (red on dev: 3 and 3; red again when the read-error probe is returned as None).
- Wording: the midpoint check covers every attempt of the first scope read (not only the first pair), and the catch-all hands any callback failure, also one after that read, over as a read error (both callers reload once, as the original route does). Docs/Architecture/console.md, ADR-126 and the docstrings say so.
- Filed: TASK-35961.2 (these regression files, and the drill test's resumed-run case, run in no PR lane).

**Merge evidence (final head, rebased on dev 416feb50ee).**
- New tests, plain command: test_character_settled_check_loop_cost.py 16 passed; on origin/dev 10 failed (the 8 original cases plus the 2 failed-read cases), 6 pass by design (4 midpoint pins, 2 dispatch cases).
- Neighbour files, one per command, branch vs a detached origin/dev worktree, compared by test id: no test fails on the branch that passes on dev (Character Backup_Recovery suites all green on both; the environmental RecoveryRequired/inventory/ratchet reds are identical ids on both arms).
- Paired in-process loop CPU, final head minus the original probe route, 2 runs x 150 checks per arm: run-time -0.012 and -0.026 ms, idle -0.044 and -0.045 ms (AC#1 still holds with the added qualification).
- Live round 2 (real app, Anthropic claude-haiku-4-5 via the review proxy, isolated profile, null keyring, 160x45, caffeinate held; dev 416feb50ee vs this head, same scenario): unchanged checks that read the scope, median loop CPU 1.31 ms on dev (n=29) vs 0.64 ms here (n=24). Changed checks: max 12.7 ms on dev, 13.0 ms here, medians 6.7 vs 7.3 ms (n=10/12). Round 1's two branch outliers (57.9 and 35.5 ms at host load about 30) did not recur, which fits host noise; this diff does not change the changed path's loop work beyond the selection.
- ./scripts/preflight.sh passes on the final head.

**Re-rebased 2026-10-27 onto dev 0923ec0ca2** (dev's two new commits touch only tests and CI; the app code is byte-identical to the head the live rounds ran). Re-run on the rebased head, one file per plain command, against a detached origin/dev worktree, compared by test id: test_character_settled_check_loop_cost.py 16 passed (dev: 10 failed, 6 pass by design); the Character Backup_Recovery and UI suites, inventories and module-size ratchet show no test failing on the branch that passes on dev (identical red ids on both arms: RecoveryRequired setup errors, inventory and ratchet rows naming other files). ./scripts/preflight.sh passes on the final head.
<!-- SECTION:NOTES:END -->
