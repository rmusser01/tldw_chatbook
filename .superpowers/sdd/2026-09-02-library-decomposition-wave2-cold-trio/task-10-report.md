# Task 10 (wave close) — as landed

Wave-2 close: recipe bookkeeping, stale-doc alignment, fresh measurement,
and full verification for `refactor/library-decomp-wave2-cold-trio` before
PR. No production code changed — this task is docs-only (recipe.md tables/
new section, two module docstrings) plus filing the wave-3 backlog task
that was already sitting untracked in the tree.

## 1. Recipe diff summary

`backlog/docs/library-decomposition-recipe.md` (+243/−14 lines):

- **§8 table** (Subsystem order): export and collections rows updated from
  "recipe rehearsal" to **complete**, each with its actual moved/excluded
  method counts (export 22/29, collections 64/3) and a pointer to its
  detail section (§12, §13–§15). The search row updated to record the
  **BLOCKED** gate outcome (57.1% direct / 45.5% conservative cross-call
  census, both past the 1/3 gate) and its deferral to wave-3's combined
  search+RAG series (task-31203). The RAG/onboarding-plumbing row got a
  one-line forward note that search merges into that pool for wave 3.
- **New §16 "Wave-2 close — summary"**, appended after §15: the
  re-derived full wave-2 pin trajectory (task-level, git-log-verified, not
  carried over from any report), a subsystem-outcomes recap, the full
  verification battery's counts, the sequential xdist paired-baseline
  sweep comparison, the probe table, and four wave-2 lessons stated with
  their incidents (verbatim-comment discipline, the `Tests/Live` third
  test root recurring a second time, sequential-vs-concurrent sweeps, the
  RED-commit criterion's actual wording).

## 2. Stale-docs fixes

Both `library_export_state.py` and `library_collections_state.py` module
docstrings carried the same stale paragraph: "`library_screen.py` keeps
every original `_library_<subsystem>_<field>` attribute name alive as a
generated getter/setter `@property` shim ... a later controller PR in
this series moves the subsystem's methods here too." That sentence
described the state PR's OWN moment in time (before either subsystem's
controller or cleanup PR had landed) and was never updated after both
landed — the screen-side shim was deleted wholesale by each subsystem's
cleanup PR (export: task 4, `cdb43ebcc`; collections: task 7,
`39a976321`), and the controller that took over each subsystem's methods
now carries the permanent generated shim in its place. Rewrote both
docstrings to the conversations-exemplar's own post-cleanup tense (that
module's docstring, `library_conversations_state.py`, is the template
this recipe already establishes for the correct wording): past tense for
what the state PR originally installed, then a sentence naming the
specific cleanup-PR commit that deleted it and the specific controller
that now owns the permanent replacement shim, plus a pointer to that
controller's own shim-block comment for the "why permanent" reasoning
(the byte-for-byte canon, recipe §1).

Only `library_collections_state.py` was named in the ledger as a deferred
minor (task 7's note), but the sweep this task's brief asked for
(`library_export_state.py`, `library_export_controller.py`,
`library_collections_state.py`, `library_collections_controller.py`, for
"delete wholesale at cleanup"/pre-cleanup phrasing) found the identical
stale sentence in `library_export_state.py` too — same defect, same fix,
one export series ahead of the collections one that was actually flagged.
Both controller modules
(`library_export_controller.py`, `library_collections_controller.py`)
were checked and already carry the correct "Permanent, not a cleanup-PR
deletion target" wording from when they were written (verified by
`grep` for `a later\|will move\|not yet\|series moves\|next task\|future
cleanup\|cleanup PR will\|to be deleted\|to be removed\|eventually\|TBD\|
TODO` across all four files — zero hits after the fix). Both edited files
re-parse cleanly (`ast.parse`) and no test asserts on either docstring's
literal content (grepped for `__doc__` references in the three most
plausible test files; none).

## 3. Fresh measurements

`_measure()` semantics (ast-walked line count + `LibraryScreen` method
count — matches `Tests/Architecture/test_screen_size_ratchet.py`'s own
helper exactly, not `wc -l`), run against the HEAD tree
(`1e466ffac` + this task's own doc-only edits, which touch no `.py` file
under `tldw_chatbook/` or `Tests/` besides the two docstrings, verified
`ast.parse`-clean and not test-observed):

```
lines: 42411
methods: 1267
```

Exact match to the recorded `_BUDGETS` row (`42411, 1267`) — **zero
drift, nothing to lower.** No fix needed.

## 4. Verification battery — every suite's counts

All commands: `.venv/bin/python -m pytest ... -p no:randomly -q`, run
from `.worktrees/library-decomp-foundation`.

| Suite | Result |
|---|---|
| Wiring suites (`test_library_collections_wiring.py` 4 + `test_library_export_wiring.py` 5 + `test_library_conversations_wiring.py` 7) | **16 passed** (only 3 such files exist for this recipe today — no search wiring suite, since Task 8 never moved anything; the conversations file gained a 7th test since §15's recorded count of 6, ordinary growth) |
| Characterization files (collections 5 + export 5 + conversations 4) | **14 passed**, matches §15's recorded count exactly |
| Both size-ratchet guards, full suite (`test_screen_size_ratchet.py`) | **3 passed, 2 failed** — exactly the two documented pre-existing `chat_screen.py` rows (`test_screen_does_not_grow_past_its_budget[chat_screen.py]`, `test_task_22507_4_does_not_worsen_chat_screen_base`), no others |
| Recompose census suite (`test_library_recompose_ratchet.py`, Task 1's guard lives here) | **6 passed** |
| Support-layer surface (`test_library_support_layer_surface.py`) | **8 passed** |
| Preflight (`./scripts/preflight.sh`) | **all six checks green** (CSS bundle, profile-owned-path census, diagnostic inventory, backlog task ids, chachanotes table allowlist, index plan pins) — run twice (before and after the doc edits), identical both times |

A note on "all four wiring suites" from this task's own brief: only three
`Tests/Architecture/test_library_*_wiring.py` files exist
(`find . -iname "*librar*wiring*"` confirms it, repo-wide) — collections,
export, conversations. No fourth was found under any name; search never
shipped a wiring suite since Task 8 was BLOCKED before any move. Ran all
three that exist rather than inventing a fourth.

## 5. Full library xdist sweep + paired pristine baseline (recipe §7, sequential)

**Branch** = this task's own tree at the time of the run (HEAD `1e466ffac`
+ the doc-only edits above, which touch no test or production-logic
file). **Baseline** = a path-scoped `git checkout 2b20ebbb9 -- tldw_chatbook
Tests` overlay of the foundation tip (`2b20ebbb9`, the wave-2 branch's own
start commit — `_measure()` on that overlay confirmed 43965/1282, an exact
match to the wave-2 plan's own recorded starting baseline), the
multi-commit-back equivalent of the per-task `git stash -u` technique used
elsewhere in this ledger, since this comparison spans the WHOLE wave
rather than one task's uncommitted diff. My own uncommitted doc edits +
the untracked task-31203 file were `git stash push -u`-ed first so the
overlay checkout wouldn't touch them; restored via `git checkout HEAD --
tldw_chatbook Tests` (re-verified `_measure()` back to 42411/1267 and
`git status` clean) then `git stash pop` afterward.

Run **SEQUENTIALLY**, not concurrently, per Task 6/7's own forward note
(recipe §7/§15) — Task 6's concurrent run measurably amplified flakiness
and cost real triage time; Task 7's sequential re-run landed back inside
the historical range.

```
.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal
```

| | Failed | Passed | Wall time |
|---|---|---|---|
| Branch (`1e466ffac` + close) | 335 | 3904 | 1261.02s (21:01) |
| Baseline (`2b20ebbb9`) | 343 | 3895 | 1252.31s (20:52) |

Diffing the two sorted failure-name sets (`comm`): **5 branch-unique**,
13 baseline-unique, 330 shared. The 5 branch-unique names:

- `Tests/UI/test_library_media_reader_no_change_sync_t22208.py::test_no_change_traversal_builds_no_preview_and_copies_no_content`
- `Tests/UI/test_library_prompts_canvas.py::test_library_prompt_history_stale_conflict_reload_refreshes_and_can_retry`
- `Tests/UI/test_library_prompts_canvas.py::test_library_prompts_stale_search_cannot_restore_an_old_filter_caret`
- `Tests/UI/test_library_shell.py::test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`
- `Tests/UI/test_library_shell.py::test_library_shell_note_id_deeplink_opens_note_editor`

Re-run single-process, all 5 together, on the branch tree: **5 passed**
(13.39s) — confirmed pure xdist ordering/parallelization noise, not a
regression. None touches Export or Collections code, this wave's own
diff, or a fixture this wave's diff shares (Media reader traversal,
Prompts canvas, Notes shell) — the names alone already say "unrelated
feature," and the clean combined re-run closes the question directly.
**Zero real regressions across the whole wave.**

## 6. Probe run

```
perl -e 'alarm 150; exec @ARGV' .venv/bin/python Helper_Scripts/library_click_probe.py
```

| interaction | settle (ms) | max gap (ms) | recompose | full-update | mounts | nodes |
|---|---|---|---|---|---|---|
| media (switch-in) | 485 | 155 | 0 | 2 | 163 | 113 |
| media (re-click same) | 328 | 54 | 0 | 2 | 79 | 113 |
| media (re-click same, 2nd) | 329 | 56 | 0 | 2 | 79 | 113 |
| notes (switch) | 413 | 195 | 0 | 1 | 110 | 110 |
| notes (re-click same) | 264 | 56 | 0 | 1 | 38 | 110 |
| media (switch-back) | 467 | 156 | 0 | 1 | 165 | 113 |
| notes (switch, 2nd) | 356 | 131 | 0 | 1 | 110 | 110 |
| media (switch-back, 2nd) | 411 | 94 | 0 | 1 | 165 | 113 |

**Finding worth flagging**: no prior run of this probe is recorded
anywhere in the repo (`grep -rln "click_probe"` over `.` finds only the
recipe's own §9 description of the script) — §9 says to run it before AND
after each controller-move PR, but neither Task 3 (export controller) nor
Task 6 (collections controller) captured one. This is genuinely the FIRST
recorded output. I did not fabricate a before/after diff against a
nonexistent number; I recorded this run as the wave-2 close baseline
instead, with the honest caveat that "within noise of the recipe's
recorded band" (this task's own brief phrasing) presumes a prior recorded
band that does not exist. What I can say with confidence: the probe
exercises ONLY the Media/Notes rail-switch path (the foundation-era
139–380 ms freeze §8's Phase C note references) — a path neither the
export nor the collections series touches — so these numbers reflect the
pre-Phase-C cost the design doc already documents, not something this
wave's moves could plausibly have changed. Every row is consistent with
that: no recompose count above 0 (this wave never adds a whole-screen
recompose site), full-update counts of 1–2 (unchanged shape), and gaps in
the same 54–195 ms range the docstring's own motivating numbers describe.

## 7. Files changed

- `backlog/docs/library-decomposition-recipe.md` (+243/−14): §8 table
  completed for export/collections/search; new §16 "Wave-2 close —
  summary".
- `tldw_chatbook/UI/Library_Modules/library_export_state.py` (docstring
  only): stale pre-cleanup shim paragraph rewritten to the post-cleanup
  mechanism.
- `tldw_chatbook/UI/Library_Modules/library_collections_state.py`
  (docstring only): same fix.
- `backlog/tasks/task-31203 - Library-decomposition-wave-3-combined-searchRAG-series.md`
  (new, was untracked from Task 8): committed as part of this close.

No test file, no production logic, no `_BUDGETS` row, no
`.git-blame-ignore-revs` entry (nothing here moves a method body — no
blame-ignore entries needed).

## 8. Self-review

- Every number in §16 and this report is either a fresh measurement taken
  during this task (the `_measure()` run, the two sweep counts, the probe
  table) or a `git log`-verified fact (the pin-trajectory chain, cross-
  checked against `Tests/Architecture/test_screen_size_ratchet.py`'s own
  inline comment history at each of the 9 relevant commits) — nothing
  copied from a prior report without independent verification.
- The "four wiring suites" / "recipe's recorded band" phrasings in this
  task's own brief did not survive contact with the actual repo state
  (only 3 wiring suites exist; no probe band was ever recorded). Both
  discrepancies are called out explicitly above rather than silently
  reconciled or silently ignored.
- The xdist sweep's baseline used a path-scoped checkout-and-restore
  (`git checkout <sha> -- tldw_chatbook Tests` / `git checkout HEAD --
  tldw_chatbook Tests`) rather than `git stash -u`, because the wave-level
  baseline is 9 commits back, not this task's own uncommitted diff —
  `git stash -u` alone cannot reach it. Verified the restore was complete
  and clean both by `_measure()` (42411/1267, exact) and `git status`
  (empty) before popping the doc-edit stash back.
- The 5 branch-unique sweep failures were confirmed noise by a real
  combined single-process re-run (not assumed from the names alone,
  though the names alone already pointed away from this wave's own
  subsystems).
- Did not touch `Tests/`, `_BUDGETS`, or any controller/state module body
  — this close is docs-only by design, matching its own "no blame-ignore
  entries (no body moves)" instruction.
