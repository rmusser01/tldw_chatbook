# Task 5 report — Wave close (recipe, task-31203, durable evidence, verification)

Wave-3 Task 5 (recipe: `backlog/docs/library-decomposition-recipe.md`). Base
commit `a150fc766` (Task 4 fix round 1, final search+RAG cleanup pins).
Scope: final bookkeeping and full verification, not new extraction work —
the series itself (Tasks 2-4) already landed and its own recipe/task-31203
updates were substantially pre-landed by Task 4's own commits. This task's
job was to VERIFY that pre-landed state against reality (git log, ratchet
comments, independent `git show | wc -l` re-derivation) and fix what had
drifted, then add durable evidence and run the wave-close verification
battery.

## 1. What was already done at BASE (found, not redone)

Task 4's own commits (`5bea63cdc`, `a150fc766`) already:
- Updated `backlog/docs/library-decomposition-recipe.md` §8's subsystem
  table (search+RAG row: "complete", actual numbers) and added the full
  §18 section (cluster derivation, single-vs-split confirmation, per-task
  fields/methods table, pin trajectory, delegator census, two genuine
  findings, the ruled docstring correction, sweep evidence, 3 lessons).
- Ticked task-31203 AC#1-3 (AC#4 was Task 1's), added consolidated
  Implementation Notes covering all 4 tasks, and set status to `Done` via
  the backlog CLI (frontmatter `updated_date: '2026-09-04 02:19'`
  confirms a real CLI write, not a hand-edit).

This is earlier than the wave's own plan intended (Task 5 was meant to do
this), but it is not wrong — the numbers were mostly accurate. Verifying
it rather than blindly redoing it avoided duplicate, possibly-conflicting
edits.

## 2. Verification of the pre-landed recipe/task-31203 state

Re-derived the pin trajectory independently from raw `git show <rev>:<path>
| wc -l`, not from ratchet comments or the recipe's own prose (the
wave's own "verified-before-written" discipline, §18 lesson 4 below):

| Commit | `library_screen.py` lines |
|---|---|
| `fa07400a1` (pre-task-2) | 43977 |
| `77750c85d` (task 2) | 43923 |
| `877eeaf9a` (task 3) | 43009 |
| `5bea63cdc` (task 4) | 42949 |
| `a150fc766` (task 4 fix round 1, final) | 42940 |

| Commit | `library_rag_search_controller.py` lines |
|---|---|
| `877eeaf9a` (born) | 1857 |
| `544664510` (task 3 fix round 1) | 1890 |
| `5bea63cdc` (task 4) | 1895 |
| `a150fc766` (final, unchanged) | 1895 |
| this task, after its own docstring fix (§3 item 2, §7a) | 1897 |

Both chains match the ratchet files' dated comments and the recipe §18
pin-trajectory line exactly at BASE. §8's subsystem table and §18's
fields/methods table were spot-checked against Task 2/3/4's own reports
and found accurate (42 moved / 8 excluded by class / single-combined-
controller decision, matching AC#1's census and AC#2/#3's landed
reality). The controller's own final line count (1897) is THIS task's own
addition (§7a) — the pin was re-verified fresh, not assumed from this
table's own BASE snapshot.

## 3. Inaccuracies found and fixed

**task-31203's own Implementation Notes were stale relative to Task 4's
own fix round.** The consolidated notes (written at commit `5bea63cdc`,
before the fix round) still said "42949/1304 (task 4 final)", "35 literal
field references across 9 ... methods", and "removed 5 dead imports" —
all three were the PRE-fix-round numbers; the fix round (`a150fc766`)
corrected them to 42940/1304, 66/11, and 14 total, respectively, in the
ratchet comments and recipe §18, but the task file's own prose was never
updated to match. Fixed via a direct file edit (not `backlog task edit
--notes`, per `lessons-backlog-hygiene.md`'s documented notes-replacement
trap) and diffed against the pre-edit version to confirm only the prose
numbers changed — frontmatter, AC block, and the Renumbering provenance
section are untouched (`git diff` shown below).

**Two module docstrings still described the (already-completed) cleanup
in future tense, naming the wrong PR type in one case:**

1. `tldw_chatbook/UI/Library_Modules/library_rag_search_state.py`'s module
   docstring said "A future controller PR (wave-3 task 3) will delete
   that screen-side shim block" — wrong on two counts: the shim was
   deleted by the CLEANUP PR (task 4), not the controller PR (task 3, per
   the recipe's own §1 series definition), and it already happened.
   Rewritten to the past-tense template `library_export_state.py`/
   `library_collections_state.py` already use ("The <subsystem> cleanup
   PR (task N) deleted that screen-side shim block entirely once the
   subsystem's methods had all moved to <Controller> (task N-1)...").
2. `library_rag_search_controller.py`'s own module docstring said "the
   same two-prefix generator shape `LibraryScreen` carries (task 2)" —
   present tense, false since Task 4 deleted that shim. Fixed to past
   tense ("task 2 installed on `LibraryScreen` (deleted at cleanup, task
   4...)"), matching `library_export_controller.py`'s own correct
   precedent ("the same generator shape task 2 installed on
   `LibraryScreen`" — past tense, survives cleanup unmodified because it
   describes what a task DID, not what currently exists).

   **Found the identical present-tense staleness pre-existing, already
   unfixed, in `library_conversations_controller.py:1712`** ("the shim
   block `LibraryScreen` carries (task 6)") — an earlier-wave file with
   the same bug, never caught at its own cleanup. Left untouched:
   out of scope for this wave's close (not a search+RAG file), noted here
   rather than silently fixed or silently ignored.

No other pre-cleanup phrasing ("not-yet-installed", "will be pruned",
"pending cleanup") found in either module or in `canvas_sync.py`'s search
branch (that file's own guard comment, added by Task 4's fix round, is
already accurate and past-tense-correct) or in
`Tests/Architecture/test_library_search_rag_wiring.py`'s own module
docstring (already correctly rewritten by Task 4 to describe the finished
3-task series).

## 4. Recipe additions — 5 new lessons in §18

Added lessons 4-8 to §18 (lessons 4-7 are the lesson-candidates named in
this task's brief; the canvas_sync controller-as-screen forwarding trap
was already lesson 2; lesson 8 was added later, from this task's own
self-caught pin drift, §7a):

4. **State-shape evidence accuracy** — Task 2's review found the
   state-shape section's "all 20 fields in one call via a continuation"
   claim was factually wrong (14/20, no continuation); the fix re-derived
   the claim from a per-field grep rather than an impression. Generalizes
   §16 lesson 1's "verbatim claim needs evidence discipline" to fan-out
   claims about what a method reads.
5. **Moved-body docstring corrections are cleanup-PR-only** — the general
   rule behind Task 3's deferred docstring-correction ruling (byte-for-byte
   canon applies to a body's docstring exactly as to its code).
6. **No-red-ships reconfirmed** — Task 3's path-census test's same-fix-round
   retarget (not deferred to cleanup) is a second, independent confirmation
   of §3's existing rule.
7. **Born-governed same-commit mechanism, first live exercise** —
   `library_rag_search_controller.py`'s glob-discovered birth and two
   same-commit re-pins (1857→1890→1895) confirm Task 1's governance design
   (§17) worked exactly as intended on the first subsystem to need it.
8. **A battery run captured before a later edit does not verify that
   edit** — this task's own docstring fix (item 2 above) grew the
   controller file +2 lines after the ratchet had already been confirmed
   green; caught only by a fresh `_measure()` call after the edit, not by
   trusting the earlier green run. Full incident in §7a.

## 5. Durable evidence

`.superpowers/` is gitignored (`.gitignore:8`); per the wave-2
durable-record mandate, force-added the wave-3 SDD directory's reports and
review diffs (previously only `progress.md` was tracked):

```
git add -f .superpowers/sdd/2026-09-03-library-decomposition-wave3-search-rag/
```

Adds: `review-*.diff` (8 files), `task-{1,2,3,4}-report.md`. 13 files,
11,128 insertions (frozen historical diffs + reports, none edited).

## 6. Fresh measurements

Both ratchet files' `_BUDGETS` rows re-confirmed against a fresh `_measure`
run (the tests themselves, which call the exact same `len(...splitlines())`
expression the recipe documents) — no drift found, no fix needed beyond
the task-31203 prose correction in §3 above.

## 7. Full verification battery

All commands from `.worktrees/library-decomp-foundation`, `.venv/bin/python`,
`-p no:randomly`.

- **Both size ratchets**
  (`test_screen_size_ratchet.py` + `test_library_modules_size_ratchet.py`):
  **30 passed / 2 failed** — both failures the documented pre-existing
  `chat_screen.py` rows.
- **All 4 wiring suites** (collections/conversations/export/search+RAG) +
  **all 3 characterization files** (collections/conversations/export; no
  dedicated search+RAG characterization file exists) + **support-layer
  surface** + **recompose census** (`test_library_recompose_ratchet.py`) +
  both size ratchets, run together (11 files): **79 passed / 2 failed**
  (same 2 chat_screen reds).
- **Preflight** (`./scripts/preflight.sh`): **6/6 green** (CSS bundle,
  profile-owned-path census, production diagnostic inventory, backlog task
  ids, chachanotes table allowlist, index plan pins).
- **Full `Tests/Architecture` suite** (550 tests): **15 failed / 534
  passed / 1 skipped**. All 15 failures identity-matched against the
  documented dev-drift baseline (2 `chat_screen.py` rows +
  `test_console_realtime_controller_boundary`,
  `test_console_review_selection_controller_boundary`,
  `test_console_wave6_closeout_inventory`,
  `test_console_wave6_inventory` x2, `test_default_timeout_session_guard`,
  `test_persistent_diagnostic_inventory` x2, `test_timer_path_static_
  update_inventory` x3, `test_worker_exclusive_group_inventory` x2) —
  zero Library-related, zero new. Passed count (534) is higher than Task
  1's own baseline (527) because wave-3 added new passing tests since.
- **Full sequential xdist paired-baseline sweep**
  (`Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal`, branch
  then a `git stash -u` pristine baseline at `a150fc766`, per recipe §7):

  | | Failed | Passed | Wall time |
  |---|---|---|---|
  | Branch (this task's tree) | 357 | 3924 | 1323.95s (22:03) |
  | Baseline (`a150fc766`, `git stash -u`) | 349 | 3932 | 1329.73s (22:09) |

  Both totals fall inside the recipe's own documented ~330-355 historical
  backdrop (§7), and the baseline number is an EXACT match to Task 4's
  own most recent baseline run (349/3932) — a strong reproducibility
  signal. Diffing the two sorted failure-name sets: **347 shared, 10
  branch-unique, 2 baseline-unique**. Both baseline-unique names are
  already documented pre-existing in §7 (from wave-2/Task 4's own
  sweeps) — noise in the opposite direction, not investigated further.
  Of the 10 branch-unique names, 2 exactly match Task 4's own two
  already-confirmed §7 entries (`test_library_media_initial_error_is_
  unknown_and_retry_is_unique` — confirmed noise; `test_library_media_
  page_error_retains_rows_and_gates_unsafe_controls` — confirmed
  pre-existing). The remaining 8 were re-run combined, single-process,
  on the CURRENT (branch) tree: **4 passed cleanly** (xdist noise —
  `test_library_file_notes_git.py::test_guarded_commit_success_renders_
  and_focuses_fresh_owner_status[size1]`, `test_library_media_reader_
  traversal_t22207.py::test_loading_banner_paints_in_place_without_body_
  rebuild`, both `test_library_prompts_canvas.py` history tests). **4
  still failed**: `test_library_notes_reader.py::test_wide_editor_deep_
  link_keeps_reader_navigation_and_local_back`, `test_screen_navigation.
  py::{test_generic_library_entry_lands_hub_on_first_visit, test_
  generic_reentry_returns_to_library_landing, test_library_screen_round_
  trip_returns_to_landing_with_rag_draft}`. Re-ran these 4 on a SECOND
  `git stash -u` to the same pristine `a150fc766` tree, combined,
  single-process: **all 4 reproduced identically** — confirmed genuinely
  pre-existing, not caused by this task or the wave-3 series. Notably,
  `..._with_rag_draft` is the SAME test name Task 4's own sweep flagged
  as BASELINE-unique in the opposite direction (§18) — flip-flopping
  which side it fails on between independent runs is itself strong
  evidence of pure flakiness, unrelated to either tree's content. **Zero
  real regressions.** All 4 newly-confirmed names added to the recipe's
  §7 documented list.
- **Probe run**
  (`perl -e 'alarm 150; exec @ARGV' .venv/bin/python Helper_Scripts/
  library_click_probe.py`):

  | interaction | settle (ms) | max gap (ms) | recompose | full-update | mounts | nodes |
  |---|---|---|---|---|---|---|
  | media (switch-in) | 464 | 132 | 0 | 2 | 178 | 118 |
  | media (re-click same) | 318 | 56 | 0 | 2 | 94 | 118 |
  | media (re-click same, 2nd) | 314 | 56 | 0 | 2 | 94 | 118 |
  | notes (switch) | 345 | 125 | 0 | 1 | 110 | 110 |
  | notes (re-click same) | 264 | 59 | 0 | 1 | 38 | 110 |
  | media (switch-back) | 422 | 121 | 0 | 1 | 180 | 118 |
  | notes (switch, 2nd) | 351 | 129 | 0 | 1 | 110 | 110 |
  | media (switch-back, 2nd) | 430 | 96 | 0 | 1 | 180 | 118 |

  Compared against the recipe's own recorded wave-2-close band (§16):
  settle 264-485ms (this run: 264-464ms, inside), max gap 54-195ms (this
  run: 56-132ms, inside), recompose 0 in both, full-update 1-2 in both.
  Media-row mount/node counts are ~5 higher than the wave-2-close
  recording (178/118 vs 163/113) — a small, consistent drift across ALL
  media rows only (Notes rows are byte-identical, 110/110 in both), most
  plausibly unrelated Media-screen growth merged from `dev` since wave-2
  closed, not a search+RAG effect (this probe's click path never touches
  Search/RAG code, and the drift's magnitude/direction is not what a
  recompose-count or full-update-count regression would look like — both
  stayed at their recorded values). Within the recipe's own noise
  tolerance; no regression signal.

## 7a. Self-caught: the controller docstring fix itself drifted the pin it was verified against

After fixing `library_rag_search_controller.py`'s docstring (§3, item 2)
and re-confirming the battery green, a later fresh-measurement pass (§6,
run via the ratchet's own `_measure()` function directly rather than only
via pytest) found the file now measured **1897 lines against a pin of
1895** — the rewritten paragraph (2 lines → 4 lines, to fit "task 2
installed on `LibraryScreen` (deleted at cleanup, task 4...)" without
cramming) grew the file by 2 lines, and an EARLIER battery run (before
this specific edit landed) had already been recorded as green, creating a
false sense that the fix was measurement-verified when it was not.
`test_controller_does_not_grow_past_its_budget[library_rag_search_
controller.py]` confirmed red on re-run. Re-pinned 1895 → 1897 in
`test_library_modules_size_ratchet.py`, same commit, dated comment, per
§17's re-pin-at-move flow — exactly the mechanism Task 3/Task 4 used for
their own docstring-correction growth. Re-ran both ratchets (30/2
restored) and the full 11-file battery (79/2) to confirm. Recorded here
rather than smoothed over: this is the wave's own "verified-before-
written" discipline applied to my own edit, not just to the prior tasks'
claims — a battery run captured BEFORE a later edit does not verify that
edit; only a fresh run captured AFTER it does.

## 8. Files changed

- `backlog/docs/library-decomposition-recipe.md` — 4 new lessons appended
  to §18; a new §7 bullet recording the wave-close sweep's 4 newly-
  confirmed pre-existing names (verified rest of the file accurate, no
  other changes needed).
- `Tests/Architecture/test_library_modules_size_ratchet.py` — re-pinned
  `library_rag_search_controller.py` 1895 → 1897 (dated comment), the
  same-commit correction for the +2-line growth this task's own
  docstring fix caused (§7a).
- `backlog/tasks/task-31203 - Library-decomposition-wave-3-combined-searchRAG-series.md`
  — 3 stale numbers in Implementation Notes corrected to their fix-round-1
  final values; status/ACs/frontmatter/provenance section untouched
  (diffed clean).
- `tldw_chatbook/UI/Library_Modules/library_rag_search_state.py` — module
  docstring's shim-deletion paragraph rewritten past-tense, correct PR
  attribution, matching the export/collections template.
- `tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py` —
  module docstring's "LibraryScreen carries" claim corrected to past
  tense with cleanup attribution.
- `.superpowers/sdd/2026-09-03-library-decomposition-wave3-search-rag/`
  — force-added `review-*.diff` (8) + `task-{1,2,3,4}-report.md` (4),
  previously untracked.
- `.superpowers/sdd/2026-09-03-library-decomposition-wave3-search-rag/progress.md`
  — Task 5 ledger entries.

## 9. Self-review

- Verified before writing: every pin-trajectory number in this report and
  in the fixes above was re-derived from `git show <rev>:<path> | wc -l`
  directly, not copied from a prior comment or report.
- Diffed the task-31203 file after the direct edit to confirm no
  unintended section was touched (frontmatter, AC block, provenance
  section all byte-identical to before).
- Did not touch `library_conversations_controller.py`'s own identical
  staleness (out of scope, flagged instead of silently fixed or ignored).
- Did not re-run the narrow single-process `-k "(search or rag) and
  library"` sweep (already run twice by Task 4, both times documented);
  relied on the required full xdist paired-baseline sweep instead, per
  this task's explicit brief.
- **Caught my own regression before it shipped** (§7a): a battery run
  captured green BEFORE a later docstring edit does not verify that
  edit — the drift was only found by taking a FRESH measurement after
  every edit landed, not by trusting the last "all green" result. Fixed
  same-commit, per the recipe's own governance design, and re-verified.
- **The initial single-test isolation check was contaminated** and its
  result discarded: I first ran one suspected-branch-unique test while
  the baseline xdist sweep's 8 workers were still actively running in
  the background, producing a spurious 30s mount-timeout that had
  nothing to do with the tree under test. Caught by checking `ps aux`
  for competing pytest processes; the correct re-run (after the baseline
  sweep genuinely finished, machine idle) is what the report's §7 numbers
  reflect — a second instance of the wave's own "concurrent runs amplify
  flakiness" lesson (recipe §7), this time self-inflicted rather than a
  finding to record.
- Did not do the "final whole-branch review" or open a PR — the wave-3
  plan's Task 5 description mentions both, but the actual dispatch brief
  for this task scoped Task 5 to bookkeeping/verification only, explicitly
  framing this as "before the wave goes to its final review and PR" (i.e.
  a separate, later step).
