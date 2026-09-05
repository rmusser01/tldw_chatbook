# Wave-5 Task 4 — Wave close

Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§20 (this task
extends §20 with a "Wave-5 close" subsection and updates §8's per-subsystem
table). Plan: `Docs/superpowers/plans/2026-09-05-library-decomposition-wave5-ingest.md`.
Ledger: `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/progress.md`.
Prior tasks: Task 1 (ingest state PR + fix round), Task 2 (ingest controller
PR + fix round), Task 3 (ingest cleanup PR + fix round) — the ingest series
is complete (Tasks 1-3, recipe §20).

## 1. Recipe close-out

### §8 (subsystem order table)

The `ingest` row (already rewritten to "complete" by Task 3) is given one
small addition: both bypass-shape findings are now explicitly marked as
review-found CRITICALs (matching the skills row's own "CRITICAL
unbound-attribute-escape" framing) and the row now points at §20's own new
"Wave-5 close" subsection alongside §20 itself.

### §20 ("Wave-5 close" subsection, new)

Added after the existing ingest-series "Lessons" list (already written by
Task 3). Contents:

- **Pin trajectory — full wave-5 chain**: re-derived directly from
  `git show <commit>:<path>` reads of both ratchet files at each commit (not
  carried over from any report), in true chronological order. Confirmed
  EXACT match to what Tasks 1-3 already documented: screen
  `41574/1302 → 41520/1302 → 40096/1302 → 40131/1302 → 40094/1296`;
  controller born `2510` (Task 2's `68a896993`) `→ 2536` (Task 2's fix
  round, `e3d85ad21`) `→ 2558` (Task 3, `e6148e29a`) `→ 2569` (Task 3's fix
  round, `5b9c7bdf4`). No corrections needed to the prior tasks' own
  framing — unlike wave-4 close, every number in the ledger's own running
  narrative matched the git-derived chain exactly.
- **Verification battery** (this task's own fresh run, not carried over):
  fresh `_measure()` on both ratchet files (exact match, zero drift); the
  combined 6-wiring-suite/4-characterization/both-guards/recompose-census/
  support-layer run (105 passed, 2 pre-existing `chat_screen.py` failures);
  the full `Tests/Architecture/` run (550 passed, 1 skipped, 16 failed —
  identical categories to wave-4 close's own documented 16, 7 more passing
  from ordinary interim `dev` growth); preflight (all six checks green,
  3,241 task files).
- **Full sequential xdist paired-baseline sweep — whole-wave span**: branch
  (`5b9c7bdf4` + this task's own doc-only edits) against an ISOLATED
  `git worktree add` + its own `uv venv` at the wave-5 START commit
  (`9e62dd8f7`) — applying this wave's OWN task-1 lesson (an in-place
  overlay is interruption-unsafe) from the start, rather than a path-scoped
  checkout overlay. Machine load recorded at the start: load average
  ~3.0-3.2, 2 `pytest` processes (one unrelated) — substantially quieter
  than wave-4 close's own ~22.7.
- **Probe run**: `Helper_Scripts/library_click_probe.py`, run separately
  from the xdist sweep to avoid CPU contention, compared against both prior
  recorded runs (wave-2 close §16, wave-4 close §19).
- **Lessons** (4, three cross-referencing existing §3/§7/wave-4-close
  content rather than duplicating it, one genuinely new): (1) the wave's
  two new bypass shapes (§3's seventh/eighth) as the headline finding,
  both review-found CRITICALs, reinforcing wave-4's own "review is a
  mandatory gate" lesson a second time; (2) a count-accuracy-discipline
  lesson naming THREE separate incidents across all three ingest tasks
  (task 1's 24→25→27 site-count correction, task 2's five-round
  78→57 mover-count churn plus a within-task module-globals
  undercount, task 3's wholesale mover-caller mislabeling across all 23
  KEEP rows plus two further count bugs it surfaced) as recurrence of
  wave-4's own already-stated rule, generalizing that restating a rule
  does not prevent its recurrence — only an independent re-derivation
  method does; (3) a NEW finding: `.git-blame-ignore-revs` is missing
  every wave's own state-PR commit except the conversations exemplar's
  backfilled one (export, collections, search+RAG, skills, and — until
  this close's own audit — ingest's `12ba4fb13`), a documentation-hygiene
  gap with no test surface that survived four consecutive waves; fixed for
  ingest's own commit here, the other four named as a lead, not
  retroactively fixed; (4) confirmation that this wave applied its own
  task-1 interruption-unsafe-baseline lesson preemptively in every
  subsequent sweep, with no redo needed.

## 2. Follow-up filing

**True max task ID swept properly this time, after an initial miss.** My
first pass used only a local filesystem scan of this worktree's
`backlog/tasks/`+`backlog/archive/tasks/` (max **31429**) and filed
`TASK-31430` directly from that — repeating exactly the trap
`backlog/docs/lessons-backlog-hygiene.md` warns against (a CLI/local-scan
answer is not a safe answer). Caught before committing anything, by running
the lessons file's own prescribed remote-ref sweep:

```bash
git for-each-ref --format='%(refname)' refs/remotes/ | while read -r b; do
  git ls-tree -r -z --name-only "$b" backlog/ | tr '\0' '\n'
done | grep -oE 'task-[0-9]+' | cut -d- -f2 | sort -n -u | tail -5
```

First sweep: true max **31566** (`TASK-31430` already 136 ids stale). A
`backlog task create` throwaway probe confirmed the CLI's own
auto-assignment is equally unsafe (offered `31431`, still far below the
swept max) — both `TASK-31430` and the probe were deleted before anything
referenced either. A SECOND sweep, run immediately before re-filing (after
`git fetch --all`), found the ceiling had already moved to **31635** in the
interim — the lessons file's own "an id scanned at the start is not safe at
the end" warning, reproduced live within this one task. Filed at
**TASK-31650** (comfortably clear of 31635, re-verified individually free
against the same remote-ref sweep immediately before the rename), by
creating via the CLI (which assigned the stale `31430` again) then
`git mv`-renaming the file and editing its frontmatter `id:` field directly
— the CLI has no explicit `--id` flag. Rendering re-verified with
`backlog task 31650 --plain` before moving on. No new
`lessons-backlog-hygiene.md` entry: this is a fresh occurrence of an
already-fully-documented pattern, caught by following that file's own
prescribed procedure, not a new failure mode.

- **TASK-31650** — "Cross-controller module-globals patch-bypass census
  (recipe eighth bypass shape) across all six Library decomposition
  controllers." Files the cross-wave follow-up the ledger flagged at wave-5
  task 2 (`e3d85ad21`) and recipe §3's eighth-bypass-shape entry: the
  ingest controller's own `_apply_library_ingest_backend_save` was found,
  by the mandated mechanical census, to read the shared
  `_sync_library_canvas` dispatcher as a bare module global — confirmed
  LATENT for ingest (10 files/38 sites, none active), but the five OTHER
  Library decomposition controllers (conversations, export, collections,
  search+RAG, skills) import the identical dispatcher the identical way
  and landed BEFORE this census existed, so none of them has ever had it
  run against their own moved-method sets. 2 tickable ACs: the census run
  against all six controllers with every finding classified ACTIVE/LATENT
  and recorded in that controller's own module docstring; any ACTIVE
  collision fixed the same way `_resolve_ingest_source` was (exclude +
  rebind + existing-file probe), in the same task.

## 3. Stale-doc sweep

Scope: the ingest state and controller modules
(`tldw_chatbook/UI/Library_Modules/library_ingest_state.py`,
`tldw_chatbook/UI/Library_Modules/library_ingest_controller.py`, plus the
lower-level `tldw_chatbook/Library/library_ingest_state.py`), checked
against the collections/search+RAG/skills post-cleanup docstrings as the
template.

**Zero stale spots found** — task 3's own fix round (`5b9c7bdf4`) had
already corrected every present-tense/future-tense/stray-count claim these
modules carried (the "56 original names" delegator claim, both stray "63"
counts, the shim-block "originally kept... now deleted" past-tense
rewrite in both state modules, and the `LibraryIngestFormState` docstring's
current-attribute-path correction). This task's own independent re-check
(content grep for `shim`, `delegat`, `63`, `every one of`, `keeps... names`,
`will be`, `once landed`, `TODO`/`FIXME`, `not yet`, `still lives on the
screen`, `Owned by the screen` across all four files) found nothing further
to fix — the only "will be" hits are unrelated forecast-messaging prose in
`Library/library_ingest_state.py` (pre-flight/submit-outcome copy, e.g.
"0 will be sent to the server"), not decomposition-state claims.
`Tests/Architecture/test_library_ingest_wiring.py`'s own two docstrings
were already corrected by task 3 too (both former "63" mentions now say
56). Confirmed all four files still parse and import cleanly.

## 4. Durable evidence

`.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/` (ledger,
three task reports, six review diffs, this report) force-added to the
wave-close commit — the directory is `.gitignore`d (`.superpowers/`) by
default, so without `git add -f` this whole audit trail would die with the
worktree.

## 5. Fresh measurements

Both ratchet files' own `_measure()` functions called directly (not through
pytest) against the current tree: screen **40094 lines / 1296 methods**,
controller **2569 lines** — EXACT match to both recorded `_BUDGETS`/pin
values. Zero drift; nothing to lower.

## 6. Full verification

- All six wiring suites + 4 characterization files + both size guards + the
  recompose-census guard + support-layer surface suite, combined single
  run: **105 passed, 2 failed** (both documented pre-existing
  `chat_screen.py` ratchet rows).
- Full `Tests/Architecture/` run: **550 passed, 1 skipped, 16 failed** —
  identical failure categories to wave-4 close's own documented 16 (Console
  realtime/review-selection boundary ×2, console wave6 closeout/inventory
  ×4, default-timeout-session-guard ×1, persistent-diagnostic-inventory ×2,
  chat_screen ratchet ×2, timer-path-static-update-inventory ×3,
  worker-exclusive-group-inventory ×2), same TASK-15743 skip reason. Zero
  Library/Ingest-scoped failures.
- `preflight`: all six derived-artifact checks green, including the backlog
  task-id sweep (3,241 files, no duplicates, including this task's own new
  filing).
- Full sequential xdist paired-baseline sweep (whole-wave span, `9e62dd8f7`
  vs. `5b9c7bdf4`+close, `Tests/UI -k "library" -p no:randomly -q -n 8
  --dist worksteal`, isolated worktree baseline): **branch 356 failed/3994
  passed (1457.61s) vs. baseline 356 failed/3985 passed (1495.87s)**; 351
  shared, 5 baseline-unique (not investigated), 5 branch-unique. 4 of the
  5 branch-unique names matched ones already documented in recipe §7
  (wave-5 task 2's own `test_loading_banner_paints_in_place_without_body_
  rebuild`, wave-3 task 5's `test_wide_editor_deep_link_keeps_reader_
  navigation_and_local_back`, and two flaky-on-rerun names from wave-4
  close/wave-5 task 1 that passed cleanly on a combined re-run here too).
  The 5th, `test_library_shell.py::test_library_note_compact_deep_link_
  intent_opens_notes_stage[context2-#library-note-body-editor-False]`, is
  genuinely new — confirmed pre-existing by reproducing identically (same
  30s DOM-mount-timeout signature) in TRUE isolation on BOTH the branch
  and the isolated baseline worktree (an initial combined-run test on the
  baseline showed it passing alongside an unrelated test, overturned by a
  true-isolation rerun — recorded as its own small methodology note: a
  combined-run pass is not sufficient evidence for a timing-sensitive
  DOM-mount test). Added to recipe §7. **Zero real regressions across the
  whole wave-5 span.**
- Probe run (`Helper_Scripts/library_click_probe.py`, run separately after
  both sweeps to avoid CPU contention; machine load 2.31/3.84/5.37,
  substantially quieter than wave-4 close's ~22.7): every row sits INSIDE
  wave-2 close's own original recorded band (settle 264-485 ms, max gap
  54-195 ms) — the closest match of any close-time probe run to that
  baseline. Load-independent columns (recompose 0, full-update
  `2/2/2/1/1/1/1/1`) match both prior closes row-for-row; mounts/nodes
  match wave-4 close's own numbers with no further node-count drift this
  time. This wave's diff touches none of the probed Media/Notes
  rail-switch path. See recipe §20 "Wave-5 close" for the full table.

## 7. Files changed

- `backlog/docs/library-decomposition-recipe.md`: §8's ingest row gains
  explicit "review-found CRITICAL" framing for both bypass shapes; §20
  gains a "Wave-5 close" subsection (pin trajectory, verification battery,
  whole-wave sweep, probe run, 4 lessons).
- `.git-blame-ignore-revs`: adds the missing `12ba4fb13` (ingest task 1's
  own pure-move commit), closing a gap this task's own audit found — the
  same gap recurs, unfixed, for four earlier waves' state-PR commits
  (named, not retroactively fixed, in the new entry's own comment).
- `backlog/tasks/task-31650 - ....md`: new follow-up filing (§2 above).
- `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/`:
  force-added for durable evidence (ledger + 3 reports + 6 review diffs +
  this report).

Commit: `225a75be1`

## 8. Self-review

- **No production behavior changed.** Every edit in this task is
  documentation (recipe, backlog filing, ledger, this report) or a
  durable-artifact-hygiene addition (`.git-blame-ignore-revs`, a comment
  metadata file, never consulted by the test suite or the shipped
  application) — verified by `git diff --stat` before committing (3 files
  modified + 1 new task file, zero `tldw_chatbook/` or `Tests/` changes)
  and by the fact that every battery number recorded above (fresh
  `_measure()`, the combined wiring/characterization/guard run, the full
  `Tests/Architecture/` run) reproduced identically to what Tasks 1-3 had
  already landed.
- **A genuine ID-collision near-miss was caught by following the project's
  own documented procedure, not by luck.** The first pass filed TASK-31430
  from a local-only scan — exactly the trap `lessons-backlog-hygiene.md`
  warns against — and the true max (swept properly, per that file's own
  remote-ref procedure) was already 136 IDs ahead at 31566, then moved
  again to 31635 within the same task after a `git fetch --all`. Caught,
  both stale filings deleted, and refiled at TASK-31650 with a live
  re-sweep immediately before the rename. No new lessons-file entry was
  warranted (the pattern and its fix are already fully documented there);
  this is the existing safeguard working as designed, recorded honestly
  rather than glossed over as if the first sweep had been sufficient.
- **A genuinely new durable-artifact gap was found and fixed, not just
  inherited.** The close-time audit of `.git-blame-ignore-revs` (prompted
  by needing to verify ingest's own three commits were all correctly
  entered) found ingest task 1's own pure-move commit missing entirely,
  and widening the check found the identical gap in every prior wave's own
  state-PR commit. Fixed for ingest's own commit (this wave's
  responsibility); the other four are named as a lead, not silently
  left for the next person to rediscover from scratch, and not
  overclaimed as fixed when they weren't touched.
- **The one genuinely new branch-unique sweep failure was verified with
  the strongest available method, not the first result that looked
  clean.** An initial combined-run test of the two "unclear" branch-unique
  names on the isolated baseline showed one PASSING; re-running that same
  name alone (true isolation) on the SAME baseline tree reversed that
  result to FAILING, matching the branch's own true-isolation failure
  exactly. Trusting the first (combined-run) result would have wrongly
  classified a pre-existing flaky test as a real regression's absence-of-
  evidence, or worse, missed recording it in §7 at all. The correction is
  recorded in the recipe's own new sweep-evidence entry, not silently
  smoothed over.
- **The stale-doc sweep found nothing to fix, and that absence was
  verified rather than assumed.** Task 3's own fix round had already
  corrected every present/future-tense claim and stray count in the
  ingest state/controller modules; this task's own independent content
  grep (a different search, not a re-read of task 3's own claim) across
  all four ingest-adjacent modules confirmed zero remaining spots — a
  negative result earned by checking, not by trusting the prior task's
  own "fixed" framing at face value.
- **The isolated-worktree baseline methodology (this wave's own task-1
  lesson) was applied preemptively for both this close's sweeps**, never
  falling back to a same-tree overlay — the exact discipline the wave's
  own lesson 4 records as a confirmation rather than a new finding.
- Open risk, unchanged from Task 3's own carried note: the cross-controller
  `_sync_library_canvas` module-globals audit across the five prior
  controllers remains unfixed — TASK-31650, filed by this task, is the
  durable tracking for it.
