# Wave-4 Task 4 — Wave close

Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§19 (this task
extends §19 with a "Wave-4 close" subsection and updates §8's per-subsystem
table). Plan: `Docs/superpowers/plans/2026-09-04-library-decomposition-wave4-skills.md`.
Ledger: `.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/progress.md`.
Prior tasks: Task 1 (state PR), Task 2 (controller PR + post-landing review
fix round), Task 3 (cleanup PR + review fix round) — the skills series is
complete (Tasks 1-3, recipe §19).

## 1. Recipe close-out

### §8 (subsystem order table)

The `skills` row (previously a bare stub: `| 3 | skills | 15 | |`) is
rewritten to match every other completed subsystem's row shape: marked
**complete** (wave-4 Tasks 1–3), and given the same field/method/exclusion
summary the export/collections/search+RAG rows already carry — 36 fields
moved via the three-way `skill_state_shim_attr()` resolver; 86 of 127
"skill"-named method candidates moved to ONE `LibrarySkillsController` (41
exclusions across 5 classes, including the new bare-self-identity-argument
hazard and the CRITICAL unbound-attribute-escape found by post-landing
review); 16 of 86 screen delegators pruned at cleanup; a pointer to §19 for
the full as-landed numbers.

### §19 ("Wave-4 close" subsection, new)

Added after the existing skills-series "Lessons" list (already written by
Task 3). Contents:

- **Pin trajectory — full wave-4 chain**: re-derived directly from `git log`
  (each commit's own `_BUDGETS` value in both ratchet files, not carried
  over from any report) rather than trusted from the prior tasks' reports.
  Confirmed EXACT match to what Tasks 1-3 already documented: screen
  `43225/1311 → 43179/1311 → 41247/1311 → 41155/1295`; controller born
  `3099` (Task 2's `60857a2be`) `→ 3131` (Task 2's fix round, `bf13b133b`)
  `→ 3140` (Task 3, `ed4c29d45`). One correction to the prior framing: Task
  2's own report describes an in-session `3181 → 3113 → 3099` draft
  sequence, but `git log` shows only `3099` at the controller's actual birth
  commit — the two earlier numbers belong to uncommitted working-tree
  drafts from the same session (Form-B/Form-C reverts), never landed as
  separate commits, so they do not appear in the git-derived chain. Noted
  in the recipe rather than silently smoothed over.
- **Verification battery** (this task's own fresh run, not carried over):
  fresh `_measure()` on both ratchet files (exact match, zero drift); the
  combined wiring/characterization/support-layer/size-guard run (83 passed,
  2 pre-existing failures); the full `Tests/Architecture/` run (543 passed,
  1 skipped, 16 failed — one fewer than Task 2's documented 17, traced to
  an external git-object-availability change in
  `test_persistent_diagnostic_inventory.py`, unrelated to this diff); the
  `-k "skill and library"` sweep (10 failed/272 passed, exact match to Task
  3's own baseline); the `Tests/Skills/` full run (538 passed/1 failed, the
  OTHER documented pre-existing failure flipped to pass this run); preflight
  (all six checks green).
- **Full sequential xdist paired-baseline sweep — whole-wave span**: unlike
  Tasks 1-3's own per-task sweeps (each comparing against its own immediate
  predecessor), this close task compares the WAVE'S full span — branch
  (`f42f75d98` + this task's own doc-only edits) against a path-scoped
  checkout of `2372ea764` (the wave-4 START commit), mirroring wave-2
  close's own (§16) precedent for a wave-level rather than per-task
  comparison. Machine load recorded at both the start of the sweep AND
  independently via the probe run (see below) — this session's own machine
  ran at a sustained ~22-23 load average with 8+ concurrent `pytest`
  processes throughout, a condition worth recording explicitly per §19
  lesson 5's own "the paired comparison is what keeps this valid regardless
  of load" discipline.
- **Probe run**: `Helper_Scripts/library_click_probe.py`, once, compared
  against the ONLY prior recorded run in this recipe (wave-2 close, §16).
  Every interaction came in slower than that band (e.g. `notes (switch,
  2nd)`'s 587 ms max gap vs. wave-2's highest recorded 195 ms) — recorded
  as evidence of ambient machine load (8+ concurrent `pytest` processes,
  ~22.7 load average measured at run time), NOT a code regression: this
  task's diff touches zero code on the Media/Notes rail-switch path the
  probe exercises, and every prior wave-4 task's own report already
  confirms the skills move touches none of that path either (a completely
  separate subsystem). Recorded honestly with the load caveat attached,
  neither silently matched to the old band nor wrongly flagged as a
  regression.
- **Lessons** (2 new, both cross-referencing existing §3/§19 content rather
  than duplicating it): (1) a "count-accuracy discipline" lesson naming
  three separate wave-4 incidents (the "three pairs" arithmetic error, the
  "18 vs. 28" restructure-count review fix, and the mover-count
  re-derivation after Forms B/C) as one recurring failure shape worth a
  single named rule; (2) a lesson on why a CRITICAL bug (the unbound
  `focused` property) survived an entire green verification battery and was
  only caught by independent review — stated as a general point about what
  a green battery does and does not prove; (3) a lesson tying the probe's
  own load-sensitivity to the xdist sweep's already-documented load-
  sensitivity (§19 lesson 5), recommending the load snapshot become a
  standing part of every future probe run.

## 2. Follow-up filings

True max task ID swept across every remote ref, every local branch, and
this worktree (NUL-safe, numeric sort, per `lessons-backlog-hygiene.md`):
**31420**. The `backlog` CLI's own auto-assignment was independently probed
(`backlog task create "PROBE throwaway id check"` → offered **31263**,
already stale by 157 IDs) and confirmed unsafe, per the lessons file's own
"never trust CLI auto-assignment" rule; the probe task was deleted before
anything referenced it. Both follow-ups were hand-filed at
`git status`/`gh pr list --search`-confirmed-clear IDs **31421** and
**31422**, one integer past the swept maximum, each with a hand-authored
two-item `## Acceptance Criteria` block (never `--ac` with commas, per the
same lessons file) and verified rendering via `backlog task <id> --plain`
before moving on.

- **TASK-31421** — "Library skills browse: loading/ready settlement race
  can drop a correct focus restore." Files the pre-existing race Task 2's
  own post-landing review fix round found while building its covering test
  (§12a of `task-2-report.md`): `LibrarySkillsBrowseController`'s two-round
  settlement (`dispatch()`'s synchronous "loading" call, then the async
  worker's own "ready" call) can land both rounds in the same event-loop
  turn against a fast skills-scope service, and `queue_after_recompose`'s
  one-pending-callback-per-host limit lets the ready round's own resync
  silently overwrite the loading round's still-pending, CORRECT focus
  restore. Confirmed genuine and reproducible with a bounded-delay fake
  service; unrelated to and unaffected by the wave-4 move (reproduces with
  the reviewed `focused` property both present and absent). 2 tickable ACs:
  the race no longer drops a correct restore under a fast service, and a
  reproducing test exists (fails before, passes after).
- **TASK-31422** — "Settle the closeout-cycle destination flake's rate
  disparity with a larger paired sample." Files Task 1's own minor,
  deferred finding (progress.md: "closeout-cycle flake triage rests on an
  11-run sample [...] a 20+-run paired quiescent sample would settle [it]")
  for `test_closeout_single_app_route_cycle` — the mechanism is
  independently ruled out (traced to the 'collections' destination step,
  unrelated to any of the 36/86 skills fields/methods this wave moved), but
  the observed rate disparity between branch and baseline in small samples
  (7/8 vs. 1/3 in Task 1's own isolated re-runs) has never been settled with
  a large-enough sample on a quiescent machine. 2 tickable ACs: a 20+-run
  paired quiescent sample is captured, and its comparison either closes the
  question or names a concrete mechanism.

## 3. Stale-doc sweep

Scope: the skills state and controller modules
(`tldw_chatbook/UI/Library_Modules/library_skills_state.py`,
`tldw_chatbook/UI/Library_Modules/library_skills_controller.py`), checked
against the collections/search+RAG post-cleanup docstrings as the template
(both already correctly rewritten to past tense after their own series
completed).

**One stale spot found and fixed**, in `library_skills_state.py`'s own
module docstring: it still described the screen-side shim block in FUTURE
tense — "A future Skills controller PR will move the subsystem's methods
off the screen and take over this shim's job [...] the Skills cleanup PR
will then delete this screen-side block" — even though both the controller
PR (Task 2) and the cleanup PR (Task 3) had already landed and done exactly
that. Rewritten to past tense, naming the actual controller
(`LibrarySkillsController`), the actual accessor
(`skills_state_accessor`), and the actual outcome, matching
`library_collections_state.py`'s and `library_rag_search_state.py`'s own
post-cleanup docstring wording almost verbatim (same three sentences: what
the screen originally did, what the cleanup PR did, why the controller's
own copy is permanent).

`library_skills_controller.py`'s own module docstring was checked line by
line against the same template and found already correctly worded
throughout (Task 3's own review fix round had already corrected its "70 of
86" delegator-count claim and its "three pairs" arithmetic error; this
task's re-check found nothing further). The controller's own generated
shim-block comment (`--- BEGIN generated skills-state shims (permanent;
byte-for-byte canon) ---`) is also already correctly worded in past/present
tense throughout.

Confirmed both files still parse and import cleanly after the edit
(`ast.parse` + a direct module import).

## 4. Durable evidence

`.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/` (ledger,
three task reports, five review diffs) force-added to the wave-close
commit — the directory is `.gitignore`d (`.superpowers/`) by default, so
without `git add -f` this whole audit trail would die with the worktree,
per `lessons-backlog-hygiene.md`'s own "gitignored working files die with
their worktree" entry.

## 5. Fresh measurements

Both ratchet files' own `_measure()` functions called directly (not through
pytest) against the current tree: screen **41155 lines / 1295 methods**,
controller **3140 lines** — EXACT match to both recorded `_BUDGETS`/pin
values. Zero drift; nothing to lower.

## 6. Full verification

- All five wiring suites + 3 characterization files + support-layer surface
  + both size guards, combined single run: **83 passed, 2 failed** (both
  documented pre-existing `chat_screen.py` ratchet rows).
- Full `Tests/Architecture/` run: **543 passed, 1 skipped, 16 failed** — see
  §1 above for the one-fewer-than-17 explanation; zero Library/Skills-scoped
  failures.
- `-k "skill and library"` sweep (`Tests/UI`+`Tests/Library`, single
  process): **10 failed, 272 passed, 22073 deselected** — exact match to
  Task 3's own documented baseline, name-for-name.
- `Tests/Skills/` full run: **538 passed, 1 failed** — the other documented
  pre-existing failure passed this run (environment-dependent, as
  characterized by every prior task).
- `preflight`: all six derived-artifact checks green, including the backlog
  task-id sweep (3188 files, no duplicates, including this task's own 2 new
  filings).
- Full sequential xdist paired-baseline sweep (whole-wave span,
  `2372ea764` vs. `f42f75d98`+close, `Tests/UI -k "library" -p no:randomly -q
  -n 8 --dist worksteal`): **branch 372 failed/3932 passed (1699.84s,
  28:19) vs. baseline 365 failed/3934 passed (1398.41s, 23:18)** — both at
  or just past the documented ~330-371 historical backdrop (recipe §7),
  under this run's own recorded machine load (~22.7 at start, spiking to
  ~47.6 mid-run per direct `uptime` snapshots, 8-9+ concurrent `pytest`
  processes throughout). 356 shared, 9 baseline-unique (noise in the
  opposite direction, not investigated further per precedent — one of the
  9, `test_closeout_single_app_route_cycle`, is TASK-31422's own subject:
  it failed on the BASELINE and passed on the branch here, the OPPOSITE
  direction from Task 1's own observation, further corroborating that
  filing's premise that the failure is direction-independent flakiness),
  16 branch-unique. Combined single-process re-run of all 16: 12 passed
  cleanly (ordinary xdist noise); the remaining 4
  (`test_library_media_reader_traversal_t22207.py::
  test_focus_traversal_builds_zero_bodies_for_pass_through_rows`,
  `test_library_prompts_canvas.py::
  test_library_prompt_undo_refreshes_applied_page_and_preserves_basket`,
  `test_library_shell.py::
  test_library_starter_production_geometry_and_focus_order[size0]`,
  `test_library_shell.py::
  test_library_starter_production_geometry_and_focus_order[size1]`) each
  passed in TRUE isolation on the branch. Zero of the 16 touches a Skills
  file or Skills-owned code; several are Media/Notes/Prompts/Collections
  tests already named as pre-existing xdist noise by wave-1/2/3's own
  sweeps. **Zero unexplained branch-unique failures across the whole
  wave-4 span.**
- Probe run: recorded above and in the recipe, with the machine-load caveat.

## 7. Files changed

- `backlog/docs/library-decomposition-recipe.md`: §8's skills row rewritten
  to "complete" with the full summary; §19 gains a "Wave-4 close"
  subsection (pin trajectory, verification battery, whole-wave sweep, probe
  run, 3 lessons).
- `backlog/tasks/task-31421 - ....md`,
  `backlog/tasks/task-31422 - ....md`: new follow-up filings (§2 above).
- `tldw_chatbook/UI/Library_Modules/library_skills_state.py`: module
  docstring's stale future-tense paragraph rewritten to past tense
  (comment-only; zero logic changed).
- `.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/`:
  force-added for durable evidence (ledger + 3 reports + 5 review diffs +
  this report).

Commit: `6c71826b1`

## 8. Self-review

- **No production behavior changed.** Every edit in this task is
  documentation (recipe, two backlog filings, one ledger entry, this
  report) or a comment-only docstring correction
  (`library_skills_state.py`) — verified by `git diff --stat` before
  committing and by `ast.parse` + a direct module import after the edit.
- **All git-plumbing risk from the whole-wave sweep was handled explicitly,
  not assumed safe.** Running `git stash -u` / a scoped `git checkout
  2372ea764 -- tldw_chatbook Tests` / `git checkout HEAD -- ...` / `git
  stash pop` while a large recipe edit was mid-flight in this same session
  created a real risk of a lost or garbled edit (the recipe file's own
  `git status` "modified on disk" warning surfaced this mid-task). Handled
  by saving the in-flight edit as a standalone patch, reverting the file to
  match what the background script expected, and re-applying the patch
  (verified via `git apply --check`) only after the script's own stash pop
  had restored everything else — confirmed via `git status`/`git stash
  list` at each step rather than assumed.
- **The coordinator's mid-task correction was warranted and acted on.** An
  early stretch of this task relied on the Monitor tool's own promised
  notification-on-completion rather than actively re-checking the running
  sweep's output file inside the turn; the coordinator flagged this as the
  project's own recurring trap and it was corrected immediately (bounded
  polling loops inside single Bash calls, `bash -c 'for i in 1..N; do grep
  ...; sleep 10; done'`, re-invoked until the marker appeared) — recorded
  here rather than smoothed over, since the correction changed how the
  remainder of this task's own long-running verification was actually
  driven.
- **The probe's elevated numbers were reported honestly, not hidden or
  misclassified.** Every interaction came in slower than wave-2's own
  recorded band. The honest reading (ambient load, not regression) rests
  on two independent facts, both checked rather than assumed: this task's
  diff touches zero code on the probed path, and the machine's own load
  was directly measured (via `uptime`/`ps aux`) at the time of the run, not
  inferred after the fact.
- **The follow-up filings used the project's own documented ID-collision
  discipline, not the CLI's default.** `backlog task create`'s own
  auto-assignment was probed and confirmed unsafe (offered 31263 against a
  swept true max of 31420) before either task was filed by hand; both
  files were verified with `backlog task <id> --plain` to confirm the
  hand-authored two-item AC blocks rendered as independently-tickable
  criteria, not a comma-joined run-on (the documented `--ac` trap this
  task avoided entirely by never invoking that flag).
- **One acknowledged sequencing note, stated precisely rather than glossed
  over**: the combined wiring/characterization/support-layer/ratchet run,
  the full `Tests/Architecture/` run, the `-k "skill and library"` sweep,
  `Tests/Skills/`, and `preflight` were all run BEFORE the whole-wave
  xdist sweep's own stash/checkout round trip (deliberately, to avoid a
  concurrent git-checkout corrupting a running suite's file reads) — not
  after, and not re-run a second time once the round trip finished. This
  is sound, not a gap: this task's own doc-only diff (the only thing that
  round trip ever touched via the stash) does not change any test's
  behavior, and the fresh `_measure()` call plus `git status`/`git stash
  list` checks performed immediately after the stash pop independently
  confirm the production+test tree came back byte-identical to what those
  five suites had already exercised. A second re-run would have added
  confirmation, not new information; recorded here as a deliberate
  ordering choice made explicit, not an oversight discovered after the
  fact.
