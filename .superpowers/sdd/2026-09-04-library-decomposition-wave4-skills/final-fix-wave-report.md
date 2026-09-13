# Wave-4 final review — fix wave + dev merge

Branch: `refactor/library-decomp-wave4-skills-ingest`. Executes the
final-review dispatch's Part A (5 findings, one commit) and Part B (ledger
append, dev merge, verification, push).

## Part A — findings

### 1. Probe-caveat wording (Important-1)

`backlog/docs/library-decomposition-recipe.md` §19 and
`task-4-report.md` §1 both claimed the skills-cleanup move "touches zero
code" on the probed Media/Notes rail-switch path. Verified FALSE: diffed
every hunk of the cleanup commit (`ed4c29d45`) against the five methods on
that path (`compose_content`,
`_select_library_rail_row_after_source_admission`,
`_toggle_library_media_reader_pane`, `restore_state`,
`_persist_library_reader_preference`) using a line-range-overlap script,
then read each hunk directly — all are mechanical receiver swaps
(`self._library_skill(s)_<field>` -> `self._skills_state.<field>`), zero
control-flow or logic changes.

Rewrote both documents: "only mechanical receiver swaps ... nanosecond-
scale, cannot explain a ~3x max-gap delta"; added the load-independent
probe columns (recompose 0->0, full-updates identical row-for-row across
both waves) as the stronger no-structural-regression evidence; and
remarked the small mounts/nodes drift (media 163->173/113->115,
notes-switch-2nd 110->115) as attributable to ordinary Media/Notes feature
drift on this branch between wave-2 close (`09a5cadff`) and wave-4 close
(`ed4c29d45`) — verified via `git log --oneline 09a5cadff..ed4c29d45` (70
non-doc commits touching the relevant files, including several
`origin/dev` merges and named Media/Notes feature commits), not left as an
unverified guess.

Also fixed a third occurrence of the same false claim in §19's own
"Lessons" subsection (lesson 3, not named in the dispatch but the same
sentence), since leaving it uncorrected in the same document would have
been inconsistent with the finding.

### 2. Minor-3 — dead import

`LIBRARY_SKILLS_IMPORT_WORKER_GROUP` had zero in-file uses in
`library_skills_controller.py` (confirmed by grep before removing;
the screen, not the controller, is the real consumer). Removed.

### 3. Minor-5 — sixth-vs-seventh bypass-shape numbering

The controller docstring called the `getattr(self, "focused", None)`
escape "a SIXTH ... hazard, distinct in shape from" the bare-self-identity
findings; `task-2-report.md` §12a called the same fix "a SEVENTH ...
instance, a NEW shape" in its header, then contradicted that framing 90
lines later in its own §12b ("its close cousin"). Recipe §3's own
framing — one shape (sixth), the getattr escape as that shape's own
"close cousin" sub-case — was picked as canonical. Reworded the controller
docstring and task-2-report §12a's header + argument paragraph to match:
the escape is the sixth bypass shape's own case, the seventh instance
counted under it (6 identity-argument names + this one), not a
distinct/new shape. Recipe §3 itself needed no edit (already correctly
framed).

### 4. Minor-4 — dead CSS selector, filed as a task

Verified `#library-skill-allowed-tools` is genuinely dead: two
`@on(Input.Changed, "#library-skill-allowed-tools")` handlers exist
(screen delegator + controller) but no widget with that id is ever
composed anywhere (the tool picker moved to a `SelectionList`-based
chooser in `library_skills_canvas.py`). Filed **TASK-31423** with 2
independently-tickable ACs (handler removal + CSS removal).

### 5. Minor-7 — hazard-census promotion, filed as a task

Filed **TASK-31424** to promote the getattr-literal-resolves and
bare-self-identity hazard censuses into a standing `Tests/Architecture`
test over every `Library_Modules` controller (the getattr/`focused` hazard
survived a full green battery and was only caught by independent review —
exactly the gap a standing test would close). 2 independently-tickable ACs.

**Task-ID sweep**: true max across every remote ref, every local branch,
and every worktree = **31422** (this worktree's own already-committed
task-31421/31422). Verified with a throwaway CLI probe (offered 31423,
deleted before use, confirming no stale collision). Filed at 31423/31424
by hand, rendering verified via `backlog task <id> --plain`.

### Part A commit

`4bf1187f8` — `docs(library): wave-4 fix wave — probe caveat corrected,
dead import pruned, hazard-census follow-ups filed`.

Controller size ratchet re-pinned in the same commit (both the import
removal and the docstring reword landed in the same file): `3140 -> 3142`
(-1 line from the import, +3 from the reworded paragraph). Verified via
`ast.parse`, a direct module import, and
`Tests/Architecture/test_library_modules_size_ratchet.py` (29 passed).

## Part B — ledger, dev merge, verification

### Ledger

Appended, committed via `git add -f` (`282b229d5`):

> FINAL REVIEW (wave 4): MERGE-READY WITH CONDITIONS (fable); fix wave +
> dev merge executed; see task-4-report + this commit.

### Dev merge

`git fetch origin && git merge origin/dev` (106 commits since this
branch's merge-base). Exactly **one** conflict, as predicted:
`Docs/security/production-diagnostic-inventory.json`. Resolved via
`git checkout --theirs` + `scripts/check_persistent_diagnostic_inventory.py`
(read the drift rows first: only our own branch's skills-controller move —
5 diagnostic calls relocated `library_screen.py` ->
`library_skills_controller.py`, screen 109->104 — dev's copy simply
predates that move since the controller file doesn't exist on dev) then
`--write`. Re-verified clean after.

**Screen ratchet re-pin**: fresh `_measure()` on the merged tree:
`LibraryScreen` 41155/1295 -> **41574/1302** — exact match to the
reviewer's predicted ≈41574/1302. Re-pinned with a dated comment
attributing the growth to ordinary dev-side feature drift (not the skills
move, and not a conflict — `library_screen.py` merged cleanly).
`chat_screen.py`'s own row left untouched, as instructed.

**`5f030c8a0` routing check**: dev's own Skills-survives-source-outages
commit calls `self._library_skills_list_canvas_kwargs()` at
`library_screen.py:~12454`. Confirmed by grep this is the SAME one-line
delegator (`library_screen.py:18218`) our own `compose_content` call
(`:14636`) already used, forwarding to `LibrarySkillsController`.

**Flat-name + pruned-delegator greps**: reconstructed the 36-field
flat-name list programmatically (`skill_state_shim_attr` over every
`LibrarySkillsState` field) and word-boundary-grepped `library_screen.py`:
**1 hit**, the same expected historical explanatory comment task 3's
cleanup already documented (`_selected_skill_name` at line 41568) — zero
real reintroductions. `test_screen_delegates_skills_handlers` (which
guards the 16 pruned delegators' continued absence) passes.

### Verification

- Five wiring suites + 3 characterization files + support-layer surface +
  both size ratchets + `Tests/UI/test_library_skills_canvas.py`, combined:
  234 passed, 9 failed. 2 of the 9 are the documented pre-existing
  `chat_screen.py` ratchet rows.
- **The other 7** (`test_library_skill_row_class_matches_prompt_row_
  visual_parity`, `test_library_skills_header_filter_empty_have_css_
  blocks`, `test_library_skill_name_input_css_blocks_match_prompt_name_
  parity`, `test_library_skills_import_row_css_blocks_match_prompt_
  parity`, `test_library_skill_trust_setup_explanation_css_block_matches_
  review_files_parity`, `test_action_library_skill_back_honors_dirty_
  guard`, `test_skill_editor_production_geometry_contains_basic_and_
  advanced_workflows[size0]`) were NOT part of the predicted contract.
  Investigated rather than assumed: the 5 CSS-parity tests all assert
  Skills selectors exist in `tldw_chatbook/css/tldw_cli_modular.tcss`,
  which — verified by direct grep — has never carried them (0 occurrences
  of `.library-skill-row`, `#library-skill-name`, etc.). The 2 Pilot-driven
  tests fail on a focus/geometry assertion. Reproduced ALL 7 identically
  on the wave-4-close commit (`29039a6ad`), BEFORE this fix wave and BEFORE
  the dev merge, using an isolated `git worktree` + its own `uv venv`
  (per the worktree-test-invocation lesson). Confirmed pre-existing to the
  whole final-review session, not introduced by anything in this task —
  but never previously surfaced in any wave-4 report, because no prior
  task ran this test file in full. Left unfixed (out of this review's
  scope); named here rather than silently folded into "documented
  backdrop."
- Full `Tests/Skills/`: 537 passed, 2 failed
  (`test_import_real_superpowers_skills_lands_trust_pending`,
  `test_uninitialized_trust_shows_setup_state_and_bootstrap_enables_
  approve_flow`) — both are the two ALREADY-documented environment-
  dependent flakes in this suite (prior tasks each saw one of the two
  flip pass/fail; this run landed both red). Not new.
- Full `Tests/Architecture/`: 543 passed, 1 skipped, 16 failed — an EXACT
  match (same counts) to the wave-4-close task's own documented backdrop.
  Zero Library/Skills-scoped failures among the 16.
- `preflight`: all six derived-artifact checks green (CSS bundle sync,
  profile-owned-path census, diagnostic inventory — 574 owners post-merge,
  backlog task-id sweep — 3234 files no duplicates, chachanotes table
  allowlist, index plan pins).

### Merge commit

`2fc0003b8` — includes the diagnostic-inventory resolution, the screen
ratchet re-pin, and the full verification narrative (see commit message).

### Push

`git push origin refactor/library-decomp-wave4-skills-ingest` — see
final status contract for the outcome.

## Self-review

- Every quantitative claim in this report was independently re-derived
  (line-range-overlap script for the probe-caveat fix, a programmatic
  36-flat-name census, a real `_measure()` call, an actual isolated
  reproduction of the 7 unexpected canvas-test failures) rather than
  trusted from the dispatch's own predictions — all of which turned out
  accurate except the "only documented reds" scope for
  `test_library_skills_canvas.py`.
- The 7 newly-surfaced `test_library_skills_canvas.py` failures were
  investigated to a specific root cause and a specific pre-existing commit
  rather than being waved through as "probably fine" or silently added to
  the fix wave's scope (which the dispatch did not authorize touching).
- No production code changed beyond the two Part-A fixes (dead import,
  docstring reword) and whatever the `dev` merge itself brought in
  (reviewed via the wiring/flat-name/delegator checks above, not assumed
  safe because "the merge is just dev's own work").
