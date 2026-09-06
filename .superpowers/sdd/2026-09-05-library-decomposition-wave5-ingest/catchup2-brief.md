# Round-2 catch-up brief — merge origin/dev (72 commits, TASK-31521 screen reuse) into wave-5 branch

A `git merge origin/dev` is ALREADY IN PROGRESS in the worktree `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation` (branch `refactor/library-decomp-wave5-ingest`, MERGE_HEAD set, one conflicted file: `tldw_chatbook/UI/Screens/library_screen.py`, 3 hunks). Do not abort it; resolve, port, verify, commit. Merge-base is `93388ba69`. The controller coordinating you has ALREADY done the semantic analysis below — your job is precise execution plus independent verification of each claim as you apply it.

This is the second dev-reconciliation of this PR (#2424). The first (commit `897ab81f7`) set every precedent you need: delegators stay, dev's edits to moved bodies get ported into `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py`, screen-owned fields the ported code touches get accessor constructor bindings (worked examples IN THE CONTROLLER: `library_ingest_analyze_outcomes_accessor` getter-only property, and in `library_skills_controller.py` the `..._accessor` + `set_...` getter/setter pairs).

## Environment
- `.venv/bin/python`; UI tests with `-p no:randomly`; `timeout` unavailable (`perl -e 'alarm N; exec @ARGV'`).
- Do NOT push. Do NOT touch `progress.md`. No subagents. Notifications never reach you — poll output files directly.

## The analysis (verify each item as you execute)

Dev's 72 commits are mostly TASK-31521 (Library route becomes reusable: suspend/resume instead of unmount) plus chunking-lab buttons and media-focus work. Ingest-relevant impact:

**Dev edited TWO moved bodies** (confirmed by AST comparison of every controller-resident method name between `93388ba69` and `origin/dev`):
1. `_handle_library_ingest_registry_changed` — two edits: (a) the `LIBRARY_ROW_INGEST_MEDIA` block gains a suspended-gate (`if self._library_screen_suspended: self._library_ingest_suspended_activity = True / else: <original dynamic-regions + shortcuts block>`); (b) `if grew:` becomes `if grew and not self._library_screen_suspended:` (+ its 3-line comment), and the landing-attention gate gains `and not self._library_screen_suspended`.
2. `_handle_library_ingest_progress_changed` — the entry guard is restructured: `is_attached` check first, then a suspended-gate block (sets `_library_ingest_suspended_activity = True` when the ingest row is selected, then returns), then the row check.

Extract dev's exact versions with `git show origin/dev:tldw_chatbook/UI/Screens/library_screen.py` and port the edits into the controller bodies **verbatim except** field access goes through the new accessors (below). After porting, AST-compare each controller body against dev's screen body the way the first reconciliation did: they must be identical modulo the accessor substitutions.

**Dev added 4 new flat screen `__init__` fields** (all NEW since merge-base — verified by diffing; everything else on the dev side of the init hunk is a moved `LibraryIngestState` field): `_library_source_snapshot_timeout_timer`, `_library_screen_suspended`, `_library_ingest_suspended_activity`, `_library_visit_entered`.

**Dev's new `on_screen_suspend`** (auto-merged in, NOT conflicted — fix it anyway) stops timers via a string loop `for attr in (...): getattr(self, attr, None)` whose tuple includes `"_library_ingest_path_debounce_timer"` — a field this branch moved into `LibraryIngestState` and whose flat shim was deleted at cleanup. On this branch that `getattr` silently returns None and dev's timer-stop fix NO-OPS for the ingest timer. This is dev's new screen-resident method (not a moved body), so edit it: remove that one name from the tuple and add an explicit stop of `self._ingest_state.path_debounce_timer` (same stop+None pattern), with a one-line comment that the field lives in ingest state.

## Resolution spec

### Hunk 1 (init block, ~line 3460)
HEAD side is empty (the 12 moved fields were extracted). Dev side mixes moved fields with the 4 new ones. Resolution: keep ONLY the 4 new fields with their exact comment blocks from dev (`git diff 93388ba69 origin/dev` on the file shows precisely which lines are additions — the `_library_ingest_path_debounce_timer` line and every other moved-field line are context, not additions; drop them all). Order the kept lines as dev has them.

### Hunks 2 and 3 (~20533, ~20755)
Keep HEAD's one-line delegators (`_handle_library_ingest_registry_changed`, `_handle_library_ingest_progress_changed`). Port dev's edits per above.

### Accessor bindings (constructor arity 38 → 40 keyword-only — measure and state the real numbers; the first reconciliation's brief got this wrong by not measuring)
- `library_screen_suspended_accessor` — getter-only (ported code only reads it). `_library_screen_suspended` is screen-wide lifecycle state (dev gates media/notes surfaces on it too), so an accessor is its correct PERMANENT shape, like `library_canvas_resync_pending_accessor`.
- `library_ingest_suspended_activity_accessor` + `set_library_ingest_suspended_activity` — getter/setter pair (ported code sets it True), mirroring the skills controller's `..._accessor`/`set_...` precedent. This field IS ingest-exclusive — an interim bridge like analyze_outcomes.
- Wire all THREE construction sites (screen, `wire_bypass_ingest_controller`, the inline-consent local helper — grep `LibraryIngestController(`). Update the controller module docstring's group counts and its divergence paragraph (now three ported-edit divergences; name TASK-31521 alongside task-28007).
- Seed the bypass helpers: `object.__new__(LibraryScreen)` screens need `_library_screen_suspended = False` and `_library_ingest_suspended_activity = False` (match dev's `__init__` defaults exactly), same two-helper pattern as the analyze_outcomes seeds.

### Census (post-resolution, before committing)
Sweep the MERGED tree (`tldw_chatbook/**` + `Tests/**`, excluding the controller/state modules) for the 20 `LibraryIngestState` field names PLUS `_library_ingest_analyze_outcomes`, in BOTH forms: attribute (`._library_ingest_<name>` / `self._library_ingest_<name>`) AND quoted string (`"_library_ingest_<name>"` / `'_library_ingest_<name>'`) — the on_screen_suspend string loop proves the attribute grep alone is blind. Every live hit gets retargeted to `_ingest_state` (or reported if non-mechanical). Also check dev's new/edited tests (auto-merged `Tests/UI/test_library_shell.py`, `test_library_ingest_canvas.py`, plus any new TASK-31521 test files) for flat uses on bypass screens.

### TASK-31651 update
Append to `backlog/tasks/task-31651*.md`: `_library_ingest_suspended_activity` is a second interim accessor-bridged field to fold into `LibraryIngestState` (with its `set_` binding to retire), and note the `on_screen_suspend` explicit state-object timer stop as a seam to simplify then. One or two ACs, ticked-off-able.

### Re-pins (same merge commit)
Fresh `_measure()` on both ratchet rows (`Tests/Architecture/test_screen_size_ratchet.py` LibraryScreen — will RISE with dev's new methods; `Tests/Architecture/test_library_modules_size_ratchet.py` ingest row — rises with ported edits). Pins exact, comment per the files' established dev-merge convention. NOTE: dev may also have moved its OWN pins for other rows — take dev's side for every row that isn't ours.

### Verification (focused — ruled by the controller: NO third full paired-baseline sweep this round; the marginal surface is two gate-edits + accessors, covered by the suites below)
1. 6 wiring suites, 4 ingest characterization files, inline-consent, both ratchets, recompose census guard, support-layer surface.
2. Dev's TASK-31521 test files/additions (find them: `git log --name-only 93388ba69..origin/dev -- Tests/ | sort -u | grep -i "test_"` filtered to library/screen-reuse names) — run each touched library test file.
3. `./scripts/preflight.sh` all green.
4. AST identity check of both ported bodies vs dev's screen bodies (modulo accessor substitution) — include the transcript in the report.

Commit the resolved merge (message: what was ported, the two new accessors with REAL arity numbers, the string-loop retarget, census result), then the TASK-31651 edit + bypass seeds either in the merge commit or one follow-up commit — your call, state it.

## Report
Full record to `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/catchup2-report.md`: per-hunk resolution, port verification transcripts, census hits and dispositions, seed additions, pin values with measures, battery numbers. Return ONLY: STATUS, commit hashes, one-line test summary, concerns.
