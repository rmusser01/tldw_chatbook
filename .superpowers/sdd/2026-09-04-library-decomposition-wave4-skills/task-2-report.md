# Wave-4 Task 2 — Skills controller move (series 2/3)

Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§18. Plan:
`Docs/superpowers/plans/2026-09-04-library-decomposition-wave4-skills.md`.
Templates: `library_rag_search_controller.py` (newest, largest prior
combined series) + `Tests/Architecture/test_library_search_rag_wiring.py`.
State layer (Task 1): `LibrarySkillsState` (36 fields, three-way prefix
mapping via `skill_state_shim_attr()`), `self._skills_state`, screen shims.

## 1. Cluster enumeration (re-derived fresh, not carried over)

`ast` walk of `LibraryScreen` for method names containing `"skill"`
(case-insensitive): **133 raw `FunctionDef` matches, 127 unique names** —
identical to Task 1's own census. The 6-match gap is SIX
`@property`/`@x.setter` pairs (each of the 6 names below contributes one
getter def + one setter def = 2 raw matches, 1 unique name — Task 1's own
report undercounted this as "three pairs" despite listing all six names;
corrected here), all thin one/two-line projections onto
`_library_skill_import_coordinator` (the existing, untouched
`LibrarySkillImportCoordinator`).

A companion `ast` sweep checked for the conversations exemplar's own
"startswith enumeration trap" in its METHOD form (a non-"skill"-named
method called exclusively by cluster members, which a substring census
would miss): **zero** found — every non-skill-named method referenced by
the cluster is either shared shell/plumbing or belongs to another
subsystem's own methods, confirmed by a reverse call-graph walk over ALL
`LibraryScreen` methods, not just the 127.

## 2. Single-vs-split decision — ONE controller, by call-graph connected components

The plan named a possible editor/trust vs. browse/list split seam as
worth checking at this scale (121 movable candidates after the 6
merely-delegate exclusions, the largest single-subsystem cluster in the
program, nearly double collections' own 64). Method: union-find over
every `self.<name>(...)` call among the 121 candidates.

**Result: one connected component of 107 members, plus 13
singleton/pair components with zero intra-cluster calls** (reached only
via `@on`/other external dispatch — isolated because they are leaf nodes,
not because they belong to a second cluster). A secondary, hand-labelled
heuristic bucket pass (editor/trust/browse/detail/import/other) showed
the same thing at a coarser grain: dense cross-calling in every
direction (editor↔trust, editor↔detail, trust↔detail, browse↔editor —
no bucket pair with zero edges). There is no subset of the cluster that
only ever calls within itself; the plan's own hypothetical seam does not
hold.

**Decision: ONE combined `LibrarySkillsController`**, matching the
plan's own "when unsure, one controller" default and reconfirming the
search+RAG series' own identical resolution at a comparable (50-candidate)
scale.

## 3. Exclusion census — 41 of 127, not moved (86 move)

### 3a. 6 merely-delegate-to-existing-controller properties
`_library_skills_import_open/_path/_status/_review_name/_in_flight/_generation`
— each a `@property`/`@x.setter` pair whose entire body is
`self._library_skill_import_coordinator.snapshot.<x>` /
`.update(<x>=value)`. Per the plan's own named exclusion class
(delegation to the pre-existing, untouched coordinator).

### 3b. 27 unbound-fake-self test-bypass exclusions
A repo-wide grep for `LibraryScreen.<name>(` across all FOUR test roots
(`Tests/UI`, `Tests/Library`, `Tests/Live`, `Tests/Skills`), for every one
of the 121 candidates, found 27 names called with a bare
`SimpleNamespace`/hand-built fake (confirmed by reading each call site's
own fixture construction, never inferred from the call shape alone —
never a real, constructed `LibraryScreen`) standing in for `self`:
`_build_library_skills_state`, `handle_library_skills_sort`,
`handle_library_skills_sort_choice`, `handle_library_skills_filter`,
`handle_library_skill_row`, `_call_library_skill_trust_service`,
`_approve_library_skill_trust`, `handle_library_skill_delete`,
`handle_library_skill_delete_confirm`, `handle_library_skill_delete_cancel`,
`handle_library_skill_trust_review`, `_reset_library_skill_editor_state`,
`handle_library_skills_import_review`, `handle_library_skills_import_browse_folder`,
`_library_skill_editor_active`, `_library_skill_save_available`,
`_begin_library_skill_save`, `action_library_skill_save`,
`_exit_library_skill_editor_guarded`, `action_library_skill_back`,
`handle_library_skills_import_browse`, `handle_library_skills_import_cancel`,
`handle_library_skills_trust_action`, `_open_first_blocked_skill`,
`handle_library_skills_trust_reset_request`,
`_apply_library_skills_import_status` (`Tests/UI/test_library_canvas_scoped_sync.py`),
`_present_library_skills_import_choice_if_needed` (`Tests/Skills/test_skills_import.py`
— the wave's own fourth-root trap, confirmed live). This roughly triples
export's own prior 9-of-51 record, confirming §12's forward note that this
shape scales with how much a subsystem's test style favors unbound
`SimpleNamespace`/`Mock` unit-style calls.

### 3c. 1 instance-attribute-monkeypatch exclusion
`_request_library_skills_browse` — `Tests/UI/test_library_skills_canvas.py`
patches `screen._request_library_skills_browse = lambda ...` on one REAL,
directly-constructed `LibraryScreen` instance, expecting the mover
`_refresh_library_skills_after_committed_mutation` to observe the patch.
`_call_library_skill_trust_service` independently matches BOTH this shape
(`Tests/UI/test_library_skills_reader.py:286`, a real Pilot-mounted
screen) and shape 3b (also called unbound with a fake elsewhere) — doubly
confirmed, not a coincidence. (`_flush_library_skill_save`, patched the
same instance-attribute way in two other test files, MOVES normally: its
only external callers — `flush_pending_work`/`_select_library_rail_row`
— are screen-resident and never move, and Python's instance-attribute
shadowing means the patch is observed regardless of where the delegator's
target body lives.)

### 3d. 1 module-globals-coupling exclusion
`_persist_library_skill_editor_mode` reads the bare name
`save_setting_to_cli_config` (an ordinary module-level import in
`library_screen.py`, resolved against the DEFINING module's `__globals__`
at call time — the search+RAG series' own `_load_library_search_history`/
`get_cli_setting` precedent, reproduced on a different free name).
`Tests/UI/test_library_skills_canvas.py` (~line 1975) patches
`library_screen_module.save_setting_to_cli_config` and presses the real
editor-mode toggle button through a full Pilot session — confirmed by
reading the test, not assumed from the census alone.

### 3e. 6 bare-self-as-identity-argument hazard exclusions — a NEW bypass shape, found in two forms
This series adds a new category to the recipe's own catalogue: passing
bare `self` not merely as an attribute-lookup receiver (safe, duck-typed,
per every prior controller's `_sync_library_canvas`-forwarding precedent)
but as an argument to code that compares it for **identity** against the
real screen — something a controller standing in for `self` can never
satisfy.

- **Form A** (found by static analysis before any code moved):
  `_refresh_library_skills_trust_posture` calls `self.workers.
  cancel_group(self, "library_skills_trust_posture")`. Textual's own
  `WorkerManager.cancel_group` (read via `inspect.getsource`, not
  assumed) filters `worker.node == node` by identity; since `run_worker`
  always forwards to `self._screen.run_worker(...)`, a worker's `.node`
  is always the SCREEN, never the controller. A verbatim move would make
  this cancellation a silent, permanent no-op.
- **Form B** (found the hard way — by running the battery after a first
  draft moved 4 names): `_library_screen_is_current(screen)`
  (`screen_helpers.py`) does `current_screen = getattr(screen.app,
  "screen", screen); return current_screen is screen`. FOUR candidates
  called it as a bare `self` forward: `handle_library_skills_import`,
  `handle_library_skills_import_path_changed`,
  `handle_library_skills_import_retry`, `_start_library_skills_import`.
  Moved verbatim, every call permanently evaluates `False`
  (`real_screen is controller` can never be true), silently no-opping
  the entire Skills import feature.
- **Form C** (found by the SAME method, on a SECOND draft, via the
  Tests/Skills fourth-root suite): `_present_library_skills_import_
  snapshot` guards its whole body — the terminal-status DOM update and
  the queued candidate-choice modal — behind `self.is_mounted and
  self.app.screen is self and ...`. Reached only externally, from the
  pre-existing `LibrarySkillImportCoordinator._settle` via
  `getattr(runtime_app.screen, "_present_library_skills_import_
  snapshot", None)` — always the real screen — so the delegator receives
  the call correctly, but the moved body's own `self.app.screen is self`
  is permanently `False` once `self` is the controller.

Both regressions were **real, not theoretical**, and were caught only by
the verification battery, never by the wiring/ratchet tests (a
same-name-forwarding regex and a byte-for-byte diff cannot see a semantic
identity bug):

- Form B's first draft: `Tests/UI/test_library_skills_reader.py::
  test_skills_mount_three_retained_roles_and_default_to_overview` and
  `Tests/UI/test_library_per_click_recompose_t21116.py::
  test_skills_import_open_and_cancel_are_canvas_scoped` both failed
  (each timing out its own 30s DOM-mount wait, having pressed the real
  `#library-skills-import` button and never seeing
  `#library-skills-import-path` mount). Confirmed as a genuine regression
  via a paired baseline: both pass in under 3s combined against the
  pre-move tree (`git stash`) and fail against the four-moved draft.
- Form C's second draft: 8 `Tests/Skills/` tests failed (7 in
  `test_skills_import.py`, 1 in `test_skills_library_flow.py`), all
  confirmed genuine via the same paired-baseline method (all 8 pass on
  the pre-move tree, fail with `_present_library_skills_import_snapshot`
  moved).

All five Form-B/C names reverted to screen-resident, full-bodied,
untouched. Zero further hazards of this shape exist in the FINAL
86-mover set — confirmed by a repeat, method-scoped AST census checking
every `ast.Compare` with `Is`/`IsNot`/`Eq`/`NotEq`/`In`/`NotIn` against a
bare `Name(id="self")` on either side, AND every free-function or
same-class-method call passing bare `self` as an argument, over the
final controller file.

**41 total exclusions: 6 + 27 + 1 + 1 + 6 = 41. 127 − 41 = 86 movers.**

## 4. Dynamic-dispatch census

Checked the 86 movers' own bodies for `getattr`/`setattr` with an
f-string or dict-literal argument, and the `dict.get(...)` → variable →
`setattr`/`getattr` two-step shape (collections series' own fix-round
lesson): **zero hits of either shape**. Every `getattr(self, "<literal>",
default)` call found is a defensive read of a constant, hard-coded
attribute/service name (e.g. `getattr(self.app_instance,
"skills_scope_service", None)`), not a runtime-computed name — a plain
grep for the literal string would find every one, so none constitute the
hazardous shape the recipe's census exists to catch. `_call_library_
skill_trust_service`'s own `getattr(trust_service, method_name, None)`
dispatches on an EXTERNAL service object's own API surface (a string
parameter, not a screen/controller attribute name) — unrelated to this
move's class boundary. The shared, multi-subsystem dynamic dispatchers
Task 1 already found and handled at the FIELD level
(`_toggle_library_media_reader_pane`, `_replace_library_reader_
preference`/`_persist_library_reader_preference`) are screen-resident
shell code, never move, and are unaffected by this task (Task 1's own
field-level accommodation already covers their skills-field access
post-move).

## 5. Byte-for-byte verification — method + result

**Method**: an `ast`-driven extraction script pulled each of the 86
movers' exact source segment (decorators through body) from the
ORIGINAL, pre-move `library_screen.py` (using the file's own line
offsets, never hand-retyped), keyed by name. A second, independent script
re-parsed the FINISHED `library_skills_controller.py`, re-extracted each
same-named method's segment the same way, and asserted string equality
per name. Re-run after every fix round (91→87→86 movers) against the
correspondingly-updated expected set.

**Result**: 0 mismatches, 0 missing, for all three iterations of the
mover set. The controller module also imports and constructs cleanly
(`LibraryScreen(_build_test_app())`), and two ad hoc runtime probes
(`screen._notify_skill_dirty_veto()`, `screen._library_skills_canvas_
kwargs()` — the latter exercising the excluded `_build_library_skills_
state` dependency) executed without exception before any test suite ran,
confirming the dependency-injection chain works at runtime, not just
statically.

## 6. Free-name walk

A script collected every `ast.Name` Load inside each of the 86 movers not
bound by a parameter/local/comprehension/lambda/nested-def, then checked
it resolves to either a builtin, `self`, or a module-level import/
assignment in the NEW controller module. Cross-checked the resulting
free-name list against `library_screen.py`'s own import table to build
the controller's complete import block (57 names, spanning `typing`,
`collections.abc`, `pathlib`, `dataclasses`, `asyncio`, `loguru`,
`textual`/`textual.widgets`/`textual.css.query`/`textual.widget`, five
Library/Widgets/Screens modules, and two sibling `Library_Modules` files).

**Result**: one apparent unresolved pair (`BuiltinToolProvider`,
`LocalToolProvider` in `_build_library_skill_tool_catalog`) — confirmed a
false positive: both are imported LOCALLY inside that method's own body
(`from ...Agents.tool_catalog import BuiltinToolProvider`), moved
byte-for-byte with it. Zero genuine unresolved names.

## 7. Bind classification (binding kinds, per recipe §1)

**Framework services (9, live-read `@property`)**: `app`, `app_instance`,
`call_after_refresh`, `is_mounted`, `is_running`, `query`, `query_one`,
`refresh`, `run_worker`. `is_running`/`app`/`refresh`/`call_after_
refresh`/`query`/`query_one` exist partly because 15 movers forward
`self` into the shared `_sync_library_canvas(screen, "skills", ...)`
dispatcher, whose own kind-independent top/exception-handler code reads/
writes them (the RAG controller's own documented reason, reconfirmed
here by reading `canvas_sync.py`'s actual `"skills"` branch, not assumed
from its signature).

**General shell helpers (8, group a)**: `_run_library_service_call`,
`_sanitize_media_field`, `_sanitize_note_content`, `_refresh_local_
source_snapshot` (one of recipe §3's four PERMANENTLY screen-routed
names), `_library_entry_route_key`, `_library_entry_reconcile_is_
current`, `_capture_library_entry_focus`, `_restore_library_entry_focus`.

**Shared shell state (group b)**: read-only —
`_library_snapshot_state_generation`, `_library_entry_reconcile_dirty`,
`_library_entry_reconcile_pending`, `_library_canvas_projection_depth`
(the last bound purely because `_sync_library_canvas`'s shared preamble
touches it, mirroring the RAG controller's identical pair — no skills
mover body references it directly). Read+write —
`_library_canvas_resync_pending` (same reason), `_library_selected_row_
id` (written once, by `_open_library_skill_editor_for_review`).

**WIRING accessor pairs (group c, 2)**: `_library_skill_import_
coordinator`, `_library_skills_browse_controller` — read+write,
mirroring the collections controller's own `_library_collections_
capture_controller` precedent exactly.

**Read-only accessors for the 6 merely-delegate properties (group d)**:
5 get-only (`_library_skills_import_in_flight`, `_open`, `_path`,
`_review_name`, `_status`) + 1 get+set (`_library_skills_import_
generation`, written by `handle_library_skills_import_path_changed`).

**Named late-binding callables for the exclusions a mover still calls
(group e, 10)**: `_approve_library_skill_trust`, `_begin_library_skill_
save`, `_build_library_skills_state`, `_call_library_skill_trust_
service`, `_exit_library_skill_editor_guarded`, `_persist_library_skill_
editor_mode`, `_refresh_library_skills_trust_posture`, `_request_
library_skills_browse`, `_reset_library_skill_editor_state`, `_start_
library_skills_import` — each a `lambda` re-reading `screen.<name>` at
CALL time, exactly why a test's `monkeypatch.setattr(screen, "<name>",
...)` keeps working. Four more exclusion names have zero mover callers
and need no binding at all (`handle_library_skills_import`,
`_path_changed`, `_retry`, `_present_library_skills_import_snapshot`).

Constructor: 40 parameters total (`self`, `screen`, `skills_state_
accessor`, plus 37 named dependencies across the five groups above).

## 8. Verification battery

- **Wiring RED→GREEN**: RED confirmed via `git stash` of the controller +
  screen edits — the 4 new controller-level tests fail (module/attribute
  not found), the 5 pre-existing Task-1 state tests keep passing
  unaffected. GREEN confirmed after each of the two fix rounds: all 9
  tests in `test_library_skills_wiring.py` pass on the final tree.
- **Both size guards + census + support-layer**: `test_screen_size_
  ratchet.py` + `test_library_modules_size_ratchet.py` + `test_library_
  support_layer_surface.py` + all 5 wiring suites (conversations/export/
  collections/search+RAG/skills): **70 passed**, 2 failed — both the
  documented pre-existing `chat_screen.py` ratchet rows (recipe §7's own
  standing list), unrelated to this diff.
- **`Tests/Architecture/` full run**: 543 passed, 1 skipped, 17 failed —
  all 17 match the SAME categories recipe §7/Task 1 already document as
  pre-existing, unrelated-file churn (Console realtime/review-selection
  boundary, console wave6 closeout/inventory ×3, default-timeout-session-
  guard, persistent-diagnostic-inventory ×3, chat_screen ratchet ×2,
  timer-path-static-update-inventory ×3, worker-exclusive-group-inventory
  ×2) — zero overlap with any file this task touched (confirmed by file
  path; not re-derived via a fresh stash since these exact categories are
  independently re-confirmed pre-existing by every prior task in this
  wave, per §7's own "check this list before re-proving" guidance).
- **`-k "skill and library"` sweep** (`Tests/UI`+`Tests/Library`, single
  process, final tree): **10 failed, 271 passed** — every one of the 10
  matches Task 1's own already-documented pre-existing set exactly (CSS-
  block/geometry-parity tests, `test_action_library_skill_back_honors_
  dirty_guard`, the command-palette test, `test_shadow_name_set_stays_
  in_sync_with_real_sources`, `test_skills_route_lands_on_library_with_
  skills_row_selected`). One flip from the FIRST post-fix run (11→10):
  `test_library_skills_manual_items_priority_survives_compact_layout_
  sync` passed this time — Task 1 already characterized this exact name
  as order-dependent xdist-adjacent noise, not investigated further.
  **This sweep also directly evidences the two regression-and-fix
  rounds**: the pre-fix run (4 hazard exclusions applied, Form C not yet
  found) showed **12 failed** including 2 genuinely new names
  (`test_skills_import_open_and_cancel_are_canvas_scoped`,
  `test_skills_mount_three_retained_roles_and_default_to_overview`);
  after excluding Form B's four names, those 2 dropped out (12→11,
  matching Task 1 exactly); the final run (Form C also excluded) is
  clean at 10/271.
- **`Tests/Skills/` full run** (fourth root): **537 passed, 2 failed** —
  matches Task 1's own documented baseline (`test_import_real_
  superpowers_skills_lands_trust_pending`, environment-dependent;
  `test_uninitialized_trust_shows_setup_state_and_bootstrap_enables_
  approve_flow`, confirmed pre-existing via a direct `git stash`
  baseline run earlier in this task). **This run is the one that
  originally surfaced Form C**: before that exclusion, the SAME command
  showed 9 failed (7 more in `test_skills_import.py`, all import-flow
  tests asserting a terminal status line that never updated), all
  confirmed genuine regressions via paired baseline, all resolved by the
  Form-C exclusion.
- **Full sequential xdist paired-baseline sweep** (`Tests/UI -k
  "library" -p no:randomly -q -n 8 --dist worksteal`, branch then a
  `git stash`-pristine baseline of the final pre-task tree — both run
  IN-PLACE in this worktree's own venv, sequentially, ~30 min each, per
  §7's own "concurrent runs amplify flakiness" lesson): **branch 370
  failed/3933 passed (1790.7s) vs. baseline 371 failed/3932 passed
  (1788.8s)**, both inside the documented ~330–371 historical backdrop.
  358 shared, 13 baseline-unique (noise in the opposite direction, not
  investigated further per precedent), **12 branch-unique**: 10 passed
  cleanly on a combined single-process re-run (ordinary xdist noise);
  `test_library_notes_reader.py::test_wide_editor_deep_link_keeps_
  reader_navigation_and_local_back` is already on the recipe's own §7
  documented pre-existing list (wave-3 task 5's own sweep); `test_
  closeout_single_app_route_cycle` is the SAME test Task 1's own report
  investigated in depth (traced to the 'collections' destination step,
  unrelated to any of this task's 86 moved methods or their construction
  ordering) and classified as pre-existing timing-sensitive flakiness
  under machine load — re-confirmed here passing on a fresh isolated
  baseline run, consistent with intermittent flakiness rather than a
  deterministic regression. **Zero unexplained branch-unique failures.**
- **preflight**: `./scripts/preflight.sh` — one real, expected finding:
  the production-diagnostic-inventory check flagged 5 `logger.warning`/
  `logger.opt(...).warning` call sites that moved from `library_screen.py`
  into the new controller. Verified via `check_persistent_diagnostic_
  inventory.py --statements ... --since <base>` that all 5 added/removed
  pairs share IDENTICAL digests (pure relocation, zero reworded/new
  logging, nothing newly interpolating user content or secrets) before
  running `--write`. Re-ran preflight after: all six checks pass clean.

## 9. Fresh pins

- **Screen** (`test_screen_size_ratchet.py`): `43179/1311 -> 41247/1311`
  (methods unchanged — pure move, 86 `FunctionDef`s out, 86 one-line
  delegators in).
- **Controller** (`test_library_modules_size_ratchet.py`, born-governed):
  `library_skills_controller.py` pinned at its exact measured line count,
  through two fix-round re-measurements: `3181 -> 3113 -> 3099` (first
  fix round reverted 4 Form-B methods; second reverted the 1 Form-C
  method and 3 now-orphaned constructor dependencies).

## 10. Files changed

- `tldw_chatbook/UI/Library_Modules/library_skills_controller.py` (new):
  `LibrarySkillsController`, 86 moved methods, full binding-kind
  constructor, generated state-shim loop (mirrors the screen's own Task-1
  shim, sharing `skill_state_shim_attr`).
- `tldw_chatbook/UI/Screens/library_screen.py`: +1 import
  (`LibrarySkillsController`); `__init__` gains `self._skills_
  controller = LibrarySkillsController(...)` right after `self._rag_
  search_controller`; 86 method bodies replaced with one-line delegators
  (5 exclusion names — `handle_library_skills_import`, `_path_changed`,
  `_retry`, `_start_library_skills_import`, `_present_library_skills_
  import_snapshot` — kept full-bodied, byte-for-byte identical to their
  pre-move originals).
- `Tests/Architecture/test_library_skills_wiring.py`: 4 new controller-PR
  tests (cluster ownership, same-name delegator forwarding, staticmethod
  class-forwarding, controller-side state-field property coverage) added
  to Task 1's own file; the 5 pre-existing state-PR tests untouched.
- `Tests/Architecture/test_screen_size_ratchet.py`,
  `Tests/Architecture/test_library_modules_size_ratchet.py`: `_BUDGETS`
  rows re-pinned, dated comments added.
- `Docs/security/production-diagnostic-inventory.json`: regenerated
  (pure relocation of 5 pre-existing log statements, verified before
  writing).
- `.git-blame-ignore-revs`: the move commit's hash appended.

Commits: `5ecf223d4` (RED — wiring test additions, screen/controller
untouched), `60857a2be` (GREEN — the move, at its final 86-mover/
41-exclusion state), `679a90d1b` (blame-ignore).

## 11. Self-review

- **The two regression rounds are the honest headline of this task, not
  a footnote.** A first working-tree draft moved 91 candidates (missing
  4 Form-B exclusions); a second moved 87 (missing the 1 Form-C
  exclusion). Neither draft was ever committed — both were caught and
  fixed inside this same session, before the GREEN commit, by running
  the verification battery the task brief required rather than trusting
  the wiring/ratchet tests' own green result. Both fixes are documented
  in full in the controller's own module docstring (exclusion 5, Forms
  A/B/C) and in this report, not smoothed into a clean-looking single
  pass.
- **The identity-comparison hazard shape is new to the recipe and likely
  to recur.** Every subsequent subsystem's controller-PR should add "grep
  for bare `self` passed to a function/method that compares it — `is`,
  `==`, `in` — against another object" to its own hazard sweep, not just
  the established `@work`/monkeypatch/module-globals shapes. The
  mechanical AST check this task wrote (`ast.Compare` with `Is`/`IsNot`/
  `Eq`/`NotEq`/`In`/`NotIn` against a bare `Name(id="self")`, plus every
  free-function/same-class-method call passing bare `self`) is cheap to
  re-run and should be run BEFORE the first move draft, not after a test
  failure, on the next subsystem.
- **Single-controller decision was verified twice**, independently
  (connected-components over the 121 candidates; a coarser heuristic
  bucket pass), matching the search+RAG series' own "confirm the
  field-level finding at the method level too" discipline rather than
  trusting either alone.
- **Byte-for-byte and free-name verification were both done by tooling**,
  not manual read-through, appropriate for a cluster this size (86
  methods, ~2,000 lines) per the collections series' own "write the
  extraction and verification as scripts" lesson for clusters exceeding
  ~40–50 methods.
- **One acknowledged gap**: the full `Tests/Architecture/` 17
  pre-existing failures were confirmed unrelated by file-path
  non-overlap with this task's diff and by citing prior tasks' own
  repeated `git stash` confirmations of the same categories, not by a
  fresh `git stash` re-derivation in this task specifically — consistent
  with recipe §7's own "check this list before spending time re-proving"
  guidance, but recorded here as a lighter-weight evidence path than the
  stash-baseline method used everywhere else in this report.

## 12. Post-landing review fix round

A coordinator review of the landed commits found 1 CRITICAL + 1 IMPORTANT
+ 3 minor findings, all addressed in this fix round.

### 12a. CRITICAL — `focused` unbound on the controller (a SEVENTH bare-self
hazard instance, a NEW shape: unbound-attribute escape)

`_sync_library_skills_browse_result` reads `focused = getattr(self,
"focused", None)`. `LibrarySkillsController` had no `focused` property --
unlike every OTHER framework service this controller (and every sibling
controller in the file, e.g. `library_rag_search_controller.py:571-573`,
`library_conversations_controller.py:544-546`) binds, this one was
missed. The `getattr` default silently returned `None` on every call, no
exception anywhere, degrading two real behaviors permanently: the
live-focus override (`if isinstance(focused, (Button, Input)) and ...:
focus_identity = live_focus_id`) and the live-caret cursor-position read
(a deliberate fix, commit `8027e99f0`). User-visible: focus dropped on
every committed-mutation refresh (`focus_identity=None` callers at this
controller's own line ~1160 and the screen's `restore_state`/`_select_
library_rail_row_after_source_admission` call sites).

**This is a DISTINCT shape from exclusion 5's Forms A/B/C** (all three of
which are bare `self` passed to something that does an IDENTITY
comparison it can never satisfy). `getattr(self, "focused", None)` is not
an identity comparison at all -- it is a plain attribute read with a
silent default, invisible to BOTH the recipe's own `self.<attr>`
`ast.Attribute` census (the name never appears as a literal expression)
and to any exception-based detection (no `AttributeError` is ever
raised). A dedicated re-scan of the whole moved-body source for `getattr(
self, "<literal-string>", ...)` calls (not just `self.<attr>` accesses)
found exactly this one instance in the final 86-mover set.

**Fix**: added the same live-read `@property` every sibling controller
already carries:
```python
@property
def focused(self) -> Any:
    return self._screen.focused
```

**Covering test**: `Tests/UI/test_library_skills_canvas.py::
test_committed_mutation_refresh_with_no_focus_identity_restores_live_
focus`. Two false starts before landing a reliable signal, both
documented in the test's own docstring and worth recording here because
they generalize:

1. **A bare `.has_focus` assertion is not discriminating.** Empirically
   confirmed: Textual's own generic "focus the rebuilt canvas's first
   focusable descendant" behavior lands on the SAME filter Input whether
   or not `focused` is bound -- an unrelated framework fallback that
   happens to coincide with this pin's own target widget, positionally.
   A test asserting only `current_filter.has_focus` passes identically
   with the property present OR absent, i.e. it would never have caught
   the bug it was written to catch.
2. **The two-round settlement (`dispatch()`'s synchronous "loading" call,
   then the async worker's own "ready" call via `apply()`) races itself,
   independent of this bug.** Against a near-instant fake service, BOTH
   rounds' own `_sync_library_canvas(..., then=restore_focus, ...)` calls
   land within the same event-loop turn; `queue_after_recompose` holds
   only one pending callback per host, so the ready round's own resync
   overwrites the loading round's still-pending, CORRECT restore before
   it ever fires -- confirmed directly via temporary instrumentation
   (`_sync_library_skills_browse_result` ran 4 times for one refresh
   call; the loading round correctly derived `focus_identity=
   "library-skills-filter"`, the immediately-following ready round
   independently re-derived `focus_identity=None` because nothing was
   focused yet). This is a genuine, PRE-EXISTING race in `library_
   skills_browse_controller.py` (untouched by this move), unrelated to
   whether `focused` is bound -- confirmed by reproducing it with the
   property both present and absent. Not fixed (out of this task's
   pure-move scope; recorded here rather than silently worked around).
   The test uses a bounded-delay fake service (`_DelayedSkillsScopeService`,
   never an indefinite wait, to rule out any deadlock risk) to separate
   the rounds, and checks only the loading round's own settlement.

The landed test instead spies on `screen.query_one`, asserting
`"#library-skills-filter"` is queried -- the exact call `restore_focus()`
only makes when `focus_identity` is truthy, i.e. only when the live read
resolved a focused `library-skills-*` control. This is a direct trace of
the CODE PATH itself, immune to both confounds above. **Revert-probe
confirmed**: removing the `focused` property makes the test FAIL
(`AssertionError: restore_focus() never queried '#library-skills-filter'`,
queried selectors `['#library-skills-canvas', '#library-skills-reader-
shell', ...]` -- the canvas's own internal lookups fired, but never the
focus-restore one); restoring the property makes it PASS. Both directions
verified live, not asserted.

### 12b. IMPORTANT — recipe lesson

Added a sixth bypass shape to recipe §3's catalogue (`backlog/docs/
library-decomposition-recipe.md`): bare `self` as an identity-compared or
screen-identity argument (Forms A/B/C, generalizing this task's own
exclusion 5 with the two battery-caught regressions' full incident
detail) AND its close cousin, the unbound-attribute-escape shape this
review's own `focused` finding added (12a above). Also recorded the
reviewer's own minor-5 note as a standing rule: a battery-found hazard
that shrinks a RED commit's own mover-count tuple after the fact is the
expected shape of this work (the census that produces a RED tuple is
necessarily static; these shapes are by definition only found by running
code), not evidence the RED commit was wrong when written -- re-deriving
the FULL set of counts after such a correction, not just the one number
that changed, is what keeps the tuple and its own narrative from
drifting apart.

### 12c. Minors

- Section 1 above ("Cluster enumeration") originally said "three
  `@property`/`@x.setter` pairs" for the 6-raw-match/6-name gap,
  inherited verbatim from Task 1's own report (which made the identical
  arithmetic error despite listing all six names). Corrected to "SIX
  pairs" with the arithmetic spelled out (2 raw defs - 1 unique name = 1
  gap, per name; 6 names = 6 gap).
- `test_library_modules_size_ratchet.py`'s own dated comment said
  "~38-parameter constructor" while this report said "40 parameters
  total" -- both were counting DIFFERENT things (38 named dependencies
  vs. 40 total args including `self`/`screen`) without saying so.
  Reworded the ratchet comment to state both numbers and what each
  counts, removing the ambiguity rather than picking one silently.

### 12d. Fresh pins after the fix round

- Controller (`test_library_modules_size_ratchet.py`): `3099 -> 3131`
  (the `focused` property + its own docstring paragraph; comment updated
  same-commit per §17's re-pin-at-move flow).
- Screen: unchanged (`focused` was purely additive to the controller; no
  screen-side line moved).

### 12e. Verification (fix round)

- New covering test: FAIL-without/PASS-with confirmed live (12a above).
- `Tests/Architecture/test_library_skills_wiring.py`: 9/9 passed.
- Both size guards + support-layer surface: 50 passed, 2 failed (the
  same documented pre-existing `chat_screen.py` ratchet rows).
- Full `Tests/UI/test_library_skills_canvas.py` (158 tests, includes the
  new test): 151 passed, 7 failed -- all 7 match the already-documented
  CSS-block/geometry-parity + `test_action_library_skill_back_honors_
  dirty_guard` pre-existing bucket exactly.
- `Tests/Skills/test_skills_import.py` + `test_skills_library_flow.py`
  (the two files Form C's own regression hit): 70 passed, 2 failed --
  the same 2 already-documented pre-existing failures.
- `./scripts/preflight.sh`: all six checks pass clean (no diagnostic-
  inventory drift -- the `focused` property adds no logging calls).

### 12f. Files changed (fix round)

- `tldw_chatbook/UI/Library_Modules/library_skills_controller.py`: `+1`
  `focused` framework-service property; module docstring exclusion-5
  section extended with Form C's own paragraph (already present from the
  original landing) plus this round's cross-reference; mover/exclusion
  counts re-verified unchanged (86/41) since `focused` is additive, not a
  mover-set change.
- `Tests/UI/test_library_skills_canvas.py`: `+1` new test
  (`test_committed_mutation_refresh_with_no_focus_identity_restores_
  live_focus`) and its own supporting fake
  (`_DelayedSkillsScopeService`).
- `backlog/docs/library-decomposition-recipe.md`: §3 gains the sixth
  bypass-shape entry (12b above).
- `Tests/Architecture/test_library_modules_size_ratchet.py`: `_BUDGETS`
  row re-pinned `3099 -> 3131`; dated comment extended and disambiguated.
- `.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills/
  task-2-report.md`: this section, plus the minor-1 correction in §1.
