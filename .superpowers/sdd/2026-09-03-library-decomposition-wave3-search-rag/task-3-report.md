# Task 3 report — Combined Search+RAG controller PR (series 2/3)

Wave-3 Task 3 (recipe: `backlog/docs/library-decomposition-recipe.md`;
export/collections controller PRs are the worked examples). Scope: the
controller move, following Task 2's state PR
(`.superpowers/sdd/2026-09-03-library-decomposition-wave3-search-rag/task-2-report.md`).

## 1. Cluster re-derivation (fresh, not carried over)

Re-ran the `ast` census from scratch against the current tree (matches
Task 2's own numbers exactly, confirming no drift since):

- 60 raw `"search"`/`"rag"` name matches (24 + 39, minus the 3-name
  overlap), minus 3 Prompts-owned (`_flush_library_prompts_search`,
  `_queue_library_prompts_search`, `_stop_library_prompts_search_debounce`)
  and 7 Media-owned (`_focus_library_media_content_search_input`,
  `_reset_library_media_search_on_mode_change`, `handle_library_media_
  content_search_{next,prev,submitted}`, `handle_library_media_trash_
  search_{changed,submitted}`) = **50 combined-cluster candidates**.
- Decorator scan of all 50 confirms exactly the same 3 `@work`-decorated
  methods Task 2 found, no others.

## 2. Single vs. split controller — reconfirmed at the METHOD level

Task 2's field-level census already found all 20 state fields consumed
inside one lock-serialized call graph. This task re-verified the SAME
conclusion independently, per the task brief's own instruction, with an
`ast` call-graph walk of all 50 candidates (every `self.<name>(...)` call
where `<name>` is itself another candidate). Full call graph (selected
edges; see `library_rag_search_controller.py`'s module docstring for the
complete reasoning):

- `handle_library_search_submitted`/`rerun_library_search_from_history`/
  `submit_library_rag_query`/`run_library_rag_query` (the "search"/generic
  entry points) all call `_start_library_rag_query` directly.
- `_start_library_rag_query` calls `_record_library_search_history` (a
  "search"-named method) as an ordinary step in its own body.
- `clear_library_search_history` calls both `_library_rag_panel_state` and
  `_refresh_library_rag_history_widget` (both "rag"-named).
- `_apply_library_rag_search_outcome` calls `_start_library_rag_answer`
  unconditionally, in the same outcome-application sequence that also
  updates retrieval state and history.
- `_sync_library_rag_scope_toggle_and_run_gate_widgets` (a scope/run-gate
  method) schedules `_mirror_library_rag_scope_recovery` as a worker.

There is no subset of the 50 that only ever calls within itself; any split
would cut through calls that exist today. **Decision: ONE combined
`LibraryRagSearchController`**, confirmed by two independent methods
(Task 2's field census, this task's method call-graph).

## 3. Exclusions — 8 of the 50, not moved

### 3a. 3 `@work`-decorated methods (framework-decorator hazard)

`_execute_library_rag_answer` (`@work(exclusive=True,
group='library_rag_answer')`), `_execute_library_rag_search`
(`@work(exclusive=True, group='library_rag_search')`),
`_save_library_search_history` (`@work(thread=True)`). Textual's `@work`
closure asserts `isinstance(self, DOMNode)` at call time; a plain
controller instance would fail it. Reached from movers via named
late-binding callables (`execute_library_rag_answer`,
`execute_library_rag_search`, `save_library_search_history`).

### 3b. 1 module-globals-coupling exclusion — found by running the battery, not by static census

`_load_library_search_history` reads the bare name `get_cli_setting` (a
plain module-level import, not `self.get_cli_setting`). Python resolves a
free name against the DEFINING module's `__globals__`, fixed at definition
time — this is recipe §3's SECOND documented bypass shape (distinct from
instance-attribute monkeypatching), previously seen for
`_read_library_ingest_options_from_config`.

**How it was found**: this method was initially MOVED (43 movers in an
earlier draft of this task). The full `-k "(search or rag) and library"`
sweep (§6 below) failed
`Tests/UI/test_library_shell.py::test_library_shell_search_history_loads_from_cli_config_fallback`
— `screen._library_search_history == ()` instead of the expected
`("alpha", "bravo")` fallback tuple. Root cause: the test does
`monkeypatch.setattr(library_screen_module, "get_cli_setting",
fake_get_cli_setting)`, and a blanket per-test isolation fixture at
`test_library_shell.py:1015` ("tests that want to exercise the CLI-config
fallback itself re-patch `library_screen_module.get_cli_setting` after
this fixture runs") depends on the SAME mechanism for isolation across
MANY tests in that file. Once `_load_library_search_history`'s body moved
to `library_rag_search_controller.py` (which has its own independent
`from ...config import get_cli_setting`), the free name inside the moved
body resolved against the CONTROLLER's globals instead, silently
bypassing every `library_screen_module.get_cli_setting` patch for this one
call site specifically (other `get_cli_setting` call sites still on
`LibraryScreen` — ingest options, rail-state — were unaffected, since they
never moved).

**Fix**: reverted. `_load_library_search_history` stays a REAL,
full-bodied `LibraryScreen` method, byte-for-byte identical to before this
task (confirmed against the RED-commit baseline). It has no mover callers
(only `LibraryScreen.__init__`'s `LibraryRagSearchState(history=...)`
computed default), so excluding it needed no named-dependency binding on
the controller side, and let the controller's construction position in
`__init__` revert to the standard "right after `self._collections_
controller`" spot (an earlier draft had moved it earlier in `__init__`
specifically to accommodate this method's now-unneeded delegation).

**A second, currently-latent instance of the same shape, deliberately NOT
excluded**: `cycle_library_rag_mode`/`toggle_library_rag_scope_source`
forward `self` into the shared `_sync_library_canvas(screen, kind, ...)`
dispatcher (`canvas_sync.py`) as a bare name — also monkeypatched at the
`library_screen_module` level in **5 sites, not 4** (corrected in fix
round 1 below — the original census missed one): 4 in
`Tests/UI/test_library_entry_compose_once.py` (`:2909`, `:2980`, `:3183`,
`:3232`) plus 1 in `Tests/UI/test_library_notes_reader.py:213` (a
`monkeypatch.context()`-scoped `patcher.setattr(library_screen_module,
"_sync_library_canvas", ...)`, functionally identical to the other 4).
Read all 5 patched tests: none presses the RAG mode-toggle/scope-toggle
buttons or asserts on a `"search"`-kind sync call; the notes-reader site
exercises a notes-flush handler, entirely screen-resident and unrelated
to this cluster; the other 4 exercise `landing`/`conversations`/`media`
kinds through screen-resident callers that still resolve correctly. This
is the IDENTICAL shape the conversations controller
ALREADY shipped in wave-2 (`_sync_library_conversation_canvas` forwards
`self` the same way) with no additional accommodation. Confirmed
unexercised by the full 3-root sweep (zero related failures). Documented,
not silently accepted, in the controller's module docstring.

### 3c. 4 test-bypass exclusions (instance-attribute monkeypatch shape)

Found by a repo-wide census of every `monkeypatch.setattr(screen,
"<name>", ...)`/`monkeypatch.setattr(LibraryScreen, "<name>", ...)`/bare
`screen.<name> = ...` site across all THREE test roots (`Tests/UI`,
`Tests/Library`, `Tests/Live`) for all 50 candidate names:

- **`_refresh_search_rag_panel_state_widgets`** (Task 2's own forward
  note) — patched as a full replacement in
  `Tests/UI/test_product_maturity_gate16_library_search_rag.py:2905` (a
  `fail_refresh` that raises if reached) and
  `Tests/UI/test_library_shell.py:7310` (counting wrapper) and `:28851`
  (`Mock(wraps=...)`, `call_count == 0`). Called internally by 5 other
  movers.
- **`_patch_sibling_library_search_input`** (Task 2's own forward note) —
  patched as a bounded wrapper in `Tests/UI/test_library_shell.py:7307`.
  Called from 2 movers.
- **`_library_rag_panel_state`** (NEW finding — Task 2's characterization
  scan was `@on`-handler-only, so it never checked this non-`@on` name) —
  spied via the `_spy_panel_statuses` helper (defined at
  `test_product_maturity_gate16_library_search_rag.py:3014`; the actual
  `monkeypatch.setattr` call inside it is at `:3030` — corrected in fix
  round 1 below, the original citation pointed at the `def` line, not the
  patch site) (2 tests assert a transient `"answering"`
  status was observed mid-refresh). Called directly by 10 other movers —
  the single most heavily-depended-on method in the cluster.
- **`_mirror_library_rag_scope_recovery`** (NEW finding) — spied via
  `Mock(wraps=...)` in `test_library_shell.py:29076`, asserting `call_count
  == 0` on a repeat steady-state snapshot. Its only caller,
  `_sync_library_rag_scope_toggle_and_run_gate_widgets` (a mover), calls it
  via `self.run_worker(self._mirror_library_rag_scope_recovery(), ...)`.

**Confirmed SAFE, no exclusion needed** (same census, two more hits) —
**corrected in fix round 1 (see that section below): both caller claims
here were wrong in the original pass.** `_focused_library_rag_result_
card_index` (patched in `test_screen_navigation.py:2430`) has FOUR
callers, not one: `LibraryScreen.check_action` (screen-resident, never
moved) plus THREE controller-internal movers on this same cluster —
`action_library_rag_result_card_select`, `action_library_rag_result_
card_open`, `action_library_rag_use_in_console`
(`library_rag_search_controller.py:~1123/~1135/~1155`). The one test that
patches this name only ever exercises the `check_action` path (a bare
`screen._focused_library_rag_result_card_index = lambda: ...` instance
assignment, never followed by pressing Enter/`o`/`u`), so it stays safe
today — but a test that DID press one of those keys while patching the
screen's copy would silently miss the three controller-internal calls,
which now resolve `self.<name>()` against the controller instance, not
the patched screen instance. `_sync_library_rag_scope_toggle_and_run_
gate_widgets` itself (`Mock`-wrapped in `test_library_shell.py:28843`) has
exactly ONE external caller, and it is NOT `_apply_local_source_snapshot`
(that claim was simply wrong) — it is `_reconcile_library_entry_state`
(`library_screen.py:~12509`), a screen-resident method that was never a
cluster candidate. It is also not protected by recipe §3's four
permanently screen-routed names (`_list_local_source_snapshot`/`_refresh_
local_source_snapshot`/`_apply_local_source_snapshot`/`_refresh_library_
note_detail` — an unrelated list); it is simply screen-resident on its
own independent merits, and the actual protection is that its caller
invokes `self.<name>()` with `self` = the real, patched screen instance,
so the instance-level `Mock` shadows the class-level delegator regardless
of where the underlying implementation lives. Both still MOVE normally —
these are corrections to the *reasoning*, not the outcome.
`_apply_library_rag_search_outcome` is patched instance- and class-level
(`test_library_shell.py:7407`/`:7556`) but every site uses a REAL,
fully-constructed `LibraryScreen` (never an unbound fake), so an unbound
class-level call still resolves correctly through the screen delegator —
MOVES normally.

**Net: 42 of 50 candidates move.**

## 4. Byte-for-byte verification — method and result

Scripted `ast` extraction (not manual Read/Edit), following the
collections-controller precedent for a cluster this size:

1. Extracted each of the 42 final movers' exact source text (decorator
   start through `end_lineno`) from the pre-move `library_screen.py`,
   using the file's own line offsets — never hand-retyped.
2. Assembled the controller module from that extracted text plus a
   hand-written header/constructor/properties/footer.
3. Re-verified with a SECOND script: re-parsed both the RED-commit
   baseline (`b61d55987`) and the finished controller module, extracted
   each of the 42 method bodies from both by AST, and asserted byte-for-
   byte string equality. **Result: 0 missing, 0 mismatches** (confirmed
   twice — once before the `_load_library_search_history` exclusion fix,
   once after, both green).
4. Confirmed the 8 excluded methods remain byte-identical to the RED-
   commit baseline on the SCREEN (a second AST diff, same technique) —
   0 differences.

## 5. Free-name resolution walk

A dedicated script walked every `self.<name>` attribute reference across
all 41 non-`__init__` methods on the finished `LibraryRagSearchController`
class (including the 20 generated state-shim properties, the 10 framework-
service properties, and the 16 named-dependency properties) and checked
each resolves to either: a `dir(Controller)` class-level member, or an
instance attribute assigned in `__init__`. **Result: 0 unresolved names.**

Complete free-name census (from the pre-move census, before the module-
globals exclusion round; the module-globals hazards below — `get_cli_
setting`, `_sync_library_canvas` — are not `self.<name>` shapes, so this
walk does not (and should not) catch them; §3b documents how those were
found instead):

- **Framework services** (10, `@property` on the controller, live-forward
  to screen): `app_instance`, `app`, `call_after_refresh`, `focused`,
  `is_mounted`, `is_running`, `query`, `query_one`, `refresh`,
  `run_worker`. `app`/`is_running`/`refresh` exist ONLY because of the
  `_sync_library_canvas` forwarding (§3b) — confirmed by reading that
  dispatcher's own body line-by-line, not assumed from its signature.
- **Named constructor dependencies** (16): `_active_library_rail`,
  `_console_setup_would_block`, `_open_library_item_by_id`, `_safe_text`,
  `_select_library_rail_row`, `_trailing_index` (6 shared/incidentally-
  exclusive shell helpers); `_library_selected_row_id` (read-only),
  `_library_canvas_projection_depth` (read-only),
  `_library_canvas_resync_pending` (get+set) (3 shared shell-state
  bindings, the last 2 needed ONLY for `_sync_library_canvas` forwarding,
  mirroring the conversations controller's identical pair); the 3
  `@work`-excluded methods; the 4 test-bypass-excluded methods.
- **20 state fields** via the generated two-prefix shim loop
  (`_library_rag_<field>`/`_library_search_history`), reading through
  `self._rag_search_state_accessor()`.

## 6. Verification battery

All commands from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`, at commit `877eeaf9a` (search+RAG controller move)
unless noted.

- **Wiring suite** (`Tests/Architecture/test_library_search_rag_wiring.py`):
  6/6 passed (state-shim-surface test unchanged from Task 2, prefix-mapping
  guard, controller-owns-cluster, screen-delegates-cluster,
  staticmethod-forwards-to-class, controller-exposes-every-state-field).
- **Both size ratchets**
  (`test_screen_size_ratchet.py` + `test_library_modules_size_ratchet.py`):
  37 passed / 2 failed (only the two documented pre-existing
  `chat_screen.py` rows).
- **Regression suites** (export/collections/conversations wiring +
  characterization, support-layer surface, recompose ratchet): 20 passed.
- **Preflight** (`./scripts/preflight.sh`): all 5 checks green. First run
  (before regenerating the pin) correctly flagged the production
  diagnostic inventory drift from the moved `logger.debug` call inside
  `_start_library_rag_answer`; verified same digest (`0c79fc7ef3985611`)
  via `scripts/check_persistent_diagnostic_inventory.py --statements`
  before running `--write`.
- **Ordering bug found and fixed** (not test-driven, found by direct
  construction): an early draft moved `_load_library_search_history` and,
  separately, had to construct `LibraryRagSearchController` BEFORE
  `self._rag_search_state` to keep the (then-delegator) eager `__init__`
  call working. Running `test_check_action_gates_rag_result_card_actions_
  to_focused_card` and
  `test_library_search_rag_worker_completion_ignores_unmounted_screen`
  surfaced `AttributeError: 'LibraryScreen' object has no attribute
  '_rag_search_controller'` at `library_screen.py:22451` — every
  `LibraryScreen()` construction was broken. Both are fixed by the same
  §3b exclusion (the method never needed to be a delegator, so the
  controller never needed to move earlier); both tests pass afterward.
- **`-k "(search or rag) and library"` across `Tests/UI`+`Tests/Library`+
  `Tests/Live`** (single-process, matching Task 2's own per-task check):
  **789 passed, 15 failed, 3 skipped, 21510 deselected** in 797.70s.
  Diffed against Task 2's own documented 10-name pre-existing list
  (recipe §7's "documented pre-existing failures," none of which need
  re-deriving):
  - **10 of 15 match Task 2's list exactly**: `test_library_rag_handoffs.py::
    test_library_use_in_console_chip_and_prompted_counts_are_honest`;
    `test_library_rag_rechunk_action.py::{test_rechunk_control_class_
    defines_all_states_with_ds_tokens, test_rechunk_summary_and_report_
    lines_use_the_styled_quiet_line_class}`; `test_library_shell.py::
    {test_library_starter_hidden_route_focuses_compact_rail_without_search,
    test_library_shell_rail_search_submit_aborts_on_note_conflict,
    test_library_shell_notes_filter_queries_search_seam}`;
    `test_screen_navigation.py::{test_action_library_list_focus_rail_
    focuses_search_input, test_library_screen_round_trip_returns_to_
    landing_with_rag_draft, test_boot_with_search_default_tab_lands_on_
    library_rag_canvas, test_search_route_round_trips_to_the_library_rag_
    row}`.
  - **1 was a real bug, found and fixed** (§3b):
    `test_library_shell_search_history_loads_from_cli_config_fallback`.
    Confirmed passing after the fix, individually and in the wiring
    re-run.
  - **1 is a known, real, NEW test-path staleness, left for Task 4**:
    `Tests/Library/test_library_rag_scope.py::
    test_library_screen_call_sites_never_pass_scope_kwarg`. AST-walks
    `library_screen.py`'s own source for `LibraryRagSearchRequest(...)`
    call sites and asserts at least one exists with no `scope=` kwarg (a
    D2 guard: the Library Search canvas must stay unscoped by omission).
    The ONE call site (inside `_start_library_rag_query`) legitimately
    moved to `library_rag_search_controller.py`, so the census over
    `library_screen.py` alone now finds zero. Confirmed via `git stash -u`
    to the RED-commit baseline: PASSES there (the call site was still in
    `library_screen.py`). Confirmed the underlying INVARIANT the test
    guards still holds at the new location (a direct AST check of the
    controller module: 1 call, kwargs
    `{'include_citations', 'top_k', 'mode', 'query', 'source_types'}`, no
    `scope`). This is a hardcoded-file-path test census, not a monkeypatch
    bypass — the recipe reserves test edits for the cleanup PR (Task 4
    should retarget this census to `library_rag_search_controller.py`,
    or scan both files). Not touched here.
  - **3 investigated individually and confirmed pre-existing/flaky, NOT
    caused by this move** (each re-run in TRUE isolation, single test,
    multiple trials, on both this branch and a `git stash -u` pristine
    baseline of the RED-commit tree):
    - `test_library_prompts_canvas.py::
      test_library_prompts_stale_search_cannot_restore_an_old_filter_
      caret` — passed in isolation on this branch (`cursor_position`
      assertion is timing-sensitive; failed only inside the large batch
      run).
    - `test_screen_navigation.py::
      test_search_all_palette_command_lands_on_library_with_honest_toast`
      — FAILED identically on the pristine baseline in true isolation
      (same assertion, same symptom).
    - `test_screen_navigation.py::test_search_route_lands_on_library_rag_
      canvas` — a genuine, pre-existing app-boot race
      (`app.handle_screen_navigation`'s own guard: `if not
      self._initial_screen_pushed: ... return`, logged as "Ignoring
      navigation to search: initial screen not yet mounted"). The test's
      own poll loop treats `type(app.screen).__name__ != "Screen"` as the
      boot-complete signal, but the authoritative signal is
      `_initial_screen_pushed`, set only after `_push_initial_screen`'s
      own `await self.push_screen(new_screen)` fully resolves — there is
      a real window where `self.screen` has already flipped away from the
      placeholder while that flag is still `False`. Confirmed via
      isolated debug instrumentation (temporary, reverted) that this
      exact log line fires. Confirmed on the pristine baseline too:
      across 8 isolated trials, baseline showed 2 passed / 6 failed
      (a genuinely flaky pre-existing race, not a deterministic pass);
      this branch showed 0 passed / 8 failed across the same 8 trials.
      Measured whether this branch's added import weight explains the
      higher fail rate: `import tldw_chatbook.UI.Screens.library_screen`
      cold-import time is statistically indistinguishable (branch
      1.055s vs. baseline 1.085s — branch is not slower). The race exists
      identically on both trees; this branch's own trials happened to hit
      it every time in this session, which the recipe's own "expect
      noise to flip in either direction" framing (§7) anticipates rather
      than rules out. Reported here in full rather than rounded down to
      "pre-existing, case closed," since the differing trial counts are a
      real observation worth a future task's attention if it recurs.

### Full sequential xdist paired-baseline sweep (recipe §7)

`Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal`, branch
(commit `f8974b9cb`) then baseline, run SEQUENTIALLY per the recipe's own
"concurrent runs amplify flakiness" lesson. Baseline = a path-scoped
`git checkout 8efc79655 -- tldw_chatbook Tests Docs/security/production-
diagnostic-inventory.json` overlay of Task 2's own final commit (this
task's true merge-base), plus manually removing the new controller module
(a `git checkout` of an old commit does not delete files the target
commit never had) — restored afterward via the same overlay technique in
reverse plus re-adding the controller file from `git show`.

**Branch**: 347 failed, 3934 passed, 97 warnings in 1370.02s (22:50).
Grepped the failure names for `search`/`rag`: 12 hits, all already
individually investigated above (10 pre-existing Task-2-documented names,
1 fixed, 1 known test-path staleness) — this wider sweep surfaced no NEW
search/RAG-named failure beyond the narrower per-task check. The
remaining 335 failures are overwhelmingly clustered in
`test_library_shell.py`'s Notes tests (177 of 347, `test_library_note_*`/
`test_library_shell_note_*`) — an entirely different subsystem this
task's diff does not touch at all.

**Baseline** (same command, same overlay commit, restored to HEAD
afterward via `git checkout HEAD -- tldw_chatbook Tests Docs/security/
production-diagnostic-inventory.json`, confirmed clean and back to
43009/1316 + 1857 by `_measure()` and `git status` afterward): 354
failed, 3927 passed, 96 warnings in 1360.62s (22:40) — both branch and
baseline totals fall inside the recipe's own documented ~330–355
historical backdrop range (§7).

**Diff**: 346 shared, 8 baseline-unique (noise in the opposite
direction), **1 branch-unique**:
`Tests/UI/test_library_prompts_canvas.py::
test_library_prompt_undo_refreshes_applied_page_and_preserves_basket` — a
Prompts-canvas test, unrelated to Search/RAG and untouched by this task's
diff. **Cleared by name/file inference alone in the original pass; fix
round 1 (below) redid this by actual fixture/content inspection, not
inference**: the test's only parameter is the plain `tmp_path` fixture
(no fixture shared with this task's diff), and a grep of the full test
body for `search`/`rag` (case-insensitive) returns zero hits — nothing in
the function references this task's cluster at all. Re-ran in true
isolation on the branch: **1 passed** (6.72s), confirming ordinary xdist
ordering/shared-state noise, not a regression, per the recipe's own
re-run-before-concluding protocol.

**Zero real regressions** across the full sequential paired-baseline
sweep — a cleaner result than several prior tasks in this program (Task
2's own sweep found 8 branch-unique names, 2 of which reproduced as
pre-existing; this task's single branch-unique name did not reproduce at
all).

## 7. Fresh pins

`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
(`Tests/Architecture/test_screen_size_ratchet.py`): `43923/1316 ->
43009/1316` (net -914 lines, methods unchanged — a pure move net of
restoring `_load_library_search_history`'s full body once excluded).

`_BUDGETS["tldw_chatbook/UI/Library_Modules/library_rag_search_
controller.py"]` (`Tests/Architecture/test_library_modules_size_ratchet.py`):
born-governed, pinned at **1857** lines (42 moved methods + constructor/
property scaffolding).

## 8. Files changed

- `tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py`
  (new) — `LibraryRagSearchController`, 42 moved methods, 10 framework-
  service properties, 16 named-dependency properties, generated
  two-prefix state shim.
- `tldw_chatbook/UI/Screens/library_screen.py` — import added; 42 method
  bodies replaced by one-line delegators; `LibraryRagSearchController`
  constructed in `__init__` right after `self._collections_controller`;
  `_load_library_search_history` unchanged (real body, never moved).
- `Tests/Architecture/test_library_search_rag_wiring.py` — added
  `_RAG_SEARCH_CLUSTER_METHOD_NAMES` (42), `_RAG_SEARCH_CLUSTER_
  STATICMETHOD_NAMES` (1), and the 4 controller-PR wiring tests (owns-
  cluster, screen-delegates, staticmethod-forwards, controller-exposes-
  state-field).
- `Tests/Architecture/test_library_modules_size_ratchet.py` — new
  `_BUDGETS` row for `library_rag_search_controller.py` (1857).
- `Tests/Architecture/test_screen_size_ratchet.py` — `_BUDGETS` row
  lowered for `library_screen.py` (43009), with a dated comment.
- `Docs/security/production-diagnostic-inventory.json` — regenerated
  (the one moved `logger.debug` call, same digest, now attributed to the
  controller file).
- `.git-blame-ignore-revs` — appended the move commit's hash.

Commits: `b61d55987` (RED wiring pins), `877eeaf9a` (move),
`750df2c8c` (blame-ignore follow-up).

## 9. Self-review

- **The census-before-move discipline caught real things, but not
  everything the first time.** The initial monkeypatch census (§3c)
  correctly found 2 NEW test-bypass exclusions beyond Task 2's forward
  note. It did NOT initially check bare module-level names (as opposed to
  `self.<name>` attributes) for module-globals coupling — that gap
  (`get_cli_setting`) was only found by actually running the battery, not
  by static analysis ahead of time, exactly matching recipe §3's own
  documented lesson about this second bypass shape. A wider up-front grep
  (every bare, non-`self.` name referenced by a mover, checked against
  `monkeypatch.setattr(library_screen_module, "<name>", ...)` across all
  three test roots) would have caught this before the first move attempt
  rather than after; noted for the next subsystem's own pre-move census.
- **The `_sync_library_canvas` module-globals risk is real but
  unexercised, and left unexcluded on purpose, not by oversight.**
  Documented explicitly rather than silently accepted, with the specific
  reasoning (matches an already-shipped conversations-controller
  precedent; confirmed zero current test failures relate to it).
- **Ordering matters for `__init__` construction, and the collections
  state PR's own documented lesson (recipe §13, "check whether anything
  runs before the position you'd naively construct at") applies to
  controller PRs too, not just state PRs.** The first attempt at this
  move got this wrong in a way that broke EVERY `LibraryScreen()`
  construction, caught immediately by two targeted test runs before ever
  reaching the full sweep. Corrected, and the correct fix (excluding the
  one method properly) made the construction-order question moot rather
  than needing a permanent deviation.
- **The `test_library_screen_call_sites_never_pass_scope_kwarg` finding
  is reported as a known gap, not silently left broken.** Verified the
  underlying invariant it guards is unaffected at the new location before
  concluding it is purely a test-staleness issue, not a behavior change.
- **Byte-for-byte verification was tooled, not manual**, for both the
  initial 43-mover draft and the corrected 42-mover final state,
  reducing the risk that a hand-edit during the fix round silently
  altered a body.
- Did not touch `Agents/tool_catalog.py`-style registration surfaces or
  anything outside the controller-move scope. The delegator-prune census
  and shim-block deletion are explicitly Task 4 (cleanup) work, per the
  recipe's own series shape, and were not attempted here.
- **The full sequential xdist paired-baseline sweep (§6) used a real
  pristine tree, not an approximation.** `git checkout <commit> --
  tldw_chatbook Tests ...` alone was insufficient for the baseline (a
  checkout of an old commit does not delete files that commit never had —
  the new controller module survived the first checkout and had to be
  removed by hand before the baseline run was valid); caught by verifying
  `_RAG_SEARCH_CLUSTER_METHOD_NAMES`/the controller import were genuinely
  absent from the checked-out tree before launching the ~23-minute sweep,
  not after. Restoration to HEAD afterward was verified the same way (a
  fresh `_measure()` matching 43009/1316 + 1857, `git status` clean)
  before trusting the branch's own state for the commits already made.

## 10. Fix round 1 (reviewer findings)

The original implementer of this task was unavailable; this round was
done by a different agent working from the reviewer's verified findings
plus a fresh read of every cited call site before editing. Seven findings,
all fixed. In-place corrections were made directly in the sections above
(each marked inline as "corrected in fix round 1"); this section is the
changelog and the verification evidence.

1. **False caller claim #1 (`_focused_library_rag_result_card_index`)** —
   `library_rag_search_controller.py`'s module docstring and this report's
   §3c both said the "only caller" was `LibraryScreen.check_action`.
   Confirmed by direct grep (`grep -rn
   "_focused_library_rag_result_card_index(" tldw_chatbook/`) that there
   are FOUR callers: `check_action` (`library_screen.py:27716`, screen-
   resident) plus three controller-internal movers on THIS class —
   `action_library_rag_result_card_select` (`:1123`),
   `action_library_rag_result_card_open` (`:1135`),
   `action_library_rag_use_in_console` (`:1155`). Read the one test that
   patches this name (`Tests/UI/test_screen_navigation.py:2430`,
   `test_check_action_gates_rag_result_card_actions_to_focused_card`): it
   assigns `screen._focused_library_rag_result_card_index = lambda: ...`
   and calls only `screen.check_action(...)`, never the three action
   methods — so today's test suite is unaffected, but the docstring's
   "only caller" framing hid a real, live bypass hazard (a future test
   that patches the screen copy and then presses Enter/`o`/`u` would
   silently miss the three controller-internal calls, which resolve
   `self.<name>()` against the controller instance). Rewrote both the
   controller docstring and the report's §3c to name all four callers and
   state the hazard honestly, mirroring the `_sync_library_canvas`
   paragraph's own treatment of the identical shape.
2. **False caller claim #2 (`_sync_library_rag_scope_toggle_and_run_gate_
   widgets`)** — both documents claimed its only external caller was
   `_apply_local_source_snapshot`, protected by being one of recipe §3's
   four permanently screen-routed names. Grepped every call site
   (`grep -rn "_sync_library_rag_scope_toggle_and_run_gate_widgets("
   tldw_chatbook/`): the only external caller is actually
   `_reconcile_library_entry_state` (`library_screen.py:12509`, inside an
   `if self._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH:`
   guard) — a name that does not appear in recipe §3's four-name list at
   all (`_list_local_source_snapshot`/`_refresh_local_source_snapshot`/
   `_apply_local_source_snapshot`/`_refresh_library_note_detail`). The
   real protection is simpler and unrelated to that list: the caller is
   screen-resident on its own merits (never a cluster candidate, never
   moved), so it invokes `self.<name>()` with `self` = the real, patched
   screen instance, and the instance-level `Mock` from
   `test_library_shell.py:28843` shadows the class-level delegator
   regardless of where the implementation lives. Corrected both documents.
3. **Self-contradictory ratchet comment** —
   `Tests/Architecture/test_screen_size_ratchet.py`'s wave-3-task-3
   `_BUDGETS` comment said "43 of the 50 candidates moved" in its opening
   sentence, then "42, not 43" three paragraphs later, with the
   delegator/FunctionDef counts (43/43) also disagreeing with the final
   42. Truth, re-verified independently: 42 moved (41 instance-forwards +
   1 class-forward for the single staticmethod), 8 excluded (3 `@work` + 1
   module-globals-coupling (`_load_library_search_history`) + 4
   test-bypass = 8; 50 − 8 = 42). Rewrote the whole comment block so every
   number agrees, and added an explicit "(3 + 4 + 1 = 8 excluded; 50 − 8 =
   42 moved)" arithmetic line so a future reader doesn't have to re-derive
   it. Also fixed the matching "43 moved names" docstrings in
   `Tests/Architecture/test_library_search_rag_wiring.py` (`:134` and
   `:153`) to "42 moved names" — the module's own top-of-file docstring
   and the `_RAG_SEARCH_CLUSTER_METHOD_NAMES` tuple itself (counted
   programmatically: 42 entries) were already correct; only the two
   per-test docstrings had drifted.
4. **Shipped-red test, fixed now per controller ruling** —
   `Tests/Library/test_library_rag_scope.py::
   test_library_screen_call_sites_never_pass_scope_kwarg` censused
   `LibraryRagSearchRequest(...)` call sites in `library_screen.py` alone;
   the sole construction site moved to
   `library_rag_search_controller.py:1036` in this task's own move,
   leaving the census with zero call sites and a failing "expected at
   least one" assertion. Retargeted the census to walk BOTH
   `library_screen.py` and `library_rag_search_controller.py` (decided
   over a controller-only retarget because the invariant is "no call site
   anywhere passes `scope=`," not "no call site in this one file" — a
   two-file census survives a future move back onto the screen, or a
   second call site appearing on either file, without another retarget).
   Assertions unchanged in meaning (still: at least one call site exists,
   none pass `scope=`). Verified green in isolation
   (`.venv/bin/python -m pytest Tests/Library/test_library_rag_scope.py::
   test_library_screen_call_sites_never_pass_scope_kwarg -p no:randomly
   -q` → 1 passed) and in the full battery below. Added one sentence to
   `backlog/docs/library-decomposition-recipe.md` §3 (a new paragraph
   after its existing bypass-shapes list): a hardcoded-file-path census is
   not a monkeypatch bypass — it ships red at the exact commit boundary
   that moves the code, so it is retargeted in the SAME PR-stage that
   moved the code (no-red-ships precedence wins over the usual
   "defer routing fixes to cleanup" doctrine), with cleanup handling the
   rest of that subsystem's routing work as usual.
5. **Incomplete patch census (`_sync_library_canvas`)** — the report's §3b
   said 4 patch sites; grepping
   `monkeypatch.setattr(library_screen_module, "_sync_library_canvas"` /
   `patcher.setattr(library_screen_module, "_sync_library_canvas"` across
   `Tests/` found 5: the 4 already named in
   `Tests/UI/test_library_entry_compose_once.py` (`:2909`, `:2980`,
   `:3183`, `:3232`) plus `Tests/UI/test_library_notes_reader.py:213`
   (inside a `monkeypatch.context()` block, same shape). Read the 5th
   site's own test: it exercises a notes-flush handler entirely
   unrelated to the Search/RAG cluster, screen-resident throughout —
   benign, same conclusion as the other 4, just an undercounted census.
   Corrected the report's count and citation list.
6. **Sweep-triage argument strengthened to the required standard** — the
   report cleared the 1 branch-unique failure
   (`test_library_prompt_undo_refreshes_applied_page_and_preserves_
   basket`) by name/file inference ("a Prompts-canvas test, unrelated to
   Search/RAG"), which the testing-evidence standard does not accept on
   its own. Ran the actual fixture/content inspection the standard
   requires: the test's only parameter is the plain `tmp_path` fixture
   (shares no fixture with this task's diff), and a case-insensitive grep
   of the full test body for `search`/`rag` returns zero hits. Added both
   facts to the report so the triage now rests on inspected evidence, not
   inference, alongside the pre-existing isolated re-run (1 passed,
   6.72s).
7. **Citation fix** — the report cited
   `test_product_maturity_gate16_library_search_rag.py:3014` as "the"
   patch site for `_library_rag_panel_state`; `:3014` is the `def
   _spy_panel_statuses(...)` line, and the actual `monkeypatch.setattr`
   call inside that helper is at `:3030` (confirmed by grep). Corrected
   the citation to point at both: the helper's definition and the patch
   call itself.

**Side effect of fixes 1–2: the controller's pinned size budget moved.**
Correcting the two false-caller paragraphs in
`library_rag_search_controller.py`'s own module docstring grew the file
from 1857 to 1890 lines (comment-only growth — no method body, no mover
count, no byte-for-byte canon content changed). Per
`backlog/docs/library-decomposition-recipe.md`'s controller-governance
re-pin rule (§17: re-measure and re-pin in the same commit, never
deferred), updated the `_BUDGETS` row in
`Tests/Architecture/test_library_modules_size_ratchet.py` from 1857 to
1890 with a dated comment. `library_screen.py` itself was not touched by
any fix here (`git diff --stat` confirms — only the controller module,
the two Architecture test files, the Library scope test, and the recipe
doc changed), so `test_screen_size_ratchet.py`'s own `library_screen.py`
budget (43009/1316) needed no change; only its wave-3-task-3 governance
*comment* (finding 3) was corrected.

### Verification

```
.venv/bin/python -m pytest Tests/Library/test_library_rag_scope.py \
  Tests/Architecture/test_library_search_rag_wiring.py \
  Tests/Architecture/test_library_modules_size_ratchet.py \
  Tests/Architecture/test_screen_size_ratchet.py \
  Tests/UI/test_library_recompose_ratchet.py -p no:randomly -q
```

Result: **63 passed, 2 failed** — the 2 failures are exactly the
pre-existing, documented `chat_screen.py` rows
(`test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/
chat_screen.py]`, `test_task_22507_4_does_not_worsen_chat_screen_base`),
unrelated to this task's scope and unrelated to the fix round. No new
failures. `./scripts/preflight.sh` run and confirmed green (see below).

### Files touched this round

- `tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py` —
  module docstring: corrected findings 1–2 (caller counts/names,
  protection rationale).
- `Tests/Architecture/test_screen_size_ratchet.py` — rewrote the
  wave-3-task-3 `_BUDGETS` comment block for internal consistency
  (finding 3); no `_BUDGETS` value changed (that file's own diff is
  comment-only).
- `Tests/Architecture/test_library_search_rag_wiring.py` — two docstrings
  "43 moved names" → "42 moved names" (finding 3).
- `Tests/Library/test_library_rag_scope.py` — retargeted
  `test_library_screen_call_sites_never_pass_scope_kwarg`'s census to both
  `library_screen.py` and `library_rag_search_controller.py` (finding 4).
- `backlog/docs/library-decomposition-recipe.md` — new paragraph at the
  end of §3 documenting the path-census no-red-ships exception (finding
  4).
- `Tests/Architecture/test_library_modules_size_ratchet.py` — re-pinned
  `library_rag_search_controller.py`'s budget 1857 → 1890 with a dated
  comment (side effect of findings 1–2, per §17's re-pin-in-the-same-
  commit rule).
- This report — in-place corrections for findings 1, 2, 5, 6, 7, plus this
  section.
