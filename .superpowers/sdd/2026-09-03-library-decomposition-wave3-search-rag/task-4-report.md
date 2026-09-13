# Task 4 report — Combined Search+RAG cleanup PR (series 3/3)

Wave-3 Task 4 (recipe: `backlog/docs/library-decomposition-recipe.md`;
export/collections cleanup PRs are the worked examples). Scope: cleanup —
the one PR type allowed to edit tests and moved-body docstrings. Follows
Task 2 (state PR) and Task 3 (controller PR); base commit `801c5375e`.

## 1. Dynamic-dispatch census (screen + all three test roots)

Ran the recipe's own dynamic-dispatch guidance (§3/§11 lesson 3, extended
by the export/collections series) against all 20 `LibraryRagSearchState`
field names, across `tldw_chatbook/` (the whole tree, not just the screen)
and all three test roots (`Tests/UI`, `Tests/Library`, `Tests/Live`):

- **`getattr`/`setattr` with an f-string or dict-literal argument**: zero
  hits touching any of the 20 fields.
- **`dict.get(...)` result flowing into `setattr`/`getattr` later in the
  same function** (the collections series' own task-6 fix-round addition,
  §14): zero hits.
- **A literal `screen.<flat_name> = value` write from a module OTHER than
  `library_screen.py`** — found **one site**:
  `tldw_chatbook/UI/Library_Modules/canvas_sync.py`'s
  `_sync_library_canvas`, in its `"search"` branch:
  `screen._library_rag_answer_render_key = None`. **This one needed
  tracing, not retargeting — the obvious fix was wrong, and shipping it
  once is the actual finding.** The function's own parameter is typed
  `screen: "LibraryScreen"`, so the first-pass fix retargeted this to
  `screen._rag_search_state.answer_render_key = None` to match the
  deleted shim. That edit passed the wiring suite and both ratchets, then
  failed `Tests/UI/test_library_canvas_scoped_sync.py::
  test_media_choice_and_rag_toggles_are_canvas_scoped` in the narrow
  `-k "(search or rag) and library"` sweep — a name neither Task 2 nor
  Task 3's documented failure lists mention. Tracing `_sync_library_
  canvas`'s actual callers for `kind == "search"` (`grep -rn
  "_sync_library_canvas(" tldw_chatbook/`) found exactly two, both
  `LibraryRagSearchController` methods (`cycle_library_rag_mode`/
  `toggle_library_rag_scope_source`), both calling `_sync_library_canvas(
  self, "search")` — `self` there is the CONTROLLER, forwarded as the
  `screen` parameter despite the type annotation. The controller has no
  `_rag_search_state` attribute by design (its own permanent shim's
  docstring: "this class has none"), so the dotted retarget raised
  `AttributeError` on every real invocation. The FLAT name was already
  correct, resolving through the controller's own permanent generated
  shim (installed by Task 3, mirroring the screen's now-deleted one, for
  exactly this `self`-forwarding shape — already documented as a known,
  deliberately-unexcluded risk in the controller's own module docstring,
  §3b). Reverted; `canvas_sync.py` needed NO change at all.
- **Name-collision false positives, confirmed by receiver, not string
  alone**: a repo-wide grep for the flat name `_library_rag_query` also
  surfaces `Tests/UI/test_console_rag_settings_modal.py:54` and
  `Tests/UI/test_console_library_search_modal.py:25`, both setting
  `controller._library_rag_query = lambda: ...` where `controller` is a
  `ConsoleRetrievalController` (`tldw_chatbook/UI/Console_Modules/
  retrieval.py`) — an entirely unrelated Console feature with its own,
  coincidentally-identically-named callable. Confirmed unrelated by
  reading the receiver's class (`object.__new__(ConsoleRetrievalController)`)
  and the surrounding test file's imports; neither touched.
- The `_replace_library_reader_preference`/`_persist_library_reader_
  preference` 7-destination shared dispatcher (already fixed by the
  conversations/collections series for their own subsystems) has **no
  `"search"`/`"rag"` entry** — confirmed by reading its dict literal;
  Search+RAG was never one of its destinations.

## 2. Screen-side retarget: flat state fields → `self._rag_search_state.<field>`

AST census of `LibraryScreen` for the 20 flat field names (before any
edit) found live consumers in exactly 9 non-cluster methods (the moved
cluster's own 42 methods never touch flat names directly — they resolve
through the controller's own permanent shim):

| Method | Fields touched |
|---|---|
| `save_state` | 11 |
| `restore_state` | 11 |
| `_library_rag_panel_state` (test-bypass exclusion, stays screen-resident) | 14 |
| `_library_continue_receipt_for_current_route` | 4 |
| `_mirror_library_rag_scope_recovery` (test-bypass exclusion) | 2 |
| `_refresh_search_rag_panel_state_widgets` (test-bypass exclusion) | 2 |
| `_reconcile_library_entry_state` | 1 |
| `_replace_library_canvas_child` | 1 |
| `_show_library_file_notes` | 1 |
| `_sync_library_rail_lifecycle_presentation` | 1 |
| `compose_content` | 2 |

**35 literal `self._library_rag_<field>`/`self._library_search_history`
occurrences retargeted** to `self._rag_search_state.<field>` — one
mechanical pass: `save_state`/`restore_state`'s multi-field blocks
hand-edited for readability (kwarg-per-line reformatting where the new
name pushed a line over-length), the remaining 25 occurrences (all
single-field, `query`/`results`/etc.) via a scripted `re.sub` with a
per-field mapping table, restricted to `self.` receivers and excluding the
shim block's own generic `getattr(self._rag_search_state, _n)` text (which
contains no literal field names to collide with). Re-verified with a fresh
AST walk: **zero remaining consumers** of any flat name outside the
soon-to-be-deleted shim block.

Six comment/docstring mentions of the flat names (no `self.` prefix, e.g.
`` `_library_rag_query` `` in a docstring) were also updated for accuracy,
since the underlying attribute stops existing once the shim is gone:
`_library_rag_panel_state`'s docstring (`restore_state` cross-reference),
`_patch_sibling_library_search_input`'s docstring, two mentions inside
`_mirror_library_rag_scope_recovery`/`_refresh_search_rag_panel_state_
widgets`'s docstrings (`_library_rag_panel_refresh_lock`), one inline
comment (`_library_rag_answer`), and one cross-module docstring
(`tldw_chatbook/Library/library_ingest_state.py`'s `LibraryIngestFormState`
docstring, which analogizes to `_library_rag_history_collapsed` by name).

## 3. Test retarget — per file, all three roots

Repo-wide grep for the 20 flat field names across the whole `Tests/` tree
(not just `Tests/UI`/`Tests/Library`, per the collections series' own
"widen the root" lesson, recipe §16 lesson 2) found 7 files with hits (6
needing a live retarget, 1 comment-only), plus 2 more (the Console modal
tests, §1) confirmed unrelated name collisions and left untouched:

| File | Retargets | Notes |
|---|---|---|
| `Tests/UI/test_library_shell.py` | 62 (56 `self.`-shaped attribute accesses across 4 receiver names `screen`/`original`/`restored`/`second` + 6 bare docstring/comment mentions) | scripted regex pass, verified zero remaining hits afterward |
| `Tests/UI/test_product_maturity_gate16_library_search_rag.py` | 56 (54 attribute accesses + 2 bare docstring mentions) | same scripted pass |
| `Tests/UI/test_library_rag_keystroke.py` | 7 (6 attribute accesses + 1 docstring mention) | hand-edited individually |
| `Tests/UI/test_screen_navigation.py` | 2 | real `app.screen`/`restored_screen` instances |
| `Tests/ProductionApp/test_reactive_ownership_maturity.py` | 2 | real `screen` instances in the full-harness maturity walkthrough |
| `Tests/UI/test_library_honesty_accessibility.py` | 1 | real `screen` instance |
| `Tests/Architecture/test_screen_size_ratchet.py` | 0 live retargets (1 historical trajectory comment, left as-is — describes what Task 2 measured at the time, still accurate) | |

**130 retargets total, zero assertion VALUE changes** — every one is a
receiver-path rewrite only (`screen._library_rag_<field>` →
`screen._rag_search_state.<field>`), confirmed by running each file before
and after. No unbound-fake-self construction touches any Search+RAG
field (all sites use a REAL, fully-constructed `LibraryScreen`), so no
fixture needed the "flat kwargs → nested `_rag_search_state=
SimpleNamespace(...)`" restructuring the conversations/export exemplars'
own cleanup PRs needed.

`Tests/Live/` and `Tests/Library/` (the collections series' own two
findings) were both re-checked and came back clean: zero hits for any of
the 20 flat field names in either root.

## 4. Screen shim block deletion

The task-2-generated `_library_rag_<field>`/`_library_search_<field>`
property-shim loop (installed at module end, `dataclasses.fields(
LibraryRagSearchState)`-driven, two-prefix) was deleted wholesale once
census #1/#2 above confirmed zero remaining consumers anywhere in
`tldw_chatbook/` or `Tests/` outside `LibraryRagSearchController`'s own
PERMANENT generated shim loop (installed by task 3, reading `self.
_rag_search_state_accessor().<field>` — untouched by this task, per the
task brief's explicit "controller shims STAY" instruction).

## 5. Delegator census (all 42)

Per-name repo-wide grep (`tldw_chatbook/`, `Tests/`, including
`Tests/Live/`) for every one of task 3's 42 moved-cluster names, split by
decorator shape:

- **14 `@on`-decorated handlers: KEEP, unconditionally.**
- **3 `action_*` handlers: KEEP, unconditionally** (Textual's own action
  dispatch resolves these by string-keyed `getattr`, not literal name
  reference — the recipe's own transform whitelist names `@on`/`action_*`
  together as always-keep). Confirmed real `Binding` entries dispatch to
  all three (`"enter"`→`library_rag_result_card_select`, `"o"`→
  `library_rag_result_card_open`, `"u"`→`library_rag_use_in_console`) and
  `check_action` gates all three by the same literal strings.
- **13 non-`@on`/`action_*` methods with a genuine external caller**
  beyond their own delegator body: KEEP.
  - Screen-resident callers: `_execute_library_rag_answer` (calls
    `_apply_library_rag_answer`), `_execute_library_rag_search` (calls
    `_apply_library_rag_search_outcome`), `_refresh_search_rag_panel_
    state_widgets` (calls all four `_refresh_library_rag_*_widgets` plus
    `_library_rag_scope_summary`), `_mirror_library_rag_scope_recovery`
    (calls `_apply_library_rag_scope_recovery_block`), and
    `_reconcile_library_entry_state` (a screen-resident method that was
    NEVER one of the 50 cluster candidates, calling `_sync_library_rag_
    scope_toggle_and_run_gate_widgets`).
  - Direct test calls on a real `screen` instance: `_apply_library_rag_
    answer`, `_apply_library_rag_search_outcome`, `_library_rag_answer_
    chat_kwargs`, `_start_library_rag_query`.
  - Instance-attribute monkeypatch relying on the delegator SLOT existing:
    `_focused_library_rag_result_card_index` (patched by `test_screen_
    navigation.py`, read by `check_action`), `_sync_library_rag_scope_
    toggle_and_run_gate_widgets` (`Mock`-wrapped by `test_library_
    shell.py`, called by `_reconcile_library_entry_state`).
- **12 non-`@on`/`action_*` methods with ZERO consumers beyond their own
  delegator body**: **PRUNED** — each confirmed by a repo-wide grep
  restricted to `*.py` outside the controller module, the screen's own
  delegator body, and this task's wiring-test cluster tuple, before
  deletion: `_focus_library_search_input`, `_open_library_rag_result_by_
  index`, `_persist_library_search_history`, `_record_library_search_
  history`, `_reset_library_rag_answer_state`, `_reset_library_rag_in_
  flight_status`, `_reset_library_rag_retrieval_state`, `_reveal_library_
  rag_results`, `_select_library_rag_result_by_index`, `_stage_library_
  rag_result_in_console`, `_start_library_rag_answer`, `_use_library_rag_
  result_in_console`. Every one of these 12 is still called internally,
  controller-to-controller, by its own sibling movers.

**Net: 30 KEEP, 12 PRUNED** (~29% of 42) — between export's 1-of-22 (~5%)
and collections' 14-of-64 (~22%)/conversations' 18-of-61 (~30%).

## 6. Shim block deletion (screen side) — confirmed

Verified via a post-deletion AST re-scan: zero `def <pruned-name>(` on
`LibraryScreen` for all 12; the 30 KEEP names still delegate correctly
(wiring test `test_screen_delegates_rag_search_handlers` covers this
mechanically).

## 7. Import verification

**5 dead imports removed**, each verified single-occurrence (import line
only) in `library_screen.py` via per-name grep, then checked against
`Tests/Architecture/test_library_support_layer_surface.py`'s `_SURFACE`
dict (the PR-0a re-export contract) before deletion — `library_rag_state`,
`library_rag_answer_service`, `library_rag_service`,
`Views.RAGSearch.search_handoff`, and `Library_Modules.library_rag_
search_state` are not among `_SURFACE`'s 5 listed modules:

- 3 already dead since Task 3's controller move (their only screen-side
  consumers moved with the cluster; Task 3 deliberately left them for this
  cleanup PR, per the export/collections Task 3/Task 4 split):
  `LIBRARY_RAG_QUERY_MAX_LENGTH`, `LIBRARY_RAG_USE_IN_CONSOLE_LOCKED_
  NOTICE`, `library_rag_scope_summary`.
- 1 newly dead as a DIRECT RESULT of this task's own delegator prune
  (`_stage_library_rag_result_in_console` was its only screen-side
  caller): `build_library_rag_console_live_work_payload`.
- 1 whose only screen-side consumer was the deleted shim block itself:
  `SEARCH_PREFIXED_STATE_FIELDS` (still imported and used by the wiring
  test directly from `library_rag_search_state`, and still the
  controller's own generated shim's source of truth — only the SCREEN's
  import became dead).

`LibraryRagSearchState` itself stays imported (used at `__init__`'s
`self._rag_search_state = LibraryRagSearchState(...)` construction).

## 8. Wiring test finalization

`Tests/Architecture/test_library_search_rag_wiring.py`:
`test_state_object_fields_match_the_shim_surface` DELETED (screen shim
gone, mirrors the conversations/export/collections precedent exactly);
`_RAG_SEARCH_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset (12 names) added,
with `test_screen_delegates_rag_search_handlers` skipping those names and
instead asserting their genuine ABSENCE from `LibraryScreen`; module
docstring rewritten to describe the finished 3-task series.
`test_search_prefixed_state_fields_are_real_state_fields` (a general
invariant on `SEARCH_PREFIXED_STATE_FIELDS` itself, unrelated to whether
the screen imports it) is unchanged. 5 of the original 6 tests remain
(the shim-surface test's removal is the only count change) — all 5 green
post-cleanup.

## 9. The ruled moved-body docstring fix

Task 3's own review found `_sync_library_rag_scope_toggle_and_run_gate_
widgets`'s moved-body docstring carrying a false caller claim ("Called
synchronously from `_apply_local_source_snapshot`'s in-place branch") —
byte-for-byte ORIGINAL text present on the screen before Task 3's move,
so Task 3's own fix round correctly declined to touch it (fixing it there
would violate the byte-for-byte canon on a body the controller-move PR
cannot edit) and ruled the correction into this cleanup PR instead,
per the wave-2 `_apply_library_row_toggle` precedent.

Fixed here: the docstring now names the actual caller,
`_reconcile_library_entry_state` (`library_screen.py`, screen-resident,
never one of the 50 cluster candidates), and clarifies the call is
scheduled via `call_later` off every snapshot-generation bump
(`_apply_local_source_snapshot` and its siblings) rather than literally
inline inside that method's own stack frame — so the "fires off the UI
thread on every ingest done-count growth" framing survives, attributed to
the right intermediate caller. Matches the module docstring's own
already-corrected paragraph (Task 3 fix round 1, lines ~226-241) rather
than introducing a second, possibly divergent, correction.

Side effect: the controller file grew by 5 lines (comment-only) —
`library_rag_search_controller.py`'s `_BUDGETS` row re-pinned 1890 → 1895
in the same commit, per §17's re-pin-in-the-same-commit rule.

## 10. Recipe update

`backlog/docs/library-decomposition-recipe.md`:
- §8's subsystem-order table: the "search — BLOCKED" row replaced with
  "search + RAG — complete", merged with the "RAG / onboarding plumbing"
  pool row it had already absorbed at wave-2 close.
- New §18: the full search+RAG series as landed (cluster derivation,
  single-vs-split confirmation, per-task fields/methods table, pin
  trajectory, delegator census, the two new findings below, the ruled
  docstring fix, sweep evidence, and 3 lessons).

Two genuinely new findings recorded for future subsystems:
1. **A flat name found outside the screen file needs its caller traced,
   not just retargeted** — a shared dispatcher's `self`-forwarding shape
   (a controller passing itself AS the `screen` parameter) can make an
   already-correct flat name LOOK stale; retargeting it without tracing
   the actual callers is the bug, not leaving it (§1 of this report, the
   `canvas_sync.py` near-regression).
2. **A flat field name is not a unique key across the whole codebase** —
   confirm the receiver's type/class before treating a grep hit as this
   subsystem's own (the Console `_library_rag_query` collision).

## 11. Fresh pins

`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
(`Tests/Architecture/test_screen_size_ratchet.py`): `43009/1316 ->
**42949/1304**` (net -60 lines, -12 methods — exactly the 12 pruned
delegators; zero method bodies touched).

`_BUDGETS["tldw_chatbook/UI/Library_Modules/library_rag_search_
controller.py"]` (`Tests/Architecture/test_library_modules_size_
ratchet.py`): `1890 -> **1895**` (comment-only growth from the ruled
docstring fix, §9).

Full pin trajectory for the search+RAG series:
`43977/1316 (task 2 start) -> 43923/1316 (task 2) -> 43009/1316 (task 3)
-> 42949/1304 (task 4, final)`; controller: born-governed 1857 -> 1890
(task 3 fix round) -> 1895 (task 4, final).

## 12. Battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`, `-p no:randomly`.

- **Wiring suite** (`Tests/Architecture/test_library_search_rag_wiring.py`):
  5/5 passed (the shim-surface test removed; prefix-mapping guard,
  controller-owns-cluster, screen-delegates-cluster (with the 12-name
  prune skip/absence check), staticmethod-forwards-to-class,
  controller-exposes-every-state-field).
- **Both size ratchets** (`test_screen_size_ratchet.py` +
  `test_library_modules_size_ratchet.py`): 30 passed / 2 failed (only the
  two documented pre-existing `chat_screen.py` rows).
- **Regression suites** (export/collections/conversations wiring +
  characterization, support-layer surface, recompose ratchet): 21 + 28 =
  49 passed (5 search+rag + 4 collections + 5 export + 7 conversations
  wiring; 5 collections + 5 export + 4 conversations characterization + 6
  recompose ratchet + 8 support-layer).
- **Preflight** (`./scripts/preflight.sh`): all 6 checks green (CSS
  bundle, profile-owned-path census, diagnostic inventory, backlog task
  ids, chachanotes table allowlist, index plan pins) — no diagnostic
  inventory drift (this task moved zero `logger.debug`/persistent-
  diagnostic call sites, unlike Task 3's own controller move).
- **`-k "(search or rag) and library"` across `Tests/UI`+`Tests/Library`+
  `Tests/Live`** (single-process, matching Task 2/3's own per-task check):
  **12 failed, 792 passed, 3 skipped, 21510 deselected in 765.50s**. A
  first run (before the `canvas_sync.py` finding below was corrected)
  also showed 12 failed/792 passed, but with
  `test_library_canvas_scoped_sync.py::
  test_media_choice_and_rag_toggles_are_canvas_scoped` among them — a
  real regression from this task's own first-draft fix, investigated and
  reverted (§1). The corrected re-run's 12 failures are **all** exactly
  matched to already-documented names: the same 10 from Task 2's own
  documented pre-existing list (`test_library_rag_handoffs.py::
  test_library_use_in_console_chip_and_prompted_counts_are_honest`;
  `test_library_rag_rechunk_action.py::{test_rechunk_control_class_
  defines_all_states_with_ds_tokens, test_rechunk_summary_and_report_
  lines_use_the_styled_quiet_line_class}`; `test_library_shell.py::
  {test_library_starter_hidden_route_focuses_compact_rail_without_search,
  test_library_shell_rail_search_submit_aborts_on_note_conflict,
  test_library_shell_notes_filter_queries_search_seam}`; `test_screen_
  navigation.py::{test_action_library_list_focus_rail_focuses_search_
  input, test_library_screen_round_trip_returns_to_landing_with_rag_
  draft, test_boot_with_search_default_tab_lands_on_library_rag_canvas,
  test_search_route_round_trips_to_the_library_rag_row}`) plus 2 of Task
  3's own "confirmed pre-existing/flaky, not caused by the move" bucket
  (`test_screen_navigation.py::{test_search_route_lands_on_library_rag_
  canvas, test_search_all_palette_command_lands_on_library_with_honest_
  toast}`, both app-boot-race/timing-sensitive, both previously confirmed
  identical on a pristine baseline by Task 3). **Zero new failures.**
- **Full sequential xdist paired-baseline sweep** (`Tests/UI -k "library"
  -p no:randomly -q -n 8 --dist worksteal`, branch then a `git stash -u`
  pristine baseline of the pre-task tree, per recipe §7, run
  SEQUENTIALLY per the recipe's own "concurrent runs amplify flakiness"
  lesson):

  | | Failed | Passed | Wall time |
  |---|---|---|---|
  | Branch (this task's tree) | 350 | 3931 | 1314.03s (21:54) |
  | Baseline (`801c5375e`, `git stash -u`) | 349 | 3932 | 1340.34s (22:20) |

  Both totals fall inside the recipe's own documented ~330–355
  historical backdrop range (§7). Diffing the two failure-name sets:
  **345 shared, 4 baseline-unique** (noise in the opposite direction —
  one of the 4, `test_library_screen_round_trip_returns_to_landing_
  with_rag_draft`, is itself a search/RAG-cluster test, failing on the
  BASELINE and not the branch, which alone rules out a regression for
  it), **5 branch-unique**:
  `test_library_shell.py::{test_library_media_initial_error_is_unknown_
  and_retry_is_unique, test_library_media_page_error_retains_rows_and_
  gates_unsafe_controls, test_library_note_compact_deep_link_intent_
  opens_notes_stage[context2-#library-note-body-editor-False],
  test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_
  back}`, `test_screen_navigation.py::test_skills_route_lands_on_
  library_with_skills_row_selected` — none touch Search/RAG (Media,
  Notes, and Skills route tests). Re-run combined, single-process,
  true isolation: **3 of 5 passed cleanly** (xdist ordering/shared-state
  noise). The remaining 2
  (`test_library_media_page_error_retains_rows_and_gates_unsafe_
  controls`, `test_skills_route_lands_on_library_with_skills_row_
  selected`) reproduced identically in a SECOND `git stash -u` to the
  same pristine `801c5375e` tree, run in the same true-isolation
  combination — confirmed genuinely pre-existing, not caused by this
  task, and unrelated to Search/RAG. **Zero real regressions.** Both
  newly-confirmed pre-existing names added to the recipe's §7
  documented list.

## 13. Files changed

- `tldw_chatbook/UI/Screens/library_screen.py` — shim block deleted; 35
  literal field references retargeted; 12 dead delegators removed; 5 dead
  imports removed; 6 docstring/comment mentions updated.
- `tldw_chatbook/Library/library_ingest_state.py` — one cross-reference
  docstring updated for accuracy.
- `tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py` —
  the ruled moved-body docstring fix (§9); no method body changed.
- `Tests/UI/test_library_shell.py`, `Tests/UI/test_product_maturity_
  gate16_library_search_rag.py`, `Tests/UI/test_library_rag_keystroke.py`,
  `Tests/UI/test_screen_navigation.py`, `Tests/ProductionApp/test_
  reactive_ownership_maturity.py`, `Tests/UI/test_library_honesty_
  accessibility.py` — 130 receiver-path retargets, zero assertion value
  changes.
- `Tests/Architecture/test_library_search_rag_wiring.py` — shim-surface
  test deleted; `_RAG_SEARCH_CLUSTER_SCREEN_DELEGATOR_PRUNED` added;
  module docstring rewritten.
- `Tests/Architecture/test_screen_size_ratchet.py` — `_BUDGETS` row
  lowered for `library_screen.py`, dated comment.
- `Tests/Architecture/test_library_modules_size_ratchet.py` — `_BUDGETS`
  row raised for `library_rag_search_controller.py`, dated comment.
- `backlog/docs/library-decomposition-recipe.md` — §8 table row updated;
  new §18.
- `backlog/tasks/task-31203 - Library-decomposition-wave-3-combined-
  searchRAG-series.md` — AC#1-3 checked, Implementation Notes appended,
  status Done.
- `.git-blame-ignore-revs` — appended the cleanup commit's hash (follow-up
  commit).

## 14. Self-review

- **The dynamic-dispatch census found a real hazard this task's own first
  fix attempt walked straight into** (§1's `canvas_sync.py` finding):
  widening the census to the whole `tldw_chatbook/` tree found the site,
  but the obvious retarget was wrong — the shared dispatcher's `self`-
  forwarding shape means `screen` there is actually the controller, not
  `LibraryScreen`. The narrow sweep caught the resulting `AttributeError`
  before it reached a commit; the fix was to revert, not to patch further.
  Recorded honestly in the recipe (§18) as a near-miss, not smoothed over.
- **Name-collision risk was checked, not assumed** — the Console
  `_library_rag_query` hits were read at the receiver level before being
  excluded, not dismissed by string-match alone.
- **All retargets are receiver-path-only; zero assertion values changed**
  — verified per file, before/after, via the battery above.
- **The moved-body docstring fix is scoped exactly to what Task 3's
  ruling deferred** — no other docstring/comment inside the controller's
  own moved bodies was touched beyond the one ruled paragraph, preserving
  the byte-for-byte canon on everything else.
- Did not touch anything outside the cleanup PR's own scope: no method
  body was re-shaped, no logic changed, no new tests added beyond the
  wiring test's mechanical prune/absence pair.
- **One more pre-existing failure found and confirmed via `git stash -u`
  against the pre-task tree** (`Tests/UI/test_library_canvas_scoped_
  sync.py::test_notes_per_click_updates_keep_screen_and_canvas_identity`,
  a Notes-only canvas-identity test with no Search/RAG interaction),
  reproduces identically on both trees; added to the recipe's §7
  documented list rather than left for the next task to rediscover.

## 15. Fix round 1 (post-review)

Reviewer (opus) found the dead-import prune incomplete and two wording
inaccuracies in durable records. Commit `a150fc766`.

**Finding 1 (Important) — 9 more dead imports left in the same import
block.** Section 7 above claimed "5 dead imports removed" and stopped
there; it missed 9 more names in the SAME `Widgets.Library` import
block (`library_screen.py` ~487-497):
`library_rag_answer_children`, `library_rag_history_children`,
`library_rag_query_quiet_text`, `library_rag_query_shows_full_recovery`,
`library_rag_query_status_children`, `library_rag_results_body_children`,
`library_rag_scope_recovery_children`, `results_heading_text`,
`scope_toggle_label`. Re-verified each individually with a per-name
grep restricted to `library_screen.py`: every one appears ONLY on its
own import line (one, `library_rag_history_children`, also has a
backtick prose mention in a docstring at the time, ~line 22487 --
prose, not a code reference). The immediate neighbour
`library_rag_scope_shows_recovery` was correctly NOT touched -- it is
live, called from `_mirror_library_rag_scope_recovery`/
`_refresh_search_rag_panel_state_widgets` at (now) lines 42446/42522.
Cross-checked against `_SURFACE` in
`Tests/Architecture/test_library_support_layer_surface.py`: none of the
9 names appear there. Removed the 9-name block; re-measured with the
ratchet's own `_measure` (`ast`-based line/method count): 42949 -> 42940
lines, methods unchanged at 1304 (exactly -9 lines, one per deleted
import). `_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
re-pinned in the same commit, dated comment appended after the
original task-4 entry rather than rewriting it in place (matches the
`chat_screen.py`/task-22507.4 precedent of layering a fix-round comment
onto an existing entry rather than editing history).

**Finding 2 (minor) — retarget-count wording was wrong, not just
imprecise.** Section 2's "35 literal ... occurrences ... across 9
screen methods" is corrected to the actual numbers. Re-derived from
first principles rather than trusting either the original or the
reviewer's number: read `LibraryRagSearchState`'s 20 field names
directly from the dataclass (`library_rag_search_state.py`), built the
19 corresponding `_library_rag_<field>`/`_library_search_history` flat
names, and AST-walked the PRE-task-4 file (`git show
801c5375e:...library_screen.py`) for `self.<flat-name>` attribute
nodes inside `LibraryScreen`'s methods. Result: **66 occurrences across
11 methods** -- exactly matching a plain-text diff check on commit
`5bea63cdc` (66 removed `self._library_rag_*`/`self._library_search_
history` lines). The 11 methods are the union of the original report's
own 11-row table (§2 above) -- so the "9 methods" prose undercounted
against the report's OWN table the whole time, a second inconsistency
the reviewer's finding didn't separately flag but this fix corrects
too. Note: the reviewer's suggested parenthetical, "35 = distinct
fields touched," does not hold up -- the state object has only 20
fields total, so 35 distinct fields is impossible by construction; the
same AST census finds only 19 distinct field names touched by these 11
methods (one field, `history_refresh_lock`, is used only inside the
moved controller cluster). Both durable records (the `_BUDGETS`
comment and recipe §18's table) now state 66/11 without repeating the
unsupported "35 = distinct fields" explanation, and the `_BUDGETS`
comment says plainly that the original 35/9 was an undercount.

**Finding 3 (minor) — canvas_sync.py guard comment.** Added a one-line
(6-line, actually) comment directly above `screen._library_rag_answer_
render_key = None` at `canvas_sync.py:467` explaining why the flat
write is deliberate: both real callers of this `"search"` branch
(`LibraryRagSearchController.cycle_library_rag_mode`/
`toggle_library_rag_scope_source`, re-confirmed via a fresh `grep -rn
"_sync_library_canvas(" tldw_chatbook/` -- still exactly these two for
`kind == "search"`) forward the CONTROLLER as `screen`, and the
controller's own permanent shim exposes this flat setter; the
controller has no `_rag_search_state` attribute at all. Cites recipe
§18. `canvas_sync.py` is not governed by
`test_library_modules_size_ratchet.py` (verified: no entry for it in
that file's `_BUDGETS`), so no re-pin was needed there.

**Recipe update.** §18's per-task table (row 4) and pin-trajectory line
updated to append the fix round's numbers (`42949/1304 -> 42940/1304`),
following the existing convention (line ~580's conversations-series
"review-fix round added +24 lines" entry) of layering the correction
onto the existing row rather than rewriting the historical record.

**Battery**: `Tests/Architecture/test_screen_size_ratchet.py
test_library_modules_size_ratchet.py test_library_search_rag_wiring.py
Tests/UI/test_library_recompose_ratchet.py
test_library_support_layer_surface.py -p no:randomly -q` -- 49 passed,
2 failed (the same pre-existing `chat_screen.py` reds documented in
§12 of the original report, unrelated to this change).
`./scripts/preflight.sh`: all 6 checks green.

**Files touched**: `tldw_chatbook/UI/Screens/library_screen.py` (9
imports removed), `Tests/Architecture/test_screen_size_ratchet.py`
(wording correction + new dated re-pin comment + budget value),
`backlog/docs/library-decomposition-recipe.md` (§18 table row +
pin-trajectory line), `tldw_chatbook/UI/Library_Modules/canvas_sync.py`
(6-line guard comment, no logic change). Nothing else in the tree
touched.
