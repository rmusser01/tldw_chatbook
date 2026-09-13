# Task 6 report — Collections controller PR (collections series 2/3)

Branch `refactor/library-decomp-wave2-cold-trio`, worktree
`.worktrees/library-decomp-foundation`. Base commit `bca923b4c` (collections
state PR, series 1/3, complete).

**Fix round 1 (post-review, no production code changes)**: review verified
the engineering at full scale (64/64 byte-identical, 64/64 delegators
AST-faithful, zero test bypasses across the whole `Tests/` tree) but found
the RECORD needed 3 corrections, all applied below and marked inline where
they occur: (1) §10's clearing rationale falsely claimed none of the 12
branch-unique sweep failures touches Collections — one
(`test_closeout_single_app_route_cycle`) genuinely does, via two of this
task's own moved methods; corrected, and strengthened with a direct
base-commit (`bca923b4c`) rerun of that one test, confirming it passes
identically at both HEAD and base; (2) §4's dynamic-dispatch census
overclaimed a clean sweep — one moved method
(`retain_library_collection_quick_capture_input`) dispatches via a
dict-lookup-into-variable-into-`setattr` shape the census's literal-
argument grep missed; safe by construction (full get+set property
shims), but the census wording is corrected; (3) the two
`test_library_media_reader_flow.py` test ids in §10 are now named
explicitly. No commit landed at task-completion time changed
`.superpowers/` (gitignored, per `.gitignore:8`); the tracked-repo
correction (recipe `backlog/docs/library-decomposition-recipe.md`) landed
in a separate `docs(library): correct the task-6 sweep-triage record (fix
round 1)` commit.

## 1. Cluster enumeration (re-derived, not trusted from task 5's snapshot)

`ast` walk of `LibraryScreen` for method names containing `"collection"`
(case-insensitive): **67 methods** — matches task 5's own 2026-09-02
snapshot exactly. Of those, **3 are Prompts-owned, not Collections-owned**
(reconfirmed by reading each body, not by name): `handle_library_prompts_
collection` (`@on`), `_apply_library_prompt_collection`,
`_sync_library_prompt_collection_label` — the unrelated "Prompt
Collections" saved-prompt-grouping feature, using `_library_prompt_
collections_controller`/`_library_prompt_browse_controller`. Untouched by
this task (confirmed via `git diff bca923b4c -- library_screen.py`: no
diff lines touch any of the three).

**64 methods remain, and every one of the 64 moves onto the new
`LibraryCollectionsController`.** Unlike the export series (29 of 51
excluded across 3 rounds), this task found **zero exclusions** among the
64 — see §2 for the battery that confirmed this, not assumed it.

## 2. Exclusion-round battery (all rounds run, all clean)

- **`@work`/framework-decorator hazard**: a full `ast` decorator-list scan
  over all 64 candidates found zero `@work`-decorated methods. None
  excluded.
- **Class-level `monkeypatch.setattr(LibraryScreen, "<name>", ...)`**: a
  script-driven regex sweep over every `.py` file under `Tests/` (not just
  `Tests/UI`/`Tests/Library` — the whole tree) for both `monkeypatch.
  setattr(screen, "<name>", ...)` and `monkeypatch.setattr(LibraryScreen,
  "<name>", ...)`, all 64 names: zero hits.
- **Instance-attribute monkeypatch** (`screen.<name> = ...` on a real
  instance): same script, `screen\.(name)\s*=` pattern, all 64 names: zero
  hits.
- **Unbound fake-`self` / silent-Mock-auto-attribution** (`LibraryScreen.
  <name>(fake, ...)`): a repo-wide grep for every one of the 64 exact
  names as the literal string `LibraryScreen.<name>(`, across all of
  `Tests/`: zero hits. (The 17 unrelated `LibraryScreen.` references this
  grep surfaced are all for OTHER clusters — `_run_library_service_call`,
  `handle_library_prompts_sort_choice`, `_apply_library_rag_search_
  outcome`, `_run_library_export_via_service`, `_apply_library_export_
  success`, `LIBRARY_SHORTCUTS`/`BINDINGS`, etc. — none is one of our 64.)
- **Recipe §3's four known screen-routed monkeypatch names**
  (`_list_local_source_snapshot`, `_refresh_local_source_snapshot`,
  `_apply_local_source_snapshot`, `_refresh_library_note_detail`): none of
  the 64 is one of these four.
- **Dynamic `getattr`/`setattr` dispatch with an f-string or dict-literal
  argument** (recipe §11 lesson 3): a full grep across `tldw_chatbook/`
  found none targeting any Collections field or method name. The one
  PRE-EXISTING dynamic-dispatch site that DOES reach a Collections name
  — `_replace_library_reader_preference`/`_persist_library_reader_
  preference`'s 7-destination `{"collections": "_library_collections_
  reader_preferences", ...}` dict — resolves to the SCREEN's own
  `_library_collections_reader_preferences` property shim (installed by
  task 5, unaffected by this move), not to any method this PR moves. Not
  a hazard for this task.

**Already-extracted-wiring check (this task's own new question, per the
brief's flagged "browse-controller-delegation" concern)**: the brief named
`library_collections_browse_controller.py` as an existing, untouched
module whose delegating callers would be EXCLUSION candidates. **No such
file exists.** The only pre-existing Collections controller in
`Library_Modules/` is `library_collections_capture_controller.py`
(`LibraryCollectionsCaptureController` — a headless, generation-fenced
orchestration engine; NOT a Textual-adjacent screen controller). A
repo-wide grep for `collections_browse_controller`/
`CollectionsBrowseController` found only ONE hit outside a backlog task
file: `Tests/UI/test_product_maturity_phase39_library_collections.py::
test_collections_route_has_no_generic_container_controller_or_panel`,
which asserts the literal string `"LibraryCollectionsBrowseController"`
(note the different name) **never appears** in `library_screen.py` — a
retired-concept guard, not a live controller to check delegation against.
The new controller is named `LibraryCollectionsController` (matching the
`LibraryCollectionsState`/`LibraryExportController`/`LibraryExportState`
naming convention), which does not collide with that guard; the test
still passes post-move (§7).

Checked instead: does any of the 64 candidates ALREADY delegate (as a
one-line forward) to `LibraryCollectionsCaptureController` or to any
other already-existing controller, making it dead-on-arrival for a
full-body move? **No.** All 64 are real, full-bodied `LibraryScreen`
methods pre-move (verified by reading every one; none is a bare
`return self.<something>.<name>(...)` one-liner). 28 of the 64 reference
`self._library_collections_capture_controller` (the headless engine) as a
collaborator — building requests, calling `controller.load_page`/
`select_item`/`scope_service.<op>`, etc. — but each such body still
carries its own request-building, validation, status-line, and
recompose-scheduling logic; the headless engine is a data/business-logic
dependency, not a wiring shortcut. None was excluded on this basis.

**Verdict: single controller, all 64 methods, zero exclusions.**

## 3. Split-or-single-controller decision

**Single controller** (`LibraryCollectionsController`), per the task's own
default ("when unsure, one controller"). Considered the two candidate
seams the brief and the conversations-exemplar precedent suggest
(capture/creation vs. reader/detail-actions), but rejected: every method
in "reader/detail" territory (mode switch, highlights, notes, content
actions, archive/delete/favorite) and every method in "capture/browse"
territory (rail scope/filter/sort/paging, quick-capture) shares the SAME
dependency shape — one `LibraryCollectionsCaptureControllerState` object,
one `_refresh_library_collections_capture_reader` recompose gate, and
cross-calls in both directions (`set_library_collection_capture_mode`
(reader-side) calls `_load_library_collection_capture_highlights`
(reader-side) but `_page_library_collection_captures`
(browse/paging-side) calls the same `_run_library_collections_capture_
transition` helper detail actions also use). Unlike the conversations
exemplar's Reader-vs-Browse split (genuinely disjoint state/concerns), no
clean seam exists here; splitting would produce two controllers each
needing to call back into the other for shared helpers, which is exactly
the "canon-inexpressible tangle" shape the task brief says to avoid.

## 4. Dynamic-dispatch census (method + field names, confirmed before moving)

See §2's dynamic-dispatch bullet for the one pre-existing reader-
preferences dict hit (not a hazard).

**Correction (fix round 1, post-review): the original version of this
section claimed no `setattr`/`getattr` dispatch with a dict-literal or
f-string argument touches a Collections name anywhere in `tldw_chatbook/`.
That was false — the census script's grep pattern only matched a
dict-literal/f-string passed DIRECTLY as the `getattr`/`setattr` call's
own argument, and missed the two-step shape where a dict literal is used
to look up a name into a local VARIABLE first, and that variable is
passed to `setattr` on a later line.** One of the 64 moved methods,
`retain_library_collection_quick_capture_input`
(`library_collections_controller.py`, ~lines 733-750), does exactly this:

```python
attributes = {
    "library-collections-capture-url": "_library_collections_quick_capture_url",
    "library-collections-capture-title": "_library_collections_quick_capture_title",
    "library-collections-capture-tags": "_library_collections_quick_capture_tags",
}
attribute = attributes.get(event.input.id or "")
if attribute is not None:
    setattr(self, attribute, event.value)
```

This dynamically dispatches to one of three Collections state-field
names, resolved from a DOM-id-keyed dict rather than an inline literal.
**It is safe**: all three target names (`_library_collections_quick_
capture_url`/`_title`/`_tags`) are among the 26 fields exposed by this
controller's own generated state-shim loop, and every shim in that loop
is a full `property(get, set)` pair (not read-only) — so
`setattr(self, "_library_collections_quick_capture_url", value)` resolves
through the property SETTER exactly as `self._library_collections_
quick_capture_url = value` would if the name were written literally.
`self` here is the CONTROLLER (this method was moved as part of this
task, byte-for-byte, so `self` inside its body is now `LibraryCollections
Controller`, not `LibraryScreen`) — the dispatch target and the dispatch
site moved together, so no cross-module resolution gap was introduced by
the move itself.

Corrected census statement: no `getattr`/`setattr` dispatch using an
f-string or a dict-literal ARGUMENT DIRECTLY IN THE CALL touches a
Collections name outside the one pre-existing reader-preferences hit
(§2) — but at least one dispatch using a dict-literal to populate an
intermediate VARIABLE, then passed to `setattr` separately, DOES exist
inside the moved cluster itself (the example above), and is safe by
construction rather than by the census's own literal-pattern match. A
future subsystem's census should grep for `\.get\(` results assigned to a
variable that flows into a `setattr`/`getattr` call within the same
function, not just an inline literal argument, to avoid re-missing this
shape.

## 5. Bind-list classification (the two binding kinds, recipe §1)

**Framework services** (live-read `@property`, six total, matching the
export/conversations precedent exactly): `app_instance`, `app`,
`call_after_refresh`, `is_mounted`, `query_one`, `refresh`.

**Named constructor dependencies** (derived by an AST script extracting
every `self.<attr>` reference across all 64 moved bodies and subtracting
the cluster's own method names and its own 26 state-field names — not a
hand list):

- `library_adaptive_reader_allocation_is_current` — general Library-wide
  shell helper (shared with Notes/File Notes/Media), used once (`_sync_
  library_collections_reader_layout_from_shell`).
- `library_selected_row_id_accessor` — the recipe's own canonical
  >=2-subsystems field (226 refs). An AST Store-context scan over all 64
  bodies confirms zero writes; read-only accessor bound (used once, in
  `_refresh_library_collections_capture_reader`).
- `library_collections_capture_controller_accessor` +
  `set_library_collections_capture_controller` — a GET+SET pair (not a
  read-only accessor) for `_library_collections_capture_controller`, the
  ONE field task 5 deliberately kept OFF `LibraryCollectionsState`
  ("wiring, not state"). An AST Store-context scan confirms exactly one
  moved body (`_ensure_library_collections_capture_controller`) writes
  it, hence the setter; 28 bodies read it.

**Own state**: all 26 `LibraryCollectionsState` fields, exposed via a
generated property loop reading `self._collections_state_accessor().
<field>` — identical generator shape to the export controller's own
block, single `_library_collections_` prefix (task 5: no field needed a
plural variant). No `_safe_text` class-binding needed: no moved body calls
`self._safe_text(...)`.

## 6. Byte-for-byte verification — method and result

**Method**: an `ast`-based script extracted, for the pre-move file, each
of the 64 methods' exact source segment (decorator(s) through
`end_lineno`, using the original file's own line offsets — not a
hand-retyped copy) and reassembled the new controller module by
concatenating a hand-written header (docstring/imports/`__init__`/
properties), the 64 extracted segments UNCHANGED, and a hand-written
footer (the generated state-shim loop). A second script then re-parsed
BOTH the original `library_screen.py` (at `HEAD`, pre-move) and the new
controller module, matched each of the 64 names' `FunctionDef`/
`AsyncFunctionDef` nodes in both files, and asserted the extracted source
text (decorator line(s) through `end_lineno`) was byte-for-byte identical
in both.

**Result**: `ALL MATCH byte-for-byte` — ("Checked 64 methods... New
controller has 74 methods total... ALL MATCH byte-for-byte", the 10 extra
being `__init__` and the 9 property accessors, exactly as expected). No
method body was hand-transcribed at any point in this task, eliminating
transcription-error risk for a move of this size.

## 7. Free-name resolution walk

A second AST script walked every one of the 64 moved bodies inside the
FINISHED controller module, collecting every `Name` node in `Load`
context, and checked each against: Python builtins, the module's own
imports/top-level defines, the function's own parameters, and every name
locally bound inside the function (assignments, `for`/`with`/`except`
targets, comprehension targets, walrus, nested lambda params). **Zero
unresolved free names** across all 64 methods (one apparent hit,
`_library_collections_capture_controller` referenced inside its own
`@_library_collections_capture_controller.setter` decorator, is a false
positive from the checker walking the decorator expression — this is the
ordinary property-getter/setter pattern, confirmed live: `LibraryCollections
Controller.__dict__['_library_collections_capture_controller']` is a
`property` with both `fget` and `fset` set, and `_ensure_library_
collections_capture_controller` — which reads AND writes it — is exercised
by `Tests/UI/test_library_collections_capture_reader.py`, all passing).

## 8. `LibraryCollectionsController`

`tldw_chatbook/UI/Library_Modules/library_collections_controller.py`
(new, ~1690 lines): the 64-method cluster, constructor + 6 framework
properties + 3 named-dependency properties (the wiring GET+SET pair
counts as one dependency, exposed as 2 properties: getter + setter) + the
generated 26-field state-shim loop. Constructed in `LibraryScreen.__init__`
as `self._collections_controller`, immediately after `self._export_
controller` (matching the export/conversations precedent's construction
position), before the shared reader-preferences tuple-unpack (position
doesn't matter here since every dependency is a deferred lambda, not an
eagerly-read value).

## 9. Screen delegators

All 64 original names became one-line delegators, generated
programmatically (not hand-typed) from each method's own AST signature:
non-static delegators `return [await ]self._collections_controller.
<name>(<forwarded args>)`; the cluster's one `@staticmethod`
(`_restore_library_collections_page`) forwards to the class:
`return LibraryCollectionsController._restore_library_collections_page(
state)`. Every `@on(...)` decorator (including the two multi-line,
multi-selector ones) was copied verbatim, byte-for-byte, from the
original decorator source lines — not reformatted or reflowed. Argument
forwarding respects parameter kind: positional-or-keyword params forward
positionally by name; keyword-only params (`_library_collections_capture_
request`'s `page`/`search`, `_library_collection_capture_filter_
request`'s `clear`) forward as `name=name`. No method in the cluster uses
`*args`/`**kwargs`/positional-only params (confirmed by an AST scan before
generating), so no forwarding edge case was needed.

## 10. Verification battery

**Wiring RED -> move -> GREEN** (separate commits, per the new standing
rule from task 5's review):

- RED commit (`806cfea6f`): added `library_collections_controller.py`
  (complete, self-contained — the meaningful gate is the SCREEN's own
  delegation, not the controller module's existence, per task 5's own
  documented precedent for this exact nuance) plus 4 new tests to
  `Tests/Architecture/test_library_collections_wiring.py`. Confirmed RED:
  `2 failed, 3 passed` — `test_screen_delegates_collections_handlers` and
  `test_collections_cluster_staticmethods_forward_to_the_controller_class`
  fail (the screen still carries the original 64 bodies); the pre-existing
  state-shim test plus the 2 new controller-only-shape tests
  (`test_collections_controller_owns_its_cluster`, `test_collections_
  controller_exposes_every_state_field`) already pass since the controller
  class is fully built and self-contained.
- Move commit (screen edit): `5 passed` — all 5 wiring tests green.

**Characterization file (`Tests/UI/test_library_collections_
characterization.py`, task 5's 17 pins)**: still green post-move (part of
the 26-test combined run, §10 below).

**Size ratchet — ceiling AND slack, fresh post-move measurement, lowered
in this same commit**: `42486` lines / `1281` methods (measured via the
ratchet's own `_measure` semantics — `len(source.splitlines())` — not
`wc -l`, confirmed identical for this file). Pin trajectory:
`43410 -> 42486` (methods unchanged: 64 `FunctionDef`s out, 64 one-line
delegators in — pure move). Both `test_screen_does_not_grow_past_its_
budget[library_screen.py]` and `test_budget_is_not_left_slack_after_a_
wave[library_screen.py]` pass; the two pre-existing `chat_screen.py`-
scoped failures in the same file remain (recipe §7's documented list,
unrelated to this diff).

**Recompose ratchet + slack guard + support-layer surface**: `Tests/UI/
test_library_recompose_ratchet.py` (6 tests) and `Tests/Architecture/
test_library_support_layer_surface.py` (8 tests, incl. `test_no_import_
cycle`) all pass — this move touches zero `refresh(recompose=True)` call
sites (pure body relocation), so the recompose census pin is unaffected.

**Export + conversations wiring/characterization regressions +
collections characterization**: `Tests/Architecture/test_library_export_
wiring.py`, `Tests/Architecture/test_library_conversations_wiring.py`,
`Tests/UI/test_library_export_characterization.py`, `Tests/UI/
test_library_conversations_characterization.py`, `Tests/UI/test_library_
collections_characterization.py` — **26 passed**.

**Collections-adjacent live-functional suites** (capture controller,
capture reader, reader geometry, the phase-39 cutover-contract guard):
`Tests/UI/test_library_collections_capture_controller.py`, `Tests/UI/
test_library_collections_capture_reader.py`, `Tests/UI/test_library_
collections_reader_geometry.py`, `Tests/UI/test_product_maturity_phase39_
library_collections.py` — **43 passed**, including the
`"LibraryCollectionsBrowseController" not in source` guard (§2).

**`-k "collection and library"` with pristine-baseline comparison** (swept
BOTH `Tests/UI` and `Tests/Library`): branch **361 passed, 3 failed**
(`test_library_starter_deep_link_opens_hidden_collection_or_note_route`,
`test_library_landing_continue_receipt_accepts_only_authoritative_source_
scopes[browse-collections-expected_scope4]`, `test_get_library_collection_
supported_types_round_trip_public_ids`) — all 3 match the recipe's own
documented pre-existing list from task 5 exactly (same names). Baseline
(pristine `bca923b4c` tree — checked out **in a separate scratch git
worktree** rather than `git stash -u`, since a stash on this branch would
have collided with the RED commit already landed; same evidentiary effect,
a clean pre-task tree — invoked as `PYTHONPATH=<baseline-worktree>
<this-worktree's venv python> -m pytest`, confirmed to resolve `tldw_
chatbook` from the baseline worktree before running): **360 passed, 4
failed** — the same 3 plus ONE extra, `Tests/UI/test_library_prompt_
collections.py::test_library_screen_membership_load_retry_and_apply_
retry_are_distinct` (a Prompts "Prompt Collections" test — the unrelated
feature this move's 3 Prompts-exclusion methods belong to, sharing only
the English word "collection" with our cluster; this move's diff touches
none of that test's code path). **Zero branch-unique failures** — the
diff runs strictly in the OTHER direction (one failure present only on
the pristine baseline, absent on the branch), the same "noise in the
opposite direction" shape both the export and collections-state sweeps
already documented. Both runs were slowed by CPU contention (~460s each,
vs. normal single-digit seconds) from running concurrently with the two
full xdist sweeps below; wall-clock time does not affect single-process
pytest correctness.

**Full xdist paired-baseline sweep** (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`), run CONCURRENTLY (branch in this
worktree, baseline in the same scratch git worktree used for the narrow
check above, both with `-n 8`) — a deviation from running them
sequentially, which produced noticeably higher failure counts than this
recipe's historical range on both sides (mutual CPU contention from 16
simultaneous xdist workers, not a code issue):

- Branch: **349 failed, 3890 passed** (1542.63s / 25:42).
- Baseline (pristine `bca923b4c`): **344 failed, 3895 passed** (1592.93s /
  26:32).
- Diff (`comm` on sorted `FAILED ...` line sets): **12 failures unique to
  branch**, **7 unique to baseline only**, the remainder (332) shared.

**Every one of the 12 branch-unique failures was re-run**, single-process
(no xdist), on the BRANCH tree, both individually (combined in one
invocation) and — for the one that reproduced — in true isolation:
`test_library_adaptive_reader_closeout.py::test_closeout_single_app_
route_cycle`, `test_library_canvas_scoped_sync.py::test_real_prompt_and_
skill_rows_keep_their_canvas_identity`, `test_library_entry_compose_
once.py::test_export_counts_leave_return_same_scope_rejects_older_
request`, `test_library_file_notes_workspace.py::test_conflict_
resolution_discard_keeps_cancel_first_confirmation`,
`test_library_media_reader_flow.py::test_edit_metadata_from_read_routes_
to_info_form_actions` and `test_library_media_reader_flow.py::
test_info_calls_empty_content_metadata_only_like_console`,
`test_library_media_reader_match_nav_
t22209.py::test_match_navigation_takes_no_document_pass_per_click`,
`test_library_media_reader_no_change_sync_t22208.py::test_no_change_
traversal_builds_no_preview_and_copies_no_content`, `test_library_media_
reader_traversal_t22207.py::test_one_megabyte_markdown_document_is_not_
reparsed_per_keystroke`, `test_library_media_trash.py::test_media_trash_
entry_requests_one_independent_initial_page`, `test_library_prompt_
collections.py::test_library_screen_membership_load_retry_and_apply_
retry_are_distinct` (the SAME Prompts test already found flipping in the
narrow-sweep comparison above — a second independent confirmation it is
pure noise, not attributable to either tree), and `test_library_prompts_
canvas.py::test_library_prompts_settlement_keeps_newer_surviving_focus`.
**11 of 12 passed cleanly** in that combined single-process run. **The
1 that reproduced** (`test_one_megabyte_markdown_document_is_not_
reparsed_per_keystroke`) was checked further: (a) run in TRUE isolation
(alone, nothing else in the same pytest session) on the branch — passed;
(b) the SAME 12-test combined invocation re-run against the PRISTINE
BASELINE tree — reproduced the identical failure there too (`test_one_
megabyte_markdown_document_is_not_reparsed_per_keystroke` failed on
BOTH trees under the same combined-run conditions, plus a second test,
`test_no_change_traversal_builds_no_preview_and_copies_no_content`, which
had passed in the branch's own combined run — a different subset flaking
each time). This is decisive: the failure is a shared-state/ordering
sensitivity to WHICH OTHER TESTS ran earlier in the same process,
identical on both code versions, not a consequence of this task's diff
(a Media-reader-cluster test; this move touches zero Media-reader code).
**Zero of the 12 branch-unique failures are real regressions.** The 7
baseline-unique failures are better-on-branch noise in the opposite
direction, not attributable to this task either.

**Correction (fix round 1, post-review): the original version of this
paragraph claimed none of the 12 branch-unique names touches Collections
at all. That was false and has been replaced.** One of the 12,
`test_library_adaptive_reader_closeout.py::test_closeout_single_app_
route_cycle`, DOES touch Collections: its own `DESTINATION_CONTRACT`
dict includes a `"collections"` entry (`#library-row-browse-collections`,
`#library-collections-reader-shell`, `#library-collections-row-1`,
`_library_collections_reader_preferences`, `_library_collections_reader_
layout`), and the test cycles every destination in that contract --
which, for "collections", traverses the shared reader-shell dispatcher
(`_sync_library_reader_preference_layout`) that calls two of THIS task's
own moved-and-delegated methods: `_sync_library_collections_reader_
layout_from_shell` and `_mirror_library_collections_reader_preference`
(screen delegators at `library_screen.py:6907`/`6921`, forwarding to
`self._collections_controller`). The test's OWN pass/fail result above
(passed cleanly, both in the 12-test combined run and again in true
isolation) already covered this path — the error was only in the
CHARACTERIZATION sentence claiming no overlap existed, not in whether the
path was exercised.

**Strengthened clearing evidence, added in this fix round**: ran this one
test, in true isolation, against the PRE-MOVE tree by temporarily
swapping `tldw_chatbook/` to base commit `bca923b4c` inside this worktree
(`git checkout bca923b4c -- tldw_chatbook`, confirmed via `git status`/
`git diff --stat` that only `library_screen.py` changed and the swap
covered the full pre-controller-move state), then restoring
(`git checkout HEAD -- tldw_chatbook`, confirmed `git status` clean
afterward, so no stray checkout state survives this report). Result:
**passes identically at both HEAD (branch) and base (`bca923b4c`)** --
`1 passed` in ~10-13s either way, no assertion difference. This is a
DIRECT, single-test-scoped pre-existing-vs-branch comparison (not an
inference from "the name doesn't mention Collections," which was the
flawed original method) and it independently confirms the same
conclusion the 12-test combined re-run already gave: this failure is
xdist ordering/shared-state flakiness from the full sweep, not a
regression from the Collections controller move -- even though, unlike
the other 11, this ONE test's own code path genuinely does exercise two
of the moved methods.

The other 11 branch-unique names remain accurately described as NOT
touching Collections (Media reader/trash, Notes workspace, Prompts
canvas/collections-the-other-feature, entry-compose-once) -- only
`test_closeout_single_app_route_cycle` (adaptive-reader-shell closeout,
which cycles every destination including Collections) and
`test_real_prompt_and_skill_rows_keep_their_canvas_identity`
(canvas-scoped sync; scoped to Prompts/Skills canvases, not Collections)
warranted this correction; the canvas-sync one does not touch Collections
and needed no re-check.

**Preflight** (`./scripts/preflight.sh`): all six checks green (CSS
bundle, profile-owned-path census, diagnostic inventory, backlog task
ids, chachanotes table allowlist, index plan pins) — run once, after the
move commit, before writing this report.

## 11. Dead imports found, deliberately left for the cleanup PR

Per the export series' own Task 3/Task 4 split (dead-import removal is a
cleanup-PR activity), 4 more names became genuinely dead in
`library_screen.py` as a direct result of this move (checked via exact
occurrence count — 1 remaining occurrence each, the import line itself):
`CAPTURE_SORTS`, `CaptureSaveRequest`, `ExternalNoteReference`,
`CollectionsCaptureError`. Left in place. (`CollectionsCaptureController
State`, already flagged dead by task 5's own report for a different
reason, remains dead — still 1 occurrence, the import line.) Three other
names moved bodies used remain genuinely alive in `library_screen.py`
because a MOVED method's own signature/return-type annotation still
carries them (the delegator's header is byte-for-byte the original
signature): `CollectionsCaptureReaderPresentation` (return type of
`_library_collections_capture_presentation`'s delegator),
`CaptureIdentity` (2 delegator signatures), `CapturePageRequest` (3
delegator signatures). These should NOT be treated as dead by task 7's
cleanup census — they are load-bearing for the SCREEN's own delegator
type hints, not leftover.

## 12. Files changed

- `tldw_chatbook/UI/Library_Modules/library_collections_controller.py`
  (new) — `LibraryCollectionsController`, 64 methods.
- `tldw_chatbook/UI/Screens/library_screen.py` (modified) — 1 import
  added; `self._collections_controller` constructed in `__init__` right
  after `self._export_controller`; all 64 method bodies replaced with
  one-line delegators (decorators/signatures preserved verbatim).
- `Tests/Architecture/test_library_collections_wiring.py` (modified) — 4
  new tests added (RED commit), all green post-move.
- `Tests/Architecture/test_screen_size_ratchet.py` (modified) —
  `_BUDGETS` row lowered to `42486/1281`.
- `backlog/docs/library-decomposition-recipe.md` (modified) — new §14
  ("collections series, task 2 (controller PR) — as landed") plus a new
  §7 documented-pre-existing-failures entry for this task's sweep
  findings.
- `.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/
  task-6-report.md` (new) — this report.

## 13. Self-review

- Cluster re-derived mechanically at execution time (67 -> 64 after the
  Prompts exclusion), not trusted from task 5's snapshot — matches
  exactly, but re-derivation (not the count alone) is what the recipe
  requires.
- Every exclusion round from the recipe's own catalogue (class-level
  monkeypatch, instance-attribute monkeypatch, unbound-fake-self/silent-
  Mock, `@work`/DOMNode assertion, recipe §3's 4 known names) was run as
  an actual script/grep against Tests/ and tldw_chatbook/, not asserted
  from precedent, and all came back clean — zero method-level TEST-BYPASS
  exclusions, recorded as a genuine finding, not skipped because
  "collections looked simple." The dynamic getattr/setattr dispatch round
  is the ONE exception: the census script's literal-argument grep missed
  a two-step dict-lookup-into-variable-into-setattr shape inside the
  cluster itself (`retain_library_collection_quick_capture_input`,
  §4's fix-round-1 correction) — safe by construction (the dispatched
  names are full get+set property shims), but the ORIGINAL claim that
  this round was clean was wrong until corrected in review.
- The brief's named "browse controller" turned out to be a naming
  mismatch for the docs/brief itself (`library_collections_capture_
  controller.py` is the only real prior controller, and it is a headless
  orchestration engine, not a delegation target for our cluster) —
  resolved by reading the actual guard test
  (`test_collections_route_has_no_generic_container_controller_or_panel`)
  rather than guessing at a file that doesn't exist.
- Byte-for-byte canon enforced by TOOLING (an AST-based extract-and-
  compare script), not by manual diff-reading of a 1300-line transcription
  — eliminates the main risk of a move this size.
- Free-name resolution walk run as a script against the FINISHED
  controller module (not skipped as "the tests will catch it") — found
  one false positive, traced to ground truth (a live property object with
  both accessors present), not waved away.
- Size ratchet measured fresh, post-edit, via the ratchet's own `_measure`
  semantics (not `wc -l` alone, though the two agreed here) — lowered in
  this same commit, per recipe §6's explicit lesson.
- Sweep evidence: both full xdist sweeps completed (branch 349f/3890p,
  baseline 344f/3895p — both above this recipe's historical ~330-340
  backdrop due to running the two sweeps CONCURRENTLY, a deliberate
  wall-clock-saving deviation this report documents rather than hides).
  12 branch-unique failures were individually AND combined re-run
  single-process: 11 passed cleanly; the 1 that reproduced
  (`test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke`)
  was confirmed pre-existing by (a) passing in true isolation and (b)
  reproducing identically on the PRISTINE baseline under the same
  combined-invocation conditions — not attributable to this task's diff,
  and not a Collections test. Zero real regressions. Recorded in the
  recipe's §7 documented list and its own new §14 forward note (prefer
  sequential sweeps, not concurrent, for future tasks).
- The recipe doc (`backlog/docs/library-decomposition-recipe.md`) was
  updated with a new §14 ("collections series, task 2 (controller PR) —
  as landed") plus a §7 documented-failures append, matching every prior
  task's own practice of feeding forward what this task's execution
  found.
