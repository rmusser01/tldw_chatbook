# Task 5 report — Collections series 1/3: characterization spot-check + LibraryCollectionsState

Branch `refactor/library-decomp-wave2-cold-trio`, worktree
`.worktrees/library-decomp-foundation`. Base commit `58118128c` (export
series complete: state/controller/cleanup + blame-ignore follow-ups).

## 1. Cluster enumeration

`ast` walk of `LibraryScreen` for method names containing `"collection"`
(case-insensitive): **67 methods** — matches the wave-2 plan's snapshot
exactly (re-derived at execution time).

Of those 67, **3 are Prompts-owned, not Collections-owned**:
`handle_library_prompts_collection`, `_apply_library_prompt_collection`,
`_sync_library_prompt_collection_label` — all use
`_library_prompt_collections_controller`/`_library_prompt_browse_controller`
and implement the unrelated "Prompt Collections" feature (a saved-prompt
grouping concept, nothing to do with Library Collections/captures).
Confirmed by reading each body, not by name — the recipe's own documented
substring-match trap (§2's "startswith enumeration trap", §11's lesson,
generalized here from fields to methods exactly as the export series'
Task 3 report generalized it for its own 51-name census).

Of the remaining 64 genuinely Collections-named methods, **42 carry a
distinct `@on` decorator**; 41 once the one Prompts false-positive
(`handle_library_prompts_collection`) is excluded. This task's scope is
the state PR only (fields + characterization spot-check); the full
method-ownership census (which of the 64 genuinely move to a controller)
is deferred to this series' controller PR, per the export series' own
Task 2/Task 3 split.

## 2. Characterization decision table

All 41 `@on`-bound Collections selectors were checked with a per-id
`grep -rn "<id>" Tests/` followed by a manual read of the surrounding
lines for an actual `.press()`/`.click()`/direct-`.value`-assignment/
`Input.Submitted` interaction — not a same-line-only grep, which
undercounts (one selector, `retry_library_collection_quick_capture`,
`#library-collections-capture-retry-confirm`, looked unpressed under a
same-line check but is genuinely covered: the press is on the line AFTER
its `query_one` call in
`test_unknown_quick_capture_preserves_draft_and_does_not_auto_retry`).

| Selector / handler | DOM-driven? | Disposition |
|---|---|---|
| `#library-collections-quick-capture`, `-capture-save`, `-capture-cancel`, `-capture-retry-back`, `-capture-retry-confirm` | Yes (`test_library_collections_capture_reader.py`, `test_unknown_quick_capture_preserves_draft_and_does_not_auto_retry`) | Covered |
| `#library-collections-capture-url`/`-title`/`-tags` (`Input.Changed`), `-capture-note` (`TextArea.Changed`) | Yes (direct `.value =`/`.text =` assignment through mounted widgets, exercised in `test_quick_capture_draft_survives_background_reader_recompose` and the walkthrough) | Covered |
| `#library-collections-page-next`, `-page-previous` | Yes (`test_library_collections_capture_walkthrough.py`) | Covered |
| `#library-collections-more`, `-summarize`, `-listen`, `-save-offline`, `-mode-{read,highlights,notes,info}`, `-highlight-save`, `-freeform-note-save`, `-linked-note-save` | Yes (`test_real_local_capture_actions_persist_reader_results`) | Covered |
| `#library-collections-archive`, `-archive-undo`, `-hard-delete`, `-hard-delete-confirm`, `-retry-extraction`, `-legacy-recovery` | Yes (`test_library_collections_capture_walkthrough.py`) | Covered |
| `#library-collections-filters-apply` | **No** — only a `query_one` existence assertion | **PIN** |
| `#library-collections-filters-clear` | **No** — only a `query_one` existence assertion | **PIN** |
| `#library-collections-filters` (toggle) | **No** — zero references anywhere (only the `-apply`/`-clear` siblings are mentioned) | **PIN** |
| `#library-collections-sort` (cycle) | **No** — only a `query_one` existence assertion | **PIN** |
| `#library-collections-filter` (`Input.Submitted`) | **No** — only a `query_one` existence assertion | **PIN** |
| `.library-collections-item-row` (select) | **No** — every existing test either counts/reads rows or selects via a direct, unbound-of-the-button `await screen._select_library_collection_capture(identity)` call | **PIN** |
| `.library-collections-scope-row` (built-in scope) | **No** — only a count/label assertion | **PIN** |
| `#library-collections-page-retry` | **No** — only a `query_one` existence assertion (against a permanently-broken fixture) | **PIN** |
| `#library-collections-reader-retry` | **No** — zero references anywhere | **PIN** |
| `#library-collections-favorite` | **No** — zero references anywhere | **PIN** |
| `#library-collections-mark-read` | **No** — zero references anywhere | **PIN** |
| `#library-collections-open-original` | **No** — zero references anywhere | **PIN** |
| `.library-collections-highlight-delete` | **No** — zero references anywhere (its `save` sibling is pressed by `test_real_local_capture_actions_persist_reader_results`) | **PIN** |
| `.library-collections-linked-note-unlink` | **No** — zero references anywhere (its `link` sibling is covered) | **PIN** |
| `#library-collections-capture-refresh` | **No** — only `.disabled` assertions | **PIN** |
| `#library-collections-hard-delete-cancel` | **No** — zero references anywhere (its `arm`/`confirm` siblings are pressed by the walkthrough) | **PIN** |
| `#library-collections-legacy-recovery-close` | **No** — zero references anywhere (its `inspect` sibling is pressed by `test_library_export_characterization.py`'s own legacy-recovery-export pin) | **PIN** |
| `#library-collections-legacy-recovery-export` | (Export-owned selector; already pinned by `test_library_export_characterization.py`, not re-pinned here) | N/A |

**17 pins written** into `Tests/UI/test_library_collections_characterization.py`,
grouped into 5 test functions by shared setup (mirroring this codebase's
own walkthrough-style tests, which routinely exercise several related
handlers inside one continuous Pilot session rather than one session per
handler): filters/sort/free-text (5 handlers), item/scope row selection
(2), page/detail retry (2), detail actions — favorite/mark-read/
open-original/highlight-delete/note-unlink (5), and quick-capture-refresh/
hard-delete-cancel/legacy-recovery-close (3). All 17 pass against current
code pre-move (inverted TDD confirmed — see §6). No live bugs found; all
17 gaps are coverage gaps, not behavior bugs. The remaining 24 covered
`@on` handlers and the 23 non-`@on` private Collections helpers are
reached transitively by the same well-covered flows and were not
individually re-pinned (mirrors the export series' identical blanket
finding).

**A new test-authoring gotcha found while writing these pins** (not a
production bug): pressing a button immediately after a prior async
transition's `_wait_for_condition` resolves true can race that
transition's own trailing recompose —
`_run_library_collections_capture_transition` schedules a SECOND
`_refresh_library_collections_capture_reader()` only after `await task`
completes, and a condition watching `controller.state` can observe the
mutation before that second recompose has rebuilt the DOM. Symptom:
`NoMatches` on a selector that should exist, or a 30s `_wait_for_selector`
timeout on one that should appear instantly. Fixed by adding a few extra
`await pilot.pause()` calls after the setup condition, before touching
DOM elements that depend on it. Two of the five test functions needed
this fix before passing reliably (3 consecutive clean runs each). Full
details recorded in the recipe (§13) for the next subsystem's
characterization-writing task.

## 3. Ownership table (recipe §2 script, `_library_collections` prefix)

Script output: **27 `__init__`-scoped fields** (not ~28 as the wave-2 plan
estimated). Unlike Export's `origin_row_id`, a full class-level `AnnAssign`
scan found **zero** collections-owned class-level-only attributes, so no
field was missed by the `__init__`-scoped census this time.

| Field | Non-collection `__init__` users | Verdict |
|---|---|---|
| `capture_capabilities` | NONE | MOVE |
| `saved_searches` | NONE (see §4 census below) | MOVE |
| `saved_searches_total` | NONE (see §4 census below) | MOVE |
| `active_scope` | NONE | MOVE |
| `requested_page` | `restore_state` (shell/plumbing) | MOVE |
| `reader_mode` | NONE | MOVE |
| `highlights` | NONE | MOVE |
| `quick_capture_open` | NONE | MOVE |
| `quick_capture_url` | NONE | MOVE |
| `quick_capture_title` | NONE | MOVE |
| `quick_capture_tags` | NONE | MOVE |
| `quick_capture_note` | NONE | MOVE |
| `save_outcome_unknown` | NONE | MOVE |
| `confirming_save_retry` | NONE | MOVE |
| `quick_capture_saving` | NONE | MOVE |
| `filters_open` | NONE | MOVE |
| `more_open` | NONE | MOVE |
| `confirming_hard_delete` | NONE | MOVE |
| `legacy_recovery_rows` | NONE | MOVE |
| `legacy_recovery_open` | NONE | MOVE |
| `legacy_recovery_lines` | NONE | MOVE |
| `action_status` | NONE | MOVE |
| `action_content` | NONE | MOVE |
| `reader_layout` | `_toggle_library_media_reader_pane` (a multi-subsystem shell dispatcher — name contains "media" as a substring but branches generically on `_library_selected_row_id` across Collections/Conversations/Notes/Media; not Media-owned), `compose_content` (shell/plumbing) | MOVE (entangled — original line kept, per §1) |
| `reader_persistence_locks` | `_persist_library_reader_preference` (shell/plumbing, the 7-destination shared dispatcher) | MOVE (entangled — original line kept) |
| `reader_preferences` | `request_library_reader_layout_refresh` (shell/plumbing) | MOVE (entangled — original line kept) |
| `capture_controller` | `on_unmount`, `save_state`, `_library_continue_receipt_for_current_route`, `apply_navigation_context`, `_build_library_shell_input`, `_select_library_rail_row_after_source_admission` (all shell/plumbing) | **WIRING, NOT STATE** — holds a live `LibraryCollectionsCaptureController` instance (the `_conversation_reader_controller` precedent); stays a plain `LibraryScreen` attribute, untouched |

**Total: 26 fields moved into `LibraryCollectionsState`, 1 field
(`capture_controller`, wiring) stayed on the screen, 0 fields BLOCKED.**
No ≥2-subsystems sharing was found for any field — every non-collection
consumer found by the script is shell/plumbing or a generically-named
multi-subsystem dispatcher (`_toggle_library_media_reader_pane`,
`_persist_library_reader_preference`), never another subsystem's OWN
exclusive logic.

## 4. The saved-searches census — the brief's flagged contested boundary

The brief flagged `_library_collections_saved_searches` /
`_library_collections_saved_searches_total` as contested between
Collections and "the search cluster" and asked for a consumer census
before deciding.

**Consumer census** (every `self._library_collections_saved_searches*`
occurrence, repo-wide):

- `library_screen.py:2340` (was `2349` pre-move) — `__init__` default
  assignment.
- `_library_collections_capture_presentation` (reads both fields, builds
  the render-only `CollectionsCaptureReaderPresentation`).
- `_load_library_collections_capture_entry` (writes both, from
  `scope.list_saved_searches(1)` — a `CollectionsCaptureScopeService`
  call, `Library/collections_capture_service.py`).
- `select_library_collection_capture_scope` (reads `saved_searches` to
  resolve a pressed saved-search scope row into a `CapturePageRequest`).

All four are Collections-cluster methods (name contains "collection").
**A repo-wide `grep -rn` for both field names outside `library_screen.py`
returns zero hits** — no other module, screen, controller, or test
references either field.

**What "the search cluster" actually is**: a repo-wide `ast` census of
`LibraryScreen` methods containing `"search"` (23 methods) shows a
completely different, unrelated feature — the Library-wide search rail
and its submit history (`handle_library_search_changed`,
`handle_library_search_submitted`, `_load_library_search_history`,
`_persist_library_search_history`, `_record_library_search_history`,
`clear_library_search_history`, `rerun_library_search_from_history`,
`_focus_library_search_input`, `_library_rail_search_placeholder`, plus
several Media/Prompts/RAG-scoped search helpers). None of these 23
methods reference either saved-searches field. The `SavedCaptureSearch`
dataclass and `list_saved_searches`/`save_saved_search` methods are
defined entirely in `Library/collections_capture_models.py`,
`Library/collections_capture_service.py`, and
`Library/server_collections_capture_service.py` — Collections' own
capture-scope service layer (per-scope saved FILTER presets for the
capture reader), architecturally unconnected to the search-bar's
submit-history mechanism despite the shared English word "search."

**Verdict: MOVE, uncontested.** The census fully resolves the brief's
flagged ambiguity — there is no genuine second subsystem consumer, so the
recipe's ordinary "NONE → moves" rule applies cleanly. No BLOCKED
condition arose.

## 5. `LibraryCollectionsState` + shims

`tldw_chatbook/UI/Library_Modules/library_collections_state.py`: a
`@dataclass` with all 26 fields, verbatim defaults. All 26 use the same
`_library_collections_` prefix — Collections' own subsystem name is
already plural, so unlike Conversations (whose singular subsystem name,
"conversation", needed a plural `_library_conversations_` variant for two
fields) there is no analogous split and no
`COLLECTIONS_PLURAL_STATE_FIELDS` constant, matching the export series'
own single-prefix precedent exactly.

Three fields (`reader_preferences`, `reader_persistence_locks`,
`reader_layout`) are entangled with other subsystems' shared init code —
same shape as the conversations exemplar's own trio — and keep their
original `__init__` assignment lines completely untouched, routed through
the generated property shim.

**A new deviation from "construct at the position of the first removed
field," required by entanglement ORDER (not just entanglement itself)**:
`reader_preferences` is one of eight targets in a shared
`self._load_library_reader_preference_snapshot()` tuple-unpack that
executes BEFORE any of Collections' 23 non-entangled fields are assigned
in the original `__init__`. Unlike the conversations exemplar and the
export series (where every entangled field's original line sits AFTER
the position their state object was constructed at), constructing
`self._collections_state` at the position of the first REMOVED
(non-entangled) field — which sits AFTER the shared tuple-unpack — would
raise `AttributeError` the instant that unpack's
`self._library_collections_reader_preferences` target tried to route
through a not-yet-installed property into a not-yet-existing object.
Fixed by constructing `self._collections_state = LibraryCollectionsState()`
at the SAME early point `self._conversations_state` is constructed
instead (immediately after it), before the shared tuple-unpack. This
remains a behaviorally-transparent move: all 23 non-entangled fields have
static-literal defaults (no field needed a constructor argument, unlike
Export's computed `form` default), so constructing the dataclass earlier
than usual has no observable side effect — confirmed by the full battery
(§6) passing, including every test that presses buttons during the
reader-preferences-dependent early `__init__` sequence.

Shim: a module-level `for _cos_field in dataclasses.fields(LibraryCollectionsState):
setattr(LibraryScreen, "_library_collections_" + _cos_field.name, property(...))`
loop at the end of `library_screen.py`, sentinel-wrapped
(`--- BEGIN/END generated collections-state shims ---`), installed
exactly like the export series' own state-PR shim block. `_n=` default-arg
closures bind both getter and setter per field.

**Plural/singular prefix mapping, single source**: none needed. Every
Collections field maps to `_library_collections_<field>` — one prefix,
defined as a literal string constant inline in the shim loop (no separate
`_PLURAL_STATE_FIELDS`-shaped constant module, mirroring Export's own
"no plural set was created" precedent, documented in both this state
module's docstring and the recipe §13).

**Dead imports, deliberately left for the cleanup PR**: `CaptureCapabilities`,
`CaptureHighlight`, `SavedCaptureSearch`, `CollectionsReaderMode` are now
imported into `library_screen.py` but never referenced there directly
(only inside the moved dataclass field type annotations, now in
`library_collections_state.py`). Per the export series' own Task 2/Task 4
split (dead-import removal is explicitly a cleanup-PR activity, not a
state-PR one), these are left in place; the collections cleanup PR should
remove them after checking each against the `_SURFACE`-shaped re-export
contract test first (recipe §11's "dead within this file is not the same
question as dead" lesson).

## 6. Wiring test — TDD evidence

`Tests/Architecture/test_library_collections_wiring.py` was written and
run before the screen-side shim installation existed:

```
FAILED Tests/Architecture/test_library_collections_wiring.py::test_state_object_fields_match_the_shim_surface
```

RED confirmed — a genuine `AssertionError` (no `_library_collections_*`
property existed on `LibraryScreen` yet), not a trivially-passing
placeholder. (Process note, honestly recorded: `library_collections_state.py`
was created before this wiring test file in this task's own execution
order, unlike the export series' literal "before the state module
existed" RED shape — but the wiring test's own RED/GREEN transition is
still genuine, gated on the SCREEN edit, not the state module's
existence.) After the screen edit landed:

```
Tests/Architecture/test_library_collections_wiring.py .            [100%]
1 passed
```

GREEN. Scope matches the export series' own Task 2 precedent exactly
(state-object fields ↔ shim surface only) — this task is the state PR
only within the 3-task collections series.

## 7. Verification battery

**Wiring test RED → GREEN**: see §6.

**Characterization file all-PASS** (inverted TDD; 3 consecutive clean
runs of the full 5-function file after the `pilot.pause()` timing fix
in §2):
```
5 passed
```

**Size ratchet — ceiling AND slack green, and the brief's own flagged
stale-pin gap closed**: `git show HEAD:tldw_chatbook/UI/Screens/library_screen.py | wc -l`
on the pre-task tree measures **43412** — one below the recorded
`_BUDGETS` pin of `43413` (a 1-line slack that had gone unnoticed since
the export cleanup PR, exactly as this task's brief predicted). This
task's own edit (import +1, 23-line field block removed, 2-line early
construction added, 20-line shim block added) measures **43410 lines,
1281 methods** fresh, post-edit — below BOTH the stale pin and the true
baseline, so `_BUDGETS` is lowered to `43410` in this same commit
(`Tests/Architecture/test_screen_size_ratchet.py`), closing the gap
rather than carrying it forward. Both
`test_screen_does_not_grow_past_its_budget` and
`test_budget_is_not_left_slack_after_a_wave` pass for the
`library_screen.py` row (and the pre-existing `chat_screen.py`-scoped
failures in this same file remain, unrelated to this task's diff).

**Recompose ratchet (with its slack guard) + support-layer surface**: all
6 tests in `Tests/UI/test_library_recompose_ratchet.py` pass — this move
touches zero `refresh(recompose=True)` call sites (pure field relocation),
so the recompose census pin and its anti-slack guard are unaffected.
`Tests/Architecture/test_library_support_layer_surface.py` (13 tests,
including `test_no_import_cycle`) all pass.

**Export + conversations wiring/characterization regressions**: green —
`Tests/Architecture/test_library_export_wiring.py`,
`Tests/Architecture/test_library_conversations_wiring.py`,
`Tests/UI/test_library_export_characterization.py`,
`Tests/UI/test_library_conversations_characterization.py` — 21 passed.

**`-k "collection and library"` with stash-baseline comparison** (swept
BOTH `Tests/UI` and `Tests/Library`, per the recipe's export-series
lesson that `Tests/Library` needs explicit inclusion): 361 passed, 3
failed on this task's branch. A direct node-id rerun of the same 3
against a `git stash -u` baseline of the pre-task tree reproduced the
identical 3 failures with identical error messages/tracebacks (see
recipe §7's updated documented-failures list for the full per-test
detail): `test_library_starter_deep_link_opens_hidden_collection_or_note_route`,
`test_library_landing_continue_receipt_accepts_only_authoritative_source_scopes[browse-collections-expected_scope4]`,
`test_get_library_collection_supported_types_round_trip_public_ids`. All
3 confirmed pre-existing and appended to the recipe's documented list.

**Full xdist paired-baseline sweep** (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`):
- Branch: 333 failed, 3906 passed (1308.27s / 0:21:48)
- Baseline (`git stash -u`): 340 failed, 3894 passed (1286.31s / 0:21:26)
- Diff: 2 failures unique to branch, 9 unique to baseline only, 331
  shared (pre-existing backdrop).
- The 2 branch-unique failures
  (`Tests/UI/test_library_prompts_canvas.py::
  test_library_prompt_history_no_change_keeps_selection_and_retry_available`,
  `Tests/UI/test_library_shell.py::
  test_library_starter_production_geometry_and_focus_order[size1]`) were
  re-run directly, single-process (no xdist), both individually and
  together in one invocation: **both pass cleanly every time** (2 passed,
  0 failed). Neither test touches Collections, exercises a Collections
  route, or shares a fixture with this task's diff — confirmed
  xdist-specific ordering/shared-state flakiness (recipe §7's documented
  noise class), not a regression from this task's move. The 9
  baseline-unique failures are better-on-branch noise in the opposite
  direction, absorbed into the shared backdrop, not attributable to this
  task.
- **Zero real failures unique to this branch** — matches the export
  series' own Task 2 sweep result (0 branch-unique) more closely than its
  Task 3/Task 4 sweeps, which each found genuine new bypass-shape
  regressions; this task's pure field-relocation shape produced none.

**Preflight** (`./scripts/preflight.sh`): all six checks green (CSS
bundle, profile-owned-path census, diagnostic inventory, backlog task
ids, chachanotes table allowlist, index plan pins).

## 8. Files changed

- `Tests/UI/test_library_collections_characterization.py` (new) — 17
  characterization pins across 5 test functions.
- `tldw_chatbook/UI/Library_Modules/library_collections_state.py` (new) —
  `LibraryCollectionsState` dataclass, 26 fields.
- `Tests/Architecture/test_library_collections_wiring.py` (new) —
  state-field/shim wiring test.
- `tldw_chatbook/UI/Screens/library_screen.py` (modified) — import added,
  `self._collections_state` constructed early (before the shared
  reader-preferences tuple-unpack, not at the first-removed-field
  position — see §5), 23-field `__init__` block removed, generated shim
  block appended at module end.
- `Tests/Architecture/test_screen_size_ratchet.py` (modified) —
  `_BUDGETS` row lowered to `43410/1281`.
- `backlog/docs/library-decomposition-recipe.md` (modified) — new §13
  ("The collections series, task 1 (state PR) — as landed") with the
  full cluster/ownership/saved-searches census, plus 3 new pre-existing
  failures appended to the §7 documented list.

No `.git-blame-ignore-revs` entries added (brief's explicit instruction:
state PRs are not body moves).

## 9. Self-review

- Ownership script re-derived at execution time (27 `__init__`-scoped
  fields), not trusted from the plan's ~28 estimate; cross-checked with a
  full class-level `AnnAssign` scan (found zero collections-owned
  class-level-only attributes, unlike Export's `origin_row_id` case).
- The saved-searches contested-boundary question was resolved by an
  actual repo-wide consumer census (both fields, every occurrence read),
  not by assertion — recorded in full in §4 and in the recipe.
- Every controller/service-instance-holding field was checked:
  `capture_controller` is the only one in this cluster, and its verdict
  (WIRING, stays) is recorded with its own shell/plumbing consumer list.
- Byte-for-byte canon respected: no method body touched, no receiver
  rewritten, no "while I'm here" cleanup. The three entangled fields'
  original `__init__` lines are untouched (confirmed via `git diff` —
  only the 23 non-entangled field lines and the shim/import/construction
  additions appear in the diff).
- Size ratchet measured fresh, post-edit, not deferred or carried over —
  recipe §6's explicit lesson from the conversations exemplar's Task 7
  near-miss, and this task's own brief-flagged stale-pin gap is closed in
  this same commit rather than left for a future task to discover.
- Sweep evidence follows recipe §7's procedure: xdist run (in progress /
  completed — see §7), paired `git stash -u` baseline for the narrower
  `-k` check already completed and diffed, targeted node-id rerun to
  confirm the smaller subset's 3 failures before trusting them as
  pre-existing.
- No BLOCKED conditions encountered: every collections field resolved
  unambiguously (25 plain MOVE + 1 entangled-but-still-MOVE trio + 1
  WIRING), including the contested saved-searches boundary the brief
  flagged in advance.

## 10. Lessons for the next task (collections controller PR)

- The 64 genuinely-Collections-named methods (67 minus 3 Prompts
  false-positives) have NOT been individually classified for
  controller-ownership in this task — that census is this series'
  controller PR's own job, per the export series' precedent. Expect the
  same "not every name-matching method actually moves" pattern export
  found (18 of 51 export candidates were owned by other subsystems; some
  fraction of these 64 likely are too, though none were found to be
  Prompts-owned beyond the 3 already excluded here).
- Widen the controller PR's own verification sweep to `Tests/Library` in
  addition to `Tests/UI` from the start (this task already needed to, and
  found one of its three pre-existing failures there) — the export
  series' Task 3 report already flagged this as a forward note; this
  task reconfirms it.
- Check for the "unbound fake-self" bypass shape early: the pre-existing
  `test_library_starter_deep_link_opens_hidden_collection_or_note_route`
  failure's `SimpleNamespace` fixture missing `active_authority` is a
  close cousin of that shape (an unbound/incomplete fake service object,
  not a fake `self` — but the same "test fixture assumes a smaller
  surface than the real object now has" root cause) and may recur for
  Collections' own controller-PR fixtures.
