# Task 2 report — Combined Search+RAG state PR (series 1/3)

Wave-3 Task 2 (recipe: `backlog/docs/library-decomposition-recipe.md`;
export/collections series are the worked examples). Scope: state layer
only, per the task brief — the controller move is Task 3.

## 1. Cluster enumeration (re-derived, no `startswith` shortcuts)

`ast`-walked `LibraryScreen` (43977 lines, 1316 methods at task start —
matches `_BUDGETS` exactly) for method names containing `"search"` and
`"rag"` (case-insensitive), independently, then unioned:

- **Raw `"search"` matches: 24** (wave-2's own census found 23 at the time;
  the file has churned since — one new method,
  `_reset_library_media_search_on_mode_change`, appeared. Verified
  Media-owned by body inspection: touches only
  `_library_media_content_match_index/_memo/_query`,
  `_library_media_reader_session`, nothing search/RAG-cluster-owned).
- **Raw `"rag"` matches: 39** — unchanged from wave-2's census, byte-for-byte
  same 39 names.
- **Overlap (both substrings): 3** —
  `_apply_library_rag_search_outcome`, `_execute_library_rag_search`,
  `_refresh_search_rag_panel_state_widgets`.
- **Combined unique candidate set: 24 + 39 − 3 = 60.**

### Exclusions (verified by reading each body, not by name)

- **3 Prompts-owned**: `_flush_library_prompts_search`,
  `_queue_library_prompts_search`, `_stop_library_prompts_search_debounce`
  (all touch `_library_prompt_browse_controller`/`_library_prompts_*`
  fields only).
- **7 Media-owned**: `_focus_library_media_content_search_input`,
  `_reset_library_media_search_on_mode_change`,
  `handle_library_media_content_search_{next,prev,submitted}`,
  `handle_library_media_trash_search_{changed,submitted}` (all touch only
  `_library_media_content_*`/`_library_media_trash_*` fields/controllers).
- **50 remain — the combined Search+RAG cluster** (14 search-cluster +
  39 rag-named − 3 overlap = 50), matching wave-2's own search-side count
  (14) exactly and reconfirming the RAG-named set unchanged (39).

The 14 search-cluster candidates (unchanged from wave-2's census):
`_apply_library_rag_search_outcome`, `_execute_library_rag_search`,
`_focus_library_search_input`, `_library_rail_search_placeholder`,
`_load_library_search_history`, `_patch_sibling_library_search_input`,
`_persist_library_search_history`, `_record_library_search_history`,
`_refresh_search_rag_panel_state_widgets`, `_save_library_search_history`,
`clear_library_search_history`, `handle_library_search_changed`,
`handle_library_search_submitted`, `rerun_library_search_from_history`.

## 2. Field ownership (recipe §2 script, `_library_search`/`_library_rag` prefixes)

**20 `__init__`-scoped fields** (19 `_library_rag_*` + 1
`_library_search_history`; no class-level-only field like Export's
`origin_row_id` — a targeted `AnnAssign`/`Assign` scan of the whole class
body found zero search/rag-named class-level attributes).

Per-field consumer census (methods other than `__init__` touching each
field, split into search-cluster / rag-named / other):

| Field | search-cluster users | rag-named users | other/shell users |
|---|---|---|---|
| `_library_rag_answer` | 0 | 3 | `restore_state`, `save_state` |
| `_library_rag_answer_in_flight` | 0 | 5 | none |
| `_library_rag_answer_in_flight_provider` | 0 | 4 | none |
| `_library_rag_answer_mode` | 0 | 3 | `restore_state`, `save_state` |
| `_library_rag_answer_query` | 0 | 3 | `restore_state`, `save_state` |
| `_library_rag_answer_render_key` | 0 | 1 | `compose_content` |
| `_library_rag_diagnostics` | 1 | 4 | `restore_state`, `save_state` |
| `_library_rag_history_collapsed` | 1 | 3 | none |
| `_library_rag_history_refresh_lock` | 0 | 1 | none |
| `_library_rag_mode` | 2 | 3 | `_library_continue_receipt_for_current_route`, `restore_state`, `save_state` |
| `_library_rag_panel_refresh_lock` | 1 | 2 | none |
| `_library_rag_query` | 3 | 2 | `_reconcile_library_entry_state`, `_replace_library_canvas_child`, `_show_library_file_notes`, `_sync_library_rail_lifecycle_presentation`, `compose_content`, `restore_state`, `save_state` |
| `_library_rag_recovery_state` | 1 | 5 | `restore_state`, `save_state` |
| `_library_rag_results` | 1 | 4 | `restore_state`, `save_state` |
| `_library_rag_retrieval_status` | 1 | 5 | `_library_continue_receipt_for_current_route`, `restore_state`, `save_state` |
| `_library_rag_scope_deselected` | 0 | 2 | `_library_continue_receipt_for_current_route`, `restore_state`, `save_state` |
| `_library_rag_scope_recovery_visible` | 1 | 3 | none |
| `_library_rag_searched_query` | 1 | 4 | `_library_continue_receipt_for_current_route`, `restore_state`, `save_state` |
| `_library_rag_selected_result_id` | 1 | 5 | `restore_state`, `save_state` |
| `_library_search_history` | 3 | 1 (`_library_rag_panel_state`) | none |

**Zero fields BLOCKED, zero shared with a second subsystem's own methods**
— every non-cluster consumer is shell/plumbing (`save_state`,
`restore_state`, `compose_content`, `_reconcile_library_entry_state`,
`_replace_library_canvas_child`, `_sync_library_rail_lifecycle_presentation`,
`_show_library_file_notes`, `_library_continue_receipt_for_current_route`).
The ≥2-subsystems rule never triggers. **All 20 fields MOVE.**

`_library_rag_searched_query` (flagged in the wave plan as needing explicit
classification): 7 users, 4 rag-named + `_apply_library_rag_search_outcome`
(a search-cluster/rag-named overlap member) + 2 shell — cluster-owned,
uncontested. `_library_search_history`: 4 users, all cluster (3
search-cluster + `_library_rag_panel_state`) — cluster-owned, uncontested.

`_library_collections_saved_searches*` reconfirmed collections-owned per
wave-2's own verdict (§13 of the recipe); not touched by this task, not
re-examined (already resolved by a prior series).

## 3. `@work` enumeration (methods that CANNOT move to a controller)

Per the export series' "framework-decorator self-type assertion" lesson
(`@work`'s closure asserts `isinstance(self, DOMNode)` at call time — a
plain controller instance would fail it) — **3 methods in the combined
cluster**, found by scanning every one of the 50 cluster candidates'
decorator lists:

- `_execute_library_rag_answer` — `@work(exclusive=True, group='library_rag_answer')`
- `_execute_library_rag_search` — `@work(exclusive=True, group='library_rag_search')`
- `_save_library_search_history` — `@work(thread=True)`

All three stay on `LibraryScreen`, unmoved, in the eventual controller PR
(Task 3). No other cluster candidate carries `@work` or any other
framework decorator with a self-type hazard.

## 4. State-shape decision: ONE combined `LibraryRagSearchState`

**Decision: single combined state object, not a search/rag split.**

**Correction (fix round 1, post-review): the original version of this
section claimed `_library_rag_panel_state` reads "all 20 fields in one
call" with "its continuation" folding in the rest. Both claims were
factually wrong** — the method has no continuation (it is a single `return
LibraryRagPanelState.from_values(...)` statement, `library_screen.py:13802`
–`13945`), and a mechanical grep of its actual body against all 20 field
names shows it reads exactly **14 of 20** directly: `mode`, `query`,
`results`, `retrieval_status`, `recovery_state`, `selected_result_id`,
`diagnostics`, `searched_query`, `answer`, `answer_in_flight`,
`answer_in_flight_provider`, `scope_deselected`, `history_collapsed`,
`history`. It does **not** reference `answer_query`, `answer_mode`,
`answer_render_key`, `history_refresh_lock`, `panel_refresh_lock`, or
`scope_recovery_visible` anywhere in its body. The corrected reasoning
below traces each of those 6 to its real consumer instead of asserting
they're read here too:

- **`answer_query`/`answer_mode`** are staleness guards. Verified by
  reading both ends: `_start_library_rag_answer` (`library_screen.py:42881
  -42882`) writes them from the just-dispatched request
  (`self._library_rag_answer_query = request.query`, `..._mode =
  request.mode`); `_apply_library_rag_answer` (`42983`, `42986`), the
  answer worker's own completion handler, reads them back and discards a
  stale answer when `request.query != self._library_rag_answer_query` or
  `request.mode != self._library_rag_answer_mode`. `_start_library_rag_
  answer` itself calls `self._library_rag_panel_state().coverage_note`
  (`42880`) to build the very answer these two fields guard — a real
  coupling to the builder, but as a CALLER of it, not as fields the
  builder itself reads.
- **`answer_render_key`** is a render-skip cache. Verified at
  `_refresh_library_rag_answer_widgets` (`43403-43420`): it takes the
  panel-state OBJECT `_library_rag_panel_state()` returns as its own
  parameter, derives a `render_key` from three of that object's fields,
  and compares it against `self._library_rag_answer_render_key` to decide
  whether to skip the rebuild.
- **`history_refresh_lock`/`panel_refresh_lock`** are `asyncio.Lock`
  synchronization primitives, not display values. Verified:
  `_refresh_library_rag_history_widget` holds `history_refresh_lock`
  (`43453`); `_refresh_search_rag_panel_state_widgets` (`43260`) and
  `_mirror_library_rag_scope_recovery` (`43181`) both hold
  `panel_refresh_lock` — and BOTH call `self._library_rag_panel_state()`
  fresh from inside the lock they hold (`43272`, `43191`), which is the
  real shape of their coupling to the builder.
- **`scope_recovery_visible`** is a change-gate cache. Verified:
  `_refresh_search_rag_panel_state_widgets` writes it (`43290`)
  immediately after calling the builder in the same method
  (`self._library_rag_scope_recovery_visible =
  library_rag_scope_shows_recovery(panel_state.scope)`);
  `_sync_library_rag_scope_toggle_and_run_gate_widgets` reads it back
  (`43126`, `if shows_recovery != self._library_rag_scope_recovery_
  visible:`) to decide whether a background snapshot changed enough to
  schedule `_mirror_library_rag_scope_recovery`.

**Corrected conclusion**: the true claim is narrower than "one method
reads all 20 fields" but still supports the same decision. All 20 fields
are consumed inside ONE tightly-coupled, lock-serialized call graph rooted
at `_refresh_search_rag_panel_state_widgets`/`_library_rag_panel_state` —
14 directly by the builder, the other 6 by its immediate callers/callees
in that same sequence (`_refresh_library_rag_answer_widgets`,
`_mirror_library_rag_scope_recovery`, `_sync_library_rag_scope_toggle_
and_run_gate_widgets`) or by the answer-worker completion pair that
produces the builder's own inputs. A two-object split would still force
those SAME functions — not a hypothetical third party — to reach across a
controller boundary repeatedly within one refresh sequence: e.g.
`_refresh_search_rag_panel_state_widgets` calls `_refresh_library_rag_
answer_widgets` (an answer-field consumer) UNCONDITIONALLY, above the
results/history gate, inside the identical lock and using the identical
`panel_state` object as the retrieval/history calls right below it — so
even the plan's own hypothetical "rag-answer pipeline vs search/history
surface" seam cuts through this one method's own internals, not between
two independent call paths. **ONE combined state object remains the
right call, on the corrected evidence.**

This is the field-level confirmation of what wave-2 task 8's method-level
census already forced: the top search bar's submit path *is* the RAG
query entry point (`handle_library_search_submitted`/`rerun_library_
search_from_history` both call `_start_library_rag_query` directly), not
a sibling of it.

**What Task 3 should weigh given this more precise picture**: the
coupling is real and still favors one controller, but it is now
documented as "one call graph, several cooperating methods" rather than
"one function, twenty fields" — Task 3's own controller-boundary decision
should re-verify the METHOD-level call graph directly (as wave-2 task 8
already did for the search/RAG split question) rather than leaning on
this field-consumer table alone, since a method-count argument and a
field-consumer argument can diverge in a way this correction's own
mistake is a live example of.

Class: `LibraryRagSearchState` in
`tldw_chatbook/UI/Library_Modules/library_rag_search_state.py`. This
previews (does not bind) Task 3's own controller-shape call — the plan's
own framing is explicit that the field-level finding here is evidence for,
not a decision that forecloses, that later call. Given the finding, a
single `LibraryRagSearchController` is the natural continuation, but
Task 3's own brief should re-verify against the METHOD-level call graph
(not just field consumers) before committing.

### Two-prefix shim mapping

All fields use prefix `_library_rag_` except one: `_library_search_history`
(dataclass field name `history`). `SEARCH_PREFIXED_STATE_FIELDS =
frozenset({"history"})` lives in `library_rag_search_state.py` itself —
the single authoritative home, imported by the screen's shim-generator
loop rather than redefined there — applying the conversations exemplar's
own `CONVERSATIONS_PLURAL_STATE_FIELDS` drift lesson (recipe §11: two
independent copies of the same set drifted silently at that task's own
fix round) from the start instead of rediscovering it.

## 5. Characterization spot-check — genuinely-unpressed `@on` handlers

Scope per brief: `@on` handlers in the combined cluster only (not
`action_*` methods, which are a Task-3 delegator-census concern). **14
`@on` handlers found**:

| Handler | Selector | Coverage found |
|---|---|---|
| `handle_library_search_changed` | `Input.Changed` `#library-search-input` | Real keystroke dispatch (`pilot.press("a","b","c")` after focus, `test_library_shell.py`) AND a direct-call test with `Input.Changed(rail_input, ...)` on a real, fully-mounted `screen` (same file, ~line 7314) |
| `handle_library_search_submitted` | `Input.Submitted` `#library-search-input` | Real `pilot.press("enter")` after value+focus (`test_library_shell.py:8467`) |
| `update_library_rag_query` | `Input.Changed` `#library-rag-query-input` | Direct call `await screen.update_library_rag_query(Input.Changed(...))` on a real instance (`test_library_rag_keystroke.py`) |
| `submit_library_rag_query` | `Input.Submitted` `#library-rag-query-input` | Direct call `await screen.submit_library_rag_query(...)` (`test_library_rag_keystroke.py:231`) |
| `run_library_rag_query` | `.press()` `#library-rag-run-query` | Real `.press()`, 6+ files |
| `open_import_export_from_library_rag` | `.press()` `#library-rag-open-import-export` | Real `.press()` (`test_library_shell.py:7155`) |
| `cycle_library_rag_mode` | `.press()` `#library-rag-mode-toggle` | Real `.press()`, 4+ files |
| `toggle_library_rag_scope_source` | `.press()` `.library-rag-scope-toggle` | Real `.press()` on `#library-rag-scope-toggle-{media,notes,conversations}` |
| `sync_library_rag_history_collapsed` | `Collapsible.Toggled` `#library-rag-history` | `collapsible.collapsed = False` fires Textual's `_watch_collapsed` → `post_message(Expanded(...))` (confirmed by reading `textual.widgets.Collapsible` source); a companion test explicitly awaits `_library_rag_history_collapsed is False` to prove the handler ran (`test_library_shell.py:15454`) |
| `clear_library_search_history` | `.press()` `#library-rag-history-clear` | Real `.press()` (`test_library_shell.py:15348`) |
| `rerun_library_search_from_history` | `.press()` `.library-rag-history-row` | Real `.press()` on `#library-rag-history-{0,1}` |
| `select_library_rag_result` | `.press()` `.library-rag-result-action` | Real `.press()` on `#library-rag-select-result-{index}` (many files) |
| `open_library_rag_result` | `.press()` `.library-rag-result-open` | Real `.press()` on `#library-rag-open-result-{index}` (`test_library_shell.py`) |
| `use_selected_library_rag_result_in_console` | `.press()` `#library-rag-use-selected-in-console` | Real `.press()` (`test_product_maturity_gate16_library_search_rag.py`, 4 tests) |

**Zero genuine gaps found — 14/14 already covered.** This is the "RAG
flows may have deep coverage" outcome the wave-3 plan explicitly flagged
as a real possibility, confirmed rather than assumed (initial substring
greps for `.library-rag-result-open`/`.library-rag-result-action` looked
like 0-press gaps at first pass; re-checking against the actual per-index
`id=` pattern the widget renders — `library-rag-open-result-{index}` /
`library-rag-select-result-{index}`, distinct strings from the CSS class
names — found real presses in both cases).

**Skip decision**: no new `Tests/UI/test_library_search_rag_characterization.py`
file. Every handler already has dedicated, real coverage; a redundant pin
would add nothing a move-time regression couldn't already trip via the
existing tests. Recorded in the characterization commit's own message in
detail (per-handler evidence), not just here.

**Forward note for Task 3 (controller move, out of this task's scope)**:
2 of the direct-call tests above monkeypatch INSTANCE attributes on the
real `screen` object (`monkeypatch.setattr(screen, "_patch_sibling_
library_search_input", ...)` and `..."_refresh_search_rag_panel_state_
widgets", ...)`) — the conversations exemplar's own "instance-attribute
monkeypatch" bypass shape (recipe §11, lesson 2). Both patched names are
themselves cluster candidates for the controller move. If either moves
onto a controller, `self` inside the moved sibling body resolves to the
CONTROLLER instance, which never saw the patch applied to the SCREEN
instance — Task 3's own cluster census should check these two names
explicitly rather than rediscovering the failure via the sweep.

## 6. State object + shims

`tldw_chatbook/UI/Library_Modules/library_rag_search_state.py`: a
`@dataclass` with all 20 fields, verbatim defaults, verbatim comments
(copy-pasted from the original `__init__` block, not retyped — checked
against the pre-edit `library_screen.py` line-by-line while writing).
`history` (`_library_search_history`) has a genuinely computed default
(`self._load_library_search_history()`, a same-subsystem method, not
entangled with another subsystem) — per the recipe's "computed defaults
become constructor arguments" rule, `__init__` still calls it at the
position of the removed line and passes the result into the
`LibraryRagSearchState(...)` constructor call.

Unlike the conversations/collections exemplars' entangled reader-preferences
trios, **all 20 fields sat in one contiguous, unentangled `__init__` block**
(`library_screen.py` lines 2146–2243 at this task's pre-edit measurement) —
nothing before or after it belongs to this cluster, and nothing in between
belongs to another subsystem. No early-construction workaround was needed;
the state object constructs in one shot at the exact position of the first
removed field.

Screen attribute: `self._rag_search_state` (matching the `self._<subsystem>_
state` convention with subsystem name `rag_search`, derived from the class
name `LibraryRagSearchState`).

Shim: a sentinel-wrapped, module-level generated-property loop (`--- BEGIN
generated search+rag-state shims (delete wholesale at cleanup) ---`),
mirroring the conversations exemplar's own two-prefix generator shape
exactly — `_library_search_` for names in `SEARCH_PREFIXED_STATE_FIELDS`,
`_library_rag_` for everything else — reading `SEARCH_PREFIXED_STATE_
FIELDS` from the state module (single source) rather than redefining it
inline on the screen.

## 7. TDD evidence

Commit 1 (`test(library): characterization + wiring pins for the
search+RAG extraction series`, `315cd4c3c`) ships the NEW state module
(`library_rag_search_state.py`) and the wiring test
(`Tests/Architecture/test_library_search_rag_wiring.py`) together, but
leaves `library_screen.py` byte-identical to its parent — the recipe §16
lesson 4 structural RED criterion ("screen untouched, tests failing at
this point", not "zero production code in the commit"), applied to a
state module exactly as it was already established for a controller
module. Confirmed RED before commit:

```
FAILED Tests/Architecture/test_library_search_rag_wiring.py::test_state_object_fields_match_the_shim_surface
AssertionError: no screen shim property found for: [... all 20 expected `_library_rag_*`/`_library_search_history` names ...]
```

Commit 2 (`refactor(library): search+RAG state object(s) + shims (series
1/3)`, `77750c85d`) edits only `library_screen.py` (import, `__init__`
block replacement, shim loop) and the `_BUDGETS` pin. Confirmed GREEN
after:

```
Tests/Architecture/test_library_search_rag_wiring.py ..            [100%]
2 passed
```

## 8. Verification battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`, at commit `77750c85d`.

- **Wiring test + guard suite**: `test_library_search_rag_wiring.py` (2),
  `test_library_export_wiring.py` (5), `test_library_collections_wiring.py`
  (4), `test_library_conversations_wiring.py` (6),
  `test_library_modules_size_ratchet.py` (task-1's controller governance
  guard — untouched by this state PR, confirmed still green),
  `test_screen_size_ratchet.py`, `Tests/UI/test_library_recompose_
  ratchet.py`, `test_library_support_layer_surface.py` — **60 passed, 2
  failed** (the two documented pre-existing `chat_screen.py` rows only).
- **Characterization suites (conversations/export/collections)**:
  `test_library_conversations_characterization.py` (4) +
  `test_library_export_characterization.py` (5) +
  `test_library_collections_characterization.py` (5) — **14 passed**.
- **Full `Tests/Architecture`**: **529 passed, 15 failed, 1 skipped** — the
  15 match this wave's own Task-1 ledger note exactly (2 `chat_screen.py`
  rows + 13 Console/timer/worker/diagnostic dev-drift rows), zero new.
- **Real-instance smoke** (constructs and drives an actual `LibraryScreen`
  through the new state object, not just class-level shim checks):
  `test_library_rag_keystroke.py` (6 passed),
  `test_product_maturity_gate16_library_search_rag.py` +
  `Tests/Library/test_library_rag_state.py` (238 passed).
- **`-k "(search or rag) and library"` across `Tests/UI` + `Tests/Library`**:
  branch **785 passed, 10 failed, 3 skipped** (20999 deselected). All 10
  reconfirmed **identical by name** on a `git checkout` pristine baseline
  of the pre-task tree (`fa07400a1`, via a direct node-id rerun of the
  same 10): `test_library_rag_handoffs.py::test_library_use_in_console_
  chip_and_prompted_counts_are_honest`,
  `test_library_rag_rechunk_action.py::{test_rechunk_control_class_
  defines_all_states_with_ds_tokens, test_rechunk_summary_and_report_
  lines_use_the_styled_quiet_line_class}`,
  `test_library_shell.py::{test_library_starter_hidden_route_focuses_
  compact_rail_without_search, test_library_shell_rail_search_submit_
  aborts_on_note_conflict, test_library_shell_notes_filter_queries_
  search_seam}`, `test_screen_navigation.py::{test_action_library_list_
  focus_rail_focuses_search_input, test_library_screen_round_trip_
  returns_to_landing_with_rag_draft, test_boot_with_search_default_tab_
  lands_on_library_rag_canvas, test_search_route_round_trips_to_the_
  library_rag_row}` — same 10, same names, 10 failed on both trees. One
  sampled traceback (`test_action_library_list_focus_rail_focuses_
  search_input`) confirms an unrelated pre-existing gap:
  `AttributeError: '_FakeInput' object has no attribute 'query_one'` at
  `library_screen.py:22604` (an unbound-fake-object test double lacking a
  method the real code calls) — nothing to do with this task's field move.
- **Full sequential xdist paired-baseline sweep** (`Tests/UI -k "library"
  -p no:randomly -q -n 8 --dist worksteal`, run SEQUENTIALLY per the
  recipe's own "concurrent runs amplify flakiness" lesson, not
  concurrently): branch **354 failed / 3927 passed** (1321s) vs pristine
  baseline (`fa07400a1`, same command) **350 failed / 3931 passed**
  (1324s) — both inside the recipe's documented ~330–355 historical
  backdrop range. Diff: **8 branch-unique, 4 baseline-unique**, 342+
  shared. All 8 branch-unique names re-run single-process, combined:
  **6 of 8 passed cleanly** (pure xdist ordering/shared-state noise); the
  remaining 2
  (`test_screen_navigation.py::test_deep_link_library_route_lands_its_
  canvas_over_restored_state`,
  `test_screen_navigation.py::test_library_screen_round_trip_returns_to_
  landing_with_rag_draft` — the latter also independently confirmed via
  the narrower `-k` check above) reproduced even combined, but **both
  reproduced IDENTICALLY on the pristine baseline under the same
  single-process combined-invocation conditions** (2 failed/2 failed,
  same names) — confirmed pre-existing, not a regression, per the
  Media-reader-cluster precedent this exact verification method follows
  (recipe §13/14 close-out). **Zero real regressions.**
- **Preflight**: `./scripts/preflight.sh` — all 5 checks passed (CSS
  bundle, profile-owned-path census, diagnostic inventory, backlog task
  ids, chachanotes table allowlist + index plan pins).

## 9. Files changed

- `tldw_chatbook/UI/Library_Modules/library_rag_search_state.py` (new) —
  `LibraryRagSearchState` dataclass, `SEARCH_PREFIXED_STATE_FIELDS`.
- `Tests/Architecture/test_library_search_rag_wiring.py` (new) — shim-surface
  wiring test + a guard on the prefix-mapping constant itself.
- `tldw_chatbook/UI/Screens/library_screen.py` — import added; 20-field
  `__init__` block replaced with one constructor call; generated shim loop
  appended at module end.
- `Tests/Architecture/test_screen_size_ratchet.py` — `_BUDGETS` row lowered
  `43977/1316 → 43923/1316` with a dated comment.

Commits: `315cd4c3c` (characterization + wiring, RED),
`77750c85d` (state object + shims, GREEN).

## 10. Self-review

- Ownership analysis re-derived from scratch against the current tree (no
  numbers carried over from the wave-2 census except as a cross-check);
  found and correctly classified the one new method
  (`_reset_library_media_search_on_mode_change`) that appeared since.
- State-shape decision backed by reading the actual bodies of the small
  cluster of methods that would have to straddle any split
  (`_library_rag_panel_state` and its immediate callers/callees in the
  same refresh sequence), not by the field-count table alone. **Correction
  (fix round 1, post-review)**: the first draft of §4 and the shipped
  module docstring overstated this evidence — claiming `_library_rag_
  panel_state` reads "all 20 fields in one call" via a "continuation" the
  method does not have (verified: it is one `return` statement,
  `library_screen.py:13802`-`13945`, reading 14 of 20 fields directly).
  Re-derived the true count with a mechanical per-field grep of the
  method's own body, then traced each of the other 6 fields to its actual
  consumer (2 answer-worker staleness guards, 1 render-skip cache, 2
  `asyncio.Lock`s, 1 change-gate cache) by reading those methods directly
  rather than asserting a plausible-sounding story. Both the docstring and
  §4 above are corrected to state exactly what the code shows; the
  ONE-state-object conclusion still holds on the corrected evidence, for a
  more precise reason (one shared call graph, not one function).
- Characterization scope followed exactly (14 `@on` handlers, not
  `action_*` methods); the "zero gaps" outcome was verified per-handler
  with real evidence (grep + source read for `Collapsible`'s message
  mechanics), not asserted from the plan's own hint.
- TDD RED was a real, git-history RED (ran the wiring test and captured
  the failure BEFORE editing the screen), not a same-commit red+green.
- Both required baseline comparisons (`-k` narrow check and the full xdist
  sweep) used a real pristine tree (`git checkout` to the parent commit,
  not an approximation), run sequentially, with every branch-unique name
  individually re-verified before being called noise.
- One deviation from a literal reading of "wiring test committed WITH the
  characterization-pins commit": no NEW characterization pins exist,
  because verification found none were needed. The commit still bundles
  the wiring test with a fully-documented characterization finding, which
  is the substance the instruction asks for; flagging the deviation here
  rather than silently reinterpreting the instruction.
- Did not touch `Agents/tool_catalog.py`-style registration surfaces,
  `task-31203`'s AC#1-3, or anything outside the state layer — Task 3's
  controller move and the delegator/action_* census are explicitly out of
  this task's scope and left untouched.
