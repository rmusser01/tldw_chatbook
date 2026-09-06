# Wave-7 Task 2 — Media controller move (media series 2/3)

**Status:** COMPLETE (not BLOCKED). The media cluster is extractable at the
method level, into ONE controller, and the evidence for "one" is
connected-components, not judgement.

**Branch:** `refactor/library-decomp-wave7-media`
**Parent (task-2 start):** `37a0934b2`
**Commits (rev-parse only, never transcribed):**

| Commit | Kind | Subject |
|---|---|---|
| `3dd7a745a` | RED / pins | `test(library): full-cluster wiring pins for the media controller move (RED)` |
| `9187bd281` | pure move | `refactor(library): media controller + delegators (media series 2/3)` |
| `e97ac3a32` | blame-ignore | `chore(library): blame-ignore the media controller-PR pure move` |
| `40fe1f5bc` | pure move (fix round 1) | `refactor(library): exclude the media viewer-exit from the move (fix round 1)` |
| `7fee56dd4` | blame-ignore | `chore(library): blame-ignore the media controller fix-round move` |

Worktree: `.worktrees/library-decomp-foundation`, `.venv/bin/python`. No
pushes. `progress.md` untouched. `timeout` unavailable; every timed run used
`perl -e 'alarm N; exec @ARGV'`. Every baseline comparison ran in an isolated
`git worktree` at `3dd7a745a` with its own `uv venv`, verified resolving its
OWN tree.

---

## 1. Census — all four spellings, re-derived at this tree

`ast` walk of every `LibraryScreen` class-body method whose name contains
"media" (case-insensitive), run at `37a0934b2`:

| Measurement | Value |
|---|---|
| media-named `FunctionDef`s | **251** |
| unique names among them | **251** (no property/setter-pair gap) |
| carrying `@on` | **79** |
| `action_*` | **12** |
| `@staticmethod` | **5** |
| `@work` | **0** |
| `on_<message>` name-dispatched | **0** |

**Task 1's "0 `on_<message>`" claim RE-VERIFIED, not carried over.** The only
`on_[a-z]` methods on `LibraryScreen` are 10 Textual lifecycle hooks
(`on_descendant_focus`, `on_key`, `on_mount`, `on_mouse_down`,
`on_mouse_scroll_down`, `on_mouse_scroll_up`, `on_resize`, `on_screen_resume`,
`on_screen_suspend`, `on_unmount`) plus the 7 prompts-owned
`on_prompt_block_editor_*`. §4's third whitelist member is inert for media.

### 1.1 Two completeness checks beyond the name census

- **Decorator scan** for a NON-media-named method carrying a
  `#library-media*`/`.library-media*`/`*Media*` `@on` selector: **zero**.
- **Reverse call-graph fixpoint** for non-media-named methods whose every
  in-class caller is (transitively) media-named: **17 names, all review-set
  members** — `_activate_review_set`, `_active_review_set_banner`,
  `_collect_review_pairs_from_scope`, `_collect_review_set_picker_rows`,
  `_create_and_open_review_set`, `_order_selected_review_pairs`,
  `_push_review_set_picker`, `_review_dismiss_receipt_name`,
  `_review_dismiss_undo_worker`, `_review_read_later_pairs`,
  `_review_selected_worker`, `_review_set_picker_worker`,
  `_review_these_name`, `_review_these_worker`, `_toggle_reviewed_unguarded`,
  `_walk_active_review_set`, `_walk_active_review_set_unguarded`.

**All 17 are deliberately NOT moved, on the One Home Rule rather than a
hazard.** Review sets are a ~25-member feature family that straddles the
media/shell boundary: eight more of its members
(`_review_set_service`, `_review_set_live_ids`, `_notify_review_set`,
`_review_set_active`, `_active_review_progress`,
`_active_review_loaded_at_last`, `_review_footer_entries`,
`_maybe_auto_resume_review_set`/`_cancel_pending_review_set_resume`) are
called by SHELL methods — `check_action`,
`_library_route_shortcuts_for_current_state`,
`_select_library_rail_row_after_source_admission`,
`_open_library_item_by_id` — and `open_chunking_lab` is reached from two
non-media-named `@on` handlers. Moving the media-reachable subset would split
one family across two homes, exactly the split task 1's own BLOCKED ruling on
`_library_pending_list_entry_media_return` refused at the field level. The
five movers that call into the family reach it through named late-binding
dependencies.

**A trap worth recording.** Run WITHOUT first removing the two media-NAMED
generic dispatchers (`_toggle_library_media_reader_pane`,
`request_library_media_layout_refresh`), the same fixpoint returns **31**
names and drags in the whole shared reader-preference machinery — including
three OTHER subsystems' `_mirror_library_*_reader_preference` methods and
`request_library_reader_layout_refresh`, which reads and writes eight
subsystems' state. An uncritical bare-name fixpoint over a cluster that owns a
generic dispatcher will claim half the screen.

### 1.2 The four census spellings, run over all 251 names

| Spelling | Result |
|---|---|
| attribute (`LibraryScreen.<name>(...)`, `ast.Call` with an `Attribute` func) | 49 names / 132 sites |
| non-call attribute (`LibraryScreen.<name>` as a bare reference) | 43 names / 82 sites |
| quoted string anywhere in the file (tokenize, STRING tokens only) | 24 names / 33 sites |
| patch-target table / distant `setattr` consumption | covered by the string pass; 3 real `for name in (...)` loops found |

The line-based and `ast` passes were reconciled name-for-name rather than
assumed to agree: the line pass produced exactly one extra name,
`_add_library_media_highlight`, whose only hit is docstring PROSE in
`Tests/Media/test_local_media_reading_service.py:1118`. The `ast` pass
rejected it.

---

## 2. The SPLIT DECISION — one controller, with component evidence

**Decision: ONE `LibraryMediaController`.**

Built the `self.<name>` reference graph among all 251 candidates (edges from
`ast.Attribute` on `self` plus `getattr(self, "<literal>")`), then computed
connected components:

| Graph | Components | Largest | Rest |
|---|---|---|---|
| call graph only | 33 | **213** | 28 singletons + 2×size-3 + 2×size-2 |
| call graph + shared-`LibraryMediaState`-field edges | 13 | **238** | 11 singletons + 1×size-2 |

**There is no second component of any size.** That is the same shape the
prompts series measured (one component of 145 of 161) and the same verdict.

**The plan's own candidate seam was probed directly, not waved past.** Four
candidate partitions, measured:

| Partition | \|A\| | A-internal edges | B-internal | CROSS | shared state fields |
|---|---|---|---|---|---|
| trash vs rest | 35 | 25 | 308 | **16** | **6** |
| viewer/reader vs rest | 46 | 50 | 224 | 75 | 20 |
| analyze vs rest | 22 | 22 | 313 | 14 | 7 |
| browse/list vs rest | 33 | 20 | 288 | 41 | 21 |

The trash partition is the least entangled and still carries 16 cross-call
edges and 6 shared state fields — a seam is *zero* of both, not "fewer". And
after the exclusions only **19 of the 140 movers** are trash-named: a second
controller for 19 methods, still coupled through six shared fields, buys
nothing the per-file `_BUDGETS` governance does not already give.

**The plan's size worry did not materialise.** It feared "~8-9k lines" for a
single controller at ~244 methods. The hazard exclusions remove 111 of the
251, so the 140 that move carry **3,166 source lines** of body and the file
measures **4,461** — within a few hundred lines of
`library_prompts_controller.py`'s 4,998, and governed by an exact row from
birth.

Contingency (c) — "STOP and gate on search-RAG-grade cross-cluster
entanglement" — does not apply: the entanglement here is INTERNAL to media
(one blob), not across subsystems. The only cross-subsystem method found,
`_sanitize_media_field`, is a single shared shell helper and is excluded.

---

## 3. MOVE / EXCLUDE — 140 move, 111 exclude

Movers: **48 `@on` + 5 `action_*` + 3 `@staticmethod` + 84 plain.**

| Class | N | Names (or method) |
|---|---|---|
| unbound-fake-self | **73** | see §3.1 |
| instance-attribute-monkeypatch | **16** | `_analyze_one_library_media_item`, `_arm_library_media_return_settlement`, `_exit_library_media_select_mode`, `_focus_library_media_grip_if_current`, `_library_media_content_signature`, `_library_media_layout_signature`, `_library_media_settlement_tree`, `_library_media_unanalyzed_ids`, `_notify_library_media_analysis_warning`, `_open_selected_media_handoff`, `_reconcile_library_media_stage_presentation`, `_request_library_media_browse`, `_request_library_media_type`, `_start_library_media_analyze`, `_sync_library_media_browse_state`, `_sync_library_media_trash_state` |
| source-census (`inspect.getsource`) | **8** | `_claim_library_media_mutation`, `handle_library_media_row`, and the six-handler tuple `handle_library_media_{bulk_delete_confirm, bulk_delete_undo, delete_confirm, edit_save, trash_restore, trash_delete_confirm}` |
| module-globals-coupling (ACTIVE) | **4** | `_dispatch_library_media_analysis`, `_library_media_analysis_provider_reason`, `handle_library_media_analysis_generate`, `_library_media_viewer_state_cached` |
| screen-identity, MEMBERSHIP form | **3** | `_commit_library_media_return`, `_library_media_focus_target_matches_receipt`, `_resolve_library_media_settlement_target` |
| class-monkeypatch | **2** | `_refresh_library_media_detail`, `_sync_library_media_reader_layout_from_shell` |
| bypassed-construction lifecycle | **2** | `_stop_library_media_selection_debounce`, `_stop_library_media_filter_timer` |
| callback-identity assertion | **1** | `_exit_library_media_viewer` |
| shared shell helper (≥2 subsystems) | **1** | `_sanitize_media_field` |
| generic dispatcher | **1** | `_toggle_library_media_reader_pane` |

`@work` check: **0 `@work` decorators anywhere in the 251**, so the
framework-decorator self-type-assertion hazard (export/RAG/ingest series) does
not arise. `DOMNode` assertions: none found beyond the three membership tests
below. Delegation-to-existing-controller: the four one-liners through
`_library_media_browse_controller`/`_library_media_trash_browse_controller`
(`_request_library_media_browse`, `_retry_library_media_browse`,
`_request_library_media_facets`, `_library_media_type_options`) were ruled
individually — two are excluded on independent hazards, two move with the
wiring accessor bound.

### 3.1 The unbound-fake-self population, by discovery shape

Media's tests favour unbound-`SimpleNamespace` unit-style calls to a degree no
prior subsystem has matched — the export series' own forward note ("expect
this shape to scale with how much a subsystem's tests favor unbound calls"),
measured:

- **49 names / 132 sites** as `LibraryScreen.<name>(fake, ...)`
  (`test_library_multiselect_media.py`, `test_library_media_trash.py`,
  `test_library_media_reader_flow.py`, `test_review_set_walker.py`,
  `test_library_shell.py`, `test_library_media_bulk_delete_real_db.py`);
- **34 names** bound onto a fake as `types.MethodType(LibraryScreen.<name>,
  fake)` / `_press(fake, LibraryScreen.<name>)` /
  `LibraryScreen.<name>.__get__(fake)`;
- **12 names** dispatched by STRING through a
  `for name in ("...", ...): setattr(fake, name, types.MethodType(getattr(
  LibraryScreen, name), fake))` loop —
  `test_library_multiselect_media.py:2515`/`:2734`,
  `test_library_media_reader_flow.py:1362`.

Union = 73 distinct names.

### 3.2 The source-census exclusions, and what they also protect

`test_library_multiselect_media.py::test_every_media_mutation_claims_the_
interlock_at_one_audited_seam` (`:2986-3028` at `37a0934b2`) does three things a move can
break:

1. asserts `"_library_media_bulk_delete_in_flight = True" in inspect.getsource(
   LibraryScreen._claim_library_media_mutation)`;
2. asserts, for each of six named handlers,
   `"_claim_library_media_mutation" in inspect.getsource(handler)` — which a
   one-line delegator's source empties;
3. asserts `inspect.getsource(library_screen_module)` contains **exactly one**
   `_library_media_bulk_delete_in_flight = True` and **exactly one**
   `group="library_media_bulk_delete"`.

Both literals in (3) live inside `_claim_library_media_mutation`
(`library_screen.py:15501` and `:15505` at `3dd7a745a`), which (1) already
keeps in place, so both module-level counts stay at 1. Five of the eight names
are independently excluded as unbound-fake-self;
`handle_library_media_edit_save` is excluded by this class alone.

### 3.3 The module-globals-coupling census (recipe §3's eighth shape), run to completion

All **71 bare module-global names** the mover bodies read, crossed against
every `library_screen`-scoped patch shape across ALL of `Tests/`, under every
alias tests actually import the module as — `library_screen`,
`library_screen_module`, `library_module`, `screen_module`, **derived by
`ast`, not assumed**. 13 names had hits; each hitting test was read.

**The census had to be redone by `ast`.** A first, line-anchored pass scored
`resolve_ingest_analysis_provider` at ZERO because its five real patch sites
spell the module alias and the name on DIFFERENT LINES of a multi-line
`monkeypatch.setattr(...)` call. That is wave-5 task 3's own multi-line lesson
firing again, and it would have shipped an ACTIVE collision.

| Name | Sites/files | Verdict |
|---|---|---|
| `chat_api_call` | 2 / 1 | **ACTIVE** — `test_library_shell.py:10555` patches it, presses the real `#library-media-analysis-generate`, and asserts `dispatched.get("api_endpoint") == "openai"`. Excludes `_dispatch_library_media_analysis`. |
| `analysis_unavailable_reason` | 8 / 3 | **ACTIVE** — `test_library_ingest_analyze_skipped.py`'s `_ready_provider`/`_unready_provider` drive a REAL pilot whose action-visibility gate IS the patched reason; `test_library_media_render_fixes.py` patches it around real screens five more times. |
| `resolve_ingest_analysis_provider` | 5 / 3 | **ACTIVE**, same tests. Together these exclude `_library_media_analysis_provider_reason` and `handle_library_media_analysis_generate`. |
| `build_library_media_viewer_state` | 2 / 1 | **ACTIVE** — `test_library_media_reader_no_change_sync_t22208.py:161` wraps it in a CALL COUNTER and asserts the count is ZERO for a pass-through traversal. A bypassed patch also counts zero: green-but-vacuous, which the recipe classes as ACTIVE. Excludes `_library_media_viewer_state_cached`. |
| `_sync_library_canvas` | **33 / 10** | **LATENT, kept.** Four movers forward bare `self` into it. Every one of the 33 sites was mapped to its ENCLOSING test function and compared against every function in the same file naming one of those four movers or pressing one of their selectors: **zero overlap, in all ten files** (`Tests/Skills/test_skills_import.py`, `test_library_canvas_scoped_sync.py`, `test_library_entry_compose_once.py`, `test_library_media_trash.py`, `test_library_note_import_flow.py`, `test_library_notes_folder_navigator.py`, `test_library_notes_reader.py`, `test_review_set_walker.py`, and two more). Same systemic shape and same verdict every sibling controller carries. |
| `_ANALYZE_ORIGIN_MEDIA`, `_ANALYZE_ORIGIN_IMPORT`, `LIBRARY_ROW_BROWSE_NOTES`/`_PROMPTS`/`_SKILLS`, `LIBRARY_NOTES_SOURCE_DATABASE`, `set_mode` | 9 / 6 | **LATENT** — plain module-attribute READS (a test asserting a constant or computing an expected value), never a patch. |
| `asyncio` | 1 / 1 | **LATENT** — patches an attribute OF the shared `asyncio` module object, the same object in every importer. |

*(A regex artifact worth naming: the first pass's two-argument shape lacked
word boundaries and matched `llm_screen_module`/`personas_screen_module`/
`evals_screen_module` for `logger`. Read, dismissed, and the `ast` pass does
not reproduce it.)*

### 3.4 The `getattr(self, "<literal>")` census — BINDING I-1, re-derived by AST

Per the review erratum, re-derived with `ast.Call` → func `getattr` → arg0
`Name('self')` → arg1 `ast.Constant(str)`, never grep. At `37a0934b2`, over the
whole file:

- `getattr(<Name>, ...)` total: **278**; receiver `self`: **122**; naming a
  media thing: **15**; of those, naming an **ATTRIBUTE: 13** (the other 2 name
  METHODS — `_cancel_library_media_selection_settlement` in
  `_toggle_library_media_select_mode`, `_select_library_media_reader_row` in
  `_undo_library_media_bulk_delete`).

The 13, with their host method and whether the host moved:

| Line(s) | Name | Host | Host moved? |
|---|---|---|---|
| 9630 | `_library_media_analyze_origin` | `on_unmount` | no (shell) |
| 10450 | `_library_media_view` | `_library_list_canvas_showing_list` | no (shell) |
| 15365 | `_library_media_analyze_origin` | `_library_media_analyze_receipt_fields` | **yes** |
| **16189-16193** | `_library_media_trash_focus_authority_generation` | `_sync_library_media_trash_state` | no (excluded) |
| 16218 | `_library_media_trash_focus_request_key` | `_sync_library_media_trash_state` | no (excluded) |
| 20125 | `_library_media_analyze_running` | `_build_library_ingest_state` | no (shell) |
| 25752, 25771 | `_library_media_view` ×2 | `check_action` | no (shell) |
| **25765-25767** | `_library_media_trash_browse_controller` | `check_action` | no (shell) |
| 32342 | `_library_media_bulk_delete_in_flight` | `handle_library_media_review_selected` | **yes** |
| 32560 | `_library_media_select_mode` | `_create_and_open_review_set` | no (review-set) |
| 33501 | `_library_media_browse_controller` | `_delete_library_media_item` | no (excluded) |
| 33564 | `_library_media_reader_session` | `_delete_library_media_item` | no (excluded) |

Both line-wrapped sites the erratum named are reproduced. **13, not 11**, and
the two that live inside movers resolve on the controller (one through the
generated state shim loop, one through it as well). `:2893` reads
`app_instance`, correctly outside this census.

Inside the final mover set the complete `getattr(self, "<literal>")` name list
is 3: `_library_media_analyze_origin`, `_library_media_bulk_delete_in_flight`
(both state fields → the shim loop) and `_library_notes_focus_intent_generation`
(a bound shell accessor). All three resolve.

---

## 4. The controller

`tldw_chatbook/UI/Library_Modules/library_media_controller.py`, **4,461
lines**, born governed and born lazy.

- **Constructor arity MEASURED** with
  `inspect.signature(LibraryMediaController.__init__)`: **85 parameters
  including `self`** — 1 positional (`screen`) + 83 keyword-only.
- **174 class-level `property` objects**: 92 hand-written bindings + the 82
  generated flat-name state shims.
- Binding groups: 15 framework services, 11 read-only shared-shell accessors,
  5 read/write pairs, 2 media-wiring accessors, 24 shell helpers, 35
  late-binding callables for the exclusions.

### 4.1 The two bindings the recipe's own census cannot produce

`_library_canvas_projection_depth` and `_library_canvas_resync_pending` appear
in **no moved body**. The shared `_sync_library_canvas` dispatcher reaches them
off the bare `self` four movers hand it, and the first is read as
`getattr(screen, "_library_canvas_projection_depth", 0)` — unbound it returns
0 forever, permanently disabling the projection-replay branch with no
exception and no red test. This wave's mechanical binding census does not
produce either name; what caught it was reading the dispatcher's own body for
the media `kind`.

**The conversations exemplar had already found and bound the identical pair**
(`library_conversations_controller.py`, groups (c)/(d) and its own two
properties), so this is a re-derivation. The standing lesson: **any controller
whose movers forward bare `self` into `_sync_library_canvas` must read that
dispatcher's body for its `kind`, because the recipe's `self.<attr>` census
over the moved bodies structurally cannot see it.** The media path needs 15
names off the object it is handed; the other 13 are covered (5 framework, 1
state shim, 5 movers, 2 named dependencies).

### 4.2 Read/write shared shell state — a shape prompts did not need

Five names are STORED by a moved body and are not this controller's own state,
so each gets a getter **and** a setter rather than a read-only accessor:
`_library_selected_row_id`, `_library_notes_programmatic_focus_target`,
`_library_notes_restoring_focus`, `_library_list_entry_focus_timer`,
`_library_canvas_resync_pending`. The prompts census found zero such stores;
the conversations exemplar's group (c) is the precedent for the shape.

### 4.3 Four constants relocated, not duplicated

`_MEDIA_VIEW_LIST`, `_MEDIA_VIEW_VIEWER`, `_ANALYZE_ORIGIN_MEDIA`,
`_ANALYZE_ORIGIN_IMPORT` were module-level assignments in `library_screen.py`
that moved bodies read as bare globals. They move verbatim (comments included)
to `screen_constants.py` — the established home, the ingest series'
dead-zone-constant precedent — and `library_screen.py` imports them back, so
`library_screen_module._ANALYZE_ORIGIN_MEDIA`/`..._IMPORT` still resolve for
`Tests/UI/test_library_ingest_analyze_skipped.py:611`/`:612`, the two
assertions task 1's forward note flagged. Re-spelling `"media"` in a second
file would have broken the single-source guarantee those constants exist to
give.

---

## 5. Two exclusion classes the STATIC census could not produce

Both were found by this task's own battery after the RED tuple was written,
which recipe §3 records as the expected shape of the work.

### 5.1 Bypassed-construction lifecycle (2 names)

`Tests/UI/test_library_screen_reuse.py::test_on_screen_suspend_stops_every_
timer_in_isolation` invokes `on_screen_suspend` — a SCREEN-RESIDENT lifecycle
hook — directly on an `object.__new__(LibraryScreen)` instance. Its two media
callees' delegators reach `self._media_controller`, which `__init__` never
built:

```
tldw_chatbook/UI/Screens/library_screen.py:8915: in _stop_library_media_selection_debounce
    return self._media_controller._stop_library_media_selection_debounce()
E   AttributeError: 'LibraryScreen' object has no attribute '_media_controller'
```

This is the CONTROLLER-PR analogue of recipe §3's SEVENTH shape (which bites a
STATE PR through the generated property shim): same bypassed-construction
cause, different attribute, equally undeferrable — red at the move commit
itself. Unlike the state PR's version there is no cheap one-line fixture seed,
because seeding a controller means constructing one with 83 keyword arguments;
the accommodation is exclusion. **Task 1's forward note had already flagged
both names as exclusion candidates for exactly this call path** — a
predecessor's forward notes are hypotheses to TEST, not facts and not noise.

Closing census, run empirically rather than statically: all **9 files** in the
repo containing `object.__new__(LibraryScreen)`/`LibraryScreen.__new__`, run
together — **1 failed / 426 passed**, the one failure being
`test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_footer_
when_retry_availability_changes`, documented in recipe §7 since wave-5 task 3.

### 5.2 Screen identity in a MEMBERSHIP form (3 names) — the expensive one

`_commit_library_media_return`, `_library_media_focus_target_matches_receipt`
and `_resolve_library_media_settlement_target` each spell
`self in <widget>.ancestors` / `self not in <widget>.ancestors`.
`Widget.ancestors` is the DOM parent chain up to the screen, so a moved body's
`self` — the CONTROLLER — is never a member: the guard flips to permanently
False (or permanently True negated) and silently changes behaviour, exactly
like recipe §3's Forms A/B/C.

**The census gap this exposes.** The recipe's sixth-shape census looks for
`... is self` / `... is not self` (an `ast.Compare` with `Is`/`IsNot`) and for
bare `self` passed as a call argument. This shape is neither: it is an
`ast.Compare` with an `In`/`NotIn` operator, and a first pass of that census
over all 251 candidates returned **zero identity hits**. The move shipped into
the battery and produced **19 failed / 144 passed in the return-settlement
cluster against a 2-failed / 161-passed isolated baseline** — 17 branch-unique,
signature assertion `AssertionError: assert 'layout-settlement-failed' ==
'exact-settled'`. After the three exclusions the same three files measure
**2 failed / 161 passed**, name-for-name identical to the baseline.

**The standing correction:** census bare `self` as an operand of ANY
comparison operator — and, stronger and cheaper, census **EVERY bare `self`
`ast.Name` in a moved body that is not the receiver of an attribute access**.
That form is exhaustive by construction. Over the final mover set it returns
exactly 10 occurrences: 3 `getattr(self, "<literal>")` reads (bound), 4
`_sync_library_canvas(self, ...)` duck-typed forwards (bound), 3 `ancestors`
membership tests (excluded). Whole class, proven rather than sampled.

### 5.3 Callback identity (1 name) — fix round 1

`_exit_library_media_viewer` schedules its continuation as
`self.call_next(self._apply_library_media_list_return, None)`, and
`Tests/UI/test_screen_navigation.py:3109` asserts
`continuation == screen._apply_library_media_list_return` on a REAL screen.
Moved, `self` is the controller, so the captured callback is the CONTROLLER's
bound method and the equality fails. It was the ONE branch-unique name in the
paired adjacent-file sweep (§7.3).

**Excluding the CALLEE would not have fixed it**: the controller's
late-binding property returns the injected `lambda`, a third object again. The
CALLER is what has to stay, so `self.<name>` resolves to a screen-bound method
whose `__self__`/`__func__` match the assertion's own.

Census added to the checklist, run over all 251 candidates across `Tests/`:
every cluster name referenced as a BARE attribute on any receiver (not the
func of a call, not an assignment target) — **10 names**, 9 already excluded on
other grounds and one (`_navigate_to_media`) where the receiver holds a
`MagicMock` set on the instance, which a delegator cannot disturb.

---

## 6. Byte-for-byte evidence

Run against `3dd7a745a` (the RED commit — screen untouched) at every stage,
final state:

```
movers checked: 140
missing from the controller: []
TEXT-identity failures: []
AST-identity failures: []
EXCLUDED methods that drifted on the screen: []
screen method-name SET identical: True | added: [] | removed: []
non-media screen methods that drifted: ['__init__']
```

Both checks run per mover: exact SOURCE-TEXT equality of the block (first
decorator line through `end_lineno`) and `ast.dump(..., include_attributes=
False)` equality. The method-NAME SET comparison is the stronger form of the
count check — a count match can hide an add-and-remove pair; a set match
cannot. `__init__` is the only non-media screen method that changed (the
construction site).

**Zero movers carry comment lines outside their own AST range** — measured,
not assumed, so unlike the prompts move nothing was orphaned behind a
delegator.

---

## 7. Verification battery

### 7.1 RED/GREEN, proven in an isolated worktree

At `3dd7a745a` in the isolated baseline worktree (own `uv venv`, verified
resolving its own tree):

```
5 failed, 6 passed
E   ModuleNotFoundError: No module named
    'tldw_chatbook.UI.Library_Modules.library_media_controller'
```
(`test_media_controller_owns_its_cluster`, `test_screen_delegates_media_
handlers`, `test_media_cluster_staticmethods_forward_to_the_controller_class`,
`test_media_controller_exposes_every_state_field`, `test_media_controller_
binds_every_name_its_moved_bodies_use`.) At HEAD: **11 passed**.

### 7.2 Guards

| Suite | Result |
|---|---|
| the 8 Library wiring suites + recompose ratchet + media characterization + ui_ready census + preimport closure + the seeded `__new__` fixture + both ratchets | **117 passed, 3 failed** |
| — the 3 failures | the 2 documented `chat_screen.py` rows and `library_media_browse_controller.py` 410-vs-371 |
| `Tests/Architecture` (whole root), branch | 19 failed / 582 passed / 1 skipped |
| `Tests/Architecture` (whole root), isolated baseline at `3dd7a745a` | 24 failed / 575 passed / 1 skipped |
| diff of the two sorted failure-name lists | **the ONLY difference is the 5 RED wiring names, baseline-unique.** Zero branch-unique. |
| `./scripts/preflight.sh` | all derived-artifact checks passed |

The 19 shared `Tests/Architecture` failures are exactly task 1's recorded set:
the 15 recipe-§7 names (including the three `(file, line)`-keyed
`test_timer_path_static_update_inventory` rows and the two
`test_worker_exclusive_group_inventory` rows — the shapes a pure move is most
likely to break, and none moved), the 3 ratchet rows, and
`test_persona_visual_runtime_boundary`.

**`library_media_browse_controller.py` was left flagged, not absorbed.** No
commit in this task edits that file, so the plan's absorb-only-if-touched rule
keeps two waves of refusal precedent intact.

**Diagnostic inventory.** The move relocated 7 `logger` statements from
`library_screen.py` (95 → 88) into the controller (0 → 7). Every one was read
by digest before regenerating, per the guard's own instruction: the 7 removed
digests are byte-identical to the 7 added ones (the digest covers indentation,
so the 4-space class indent is confirmed preserved too), 0 moved-only, 0
reworded, none newly interpolating user content, a secret, a path or a URL.
Only then `--write`. Fix round 1 reported 7 → 7 with zero added/removed/moved,
so no second regeneration was needed.

### 7.3 Regression evidence — paired, sequential, isolated baseline

All sweeps run SEQUENTIALLY (never two 8-worker invocations on one machine).

**Sweep A — the 17 dedicated media suites** (`-n 8 --dist worksteal`, branch):
**8 failed / 530 passed** in 220s. Every one of the 8 is a documented §7 name
or a task-1 §7 addition:
`test_a_new_document_rescans_for_the_same_query`,
`test_no_change_traversal_builds_no_preview_and_copies_no_content`,
`test_focus_traversal_builds_zero_bodies_for_pass_through_rows`,
`test_viewer_return_waits_for_geometry_then_scrolls_before_row_focus`,
`test_authoritative_recompose_rearms_before_replacement_geometry`,
`test_loading_banner_paints_in_place_without_body_rebuild`,
`test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke`,
`test_image_item_traversal_wall_time_probe`. **Zero undocumented.**

**Sweep B — 19 adjacent files** referencing moved names, identical
single-process command on both trees:

| | failed | passed | wall |
|---|---|---|---|
| branch (post-fix, `7fee56dd4`) | 40 | 719 | 763.3s |
| isolated baseline (`3dd7a745a`) | 42 | 717 | 723.8s |

**40 shared, ZERO branch-unique, 2 baseline-unique**
(`test_screen_navigation.py::test_search_route_round_trips_to_the_library_rag_
row` — §7, wave-5 task 1; `::test_skills_route_lands_on_library_with_skills_
row_selected` — §7, wave-3 task 4). *Before* fix round 1 the same pairing was
43-vs-42 with 1 branch-unique — the callback-identity finding of §5.3.

**Sweep C — `Tests/UI/test_library_shell.py -k media`**, paired:

| | failed | passed | deselected |
|---|---|---|---|
| branch | 6 | 121 | 698 |
| isolated baseline | 6 | 121 | 698 |

**6 shared, ZERO branch-unique, ZERO baseline-unique** — the two failure-name
lists are identical. *(The unfiltered whole-file run was attempted and reached
~85% before the harness stopped the background process; it was reproducing the
fd-exhaustion cascade recipe §7 already documents for this file — task 1's own
paired unfiltered run measured 228-vs-229 with zero branch-unique. The
media-filtered pairing above is the targeted substitute and is the stronger
signal for this diff.)*

**Return-settlement cluster, single-process, both trees** (the check that
found §5.2): after the fix, branch **2 failed / 161 passed**, isolated
baseline **2 failed / 161 passed**, same two names.

**The 9 `__new__`-bypass files**: 1 failed / 426 passed (documented).

### 7.4 Probe (§9) — paired, and the pair run TWICE with the order swapped

| Run | order | Σ settle (ms) |
|---|---|---|
| branch | 1st | 2,916 |
| baseline | 2nd | 2,847 |
| baseline | 1st | 2,887 |
| branch | 2nd | 2,880 |

The wall-clock penalty follows the ORDER, not the tree (branch +2.4% when it
ran first, −0.2% when it ran second) — the warm-up artifact §9's order-swap
rule exists to expose.

The load-INDEPENDENT columns are the verdict, and they are flat:
`recompose = 0` on all eight interactions in all four runs; `full-update`
identical per interaction across all four (2/2/2/1/1/1/1/1); `nodes` identical
(119/119/119/114/114/119/114/119); mount counts vary by ±4 in BOTH directions
on BOTH trees (85↔89, 175↔179), i.e. run-to-run. **No per-frame regression.**

---

## 8. Pin trajectory

`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`:
**37333/1282 → 34754/1282** (method count unchanged, as every pure controller
move's must be).

`_BUDGETS[".../library_media_controller.py"]`: born **4461**.

Line delta **−2579**, reconciled term by term off measurements, not estimates:

| Term | Lines |
|---|---|
| moved mover blocks (first decorator → `end_lineno`) | −3202 |
| delegator lines (2–7 each; 3 staticmethods carry a 4-line function-local import) | +343 |
| four relocated module constants (with comments) | −12 |
| those four names added to the `screen_constants` import block | +4 |
| born-lazy `LibraryMediaController` import | +3 |
| construction site (92 named dependencies, under a 3-line comment) | +249 |
| **net** | **−2579** |

All three in-file copies of each figure were updated together (row, arithmetic
derivation, narrative line), per §6's three-places rule; the commit messages'
numbers were re-derived last, after the final edit.

---

## 9. Deviations, errata and process notes

- **The RED tuple was amended twice**, in the move commit (146 → 141: the two
  bypassed-construction-lifecycle names and the three membership-identity
  names) and in fix round 1 (141 → 140: the callback-identity name). Recipe §3
  names this the expected shape of the work; every count in the wiring test,
  the controller docstring, both ratchet comments and the report was
  re-derived after each correction rather than patched piecemeal.
- **A first draft of the module-globals census was line-anchored and wrong.**
  It scored `resolve_ingest_analysis_provider` at zero because its five patch
  sites span two lines. Redone by `ast`. Recorded because the same class of
  miss is already in the recipe from wave-5 and recurred anyway.
- **`git checkout HEAD -- .` + full regeneration** was used for each mover-set
  correction rather than hand-patching the generated files, so every
  intermediate state is reproducible from the scripts and the byte-for-byte
  proof re-ran clean each time.
- One background sweep (`Tests/UI/test_library_shell.py`, unfiltered) was
  stopped by the harness at ~85%; a targeted `-k media` pairing replaced it
  (§7.3). An earlier attempt at the same file's baseline half was killed by a
  foreground 10-minute tool timeout propagating SIGTERM to a `nohup`'d child —
  the fix was to use the tool's own detached background mode, and the lesson
  is that `nohup … &` inside a foreground call is NOT detached from the call's
  own timeout.
- **Nothing was pushed. `progress.md` was not touched.**
- The isolated baseline worktree (`w7t2base` @ `3dd7a745a`, own `uv venv`) is
  removed at task close.

---

## 10. Concerns handed forward to task 3

1. **The delegator-prune census is over 140 names, and §4's third whitelist
   member is inert here** (0 `on_<message>` handlers, re-verified) — but the
   `@on` (48) and `action_*` (5) members are not: 53 of the 140 KEEP
   unconditionally.
2. **The test retarget is still the largest of the program** (task 1's 1,214
   boundary-matched flat-name occurrences across 36 files in five roots), and
   the shim block that makes them work is the one task 3 deletes.
3. **111 exclusions stay screen-resident and full-bodied.** Task 3's dead-import
   sweep must not treat their imports as dead, and its `_SURFACE` check must be
   by exact name, not a lowercase `media` grep (`LIBRARY_MEDIA_*` names spell
   it uppercase — the prompts series lost 5 saves to exactly that).
4. **Four constants now live in `screen_constants.py`.** `library_media_state.py`'s
   own docstring still says `_ANALYZE_ORIGIN_MEDIA` "stays a `library_screen`
   module" constant — true of the ATTRIBUTE (it is imported back and still
   resolves at that path) but no longer of the DEFINITION. A one-phrase
   correction for task 3's stale-prose sweep.
5. **`library_media_browse_controller.py`'s standing red is still standing.**
   This task edited no controller file other than the new one.
6. **Recipe §7 additions this task earned** (task 3 or 4's scope to write):
   the `test_library_media_return_settlement.py` pair is already task 1's
   addition; nothing new to add — every failure this task saw was already
   documented.
