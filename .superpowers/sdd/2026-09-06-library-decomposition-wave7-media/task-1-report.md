# Wave-7 Task 1 — Media state PR (media series 1/3)

**Status:** COMPLETE (not BLOCKED). The media cluster is cleanly extractable at
the field level.

**Branch:** `refactor/library-decomp-wave7-media`
**Parent (wave-7 start):** `83e17323e`
**Commits (rev-parse only, never transcribed):**

| Commit | Kind | Subject |
|---|---|---|
| `600192d6e` | RED / pins | `test(library): characterization + wiring pins for the media extraction series (RED)` |
| `76b248c05` | pure move | `refactor(library): media state object + shims (media series 1/3)` |
| `23e442ad9` | blame-ignore | `chore(library): blame-ignore the media state-PR pure move` |

Worktree: `.worktrees/library-decomp-foundation`, `.venv/bin/python` (verified
resolving its own tree). No pushes. `timeout` unavailable; every timed run used
`perl -e 'alarm N; exec @ARGV'`.

---

## 1. Ownership analysis

### 1.1 The census, re-derived (never carried over)

The plan's rough measure was "~122 distinct flat `_library_media_*` attribute
names". **122 is the count of `_library_media_*` names appearing as
`self.<attr>` ANYWHERE in the class body — 39 of them are METHOD references,
not fields.** Re-derived mechanically at the parent tree:

| Measurement | Value | How |
|---|---|---|
| `self.<attr>` names anywhere in `LibraryScreen` | 1,095 | `ast` walk, all methods |
| …of which `_library_media_*` | 122 | (the plan's number) |
| …of those, `__init__`-stored | 82 | `ast` walk, `Store` ctx, `__init__` only |
| …of those, a `LibraryScreen` method name | 39 | cross-checked against the class's `FunctionDef` set |
| …of those, a genuine class-body attribute | 1 | `_library_media_arrival_note` |
| `__init__`-stored attributes containing "media" (substring, NOT `startswith`) | 84 | adds `_selected_media_id`, `_library_pending_list_entry_media_return` |
| **Total media-cluster attribute candidates** | **85** | 84 + the class-body one |

The substring (not `startswith`) form is the conversations exemplar's
"startswith enumeration trap" lesson; it is what found `_selected_media_id`.
The class-body `Assign`/`AnnAssign` scan is what found
`_library_media_arrival_note` — the export series' `origin_row_id` shape,
invisible to an `__init__`-only census.

A separate scan of all 202 non-media-named `__init__` fields for any whose
**every** consumer is media-named returned **0**, so no bare-named
media-cluster field exists beyond `_selected_media_id`.

### 1.2 Verdicts

**82 MOVE / 2 WIRING / 1 BLOCKED / 0 blocked by the ≥2-subsystems rule.**

**WIRING (2)** — live prior-extracted controller instances, the
`_conversation_reader_controller`/`_library_collections_capture_controller`/
`_library_prompt_browse_controller` precedent. Constructed in place with
late-binding lambdas, untouched:

- `_library_media_browse_controller` (`LibraryMediaBrowseController`)
- `_library_media_trash_browse_controller` (`LibraryMediaTrashBrowseController`)

**BLOCKED (1) — `_library_pending_list_entry_media_return`.** Media-named,
holds a `_LibraryMediaReturnReceipt`, and the census's own tagging heuristic
scores its direct users as shell/plumbing — but the field is shared shell
state:

- Its **only two writers** are `_arm_library_list_entry_focus` and
  `_disarm_library_list_entry_focus`. `_arm_library_list_entry_focus`'s own
  docstring names its callers: *"a fresh rail-row press landing on
  Media/Notes/Prompts/Skills, and every 'back to list' exit from that canvas's
  viewer/editor"* — four subsystems, which is the recipe §2 ≥2-subsystems rule
  by construction. `library_prompts_controller.py` already binds
  `_arm_library_list_entry_focus` as a named constructor dependency, so a
  Prompts code path demonstrably drives this write today.
- It is one of four members of a single shell family
  (`_library_pending_list_entry_focus`,
  `_library_pending_list_entry_focus_anchor`,
  `_library_list_entry_focus_generation`) assigned and cleared in the same two
  statements. Moving one member into a subsystem state object while its
  siblings stay on the screen is the split the One Home Rule exists to prevent.
- Media's OWN receipts of the identical type — `_library_media_viewer_return`
  and `_library_media_trash_return`, assigned two lines later — **do** move.
  That is the clean line: the shell's copy stays, the subsystem's copies go.

**MOVE (82).** The recipe §2 script output over all 85 candidates found
**exactly one non-media, non-shell user in the whole cluster**:
`_record_library_notes_focus_interaction` (Notes-named) reads
`_library_media_view`. Read: it is a read-only route guard —
`self._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA and
self._library_media_view == "list"` — consulting where Media is before deciding
whether a Notes focus interaction applies, exactly as the fifteen shell guards
in the same list do. All four categories of writer are media-named or
shell/plumbing; no Notes method ever writes it. This is the prompts series'
already-landed `_library_prompts_mutation_in_flight` precedent (three
Notes-named readers, zero Notes writers), applied identically.

Two further cross-reader groups recorded rather than waved past:

- **The review-set cluster** (`_review_set_picker_worker`,
  `_review_dismiss_undo_worker`, `_review_dismiss_receipt_name`,
  `_walk_active_review_set_unguarded`, `_active_review_set_banner`,
  `_active_review_loaded_at_last`, `_toggle_reviewed_unguarded`,
  `_review_these_worker`, `_order_selected_review_pairs`, `open_chunking_lab`)
  reads and writes `review_dismiss_receipt` and reads `reader_session`. Review
  sets are **not** one of the eleven subsystems in recipe §8 — they are a
  media-only feature built ON the media reader (every item is a
  `backing_media_id`; every walk step calls `_select_library_media_reader_row`;
  `open_chunking_lab` reads `reader.loaded_backing_id`). They tag
  shell/plumbing and their references move with Media, retargeted at cleanup.
- **`reader_layout`** is read by `_toggle_library_media_reader_pane` — the
  generic multi-subsystem pane dispatcher which is media-NAMED but reads four
  state objects. Collections, conversations, skills and prompts all already
  moved their identically-named field past this same reader; it stays
  screen-resident and becomes a controller-PR exclusion, per the prompts ruling.

### 1.3 The suspend/resume seam (dev's TASK-31521), ruled field by field

This was the wave's flagged high-risk analysis. The result is unusually clean,
and it is a *measured* result, not an assumption. Media-named attribute
references, per screen-lifecycle method (`ast`, at the parent tree):

| Lifecycle method | Media FIELDS touched | Verdict |
|---|---|---|
| `on_screen_suspend` | **none** — it references only two media METHOD names, `_stop_library_media_selection_debounce` and `_stop_library_media_filter_timer` | no field-level entanglement at all |
| `on_screen_resume` | **none** | — |
| `_refresh_library_visit_surfaces` (the resume dispatch seam) | writes `trash_query_draft`, `trash_input_error`, `trash_type_choices_visible`; reads `view` (+ the 2 WIRING controllers) | shell/plumbing writers → MOVE (recipe §2), retargeted at cleanup |
| `on_unmount` | writes `current_owner`, `geometry_floor`, `geometry_floor_owner_identity`, `last_settlement_outcome`, `lifecycle_generation`, `progress_inflight_write`, `return_settlement`, `trash_mounted_authority`; reads 5 more (`analyze_done`, `analyze_running`, `analyze_total`, `progress_pending_writes`, `progress_write_worker`) | shell/plumbing → MOVE |
| `on_mount` | writes `reader_session`, `trash_mounted_authority`; reads `detail`, `view`, `_selected_media_id` | shell/plumbing → MOVE |
| `save_state` / `restore_state` / `_apply_navigation_context_state` | `view`, `type_filter`, `_selected_media_id` | shell/plumbing → MOVE |
| `compose_content` / `check_action` / `on_descendant_focus` | `view`, `reader_layout`, `return_settlement`, `_selected_media_id`, `select_mode`, `bulk_delete_in_flight`, `confirming_bulk_delete`, `reader_session` | shell/plumbing → MOVE |
| `apply_navigation_context`, `flush_pending_work` | none | — |

**Why `on_screen_suspend` needs nothing from this PR:** it reaches
`selection_timer`/`filter_timer` only *through* the two media methods, whose
bodies are untouched by a state PR and keep resolving the flat names through the
generated shim. Its own flat-name `getattr` string loop lists only
`_library_notes_autosave_timer` and `_library_source_snapshot_timeout_timer` —
no media name — so the "a `getattr` for the old flat name silently returns
None" hazard the ingest and prompts cleanups each had to fix does not arise
here (it becomes a cleanup-PR concern for THIS series only if task 3 deletes
the shim, which is exactly what the two comments already in that method
describe).

**The Timer/Worker question, decided by precedent rather than surface
analogy.** The brief asked specifically how ingest's `path_debounce_timer`
(moved, state) was ruled versus the WIRING controller instances. Checked
against the live modules, not from memory:

- `library_ingest_state.py:137` `preflight_worker: Worker | None = None`
- `library_ingest_state.py:148` `path_debounce_timer: Timer | None = None`
- `library_prompts_state.py:222` `debounce_timer: Timer | None = None`

Both prior series moved Timer- and Worker-*handle* fields into state, and both
are stopped from `on_screen_suspend`/`on_unmount` — the same shape as media's.
The WIRING exclusion is specifically for a live **controller instance**
constructed with screen-closing lambdas, which is a wiring graph edge, not
data. A `Timer`/`Worker` handle is a value. Verdict, following the reasoning:

- `selection_timer`, `filter_timer` (`Timer | None`) → **MOVE**
- `progress_write_worker` (`Worker | None`) → **MOVE**
- `preview_factory` (a `Callable[..., Widget] | None` injected via
  `__init__`'s own keyword parameter) → **MOVE**, as a constructor argument. It
  is injected test data, not a controller instance; the WIRING precedent does
  not reach it.
- `reader_session` (`LibraryMediaReaderSessionState`, a frozen dataclass) →
  **MOVE**, the prompts `block_state: PromptBlockEditorState` shape.

The screen-lifecycle fields themselves — `_library_screen_suspended`,
`_library_visit_entered` — are not media-named, are not in this census, and
stay screen-owned, accessor-at-most, per the plan's Global Constraints.

### 1.4 Construction position, placeholders and constructor arguments

`self._media_state = LibraryMediaState(...)` is constructed immediately after
`self._prompts_state`, i.e. **before** the shared
`_load_library_reader_preference_snapshot()` tuple-unpack — the forced-early
construction the collections, skills and prompts series each document.

**4 fields keep their original `__init__` line** (dataclass defaults are
momentary placeholders, overwritten through the shim when each original line
runs):

| Field | Why it cannot be a constructor argument |
|---|---|
| `reader_preferences` | one of eight targets of the shared tuple-unpack |
| `reader_persistence_locks` | shares the local `library_pane_persistence_lock` |
| `layout_refresh_generation` | mirrors `_library_reader_layout_refresh_generation`, itself read off `app_instance` further down |
| `reader_layout` | derived from `reader_preferences` after the unpack settles |

**3 fields become constructor arguments** (the export series' `form`
precedent — their original lines are deleted):

| Field | Source available at the early construction point |
|---|---|
| `preview_factory` | `__init__`'s `preview_widget_factory` keyword parameter |
| `preview_factory_injected` | `preview_widget_factory is not None` |
| `analyze_origin` | the module constant `_ANALYZE_ORIGIN_MEDIA` |

`_ANALYZE_ORIGIN_MEDIA` deliberately **stays in `library_screen.py`**:
`Tests/UI/test_library_ingest_analyze_skipped.py:611` asserts
`library_screen_module._ANALYZE_ORIGIN_MEDIA == "media"`, and the constant's own
comment says it exists so "the init default … and its two readers cannot drift
apart". Passing it preserves that single source of truth; re-spelling `"media"`
in the state module would have broken it silently.

**4 pure factory calls folded into `default_factory`** — `RowSelection("media")`
(a plain accumulator, `__init__` sets two attributes), `LibraryMediaReaderSessionState()`
and `MediaBrowseScope()` (both frozen dataclasses with validating
`__post_init__`s and no side effects), and the `object()` memo sentinel. Each
was read before folding, matching the prompts series' own
`PromptSelectionBasket`/`local_prompt_capabilities` folds.

**75 remaining original lines deleted outright.** 75 + 3 = 78 assignments
removed; 78 + 4 kept = 82 moved fields.

### 1.5 Prefixes — TWO families, not three

- `_library_media_` — 81 of 82 fields. **No plural variant**: "media" is
  already both singular and plural, so unlike Skills and Prompts there is no
  `MEDIA_PLURAL_STATE_FIELDS` set.
- bare `_` — exactly one field, `selected_media_id` (original attribute
  `_selected_media_id`), the `_selected_skill_name`/`_selected_prompt_id`
  third-prefix shape.

`media_state_shim_attr()` is the single authoritative resolver, shared by the
screen's shim loop and the wiring test.

---

## 2. Characterization spot-check

**Coverage was verified by reading before anything new was pinned**, as the
brief required, and every count below is re-derived.

Cluster shape at this tree: **251 media-named `FunctionDef`s** (251 unique
names — no property/setter-pair gap), of which **79 carry an `@on` decorator**,
**12 are `action_*`**, and — worth recording for task 2 — **0 are
`on_<message>` name-dispatched handlers.** (The plan warned media "almost
certainly owns name-dispatched handlers"; it does not. The only `on_[a-z]`
methods on `LibraryScreen` are 10 Textual lifecycle hooks and the 7
prompts-owned `on_prompt_block_editor_*`.)

Of the 79 `@on` handlers, **76 carry a string SELECTOR** and **3 are
Message-typed** (`_toggle_library_media_reader_pane` ← `PaneToggleRequested`,
`_resize_library_media_reader_shell` ← `MediaShellResized`,
`_handle_library_media_row_geometry_changed` ←
`LibraryMediaRowGeometryChanged`). The 3 were censused separately by message
type: all three are referenced in `Tests/` at the message level
(`test_library_adaptive_reader_shell.py` captures `PaneToggleRequested`;
`test_library_media_return_settlement.py` posts `MediaShellResized` and names
`LibraryMediaRowGeometryChanged`), so none is an unpressed gap — and
`_toggle_library_media_reader_pane` is a controller-PR exclusion in any case.
The 76 selector-bound handlers are what the selector census below measures.

**Census method** — four passes, because the obvious one is wrong:

1. Exact id/class **boundary** matching. A substring match scores
   `#library-media-review` as covered off `#library-media-review-selected`; it
   inflated the first pass by 10 hits.
2. A ±4-line press window around each selector hit (`.press(`, `.click(`,
   `pilot.click`, `.value =`, `post_message`, `OptionSelected`, `Submitted`,
   `.focus(`) — wider than a same-line grep, per the collections series' lesson.
3. For the four CLASS-bound handlers, the derived `#<id>-N` **row** spelling,
   because rows are pressed by id and never by class.
4. A separate per-handler **METHOD NAME** grep for the unbound
   `LibraryScreen.<name>(fake, event)` call shape several media suites favour.

**What was read and what was not, stated plainly:** the *covered* verdicts are
the heuristic's (a press-shaped token within ±4 lines of an exactly-bounded
selector), the same standard the collections and prompts series used. Every
*non-covered* verdict, and every verdict a later pass *changed*, was read
against the hitting test before being written down. That asymmetry is
deliberate — a false "covered" costs a missing pin, a false "gap" costs a
wasted test, and only the second is cheap to detect by reading.

**Result at the PARENT tree (re-run there, in the isolated worktree, so the
numbers describe pre-existing coverage rather than this task's own additions):
71 of the 76 selector-bound handlers were already covered** — **62** by a real
selector press, **9 more only by a direct unbound
`LibraryScreen.<name>(fake, event)` call** (all in
`test_library_multiselect_media.py`, `test_review_set_walker.py`,
`test_library_media_trash.py` — that bypass shape's signature, and a forward
note for task 2's exclusion set). Five were uncovered by either route.

Two of the five are **not state-PR concerns**, confirmed by reading each body
rather than inferred: `handle_library_media_review_these` and
`handle_library_media_review_sets` are each a bare `event.stop()` plus
`run_worker`, touching **no** moved field.

Three handlers the FIRST census pass wrongly scored as gaps were overturned by
passes 1–3, which is why those passes exist:
`handle_library_media_reader_mode` (bound by class
`.library-media-reader-mode`, pressed by the id
`#library-media-reader-select-analysis` in
`test_library_reader_press_scope_t22228.py:99` — the prompts series'
bound-by-class/pressed-by-id shape, reproduced exactly),
`handle_library_media_highlight_delete` and `handle_library_media_trash_row`
(both reached by the derived `#<id>-N` row spelling). Each was confirmed by
reading the hitting test, not by the grep verdict alone.

**Three genuine gaps, all pinned** in
`Tests/UI/test_library_media_characterization.py` (3 tests, all GREEN at the
RED commit — which is what a characterization pin is for):

| Handler | Moved state it reads | What existing tests did instead |
|---|---|---|
| `handle_library_media_previous` | `bulk_delete_in_flight` | 8 selector hits across 2 files, all `.disabled`/`.tooltip`/focus-identity assertions; pressed zero times |
| `handle_library_media_review_selected` | `bulk_delete_in_flight` (via a `getattr` literal), `row_selection` | 2 hits in `test_library_media_toolbar_adapt.py`, both label/geometry, on a standalone canvas host that cannot dispatch a SCREEN `@on` handler at all |
| `handle_library_media_open_original` | `detail` | zero references anywhere in the repo outside its own decorator and the widget that yields the button |

Each test asserts through a signal **provably tied to the handler's own logic**,
not a DOM end-state (recipe §3's false-negative warning): the applied pager
scope and rendered page status; the exact `id_allowlist` the selection produces
in the media scope service's recorded calls; the URL handed to
`webbrowser.open`.

`webbrowser.open` is patched **on the `webbrowser` module itself**, never at the
`library_screen`-scoped path — so the pin survives this series' controller PR
moving that body to another module. Recipe §3's eighth bypass shape, avoided by
construction rather than discovered later. No new unbound-fake-self call was
introduced, so the pin adds nothing to task 2's exclusion set.

**No live bugs found**; all three are coverage gaps.

*Stability note:* the very first invocation of the new file (immediately after
writing it, in a session where pytest's own tmpdir garbage collector was
clearing several hundred stale directories) failed
`test_media_previous_page_returns_to_the_page_before_it` once. It was hardened
to wait for `#library-media-next` to be *enabled* (not merely for page 1 to be
applied) before pressing, and has since run **14 consecutive green** (9 before
the change, 5 after).

---

## 3. The state module, the screen, and the shims

`tldw_chatbook/UI/Library_Modules/library_media_state.py` — a `@dataclass` with
all **82** fields in original `__init__` order, verbatim defaults and comments,
plus `MEDIA_UNPREFIXED_STATE_FIELDS` and `media_state_shim_attr()`.

**Byte-for-byte comment verification, run in BOTH directions:**

- Every comment line in the dataclass body was checked against the screen's
  three media regions at the parent tree. 150 comment lines total; 19 are new
  (the placeholder / constructor-argument explanations, and two `#` separators
  inside them); the other **131 are byte-for-byte identical** to their originals.
- The converse: every comment line in those regions that is *not* in the state
  module was listed and read. All **14** belong to fields that did not move
  (`_review_dismiss_undo_in_flight` ×2, `_library_ingest_analyze_outcomes` ×9,
  `_library_list_entry_focus_timer` ×3). Nothing was orphaned and nothing was
  taken that should have stayed.
- The deletion pass itself refuses to run if a comment line immediately above a
  to-delete statement is not present in the state module (it raises rather than
  deleting), so the two counts reconcile by construction: **131 comment lines
  deleted from the screen = 131 relocated occurrences.**

*(A first draft of that pass swept the WHOLE file for matching comment text and
wrongly deleted 8 unrelated lines elsewhere — including bare `#` separators
inside other subsystems' comment blocks. It was caught by reading the diff,
reverted with `git checkout`, and rewritten to walk upward only from each
to-delete statement. Recorded because the failure mode is silent: those 8
deletions parsed fine and no test would have noticed.)*

**Shim block** (`library_screen.py`, module end, sentinel-wrapped, 20 lines):
the established generated-property loop over `dataclasses.fields`, with
`_n=_lms_field.name` **default-argument closure binding on BOTH the getter and
the setter** (the closure-binding trap that would otherwise alias all 82 names
onto the last field), the flat name resolved through `media_state_shim_attr()`.

**Bypass-fixture census (recipe §3's seventh shape).** Content grep across ALL
of `Tests/` — never a `-k` name filter, per that shape's own filter-blindness
lesson — for `object.__new__(LibraryScreen)` / `LibraryScreen.__new__` /
`__new__(LibraryScreen`:

- **9 files** contain such a spelling.
- **2 co-occur** with a moved flat name.
- **1 is real:** `Tests/UI/test_library_screen_reuse.py:224`
  `test_on_screen_suspend_stops_every_timer_in_isolation` — it `setattr`s
  `_library_media_selection_timer` and `_library_media_filter_timer` and three
  settlement fields that `_disarm_library_list_entry_focus` resets. Fixed with
  **one** seeding line (`screen._media_state = LibraryMediaState()`) immediately
  after the bypassing construction, mirroring the `_ingest_state`/`_prompts_state`
  seeds already beside it. Zero assertions touched. (4 passed.)
- **1 is a false positive:** `Tests/UI/test_library_shell.py:24226` — the string
  appears only inside `wire_bypass_ingest_controller`'s own docstring prose;
  every screen in that file is a real `LibraryScreen(app)`. Confirmed by
  reading the fixture, not by the grep.
- The **other 7 `__new__` files were RUN** to rule out an indirect write through
  a media shim on a bypassed screen: **422 passed, 1 failed**, and that one is a
  documented pre-existing red (see §5).

**Dynamic-dispatch string sites recorded for task 3** (all work unchanged
through the shim now; each will need a retarget when the shim is deleted):
`_library_media_reader_preferences` as a dict value in
`_replace_library_reader_preference` and `_persist_library_reader_preference`;
`_library_media_type_choices_visible`/`_library_media_sort_choices_visible` in
`_library_open_choice_strip`'s return tuple and its `canvas_kind` dict; and 11
`getattr(self, "_library_media_*", <default>)` sites. Both dict paths already
resolve dotted strings (`operator.attrgetter` /
`_assign_library_reader_preferences_attribute`), so the retarget is a value
edit, not a mechanism change.

---

## 4. Size ratchet

Fresh `_measure()` through the guard's own function, post-edit, after the RED
commit:

`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`:
**37537/1282 → 37333/1282** (methods unchanged — a pure field move touches zero
`FunctionDef`s).

Delta **−204**, reconciled term by term off `git diff --numstat` rather than
estimated:

| Term | Lines |
|---|---|
| removed field-statement lines (78 assignments) | −110 |
| relocated comment lines | −131 |
| one blank line left doubled by the class-body removal | −1 |
| `library_media_state` import | +4 |
| construction site + its 9-line why-this-is-early comment | +14 |
| sentinel-wrapped generated shim block | +20 |
| **net** | **−204** |

All three in-file copies of the figure were updated together (row, arithmetic
derivation, narrative `Fresh _measure()` line) — the §6 three-places rule. The
commit message's numbers were re-derived last, after the final edit; the
`-203`/`37334` figures from an intermediate draft (before the doubled blank
line was removed) exist nowhere in the committed tree.

`Tests/Architecture/test_library_modules_size_ratchet.py` globs
`tldw_chatbook/UI/Library_Modules/*_controller.py` only, so a **state** module
needs no row there — verified by reading that file's own docstring, not assumed.

**Pure-move proof, stronger than the method COUNT.** The method-NAME SET of
`LibraryScreen` was compared between `83e17323e` and HEAD by `ast`: **identical
in both directions** (zero added, zero removed). The media-named subset is 251
at both revisions, with 79 `@on`, 12 `action_*` and 0 `on_<message>`. A count
match can hide an add-and-remove pair; a set match cannot.

---

## 5. Verification battery

### 5.1 RED/GREEN evidence for the wiring test

Proven in a **dedicated isolated `git worktree` checked out at the RED commit
`600192d6e`** (parent tree + the three new files only; `library_screen.py`
untouched), with its own `uv venv` verified resolving that tree:

```
E   assert not ['_library_media_analyze_choice', '_library_media_analyze_done',
                '_library_media_analyze_failed_ids', '_library_media_analyze_origin',
                '_library_media_analyze_reason_cache', '_library_media_analyze_running', ...]
FAILED Tests/Architecture/test_library_media_wiring.py::test_state_object_fields_match_the_shim_surface
FAILED Tests/Architecture/test_library_media_wiring.py::test_every_shim_reads_and_writes_its_own_state_field
2 failed, 3 passed in 6.13s
```

(The same 2-failed/3-passed result was observed in the working tree at the
moment the RED commit was made; the isolated-worktree run above is the
recorded evidence, so the claim does not rest on a same-tree observation.)

The other 3 pass by construction — they check the dataclass and the prefix
mapping, not the screen. After the move commit `76b248c05`:

```
Tests/Architecture/test_library_media_wiring.py .....            [100%]
5 passed in 0.84s
```

This is exactly the prompts series' own 2-failed/3-passed → 5-passed shape.
The literal per-prefix-family mapping assertion
(`test_media_state_shim_attr_maps_each_prefix_family_to_its_literal_name`)
ships from **birth** here rather than being added by a later review round —
the prompts series proved that hole is real (deleting a branch from the shared
resolver leaves a self-referential sweep green).

### 5.2 Default and annotation fidelity, verified mechanically

Two independent `ast` comparisons against the parent tree, not eyeballing:

- **Defaults.** All 75 fields whose original line was deleted outright were
  compared value-for-value. 70 match textually and exactly. The other 5 are the
  `default_factory` folds (`RowSelection("media")`,
  `LibraryMediaReaderSessionState()`, `MediaBrowseScope()`, `set()`,
  `object()`) — semantically identical fresh instances; `object()` is
  deliberately a distinct sentinel per instance in both versions.
- **Annotations.** Of the 82 fields, **70 carried an annotation in
  `__init__`; all 70 are byte-identical** (whitespace-normalized) in the
  dataclass. The other 12 had no original annotation (`self._x = value` form)
  and were given minimal accurate ones, which a dataclass requires — an
  annotation addition, never a default change, and each of those 12 values was
  covered by the default comparison above.

### 5.2b Dead imports — none, checked rather than assumed

The prompts state PR left 2 imports dead for its cleanup task. Here, each of
the 12 type/factory names the moved fields used was counted in
`library_screen.py` after the move: none dropped to 1 occurrence (its own
import line), so **zero dead imports** are handed to task 3 from this commit.
The state module itself imports nothing it does not use (`ast` check: empty
unused set).

### 5.3 Guard battery

| Suite | Result |
|---|---|
| `Tests/Architecture/test_library_media_wiring.py` (new) | **5 passed** |
| The 7 pre-existing library wiring suites + preimport-closure + recompose ratchet | **59 passed** (run together) |
| `Tests/Architecture/test_screen_size_ratchet.py` | 3 passed, **2 failed** — both documented `chat_screen.py` rows; the `library_screen.py` row is GREEN at its new pin |
| `Tests/Architecture/test_library_modules_size_ratchet.py` | **1 failed** — `library_media_browse_controller.py` 410 vs 371, the standing dev-owned red; **left flagged, not absorbed** (this task edits no controller file) |
| Both ratchets at the parent, isolated worktree | **3 failed / 39 passed — identical name-for-name to the branch** |
| `Tests/Performance/test_ui_ready_module_census.py` | **4 passed** (did not flap in this run; the state module is imported by `library_screen` which is already in that window) |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** (CSS bundles, profile-owned paths, diagnostic inventory 584 owners, 3,365 backlog ids, chachanotes allowlist, index plan pins) |
| `Tests/UI/test_library_media_characterization.py` (new) | **3 passed**, 14 consecutive runs |
| `Tests/UI/test_library_screen_reuse.py` (the seeded bypass fixture) | **4 passed** |

### 5.4 Regression evidence — paired xdist sweeps, branch vs isolated baseline

All sweeps run SEQUENTIALLY, never concurrently (wave-2 task 6's own lesson:
two 8-worker invocations sharing one machine measurably amplify the flakiness
backdrop and cost real triage effort).

**Sweep A — the 17 dedicated media suites** (`-n 8 --dist worksteal`, branch):
**533 passed / 5 failed** in 215s. Every one of the 5 resolved:

- 2 × `test_library_media_return_settlement.py` — re-ran the whole file
  single-process on the branch (**2 failed / 42 passed**) and on the isolated
  pristine baseline (**2 failed / 42 passed, same two names**), with the
  byte-identical assertion `assert (0, 45) == (0, 42)`. Pre-existing; new
  entries for §7 (see §5.5).
- 3 already documented in recipe §7 (`test_a_new_document_rescans_for_the_
  same_query`, `test_no_change_traversal_builds_no_preview_and_copies_no_
  content`, `test_image_item_traversal_wall_time_probe`), reproduced in a
  single-process combined re-run on the branch.

**Sweep B — the 19 adjacent files that reference moved names**
(`test_screen_navigation.py`, `test_review_set_walker.py`,
`test_library_per_click_recompose_t21116.py`, the adaptive-reader pair,
`Tests/Library`/`Tests/Media` members, …), identical command on both trees:

| | failed | passed |
|---|---|---|
| branch (`999bd269d`) | 45 | 704 |
| isolated baseline (`83e17323e`) | 44 | 705 |

**43 shared, 2 branch-unique, 1 baseline-unique.** Both branch-unique names —
`test_screen_navigation.py::test_overlapping_navigate_requests_complete_in_
fifo_order` and `::test_research_workspace_runs_round_trip_restores_
independent_context` — **reproduce identically on the isolated pristine
baseline** in the same combined single-process re-run (2 failed on each tree).
**Zero real regressions.**

**Sweep C — `Tests/UI/test_library_shell.py`** (173 moved-name references, the
single heaviest consumer): see §5.6.

### 5.5 Documented pre-existing reds (each verified, not assumed)

Every dismissal below was proven at the parent `83e17323e` in an **isolated
`git worktree` with its own `uv venv`** (verified resolving its OWN tree with
`python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"`), per recipe
§3's now-unconditional rule. No `git stash`, no same-tree checkout overlay, at
any duration.

| Red | Status |
|---|---|
| `test_screen_size_ratchet.py` — the two `chat_screen.py` rows | documented in §7; reproduced identically at the parent |
| `test_library_modules_size_ratchet.py[library_media_browse_controller.py]` (410 vs 371) | documented in §7; reproduced identically at the parent. **Left flagged, not absorbed** — this task edits no controller file, so the plan's absorb-only-if-touched rule keeps two waves of refusal precedent intact |
| `test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_footer_when_retry_availability_changes` | documented in §7 (wave-5 task 3); re-confirmed at the parent (1 failed / 12 passed, both sides) |
| `test_library_media_reader_match_nav_t22209.py::test_a_new_document_rescans_for_the_same_query` | documented in §7 (wave-5 task 2) |
| `test_library_media_reader_no_change_sync_t22208.py::test_no_change_traversal_builds_no_preview_and_copies_no_content` | documented in §7 (wave-2 close, wave-5 task 1) |
| `test_library_media_reader_no_change_sync_t22208.py::test_image_item_traversal_wall_time_probe` | documented in §7 (wave-5 task 1) — a wall-clock timing probe by design |

**THREE NEW ENTRIES for recipe §7's documented list** (found by this task, not
previously recorded anywhere):

- `Tests/UI/test_library_media_return_settlement.py::test_viewer_return_waits_for_geometry_then_scrolls_before_row_focus`
- `Tests/UI/test_library_media_return_settlement.py::test_authoritative_recompose_rearms_before_replacement_geometry`
- `Tests/Architecture/test_persona_visual_runtime_boundary.py::test_persona_visual_modules_stay_outside_other_runtime_and_ui_boundaries`
  (see §10 — proven pre-existing at the parent in the isolated worktree)

Both fail with the **byte-identical** assertion `assert (0, 45) == (0, 42)` —
a scroll-offset geometry comparison — and the whole file measures **2 failed /
42 passed on the branch AND 2 failed / 42 passed on the isolated pristine
baseline**, same two names. These matter more than an ordinary flake entry
because they sit squarely in the settlement cluster this PR moves
(`return_settlement`, `last_exact_settlement`, `successful_focus_ownership`),
which is exactly the shape a reviewer would suspect first; they are proven
pre-existing rather than argued to be. (The recipe §7 edit itself is task 3/4's
scope per the plan, not this task's.)

---

## 6. Forward notes for tasks 2 and 3

Facts established here that the next two tasks should not re-derive:

1. **Cluster size, re-measured:** **251** media-named `FunctionDef`s, 251
   unique names (no property/setter-pair gap). The plan's estimate was ~244.
   Of these: **79 `@on`**, **12 `action_*`**, **0 `on_<message>`
   name-dispatched**. The plan predicted media would own name-dispatched
   handlers; **it owns none** — the only `on_[a-z]` methods on `LibraryScreen`
   are 10 Textual lifecycle hooks plus the 7 prompts-owned
   `on_prompt_block_editor_*`. The §4 whitelist's third member is therefore
   inert for media (it will still matter for notes).
2. **Exclusion candidates already visible:**
   `_toggle_library_media_reader_pane` (the wave-6-verified generic dispatcher
   reading four state objects — stays screen-resident);
   `_stop_library_media_selection_debounce` and
   `_stop_library_media_filter_timer` (called directly by `on_screen_suspend`,
   a screen-lifecycle method); and a large unbound-fake-self population —
   **9 handlers are covered ONLY by `LibraryScreen.<name>(fake, event)`
   calls** in `test_library_multiselect_media.py`, `test_review_set_walker.py`
   and `test_library_media_trash.py`, which is that bypass shape's signature —
   `handle_library_media_{analyze_overwrite, analyze_receipt_dismiss,
   analyze_retry, analyze_skip, bulk_delete_cancel, bulk_delete_undo,
   empty_clear_type, review_dismiss_receipt_close, review_dismiss_undo}`. That
   is a floor, not the full unbound-fake-self population: it counts only
   `@on` handlers whose selector has no press anywhere, so movers reached by a
   fake alongside a real press are not in it.
3. **`getattr(self, "<literal>")` sites to bind on the controller** (recipe
   §3's unbound-attribute-escape shape, the one that cost the skills series a
   silent production regression): 11 in `library_screen.py` today, naming
   `_library_media_analyze_origin` (x2), `_library_media_view` (x3),
   `_library_media_trash_focus_request_key`, `_library_media_analyze_running`,
   `_library_media_bulk_delete_in_flight`, `_library_media_select_mode`,
   `_library_media_browse_controller`, `_library_media_reader_session` — plus
   the `getattr(self, "focused", None)`-shaped names to re-scan once the mover
   set is final.
4. **Dynamic-dispatch string sites for the cleanup PR:** the two
   reader-preference dicts (`_replace_library_reader_preference`,
   `_persist_library_reader_preference`), the choice-strip return tuple and its
   `canvas_kind` dict (`_library_media_type_choices_visible`,
   `_library_media_sort_choices_visible`), and the `getattr` literals above.
   Both dict paths already resolve DOTTED strings, so the retarget is a value
   edit, not a mechanism change.
5. **Test-side retarget scope for the cleanup PR:** the 82 moved flat names
   appear **1,214 times across 36 files** in `Tests/` today, counted with
   IDENTIFIER BOUNDARIES (a plain substring count says 1,426 across 39 files
   and is wrong — several moved field names are substrings of method names,
   e.g. `_library_media_detail` inside `_refresh_library_media_detail`; a
   cleanup PR that retargets on substring match will corrupt method names).
   Either way it is far larger than prompts' 465 across 11 files, the prior
   record. The heaviest are `test_library_multiselect_media.py` (228),
   `test_library_media_return_settlement.py` (204),
   `test_library_media_reader_flow.py` (154), `test_library_shell.py` (153),
   `test_library_media_trash.py` (109) and `test_screen_navigation.py` (48).
   Roots span `Tests/UI`, `Tests/Library`, `Tests/Live`, `Tests/Media` and
   `Tests/Architecture`.
6. **The split decision (task 2's own call) is NOT pre-empted here.** This
   task's ownership analysis found ONE state object is right — every field
   belongs to a single subsystem, and the trash/viewer/browse/analyze/progress
   sub-clusters share `view`, `_selected_media_id`, `bulk_delete_in_flight` and
   `reader_session` freely. That is evidence about STATE, not about the method
   graph; task 2 must run its own connected-components analysis over the 251
   candidates.
7. **`_ANALYZE_ORIGIN_MEDIA` must stay in `library_screen.py`.**
   `Tests/UI/test_library_ingest_analyze_skipped.py:611` asserts it at that
   module path; the state module receives it only as a constructor argument
   passed from `__init__`, never by import.

---

## 7. Deviations, errata and process notes

- **Erratum 1 — the RED commit's characterization counts.** `600192d6e`'s
  message says "73 of 79 are already covered (60 by a real press, 13 only by a
  direct unbound call)" and groups `handle_library_media_reader_mode` with the
  two review handlers as "three [that] touch no moved field". Both are wrong,
  and re-derived at the parent in the isolated worktree the true figures are:
  **76 of the 79 handlers carry a selector** (3 are Message-typed and were
  censused separately); of those 76, **62 covered by a real press, 9 by an
  unbound call only, 5 uncovered**. `handle_library_media_reader_mode` is
  **covered**, not a gap — it belongs with `handle_library_media_highlight_delete`
  and `handle_library_media_trash_row` in the "first pass scored it a gap,
  passes 1–3 overturned it" group, not with the two review handlers. The
  operative conclusion — **3 genuine gaps touching moved state, all pinned** —
  is unchanged, and §2 above carries the corrected accounting. `600192d6e` is
  the parent of `76b248c05`, which `.git-blame-ignore-revs` records by hash, so
  amending it would rewrite that hash and orphan the entry (recipe §6);
  the correction of record lives here.
- **Erratum 2 — the move commit's fold accounting.** The RED commit's
  state-module docstring said the 75 deleted-outright original lines were "71
  static literals plus four pure factory calls". Re-derived by `ast`, the split is
  **70 literals / collection displays and 5 factory calls** — `set()` is also
  folded into `default_factory` and was omitted from the list. Fixed in
  `999bd269d` (doc-only). `76b248c05` is recorded in `.git-blame-ignore-revs`,
  so amending it would orphan the entry (recipe §6); the size-ratchet
  derivation's own totals were already correct and are unchanged.
- **A near-miss worth recording as a lesson.** The first draft of the
  comment-relocation deletion pass searched the WHOLE file for comment text
  matching the state module's, and deleted **8 unrelated lines** in other
  subsystems' comment blocks — including bare `#` separators, which match
  anything. It was caught by reading the diff (not by any test: the result
  parsed fine, and no guard reads a comment), reverted with `git checkout`, and
  rewritten to walk upward only from each to-delete statement, raising rather
  than deleting if it meets a comment that is not in the state module. **A
  content-equality sweep needs a positional bound; a short comment line is not
  a unique key.**
- **Nothing was pushed. `progress.md` was not touched.**
- The isolated baseline worktree (`w7base` @ `83e17323e`, own `uv venv`) is
  removed at task close.

---

## 8. Concerns handed forward

1. **Task 3's test retarget is the largest of the program** — 1,214
   boundary-matched occurrences across 36 files, in five roots. Budget for it accordingly, and
   expect `SimpleNamespace`/fake-harness fixture restructuring at prompts'
   scale or beyond.
2. **The `library_media_browse_controller` standing red is still standing.**
   Task 2 WILL be in that neighbourhood; the plan's rule then requires
   re-measuring and re-pinning it in the same commit that legitimately edits
   the file, with an attribution comment. It was deliberately not touched here.
3. **`test_library_media_return_settlement.py`'s two geometry failures** are
   pre-existing but sit in the cluster this series moves. Task 2 should expect
   to see them again and should not spend a second proving them once more — the
   isolated-baseline pairing is recorded above.
4. **`Tests/Performance/test_ui_ready_module_census.py` did not flap in this
   task's run** (4 passed). It is a known zero-headroom flapper (recipe §7,
   TASK-31816); a future task seeing it red should report it rather than
   treating this task's green as evidence it is fixed.

---

## 9. Sweep C — `Tests/UI/test_library_shell.py`, paired (added after §5.5)

The single heaviest consumer of moved flat names (153 boundary-matched
occurrences). Identical command on both trees, run SEQUENTIALLY:

| | failed | passed | wall |
|---|---|---|---|
| branch (`999bd269d`) | 228 | 597 | 622.5s |
| isolated baseline (`83e17323e`) | 229 | 596 | 623.2s |

**228 shared, ZERO branch-unique, 1 baseline-unique**
(`test_library_media_initial_error_is_unknown_and_retry_is_unique` — already
documented in recipe §7 from wave-5 task 1 as load-adjacent flakiness).

The 228-failure backdrop is the fd-exhaustion cascade recipe §7 already
records for this file ("BOTH runs independently developed a massive,
near-identical burst of `test_library_shell.py::test_library_note_*`
DOM-mount-timeout failures concentrated in the final ~3% of each run"): **170
of the 228 branch failures are note-named and only 6 are media-named**, and
the two trees' failure sets are identical but for one name. Wall times differ
by 0.7s.

**Zero real regressions.** Across all three paired sweeps (17 media suites,
19 adjacent files, `test_library_shell.py`) plus the 7 `__new__` files and the
guard battery, **not one failure is attributable to this diff**, and every
branch-unique name that appeared was reproduced on the isolated pristine
baseline.

---

## 10. Full `Tests/Architecture` run (added after §9)

`.venv/bin/python -m pytest Tests/Architecture -p no:randomly -q` on the
branch: **19 failed / 574 passed / 1 skipped** in 336s. All 19 accounted for,
none attributable to this diff:

- **15** are the exact set recipe §7 records as pre-existing NAME-FOR-NAME at
  the wave-6 base (`test_console_realtime_controller_boundary` 1,
  `test_console_review_selection_controller_boundary` 1,
  `test_console_wave6_closeout_inventory` 1, `test_console_wave6_inventory` 3,
  `test_default_timeout_session_guard` 1,
  `test_persistent_diagnostic_inventory` 2, `test_progress_widget_clock_guard`
  1, `test_timer_path_static_update_inventory` 3,
  `test_worker_exclusive_group_inventory` 2). Three of those key on
  `(file, line)` tuples — the shape a pure move is most likely to break — and
  none moved.
- **3** are the documented ratchet rows (2 × `chat_screen.py`, 1 ×
  `library_media_browse_controller.py`), verified identical at the parent in
  the isolated worktree.
- **1 is new to this list and was paired directly:**
  `test_persona_visual_runtime_boundary.py::test_persona_visual_modules_stay_outside_other_runtime_and_ui_boundaries`
  — **1 failed / 2 passed at the parent `83e17323e` in the isolated worktree**,
  identical. Pre-existing, unrelated to Library (a Persona-visual module
  boundary check). A candidate row for recipe §7 alongside the two
  `test_library_media_return_settlement.py` names.

`library_screen.py`'s own ratchet row is GREEN at its new pin in this run.

## Review erratum (appended by the coordinator after the task review — PASS + Approved, no fix round required)
- I-1 (Important, FORWARD-BINDING on task 2): §6.3's "11 getattr(self, "<literal>") sites" is an under-count from a single-line grep that missed line-wrapped calls. AST truth: 15 calls, 13 naming ATTRIBUTES (the missing two: library_screen.py:16189-16193 getattr on the MOVED `_library_media_trash_focus_authority_generation`; :25765-25767 on the WIRING `_library_media_trash_browse_controller`). Task 2 MUST re-derive this census by AST (ast.Call -> getattr/self/ast.Constant) and bind 13, not 11. (:2893 reads app_instance, correctly excluded.)
- M-1: §5.3's "59 passed" belongs to the 10-suite row (incl. media's 5); the 9 pre-existing suites alone measure 54.
- M-2: §5.4 Sweep C header: 153 boundary / 169 substring (not 173); heaviest consumer is test_library_multiselect_media.py at 228 (as §6.5/§9 say); the §5.6 pointer means §9.
- M-3: §5.2 "70 match textually and exactly" overclaims: 62 textual + 8 empty-collection displays rendered as default_factory + 5 factory folds; semantics identical for all 82 (reviewer's dual-tree runtime dump).
- M-4: .git-blame-ignore-revs prose said 78/75 __init__ assignments; correct 77/74 (fixed in the tracked file by the coordinator, one word each).
- M-5: test_library_screen_reuse.py:226 comment said "the four settlement fields"; the test seeds three, the helper resets five (fixed in the tracked file).
- M-6: substring foil is 1,427 per-name (1,402 non-overlapping), not 1,426; "~20 dedicated media test files" is 16.
