# Wave-6 Task 2 — Prompts controller move (prompts series 2/3)

Branch: `refactor/library-decomp-wave6-prompts`
Worktree: `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`
Parent (task 1 tip): `471813f21`

| Commit | Subject |
|---|---|
| `8ae30f490` | `test(library): full-cluster wiring pins for the prompts controller (RED)` |
| `d0ec95b16` | `refactor(library): prompts controller (prompts series 2/3)` |
| `926b0431c` | `chore(library): blame-ignore the prompts controller pure move` |
| `52268ea42` | `docs(library): correct one stale line reference in the prompts controller` (fix round 1, comment-only) |

Every number below was re-derived against the tree at execution time. Where a
figure changed mid-task (it did, three times), the correction is recorded
rather than the first draft silently replaced.

> ### Session interruption — what is verified evidence and what is not
>
> This task's session was terminated by an API session limit while the full
> `Tests/UI -k "library"` xdist sweep was running. On resumption, ground truth
> was re-established from `git log`/`git status` rather than from memory, and
> **every claim below is either backed by a transcript captured before the
> interruption (files still on disk, timestamps intact) or was re-run
> foreground afterwards.** Sections carrying post-interruption evidence are
> marked **[re-verified post-interruption]**. The one item that did NOT
> complete is stated as such in §9.5 rather than reported as a result.

---

## 1. Method census — 161 candidates

An `ast` scan of every `LibraryScreen` **class-body** `FunctionDef`/
`AsyncFunctionDef` whose name contains `"prompt"` (case-insensitive, a
SUBSTRING match — never a `startswith` prefix filter, per the conversations
exemplar's own enumeration trap, recipe §11):

**161 raw matches, 161 unique names** — no property/setter-pair gap (unlike
Skills' 133 raw / 127 unique). The class as a whole has 1,321 `FunctionDef`s
and 1,308 unique names, so the cluster is ~12% of the screen's method surface.

### 1.1 Two completeness checks, both empty

The brief asks explicitly for "any bare-named members of the cluster found by
call-graph, e.g. handlers reached only from prompt rows". Two independent
scans:

- **Reverse call-graph.** For every NON-prompt-named `LibraryScreen` method,
  compute the set of methods that reference it as `self.<name>`; keep the ones
  whose reference set is non-empty and entirely prompt-named. **Zero.** (This
  is the method-level analogue of the exemplar's `_conversation_records` miss.)
- **Decorator / naming-convention scan.** Any non-prompt-named method carrying
  an `@on(...)` selector mentioning `prompt`/`recipe`: **zero**. Every
  `on_<message>` naming-convention handler on the class was listed and read;
  the only prompt-related ones are the six `on_prompt_block_editor_*`, which
  already contain "prompt" and are therefore already inside the 161.

### 1.2 Shape of the 161

| Shape | Count |
|---|---|
| `@on(...)`-decorated handlers | 54 |
| `on_<message>` naming-convention handlers (no decorator) | 7 |
| `action_*` | 1 |
| `@staticmethod` | 1 |
| `@property` | 1 |
| plain | 97 |
| **total** | **161** |
| **`@work`-decorated** | **0** |

> **Correction of record.** A first draft of this table said 48 `@on` / 103
> plain. 48 is a *different* measurement — task 1's count of `@on` handlers
> that touch at least one MOVED PROMPT FIELD — and reusing it here silently
> mixed two censuses. Re-derived directly from the pre-move tree
> (`git show 8ae30f490:…`): **54 / 7 / 1 / 1 / 1 / 97 = 161**. Recorded rather
> than quietly replaced, per this program's own count-instability lesson.

`@work` deserves an explicit zero: it is the export series' own
"framework-decorator self-type assertion" hazard (Textual's `@work` asserts
`isinstance(self, DOMNode)` at call time), and it cost ingest 4 exclusions and
search+RAG 3. Prompts has none — the cluster's async work is dispatched via
`self.run_worker(<coroutine>)` rather than by decorating the method.

---

## 2. Single vs. split controller — SINGLE, by measurement

The plan explicitly leaves a split on the table at this scale ("split ONLY on a
clean ownership seam; when unsure, one controller"), and names a plausible
seam: editor/studio vs. browse/list wiring.

**The seam does not exist.** Building the undirected `self.<name>` reference
graph over all 161 candidates and taking connected components gives:

```
components: 17   sizes: [145, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

**One component of 145 names, plus 16 isolated singletons** (leaves that
neither call nor are called by a prompt sibling: `handle_library_prompts_sort`,
`handle_library_prompts_export`, `_library_prompt_delete_fingerprint`,
`_mirror_library_prompts_reader_preference`, …). There is no second component
of any size — no subset that only ever calls within itself.

Reading the hub edges confirms why: the editor's own guarded exit
(`_exit_library_prompt_editor_guarded`) drives the browse refetch; the detail
loader (`_refresh_library_prompt_detail`) is reached from the row handler, the
retry handler, the conflict resolvers and the membership-apply refresh; and the
delete/undo flow writes state both the list and the editor read.

**Decision: ONE `LibraryPromptsController`.** This matches the skills (86
methods) and search+RAG (42) precedents' identical resolution at comparable
scale, and the plan's own "when unsure, one controller" default — except this
one is not "unsure", it is measured.

---

## 3. Exclusions — 22 of 161 (139 move)

Counts by class: **2 screen-identity + 14 unbound-fake-self + 3
instance-attribute-monkeypatch + 2 module-globals-coupling + 1
merely-delegate-to-existing-controller property = 22.**

### 3.1 Screen-identity hazard — 2 (recipe §3's sixth shape, Form C)

An `ast` sweep of all 161 bodies for every bare `self` `Name` node that is NOT
the receiver of an `Attribute` found 20 such uses. Classified:

- 11 × `_sync_library_canvas(self, "prompts", …)` — duck-typed forwarding into
  the shared dispatcher, **not** a hazard (see §4.1);
- 7 × `getattr(self, "<literal>", default)` — an unbound-attribute escape,
  handled by binding (see §4.2);
- **2 × `self.app.screen is not self`** — a genuine, silent production
  regression if moved.

| Method | Line | Expression |
|---|---|---|
| `_apply_library_prompts_import_status` | 27601 | `or self.app.screen is not self` |
| `_run_library_prompts_import` | 27798 | `or self.app.screen is not self` |

`real_screen is controller` can never be true, so both guarded branches would
silently take the "not current" path forever. This is Form C exactly — the
shape the skills series' SECOND draft shipped and had to revert after 8
`Tests/Skills/` tests failed. There is no accommodation (identity cannot be
satisfied by a proxy), so both stay screen-resident, full-bodied, untouched.
Found by **static analysis before the move**, not by the battery.

`_run_library_prompts_import` is doubly disqualified — it also reads the bare
module global `validate_path_simple` (§3.4).

Also checked and clean: no mover calls the shared `_library_screen_is_current(
self)` helper (Form B), and no mover carries `self.workers.cancel_group(self,
…)` (Form A). `self.workers` appears once, in
`_library_prompt_write_worker_is_active`, only as an iterable of workers —
no identity filter.

### 3.2 Unbound fake-self — 14

A **content grep across ALL of `Tests/`** (never a `-k`-filtered subset, per
the seventh shape's own filter-blindness rule) for `LibraryScreen.<name>(`
found **19 direct call sites across 3 files, covering 10 names** — exactly
reproducing task 1's §11 lead list:

| File | Sites | Names |
|---|---|---|
| `Tests/UI/test_library_prompts_canvas.py` | 14 | `handle_library_prompts_empty_new`, `_build_library_prompts_state` ×2, `handle_library_prompts_sort` ×2, `handle_library_prompts_filter`, `_settle_library_prompt_delete`, `handle_library_prompt_insert_console` ×6, `on_prompt_block_editor_apply_requested` |
| `Tests/UI/test_library_canvas_scoped_sync.py` | 3 | `handle_library_prompt_row`, `_apply_library_prompts_import_status` ×2 |
| `Tests/UI/test_library_choice_strips.py` | 2 | `handle_library_prompts_sort_choice` ×2 |

**Four more names reach the identical shape through indirections a
`LibraryScreen.<name>(` grep is structurally incapable of seeing.** Each was
found by a *different* census this task ran precisely because the direct grep
is a discovery aid, not a completeness proof:

1. **Fake-harness CLASS ATTRIBUTE** — found by an AST census of NON-CALL
   attribute references to a cluster name across `Tests/`:
   ```python
   class _LibraryPromptHandlerHarness(SimpleNamespace):
       _library_prompts_mutation_in_flight = False
       _stage_library_prompt_for_console = LibraryScreen._stage_library_prompt_for_console
   ```
   (`test_library_prompts_canvas.py:13341-13343`). The harness binds the
   ORIGINAL function onto itself; a delegator would reach for
   `self._prompts_controller` on a `SimpleNamespace`. → **`_stage_library_
   prompt_for_console` excluded.**
2. **`parametrize` tuple of unbound functions** — same census:
   ```python
   @pytest.mark.parametrize(("handler", "target_page"), [
       (LibraryScreen.handle_library_prompts_page_previous, 1),
       (LibraryScreen.handle_library_prompts_page_next, 3),
   ])
   ...
   handler(fake, event)          # :2914
   ```
   (`test_library_prompts_canvas.py:2873-2914`). → **both `handle_library_
   prompts_page_{previous,next}` excluded.**
3. **STRING-NAME `getattr` dispatch** — found by a census of every cluster
   method name appearing as a **string literal** anywhere in `Tests/`, the
   "quoted-string form" sweep this program's standing checklist requires,
   applied here to METHOD names rather than field names:
   ```python
   getattr(LibraryScreen, handler_name)(fake, SimpleNamespace(stop=lambda: None))
   ```
   (`test_library_prompts_canvas.py:2044`, with `handler_name` supplied by the
   parametrize table at `:2016`/`:2022`). → **both `handle_library_prompts_
   empty_{clear_filter,all_prompts}` excluded.**

That string census returned 18 hits total; the other 12 are the three
instance-attribute-monkeypatch names below (`monkeypatch.setattr(screen,
"<name>", …)`) and 5 rows of `test_library_modal_dismissal.py`'s declared
modal-edge table (§7.2). A matching census over `tldw_chatbook/` found **zero**
prompt method names as string literals in production code.

**The names this shape reaches — 15 rows, of which 14 count HERE** (the 15th,
`_apply_library_prompts_import_status`, is tallied under §3.1's
screen-identity class so it is never double-counted; it trips both shapes):
`_apply_library_prompts_import_status` (counted under
§3.1), `_build_library_prompts_state`, `_settle_library_prompt_delete`,
`_stage_library_prompt_for_console`, `handle_library_prompt_insert_console`,
`handle_library_prompt_row`, `handle_library_prompts_empty_new`,
`handle_library_prompts_empty_all_prompts`,
`handle_library_prompts_empty_clear_filter`, `handle_library_prompts_filter`,
`handle_library_prompts_page_next`, `handle_library_prompts_page_previous`,
`handle_library_prompts_sort`, `handle_library_prompts_sort_choice`,
`on_prompt_block_editor_apply_requested`.

Also swept and empty: **`object.__new__(LibraryScreen)`/`LibraryScreen.
__new__` bypass screens making BOUND calls to a prompt method.** All 8 files in
the repo carrying such a construction were AST-walked, resolving each bypass
assignment target and searching the same test-function body for
`<var>.<promptname>(` calls or references: **zero hits** (they are all Ingest/
Parakeet fixtures). This is the shape that cost the ingest series 9 exclusions;
Prompts has none.

### 3.3 Instance-attribute monkeypatch — 3

Two censuses: an AST scan for `<recv>.<promptname> = …` assignments, and a
regex for `monkeypatch.setattr|setattr|patch.object|patcher.setattr(<recv>,
"<promptname>", …)`. 4 + 10 = **14 sites, 3 names**, every receiver a REAL,
`__init__`-constructed `LibraryScreen`:

| Name | Sites |
|---|---|
| `_flush_library_prompt_save` | `test_screen_navigation.py:2021`, `:3239`, `:3251`; `test_library_prompts_canvas.py:4038`, `:6777` |
| `_request_library_prompts_browse` | `test_screen_navigation.py:3261`; `test_library_prompts_canvas.py:4492`, `:5821`, `:10377` (+ `Mock(wraps=screen._request_library_prompts_browse)` recorders at `:5809`, `:10365`); `test_library_shell.py:4274`, `:5146` |
| `_reset_library_prompt_editor_state` | `test_screen_navigation.py:3252` |

All three are called internally by movers — `_exit_library_prompt_editor_
guarded` calls all three in sequence — so each is reached through a named
late-binding dependency that re-reads `screen.<name>` at CALL time, which is
exactly why the patches keep working after the move. **Class-level**
monkeypatching of a cluster name (`monkeypatch.setattr(LibraryScreen, "<name>",
…)`, the fully-qualified string form, and `LibraryScreen.<name> = …`) was
censused separately: **zero in the repo.**

### 3.4 Module-globals coupling — 2 (recipe §3's eighth shape, run to completion)

The mechanical 4-step census, run in full rather than stopped at the first hit:

1. **Free-name extraction.** Walked all 161 candidate bodies' `ast.Name` Load
   nodes, excluding `self`/`cls`, locals/parameters/nested-function names and
   builtins, keeping only names that are module-level imports or definitions in
   `library_screen.py`: **82 names.**
2. **Alias derivation first, then the grep.** Every test file importing the
   screen module object was enumerated and its alias recorded — **38 files, 4
   distinct aliases: `library_screen`, `library_screen_module`, `screen_module`,
   `library_module`** (`library_module` and `screen_module` are exactly the
   spellings wave-5 task 3's own correction found a fixed-string grep missing).
   All 82 names were then searched across ALL of `Tests/` in the three patch
   shapes: direct-attribute (`<alias>.<name>`), fully-qualified string
   (`"tldw_chatbook.UI.Screens.library_screen.<name>"`), and the two-argument
   `setattr`/`patch.object(<alias>, "<name>", …)` form.
3. **Read every hit.** 7 names had hits.
4. **Classify.**

| Free name | Sites / files | Reading verdict |
|---|---|---|
| `validate_path_simple` | 4 / 3 | **ACTIVE** → exclude `_write_library_prompt_export_file` |
| `save_setting_to_cli_config` | 52 / 11 | **ACTIVE** → exclude `_persist_library_prompt_editor_mode` |
| `_sync_library_canvas` | 33 / 10 | **LATENT, kept** |
| `LIBRARY_ROW_BROWSE_PROMPTS` | 2 / 2 | LATENT (plain reads, not patches) |
| `_LIBRARY_PROMPTS_IMPORT_WORKER_GROUP` | 1 / 1 | LATENT (plain read) |
| `resolve_adaptive_reader_layout` | 1 / 1 | LATENT (plain read, and Notes-scoped) |
| `asyncio` | 1 / 1 | LATENT (patches `library_screen_module.asyncio.to_thread` — an attribute of the SHARED module object, identical in every importer, so a move cannot bypass it) |

**`validate_path_simple` — ACTIVE, and this one fails LOUDLY, not vacuously.**
`test_library_prompts_canvas.py::test_library_prompt_write_export_file_rejects_
invalid_path` (`:10976`) patches `library_screen_module.validate_path_simple`
with a stub that unconditionally raises, then calls
`screen._write_library_prompt_export_file(destination, …)` with a perfectly
VALID `tmp_path` destination and asserts `not destination.exists()` plus a
"Rejected export path" warning. Moved, the controller's own freshly-imported
`validate_path_simple` would win, the real validator would ACCEPT the tmp path,
the file would be written, and the assertion would fail. (Contrast the ingest
series' own `_resolve_ingest_source`, where a nonexistent path failed the real
validator identically to the stub and the test stayed green-but-vacuous. This
one is the same coupling with a louder failure mode — which is a property of
the test's inputs, not of the coupling.) → excluded; its one mover caller
(`_export_library_prompt`) reaches it through a named dependency.

**`save_setting_to_cli_config` — ACTIVE.** Two tests:
`test_library_prompt_mode_persistence_failure_keeps_live_mode_and_warns`
(`:689`) patches it to return `False` and awaits
`screen._persist_library_prompt_editor_mode("advanced")` directly on a real
screen, asserting the warning notice; a second (`:5585`) patches it with a
thread-recording stub and clicks the real `#library-prompt-mode-advanced`
button. The body passes the bare name into `asyncio.to_thread(
save_setting_to_cli_config, …)`. → excluded; its one mover caller
(`handle_library_prompt_editor_mode`) reaches it through a named dependency.

**`_sync_library_canvas` — LATENT, kept (recorded, per the eighth shape's own
"record which test files were checked and why none applies" rule).** 33 sites
across 10 files. Every one was located by enclosing test function and checked
against the 139-mover set. Only ONE test function mentions any mover name at
all — `test_library_canvas_scoped_sync.py::test_prompt_and_skill_row_handlers_
route_to_their_canvas` — and the three mover names it contains
(`_clear_library_prompt_selection`, `_invalidate_library_prompts_browse`,
`_refresh_library_prompt_detail`) appear only as `Mock()` kwargs on the
`SimpleNamespace` fake; the method it actually invokes is
`handle_library_prompt_row`, an EXCLUDED name whose body never left
`library_screen.py` and therefore still resolves `_sync_library_canvas` through
that module's own globals. The other two Prompts-touching sites in that file
(`:342`, `:418`) exercise `handle_library_ingest_option_value_changed` and
`_apply_library_prompts_import_status` — also excluded. The remaining 7 files
(`Tests/Skills/test_skills_import.py`, `test_library_entry_compose_once.py`,
`test_library_file_notes_workspace.py`, `test_library_media_trash.py`,
`test_library_note_import_flow.py`, `test_library_notes_folder_navigator.py`,
`test_library_notes_reader.py`, `test_library_review_round_t21116.py`,
`test_review_set_walker.py`) patch it for notes/media/skills canvas syncs.
**Zero reach any of the 11 movers that read this name.** Same systemic
bare-function shape every sibling controller already carries; same verdict the
ingest series recorded.

### 3.5 Merely-delegate-to-existing-controller property — 1

`_library_prompt_history_state` is a `@property` whose ENTIRE body is
`return self._library_prompt_history_controller.state`. This is the skills
series' own named exclusion class (six there). It stays screen-resident; tests
read it 64 times directly off the screen (`test_library_prompts_canvas.py`).
The controller reaches it read-only through an injected accessor.

The other four one-line delegates to the history controller
(`_invalidate_library_prompt_history`, `_initialize_library_prompt_history`,
`_request_library_prompt_history_count`, `_request_library_prompt_history_page`)
are ordinary METHODS, not properties, and move cleanly — they reach the wiring
controller through the group-(c) accessor.

---

## 4. The 139 movers and their binding surface

| Shape | Count |
|---|---|
| `@on(...)` handlers | 44 |
| `on_<message>` naming-convention handlers | 6 |
| `action_*` | 1 |
| `@staticmethod` | 1 |
| plain | 87 |
| **total** | **139** |

The binding surface was derived **mechanically**, not by reading: an `ast` walk
of all 139 moved bodies collecting every `self.<attr>` load/store plus every
`getattr(self, "<literal>")`, minus this controller's own state fields and the
movers themselves. **42 names**, and — a notable result — **zero stores to any
non-own-state name**, so every group-(b)/(c)/(d) accessor is getter-only.

| Group | Count | Names |
|---|---|---|
| Framework services (live-read `@property`) | 12 | `app`, `app_instance`, `call_after_refresh`, `focused`, `is_mounted`, `is_running`, `query`, `query_one`, `refresh`, `run_worker`, `set_timer`, `workers` |
| (a) General shell helpers | 12 | `_arm_library_list_entry_focus`, `_focus_library_control`, `_library_entry_reconcile_is_current`, `_library_entry_route_key`, `_library_list_canvas_showing_list`, `_library_note_keywords_from_input`, `_open_library_export_canvas`, `_refresh_local_source_snapshot`, `_run_library_service_call`, `_safe_text`, `_sanitize_media_field`, `_sanitize_note_content` |
| (b) Shared shell state (read-only) | 4 | `_library_pending_list_entry_focus`, `_library_selected_row_id`, `_library_snapshot_state_generation`, `_local_source_counts` |
| (c) Prompt wiring controllers | 3 | `_library_prompt_browse_controller` (23 movers — the cluster's most-referenced single name), `_library_prompt_collections_controller` (10), `_library_prompt_history_controller` (14) |
| (d) Merely-delegate property | 1 | `_library_prompt_history_state` |
| (e) Late-binding callables for exclusions | 10 | the 10 excluded methods a mover still calls |

Constructor arity **measured with `inspect.signature`**, never hand-counted:
**33 parameters including `self`** — 1 positional (`screen`) + 31 keyword-only
(1 state accessor + 12 + 4 + 3 + 1 + 10). 85 class-level `property` objects: 42
hand-written bindings + 43 generated flat-name state shims.

`_local_source_counts` is getter-only despite two movers writing into it
(`self._local_source_counts["prompts"] = …` in `_delete_library_prompts` and
`_undo_library_prompt_delete`): a `dict` mutates in place through the getter —
the ingest controller's `_library_ingest_analyze_outcomes` precedent.

### 4.1 `_sync_library_canvas(self, "prompts")` — what the controller must satisfy

11 movers forward bare `self` into the shared dispatcher. Reading
`canvas_sync.py`'s `_sync_library_canvas` rather than assuming, the prompts
path touches: `query_one`, `query`, `refresh`, `call_after_refresh`, `app`,
`is_running`, `getattr(screen, "_library_canvas_projection_depth", 0)`,
`screen._library_canvas_resync_pending`, and three PROMPT-named screen methods
(`_library_prompts_list_canvas_kwargs`, `_library_prompt_work_pane_kwargs`,
`_sync_library_prompts_reader_layout_from_shell`). The three prompt methods are
all movers, so they resolve on the controller; the framework services are
bound. `_library_canvas_projection_depth` / `_library_canvas_resync_pending`
resolve on the SCREEN in every real call path (the dispatcher's own
`getattr(..., 0)` default and the assignment both run against whatever object
was passed) — the same standing every sibling controller has. Crucially,
`canvas_sync.py` contains **no** `_library_screen_is_current` call and **no**
`is screen` identity comparison, so Form B does not apply here.

### 4.2 `getattr(self, "<literal>")` — the skills series' silent regression, closed by construction

Three literal names appear across the moved bodies. All three were re-scanned
against the finished controller class:

| Literal | Movers using it | Resolves on `LibraryPromptsController`? |
|---|---|---|
| `focused` | `_sync_library_prompt_selection`, `_library_prompts_focus_identity`, `_capture_library_prompts_filter_cursor`, `_sync_library_prompts_browse_result` | ✅ `property` |
| `_library_prompts_view` | `_library_prompt_editor_active` | ✅ `property` (generated state shim) |
| `_library_prompt_block_state` | `_current_library_prompt_editor_state`, `_save_library_prompt` | ✅ `property` (generated state shim) |

`focused` is the exact name the skills series shipped UNBOUND, degrading a real
focus-restore path permanently and silently (recipe §3's unbound-attribute
escape). Here it is bound from the first commit **and pinned by a wiring test**
(`test_prompts_controller_binds_every_name_its_moved_bodies_use`) rather than
left to a reviewer to notice — the first wiring suite in this program to assert
the constructor-binding surface at all.

---

## 5. RED wiring commit — proven red at the parent

`8ae30f490` changes **one file** (`git show --stat`: `Tests/Architecture/
test_library_prompts_wiring.py`, 428 insertions / 2 deletions). `library_
screen.py` is untouched, and `git diff --stat 8ae30f490 d0ec95b16 --
Tests/Architecture/test_library_prompts_wiring.py` is **empty** — the move
commit never edited the pins it was written against.

Run at `8ae30f490`:

```
5 failed, 7 passed, 94 warnings in 1.70s

FAILED test_prompts_controller_owns_its_cluster
FAILED test_screen_delegates_prompt_handlers
FAILED test_prompts_cluster_staticmethods_forward_to_the_controller_class
FAILED test_prompts_controller_exposes_every_state_field
FAILED test_prompts_controller_binds_every_name_its_moved_bodies_use
```

with, verbatim:

```
E  ModuleNotFoundError: No module named
   'tldw_chatbook.UI.Library_Modules.library_prompts_controller'
E  AssertionError: not delegators yet: [<all 139 names>]
E  AssertionError: expected class-forwarding delegators:
   ['_restore_library_prompts_scope']
```

The 7 that pass are task 1's five state-PR pins plus the two census-drift
guards, none of which depend on the controller existing.

After `d0ec95b16`: **12 passed.**

### What the new pins add beyond the prior series' shape

- `_PROMPTS_CLUSTER_METHOD_NAMES` (139), with a name-shape guard and an exact
  count, plus a duplicate-entry guard.
- `_PROMPTS_CLUSTER_STATICMETHOD_NAMES` (1) and the class-forwarding check.
- `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED` — **deliberately empty**, with the
  skip/absence-assertion pair already wired, so task 3's prune is a one-line
  frozenset edit rather than a test restructuring.
- `_PROMPTS_CONTROLLER_BOUND_NAMES` (42) — new to this program. It asserts at
  the CLASS level (`isinstance(getattr(C, name, None), property)`) rather than
  on a constructed instance, deliberately: `workers` raises off the app tree
  (its one caller wraps it in `try`/`except`), so an instance probe would
  report a false failure for a correct binding.

---

## 6. Move commit — byte-for-byte evidence

The move was performed by **script, not by hand-editing** (the collections
series' own lesson for clusters past ~40-50 methods, here at 139): each
mover's exact source segment was extracted using the ORIGINAL file's line
offsets (first decorator line through `end_lineno`), the controller module was
assembled from that extracted text plus a hand-written header/footer, and the
screen was rewritten by splicing generated delegators into the same offsets.

### 6.1 Verification transcript **[re-verified post-interruption]**

A SECOND, independent script re-parsed the pre-move screen (read from
`git show 8ae30f490:…`, not from a saved copy) and the finished controller, and
compared each of the 139 methods two ways. Re-run at the final tree
(`52268ea42`) after the session interruption:

```
=== BYTE-FOR-BYTE TRANSCRIPT (re-run post-interruption, HEAD 52268ea42) ===
movers compared:                     139
TEXT (source-segment) mismatches:    0 []
AST (normalized-dump) mismatches:    0 []
screen names NOT a 1-stmt delegator: 1 ['_restore_library_prompts_scope']
excluded methods compared:           22; changed: []
screen measure: 37722 lines / 1321 methods
controller measure: 4956 lines
```

The single non-one-statement delegator is expected and correct:
`_restore_library_prompts_scope` is the cluster's `@staticmethod`, and its
class-forwarding delegator needs its own function-local import of the
controller class (§6.2) — two statements, not one. An earlier run of this same
script, taken BEFORE that fix, reported zero such rows; that earlier transcript
is superseded by the one above rather than kept, because it described a tree
that shipped a `NameError`.

- **TEXT** = the raw source segment (decorators through `end_lineno`), compared
  character for character.
- **AST** = `ast.dump(ast.parse(ast.unparse(node)))`, which normalises
  formatting and would catch a semantic difference a text compare could not
  reach (it cannot differ here, but it is the independent second method the
  program's own count-instability lessons call for).
- All 139 screen names are now one-statement delegators; all 22 exclusions are
  byte-identical to their pre-move text.

**One body carried trailing comment lines outside its own AST range.**
`_save_library_prompt` ends at line 29518, but lines 29519–29524 are six
`#`-comment lines at body indent explaining why the broader local-source
snapshot is deliberately not refreshed there. `end_lineno` does not include
them. A whole-cluster scan found this is the only such case; those six lines
were moved WITH the body rather than left orphaned inside a two-line delegator.

Undefined-name and dead-import checks on the finished controller (no `ruff`/
`pyflakes` in this venv, so a `symtable`-based equivalent):

```
UNDEFINED global names referenced: 0
possibly unused imports: []
```

### 6.2 A real bug the battery caught, and its fix

The generated `@staticmethod` delegator for `_restore_library_prompts_scope`
first read:

```python
    @staticmethod
    def _restore_library_prompts_scope(state: Mapping[str, Any]) -> PromptBrowseScope:
        return LibraryPromptsController._restore_library_prompts_scope(state)
```

`LibraryPromptsController` is **not** a module-level name in
`library_screen.py` — the whole point of the born-lazy import. 8 tests failed
with `NameError: name 'LibraryPromptsController' is not defined`. Fixed by
adding the function-local import the two sibling static delegators immediately
below it in the same file (`_restore_library_skills_scope`,
`_restore_library_collections_page`) already use. **Recorded because it is a
generalizable trap: the static-method delegator pattern (recipe §11) and the
born-lazy-import constraint interact — a class-forwarding delegator needs its
own local import, and a code generator that emits the forwarding line without
it produces a file that imports fine and fails only at call time.**

This is also why the screen line count changed twice: 37718 (first draft) →
37722 (with the 4-line local import). The `_BUDGETS` row and its arithmetic
comment were corrected before the commit landed; the first figure never
shipped.

### 6.3 Screen line-delta reconciliation (every term measured)

```
removed=4061   added_delegators=333   lazy_import=3   construction=88
-4061 + 333 + 3 + 88 = -3637     41359 - 3637 = 37722  ✅
```

Method count **1321 → 1321**, as every pure controller move's must be.

---

## 7. Two test files this move touched, and why

The recipe allows a move PR to edit tests in exactly one circumstance: a census
that goes **loudly RED at the move boundary** rather than silently green
(recipe §3's fifth shape). One file qualified; a second was examined and
deliberately left alone.

### 7.1 `Tests/UI/test_prompt_review_docstrings.py` — retargeted, same commit

`test_prompt_library_button_handlers_document_event_argument` asserts
`inspect.getdoc(LibraryScreen.handle_library_prompt_{copy,delete})` contains
`"Args:"` and `"event:"`. A delegator carries no docstring, so both
parametrizations go red at exactly `d0ec95b16` (confirmed: 2 failed / 10 passed
before the fix). The two parametrize rows now name
`LibraryPromptsController`, where the byte-for-byte-moved docstrings actually
live. **Both assertions are unchanged, character for character**; only the
owner moved, and the module docstring records the incident. 4 passed after.

### 7.2 `Tests/UI/test_library_modal_dismissal.py` — pre-existing RED, deliberately NOT touched

`LIBRARY_MODAL_LAUNCH_EDGES` declares 33 `(file, class, presenter, modal-type)`
rows and `test_library_modal_inventory_matches_declared_edges_bidirectionally`
AST-parses `library_screen.py` to rediscover them. Four rows name prompt movers
(`handle_library_prompts_import_browse`, `handle_library_prompt_history_
restore`, `_export_library_prompt`, `_open_library_prompt_delete_confirmation`;
a fifth, `_stage_library_prompt_for_console`, is an exclusion and stays).

**That test already fails at HEAD, before this move, for an unrelated reason**,
and fails *before* the edge comparison is ever reached:

```
E  AssertionError: unresolved modal constructor in supported presenter:
   tldw_chatbook/UI/Screens/library_screen.py:LibraryScreen.
   _present_library_skills_import_choice_if_needed
   (SkillImportChoiceModal(snapshot.candidates))
E  assert None is not None
```

Verified reproducing identically at the parent commit. Its declared table is
also already stale from prior waves (`handle_library_ingest_browse`,
`_request_library_skill_trust_passphrase`, `_request_library_skill_trust_
bootstrap_passphrase` are delegators today, yet still declared as
`LibraryScreen` edges).

**Decision: leave it, document it, hand it forward.** Retargeting only *my*
four rows could not make the test green (it dies at discovery), would mix an
unrelated pre-existing repair into a pure-move commit, and could not be
verified by running anything. The honest record is: this move adds 4 more stale
declared edges to an inventory that is already stale by 3 and already red for a
third reason. **Forward action for task 3 / the wave close:** add an
`_OwnerScope` row for each landed controller module, repoint every migrated
edge, and fix the `SkillImportChoiceModal(snapshot.candidates)` resolution —
one coherent repair, attributable, in a commit that can actually prove itself.

---

### 7.3 Fix round 1 — `52268ea42`, comment-only

The controller's module docstring cited `library_screen.py:9434` as
`_PROMPTS_WORKBENCH_FOCUS_TARGETS`'s sole reader — a PRE-move offset, taken
from the census that produced the exclusion list. The move itself shifted that
line to **9472** (re-measured with `grep`, not inferred). One character
changed; the file line count is unchanged at 4956, so no `_BUDGETS` re-pin is
owed, and the claim it supports (the reader is a shell method, not a mover, so
the class constant stays on the screen) is unaffected. Not added to
`.git-blame-ignore-revs` — a comment fix is not a pure move.

Every other file:line reference in that docstring points into TEST files, which
this move does not renumber; all were re-checked.

---

## 8. Size ratchets — re-pinned in the move commit

| File | Before | After |
|---|---|---|
| `tldw_chatbook/UI/Screens/library_screen.py` | 41359 / 1321 | **37722 / 1321** |
| `tldw_chatbook/UI/Library_Modules/library_prompts_controller.py` | — (did not exist) | **4956** (born-governed) |

Pin trajectory this wave: `41393/1321 → 41359/1321` (task 1) `→ 37722/1321`
(task 2). The controller is born governed by
`test_library_modules_size_ratchet.py`'s glob — nothing had to remember to add
the row; the guard named the file and the row was set to its exact measurement,
in the same commit, with a dated comment carrying the measured constructor
arity and property counts.

---

## 9. Battery

All commands from the branch worktree, `.venv/bin/python`, `-p no:randomly`,
`timeout` unavailable (`perl -e 'alarm N; exec @ARGV'`).

**[all rows re-verified post-interruption, foreground]**

| Suite (one combined invocation per row) | Result |
|---|---|
| extended prompts wiring + the 6 other `test_library_*_wiring.py` + `test_library_support_layer_surface.py` + BOTH size ratchets + `test_library_recompose_ratchet.py` | **95 passed, 3 failed** — the pre-authorized `chat_screen.py` ×2 and `library_media_browse_controller.py` rows, and nothing else. The `library_screen.py` ratchet row is GREEN at the new pin. |
| `test_library_preimport_closure.py` + `test_ui_ready_module_census.py` + `test_library_prompts_characterization.py` + `test_library_screen_reuse.py` + `test_prompt_review_docstrings.py` | **16 passed, 1 failed** — the ui-ready census only (§9.2); preimport-closure, all 4 task-1 characterization pins, the `__new__`-bypass fixture and the retargeted docstring contract all green. |
| Full `Tests/Architecture/` | **565 passed, 1 skipped, 18 failed — all 18 name-for-name identical to the set already proven at the parent** (§9.1) |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** |

For reference, the pre-interruption numbers for the same surfaces
(`12 passed` on the prompts wiring alone against `5 failed / 7 passed` at the
RED parent; `53 passed` on the wiring+closure set; `588 passed / 1 skipped /
18 failed` on the wider Architecture+extras set) are consistent with the
re-runs above — the row counts differ only because the files were grouped
differently.

### 9.1 The 18 Architecture failures, proven at the parent **[re-verified post-interruption]**

Rather than infer from names, the identical 18 node-ids were re-run **at the
parent commit** (`git checkout 8ae30f490 -- tldw_chatbook Tests Docs`, no other
job running, tree restored and `git status` verified clean afterwards):
**18 failed**, and a `diff` of the two sorted name lists reports **IDENTICAL**.

They are the same 18 task 1 documented: `test_console_realtime_controller_
boundary` (1), `test_console_review_selection_controller_boundary` (1),
`test_console_wave6_closeout_inventory` (1), `test_console_wave6_inventory` (3),
`test_default_timeout_session_guard` (1), `test_persistent_diagnostic_
inventory` (2), `test_progress_widget_clock_guard` (1), `test_timer_path_
static_update_inventory` (3), `test_worker_exclusive_group_inventory` (2), the
two `chat_screen.py` ratchet rows, and `library_media_browse_controller.py`.

Worth naming explicitly, because they were the plausible-regression
candidates: the three `test_timer_path_static_update_inventory` failures key
`CLASSIFIED_SITES` on `(file, line)` tuples — exactly the shape a 3,637-line
shift could break — and they fail identically with the screen untouched.

The post-interruption full `Tests/Architecture/` re-run (`565 passed, 1
skipped, 18 failed` in 1134.82s under heavy ambient load) produced a failure
set that `diff`s **IDENTICAL** to that same proven-at-parent 18-name list, and
contains **zero** `library_screen.py`-scoped rows.

### 9.2 The `_ui_ready` census — zero headroom, breached identically on both trees **[re-verified post-interruption]**

This guard failed intermittently throughout the task. It is a **pre-existing
zero-headroom pin on `dev`**, proven two independent ways:

1. **Equal measurement, quiet machine** (pre-interruption, run alone with `-s`
   on each tree):
   ```
   branch:   ui-ready-census: 972/972 modules (headroom 0); snapshot drift +23/-19
   parent:   ui-ready-census: 972/972 modules (headroom 0); snapshot drift +23/-19
   ```
2. **Equal BREACH, loaded machine** (post-interruption, run alone on the branch
   worktree and then on the ISOLATED baseline worktree at `8ae30f490`, which
   has its own venv):
   ```
   branch (52268ea42):  E  AssertionError: 974 tldw_chatbook modules resident at
                           _ui_ready (ratchet limit 972).
   parent (8ae30f490):  E  AssertionError: 974 tldw_chatbook modules resident at
                           _ui_ready (ratchet limit 972).
   ```

**The same number, the same message, on a tree that does not contain this
move.** The move adds exactly zero modules to the first-paint window — which is
the born-lazy controller import working as specified. The pin is 972 and the
warm-boot measurement is 972–974 depending on machine load, against a guard
whose own docstring documents ±1 run-to-run wobble; that is `dev`'s budget to
tighten or raise, not this move's. Recorded as a lead for the wave close.

### 9.3 Prompts regression battery — paired baseline, zero branch-unique
*(transcripts captured pre-interruption; files intact and re-checked on resume)*

Five prompt-heavy files (`test_library_prompts_canvas.py`,
`test_library_prompts_reader.py`, `test_library_prompt_collections.py`,
`test_library_prompt_browse_controller.py`, `test_library_choice_strips.py`),
`-n 8 --dist worksteal`, branch then a `git stash -u` of the identical tree at
the RED commit, run **sequentially**:

| | failed | passed | wall |
|---|---|---|---|
| branch (`d0ec95b16` content) | 22 | 436 | 142.11s |
| parent (`8ae30f490`) | 24 | 434 | 143.87s |

**22 shared, 0 branch-unique, 2 baseline-unique**
(`test_library_prompts_settlement_keeps_newer_surviving_focus`,
`test_library_prompts_stale_search_cannot_restore_an_old_filter_caret` — the
latter already in recipe §7 as run-to-run noise).

An earlier branch-only run of the same five files posted 36 failed; the delta
was the `_restore_library_prompts_scope` `NameError` (§6.2), fixed before the
commit. One of the survivors,
`test_prompt_selection_clear_boundaries_and_invalid_row_fail_closed`, was
individually re-run at the parent and fails there identically ("Unsaved Prompt
changes" veto notice) — pre-existing.

### 9.4 Nine adjacent Library files — paired baseline
*(transcripts captured pre-interruption; files intact and re-checked on resume)*

`test_library_canvas_scoped_sync.py`, `test_screen_navigation.py`,
`test_library_modal_dismissal.py`, `test_library_screen.py`,
`test_library_adaptive_reader_closeout.py`,
`test_library_resize_focus_gates_t23025.py`,
`test_library_entry_compose_once.py`,
`test_library_per_click_recompose_t21116.py`,
`test_library_honesty_accessibility.py`, `-n 8 --dist worksteal`, sequential:

| | failed | passed | wall |
|---|---|---|---|
| branch | 45 | 465 | 111.35s |
| parent | 45 | 465 | 108.86s |

**42 shared, 3 branch-unique, 3 baseline-unique.** All 3 branch-unique
resolved:

- `test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_
  cycle` and `test_screen_navigation.py::test_overlapping_navigate_requests_
  complete_in_fifo_order` — passed cleanly on a combined single-process re-run.
  (The closeout test is the one recipe §7 warns is NOT unrelated to any
  subsystem — its `DESTINATION_CONTRACT` cycles every destination including
  prompts — so it was re-run rather than dismissed by name.)
- `test_screen_navigation.py::test_boot_with_search_default_tab_lands_on_
  library_rag_canvas` — reproduced **3 of 3 in TRUE isolation on the branch AND
  3 of 3 in true isolation at the parent**. Pre-existing; it merely happened to
  win its ordering lottery in the baseline xdist run.

### 9.5 Full sequential xdist paired-baseline sweep — **NOT COMPLETED**

This is the one piece of the brief's battery this task did not deliver, and it
is reported as not-done rather than as a result.

**Setup that WAS completed** (and is reusable): an isolated baseline worktree
at `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/
w6t2-baseline`, checked out at `8ae30f490` (the RED commit — screen untouched)
with its own `uv venv` and `uv pip install -e ".[dev]"` — never a same-tree
checkout overlay, per wave-5 task 1's own interruption-safety lesson. It was
used successfully for the §9.2 `_ui_ready` parent measurement.

**What happened.** The branch half of
`pytest Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal` was
launched under a 3000 s (50 min) alarm. Prior series record this sweep at
~24 min on a quiet machine; this machine was not quiet. `ps aux` at launch and
`uptime` during the run recorded **load averages of 20.8–52.1 with 8–10
unrelated `pytest` processes from other repo checkouts and other sessions**
(`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/`, `tldw_Server_API`, a
second worktree's scratchpad venv — one of them a full `pytest Tests -n` run).
The alarm fired before pytest reached its summary, so the captured file is
**empty** — not "zero failures", *no result at all*. The session was then
terminated by an API session limit. On resumption the run was not restarted:
each side needs ~50+ min against a 10-min foreground command ceiling, and the
resumption instruction is explicit that background waits are not available.

**What stands in its place**, all with transcripts and all paired against the
same parent commit:

- §9.3 — the five prompt-heavy files, xdist, sequential paired baseline:
  **0 branch-unique** (22 shared, 2 baseline-unique).
- §9.4 — nine adjacent Library files (canvas-sync, screen-navigation,
  modal-dismissal, library-screen, adaptive-reader-closeout, resize-focus
  gates, entry-compose-once, per-click-recompose, honesty/accessibility),
  xdist, sequential paired baseline: **3 branch-unique, all resolved** (2 pass
  on a single-process re-run; 1 reproduces 3/3 in true isolation on BOTH
  trees).
- §9.1 — the full `Tests/Architecture/` failure set, `diff`-identical to the
  18 names re-run at the parent commit.
- §9.2 — the `_ui_ready` census breaching at the same number on both trees.

- §9.6 — `Tests/UI/test_library_shell.py`, the largest Library test file and
  the one the full sweep would otherwise have been the only cover for, run in
  two paired slices (added post-interruption specifically to close this gap).

Between them these cover every file the cluster's own tests live in, the
adjacent shell/navigation surfaces a 3,637-line relocation could plausibly
disturb, and 571 of `test_library_shell.py`'s 825 tests.

**Residual gap, stated plainly:** the `-k "note"` half of
`test_library_shell.py` (254 tests) exceeded the 10-minute foreground budget
twice and was not run paired. That is the Notes cluster whose DOM-mount
timeouts recipe §7 already documents as the dominant sweep backdrop (with an
`fd_leak_sentinel` "open file descriptors grew by 274" diagnosis attached); it
touches no prompt code and no line this move changed.

**Carry-forward for task 3 / the wave close:** run the full sequential paired
sweep against this wave's span on a quiet machine, using the `w6t2-baseline`
worktree pattern (re-created at the wave-6 start commit `e5e03846a`), and cover
the `test_library_shell.py -k "note"` half explicitly if the full sweep is
again infeasible.

### 9.6 `Tests/UI/test_library_shell.py` — two paired slices **[post-interruption]**

Both slices run branch-first, then the ISOLATED `w6t2-baseline` worktree at
`8ae30f490` with its own venv, sequentially, `-n 8 --dist worksteal`.

| Slice | Branch | Baseline (`8ae30f490`) |
|---|---|---|
| `-k "prompt"` (11 tests) | 1 failed, 10 passed (33.93s) | 1 failed, 10 passed (37.62s) |
| `-k "not note"` (571 tests) | 60 failed, 511 passed (264.43s) | 60 failed, 511 passed (392.02s) |

**`-k "prompt"`: the SAME single failure name on both trees** —
`test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-prompts-
#library-prompts-reader-shell]` (the browse-prompts parametrization of a family
whose `browse-conversations` sibling recipe §7 already documents). **Zero
branch-unique.**

**`-k "not note"`: 58 shared, 2 branch-unique, 2 baseline-unique.** Both
branch-unique names — `test_library_conversation_retry_first_failure_has_no_
applied_metadata` and `test_library_shell_reused_screen_reentry_serializes_
draining_retrieval` — **passed cleanly (2 passed) on a combined single-process
re-run**, the recipe's own disposition for a name that does not reproduce.
Neither is prompt-related (a Conversations retry-metadata test and a
screen-reuse retrieval-serialization test). **Zero real regressions.**

---

## 10. Notes for task 3 (cleanup PR)

- **Delegator prune census** is not run yet; `_PROMPTS_CLUSTER_SCREEN_
  DELEGATOR_PRUNED` is an empty frozenset with its skip/absence assertions
  already wired. Heed the skills series' lesson 2: sanity-check the census
  regex against a name KNOWN to have a `self.<name>(` caller before trusting
  any "zero references" verdict.
- **The 14 quoted-string field sites** task 1 recorded (§6 of task-1-report),
  two of which (`getattr(self, "_library_prompts_view", …)`,
  `getattr(self, "_library_prompt_block_state", …)`) need the RECEIVER fix, not
  just a string swap.
- **Dead imports** left by THIS move: re-derive against `library_screen.py`
  after the shim deletion, and check every candidate against PR-0a's `_SURFACE`
  re-export contract individually (the shape that has bitten two series).
- **`test_library_modal_dismissal.py`** — see §7.2. The coherent repair
  (per-controller `_OwnerScope` rows + edge repointing + the
  `SkillImportChoiceModal` resolution fix) is a good task-3 or wave-close item.
- **The `_ui_ready` census has ZERO headroom and already breaches under load**
  (pin 972, measured 972 quiet / 974 loaded, identically on both trees). Any
  wave adding a single mount-leg module will trip it, and it will keep failing
  intermittently until `dev` either sheds mount-leg cost or re-pins. Not this
  wave's to fix; named here so the next task does not re-derive it as a
  regression.
- **The isolated baseline worktree** at `.worktrees/w6t2-baseline`
  (`8ae30f490`, own venv) was removed at the end of this task. Re-create it at
  `e5e03846a` for the wave-close sweep rather than reusing a same-tree overlay.
- **The 22 exclusions' own field references** will need retargeting when the
  screen shim block is deleted; several are exercised through
  `SimpleNamespace` fakes carrying FLAT prompt kwargs, which will need the
  "flat kwargs → nested `_prompts_state=SimpleNamespace(...)`" restructuring.
  Count the call sites before deciding by-hand vs. scripted (skills' ~28
  threshold).
