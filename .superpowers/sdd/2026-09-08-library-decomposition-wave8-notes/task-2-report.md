# Wave-8 Task 2 — Notes controller move (notes series 2/N)

Branch `refactor/library-decomp-wave8-notes`, worktree
`/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`,
task parent `915e76b71` (task 1's close). **The FINAL controller move of the
eight-wave Library decomposition program.**

| Commit | What |
|---|---|
| `afaf2320c` | RED/pins: full-cluster controller pins in `Tests/Architecture/test_library_notes_wiring.py` (RED at parent: 6 failed / 9 passed, screen untouched) |
| `9e13f0207` | Move: 186 methods → `LibraryNotesController`, 186 delegators, born-lazy import, preimport suffix, both ratchets re-pinned, diagnostic inventory regenerated |
| `a1bdd25a0` | `.git-blame-ignore-revs` row for `9e13f0207` (hash via `git rev-parse`) |
| `a529dbd3e` | Fix round 1: one mover reverted to the screen (battery-found `partial(LibraryScreen.<x>, self, …)` hazard); two path censuses retargeted; both ratchets re-pinned |
| `56999dbf6` | `.git-blame-ignore-revs` row for `a529dbd3e` (hash via `git rev-parse`) |
| `6f50060f7` | Docstring correction: the generalized `LibraryScreen.<x>` census figure re-derived over the whole module (26/23 → 27/22); controller re-pinned 5214 → 5216 |
| `991758c1f` | Fix round 2 (review), doc-only: the two stale ratchet narrative figures, the split-decision set, the withdrawn Form-E mechanism, and 4 minors; controller re-pinned 5216 → 5254 |

Nothing pushed. The isolated baseline worktree was removed at task close
(`git worktree list` verified back to the three pre-existing entries).

**Final shape: 185 of 285 candidates moved, 100 excluded.**

> **Review round (fix round 2) — doc-only.** Five Important and eight Minor
> findings, every one an evidence/accuracy defect rather than a code defect:
> the move itself was verified byte-perfect over the whole set. Each is
> corrected in place below and marked *(Correction, review round)* rather than
> silently rewritten, so the delta between what this report first claimed and
> what is true stays legible. The two that matter most: the callback-identity
> exclusion carried a **wrong mechanism** (§1 class 11), and a red I called
> **new** is documented three places in the recipe (§8).

---

## 1. Method census

### Method

Recipe §2's `ast` walk over every `LibraryScreen` class-body method, matched
as a **SUBSTRING on `"note"`** (case-insensitive), never a `startswith`
filter. Re-derived at this task's own execution time per §6.

- class-body `FunctionDef`/`AsyncFunctionDef` total: **1,290**
- containing `"note"`: **291 raw matches, 285 unique names**

**The 6-name gap is not a property/setter pair, as in every prior series — it
is a byte-identical DUPLICATE block dev shipped twice.**
`_library_notes_work_session_reader_width`,
`_dispatch_library_notes_work_session`,
`_library_notes_work_first_preferences`, `_set_library_notes_source`,
`_dispatch_database_note_identity_cleared` and
`_activate_database_note_work_session` were each defined twice, at
`library_screen.py:6557-6662` and `:6663-6768` (at the parent `afaf2320c`).
The two 106-line blocks are byte-for-byte identical — verified by string
comparison, not by eye — so the second definition silently won at class
creation and behaviour never depended on which. `git log -L` attributes the
first block to `7cf89de6c` ("consolidate long-lived branch changes") and the
second to `fd505637f` ("apply work-first Notes sessions"): a dev-side merge
artefact, not this program's. A move cannot preserve it (two delegators
cannot share one name), so **both copies were removed and one delegator per
name installed**; the definition on the controller is byte-identical to both,
asserted explicitly in the verification script. This is the whole reason a
PURE MOVE moves the method count (1290 → 1284).

### Decorator/shape breakdown of the 285 candidates

93 carry `@on`, 6 are `action_*`, 7 are `@staticmethod`, 5 are `@property`,
**0 are `@work`**.

### The 35 Message-typed `@on` handlers, and §4's third whitelist member

The brief required these enumerated and classified with whitelist evidence
per name. **They are decorator-dispatched (whitelist member 1), not
name-dispatched, and §4's third member is INERT for notes** — the opposite of
what the wave-8 plan predicted and of what task 1's hand-off repeated.

The 93 `@on` methods carry **99 decorators: 63 selector-bound, 36
Message-typed** across **35 unique method names**
(`handle_library_notes_lasting_back` carries two: `LibraryNotesAddFromFiles
Canvas.BackRequested` and `LibraryNotesSyncRootsCanvas.BackRequested`).

Every one of those 36 Message classes was **imported and its `handler_name`
read** — the exact resolution recipe §4 prescribes ("construct the `Message`
subclass it would receive and confirm `handler_name` matches"). Sample of the
full table:

| Handler method | Message | `Message.handler_name` | Matches? |
|---|---|---|---|
| `_handle_file_notes_editable_opened` | `FileNotesEditableOpened` | `on_file_notes_editable_opened` | no |
| `_handle_file_notes_identity_cleared` | `FileNotesIdentityCleared` | `on_file_notes_identity_cleared` | no |
| `_handle_file_notes_reload_confirmation_changed` | `FileNotesReloadConfirmationChanged` | `on_file_notes_reload_confirmation_changed` | no |
| `_handle_file_notes_root_changed` | `FileNotesRootChanged` | `on_file_notes_root_changed` | no |
| `handle_library_note_work_pane_editor_ready` | `LibraryNoteWorkPane.EditorReady` | `on_library_note_work_pane_editor_ready` | no |
| `handle_library_note_import_execute` | `LibraryNoteImportCanvas.ImportRequested` | `on_library_note_import_canvas_import_requested` | no |
| `handle_library_note_import_add_source` | `LibraryNoteImportCanvas.AddSourceRequested` | `on_library_note_import_canvas_add_source_requested` | no |
| `handle_library_notes_lasting_apply` | `LibraryNotesAddFromFilesCanvas.ApplyRequested` | `on_library_notes_add_from_files_canvas_apply_requested` | no |
| `handle_library_notes_lasting_root_action` | `LibraryNotesSyncRootsCanvas.RootActionRequested` | `on_library_notes_sync_roots_canvas_root_action_requested` | no |
| … 27 more, same shape | | | no |

The table shows 9 of the 35 names; **the remaining 26** follow the identical
pattern. By owning widget, the **36 decorators over those 35 names** are: 4
module-level `FileNotes*` events (`on_file_notes_*`), 12
`LibraryNoteImportCanvas.*` (`on_library_note_import_canvas_*`), **1
`LibraryNoteWorkPane.EditorReady`** (`on_library_note_work_pane_editor_ready`),
16 `LibraryNotesAddFromFilesCanvas.*`
(`on_library_notes_add_from_files_canvas_*`) and 3
`LibraryNotesSyncRootsCanvas.*` (`on_library_notes_sync_roots_canvas_*`) —
4+12+1+16+3 = 36, one more than 35 because `handle_library_notes_lasting_back`
carries two. *(Correction, review round: this paragraph shipped saying
"remaining 27" — 35 − 9 is 26 — and its owner list omitted `LibraryNoteWorkPane`
entirely, so the four groups it named summed to 35 while double-counting the 9
already tabled. Both fixed by recounting rather than re-reading.)*
**Not one equals the name of the method decorated with it**, so every one is
reached only through `cls.__dict__["_decorated_handlers"]`, and every one is
kept unconditionally as a screen delegator under whitelist member 1.

Run the other direction — over every `Message` subclass in
`tldw_chatbook/Widgets/Library` (**113 distinct `handler_name`s**) — the only
ones that ARE `LibraryScreen` methods are 4 Textual builtins (`on_resize`,
`on_key`, `on_mouse_down`, `on_descendant_focus`) and the 7 prompts-owned
`on_prompt_block_editor_*`. `LibraryScreen` carries **17** `on_*` methods in
total: 10 lifecycle hooks and those 7. **Zero are notes-owned.**

`test_no_notes_handler_is_name_dispatched_by_textual` pins both directions,
so a later notes widget that DOES introduce a name-dispatched handler fails
loudly instead of being pruned silently in task 3.

### Completeness checks

- **Non-notes-named methods with a notes-ish `@on` selector:** one hit, a
  false positive — `set_library_collection_capture_mode`'s four-selector list
  includes `#library-collections-mode-notes`.
- **Reverse call-graph fixpoint** (non-notes-named methods whose every
  in-class caller is transitively notes-named — the conversations exemplar's
  `_conversation_records` shape): **6 names**, every one SHELL-named, none
  moved. `_apply_library_emergency_geometry` with its two helpers
  `_capture_library_emergency_restore_receipt` and
  `_restore_library_emergency_receipt`; `_library_compose_scoped_ref`;
  `_library_landing_control_row_id`; `_library_landing_focus_control_id`.
  Each is a general shell primitive with one live caller today; moving them
  would relocate shell infrastructure into a subsystem controller — the One
  Home Rule ruling the media series applied to its review-set family. The one
  a mover calls is bound as a named late-binding dependency.

### MOVE/EXCLUDE verdicts — 185 MOVE, 100 EXCLUDE

Disjoint classes, in the order applied, so the numbers sum. Classes 1–9 come
from the static censuses; class 10 came from this task's own **battery**,
which recipe §3 records as the expected shape of the work.

| # | Class | Count | Reason |
|---|---|---|---|
| 1 | unbound-fake-self (`Tests/`) | 57 | §3 shape 1: `LibraryScreen.<name>(fake, …)` / `types.MethodType(...)` / `getattr(LibraryScreen, "<name>")` across ALL 3,372 `Tests/` files |
| 2 | in-file `LibraryScreen.<x>(self, …)` TARGETS | 11 further | a delegator called unbound with a foreign `self` reaches for `_notes_controller` on that object |
| 3 | …and CALLERS of that shape | 8 further | moved, their `self` is the controller, and the screen-resident target runs against a non-screen |
| 4 | instance-attribute monkeypatch on a REAL screen | 8 further | §3's opening rule (13 names total; 6 have a mover caller, 7 held by the conservative rule) |
| 5 | not-notes-owned | 5 | Export's `_resolve_library_export_chachanotes_db` + the 4 Collections quick-capture methods (already pinned by `test_library_collections_wiring.py`) |
| 6 | `_library_note_session` projection-property family | 4 further | 5 contiguous `@property` members at `:3944-3985`; `_library_note_dirty` is class-patched, so moving the other 4 splits one family |
| 7 | shared shell helper (≥2 subsystems, method-level) | 3 | `_sanitize_note_content` (prompts+skills controllers bind it), `_library_note_keywords_from_input` (prompts), `_sync_library_notes_reader_layout_from_shell` (media + the generic pane dispatcher + `_sync_library_canvas`) |
| 8 | class monkeypatch | 1 further | `_library_note_dirty` (`monkeypatch.setattr(LibraryScreen, …)` at `test_library_shell.py:2419` and `monkeypatch.setattr(type(screen), "_library_note_dirty", property(...))` at `:17652-17656` — **not** `patch.object`, as the controller docstring first said); `_refresh_library_note_detail` is also one of §3's four permanently screen-routed names |
| 9 | module-globals coupling | 1 | `_write_library_note_export_file` reads bare `validate_path_simple`, ACTIVE at `test_library_shell.py:22311` |
| 10 | `partial(LibraryScreen.<x>, self, …)` target | 1 | `_restore_library_notes_browse_return_receipt`; **found by the battery**, see §6 |
| 11 | test-bound-and-captured | 1 | `_exit_library_note_editor_guarded`, held by §3's conservative opening rule — **not** the Form E claim this report first made, see below |
| | **TOTAL EXCLUDED** | **100** | |

**Movers: 185** — 70 `@on`, 4 `action_*`, 3 `@staticmethod`, 108 plain (35 of
the 185 are `async def`). 3,934 source lines of body.

### Module-globals census (§3's eighth shape, mandatory, run to completion)

All **82 bare module-global names** the mover bodies read, crossed against
every `library_screen`-scoped patch shape (direct-attribute,
fully-qualified string, two-argument `monkeypatch.setattr`/`patch.object`)
across ALL of `Tests/`, under every alias tests actually import the module as
— `library_screen`, `library_screen_module`, `library_module`,
`screen_module`, the list DERIVED by `ast` rather than assumed. **10 names had
hits; each hitting test was read.**

- `validate_path_simple` → **ACTIVE** (3 sites, 2 files).
  `test_library_shell.py::test_library_shell_note_write_export_file_rejects_
  invalid_path` patches `library_screen_module.validate_path_simple` with a
  raising stub, calls `screen._write_library_note_export_file(...)` on a REAL
  screen, and asserts `not destination.exists()` plus the warning copy. Moved,
  the real validator would accept the `tmp_path` destination and the file
  would be written. **`_write_library_note_export_file` excluded.**
- `_sync_library_canvas` → **LATENT, kept** (33 sites across 10 files, "site"
  per the ingest controller's definition). 31 movers forward bare `self` into
  it; no site's enclosing test reaches a notes mover's own call path. Same
  systemic verdict every sibling controller carries.
- `_LibraryNotesRestoreGuard` (3), `LIBRARY_ROW_BROWSE_NOTES` (2),
  `LIBRARY_NOTES_SOURCE_DATABASE`, `LIBRARY_NOTES_SOURCE_FILES`,
  `NoteFlushOutcomeKind`, `build_library_shell_state`,
  `resolve_adaptive_reader_layout`, `asyncio` → **LATENT**: every one a plain
  module-attribute READ (constructing a guard for an assertion, or naming a
  constant), never a patch. `asyncio`'s hit patches an attribute OF the shared
  module object.

The census also earned a cleanup: `validate_path_simple` and (after fix round
1) `LibraryNotesTreeReceipt` became unused imports in the controller and were
removed, verified by a per-name occurrence count.

### Class 11 restated — the Form-E claim was WRONG, and is withdrawn

**Correction (review round).** This report and the controller docstring first
justified excluding `_exit_library_note_editor_guarded` as recipe §3's TENTH
shape (the media series' Form E, callback identity): that
`test_screen_navigation.py:3225`'s `focus_calls ==
[screen._restore_library_notes_focus_identity]` would compare against a
CONTROLLER-bound method once the scheduling body moved. **It would not**, on
two independent grounds, and I established neither before writing it:

1. **The captured callback is not a bound method at all.** Read at the tree
   rather than inferred from the assertion text, the object that assertion
   receives is `finish_list_projection` — a CLOSURE defined inside
   `_exit_library_note_editor_guarded`. No move could change which object it
   captures, so the Form-E mechanism never applied.
2. **The test is RED at the parent, on that exact assertion.** Verified in a
   fresh isolated worktree at `afaf2320c` (3.14.2, own venv): `At index 0
   diff: <function LibraryScreen._exit_library_note_editor_guarded.<locals>.
   finish_list_projection> != <bound method ...._restore_library_notes_focus_
   identity>` — byte-identical to the branch. `test_screen_navigation.py` is
   on §7's documented list at recipe:1121 with a **stable 32-name failure set
   across trees**, and that entry says in terms that a future wave should
   compare name sets and "should not read a green expectation into it". I read
   a mechanism into a red I had never checked against the parent.

**The exclusion is retained, on the rule that actually governs**:
`_exit_library_note_editor_guarded` is awaited directly on a REAL screen at
`test_library_notes_reader.py:1532` and injected into another controller as a
named dependency by `library_screen.py:3223-3225`, so §3's conservative
OPENING rule — a name a test binds or captures stays screen-routed until that
subsystem's cleanup PR retargets the fixtures — holds it. It is a MOVE
candidate for a later series once those fixtures retarget, the same
disclosure media made for 7 of its own 16. Recipe §3's own words are the
reason this is rewritten rather than quietly left standing: **"an exclusion's
justification is a claim, and it is held to the same evidence standard as a
test-passing claim; if you assert the narrow condition, verify it name by
name, otherwise cite the broad rule that actually governs."** The exclusion
set was right; the stated reason was wrong, and a wrong reason is worse than a
thin one because the next series inherits it.

**Notes therefore carries ZERO Form-E callback-identity carriers.** The census
that would find one returned exactly this one candidate, and reading it
disqualified it.

### Bare-`self` census (§3's generalized standing correction) — both figures

Over all **285 candidates**: **174 occurrences in 74 methods**. Over the
**final 185 movers**: **37** — 31 `_sync_library_canvas(self, "notes", …)`
duck-typed forwards and 6 `getattr(self, "<literal>")` reads, every one bound.
Both figures are stated because only the pair shows the census working.

**Zero identity comparisons and zero `ancestors` membership tests survive
into the mover set.** The screen carries 11 such sites
(`self.app.screen is not self` ×5, `node is self`, `self not in
<w>.ancestors` ×2, `self in <w>.ancestors` ×2, `focused is not self.focused`)
and every one lives in a NON-candidate method
(`_library_emergency_return_eligibility`, `_library_ref_is_live`,
`on_descendant_focus`, and eight media/prompts/skills-named methods). The four
`event.control is self._library_file_notes_workspace` comparisons inside
movers compare against a `LibraryNotesState` FIELD, which resolves through the
controller's shim loop to the identical object.

### Sync-engine / watcher-thread census (the brief's explicit caution)

`Notes/sync_engine.py` no longer exists
(`Tests/Notes/test_notes_sync_cutover.py::test_cutover_release_deletes_every_
legacy_writer_module` enforces its absence, and it passes). The lasting-sync
runtime lives on the **app**; the screen holds a controller (WIRING) plus a
projected snapshot. **Zero movers are `@work`-decorated** and **zero movers
are reached from a thread hand-off seam.**

*Correction (review round) — the first census asked too narrow a question.* It
swept only the `call_from_thread` attribute (**296 sites** across
`tldw_chatbook/` on a re-count; the report first said 279, from a run that
scanned a subset). That spelling is one of several: `loop.call_soon_threadsafe`,
`asyncio.run_coroutine_threadsafe`, `asyncio.to_thread`,
`loop.run_in_executor`, `Executor.submit`, `threading.Thread`,
`threading.Timer`, `_thread.start_new_thread` and `apply_async` all hand work
across a thread boundary and are **invisible to a `call_from_thread` grep**.
Re-run over all ten spellings the sweep finds **1,478 sites** (`to_thread` 867,
`call_from_thread` 296, `Thread` 100, `Timer` 77, `submit` 59,
`run_in_executor` 48, `call_soon_threadsafe` 23, `run_coroutine_threadsafe` 7,
`apply_async` 1) and **zero** whose ±5-line neighbourhood names a notes cluster
member. *(An independent re-derivation with a slightly different spelling set
measures 1,419 sites and the same zero; the instrument differs, the answer does
not.)* **The widened sweep is the census that answers the question** — the
narrow one merely failed to contradict it. The spelling set belongs in the
recipe at wave close, beside §3's own census-spelling catalogue, for the same
reason the fifth and sixth attribute spellings do: a census is only as good as
the spellings it searches for.

The two publish callbacks the screen hands the
controllers (`_publish_library_note_import_snapshot`,
`_publish_library_notes_lasting_sync_snapshot` — both movers) touch the DOM
(`_sync_library_canvas`, `_apply_library_notes_footer_context`) and are
therefore already UI-thread-only, exactly as task 1 established at field
level. `run_worker` is bound as a framework service (13 movers use it), and
every worker it starts is a `LibraryScreen`-owned Textual worker whose
callback path is the same UI thread. **No moved body is reached from a
non-UI thread through a deleted flat name.**

---

## 2. Split decision — SINGLE controller, by connected components

**Correction (review round): the published component figures are the
196-name set, not the 185 movers.** The split decision was taken at the point
it had to be — over the candidate MOVE set as it stood then, 196 names, before
exclusion classes 2/3 and everything after them trimmed it to 185. Over those
**196**: **one component of 135, one of 9, one of 2, and 50 singletons**
(135+9+2+50 = 196); with shared-field edges, **154 + 9 + 2 + 31** (= 196). The
sums are the tell, and I did not check them.

Re-derived over the **final 185 movers** with the identical instrument (a
`self.<attr>` walk PLUS the `getattr(self, "<literal>")` spelling, since that
is the spelling the binding census must honour): **126 + 9 + 2 + 48
singletons**, and **146 + 9 + 2 + 28** with shared-field edges. An independent
re-derivation counting only plain `self.<attr>` edges measures **142 + 9 + 3 +
2 + 29**; I reproduced that exact profile by dropping the literal-`getattr`
edges, so the delta is entirely that one spelling and nothing else. All four
profiles are recorded.

**The verdict is unaffected by which set or which instrument is used** —
every profile is one dominant component, the same 9-member note-import group,
one 2-member pair, and singletons. Nothing here is a second seam.

The only candidate seam of any size is the 9-member note-import group
(`handle_library_note_import_{check,collision_choice,confirm_match,
destination,item_action,item_choice,retry}`, `_run_library_note_import_check`,
`_notify_library_note_import_failure`). It does not hold: four of the five
other `LibraryNoteImportCanvas` handlers (`…add_source`, `…cancel`,
`…collision_name`, `…page`) are singletons only because they touch nothing
else, and every one of the nine reaches the same
`_library_note_import_controller` WIRING instance and the same
`_library_note_import_*` state fields. Splitting would produce a second
controller of ~13 methods sharing a wiring handle and a state object with the
first — nothing §17's per-file governance does not already give.

The feared size did not materialise: 100 exclusions leave 3,934 lines of moved
body, putting the file (5,214 lines) between `library_media_controller.py`
(4,630) and `library_prompts_controller.py` (4,998). Wave 3's split gate was
not triggered: **no non-notes tangle emerged** — the fixpoint's 6 extra names
are shell primitives, not another subsystem.

---

## 3. RED wiring commit (`afaf2320c`)

`Tests/Architecture/test_library_notes_wiring.py` gained
`_NOTES_CLUSTER_METHOD_NAMES`, `_NOTES_CLUSTER_STATICMETHOD_NAMES`,
`_NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED` (empty until task 3),
`_NOTES_CONTROLLER_BOUND_NAMES`, and 7 tests.

**RED proven at the parent with the screen untouched** (`git status` showed
one modified file): **6 failed / 9 passed**. The six reds are the six checks
that need the controller module; `test_no_notes_handler_is_name_dispatched_by_
textual` passes at the parent by design — it is a standing invariant about
Textual dispatch, not a controller-existence check.

---

## 4. Move commit (`9e13f0207`) + fix round (`a529dbd3e`)

### Byte-for-byte evidence

Extraction and verification were both **scripted**, per §14's rule for
clusters over ~40–50 methods: each method's exact source segment was cut from
the original file by `ast` line offsets (decorators included), assembled with
a hand-written header/footer, then a SECOND script re-parsed the pre-move
screen (`git show afaf2320c:…`) and the finished controller and asserted
byte-for-byte identity per method.

- **185 / 185 movers byte-identical.**
- All 6 duplicate-definition movers: 1 distinct text across both original
  copies, and the controller matches it.
- **Zero non-mover screen methods changed** (every remaining `LibraryScreen`
  method's source segment is byte-identical to the parent's, `__init__`
  excepted).
- Exactly one delegator per mover; **no delegator longer than 6 lines**.

### Binding surface — 102 names, verified in BOTH directions

Derived mechanically by walking all mover bodies for every `self.<attr>`
load/store and every `getattr(self, "<literal>")`, subtracting the 100 state
fields and the movers themselves — **99 names** — plus **3 the walk
structurally cannot see**: `is_running`, `_library_canvas_projection_depth`
and `_library_canvas_resync_pending`, which the shared `_sync_library_canvas`
dispatcher reaches off the bare `self` 31 movers hand it (the conversations
and media controllers each bind the same pair, for the same reason). The
`kind == "notes"` path of that dispatcher needs 16 names off the object it is
handed; the other 13 are covered by framework services, the state shim loop,
movers and named dependencies — enumerated in the controller docstring.

Groups: 16 framework services (live-read `@property`), 10 shared shell reads,
**6 read+write** (getter **and** setter: `_library_canvas_resync_pending`,
`_library_navigation_context_generation`,
`_library_notes_programmatic_focus_target`, `_library_notes_restoring_focus`,
`_library_selected_row_id`, `_pending_library_source_open`), 3 WIRING
accessors, 65 named late-binding callables.

**Hand-off (b) discharged, and measured rather than inherited.** Task 1
handed forward "5 late-binding hand-off methods (4 write, 1 reads)" for the 2
BLOCKED fields. Re-derived at this tree: `_library_notes_restoring_focus` is
written by 5 movers and `_library_notes_programmatic_focus_target` by 1, so
both got a getter **and** a setter accessor per the
`library_screen.py:2844-2852` media precedent — and the read-only variant task
1 predicted for `_queue_library_notes_scroll_interaction` is subsumed by the
same getter.

**The both-directions AST resolver check** (the brief's real check, since the
pin is one-directional per TASK-32013):

- *referenced → resolves*: **245 distinct names the moved bodies actually
  spell** — 244 as `self.<attr>` plus 6 as `getattr(self, "<literal>")`, union
  245, measured over the controller's own class body. **0 unresolved on the
  controller. 0 written through a setter-less property.** *(Correction, review
  round: this shipped as "250". 250 is that union PLUS the 5 dispatcher-only
  names the resolver script injects before checking — a figure of the
  instrument, not of the bodies. Both are stated with the method that produces
  them.)*
- *listed → referenced*: **3 pinned bound names are spelled by no moved body**
  — `is_running`, `_library_canvas_projection_depth` and
  `_library_canvas_resync_pending`, precisely the structurally-invisible trio
  this report names two paragraphs above and the wiring test pins by name.
  *(Correction, review round: this shipped as "0", because the resolver
  injected those three into the referenced set before differencing — the check
  answered a question it had already prejudiced. Re-run without the injection
  it returns exactly those 3 and nothing else, which is the right answer: 3
  deliberate, documented, dispatcher-only bindings and no decoration.)* The
  same direction, run before the injection was noticed, is still what caught
  the now-dead `reload_library_notes_browse_return_receipt` dependency after
  fix round 1; it was removed rather than left as decoration.
- *free names*: every bare module global the moved bodies read resolves in the
  controller module's own globals (8 apparent misses are the six generated
  setter-property names at class scope plus two lambda parameters, all
  confirmed false positives by reading); **zero unused imports**.

### Screen change

- Function-local import of `LibraryNotesController` inside `__init__`'s
  established lazy-import block (**born-lazy**), never at module level.
- `self._notes_controller = LibraryNotesController(...)` immediately after
  `self._media_controller`, 92 named dependencies (289 lines incl. a 3-line
  comment).
- **Preimport-closure suffix**: `"notes"` added to the deferred-suffix tuple
  in `Tests/Packaging/test_library_preimport_closure.py`.
- **Born-governed row**: `library_notes_controller.py` added to
  `Tests/Architecture/test_library_modules_size_ratchet.py` at its exact
  measurement — the glob would have failed by name otherwise.
- **Screen re-pin**: `35621/1290 → 32325/1284`, delta **-3296**, reconciled
  term by term off the tree (not estimated): −4034 removed (191 `FunctionDef`
  segments), −6 (the blank line after each of the 6 second copies), +452 (185
  delegators), +3 (import), +289 (construction site).
- **Production diagnostic inventory** regenerated after reading the rows: 8
  statements moved `library_screen.py` (88→80) → `library_notes_controller.py`
  (0→8), **all 8 with identical digests on both sides** (`--statements
  --since` confirms "moved only", zero reworded, zero new interpolation, no
  new sink).

### Hand-off (a) — NO seed needed, and here is why

Task 1's **TASK-2 MUST** required seeding `_notes_controller` in
`Tests/UI/test_library_ingest_inline_consent.py` *when
`_apply_library_notes_stage_visibility` moves*. **It does not move.** It is
monkeypatched on REAL, `__init__`-constructed screens at
`test_library_resize_focus_gates_t23025.py:165`/`:255` (`screen = await
_settled_library(host, pilot)`) and
`test_library_media_return_settlement.py:95`/`:118` (`screen =
LibraryScreen(app)`), so §3's opening rule keeps it screen-resident. The
ingest controller's injected lambda therefore still reaches a real,
full-bodied screen method, and the bypassed fixture is untouched. Confirmed
empirically: the 7-file `__new__` battery is **423 passed / 1 failed**, the
one failure being the documented pre-existing
`test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_footer_
when_retry_availability_changes` — byte-identical to task 1's own measurement
of the same battery.

### The `canvas_sync.py` notes branch — deliberately NOT added

Per task 1's binding correction (and `canvas_sync.py`'s own comment dating the
media precedent to the CLEANUP PR `5dd2e71cf`), the notes dotted branch at
`canvas_sync.py:227` lands at **shim-deletion time (task 3)**, together with
its mutation-verified guard beside the conversations and media siblings in
`Tests/UI/test_library_selection_updates.py`. The composed name
`_library_notes_row_selection` still resolves through the live screen shim
today, and — newly relevant after this move — also through the **controller's**
own generated shim loop when `_apply_library_row_toggle` is handed a
controller `self` from `handle_library_notes_row`. Adding the branch now would
be a no-op edit outside §4's transform whitelist. **Handed forward to task 3
unchanged.**

---

## 5. `.git-blame-ignore-revs`

Two rows appended, each hash taken from `git rev-parse` and never transcribed:
`9e13f0207cadb0de5eb7559faa4ada234e3a6f6c` and the fix round's own. Both rows
state what the commit did and that zero method bodies were edited.

---

## 6. The battery-found hazard (fix round 1) — the one real regression

The paired sweep produced **4 branch-unique failures**, all real, in two
kinds.

**(a) A silent-then-loud mover hazard the static census could not see.**
`handle_library_notes_filter` and `handle_library_notes_filter_clear` (both
exclusions) build

```python
restore = (
    partial(
        LibraryScreen._restore_library_notes_browse_return_receipt,
        self,
        browse_receipt,
    )
    if browse_receipt is not None
    else self._focus_library_notes_filter_input
)
```

(Three arguments, not four — `library_screen.py:21292-21297` and `:21329-21334`
at the current tree. An earlier draft of this report and commit `a529dbd3e`'s
message both rendered it `(..., self, receipt, guard)`, a signature that does
not exist. The mechanism is unchanged; the quotation was wrong.)

The census behind exclusion classes 2 and 3 walked every bare `self`
`ast.Name` and asked *what CALL is it a direct argument of* — the answer here
is `partial`, not `LibraryScreen.<x>`, so the TARGET scored as an ordinary
mover. As a delegator it then ran with whatever object the partial was handed:
`AttributeError: 'types.SimpleNamespace' object has no attribute
'_notes_controller'` in
`test_library_notes_folder_navigator.py::test_clearing_filter_restores_same_
epoch_browse_receipt_without_touching_ranges[button|empty_submit]`.

**The standing correction, stronger and cheaper than enumerating call shapes:
census EVERY `LibraryScreen.<name>` `ast.Attribute` reference anywhere in
`library_screen.py`, whatever expression encloses it.** Exhaustive by
construction — it cannot be defeated by `partial`, a variable assignment, a
list literal, or the next wrapper someone introduces. Run that way at the
parent over the WHOLE module (not only the class body) it returns **27
targets**, **22** of them notes candidates, exactly **one** of which was a
mover: this name. Both figures belong in the record — the
direct-call-argument census found **20 of the 22**, and the two it missed were
`_focus_library_notes_tree_after_page` (already excluded as a monkeypatch, so
its miss was harmless and over-determined) and this one, which was not.
*(Correction of record, per §6: commit `a529dbd3e`'s own message says "26
targets" — the figure from a first run that walked only the class body, one
target short. A commit message is the one place a correction cannot be
applied, so it is recorded here instead; the controller docstring and this
report both carry 27/22.)* `_restore_library_notes_browse_return_
receipt` is now excluded (class 10); no mover calls it, so it needed no
binding, and the dependency it was the sole consumer of was removed with it.

**(b) Two hardcoded-file-path censuses, retargeted in this stage.**
`test_library_notes_lasting_sync_flow.py::test_production_screen_derives_
lasting_route_from_the_app_runtime` and
`::test_relationship_chooser_reuses_the_one_existing_import_controller` read
`library_screen.py` by path and assert source substrings
(`self._library_notes_sync_controller.choose_relationship`,
`choose_relationship(`, the two canvas class names). Recipe §3 is explicit
that this shape has **no grace period** — it ships RED at the move boundary
rather than staying silently green, so the no-red-ships rule fixes it in the
stage that moved the code. Both were retargeted at
`library_notes_controller.py` for the cluster half of their invariant,
assertions kept, and the "no second import controller" half is now asserted
over **both** files so it cannot be satisfied by the name merely having left
the file the guard used to read.

Recipe §3's own amended-RED-tuple rule covers the bookkeeping: the mover
tuple, the "N of 285" counts, the binding-name count, both ratchet rows, the
controller docstring and the wiring test were re-derived after the correction
rather than patched by one number.

**Erratum (review round): that claim was not true of two lines, and the
exception is the whole point of the rule.** `Tests/Architecture/test_screen_
size_ratchet.py:769-770` still read *"Fresh `_measure()`: 35621/1290 ->
32286/1284. Line delta -3335 reconciles EXACTLY"* — the GREEN commit's
figures — sitting eighteen lines above a per-term derivation that correctly
summed to **-3296** and a row correctly pinned at **32325**. The fix round
re-derived the terms and the row and left the narrative sentence that
introduces them. `Tests/Architecture/test_library_modules_size_ratchet.py:331`
carried the same shape: *"the 93-dependency constructor, the 103 binding
properties"* against a measured 92 and 102. Both are corrected in this round,
each with an in-file erratum comment.

This is **recipe §6's three-places-inside-the-guard hazard**, which the recipe
records from wave-6 and which I quoted approvingly in this very report before
reproducing it: *"correcting two of three sites and assuming the third feels
identical to correcting all three."* The concrete rule §6 already gives and I
did not follow: **when a measurement changes mid-task, `grep` the guard file
for the OLD number before committing.** Doing that for `32286` and `3335`
would have taken one command and found both lines.

---

## 7. Battery

All runs `-p no:randomly`, branch venv Python **3.14.2**. The isolated
baseline worktree sat at `afaf2320c` with its own `uv venv --python 3.14` —
**interpreter parity proven** (3.14.2 on both), **package parity proven**
(`uv pip list` name lists `diff`ed to identical, 106 rows), and the isolated
venv verified to resolve its OWN tree.

| Check | Result |
|---|---|
| `Tests/Architecture/test_library_notes_wiring.py` | **15 passed** (6 failed / 9 passed at the parent — the RED proven first) |
| 9 wiring suites (collections 4, conversations 7, export 5, ingest 5, media 12, notes 15, prompts 13, search+RAG 5, skills 8) | **74 passed**, 0 failed |
| `Tests/Architecture/test_screen_size_ratchet.py` | 3 passed, **2 failed** — both documented `chat_screen.py` rows |
| `Tests/Architecture/test_library_modules_size_ratchet.py` | **2 failed** — both documented dev-owned rows (`library_media_browse_controller` 371-pin; `library_conversations_controller` slack mirror). The new `library_notes_controller` row passes both checks |
| `Tests/UI/test_library_recompose_ratchet.py` | **1 failed** — `66 found, 63 allowed`, **byte-identical at the isolated parent**; task 1's own new §7 entry |
| `Tests/Packaging/test_library_preimport_closure.py` | passed (with `"notes"` added to the deferred tuple) |
| `Tests/Performance/test_ui_ready_module_census.py` | **passed** — the zero-headroom 972 pin; the controller is born-lazy, so nothing it imports enters the first-paint window |
| `Tests/UI/test_library_screen_reuse.py` | 3 passed, **1 failed** — `test_on_screen_suspend_stops_every_timer_in_isolation`, `AttributeError: … no attribute '_unavailable_navigation'`, **identical at the isolated parent** (dev's TASK-31249 red) |
| the 7 `__new__`-bypass files | **423 passed, 1 failed** — the documented `test_library_ingest_retry_last.py` red; identical to task 1's own run |
| `Tests/UI/test_library_notes_characterization.py` | **3 passed** |
| `Tests/Notes/test_notes_sync_cutover.py` | passed (see the guard note below) |
| `Tests/UI/test_library_selection_updates.py` | **1 failed** — `test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`, **identical at the isolated parent**. NEW documented pre-existing red |
| `Tests/UI/test_library_notes_lasting_sync_flow.py` | **7 passed** after the retarget |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** |
| End-to-end construction probe | A real `LibraryScreen(MagicMock())` was constructed; all 185 movers callable on the controller AND on the screen; all 102 pinned bindings resolve on the INSTANCE (sole exception `app`, which raises `NoActiveAppError` off the app tree — the documented reason the pin asserts at CLASS level); all 100 state fields round-trip controller → state → screen shim; all 3 WIRING accessors return the identical object the screen holds; both BLOCKED setters write through to the screen; a `@staticmethod` delegator returns through the controller CLASS (`LibraryScreen._note_word_count("a b c") == 3`) |

**Documented reds only.** Every failure above is either in recipe §7's list,
is one of task 1's two new entries, or is the one NEW pre-existing red
recorded here — each verified at the isolated parent, never assumed.

### Paired sweep — three matched batches, sequential, isolated

Recipe §7's method, narrowed to `Tests/UI -k "note and library"` (1,131
tests) as task 1 did, split into three file batches so each run completes in
the foreground. The selection is identical on both trees, so the comparison
is exact.

| Batch | Branch | Isolated parent `afaf2320c` | Unique |
|---|---|---|---|
| 1 — `test_library_shell.py` | **171 failed / 83 passed** (498s) | **171 failed / 83 passed** (499s) | **0 / 0** |
| 2 — the three `file_notes_{workspace,git,git_push}` files | **23 failed / 349 passed** (91s) | **23 failed / 349 passed** (94s) | **0 / 0** |
| 3 — everything else (pre-fix) | 30 failed / 475 passed | 26 failed / 479 passed | **4 branch-unique / 0** |
| 3 — everything else (post-fix) | **26 failed / 479 passed** (135s) | **26 failed / 479 passed** (162s) | **0 / 0** |

Batch 1 was re-run on the branch AFTER fix round 1 and stayed at exactly
171/83 with a zero-name diff both ways.

**Zero real regressions after fix round 1.** The four branch-unique names
before the fix were the two `AttributeError`s and the two path censuses in §6,
both dispositioned by READING, not by re-running — neither is timing-shaped, so
the n>1/interleaved-round-robin protocol did not apply. No timing- or
geometry-shaped disposition arose in this task; had one arisen it would have
been run round-robin per §7's third level.

### The cutover guard's task-1-disclosed vacuous flip — status after this move

`Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_legacy_
timer_worker_or_mutating_handler` was red at `889e12b86` and turned green in
task 1 for a reason it did not intend (the shim loop *composes* the name it
censuses). **This move does not change that**, and I re-derived rather than
assumed it: no mover's name matches the guard's `handle_library_notes_sync_`
/ `_library_notes_sync_` prefixes, the controller contains no
`_library_notes_auto_sync_timer` attribute, no `group="library_notes_sync"`
worker keyword, and none of `NotesSyncEngine`/`NotesSyncService`/
`sync_service`. The guard's sibling censuses over `library_screen.py`
(`test_retired_product_surfaces_do_not_read_or_write_legacy_sync_config`,
`test_missing_lasting_runtime_has_no_legacy_fallback`) are equally unaffected:
the controller's string constants intersect the legacy config keys in **zero**
places, and `InertLastingSyncRuntime` is still spelled in `library_screen.py`.

**One new fact for task 4's filing:** pointed at the controller file, the
guard's method-name prefix census would report exactly one violation —
`_library_notes_sync_controller:988`, the WIRING binding accessor. A naive
retarget therefore false-positives, which is one more reason the right fix is
a product decision rather than a mechanical repoint.

---

## 8. Findings and hand-offs

### WITHDRAWN: the "NEW §7 red" finding, and the meta-lesson it earns

**`Tests/UI/test_library_selection_updates.py::test_tier1_toggle_falls_back_
to_recompose_on_query_one_failure` is NOT new.** It is already in §7's
documented list — recorded at **recipe:1062** (found by wave-1 Task 9,
confirmed on a pristine baseline), re-recorded at **recipe:1594-1600**, and
counted among the wave-7 close's own documented reds at **recipe:5937**. The
finding is withdrawn and the task-4 hand-off row is dropped.

**The meta-lesson, and it is the point of the entry I walked past.**
recipe:1594-1600 exists *precisely because wave-7 task 3 met this same test as
an apparently-NEW red and had to re-derive it* — that entry's own words are
"recorded here only because wave-7 task 3 met it as an apparently NEW red and
had to re-derive it." I then did the identical thing, which makes this the
**third** time this one name has been reported as new. My stated reason was
"task 1 did not run this file", and that reason is exactly the error:
**§7's list is the authority on what is pre-existing, not the set of files an
earlier task in the same wave happened to run.** A red must be checked against
§7 by NAME before it is called new — a parent-tree re-run proves only that it
is not this task's regression, never that it is undocumented.

There is a second, sharper mechanism underneath it, worth recording because it
generalises: this test sat INSIDE my own swept batch, so its failure was
absorbed into the 26-name both-trees backdrop and never surfaced as
branch-unique. A paired sweep is built to make branch-unique names visible; it
is structurally blind to *"already documented"* as a property. That is the
same absorption §7:1062 warns about — and it means the §7 name-check has to be
run deliberately, not expected to fall out of the sweep.

### NEW recipe finding — a SIXTH census spelling, and its generalized form

§3's catalogue has five spellings for finding a name a move would break;
none covers **an unbound `LibraryScreen.<name>` reference inside another
expression**. `partial(LibraryScreen._x, self, …)` is the instance this task
hit; a variable assignment, a list literal or a decorator argument would
behave identically. The generalized census is one line of `ast` and is
exhaustive by construction: **every `ast.Attribute` whose `value` is
`Name("LibraryScreen")`, anywhere in `library_screen.py`, whatever encloses
it.** At the parent it returns 27 targets, 22 of them notes candidates. Task 4 should write this into §3
beside the five existing spellings, with the incident.

### NEW recipe finding — a duplicated definition block defeats "a pure move never changes the method count"

Six notes methods were defined twice, byte-identically, and the class-body
`FunctionDef` count therefore over-counted the real surface by 6. Every prior
series' ratchet narrative asserts "the METHOD count is unchanged, as every
pure move's must be". That is now false in one measurable case, and the guard
comment says so explicitly. Task 4 should record the shape: **before writing
the "method count unchanged" line, check `Counter(m.name for m in class body)`
for duplicates, and if any mover is duplicated, state the collapse in the
same breath as the count.**

### Hand-offs to task 3 (cleanup)

- **The `canvas_sync.py` notes dotted branch** (`:227` at `afaf2320c`) plus
  its mutation-verified guard in `Tests/UI/test_library_selection_updates.py`
  land in the SHIM-DELETION commit, per task 1's correction of the plan.
  **New for task 3:** after this move the composed name must resolve on TWO
  objects, not one — the screen (via task 1's shim block) and the CONTROLLER
  (via the controller's own permanent shim loop), because
  `handle_library_notes_row` — an exclusion — forwards bare `self` into
  `_apply_library_row_toggle` while `_sync_library_canvas`'s notes path is
  reached from 31 movers with a controller `self`. Deleting the SCREEN shim
  therefore does not make the composed name unresolvable on the controller;
  the dotted branch must handle both receivers, and the guard must exercise
  the controller receiver too or it will pass vacuously.
- **`on_screen_suspend`'s flat-name string loop** still needs its explicit
  `self._notes_state.autosave_timer` block in the shim-deletion commit, plus
  the two `Tests/UI/test_library_screen_reuse.py` retargets (task 1's
  hand-off, unchanged by this move — `on_screen_suspend` calls no mover).
- **Delegator-prune census**: `_NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED` is
  empty and must be filled from a fresh `ast` census over `tldw_chatbook/` +
  every `Tests/` root + `Docs/` + `scripts/` + `Helper_Scripts/`. **All 74
  whitelist-exempt names KEEP unconditionally** — 70 `@on` + 4 `action_*`;
  **§4's third member (`on_<message>`) contributes zero, proven, and
  `test_no_notes_handler_is_name_dispatched_by_textual` keeps it proven.**
- **Re-derive every inherited number.** The bare-quoted-string /
  dict-of-name-string inventory, `library_inspection_admission._SCREEN_FIELDS`,
  `canvas_sync.py:456`'s `partial(getattr, screen, …)`, and
  `library_unavailable_navigation.py`'s direct `self._library_note*`
  references are all LEAD LISTS from task 1 — re-derive at task 3's own tree
  with a stated method.
- **`RowSelection` (`library_screen.py:358`)** is on task 1's dead-import
  list; re-verify after this move, since the move changed which names
  `library_screen.py` still uses.

### Hand-offs to task 4 (wave close)

- ~~Add the new §7 row (`test_library_selection_updates.py`).~~ **Withdrawn**
  — already documented at recipe:1062/:1594-1600/:5937. What task 4 SHOULD add
  is the meta-note in §7 above the list: *check a red against §7 by NAME before
  calling it new; the set of files an earlier task in the same wave ran is not
  evidence of anything.* This name has now been reported as new three times,
  and :1594 was written after the second.
- Write the sixth census spelling and the duplicate-block finding into §3/§6,
  **and the thread-handoff spelling set** (ten spellings, §8's thread census
  above) beside them.
- File the cutover-guard task noting the controller-file false positive
  (`_library_notes_sync_controller`).
- Record this series' own §2 table row: 285 candidates → 185 moved / 100
  excluded across 11 disjoint classes; prune fraction is task 3's to measure.
