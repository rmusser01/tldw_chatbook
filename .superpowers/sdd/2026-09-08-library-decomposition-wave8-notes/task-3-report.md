# Wave-8 Task 3 — Notes cleanup (notes series 3/3)

Branch `refactor/library-decomp-wave8-notes`, worktree
`/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`,
task parent `81bffe809` (task 2's close), wave start `889e12b86`.

**The FINAL cleanup PR of the eight-wave Library decomposition program.**
After it, `library_screen.py` carries no generated flat-state shim block for
any subsystem, and zero flat notes-name references in any of the six census
spellings.

| Commit | What |
|---|---|
| `3d670ee38` | `refactor(library): notes cleanup (notes series 3/3)` — 42 files |
| `d3528342c` | `chore(library): blame-ignore the notes cleanup` |
| `9d8244c72` | Fix round 1: three `_notes_state` seeds for media-trash fakes the retarget reached (a REAL 3-test regression the paired baseline caught), the wiring test's task-3 docstring paragraph, and recipe §23's sweep/seed/disposition findings |
| `4aca7171a` | `chore(library): blame-ignore the notes cleanup fix round` |

**No follow-on move commit**: a cleanup PR's retargets are FIELD retargets and
none of them removes a method-monkeypatch, so none of task 2's exclusions
became movable here — the same verdict the media series reached for its own 7
deferred candidates, re-derived rather than inherited (§6).

Nothing pushed. `progress.md` was not touched. The isolated baseline worktree
was removed at task close.

---

## 1. Census FIRST — SIX spellings, re-derived at this tree

Nothing was carried over from tasks 1/2. Name sets derived live: the **100**
field names from `dataclasses.fields(LibraryNotesState)` through
`notes_state_shim_attr()`; the **185** mover names from
`_NOTES_CLUSTER_METHOD_NAMES`; the **285** notes-named `FunctionDef`s on
`LibraryScreen` by `ast` (185 delegators + 100 full-bodied exclusions —
`set(candidates) - set(movers)` is exactly 100, and reconciles with task 2's
own figures with no arithmetic left over).

**Every census is `ast`-based, never substring or regex-over-source.** The
wave-7 lesson bites harder here than it did there: **5 of the 100 field names
are a proper PREFIX of another field name, across 8 colliding pairs** —
`_library_notes_filter` of `_library_notes_filter_records`,
`…_filter_generation`, `…_filter_navigation_generation` and
`…_filter_browse_receipt`; `_library_file_notes_workspace` of
`…_workspace_factory`; `_library_notes_operation` of `…_operation_counter`;
`_library_notes_sort` of `…_sort_choices_visible`; `_library_notes_stage` of
`…_stage_applied_signature`. Every edit in this task is applied at an `ast`
node's own span, so boundary safety is a property of the tool rather than of a
pattern.

### 1.1 The six spellings, BEFORE

Over `tldw_chatbook/` + all of `Tests/` + `Docs/` + `backlog/` + `scripts/` +
`Helper_Scripts/`:

| Spelling | 100 field names | 185 mover names | 100 exclusion names |
|---|---|---|---|
| attribute (`<recv>.<name>`) | **2,061** | 685 | 492 |
| `LibraryScreen.<name>` attribute (the SIXTH spelling) | 0 | **0** | 196 |
| bare quoted string (patch-target tables included) | 73 | 194 | 79 |
| kwarg | 61 | 12 | 17 |
| `def` | 4 | 371 | 149 |
| bare `ast.Name` / parameter | 0 | 0 | 8 |
| runtime f-string (`ast.JoinedStr`) | 3 | 1 | 1 |
| **total** | **2,202** | **1,263** | **942** |

Two rows carry the findings.

**The sixth spelling** — task 2's own hand-off, the generalized form of the
`partial(LibraryScreen._restore_library_notes_browse_return_receipt, self, …)`
hazard its battery found — returns **zero for the movers**. That is the answer
the census exists to produce: after task 2's fix round, no name that left the
screen is still reached as an unbound `LibraryScreen.<name>` from any
enclosing expression. The 196 exclusion hits are the population that shape
lives in, and they are exactly why 19 of task 2's 100 exclusions exist.

**The field `def` row is 4, not 0**, and all four are on OTHER subsystems'
controllers: `library_media_controller.py` declares three accessor
`@property` names — `_library_notes_compact`,
`_library_notes_focus_intent_generation` and `_library_notes_source` — and
`library_conversation_reader_controller.py` declares a fourth that REPEATS
media's second, `_library_notes_focus_intent_generation`. *(Corrected at
review: this shipped saying the reader controller "declares the third", i.e.
`_library_notes_source`. It does not; measured by an `ast` sweep of both class
bodies against the 100-name set, the reader controller's only such property is
the focus-intent one. The 4 is right; the attribution was not.)* Those PROPERTY
NAMES do not change; what changed is the screen-side lambda each one calls
(§2).

### 1.2 The six spellings, AFTER — 467 hits, every one accounted for

| Where | Hits | Verdict |
|---|---|---|
| `library_notes_controller.py` | 410 | its own PERMANENT generated shim loop — the mechanism that keeps 185 moved bodies byte-for-byte. Never retargeted |
| `Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py` | 17 | frozen evidence script; the standing precedent that still leaves `screen._library_skill_editor_state` there four waves after the skills cleanup |
| `library_media_controller.py` | 13 | 3 accessor `@property` NAMES + their own bodies' reads (moved bodies of the media series) |
| `Research_Workspace_Modules/quick_notes_section.py` | 10 | **FALSE POSITIVE** — a Research-Workspace widget's own `self._selected_note_id`, a different class entirely (recipe §18: check the RECEIVER, not just the string). Untouched |
| `Tests/Architecture/test_library_notes_wiring.py` | 9 | this series' own literal per-family mapping pins |
| `library_conversation_reader_controller.py` | 3 | 1 accessor property name + its 2 body reads |
| `Tests/Architecture/test_library_media_wiring.py` | 3 | the media wiring test's pins on those 3 accessor property names |
| `canvas_sync.py` | 1 | the `f"_library_{kind}_row_selection"` leg, now unreachable for `kind == "notes"` |
| `Tests/Notes/test_notes_sync_cutover.py` | 1 | the cutover guard's `_library_notes_auto_sync_timer` prefix string — task 4's filing, deliberately untouched (§6) |

**`library_screen.py`: 619 → 0.** Verified at runtime as well as statically:
`0 of 100` flat names are a `property` on `LibraryScreen`, `100 of 100` are
properties on `LibraryNotesController`.

### 1.3 The fifth spelling, in its first DUAL-RECEIVER form

Notes is the third and last kind to need `canvas_sync.py`'s dotted branch, and
the first where the dotted spelling has to resolve on TWO receivers. I
re-derived the receiver of each site rather than inheriting the brief's
framing, and the framing needed correcting:

- **`_apply_library_row_toggle` is handed the SCREEN in production, always.**
  All four production call sites are in `library_screen.py`
  (`:19903`, `:20144`, `:20258`, `:26518` at the parent), and the notes one
  sits inside `handle_library_notes_row` — a task-2 EXCLUSION, so its `self`
  is a real `LibraryScreen`. It is also driven with an `App` double from
  `test_library_multiselect_notes.py`.
- **`_sync_library_canvas` is handed a bare CONTROLLER `self` at 31 call
  sites, spread across 26 of the 185 movers**, and its notes leg reads
  `_library_notes_focus_intent_generation` off that receiver
  (`canvas_sync.py:456`'s `partial(getattr, screen, …)`). *(Corrected at
  review: "31 movers" is a call-SITE count wearing a method count's clothes.
  Re-derived by an `ast` walk of the controller class body for
  `_sync_library_canvas(self, …)` calls: **31 sites, 26 distinct enclosing
  methods, every one a mover, every one with `kind="notes"`.** The
  dual-receiver conclusion is unchanged — 26 is still "most of the cluster's
  DOM-touching half" — but a site count and a method count are different
  measurements and the smaller one is the load-bearing one here, because what
  needs a resolvable receiver is the METHOD's `self`.)*

So the dual-receiver requirement is real, but its mechanism is the inverse of
"the flat name stops resolving": on a controller the flat name KEEPS resolving
(its own shim loop), and it is the NEW dotted spelling that would have broken
there. Three changes, together:

1. `_apply_library_row_toggle`'s branch gains
   `else "_notes_state.row_selection" if kind == "notes"`.
2. `canvas_sync.py:456` becomes
   `partial(operator.attrgetter("_notes_state.focus_intent_generation"), screen)`
   — `attrgetter` re-reads BOTH hops on every call, exactly as the
   `partial(getattr, screen, "<flat>")` it replaces did, so the late-binding
   property is preserved.
3. `LibraryNotesController` gains a `_notes_state` accessor property, the
   file's own existing `_media_state`/`_prompts_state` shape. **No moved body
   spells `self._notes_state`** — they all read flat properties from the shim
   loop, which is what keeps them byte-for-byte — so this accessor exists
   solely for the shared dispatchers.

**The guard is parametrized over the receiver, and BOTH legs are
mutation-verified.**
`Tests/UI/test_library_selection_updates.py::test_notes_row_toggle_resolves_
the_dotted_state_path[screen|controller]`. The `[controller]` leg subclasses
the REAL `LibraryNotesController`, overriding only `query`/`query_one`/
`refresh` (framework-service properties a double cannot assign over), so the
`_notes_state` accessor under test is the production one.

| Mutation | `[screen]` | `[controller]` | conversations + media precedents |
|---|---|---|---|
| notes branch reverted in `canvas_sync.py` | **FAILS** — `AssertionError: fallback recompose must not fire` | passes (controller shim still resolves the flat name) | both GREEN |
| controller's `_notes_state` accessor removed | passes | **FAILS** — same fallback assertion | both GREEN |
| neither | passes | passes | both GREEN |

The second row is the point the brief asked for, measured: **a screen-only
guard passes while the controller receiver silently takes the full-screen
recompose fallback.** `canvas_sync.py` and `library_notes_controller.py` were
each restored byte-for-byte after their mutation (`git diff` re-checked).

### 1.4 A LIVE defect the same receiver analysis found in the MEDIA branch

Running the identical question against the sibling branch shows wave-7's own
dotted retarget is incomplete, and the failure mode is the silent one:

- `canvas_sync.py:437` (inside `_sync_library_canvas`'s media leg) does
  `screen._media_state.selected_media_id = media_state.selected_id`.
- `library_media_controller.py:2692` and `:2701`
  (`handle_library_media_select_all` / `handle_library_media_select_clear`,
  both movers, both reached through a real screen delegator's
  `@on(Button.Pressed)`) call `_sync_library_canvas(self, "media")` with a
  CONTROLLER `self`.
- `LibraryMediaController` has **no** `_media_state` — only
  `_media_state_accessor`. Verified at runtime:
  `hasattr(LibraryMediaController, "_media_state") is False`, against
  `hasattr(LibraryNotesController, "_media_state") is True` (the notes
  controller declares the accessor; the media one does not declare its own).
- `_sync_library_canvas`'s outer `except Exception` (`canvas_sync.py:710`)
  swallows the `AttributeError` into
  `logger.debug("Library media canvas sync failed.")` plus
  `screen.refresh(recompose=True)`.

So every media "Select all" / "Clear" press takes the whole-screen recompose
the Tier-1 targeted-sync design exists to avoid, with no exception and no red
test — recipe §3's fifth-spelling failure mode, one function along.

**Deliberately NOT fixed here.** It is a different subsystem's file and a
behaviour fix rather than a mechanical retarget, i.e. outside §4's transform
whitelist and outside this task's scope; recipe §5's rollback-not-fix-forward
posture applies to the series that owns it. **Filed to task 4** with the
four-line remedy (give `LibraryMediaController` the same `_media_state`
accessor property, and give the media guard a `[controller]` leg). The wider
lesson belongs in §3: **when a subsystem's movers forward a bare `self` into a
shared dispatcher, census the dispatcher's RECEIVERS, not just its spellings**
— the dotted retarget is only correct for the receiver it was written against.

### 1.5 Field-name prose sweep

Tokenize-based (COMMENT and STRING tokens only, so a live attribute access
cannot masquerade as prose) over the **100 deleted field names AND the 26
pruned method names**, across `tldw_chatbook/`, all of `Tests/`, `Docs/`,
`backlog/`, `scripts/` and `Helper_Scripts/`, plus a plain line scan of
`*.md`/`*.toml`/`*.tsv`/`*.txt`.

**462 raw prose occurrences across 112 files** (178 Python, 254 markdown, 30
text). The recipe's ±3-line screen-attribution filter narrows that to **87**;
excluding frozen historical records (`Docs/superpowers/{plans,specs,reviews}/`,
`backlog/tasks/`, `backlog/docs/test-health-baseline-*`,
`backlog/docs/lessons-*`, `.superpowers/`) leaves **9 candidates, every one
read**.

**6 are class-3 present-tense mis-attributions and are fixed, all
line-neutral:**

| File:line (at the landed tree) | Defect | Fix |
|---|---|---|
| `library_conversation_reader_controller.py:68` | module docstring: `_selected_conversation_id` "shared with `_media_state.selected_media_id`/`_selected_note_id` in the screen's save/restore" | → `_notes_state.selected_note_id` |
| `library_conversation_reader_controller.py:202` | `notes_focus_intent_generation_accessor: Reads ``LibraryScreen._library_notes_focus_intent_generation``` | → `LibraryScreen._notes_state.focus_intent_generation` |
| `library_conversation_reader_controller.py:216` | same shape in the `__init__` docstring | → `_notes_state.selected_note_id` |
| `library_conversations_controller.py:364` | a THIRD subsystem's docstring naming `_selected_note_id` as live screen state | → `_notes_state.selected_note_id` |
| `library_notes_state.py:11-17` | "The state PR **keeps** every original … name alive as a shim on `LibraryScreen`" | past-tensed, and pointed at the controller's surviving copy |
| `Tests/UI/test_library_shell.py:20769` | "see ``_library_note_editor_armed``" — a pointer to a name that no longer exists in any spelling | → `_notes_state.editor_armed` |

Four of the six live in files belonging to OTHER subsystems — exactly the
class the prompts series' review found, and the reason this census is
repo-wide rather than subsystem-scoped. Both
`library_conversation_reader_controller.py` (governed at 943) and
`library_conversations_controller.py` were kept LINE-NEUTRAL so no prose
correction moves a pin.

The remaining 3 are class 1/2 and were left: the successor comment in
`library_screen.py` (self-labelled), and `library_notes_state.py`'s two-line
past-tense narrative of the cutover guard's flipped red, which carries its own
commit citation (`889e12b86`).

**Re-run over the LANDED tree, the same census returns 6 candidates, not 9** —
the six class-3 defects are gone and what remains is class 1/2 by
construction: `library_notes_state.py:13` (now self-labelled past tense),
`:194`/`:196` (the cutover-guard incident narrative with its own commit
citation), `library_screen.py`'s successor comment, and this series' own two
new recipe passages. A prose census that does not go DOWN after its fixes has
not fixed anything.

**A refinement to the census's own method, recorded because the raw number is
useless without it:** a third of the 462 raw hits are FILENAME matches
(`test_library_file_notes_workspace.py` literally contains the field name
`_library_file_notes_workspace`), and 78 of the 87 attribution-filtered
candidates are frozen-record noise. Quote the filters with the count, or the
count means nothing.

---

## 2. Screen-side retargets

`tldw_chatbook/UI/Screens/library_screen.py`:

- **571** `self.<flat>` → `self._notes_state.<field>`, applied at `ast`
  attribute spans, back-to-front, single-line only (**zero refusals** — the
  tool reports and refuses rather than guessing). Line-neutral by
  construction.
- **43** `getattr(self, "<flat>", <default>)` RECEIVER fixes →
  `getattr(self._notes_state, "<field>", <default>)`, the skills-series shape:
  `_library_notes_navigation_generation`, `_library_notes_tree_branches`,
  `_library_notes_tree_request_generations`, `_library_notes_tree_navigation_
  requests`, `_library_notes_filter_navigation_generation`, `_library_notes_
  tree_filter_state`, `_library_notes_filter_generation`, `_library_notes_
  source`, `_library_file_notes_workspace`, `_library_notes_view`,
  `_library_notes_focus_intent_generation`, `_library_notes_tree_expanded_ids`,
  `_library_notes_tree_protected_folder_ids`, `_library_notes_tree_inactive_
  managed_folder_ids`, `_library_notes_navigation_status`, `_library_notes_
  tree_topology_epoch`, `_library_notes_filter` (43 sites over those names).
- **4** dynamic-dispatch string VALUES dotted — `_replace_library_reader_
  preference` and `_persist_library_reader_preference`'s `"notes"` /
  `"notes_files"` entries → `"_notes_state.reader_preferences"` /
  `"_notes_state.file_notes_reader_preferences"`. Mechanism unchanged: both
  dict paths already resolve dotted strings through `operator.attrgetter` /
  `_assign_library_reader_preferences_attribute`.
- **`on_screen_suspend`**: `_library_notes_autosave_timer` leaves the
  flat-name string loop (which collapses to a 1-line single-entry tuple) for
  its own explicit `self._notes_state.autosave_timer` stop block, exactly like
  the ingest, prompts and media timers three lines away.

**Task 1's hand-off list, re-derived rather than inherited:**

- `_replace_library_reader_preference` / `_persist_library_reader_preference`
  — 4 string values, dotted (above).
- `_close_open_library_choice_strip` — **no notes row exists.** The converged
  choice-strip dispatcher (`_library_open_choice_strip`) carries media ×2,
  prompts, skills and export; the file's own comment says the Notes Sort strip
  "keeps its own pre-existing wiring", and the census confirms it: notes'
  `sort_choices_visible` is only ever spelled as an attribute, never as a
  dispatch string. Nothing to do, stated as a verdict rather than skipped.
- `canvas_sync.py:456`'s `partial(getattr, screen, …)` — §1.3.
- `library_unavailable_navigation.py` — **16** direct `self._library_note*`
  references (task 1 said 15; re-derived at this tree it is 16), all
  retargeted. `self._library_note_session` is WIRING and stays flat.
- `library_inspection_admission.py::_SCREEN_FIELDS` — the two notes names left
  `_SCREEN_FIELDS` for their own `_NOTES_FIELDS` group on
  `screen._notes_state`, which is the file's OWN existing idiom (the
  conversations state is already a separate group with its own identity check
  in `matches()`), rather than a dotted-path variant of `getattr`/`setattr`.
  The `delattr` branch cannot fire for a dataclass field, so the group form is
  also strictly safer than the string form it replaces.
- `library_media_controller.py` / `library_conversation_reader_controller.py`
  — their three cross-subsystem accessor lambdas are built in
  `library_screen.__init__` (`:2283`, `:2816-2820`) and were retargeted with
  the other 571. Neither controller's own body was touched: their PROPERTY
  names are unchanged and their bodies are moved bodies of prior series.

**After the retarget the screen carries ZERO flat notes-name references in
any spelling.**

---

## 3. Test-side retargets — 1,031 occurrences across 26 files, four roots

| Transform | Count |
|---|---|
| attribute-path retargets (`<recv>.<flat>` → `<recv>._notes_state.<field>`) | **1,031** |
| flat kwargs nested into `_notes_state=SimpleNamespace(...)` | **61**, in **12** fixtures |
| string-spelling retargets | **4** |
| `_notes_state` seeds for non-`LibraryScreen` doubles | **6** |
| wiring-test literal pins deliberately KEPT | **9** |

**Two counts, and they are different sets — say which one each fact belongs
to** (recipe §22's own closing lesson, from the media series' two 36s). The
RETARGET set is **26 files across THREE roots**: `Tests/UI` 24,
`Tests/Live` 1, `Tests/ProductionApp` 1 — zero in `Tests/Architecture`,
`Tests/Library`, `Tests/Notes` or `Tests/Media`. The CHANGED set is **32 files
across FOUR roots**: `Tests/UI` 27, `Tests/Architecture` 3 (the wiring test
and both ratchets), `Tests/Live` 1, `Tests/ProductionApp` 1 — the six extras
being the two guard files, the wiring test, the modal inventory, the reuse
pins and the new guard's home, none of which is a field retarget.

**Erratum of record (recipe §6: a commit message is the one place a correction
cannot be applied), TWO items, both in `3d670ee38`'s own message:**

1. It says "1,031 attribute retargets across 26 files in four roots". The
   26-file retarget set spans **THREE** roots; four is the CHANGED set's
   figure. Exactly the mistake §22 warns about — two derived counts quoted
   interchangeably.
2. It says "61 flat kwargs nested into 11 `_notes_state=SimpleNamespace(...)`
   fixtures (8 mechanical, 3 hand-reordered)". The nested count is **12**
   (9 mechanical + 3 hand-reordered); 11 came from a first pass that
   miscounted the per-file call list. Re-derived from the landed tree by an
   `ast` sweep for `_notes_state=` keywords: 12 across 6 files. 61 is right.

3. It says the notes cluster hands the shared dispatchers a bare `self` "from
   31 of the 185 movers". **31 is a call-SITE count**; re-derived by an `ast`
   walk it is **31 sites across 26 distinct movers** (all `kind="notes"`).

All three are recorded here rather than amended, because
`.git-blame-ignore-revs` records that hash and amending would orphan the entry
(§10). Every live copy of all three claims is corrected in the fix-round
commit.

### 3.1 Fixture restructuring — 9 mechanical, 3 hand-reordered

61 flat kwargs became 12 nested `_notes_state=SimpleNamespace(...)` blocks
(`test_library_notes_reader.py` 4, `test_library_notes_folder_navigator.py` 3,
`test_library_media_trash.py` 2, and one each in
`test_library_honesty_accessibility.py`, `test_library_multiselect_notes.py`
and `test_library_notes_rename_propagation_t31796.py` — re-counted from the
landed tree by an `ast` sweep for `_notes_state=` keywords, not carried over),
applied at `ast.keyword` spans with interleaved comment lines preserved in
place. **9 runs were contiguous and nested mechanically. 3 were not**, each
because a NON-field entry sat inside the run, and each was hand-reordered
first with a one-line comment saying why the moved kwarg is not a state field:

| Fixture | Interloper(s) | Class |
|---|---|---|
| `test_library_notes_folder_navigator.py:227` (`_branch_screen_fake`, 23 fields) | `_library_notes_user_id`, `_capture_library_notes_focus_identity` (method stand-ins), `_repaints`, `_focus_calls` (test-local) | moved BELOW the run |
| `test_library_honesty_accessibility.py:525` (7 fields) | `_library_notes_workflow_active`, `_library_notes_focus_region` (method stand-ins), **`_library_note_session`** (one of the three WIRING attributes the state PR left on the screen) | moved ABOVE the run |
| `test_library_multiselect_notes.py:35` (`_fake`, 5 fields) | **`_library_note_dirty`** (one of the five `_library_note_session` projection `@property` members task 2 left screen-resident), `_refreshed`/`_opened`/`_flushed` (test-local) | moved ABOVE the run |

The two bolded interlopers are the interesting ones: a kwarg census that
matched on the `_library_note*` prefix rather than on the exact 100-name set
would have swept a WIRING attribute and a projection property into the state
namespace, and both would have failed silently (the fixtures are duck types).

### 3.2 The SIX `_notes_state` seeds — and the census that should have found them all

**The first two came from a static census; the third and fourth came from
paired baselines, and the fourth is the one that matters.** Recorded in the
order they were found, because the sequence is the lesson.

The census I ran first was media's own: every `<recv>._notes_state.<field>`
access in `Tests/`, split Load/Store, each receiver resolved to its binding
site. It found two doubles that are not `LibraryScreen`s. It structurally
**cannot** find the other two:

1. It enumerates receiver NAMES, so a second class binding the same name in a
   different scope is invisible — that is how
   `test_library_multiselect_notes.py`'s SECOND `App` stand-in
   (`_PaintableNotesCanvasApp`, seeded at the call site rather than in
   `__init__`) survived to become §8.1's single branch-unique failure.
2. It matches only the DOTTED shape `<recv>._notes_state.<field>`, so the 43
   `getattr(self._notes_state, "<field>", <default>)` RECEIVER fixes are
   invisible to it — the `_notes_state` node there is a `Call` argument, not
   the value of an enclosing `Attribute`. That is how
   `test_library_media_trash.py`'s `_restore_fake` survived to become three
   branch-unique failures in §8.3.

**The census that finds all four, and which is the one to run:** for every
screen method whose body now spells `self._notes_state` **in any form** (112
of them), find every UNBOUND `LibraryScreen.<name>(<first-arg>, …)` call in
`Tests/` and resolve `<first-arg>`. Run at the landed tree it returns 25
(file, receiver) pairs across 12 files — including five files this task never
edited at all (`test_library_media_reader_flow.py`,
`test_library_multiselect_media.py`, `test_library_skills_canvas.py`,
`test_review_set_walker.py`,
`test_library_open_in_library_items_t31797.py`), each driving a retargeted
screen method with a duck-typed fake. **A cleanup PR's blast radius is not its
changed-file set**: retargeting a screen method's receiver reaches every test
that calls that method unbound, whichever subsystem the test belongs to.

The four seeds:

| Site | Double | Fix | Found by |
|---|---|---|---|
| `test_library_resize_focus_gates_t23025.py` (`__init__`, 5 Stores) | a stage-signature fake class | `self._notes_state = SimpleNamespace()` before the five field writes | dotted receiver census |
| `test_library_multiselect_notes.py` `_DuplicatePlacementNotesCanvasApp` | an `App` stand-in `_apply_library_row_toggle` is driven against | `self._notes_state = SimpleNamespace(row_selection=RowSelection("notes"))` | dotted receiver census |
| `test_library_multiselect_notes.py` `_PaintableNotesCanvasApp` | ditto, seeded at the CALL SITE | same, at the call site | **§8.1 paired baseline** |
| `test_library_media_trash.py` `_restore_fake` | a MEDIA fake driving `LibraryScreen._restore_library_media_from_trash`, whose failure legs stamp `_media_state.trash_focus_authority_generation` from the NOTES focus counter | `_notes_state=SimpleNamespace(focus_intent_generation=0)` | **§8.3 paired baseline** |
| `test_library_media_trash.py` `_permanently_delete…` fake | same shape, `LibraryScreen._permanently_delete_library_media_from_trash` | same | **§8.3 paired baseline, second round** |
| `test_library_media_trash.py` `_failed_trash_restore_screen_fake` | same shape, `LibraryScreen._sync_library_media_trash_state` | same | **§8.3 paired baseline, second round** |

**Six seeds, not three, and four of the six were found by running code.** The
last two arrived on a SECOND paired round after the first media-trash seed
fixed only one of that file's three branch-unique names — which is its own
small lesson: fixing the first instance of a shape does not close the shape,
and the re-pairing is what proves it did.

### 3.2.1 The two lessons, stated separately

**(a) A receiver census that enumerates receiver NAMES under-counts when the
same name binds a different object in a different scope.** The count to trust
is per binding SITE. `test_library_multiselect_notes.py` reported one
`recv=self` Store and three `recv=app` accesses; I resolved `self` (an
`__init__`) and stopped, because both belong to classes of the same shape —
and the second class was a different class.

**(b) The dotted-shape census cannot see a `getattr` RECEIVER, and this
cleanup created 43 of them.** `getattr(self._notes_state, "<field>",
<default>)` puts the `_notes_state` node in a `Call`'s argument list, not as
the value of an enclosing `Attribute`, so a census keyed on
`<recv>._notes_state.<field>` scores it zero. **The generalized census is the
same shape §3's own sixth spelling took: walk EVERY `<recv>._notes_state`
`ast.Attribute` node, whatever encloses it.** Run that way over the landed
tree it separates cleanly — `library_screen.py` shows 573 dotted accesses, 43
`Call` receivers, 1 `Lambda` and 1 `Assign`; the 43 are exactly the shape the
narrow census missed.

**Both were caught by the paired baseline, not by a census** — which is the
argument for running the baseline before believing a census rather than after,
and the reason recipe §7's pairing is not optional evidence.

### 3.3 The four string-spelling retargets, and why two are load-bearing

| File | Site | Fix |
|---|---|---|
| `test_library_adaptive_reader_closeout.py` | `DESTINATION_CONTRACT["notes"]`'s two attribute names | → `"_notes_state.reader_preferences"` / `"_notes_state.reader_layout"`. **Notes is the EIGHTH and last destination** to take the dotted form, so every entry in that contract is dotted now and the `operator.attrgetter` reads that made the passthrough possible are no longer load-bearing for any of them |
| `test_screen_navigation.py` | `test_files_back_navigation_workspace_contract_matches_real_workspace`'s AST visitor matches its receiver by the literal string `"_library_file_notes_workspace"` | → `"file_notes_workspace"`, because the four seams it walks now spell `self._notes_state.file_notes_workspace.<name>`. Left alone the guard goes LOUDLY red at the retarget commit rather than passing vacuously, which is why it belongs in this commit rather than deferred. *(Corrected at review: this shipped as "the contract set goes EMPTY". Mutation-measured live, the stale string yields a **3-element** set — `{cancel_path_task, cancel_reload_confirmation, display}` — because the visitor's OTHER branch (`isinstance(v, ast.Name) and v.id == "workspace"`, the local-variable form) is independent of the receiver string and keeps matching. It is still an unambiguous red: the pinned contract is `{flush_pending_work, acquire_transition, cancel_reload_confirmation}`, so the assertion fails on two missing names and two extra ones. "Loud, not vacuous" holds; "empty" was an assumption I never ran.)* |
| `test_library_screen_reuse.py` | the notes timer name inside `on_screen_suspend`'s two flat-name string loops | §3.4 |

`Tests/Notes/test_notes_sync_cutover.py:82`'s `_library_notes_auto_sync_timer`
prefix string is the one string site deliberately NOT retargeted — see §6.

### 3.4 The notes autosave timer leaves the flat-name string loop

`test_library_screen_reuse.py` pins `on_screen_suspend` twice — once against a
real booted app, once against a `LibraryScreen.__new__` screen — both via a
tuple of flat timer attribute names driven through `setattr`/`getattr`. After
the shim deletion that name would arm and assert a field the hook never reads
(a VACUOUS pass, not a failure). Both loops now carry the notes timer in their
own explicit `_notes_state` block, exactly as the ingest, prompts and media
timers already do, and task 1's own seed comment was corrected to match
(it said the shim keeps the loop non-vacuous "until this series' cleanup PR
deletes it" — this is that PR). The `__new__` fixture also gains the
`_RecordingTimer` arm and the two assertions the flat name used to carry, so
the pin's arity is unchanged: the docstring's "all seven" is still seven.

### 3.5 No receiver fix was needed for a moved METHOD

Censused directly, the media §3.6 check: every `<recv>.<mover> = …` and
`monkeypatch.setattr(<recv>, "<mover>", …)` site in `Tests/`. **Eight mover
names have such a site, and all eight receivers are duck-typed
`SimpleNamespace` fakes** — `_focus_library_notes_filter_input`,
`_library_note_import_execution_active`, `_library_notes_focus_region`,
`_library_notes_mutation_fenced`, `_library_notes_operation_for_active_region`,
`_library_notes_scroll_owner`, `_restore_library_notes_focus_identity`,
`_selected_library_notes_tree_row`, across
`test_library_notes_folder_navigator.py`, `test_library_multiselect_notes.py`,
`test_library_honesty_accessibility.py` and `test_library_notes_reader.py`.
None is a stub on a REAL screen, so none needs a receiver fix, and all eight
are correspondingly in the KEEP set (each has a genuine reference).

### 3.6 Zero assertion VALUES changed — proven mechanically

The same transform (attribute rewrite + `getattr` receiver fix + contiguous
kwarg nesting) was applied to each changed `.py` file's `81bffe809` text in a
scratch copy and `ast.dump`-compared against the current file:

- **20 of the 40 changed `.py` files reproduce EXACTLY** — zero hand edits at
  all, including `test_library_file_notes_workspace.py` (87 retargets),
  `test_library_notes_reader.py` (56), `Tests/Live/test_library_notes_tree_
  paging_live.py` (22) and `library_unavailable_navigation.py` (16).
- The **20 that differ** each carry one reviewed hand edit, and their
  non-docstring `ast.Constant` multisets were diffed: **7 have ZERO literal
  deltas** (`test_library_shell.py` with its 339 retargets among them — its
  only hand edit is a prose fix), and in the other 13 every delta is a pin
  number, a name-string retarget, a new assertion message, or a literal
  belonging to the new guard. **No assertion VALUE changed anywhere.**
- `library_notes_controller.py`'s delta is the tell that the byte-for-byte
  canon held: the scratch copy (transform applied) carries `"view"`,
  `"source"`, `"mutation_in_flight"`, `"tree_selected_placement_id"` and
  `"deleted_folder_receipt"` where the real file still carries the FLAT
  spellings — i.e. the real controller was not retargeted, which is exactly
  right.

---

## 4. Shim deletion, delegator prune, dead imports

### 4.1 Shim block deleted at zero consumers

The 20-line sentinel-wrapped block is replaced by the 11-line "deleted, and
why" successor comment every prior series leaves in its place, noting that
this is the LAST of the eight. Verified at runtime: **0 of 100** flat names is
a `property` on `LibraryScreen`; **100 of 100** are properties on
`LibraryNotesController`.

### 4.2 Delegator census — 159 KEEP / 26 PRUNE (14.05%)

`ast`-based (attribute + `LibraryScreen.<name>` + quoted-string + kwarg +
bare-name + `def`), over `tldw_chatbook/` + every `Tests/` root + `Docs/` +
`backlog/` + `scripts/` + `Helper_Scripts/`, excluding only the controller
module, each name's own delegator body, and the wiring test's own pin tuple. A
`def` is never counted as a caller.

| Class | N |
|---|---|
| KEEP — §4 whitelist, unconditional (**70 `@on` + 4 `action_*`**) | **74** |
| KEEP — genuine external caller | **85** |
| PRUNE — zero references, not whitelisted | **26** |

`on_<message>` NAME-dispatched handlers: **0**. §4's third whitelist member is
inert for notes, as it was for media — and task 2 proved it in BOTH directions
(35 Message-typed decorators over 35 handler names, not one
`Message.handler_name` equal to its own method's name; and of the 113 distinct
`handler_name`s across `tldw_chatbook/Widgets/Library`, the only ones that ARE
`LibraryScreen` methods are 4 Textual builtins and the 7 prompts-owned
`on_prompt_block_editor_*`). `test_no_notes_handler_is_name_dispatched_by_
textual` keeps it proven, so a later notes widget that introduces one fails
loudly instead of being pruned silently.

**Every one of the 26 was then re-checked by a broad `git grep` over the whole
repo, all file types.** Outside the controller, each name's only hits are its
own delegator `def`+body, the wiring test's pin tuple, and prose in `Docs/`,
`backlog/`, this series' own reports, or a test docstring. No `Docs/*.py`
executable script calls any of them.

The 26 pruned:

```
_apply_library_note_saved_presentation    _move_library_notes_operation
_apply_library_notes_operation_state      _note_word_count
_defer_library_notes_settled_focus_restore _queue_library_notes_scroll_interaction
_exit_library_notes_lasting_sync          _record_library_notes_presented_focus
_focus_library_note_import_control        _record_library_notes_scroll_interaction
_focus_library_notes_lasting_control      _remember_library_notes_responsive_focus
_install_library_notes_scroll_observers   _resolve_library_note_conflict
_library_note_meta_base_line              _restore_library_notes_final_scroll
_library_note_session_is_unsafe           _restore_library_notes_scroll_after_layout
_library_note_status_line                 _restore_library_notes_scroll_offset
_library_notes_authority_root             _run_library_note_import_check
_library_notes_fallback_focus_target      _selected_library_note_handoff_payload
_library_notes_operation_is_current_and_active _settle_library_notes_final_scroll
```

Deleted by `ast` span (`def` line → `end_lineno`, plus the one blank
separator), with an assertion that none carries a decorator other than
`@staticmethod` — `_note_word_count` is the one `@staticmethod` among them, at
5 lines rather than 3. 80 lines, 26 methods, `1284 → 1258` `FunctionDef`s,
both directions of the method-name set difference otherwise empty, and the
deleted set is EQUAL to the census's prune set (asserted, not eyeballed).

**The prune is CLOSED, re-derived on the post-cleanup tree**: 159 survivors,
of which 74 are whitelist members and 85 have a genuine caller — **zero
further would-prune names**. Prune fraction **26/185 = 14.05%**, second lowest
of the program: export 4.55% < ingest 10.71% < **notes 14.05%** < media 15.71%
< skills 18.60% < collections 21.88% < prompts 28.06% < search+RAG 28.57% <
conversations 29.51%.

### 4.3 Import verification — 11 dead, 1 saved, and a census that needed running twice

Derived as a DIFFERENCE (AST-unused at the wave start `889e12b86` vs. now),
never as an absolute unused list: 41/43 names are unused at BOTH ends and
belong to prior series or to `__future__`/`_SURFACE` bookkeeping.

**The first pass returned 10 and was wrong by two.** It treated a name
appearing inside ANY string constant as "used" — a deliberate conservatism for
`TYPE_CHECKING` forward references — which hides any name that survives only
in long docstring prose. Re-run counting only annotation-shaped strings (≤80
chars) it returns **12**, the two extras being `RowSelection` (which task 1
had explicitly predicted would go dead, and which the lenient pass would have
silently dropped from the list) and `FileSave` (dead since task 2 moved
`_export_library_note`). Both were confirmed by reading: their only remaining
occurrences in `library_screen.py` are docstring prose. **A string-tolerant
unused-import census is a lower bound; run the strict variant too and
adjudicate the difference by reading.**

The `_SURFACE` check by **EXACT NAME** against
`Tests/Architecture/test_library_support_layer_surface.py` saved **1 of 12**:
`LIBRARY_NOTE_CONTENT_MAX_CHARS`, which spells the subsystem word UPPERCASE —
the same trap that nearly cost the prompts series five saves. The
alias-scoped re-export grep (wave-7's added rule: search every alias tests
import the module AS, derived by `ast` rather than assumed — here
`library_screen`, `library_screen_module`, `library_module`, `screen_module`)
returned **zero** live consumers for the other 11.

The 11 deleted, each independently confirmed live inside the controller by a
per-name occurrence count (2–13 uses each): `ConflictAction`,
`ConflictOutcomeKind`, `DestructiveAdmissionOutcomeKind`, `NoteSaveOutcomeKind`,
`build_library_note_editor_state`, `notes_autosave_status_text`,
`notes_state_shim_attr`, `reduce_notes_work_session`,
`resolve_database_note_status_channels`, `RowSelection`, `FileSave`. Ten cost
a whole line each; `FileSave` was removed in place from a shared import line
(`FileOpen` and `SelectDirectory` on it are still live). `notes_state_shim_attr`
is safe to drop because the wiring test imports it from the STATE module, not
from `library_screen` — checked, not assumed.

### 4.4 Modal inventory — 2 rows repointed, proven by construction

`Tests/UI/test_library_modal_dismissal.py` gained
`_OwnerScope(_NOTES_CONTROLLER_FILE, "LibraryNotesController")` FIRST (without
it a repointed edge is never discovered and the bidirectional assertion fails
the other way — the prompts precedent), then two of the eight notes rows were
repointed: `_export_library_note` (`FileSave`) and
`handle_library_notes_lasting_folder_requested` (`FileOpen`), the only two
notes presenters among task 2's movers. The other six
(`_push_library_note_import_picker`, `handle_library_notes_folder_{new,rename,
move,remove}`, `_choose_library_notes_placement_target`) are all task-2
exclusions, are still screen-resident, and were correctly left alone.

That test **cannot be proven green end-to-end** — it is still the TASK-31815
blocked guard, pre-RED at discovery on the unrelated skills-era
`_present_library_skills_import_choice_if_needed` constructor, which aborts
before the comparison. Proven by CONSTRUCTION instead, running the file's own
`_discover_library_modal_edges` against the notes scope alone:

```
notes scope: (_OwnerScope('tldw_chatbook/UI/Library_Modules/library_notes_controller.py',
                          'LibraryNotesController'),)
discovered: 2   declared: 2   undeclared: set()   missing: set()   EXACT MATCH: True
   _export_library_note                          -> FileSave
   handle_library_notes_lasting_folder_requested -> FileOpen
```

---

## 5. Pins — both guard files, same commit

`Tests/Architecture/test_screen_size_ratchet.py`:
`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
**32325/1284 → 32230/1258**, measured with the ratchet's own `_measure()`.

Line delta **−95**, each term measured off the tree (the diff's own hunk
arithmetic, not an estimate):

| Term | Lines |
|---|---|
| 10 dead imports, one whole line each (`FileSave` removed in place from a shared line, so it costs none) | −10 |
| 26 pruned delegators (25 × 3 + `_note_word_count`'s 5) | −80 |
| `on_screen_suspend`: 4-line tuple → 1 line (−3), plus the 7-line explicit notes block (+7) | +4 |
| 20-line shim block → its 11-line successor comment | −9 |
| **net** | **−95** |

The 571 attribute retargets, 43 `getattr` receiver fixes and 4 dotted string
values are line-neutral by construction (632 insertions / 632 deletions,
measured before any deletion landed). Method delta is exactly the 26 pruned
delegators.

`Tests/Architecture/test_library_modules_size_ratchet.py`:
`library_notes_controller.py` **5254 → 5276** (+22) — one accessor property
plus its docstring, and nothing else; no moved body was touched.
`library_conversation_reader_controller.py` stays **943** and
`library_conversations_controller.py` stays at its (pre-existing, dev-owned,
already-off-pin) row because both prose fixes were deliberately kept
line-neutral. `library_unavailable_navigation.py` stays **817** — its 16
retargets are line-neutral.

**Recipe §6's three-places-inside-the-guard check was run**: `grep`ped both
guard files for the OLD numbers (`32325`, `1284`, `5254`) before committing.
The surviving occurrences are all in task 2's own historical narrative and its
erratum, where they are correct as history.

---

## 6. Findings and hand-offs

### TASK-4 MUST **FIX** (coordinator ruling): the media dotted branch breaks its CONTROLLER receiver

§1.4 in full. `canvas_sync.py:437` assigns through
`screen._media_state.selected_media_id`; `library_media_controller.py:2692`
and `:2701` hand that dispatcher a controller `self`; `LibraryMediaController`
has no `_media_state`; the dispatcher's own `except Exception` swallows the
`AttributeError` into a full-screen recompose. Live, silent, and reached by
every media "Select all"/"Clear" press. **Not fixed here** — different
subsystem, behaviour fix, outside §4's whitelist for a cleanup PR.

**Coordinator ruling, recorded: task 4 FIXES this, it does not merely file
it** — as its own clearly-labeled behaviour-fix commit, separate from any
pure-move or doc work, because the program CLOSES after task 4 and the defect
silently fires the exact whole-screen recompose phase C exists to eliminate.
The reviewer reproduced it end-to-end at runtime, so it ships as a confirmed
defect rather than a static inference. Remedy ingredients, confirmed by the
reviewer against the live tree:

1. `LibraryMediaController` gains the accessor property, in the shape its own
   sibling already uses — the notes analogue is literally
   `@property def _media_state(self): return self._media_state_accessor()`,
   and the underlying `_media_state_accessor` **already exists on the
   instance** (`library_media_controller.py:691`), so the fix adds a property
   and nothing else.
2. `test_library_selection_updates.py`'s media guard gains a `[controller]`
   leg, parametrized exactly like this series'
   `test_notes_row_toggle_resolves_the_dotted_state_path` — subclass the REAL
   `LibraryMediaController`, override only the framework-service properties a
   double cannot assign over, and leave `_media_state` inherited so the
   accessor under test is the production one.
3. **Mutation-verify both ways**, as this series did: removing the accessor
   must red ONLY the `[controller]` leg, and reverting the media branch in
   `canvas_sync.py` must red ONLY the `[screen]` leg. Without the second
   direction the new leg could pass for the wrong reason.

The general rule this earns for §3, and which task 4 should write there:
**when a subsystem's movers forward a bare `self` into a shared dispatcher,
census the dispatcher's RECEIVERS, not just its spellings** — a dotted
retarget is only correct for the receiver it was written against, and the
wrong one fails silently behind that dispatcher's own `except`.

### The cutover guard's vacuity — final shape, re-verified at THIS tree

Three measurements, taken on the landed tree rather than inherited:

```
library_screen.py occurrences of "_library_notes_auto_sync_timer":        0
ast.Attribute census over library_screen.py for that prefix:             []
controller method names matching the guard's own prefixes:  ['_library_notes_sync_controller']
library_notes_controller.py occurrences of the timer name:                0
```



`Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_legacy_
timer_worker_or_mutating_handler` AST-walks `library_screen.py` for any
`ast.Attribute` whose name starts with `_library_notes_auto_sync_timer`. Task 1
disclosed that folding the field into the dataclass turned it green for a
reason it did not intend. **This task makes that permanent and I did not touch
it**, per the brief:

- After the shim deletion the field is spelled only as
  `self._notes_state.auto_sync_timer`, whose `ast.Attribute.attr` is
  `auto_sync_timer` — which does not start with the guard's prefix — so the
  census can never see it again, in any file, at any revision.
- The guard's own file still spells the prefix once (`:82`, its census
  constant), which is the single quoted-string hit §1.2 reports and the one
  string site deliberately left unretargeted.
- **Task 2's disclosed false positive is unchanged and re-verified at this
  tree**: pointed at `library_notes_controller.py`, the guard's method-name
  prefix census reports exactly one violation, `_library_notes_sync_controller`
  — the WIRING binding accessor, not a legacy handler. A naive retarget
  therefore goes red on a false positive. The guard's state at the landed tree
  is: **passing, permanently vacuous with respect to the field it was written
  to catch, and not naively retargetable.** The right fix is a
  notes-sync-cutover product decision; it belongs to the filing.

### Recipe

- **§8**'s subsystem table: the notes row filled in (four prefix families, the
  2-BLOCKED/10-MOVE cross-tag split, the 100-exclusion breakdown, the 14.05%
  prune fraction, the zero-`on_<message>` finding, the sixth spelling and the
  dual-receiver form of the fifth).
- **§23** added — "The notes series, as landed": per-task table, the
  six-spelling census before/after, the dual-receiver guard with its mutation
  table, the media defect, the 159/26 delegator census, the import
  verification and its two-pass lesson, the test retarget with its three
  seeds, the modal inventory, and the prose sweep.

### Hand-offs to task 4 (wave close)

1. **FIX the media `_media_state`-on-the-controller defect** (§1.4), per the
   coordinator ruling recorded above — its own behaviour-fix commit, accessor
   property + a `[controller]` leg on the media guard, mutation-verified both
   ways — and write the receiver-census lesson into §3.
2. **File the cutover-guard decision** (above), including the
   `_library_notes_sync_controller` false positive and the now-permanent
   vacuity.
3. Write the SIXTH census spelling into §3 beside the five (task 2's
   hand-off), and the **receiver** dimension of the fifth: a dotted retarget is
   only correct for the receiver it was written against.
4. Record the two-pass unused-import lesson (§4.3) — a string-tolerant census
   is a lower bound.
5. Record the per-binding-SITE receiver-census lesson (§3.2) — the miss this
   task's own paired baseline caught.
6. The three documented reds this task re-confirmed at the isolated parent are
   already in §7 or in task 1's list; **nothing new was found** (checked
   against §7 BY NAME first, per task 2's meta-lesson).
7. **Three class-3 prose survivors the ±3-line filter structurally dropped —
   hand-off, deliberately NOT fixed here.** Each names a DELETED flat field as
   a live cross-reference with no screen attribution anywhere near it, which is
   exactly the blind spot §1.5's mechanical narrowing has; widening the filter
   by hand inside a cleanup PR is how a prose sweep stops being reproducible
   (the media series' own stated reason for forward-noting its own three). Line
   numbers read off the live tree at this task's tip:

   | File:line | Prose |
   |---|---|
   | `tldw_chatbook/UI/Library_Modules/library_prompts_state.py:297` | "mirroring ``_library_note_editor_armed``" — a DIFFERENT subsystem's state module cross-referencing a deleted notes field |
   | `tldw_chatbook/UI/Library_Modules/screen_constants.py:167` | "see ``_library_note_pending_blank_gc_id``" |
   | `tldw_chatbook/UI/Library_Modules/library_unavailable_navigation.py:716` | "since the reset flips `_library_notes_view` back to \"list\"" — in a file this task DID edit, which is the sharper half of the point: the retarget fixed the code on that comment's own neighbours and left the comment naming the old spelling |

   All three are one-phrase, line-neutral fixes, and the first two live in
   files with no notes ownership at all — the same cross-subsystem class
   §1.5's six fixed defects came from. The wave-close sweep is the right home:
   per §21 it runs over BOTH name sets and is where two of the prompts series'
   six defects were found in files no task's ruled scope covered.

---

## 7. Battery

All runs `-p no:randomly`, `.venv/bin/python -m pytest`, branch venv Python
**3.14.2**. The isolated baseline worktree sat at `81bffe809` with its own
`uv venv --python 3.14`: **interpreter parity proven** (3.14.2 both),
**package parity proven** (`uv pip list` name lists `diff`ed to identical, 106
rows), and the isolated venv verified to resolve its OWN tree.

| Check | Result |
|---|---|
| **10 guard suites** — 9 wiring (collections 4, conversations 7, export 5, ingest 5, media 12, notes 15, prompts 13, search+RAG 5, skills 8) + `test_library_support_layer_surface.py` | **82 passed**, 0 failed |
| `Tests/Architecture/test_screen_size_ratchet.py` | 3 passed, **2 failed** — both documented `chat_screen.py` rows. `library_screen.py`'s own row GREEN at 32230/1258 |
| `Tests/Architecture/test_library_modules_size_ratchet.py` | **2 failed** — both documented dev-owned rows (`library_media_browse_controller` 371-pin; `library_conversations_controller` slack mirror). `library_notes_controller`'s row GREEN at 5276 |
| `Tests/UI/test_library_recompose_ratchet.py` | **1 failed** — `66 found, 63 allowed`, byte-identical to tasks 1 and 2. Task 1's own §7 entry; unchanged by this task |
| `Tests/Packaging/test_library_preimport_closure.py` | passed |
| `Tests/Performance/test_ui_ready_module_census.py` | passed — the zero-headroom 972 pin |
| `Tests/UI/test_library_screen_reuse.py` (with this task's retargets) | 3 passed, **1 failed** — `test_on_screen_suspend_stops_every_timer_in_isolation`, `AttributeError: … '_unavailable_navigation'` at `library_screen.py:7867`, i.e. the SAME dev TASK-31249 red tasks 1 and 2 both measured, failing before the timer loop is reached |
| `Tests/UI/test_library_notes_characterization.py` + `Tests/UI/test_library_inspection_admission.py` (same process) | **50 passed** total with the reuse suite |
| `Tests/UI/test_library_selection_updates.py` (the new guard's home) | **8 passed / 1 failed** — the failure is `test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`, recipe §7's documented red at :1062/:1594/:5937. Both new parametrizations pass |
| mutation verification of the new guard | both directions, both legs — §1.3's table. `canvas_sync.py` and `library_notes_controller.py` each restored byte-for-byte afterwards |
| modal-inventory construction proof | 2 discovered / 2 declared / exact match |
| **End-to-end construction probe** on a real `LibraryScreen(MagicMock())` | `controller._notes_state IS screen._notes_state` (the accessor returns the very object, not a copy); all 100 fields round-trip controller → state; **0 of 100** flat names resolve on `LibraryScreen`; all **159** kept delegators callable on the screen and all **26** pruned names absent from it; all **185** movers callable on the controller; the pruned `@staticmethod` still reachable through the controller CLASS (`LibraryNotesController._note_word_count("a b c") == 3`); and `operator.attrgetter` resolves BOTH dotted paths (`_notes_state.row_selection`, `_notes_state.focus_intent_generation`) on the screen AND on the controller |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** (after regenerating the production diagnostic inventory — see below); re-run after the fix round, still all green |
| **FRESH consolidated battery after the fix round** — the 10 guard suites + BOTH ratchets + preimport closure + ui-ready census + recompose ratchet + `test_library_selection_updates.py`, in ONE process | **144 passed / 6 failed**, 33.5s — and the 6 are EXACTLY the documented standing reds (2 × `chat_screen.py`, `library_media_browse_controller`'s 371-pin, `library_conversations_controller`'s slack mirror, the 66/63 recompose ratchet, and `test_tier1_toggle_falls_back_to_recompose_on_query_one_failure`). `library_screen.py`'s row is GREEN at 32230/1258 and `library_notes_controller.py`'s at 5276 |

**Production diagnostic inventory.** `library_screen.py`'s row changed with an
UNCHANGED count (80 → 80). Read before regenerating, per CLAUDE.md: 3 removed
/ 3 added, and all three are the SAME statements with the receiver retargeted
(`getattr(self, "_library_notes_navigation_generation", None)` →
`getattr(self._notes_state, "navigation_generation", None)`, and two
`logger.warning("Database Notes note {} failed; epoch={} …",
self._library_notes_tree_topology_epoch, …)` → `self._notes_state.tree_
topology_epoch`). Same level, same message strings, **no new interpolation of
user content, secrets, paths or URLs, and no new sink**. Regenerated with
`--write` only after that read.

**Documented reds only.** Every failure above was checked against recipe §7 BY
NAME before being called anything (task 2's meta-lesson, after that one name
was reported as new three times), and each was additionally re-confirmed at
the isolated parent where this task's own batches covered it.

---

## 8. Paired baseline — isolated worktree, sequential, parity-proven

`git worktree add <scratch>/w8t3base 81bffe809` + its own `uv venv --python
3.14` + `uv pip install -e ".[dev]"`, verified to resolve its OWN tree
(`tldw_chatbook.__file__` under the scratch path). Created once, reused,
removed at close. No same-tree checkout overlay was used at any point.

### 8.1 The three heaviest notes suites

`test_library_notes_folder_navigator.py` (342 retargets + 36 kwargs),
`test_library_multiselect_notes.py`, `test_library_notes_reader.py` (56 + 6).
Identical command and file order on both trees, run SEQUENTIALLY.

| | failed | passed | wall |
|---|---|---|---|
| branch | **15** | 146 | 431.44s |
| isolated parent `81bffe809` | **14** | 147 | 441.49s |

**14 shared, 1 branch-unique, 0 parent-unique.** The one branch-unique name
was `test_notes_export_label_holds_its_column_across_the_first_selection`
(`AttributeError: '_PaintableNotesCanvasApp' object has no attribute
'_notes_state'`) — a REAL miss, the third seed in §3.2, dispositioned by
READING rather than re-running because an `AttributeError` on a missing seed
is not timing-shaped. Fixed with one line; the file is **10 passed**
afterwards. No timing- or geometry-shaped disposition arose in this task; had
one arisen it would have been run interleaved round-robin per §7's third
level.

The 14 shared failures are the pre-existing backdrop task 1's own sweep
already characterised (13 `test_library_notes_reader.py` names in its 220-name
both-trees set). Two were additionally checked BY HAND at the parent rather
than trusted to the batch, because both are source-census shapes this task
could plausibly have broken:

- `test_unmount_invalidates_notes_authority_before_first_await` — asserts
  `inspect.getsource(LibraryScreen.on_unmount).index("_invalidate_library_
  notes_tree_for_unmount") < .index("await ")`. Measured on both trees:
  **first `"await "` at 1568, name at 3730, IDENTICAL**. (The `"await "` it
  finds is prose inside a docstring — "Say where it stopped BEFORE any await
  below" — so the guard has been measuring comment position for some time.)
  *(Corrected at review: this shipped with the supporting claim "`on_unmount`
  contains no notes flat name, so this task did not move a single character of
  it." **That is false.** `on_unmount` carries exactly one —
  `workspace = self._library_file_notes_workspace` — and this task retargeted
  it to `self._notes_state.file_notes_workspace`, a +2-character edit. The
  CONCLUSION survives on a different argument, and it is the argument that
  should have been given: the edited line begins at byte offset **6764** of
  `inspect.getsource(LibraryScreen.on_unmount)`, i.e. **after both measured
  offsets (1568 and 3730)**, so no edit past 3730 can move either — which is
  why the two numbers are identical on both trees, as measured. Recipe §3's
  own rule applies to a supporting claim exactly as it does to an exclusion's
  justification: **a wrong reason is worse than a thin one, because the next
  reader inherits it and re-derives the wrong test.** The right thin reason
  was "identical on both trees, measured"; the wrong sharp one was "untouched".)*
- `test_folder_files_reader_authority_scaffold_is_distinct` — asserts
  `set(get_args(LibraryReaderDestination))` equals a 6-member set. The type
  has **7** members on both trees (`collections` is the extra), verified in
  the isolated parent's own interpreter.

### 8.2 `test_library_shell.py -k "note"` — the single heaviest consumer

339 attribute retargets, the largest of any test file. Taken as a notes-scoped
subset rather than the whole file, per task 2's own precedent and for the
reason recipe §7 records: the whole-file run is a 171-failure
file-descriptor-exhaustion cascade that drowns the signal.

Identical command and selection on both trees, run SEQUENTIALLY, `-n 8
--dist worksteal`:

| | failed | passed | wall |
|---|---|---|---|
| branch | **171** | 83 | 499.81s |
| isolated parent `81bffe809` | **171** | 83 | 499.26s |

**171 shared, ZERO branch-unique, ZERO parent-unique — the failure sets are
byte-for-byte IDENTICAL**, and the wall times differ by 0.55s (0.1%). This is
the strongest available result for the file carrying a third of the whole
test-side change.

The 171-failure backdrop is the file-descriptor-exhaustion cascade recipe §7
characterises, and it is the SAME 171/83 task 2 measured for this file at its
own parent — i.e. the notes-scoped subset IS the cascade, not a notes-specific
effect.

### 8.3 The remaining 17 touched test files

Everything in the changed set not already covered, in one batch, identical
command and file order on both trees, `-n 8 --dist worksteal`, sequential
(`test_library_file_notes_workspace.py`, `test_screen_navigation.py`,
`test_library_adaptive_reader_closeout.py` and `test_library_modal_dismissal.py`
were split out — see §8.5 — and are covered as noted in §8.4):

| | failed | passed | wall |
|---|---|---|---|
| branch | **52** | 760 | 194.44s |
| isolated parent `81bffe809` | **47** | 765 | 196.58s |

**46 shared, 6 branch-unique, 1 parent-unique.** The bidirectional split is
§7's own signature of xdist run-to-run noise rather than a regression tied to
either tree, and the unique names concentrate in exactly two files — neither
of them notes-named, and each carrying only ONE or TWO of this task's 1,031
retargets (`test_library_media_trash.py` 2, `test_library_prompts_canvas.py`
1) — and all seven are failure-path / stale-state / focus-survival tests, the
order-sensitive shape §7 records for this suite.

Dispositioned per §7's THIRD level — each node run ALONE, n=3, with the trees
INTERLEAVED inside one load window rather than back-to-back. **The result was
not what the "bidirectional split means noise" reading predicted, and that is
the finding:**

| Node | branch | parent | verdict |
|---|---|---|---|
| `test_library_media_trash.py::test_media_trash_permanent_failure_keeps_fresh_row_and_skips_refresh` | **3/3 FAILED** | 0/3 | **REAL REGRESSION** |
| `test_library_media_trash.py::test_media_trash_restore_failure_preserves_restore_focus_not_retry` | **3/3 FAILED** | 0/3 | **REAL REGRESSION** |
| `test_library_media_trash.py::test_restore_failure_warns_keeps_row_and_clears_flag` | **3/3 FAILED** | 0/3 | **REAL REGRESSION** |
| `test_library_prompts_canvas.py::…history_restore_failures…` | 0/3 | 0/3 | xdist noise |
| `test_library_prompts_canvas.py::…pager_first_and_filter_failure_states[size0]` | 0/3 | 0/3 | xdist noise |
| `test_library_prompts_canvas.py::…stale_search_cannot_restore_an_old_filter_caret` | 1/3 | 0/3 | see below |
| `test_library_prompts_canvas.py::…page_focus_survives_loading_recompose` (parent-unique) | 0/3 | 0/3 | xdist noise |

**The three media-trash names were a REAL regression this task introduced**,
and the mechanism is §3.2's second census hole:
`LibraryScreen._restore_library_media_from_trash`,
`._permanently_delete_library_media_from_trash` and
`._sync_library_media_trash_state` — all MEDIA-named, all screen-resident, all
retargeted by this task because their failure legs stamp
`_media_state.trash_focus_authority_generation` from the NOTES focus counter —
are each driven UNBOUND with a `SimpleNamespace` fake in that file, and
`getattr(self._notes_state, "focus_intent_generation", 0)` needs a
`_notes_state` those fakes never carried. Three seeds, three comment lines,
zero assertions touched. **Had the batch been read as "6 branch-unique + 1
parent-unique ⇒ bidirectional ⇒ noise", a real three-test regression would
have shipped.** §7's third level is what separated them, and it is the reason
the rule says report RATES rather than a verdict.

Re-paired after the seeds, the six fake-driven files (§8.6) measure **branch
15 failed / 494 passed vs parent 16 / 493 — ZERO branch-unique, one
PARENT-unique**, and the full §8.3 batch re-paired measures:

| | failed | passed | wall |
|---|---|---|---|
| branch | **49** | 763 | 207.70s |
| isolated parent `81bffe809` | **48** | 764 | 193.86s |

**47 shared, 2 branch-unique, 1 parent-unique** — and the branch-unique SET is
different from the previous run's, which is itself the xdist-churn signature.
All three dispositioned interleaved:

| Node | branch | parent | verdict |
|---|---|---|---|
| `test_library_prompts_canvas.py::…stale_search_cannot_restore_an_old_filter_caret` | **0/10** | **0/10** | noise (n raised to 10 because its first pass was 1/3) |
| `test_library_prompts_canvas.py::…undo_refreshes_applied_page_and_preserves_basket` | 0/3 | 0/3 | noise |
| `test_library_canvas_sync_defects.py::test_notes_row_press_to_editor_keeps_focus_inside_the_canvas` (PARENT-unique in the batch) | **1/10** | **0/10** | flaky on both trees — it failed at the PARENT in the batch and on the BRANCH in isolation, i.e. bidirectional across two instruments, §7's own "strong independent evidence of pure run-to-run flakiness". Both rates reported rather than the convenient one |

**Zero real regressions remain.** The three that were real are fixed and
re-paired.

Three earlier-observed names are in the SHARED set and therefore already
dispositioned as pre-existing by this pairing:
`test_library_canvas_scoped_sync.py::test_notes_per_click_updates_keep_screen_
and_canvas_identity` (`NoMatches: No nodes match '#library-notes-row-0'` — the
same DOM-mount family as the 12 shared `test_library_notes_reader.py` names in
§8.1) and the two `test_library_notes_files_sync_journey.py` names. The first
of the three was additionally checked by CONSTRUCTION, because it is the one
test in the batch that presses a notes row through the canvas-sync path this
task changed: `partial(operator.attrgetter("_notes_state.focus_intent_
generation"), screen)` and `partial(getattr, screen, "_library_notes_focus_
intent_generation")` return the same value, both re-read on every call, and the
attrgetter form additionally resolves on a controller-shaped receiver.

### 8.6 The five files this task never edited, paired anyway

The unbound-fake census (§3.2) named five files that DRIVE a retargeted screen
method with a duck-typed fake but carry no retarget of their own —
`test_library_media_reader_flow.py`, `test_library_multiselect_media.py`,
`test_library_skills_canvas.py`, `test_review_set_walker.py`,
`test_library_open_in_library_items_t31797.py` — plus
`test_library_media_trash.py`, which does. Paired as their own batch, post-fix:
**branch 15 failed / 494 passed vs parent 16 / 493, ZERO branch-unique, one
parent-unique.** None of the five needed a seed; the sixth needed three. Had
this batch not been run, the media-trash regression would have been visible
only through §8.3's larger batch, where its three names sat inside a
"bidirectional ⇒ noise"-shaped split.

### 8.4 Coverage of the changed test files, stated explicitly

Of the **32** changed `Tests/` files: 3 in §8.1 (paired), 1 in §8.2 (paired),
21 in §8.3 (paired), and **7 covered by the guard battery instead**, each run
individually or in the §7 battery rather than in a paired batch — the two size
ratchets, the notes wiring test, the modal inventory (proven by construction),
`test_library_notes_characterization.py` (task 1's 3 pins),
`test_library_screen_reuse.py` (this task's own retargets) and
`test_library_selection_updates.py` (the new guard's home, itself paired in
task 2's battery and re-run with mutation verification here). 3 + 1 + 21 + 7 =
32, and the arithmetic is stated rather than implied — the wave-7 fix-round
finding.

### 8.5 Batch composition — a measured process note worth recording

The first attempt at §8.3 ran all 21 files **single-process**, and after ~75
minutes of wall clock it had reached 22%. The cause is not the retarget: these
suites' pre-existing failures are DOM-mount pollers that each burn a **30-second
timeout** before failing (`#library-notes-row-0 never mounted within 30.0s`),
so a batch's wall time is dominated by its failure count, not its test count.
Re-run with recipe §7's own prescribed `-n 8 --dist worksteal` the same batch
is tractable. **Recipe §7 says to use xdist and this task started without it;
the lesson is that the xdist prescription is not a nicety for large batches —
it is what makes a batch containing timeout-shaped pre-existing failures
finish at all.**

---

## 9. `.git-blame-ignore-revs`

TWO rows appended, each hash taken from `git rev-parse` and never transcribed:
`3d670ee38d828424a4371222c03eb03614f6f668` (the cleanup) and the fix round's
own. Each row states what its commit did and that no method body was edited.
The cleanup commit was NOT amended after the fix round — amending would orphan
the first row, which is §10's own rule and the reason both commit-message
errata (§3) are recorded here instead.

---

## 10. Deviations, errata and process notes

- **The brief's dual-receiver premise needed correcting, and the correction is
  in the evidence rather than the framing.** The brief said 31 movers hand
  *the dispatcher* a controller `self` and that a screen-only guard "passes
  vacuously". Both halves are true of `_sync_library_canvas`; neither is true
  of `_apply_library_row_toggle`, whose four production call sites are all
  screen methods. The mechanism is also the inverse of the natural reading:
  on a controller the FLAT name keeps resolving, so what the dual-receiver
  work protects against is the new DOTTED spelling failing there. Both
  statements are backed by mutation runs (§1.3), not by re-reading the brief.
  **The brief's "31" is also a call-SITE count, and I propagated it as a
  method count into seven shipped places** (the controller docstring ×3, the
  module ratchet comment, the new guard's docstring, the wiring test, and
  `canvas_sync.py`'s own comment) before re-deriving it. All seven now say 26
  movers / 31 call sites; the instance in commit `3d670ee38`'s MESSAGE is a
  third erratum of record (§3), since that hash is blame-ignored.
- **`Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py` (17
  hits) was deliberately left**, on the standing precedent that still leaves
  `screen._library_skill_editor_state` there four waves after the skills
  cleanup. Recorded as a verdict rather than skipped silently.
- **`quick_notes_section.py`'s 10 hits are a false positive** and were
  verified by reading the receiver: a Research-Workspace widget's own
  `self._selected_note_id`, no relation to `LibraryScreen`. Retargeting them
  would have been a silent cross-subsystem bug.
- **No `Docs/User_Guide/` page was touched, deliberately.** CLAUDE.md asks for
  a User-Guide update on PRs that change a screen's UI; this PR changes where
  state lives and deletes unreferenced delegators, and renders nothing
  differently. Considered and declined, consistent with all seven prior
  cleanup PRs.
- **Nothing was pushed. `progress.md` was not touched.** Staging was done from
  an explicit path list, never `git add -A` (wave-7's own erratum), and
  `git status` was checked for untracked files before each commit.
- **A harness process note, not a repo one.** This task's first §8.3 batch ran
  single-process and was abandoned at 22% after ~75 minutes; the re-run under
  `-n 8 --dist worksteal` finished the same 17 files in 3m14s. The cause is
  §8.5's 30-second DOM-mount timeouts, and the practical rule is in the recipe.
- **Four background sweeps were killed and re-run** (the abandoned
  single-process batch, its xdist re-run whose composition I then changed, and
  two waiter tasks). Every number quoted in §8 comes from a run that reached
  its own summary line; nothing is extrapolated from a progress percentage.
- The isolated baseline worktree is removed at task close.
