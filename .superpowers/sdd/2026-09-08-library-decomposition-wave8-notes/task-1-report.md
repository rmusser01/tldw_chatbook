# Wave-8 Task 1 — Notes state PR (notes series 1/N)

Branch `refactor/library-decomp-wave8-notes`, worktree
`/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`,
parent `889e12b86`.

| Commit | What |
|---|---|
| `f2b6de768` | RED/pins: `LibraryNotesState` + wiring test (RED at parent: 2 failed / 5 passed) + 3 characterization pins (green) |
| `ec7318181` | Move: 100 fields out of `__init__`, `self._notes_state`, generated shim block, `__new__`-bypass seed, ratchet re-pin 35777 → 35621 |
| `2641ff0a0` | `.git-blame-ignore-revs` row for `ec7318181` (hash via `git rev-parse`, never transcribed) |

Nothing pushed. The isolated baseline worktree was removed at task close
(`git worktree list` verified back to the three pre-existing entries).

---

## 1. Ownership analysis

### Census method

Recipe §2 script, run as a **SUBSTRING match on `"note"`** over every
`__init__`-stored `self.<attr>` of `LibraryScreen` (never a `startswith`
filter — the conversations exemplar's enumeration trap), plus the full
class-body `Assign`/`AnnAssign` scan the recipe requires.

- `__init__`-stored attributes total: **211**
- containing `"note"` (case-insensitive): **105**
- class-body `Assign`/`AnnAssign` with `"note"`: **13, all excluded by rule**
  (`LIBRARY_NOTES_*_SHORTCUTS` × 12 plus `_NOTES_WORKBENCH_FOCUS_TARGETS`) — all
  36 class-body assigns on `LibraryScreen` are ALL-UPPERCASE constant tuples,
  zero are instance state. So **0 class-body attributes move**, unlike Media's
  `_library_media_arrival_note`; the exclusion is stated in the number rather
  than hidden behind it.

The census was then **extended past the screen**, because seven subsystems'
bodies have already left this file and a screen-only census cannot see them.
Per-file reference sweep over `tldw_chatbook/UI/Library_Modules/*.py`:

| Module | Live notes-field references | Kind |
|---|---|---|
| `library_media_controller.py` | 5 (`compact`, `focus_intent_generation`, `source` read-only; `programmatic_focus_target`, `restoring_focus` read **and written**) | another subsystem |
| `library_conversation_reader_controller.py` | 1 (`focus_intent_generation`, getter-only property) | another subsystem |
| `library_unavailable_navigation.py` | 15, direct `self._library_note*` on the screen | shell/plumbing |
| `library_inspection_admission.py` | 2, by name-string in `_SCREEN_FIELDS` | shell/plumbing |
| `canvas_sync.py` | `_library_note_session` (WIRING) + `partial(getattr, screen, "_library_notes_focus_intent_generation")` | shared dispatcher |

Eight further apparent hits (`library_collections_controller`,
`library_conversations_controller`, `library_export_controller`,
`library_ingest_controller`, `library_prompts_controller`,
`library_rag_search_controller`, `library_prompts_state`, `screen_constants`,
`library_note_import_controller`) were **read** and are all false positives —
docstring prose, or a field name that is a substring of a METHOD name
(`_apply_library_notes_stage_visibility` contains `_library_notes_stage`;
`project_library_note_import_snapshot` contains `_library_note_import_snapshot`).

### Verdicts — 100 MOVE, 3 WIRING, 2 BLOCKED

**WIRING (3)** — live prior-extracted coordinator instances built with
callables that close over the screen; not data. They keep their original
`__init__` positions and are untouched:

| Attribute | Holds | Prior-extracted module |
|---|---|---|
| `_library_note_import_controller` | `LibraryNoteImportController` | `library_note_import_controller.py` |
| `_library_notes_sync_controller` | `LibraryNotesSyncController` | `library_notes_sync_controller.py` |
| `_library_note_session` | `DatabaseNoteSessionCoordinator` over `_LibraryDatabaseNoteSessionPort(run_service_call=self._run_library_service_call, …)` | `note_session_port.py` |

The plan named **four** prior-extracted Notes modules. The fourth,
`library_notes_work_session.py`, is different in kind: it exports a pure
reducer (`reduce_notes_work_session`) plus two enums, and the screen's two
work-session attributes hold **data** (a `NotesWorkSessionPhase` member and a
bool), not a live object. Both therefore **MOVE**, like any other field. The
two derived snapshot fields (`_library_note_import_snapshot`,
`_library_notes_lasting_sync_snapshot`) also MOVE — they are frozen projections
the controllers hand over, not the controllers.

**BLOCKED (2)** — `_library_notes_programmatic_focus_target` and
`_library_notes_restoring_focus`.

Both carry a notes name and Notes' own methods write them — but so does
**Media**, from bodies that already left this screen.
`library_media_controller.py` binds each with a getter **and a setter**
accessor and names them, in its own module docstring, among the *"5 shared
shell state [names] this cluster also WRITES"*, beside `_library_selected_row_id`
— the recipe's canonical ≥2-subsystems example. The screen-resident writers
confirm it from the other side: `_commit_library_media_return`,
`_sync_library_media_browse_state`, `_sync_library_media_trash_state` (media)
and `_focus_library_list_entry` (shell). They stay screen-owned.

**The line drawn — corrected at review: the cross-tagged set is TWELVE, not
five.** My first sweep enumerated only the *already-moved module bodies*
(`library_media_controller.py`, `library_conversation_reader_controller.py`,
…) and reported the 5 fields those bodies bind. That undercounts: the recipe's
own §2 script tags a field whenever ANY user method's NAME matches another
subsystem's prefix, including methods still resident on the screen. Re-run
verbatim against the parent `889e12b86`, it flags **12**:

| Field | Other-subsystem-named users | WRITTEN by them? | Verdict |
|---|---|---|---|
| `_library_notes_programmatic_focus_target` | `_commit_library_media_return`, `_sync_library_media_trash_state` (+ media controller setter accessor) | **YES** | **BLOCKED** |
| `_library_notes_restoring_focus` | `_sync_library_media_browse_state`, `_sync_library_media_trash_state` (+ media controller setter accessor) | **YES** | **BLOCKED** |
| `_library_notes_focus_intent_generation` | `_arm_library_media_return_settlement`, `_focus_library_media_grip_if_current`, `_sync_library_media_reader_layout_from_shell`, `handle_library_media_trash_open`, `handle_library_media_trash_row` (+ media & conversation-reader getter-only accessors) | no | MOVE |
| `_library_notes_resize_settling` | `_focus_library_media_grip_if_current`, `_sync_library_media_reader_layout_from_shell` | no | MOVE |
| `_library_notes_compact` | `_library_media_layout_signature` (+ media controller getter-only) | no | MOVE |
| `_library_notes_source` | `_toggle_library_media_reader_pane` (+ media controller getter-only) | no | MOVE |
| `_library_notes_view` | `_toggle_library_media_reader_pane` | no | MOVE |
| `_library_notes_work_session_phase` | `_toggle_library_media_reader_pane` | no | MOVE |
| `_library_notes_reader_preferences` | `_toggle_library_media_reader_pane` | no | MOVE |
| `_library_file_notes_reader_preferences` | `_toggle_library_media_reader_pane` | no | MOVE |
| `_library_notes_reader_layout` | `_toggle_library_media_reader_pane` | no | MOVE |
| `_library_file_notes_reader_layout` | `_toggle_library_media_reader_pane` | no | MOVE |

The seven newly-disclosed rows all moved on the same rationale as the three I
did disclose, and a mechanical write/read split over all twelve reproduces the
verdicts exactly: **the 2 with an other-subsystem-named WRITER are precisely
the 2 BLOCKED; the other 10 are read-only.** Most of those reads come from
`_toggle_library_media_reader_pane`, the generic multi-subsystem pane
dispatcher that is media-NAMED but reads four subsystems' state and stays
screen-resident (the media series' own ruling). The verdicts were right; the
disclosure was incomplete, and the incompleteness came from censusing the
wrong population — moved bodies instead of the §2 script's own tagging.

**THE RULE, stated explicitly because three waves have now relied on it
implicitly: an other-subsystem-named WRITER ⇒ BLOCKED; other-subsystem-named
READS are route-guard precedent and do not block a move.** The landed
precedent for the read side is
`_library_prompts_mutation_in_flight`: the prompts series moved it into
`LibraryPromptsState`, and three Notes-named screen methods
(`_show_library_file_notes`, `_show_library_database_notes`,
`_return_to_library_database_notes`) plus one `__init__` binding lambda read
`self._prompts_state.mutation_in_flight` to this day — 21 such call sites in
`library_screen.py` today. The media series drew the same line for its own
`view`, read by a Notes-named guard without being owned by it.

### The four explicit rulings the brief asked for

**(a) The four prior-extracted wiring instances.** Three are WIRING (above).
The fourth module's screen state (`work_session_phase`,
`work_session_activation_pending`) is data and MOVES; its by-string consumer in
`library_inspection_admission.py::_SCREEN_FIELDS` (`getattr`/`setattr` by name,
with a `delattr` branch reachable only for a genuinely-absent name) keeps
resolving through the shim for the whole state PR. Retargeting it — the same
possibly-dotted-path shape `_assign_library_reader_preferences_attribute`
already solves for four subsystems — is the cleanup PR's job. The `delattr`
branch cannot fire for a shimmed name on a real screen: `capture()` reads with
`getattr(screen, name, _ABSENT)`, which always succeeds once `_notes_state`
exists.

**(b) Every sync-engine-adjacent field.** `Notes/sync_engine.py` no longer
exists (`Tests/Notes/test_notes_sync_cutover.py::test_cutover_release_deletes_
every_legacy_writer_module` enforces its absence). The lasting-sync runtime
lives on the **app**, and the screen holds only a controller (WIRING) plus a
projected snapshot. The two publish callbacks the screen hands the controllers
— `_publish_library_note_import_snapshot` and
`_publish_library_notes_lasting_sync_snapshot` — both touch the DOM
(`_sync_library_canvas`, `_apply_library_notes_footer_context`) and are
therefore already UI-thread-only; no watcher thread writes a notes field.
Independently, a property shim is exactly as atomic as the plain attribute
store it replaces (one store, one object), so the move introduces no new
cross-thread hazard even if such a writer appeared. The eleven
`_library_notes_sync_*` panel fields have **zero** readers or writers among
`LibraryScreen`'s own methods (measured) — they are inert panel state the
cutover left behind — and MOVE without ceremony.

**(c) The file-notes workspace seams.** Workspace-owned pixels stay with the
widget: this PR moves only the five `_library_file_notes_*` **screen**
attributes (the lazily-constructed workspace instance, its factory, and the
Folder-Files reader-preferences trio). `LibraryFileNotesWorkspace` itself is
untouched, and the state module keeps its import **TYPE_CHECKING-only**,
exactly as `library_screen.py` does, so the widget stays out of the
first-paint module census.

**(d) The four-member shell family — ruled INAPPLICABLE, with evidence.**
Re-derived at this commit, the family is
`_library_pending_list_entry_focus`, `_library_pending_list_entry_focus_anchor`,
`_library_pending_list_entry_media_return` and
`_library_list_entry_focus_generation` — assigned together at
`library_screen.py:10664-10667` (`_arm_library_list_entry_focus`) and cleared
together at `:10705-10708` (`_disarm_library_list_entry_focus`), at the
pre-move tree. **There is no notes twin.** Only the media member is
subsystem-named, and none of the four contains `"note"`, so none was ever in
this wave's candidate set. The family stays screen-owned and whole; nothing in
this PR touches it, and
`test_the_four_member_list_entry_focus_family_stays_screen_owned` pins that so
a later widening of the census substring cannot capture it silently.

**(e) Every suspend/resume-adjacent field.** `on_screen_suspend`
(`library_screen.py:9389`) reaches `_library_notes_autosave_timer` through a
**flat-name string loop** (`for attr in (…): getattr` then `setattr`). Both
halves keep working through the generated shim, so this PR needs no edit there
and makes none. `Tests/UI/test_library_screen_reuse.py`'s current shape — read
first, as instructed — is what settles the timing: each of the ingest, prompts
and media timers got its explicit `self._<subsystem>_state.<timer>` block in
the commit that **DELETED** its shim, and that file's own comments say so
verbatim ("the screen's generated shim block was deleted in the X cleanup PR,
so a `getattr` on the old flat name passes VACUOUSLY"). While the shim is live,
the loop and both of that file's assertions stay **non-vacuous**; adding a
duplicate explicit block now would be a body edit outside the recipe §4
transform whitelist that buys nothing. Wave-8 task 3 owns that block and the
matching retarget of both assertions. The one same-commit action this seam DID
require — the `__new__`-bypass seed — is done (§4 below).
`_refresh_library_visit_surfaces` (the `on_screen_resume` dispatch) reads
`source`/`view`/`stage` and writes no notes field.

### Prefix families — FOUR, one more than any prior subsystem

| Family | Count | Mapping |
|---|---|---|
| `_library_notes_` (default) | 73 | `"_library_notes_" + field` |
| `_library_note_` (singular) | 21 | `"_library_note_" + field` |
| `_library_file_notes_` | 5 | `"_library_" + field` (names keep the `file_notes_` marker) |
| bare `_` | 1 (`_selected_note_id`) | `"_" + field` |

The Folder-Files family is the new wrinkle. Notes owns **two** reader
destinations — `"notes"` and `"notes_files"` in
`_replace_library_reader_preference`'s own dispatch dict — so
`reader_preferences`, `reader_layout` and `reader_persistence_locks` each exist
twice. Stripping the family marker (as every other family strips its own) would
collapse all three pairs onto one field each and silently give the Folder-Files
pane the Database-Notes pane's geometry. Keeping the marker makes that
impossible, and
`test_the_two_reader_destinations_never_collapse_onto_one_field` pins the three
pairs apart at both ends of the mapping.

---

## 2. Characterization spot-check

Existing coverage was read first: **12** dedicated
`Tests/UI/test_library_*note*.py` files (17 match the looser
`Tests/UI/*note*.py` glob — the earlier "16" was that wider count, miscited), 6 under `Tests/Library/`, a whole `Tests/Notes/` root, the canvas suites
under `Tests/Widgets/Library/`, plus `test_library_shell.py`.

Census over the **93** `@on`-decorated notes-named handlers on `LibraryScreen`
(58 selector-bound, 35 Message-typed), against ALL of `Tests/`, with exact
id/class boundaries, the derived `#<id>-N` row spelling, and a ±4-line
press/click/`post_message` window: **48 of 58** selector-bound handlers carry
automatic press evidence. Of the ten remaining, **two were overturned by
READING**, six touch no moved field, and — **corrected at review** — the last
two were overturned as well, by a census spelling I never ran:

- `handle_library_notes_sort_choice` — bound on the class
  `.library-notes-sort-choice`, pressed by the derived id
  `#library-notes-sort-oldest` (`test_library_shell.py:18785`). The
  bound-by-class/pressed-by-id shape the prompts and media series each recorded.
- `handle_library_notes_select_clear` — activated by KEYBOARD via
  `_task10_activate_with_keyboard(screen, pilot, "#library-notes-select-clear")`
  with the selection count asserted either side (`test_library_shell.py:33839`)
  — invisible to a `.press()`-anchored window.
- Six others touch **no** field this PR moves, so a pin there would prove
  nothing about the move.
- **`handle_library_notes_folder_restore` AND `handle_library_notes_folder_new`
  are both reached by unbound `LibraryScreen.<name>(fake, event)` calls** —
  `test_library_notes_reader.py:318` and `:280` respectively, each asserting the
  exact scheduled mutation tuple (`("restore_folder", …)`, `("create_folder",
  …)`); `folder_new` is additionally named in
  `test_library_modal_dismissal.py:668`. **My census never ran the
  method-NAME/unbound-call spelling at all**, even though the media series'
  own characterization census lists it explicitly ("separately grepped each
  handler's own METHOD NAME for the unbound-`LibraryScreen.<name>(fake, event)`
  call shape"). I grepped the button ids and stopped.

**The consequence, disclosed rather than tidied away: one of my three pins was
mislabelled.** `test_notes_new_folder_opens_the_folder_name_dialog` was written
up as covering a "genuine gap"; it is not one. The pin is **kept**, because what
it covers is still uncovered — the unbound call bypasses screen `@on` dispatch
entirely, so nothing else in the repo drives this handler through a real button
press, a real `Button.Pressed` dispatch and a real `app.push_screen`. But it is
a DISPATCH-path pin, not a gap pin, and the asymmetry is real: `folder_restore`
has the identical unbound-call shape and got no pin, purely because its button
does not mount on the default notes route (it needs a deleted-folder receipt).
Neither handler was ever the "genuine gap" I claimed.

Three pins landed (`Tests/UI/test_library_notes_characterization.py`):

| Test | Handler | What it actually covers | Signal |
|---|---|---|---|
| `test_notes_new_folder_opens_the_folder_name_dialog` | `handle_library_notes_folder_new` | **dispatch path**, not a gap — an unbound call at `test_library_notes_reader.py:280` already drives the body; nothing else drives a real press through screen `@on` dispatch to a real modal push | the exact modal type the press pushes (`LibraryNoteFolderNameDialog`) |
| `test_work_pane_editor_ready_arms_dirty_tracking_on_the_editor_route` | `handle_library_note_work_pane_editor_ready` | a genuine gap, re-verified: both `LibraryNoteWorkPane.EditorReady` and the handler's own method name occur in ZERO test files other than this new one | the armed flag flipping from a deliberately-disarmed start |
| `test_work_pane_editor_ready_is_ignored_off_the_editor_route` | same | same handler's route guard, which reads two moved fields | the same flag NOT flipping, same start value, only `view` differs |

No live bugs found. GREEN, 3 passed on 5 consecutive runs at the RED commit.

---

## 3. `LibraryNotesState`

`tldw_chatbook/UI/Library_Modules/library_notes_state.py`. Basename twin
confirmed and documented: `tldw_chatbook/Library/library_notes_state.py`
already exists (the Notes domain layer — `LibraryNoteDeleteReceipt`,
`LibraryNotesFocusIdentity`, `LibraryNotesOperationState`, …), which this file
imports FROM, package-qualified. Same inert collision media and prompts already
carry.

- **100 fields**, verified equal to the census MOVE set both ways (zero
  missing, zero extra).
- Defaults and comments moved **verbatim**: 83 relocated comment lines, every
  one verified byte-for-byte identical in **both directions** (no removed
  comment absent from the state module; no state-module comment without a
  removed original, apart from three clearly-labelled placeholder notes).
- **9 fields keep their original `__init__` assignment lines** (placeholder
  defaults): the workspace-factory `if`/`else` that defines a closure, the two
  reader-preferences targets of the shared 8-way tuple-unpack, the two layouts
  derived from them, the two persistence-lock dicts sharing a local
  `asyncio.Lock`, and the two controller snapshots.
- **Zero constructor arguments** — unlike Media's three. Every folded default
  is a static literal or a pure no-argument/one-literal-argument factory call:
  **81 static literals + 10 `default_factory` fields** (corrected at review; the
  earlier "86 + 5" undercounted the factories by miscounting the five mutable
  dict/`{}` displays as literals). The 10 are `row_selection`
  (`RowSelection("notes")`), `tree_expanded_ids` (`set()`),
  `tree_protected_folder_ids` and `tree_inactive_managed_folder_ids`
  (`frozenset()`), `authority_focus` (the two-key `{"database": None, "files":
  None}` display) and five plain `{}` dict displays (`tree_branches`,
  `tree_request_generations`, `tree_navigation_requests`,
  `tree_target_offsets`, `tree_status_by_slice`). 81 + 10 = 91 folded fields.
  (The state module carries 16 `default_factory` uses in total; the other 6
  belong to the placeholder KEEP fields, which are not folds.)
- Two placeholders are `None` with a widened annotation, because
  `LibraryNoteImportSnapshot` and `LibraryNotesLastingSyncSnapshot` require 12
  and 6 constructor arguments respectively (checked against the live classes),
  so no cheap side-effect-free stand-in exists.
- Imports add **no new module to the first-paint window**: every runtime import
  is one `library_screen.py` already makes at module level, and
  `LibraryFileNotesWorkspace` stays `TYPE_CHECKING`-only.

`notes_state_shim_attr()` is the single source both the screen's shim loop and
the wiring test call. Literal per-family mapping assertions ship **from birth**
(the prompts series' proof that a self-referential sweep passes even when the
shared resolver is wrong), and Notes' four families are exactly the condition
that makes that hole cheap to fall into.

---

## 4. Screen change and the bypass census

`self._notes_state = LibraryNotesState()` sits at the position of the **first
removed field** — above the Folder-Files workspace-factory `if`/`else`, which
is the earliest of the nine kept lines.

**Bypass census (recipe §3 shape 7)** — a CONTENT grep across the WHOLE
`Tests/` tree (never a `-k` name filter), for a real
`object.__new__(LibraryScreen)` / `LibraryScreen.__new__` construction
co-occurring with any of the 100 moved flat names. Ten files carry a `__new__`
spelling; four co-occur:

| File | Verdict |
|---|---|
| `Tests/UI/test_library_screen_reuse.py` | **REAL** — `setattr(screen, "_library_notes_autosave_timer", …)` on a `__new__` screen. **Seeded** with one line: `screen._notes_state = LibraryNotesState()`, zero assertions touched. |
| `Tests/Architecture/test_library_notes_wiring.py` | this task's own file; seeds `_notes_state` explicitly by construction |
| `Tests/UI/test_library_shell.py` | false positive — both `__new__` mentions are inside `wire_bypass_ingest_controller`'s own docstring prose; every screen in the file is a real `LibraryScreen(app)` |
| `Tests/UI/test_library_ingest_inline_consent.py` | false positive — `_library_notes_stage` occurs only as a substring of the method name `_apply_library_notes_stage_visibility` |

`Tests/UI/test_library_screen.py`'s `object.__new__` mention is the documented
docstring-prose false positive the recipe already records.

**Shared-seam analysis (wave-7 inheritance #3) — amended at review; my first
pass had a real hole.** The failure mode requires a **real** `LibraryScreen`
whose `__init__` never ran; a `SimpleNamespace`/`Mock` fake has no shim at all.
I enumerated the 13 **non-notes-named** screen methods that WRITE a notes field
(`_commit_library_media_return`, `request_library_reader_layout_refresh`,
`on_resize`, `refresh`, `restore_state`, `on_descendant_focus`,
`_focus_library_list_entry`, `compose_content`,
`_sync_library_media_browse_state`, `_sync_library_media_trash_state`,
`_continue_library_landing`,
`_select_library_rail_row_after_source_admission`, `_open_library_item_by_id`),
found none reachable from a bypassed fixture, and wrote that up as if it closed
the class. **It does not: restricting the writer census to NON-notes-named
methods is the hole.** A NOTES-named writer injected into another subsystem's
controller reaches a bypassed fixture exactly the same way, and one does:

- `Tests/UI/test_library_ingest_inline_consent.py:242` builds
  `object.__new__(LibraryScreen)` (`_minimal_library_screen()`);
- its `_wire_bypass_ingest_controller` helper (`:113-118`) passes
  `apply_library_notes_stage_visibility=lambda *a, **k:
  screen._apply_library_notes_stage_visibility(*a, **k)` into
  `LibraryIngestController`;
- the controller calls that binding at `library_ingest_controller.py:988` and
  `:994` (the rail auto-collapse legs);
- and `_apply_library_notes_stage_visibility` WRITES the moved
  `_library_notes_stage_applied_signature` (verified by AST over the parent
  tree).

**No live failure today** — the auto-collapse legs are not reached by any test
in that file, and the whole 7-file `__new__` battery is green apart from the
documented ingest-retry red (**423 passed, 1 failed**,
`test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_footer_
when_retry_availability_changes`, re-confirmed at the isolated parent
`889e12b86`). It is a latent seam, not a shipped bug, and it stays latent for
as long as the method remains screen-resident behind a live shim. But the
correct claim is "one notes-named writer DOES reach a bypassed fixture and
happens not to fire", not "none is reached" — and it becomes live the moment
that method moves (see the TASK-2 MUST below).

There is exactly one `LibraryScreen` subclass anywhere in `Tests/`
(`_SelectionLibraryScreen`, `test_library_prompts_canvas.py:386`) and it
inherits `__init__` unchanged.

**Other census spellings, checked and recorded for task 3 (all inert while the
shim is live):**

- *Fifth spelling (runtime f-string).* An AST walk of every `ast.JoinedStr`
  across `tldw_chatbook/` and all of `Tests/`, pattern-matched against the 105
  candidate names, returns exactly **one** real hit:
  `canvas_sync.py:227` builds `f"_library_{kind}_row_selection"`, which composes
  `_library_notes_row_selection` for `kind == "notes"`. (`:227` at `889e12b86`;
  the plan's `:219` was the figure at `5dd2e71cf`.) It resolves through the
  shim today; the notes dotted branch and its mutation-verified guard belong to
  whichever commit DELETES the shim, exactly as the conversations and media
  branches there did. The four other JoinedStr hits are `f'_{x}'` catch-alls in
  unrelated modules (MCP tool naming, `Utils.Utils`, console chat controller,
  a console inventory test).
- *Bare-quoted-string / patch-target-table spellings.* 60 sites in
  `library_screen.py`, 4 in `Library_Modules/`, 8 in `Tests/` — including
  `_replace_library_reader_preference`'s `{destination: attribute-name}` dict
  (`"notes"`, `"notes_files"`), `library_inspection_admission._SCREEN_FIELDS`,
  and `canvas_sync.py:456`'s `partial(getattr, screen, "…")`. All resolve
  through the shim; all are cleanup-PR retargets.

---

## 5. Battery

All runs `-p no:randomly`, branch venv Python **3.14.2**, isolated baseline
worktree at `889e12b86` with its own `uv venv --python 3.14` — **interpreter
parity proven** (`3.14.2` on both) and **package parity proven** (`uv pip list`
name lists `diff`ed to identical). Note for the reader: for part of this task a
**second, unrelated session was running its own pytest suites on the same
machine** (`--basetemp=/private/tmp/pr2427-*`), so wall-clock and any
timing-shaped result here sits under heavier load than usual.

| Check | Result |
|---|---|
| `Tests/Architecture/test_library_notes_wiring.py` (new) | **7 passed** (2 failed / 5 passed at the parent — the RED that was proven first) |
| 8 existing wiring suites (collections 4, conversations 7, export 5, ingest 5, media 12, prompts 13, search+RAG 5, skills 8) + mine (7) | **66 passed**, 0 failed (the earlier "105" was this battery plus `test_library_modules_size_ratchet.py`'s 41, conflated in one count) |
| `Tests/Architecture/test_screen_size_ratchet.py` | 3 passed, **2 failed** — both documented `chat_screen.py` rows |
| `Tests/Architecture/test_library_modules_size_ratchet.py` | **2 failed** — both documented dev-owned rows (`library_media_browse_controller` 371-pin; `library_conversations_controller` slack mirror) |
| `Tests/UI/test_library_recompose_ratchet.py` | **1 failed** — `66 found, 63 allowed`, **byte-identical at the isolated parent** (66/63 there too). NEW documented pre-existing red; see §6 |
| `Tests/Packaging/test_library_preimport_closure.py` | passed |
| `Tests/Performance/test_ui_ready_module_census.py` | **passed** (the zero-headroom 972 pin; the state module adds no new first-paint module — every runtime import it makes is one `library_screen.py` already made, and `LibraryFileNotesWorkspace` stays `TYPE_CHECKING`-only) |
| `Tests/UI/test_library_screen_reuse.py` | 3 passed, **1 failed** — `test_on_screen_suspend_stops_every_timer_in_isolation`, which fails at the isolated parent too, with the **same** `AttributeError: … no attribute '_unavailable_navigation'` (dev's TASK-31249 red). Before the seed it failed EARLIER, on `_notes_state`; the seed restores the parent's exact failure mode |
| the 7 other `__new__`-bypass files | **423 passed, 1 failed** — `test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_footer_when_retry_availability_changes`, documented in recipe §7 and re-confirmed at the isolated parent |
| `Tests/UI/test_library_notes_characterization.py` (new) | **3 passed**, 5 consecutive runs, at the RED commit and after the move |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** |
| End-to-end construction probe | A real `LibraryScreen(MagicMock())` was constructed and every one of the 9 kept-line fields verified to have written THROUGH the shim into `_notes_state` (including that the two persistence-lock dicts still share the SAME `library_pane_persistence_lock` object, which only the original line can produce), both snapshot placeholders replaced by their real types, all 3 WIRING and both BLOCKED attributes still plain, and a write-through round-trip |

**Documented reds only.** Every failure above is either in recipe §7's list
(verified at the parent `889e12b86` in the isolated worktree, not assumed) or is
the one NEW pre-existing red recorded in §6.

### Notes-scoped regression sweep — paired, isolated, sequential

Recipe §7's method, narrowed from `Tests/UI -k "library"` to
`Tests/UI -k "note and library"` (1,131 tests) so the whole pairing fits in
~25 minutes of wall clock instead of ~50, and so every test it runs is one
this PR could plausibly touch:

```
.venv/bin/python -m pytest Tests/UI -k "note and library" \
    -p no:randomly -q -n 8 --dist worksteal --tb=no -rf
```

Run **sequentially**, never concurrently (§7's own lesson about concurrent
runs amplifying flakiness), branch first, then the identical command in the
isolated `889e12b86` worktree with its own venv.

| Tree | Result | Wall |
|---|---|---|
| Branch (`ec7318181`), run 1 | **220 failed / 911 passed** | 744.94s |
| Isolated baseline `889e12b86` | **220 failed / 908 passed** | 764.45s |
| Branch (`ec7318181`), run 2 (with `-rf`, for the name list) | **221 failed / 910 passed** | 733.19s |

Run 1 and the baseline have the **identical failure count**, and the branch
passes exactly 3 more — the 3 new characterization tests this PR adds. Run 2
exists only because run 1's output was captured through a `tail` that
discarded the `-rf` name list; it re-measured 221/910, i.e. **one name flipped
between two runs of the SAME tree**, which is §7's own bidirectional signature
of run-to-run flakiness before any tree comparison is made.

**Name diff (branch run 2 vs baseline): 220 shared, 0 baseline-unique, 1
branch-unique.** Every one of the baseline's 220 failures is also a branch
failure. They concentrate as 171 `test_library_shell.py`, 13
`test_library_notes_reader.py`, 12 `test_library_file_notes_git_push.py`, 11
`test_library_file_notes_workspace.py` and 13 singletons across 8 other files,
and BOTH runs develop the same late burst of `F`s in the final ~10% of the
progress line — the DOM-mount-timeout cascade recipe §7 already characterizes
as file-descriptor exhaustion accumulating over a long session ("open file
descriptors grew by 274 over the test session"), not a code effect.

**The single branch-unique name, dispositioned per §7's THIRD level** (alone,
n=10, matched batch, trees **INTERLEAVED** inside one load window — it is a
paint/geometry test, exactly the shape for which sequential batches mislead):

`Tests/UI/test_library_file_notes_workspace.py::test_production_compact_folder_
files_disclosure_and_states_are_painted`

| | Branch | Isolated parent `889e12b86` |
|---|---|---|
| red, 10 interleaved rounds | **6/10** | **3/10** |

Flaky on **both** trees, with a **byte-identical** assertion on each:

```
AssertionError: assert 'Export exact copy' in 'Saved'
 +  where 'Saved' = _painted_text_in_region(TldwCli(…), Region(x=10, y=12, width=28, height=1))
 +    and   Region(…) = Button(id='file-notes-save-copy', …).region
```

That is a transient button LABEL still reading "Saved" when the assertion
samples the painted cells — a settle race in `LibraryFileNotesWorkspace`'s own
paint, a widget this PR does not touch at all (it moves only the SCREEN's
handle to it, behind a live shim). It earns an entry in recipe §7 as a **new**
documented pre-existing failure; it is not a regression. Both rates are
reported, per §7's own rule that a single outcome is not a disposition — and
the branch's 6/10 is deliberately not soft-pedalled, it is simply higher than
the parent's 3/10 on a test that is red on the parent too, with the same
assertion, in a matched interleaved batch.

**Zero real regressions.**

---

## 6. Findings for the recipe, and hand-offs

### NEW documented pre-existing failures (for recipe §7)

**1. `Tests/UI/test_library_recompose_ratchet.py::test_library_screen_whole_
screen_recompose_count_is_ratcheted`** — *"statement-level whole-screen
recompose sites: **66 found, 63 allowed**"*, byte-identically on the branch and
at an isolated worktree on `889e12b86`. Dev-side creep against a stale pin; no
Library extraction PR added a recompose site (this one moves fields only, and
the method count is unchanged at 1290). Not previously in §7's list.

**2. `Tests/UI/test_library_file_notes_workspace.py::test_production_compact_
folder_files_disclosure_and_states_are_painted`** — a paint settle race
(`assert 'Export exact copy' in 'Saved'` on
`Button(id='file-notes-save-copy')`), red on BOTH trees with the identical
assertion: **6/10 branch, 3/10 isolated parent** in a matched, interleaved
batch. Reported with both rates, per §7.

**3. `Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_legacy_
timer_worker_or_mutating_handler`** — red at the parent (see the disclosure
below); this PR turns it green for a reason the guard did not intend.

Wave 8's task 4 should add all three.

### A pre-existing red this move flips GREEN, disclosed rather than banked

`Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_legacy_
timer_worker_or_mutating_handler` AST-walks `library_screen.py` for any
`ast.Attribute` whose name starts with `_library_notes_auto_sync_timer`. At the
parent it fails with exactly one violation
(`assert ['_library_notes_auto_sync_timer:3580'] == []`) — the `__init__`
assignment. Folding that field into `LibraryNotesState` removes the only literal
spelling from that file, so the guard now **passes** — for a reason it did not
intend: the field still exists, one indirection away, and the guard's own census
cannot see a name the shim loop **composes**. That is recipe §3's fifth-spelling
problem pointed at a guard rather than at production code.

This is recorded, not banked: the state module's docstring says so, the move
commit message says so, and it is named here as the thing wave-8 task 4 should
file. Retargeting that census belongs with the notes-sync cutover work that owns
it, not with a pure field move. The field itself is unambiguously notes-owned —
**zero** readers or writers among `LibraryScreen`'s own methods — so blocking it
for a test-guard reason would have been a wrong ownership verdict dressed up as
caution.

### Hand-offs to task 2 (controller move)

- **TASK-2 MUST — seed `_notes_controller` in
  `Tests/UI/test_library_ingest_inline_consent.py` in the SAME commit that moves
  `_apply_library_notes_stage_visibility`.** That file's
  `_minimal_library_screen()` (`:242`) builds `object.__new__(LibraryScreen)`
  and its `_wire_bypass_ingest_controller` (`:113-118`) injects
  `screen._apply_library_notes_stage_visibility` into `LibraryIngestController`,
  which calls it at `library_ingest_controller.py:988`/`:994`. While the method
  is screen-resident this is inert; the instant it becomes a delegator reaching
  `self._notes_controller`, that lambda raises `AttributeError` on a screen
  whose `__init__` never ran. This is recipe §3's seventh bypass shape **at
  controller level** — the exact recurrence the ingest series already documented
  (`recipe:4263`, and `test_library_shell.py`'s own
  `wire_bypass_ingest_controller` docstring describes the same fix). One seed
  call after the bypassing construction; zero assertions touched.

- **The `canvas_sync.py` notes dotted branch belongs to the SHIM-DELETION
  commit, not the move commit — the plan's wording is wrong and this
  reconciliation is deliberate so task 2's reviewer does not score it missing.**
  The wave-8 plan's Global Constraint #2 says "add the notes branch in the move
  commit". The precedent it cites says otherwise, in `canvas_sync.py`'s own
  comment: *"(wave-7 task 3) Media is the second kind to need the dotted form"*
  — i.e. media's branch landed in the **cleanup** PR (`5dd2e71cf`), because the
  composed name `f"_library_{kind}_row_selection"` keeps resolving through the
  live screen shim for as long as that shim exists, and only stops resolving
  when the shim is deleted. Same for notes: `_library_notes_row_selection`
  resolves today and will keep resolving through task 2. Adding the branch
  earlier would be a no-op edit outside the §4 transform whitelist; adding it
  later than shim-deletion would be a silent production defect. **It lands in
  whichever commit deletes the shim (task 3), together with its
  mutation-verified guard beside the conversations and media siblings in
  `Tests/UI/test_library_selection_updates.py`.**
  The composed name is built at `canvas_sync.py:227` (at `889e12b86`; the plan's
  `:219` was the figure at `5dd2e71cf`). The census that found it (an AST walk
  of every `ast.JoinedStr` across `tldw_chatbook/` and all of `Tests/`,
  pattern-matched against the candidate names) returned exactly one real hit and
  four trivially-excluded `f'_{x}'` catch-alls.
- Notes **does** own `on_<message>`-shaped handlers, so recipe §4's third
  delegator-prune whitelist member is live for this wave (media's was inert).
  The 93 `@on`-decorated notes-named handlers split 58 selector-bound / 35
  Message-typed; several of the Message-typed ones are dispatched purely by name.
- Two BLOCKED fields mean extra named late-binding dependencies for the notes
  methods that touch `_library_notes_programmatic_focus_target` /
  `_library_notes_restoring_focus`. **Four WRITE them** —
  `_restore_library_notes_focus_identity` (both),
  `_restore_library_notes_final_scroll`, `_settle_library_notes_final_scroll`
  and `_restore_library_notes_scroll_after_layout` (`restoring_focus`) — and a
  fifth, `_queue_library_notes_scroll_interaction`, **only READS**
  `restoring_focus`, so it needs a getter-only accessor, not a setter
  (corrected at review; the earlier text said five writers).

### Hand-offs to task 4 (wave close)

- **TASK-4 MUST — amend recipe §2's literal text with the read/write
  refinement.** §2 today says, flatly: *"a field referenced by two or more
  subsystems is shared shell state by definition and is never moved by an
  extraction PR."* Read literally that would have BLOCKED all 12 cross-tagged
  notes fields, and by the same token media's own `view` and the prompts
  series' `_library_prompts_mutation_in_flight` — both of which moved, and both
  of which are load-bearing precedents today. What three waves have actually
  applied is the refinement: **an other-subsystem-named WRITER blocks a move; an
  other-subsystem-named READ is a route guard and does not.** Write it into §2
  so the fourth wave does not have to re-derive it from precedent, and record
  this series' own decision (12 cross-tagged → 2 BLOCKED / 10 MOVE, with the
  writer/reader split) in §2's per-subsystem table.

- **TASK-4 MUST — file a backlog task for the permanently-vacuous cutover
  guard.** `Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_
  legacy_timer_worker_or_mutating_handler` is red at `889e12b86` and green after
  `ec7318181` for a reason it did not intend (below). **It gets worse after task
  3, not better:** once the shim block is deleted, `auto_sync_timer` lives only
  as `self._notes_state.auto_sync_timer`, whose `ast.Attribute.attr` is
  `auto_sync_timer` — which does not start with `_library_notes_auto_sync_timer`
  — so the guard's `startswith` census can never see it again, in any file, at
  any revision. The guard becomes permanently vacuous with respect to the thing
  it was written to catch. **And the honest retarget is not obviously the right
  call either**: a census that follows the field to its new home would go RED
  again, because the legacy timer field genuinely still exists (dead, zero
  readers or writers in production, but present). Whether the right fix is
  deleting the vestigial field, retargeting the census, or retiring the guard is
  a notes-sync-cutover decision with a product component — **it belongs to the
  filing, not to this extraction series**, which is precisely why this is a
  "file it" MUST rather than a "fix it" one.

- **TASK-4 MUST — add the three new §7 rows** listed above.

### Hand-offs to task 3 (cleanup)

- **`on_screen_suspend`'s flat-name string loop** must gain an explicit
  `self._notes_state.autosave_timer` block in the SAME commit that deletes the
  shim, plus the two `Tests/UI/test_library_screen_reuse.py` assertion retargets
  (`getattr(library, "_library_notes_autosave_timer")` → `library._notes_state.
  autosave_timer`, and the `timer_attrs` seed). Exactly the ingest/prompts/media
  treatment, at exactly their timing.
- **`library_inspection_admission.py::_SCREEN_FIELDS`** holds two notes names as
  bare strings consumed by `getattr`/`setattr` (and a `delattr` branch). It needs
  the possibly-dotted-path treatment
  `_assign_library_reader_preferences_attribute` already gives four subsystems.
- **`_replace_library_reader_preference` / `_persist_library_reader_preference`
  / `_close_open_library_choice_strip`** dispatch dicts carry the `"notes"` and
  `"notes_files"` rows. Both notes destinations must be repointed, and they are
  the reason the Folder-Files fields keep their `file_notes_` marker.
- **`canvas_sync.py:456`** hands `partial(getattr, screen,
  "_library_notes_focus_intent_generation")` to the notes canvas.
- **`library_unavailable_navigation.py`** carries 15 direct
  `self._library_note*` references (shell/plumbing module, screen passed as
  `self`); **`library_media_controller.py`** and
  **`library_conversation_reader_controller.py`** carry the three read-only
  cross-subsystem accessor lambdas built in `library_screen.__init__`, which the
  cleanup PR retargets to `self._notes_state.<field>` — the landed
  `_prompts_state.mutation_in_flight` shape.
- **Bare-quoted-string / dict-of-name-string inventory, re-derived at
  `2641ff0a0`** by an exact-match AST census (every `ast.Constant` whose `str`
  value equals one of the **100 moved** flat names, over `tldw_chatbook/`,
  `Tests/`, `scripts/`, `Helper_Scripts/`): **57 in `tldw_chatbook/`** (53
  `library_screen.py`, 2 `library_inspection_admission.py`, 1 `canvas_sync.py`,
  1 `library_media_controller.py`) and **16 in `Tests/`** (7 this series' own
  new wiring file, 3 `test_library_media_wiring.py`, 2
  `test_library_adaptive_reader_closeout.py`, 2 `test_library_screen_reuse.py`,
  1 `test_notes_sync_cutover.py`, 1 `test_screen_navigation.py`).
  My pre-move figures (60/4/8) were taken over the **105 candidate** names, so
  they included the 3 WIRING and 2 BLOCKED spellings that never moved — that is
  the whole delta, not a recount. The review cited 56/25 on a third scoping
  (and 12 rather than 7 in the wiring file, which is the exact-match 7 plus the
  f-string-built `f"_library_notes_{stem}"` pairs that are not `ast.Constant`
  nodes at all). **Task 3 must re-derive at its own tree with its own stated
  method rather than inherit any of these three numbers** — a count is only true
  of a stated revision AND a stated matching rule, and this row now has three
  defensible values for exactly that reason.
- **`RowSelection` (`library_screen.py:358`) is newly DEAD** and must go on the
  dead-import list: after the move its only remaining occurrences in that file
  are two docstring mentions (`:31581`, `:33595`), with no live code use. It is
  **not** an `_SURFACE`-named import (the only `_SURFACE` import there is the
  unrelated `_LIBRARY_HELP_SURFACE_LABELS`), so the prompts series' `_SURFACE`
  re-export check does not protect it and does not flag it either — it needs the
  ordinary per-name deadness proof.
