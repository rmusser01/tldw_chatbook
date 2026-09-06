# Wave-6 Task 1 — Prompts state PR (prompts series 1/3)

Branch: `refactor/library-decomp-wave6-prompts`
Worktree: `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`
Base (wave-6 start): `e5e03846a` (`docs(sdd): wave-6 ledger open + pre-flight scan`)

| Commit | Subject |
|---|---|
| `a41cbe8c6` | `test(library): characterization + wiring pins for the prompts extraction series (RED)` |
| `f59db7c94` | `refactor(library): prompts state object + shims (prompts series 1/3)` |

Every number below was re-derived against the tree at execution time; none is
carried over from the plan or from a prior wave.

---

## 1. Pre-task measurement

| Metric | Value | How |
|---|---|---|
| `library_screen.py` lines | 41393 | `wc -l` |
| `LibraryScreen` class-body `FunctionDef`s | 1321 | `ast` walk of the class body |
| `_BUDGETS` row before | `("LibraryScreen", 41393, 1321)` | `Tests/Architecture/test_screen_size_ratchet.py:497` |
| `__init__`-stored attributes (all subsystems) | 326 | recipe §2 script |

The pre-task `_BUDGETS` row matched the true measurement exactly — no stale-pin
slack to close (unlike the collections series' 1-line gap).

---

## 2. Ownership analysis

### 2.1 Census method

The recipe §2 script, run as a **substring match on `"prompt"`** over every
`__init__`-stored `self.<attr>` (NOT a `startswith` filter on the two
`_library_prompt(s)_` prefixes) — the conversations exemplar's own "startswith
enumeration trap" (recipe §11). Result: **46 fields**.

Two completeness checks were run alongside it:

- **Class-level attributes:** a full `AnnAssign`/`Assign` scan of the
  `LibraryScreen` class body found **zero** prompts-owned class-level-only
  state attributes (only constants such as `_PROMPTS_WORKBENCH_FOCUS_TARGETS`,
  which is a focus-target table, not state).
- **Hidden prompts fields whose name lacks "prompt":** for every one of the
  other 280 `__init__` fields, the census checked whether ALL of its non-`__init__`
  consumers are prompt-named. **Zero** such fields — i.e. the skills series'
  `_selected_skill_name` analogue here (`_selected_prompt_id`) is the only
  bare-prefix field and it *does* contain "prompt".

### 2.2 Classification — 46 fields: 43 MOVE, 3 WIRING, 0 BLOCKED

`users` = number of `LibraryScreen` methods (excluding `__init__`) referencing
the field. `non-prompt users` = those whose own name lacks "prompt", tagged by
the recipe script's name-based heuristic and then **read** where the tag named
another subsystem.

| # | Field | Prefix | users | non-prompt users | Verdict |
|---|---|---|---|---|---|
| 1 | `_library_prompt_block_state` | sing | 23 | NONE | MOVE |
| 2 | `_library_prompt_browse_controller` | sing | 41 | 7 shell/plumbing | **WIRING** |
| 3 | `_library_prompt_capabilities` | sing | 6 | NONE | MOVE |
| 4 | `_library_prompt_collections_controller` | sing | 13 | 2 shell/plumbing | **WIRING** |
| 5 | `_library_prompt_conflict_snapshot` | sing | 14 | 2 shell/plumbing | MOVE |
| 6 | `_library_prompt_delete_inflight_fingerprint` | sing | 5 | NONE | MOVE |
| 7 | `_library_prompt_delete_pending_editor_prompt_id` | sing | 3 | NONE | MOVE |
| 8 | `_library_prompt_delete_pending_entries` | sing | 3 | NONE | MOVE |
| 9 | `_library_prompt_delete_pending_fingerprint` | sing | 4 | NONE | MOVE |
| 10 | `_library_prompt_delete_pending_selection_generation` | sing | 3 | NONE | MOVE |
| 11 | `_library_prompt_delete_pending_targets` | sing | 3 | NONE | MOVE |
| 12 | `_library_prompt_delete_receipt` | sing | 6 | 1 shell/plumbing | MOVE |
| 13 | `_library_prompt_detached_structured` | sing | 12 | NONE | MOVE |
| 14 | `_library_prompt_detail` | sing | 22 | 2 shell/plumbing | MOVE |
| 15 | `_library_prompt_detail_error` | sing | 7 | NONE | MOVE |
| 16 | `_library_prompt_detail_generation` | sing | 3 | NONE | MOVE |
| 17 | `_library_prompt_detail_loading` | sing | 5 | NONE | MOVE |
| 18 | `_library_prompt_detail_retryable` | sing | 7 | NONE | MOVE |
| 19 | `_library_prompt_detail_selected_name` | sing | 5 | NONE | MOVE |
| 20 | `_library_prompt_dirty` | sing | 25 | 2 shell/plumbing | MOVE |
| 21 | `_library_prompt_editor_armed` | sing | 12 | NONE | MOVE |
| 22 | `_library_prompt_editor_mode` | sing | 3 | NONE | MOVE (entangled line kept) |
| 23 | `_library_prompt_history_controller` | sing | 15 | NONE | **WIRING** |
| 24 | `_library_prompt_include_starter_content` | sing | 6 | NONE | MOVE |
| 25 | `_library_prompt_loaded_id` | sing | 8 | NONE | MOVE |
| 26 | `_library_prompt_mutation_disabled_states` | sing | 3 | 1 shell/plumbing | MOVE |
| 27 | `_library_prompt_mutation_generation` | sing | 4 | NONE | MOVE |
| 28 | `_library_prompt_mutation_status` | sing | 8 | 1 shell/plumbing | MOVE |
| 29 | `_library_prompt_name_in_use` | sing | 3 | NONE | MOVE |
| 30 | `_library_prompt_original_name` | sing | 7 | NONE | MOVE |
| 31 | `_library_prompt_select_mode` | sing | 11 | 1 shell/plumbing | MOVE |
| 32 | `_library_prompt_selection` | sing | 10 | NONE | MOVE |
| 33 | `_library_prompt_status` | sing | 11 | NONE | MOVE |
| 34 | `_library_prompt_version` | sing | 9 | NONE | MOVE |
| 35 | `_library_prompts_debounce_timer` | plur | 3 | NONE | MOVE |
| 36 | `_library_prompts_filter_cursor_context` | plur | 3 | NONE | MOVE |
| 37 | `_library_prompts_import_open` | plur | 5 | 1 shell/plumbing | MOVE |
| 38 | `_library_prompts_import_path` | plur | 8 | 1 shell/plumbing | MOVE |
| 39 | `_library_prompts_import_status` | plur | 5 | 1 shell/plumbing | MOVE |
| 40 | `_library_prompts_mutation_in_flight` | plur | 94 | 13 shell/plumbing + **3 Notes-named** | MOVE (see §2.3) |
| 41 | `_library_prompts_reader_layout` | plur | 3 | 1 shell/plumbing + **1 Media-named** | MOVE (see §2.3), entangled line kept |
| 42 | `_library_prompts_reader_persistence_locks` | plur | 1 | 1 shell/plumbing | MOVE, entangled line kept |
| 43 | `_library_prompts_reader_preferences` | plur | 2 | 1 shell/plumbing | MOVE, entangled line kept |
| 44 | `_library_prompts_sort_choices_visible` | plur | 6 | 2 shell/plumbing | MOVE |
| 45 | `_library_prompts_view` | plur | 30 | 9 shell/plumbing | MOVE |
| 46 | `_selected_prompt_id` | bare | 27 | 5 shell/plumbing | MOVE |

**Totals (re-derived): 46 = 43 MOVE + 3 WIRING + 0 BLOCKED.**

**WIRING (3)** — each holds a live controller instance built with lambdas
closing over the screen; the `_conversation_reader_controller` /
`_library_collections_capture_controller` / `_library_skill_import_coordinator`
precedent. Each stays a plain `LibraryScreen` attribute at its original
`__init__` position, untouched: `_library_prompt_history_controller`
(`LibraryPromptHistoryController`), `_library_prompt_browse_controller`
(`LibraryPromptBrowseController` — the prior-extracted browse wiring the plan
named in advance, in `library_prompt_browse_controller.py`, untouched),
`_library_prompt_collections_controller`
(`LibraryPromptCollectionsController`).

### 2.3 The two cross-subsystem readers (the ≥2-subsystems rule, applied by body)

The recipe's own caveat is that the script's tags are name-based, not
body-based, so both flagged fields were resolved by **reading every hitting
body**, not by the tag.

**`_library_prompts_mutation_in_flight` — MOVE.**
Load/store split, re-derived: **4 stores, 93 loads** (94 distinct user methods).
All 4 writers are prompt-named: `_delete_library_prompts`,
`_settle_library_prompt_delete`, `_undo_library_prompt_delete`,
`handle_library_prompt_delete_undo` — i.e. the prompt delete/undo flow, which
is what the flag *means* (`TASK-15101 / ADR-055`, its own comment). Of the 93
readers, 77 are prompt-named; the other 16 are:

- 13 shell/plumbing navigation guards: `apply_navigation_context`,
  `_apply_navigation_context_after_flush`,
  `_apply_navigation_context_after_source_admission`,
  `_apply_navigation_context_state`, `acquire_navigation_transition`,
  `flush_pending_work`, `_open_pending_library_source`, `_open_library_item_by_id`,
  `handle_library_rail_row`, `_select_library_rail_row`,
  `_select_library_rail_row_after_source_admission`, `compose_content`,
  `_library_emergency_return_eligibility`.
- 3 Notes-named: `_show_library_file_notes` (`library_screen.py:22695`),
  `_show_library_database_notes` (`:22809`),
  `_return_to_library_database_notes` (`:22870`). All three were read: each
  uses the **identical** read-only shape `if self._library_prompts_mutation_in_flight: return`
  as the thirteen shell guards — "don't leave/enter a Notes surface while a
  Prompt mutation is in flight". No Notes method writes it, and Notes has its
  own separate dirty/flush guards (`_library_note_dirty`,
  `_flush_library_note_save`) on the very next lines.

Precedent check (run mechanically over the six already-landed state objects,
looking for cross-subsystem-named consumers of an already-moved field): the
search+RAG series already moved `query` past exactly this reader —
`self._rag_search_state.query` is read by `_show_library_file_notes`
(`library_screen.py:22743`). Verdict: MOVE, recorded, not BLOCKED.

**`_library_prompts_reader_layout` — MOVE.** Its one cross-subsystem reader,
`_toggle_library_media_reader_pane`, is the generic multi-subsystem pane
dispatcher the collections series already documented as a name-only false
positive; the same mechanical scan confirms `_collections_state.reader_layout`,
`_conversations_state.reader_layout` and `_skills_state.reader_layout` are ALL
already read by that same method post-move.

### 2.4 Three prefix families

| Prefix | Count | Field names |
|---|---|---|
| `_library_prompt_` (singular, default) | 31 | the editor/detail/delete-batch cluster |
| `_library_prompts_` (plural) | 11 | `debounce_timer`, `filter_cursor_context`, `view`, `sort_choices_visible`, `mutation_in_flight`, `import_open`, `import_path`, `import_status`, `reader_preferences`, `reader_layout`, `reader_persistence_locks` |
| `_` (bare) | 1 | `selected_prompt_id` |

43 fields → 43 distinct flat names, **zero collisions** across the three
prefixes (verified programmatically: no `X` exists as both
`_library_prompt_X` and `_library_prompts_X`). Resolved by the single-source
`prompt_state_shim_attr()` — the skills series' `skill_state_shim_attr()` shape.

### 2.5 Basename-collision handling (package-qualified census)

`tldw_chatbook/Library/library_prompts_state.py` (92 KB, the Prompts **domain**
layer) already existed. A repo-wide grep for `library_prompts_state`
(excluding the new module's own file and excluding `_build_library_prompts_state`)
finds, **within `*.py` files** (see §11 for the 24th mention, in a non-Python
snapshot artifact), **23 mentions: 20 import statements + 3 docstring-prose
mentions**. Of the 20 imports, **18 target the DOMAIN module and 2 the new UI
module — all 20 fully package-qualified, zero bare/ambiguous**:

- domain (18): `UI/Library_Modules/prompt_history.py:17`,
  `prompt_history_region.py:16`, `prompt_collections.py:12`,
  `prompt_collection_manager_modal.py:19`,
  `library_prompt_browse_controller.py:12`, `UI/Screens/library_screen.py:266`,
  `UI/Console_Modules/prompts.py:126`,
  `Widgets/Library/library_prompts_canvas.py:21`, and 10 test imports across
  `Tests/UI/test_library_prompt_collections.py`, `test_library_choice_strips.py`
  (×3), `test_library_modal_dismissal.py`,
  `test_library_prompt_browse_controller.py`, `test_library_prompts_canvas.py`,
  `test_library_shell.py`, `Tests/Library/test_library_prompts_state.py` (×2).
- new UI module (2): `UI/Screens/library_screen.py:539` and
  `Tests/Architecture/test_library_prompts_wiring.py:28`.
- prose (3): `Prompt_Management/prompt_markdown_export.py:76` and two lines of
  the new wiring test's own docstring.

The new UI state module imports the domain module by fully-qualified relative
path (`from ...Library.library_prompts_state import ...`), so the shared
basename is inert — the same pair already exists for six other subsystems.

`_build_library_prompts_state` — **8 production references** (7 call sites +
its own `def` at `library_screen.py:18684`) and **9 test references**
(including a subclass override + `super()` call in
`Tests/UI/test_library_prompts_canvas.py:387-388`) — is a SCREEN METHOD
returning the domain `PromptsListState`; it is **not** part of this extraction
and was not touched.

---

## 3. Characterization spot-check

**48** `@on`-decorated `LibraryScreen` methods touch at least one moved prompt
field; **44** of them are prompt-named (the other 4 are the shell/plumbing
readers from §2.3: `_toggle_library_media_reader_pane`,
`_show_library_file_notes`, `_show_library_database_notes`,
`handle_library_rail_row`).

Method: for each of the 44 handlers' `@on` selector(s) (or message class),
a content grep across **all of `Tests/`** — explicitly including the four extra
prompt roots — followed by a **read** of every hit for a real
press/click/value-set/keyboard-activation on a **real `LibraryScreen`** (a
standalone `_CanvasHost` test cannot dispatch a SCREEN `@on` handler, and a raw
`LibraryScreen.handler(fake, event)` call exercises the body but never the
selector dispatch).

**The four extra roots carry zero `LibraryScreen` consumers between them**:
`Tests/Prompt_Management/` (18 files), `Tests/Prompt_Studio/` (2),
`Tests/Internal_Prompts/` (13), `Tests/Prompts_DB/` (9) — the only
`library_screen` mention in any of them is one docstring line in
`Tests/Prompts_DB/test_prompts_db_legacy.py:367`. Their `_library_prompt*`
matches are service/DB method names (`list_library_prompts_page`,
`_seed_library_prompt`, …), not screen attributes. So: no existing coverage
found there, no test-side retargets owed there for task 3, and no bypass
fixtures there.

**Result: 41 of 44 genuinely covered, 3 genuine gaps.**

> **Correction (post-review).** This section's first draft said *40 of 44,
> 4 gaps*. It wrongly counted `handle_library_prompt_discard` as a gap. The
> review pass found the coverage I missed —
> `Tests/UI/test_library_prompts_canvas.py::test_library_prompt_compatibility_editor_discard_returns_to_current_list`
> (lines 10310-10404) dirties a real editor and presses the real button
> (`discard.press()`, line 10380) on a real `LibraryScreen` — and proved it is
> load-bearing by **mutation**: no-oping `_reset_library_prompt_editor_state`
> makes that pre-existing test fail. Credit to the review for both the find
> and the mutation evidence.
>
> **Why my census missed it** (the reusable lesson): my press-detection pass
> looked for `.press()` within a few lines of the *selector string*. Here the
> selector is at line 10358 (`discard = screen.query_one("#library-prompt-discard", Button)`)
> and the press lands 22 lines later, at line 10380, through the local variable
> `discard`. A
> proximity window anchored on the selector cannot see a press made through a
> variable bound earlier — the collections series' "same-line-only grep
> undercounts coverage" trap, one indirection further out. **A future
> subsystem's spot-check should resolve the variable, not just the selector's
> neighbourhood** — or, cheaper, treat every "uncovered" verdict as a
> hypothesis to be killed by reading the whole enclosing test, which is what
> the review did.
>
> The new pin is **kept** — it is deepened coverage, not new coverage, and it
> asserts strictly more than the pre-existing test: the editor projection is
> fully torn down (`_library_prompt_detail` and `_library_prompt_block_state`
> both back to `None` — two moved fields) and the discarded edit never reached
> the database. Both the pin's own docstring and the characterization module
> docstring now say so explicitly, so nobody re-derives it as a gap.

Three of the 41 were near-misses a same-line grep reports as uncovered — each
resolved by reading, not by the grep (the collections series' own
"same-line-only grep undercounts coverage" trap):

| Handler | Selector | Why the grep was wrong |
|---|---|---|
| `handle_library_prompts_empty_new` | `#library-prompts-empty-new` | Activated by a focused-Button `pilot.press("enter")` in `Tests/UI/test_library_shell.py::test_library_paged_empty_recovery_is_painted_and_keyboard_reachable[prompts]` (`:11593`), which then waits for `#library-prompt-name` to mount |
| `handle_library_prompts_empty_clear_filter` | `#library-prompts-empty-clear-filter` | Same test (`:11566`), same keyboard activation |
| `handle_library_prompts_sort_choice` | `.library-prompts-sort-choice` | Bound by CLASS, pressed by ID: `screen.query_one("#library-prompts-sort-name", Button).press()` in `Tests/UI/test_library_prompts_reader.py:579` (a real screen; the test then asserts a real browse request settles) |

**The 3 genuine gaps**, plus **1 deepened** handler, all pinned in
`Tests/UI/test_library_prompts_characterization.py` (4 tests, confirmed
**PASSING pre-change** — 4 passed at `a41cbe8c6`, before any screen edit):

| Handler | `@on` | Prior state | Classification |
|---|---|---|---|
| `handle_library_prompts_import_path_changed` | `Input.Changed, "#library-prompts-import-path"` | The field was only ever asserted MOUNTED; no test typed into it on a real screen | **GAP** |
| `handle_library_prompts_import_path_submitted` | `Input.Submitted, "#library-prompts-import-path"` | Never submitted anywhere | **GAP** |
| `handle_library_prompts_import_run` | `Button.Pressed, "#library-prompts-import-run"` | Only queried for presence/parent, on a standalone canvas host | **GAP** |
| `handle_library_prompt_discard` | `Button.Pressed, "#library-prompt-discard"` | **Already covered** by `test_library_prompt_compatibility_editor_discard_returns_to_current_list` (browse/refresh/focus side of the exit) | **DEEPENED**, not a gap — the new pin adds the editor-projection teardown (`_library_prompt_detail`/`_library_prompt_block_state` → `None`) and the DB-row-unchanged assertion |

**No live bugs found** — the 3 gaps are coverage gaps, not behavior bugs, and
the 4th test deepens existing coverage. Each pin asserts SCREEN-owned state the
handler mutates, so it stays meaningful after the shims and after task 3's
retarget.

---

## 4. `LibraryPromptsState` + shims

`tldw_chatbook/UI/Library_Modules/library_prompts_state.py`:
`PROMPTS_PLURAL_STATE_FIELDS` (11), `PROMPT_UNPREFIXED_STATE_FIELDS` (1),
`prompt_state_shim_attr()` (single-source three-way mapping), and the
`@dataclass LibraryPromptsState` with all 43 fields.

**Byte-for-byte canon, verified mechanically, not by eye:** every one of the
**23 comment lines** removed from `__init__` by this move appears verbatim
(after strip) in the new module — checked by diffing the commit's own removed
`#` lines against the module's line set: **0 not found**.

**Construction position.** `self._prompts_state = LibraryPromptsState()` is
constructed immediately after `self._skills_state = LibrarySkillsState()`
(`library_screen.py:2208`), i.e. BEFORE the shared reader-preferences
tuple-unpack — the collections/skills forced-early-construction shape. Four
fields' original `__init__` lines run after that point and are therefore left
**completely untouched**, writing through the newly installed shim:

| Field | Original line | Why it cannot fold |
|---|---|---|
| `reader_preferences` | tuple-unpack, 8 shared targets | shared `_load_library_reader_preference_snapshot()` call |
| `reader_layout` | `resolve_adaptive_reader_layout(0, …reader_preferences, LIBRARY_PROMPTS_READER_PROFILE)` | derived from the above |
| `reader_persistence_locks` | `{"library": library_pane_persistence_lock, "items": asyncio.Lock()}` | shares a local lock with 5 other subsystems |
| `editor_mode` | `coerce_prompt_editor_mode(library_config.get("prompt_editor_mode") …)` | reads live `library_config` |

The other 39 original lines are deleted outright: 37 static literals plus two
pure no-argument factory calls folded to `default_factory`
(`PromptSelectionBasket`, `local_prompt_capabilities` — both read only module
constants and have no side effects, so evaluating them at the earlier
construction point is behaviorally transparent; the ingest series' own
`LibraryIngestFormState()`/`threading.Lock()` fold).

**Behavioral proof of the fold, base vs branch:** a real
`LibraryScreen(MagicMock())` was constructed on BOTH trees and all 43 flat
attribute values dumped and diffed. **All 43 identical.** (The only textual
difference in the first pass was a `frozenset` repr ordering inside
`_library_prompt_capabilities` — pure `PYTHONHASHSEED` artifact, not a value
difference; re-run with `PYTHONHASHSEED=0` on both sides to confirm.)

**Shim block** (sentinel-wrapped, module end): the established generated loop
with `_n=_lps_field.name` default-argument closure binding on **both** the
getter and the setter, resolving the flat name through `prompt_state_shim_attr()`
rather than a second copy of the branch.

---

## 5. Bypass-fixture sweep (recipe §3's SEVENTH shape)

Census: a **content grep across ALL of `Tests/`** (never a `-k` name filter,
per that shape's own filter-blindness lesson) for a file containing a `__new__`
construction AND any of the 43 moved flat names.

| File | Real `object.__new__(LibraryScreen)` / `LibraryScreen.__new__`? | Moved fields present | Verdict |
|---|---|---|---|
| `Tests/UI/test_library_screen_reuse.py` | yes (`:216`) | `_library_prompts_debounce_timer` | **REAL HIT — fixed** |
| `Tests/UI/test_library_shell.py` | no — the only `__new__` strings are inside `wire_bypass_ingest_controller`'s own docstring prose (`:24223`, `:24226`); every screen it builds is a real `LibraryScreen(app)` | 7 | false positive, confirmed by reading |

The real hit failed exactly as predicted the moment the shim installed:

```
E   AttributeError: 'LibraryScreen' object has no attribute '_prompts_state'
tldw_chatbook/UI/Screens/library_screen.py:41354: AttributeError
```

Fixed in the SAME commit as the shim (not deferrable): one seeding line
(`screen._prompts_state = LibraryPromptsState()`) inserted immediately after
the bypassing construction, mirroring the `_ingest_state` seed already in that
fixture. Zero assertions touched. `Tests/UI/test_library_screen_reuse.py`:
**4 passed**.

Additional safety net for the *indirect* form of this shape (a bypassed screen
calling a method that WRITES a prompt field): every ingest-owned
`object.__new__` file was RUN — `test_library_ingest_canvas.py`,
`test_library_ingest_inline_consent.py`, `test_library_ingest_retry_last.py`,
`test_parakeet_v2_install_ui.py`, `Tests/App/test_submit_library_ingest_job.py`,
`Tests/integration/test_library_ingest_flow.py` → **390 passed, 1 failed**, the
one failure being the recipe §7-documented pre-existing
`test_registry_ticks_only_reflow_footer_when_retry_availability_changes`.

Non-hazards, checked and dismissed: class-level patches of a moved name
(`monkeypatch.setattr(LibraryScreen, "_library_prompt…")`) — **zero** in the
repo; and `hasattr`/`getattr(screen, "<flat>", default)` on a bypassed screen,
which returns the same default before and after (the property getter's
`AttributeError` is swallowed identically).

---

## 6. Quoted-string / dynamic-dispatch census (recorded for task 3)

All 14 sites where a moved flat name appears as a **string literal**. None is a
state-PR hazard (a `getattr`/`setattr` by name resolves the property
identically); all are task-3 retarget targets, and each `getattr(self, "<flat>", …)`
will need the skills series' **receiver** fix, not just a string swap.

| File:line | Shape |
|---|---|
| `library_screen.py:7341`, `:7485` | `"prompts": "_library_prompts_reader_preferences"` (the two shared reader-preference dispatch dicts, already generic `operator.attrgetter`/`_assign_…` passthroughs) |
| `library_screen.py:9541` | `"_library_prompts_debounce_timer"` in the suspend-hook timer-attr tuple |
| `library_screen.py:10711`, `:10742` | `getattr(self, "_library_prompts_view", "list")` — **receiver fix needed** |
| `library_screen.py:27884`, `:29278` | `getattr(self, "_library_prompt_block_state", None)` — **receiver fix needed** |
| `library_screen.py:30002`, `:30041` | choice-strip visibility/canvas-kind dicts keyed on `"_library_prompts_sort_choices_visible"` |
| `Tests/UI/test_library_adaptive_reader_closeout.py:102-103` | `DESTINATION_CONTRACT` dotted-vs-flat fixture table (the 5th subsystem in a row to need this entry) |
| `Tests/UI/test_screen_navigation.py:3253` | `setattr(screen, "_library_prompts_view", "list")` on a real screen |
| `Tests/UI/test_library_screen_reuse.py:109`, `:221` | timer-attr tuples |

---

## 7. RED-at-parent proof

`Tests/Architecture/test_library_prompts_wiring.py` landed in `a41cbe8c6`, a
commit in which `library_screen.py` is **untouched** (`git show --stat`:
3 files, all additions — the state module, the wiring test, the
characterization file).

Run at the parent commit, with the base tree checked out over the worktree
(`git checkout a41cbe8c6 -- tldw_chatbook Tests`):

```
FAILED Tests/Architecture/test_library_prompts_wiring.py::test_state_object_fields_match_the_shim_surface
FAILED Tests/Architecture/test_library_prompts_wiring.py::test_every_shim_reads_and_writes_its_own_state_field
2 failed, 3 passed, 94 warnings in 1.12s
```

with, verbatim:

```
E  AssertionError: no screen shim property found for: ['_library_prompt_block_state', …]   (all 43)
E  AttributeError: 'LibraryScreen' object has no attribute '_library_prompt_block_state'
```

The 3 that pass at the parent are the ones that do not depend on the screen
edit: the two prefix-set drift guards and the wiring-fields-stay-off-the-state-object
guard.

After the screen commit (`f59db7c94`): **5 passed**.

---

## 8. Battery

| Suite | Result |
|---|---|
| `Tests/Architecture/test_library_prompts_wiring.py` (new) | **5 passed** |
| the 6 existing `test_library_*_wiring.py` + `test_library_support_layer_surface.py` | **47 passed** (all 8 files together) |
| `Tests/Architecture/test_screen_size_ratchet.py` | 3 passed, **2 failed** — the two documented pre-existing `chat_screen.py` rows (recipe §7); the `library_screen.py` row is GREEN at the new pin |
| `Tests/Architecture/test_library_modules_size_ratchet.py` | 30 passed, **1 failed** — the documented pre-existing `library_media_browse_controller.py` row; its glob is `*_controller.py`, so a state module needs no row |
| `Tests/UI/test_library_recompose_ratchet.py` + `Tests/Packaging/test_library_preimport_closure.py` + `Tests/Performance/test_ui_ready_module_census.py` | **11 passed** |
| `Tests/UI/test_library_prompts_characterization.py` (new) | **4 passed** |
| `Tests/UI/test_library_screen_reuse.py` (bypass seed) | **4 passed** |
| ingest `object.__new__` files (6 files) | 390 passed, 1 documented pre-existing failure |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** |
| full `Tests/Architecture/` (fix round 1) | 557 passed, 1 skipped, **18 failed — all 18 proven pre-existing**, see below |

**Full `Tests/Architecture/` paired baseline (fix round 1).** The whole
directory was run rather than only the prompts-relevant files, because three of
its guards are package-wide AST censuses that a 34-line shift in
`library_screen.py` could in principle disturb — `test_timer_path_static_update_inventory.py`
keys `CLASSIFIED_SITES` on `(file, line)` tuples, exactly the shape a pure move
breaks. Result: **18 failed / 557 passed / 1 skipped (358 s)**.

- **3** are the pre-authorized documented rows (2 `chat_screen.py` +
  `library_media_browse_controller.py`), proven unaffected structurally: this
  task's entire diff touches neither those files nor their budget rows.
- **15** are everything else. Rather than infer from their names, the identical
  15-name set was re-run **at the base commit** (`a41cbe8c6`, overlay applied
  with no other job running, tree restored and verified clean afterwards):
  **15 failed / 166 passed / 1 skipped (215 s) — the same 15 names,
  name-for-name.** They are `test_console_realtime_controller_boundary` (1),
  `test_console_review_selection_controller_boundary` (1),
  `test_console_wave6_closeout_inventory` (1), `test_console_wave6_inventory` (3),
  `test_default_timeout_session_guard` (1),
  `test_persistent_diagnostic_inventory` (2),
  `test_progress_widget_clock_guard` (1),
  `test_timer_path_static_update_inventory` (3) and
  `test_worker_exclusive_group_inventory` (2).

The three timer-census failures are the ones worth naming explicitly, since
they were the plausible-regression candidates: they fail identically with the
screen untouched, and independently, neither
`test_timer_path_static_update_inventory.py` nor
`test_worker_exclusive_group_inventory.py` contains a single reference to
`library_screen` — so no `(file, line)` key of theirs could have gone stale.
**Zero of the 18 is attributable to this task.**

### Prompts regression sweep (paired baseline)

**Discovery run (branch, xdist).** The five prompt-heavy files
(`test_library_prompts_canvas.py`, `test_library_prompts_reader.py`,
`test_library_prompt_collections.py`, `test_library_prompt_browse_controller.py`,
`test_library_choice_strips.py`) under `-p no:randomly -q -n 8 --dist worksteal`:
**23 failed / 435 passed** (127.8 s).

**Paired baseline, single-process, identical node-id set (23 names).** Base tree
= `a41cbe8c6` (the RED commit — screen untouched), applied as a
`git checkout a41cbe8c6 -- tldw_chatbook Tests` overlay with **no background
job running** and restored to a verified-clean tree afterwards (the recipe's
overlay-fragility warning applies to long unattended sweeps; these are 2m40s
foreground runs and the restore was confirmed with `git status`).

| | failed | passed | wall | vs base |
|---|---|---|---|---|
| base (`a41cbe8c6`) | 18 | 13 | 160.3 s | — |
| branch run 1 (`f59db7c94`) | 19 | 12 | 160.9 s | 18 shared, 0 base-unique, **1 branch-unique** |
| branch run 2 (same tree, same command) | **18** | **13** | 160.0 s | **failure sets IDENTICAL to base — 0 branch-unique** |

**The 1 branch-unique name from run 1** — resolved twice over, first by
isolation reruns, then decisively by run 2 reproducing the base's exact failure
set on the unchanged branch tree:
`test_library_prompts_canvas.py::test_library_prompt_history_no_change_keeps_selection_and_retry_available`
— already documented **twice** in recipe §7 as run-to-run noise that passes on
rerun (wave-2 task 5's own sweep listed it as one of exactly two branch-unique
names "confirmed pure xdist ordering/shared-state noise by a direct
single-process rerun"; wave-5 task 3's sweep listed it again among 5
branch-unique names "all resolved on a combined single-process rerun"). Here it
**passed 3 of 3 true-isolation reruns on the branch** (`1 passed` × 3) **and
did not fail at all in the identical second combined run** (run 2 above, whose
19→18 failure set is byte-identical to the base's). Its own subject — Prompt
*history* no-change selection retention — touches none of this task's diff
beyond reading shimmed fields the wiring test proves round-trip correctly, and
the 43-value base-vs-branch default dump (§4) is identical. **Not a
regression.**

**The 18 shared failures** are pre-existing on dev (they fail byte-identically
with the screen untouched), in four clusters:

- 5 CSS-parity/`_css_block` assertions reading
  `tldw_chatbook/css/tldw_cli_modular.tcss` + `_agentic_terminal.tcss` for
  `#library-prompts-*`/`.library-prompt-row` blocks that now live in the
  per-screen bundles (`screen_agentic_library.tcss`) —
  `test_library_prompt_row_class_matches_notes_row_visual_parity`,
  `test_library_prompts_header_filter_empty_have_css_blocks`,
  `test_library_prompt_editor_field_css_blocks_match_notes_editor_parity`,
  `test_library_prompt_field_hint_css_block_matches_field_label_parity`,
  `test_library_prompts_import_row_css_blocks_match_filter_status_parity`.
- 4 `test_library_prompt_history_geometry_uses_only_the_outer_editor_scroll[dirty-size0..3]`.
- 4 `test_library_prompt_bulk_delete_focus_and_refresh_are_exactly_once[…]`.
- 5 others: `test_library_prompts_unmount_revokes_late_apply_before_workspace_shutdown`,
  `test_library_shell_create_prompt_save_creates_and_increments_count`,
  `test_library_prompt_import_blocks_undo_until_import_settles`,
  `test_cancelled_prompt_import_retains_writer_ownership_until_commit`, and
  `test_library_choice_strips.py::test_media_type_strip_works_in_both_layouts`
  (the last already documented in recipe §7, wave-2 task 4).

These 18 are **new to this recipe's §7 documented-failure list** (except the
`test_media_type_strip` row) and should be added there at the wave close — they
are a dev-state backdrop this wave inherits, not anything this task caused.
**Zero real regressions.**

**The three pre-existing reds the brief pre-authorized** were confirmed by
construction rather than by a re-run, which is stronger: `git diff --name-only
e5e03846a HEAD` shows this task's ENTIRE diff is 7 files, and **none** of them
is `chat_screen.py`, `library_media_browse_controller.py`, or
`test_library_modules_size_ratchet.py`; `git diff e5e03846a HEAD --
test_screen_size_ratchet.py` shows the `chat_screen.py` `_BUDGETS` row
untouched. Neither the measured file nor its budget changed, so all three rows
necessarily fail identically at the wave-6 base commit.

---

## 9. Size ratchet

Measured fresh, post-edit:

| | lines | methods |
|---|---|---|
| before (`a41cbe8c6`) | 41393 | 1321 |
| after (`f59db7c94`) | **41359** | **1321** |

`git diff --numstat a41cbe8c6 f59db7c94 -- …/library_screen.py` → **34 added,
68 deleted**, net **−34**: −68 `__init__` field/comment lines; +3 replacement
comment lines, +6 construction lines (1 statement + its 5-line comment),
+4 import lines, +21 shim block (7 comment + 12 code + `del` + blank
separator, sentinel lines included). Methods unchanged — a pure field move
touches zero `FunctionDef`s. `_BUDGETS` lowered to
`("LibraryScreen", 41359, 1321)` in the same commit (recipe §6).

> **ERRATUM — commit `f59db7c94`'s message carries a stale line breakdown.**
> Its body says "−63 `__init__` lines, +6 construction/comment lines, +4 import
> lines, +21 shim block, +2 replacement comment lines", which sums to **−30**,
> not the real net **−34**. The correct breakdown is the one in the table
> above: **−68, +3, +6, +4, +21**. The totals the commit states elsewhere
> (41393 → 41359, methods 1321) and the `_BUDGETS` row it lands are **correct
> and verified**; only the intermediate per-category figures are wrong, an
> artifact of an early draft of the breakdown written before the final edit
> pass. **The commit is deliberately NOT amended**: its hash is load-bearing in
> `.git-blame-ignore-revs` (and is quoted in this report, the blame-ignore
> entry's own comment, and the wiring/characterization docstrings), so
> rewriting it to fix prose would break a durable reference for no behavioral
> gain. This erratum is the correction of record.

Pin trajectory so far: `41393/1321 → 41359/1321`.

---

## 10. Deferred to this series' later tasks

- **Dead imports** left by this move: `PromptSourceCapabilities` and
  `local_prompt_capabilities` are each now at exactly ONE occurrence in
  `library_screen.py` (their own import line). Deferred to the cleanup PR
  (task 3), per the export/collections/skills/ingest Task-3 precedent —
  and each must first be checked against PR-0a's `_SURFACE` re-export
  contract, which has bitten two prior series.
- The 14 quoted-string sites in §6 (2 of them needing a receiver fix).
- Screen-side flat-reference retargets and shim-block deletion (task 3).
- The 3 WIRING fields stay out of scope permanently.

## 11. Notes for task 2 (controller PR)

> **Scope warning:** this section is a **lead list, not a completed census**.
> The state PR only needed the bypass shapes a *field* move can trip; the
> unbound-fake-self and module-globals censuses that gate a *method* move are
> task 2's own mandatory work (recipe §3), and the numbers below are a starting
> point for them, not a substitute. An earlier draft of this section named a
> single call site and read as if the sweep were done — corrected after review.

**Unbound fake-self call sites — 19 across 3 files** (re-derived for this
correction: an `ast`-free regex census for
`LibraryScreen.<prompt-named method>(` across all of `Tests/`). Every one is an
exclusion candidate for the controller move (recipe §3 shape 1), and all are
harmless to *this* state PR because the receiver is a `SimpleNamespace`/fake,
not a `LibraryScreen`, so no shim applies:

| File | Sites | Methods |
|---|---|---|
| `Tests/UI/test_library_prompts_canvas.py` | 14 | `handle_library_prompts_empty_new` (:2005), `_build_library_prompts_state` (:2754, :2784), `handle_library_prompts_sort` (:2854, :2857), `handle_library_prompts_filter` (:2869), `_settle_library_prompt_delete` (:9960), `handle_library_prompt_insert_console` (:13323, :13408, :13450, :13515, :13556, :13622), `on_prompt_block_editor_apply_requested` (:13363) |
| `Tests/UI/test_library_canvas_scoped_sync.py` | 3 | `handle_library_prompt_row` (:259), `_apply_library_prompts_import_status` (:423, :429) |
| `Tests/UI/test_library_choice_strips.py` | 2 | `handle_library_prompts_sort_choice` (:591, :605) |

The one to look at first is
**`Tests/UI/test_library_canvas_scoped_sync.py:259`** — `await
LibraryScreen.handle_library_prompt_row(prompt_screen, prompt_event)` against a
`SimpleNamespace` built at `:232-247` that hand-carries **6 moved state fields**
(`_library_prompts_mutation_in_flight`, `_library_prompt_detail`,
`_library_prompt_detail_selected_name`, `_selected_prompt_id`,
`_library_prompt_select_mode`, `_library_prompts_view`) alongside the WIRING
`_library_prompt_browse_controller`, the shared-shell `_library_selected_row_id`,
5 method mocks and `run_worker`. (The review's hand-off said "8 moved flat
names"; the verified split is 6 moved fields + 2 prompt-adjacent non-moved
attributes — recorded precisely because this program's recurring failure is
count drift.) That same test also patches `_sync_library_canvas` on the
`library_screen` module object, so it is simultaneously a **module-globals**
candidate for the §3 eighth-shape census — the exact shape the ingest series'
own review found late.

**Other task-2 leads:**

- `library_prompt_browse_controller.py` is prior-extracted wiring; screen
  methods that merely delegate to it (or to
  `_library_prompt_collections_controller`/`_library_prompt_history_controller`)
  are exclusion candidates, per the skills-series browse/import-coordinator
  precedent.
- The three WIRING fields stay out of scope permanently, but a moved body
  reading `self._library_prompt_browse_controller` needs a named late-binding
  dependency — this is the single most-referenced prompt attribute (41 users).
- `library_export_controller.py` already carries a read-only
  `_library_prompts_mutation_in_flight` accessor fed by a late-binding lambda
  from `library_screen.py:2343`. It resolves through the new shim unchanged and
  is a ready-made template for the same binding in the prompts controller.

**Basename-collision addendum — the 24th mention, outside `.py`.** §2.5's
census used `--include="*.py"` and so reported 23 mentions. There is one more,
in a non-Python artifact:
`Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt:317` —
`tldw_chatbook.Library.library_prompts_state`, i.e. the **domain** module.
The NEW UI module is **absent** from that snapshot (as is
`UI.Library_Modules.library_ingest_state`, its precedent), and that absence is
exactly what keeps this wave's module-level state import safe: `library_screen.py`
is not imported at `_ui_ready`, so a module-level state import adds nothing to
the boot budget. Confirmed empirically — `Tests/Performance/test_ui_ready_module_census.py`
is green in the battery. Task 2's controller import must stay **function-local**
for the same reason (a Global Constraint), and this snapshot is the guard that
would catch a violation.
