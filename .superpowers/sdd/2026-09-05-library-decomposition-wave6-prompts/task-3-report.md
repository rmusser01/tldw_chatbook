# Wave-6 Task 3 — Prompts cleanup (prompts series 3/3)

**Branch:** `refactor/library-decomp-wave6-prompts`
**Worktree:** `/Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook/.worktrees/library-decomp-foundation`
**Parent:** `bcf0631f7` (`docs(sdd): wave-6 task-2 close — ledger, report line-range correction`)

Every hash in this report came from `git rev-parse` / `git log` output, never
from memory. Every count was re-derived from the tree at the moment of
writing, by script, and the scripts are reproduced or described precisely
enough to re-run.

---

## 0. Headline numbers

| Quantity | Value | How derived |
|---|---|---|
| State fields in scope | **43** | `dataclasses.fields(LibraryPromptsState)` |
| Moved method names in scope | **139** | `_PROMPTS_CLUSTER_METHOD_NAMES`, AST-extracted |
| Screen-side flat attribute retargets | **128** | regex pass, counted by the substitution callback |
| Screen-side dynamic-dispatch string retargets | **4** | hand-applied, each asserted `count(old) == 1` |
| Screen-side `getattr` RECEIVER fixes | **1** | hand-applied |
| Test-side attribute-path retargets | **465** across 11 files | regex pass, counted per file |
| `SimpleNamespace` fixture restructurings | **13** fixtures / **27** flat kwargs, + 1 class-attribute harness (5 construction sites) | AST-driven pass + hand edit |
| Screen delegators pruned | **39** of 139 (~28%) | census below |
| Dead imports removed | **25** (5 more SAVED by `_SURFACE`) | AST-diff vs wave-6 start |
| Screen pin | **37722/1321 → 37574/1282** | the ratchet's own `_measure()` |
| Controller pin | **4991 → 4998** | the modules ratchet's own `_measure()` |

---

## 1. Dynamic-dispatch census — FIRST, all four spellings, both name sets

Run before any edit, over **the 43 state fields AND the 139 mover names**, in
all four spellings the recipe's §3 requires:

**A — attribute form** (`<recv>.<name>`), **B — quoted-string form**
(`"<name>"` / `'<name>'`), **C — bare-assignment / kwarg form**
(`<name>=` not preceded by a dot), **D — patch-target-table form** (a
`(target, "<name>", key)` row consumed by a *distant* `monkeypatch.setattr`;
exemplar `Tests/UI/test_library_shell.py:5146`).

Roots swept: `tldw_chatbook/`, `Tests/`, `Docs/`, `backlog/`, `scripts/`,
`Helper_Scripts/` — content greps, never `-k` filters.

### 1.1 The 43 field names — 1195 raw hits over 33 files

Bucketed by spelling (the classifier looked at the character immediately
before the match, so `"` → QUOTED, `.` → ATTR, else BARE):

```
########## QUOTED  (17)
tldw_chatbook/UI/Library_Modules/library_prompts_controller.py:873: [_library_prompts_view] and getattr(self, "_library_prompts_view", "list") == "editor"
tldw_chatbook/UI/Library_Modules/library_prompts_controller.py:1717: [_library_prompt_block_state] block_state = getattr(self, "_library_prompt_block_state", None)
tldw_chatbook/UI/Library_Modules/library_prompts_controller.py:2975: [_library_prompt_block_state] block_state = getattr(self, "_library_prompt_block_state", None)
tldw_chatbook/UI/Screens/library_screen.py:7324: [_library_prompts_reader_preferences] "prompts": "_library_prompts_reader_preferences",
tldw_chatbook/UI/Screens/library_screen.py:7468: [_library_prompts_reader_preferences] "prompts": "_library_prompts_reader_preferences",
tldw_chatbook/UI/Screens/library_screen.py:9524: [_library_prompts_debounce_timer] "_library_prompts_debounce_timer",
tldw_chatbook/UI/Screens/library_screen.py:10685: [_library_prompts_view] return getattr(self, "_library_prompts_view", "list") == "list"
tldw_chatbook/UI/Screens/library_screen.py:27693: [_library_prompts_sort_choices_visible] "_library_prompts_sort_choices_visible",
tldw_chatbook/UI/Screens/library_screen.py:27732: [_library_prompts_sort_choices_visible] "_library_prompts_sort_choices_visible": "prompts",
Tests/UI/test_library_adaptive_reader_closeout.py:102: [_library_prompts_reader_preferences] "_library_prompts_reader_preferences",
Tests/UI/test_library_adaptive_reader_closeout.py:103: [_library_prompts_reader_layout] "_library_prompts_reader_layout",
Tests/UI/test_screen_navigation.py:3253: [_library_prompts_view] lambda: setattr(screen, "_library_prompts_view", "list")
Tests/UI/test_library_screen_reuse.py:109: [_library_prompts_debounce_timer] "_library_prompts_debounce_timer",
Tests/UI/test_library_screen_reuse.py:228: [_library_prompts_debounce_timer] "_library_prompts_debounce_timer",
Tests/Architecture/test_library_prompts_wiring.py:150: [_library_prompts_view] assert prompt_state_shim_attr("view") == "_library_prompts_view"
Tests/Architecture/test_library_prompts_wiring.py:152: [_library_prompt_dirty] assert prompt_state_shim_attr("dirty") == "_library_prompt_dirty"
Tests/Architecture/test_library_prompts_wiring.py:154: [_selected_prompt_id] assert prompt_state_shim_attr("selected_prompt_id") == "_selected_prompt_id"
```

Dispositions:

| Sites | Disposition |
|---|---|
| controller `:873`, `:1717`, `:2975` | **STAY untouched.** These are moved bodies, and they resolve through `LibraryPromptsController`'s own PERMANENT generated shim loop (the ingest precedent: "controller shims STAY"). Task 1 flagged 4 receiver-fix sites; task 2's move took 3 of them here. |
| screen `:7324`, `:7468` | Retargeted to `"_prompts_state.reader_preferences"` — collections/skills precedent, both dicts already read via `operator.attrgetter` and written via `_assign_library_reader_preferences_attribute` (a dotted-path passthrough). Zero consumption-site changes. |
| screen `:27693`, `:27732` | Retargeted to `"_prompts_state.sort_choices_visible"`, sitting beside the identical skills row. |
| screen `:10685` | **RECEIVER fix**, not a string swap: `getattr(self._prompts_state, "view", "list")`. |
| screen `:9524` + tests `:109`, `:228` | Timer-attr tuple — the string loop uses plain `getattr`/`setattr`, which is NOT dotted-aware. Followed the ingest path-debounce-timer precedent three lines below it: the name leaves the tuple and gets its own explicit `_prompts_state` block, with the matching restructuring in both test pins. |
| tests `:102`, `:103` | `DESTINATION_CONTRACT` dotted (sixth consecutive subsystem to need this one entry). |
| tests `:3253` | RECEIVER fix: `setattr(screen._prompts_state, "view", "list")`. |
| wiring `:150`, `:152`, `:154` | **KEEP.** These are the literal-string pins on `prompt_state_shim_attr()`, which the CONTROLLER's shim loop still calls. Deleting them would reopen the "screen and test agree on the same wrong answer" hole the file's own docstring documents. |

**BARE form — 78 hits.** 40 are docstring/comment prose (left as historical
prose per the ingest precedent, except the 10 corrected in §8). 37 are
`SimpleNamespace` kwargs / one class attribute — the fixture restructuring in
§4.2. One (`library_export_controller.py:545`,
`def _library_prompts_mutation_in_flight`) is the EXPORT controller's own
read-only accessor property name, fed by a late-binding lambda from
`library_screen.py` — its own API, untouched; the lambda's body was retargeted
with the rest of the screen.

**ATTR form — 1100 hits**: 439 inside the controller (moved bodies, resolve
through the controller shim, untouched), 128 on the screen, 465 in `Tests/`,
64 in frozen `Docs/` artifacts (§1.3), 4 elsewhere (all prose).

### 1.2 The 139 mover names — clean in every non-attribute spelling

The string-literal sweep over all 139 movers across `tldw_chatbook/`,
`Tests/` and `Docs/` returned **exactly two groups and nothing else**: the
wiring test's own 139-name pin tuple (`:224-362`) plus its 1-name
staticmethod frozenset (`:372`), and four rows in
`Tests/UI/test_library_modal_dismissal.py` (`:581`, `:587`, `:599`, `:605`) —
the deferred task-2-§10 inventory, handled in §7.

**Spelling D specifically:** the patch-target-table exemplar at
`Tests/UI/test_library_shell.py:5146` DOES name a prompt method —
`(screen, "_request_library_prompts_browse", "load")` — but
`_request_library_prompts_browse` is one of task 2's **22 exclusions**, so it
never moved and needed nothing. No mover name appears in any patch-target
table, and there is no `monkeypatch.setattr(screen|LibraryScreen, "<mover>",
...)` anywhere in the repo.

### 1.3 Frozen `Docs/` artifacts — not retargeted, by established precedent

64 flat-name ATTR hits live in three uncollected scripts under `Docs/`
(`qa/console-prompt-improvement-2026-08/capture_qa.py`,
`reviews/evidence/task-22033/task22033_live_matrix_runner.py`,
`reviews/evidence/task-23019/task23019_scenarios.py`) plus prose in
`Docs/superpowers/plans/*`. `pyproject.toml` sets `testpaths = ["Tests"]`, so
none is ever collected, and the precedent is explicit rather than assumed:
`task23019_scenarios.py:209` still reads `screen._library_skill_editor_state`
and `:218` `screen._library_skill_reader_mode` **two waves after the skills
cleanup deleted those shims**. Left alone, exactly as skills and ingest left
them. (One of these hits DID change a prune verdict — see §3.3.)

---

## 2. Screen-side retargets

**128 literal `self.<flat>` occurrences** rewritten to
`self._prompts_state.<field>` in one word-boundary-anchored,
longest-match-first regex pass over a 43-name mapping table. Two guards ran
first: a receiver check (**0** non-`self` receivers of any flat name in
`library_screen.py`) and a collision check (**0** overlaps between the 43
field names and the 139 mover names). A zero-result re-grep for all 43 flat
names over the whole file confirmed the pass afterwards.

The 128 live in `__init__`, in the 22 still-screen-resident exclusions, and in
shell/plumbing methods (`save_state`/`restore_state`, `check_action`,
`compose_content`, `flush_pending_work`, the reader-preference dispatchers,
`on_screen_suspend`, the rail-selection guards).

Plus the 4 string retargets and 1 `getattr` receiver fix from §1.1, and the
timer-block restructuring.

---

## 3. Delegator census — 100 KEEP, 39 PRUNED (~28%)

### 3.1 Method

For each of the 139 movers, a **tokenize-based** census (Python `tokenize`,
so a NAME token is a real identifier reference and a docstring mention is
bucketed as STRING, never as a call) over `tldw_chatbook/` + all of `Tests/`,
excluding only (a) `library_prompts_controller.py` and (b) each name's own
delegator body (`node.lineno..node.end_lineno` in `library_screen.py`). Then
a **second, broader** pass — "name appears anywhere on the line" — over
`Docs/`, `backlog/`, `scripts/`, `Helper_Scripts/`, `.github/`.

**Sanity control (the skills-series lesson-2 requirement), run before
trusting any verdict:**

```
_current_library_prompt_editor_state: verdict='KEEP (17 code refs)'; self.<name>( grep hits=11
_read_library_prompt_editor_fields:   verdict='KEEP (8 code refs)';  self.<name>( grep hits=10
_clear_library_prompt_selection:      verdict='KEEP (8 code refs)';  self.<name>( grep hits=10
_start_library_prompts_import:        verdict='KEEP (11 code refs)'; self.<name>( grep hits=2
_library_prompt_work_pane_kwargs:     verdict='KEEP (3 code refs)';  self.<name>( grep hits=1
```

Five names known to have `self.<name>(` callers, all correctly KEEP.

### 3.2 A genuinely NEW hazard: Textual's `on_<Message>` name dispatch

The recipe's transform whitelist names `@on` and `action_*`. This cluster has
a THIRD unconditional-KEEP class the whitelist does not:
**`on_prompt_block_editor_back_requested`,
`on_prompt_block_editor_block_action_requested`,
`on_prompt_block_editor_block_field_changed`,
`on_prompt_block_editor_save_as_prompt_requested`,
`on_prompt_block_editor_save_as_recipe_requested`,
`on_prompt_block_editor_update_original_requested`.**

Evidence they are dispatched purely by NAME, read out of the installed
Textual (8.x) rather than assumed:

- `.venv/lib/python3.14/site-packages/textual/message.py:86` —
  `cls.handler_name = f"on_{name}"`
- `.venv/lib/python3.14/site-packages/textual/message_pump.py:817-821` —
  `handler_name = message.handler_name` … `for cls, method in
  self._get_dispatch_methods(handler_name, message):`
- `message_pump.py:743-758` — `_get_dispatch_methods(method_name, message)`
  walks `self.__class__.__mro__` looking that name up.

None carries a decorator (`decorator_list == []` for all six, measured), so a
decorator-driven whitelist misses them. And **no code anywhere spells
`screen.on_prompt_block_editor_*`**, so a pure reference count marks all six
PRUNE. Deleting them would have silently unhooked `LibraryScreen` from six
messages.

A naive census even reports them as "having code references" — but every hit
is a `def` of the same name in a DIFFERENT class: 2 in
`Widgets/Library/library_prompts_canvas.py` (`:1699`, `:1705`) and 4 more in
a `Tests/UI/test_prompt_block_editor.py` harness (`:112`-`:142`, verified by
grep). That same coincidence is why the deletion would likely have stayed
green in the suite: the canvas keeps handling its own two one level down, and
the harness's four never involve `LibraryScreen` at all.

**Rule for the next series:** before pruning any moved name matching
`^on_[a-z]`, check it against Textual's name-based dispatch, not against the
reference census.

### 3.3 A second finding: an uncollected-but-executable `Docs/` script

`_library_prompt_can_update_original` has **zero** references in
`tldw_chatbook/` and `Tests/` — but
`Docs/superpowers/reviews/evidence/task-22033/task22033_live_matrix_runner.py:262`
reads:

```python
f"can_update={screen._library_prompt_can_update_original()!r}, "
```

on a real screen. That file is never collected (`testpaths = ["Tests"]`), and
§1.3's precedent says frozen `Docs/` scripts are not retargeted. But the prune
rule is "ZERO references outside its own body **anywhere in the repo**", and
this is a reference in executable form. **Delegator KEPT.** First time a
`Docs/` hit has changed a prune verdict; recorded in the recipe.

A check for the same shape across the other 39 candidates found only Markdown
prose in `Docs/superpowers/plans/*` (6 names), no further executable hits.

### 3.4 The bare-callable catch — why the broad pass earns its keep

Three KEEP names have code references that a call-shaped
`<name>\s*\(` regex scores as **zero**:

| Name | Its only reference |
|---|---|
| `_arm_library_prompt_editor` | `self.call_after_refresh(self._arm_library_prompt_editor)` (+ the same shape in `test_library_prompts_canvas.py`) |
| `_restore_library_prompts_focus` | passed as a positional argument on its own line |
| `_sync_library_prompt_memberships` | `sync_memberships=lambda: self._sync_library_prompt_memberships,` |

A call-shaped census marks all three PRUNE and deletes three live callbacks.
This is the skills-series lesson firing for real rather than passing
vacuously.

### 3.5 Verdicts

| Class | Count | Basis |
|---|---|---|
| KEEP — `@on` handler | 44 | transform whitelist (§4) |
| KEEP — `action_*` | 1 | transform whitelist (§4) |
| KEEP — `on_<Message>` name-dispatched | 6 | §3.2 |
| KEEP — genuine external caller | 49 | tokenize census + broad pass |
| **PRUNE** | **39** | zero references, all four spellings, whole repo |
| **Total** | **139** | |

Whitelist = 51; non-whitelist candidates = 88 (1 `@staticmethod`
`_restore_library_prompts_scope`, which has a real caller at
`library_screen.py` and is KEPT, + 87 plain). 49 + 39 = 88. ✓

The 39 pruned:

```
_apply_library_prompt_detail_failure            _notify_library_prompt_unsupported_artifact_type
_apply_library_prompt_save_outcome              _open_library_prompt_colliding_with_current_name
_await_library_prompt_durable_call              _open_library_prompt_delete_confirmation
_await_library_prompt_save_call                 _reconcile_library_prompt_history_region
_capture_library_prompt_block_state             _reconcile_library_prompt_memberships
_claim_library_prompt_detail_generation         _request_library_prompt_history_count
_detach_library_prompt_working_copy             _request_library_prompt_history_page
_exit_library_prompt_editor_guarded             _resolve_library_prompt_create_conflict
_initialize_library_prompt_history              _restore_library_prompt_history
_library_prompt_action_artifact_type            _return_to_library_prompt_create_draft
_library_prompt_artifact_fields                 _save_library_prompt
_library_prompt_detail_failure_notice           _set_library_prompt_discard_enabled
_library_prompt_detail_request_is_current       _sync_library_prompt_open_existing_button
_library_prompt_history_action_is_current       _sync_library_prompt_save_action_widgets
_library_prompt_legacy_recipe_requires_conversion  _undo_library_prompt_delete
_library_prompt_loading_notice                  _update_library_prompt_meta_static
_library_prompt_markdown_artifact_fields
_library_prompt_mutation_is_current
_library_prompt_text_fields_match_state
_load_library_prompt_memberships
_mark_library_prompt_dirty
_notify_library_prompt_legacy_recipe_requires_conversion
_notify_library_prompt_unrepresentable_markdown
```

**Post-deletion verification** (re-run against the edited tree):

```
pruned names still present on LibraryScreen: []
movers still delegating: 100 (expected 100)
non-controller/non-pin references to pruned names: 15 (all prose)
```

All 15 are comments/docstrings. Two of them now mis-attribute a moved method
to the screen —
`Widgets/Library/library_prompts_canvas.py:131`
(``LibraryScreen._update_library_prompt_meta_static``) and
`UI/Console_Modules/prompts.py:1854`
(``library_screen._save_library_prompt``). Both are in files outside this
task's scope; left with a forward note, exactly as the skills series left
`library_rag_search_controller.py`'s equivalent stale claim.

Prune fraction 39/139 ≈ 28%: export ~5% < ingest ~11% < skills ~19% <
collections ~22% < **prompts ~28%** < search+RAG ~29% < conversations ~30%.

---

## 4. Test retargets

### 4.1 Attribute paths — 465 across 11 files

| File | Retargets |
|---|---|
| `Tests/UI/test_library_prompts_canvas.py` | 337 |
| `Tests/UI/test_library_prompts_reader.py` | 48 |
| `Tests/UI/test_library_prompt_collections.py` | 14 |
| `Tests/UI/test_library_prompts_characterization.py` | 14 |
| `Tests/ProductionApp/test_personas_library_root_state.py` | 13 |
| `Tests/UI/test_library_adaptive_reader_closeout.py` | 13 |
| `Tests/UI/test_screen_navigation.py` | 11 |
| `Tests/UI/test_library_shell.py` | 10 |
| `Tests/UI/test_library_canvas_scoped_sync.py` | 2 |
| `Tests/UI/test_library_choice_strips.py` | 2 |
| `Tests/UI/test_library_entry_compose_once.py` | 1 |

### 4.2 Fixture restructuring — 13 `SimpleNamespace` + 1 class attribute

Flat kwargs → nested `_prompts_state=SimpleNamespace(...)`, driven by an AST
pass that located every `SimpleNamespace(...)` call carrying a moved flat
kwarg, then rewrote it preserving each value's exact source segment. 27 flat
kwargs across 13 fixtures in 3 files (`test_library_prompts_canvas.py` 10/17,
`test_library_canvas_scoped_sync.py` 2/8, `test_library_choice_strips.py`
1/2). Every one of these fakes stands in for an unbound `self` on one of the
22 EXCLUSIONS, which now read `self._prompts_state.<field>`.

The 14th is the shape the wiring test's own docstring calls "a fake-harness
CLASS ATTRIBUTE": `_LibraryPromptHandlerHarness(SimpleNamespace)` carried
`_library_prompts_mutation_in_flight = False` as a class attribute while its 5
constructions passed `_library_prompts_view`/`_library_prompt_dirty` as
instance kwargs. Nesting naively would have let the per-instance
`_prompts_state` **shadow** the class-level one and silently drop
`mutation_in_flight`; instead the class attribute became
`_prompts_state = SimpleNamespace(mutation_in_flight=False)` and each of the 5
constructions passes a fuller nested object that re-states it.

### 4.3 Assertions byte-for-byte — verified mechanically

Every retargeted test file was diffed against `HEAD` after normalizing BOTH
sides through the receiver rewrite (`.<flat>` and `._prompts_state.<field>`
both collapsed to a canonical marker). Residual lines, per file:

```
Tests/ProductionApp/test_personas_library_root_state.py: 0
Tests/UI/test_library_entry_compose_once.py:             0
Tests/UI/test_library_prompt_collections.py:             0
Tests/UI/test_library_prompts_characterization.py:       0
Tests/UI/test_library_prompts_reader.py:                 0
Tests/UI/test_library_shell.py:                          0
Tests/UI/test_library_adaptive_reader_closeout.py:       8   (DESTINATION_CONTRACT dotting + its comment)
Tests/UI/test_library_choice_strips.py:                  6   (one fixture nested)
Tests/UI/test_library_canvas_scoped_sync.py:            20   (two fixtures nested)
Tests/UI/test_library_prompts_canvas.py:                95   (ten fixtures + the harness class)
Tests/UI/test_library_screen_reuse.py:                  32   (timer-tuple restructuring + comments)
Tests/UI/test_screen_navigation.py:                      2   (the setattr receiver fix)
```

Six files residual-ZERO; the other six residual lines are, one for one, the
intentional structural edits listed above. **Zero assertion VALUES changed
anywhere.**

---

## 5. Shim deletion, and the wiring test's flip to asserting absence

The task-1 generated block (`for _lps_field in dataclasses.fields(
LibraryPromptsState): setattr(LibraryScreen, prompt_state_shim_attr(...),
property(...))`, 20 lines) was deleted wholesale and replaced by a 10-line
"deleted here, and why" marker stacked under the identical
collections/search+RAG/skills/ingest markers already at the end of the file.

`Tests/Architecture/test_library_prompts_wiring.py` — 12 tests before, **13
after**, all green:

- `test_state_object_fields_match_the_shim_surface` → narrowed to
  **`test_state_object_declares_the_censused_field_count`** (the half that
  never depended on the screen).
- `test_every_shim_reads_and_writes_its_own_state_field` → **re-aimed at the
  controller** as `test_every_controller_shim_reads_and_writes_its_own_state_
  field`, rather than deleted. **Deliberate deviation from the prior series'
  delete-it precedent, and the reason is specific:** this is the only test in
  the file that would catch the closure-binding trap (43 generated properties
  all capturing the LAST field), the controller's own permanent loop is
  generated by the identical `dataclasses.fields` iteration and carries the
  identical trap, and the surviving
  `test_prompts_controller_exposes_every_state_field` is exactly the weaker
  `isinstance(..., property)` check this one exists to strengthen. Prior
  series had no equivalent second test to lose, so there is no precedent
  being contradicted — only a gap not being opened.
- **`test_the_screen_no_longer_carries_a_prompt_state_shim` added**, asserting
  ABSENCE of all 43 flat names on `LibraryScreen` in any form.
- `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED` filled with the 39 names. Both
  `test_screen_delegates_prompt_handlers` and
  `test_prompts_cluster_staticmethods_forward_to_the_controller_class` were
  already wired by task 2 to skip pruned names and assert genuine absence
  instead, so this was the one-line frozenset change task 2 designed for.
- Module docstring rewritten (it described the screen shim as live).

---

## 6. Dead imports — the `_SURFACE` check saved 5 of 30

Derived as a **difference**, not an absolute list: the AST-unused imported-name
set in `library_screen.py` at the wave-6 start commit `e5e03846a` versus at
this commit. 38 names are unused at BOTH ends (prior series' residue,
`__future__`, `_SURFACE` bookkeeping) and are out of scope; **30 are newly
dead in wave 6**; 0 were resurrected.

Each of the 30 was checked **individually by exact-name lookup** against
`Tests/Architecture/test_library_support_layer_surface.py`'s `_SURFACE`. Five
came back pinned and were KEPT:

```
LIBRARY_PROMPT_DIRTY_VETO_COPY            (_SURFACE :64)
LIBRARY_PROMPT_SAVE_STATUS_COPY           (_SURFACE :62)
_LIBRARY_PROMPTS_IMPORT_WORKER_GROUP      (_SURFACE :48)
_LIBRARY_PROMPTS_SEARCH_DEBOUNCE_SECONDS  (_SURFACE :45)
_LIBRARY_PROMPT_WRITE_WORKER_GROUPS       (_SURFACE :49)
```

`test_screen_still_re_exports_every_moved_name` asserts the MODULE surface,
not live usage — the shape that has now bitten/nearly-bitten three series
running, and the largest hit yet (ingest saved 1 of 9).

**And it nearly escaped a second way.** The first `_SURFACE` check was a
case-sensitive grep for `prompt` over that file, which returned **zero
matches** — because all five names spell it `PROMPT`. Had the check stopped
there, all five would have been deleted and
`test_screen_still_re_exports_every_moved_name` would have gone red.
**Lesson: do the `_SURFACE` check by exact-name lookup, never by a lowercase
subsystem-word grep.**

The remaining **25 were deleted**: 23 confirmed already re-imported and live
inside `library_prompts_controller.py`, and 2
(`PromptSourceCapabilities`, `local_prompt_capabilities` — the pair task 1
deferred) genuinely needed nowhere else, since they now live only in
`library_prompts_state.py`, which owns the field whose `default_factory`
calls them. Post-deletion `import tldw_chatbook.UI.Screens.library_screen`
succeeds and all 5 pinned names are still re-exported (asserted directly).

---

## 7. `Tests/UI/test_library_modal_dismissal.py` — the §10 MUST

Order of work, exactly as task 2's §10 specified:

1. **`_OwnerScope` row added first.** `_SUPPORTED_OWNER_SCOPES` spans
   **`:520-526`** in the pre-edit file (task 2's corrected range — re-verified
   against the live file before editing, per the line-range lesson: `:520` is
   `_SUPPORTED_OWNER_SCOPES = (`, `:521-525` the five `_OwnerScope` rows,
   `:526` the closing `)`). A new
   `_PROMPTS_CONTROLLER_FILE` constant and an
   `_OwnerScope(_PROMPTS_CONTROLLER_FILE, "LibraryPromptsController")` row
   went in, making the file six scopes.
2. **The four rows repointed** (`handle_library_prompts_import_browse`,
   `handle_library_prompt_history_restore`, `_export_library_prompt`,
   `_open_library_prompt_delete_confirmation`), file + class both.
3. **`_stage_library_prompt_for_console` left alone** — it is one of the 22
   exclusions, still screen-resident, still correctly declared under
   `LibraryScreen`. Verified untouched.

### 7.1 The limitation, stated plainly

`test_library_modal_inventory_matches_declared_edges_bidirectionally` **cannot
be proven green end-to-end**, because it is pre-RED at this task's parent for
an unrelated skills-era failure that aborts discovery before any comparison:

```
unresolved modal constructor in supported presenter:
tldw_chatbook/UI/Screens/library_screen.py:LibraryScreen._present_library_skills_import_choice_if_needed
(SkillImportChoiceModal(snapshot.candidates))
```

Measured on both trees:

| Tree | Result |
|---|---|
| Baseline worktree at `bcf0631f7` | **1 failed, 169 passed** (108.77 s) |
| This branch | **1 failed, 169 passed** (112.79 s) |

Same single failing node name. The retarget neither fixes nor worsens it. Per
task 2's §10 instruction, the cross-wave repair (the skills blocker plus
`handle_library_ingest_browse` and the two skill-trust passphrase presenters,
all equally stale delegators) is **NOT attempted here** — filed for wave close.

### 7.2 Construction proof, row by row

Running the file's own `_discover_library_modal_edges` against the new scope
alone, and comparing to the four declared rows:

```
DISCOVERED: 4
    ('tldw_chatbook/UI/Library_Modules/library_prompts_controller.py', 'LibraryPromptsController', '_export_library_prompt', <class '...file_save.FileSave'>)
    ('tldw_chatbook/UI/Library_Modules/library_prompts_controller.py', 'LibraryPromptsController', '_open_library_prompt_delete_confirmation', <class '...prompt_delete_confirmation_modal.PromptDeleteConfirmationModal'>)
    ('tldw_chatbook/UI/Library_Modules/library_prompts_controller.py', 'LibraryPromptsController', 'handle_library_prompt_history_restore', <class '...confirmation_dialog.ConfirmationDialog'>)
    ('tldw_chatbook/UI/Library_Modules/library_prompts_controller.py', 'LibraryPromptsController', 'handle_library_prompts_import_browse', <class '...file_open.FileOpen'>)
DECLARED: 4   (identical four tuples)
undeclared: []
missing: []
EXACT MATCH: True
```

Every field of every row matches — path, class name, presenter name, and the
resolved concrete modal type. The same probe against the `LibraryScreen`
scope raises the pre-existing skills assertion, confirming that failure is
the sole reason the end-to-end node stays red.

Note `_open_library_prompt_delete_confirmation` is ALSO one of the 39 pruned
delegators; the two changes are consistent by construction — the presenter now
exists only on the controller, and the row now names the controller.

---

## 8. Docstring / comment staleness — 10 corrections, canon-safe

Every one is on a NON-moved body (a module docstring, a class docstring, or a
comment in a still-screen-resident method), so the byte-for-byte canon is not
in tension; no moved body was touched.

| File | # | What was false |
|---|---|---|
| `library_prompts_controller.py` | 2 | "`LibraryScreen` keeps one-line delegators under every one of these 139 original names" (module docstring) and the same claim again in the class docstring — the exact shape skills and ingest each had to fix. Corrected to 100-of-139 with the whitelist breakdown and a pointer to `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED`. |
| `library_prompts_state.py` | 5 | The module docstring said `library_screen.py` "keeps every original … name alive as a generated … shim"; three more spots credited "the screen's generated shim loop" / "the screen's generated property shim"; `prompt_state_shim_attr`'s own docstring named the screen loop as a caller. All re-pointed at the controller's permanent loop, with the deletion recorded. |
| `library_screen.py` | 3 | A comment naming "the stale `_library_prompt_detail`"; `_flush_library_prompt_save`'s docstring naming "``_library_prompt_dirty``" (both attributes that no longer exist); and `__init__`'s `self._prompts_state = LibraryPromptsState()` comment, which said the four forced-late original lines "route through the generated shim into this object" — they now assign into it directly. |

Plus the wiring test's module docstring (§5) and two comments in
`test_library_screen_reuse.py` (the wave-6-task-1 shim note, now describing
task 3's deletion; and the "seventh timer" ordinal, wrong once the tuple lost
an entry).

---

## 9. Recipe

- **§8 subsystem table**: the `| 4 | prompts | 41 | |` stub filled with the
  series' as-landed numbers (43 fields / three prefixes / 3 WIRING; 139 of 161
  methods with the 22-exclusion breakdown; 39 of 139 delegators pruned; the
  `on_<Message>` finding), pointing at §21.
- **§21 added** — "The prompts series, as landed — the sixth rehearsal, and
  the largest single move of this program": per-task fields/methods table,
  pin trajectory, the delegator census with both novel findings, the
  four-spelling dynamic-dispatch census, the `_SURFACE` import section, the
  modal-inventory construction proof, and the wiring-test finalization. The
  "Wave-6 close" subsection is deliberately left for task 4, matching how
  §20's own close subsection sits inside the ingest series' section.

---

## 10. Fresh pins — both guard files, same commit

**`Tests/Architecture/test_screen_size_ratchet.py`** — `37722/1321 →
37574/1282`, measured with the ratchet's own `_measure()` (`ast`-walked line
count + `LibraryScreen` method count, not `wc -l`).

Method delta **-39** = exactly the pruned delegator count: a pure deletion
with no replacement.

Line delta **-148**, every term measured:

| Term | Lines |
|---|---|
| 39 pruned delegators (3 each: `def` + forwarding `return` + blank separator; none decorated) | −117 |
| dead-import lines (25 names; 2 of the lines are whole single-name `from … import X` statements, and 2 more come from collapsing the now-single-name `library_prompts_state` import back to one line) | −29 |
| generated shim block (20 out, 10-line marker in) | −10 |
| prompts debounce timer lifted into its own explicit block | +6 |
| two corrected comments that each grew a line when re-wrapped | +2 |
| **total** | **−148** ✓ |

**`Tests/Architecture/test_library_modules_size_ratchet.py`** — `4991 → 4998`,
comment-only growth from the two controller docstring corrections (§8);
§17's re-pin-at-move flow applied to a docstring-only delta.

Both re-pins land in the same commit as the change that moved them.

---

## 11. Battery

| Suite | Result |
|---|---|
| the 7 `test_library_*_wiring.py` + `test_library_support_layer_surface.py` | **55 passed** |
| `test_library_prompts_wiring.py` alone | **13 passed** |
| `test_screen_size_ratchet.py` + `test_library_modules_size_ratchet.py` + `test_library_recompose_ratchet.py` + `test_library_preimport_closure.py` + `test_ui_ready_module_census.py` | 46 passed, **3 failed — all 3 the documented pre-existing rows** (2 × `chat_screen.py`, 1 × `library_media_browse_controller.py`); every `library_screen.py`/`library_prompts_controller.py` row GREEN |
| `./scripts/preflight.sh` | **all derived-artifact checks passed** |
| `Tests/UI/test_library_modal_dismissal.py` | 1 failed / 169 passed — **identical on both trees** (§7.1) |
| `Tests/UI/test_library_prompts_characterization.py` (task 1's characterization file) | **4 passed** |
| `Tests/UI/test_library_choice_strips.py` + `test_library_canvas_scoped_sync.py` + `test_library_screen_reuse.py` (the three restructured-fixture files) | 27 passed, **2 failed** — `test_media_type_strip_works_in_both_layouts` and `test_notes_per_click_updates_keep_screen_and_canvas_identity`, both reproduced at the parent in the baseline worktree (**2 failed** there too, same node names) |
| `Tests/UI/test_library_shell.py -k "prompt"` | **1 failed / 10 passed** on BOTH trees — `test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-prompts-#library-prompts-reader-shell]`, same node name, zero branch-unique |
| `Tests/Performance/test_ui_ready_module_census.py` re-run under load | **974 vs pin 972 on BOTH trees**, byte-identical assertion text; the zero-headroom breach task 2 documented. The 25 modules it names as consuming the headroom are all `tldw_chatbook.Chat.console_*` — nothing this wave imports. Green when run in a quiet process. |
| the 5 prompt-heavy files, paired (§12) | branch **24 failed / 404 passed**, baseline **29 failed / 399 passed** — **zero real branch-unique** |
| `Tests/UI/test_library_adaptive_reader_closeout.py` + `test_library_entry_compose_once.py` + `test_screen_navigation.py`, paired (§12) | branch **35 failed / 209 passed**, baseline **34 failed / 210 passed** — **zero real branch-unique** |

Every dismissed red was proven pre-existing by running the identical node
selection in an ISOLATED worktree at the parent commit `bcf0631f7`
(`.worktrees/w6t3-baseline`, its own `uv venv`, verified to resolve its own
tree: `tldw_chatbook.__file__` under `w6t3-baseline/`). No stash overlay was
used anywhere in this task.

---

## 12. Paired baseline

Run in an ISOLATED worktree with its OWN `uv venv` — `.worktrees/w6t3-baseline`,
detached at the parent `bcf0631f7`, `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"`.
The same-tree stash-overlay carve-out was NOT used: the editable-finder trap
means a reused venv resolves the wrong tree, so the venv was verified first:

```
/Users/.../w6t3-baseline/.venv
/Users/.../w6t3-baseline/tldw_chatbook/__init__.py
```

The machine was under heavy contention throughout (up to 17 concurrent
pytest processes from unrelated sessions), so both trees were run with the
SAME contention and compared by NAME SET, never by count alone — the counts
move run-to-run on both trees, and did.

### 12.1 The 5 prompt-heavy files

`test_library_prompts_canvas.py`, `test_library_prompts_reader.py`,
`test_library_prompts_characterization.py`, `test_library_prompt_collections.py`,
`Tests/ProductionApp/test_personas_library_root_state.py`.

| Tree | Result |
|---|---|
| branch (`6fd2b753a`) | **24 failed, 404 passed** (1028.68 s) |
| baseline (`bcf0631f7`) | **29 failed, 399 passed** (1070.86 s) |

22 failing node names are common to both. **2 appear only on the branch**,
**7 only on the baseline** — the asymmetry alone shows the noise floor. Both
branch-only names were then run in ISOLATION on both trees and **both fail at
the parent**:

| Branch-only name | Isolated at parent |
|---|---|
| `test_library_prompt_pager_first_and_filter_failure_states[size0]` | **FAILED** (in the same 2-test isolated run; and it passed 8/8 in later repeats on the baseline, i.e. flaky on the parent, not introduced) |
| `test_library_prompts_stale_search_cannot_restore_an_old_filter_caret` | **FAILED** (and 1-of-5 in a repeat loop on the baseline — flaky on the parent) |

**Zero real branch-unique failures.**

Two of the common 22 are worth naming because their shape rules out a
timing explanation and so is the strongest single piece of evidence here:
`test_library_prompt_editor_field_css_blocks_match_notes_editor_parity` and
`test_library_prompt_row_class_matches_notes_row_visual_parity` are static
CSS-text parity checks with no app boot at all. Run isolated: **2 failed on
the branch, 2 failed on the baseline**, same two names.

### 12.2 `Tests/UI/test_library_shell.py -k "prompt"`

| Tree | Result |
|---|---|
| branch | **1 failed, 10 passed** (29.55 s) |
| baseline | **1 failed, 10 passed** (29.03 s) |

Same node both sides:
`test_adaptive_routes_never_receive_ordinary_emergency_geometry[browse-prompts-#library-prompts-reader-shell]`.
**Zero branch-unique.**

### 12.3 The three other retargeted files (extra, beyond the brief's set)

Because this task retargeted `test_library_adaptive_reader_closeout.py`,
`test_library_entry_compose_once.py` and `test_screen_navigation.py` too,
they were paired as well.

| Tree | Result |
|---|---|
| branch | **35 failed, 209 passed** (619.65 s) |
| baseline | **34 failed, 210 passed** (587.49 s) |

32 names common; **3 branch-only, 2 baseline-only**. All 3 branch-only
resolved against the parent:

| Branch-only name | Verdict |
|---|---|
| `test_prompt_receipt_owner_vetoes_real_app_navigation_until_settlement` | **FAILS at the parent** when run isolated (1 failed) — pre-existing |
| `test_study_screen_escape_returns_to_library_study_staging_canvas` | **FAILS 4/4 at the parent** isolated (and 4/4 on the branch) — pre-existing, order-dependent |
| `test_source_worker_completion_during_resume_dispatch_reconciles_once` | **PASSES on both trees** isolated — a load flake |

**Zero real branch-unique failures.** Note the baseline alone produces 34
failures in this trio under load, which is the honest characterisation of
this file set on this machine right now: it is noisy, and only the paired
name-set comparison says anything.

---

## 13. Commits

| Hash | Subject |
|---|---|
| `6fd2b753aaa90160c286579fe07257cb72917f32` | `refactor(library): prompts cleanup (prompts series 3/3)` |
| `d838b7d632a905018926834ae9cf7a8701028951` | `chore(library): blame-ignore the prompts cleanup` |

Both hashes read out of `git rev-parse HEAD` / `git log --oneline` at the
moment each commit was made. The blame-ignore entry was appended AFTER the
cleanup commit existed (its hash cannot be known before), matching every
prior series' two-commit shape; all entries in `.git-blame-ignore-revs`
re-verified to resolve with `git rev-parse --verify <h>^{commit}`.

`./scripts/preflight.sh` re-run after the commit: **all derived-artifact
checks passed.**

---

## 14. Concerns / hand-offs for task 4 (wave close)

1. **`Tests/UI/test_library_modal_dismissal.py` is still 1-red**, on the
   skills-era `_present_library_skills_import_choice_if_needed` blocker
   (§7.1). The cross-wave repair also needs `handle_library_ingest_browse`,
   `_request_library_skill_trust_passphrase` and
   `_request_library_skill_trust_bootstrap_passphrase` repointed — all three
   are delegators today and equally stale. Deliberately not attempted here.
2. **Two out-of-scope stale prose mentions** (§3.5):
   `Widgets/Library/library_prompts_canvas.py:131` and
   `UI/Console_Modules/prompts.py:1854` still attribute a moved method to
   `LibraryScreen`/`library_screen`. Cosmetic; touching them would widen the
   diff past this task's file scope.
3. **The `_ui_ready` census still has ZERO headroom** (pin 972). Green in this
   task's battery, but task 2 measured it breaching under load on BOTH trees.
   Not this wave's to fix.
4. **The `on_<Message>` whitelist gap is now recorded in the recipe (§21)
   but NOT yet in §4's transform-whitelist text itself.** §4 still says
   "`@on`/`action_*`/anything a test reaches directly". Media and notes both
   almost certainly own name-dispatched handlers; folding the third class into
   §4 proper is a small, high-value edit for the wave close.
5. **The `_SURFACE` case-sensitivity trap** (§6) is recorded in §21; it
   belongs in the recipe's own import-verification guidance too, since the
   "check every candidate individually" rule is already stated there and this
   is about HOW to check.
6. **Baseline worktree** `.worktrees/w6t3-baseline` (at `bcf0631f7`, own
   `uv venv`) was REMOVED at the end of this task, matching task 2's own
   disposal. Task 4 needs a baseline at `e5e03846a` (the wave-6 start), not
   at this parent, so re-create rather than reuse — and give it its own
   venv; the editable-finder trap makes a shared one resolve the wrong
   tree.
7. **This machine's UI-test noise floor is currently high** — the baseline
   tree alone produced 29 failures in the 5 prompt-heavy files and 34 in the
   nav trio under contention, and individual nodes flipped between runs on
   BOTH trees. Every dismissal in §12 therefore rests on an isolated re-run
   at the parent, not on a count comparison. Task 4's whole-wave sweep
   should budget for this and expect to do the same per-name resolution.
