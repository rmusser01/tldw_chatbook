# Wave-4 Task 3 — Skills cleanup (series 3/3)

Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§19 (this task
adds §19). Plan: `Docs/superpowers/plans/2026-09-04-library-decomposition-wave4-skills.md`.
Templates: search+RAG cleanup (§18's Task 4) + collections cleanup (§15).
Prior tasks this series: Task 1 (state PR, `LibrarySkillsState`, 36 fields,
three-way prefix shim), Task 2 (controller PR, `LibrarySkillsController`,
86/127 moved, 41 exclusions, born-governed at 3131 after its own
post-landing review fix round).

## 1. Dynamic-dispatch census (screen + four test roots)

Re-derived Task 2's own dynamic-dispatch findings before deleting the shim,
plus the collections/search+RAG series' `dict.get(...)` → variable →
`setattr`/`getattr` two-step guidance, over `library_screen.py` and all four
named test roots (`Tests/UI`, `Tests/Library`, `Tests/Live`, `Tests/Skills`):

- **Zero** new hazardous dynamic-dispatch shapes touching any of the 36
  Skills state fields.
- **One shared-dispatcher fix, expected shape**:
  `_replace_library_reader_preference`/`_persist_library_reader_preference`'s
  destination dicts had a `"skills": "_library_skills_reader_preferences"`
  string value (and a `"skills": self._library_skills_reader_persistence_
  locks` literal in the `locks` dict) — retargeted to
  `"_skills_state.reader_preferences"` / `self._skills_state.
  reader_persistence_locks`; both already fully generic dotted-vs-flat
  passthroughs via `operator.attrgetter`, so only the string/receiver
  changed, no logic edit. The skills-list choice-strip helper
  (`_library_open_choice_strip`/`_close_open_library_choice_strip`) has an
  IDENTICAL pair (a returned tuple's third element AND a `{flat_name:
  canvas_kind}` dict key) — both updated to `"_skills_state.
  sort_choices_visible"` consistently.
- **The SAME fixture-side shape recurred**, a 4th time in a row, in
  `Tests/UI/test_library_adaptive_reader_closeout.py`'s own
  `DESTINATION_CONTRACT` dict — the `"skills"` entry's two flat strings
  retargeted to `"_skills_state.reader_preferences"`/`"_skills_state.
  reader_layout"`, matching the collections/conversations/search+RAG
  precedent exactly (every consumer already goes through
  `operator.attrgetter`, so zero consumption-site changes needed).
- **A genuinely new finding, caught by re-reading the transformed line, not
  by a test failure**: `_library_list_canvas_showing_list`'s
  `getattr(self, "_library_skills_view", "list")` call. Retargeting only
  the STRING (`"_library_skills_view"` → `"_skills_state.view"`) would have
  been silently wrong — `getattr` performs a single attribute lookup, not
  dotted-path traversal, so `getattr(self, "_skills_state.view", "list")`
  would have permanently returned the literal default `"list"` forever, no
  exception anywhere (the exact "unbound-attribute escape" shape recipe
  §3's sixth bypass class already names, found here on the SCREEN side
  rather than inside a moved controller body). Fixed by changing the
  RECEIVER too: `getattr(self._skills_state, "view", "list")`.

## 2. Screen-side retarget

**130 pre-existing flat-name occurrences** in `library_screen.py` (`ast`-
derived field list from `LibrarySkillsState`, word-boundary regex census,
not a `startswith` shortcut): 121 literal `self.<flat_name>` attribute
accesses + 5 dotted-vs-flat dispatch-dict string values (the two reader-
preference dispatcher methods + the choice-strip helper's dict/tuple pair,
above) + 2 prose-comment mentions + 1 `getattr` call needing the receiver
fix (above) + the shim block's own internal comments/code (deleted
wholesale, not retargeted). A single per-field regex pass (36-field
mapping, longest-match-first via word boundaries, so `_library_skill_
detail` never partial-matches `_library_skill_detail_generation`) handled
the 121 attribute accesses + 5 string values mechanically; the `getattr`
receiver fix and the 2 comment rewords were done by hand. Re-verified
afterward with a zero-remaining-occurrence grep for all 36 flat names over
the whole file (the 1 hit that remains is this task's own explanatory
comment at the shim-deletion site, naming the old field prefixes in past
tense — the same retained-history-comment shape every prior cleanup PR
leaves behind).

This included THREE `__init__` lines new to this series (beyond the usual
reader-preferences/layout/locks trio every subsystem's cleanup retargets):
`editor_mode` and `reader_mode` both keep their ORIGINAL `__init__` lines
(Task 1's own "forced-early-construction-point" finding — the state object
must construct before these two fields' own lines run, which sit AFTER
that point positionally, not because of cross-subsystem entanglement like
the trio) — both retargeted from `self._library_skill_<field> = ...` to
`self._skills_state.<field> = ...` in this pass, exactly like the trio.

## 3. Test retarget — three roots, 269 retargets, 28 fixture restructurings

Repo-wide census across all FOUR named roots (`Tests/UI`, `Tests/Library`,
`Tests/Live`, `Tests/Skills`) found flat-name consumers in 9 files spanning
two roots — `Tests/Live` had zero (confirmed directly against its own
`test_library_adaptive_reader_closeout.py`, a 7390-line file that mentions
"skill" 30 times but never one of the 36 flat field names — verified by
reading what those 30 mentions actually are: test-name/coverage-inventory
string literals, not attribute accesses); `Tests/Library` had exactly 1
hit, a pre-existing PROSE comment in `test_library_shell_state.py`
referencing the concept by name (not a code reference to a `LibraryScreen`
attribute at all — that file tests an unrelated `library_shell_state.py`
helper), left untouched as out of scope:

| File | Retargets | Notes |
|---|---|---|
| `Tests/UI/test_library_skills_canvas.py` | 162 | 25 `SimpleNamespace(...)` fixture blocks restructured (flat kwargs → nested `_skills_state=SimpleNamespace(...)`), plus every post-construction attribute read/write/assertion |
| `Tests/UI/test_library_skills_reader.py` | 25 | real-screen attribute accesses only |
| `Tests/UI/test_library_adaptive_reader_closeout.py` | 16 | 14 `screen.` accesses + the 2 `DESTINATION_CONTRACT` dict strings (§1 above) |
| `Tests/UI/test_library_entry_compose_once.py` | 6 | real-screen accesses (trust-posture tests) |
| `Tests/UI/test_screen_navigation.py` | 5 | real-screen accesses |
| `Tests/UI/test_library_canvas_scoped_sync.py` | 3 | 1 `SimpleNamespace` block (2 kwargs) + 1 post-construction assertion |
| `Tests/UI/test_library_choice_strips.py` | 2 | real-screen accesses |
| `Tests/UI/test_library_shell.py` | 2 | real-screen accesses, inside a shared multi-parametrization helper |
| `Tests/Skills/test_skills_library_flow.py` | 46 | real-screen accesses (fourth-root trap the wave's own task 2 already found once) |
| `Tests/Skills/test_skills_import.py` | 2 | 2 `SimpleNamespace` blocks (1 kwarg each) |

**269 retargets total, zero assertion VALUE changes** — every edit is a
receiver-path rewrite only (`screen._library_skill_<field>` /
`fake._library_skill_<field>` → `<receiver>._skills_state.<field>`, or a
flat kwarg restructured into a nested `_skills_state=SimpleNamespace(...)`
kwarg), confirmed by running the full affected-file battery before and
after (§8 below) and diffing the pass/fail set.

**The fixture-restructuring scale is new to this series**: 27 of the 86
movers are unbound-fake-self exclusions (Task 2's own census, roughly
triple export's prior 9-of-51 record), and a large fraction are tested via
`SimpleNamespace(...)` fakes carrying flat skills kwargs directly. 28
separate call sites needed restructuring (25 in `Tests/UI/test_library_
skills_canvas.py`, 1 in `Tests/UI/test_library_canvas_scoped_sync.py`, 2
in `Tests/Skills/test_skills_import.py` — verified with `grep -c
'_skills_state=SimpleNamespace(' <file>` against the finished files, not
recalled from memory). Handled with a small, generic,
line-oriented script (collect every `<flat_name>=<value>,` kwarg line
inside a `SimpleNamespace(` call block — regardless of whether the matched
kwargs are contiguous with unrelated kwargs — and re-emit them as one
`_skills_state=SimpleNamespace(<field>=<value>, ...)` kwarg at the first
match's position), verified by `ast.parse` before writing and a full
pytest run after, rather than done by hand one call site at a time.

## 4. Delegator census — 70 KEEP, 16 PRUNED (~19%)

Of the 86 moved names: **30 `@on` handlers KEEP unconditionally**. Of the
remaining 56 (55 plain + 1 `@staticmethod`, `_restore_library_skills_scope`
— KEEP, real screen-resident caller), a repo-wide grep for `\bname\b`
(matching EVERY occurrence shape — call sites, kwarg names, string
literals — not just bare-word occurrences) across `tldw_chatbook/` and all
four test roots, excluding the controller's own internal calls and the
screen delegator's own 2-line body, found:

- **40 with a genuine external caller**: an excluded, still-screen-resident
  method calling `self.<name>()` (e.g. `handle_library_skills_trust_action`
  calling `_begin_library_skill_trust_setup`/`_unlock_library_skill_trust`/
  `_refresh_library_skills_trust_posture`/`_open_first_blocked_skill`), or a
  test that calls/patches the screen delegator directly on a real instance.
- **16 with zero references anywhere outside their own delegator body and
  the controller's own internal calls**: `_apply_library_skill_detail`,
  `_apply_library_skill_detail_failure`, `_bootstrap_library_skill_trust`,
  `_build_library_skill_tool_catalog`, `_claim_library_skill_detail_
  generation`, `_do_library_skill_trust_reset`, `_focus_library_skills_
  page_control`, `_library_skill_text_fields_match_state`, `_load_library_
  skill_script_grant`, `_mark_library_skill_dirty`, `_read_library_skill_
  editor_fields`, `_read_library_skill_live_name`, `_revoke_library_skill_
  script_grant`, `_setup_library_skill_trust`, `_sync_library_skill_
  description_hint`, `_update_library_skill_toggle_buttons`. Each
  double-checked directly (`grep -c` on `library_screen.py` showed exactly
  2 occurrences — the `def` line + the `return self._skills_controller.
  <name>(...)` line — before deletion; 0 after).

**A methodology bug in this census's own first draft, caught before acting
on it**: the first grep script used a negative lookbehind
(`(?<![\w.])name\b`) meant to suppress noise from bare kwarg names/string
literals, which silently ALSO suppressed every `self.<name>(`/
`<receiver>.<name>(` call-site match (since `self.` ends in a `.`, a
lookbehind character the pattern explicitly excluded). This produced a
FALSE "zero external callers" reading for `_begin_library_skill_trust_
setup`, when `handle_library_skills_trust_action` (screen-resident,
excluded, still full-bodied) calls `self._begin_library_skill_trust_
setup()` directly at its own line 27022. Caught by manually checking one
suspicious hit (a test fixture overriding the name with a lambda, which
only makes sense if something calls it) before trusting the flawed
census's output, not by any test failure — the mechanically pruned version
would have broken `handle_library_skills_trust_action`'s "setup"/"resetup"
branch at runtime with no test catching it (no test drives that branch
through the real screen delegator; the excluded method's own coverage
mocks the target out). Rewritten to match `\bname\b` everywhere (all
receiver shapes) and classify hits by file/context afterward, not filter
them out of the pattern; re-run against ALL 56 candidates before finalizing
the 16-name prune list.

**41-of-127 exclusion count → 16-of-86 prune fraction (~19%)**: consistent
with recipe §15 lesson 3's inverse relationship (a LARGER controller-PR
exclusion count keeps MORE screen delegators alive, since an excluded,
screen-resident sibling method calling its moved neighbor is exactly what
keeps a delegator's reference count above zero) — skills' 41 exclusions
(the largest of any series so far) produced the SECOND-SMALLEST prune
fraction (below collections' 22%/conversations' 30%/search+RAG's 29%,
above only export's 5%).

## 5. Shim block deletion

The Task-1-generated `_library_skill_<field>`/`_library_skills_<field>`/
`_selected_skill_name` property-shim loop (36 fields, module end) deleted
wholesale once §2-§4's census confirmed zero remaining consumers anywhere
outside `LibrarySkillsController`'s own PERMANENT generated shim loop
(Task 2, untouched — controller shims stay, per the task brief).

## 6. Import verification — full AST-derived candidate list

**28 dead imports removed**, each independently confirmed single-occurrence
via an `ast.Name`-usage-count script (not a bare grep — annotations,
default values, and nested local imports all counted), then checked
against `Tests/Architecture/test_library_support_layer_surface.py`'s
`_SURFACE` re-export contract before deletion:

- **1 newly dead from this task's own shim deletion**: `skill_state_
  shim_attr` (the three-way prefix resolver; still live in the
  controller's own permanent shim + the wiring test).
- **27 left dead by Task 2's own controller move**, deliberately deferred
  to this cleanup PR (export/collections/search+RAG precedent): 15 from
  `Widgets.Library` (`SKILL_DISCARD_TOOLTIP_CLEAN`, `SKILL_DISCARD_
  TOOLTIP_DIRTY`, `next_skill_context`, `skill_context_toggle_label`,
  `skill_disable_model_label`, `skill_script_grant_line`, `skill_trust_
  approve_tooltip`, `skill_trust_panel_remediation_copy`, `skill_trust_
  review_enabled`, `skill_trust_review_preview`, `skill_trust_review_
  tooltip`, `skill_trust_state_line`, `skill_trust_unlock_enabled`,
  `skill_trust_unlock_tooltip`, `skill_user_invocable_label`); 10 from
  `Library.library_skills_state` (`DEFAULT_SKILL_BROWSE_PAGE_SIZE`, `MAX_
  SKILL_BROWSE_PAGE`, `SkillEditorState`, `build_skill_editor_state`,
  `classify_skill_save_error`, `compose_skill_markdown`, `reconcile_
  skill_allowed_tools`, `skill_allowed_tools_sequence`, `skill_invocation_
  copy`, `skill_review_identity_line`); 2 from `.skills_screen`
  (`SkillTrustBootstrapModal`, `SkillTrustPassphraseModal`).

27 of the 28 individually confirmed still LIVE inside `library_skills_
controller.py` (already independently re-imported there by Task 2 — the
"free-name walk" its own report documents) before removal from the screen.
The one exception, `SkillEditorState`, is NOT used in the controller at
all (`grep -c` returns 0) — it is live in `library_skills_state.py` (the
`UI/Library_Modules` state object, via the `editor_state: SkillEditorState
| None` field annotation) and in `Widgets/Library/library_skills_
canvas.py` (three signatures), both already-independent imports unrelated
to the controller's own move. The removal from `library_screen.py` was
still correct (0 occurrences there beyond the import line either way);
only the "where is it still live" attribution needed correcting.
3 SKILLS constants ARE `_SURFACE`-pinned (`LIBRARY_SKILL_TEXT_MAX_CHARS`,
`LIBRARY_SKILL_DIRTY_VETO_COPY`, `LIBRARY_SKILL_SAVE_STATUS_COPY`) —
individually re-checked and left untouched. One `Widgets.Library` name
(`skill_editor_warning_lines`) was individually re-checked despite sharing
its import block with 15 pruned siblings and confirmed still live via a
non-`_SURFACE` screen-resident consumer — not removed (mirrors the
collections series' own `CaptureIdentity`/`CollectionsCaptureReader
Presentation` "checked individually, not assumed dead-by-association"
precedent).

## 7. Docstring updates

Two inaccuracies in `library_skills_controller.py`'s own MODULE docstring
(not a moved method body — freely editable, no byte-for-byte canon
deferral) fixed:

1. An arithmetic error inherited unfixed from Task 1's own report into the
   controller's docstring ("the 6-match gap is three `@property`/
   `@x.setter` pairs" — should be SIX; Task 2's own post-landing review fix
   round corrected the identical error in its OWN report text but never
   propagated the fix into this docstring copy).
2. A now-false claim ("`LibraryScreen` keeps one-line delegators under
   every one of these original names") — corrected to name the 70-of-86
   count and point at `_SKILLS_CLUSTER_SCREEN_DELEGATOR_PRUNED`.

**Forward note, not fixed here**: `library_rag_search_controller.py`
(search+RAG) carries the IDENTICAL stale claim in its own module docstring
(12 of its own 42 movers were pruned by that series' own cleanup task, and
the claim was never corrected there) — confirmed by reading the file.
Recorded in recipe §19 as an open item for a future pass through that file,
not silently treated as license to leave this series' own copy stale too.

Two comment-prose corrections in `library_screen.py` itself (screen-
resident code, not moved-body — always freely editable): a comment
referencing `_library_skill_dirty` in prose, and two referencing
`_selected_skill_name` in prose, reworded to the dotted form. Two similar
prose corrections in test docstrings (`Tests/UI/test_library_skills_
canvas.py`, `Tests/Skills/test_skills_library_flow.py`).

## 8. Fresh measurements + pins

- **Screen** (`test_screen_size_ratchet.py`): `41247/1311 → 41155/1295`
  (16 fewer `FunctionDef`s — exactly the 16 pruned delegators, pure
  deletion, no replacement).
- **Controller** (`test_library_modules_size_ratchet.py`, born-governed):
  `3131 → 3140` (comment-only growth — the two module-docstring
  corrections; zero method bodies touched, mover/exclusion counts
  unchanged at 86/41).
- **Full pin trajectory, skills series**: `43225/1311 (pre-task-1) →
  43179/1311 (task 1) → 41247/1311 (task 2) → 41155/1295 (task 3, final)`.

## 9. Verification battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`.

- **`test_library_skills_wiring.py`**: 8/8 passed (9 → 8: the deleted
  shim-surface test is the only count change).
- **All five wiring suites + 3 characterization files + support-layer
  surface**, combined single run: **51 passed** (`test_library_skills_
  wiring.py` 8, `test_library_collections_wiring.py` 4, `test_library_
  conversations_wiring.py` 6, `test_library_export_wiring.py` 5,
  `test_library_search_rag_wiring.py` 8, `test_library_collections_
  characterization.py` + `test_library_conversations_characterization.py`
  + `test_library_export_characterization.py`, `test_library_support_
  layer_surface.py` 8).
- **Both size guards, full suite**: 32 passed, 2 failed — both the
  documented pre-existing `chat_screen.py` ratchet rows (recipe §7's
  standing list), unrelated to this diff.
- **`-k "skill and library"` sweep** (`Tests/UI`+`Tests/Library`, single
  process, final tree): **10 failed, 272 passed, 22073 deselected**. All 10
  failures match Task 1/2's own already-documented pre-existing bucket
  name-for-name: `test_palette_library_skills_command_opens_hidden_
  starter_route` (command-palette test), `test_skill_editor_production_
  geometry_contains_basic_and_advanced_workflows[size0]`, `test_library_
  skill_row_class_matches_prompt_row_visual_parity`, `test_library_
  skills_header_filter_empty_have_css_blocks`, `test_library_skill_name_
  input_css_blocks_match_prompt_name_parity`, `test_library_skills_
  import_row_css_blocks_match_prompt_parity`, `test_library_skill_trust_
  setup_explanation_css_block_matches_review_files_parity` (CSS-block/
  geometry-parity bucket), `test_action_library_skill_back_honors_dirty_
  guard`, `test_skills_route_lands_on_library_with_skills_row_selected`,
  `test_shadow_name_set_stays_in_sync_with_real_sources`. Zero new
  failures; 1 more passed than Task 2's own 271 (`test_library_skills_
  manual_items_priority_survives_compact_layout_sync` flipped to pass this
  run — Task 1 already characterized this exact name as order-dependent
  xdist-adjacent noise).
- **`Tests/Skills/` full run** (fourth root): **537 passed, 2 failed** —
  EXACT match to Task 1/2's own documented baseline (`test_import_real_
  superpowers_skills_lands_trust_pending`, environment-dependent;
  `test_uninitialized_trust_shows_setup_state_and_bootstrap_enables_
  approve_flow`, confirmed pre-existing by both prior tasks). Zero new
  failures.
- **Full sequential xdist paired-baseline sweep** (`Tests/UI -k "library"
  -p no:randomly -q -n 8 --dist worksteal`, branch then a `git stash -u`
  pristine baseline of the same pre-task tree, run SEQUENTIALLY per §7's
  own "concurrent runs amplify flakiness" lesson -- both IN-PLACE in this
  worktree's own venv, under sustained heavy CONCURRENT machine load from
  several unrelated long-running pytest processes already active on this
  machine for hours, confirmed via `ps aux` before and after): **branch
  367 failed/3937 passed (1930.5s) vs. baseline 370 failed/3934 passed
  (1949.1s)**, both inside the documented ~330-371 historical backdrop
  (recipe §7) despite the elevated absolute counts this run's heavier-than-
  usual concurrent load produced. 361 shared, 9 baseline-unique (noise in
  the opposite direction, not investigated further per precedent), **6
  branch-unique**: 3 (`test_library_media_trash.py::test_media_trash_
  commit_unknown_blocks_back_but_refresh_can_be_abandoned`, `test_library_
  prompt_collections.py::test_library_screen_membership_load_retry_and_
  apply_retry_are_distinct`, `test_library_prompts_canvas.py::test_
  library_prompt_pager_first_and_filter_failure_states[size0]`) passed
  cleanly on a combined single-process re-run (ordinary xdist noise); the
  remaining 3 (`test_screen_navigation.py::test_generic_reentry_returns_
  to_library_landing`, `test_study_origin_navigation.py::test_home_
  origin_does_not_leak_into_later_library_entry`, `test_study_origin_
  navigation.py::test_library_origin_study_round_trip_still_returns_to_
  library`) reproduced in the combined re-run and were individually
  investigated in depth (all three fail with the SAME "app never finished
  pushing its initial screen" timeout signature -- a generic Textual-app-
  startup race, not anything skills-specific): a fresh `git stash -u` to
  the SAME pristine pre-task tree reproduced 2 of the 3 immediately (`test_
  home_origin_does_not_leak_into_later_library_entry`, `test_library_
  origin_study_round_trip_still_returns_to_library`); the third (`test_
  generic_reentry_returns_to_library_landing`) initially passed once on
  that baseline stash (combined with the other two), so it was re-run 3x
  in ISOLATION on each tree to settle it -- 3/3 failures on the branch, 3/3
  failures on the SAME pristine baseline tree, both with the identical
  timeout signature. **Zero unexplained branch-unique failures** -- none
  of the 6 touches Skills code or this task's diff, and every one that
  reproduced at all reproduces identically on the pristine pre-task tree
  under the same (heavily loaded) conditions.
- **preflight**: `./scripts/preflight.sh` — all six checks green (no
  diagnostic-inventory drift; this task's diff touches zero
  `logger.warning`/persistent-diagnostic call sites).

## 10. Files changed

- `tldw_chatbook/UI/Screens/library_screen.py`: shim block (36 properties)
  deleted; 130 flat-name occurrences retargeted/reworded; 16 dead delegator
  methods (32 lines) deleted; 28 dead imports removed.
- `tldw_chatbook/UI/Library_Modules/library_skills_controller.py`: 2
  module-docstring corrections (comment-only, +9 lines); zero method
  bodies touched (86 movers, 41 exclusions unchanged).
- `Tests/UI/test_library_skills_canvas.py`,
  `Tests/UI/test_library_skills_reader.py`,
  `Tests/UI/test_library_adaptive_reader_closeout.py`,
  `Tests/UI/test_library_entry_compose_once.py`,
  `Tests/UI/test_screen_navigation.py`,
  `Tests/UI/test_library_canvas_scoped_sync.py`,
  `Tests/UI/test_library_choice_strips.py`,
  `Tests/UI/test_library_shell.py`,
  `Tests/Skills/test_skills_library_flow.py`,
  `Tests/Skills/test_skills_import.py`: 269 retargets total (28
  `SimpleNamespace` fixture blocks restructured; the rest are receiver-path
  rewrites), zero assertion value changes.
- `Tests/Architecture/test_library_skills_wiring.py`: shim-surface test
  deleted; `_SKILLS_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset (16 names)
  added; `test_screen_delegates_skills_handlers` updated to skip/assert-
  absence for the 16; module docstring rewritten for the finished 3-task
  series.
- `Tests/Architecture/test_screen_size_ratchet.py`,
  `Tests/Architecture/test_library_modules_size_ratchet.py`: `_BUDGETS`
  rows re-pinned, dated comments added.
- `backlog/docs/library-decomposition-recipe.md`: new §19 ("The skills
  series, as landed").

Commits: `ed4c29d45` (skills cleanup — shims out, ratchet lowered),
`2a744c434` (blame-ignore follow-up).

## 11. Self-review

- **Two methodology bugs in this task's own automation were caught and
  fixed BEFORE acting on their output, not after a test failure exposed
  them** — the honest headline of this task, not a footnote, mirroring
  Task 2's own "the two regression rounds are the honest headline"
  self-review framing:
  1. The `getattr(self, "_library_skills_view", "list")` receiver bug
     (§1) — caught by re-reading the transformed line before running any
     test. This shape is specifically documented (recipe §3's sixth
     bypass class) as one that does NOT reliably produce a failing test,
     so catching it required deliberate reading, not trust in the
     battery.
  2. The delegator-census negative-lookbehind bug (§4) — caught by
     manually verifying one suspicious census result (a test fixture
     overriding a name with a lambda, which only makes sense if something
     calls it) before trusting the census's "zero callers" output for the
     other 55 names. Had this gone unnoticed, `_begin_library_skill_
     trust_setup`'s delegator would have been pruned while a real,
     screen-resident caller (`handle_library_skills_trust_action`) still
     needed it — a genuine runtime regression with no test in the
     existing suite positioned to catch it.
- **The delegator census's own regex is now flagged as its own hazard
  surface** (recipe §19 lesson 2) — worth re-checking on every future
  subsystem's cleanup PR, not just trusted because it is "the mechanical
  step."
- **Every retarget was verified with a fresh, whole-file, zero-remaining-
  occurrence grep**, not assumed complete from the transform script's own
  reported count — both for the screen (130 → 1, the task's own retained-
  history comment) and for each of the 9 test files (all → 0 net-new,
  after accounting for this task's own explanatory comments).
- **One acknowledged scope decision**: this task fixed its OWN copy of the
  stale "keeps one-line delegators under every one of these" module-
  docstring claim but left the search+RAG controller's identical, equally
  stale copy untouched (out of this task's own file scope) — recorded as
  an explicit forward note in the recipe rather than silently accepted as
  "a prior series left it stale, so this one can too," and not silently
  fixed either (which would have been outside this task's own diff
  boundary for an unrelated, already-landed series).
