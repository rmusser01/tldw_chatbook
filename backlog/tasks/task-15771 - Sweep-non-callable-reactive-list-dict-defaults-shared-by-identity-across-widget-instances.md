---
id: TASK-15771
title: Sweep non-callable reactive list/dict defaults shared by identity across widget instances
status: Done
assignee:
  - '@claude'
created_date: '2026-08-13 12:31'
labels:
  - bug
  - textual
priority: medium
---

## Description

Found and confirmed during task-15479 (input-latency burn-down), flagged as
independent of that task's own fix and explicitly out of scope: Textual's
`Reactive._initialize_reactive` installs `default_or_callable() if
callable(...) else default_or_callable` — since a bare `[]` or `{}` literal
is not callable, the exact same list/dict object is installed as the
reactive's backing attribute on every widget instance that has not
explicitly reassigned it. This is the classic Python mutable-default-argument
trap, applied to a Textual `reactive()` default.

The proven instance is `Widgets/TTS/character_voice_widget.py`'s
`characters = reactive([])`: two `CharacterVoiceWidget` instances that both
start from the un-set default and never reassign `characters` share and
cross-mutate the same underlying list. Task-15479 found this only because it
made its own new tests order-dependent within one pytest session (one test's
`.append()` leaked into a later test's row count via the shared default);
its tests work around it defensively by assigning a fresh list before use,
documented in that file's test-module docstring. This is a live production
hazard, not just a test-isolation nuisance: any two instances of the same
widget class that both rely on the declared default will alias state.

## Acceptance Criteria

- [x] `character_voice_widget.py`'s `characters` reactive uses a factory
      (e.g. `reactive(list)` / a `default=` callable) so each instance gets
      its own list, not a shared one
- [x] A test proves two `CharacterVoiceWidget` instances no longer alias:
      mutating one's `characters` does not affect the other's
- [x] A repo-wide sweep for the same shape — `reactive([...])` /
      `reactive({...})` with a literal (not a callable) default — classifies
      each hit and fixes any other widget instantiated more than once per
      app session where cross-instance aliasing is plausible; sites that are
      genuinely singleton-per-app or already reassign the attribute before
      first read are recorded as reviewed, not touched
      (note: ALL 41 hits were converted to callable defaults, not only the
      live ones — the classification below records live vs latent, but the
      "already reassigns before first read" defense turned out to be unsound
      for empty-equal reassignment, see Implementation Notes, so leaving
      latent literals in place would have left loaded guns. The population
      is 41, not the first sweep's 27: the task-15771 review found the first
      sweep structurally blind to the subscripted-generic spelling
      `reactive[list[dict[str, Any]]]([])` — 14 more sites, all in
      `UI/Watchlists_Modules/`, fixed in the review round)
- [x] Existing `character_voice_widget` and STTS test suites stay green

## Implementation Plan

1. AST sweep (not grep) over `tldw_chatbook/` for every `reactive(...)`/`var(...)`
   whose first arg (or `default=`) is a List/Dict/Set literal, a comprehension, a
   module-level shared mutable, or a `list()/dict()/set()` call result. Produce
   the full site table.
2. Classify each hit live vs latent by tracing in-place mutation sites
   (`.append`/`[k]=`/`.update`/`.pop`/`.insert`/`.clear`/`del`/`.sort`) on the
   reactive's value, same-file and cross-file, without an intervening
   reassignment. Key mechanic (verified against installed Textual 8.2.8
   `Reactive._set`): the internal `setattr` only runs when
   `current_value != value`, so reassigning `[]`/`{}` over the pristine shared
   default is a NO-OP and the instance keeps aliasing the class-shared object —
   "reassigns in `__init__`/reset" is not a defense when the value is
   empty-equal.
3. Born-red two-instance leak tests for the most user-facing live cases
   (CharacterVoiceWidget.characters via the real add handler;
   ChapterEditorWidget.chapters; CollectionsTagWindow.selected_keywords) BEFORE
   fixing; run against the pre-fix tree and capture the red.
4. Fix ALL hits to callable defaults (`reactive(list)` / `reactive(dict)` — all
   27 hits use empty literals, so no lambda wrappers needed), checking for any
   code depending on the shared identity first.
5. Permanent guard: the AST sweep as
   `Tests/Architecture/test_reactive_mutable_default_inventory.py` asserting
   zero non-callable mutable reactive defaults in the package; born-red against
   the pre-fix tree.
6. Verify one `recompose=True` site still recomposes after the fix
   (FileListEnhanced/EnhancedStatusWidget), run touched-widget suites +
   `Tests/UI` collect-only sweep + heaviest touched suites, ruff on touched
   files.

## Implementation Notes

Converted every non-callable mutable `reactive()` default in the package — **41
sites, in two rounds** — to a callable factory (`reactive(list)` /
`reactive(dict)`; all 41 were empty literals, so no lambda wrappers were
needed), demonstrated the cross-instance leak born-red on the three most
user-facing live cases before fixing, and pinned the bug class with a
permanent AST inventory guard.

Round 1 (commit `ae4bfb532`) found and fixed 27 sites. The task review then
proved the sweep — and the shipped guard — structurally blind to the
subscripted-generic spelling `reactive[list[dict[str, Any]]]([])`: the
guard's `_call_name()` handled only `ast.Name`/`ast.Attribute`, and
`reactive[T](...)` parses as `Call(func=Subscript(...))`. The review's
runtime probe confirmed the form produces the identical shared-default bug.
Round 2 taught `_call_name` to unwrap `ast.Subscript` (also covering
`Reactive[list]([])`), re-ran the guard born-red against the round-1 tree —
it failed listing **exactly the 14 missed sites**, all in
`UI/Watchlists_Modules/` — and converted all 14.

### The mechanism, including the half that reverses classifications

`Reactive._initialize_reactive` (Textual 8.2.8) installs
`default_or_callable() if callable(...) else default_or_callable` — a literal
`[]`/`{}` is one shared object across all instances. The critical second half:
`Reactive._set` only stores the new value inside
`if always or self._always_update or current_value != value:` — so
**reassigning an empty-equal value (`self.attr = []` over the pristine shared
`[]`) is a complete no-op** and the instance keeps aliasing the shared object.
"Reassigns in `__init__`/reset before use" is therefore NOT a defense:
`ChapterEditorWidget.__init__`'s `self.chapters = chapters if chapters else []`
read as safe and was not (born-red test proved the leak through it).

### Sweep table (site → classification → mutation evidence)

LIVE (in-place mutation of a possibly-shared value reachable):
1. `Widgets/TTS/character_voice_widget.py:143` `characters` — `.append` (469), `.pop` (481); proven leak (task-15479 test order-dependence)
2. `Widgets/TTS/character_voice_widget.py:145` `voice_assignments` — `[k]=` (552/611/678), `del` (484), `.clear()` (496)
3. `Widgets/TTS/chapter_editor_widget.py:129` `chapters` (recompose=True) — `.insert` (362/394), `.pop` (416/432); init's `= []` is equality-skipped
4. `Widgets/CCP_Widgets/ccp_dictionary_editor_widget.py:385` `entries` — `[k]=` (818/835), `del` (849); `= {}` (671) equality-skipped
5. `Widgets/CCP_Widgets/ccp_prompt_editor_widget.py:347` `variables` — `.append` (841), `del` (858); `= []` (644) equality-skipped
6. `Widgets/collections_tag_window.py:134` `selected_keywords` — `.append` (296/311); `= []` equality-skipped
7. `Widgets/settings_theme_editor.py:38` `current_theme_data` — `[k]=` (467/772), `.update()` (815); exposure gated by on_mount's non-empty extract, mutation sites nevertheless present
8. `Widgets/Media/media_viewer_panel.py:582` `all_analyses` — external `.insert(0, …)` from `UI/MediaWindow_v2.py:1947` reachable while the value is the equality-skipped shared default; internal `.pop` (2122)

LATENT (never mutated in place; only ever reassigned or read — fixed anyway,
uniform rule): `UI/Chatbooks_Window.py:60` `chatbooks`,
`UI/Chatbooks_Window_Improved.py:382` `chatbooks` (both mutate a LOCAL list
then assign), `UI/Screens/chat_screen.py:3326` `sidebar_state` (always
reassigned via `dict(...)`; its own comments already document the
equality-no-op), `UI/Screens/chatbooks_screen.py:25` `chatbook_list`,
`UI/Screens/study_screen.py:92` `study_materials`,
`UI/Screens/watchlists_collections_screen.py:528` `overview_data`,
`UI/Watchlists_Modules/overview_pane.py:16` `data` (recompose=True, read-only
in pane), `UI/Wizards/BaseWizard.py:100` `validation_errors` / `:324`
`step_titles`, `ccp_dictionary_editor_widget.py:382` `dictionary_data`
(always `.copy()`d), `ccp_prompt_editor_widget.py:344` `prompt_data`
(mutation at 880 is on a `get_prompt_data()` copy),
`Widgets/Media/media_list_panel.py:158` `items`,
`Widgets/Media/media_viewer_panel.py:575` `search_matches`,
`Widgets/chunk_preview.py:26` `chunks` (`chunk_preview_modal.py`'s
`.append`s are a different class's plain attribute),
`Widgets/dictation_performance_widget.py:109` `metrics_data`,
`Widgets/file_extraction_dialog.py:77` `extracted_files` (element-field
mutation only, list itself never mutated),
`Widgets/file_list_item_enhanced.py:173` `files` (add path reassigns
`self.files + [x]`), `Widgets/media_details_widget.py:47` `search_matches`,
`Widgets/status_widget.py:82` `messages` (assigned from `_messages.copy()`).
Cross-file grep hits on generic names (`activity_log.entries` deque,
`state/chat_state.messages`, `Tamagotchi/tamagotchi_storage.data`,
`voice_bundle_service operation.files`, `world_info_processor.entries`,
`Chat_Dictionary_Lib diagnostics.entries`, `chatbook_creator
content.characters`) were verified to be OTHER classes' plain attributes.

ROUND 2 — the 14 subscripted-generic sites (`reactive[T](...)`) the first
sweep missed, all LATENT: `UI/Watchlists_Modules/article_list.py:257`
`items`, `artifacts_pane.py:655/677/706/788` `briefings`/`presets`/
`scripts`/`citations` + `:732` `scripts_with_audio` (`{}`),
`inspector_pane.py:225` `breadcrumb_labels`, `items_pane.py:76` `items`,
`notifications_pane.py:51` `notifications`, `rules_pane.py:72` `rules`,
`runs_pane.py:70` `runs` + `:79` `run_items`, `sources_pane.py:120`
`sources` + `:174` `watchlist_choices`. Mutation trace over all 14 names
(Watchlists modules + the collections screen, plus package-wide on the
distinctive names) found exactly one in-place candidate —
`runs_pane.py:548` `self.runs[index] = run` — and it sits inside
`for index, candidate in enumerate(self.runs)`, unreachable while the value
is the empty shared default: latent (independently re-derived; matches the
review's finding). No user-facing leak shipped from the missed 14.

No code depended on the shared identity: no `is` comparisons against these
attrs, no class-level default reads (`mutate_reactive(Class.attr)` passes the
descriptor and is unaffected). The only identity-coupled tests in the repo —
`Tests/Watchlists/test_watchlists_pagination.py`'s nine
`assert pane.items is prior_rows` assertions on `ItemsPane.items` (a round-2
site) — pin identity-across-re-renders of assigned non-empty rows, not
identity-with-the-default, and the file passes 34/34 after the conversion
(run, not assumed).

### Born-red evidence

`Tests/Widgets/test_reactive_default_aliasing.py` (new) run against the
pre-fix tree: 4 leak tests FAILED exactly as the bug predicts (a pristine
instance B observed instance A's mutation — e.g.
`assert [...leak...] is not [...leak...]` on two distinct
`CollectionsTagWindow`s) while the recompose-behavior baseline passed;
`Tests/Architecture/test_reactive_mutable_default_inventory.py` (new AST
guard, list/dict/set literals + comprehensions + module-level shared mutables
+ `list()`/`dict()`/`set()` call results, in `reactive`/`var`/`Reactive`
positional-or-`default=` position, bare and subscripted-generic call forms)
was born red twice: round 1 FAILED listing the 27 bare-form sites; round 2,
after the Subscript fix, FAILED against the round-1 tree listing exactly the
14 missed `reactive[T](...)` sites. Post-fix: green at zero. The guard pins
the class at zero for the forms it detects — literals, comprehensions,
module-level shared mutables, and `list()/dict()/set()` call results, in
both call spellings. It does NOT detect shared mutable *instance* defaults
(`reactive(SomeClass())`; 5 known occurrences per the review, e.g.
`reactive(RegionLayout())`), which is the natural next hole — follow-up to
be filed.

### recompose verification

`OverviewPane.data` (`reactive(dict, recompose=True)`) verified end-to-end
post-fix: loading state composed from the per-instance default `{}`,
reassignment recomposes to the dashboard grid
(`test_recompose_reactive_still_recomposes_with_callable_default`).

### Verification

- Round 1: targeted suites over every touched widget/screen (TTS widgets,
  theme editor, media panels/windows, collections tags, chat sidebar-state,
  watchlists overview, full `Tests/Wizards`, study, chatbooks, file
  extraction, processing dashboard, STTS profile library, console speech
  snapshot): 1,721 tests reported passed across four batch runs **in this
  session's specific runs** — honesty caveat from the review (F4):
  `Tests/UI/test_stts_profile_library.py` is a pre-existing flake family
  that the reviewer could NOT get clean on either tree (two recurring ids
  fail on base under load), so "passed" for that file describes these runs,
  not a stable property; it is being filed separately. The review also
  baselined **3 pre-existing `Tests/Architecture` failures at `c03a2fef5`**
  (console_wave6 inventory, persistent-diagnostic inventory, chat_screen
  size ratchet) — same ids fail on base, unrelated to this change, and not
  runs this session's own batches covered.
- Round 2: `test_watchlists_pagination.py` 34/34 (the nine identity
  assertions survive); pane-suite sample over all 8 modules holding the 14
  sites (items/sources/runs/rules/notifications/artifacts panes,
  article_list, filter-in-place, scoped rebuilds, inspector): **333
  passed**; guard + leak tests **6 passed**.
- `Tests/UI` collect-only sweep: 12,532 collected, exit 0.
- Two `test_first_run_wizard_live_contract.py` failures in one batch were
  baselined against a detached worktree at dev `c03a2fef5`: the identical
  batch fails 2 tests of the same file there too — **different test ids** —
  and both pairs pass in isolation on both trees; the file passes 78/78
  standalone on the fixed tree. Pre-existing order/timing flakiness on dev,
  not a regression from this change.
- ruff check clean on all touched files (including fixing a pre-existing
  duplicate `textual.events` import in `settings_theme_editor.py`); the 4
  files `ruff format` would reformat carry pre-existing drift verified
  present at the dev baseline (untouched regions; my lines are format-clean).

### Incidental pre-existing defect (out of scope, not fixed here)

`FileListItemEnhanced.compose` passes `tooltip=` to `Static` (line 126) —
Textual 8.2.8's `Static.__init__` has no such kwarg, so any
`FileListEnhanced` with a non-empty file list crashes on compose. Found when
the first recompose-verification attempt used that widget. Needs its own
task.

### Files

- Fixed (41 sites): see LIVE/LATENT lists above (24 files in round 1 + 8
  Watchlists module files in round 2, under `tldw_chatbook/`).
- New: `Tests/Widgets/test_reactive_default_aliasing.py`,
  `Tests/Architecture/test_reactive_mutable_default_inventory.py`.
- Updated: `Tests/Widgets/test_character_voice_widget.py` (stale workaround
  docstring), `backlog/docs/lessons-textual.md` (equality-skip entry).
