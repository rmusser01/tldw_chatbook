# Wave-4 Task 1 — Skills state PR (series 1/3)

Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§18. Plan:
`Docs/superpowers/plans/2026-09-04-library-decomposition-wave4-skills.md`.
Templates: `library_rag_search_state.py` / `Tests/Architecture/
test_library_search_rag_wiring.py` (search+RAG series, the newest prior
rehearsal — its Task 2/state-PR-era commits, `315cd4c3c`/`77750c85d`, are
the exact shape this task's own two commits mirror).

## 1. Cluster enumeration (both prefix families, no `startswith` shortcuts)

`ast` walk of `LibraryScreen` for method names containing `"skill"`
(case-insensitive): **133 raw `FunctionDef` matches** (matches the plan's
2026-09-04 measure exactly), **127 unique names** — the 6-match gap is
three `@property`/`@x.setter` pairs (`_library_skills_import_open`,
`_library_skills_import_path`, `_library_skills_import_status`,
`_library_skills_import_review_name`, `_library_skills_import_in_flight`,
`_library_skills_import_generation`), all six of them projections onto
`_library_skill_import_coordinator` (the WIRING field, §2), not
`__init__`-scoped state.

Of the 127 unique names: **47 carry a distinct `@on` decorator** (one,
`handle_library_skill_input_changed`, carries two — see §4), **2** are
`action_*` handlers, the rest are plain methods, `@property`/`@staticmethod`
helpers, or the delegator-shaped `_library_skills_import_*` pairs above.
No Prompts/Media/other-subsystem false positives were found in the
skill-cluster method names themselves (unlike collections' own 3
`handle_library_prompts_collection`-family false positives) — this task's
scope is the **state** PR only, so no method-body-ownership census beyond
what field-ownership required was run; the full cluster-to-controller
ownership census (which methods move where, and any false positives in
the raw match) is Task 2's job.

## 2. Field ownership (recipe §2 script, `_library_skill`/`_library_skills` prefixes)

Running the recipe's own script with `f.startswith(("_library_skill",
"_library_skills"))` found **37** fields. Re-deriving without the
`startswith` shortcut — matching `"skill" in f.lower()` against the full
`__init__`-scoped field set — found **38**, +1:
`_selected_skill_name` (no `_library_` prefix at all). This is the
conversations exemplar's own recipe §11 "startswith enumeration trap"
lesson, reproduced here on a THIRD prefix shape rather than a field
missed within a known prefix. A full class-level `AnnAssign` scan found
zero additional skill-owned class-level-only attributes (matching the
collections precedent's own clean result).

### Full ownership table (38 fields)

| Field (original attr) | Non-skill users (census) | Verdict | Treatment |
|---|---|---|---|
| `_library_skill_import_coordinator` | `compose_content`, `_select_library_rail_row_after_source_admission` (shell) | **WIRING** | stays untouched — capture-controller precedent (plan's own ruling) |
| `_library_skills_browse_controller` | `on_mount`, `on_unmount`, `save_state`, `restore_state`, `_library_continue_receipt_for_current_route`, `_select_library_rail_row_after_source_admission` (all shell) | **WIRING** | stays untouched — SAME capture-controller precedent, established here for its sibling (holds the live `LibrarySkillsBrowseController` instance) |
| `_library_skill_choice_presented_generation` | NONE | MOVE | static (`-1`), line deleted |
| `_library_skill_confirming_delete` | `_library_emergency_return_eligibility`, `compose_content` (shell) | MOVE | static (`False`), deleted |
| `_library_skill_conflict` | `_library_emergency_return_eligibility`, `compose_content` (shell) | MOVE | static (`False`), deleted |
| `_library_skill_detail` | NONE | MOVE | static (`None`), deleted |
| `_library_skill_detail_error` | NONE | MOVE | static (`""`), deleted |
| `_library_skill_detail_generation` | NONE | MOVE | static (`0`), deleted |
| `_library_skill_detail_loading` | NONE | MOVE | static (`False`), deleted |
| `_library_skill_detail_retryable` | NONE | MOVE | static (`False`), deleted |
| `_library_skill_dirty` | `_library_emergency_return_eligibility`, `compose_content` (shell) | MOVE | static (`False`), deleted |
| `_library_skill_editor_armed` | NONE | MOVE | static (`False`), deleted |
| `_library_skill_editor_mode` | NONE (own-init only) | MOVE | **COMPUTED** (`coerce_skill_editor_mode(config)`) — original line kept, routes through shim |
| `_library_skill_editor_state` | `compose_content` (shell) | MOVE | static (`None`), deleted |
| `_library_skill_more_actions_open` | `_library_route_shortcuts_for_current_state`, `_library_emergency_return_eligibility` (shell) | MOVE | static (`False`), deleted |
| `_library_skill_mutation_in_flight` | `_library_emergency_return_eligibility` (shell) | MOVE | static (`False`), deleted |
| `_library_skill_original_name` | NONE | MOVE | static (`""`), deleted |
| `_library_skill_reader_mode` | NONE | MOVE | **COMPUTED** (`coerce_skill_reader_mode(None)`) — original line kept |
| `_library_skill_script_grant` | NONE | MOVE | static (`False`), deleted |
| `_library_skill_scroll_pending` | NONE | MOVE | static (`False`), deleted |
| `_library_skill_status` | `compose_content` (shell) | MOVE | static (`""`), deleted |
| `_library_skill_tool_captured` | NONE | MOVE | static (`()`), deleted |
| `_library_skill_tool_catalog` | NONE | MOVE | static (`()`), deleted |
| `_library_skill_tool_filter` | NONE | MOVE | static (`""`), deleted |
| `_library_skill_tool_picker_changed` | NONE | MOVE | static (`False`), deleted |
| `_library_skill_trust_confirming_reset` | `compose_content`, `_select_library_rail_row_after_source_admission` (shell — rail-switch-touched, the plan's own named "census decides" case) | MOVE (shell/plumbing only; ≥2-subsystems rule does not fire — no OTHER subsystem's own method reads it) | static (`False`), deleted |
| `_library_skill_trust_details_open` | NONE | MOVE | static (`False`), deleted |
| `_library_skill_active_review` | NONE | MOVE | static (`None`), deleted |
| `_library_skills_filter` | `restore_state`, `compose_content` (shell) | MOVE | static (`""`), deleted |
| `_library_skills_filter_cursor_context` | `_select_library_rail_row_after_source_admission` (shell) | MOVE | static (`None`), deleted |
| `_library_skills_reader_layout` | `compose_content` (shell); `_toggle_library_media_reader_pane` — **reclassified**: reads as Media-owned by name, but its body is a shared, multi-subsystem pane-toggle dispatcher (Collections/Conversations/Notes/Prompts/Skills/Media branches) with a genuine Skills branch, confirmed by reading the body, not the name | MOVE (shell/plumbing) | **ENTANGLED trio** — original line kept |
| `_library_skills_reader_persistence_locks` | `_persist_library_reader_preference` (shell) | MOVE | **ENTANGLED trio** — original line kept |
| `_library_skills_reader_preferences` | `request_library_reader_layout_refresh` (shell) | MOVE | **ENTANGLED trio** — original line kept |
| `_library_skills_sort` | `restore_state`, `compose_content` (shell) | MOVE | static (`"name"`), deleted |
| `_library_skills_sort_choices_visible` | `compose_content`, `_library_open_choice_strip` (shell) | MOVE | static (`False`), deleted |
| `_library_skills_trust_posture` | `compose_content` (shell) | MOVE | static (`""`), deleted |
| `_library_skills_view` | `on_mount`, `restore_state`, `_library_entry_route_key`, `compose_content`, `_library_continue_receipt_for_current_route`, `_select_library_rail_row_after_source_admission`, `_library_open_choice_strip` (all shell) | MOVE | static (`"list"`), deleted |
| `_selected_skill_name` | `restore_state`, `_library_entry_route_key`, `compose_content` (shell) + many skill-cluster methods (own) | MOVE | static (`""`), deleted |

**Totals: 2 WIRING, 31 static-deleted, 3 entangled-trio-kept, 2 computed-kept = 38.**
Zero fields BLOCKED by the ≥2-subsystems rule — every non-skill consumer
resolved to shell/plumbing on inspection, including the two the plan
flagged by name (`_library_skill_trust_confirming_reset`,
`_library_skills_reader_layout`'s `_toggle_library_media_reader_pane`
consumer).

### Existing-controller delegation note

`library_skill_import_controller.py` (`LibrarySkillImportCoordinator`) and
`library_skills_browse_controller.py` (`LibrarySkillsBrowseController`)
are untouched, per the plan. This task moved zero method bodies (state PR
only), so no delegation-to-existing-controller exclusion census was
needed yet — every skill-cluster method still lives, unmoved, on
`LibraryScreen`. Task 2 (controller move) is where methods delegating to
either existing controller become exclusion candidates.

## 3. A new wrinkle: three prefixes, not two

The plan's own framing ("the two-prefix mapping... the conversations
plural-set pattern is the precedent for mixed prefixes") undercounted by
one. The actual census needs a three-way split:

- `_library_skill_` (singular, DEFAULT) — 26 fields.
- `_library_skills_` (plural) — 9 fields (`sort`, `filter`,
  `filter_cursor_context`, `view`, `sort_choices_visible`,
  `trust_posture`, plus the reader-layout trio).
- bare `_` (no "skill(s)" word at all) — 1 field (`selected_skill_name`).

`skill_state_shim_attr(name)` in `library_skills_state.py` is the single
function resolving all three (not two independent frozensets each hand-
checked inline) — used by both the screen's generated shim loop and this
task's wiring test, closing the drift risk the conversations exemplar's
own task-8 fix round found (two independent copies of a two-name set
silently diverging).

## 4. The construction-ordering wrinkle (generalizing the entangled-trio mechanism)

`self._skills_state` must be constructed **before** the shared reader-
preferences tuple-unpack (line 2392 at this task's measurement) — the
same ordering constraint the collections series' own state PR documents
for its identical trio. That forces construction to the SAME early point
`self._collections_state` is (right after it, well before the bulk of the
skills fields at lines ~2913–3051).

This is EARLIER than two more fields' own original lines:
`_library_skill_editor_mode` (line ~2913, computed via
`coerce_skill_editor_mode(library_config...)`) and `_library_skill_reader_
mode` (line ~2919 pre-edit, computed via `coerce_skill_reader_mode(None)`).
Neither is entangled with another subsystem's init code — but because
construction has already happened by the time their lines run, the
recipe's usual "computed defaults become constructor arguments" rule
cannot apply (there is no later constructor call to pass them into). The
only behaviorally-transparent option is the SAME mechanism the entangled
trio already uses: leave the original line completely untouched, let the
newly-installed property shim silently route the assignment into
`self._skills_state`. This generalizes the trio's own mechanism (recipe
described it as being ABOUT entanglement) to: *any field whose original
position falls after the forced-early construction point needs this
treatment, entangled or not* — confirmed safe here because every field
BEFORE that point (only `_library_skill_import_coordinator`, WIRING, and
`_library_skill_choice_presented_generation`, static) needed no such
accommodation.

## 5. Characterization spot-check — four roots, genuinely-unpressed `@on` handlers

Per-handler CSS-selector-level census (not a same-line method-name grep,
which the collections series' own report already warned undercounts —
confirmed again here: several handlers' selectors are ONLY referenced via
widget-existence/disabled-state assertions on a bare `_EditorHost`/
`_CanvasHost` widget host with no real `LibraryScreen` behind it, which a
method-name grep cannot distinguish from genuine press coverage) across
`Tests/UI`, `Tests/Library`, `Tests/Live`, and `Tests/Skills` (the wave's
own named fourth-root trap) for all 47 `@on` handlers + 2 `action_*`
handlers.

**6 genuinely unpressed anywhere, all pinned, all confirmed PASSING
pre-change:**

| Handler(s) | New test | File |
|---|---|---|
| `handle_library_skills_trust_reset_cancel` | `test_trust_reset_cancel_backs_out_without_touching_trust_state` | `Tests/Skills/test_skills_library_flow.py` |
| `handle_library_skills_trust_reset_confirm` | `test_trust_reset_confirm_wipes_trust_state` | `Tests/Skills/test_skills_library_flow.py` |
| `handle_library_skills_page_previous` | `test_library_skills_page_previous_returns_to_the_prior_exact_page` | `Tests/UI/test_library_skills_canvas.py` |
| `handle_library_skills_retry` | `test_library_skills_retry_recovers_transient_list_failure` | `Tests/UI/test_library_skills_canvas.py` |
| `handle_library_skill_tool_filter`, `_user_invocable_toggle`, `_disable_model_toggle`, `_discard` (4-in-1) | `test_skill_editor_tool_filter_toggles_and_discard_are_genuinely_pressed` | `Tests/UI/test_library_skills_canvas.py` |
| `handle_library_skill_conflict_reload` | `test_skill_editor_conflict_reload_clears_conflict_and_refetches` | `Tests/UI/test_library_skills_canvas.py` |

Both trust-reset pins use a real, already-unlocked-then-orphaned
`SkillTrustService` reaching `needs_resetup` posture (mirrors
`test_orphaned_manifest_is_one_click_resetup`'s own setup, which
established that posture renders BOTH the one-click "resetup" action
button, already covered, AND the standalone confirm-gated Reset button
this pin's flow exercises — two independent code paths, not the same
handler under two names). The retry pin uses a fail-once-then-recover
fake service (`_FailOnceSkillsScopeService`, mirrors
`_RecoveringLibraryNoteDetailService` in `test_library_shell.py`). The
grouped tool_filter/toggles/discard pin and the conflict_reload pin both
drive a real, fully-mounted `LibraryScreen` opened via `_open_real_skill_
editor`; conflict_reload's precondition (`_library_skill_conflict = True`)
is set directly on the real screen instance rather than produced through a
genuine `expected_version`-mismatch save race (out of this pin's scope,
documented in the test's own docstring) — sufficient to characterize the
BUTTON's own current behavior, which is all a pure-move state PR needs to
protect.

**Skip decisions (no new pin), recorded:**

1. **A genuinely dead/unreachable selector, not fixed.**
   `handle_library_skill_input_changed`'s SECOND `@on` decorator listens
   for `#library-skill-allowed-tools`. A repo-wide grep across
   `tldw_chatbook/` finds this string NOWHERE else — no widget anywhere
   carries that id (the allowed-tools UI is exclusively the
   `#library-skill-tool-picker` `SelectionList`, handled by the sibling
   `handle_library_skill_tool_selection`, which IS genuinely covered).
   This branch can never fire in the real UI. A pre-existing latent bug,
   out of this task's scope; not fixed (pure-move discipline), and not
   pinnable via `.press()` since nothing can press it.
2. **Already sufficiently characterized via unbound-fake-`self` direct
   calls** (consistent standard applied throughout, not just to these):
   `handle_library_skills_trust_action` (`test_trust_action_setup_
   dispatches_bootstrap`), `handle_library_skills_import_review`
   (`test_handle_library_skills_import_review_opens_editor`),
   `handle_library_skills_import_browse`/`_browse_folder`
   (`test_handle_library_skills_import_browse_folder_pushes_directory_
   dialog` + 2 more sites), `action_library_skill_save`/`action_library_
   skill_back` (3 existing direct-call tests). None reach the RAG
   precedent's own "real, fully-mounted screen" bar, but all have SOME
   existing behavioral coverage — a new pin would add nothing a
   move-time regression couldn't already trip via the existing one.
3. **Zero coverage, but the field it touches never moves.**
   `handle_library_skills_import_retry` reads
   `self._library_skill_import_coordinator.claim_retry()` exclusively —
   the ONE field this task classifies WIRING (§2). A gap here poses no
   risk to this task's own diff (a pure field move); left for the
   controller-move task to address if it proves worth the setup cost of
   a genuine claim-retry scenario.

No live bugs pinned as bugs (item 1 above is dead code, not a behavior
bug reachable through the UI).

## 6. `LibrarySkillsState` + shims

`tldw_chatbook/UI/Library_Modules/library_skills_state.py`: a `@dataclass`
with the 36 moving fields, verbatim defaults (§2's table), matching the
search+RAG precedent's mixed-prefix shape but generalized to three
prefixes (§3). `skill_state_shim_attr()` is the single-source resolver.

Shim: the same programmatic property-loop shape every prior series uses
(`for _lss_field in dataclasses.fields(LibrarySkillsState): setattr(
LibraryScreen, skill_state_shim_attr(_lss_field.name), property(...))`),
appended at the end of `library_screen.py` inside a
`--- BEGIN/END generated skills-state shims (delete wholesale at
cleanup) ---` sentinel block — this wave's FIRST currently-active shim
block (conversations/export/collections/search+RAG's own blocks were all
deleted by their respective cleanup tasks already).

## 7. TDD evidence

- RED commit (`ef289548a`): ships `Tests/Architecture/
  test_library_skills_wiring.py` + `library_skills_state.py` (production
  code, per the recipe's own structural-RED criterion — the SCREEN itself
  is what must be untouched) + the 6 characterization pins. Confirmed RED
  by re-running the wiring test against a `git stash -u` of everything
  after this commit: `test_state_object_fields_match_the_shim_surface`
  fails (`AssertionError: no screen shim property found for: [...36
  names...]`); the other 3 wiring-shape tests (state-object self-checks)
  pass unaffected. All 6 characterization pins independently confirmed
  PASSING before this commit (run standalone, then as part of their full
  files, both before ANY screen edit).
- GREEN commit (`87c318d57`): screen edits (import + `__init__` +
  trailing shim block) + `_BUDGETS` lower. Wiring test turns green (4/4).

## 8. Verification battery

- **Wiring RED→GREEN**: confirmed both directions (§7).
- **Characterization all-PASS pre-change**: all 6 new pins confirmed
  passing before the GREEN commit landed (§7); the full `Tests/UI/
  test_library_skills_canvas.py` + `Tests/Skills/test_skills_library_
  flow.py` files run together: 8 pre-existing failures (CSS-block/
  geometry-parity tests + `test_action_library_skill_back_honors_dirty_
  guard`), each confirmed identical on a `git stash -u` pristine baseline
  of the pre-task tree; 172 passed (single run), 537 passed/1 skipped/1
  more pre-existing failure (`test_import_real_superpowers_skills_lands_
  trust_pending`, environment-dependent, confirmed identical on baseline)
  in the full `Tests/Skills/` run.
- **Screen ratchet ceiling+slack**: `library_screen.py` row: `43225/1311
  -> 43179/1311` (methods unchanged — pure field move, zero `FunctionDef`s
  touched). Both `test_screen_does_not_grow_past_its_budget` and
  `test_budget_is_not_left_slack_after_a_wave` pass for this row.
- **Controller ratchet untouched-and-green**: `test_library_modules_size_
  ratchet.py` (the controller-file governance guard, task-31203 AC#4) —
  green; this task touched zero controller files.
- **Census + support-layer + ALL prior wiring/characterization suites
  green**: `Tests/Architecture/test_library_support_layer_surface.py`,
  `test_library_modules_size_ratchet.py`, and all four prior subsystems'
  wiring tests (`test_library_conversations_wiring.py`,
  `test_library_export_wiring.py`, `test_library_collections_wiring.py`,
  `test_library_search_rag_wiring.py`) plus this task's own
  `test_library_skills_wiring.py`: **60 passed**, 0 failed.
- **Full `Tests/Architecture/` run**: 16 failed, 537 passed, 1 skipped.
  All 16 failures confirmed pre-existing: the 2 documented `chat_screen.py`
  ratchet rows (recipe §7's own standing list) plus 14 more
  (`test_console_realtime_controller_boundary`,
  `test_console_review_selection_controller_boundary`,
  `test_console_wave6_closeout_inventory`/`test_console_wave6_inventory`
  ×3, `test_default_timeout_session_guard`,
  `test_persistent_diagnostic_inventory` ×2,
  `test_timer_path_static_update_inventory` ×3,
  `test_worker_exclusive_group_inventory` ×2) — none Library/Skills-scoped;
  8 of the 14 spot-checked directly against a `git stash -u` pristine
  baseline and reproduce identically. Matches wave-3's own documented
  "13 Console/timer/worker/diagnostic rows" backdrop almost exactly (+1,
  expected given the file's own ~14-commits/day churn rate).
- **`-k "skill and library"` sweep** (`Tests/UI` + `Tests/Library`, single
  process): 270 passed, 11 failed. 10 confirmed pre-existing — 8 already
  reconfirmed above, plus `test_command_palette_providers.py::
  TestTabNavigationProvider::test_palette_library_skills_command_opens_
  hidden_starter_route` and `Tests/Library/test_library_skills_state.py::
  test_shadow_name_set_stays_in_sync_with_real_sources` (both confirmed
  identical via `git stash -u`), plus `Tests/UI/test_screen_navigation.py::
  test_skills_route_lands_on_library_with_skills_row_selected` (already on
  the recipe's own §7 documented list from wave-3 task 4/5's own sweeps).
  The 11th, `test_library_skills_manual_items_priority_survives_compact_
  layout_sync`, passed in TRUE isolation (both standalone and as the sole
  selected test) — order-dependent flakiness within that single-process
  sweep, not a regression; this task's diff touches none of its
  assertions or the compact-layout-sync code path.
- **`Tests/Skills/` full run** (the wave's own fourth root): 537 passed, 1
  skipped, 2 failed — both confirmed pre-existing (§8 above).
- **Full sequential xdist paired-baseline sweep** (`Tests/UI -k "library"
  -p no:randomly -q -n 8 --dist worksteal`, branch then a pristine baseline
  checked out to `2372ea764` in-place — this task's changes are all
  COMMITTED, not stashed, so the baseline comparison used a temporary
  detached `git checkout 2372ea764` / `git checkout <branch>` round trip
  rather than `git stash -u`, per recipe §7's own "or check out ... in a
  scratch worktree" alternative, adapted here to reuse this worktree's
  existing venv rather than build a second one — a worktree's own uv venv
  cannot be skipped by pointing a second worktree's PYTHONPATH at it, per
  this project's own recorded lesson, but an IN-PLACE detached checkout
  keeps the SAME venv's editable-install path valid throughout):
  branch **370 failed / 3933 passed** (1626.63s); baseline **371 failed /
  3928 passed** (1966.45s). Both inside the documented ~330–370 historical
  backdrop (recipe §7's own prior sweeps: 330–357). 361 shared, 10
  baseline-unique (noise in the opposite direction, not investigated
  further per §7's own precedent), **9 branch-unique**:

  - 5 passed cleanly on re-run (combined single-process, or individual
    isolation) — ordinary xdist noise, not investigated further:
    `test_library_media_side_by_side.py::test_compact_media_pager_
    receipt_and_empty_states_remain_contained`,
    `test_library_prompt_collections.py::
    test_library_screen_membership_load_retry_and_apply_retry_are_
    distinct`, `test_library_prompt_collections.py::
    test_successful_mutation_with_refresh_failure_retries_catalog_only`,
    `test_library_prompts_canvas.py::
    test_library_prompt_undo_refreshes_applied_page_and_preserves_
    basket`, `test_library_prompts_canvas.py::
    test_library_prompt_pager_first_and_filter_failure_states[size0]`.
  - 3 reproduce identically in TRUE isolation on BOTH the branch and a
    freshly-checked-out pristine baseline — confirmed pre-existing, this
    task's diff touches none of their assertions or code paths:
    `test_library_entry_compose_once.py::
    test_source_worker_completion_during_mount_dispatch_reconciles_once`,
    `test_library_prompts_canvas.py::
    test_library_prompt_page_focus_survives_loading_recompose`,
    `test_library_shell.py::
    test_library_starter_production_geometry_and_focus_order[size0]`.
  - 1, `test_library_adaptive_reader_closeout.py::
    test_closeout_single_app_route_cycle`, needed deeper investigation:
    it is THE SAME test recipe §16 already documents as a destination-
    cycling trap that can look related to an unrelated change without
    being one. Traced `_focus_closeout_work_via_f6`'s failure
    (`"collections has no reachable Work focus target"`) to the
    'collections' step specifically — the SECOND destination in
    `DESTINATIONS = ("media", "collections", "conversations", "notes",
    "prompts", "skills")`, processed BEFORE "skills" is ever touched in
    the same cycle, and its own focus-target dispatch
    (`_library_workbench_focus_targets`) is a flat `_library_selected_
    row_id`-keyed branch with zero dependency on any of this task's 36
    moved fields. Repeated isolated single-process runs on both trees (8
    on branch: 7 failed/1 passed; 3 on baseline: 1 failed/2 passed) show
    the SAME assertion failing at the SAME step on BOTH trees, at
    different observed rates in these small samples — not a new failure
    mode, and mechanically unreachable from this task's diff (the trio's
    forced-early construction point was independently re-verified correct
    in the actual edited file: `self._skills_state = LibrarySkillsState()`
    at line 2172, before the shared reader-preferences tuple-unpack at
    line 2401). Classified as pre-existing, timing-sensitive flakiness in
    a wait-then-immediately-assert-on-DOM test (the exact class recipe
    documents elsewhere: a `_wait_for_condition` resolving true before a
    triggered recompose has actually finished), whose observed rate is
    plausibly sensitive to this session's own heavily fluctuating machine
    load (load average observed between ~19 and ~51 over the course of
    this task) rather than to this diff's content — not a conclusive
    zero-risk finding, but the most rigorous conclusion the evidence
    supports, recorded here rather than asserted silently. **Zero
    unexplained branch-unique failures; zero new failure MODES.**
- **preflight**: `./scripts/preflight.sh` — all six derived-artifact
  checks pass (CSS bundle, profile-owned-path census, diagnostic
  inventory, backlog task-id sweep, chachanotes table allowlist, index
  plan pins).

## 9. Files changed

- `tldw_chatbook/UI/Library_Modules/library_skills_state.py` (new, RED
  commit): `LibrarySkillsState` dataclass, `SKILLS_PLURAL_STATE_FIELDS`,
  `SKILL_UNPREFIXED_STATE_FIELDS`, `skill_state_shim_attr()`.
- `Tests/Architecture/test_library_skills_wiring.py` (new, RED commit):
  4 wiring-shape tests.
- `Tests/Skills/test_skills_library_flow.py` (RED commit): 2 new
  characterization pins.
- `Tests/UI/test_library_skills_canvas.py` (RED commit): 4 new
  characterization pins (one covering 4 handlers).
- `tldw_chatbook/UI/Screens/library_screen.py` (GREEN commit): +1 import,
  36-field `__init__` block collapsed to 1 constructor call (5 fields'
  original lines kept, routed through the new shim; 31 deleted), trailing
  generated shim block appended. Net **43225 → 43179 lines**, **1311
  methods unchanged**.
- `Tests/Architecture/test_screen_size_ratchet.py` (GREEN commit):
  `_BUDGETS` row lowered, dated comment added.

No blame-ignore entries (state PR, not a body move). No delegator
census/dead-import prune (nothing deleted from the screen's own body
except the 31 static-default lines, which had no external references —
they were assignment targets, not callable names).

## 10. Self-review

- Every field's classification is backed by a body-read, not a name
  guess — the two "reclassify a name-implied-subsystem consumer as
  shell/plumbing" calls (`_toggle_library_media_reader_pane`,
  `_library_skill_trust_confirming_reset`'s rail-switch touch) were each
  individually verified by reading the actual method body, not inferred
  from the plan's own framing.
- The three-prefix finding is a genuine deviation from the plan's "two-
  prefix" framing, caught by running the mechanical census rather than
  trusting the plan's shorthand — consistent with the recipe's own
  "verify callers" mandate.
- Characterization coverage is evidenced at the CSS-selector level, not
  the method-name level, closing the exact undercounting gap the
  collections series' own report flagged for method-name-only greps.
- Six new pins is more than any prior series' state-PR task needed
  (RAG found zero gaps in 14 handlers) — proportionate to skills' much
  larger 47-handler surface and its own pre-existing test style (two
  dense, real-service end-to-end files covering most of the surface,
  leaving a long tail of individually-untested siblings rather than a
  systematically-undertested cluster).
- One item is recorded as a live (but dead-code, unreachable) bug, not
  fixed — consistent with "live bugs pinned never fixed" in spirit (no
  fix landed), though it was documented rather than pinned since nothing
  can press an unreachable selector.
- No method bodies were touched in either commit; the whitelist
  (imports, constructor bindings, the two binding kinds) was respected
  throughout.
- **Process note, recorded honestly rather than smoothed over**: the
  baseline half of the sequential sweep required a detached `git checkout
  2372ea764` (both this task's commits were already landed, so `git stash
  -u` alone — the recipe §7 default — could not produce a true pre-task
  tree). Mid-investigation of the `test_closeout_single_app_route_cycle`
  discrepancy, one `git checkout <sha> -- .` (a mistaken partial-restore
  attempt) and one subsequent grep were run while still on the detached
  baseline commit rather than back on the branch — caught immediately when
  a grep for `_skills_state` (which should exist on the branch) came back
  empty, corrected via `git checkout HEAD -- .` / `git checkout <branch>`,
  and re-verified (line 2172 constructor call, ordering vs. line 2401's
  tuple-unpack) before trusting any further conclusion. No data collected
  while on the wrong commit was used in this report's own findings — the
  affected grep (`DESTINATION_CONTRACT` order) happened to read a test
  file identical on both trees, and the affected `_skills_state` grep
  simply produced the empty result that surfaced the mistake. Flagged here
  per this project's own evidence-integrity standard: a process error
  that self-corrected and left no false claim in the record is still
  worth naming, not quietly absorbed.
