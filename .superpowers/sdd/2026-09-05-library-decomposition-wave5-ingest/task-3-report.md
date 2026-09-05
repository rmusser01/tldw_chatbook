# Wave-5 Task 3 report — Ingest cleanup PR (series 3/3)

Plan: `Docs/superpowers/plans/2026-09-05-library-decomposition-wave5-ingest.md`.
Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§20 (mechanics
authority; this task adds §20 documenting the series as landed). Branch:
`refactor/library-decomp-wave5-ingest`. Worktree:
`.worktrees/library-decomp-foundation`. Base: `e3d85ad21` (task 2's own
fix-round-1 closing commit).

This is the one PR type in the recipe allowed to edit tests. Unlike the
state/controller PRs, it is a single commit (no RED/GREEN split — nothing
here is a pure move with a covering-test precondition; every change is a
retarget, a deletion, or a docstring correction, verified by the existing
battery staying green throughout).

## 1. Dynamic-dispatch census (first, per the task's mandated order)

Content-grepped (never `-k`) all of `tldw_chatbook/` and `Tests/` for the
20 `LibraryIngestState` field names in three dynamic-dispatch shapes: a
quoted string literal `"_library_ingest_<field>"` (dict keys/values,
`getattr`/`setattr` second argument), an f-string constructing a dynamic
attribute name, and a dict literal keying off a field name (the shared
reader-preference dispatcher / choice-strip visibility pattern every
prior subsystem's cleanup has had to update for its own fields).

**Zero hazard shapes found.** Ingest fields never intersect
`_replace_library_reader_preference`/`_persist_library_reader_preference`
(read via `grep -n` — no `ingest` name inside either method) or any
choice-strip visibility dict (Ingest has no "reader" mode and no
per-type choice-strip visibility field). The only screen-side dynamic
call touching Ingest at all is `canvas_sync.py`'s own `kind == "ingest"`
branch (`elif kind == "ingest": ... sync_args =
(screen._build_library_ingest_state(),)`) — a METHOD call, not a flat
field-name string, so no retarget applies.

**What the census DID find**: 6 `getattr(self, "_library_ingest_<field>",
default)` call sites in `library_screen.py` needing the skills series' own
"receiver, not just string" fix (`getattr` performs one attribute lookup,
not dotted traversal) — see §3 below. A 7th `getattr` hit
(`self._save_library_ingest_options({f"library.ingest_options.{group}":
{}})`) was a false positive: `_save_library_ingest_options` is a METHOD
name matching the fields regex's "options" substring incidentally; read
and confirmed not a field access.

**Out-of-file reads**: a repo-wide grep of `tldw_chatbook/` (excluding
`library_screen.py`, `library_ingest_controller.py`,
`library_ingest_state.py`) for any of the 20 field names found exactly
one hit — a prose-comment mention of `self._library_ingest_form` in
`tldw_chatbook/Library/library_ingest_state.py`'s own
`LibraryIngestFormState` docstring (a lower-level, data-only module,
unrelated to the screen/controller state object of the same name).
Corrected (§7 below); no code-level out-of-file consumer exists.

## 2. Screen-side field retarget

**37 pre-existing flat-name occurrences**, all inside 12 still-screen-
resident methods (11 EXCLUDED movers + 1 shell/plumbing method reading an
Ingest field incidentally) — none inside a MOVED method body (task 2's
own controller move already deleted those):

| Method | Occurrences | Why still screen-resident |
|---|---|---|
| `_build_library_ingest_state` | 7 (3 as `getattr`) | instance-attribute-monkeypatch exclusion |
| `handle_library_ingest_backend_switch` | 4 (2 as `getattr`) | unbound-fake-self exclusion |
| `_enqueue_library_ingest_snapshot` | 4 | unbound-fake-self exclusion |
| `_save_library_ingest_backend` | 2 | `@work`-hazard exclusion |
| `_do_submit_ingest` | 2 | unbound-fake-self exclusion |
| `_apply_library_external_preparation` | 2 | separate "external source" onboarding feature (shares 2 fields) |
| `_run_debounced_library_ingest_preflight` | 2 | unbound-fake-self exclusion |
| `_build_ingest_options_snapshot` | 2 | unbound-fake-self exclusion |
| `_library_ingest_browse_location` | 1 | unbound-fake-self exclusion |
| `handle_library_ingest_option_value_changed` | 2 | unbound-fake-self exclusion |
| `_load_library_ingest_options_from_config` | 1 | module-globals-coupling exclusion |
| `check_action` (as `getattr`) | 1 | shell/plumbing |
| `on_mount` | 1 | shell/plumbing |
| `_set_library_rail_collapsed` | 1 | shell/plumbing |
| `_library_emergency_return_eligibility` | 1 | shell/plumbing |
| `_library_resize_layout_signature` | 1 | shell/plumbing |
| `_on_preflight_retry` | 1 | shell/plumbing |
| `_apply_parakeet_v2_install_result` | 1 | unrelated model-install handler reading the `form` field |

A single mechanical regex pass (`(\w+)\._library_ingest_(field)\b` →
`\1._ingest_state.\2`) rewrote 31 of the 37; a second pass
(`getattr(self, "_library_ingest_(field)"` → `getattr(self._ingest_state,
"(field)"`) rewrote the remaining 6, fixing the RECEIVER as well as the
string (the skills series' own lesson: `getattr` does not do dotted
traversal, so a string-only swap would have silently returned the
default forever). One multi-line `getattr(...)` call the regex joined
onto one physical line was reformatted for readability (no logic change).
Re-verified with a zero-result repo-wide grep for every one of the 20
flat field names over the whole file (excluding the shim-deletion
comment's own historical prose, which intentionally still names the old
flat spellings).

## 3. Test retargets — 17 files, 301 sites, byte-for-byte assertions

Content census (never `-k`) across ALL of `Tests/` for the 20 field
names found flat-name MENTIONS in 19 files; 2
(`test_library_url_ingest_submit.py`, `test_library_ingest_wiring.py`)
turned out to be false-positive moved-method-name mentions only
(`_submit_library_ingest_form()` and similar — methods, not fields),
zero real retargets. The remaining 17 needed real changes, classified
into two shapes (a third, quoted string-literal patch target, matched
nowhere):

| Shape | Sites | Files |
|---|---|---|
| Direct attribute (`screen._library_ingest_<field>`, incl. `getattr(screen, "...")`) | 297 | 15 |
| `SimpleNamespace(...)` flat kwarg (`_library_ingest_<field>=value`) | 4 (3 fixtures) | 3 (1 overlaps the 15: `test_library_canvas_scoped_sync.py`) |

**297 direct-attribute retargets** (15 files), one mechanical regex pass
identical in shape to §2's screen-side pass, across:
`test_library_ingest_inline_consent.py` (95), `test_library_shell.py`
(52), `test_library_ingest_canvas.py` (40), `test_library_ingest_retry_
last.py` (29), `test_library_screen.py` (26), `test_library_ingest_
flow.py` (21, `Tests/integration`), `test_library_ingest_
characterization.py` (11), `test_submit_library_ingest_job.py` (6,
`Tests/App`), `test_library_ingest_clear_focus.py` (5), `test_parakeet_
v2_install_ui.py` (4), `test_library_resize_focus_gates_t23025.py` (2),
`test_library_ingest_structural.py` (2), `test_retired_tldw_api_worker_
pipeline.py` (2, `Tests/ProductionApp`), `test_library_ingest_
keyboard.py` (1), `test_library_canvas_scoped_sync.py` (1). The other 2
of the 17 (`test_ingest_preflight_egress.py` in `Tests/Library`,
`test_config_nested_settings.py` in `Tests/Utils`) needed ONLY the
`SimpleNamespace` restructuring below, zero plain attribute-path
retargets. `test_library_ingest_wiring.py`
— its own `_INGEST_CLUSTER_METHOD_NAMES` docstring prose mentions the
field name `form`/`last_submission`/`start_consent` incidentally, all
false positives, zero code change there (that file's real edits are in
§8 below). **Zero assertion VALUE changes** — every one of the 297 is a
receiver-path rewrite only, verified by running the full affected-file
battery before and after and diffing pass/fail (§10).

Safety precondition for the mechanical pass: every receiver in every hit
is either a real, `__init__`-constructed `LibraryScreen` (e.g.
`_minimal_ingest_screen()`, `test-3022`'s own real-constructor fixture)
or an `object.__new__`-bypass instance task 1's own fix round already
seeded with a real `_ingest_state = LibraryIngestState()` — confirmed by
reading each fixture helper, not assumed. No receiver in this set is a
bare `SimpleNamespace` needing restructuring.

**3 `SimpleNamespace` fixture restructurings** (4 flat kwargs total) for
the fixtures that DO stand in for an unbound `self` with a bare
`SimpleNamespace`, not an `_ingest_state`-seeded screen:

1. `Tests/UI/test_library_canvas_scoped_sync.py::
   test_ingest_checkbox_routes_to_ingest_canvas_sync` — the fake `self`
   for the excluded `handle_library_ingest_option_value_changed`:
   `_library_ingest_form=form` → `_ingest_state=SimpleNamespace(form=form)`.
2. `Tests/Library/test_ingest_preflight_egress.py::
   test_the_typing_debounce_forbids_probing_even_when_it_is_enabled` —
   the fake `self` for the excluded `_run_debounced_library_ingest_
   preflight`: `_library_ingest_path_debounce_timer=object(),
   _library_ingest_form=SimpleNamespace(path=...)` → both nested under
   one `_ingest_state=SimpleNamespace(path_debounce_timer=object(),
   form=SimpleNamespace(path=...))`.
3. `Tests/Utils/test_config_nested_settings.py::
   test_library_ingest_browse_location_audit_fix` — the fake `self` for
   the excluded `_library_ingest_browse_location`:
   `_library_ingest_form=SimpleNamespace(path="")` → nested under
   `_ingest_state=SimpleNamespace(form=SimpleNamespace(path=""))`. Found
   OUTSIDE every canonical ingest test root — the same file task 1's own
   seventh-bypass-shape fix round already flagged as evidence for
   "widen beyond the obvious roots."

**Zero quoted-string-literal patch targets** — no test monkeypatches an
Ingest field by string name (`monkeypatch.setattr(obj, "_library_ingest_
<field>", ...)` or a fully-qualified string); the census's `strlit`
pattern matched nothing.

**False-positive check**: every remaining `_library_ingest_<name>`
mention that is NOT one of the 20 fields (63 occurrences) is either a
call site for a MOVED method's own name (`_submit_library_ingest_form()`,
`_current_library_ingest_start_consent()`, `_restage_library_ingest_
last_submission()`, etc. — methods keep their flat spelling; only FIELDS
were retargeted) or a prose-comment mention — each individually confirmed
by grep context, zero required a change.

## 4. Delegator census — 50 KEEP, 6 PRUNED

Of the 56 moved names: 25 `@on` handlers + 2 `action_*` methods KEEP
unconditionally (recipe's transform whitelist). Of the remaining 29 (28
plain + 1 staticmethod), a repo-wide grep for each name — `receiver.name(`
call sites AND a bare "name appears anywhere in the line" second pass,
across `tldw_chatbook/` and every `Tests/` root, excluding
`library_ingest_controller.py`'s own internal calls and each name's own
one-line screen-delegator body — found:

**Correction (fix round 1, post-review)**: every "mover caller"/"listener
registration"-shaped label below was WRONG in the original draft of this
table — a moved method's body no longer lives on the screen to make such
a call (only the controller's own internal calls could do that, and those
are explicitly excluded from this census by definition). Every real
screen-side caller found here is either an EXCLUDED method (still
full-bodied on the screen) or an unrelated SHELL method — re-verified
name-by-name via `ast`-derived containing-method lookup, not re-guessed.
Re-verifying also caught two count errors: `_apply_library_ingest_
preflight_result` (11 → 10: the dropped 1 was a controller-internal prose
mention, out of this census's own scope) and `_library_ingest_registry`
(33 → 11: the original count was contaminated by `_library_ingest_
registry` being a literal substring of `_handle_library_ingest_registry_
changed`, so a substring-based grep silently double-counted that name's
own 22 references as this one's).

| Name | External references | Verdict |
|---|---|---|
| `_apply_library_ingest_backend_save` | 1 (excluded caller: `_save_library_ingest_backend`) | KEEP |
| `_apply_library_ingest_preflight_result` | 10 (1 excluded caller via `call_from_thread`: `_run_library_ingest_preflight` + 9 test call sites) | KEEP |
| `_authoritative_library_ingest_consent_is_current` | 1 (shell caller: `_apply_library_external_preparation`) | KEEP |
| `_cancel_library_ingest_preflight` | 3 (2 test monkeypatches + 1 prose) | KEEP |
| `_current_library_ingest_start_consent` | 13 (1 shell caller: `_apply_library_external_preparation` + 1 excluded caller: `_enqueue_library_ingest_snapshot` + 11 test call sites) | KEEP |
| `_disarm_library_ingest_retry_confirm` | 2 (1 excluded caller: `handle_library_ingest_option_value_changed` + 1 test fixture kwarg) | KEEP |
| `_disarm_library_ingest_start_confirm` | 5 (4 excluded callers: `handle_library_ingest_backend_switch`, `handle_library_ingest_option_value_changed`, `handle_library_ingest_directory_browse`, `handle_library_ingest_option_reset` + 1 test fixture kwarg) | KEEP |
| `_focus_library_ingest_path` | 1 (shell caller via `call_after_refresh`: `_select_library_rail_row_after_source_admission`) | KEEP |
| `_handle_library_ingest_progress_changed` | 3 (2 shell listener registrations in `on_mount`/`on_unmount` + 1 test call) | KEEP |
| `_handle_library_ingest_registry_changed` | 19 (2 shell listener registrations in `on_mount`/`on_unmount` + 1 shell prose comment + 16 test call sites) | KEEP |
| `_invalidate_library_ingest_preflight` | 5 (1 excluded caller: `_enqueue_library_ingest_snapshot` + 4 test sites) | KEEP |
| `_library_ingest_registry` | 11 (7 screen callers -- `on_mount`, `on_unmount`, `_library_landing_attention_action`, `check_action` (shell) + `_build_library_ingest_state`, `_enqueue_library_ingest_snapshot`, `_library_ingest_job_by_id` (excluded) -- + 4 test sites) | KEEP |
| `_library_ingest_shortcuts_for_current_state` | 4 (1 shell caller: `_library_route_shortcuts_for_current_state` + 3 test sites) | KEEP |
| `_pause_library_ingest_transient_ui` | 1 (shell caller: `_select_library_rail_row_after_source_admission`) | KEEP |
| `_reset_library_ingest_transient_state` | 3 (1 shell caller: `_apply_navigation_context_state` + 2 prose) | KEEP |
| `_restore_library_ingest_canvas_context` | 2 (1 excluded caller: `_refresh_library_ingest_canvas_preserving_context` + 1 test call) | KEEP |
| `_scroll_library_ingest_queue_into_view` | 1 (excluded caller via `call_after_refresh`: `_enqueue_library_ingest_snapshot`) | KEEP |
| `_submit_library_ingest_form` | 47 (test call sites only, zero screen-side callers) | KEEP |
| `_sync_library_ingest_rail_for_width` | 6 (3 shell callers: `_update_library_notes_responsive_state`, `on_resize`, `_select_library_rail_row_after_source_admission` + 1 test call + 2 prose) | KEEP |
| `_sync_library_ingest_rail_from_shell` | 1 (shell caller via `call_after_refresh`: `_select_library_rail_row_after_source_admission`) | KEEP |
| `_trigger_library_ingest_preflight` | 12 (1 excluded caller: `_run_debounced_library_ingest_preflight` + 1 shell caller: `_trigger_preflight`, its own alias + 1 prose (that alias's own docstring) + 9 test sites) | KEEP |
| `_update_library_ingest_fold_hint` | 1 (excluded caller via `call_after_refresh`: `_update_library_ingest_gate`) | KEEP |
| `_update_library_ingest_group_receipt` | 1 (excluded caller: `handle_library_ingest_option_value_changed`) | KEEP |
| **`_adopt_library_ingest_path`** | **0** | **PRUNED** |
| **`_ingest_job_id_from_button`** (staticmethod) | **0** (internal-only `self.<name>()` calls on the controller instance) | **PRUNED** |
| **`_library_ingest_restage_discards_work`** | **0** | **PRUNED** |
| **`_restage_library_ingest_last_submission`** | **0** | **PRUNED** |
| **`_set_library_ingest_panels_collapsed`** | **0** | **PRUNED** |
| **`_update_library_ingest_retry_label`** | **0** | **PRUNED** |

**6 of 56 (~11%)** — the LOW end of every prior series' recorded fraction
(export 1-of-22 ~5% < ingest 6-of-56 ~11% < skills 16-of-86 ~19% <
collections 14-of-64 ~22% < search+RAG 12-of-42 ~29% < conversations
18-of-61 ~30%), consistent with the recipe's inverse-relationship finding
(§15 lesson 3): only 22 of 78 candidates were excluded here (a small
exclusion count next to skills' 41-of-127), so fewer excluded,
screen-resident methods exist to keep calling their moved siblings — the
mechanism that keeps a delegator's reference count above zero.

Verified each "0 references" verdict a second way before deletion: a
broader grep matching the bare name ANYWHERE in a line (not just
call-shaped), still zero for all 6 — guards against the skills series'
own methodology trap (an over-eager exclusion pattern silently dropping
`self.<name>(` call sites).

## 5. Screen shim block deletion

Deleted the task-1-generated `_library_ingest_<field>` property-shim loop
(module end, `dataclasses.fields(LibraryIngestState)`-driven, single
prefix, 20 properties) once §1–§4 confirmed zero remaining consumers
anywhere in `tldw_chatbook/` or `Tests/` outside `LibraryIngestController`'s
own PERMANENT generated shim loop (task 2's own — untouched, reads
`self._ingest_state_accessor().<field>`). Replaced with a one-paragraph
"deleted at cleanup" comment matching the collections/search+RAG/skills
precedent's exact wording shape. Module still imports and constructs
`LibraryIngestState` in `__init__` (`self._ingest_state =
LibraryIngestState()`) — that import stays live.

## 6. AST-derived dead-import prune

Ran an AST census (every module-level import name in `library_screen.py`,
zero `ast.Name`-Load usages anywhere in the module) and filtered to
INGEST-related names only (the census also surfaced ~35 unrelated dead
names from other subsystems' historical residue — explicitly out of
scope for this task; not touched).

**8 removed**, each independently confirmed already re-imported and live
inside `library_ingest_controller.py` before deletion from the screen:

| Name | Source module |
|---|---|
| `ACTIVE_INGEST_STATES` | `Library.library_ingest_jobs` |
| `normalize_active_ingest_source` | `Library.library_ingest_jobs` |
| `LibraryIngestFormState` | `Library.library_ingest_state` |
| `build_ingest_forecast` | `Library.library_ingest_state` |
| `format_ingest_progress_line` | `Library.library_ingest_state` |
| `ingest_progress_action_signature` | `Library.library_ingest_state` |
| `build_type_group_title` | `Widgets.Library.library_ingest_canvas` |
| `capabilities_for_backend` | `Library.ingest_capabilities` |

(8 rows shown; `library_ingest_jobs`'s other still-live names —
`ActiveIngestConsentScope`, `ActiveIngestSubmissionRefused`,
`IngestJobState`, `LibraryIngestJob`, `build_active_ingest_consent_
scope`, `count_duplicate_done_jobs` — each independently re-verified live
via `grep -c` ≥2 before leaving them in place; same for `library_ingest_
state`'s `validate_ingest_option_value`, `INGEST_UNAVAILABLE_COPY`,
`LibraryIngestCanvasState`, `LibraryIngestLastSubmission`, `active_
ingest_start_confirm_line`, `build_library_ingest_state`, `clamp_chunk_
size`, `library_ingest_retry_available`, `library_ingest_retry_label`,
`parse_keywords`, and `ingest_capabilities`'s `get_capabilities`, `list_
type_groups`, and `library_ingest_canvas`'s `ingest_scope_label`.)

**1 more candidate (the 9th of 9 total) found dead but deliberately
KEPT**: `_ingestible_file_
filters` (`Library_Modules.screen_helpers`) — checked against `Tests/
Architecture/test_library_support_layer_surface.py`'s `_SURFACE` dict (the
PR-0a re-export contract) BEFORE removal and found pinned there;
`test_screen_still_re_exports_every_moved_name` asserts
`hasattr(library_screen_module, "_ingestible_file_filters")` regardless of
live screen-side usage. Removed once, broke that test, re-added
immediately (confirmed via a clean `git diff` on that import block after
reverting). **Correction to an earlier draft of this report**: this is
NOT the first time this shape has fired — the conversations exemplar's
own Task 7 hit the identical collision first
(`LIBRARY_CONVERSATION_READER_MAX_CHARS`, recipe §11's own "'dead within
this file' is not the same question as 'dead'" lesson); the four
INTERVENING cleanup tasks (export, collections, search+RAG, skills) each
recorded zero collisions because they ran the check that lesson
mandated, not because the shape stopped recurring. Caught only by
re-reading §11 before finalizing this report rather than trusting the
first draft's own "first collision" framing.

## 7. Docstring / census-listing corrections (canon-scope, freely editable)

Four corrections in `library_ingest_controller.py`'s own MODULE and
CONSTRUCTOR docstrings (not moved method bodies):

1. Module docstring: `"LibraryScreen keeps one-line delegators under
   every one of these 56 original names"` → now names the post-prune
   50-of-56 count and points at
   `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED`.
2. Class docstring: same claim, same fix, second location.
3. Constructor docstring: `"Every one of the 63 method bodies..."` → `56`
   (a stray arithmetic slip that never corresponded to any real
   derivation — not 78-22, not any other combination).
4. The `_apply_library_ingest_backend_save`/`_sync_library_canvas`
   module-globals census evidence: corrected from "7 files, ~20 sites"
   to the true **10 files, 38 sites** — 3 files (`test_library_canvas_
   scoped_sync.py`, `test_library_notes_reader.py`, `test_review_set_
   walker.py`) were missed by task 2's own grep because their patch
   sites used an import alias other than `library_screen`/`library_
   screen_module` (`screen_module`, or the same alias reached through
   `monkeypatch.context()`'s own `patcher.setattr(...)` inside a
   multi-line call). **"Site" defined and the per-file breakdown made
   reproducible** (post-review fix round 1): a site is one match, one
   line, of the census's own 3-shape pattern set (direct-attribute form,
   fully-qualified string form, or the two-argument `monkeypatch.
   setattr`/`patch.object` form), deduplicated by line number within a
   file — see `library_ingest_controller.py`'s own module docstring for
   the exact 10-file breakdown that sums to 38. Re-read all 10; all
   still LATENT (none reaches `_apply_library_ingest_backend_save`'s own
   call path) — verdict unchanged (KEEP as a mover), only the recorded
   evidence corrected.
   This is the carried minor from task 2's own re-review. Also corrected
   in the wiring test's own two docstring mentions of "63" and in
   `task-2-report.md` (an appended correction note, not a rewrite of the
   historical claim) and the recipe's §3 eighth-bypass-shape entry.

Plus two corrections outside the controller:

5. `tldw_chatbook/UI/Library_Modules/library_ingest_state.py` (task 1's
   own state module): its own docstring said "`library_screen.py` keeps
   every original `_library_ingest_<field>` attribute name alive as a
   generated ... shim" (present tense, now false) — updated to past
   tense, mirroring the skills series' own precedent for the identical
   correction on `library_skills_state.py`.
6. `tldw_chatbook/Library/library_ingest_state.py` (a different, lower-
   level, data-only module): `LibraryIngestFormState`'s own docstring
   said "Owned by the screen as a single bundled field (`self._library_
   ingest_form`)" — updated to name `self._ingest_state.form` as the
   current path, `self._library_ingest_form` as the prior one.

## 8. Wiring test finalization

`Tests/Architecture/test_library_ingest_wiring.py`:
- `test_state_object_fields_match_the_shim_surface` DELETED (screen shim
  gone; `test_ingest_controller_exposes_every_state_field` already covers
  the controller-side equivalent, needed no change).
- `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset added (6 names,
  including the cluster's one staticmethod).
- `test_screen_delegates_ingest_handlers` now skips names in that
  frozenset and asserts their genuine ABSENCE from `LibraryScreen`
  instead (a future accidental re-add fails loudly here).
- `test_ingest_cluster_staticmethods_forward_to_the_controller_class`
  given the identical skip/absence treatment for its own 1-name set
  (pruning the staticmethod needed this SEPARATE fix — `inspect.
  getsource(None)` raises `TypeError` rather than failing the assertion
  cleanly, so the plain-delegator skip pattern alone does not cover it).
- Both remaining "63" docstring mentions corrected to 56 (§7).
- Module docstring rewritten to describe the finished 3-task series.

5 of the original 6 tests remain (the shim-surface test's removal is the
only count change) — all 5 green post-cleanup.

## 9. Fresh pins (post-cleanup, re-derived via the ratchet's own `_measure`)

| File | Before | After |
|---|---|---|
| `tldw_chatbook/UI/Screens/library_screen.py` (lines/methods) | 40131/1302 | **40094/1296** |
| `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py` (lines) | 2536 | **2558** → **2569** (fix round 1) |

1296 = 1302 − 6 (exactly the pruned-delegator count — a pure deletion, no
replacement). Controller growth (+22) is comment-only (§7's four
corrections) — no method body touched, no mover count change (56
unchanged). **Fix round 1 (post-review) grew it a further +11**, also
comment-only: the `_sync_library_canvas` census's own "site" definition
and reproducible 10-file breakdown, added to close item 3 of the review.
Both re-pinned in this same commit per recipe §6/§17;
`Tests/Architecture/test_screen_size_ratchet.py::test_budget_is_not_
left_slack_after_a_wave` and `test_library_modules_size_ratchet.py::
test_budget_is_not_left_slack_after_a_move` both pass at the new pins
(zero slack).

## 10. Battery

All commands run from `.worktrees/library-decomp-foundation`,
`.venv/bin/python`.

- **6 wiring suites** (`test_library_collections_wiring.py`,
  `test_library_conversations_wiring.py`, `test_library_export_
  wiring.py`, `test_library_ingest_wiring.py`, `test_library_search_
  rag_wiring.py`, `test_library_skills_wiring.py`) + **4
  characterization files** (`test_library_ingest_characterization.py`,
  `test_library_collections_characterization.py`, `test_library_
  conversations_characterization.py`, `test_library_export_
  characterization.py`) + **both size ratchets** + **support-layer
  surface suite** (`test_library_support_layer_surface.py`, 8 tests) —
  run together: **99 passed, 2 failed** (both the documented
  pre-existing `chat_screen.py` ratchet rows).
- **`-k "ingest and library"`** (`Tests/UI` + `Tests/Library`, single
  process): **7 failed, 1143 passed, 1 skipped, 21718 deselected** (700s).
  All 7 failures are a SUBSET of the 8 content-grep-confirmed
  pre-existing names below (`-k "ingest"` cannot select `test_library_
  resize_focus_gates_t23025.py::test_tab_focus_path_library_query_
  volume_is_bounded`, whose name contains neither "ingest" nor is
  selected together with "library" the same way — the exact filter-
  blindness shape recipe §3's seventh-bypass-shape entry already
  documents for a DIFFERENT file; caught here only because the wider
  content-grep battery below also ran it directly). Zero new failures.
- **Content-grep-derived file runs** (parakeet lesson — the exact 19
  files §3's census found, run together as a targeted battery, not just
  the aggregate `-k` sweep): **488 passed, 9 failed** in the first
  combined run. 8 of the 9 confirmed pre-existing via a `git stash -u`
  pristine-baseline rerun of the SAME 8 node-ids (identical failures on
  both trees): 2 CSS-color-parity rows in `test_library_ingest_
  canvas.py::test_progress_detail_paints_below_row_without_obscuring_
  actions_or_neighbor[size0/size1]`; a footer-shortcut-list assertion in
  `test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_
  footer_when_retry_availability_changes`; a DOM-query-volume budget in
  `test_library_resize_focus_gates_t23025.py::test_tab_focus_path_
  library_query_volume_is_bounded`; 3 CSS-geometry/focus-styling rows in
  `test_library_ingest_structural.py` (`test_fold_hint_is_pinned_not_
  scrolled`, `test_outcome_lines_paint_heavier_than_the_tooling_
  summary`, `test_every_canvas_focusable_changes_at_the_glyph_level_on_
  focus`); and an environment-dependent network-egress-probe outcome in
  `test_ingest_preflight_egress.py::test_the_probe_reports_a_redirect_
  as_an_answered_status_not_an_error`. The 9th,
  `test_library_canvas_scoped_sync.py::test_notes_per_click_updates_
  keep_screen_and_canvas_identity`, is the SAME name already documented
  in recipe §7 (wave-3 task 4) as a Notes-only characterization test
  unrelated to any Ingest diff. **Zero real regressions.** All 8 new
  pre-existing names added to recipe §7.
- **Full sequential xdist paired-baseline sweep**: see §11.
- **preflight**: `./scripts/preflight.sh` — all checks green (no
  diagnostic-inventory drift; CSS bundle, profile-owned-path census,
  backlog task ids, chachanotes table allowlist, and index plan pins all
  passed).

## 11. Full sequential xdist paired-baseline sweep (recipe §7)

`.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q -n 8
--dist worksteal`, run SEQUENTIALLY (branch first, then an isolated
`git worktree add` + its own `uv venv`/`pip install -e ".[dev]"` at the
pristine base `e3d85ad21` — never a same-tree checkout overlay, per the
recipe's own isolated-worktree-baseline lesson):

- **Branch**: 356 failed, 3994 passed, 98 warnings, 1435.08s (~24 min).
- **Baseline** (`e3d85ad21`, isolated worktree): 358 failed, 3992 passed,
  113 warnings, 1499.26s (~25 min).
- **351 shared failures** (including a massive, near-identical burst of
  `test_library_shell.py::test_library_note_*` DOM-mount-timeout failures
  concentrated in the last ~3% of BOTH runs — confirmed load-driven, not
  diff-driven, by its near-total overlap between trees), **5
  branch-unique**, **7 baseline-unique** (not investigated, per the
  recipe's own established precedent for baseline-unique names).

**All 5 branch-unique names resolved, zero real regressions**, verified
by a combined single-process re-run (all 6 node-ids together, including
both parametrizations of the pager test):
`test_library_media_reader_traversal_t22207.py::test_focus_traversal_
builds_zero_bodies_for_pass_through_rows`, `test_library_media_reader_
traversal_t22207.py::test_one_megabyte_markdown_document_is_not_
reparsed_per_keystroke` (already documented in recipe §7 as a wave-2-
task-6/wave-5-task-1 branch-unique name that passed cleanly on rerun
each time — reconfirmed a third time here), `test_library_prompts_
canvas.py::test_library_prompt_history_no_change_keeps_selection_and_
retry_available`, `test_library_prompts_canvas.py::test_library_prompt_
pager_first_and_filter_failure_states[size0]` (the SAME name already
documented in recipe §7, wave-5 task 2, as passing cleanly on a combined
rerun there too), and `test_library_prompts_canvas.py::test_library_
prompts_stale_search_cannot_restore_an_old_filter_caret`. **6 passed, 0
failed** in the combined re-run. None of the 5 touches Ingest code or
this task's own diff (all are Media-reader or Prompts-canvas tests).
**Zero real regressions.**

**Coverage note for the 2 largest-diff files not in the 16-file targeted
battery** (`test_library_screen.py`, 26 retargets; `test_library_shell.py`,
52 retargets — both were part of the 297-site content census but not the
targeted-battery command, an oversight caught during self-review, not by
the plan): `test_library_screen.py` was run standalone afterward (**32
passed, 0 failed**). `test_library_shell.py` is too large to complete a
standalone timed rerun cleanly (a `-n 4` attempt hit its own 590s time
budget mid-run before printing a summary); its coverage instead comes
from the full xdist sweep above, which already exercises every test in
the file as part of `Tests/UI -k "library"` and found zero branch-unique
failures attributable to it.

**New forward observation, not chased further (out of scope)**: an
unrelated standalone rerun of `test_library_shell.py` triggered `pytest`'s
own `fd_leak_sentinel` plugin: "open file descriptors grew by 274 over
the test session (start=14, end=288, limit=200)". This is consistent
with — and a plausible root cause for — the massive, near-identical
`test_library_note_*` DOM-mount-timeout failure burst concentrated in the
final ~3% of BOTH the branch and baseline full sweeps (351 of the
356/358 total failures overlap almost entirely from this one cluster):
file-descriptor exhaustion accumulating over a long single-process test
session, not this task's diff. Recorded in recipe §7 as a lead for
whichever future task next investigates `Tests/UI`'s own sweep-flakiness
backdrop.

Isolated worktree cleaned up (`git worktree remove --force`) after the
sweep completed.

## 12. Files changed

- `tldw_chatbook/UI/Screens/library_screen.py` — shim block deleted;
  37 screen-side field retargets; 6 delegators deleted; 8 dead imports
  removed (1 more candidate kept, `_SURFACE`-pinned).
- `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py` —
  4 docstring corrections (no method body touched).
- `tldw_chatbook/UI/Library_Modules/library_ingest_state.py` — 1
  docstring correction (past-tense shim description).
- `tldw_chatbook/Library/library_ingest_state.py` — 1 docstring
  correction (current attribute path).
- `Tests/Architecture/test_library_ingest_wiring.py` — shim-surface test
  deleted; pruned-delegator/staticmethod skip+absence pairs added; 2
  stray "63" counts fixed; module docstring rewritten.
- `Tests/Architecture/test_screen_size_ratchet.py` /
  `test_library_modules_size_ratchet.py` — pins lowered/raised
  (40131/1302 → 40094/1296; 2536 → 2558), dated comments.
- 17 test files: 297 attribute-path retargets + 3 `SimpleNamespace`
  fixture restructurings (§3) — `test_library_ingest_inline_consent.py`,
  `test_library_shell.py`, `test_library_ingest_canvas.py`,
  `test_library_ingest_retry_last.py`, `test_library_screen.py`,
  `test_library_ingest_flow.py`, `test_library_ingest_
  characterization.py`, `test_submit_library_ingest_job.py`,
  `test_library_ingest_clear_focus.py`, `test_parakeet_v2_install_
  ui.py`, `test_library_resize_focus_gates_t23025.py`,
  `test_library_ingest_structural.py`, `test_retired_tldw_api_worker_
  pipeline.py`, `test_library_ingest_keyboard.py`,
  `test_library_canvas_scoped_sync.py`, `test_ingest_preflight_
  egress.py`, `test_config_nested_settings.py`.
- `backlog/docs/library-decomposition-recipe.md` — new §20 (ingest
  series as landed); §8 table row updated; §3 eighth-bypass-shape
  census-listing correction; two new §7 sweep-evidence entries.
- `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/
  task-2-report.md` — appended correction note (census-listing minor).

## 13. Self-review

- Byte-for-byte canon: N/A to this task — no method bodies moved or
  edited; only field-access receivers inside already-screen-resident
  (excluded) methods, delegator deletions, and comment/docstring text
  were touched, all explicitly within the cleanup PR's own transform
  scope per recipe §1 step 3.
- No assertion VALUES changed anywhere in the 301 test-side retargets —
  verified by running the affected-file battery before AND after and
  diffing pass/fail sets (§10); every failure present after retargeting
  is confirmed identical on the pristine pre-task tree.
- Every "0 external references" delegator-prune verdict double-checked
  with a second, broader grep pattern before deletion (skills'
  methodology-trap lesson).
- Every dead-import candidate checked against `_SURFACE` individually
  before deletion — caught one collision (`_ingestible_file_filters`),
  reconfirming (not newly discovering — see the correction in §6) the
  conversations exemplar's own Task 7 precedent.
- Census-listing correction (task 2's carried minor) verified
  independently via a fresh, from-scratch re-derivation (not copied from
  the task brief's own claim) — found the exact 3 files named, confirmed
  each is LATENT by reading its own patch call site.
- Open risk: the cross-controller `_sync_library_canvas` module-globals
  audit (flagged by task 2, reaffirmed here) remains unfixed across the
  other five controllers — explicitly out of this task's own scope per
  the review that filed it.
- Two genuine mistakes caught during self-review before finalizing this
  report, neither caught by the battery itself: (1) an initial "targeted
  battery" run of 16 files omitted `test_library_screen.py` and
  `test_library_shell.py` — the two LARGEST diffs (26 and 52 retargets)
  — entirely; closed by a standalone `test_library_screen.py` run (32/32
  passed) and by confirming the full xdist sweep already covers `test_
  library_shell.py` comprehensively. (2) a first draft of the recipe's
  own §20/report §6 claimed this task's `_SURFACE` collision
  (`_ingestible_file_filters`) was the FIRST such collision across all
  cleanup tasks — false; re-reading recipe §11 found the conversations
  exemplar's own Task 7 hit the identical shape first
  (`LIBRARY_CONVERSATION_READER_MAX_CHARS`). Both corrected in place
  before this report's own final pass, per the same "verify a claim
  against the recipe's own prior sections before recording it" discipline
  §11's lesson itself prescribes.
