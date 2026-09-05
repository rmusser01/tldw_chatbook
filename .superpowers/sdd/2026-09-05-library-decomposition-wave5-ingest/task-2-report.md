# Wave-5 Task 2 report — Ingest controller PR (series 2/3)

Plan: `Docs/superpowers/plans/2026-09-05-library-decomposition-wave5-ingest.md`.
Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§19 (mechanics
authority). Branch: `refactor/library-decomp-wave5-ingest`. Worktree:
`.worktrees/library-decomp-foundation`. Base: `74a6f5774` (task 1's own
closing commit).

Commits:
- `44ab7383b` — `test(library): handler pins + wiring pins for the ingest
  controller (RED)`
- `68a896993` — `refactor(library): ingest controller (ingest series 2/3)`
- `18e9c60f7` — `docs(library): blame-ignore the ingest controller move
  commit`

## 1. The hard precondition — 5 real `.press()`/message pins, RED commit

Per the reviewer-mandated precondition (task 1's own deferral), all 5
handlers task 1 left as coverage debt got a real DOM-driven pin in the RED
commit, before any body moved:

| Handler | Selector | Fixture technique | Result pre-move |
|---|---|---|---|
| `handle_library_ingest_cancel` | `.library-ingest-cancel-{job_id}` | `LibraryIngestJobRegistry.submit()` + `mark_parsing()` + `update_progress(phase="transcribing")`, then a real `.press()`; `app_instance.cancel_local_ingest_job` mocked | `test_cancel_button_requests_cancellation_of_the_active_local_attempt` — pass |
| `handle_library_ingest_force_stop` | `.library-ingest-force-stop-{job_id}` | Same registry setup + `cancel_requested: True` in progress; `app_instance.force_stop_local_ingest_job` mocked | `test_force_stop_button_force_stops_the_active_local_attempt` — pass |
| `handle_library_ingest_retry_faster_whisper` | `.library-ingest-retry-faster-whisper-{job_id}` | `mark_failed(error_detail={"category": "stt_failure", "actions": ["retry_faster_whisper"]})`; `app_instance.retry_library_ingest_job_with_provider` mocked | `test_retry_faster_whisper_button_retries_with_the_named_provider` — pass |
| `handle_library_ingest_option_reset` | `#opt-generic-reset` | Real canvas mount, staged non-default `type_options["generic"]`, real press | `test_option_reset_button_wipes_the_generic_panel_to_defaults` — pass (assertion corrected in-flight: the generic mirror re-populates from the state builder after reset, not `{}` — read, not guessed) |
| `handle_library_ingest_directory_browse` | `#opt-audio_video-transcription_model_dir-browse` (message-based `@on`) | Real canvas mount with `transcription_provider="parakeet-onnx"`, `_is_installed` patched True, real press captures the `push_screen` callback | `test_directory_browse_button_opens_a_real_directory_picker` — pass |

All 5 attempted (and succeeded at) real `.press()`/message-driven dispatch,
per the plan's own "attempt `.press()` first" instruction — none needed the
direct-call fallback. Confirmed via `Tests/UI/test_library_ingest_
characterization.py` at RED (9/9 passed, before the screen was touched) and
again at GREEN (9/9 passed, unchanged).

Two of the five (`_option_reset`, `_directory_browse`) turned out to be
EXCLUDED from the move anyway (an unrelated `object.__new__`-bypass hazard
— see §3) — the plan's own dispatch note ("the pin's value does not depend
on that outcome") is exactly right: these two pins now stand as the ONLY
real DOM-dispatch coverage those two permanently screen-resident handlers
have, regardless of where their bodies live.

## 2. Final cluster + every exclusion decision (78 candidates)

Fresh census at execution time (recipe's own "never trust a carried-over
count" rule): `ast` scan of `LibraryScreen` for method names containing
"ingest" (case-insensitive) — **78 raw `FunctionDef` matches, 78 unique**
(no property/setter-pair gap, unlike Skills' 133/127), matching task 1's
own count exactly. Reverse oddball scan (any non-"ingest"-named method
called by 2+ ingest-named methods; any ingest-named method reaching a field
outside the `_library_ingest_` prefix) found nothing beyond task 1's own
already-excluded shared-shell fields.

**57 move. 21 excluded, in four shapes:**

### (a) 4 `@work(thread=True)` framework-decorator hazard
`_save_library_ingest_backend`, `_persist_library_ingest_location`,
`_run_library_ingest_preflight`, `_save_library_ingest_options`. Textual's
`@work` decorator asserts `isinstance(self, DOMNode)` at call time (export
series' own precedent, recipe §12) — a plain controller object fails this
unconditionally. Stay screen-resident, decorator and body untouched. Two
have a mover caller (`_persist_library_ingest_location` ← `handle_library_
ingest_browse`; `_run_library_ingest_preflight` ← `_trigger_library_
ingest_preflight`), reached via a named dependency; the other two have zero
mover callers (their only callers are ALSO excluded, for other reasons).

### (b) 2 module-globals-coupling
- `_remember_library_ingest_location` — reads the bare name `save_setting_
  to_cli_config` (an ordinary `from ...config import (...)` name in
  `library_screen.py`, resolved against the DEFINING module's `__globals__`
  at call time). `Tests/UI/test_library_screen.py::test_ingest_browse_
  remembers_the_directory_of_the_picked_file` patches `tldw_chatbook.UI.
  Screens.library_screen.save_setting_to_cli_config` and calls the real,
  `__init__`-constructed screen's method directly, expecting the internal
  free-name call to observe the patch — moving the body would silently
  repoint it. Its only caller, the ALSO-excluded (shape a) `_persist_
  library_ingest_location`, needs no binding.
- `_load_library_ingest_options_from_config` — calls the bare name
  `_library_ingest_options_for`, one of recipe §3's own permanently
  screen-routed trio (`library_screen.py:605-692`, confirmed untouched).
  Unlike the trio's own hazard (about the trio's INTERNAL mutual
  resolution), moving this METHOD is safe for test-patch-reach purposes —
  `_library_ingest_options_for`'s own body still resolves through ITS OWN
  module's globals regardless of caller. The real blocker is CIRCULAR
  IMPORT: every controller import in `library_screen.py` sits above line
  605 where the trio is defined; a module-level `from ..Screens.library_
  screen import _library_ingest_options_for` in the new controller module
  would try to import a name that does not exist yet on a module still
  mid-execution. A deferred import inside the moved body would violate the
  byte-for-byte canon. Excluded; its only caller is `on_mount` (a shell
  method, not a mover), so no binding needed.

### (c) 9 unbound-fake-self / `object.__new__`-bypass
Repo-wide content grep (never `-k`), two passes: (1) `grep -rn
"LibraryScreen\.<name>\("` across all of `Tests/` for every one of the 78
candidates, catching literal unbound calls; (2) for every `object.__new__(
LibraryScreen)`/`LibraryScreen.__new__(LibraryScreen)` assignment found in
the same census, a second pass tracking that variable name and searching
the SAME enclosing function body for `<var>.<name>(` BOUND calls — the
unbound-only grep alone missed these (found via `_build_ingest_options_
snapshot`'s 5 sites in `Tests/App/test_submit_library_ingest_job.py`).

- `_do_submit_ingest` — `Tests/UI/test_library_ingest_canvas.py` (4 sites),
  `Tests/integration/test_library_ingest_flow.py` (2 sites), all `object.
  __new__`/`.LibraryScreen.__new__` bypass screens called unbound. Caller
  `_submit_library_ingest_form` (mover) reaches it via a named dependency.
- `_enqueue_library_ingest_snapshot` — 1 site, same shape. Only caller is
  the ALSO-excluded `_do_submit_ingest`; no binding needed.
- `_build_ingest_options_snapshot` — 5 sites (bound calls on `object.
  __new__` bypasses, found only by the second census pass). Caller
  `_current_library_ingest_start_consent` (mover) reaches it via a named
  dependency.
- `_library_ingest_browse_location` — `Tests/Utils/test_config_nested_
  settings.py::test_library_ingest_browse_location_audit_fix`, unbound call
  on a bare `SimpleNamespace` — found OUTSIDE every one of the recipe's own
  four canonical test roots, confirming the plan's "widen beyond the
  obvious roots" mandate genuinely mattered here. Caller `handle_library_
  ingest_browse` (mover) reaches it via a named dependency.
- `_run_debounced_library_ingest_preflight` — `Tests/Library/test_ingest_
  preflight_egress.py::test_the_typing_debounce_forbids_probing_even_when_
  it_is_enabled`, unbound call on a bare `SimpleNamespace`. REFERENCED (not
  called) by `handle_library_ingest_path_changed` (mover, as a `set_timer`
  callback) — reached via a named dependency.
- `handle_library_ingest_backend_switch` — `Tests/UI/test_library_ingest_
  canvas.py`, unbound call on an `object.__new__` bypass. Zero mover
  callers (`@on`-dispatched only).
- `handle_library_ingest_directory_browse` — same file, same shape. Zero
  mover callers. One of the plan's 5 hard-precondition handlers; pinned
  regardless (§1).
- `handle_library_ingest_option_reset` — same file, same shape. Zero mover
  callers. Also one of the 5 hard-precondition handlers; pinned regardless.
- `handle_library_ingest_option_value_changed` — `Tests/UI/test_library_
  canvas_scoped_sync.py::test_ingest_checkbox_routes_to_ingest_canvas_
  sync`, unbound call on a bare `SimpleNamespace`. Zero mover callers.

### (d) 6 instance-attribute-monkeypatch — found ONLY by running the battery
Recipe §3 shape 2 (skills' own `_request_library_skills_browse`
precedent), but at nearly triple the prior high-water mark. None of these
were visible to the static census; each was found by a real test failure
across five successive draft rounds (see §5 for the exact evidence chain):

- `_build_library_ingest_state` — 13 mover callers (`_current_library_
  ingest_start_consent`, `_submit_library_ingest_form`, `_update_library_
  ingest_dynamic_regions`, `action_library_ingest_back`, and the
  `handle_library_ingest_{title,author,keywords}_changed`/`_path_changed`/
  `_path_submitted`/`_clear_path`/`_expand_all`/`_collapse_all`/`_clear_
  finished` family). `Tests/UI/test_library_ingest_inline_consent.py`'s
  own `_minimal_library_screen()` (an `object.__new__` bypass shared by
  ~55 tests) does `screen._build_library_ingest_state = MagicMock(...)`.
- `_notify_library_ingest_warning` — 3 mover callers (`handle_library_
  ingest_cancel`, `_submit_library_ingest_form`, `_resolve_ingest_source`).
  Same shared fixture.
- `_update_library_ingest_gate` — 7 mover callers (`handle_library_ingest_
  {title,author,keywords}_changed`, `_update_library_ingest_dynamic_
  regions`, `_submit_library_ingest_form`, `handle_library_ingest_path_
  changed`, `action_library_ingest_back`). Same shared fixture.
- `_refresh_library_ingest_canvas_preserving_context` — 2 mover callers
  (`_update_library_ingest_dynamic_regions`, `_restage_library_ingest_
  last_submission`). Same shared fixture.
- `_update_library_ingest_dynamic_regions` — 13 mover callers (`_apply_
  library_ingest_preflight_result`, `_handle_library_ingest_progress_
  changed`, `_handle_library_ingest_registry_changed`, `_on_ingest_job_
  details`, `_trigger_library_ingest_preflight`, and the `handle_library_
  ingest_{clear_finished,clear_path,collapse_all,dismiss,expand_all,path_
  changed,retry,retry_faster_whisper}` family). Found INDEPENDENTLY, by a
  real Pilot test on a REAL, mounted screen: `Tests/UI/test_library_
  shell.py::test_library_ingest_progress_action_change_recomposes_
  dynamic_regions` does `screen._update_library_ingest_dynamic_regions =
  Mock(wraps=screen._update_library_ingest_dynamic_regions)`, presses a
  real backend-switch button, and asserts the mock observes the resulting
  registry-listener-driven call.
- `_library_ingest_job_by_id` — 2 mover callers (`handle_library_ingest_
  open`, `handle_library_ingest_view_on_server`). Found independently
  again, on a REAL `LibraryScreen(MagicMock())` fixture: `Tests/UI/test_
  library_screen.py::test_handle_library_ingest_open_wires_to_open_job_
  in_library` patches it, then calls `await screen.handle_library_ingest_
  open(event)` and asserts the mock was consulted.

All 6 stay on `LibraryScreen`, UNMOVED, full-bodied; every mover caller
reaches each through a named late-binding lambda (recipe's own binding
kind 2 — re-reads `screen.<name>` at CALL time, which is exactly why every
one of these bypass fixtures keeps working unmodified after the move).

**A repeat, method-scoped AST census for the recipe's sixth bypass shape**
(bare `self` as an identity-compared argument, or an unbound-attribute
escape via `getattr(self, "<literal>", default)`) found ZERO instances in
the final 57-mover set — no `cancel_group`/`_library_screen_is_current`
call, no bare-`self` `ast.Compare`, and every `getattr(self, "<literal>",
default)` in the moved bodies resolves to either a real `_library_ingest_
<field>` state property or the `app_instance` framework service, both of
which the controller genuinely carries.

## 3. Single vs. split — single, by call-graph connected-components

A `self.<name>(...)` internal call-graph among the original 78 candidates
(pre-exclusion) is ONE dense connected component: hub names (`_update_
library_ingest_dynamic_regions`, `_build_library_ingest_state`, `_library_
ingest_registry`, `_disarm_library_ingest_start_confirm`) are each called
from 3–13 different sibling methods spanning path-entry, pre-flight,
submit, and queue-row concerns. No subset of the cluster calls only within
itself. **Decision: ONE combined `LibraryIngestController`**, matching the
skills/search+RAG precedent's identical resolution at comparable scale.

## 4. Bind classification (byte-for-byte canon, both binding kinds)

**Kind 1 — 15 framework services**, live-read `@property` on every access:
`app`, `app_instance`, `call_after_refresh`, `is_attached`, `is_mounted`,
`is_running`, `notify`, `query`, `query_one`, `refresh`, `register_footer_
shortcuts`, `run_worker`, `set_focus`, `set_timer`, `size`. `is_running`
was added mid-task (§5) once the battery found `_apply_library_ingest_
backend_save` forwarding bare `self` into the shared `_sync_library_canvas`
dispatcher. One more name joins this group for a narrower reason:
`LIBRARY_INGEST_SHORTCUTS` is a `LibraryScreen` CLASS attribute (a literal
tuple, not an `__init__` field) `Tests/UI/test_library_ingest_keyboard.py`
reads directly off the screen — kept there permanently, exposed via the
same live-read pass-through shape rather than a duplicated literal.

**Kind 2 — named constructor dependencies:**
- (a) 13 general Library-wide shell helpers: `_apply_library_notes_stage_
  visibility`, `_focus_library_hub_entry`, `_invalidate_library_external_
  submission`, `_library_landing_attention_action`, `_open_job_in_
  library`, `_open_library_external_media_detail`, `_open_transcribe_cpp_
  gguf_picker`, `_refresh_local_source_snapshot` (one of recipe §3's four
  PERMANENTLY screen-routed monkeypatch names), `_safe_text`, `_select_
  library_rail_row`, `_server_binding_is_shipped_placeholder`, `_sync_
  library_emergency_guard_presentation`, `_sync_library_landing_lifecycle_
  presentation`.
- (b) 7 shared shell state accessors: reads `_library_selected_row_id`,
  `_transcribe_cpp_configured`, `_footer_shortcut_registration`, `_library_
  canvas_projection_depth`; reads+writes `_library_rail_collapsed`,
  `_library_landing_attention_signature`, `_library_canvas_resync_
  pending` (the last pair added alongside `is_running`, same `_sync_
  library_canvas` forwarding cause, mirroring skills/RAG's own identical
  pair exactly).
- (c) NONE — no wiring accessor pair exists for Ingest (task 1's own
  finding: no field holds a live controller/coordinator instance).
- (d) N/A — no merely-delegate-to-existing-controller properties exist for
  Ingest (unlike Skills' import-coordinator precedent).
- (e) 12 named late-binding callables for the exclusions above that a
  MOVER still calls/references internally: `_build_ingest_options_
  snapshot`, `_build_library_ingest_state`, `_do_submit_ingest`, `_library_
  ingest_browse_location`, `_library_ingest_job_by_id`, `_notify_library_
  ingest_warning`, `_persist_library_ingest_location`, `_refresh_library_
  ingest_canvas_preserving_context`, `_run_debounced_library_ingest_
  preflight`, `_run_library_ingest_preflight`, `_update_library_ingest_
  dynamic_regions`, `_update_library_ingest_gate`.

**Class-level constants, one exception aside.** Three `LibraryScreen`
class-body literals (`_RETRY_CONFIRM_DEAD_ZONE_SECONDS`, `_START_CONFIRM_
DEAD_ZONE_SECONDS`, `_CLEAR_FINISHED_DEAD_ZONE_SECONDS` — each a bare `0.3`
used by exactly one, now-moved, method and referenced NOWHERE else,
confirmed by repo-wide grep) are DELETED from the screen and declared
fresh on the controller — the class-constant analogue of a state-PR field
deletion for a zero-external-reference field. `LIBRARY_INGEST_SHORTCUTS`
is the one exception (kind 1 above): permanent screen-side test
dependency, never deleted, never duplicated.

**Construction order.** `LibraryScreen.__init__` builds `self._ingest_
controller` right after `self._skills_controller`, matching every other
controller in the file.

## 5. Exclusion-count trajectory — the battery-driven correction chain

The recipe's own "a battery-found hazard shrinking the mover set
legitimately amends the RED tuple" rule (§3) applied FIVE times in this
task, each correction re-deriving the full count rather than patching one
number:

| Round | Trigger | Movers | Exclusions |
|---|---|---|---|
| 0 (initial build) | Static census only | 63 | 15 |
| 1 | `test_submit_without_warnings_is_a_single_press` etc. — `_build_library_ingest_state` monkeypatch bypassed | 62 | 16 |
| 2 | Same shared fixture — `_notify_library_ingest_warning`, `_update_library_ingest_gate`, `_refresh_library_ingest_canvas_preserving_context` all bypassed identically | 59 | 19 |
| 3 | `test_library_ingest_progress_action_change_recomposes_dynamic_regions` (real Pilot, real screen) — `_update_library_ingest_dynamic_regions` bypassed | 58 | 20 |
| 4 | `test_handle_library_ingest_open_wires_to_open_job_in_library` (real screen, different file) — `_library_ingest_job_by_id` bypassed | 57 | 21 |
| 5 | `is_running`/`_library_canvas_projection_depth`/`_library_canvas_resync_pending` added as bindings (no mover count change — `_apply_library_ingest_backend_save`'s `_sync_library_canvas` forwarding was already a mover, just missing a dependency) | 57 | 21 |

Each round required a FULL rebuild from the pre-move tree (never patching
the already-modified `library_screen.py` in place), re-running the AST
extraction, re-generating the controller file, re-running the delegator
transform, and re-verifying byte-for-byte identity — done via `git
checkout HEAD -- tldw_chatbook/UI/Screens/library_screen.py` before each
round, never by hand-editing the intermediate result.

## 6. Byte-for-byte verification

Verified programmatically: for each of the 57 final movers, extracted the
method's source (decorators through end, using the SAME `ast`-derived line
range) from BOTH the pre-move tree (`git show 74a6f5774:tldw_chatbook/UI/
Screens/library_screen.py`) and the new controller file, and compared the
raw text.

```
total movers checked: 57
mismatches: []
```

Zero differences — every moved body, including its decorator line(s) and
docstring, is character-identical to the pre-move original.

## 7. Free-name walk

A `LOAD_GLOBAL`-style census (walk every `ast.Name` with `Load` context in
each of the 57 movers, excluding `self`, local variables/params, nested
function names, and builtins) against the controller module's own
namespace after import found only two hits, both confirmed false
positives from walking a property's OWN `@x.setter` decorator expression
(`_library_rail_collapsed`, `_library_landing_attention_signature`, and
`_library_canvas_resync_pending` — each name legitimately appears as a
bare `Name` inside its own setter's decorator, not as an undefined
reference inside a method body). Two real gaps were found and fixed
DURING this same census discipline before the final build: `ingest_scope_
label` (missing from the `library_ingest_canvas` import) and a missing
`json` stdlib import — both closed in the header before the final
controller file was written; the census re-run after each fix confirmed
zero remaining gaps.

## 8. Verification battery

All commands from `.worktrees/library-decomp-foundation`, `.venv/bin/
python`, `-p no:randomly` where applicable.

**RED commit (`44ab7383b`)**, verified before committing: `Tests/
Architecture/test_library_ingest_wiring.py` — 4 of 6 tests fail
(`test_ingest_controller_owns_its_cluster`, `test_screen_delegates_ingest_
handlers`, `test_ingest_cluster_staticmethods_forward_to_the_controller_
class`, `test_ingest_controller_exposes_every_state_field`; the 2
state-only tests, unaffected, pass). `Tests/UI/test_library_ingest_
characterization.py` — 9/9 passed pre-move (confirms the 5 new
hard-precondition pins characterize CURRENT, unmoved behavior).

**GREEN commit (`68a896993`)**, verified before committing:
- Wiring suite: 6/6 passed.
- Characterization: 9/9 passed.
- `test_library_ingest_inline_consent.py` (the file most exposed to the
  instance-attribute-monkeypatch shape): 55/55 passed.
- Both size ratchets: pass (`library_screen.py` 41520/1302 → 40096/1302;
  `library_ingest_controller.py` born-governed at 2510 lines/37-parameter
  constructor).
- `-k "ingest and library"` across `Tests/UI`+`Tests/Library`+`Tests/App`+
  `Tests/integration` (1303 passed/7 failed) — all 7 failures confirmed
  IDENTICAL to task 1's own documented pre-existing list (`test_the_probe_
  reports_a_redirect_as_an_answered_status_not_an_error`, `test_progress_
  detail_paints_below_row_without_obscuring_actions_or_neighbor[size0/
  size1]`, `test_fold_hint_is_pinned_not_scrolled`, `test_outcome_lines_
  paint_heavier_than_the_tooling_summary`, `test_every_canvas_focusable_
  changes_at_the_glyph_level_on_focus`, `test_registry_ticks_only_reflow_
  footer_when_retry_availability_changes`).
- Full `Tests/Architecture/`: 551 passed/16 failed/1 skipped — all 16
  category-matched to wave-4 close's own documented pre-existing bucket
  (Console realtime ×1, review-selection ×1, wave6 closeout ×1, wave6
  inventory ×3, default-timeout-session-guard ×1, persistent-diagnostic-
  inventory ×2, chat_screen ratchet ×2, timer-path-static-update-inventory
  ×3, worker-exclusive-group-inventory ×2). One INITIALLY-3rd diagnostic-
  inventory failure (`test_production_diagnostic_inventory_and_sink_
  topology_are_unchanged`) traced to the moved `_resolve_ingest_source`
  body's own `logger.opt(exception=True).warning(...)` call relocating
  verbatim from `library_screen.py` to the new controller file (digest
  `1abdbd0be7261096` unchanged, confirmed via `scripts/check_persistent_
  diagnostic_inventory.py --statements ... --since <pin-commit>` before
  regenerating) — fixed by `--write`-regenerating `Docs/security/
  production-diagnostic-inventory.json` in the GREEN commit, per that
  script's own review-then-regenerate contract; re-confirmed flaky/
  order-dependent on the PRISTINE baseline too (passed on an isolated
  single-file rerun there), not a real regression either way.
- `preflight.sh`: all six derived-artifact checks pass (CSS bundle,
  profile-owned-path census [48/18/46], production diagnostic inventory
  [574 owners, 1338 TASK-492, 7599 TASK-494, 10 sinks], backlog task-id
  sweep [3240 files], chachanotes table allowlist [105 tables], index
  plan pins [270/270, 57 pinned]).

**Full sequential xdist paired-baseline sweep** (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`), branch then an ISOLATED `git
worktree add /tmp/w5task2base 74a6f5774` + its own `uv venv` + `uv pip
install -e ".[dev]"` baseline (never a same-tree overlay, per the
recipe's own interruption-safety lesson):

| | Failed | Passed | Wall time |
|---|---|---|---|
| Branch (`68a896993`) | 358 | 3992 | 1473.71s (24:34) |
| Baseline (`74a6f5774`, isolated worktree) | 356 | 3989 | 1521.53s (25:22) |

Both inside the documented ~330–371 historical backdrop. 350 shared, 6
baseline-unique (not investigated further, per §7's own established
precedent), **8 branch-unique** — all resolved, zero unexplained:

- 6 passed cleanly on a combined single-process re-run (ordinary xdist
  noise): `test_library_media_reader_match_nav_t22209.py::test_a_new_
  document_rescans_for_the_same_query`, `test_library_prompt_
  collections.py::test_library_screen_manager_create_search_rename_and_
  explicit_all`, `test_library_prompts_canvas.py::{test_library_prompt_
  pager_first_and_filter_failure_states[size0], test_library_prompt_undo_
  refreshes_applied_page_and_preserves_basket}`, `test_library_shell.py::
  {test_library_media_durable_mutation_gates_and_refreshes_applied_
  scope[True], test_library_shell_blank_note_typed_then_deleted_all_is_
  gc_from_real_db}`.
- 2 reproduced identically in the same combined re-run:
  `test_library_notes_reader.py::test_wide_editor_deep_link_keeps_reader_
  navigation_and_local_back` is the SAME name already documented (wave-3
  task 5) as bidirectional run-to-run flakiness, reconfirmed here too;
  `test_library_media_reader_traversal_t22207.py::test_loading_banner_
  paints_in_place_without_body_rebuild` is a NEW name, confirmed
  pre-existing by reproducing identically in TRUE isolation on the
  isolated pristine-baseline worktree (a Media-reader loading-banner paint
  test; this task's own diff touches zero Media-reader code). Recipe §7
  updated with this task's sweep-evidence entry.

None of the 8 touches Ingest code or this task's own diff. **Zero real
regressions.**

## 9. Fresh pins (post-move, re-derived)

`Tests/Architecture/test_library_ingest_wiring.py`'s `_INGEST_CLUSTER_
METHOD_NAMES` tuple holds exactly the 57 final movers (re-derived from the
Task-2-corrected exclusion set, not the original 63/62/59/58-mover drafts);
`test_ingest_cluster_method_names_are_genuinely_ingest_named` asserts the
count (`== 57`) and the naming invariant (every name contains "ingest").
`test_ingest_controller_owns_its_cluster`/`test_screen_delegates_ingest_
handlers`/`test_ingest_cluster_staticmethods_forward_to_the_controller_
class`/`test_ingest_controller_exposes_every_state_field` all pass against
the landed controller. No delegators are pruned yet (Task 3's own scope,
not this one) — `test_screen_delegates_ingest_handlers` confirms all 57
delegators exist and forward by same-name pattern-match against the
source, not a loose substring check.

## 10. Files changed

- `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py` (new,
  2510 lines) — `LibraryIngestController`, 57 moved methods.
- `tldw_chatbook/UI/Screens/library_screen.py` — import + controller
  construction added; 57 method bodies replaced by one-line delegators; 3
  dead class-level constants deleted. 41520 → 40096 lines, 1302 methods
  unchanged (pure move).
- `Tests/Architecture/test_library_ingest_wiring.py` — rewritten to the
  controller-PR shape (58→57-name cluster, corrected across the 5
  battery-driven rounds).
- `Tests/UI/test_library_ingest_characterization.py` — 5 new `.press()`/
  message-driven pins for the hard-precondition handlers; module docstring
  extended.
- `Tests/UI/test_library_shell.py` — new shared `wire_bypass_ingest_
  controller(screen)` helper + import.
- `Tests/UI/test_library_ingest_canvas.py` (16 call sites), `Tests/App/
  test_submit_library_ingest_job.py` (5), `Tests/integration/test_library_
  ingest_flow.py` (2), `Tests/UI/test_library_ingest_retry_last.py` (1) —
  one `wire_bypass_ingest_controller(screen)` call inserted immediately
  after each existing `screen._ingest_state = LibraryIngestState()` seed
  line (task 1's own seed points), plus the shared-helper import.
- `Tests/UI/test_library_ingest_inline_consent.py` — an equivalent LOCAL
  `_wire_bypass_ingest_controller` helper (this file's own shared bypass
  fixture, `_minimal_library_screen()`, is used ~55 times in-file) + one
  call site + `LibraryIngestController` import.
- `Tests/Architecture/test_screen_size_ratchet.py` — `_BUDGETS` row
  lowered, comment appended.
- `Tests/Architecture/test_library_modules_size_ratchet.py` — new
  born-governed row for `library_ingest_controller.py`.
- `Docs/security/production-diagnostic-inventory.json` — regenerated
  (one statement's owner file changed; digest unchanged).
- `.git-blame-ignore-revs` — the GREEN commit's hash appended.
- `backlog/docs/library-decomposition-recipe.md` — §7 sweep-evidence list
  extended with this task's own entry.
- `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/
  progress.md` — task log updated.

## 11. Self-review

- **Hard precondition met**: all 5 deferred handlers got a real `.press()`/
  message-driven pin in the RED commit, verified passing BEFORE any body
  moved (not merely written — actually run against the pre-move tree).
  Two turned out excluded from the move anyway; the pins stand
  independent of that outcome, exactly as the plan anticipated. ✅
- **All bypass censuses by content-grep, never `-k`**: confirmed for
  every shape (unbound-fake-self/`object.__new__`, module-globals,
  instance-attribute-monkeypatch) — the ONE shape genuinely invisible to
  static grep (instance-attribute-monkeypatch) was closed by running the
  FULL file/suite repeatedly, not by inventing a smarter grep for it
  (recipe's own point: this shape is fundamentally battery-found, not
  census-found). ✅
- **Byte-for-byte canon**: verified programmatically (not by eye), 0
  mismatches across all 57 movers including decorators. ✅
- **Free-name walk**: ran to completion, found and fixed 2 genuine import
  gaps before the final build, re-ran clean after. ✅
- **Single-vs-split**: derived from an actual call-graph connected-
  components computation, not assumed. ✅
- **Born-governed row same commit**: `library_ingest_controller.py`'s
  ratchet row landed in the SAME commit that created the file. ✅
- **Fresh pins**: the wiring test's cluster tuple reflects the FINAL
  57-mover set, not an intermediate draft — re-derived, not hand-patched,
  after each of the 5 correction rounds. ✅
- **The largest risk this task carried**: the instance-attribute-
  monkeypatch shape recurring SIX times (triple the prior high-water
  mark, skills' own 1-instance precedent) across FIVE separate discovery
  rounds, two of which required a full tree-rebuild-and-reverify cycle
  each. Every one was found by an ACTUAL test failure (never assumed,
  never "probably also affected"), and the final battery — full `-k`
  sweep, full `Tests/Architecture/`, and the full sequential xdist
  paired-baseline sweep — surfaced zero further instances of this or any
  other shape. I'm confident this class is fully closed for this
  subsystem's controller PR.
- **Known, intentional deferral**: no screen delegator is pruned yet (57
  of 57 stay); dead-delegator pruning, receiver-normalization to `self.
  _ingest_state.<field>` in remaining screen code, and shim-block deletion
  are Task 3's (cleanup, series 3/3) explicit scope, matching every prior
  subsystem's own task boundary.
- **One thing I would flag for the wave-close review**: the exclusion
  count moved five times in-session (63→62→59→58→57). Each move is
  individually well-evidenced (see §5's table) and the final commit's own
  numbers are internally consistent across the module docstring, the
  wiring test, and both ratchet comments (cross-checked once more before
  writing this report) — but a reviewer re-deriving the census from
  scratch should expect to land on 57/21, not the plan's own pre-task
  estimate, and should treat that as expected convergence, not drift.

## 12. Fix round 1 (post-review)

Coordinator review round 6 (the mandated exhaustive module-globals census)
found 1 CRITICAL + 2 Important + 2 minors against the original report/
commits above. All addressed in commit `fix(library): exclude _resolve_
ingest_source (module-globals patch bypass); mechanical globals census
(fix round 1)`.

### CRITICAL — `_resolve_ingest_source` was a green-but-vacuous move

**The method.** The coordinator's own mechanical module-globals census
(now recipe §3's newest numbered "eighth bypass shape" entry, added in
this fix round — see below): enumerate every bare module-global name a
moved body reads (an `ast.Name` `Load` matching a `from x import name` at
the NEW module's top, never a `self.<name>` attribute — those are already
covered by the two established binding kinds), then grep ALL of `Tests/`
for a `library_screen`-scoped patch target among them, checking THREE
spellings: the direct-attribute form (`library_screen_module.<name>`),
the fully-qualified string-patch form (`"tldw_chatbook.UI.Screens.
library_screen.<name>"`), and the two-argument `setattr`/`patch.object`
form (`library_screen_module, "<name>"` — a bare STRING argument, the one
spelling my OWN first pass at this census (during the original build)
never checked, silently undercounting `validate_path_simple`'s 3 real
patch sites to 0).

**The finding.** `_resolve_ingest_source` reads the bare names `validate_
path_simple`/`validate_url`. `Tests/UI/test_library_shell.py::test_
library_shell_ingest_canvas_invalid_path_notifies_and_submits_nothing`
patches `tldw_chatbook.UI.Screens.library_screen.validate_path_simple`
with a stub that unconditionally raises `ValueError`, then presses
`#library-ingest-start` with a source path `"/tmp/whatever.txt"` that
does not exist on disk. Once the body moved, the controller's own
separately-imported `validate_path_simple` binding was the one actually
called -- the SCREEN-scoped stub was never reached. The test still PASSED
(warning fires, zero jobs submitted) because the REAL validator's own
"file does not exist" `ValueError` produces the identical observable
outcome as the stub's unconditional one -- a green-but-vacuous test, not
a red one, and therefore invisible to every RED/GREEN check, wiring
check, and xdist sweep this task's own battery ran.

**Confirmed genuine, not assumed**, by an existing-file probe (per the
review's own suggested method; the probe itself is a one-off `python -c`
script, never committed): created a REAL temp file, patched
`tldw_chatbook.UI.Screens.library_screen.validate_path_simple` to reject
it unconditionally, and called `_resolve_ingest_source` both directly on
the screen and through `screen._ingest_controller._resolve_ingest_source`.
With the body still moved, the direct screen call raised the stub's
rejection but the controller-forwarded call did NOT (proving the
patch-bypass); after the fix below, both paths correctly observe the
patch and return `None` with the expected warning.

**Fix.** Reverted `_resolve_ingest_source` to `LibraryScreen`, byte-for-
byte (re-verified against the pre-move `74a6f5774` original -- exact
match). Removed from `LibraryIngestController` entirely (including its
now-dead `logger`/`validate_path_simple`/`validate_url` imports -- all
three had zero other callers in the file, confirmed by grep before
removing). Its one mover caller, `_submit_library_ingest_form`, reaches
it through a new named late-binding dependency (`resolve_ingest_source`),
identical in shape to the 6 existing instance-attribute-monkeypatch
exclusions' own bindings. Mover count: 57 -> 56; exclusions: 21 -> 22
(3 module-globals-coupling now, was 2).

### IMPORTANT — `_apply_library_ingest_backend_save` / `_sync_library_canvas`

Same census, same shape: `_apply_library_ingest_backend_save` reads the
bare module global `_sync_library_canvas` (the shared cross-subsystem
canvas-sync dispatcher). The widened 3-spelling grep found ~20 patch
sites across 7 files (`test_library_file_notes_workspace.py`, `test_
library_entry_compose_once.py`, `test_library_note_import_flow.py`,
`test_library_review_round_t21116.py`, `test_library_media_trash.py`,
`test_library_notes_folder_navigator.py`, `Tests/Skills/test_skills_
import.py` -- the last 3 newly found by the widened grep, missed by my
original narrower one). Read every one: **zero is ingest-related** --
all 7 patch it for notes/media/skills canvas syncs. **Verdict: KEEP as a
mover**, recorded explicitly in the controller's own module docstring.
Rationale: (1) no ACTIVE test collision exists to fix, unlike `_resolve_
ingest_source`; (2) mechanically, `_sync_library_canvas` is a bare
FUNCTION call (not `self.<name>`), so it cannot be late-bound as a named
dependency without editing the moved body -- the only two accommodations
are "exclude the whole method" or "leave it, documented," and excluding
a working, correctly-tested method to guard a theoretical, zero-evidence
risk would be over-conservative; (3) the identical bare-`_sync_library_
canvas` shape exists in all five PRIOR controllers (conversations,
export, collections, search+RAG, skills) that call this same dispatcher
-- this is a systemic pattern, not an ingest defect. A cross-controller
audit of all six is recorded as a follow-up (recipe's new §3 entry
names it explicitly), NOT fixed retroactively here, per the review's own
scope instruction.

**Correction, filed by task 3 (ingest cleanup):** this 7-file/~20-site
count itself undercounted -- 3 more files (`test_library_canvas_scoped_
sync.py`, `test_library_notes_reader.py`, `test_review_set_walker.py`)
patch the same name via a variable name other than `library_screen`/
`library_screen_module` (`screen_module`, or the same name reached through
`monkeypatch.context()`'s own `patcher.setattr(...)` inside a multi-line
call my grep did not span), bringing the true count to 10 files/38 sites.
Re-read, all 10 remain confirmed LATENT with respect to this mover's own
call path (see `library_ingest_controller.py`'s own module docstring and
the recipe's §3 for the corrected, final listing) -- the verdict (KEEP as
a mover) is unchanged; only the recorded evidence was wrong.

### IMPORTANT — recipe gains the mechanical module-globals census (new numbered shape)

Added to `backlog/docs/library-decomposition-recipe.md` §3: the "eighth
bypass shape" entry, with the 4-step method (enumerate bare module
globals moved bodies read; grep all 3 patch spellings across ALL of
`Tests/`; read every hit to classify active-vs-latent; exclude on active,
document-and-keep on latent), this task's own CRITICAL finding as the
worked "active" example, and the `_apply_library_ingest_backend_save`
finding as the worked "latent" example. Also states the general rule:
this census is now a MANDATORY step of every future subsystem's
controller-PR sweep, run to completion regardless of how clean the
ordinary battery comes back -- a green-but-vacuous test is by
construction indistinguishable from a genuinely-passing one without
actually reading it.

### Minors

- **Ratchet comment arity**: re-verified with `inspect.signature(
  LibraryIngestController.__init__)` rather than hand-counted. Before this
  fix round: 37 params excl. `self` (1 positional `screen` + 36 keyword-
  only), confirmed via the SAME method against the pre-fix-round commit.
  After: 38 (1 + 37) -- a clean +1 matching the one new `resolve_ingest_
  source` dependency, no other drift. Both ratchet files' comments
  updated to state the verified numbers and the method used, not a
  hand-count.
- **`Tests/UI/test_parakeet_v2_install_ui.py`** named explicitly: run in
  this fix round (`25 passed`) as part of the closing verification --
  named here because its own filename/test names contain neither
  "ingest" nor "library" (task 1's own filter-blindness precedent file),
  so it is invisible to every `-k`-filtered sweep this task ran and must
  be checked by an explicit, content-grep-justified inclusion rather than
  assumed covered.

### Verification (fix round 1)

```
$ .venv/bin/python -m pytest Tests/Architecture/test_library_ingest_wiring.py \
    Tests/UI/test_library_ingest_characterization.py \
    Tests/UI/test_library_ingest_inline_consent.py \
    Tests/Architecture/test_screen_size_ratchet.py \
    Tests/Architecture/test_library_modules_size_ratchet.py \
    Tests/UI/test_parakeet_v2_install_ui.py \
    "Tests/UI/test_library_shell.py::test_library_shell_ingest_canvas_invalid_path_notifies_and_submits_nothing" \
    -p no:randomly -q
2 failed (both the documented pre-existing chat_screen.py ratchet rows), 130 passed
```

- Fixed module-globals census re-run: zero remaining ACTIVE collisions
  (the two hits still present -- `LIBRARY_ROW_INGEST_MEDIA`, a plain
  constant value-read, and `_sync_library_canvas`, the documented latent
  finding -- both confirmed non-actionable by reading every hit).
- Existing-file stub-fires probe: confirmed the fix (see CRITICAL section
  above); probe itself not committed.
- `-k "ingest and library"` across all four roots: 1303 passed/7 failed,
  all 7 identical to task 1's own documented pre-existing list (unchanged
  from before this fix round).
- Full `Tests/Architecture/`: 551 passed/16 failed/1 skipped -- same
  count and category as the original report's own §8 entry; the
  diagnostic-inventory regeneration (below) removed the one transient
  3rd failure that round had, and the two genuinely-flaky diagnostic-
  inventory tests remain (confirmed flaky, not caused by this fix, in the
  original report's own §8).
- `Docs/security/production-diagnostic-inventory.json` regenerated a
  second time: `_resolve_ingest_source`'s own `logger.opt(exception=True).
  warning(...)` statement moved back from the controller to the screen
  verbatim (digest `1abdbd0be7261096` unchanged, confirmed via `--
  statements --since <prior-pin-commit>` before writing).
- `preflight.sh`: all six derived-artifact checks pass.

### Files changed (fix round 1)

- `tldw_chatbook/UI/Screens/library_screen.py` — `_resolve_ingest_source`
  reverted to its full, byte-for-byte original body; one new named
  dependency (`resolve_ingest_source`) added to the `LibraryIngestController(...)`
  constructor call.
- `tldw_chatbook/UI/Library_Modules/library_ingest_controller.py` —
  `_resolve_ingest_source`'s moved body removed; a named-dependency
  constructor param/property added in its place; dead `logger`/`validate_
  path_simple`/`validate_url` imports removed; module docstring updated
  (exclusion counts, the new module-globals-coupling entry, and the
  `_apply_library_ingest_backend_save`/`_sync_library_canvas` latent-
  finding verdict).
- `Tests/Architecture/test_library_ingest_wiring.py` — `_resolve_ingest_
  source` removed from the cluster tuple; counts and docstrings corrected
  (56/22).
- `Tests/UI/test_library_shell.py`, `Tests/UI/test_library_ingest_inline_
  consent.py` — their respective `wire_bypass_ingest_controller`/`_wire_
  bypass_ingest_controller` helpers gained the new `resolve_ingest_source`
  binding (the constructor now requires it).
- `Tests/Architecture/test_screen_size_ratchet.py`,
  `Tests/Architecture/test_library_modules_size_ratchet.py` — both rows
  re-measured and re-pinned fresh (40096→40131 lines for the screen;
  2510→2536 lines and 37→38 verified constructor arity for the
  controller).
- `Docs/security/production-diagnostic-inventory.json` — regenerated.
- `backlog/docs/library-decomposition-recipe.md` — new §3 "eighth bypass
  shape" entry (the mechanical module-globals census, its 3-spelling grep
  method, and both this task's worked examples).

### Confirmation

The task's own commits (`44ab7383b`, `68a896993`, `18e9c60f7`,
`2b790783b`) are NOT amended -- this fix round lands as new, additional
commits on top, per the "never amend, always a new commit" discipline.
The RED/GREEN pair those commits establish is unaffected by this fix
round (the RED commit's own wiring-test rewrite and 5 characterization
pins never referenced `_resolve_ingest_source`'s eventual mover-vs-
exclusion status).
