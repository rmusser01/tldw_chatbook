# Wave-5 Task 1 report — Ingest state PR (series 1/3)

Plan: `Docs/superpowers/plans/2026-09-05-library-decomposition-wave5-ingest.md`.
Recipe: `backlog/docs/library-decomposition-recipe.md` §1–§19 (mechanics
authority). Branch: `refactor/library-decomp-wave5-ingest`. Worktree:
`.worktrees/library-decomp-foundation`. Base: `9e62dd8f7`.

Commits:
- `a11220648` — `test(library): characterization + wiring pins for the
  ingest extraction series (RED)`
- `12ba4fb13` — `refactor(library): ingest state object + shims (ingest
  series 1/3)`
- `3b83eab93` — `docs(library): record wave-5 task-1 sweep evidence and
  ledger close` (recipe/ledger documentation only, not part of the
  RED/GREEN pair the plan named)

## 1. Cluster enumeration + ownership analysis

`ast` walk of `LibraryScreen.__init__` for `_library_ingest_*` store
targets: **20 fields**, matching the plan's own 2026-09-04 measure exactly.
Screen measured fresh before any edit: **41574 lines, 1302 methods**
(matches the SDD ledger's own recorded baseline — no drift from the two
prior commits on this branch, neither of which touches `library_screen.py`).

### Oddball check (no prefix shortcuts)

- Substring scan for `"ingest"` anywhere in `LibraryScreen`'s `__init__`
  store targets, outside the `_library_ingest_` prefix: **zero** hits. No
  conversations-exemplar-style "startswith trap" here.
- Reverse check — do any of the 78 ingest-*named methods* reach into fields
  of a *different* prefix family: **yes, two adjacent clusters**, both
  confirmed NOT ingest-owned by reading every consumer (not assumed by
  name/proximity):
  - `_library_external_submit_generation/_scope_id/_worker/_backend/
    _consent/_busy/_status` (7 fields) — a separate "external source"
    onboarding feature (VAD-consent preparation). Only 2 of 78 ingest-named
    methods (`_do_submit_ingest`, `_enqueue_library_ingest_snapshot`) touch
    these; every field's *majority* consumer set is external-preparation-
    owned (`_apply_library_external_preparation`,
    `_apply_library_external_vad_progress`, `_confirm_library_external_vad`,
    `_invalidate_library_external_submission`, `_set_library_external_
    status`, `compose_content`). Stays screen-resident shared shell state.
  - `_transcribe_cpp_configured` — read by 2 ingest-named methods
    (`_build_library_ingest_state`, `_load_library_ingest_options_from_
    config`) but also by the unrelated `_apply_transcribe_cpp_gguf_result`
    model-install handler. Stays screen-resident.
  - `_library_model_install_progress_label/_owner` — zero ingest-named
    consumers at all (false lead from init-block proximity).

### The ingest-options trio — verified location, untouched

`_INGEST_OPTIONS_CACHE_ATTR` (`library_screen.py:605`),
`_read_library_ingest_options_from_config` (`:608`) and
`_library_ingest_options_for` (`:661`) are module-level `FunctionDef`s,
still resident in `library_screen.py`, with the file's own comment block
(`:714+`) explaining why they must stay together there (tests monkeypatch
`get_cli_setting`/`_read_library_ingest_options_from_config` on the
`library_screen` module object, and the patch only reaches `_library_
ingest_options_for`'s internal free-name call while both share that
module's globals). Read both function bodies: **neither references any
`self._library_ingest_*` instance field** — this state move touches
neither the trio nor anything it depends on. No move, no shim, nothing to
verify beyond confirming the location, which is done.

### `@work`-heavy flows (enumerated for the record; not this task's scope)

4 `@work(thread=True)`-decorated ingest methods, no `group=` kwarg on any:
`_save_library_ingest_backend`, `_persist_library_ingest_location`,
`_run_library_ingest_preflight`, `_save_library_ingest_options`. Relevant
to Task 2 (controller PR), not this state-only task.

### Shell-touched candidates named in the plan

- `_sync_library_ingest_rail_for_width` — touches exactly one ingest field
  (`auto_collapsed_rail`) plus shared shell state
  (`_library_rail_collapsed`, `_library_selected_row_id` — the canonical
  ≥2-subsystems field). Ingest-named but mixed shell+ingest body; census
  verdict for Task 2 to make, doesn't change field ownership here.
- `_pause_library_ingest_transient_ui` — touches 3 ingest fields
  (`clear_finished_armed`, `path_debounce_timer`, `preflight_generation`)
  plus calls into other ingest-named siblings and the external cluster.
  Called from `_select_library_rail_row_after_source_admission` (the
  shared shell rail-switch-hygiene dispatcher — same call site that also
  runs skills'/export's own per-subsystem hygiene resets). This is the
  task-2043 form-persistence contract's own implementation seam: it
  deliberately does NOT touch `form`, which is why the form survives rail
  switches.

### Full field-ownership table (recipe §2 script output)

All 20 fields: `NONE` or shell/plumbing-only non-ingest users, zero fields
claimed by another subsystem's own method-name prefix, zero fields BLOCKED
by the ≥2-subsystems rule. **20/20 MOVE, 0 wiring, 0 BLOCKED** — no field
holds a live controller/coordinator instance (unlike the `_conversation_
reader_controller`/`_library_collections_capture_controller`/`_library_
skill_import_coordinator` precedent), so there is no wiring-field exclusion
in this state PR at all — the simplest ownership shape of any subsystem to
date.

| Field | Non-ingest users found | Verdict |
|---|---|---|
| `auto_collapsed_rail` | `_library_resize_layout_signature`, `_set_library_rail_collapsed` (shell) | MOVE |
| `backend_generation` | none | MOVE |
| `backend_save_lock` | none | MOVE |
| `backend_target` | none | MOVE |
| `batch_baseline` | none | MOVE |
| `clear_finished_armed` | none | MOVE |
| `clear_finished_armed_at` | none | MOVE |
| `expanded_details` | none | MOVE |
| `form` | `_apply_parakeet_v2_install_result`, `_on_preflight_retry` (shell) | MOVE |
| `last_active_count` | none | MOVE |
| `last_done_count` | `on_mount` (shell) | MOVE |
| `last_submission` | none | MOVE |
| `path_debounce_timer` | none | MOVE |
| `preflight_generation` | none | MOVE |
| `preflight_worker` | none | MOVE |
| `recent_ledger` | none | MOVE |
| `retry_confirm_armed` | none | MOVE |
| `retry_confirm_armed_at` | none | MOVE |
| `start_confirm_armed_at` | `_apply_library_external_preparation` (shell) | MOVE |
| `start_consent` | `_library_emergency_return_eligibility`, `_apply_library_external_preparation` (shell) | MOVE |

All 20 original `__init__` lines are static literals or no-argument
factory calls (`threading.Lock()`, `LibraryIngestFormState()`, `[]`,
`set()`) with **zero entanglement** with any other subsystem's shared init
code (no reader-preferences-trio-shaped complication anywhere in this
cluster) — the simplest possible state-PR shape in this recipe to date.
`LibraryIngestState()` is constructed with **no constructor arguments**, at
the position of the first removed field (`auto_collapsed_rail`'s original
line, `library_screen.py:3111` pre-edit).

## 2. Characterization spot-check

Test roots checked: `Tests/UI`, `Tests/Library`, `Tests/Live`, `Tests/App`,
`Tests/integration` (no dedicated `Tests/Ingest/` tree exists — ingest
coverage is distributed across `Tests/UI/test_library_ingest_*.py` (8
files), `Tests/Library/test_library_ingest_*.py`, `Tests/App/
test_submit_library_ingest_job.py`, `Tests/integration/
test_library_ingest_flow.py`; `Tests/Live` has **zero** ingest references
of any kind).

29 unique `@on`-bound Ingest handlers (30 raw decorator matches;
`handle_library_ingest_browse` carries two selectors,
`#ingest-preflight-choose` + `#library-ingest-browse`, over one shared
body). Per-selector `grep -rn` across all five roots, followed by a manual
read for an actual `.press()`/`.click()`/message-bubble-through-a-real-
Collapsible interaction (not a same-line-only grep, and not a raw
`screen.handle_x(event)` call with a hand-built fake event, which exercises
handler logic but never the `@on` CSS-selector dispatch itself):

**24 of 29 genuinely covered**, including one initially-missed case caught
only by tracing the FULL interaction chain rather than grepping the
handler/message name literally:
`sync_library_ingest_type_group_expanded` (`@on(LibraryIngestCanvas.
OptionPanelToggled)`) looked unpressed under a literal name/message-class
grep, but is genuinely covered by `test_library_shell_ingest_type_group_
panel_expand_survives_recompose` (`Tests/UI/test_library_shell.py:24931`),
which toggles a real `#type-group-generic` `Collapsible` and asserts
`screen._library_ingest_form.expanded_type_groups` updates and survives a
recompose — the exact same "a same-line-only grep undercounts coverage"
trap the collections series' own report already named, reproduced here on
a message-based handler instead of a button id.

**5 genuine gaps**, all pinned into
`Tests/UI/test_library_ingest_characterization.py` (new file, 4 test
functions covering 5 handlers — one function double-checks 2 related
assertions), confirmed **PASSING pre-change**:

1. `_on_library_ingest_top_button` (`#library-ingest-top-button`) — the one
   existing test naming this id (`test_ingest_button_opens_canvas`,
   `Tests/integration/test_library_ingest_flow.py`) explicitly bypasses a
   real press ("button.press() is unreliable for async handlers in the
   test harness"). Disproven: `.press()` works identically to every other
   async `@on` handler already `.press()`-tested in this codebase (e.g.
   `_return_library_rail_to_starter` via `#library-rail-back-to-starter`,
   `test_library_shell.py:2940`).
2. `sync_library_ingest_tooling_detail_expanded` (`@on(LibraryIngestCanvas.
   ToolingDetailToggled)`) — its sibling above is covered; this one has no
   screen-level equivalent (only a standalone-canvas-host test in
   `test_library_ingest_canvas.py` proving the CANVAS posts the message,
   never that the SCREEN handles it).
3. `handle_library_ingest_view_on_server` (`.library-ingest-view-server`) —
   zero test references of any kind.
4. `handle_library_ingest_choose_gguf` (`.library-ingest-choose-gguf`) —
   only ever `query_one`-d for a button-presence/removal assertion, never
   pressed.

New pins use direct registry state injection (`LibraryIngestJobRegistry.
submit(origin="server")` + `mark_remote_done(...)` for the view-on-server
case; `.submit()` + `mark_failed(error_detail={"category": "stt_failure",
"actions": [...]})` for the choose-gguf case) rather than driving a real
async worker pipeline to a specific failure classification — the same
registry-direct-mutation pattern already used throughout
`test_library_shell.py`'s own existing ingest tests.

**5 more, documented but NOT pinned** (a "spot-check", not an exhaustive
re-drive of every already-somewhat-tested handler into a full DOM press):
`handle_library_ingest_cancel`, `_force_stop`, `_retry_faster_whisper`,
`_option_reset`, `_directory_browse` are each exercised only via a raw
`screen.handle_x(event)` (or `LibraryScreen.handle_x(screen, event)`) call
with a hand-built `MagicMock` event on a REAL, fully-`__init__`-ed screen
instance — genuine logic coverage, but the `@on` CSS-selector dispatch
itself is unverified. Recorded as known, bounded coverage debt in the new
characterization file's own module docstring rather than silently counted
as "24 of 29 covered" or exhaustively closed at this task's own expense —
each needs a specific backing `LibraryIngestJob` state composed through the
real registry, exactly the added machinery the 2 pinned per-job-row tests
above demonstrate is tractable, for a future task.

### Form-persistence contract (task-2043)

**Already pinned, meets the `.press()` standard — no new pin needed.**
`test_rail_switch_preserves_staged_ingest_form`
(`Tests/UI/test_library_shell.py:29585`) types into two real `Input`
widgets, presses a real rail-row button (`#library-row-browse-media`) to
switch away from Ingest — the exact call chain that runs `_pause_library_
ingest_transient_ui()` — reopens the Ingest canvas, and asserts BOTH
`screen._library_ingest_form` state AND the re-rendered widget's `.value`
survived. Confirmed passing pre-change.

## 3. `LibraryIngestState`

`tldw_chatbook/UI/Library_Modules/library_ingest_state.py` (new file): a
`@dataclass` with all 20 fields, verbatim defaults, single
`_library_ingest_` prefix (no plural variant, no wiring exclusion — the
export series' own simplest-case shape). Imports `LibraryIngestFormState`/
`LibraryIngestLastSubmission` from the pre-existing `tldw_chatbook/
Library/library_ingest_state.py` (a DIFFERENT module, same basename, the
established `UI/Library_Modules/` ↔ `Library/` pairing precedent already
used by 6 other subsystems) and `_LibraryIngestStartConsent` from
`UI/Library_Modules/screen_support_types.py` (PR 0a's own foundation
module).

## 4. Programmatic screen shims

Sentinel-wrapped block (`--- BEGIN/END generated ingest-state shims ---`)
appended at module end, mirroring the export/collections generator loop
exactly: `for _lis_field in dataclasses.fields(LibraryIngestState): setattr(
LibraryScreen, "_library_ingest_" + _lis_field.name, property(getter,
setter))`, `_n=` binding on both lambdas.

## 5. A new bypass shape, found by the battery (not by static census)

**Corrected in fix round 1 (§11) — this section's ORIGINAL count was wrong
by both digits ("24 sites across 4 files"); see §11 for the verified true
numbers (27 sites across 6 files) and the count-accuracy incident itself.
Left below in its original, uncorrected form except for this notice, per
the review's own instruction to fix the count rather than quietly patch
history.**

Not in recipe §3's prior catalogue: **`object.__new__(LibraryScreen)` /
`LibraryScreen.__new__(LibraryScreen)` `__init__`-bypass fixtures.**
24 call sites across 4 test files (`Tests/UI/test_library_ingest_canvas.py`
[16], `Tests/App/test_submit_library_ingest_job.py` [5], `Tests/UI/
test_library_ingest_inline_consent.py` [1], `Tests/UI/test_library_ingest_
retry_last.py` [1], `Tests/integration/test_library_ingest_flow.py` [2, the
`.LibraryScreen.__new__(LibraryScreen)` spelling]) skip `__init__` entirely
and hand-set flat `_library_ingest_<field>` names as PLAIN instance
attributes — safe before this move (no property existed, so the assignment
just created an instance attribute), but `AttributeError: 'LibraryScreen'
object has no attribute '_ingest_state'` immediately once the property
setter tries to route through a never-constructed state object.

Unlike every previously-catalogued bypass shape (unbound fake-self,
instance-attribute monkeypatch, module-globals coupling, bare-self
identity, unbound-attribute escape — all of which stay latent until a
CONTROLLER-PR's method move, deferred to that series' own cleanup task by
design), this one fails immediately, at the STATE PR itself, for every
affected test — a no-red-ships violation if shipped unfixed. **63 tests
were RED** after the screen edit landed, before this fix (this count is
ALSO incomplete — see §11: it is the count `-k "ingest and library"`
could see, and 2 more RED tests existed in a file that filter could not
collect at all).

Fixed mechanically and minimally, in the same GREEN commit: one line
inserted after each `__new__` call (`screen._ingest_state =
LibraryIngestState()`), zero assertions, call sites, or other lines
touched anywhere. Re-ran the same 63 (and the surrounding `-k "ingest and
library"` sweep) after the fix: **0 failures caused by this shape
remained** *within that sweep's own reach* (§11 corrects this claim: 2
more remained, outside that sweep's reach, until fix round 1). Recorded in
`library-decomposition-recipe.md`'s own §3-shaped catalogue is left as a
forward note here rather than a formal new numbered entry (out of this
task's own file-edit scope — the recipe edit in this task's third commit
is scoped to §7's sweep-evidence list per its own explicit "add to it"
instruction, not a new bypass-shape entry). **Also corrected in fix round
1**: this shape IS now a formal, numbered §3 catalogue entry (the
"seventh bypass shape") — the review's own Important #2 overrode the
scope call made here.

## 6. TDD evidence

RED (`a11220648`): `Tests/Architecture/test_library_ingest_wiring.py`
watched failing against the untouched screen —
`AssertionError: no screen shim property found for: [...]` (all 20 expected
shim properties missing), confirmed with `library_ingest_state.py` already
present. Screen untouched in this commit (verified via `git status`/`git
diff --stat` before committing).

GREEN (`12ba4fb13`): same test passes after the screen edit.

```
$ .venv/bin/python -m pytest Tests/Architecture/test_library_ingest_wiring.py -p no:randomly -q
# before the GREEN commit:
FAILED Tests/Architecture/test_library_ingest_wiring.py::test_state_object_fields_match_the_shim_surface
1 failed
# after:
1 passed
```

## 7. Size ratchet

Fresh `_measure()`: **41520 lines, 1302 methods** (pure field move — zero
`FunctionDef`s touched, methods unchanged). `_BUDGETS` lowered in the GREEN
commit per recipe §6: `41574/1302 → 41520/1302`.

Not rebased onto `origin/dev`'s 4 newer commits before this measurement:
confirmed via `git diff --stat HEAD..origin/dev` that they are
Console-scoped only (`console_cost_tracker.py`, `console_spend_
projection.py`, `console_status_chips.py`, `console_context_controls.py`,
`console_model_popover.py`, associated tests/CSS) — touching neither
`library_screen.py` nor `test_screen_size_ratchet.py`. This is a mid-series
task (1 of 3), not the wave's own closing PR; the dev-merge reconciliation
is deferred to that task, matching wave-4's own precedent (skills task 1
measured against its branch tip; the `origin/dev` merge landed at wave-4's
close, task 4, per recipe §19).

Both `test_screen_does_not_grow_past_its_budget[library_screen.py]` and
`test_budget_is_not_left_slack_after_a_wave[library_screen.py]` pass — the
pin is exact, no slack.

`Tests/Architecture/test_library_modules_size_ratchet.py` (controller-file
governance, §17): 29 passed, unaffected — no controller file exists for
Ingest yet (Task 2's own scope).

## 8. Verification battery

All commands from `.worktrees/library-decomp-foundation`, `.venv/bin/
python`, `-p no:randomly` where applicable.

**Wiring + characterization + ratchet + support-layer, combined**: all six
wiring suites (conversations/export/collections/search+RAG/skills/ingest)
+ 3 characterization files (collections/export/ingest) + support-layer
surface + both size-ratchet rows: **84 passed, 2 failed** — both the
documented pre-existing `chat_screen.py` ratchet rows (recipe §7's own
standing list), unrelated to this diff.

**Full `Tests/Architecture/`**: **16 failed, 544 passed, 1 skipped** —
exact category-for-category match to wave-4 close's own documented
pre-existing bucket (Console realtime ×1, review-selection ×1, wave6
closeout ×1, wave6 inventory ×2, default-timeout-session-guard ×1,
persistent-diagnostic-inventory ×2, chat_screen ratchet ×2, timer-path-
static-update-inventory ×3, worker-exclusive-group-inventory ×2 = 16), +1
pass versus wave-4 close's own 543 (this task's own new wiring test). Zero
Library/Ingest-scoped failures.

**`-k "ingest and library"`** across `Tests/UI` + `Tests/Library` +
`Tests/App` + `Tests/integration` (1306 collected): first pass found **63
failed** (the `object.__new__` bypass shape above, §5); after the fix,
**7 failed, 1298 passed, 1 skipped** — all 7 confirmed identical on a
`git stash -u` pristine baseline of the pre-GREEN-edit tree (i.e. the RED
commit's own tree, where `_library_ingest_form` etc. are still plain
unshimmed attributes, ruling out the property mechanism as the cause):
`test_library_ingest_canvas.py::test_progress_detail_paints_below_row_
without_obscuring_actions_or_neighbor[size0/size1]` (Color assertion),
`test_library_ingest_retry_last.py::test_registry_ticks_only_reflow_
footer_when_retry_availability_changes` (unrelated Prompt/Study
backend-unavailable `ValueError`s bleeding into an unrelated assertion),
`test_library_ingest_structural.py::{test_fold_hint_is_pinned_not_scrolled,
test_outcome_lines_paint_heavier_than_the_tooling_summary,
test_every_canvas_focusable_changes_at_the_glyph_level_on_focus}`
(geometry/weight/focus-color assertions), `test_ingest_preflight_egress.py
::test_the_probe_reports_a_redirect_as_an_answered_status_not_an_error`
(capability-warning-count, environment-dependent). None touches the state
move.

**Full sequential xdist paired-baseline sweep** (`Tests/UI -k "library" -p
no:randomly -q -n 8 --dist worksteal`), branch then baseline, per recipe
§7:

| | Failed | Passed | Wall time |
|---|---|---|---|
| Branch (`12ba4fb13`) | 356 | 3989 | 1616.20s (26:56) |
| Baseline (`9e62dd8f7`, isolated `git worktree`) | 370 | 3971 | 2355.11s (39:15) |

Both inside the documented ~330–371 historical backdrop. 348 shared, 22
baseline-unique (not investigated further per §7's own precedent), **8
branch-unique** — all resolved, zero unexplained:

- 4 passed cleanly on a combined single-process re-run (ordinary xdist
  noise): `test_audio_cpp_model_library_handoff.py::
  test_audio_cpp_presentation_reveals_slow_load_once_and_keeps_error_retry`,
  `test_library_media_reader_flow.py::
  test_edit_metadata_from_read_routes_to_info_form_actions`,
  `test_library_media_reader_traversal_t22207.py::
  test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke`
  (already in the recipe's §7 list, wave-2 task 6), `test_screen_
  navigation.py::test_search_route_round_trips_to_the_library_rag_row`.
- 2 reproduced identically in TRUE isolation on BOTH the branch and the
  isolated pristine-baseline worktree: `test_library_media_reader_no_
  change_sync_t22208.py::test_image_item_traversal_wall_time_probe` (a
  wall-clock timing probe by design) and its sibling `test_no_change_
  traversal_builds_no_preview_and_copies_no_content` (already in the
  recipe's list as a name that passed cleanly on rerun at wave-2 close;
  this run's deterministic reproduction on both trees is a stronger,
  consistent escalation of the same characterization, not a
  contradiction).
- 1 reproduced identically on both trees and is the SAME name already
  documented (wave-3 task 5) as bidirectional run-to-run flakiness:
  `test_screen_navigation.py::test_library_screen_round_trip_returns_to_
  landing_with_rag_draft`.
- 1, `test_library_shell.py::test_library_media_initial_error_is_unknown_
  and_retry_is_unique`, failed once in the combined re-run (immediately
  after the 3-minute wall-time-probe test) but passed 6 of 7 further
  isolated re-runs on the branch and 3 of 3 on the isolated baseline —
  ordinary load-adjacent flakiness, no causal link to this task's diff
  (Media error-retry logic untouched).

None of the 8 touches Ingest code or this task's own diff. **Zero real
regressions.**

Method note: the FIRST attempt at this comparison used an in-place
`git checkout 9e62dd8f7 -- tldw_chatbook Tests` overlay of this same
worktree, and was silently invalidated when a session-usage-limit
interruption's recovery restored the shared worktree to `HEAD` while that
background sweep was still reading files mid-run. Redone with an isolated
`git worktree add /tmp/w5base 9e62dd8f7` + its own `uv venv` +
`pip install -e ".[dev]"` — immune to that class of corruption. Recorded as
a forward-looking lesson in the recipe (commit `3b83eab93`).

**preflight**: `PYTHON=.venv/bin/python ./scripts/preflight.sh` — all six
derived-artifact checks pass (CSS bundle, profile-owned-path census [48
occurrences/18 files/46 exceptions], production diagnostic inventory [573
owners], backlog task-id sweep [3240 files, zero duplicates], chachanotes
table allowlist [105 tables], index plan pins [270/270, 57 pinned]).

## 9. Files changed

- `tldw_chatbook/UI/Library_Modules/library_ingest_state.py` (new) —
  `LibraryIngestState` dataclass, 20 fields.
- `tldw_chatbook/UI/Screens/library_screen.py` — import added; 20
  `__init__` lines collapsed into 1 constructor call; generated shim block
  appended at module end.
- `Tests/Architecture/test_library_ingest_wiring.py` (new) — wiring test
  (shim-surface check, export-precedent shape).
- `Tests/Architecture/test_screen_size_ratchet.py` — `_BUDGETS` row
  lowered, comment appended.
- `Tests/UI/test_library_ingest_characterization.py` (new) — 4 test
  functions pinning the 5 genuine coverage gaps.
- `Tests/UI/test_library_ingest_canvas.py`,
  `Tests/UI/test_library_ingest_inline_consent.py`,
  `Tests/UI/test_library_ingest_retry_last.py`,
  `Tests/App/test_submit_library_ingest_job.py`,
  `Tests/integration/test_library_ingest_flow.py` — 24 `object.__new__`
  bypass fixtures given a `screen._ingest_state = LibraryIngestState()`
  seed line each; matching imports added. Zero assertions changed.
- `backlog/docs/library-decomposition-recipe.md` — §7 sweep-evidence list
  extended; isolated-worktree-baseline lesson added.
- `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/
  progress.md` — ledger closed out.

## 10. Self-review

- **Ingest-options trio**: verified current location (module-level,
  `library_screen.py:605-692`), verified neither function touches any
  `self._library_ingest_*` field, untouched by this diff. ✅.
- **All 20 fields accounted for**, verdicts derived mechanically (recipe §2
  script), not hand-listed; oddball scan run in both directions (fields
  named outside the prefix, and ingest-named methods reaching outside
  fields). ✅.
- **Form-persistence contract**: found already covered to the `.press()`
  standard; did not write a redundant pin. ✅.
- **Wiring test genuinely RED then GREEN**, screen provably untouched in
  the RED commit (`git status`/`git diff --stat` checked before
  committing). ✅.
- **Ratchet lowered in the same (GREEN) commit**, not deferred; both
  ceiling and no-slack rows pass. ✅.
- **A real, battery-found regression (63 tests) was caught and fixed
  before landing** — not shipped, not deferred, not hand-waved. The fix is
  minimal (one line per call site, zero assertions touched) and verified
  by re-running the exact same failing set. This is the single largest risk
  this task carried; I'm confident it's fully closed (re-verified via the
  full `-k "ingest and library"` sweep post-fix: 0 attributable failures).
- **Known, intentional gap**: 5 handlers (cancel/force-stop/retry-faster-
  whisper/option-reset/directory-browse) remain covered only by raw
  handler calls, not real `.press()`. **Condition confirmed in fix round 1
  (§11): the reviewer endorsed deferring these 5 past this task ONLY on
  the condition that Task 2 (the controller-move PR) pins all 5 via a real
  `.press()` BEFORE any of their bodies move** — the same registry-
  injection technique the 2 new per-job-row pins in this task's own
  characterization file already demonstrate is tractable. This is a hard
  precondition for Task 2's own RED wiring commit, not optional cleanup;
  Task 2 must not move `handle_library_ingest_cancel`/`_force_stop`/
  `_retry_faster_whisper`/`_option_reset`/`_directory_browse` until each
  has a genuine DOM-press pin passing pre-move.
- **Sweep baseline method**: the first attempt (in-place checkout) was
  invalidated by a session interruption; redone correctly with an isolated
  worktree per the coordinator's explicit direction, and the lesson is
  recorded for future tasks. The two pre-interruption stash-based checks
  used to isolate the `object.__new__` bypass fix (short-lived,
  foreground, `git stash pop`ped immediately after each) are unaffected by
  that risk class — verified clean (`git status`) after each pop.

## 11. Fix round 1 (post-review)

Coordinator review found 1 CRITICAL + 2 Important issues against the
original report/commits above. All three addressed in commit
`fix(library): seed the missed object.__new__ fixtures; recipe gains the
fixture-bypass shape (fix round 1)`.

### CRITICAL — 1 missed file, 2 RED tests at HEAD (no-red-ships violation)

`Tests/UI/test_parakeet_v2_install_ui.py` has 11 total `object.__new__(
LibraryScreen)` constructions, but only 2 (lines 482 and 517 in the
pre-fix file) touch `_library_ingest_form` — the other 9 construct a
screen only to exercise `_parakeet_v2_install_worker`
(`handle_parakeet_v2_install_requested`/preflight-result-modal/GGUF-picker
tests), confirmed unrelated by reading each. Those 2 sites never got a
`screen._ingest_state = LibraryIngestState()` seed line in the original
GREEN commit and were RED at HEAD:
`test_install_result_notify_text_uses_mapped_message_not_raw_exception`
and `test_successful_install_prefers_managed_and_clears_external_override`,
both failing with the exact same `AttributeError: 'LibraryScreen' object
has no attribute '_ingest_state'` signature as the 63 originally caught.

**Root cause of the miss**: every sweep this task ran to hunt for this
bypass shape's fallout (`-k "ingest and library"`, `-k "ingest"`) is a
NAME-based pytest filter, and `test_parakeet_v2_install_ui.py`'s own
filename and test names contain neither "ingest" nor "library" — 0 tests
from this file were ever collected by any of those filters, so the 2 RED
tests were invisible to every verification this task ran, despite genuinely
touching ingest state. Only a repo-wide CONTENT grep (`object.__new__(
LibraryScreen)` OR `LibraryScreen.__new__` co-occurring with
`_library_ingest_`, across ALL of `Tests/`, not a keyword subset) finds it
— exactly the check the reviewer ran and this fix round re-ran to confirm
completeness:

```
$ grep -rlE "\.__new__\(\s*LibraryScreen\s*\)|LibraryScreen\.__new__" Tests/ \
    | xargs grep -l "_library_ingest_"
Tests/App/test_submit_library_ingest_job.py
Tests/integration/test_library_ingest_flow.py
Tests/UI/test_library_ingest_canvas.py
Tests/UI/test_library_ingest_inline_consent.py
Tests/UI/test_library_ingest_retry_last.py
Tests/UI/test_library_screen.py          # false positive, see below
Tests/UI/test_parakeet_v2_install_ui.py  # the miss
```

`Tests/UI/test_library_screen.py` reconfirmed a false positive by reading
its fixture body, not just the grep hit: its own docstring mentions
`object.__new__(LibraryScreen)` in PROSE (describing a PRIOR, already-fixed
bypass, task-3022), but the actual fixture (`_minimal_ingest_screen`) calls
`screen = LibraryScreen(MagicMock())` — a real, `__init__`-routed
constructor, so `_ingest_state` is genuinely present and every subsequent
flat-name write already routes through a working shim. **7 files matched
the content grep; exactly 1 (`test_parakeet_v2_install_ui.py`) needed a
fix; the reviewer's own scan result is confirmed correct.**

Fix: added `LibraryIngestState` to the file's existing `from
tldw_chatbook.UI.Screens.library_screen import (...)` block, and inserted
`screen._ingest_state = LibraryIngestState()` immediately after each of the
2 affected `object.__new__(LibraryScreen)` calls. Zero assertions or other
lines touched.

```
$ .venv/bin/python -m pytest Tests/UI/test_parakeet_v2_install_ui.py -p no:randomly -q
25 passed
```

### IMPORTANT — recipe §3 permanent catalogue entry

Added a new, formally numbered "seventh bypass shape" entry to
`backlog/docs/library-decomposition-recipe.md` §3 (previously left as an
informal forward note in this report's own §5, out-of-scope per the
original commit's own file-edit boundary — the review overrode that scope
call, correctly: this shape is now proven to recur across an entire
subsystem's test suite regardless of controller/state-PR boundary, and
belongs in the permanent, cross-subsystem catalogue, not one task's
report). The entry states: the shape itself (`object.__new__(<Screen>)`/
`<Screen>.__new__(<Screen>)` bypass + flat-attribute hand-set, breaking the
instant a state-PR shim installs, unlike every prior shape which stays
latent until a controller-PR's method move); why it cannot be deferred to
cleanup (no-red-ships fires at the state PR itself); the filter-blindness
lesson stated as its own standing rule (a content-grep across all of
`Tests/`, never a `-k` name filter, is the only sound completeness check
for this shape — the exact rule whose absence let the Critical ship); and
the corrected 27-sites/6-files accounting. Folded into the same entry: the
interruption/isolated-worktree lesson already recorded in this task's own
prior commit (`3b83eab93`, §7's sweep-evidence list) is cross-referenced
and restated as a general, standing rule ("an isolated worktree, not a
same-tree checkout overlay, is the default method for any baseline
comparison expected to run unattended for more than a couple of minutes"),
rather than left as a footnote local to one task's own sweep entry.

### IMPORTANT — count accuracy

The original §5 said "24 call sites across 4 test files" while its own
bracketed list named 5 files summing to 25 (5 + 16 + 1 + 1 + 2) — an
arithmetic error in the prose, not the list. Re-verified the TRUE numbers
by direct count (`grep -c` per file, cross-checked against the actual
`_ingest_state = ...`/`_ingest_state = library_screen_module.
LibraryIngestState()` seed lines present):

| File | Bypass constructions | Touch `_library_ingest_*`? | Seeded |
|---|---|---|---|
| `Tests/App/test_submit_library_ingest_job.py` | 5 | yes (5) | 5 |
| `Tests/integration/test_library_ingest_flow.py` | 2 | yes (2) | 2 |
| `Tests/UI/test_library_ingest_canvas.py` | 16 | yes (16) | 16 |
| `Tests/UI/test_library_ingest_inline_consent.py` | 1 | yes (1) | 1 |
| `Tests/UI/test_library_ingest_retry_last.py` | 1 | yes (1) | 1 |
| `Tests/UI/test_parakeet_v2_install_ui.py` | 11 | only 2 of 11 | 2 |
| `Tests/UI/test_library_screen.py` | 0 (prose-only mention) | n/a | n/a (real constructor) |

**True total: 27 seeded sites across 6 files** (25 in the original GREEN
commit + 2 in this fix round), not "24 across 4." Corrected in §5 above
(left the original wrong sentence in place with a notice, per not quietly
rewriting a discovered error) and in this section's own accounting. No
other report section restated the wrong count as a hard number requiring
its own correction (§8's "63 failed"/"63 tests" language is now annotated
in §5 as ALSO incomplete — that count was always accurate for what `-k
"ingest and library"` could see, 63, but incomplete as a total, since 2
more existed outside that filter's reach; both counts, "63" and "then 2
more," are separately true and are kept distinct rather than merged into
a single revised number, since they were found via different methods at
different times and merging them would obscure exactly the filter-
blindness lesson this fix round exists to record).

### Verification (fix round 1)

```
$ .venv/bin/python -m pytest Tests/UI/test_parakeet_v2_install_ui.py -p no:randomly -q
25 passed

$ .venv/bin/python -m pytest Tests/Architecture/test_library_ingest_wiring.py -p no:randomly -q
1 passed

$ .venv/bin/python -m pytest Tests/Architecture/test_screen_size_ratchet.py Tests/Architecture/test_library_modules_size_ratchet.py -p no:randomly -q
# both library_screen.py ratchet rows pass; pre-existing chat_screen.py
# failures only; controller-file governance ratchet unaffected (29 passed)

$ PYTHON=.venv/bin/python ./scripts/preflight.sh
preflight: all derived-artifact checks passed.
```

Repo-wide content-grep re-run one final time after the fix (shown above,
§ "Root cause of the miss") to confirm zero remaining `object.__new__`/
`_library_ingest_` co-occurrences without a seed line anywhere in `Tests/`.

### Files changed (fix round 1)

- `Tests/UI/test_parakeet_v2_install_ui.py` — import + 2 seed lines.
- `backlog/docs/library-decomposition-recipe.md` — new §3 "seventh bypass
  shape" entry (fixture bypass + filter-blindness lesson + isolated-
  worktree cross-reference).
- `.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/
  task-1-report.md` — this section, plus in-place correction notices in
  §5.

### Confirmation for Task 2 dispatch

The 5 deferred handlers (`handle_library_ingest_cancel`,
`_force_stop`, `_retry_faster_whisper`, `_option_reset`,
`_directory_browse`) must each get a real `.press()`-driven characterization
pin — using the same `LibraryIngestJobRegistry.submit()`/`mark_failed()`/
direct-state-injection technique this task's own 2 new per-job-row pins
already establish as tractable — BEFORE Task 2 moves any of their bodies
into the controller. This is a precondition on Task 2's own RED wiring
commit, not a nice-to-have; Task 2 should not proceed past its own census
step until these 5 pins exist and pass pre-move.
