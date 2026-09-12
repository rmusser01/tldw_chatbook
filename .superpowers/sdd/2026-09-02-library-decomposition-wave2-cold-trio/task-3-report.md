# Task 3 report — Export series 2/3: LibraryExportController (export controller move)

Branch `refactor/library-decomp-wave2-cold-trio`, worktree
`.worktrees/library-decomp-foundation`. Base commit `264314c5f` (Task 2, export
state fix round 1). Two commits (implementation, then blame-ignore follow-up)
land on top of it.

## 1. Cluster derivation — final list, every decision recorded

Mechanical `ast` scan of `LibraryScreen` for method names containing
`"export"` (case-insensitive, no prefix shortcut): **51 methods** — matches
Task 2's own census exactly (re-derived at execution time, not trusted from
the brief).

Naively moving all 51 would have been wrong. Reading every one of the 51
bodies (not just the ones with `_library_export_` state) shows the name
match is frequently coincidental: many are a DIFFERENT subsystem's own
"Export…" button handler whose guard reads THAT subsystem's state, and which
merely calls the shared `_open_library_export_canvas` opener. The final
split, in the order the investigation found it:

### Round 1 — ownership (18 excluded)

Verified by reading each body's actual state reads/writes, not by name:

| Name | True owner | Evidence |
|---|---|---|
| `_export_library_note`, `_write_library_note_export_file`, `handle_library_note_export_markdown`, `handle_library_note_export_text` | Notes | body reads `_library_notes_view`/`_selected_note_id`/`_begin_library_notes_operation`, zero `_library_export_*` touches |
| `handle_library_notes_export` (`@on`), `handle_library_notes_export_selected` (`@on`) | Notes | guard reads `_library_notes_mutation_fenced()`/`_library_notes_row_selection`; body is a thin `_open_library_export_canvas(...)` call |
| `handle_library_prompt_export` (`@on`), `_export_library_prompt`, `_write_library_prompt_export_file` | Prompts | body reads `_library_prompts_view`/`_selected_prompt_id`/prompt editor fields |
| `handle_library_prompts_export` (`@on`), `handle_library_prompts_export_selected` (`@on`) | Prompts | guard reads `_library_prompts_mutation_in_flight`/`_library_prompt_browse_controller.freshness`/`_library_prompt_selection` |
| `handle_library_media_export` (`@on`), `handle_library_media_export_selected` (`@on`) | Media | guard/scope reads `_library_media_type_filter`/`_library_media_row_selection`/`_library_media_bulk_delete_in_flight` |
| `handle_library_conversations_export` | Conversations | **already** a one-line delegator to `LibraryConversationsController` (task 8) — export makes no claim |
| `handle_library_conversations_export_selected` | Conversations | one of task 8's own 5 unbound-fake-self exclusions (`test_library_multiselect_conversations.py`); stays real, full-bodied on `LibraryScreen` — this task does not re-litigate that decision |
| `choose_library_collection_legacy_recovery_export` (`@on`), `_export_library_collection_legacy_recovery` | Collections | entirely different mechanism (legacy JSON recovery), reads `_library_collections_legacy_recovery_open`/`collections_legacy_recovery_service`, never touches the chatbook-zip Export canvas |
| `open_import_export_from_library_rag` (`@on`) | Search/RAG (shell) | body only does `await self._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)` — a retired-feature stub, touches zero `_library_export_*` state |

### Round 2 — a framework-decorator hazard (2 excluded, genuinely Export-owned)

`_run_library_export_counts_worker` and `_run_library_export_worker`, both
`@work(thread=True, exclusive=True, group=...)`. Read Textual's own
`work()` decorator source
(`.venv/lib/python3.14/site-packages/textual/_work_decorator.py` via
`inspect.getsource`): the wrapper does
`self = args[0]; assert isinstance(self, DOMNode)` at CALL time. A plain
`LibraryExportController` instance is not a `DOMNode`; calling either
through `self.<name>(...)` on the controller would raise `AssertionError`
synchronously on every call — not a test artifact, a permanent runtime
contract. Named this bypass shape **"framework-decorator self-type
assertion"**, new to the recipe's §3/§11 catalogue. Both stay on
`LibraryScreen`, UNMOVED, decorator and body byte-for-byte untouched. Their
sole callers (`_start_library_export_counts_worker`,
`_start_library_export_worker`) turned out to ALSO be excluded in round 3
below, so this controller ends up with zero live references to either name.

### Round 3 — unbound fake-self, found only by running the battery (9 excluded, genuinely Export-owned)

Of the 31 remaining after rounds 1–2, running the verification battery
(§6) — not static analysis — surfaced 9 more methods reached by
`LibraryScreen.<name>(fake, ...)` calls where `fake` lacks
`_export_controller`. Confirmed each via `git stash -u` to PASS on a
pristine baseline before excluding (i.e. genuine regressions this move
introduced, not pre-existing reds):

| Excluded method | Bypassing test(s) | Shape |
|---|---|---|
| `_apply_library_export_cancelled` | `test_library_export_cancel.py` ×2 | unbound `SimpleNamespace` |
| `handle_library_export_cancel` | `test_library_export_cancel.py` ×1 | unbound `SimpleNamespace` |
| `_apply_library_export_progress` | `test_library_export_progress_apply.py` ×3 | unbound `SimpleNamespace` |
| `_apply_library_export_counts` | `test_library_export_receipt.py` ×1 | unbound `SimpleNamespace` |
| `_build_library_export_state` | `test_library_export_receipt.py` ×2 | **indirect**: `fake._build_library_export_state = (lambda: LibraryScreen._build_library_export_state(fake))`, invoked only when `_apply_library_export_counts`'s (also excluded) body calls `self._build_library_export_state()` with `self=fake` |
| `_update_library_export_canvas_after_run` | `test_library_export_receipt.py` ×1 | unbound `SimpleNamespace` |
| `_start_library_export_counts_worker` | `Tests/Library/test_library_export_execution.py::test_prompt_memory_database_forces_inline_count_resolution` | unbound `SimpleNamespace` |
| `_start_library_export_worker` | `Tests/Library/test_library_export_execution.py` ×2 | unbound `SimpleNamespace` |
| `_apply_library_export_success` | `test_library_shell.py::test_library_export_registry_failure_warns_it_wont_appear_in_artifacts` | **new 4th bypass shape**: `screen = Mock()` — `unittest.mock.Mock` auto-creates `screen._export_controller` as ANOTHER Mock rather than raising `AttributeError`, so a delegator "succeeds" silently while never running real logic. Named **"silent Mock auto-attribution"**. |

Four of these nine (`_start_library_export_counts_worker`,
`_start_library_export_worker`, and the two `Tests/Library/` tests they
belong to) live OUTSIDE `Tests/UI/` — the recipe's canonical
`-k "library"` sweep root — and were found only once the search was
deliberately widened. Recorded as a forward note in
`backlog/docs/library-decomposition-recipe.md` §7 for future subsystem
tasks.

### Net result

**51 → 18 (other-subsystem) + 2 (framework-decorator) + 9 (unbound-fake-self)
excluded = 22 move onto `LibraryExportController`.**

The 22: `_default_library_export_form`, `_reset_library_export_transient_state`,
`_open_library_export_canvas`, `_library_export_is_server_mode`,
`_resolve_library_export_chachanotes_db`, `_compute_library_export_counts`,
`handle_library_export_submit`, `_build_library_export_payload`,
`_run_library_export_via_service`, `_marshal_library_export_success`,
`_marshal_library_export_failure`, `_marshal_library_export_cancelled`,
`_build_library_export_success_message`, `_apply_library_export_failure`,
`_refresh_library_export_status_line`, `action_library_export_back`,
`handle_library_export_name_changed`, `handle_library_export_description_changed`,
`handle_library_export_quality`, `handle_library_export_quality_choice`,
`handle_library_export_choose_destination`, `_apply_library_export_destination`.

All 29 excluded methods (18 + 2 + 9) are, as of the final commit, verified
byte-for-byte identical (raw text, decorators/signature/docstring/body,
comments included) to base commit `264314c5f` — see §5. A first pass had
mistakenly added "excluded because…" docstring paragraphs to the 9 round-3
methods; caught in self-review (not by a reviewer) and reverted before
committing, since the transform whitelist (recipe §4) has no entry for
"document why an unmoved method stays," and the conversations exemplar's own
precedent language is "body untouched" — the reasoning lives entirely in
`library_export_controller.py`'s module docstring and the wiring test's
comments instead.

## 2. Dynamic-dispatch census (recipe §3's 4th-bypass-shape forward note)

Task 2's report flagged `_close_open_library_choice_strip`'s
`setattr(self, visibility_attr, False)` (with `visibility_attr` possibly the
literal `"_library_export_quality_choices_visible"`) as a forward risk.
Investigated before moving anything:

- `_close_open_library_choice_strip` itself is shell/plumbing (branches
  across media/prompts/skills/export), stays on `LibraryScreen`, unmoved —
  not a candidate for this move at all. The field it sets already has its
  screen-facing property shim from Task 2, untouched by this PR. **Not a
  hazard.**
- Grepped `getattr(self,`/`getattr(screen,`/`setattr(self,`/`setattr(screen,`
  with an f-string or dict-literal argument across `library_screen.py` and
  `canvas_sync.py`: two more sites found
  (`_library_rail_preferences()`'s `f"{section_id}_open"` lookup;
  `_replace_library_reader_preference`'s 7-destination dict). Neither
  destination set includes any Export name (Export has no rail-preferences/
  reader-pane concept). **Not a hazard.**
- `canvas_sync.py`'s `_sync_library_canvas` dispatcher has a literal
  `kind == "export"` branch calling `screen._build_library_export_state()`.
  Repo-wide grep for `_sync_library_canvas(..., "export"...)` call sites:
  **zero** — unreached by any caller, export-cluster or otherwise. No live
  coupling for this move to break.

## 3. Bind-list classification (final, 22-method cluster)

**Framework services** (live-read `@property` from screen, never
snapshotted): `app_instance`, `app`, `call_after_refresh`, `is_mounted`,
`query_one`, `refresh`.

**Named constructor dependencies** — general shell helpers the moved bodies
call with explicit args: `apply_open_item_surface`, `flush_note_save`,
`set_library_destination_with_conversation_fence`,
`sync_library_emergency_guard_presentation`, `close_open_library_choice_strip`,
`focus_library_hub_entry`, `select_library_rail_row`,
`focus_library_choice_strip_active`, `focus_library_control`.

**Read-only shared-state accessors**: `library_selected_row_id_accessor`
(the recipe's own canonical ≥2-subsystems field, 226 refs — confirmed via
AST Store-context check: no moved body writes it directly, only through the
two callables above), `library_prompts_mutation_in_flight_accessor` (a
DIFFERENT subsystem's own state, read as a guard by
`_open_library_export_canvas`).

**Screen-resident siblings** (round-2/round-3 exclusions, bound as named
callables exactly like any other general dependency despite staying
screen-resident): `build_library_export_state`,
`start_library_export_counts_worker`, `start_library_export_worker`,
`apply_library_export_success`, `apply_library_export_cancelled`,
`update_library_export_canvas_after_run`, `handle_library_export_cancel`.
(The two `@work` names from round 2 need NO binding at all — their only
callers are themselves excluded in round 3.)

**Class-level binding (not a constructor dependency)**: `_safe_text`,
identical shape to the conversations controller's own —
`LibraryExportController._safe_text = staticmethod(LibraryScreen._safe_text)`,
one line added in `library_screen.py`'s trailing module code, right after
the existing `LibraryConversationsController._safe_text` line.

## 4. Free-name resolution check

Ran the recipe's documented AST census (Load-context `Name` nodes not bound
locally, not builtins, not `self`/`cls`) against the final 22-method set
twice — once mid-investigation (31 methods), once after the round-3
reduction to 22 — and cross-checked every resulting name against
`library_screen.py`'s own import block, copying the identical import
statement (adjusting only relative-import depth where the target module
moved from `UI/Screens/` to `UI/Library_Modules/`, e.g.
`LibraryEntryReconcileResult`'s `..Library_Modules.screen_support_types` →
`.screen_support_types`). Final import set: `ContentType`, `ExportScope`,
`count_export_scope`, `DEFAULT_MEDIA_QUALITY`, `MEDIA_QUALITY_OPTIONS`,
`default_export_name`, `normalize_export_destination`,
`NoteFlushOutcomeKind`, `LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP`,
`LIBRARY_ROW_INGEST_EXPORT`, `FileSave`, `validate_path_simple`,
`LibraryExportCanvas`, plus stdlib (`asyncio`, `threading`, `Path`) and
framework (`Button`, `Input`, `Static`, `NoMatches`, `QueryError`, `on`,
`logger`, `escape_markup`). A post-write independent AST verification
(walking every function body in the compiled controller module, resolving
every `Name.Load` against imports/module-level defs/`self`/`cls`/locally-
bound names) confirmed **zero unresolved free names** — run twice, once
after the 31-method draft, once after the final 22-method version (the
reduction dropped `LibraryEntryReconcileResult`, `LibraryExportFormState`,
`build_library_export_form_state`, `format_last_export_line`,
`apply_library_export_submit_gate`, `format_export_progress_line`,
`resolve_export_selections`, `time` — all exclusively used by the 9
round-3-excluded methods that no longer live in this module).

## 5. Byte-for-byte verification — method and result

Two independent AST-based checks, both against `git show 264314c5f:
tldw_chatbook/UI/Screens/library_screen.py` (the pre-task base):

1. **AST-unparse comparison** (catches logic differences, blind to
   comment/whitespace text): for all 22 moved methods, `ast.unparse` of the
   body, full signature (`ast.unparse(fn.args)`), and decorator list from
   base vs. controller — **all 22 matched exactly**.
2. **Raw-text comparison** (decorators through `end_lineno`, comments and
   whitespace included) — **all 22 raw text bodies identical, byte-for-byte**.

The SAME raw-text check, run a second time after reverting the 9 round-3
docstring additions (see §1's note), confirmed those 9 excluded methods are
now **also** byte-for-byte identical to base — i.e. every one of the 51
original candidates that isn't a delegator on the screen is, in the final
commit, unedited from what `264314c5f` already had.

## 6. TDD evidence

`Tests/Architecture/test_library_export_wiring.py` extended with 5 new
tests (`test_export_controller_owns_its_cluster`,
`test_screen_delegates_export_handlers`,
`test_export_cluster_staticmethods_forward_to_the_controller_class`,
`test_export_controller_exposes_every_state_field`,
`test_export_controller_safe_text_is_bound_via_screen_import`) plus the
existing state-field test, mirroring the conversations exemplar's
`_BROWSE_CLUSTER_METHOD_NAMES` shape.

RED demonstrated by temporarily inserting a fake 32nd name
(`"_this_name_does_not_exist_on_the_controller"`) into
`_EXPORT_CLUSTER_METHOD_NAMES` and confirming exactly the two tests that
check the cluster/delegator surface fail
(`test_export_controller_owns_its_cluster`,
`test_screen_delegates_export_handlers`), 2 failed / 4 passed; reverted and
re-ran to confirm 6/6 GREEN.

## 7. Verification battery

All commands run from `.worktrees/library-decomp-foundation`, `.venv/bin/python`.

**Wiring** — `Tests/Architecture/test_library_export_wiring.py` +
`Tests/Architecture/test_library_conversations_wiring.py` (regression: the
`__init__` edit touches shared wiring) + `Tests/UI/test_library_export_
characterization.py`: **18 passed**.

**Both size-ratchet guards** (`test_screen_does_not_grow_past_its_budget`,
`test_budget_is_not_left_slack_after_a_wave`), full suite:
`Tests/Architecture/test_screen_size_ratchet.py` — 3 passed, 2 failed
(`chat_screen.py`'s two documented-pre-existing rows, unrelated churn on
`dev`), library_screen.py's own 2 rows green.

**Recompose census + its anti-slack guard** (`Tests/UI/
test_library_recompose_ratchet.py`) + **support-layer surface**
(`Tests/Architecture/test_library_support_layer_surface.py`): **14 passed**
— this move touches zero `refresh(recompose=True)` call sites (pure
relocation), pin (63) and its slack guard unaffected.

**Regression re-run of the 9 round-3-excluded methods' own test files**
(post-fix): `test_library_export_cancel.py` + `test_library_export_progress_
apply.py` + `test_library_export_receipt.py` +
`test_library_shell.py::test_library_export_registry_failure_warns_it_
wont_appear_in_artifacts` + `Tests/Library/test_library_export_execution.py`
+ `Tests/Library/test_library_export_roundtrip.py`: **43 passed, 2 failed**
— both failures independently confirmed via `git stash -u` to be
pre-existing (identical failure on pristine baseline, unrelated to this
move): `test_cancel_apply_current_run_sets_cancelled_status` (already
recipe-documented from Task 2) and `test_library_export_success_records_a_
durable_receipt_with_the_real_path` (newly found by this task, same root
cause — a stale fake missing `_sync_library_emergency_guard_presentation`
— appended to recipe §7).

**Class-level/instance-level monkeypatch spot-check** (recipe §3's own
concern, since `_compute_library_export_counts` moved and its sole callers
`_start_library_export_counts_worker`/`_run_library_export_counts_worker`
both stay screen-resident): `Tests/UI/test_library_entry_compose_once.py`'s
4 tests that `monkeypatch.setattr(LibraryScreen, "_compute_library_export_
counts", ...)` (class-level) and `monkeypatch.setattr(screen, "_run_library_
export_counts_worker", ...)` (instance-level on a real screen) — **22
passed**. Confirmed both patches are still correctly observed: the whole
call graph stays screen-routed because the CALLER
(`_start_library_export_counts_worker`) is itself a round-3 exclusion.

**`-k "export and library"` under `Tests/UI`** (the recipe's per-task
narrow check): **14 failed, 117 passed** — the failure NAMES match the
recipe's documented pre-existing 14-item list from Task 2 **exactly**, no
more, no fewer. Zero branch-caused failures.

**Free-name resolution + byte-for-byte**: §4/§5 above.

**Preflight** (`./scripts/preflight.sh`): all six checks green — CSS bundle
sync, profile-owned-path census, production diagnostic inventory (regen'd
via `check_persistent_diagnostic_inventory.py --write` after confirming,
via `--statements ... --since <base>`, that the delta is exactly 5
`logger.*` statements relocating with IDENTICAL digests — no rewording, no
new interpolation — from `library_screen.py` to the controller module,
matching the final 22-method cluster's own membership exactly), backlog
task ids, chachanotes table allowlist, index plan pins.

**Full xdist paired-baseline sweep** (recipe §7,
`Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal`):

- Branch: **333 failed, 3901 passed** (1280.15s / 21m20s).
- Baseline (`git stash -u` to `264314c5f`, identical command, my stash
  popped back afterward and re-verified with a fresh preflight run):
  **332 failed, 3902 passed** (1249.82s / 20m49s).
- Diff (`comm` on the two sorted unique-failure-name sets): **328 shared**
  (the pre-existing backdrop), **5 branch-unique**, **4 baseline-unique** —
  small, roughly balanced counts flipping in both directions, exactly the
  xdist-noise shape recipe §7 describes. None of the 9 (5+4) touches any
  export-cluster name; they span Media reader traversal, Notes reader,
  Prompts canvas/reader, and two Notes-shell tests — all subsystems this
  task never touched.
- Recipe §7 step 4 (re-run every branch-unique failure directly,
  single-process, individually AND combined, before trusting it as real):
  all 5 branch-unique names —
  `test_library_media_reader_traversal_t22207.py::
  test_loading_banner_paints_in_place_without_body_rebuild`,
  `test_library_notes_reader.py::
  test_wide_editor_deep_link_keeps_reader_navigation_and_local_back`,
  `test_library_prompts_canvas.py::
  test_library_prompt_history_no_change_keeps_selection_and_retry_available`,
  `test_library_shell.py::
  test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`,
  `test_library_shell.py::test_library_shell_note_id_deeplink_opens_note_editor`
  — run together in one single-process command: 2 failed (the notes-reader
  and blank-note-GC ones), 3 passed outright. Each of those 2 re-run FULLY
  ISOLATED (its own single-process command, nothing else in the run):
  **both passed**. Zero of the 5 is a deterministic regression; all 5 are
  xdist-specific ordering/shared-state flakiness the recipe's own §7
  explicitly warns is expected and not attributable to this task.

## 8. Measurement + pin rows

`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
(`Tests/Architecture/test_screen_size_ratchet.py`): `43930/1282 →
43432/1282`. Method count unchanged (a pure move: 22 `FunctionDef`s out as
bodies, 22 back in as one-line delegators; the 9 round-3-excluded methods
were never removed at all — a temporary docstring-addition detour, reverted
before commit). Both the ceiling guard and the no-slack guard pass at the
new value.

Recompose census pin (`Tests/UI/test_library_recompose_ratchet.py`):
unchanged at 63 — this move relocates zero `refresh(recompose=True)` call
sites.

## 9. Files changed

- `tldw_chatbook/UI/Library_Modules/library_export_controller.py` (new,
  1307 lines) — `LibraryExportController`, 22 moved methods, framework/
  named-dependency properties, generated own-state shim loop.
- `tldw_chatbook/UI/Screens/library_screen.py` (modified) — import added;
  `self._export_controller = LibraryExportController(...)` constructed in
  `__init__` right after `self._conversations_controller`; 22 one-line
  delegators replacing the moved bodies; the 9 round-3-excluded + 2
  round-2-excluded methods left byte-for-byte untouched;
  `LibraryExportController._safe_text = staticmethod(LibraryScreen.
  _safe_text)` added after the existing conversations-controller line.
- `Tests/Architecture/test_library_export_wiring.py` (modified) —
  `_EXPORT_CLUSTER_METHOD_NAMES` (22), `_EXPORT_CLUSTER_STATICMETHOD_NAMES`
  (5), 5 new tests.
- `Tests/Architecture/test_screen_size_ratchet.py` (modified) — `_BUDGETS`
  row lowered to `43432, 1282`, with a comment recording the full
  round-1/2/3 accounting.
- `Docs/security/production-diagnostic-inventory.json` (regenerated) — 5
  `logger.*` call sites relocated to the controller module (verified
  identical text, not reworded).
- `backlog/docs/library-decomposition-recipe.md` (modified) — one new
  pre-existing-failure entry appended to §7's documented list, plus a
  forward note about the `Tests/Library/`-widening lesson for future
  subsystem sweeps.

No `.git-blame-ignore-revs` entry yet in this draft — the move commit's
hash is appended via `git rev-parse HEAD` (never typed from memory) in a
same-PR follow-up commit once the implementation commit lands; see §11.

## 10. Self-review

- Ownership re-derived mechanically at execution time (51-method census
  matches Task 2's own independently), not trusted from the brief's ~51
  estimate.
- Every exclusion decision (29 methods across 3 rounds) is backed by
  concrete evidence read from the actual method body or an actual test
  failure — none guessed from a name pattern.
- **Caught and corrected a self-introduced canon violation before
  committing**: my first pass added "why this stays" docstring paragraphs
  to the 9 round-3-excluded screen methods. On reflection against the
  recipe's "body untouched" precedent language and the transform
  whitelist's absence of a "document an unmoved method" entry, reverted
  all 9 to byte-for-byte-identical-to-base text and moved that reasoning
  into the controller's module docstring instead (§1, §5) — re-verified
  with the same raw-text AST diff used for the moved 22, not just visual
  inspection.
- Every `git rev-parse`/`git show`/`git stash` output cited in this report
  was read from actual command output, never typed from memory (the
  recipe's own documented fabrication incident this instruction exists to
  prevent).
- Sweep evidence follows recipe §7's procedure: narrow `-k` check first
  (name-for-name match against the documented pre-existing list), THEN the
  full xdist paired-baseline sweep — not skipped, not summarized before
  actually running.
- The full xdist branch/baseline diff (§7) and the blame-ignore commit
  hash (§9/§11) were both genuinely pending earlier drafts of this report
  (the sweep was still running; no commit existed yet) — flagged explicitly
  in those drafts rather than filled with a guess, per the recipe's own
  "self-review overclaimed"/fabrication incidents. Both are now filled in
  with real command output (§7's sweep numbers, §11's `git rev-parse`
  output) after the fact, not backfilled from memory.

## 11. Commits

- `4cc9b6109b47d39e4622d6db9f4d1158e1d9cdd0` — `refactor(library): export
  controller (export series 2/3)` (implementation; confirmed via
  `git rev-parse HEAD` immediately after committing, never typed from
  memory).
- Blame-ignore follow-up (this task's second commit) — appends the hash
  above to `.git-blame-ignore-revs` with an explanatory comment, same
  shape as the conversations exemplar's own entries.
