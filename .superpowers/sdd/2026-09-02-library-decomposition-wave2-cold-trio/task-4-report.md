# Task 4 report — Export series 3/3: cleanup PR (shims out, delegator pruning, ratchet lowered)

Branch `refactor/library-decomp-wave2-cold-trio`, worktree
`.worktrees/library-decomp-foundation`. Base commit `5e74c64cb` (Task 3, export
controller move + blame-ignore follow-up). Recipe:
`backlog/docs/library-decomposition-recipe.md`, conversations exemplar's Task 9
is the worked example this mirrors.

## 1. Dynamic-dispatch census (before touching anything)

Re-derived Task 3's own forward-flagged risk before deleting the shim block:

- Repo-wide grep for `_library_export_<field>` across `tldw_chatbook/` and
  `Tests/`, split by receiver shape. All literal `self._library_export_<field>`
  Attribute nodes in `library_screen.py` (42, via an AST census — see §3) are
  a real code reference and must move; the controller module's OWN 13 uses of
  the same flat names (its own generated shim reading
  `self._export_state_accessor().<field>`, installed by Task 3) are the
  permanent enabling layer and were left untouched, per the task's own
  "controller shims STAY" instruction.
- **The one genuine dynamic-dispatch site**: `_library_open_choice_strip`
  (returns `(subject, opener_selector, visibility_attr)` for whichever of
  media/prompts/skills/export's converged choice strip is open) and
  `_close_open_library_choice_strip` (`setattr(self, visibility_attr,
  False)` + a `{name: canvas_kind}` dict keyed by the same string) — both
  shell/plumbing, shared across 4 subsystems, never move. Export's own
  return value used the literal string `"_library_export_quality_choices_
  visible"` for `visibility_attr`; once the screen shim is deleted that flat
  name no longer resolves. Fixed by changing the STRING (not the mechanism)
  to `"_export_state.quality_choices_visible"` in both places it must match
  (`_library_open_choice_strip`'s return tuple and
  `_close_open_library_choice_strip`'s dict key), and routing the write
  through the conversations exemplar's own `_assign_library_reader_
  preferences_attribute(owner, attribute, value)` helper (already a fully
  generic dotted-vs-flat passthrough) instead of a bare `setattr` — its
  docstring was extended with a short paragraph documenting this second,
  unrelated caller rather than adding a second near-identical helper. The
  plain-read reference to the same field inside `_library_open_choice_strip`
  (`self._library_export_quality_choices_visible` → `self._export_state.
  quality_choices_visible`) is one of the 42 AST-found sites, handled by the
  same mechanical pass as every other screen-side reference.
- Re-verified `canvas_sync.py`'s `_sync_library_canvas` "export" branch and
  the two other dynamic-dispatch sites Task 3's own census found
  (`_library_rail_preferences()`'s `f"{section_id}_open"` lookup,
  `_replace_library_reader_preference`'s 7-destination dict): neither
  destination set includes any Export name; still **not a hazard**, confirmed
  by re-grep, not assumed from Task 3's report.

## 2. Screen-side field retarget

Mechanical `ast`-driven census of every `self._library_export_<field>`
Attribute node in `library_screen.py` (excluding the shim block itself, which
uses dynamic string concatenation, not literal attribute syntax, so it never
appeared in this census):

**42 literal `self._library_export_<field>` sites**, across 19 methods
(`_library_route_shortcuts_for_current_state`,
`_library_emergency_return_eligibility`, `on_mount`, `save_state`,
`restore_state`, `_start_library_export_counts_worker`,
`_apply_library_export_counts`, `_build_library_export_state`,
`handle_library_export_cancel`, `_apply_library_export_cancelled`,
`_apply_library_export_success`, `_apply_library_export_progress`,
`_select_library_rail_row_after_source_admission`,
`_library_open_choice_strip`, plus a handful more) — a single Python
regex/AST-verified script (word-boundary-anchored per field name, so
`counts` never over-matches `counts_request_id`) rewrote all 42 to
`self._export_state.<field>` in one pass; re-verified afterward with a
zero-result `grep -n "self\._library_export_"` over the whole file. Two of
the 42 sites are the dynamic-dispatch reads described in §1.

Plus the 2 dynamic-dispatch STRING-LITERAL sites (§1) and the 1 `setattr` →
`_assign_library_reader_preferences_attribute` call-site rewrite, handled
separately since they are not literal attribute expressions an AST census
finds.

**Docstring/comment accuracy pass** (not load-bearing, but the recipe's own
precedent — see conversations Task 9's report — treats stale-name prose as
worth fixing during cleanup): 8 comments/docstrings in `library_screen.py`
that quoted the old flat name in prose (e.g. "`_library_export_run_id`'s
docstring", "stomp `_library_export_running`/`_error`/`_status`") were
updated to the new `_export_state.<field>` spelling; one historical mention
was left in a test file's comment, explicitly labeled "pre-Task-4-cleanup",
since it is describing what the code used to say, not what it says now.

## 3. Delegator census (all 22)

Per-name repo-wide grep (`tldw_chatbook/`, `Tests/`) for every one of Task 3's
22 moved-cluster names, cross-checked against which SCREEN-resident method
(if any) calls it, since the round-2/round-3 exclusions (11 of the original
51 candidates) stayed screen-resident and call several of these delegators
internally:

| Delegator | Caller(s) found | Verdict |
|---|---|---|
| `_default_library_export_form` | `__init__` (computed default for `LibraryExportState.form`) | KEEP |
| `_reset_library_export_transient_state` | `_select_library_rail_row_after_source_admission`; `Tests/UI/test_library_export_receipt.py` (direct) | KEEP |
| `_open_library_export_canvas` | 8 screen call sites (every browse canvas's "Export…" action) + `LibraryConversationsController.__init__`'s wiring lambda; many tests | KEEP |
| `_library_export_is_server_mode` | **NONE** outside the controller's own internal `self.<name>()` (×2) and its own delegator definition | **PRUNED** |
| `_resolve_library_export_chachanotes_db` | `_start_library_export_counts_worker`, `_start_library_export_worker` (both screen-resident, round-3-excluded); `Tests/Library/` fakes | KEEP |
| `_compute_library_export_counts` (static) | `_start_library_export_counts_worker`, `_run_library_export_counts_worker`; class-level-monkeypatched + directly called by tests | KEEP |
| `handle_library_export_submit` | `@on` handler; tests call directly | KEEP |
| `_build_library_export_payload` (static) | `_run_library_export_worker`; tests | KEEP |
| `_run_library_export_via_service` (static) | `_run_library_export_worker`; tests | KEEP |
| `_marshal_library_export_success` | `_run_library_export_worker` | KEEP |
| `_marshal_library_export_failure` | `_run_library_export_worker` (×3 call sites); `Tests/Library/` fake kwarg | KEEP |
| `_marshal_library_export_cancelled` | `_run_library_export_worker` | KEEP |
| `_build_library_export_success_message` (static) | `_apply_library_export_success`; tests (direct + monkeypatch) | KEEP |
| `_apply_library_export_failure` | `_start_library_export_worker`; `Tests/Library/` fake kwargs ×2 | KEEP |
| `_refresh_library_export_status_line` | `handle_library_export_cancel`, `_apply_library_export_progress`; `Tests/UI/` fake kwargs ×2 | KEEP |
| `action_library_export_back` | `action_*` rule (always kept); tests call directly | KEEP |
| `handle_library_export_name_changed` | `@on` handler | KEEP |
| `handle_library_export_description_changed` | `@on` handler | KEEP |
| `handle_library_export_quality` | `@on` handler | KEEP |
| `handle_library_export_quality_choice` | `@on` handler | KEEP |
| `handle_library_export_choose_destination` | `@on` handler; referenced in a modal-dismissal census table | KEEP |
| `_apply_library_export_destination` | `Tests/UI/test_library_shell.py` (×7 direct calls) + `test_library_export_characterization.py` | KEEP |

**Net: 21 KEEP, 1 PRUNED** (`_library_export_is_server_mode`). This is a
sharply higher keep-ratio than the conversations exemplar's 43-of-61 — not a
shallower census (every one of the 22 was individually checked, not assumed
live), but a direct consequence of Export's own round-2 (`@work` self-type
assertion)/round-3 (unbound-fake-self) exclusions: 11 of the 51 naive
"export"-named candidates stayed screen-resident specifically because a test
or a framework decorator reaches them unbound, and every one of those 11
calls its sibling delegators internally via `self.<name>()` — exactly the
shape that keeps a delegator alive. Conversations had no exclusion class this
large, so its moved cluster called itself controller-to-controller far more
often, orphaning far more one-liners.

## 4. Shim block deletion

The Task-2-generated `# --- BEGIN/END generated export-state shims ---` block
(13 dynamically-installed `property` descriptors, one per `LibraryExportState`
field) was deleted wholesale from `library_screen.py`'s trailing module code,
once §2/§3 confirmed zero remaining consumers of the flat names anywhere in
`tldw_chatbook/` or `Tests/` outside the controller's own permanent shim
layer (which reads `self._export_state_accessor().<field>`, untouched by
this task).

## 5. Test retarget — per file

Every test file reaching a flat `_library_export_<field>` name on a REAL
`LibraryScreen`/`restored` instance was retargeted to
`screen._export_state.<field>` / `restored._export_state.<field>` with a
scoped regex (receiver-name-anchored, so `LibraryScreen._compute_library_
export_counts` method references were never touched). Every test building an
UNBOUND `SimpleNamespace` fake with flat `_library_export_<field>=...`
kwargs (the recipe §11 "unbound fake-self" precedent) had those kwargs
regrouped into a nested `_export_state=SimpleNamespace(...)` constructor
argument, matching the moved body's new `self._export_state.<field>` reads —
**assertions kept byte-for-byte**; only the receiver path changed.

| File | Real-instance retargets | Fake-restructure retargets | Notes |
|---|---|---|---|
| `Tests/UI/test_library_entry_compose_once.py` | 7 (`screen`/`active_screen`) | 0 | |
| `Tests/UI/test_library_export_characterization.py` | 12 (`screen`) | 0 | module docstring's "shim keeps resolving identically" claim corrected to describe the Task-4 retarget |
| `Tests/UI/test_library_prompts_canvas.py` | 5 (`screen`) | 0 | |
| `Tests/UI/test_library_choice_strips.py` | 2 (`screen`) | 0 | |
| `Tests/UI/test_library_honesty_accessibility.py` | 1 (`screen`) | 0 | 1 stale-name comment fixed |
| `Tests/UI/test_library_shell.py` | 60 (`screen`) | 0 | 3 stale-name docstring/comment mentions fixed |
| `Tests/UI/test_library_export_receipt.py` | 14 (`screen`/`restored`) | 2 fakes (13 fields total moved into nested `_export_state`) | 1 docstring fix |
| `Tests/UI/test_library_export_cancel.py` | 0 | 3 fakes (10 field-assignments total) | includes the recipe-documented pre-existing `test_cancel_apply_current_run_sets_cancelled_status` failure, unaffected in cause |
| `Tests/UI/test_library_export_progress_apply.py` | 0 | 1 fake-builder helper (3 fields, shared by 3 tests) | |
| `Tests/Library/test_library_export_execution.py` | 0 | 1 fake (2 fields) | |
| `Tests/Library/test_library_export_roundtrip.py` | 0 | 1 fake (3 fields) + 3 assertion-path updates | includes the recipe-documented pre-existing `test_library_export_success_records_a_durable_receipt_with_the_real_path` failure, unaffected in cause |
| `Tests/Architecture/test_library_export_wiring.py` | — | — | `test_state_object_fields_match_the_shim_surface` DELETED (screen shim gone, mirrors conversations Task 9); `_EXPORT_CLUSTER_SCREEN_DELEGATOR_PRUNED` frozenset + skip/absence-assert added to `test_screen_delegates_export_handlers`; module docstring rewritten |

No test file needed an assertion VALUE change — every retarget is a receiver-
path rewrite only, confirmed by running each file before and after and
diffing the pass/fail set (see §7).

## 6. Import verification

The 5 named dead imports (task brief) verified single-occurrence in
`library_screen.py` (their own import line only) via per-name grep, then
checked against `Tests/Architecture/test_library_support_layer_surface.py`'s
`_SURFACE` dict (the PR-0a re-export contract the conversations exemplar's
own "dead within this file is not the same question as dead" lesson warns
about) — none of the 5 belong to any of `_SURFACE`'s 5 listed modules
(`screen_constants`, `screen_support_types`, `note_session_port`,
`canvas_sync`, `screen_helpers`), all of which are PR-0a support modules
unrelated to `Library/library_export_scope.py`/`library_export_state.py`/
`library_shell_state.py` (where these 5 names actually live). Deleted:
`LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP`, `MEDIA_QUALITY_OPTIONS`,
`count_export_scope`, `default_export_name`, `normalize_export_destination`.

No newly-dead imports were produced by the field-retarget pass itself: the
42 substitutions only changed an attribute RECEIVER
(`self._library_export_x` → `self._export_state.x`), never removed a call to
an imported name. `LibraryExportState` and `dataclasses` (the shim block's
only two import-level dependents) both remain used elsewhere
(`LibraryExportState(...)` is still constructed in `__init__`;
`dataclasses.replace`/`dataclasses.asdict` are used ~80 times elsewhere in
the file).

## 7. Verification battery

All commands run from `.worktrees/library-decomp-foundation`, `.venv/bin/python`.

**Wiring suites**: `Tests/Architecture/test_library_export_wiring.py` (5
passed, post-deletion/post-pruning) + `Tests/Architecture/test_library_
conversations_wiring.py` (6 passed, regression check) — **11 passed**.

**Characterization files**: `Tests/UI/test_library_export_characterization.py`
(5) + `Tests/UI/test_library_conversations_characterization.py` (4) — **9
passed**, run together to reconfirm the documented split.

**Both size-ratchet guards**, full suite
(`Tests/Architecture/test_screen_size_ratchet.py`): 3 passed, 2 failed — the
two documented-pre-existing `chat_screen.py` rows (unrelated concurrent dev
growth); both `library_screen.py` rows pass at the new value.

**Recompose census + its anti-slack guard**
(`Tests/UI/test_library_recompose_ratchet.py`) + **support-layer surface**
(`Tests/Architecture/test_library_support_layer_surface.py`): **14 passed** —
this cleanup touches zero `refresh(recompose=True)` call sites; pin (63)
unaffected.

**Per-file regression re-run of every retargeted file**
(`test_library_entry_compose_once.py`, `test_library_export_characterization.py`,
`test_library_prompts_canvas.py`, `test_library_choice_strips.py`,
`test_library_honesty_accessibility.py`, `test_library_export_receipt.py`,
`test_library_export_cancel.py`, `test_library_export_progress_apply.py`,
`Tests/Library/test_library_export_execution.py`, `Tests/Library/
test_library_export_roundtrip.py`, plus both wiring files): **24 failed, 511
passed** in one combined run. All 24 failures independently confirmed
pre-existing:

- 2 already recipe-documented (`test_cancel_apply_current_run_sets_cancelled_
  status`, `test_library_export_success_records_a_durable_receipt_with_the_
  real_path` — the stale-fake `_sync_library_emergency_guard_presentation`
  gap, unrelated to this task's field retarget).
- 1 confirmed via a fresh `git stash -u` to the pristine pre-task tree,
  identical failure on both trees: `test_library_choice_strips.py::
  test_media_type_strip_works_in_both_layouts` (a narrow-layout CSS-class
  mount timeout, unrelated to Export or the choice-strip dynamic-dispatch
  fix) — newly appended to recipe §7's documented list.
- The remaining 21 (`test_library_entry_compose_once.py` ×3, `test_library_
  prompts_canvas.py` ×13, `test_library_honesty_accessibility.py` ×4)
  independently confirmed via a SECOND `git stash -u` run of exactly these 3
  files on the pristine tree: **24 failures on baseline too**, and a
  `comm`-diff of the two 3-file failure-name sets shows **zero branch-unique,
  3 baseline-unique** (2 in `test_library_prompt_history_collapse_during_
  restore_detail_fetch_stays_closed`-shaped tests and 1 stale-search-caret
  test that pass on branch but fail on baseline — noise in the OTHER
  direction, not attributable to this task).

**`-k "export and library"` under `Tests/UI`** (recipe's per-task narrow
check): **14 failed, 117 passed** — failure NAMES match the recipe's
documented pre-existing 14-item list from Task 2 **exactly**, no more, no
fewer (verified name-for-name, not just the count).

**Preflight** (`./scripts/preflight.sh`): all six checks green — CSS bundle
sync, profile-owned-path census, production diagnostic inventory (unchanged;
this task relocates zero `logger.*` call sites), backlog task ids,
chachanotes table allowlist, index plan pins.

**Full xdist paired-baseline sweep** (recipe §7,
`Tests/UI -k "library" -p no:randomly -q -n 8 --dist worksteal`):

- Branch: **329 failed, 3905 passed** (1256.96s / 20m57s).
- Baseline (`git stash -u` to `5e74c64cb`, identical command, stash popped
  back afterward and re-verified with `ast.parse` + a fresh preflight run):
  **331 failed, 3903 passed** (1275.73s / 21m16s).
- Diff (`comm` on the two sorted unique-failure-name sets): **329 shared**
  (the pre-existing backdrop), **0 branch-unique**, **2 baseline-unique**
  (`test_library_notes_reader.py::test_wide_editor_deep_link_keeps_reader_
  navigation_and_local_back`, `test_library_shell.py::test_library_shell_
  note_id_deeplink_opens_note_editor` — fail on baseline, pass on branch;
  noise in the opposite direction).
- Recipe §7 step 4 (re-run every branch-unique failure) is vacuous here:
  **zero branch-unique failures to re-run.** This is a cleaner result than
  the conversations exemplar's own 5+4 split, but the sweep was not
  skippable on that basis — the field-retarget pass touched 13 test files
  and one 4-subsystem-shared dynamic-dispatch site, either of which could
  plausibly have broken something the narrower `-k` checks don't cover.

## 8. Measurement + pin rows

`_BUDGETS["tldw_chatbook/UI/Screens/library_screen.py"]`
(`Tests/Architecture/test_screen_size_ratchet.py`): `43432/1282 →
43413/1281`. Re-measured with the same `ast`-walk script the recipe
prescribes, on the final tree (no rebase was needed — no upstream Library
churn landed on `dev` during this task's execution window). Method count
drops by exactly 1, matching the 1 pruned delegator (`_library_export_is_
server_mode`) 1-for-1 — a pure deletion, nothing replaces it. Both the
ceiling guard and the no-slack guard pass at the new value.

Full pin trajectory for the export series: `43965/1282 (pre-Task-2) →
43930/1282 (Task 2, state PR) → 43432/1282 (Task 3, controller PR) →
43413/1281 (Task 4, cleanup PR, final)`.

Recompose census pin (`Tests/UI/test_library_recompose_ratchet.py`):
unchanged at 63 — this cleanup relocates zero `refresh(recompose=True)` call
sites and adds/removes none.

## 9. Recipe diff summary

`backlog/docs/library-decomposition-recipe.md`:

- §7's documented pre-existing-failures list: 1 new entry appended
  (`test_media_type_strip_works_in_both_layouts`, confirmed via `git stash
  -u`).
- New §12, "The export series, as landed": the fields/methods-moved table
  (state/controller/cleanup, mirroring §11's shape), the pin trajectory, the
  22-delegator/21-keep/1-prune accounting (with the ratio contrast against
  conversations' 43/61 explained), and five lessons:
  1. A FOURTH bypass shape, new to this series: "framework-decorator
     self-type assertion" (Textual's `@work` decorator's
     `isinstance(self, DOMNode)` runtime check) — a permanent runtime
     contract, not a test artifact, discovered by reading the decorator's
     own source rather than assumed.
  2. The "unbound fake-self" shape scaling past the exemplar's own
     precedent (5 → 9 methods, 1 → 6 test files, into `Tests/Library/` which
     the canonical sweep root does not cover) — a forward note for future
     subsystems' controller-PR batteries to widen their `-k` search
     deliberately.
  3. The dynamic-dispatch fix's helper-reuse decision: extend an existing
     fully-generic `_assign_...attribute` helper's docstring for a second,
     unrelated dispatch mechanism rather than writing a near-duplicate.
  4. The screen-shim wiring test's retirement (delete, don't retarget)
     reconfirmed a second time, independent of the conversations exemplar.
  5. Zero-branch-unique sweep evidence is not proof the sweep step is
     skippable for future "small" cleanup PRs.

## 10. Self-review

- Every retarget was verified by re-running the affected test file (or a
  scoped group) before moving to the next, not batched blind — the single
  documented pre-existing failure was independently confirmed via `git
  stash -u` at the point it was first encountered
  (`test_media_type_strip_works_in_both_layouts`), not assumed benign.
- The delegator census (§3) reached a conclusion (21 keep, 1 prune) that
  looked suspicious against the conversations exemplar's own 43/61 ratio;
  rather than treat the 21/22 KEEP result as evidence of a shallow census,
  each of the 22 was individually traced to a concrete caller (a screen-
  resident sibling, an `@on` decorator, or a direct test reference) before
  being kept — the disproportion is explained by Export's own round-2/
  round-3 exclusion classes (§3), not asserted without evidence.
- The dynamic-dispatch fix (§1) reused an EXISTING helper rather than
  writing a new one; this decision is recorded with its reasoning in both
  the code comment and this report, not silently done.
- `git rev-parse`/`git stash`/`git diff --stat` outputs cited in this report
  were read from actual command output, never typed from memory (the
  recipe's own documented fabrication-avoidance instruction).
- The full xdist paired-baseline sweep (§7) and the final ratchet
  measurement were both genuinely pending in earlier drafts of this report
  (the sweep was still running) — not filled with a guess; both are now
  real command output.
- Commit hashes below were read via `git rev-parse HEAD` immediately after
  each commit, not typed from memory.

## 11. Commits

- `cdb43ebcc1a1e2e113ac78d0e16f2c88ac016b3f` — `refactor(library): export
  cleanup — shims out, ratchet lowered (export series 3/3)` (implementation;
  confirmed via `git rev-parse HEAD` immediately after committing).
- `58118128c479d05dfd7450daa5b59a03c7be842a` — `chore(library): blame-ignore
  follow-up for export cleanup PR` (appends the hash above to
  `.git-blame-ignore-revs`; confirmed via `git rev-parse HEAD`).
