# Task 7 implementation report — TASK-38542

Implementation is complete and remains In Progress for controller review. ADR required: no; this directly implements `backlog/decisions/286-console-prompt-mix-ins.md`.

## Delivered behavior

- Added `PromptMixinEditor`: Name, one literal Instructions TextArea and Before/After Select, with stable `mixin-name`, `mixin-body`, `mixin-placement` IDs and parent-request save/cancel events. The approved create-only nullable definition renders an empty body with Before default; Save always constructs a strict `LiteralMixinDefinition`. Malformed/future stored definitions never become blank creates.
- Library retains raw invalid drafts privately (repr hidden), metadata, expected version, source identity, dirty Escape/Discard/Cancel and conflict authority. Mix-in saves preserve literal whitespace/braces/headings, carry expected_version and source capability/size checks, and refuse source changes. Existing inherited Work-pane composition is reused without modifying `library_prompt_work_pane.py`.
- Added explicit All types / Prompts / Recipes / Mix-ins filtering to the existing exact browse scope; count, query, collection, pagination, restore and empty-filter clearing carry the same type. A real SQLite journey verifies 22 filtered rows remain 20+2 across pages.
- Added `ConsoleMixinEnableRequest` in the existing `CONSOLE_PROMPT_INSERT` slot: body-free source ref, exact target session/incarnation/preference revision, existing TTL, and repr-hidden transient `(scope, backend, DB)` identity fences. Copies preserve these identity objects without deepcopy or persistence. Ordinary Prompt requests retain their existing positive type path; a newer Prompt or Mix-in intent supersedes the earlier one.
- Runtime `enable_reference` resolves through PromptScopeService, verifies exact source objects before awaiting and after the read, then uses the existing preference CAS/save owner. New entries append to their placement group's end; existing entries are enabled in place without duplication/reordering or last-used publication. Composer and System are untouched. A named accessor is injected by the existing Console wiring owner.
- Added source/absent-target remedies, guarded unsupported states and plain literal-conflict recovery copy. Existing token-backed forms/classes, keyboard conventions and production stylesheet sequence are reused. No TCSS/bundle changes or new dependencies.

## Interface decisions

Controller explicitly approved: nullable definition for explicit create/draft only; shared Prompt pending slot with latest-intent mutual supersession; thin Library screen dispatch plus sanitized screen-state projection and Console wiring ownership; transient three-object source fences to reject same-path DB reopen/backend replacement and scope-wrapper replacement. These rulings were recorded in the task brief and task plan before the corresponding contract changes.

The reused public runtime seam is:
`await service.enable_reference(session_id, source=ref, target_incarnation_id=..., expected_revision=..., expected_source_fences=(scope, backend, db), is_current=...)`.
Direct callers may omit the transient fences to start a fresh immediate operation; typed Library handoffs require them. Identity fields are never inferred from a legacy ordinary Prompt target.

## TDD evidence

Initial RED:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompt_mixins.py -q --basetemp=/private/tmp/task7-red -p no:cacheprovider`
Failed collection with `ModuleNotFoundError: ...prompt_mixin_editor`, as expected before implementation. Initial editor/filter/state GREEN: 3 passed.

Final exact-source RED:
`...python -m pytest Tests/UI/test_library_prompt_mixins.py -k same_authority_replacement -q --basetemp=/private/tmp/task7-source-red -p no:cacheprovider --tb=short --show-capture=no`
Three failures: unexpected `expected_source_fences` keyword before implementing the missing contract (backend, DB, wrapper replacements). Final tests verify refusal before any read call, plus identity-preserving pending copies.

## Verification commands and results

All commands use the shared interpreter above, private basetemp, and `-p no:cacheprovider`; no full suite was run.

1. `pytest Tests/UI/test_library_prompts_reader.py Tests/UI/test_library_prompt_mixins.py Tests/Library/test_library_prompts_state.py Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py -q --basetemp=/private/tmp/task7-green2 --tb=short --show-capture=no`
   **372 passed**. This broader targeted run covered normal Basic/Advanced/Info, F6, dirty navigation, retained readers, detail generations and ordinary one-shot storage. The planned reader test module now uses the established `bootstrap_profile` marker because all its nodes mount the real guarded app factory. Original unmarked admission failure was reproduced on exact BASE rather than bypassed.
2. `pytest Tests/UI/test_library_prompt_mixins.py Tests/UI/test_library_prompts_reader.py::test_mixin_enable_refuses_absent_target_and_replaced_source Tests/Library/test_library_prompts_state.py::test_literal_mixin_save_has_exact_projection_and_shared_source_limits Tests/Library/test_library_prompts_state.py::test_mixin_type_filter_empty_result_is_no_matches_and_scope_restore_keeps_type Tests/UI/test_library_prompts_canvas.py -k 'mixin or insert_console or dialog_rechecks_projection' -q --basetemp=/private/tmp/task7-confirm-final --tb=short --show-capture=no`
   **29 passed, 335 deselected**. Includes real Library save/conflict/reload with exact kept name/body/placement; actual Escape and Cancel; strict blank validation; real mounted Console enable; absent target; positive staged handoff; source switch; malformed/future records; ordinary Prompt/system-only/variable dialog regressions.
3. `pytest Tests/UI/test_library_prompt_mixins.py::test_real_console_mixin_consumer_enables_without_touching_composer_or_system Tests/UI/test_console_prompts_controller.py::test_library_prompt_insert_handoff_appends_onto_the_live_draft Tests/UI/test_console_prompts_controller.py::test_library_prompt_insert_handoff_is_blocked_before_setup_completes Tests/UI/test_console_prompts_controller.py::test_prompts_controller_is_wired_in_chat_screen_init -q --basetemp=/private/tmp/task7-wiring-final --tb=short --show-capture=no`
   **4 passed**, proving the named accessor reaches only the Prompt controller and ordinary handoffs remain intact.
4. `pytest Tests/UI/test_library_prompts_reader.py Tests/UI/test_console_prompts_controller.py Tests/Prompt_Management/test_prompt_mixins.py Tests/Prompt_Management/test_prompt_chatbook_record.py Tests/Prompt_Management/test_prompt_chatbook_scope_service.py -q --basetemp=/private/tmp/task7-regressions --tb=short --show-capture=no`
   Original qualification: **157 passed, 23 failed**. Twenty reader failures were the original config-admission issue addressed by the legitimate module marker and green reader rerun. Three unrelated existing Console improvement nodes fail with `NoMatches(#console-left-rail)`; see exact-BASE qualifier below. All selected source/Markdown/Chatbook paths passed.
5. Final added-source/fast-create ordering confirmation is recorded below. The source-final set already passed all three source replacement cases and the real held consumer; its first run had **153 passed, 1 failed** in fast Cancel→Create arming, which was investigated and fixed through the existing canvas post-recompose callback (not a sleep or dropped assertion).

## Exact BASE qualifiers (not green)

Archived BASE `6f5bfae0c191909bb676dae97e5071d6b4f56d61` source/tests to `/private/tmp/task7-baseline-evidence` and ran the original reader admission node plus these exact three Console nodes using the same interpreter/harness:
- `test_improvement_target_is_pinned_once_and_shared_across_callbacks`
- `test_improvement_activation_refuses_a_moved_system_prompt`
- `test_improvement_snapshot_refuses_an_unpinned_provider_target`

BASE output `/private/tmp/task7-baseline-original.log`: **4 failed**, original reader `raw_source_selection_changed` plus three `NoMatches(#console-left-rail)`. Current exact-three rerun `/private/tmp/task7-current-baseline-check.log`: **3 failed** with the same NoMatches signature. Those are inherited qualification failures, not reported as green and not repaired in this slice.

## Static checks

Whole-file union, not changed-line lint: `/private/tmp/task7-lint.py` archives every touched existing Python file at exact BASE and runs Ruff over complete BASE/current files with the repository pyproject, then compares full-file `(path, code, message)` diagnostic multisets. Current union includes wiring, whose inherited diagnostics increase the earlier 266-file-set total to **269 BASE and 269 current; zero introduced**. Artifacts: `/private/tmp/task7-lint-base.json`, `/private/tmp/task7-lint-current.json`, `/private/tmp/task7-python-files.json`.

New widget and test file independently pass `ruff check --no-cache` and `ruff format --check --no-cache`. `git diff --check` passes. No token/CSS build checks were needed because no stylesheet/token values changed.

## Visual evidence

Native Textual real-PromptsDatabase / production stylesheet harness, legitimate bootstrap profile; headless Pilot captures, not a claim of an OS-attached terminal/provider session. One batched inspection of six captures and one combined confirmation inspected Name/Instructions/Placement, exact literal copy, focus, dirty save/discard actions, narrow vertical scrolling and adaptive pane collapse/restoration at **80x24, 120x40, 170x48**. Placement is keyboard reachable at 80x24 while actions remain fixed. Incumbent tokens, theme, layout and native control glyphs were retained.

Artifacts are under `/private/tmp/task7-captures/`:
`mixin-{80x24,120x40,170x48}-{clean,dirty}.svg` and matching PNGs;
`contact-sheet.png`, `contact-sheet-confirm.png`;
`source-manifest.json` records per-file SHA-256 source/capture identities from the visual confirmation (manifest digest `f5c5bf08374604b28d855c5f3191d3685c58652cdd26910641f1286dc587be0c`). The subsequent transient source-fence fix affects routing/runtime only; final capture rerun/source manifest details are appended below. Task8/9 still own their requested **160x45 / 100x30 / 80x24** broader native qualification gates.

## Files and self-review

New widget/test: `Widgets/Prompts/prompt_mixin_editor.py`, `Tests/UI/test_library_prompt_mixins.py`.
Existing owners: Library display/screen state; Prompt browse/controller/canvas; Library screen thin event/focus/complete-scope projections; pending handoff and screen-state target stores; Console prompts/wiring; runtime mix-in service; Library state/reader and screen-state tests. Work pane inherits the existing canvas and required no direct change. Docs: Library guide/architecture, observed Select constructor-value lifecycle lesson, TASK-38542 plan/AC/implementation notes.

Self-review checked every changed constructor/caller, ordinary Prompt positive type narrowing, latest-slot supersession, exact session/incarnation/revision/source objects, pre/post-await authority, literal save/export custody, invalid raw drafts, conditional conflicts, source switch and no instruction body/credential/path in request fields or diagnostics. The final exact-origin correction was escalated to the controller and approved before implementation. No helper reviewers/subagents, new timer/save engine, Workbench expansion, branch/push/merge, stash/reset/clean or unrelated cleanup.

## Remaining concerns

Three exact-BASE Console improvement harness failures remain as qualified above. Full suite, external provider/server calls and OS-attached terminal verification were not run. Server revision-qualified mix-in reads remain intentionally unsupported. Task is left In Progress for controller review.

## Final material corrections and targeted control

The final request now retains exactly three repr-hidden source objects `(scope, backend, DB)` and passes them to runtime admission. Three RED replacement tests failed on the absent keyword before implementation. The real mounted Console test additionally disables a selected entry, stages the original request, then verifies same-path native DB reopen and a new scope wrapper cannot re-enable it; a held real source read followed by a wrapper replacement also leaves the entry disabled and composer/System unchanged.

The observed Cancel→Create race used a whole-screen `call_after_refresh` arm while independent canvas recomposition could still be pending. The surrounding detail owner already documents and uses the canonical canvas post-recompose `then` seam. New mix-in creation now uses that same seam. The regression holds Cancel's older source snapshot, physically clicks the visibly enabled Create action, releases the old snapshot and checks a fully mounted armed literal editor before strict validation/save. No sleep or assertion removal.

`...python -m pytest Tests/UI/test_library_prompt_mixins.py::test_library_mixin_create_and_filtered_pagination_totals Tests/UI/test_library_prompt_mixins.py::test_real_console_mixin_consumer_enables_without_touching_composer_or_system Tests/UI/test_library_prompt_mixins.py::test_enable_refuses_same_authority_replacement_before_consumption -q --basetemp=/private/tmp/task7-final-race -p no:cacheprovider --tb=short --show-capture=no` → **5 passed**. Log: `/private/tmp/task7-final-race.log`.

## Final qualification and artifact identity

The final Create busy state is patched synchronously before Items recomposition. Existing controls stay disabled through read completion; the existing `PostRecomposeCallback` enables only final mounted children. The deterministic test holds both Cancel's browse read and older source snapshot, checks Loading/disabled state, releases the read, verifies the old button remains disabled, then clicks a distinct mounted enabled replacement and releases the older snapshot.

`...python -m pytest Tests/UI/test_library_prompt_mixins.py::test_library_mixin_create_and_filtered_pagination_totals -q --basetemp=/private/tmp/task7-ordering-boundary -p no:cacheprovider --tb=short --show-capture=no` → **1 passed**.

Final covering command:
`...python -m pytest Tests/UI/test_library_prompt_mixins.py Tests/UI/test_library_prompts_reader.py::test_mixin_enable_refuses_absent_target_and_replaced_source Tests/UI/test_library_prompts_reader.py::test_prompts_global_f6_reaches_permanent_work_region Tests/UI/test_library_prompts_reader.py::test_basic_save_preserves_advanced_only_prompt_fields Tests/UI/test_library_prompts_reader.py::test_detail_failure_keeps_prior_prompt_locked_and_retry_loads_selection Tests/UI/test_library_prompts_canvas.py -k 'mixin or insert_console or dialog_rechecks_projection or prompts_global_f6_reaches_permanent_work_region or basic_save_preserves_advanced_only_prompt_fields or detail_failure_keeps_prior_prompt_locked_and_retry_loads_selection' -q --basetemp=/private/tmp/task7-final-qualified -p no:cacheprovider --tb=short --show-capture=no` → **33 passed, 335 deselected in 37.61s**.

Final log: `/private/tmp/task7-final-qualified.log`. This final run includes literal mode-change guards, strict save/cancel, final busy ordering, source fences, actual held consumer, ordinary F6/save/detail recovery and ordinary Prompt handoffs. Final whole-file lint remains **269 BASE / 269 current, zero introduced**; Ruff new-file checks/formatter and `git diff --check` pass.

Final six SVG captures and matching PNGs were regenerated by that exact final run. Final contact sheet: `/private/tmp/task7-captures/contact-sheet-final.png`. Exact final source/capture SHA-256 manifest: `/private/tmp/task7-captures/source-manifest-final.json`, digest **b895b6b39277685c6930cdcad702eaa405692e36c0b478e983d7735f7502e668**. Initial batched visual inspection and one confirmation remain recorded above; subsequent changes concern transient source identity, supported literal mode guards and the approved Create busy ordering, with no token/style redesign. Reviewers can inspect the final packet directly.

## Local commit and handoff

Commit **398e79d50b6a3232d63ae37a8c925ca0e765e36f** — `feat: author and enable library prompt mix-ins`. Twenty scoped paths committed; working tree clean. TASK-38542 remains **In Progress** for controller review. Final capture manifest source hashes were verified against the committed files; `/private/tmp/task7-captures/commit.txt` records this commit. No further tests or product edits followed the final qualification.

## Review fix round 1 — invalid copy actions

FIX_BASE: **398e79d50b6a3232d63ae37a8c925ca0e765e36f**. Reviewer important finding reproduced in real DB / production-CSS UI at both named boundaries. Strict serialization is unchanged; `Duplicate` and conflict `Save as new` now catch validation before constructing/detaching a copy. A shared feedback helper uses the ordinary Save Instructions message and body focus, and updates the existing conflict text so that validation is visible without changing conflict/recovery ownership. Raw name/body/placement, metadata, dirty/conflict state, source identity and version remain unchanged on rejection.

The required valid recovery test exposed the obsolete raw parent name overriding Duplicate's generated `(copy)` name after mount. Once strict validation succeeds and the exact detached copy has been fully constructed and adopted by the existing owner, Duplicate clears that obsolete raw draft. The canonical detached definition/name/metadata remain available for recovery/save; invalid attempts never reach detachment. No source fence, pending handoff, strict model, stylesheet, timer or provider changes.

Added parametrized real-widget-message regression `test_invalid_mixin_copy_action_keeps_raw_draft_then_recovers[save_as_new|duplicate]`. Duplicate covers an already queued action when the child's invalid change hides More actions. Both reject whitespace-only Instructions without a message-pump exception, retain the exact live raw fields/conflict/source/version and unchanged DB row, then accept corrected literal text. Recovery saves a new schema-v3 text_mixin with exact body/placement, correct generated name for Duplicate, local source identity and author/description/keywords.

RED command:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompt_mixins.py -k invalid_mixin_copy_action -q --basetemp=/private/tmp/task7-fix1-red -p no:cacheprovider --tb=short --show-capture=no`
Output: **2 failed, 20 deselected** with `ValueError: Mix-in body must be a non-empty string.` escaping both action handlers through Textual's message pump. Log: `/private/tmp/task7-fix1-red.log`.

First boundary-fix run: **1 passed, 1 failed**; valid Duplicate mounted `Recovered mix-in` instead of `Recovered mix-in (copy)`, identifying the stale raw-name custody defect. Log: `/private/tmp/task7-fix1-green-attempt.log`.

Final GREEN command:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompt_mixins.py -k 'invalid_mixin_copy_action or save_conflict_export_and_dirty_escape or create_and_filtered_pagination_totals' -q --basetemp=/private/tmp/task7-fix1-green -p no:cacheprovider --tb=short --show-capture=no`
Output: **4 passed, 18 deselected in 16.10s**. Log: `/private/tmp/task7-fix1-green.log`. This is the bounded affected copy/save/conflict/create selection; no 372/33-node rerun.

Full-file FIX_BASE lint comparison: `/private/tmp/task7-fix1-lint.py` reads exact FIX_BASE versions of both complete touched Python files, runs Ruff with the repository config on complete FIX_BASE/current files and compares `(path, code, message)` diagnostic multisets: **17 BASE / 17 current, zero introduced**. JSON evidence: `/private/tmp/task7-fix1-lint-base.json`, `/private/tmp/task7-fix1-lint-current.json`. New tests independently pass Ruff and formatter; `git diff --check` passes.

Files: `UI/Library_Modules/library_prompts_controller.py`, `Tests/UI/test_library_prompt_mixins.py`, TASK-38542 plan/notes. ADR required: no, routine validation-boundary fix under ADR-286. Task remains In Progress for independent controller review. No restyle or new capture was required; functional feedback is asserted in the existing status/conflict Static. Previously qualified full-suite/provider/OS and exact-BASE Console limitations are unchanged.

Fix-round-1 commit: **f0e1d640c4ae20c6975b5fefe66587dec3694b84** — `fix: preserve invalid mix-in copy drafts`. Three scoped files committed. Git reported an existing auto-GC/unreachable-object maintenance warning; commit succeeded and no cleanup/prune action was taken. Task remains In Progress.
