# TASK-37031 implementation report

Status: implementation ready for independent review/CI with disclosed native legacy/governance concerns. Task remains In Progress; all four acceptance criteria have qualified implementation evidence. Task remains In Progress for root review/CI.

## Scope and decision

Preserved later accepted word-bench input across queued successful Save publication. `BenchEditor.Saved` now carries the originating editor and raw form snapshot. The existing serialized region swap checks origin, selection and raw form immediately before detail replacement, after the awaited Catalog update. Catalog and Inspector still reflect persisted data. Save refresh shares the current selection revision, so it cannot supersede a pending user selection body worker. Ordinary unchanged Save reloads cleaned persisted fields and retains existing mini-form discard semantics. The successful submission advances the editor dirty baseline. Validation, write failures, Revert and run completion guards retain their paths.

ADR required: no. ADR path: N/A. Reason: bounded repair of existing ephemeral editor/screen owner and service contracts; no storage/runtime/application structure change. Existing governing paths:

- backlog/decisions/011-chatbook-workbench-ui-system.md
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/150-design-token-system-and-design-language.md
- backlog/decisions/161-component-pattern-library.md

Read relevant Evals Architecture, design-language/component-pattern guidance, testing/live-verification lessons, TDD/writing-good-tests, systematic-debugging, verification-before-completion and Ponytail. No new dependency, boot import, global state, stylesheet or token was added.

## Environment and commands

All commands ran with explicit cwd `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui` using native scoped escalation, exact primary venv Python and the shared private TEMP/profile runner. No full suite was run. Git always used both `-c safe.directory=C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui` and `-c gc.auto=0`. Primary checkout untouched. Before any product/test edit, the task CLI set TASK-37031 In Progress, assigned @codex and saved the Implementation Plan and exact existing ADR paths.

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task1-red04 Tests/UI/test_evals_bench_save_publication.py
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task1-departure-red02 Tests/UI/test_evals_bench_save_publication.py -k departure-pending
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task1-green03 Tests/UI/test_evals_bench_save_publication.py
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task1-size02 'Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/evals_screen.py]' 'Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/UI/Screens/evals_screen.py]'
```

The corrected task1-red04 runner ran against the exact `905d474e1353e2dce7786cc4c682f858d24eee2b` Git bytes of the two owned product modules, supplied by `git show BASE:path`; a native Python `try/finally` restored the candidate bytes after subprocess completion. No other source was changed. Earlier task1-red01 already reproduced intended Saved vs Saved-new and departed selection failures, but task1-red04 is the final qualified original-source receipt.

## TDD and actual child evidence

- task1-red04: parent 4 cases: 1 passed, 3 intended failures, 0 errors/skips. Four actual children, each exactly 1 case: ordinary cleaned whitespace Save passes; held-parent later input and rail-await later input both fail `Saved` vs `Saved-new`; departure fails `bench` vs `dataset`.
- task1-departure-red02: parent and actual child each 1 intended failure, 0 errors/skips: current selection is dataset but old BenchEditor remains mounted because the first candidate superseded its pending selection worker.
- task1-green03: parent 5 passed, 0 failures/errors/skips; five actual child reports each exactly 1 passed, 0 failures/errors/skips. Cases: ordinary cleaned Save, held-parent later input, input during awaited Catalog refresh, settled departure, pending departure. Both later-input cases assert DB Saved, live Saved-new, same live child, focus, caret 9, exact native selection (9,9), dirty state and explicit subsequent Save persisting Saved-new.
- task1-size02: 2 passed, 0 failures/errors/skips, no private children. Evals source 4212 lines; only its own row lowered from4236 to4212 per root authorization.

Receipts are retained under `.test-tmp/textual-37031-37034/<label>/verification.json`, outer `result.xml`, and actual child `pytest/test_save_publication_retains_<n>/private-profile-process/pytest.xml` with raw child `pytest.log`. The final five children have been inspected directly; wrapper counts are not substituted for their outcomes. Workers drain and all explicit gates/locks are released in finally blocks. Runtime imports assert source identities within this worktree.

## Excluded attempts and setup corrections

- task1-red02/red03 and task1-green01 later-input cases used nonprintable selection Key events posted directly to Input while its parent was held. Those events require bubbled App binding dispatch; their selection predicate timed out. They are harness setup exclusions, not behavioral RED/GREEN evidence. The corrected witness uses actual printable focused-child input and asserts its natural exact insertion selection (9,9); no timeout was raised or assertion weakened at the product boundary.
- task1-departure-red01 selected no case (exit5) because the fifth parameter was not yet present; excluded, superseded by red02.
- task1-green02: 4 new private children passed, but102 legacy setup errors were raw_source_selection_changed at Tests/UI/conftest.py193, before the legacy bodies. task1-bench01:34 same setup errors. These are excluded from legacy behavior qualification. Root prescribed the existing CI collection-time App import prefix, with no marks/plugins/fixture bypasses.
- Plan expected-output correction: the original brief expected the departure control to qualify; the original product also unconditionally yanked departed selection, so that leg is an intended RED, approved by root. The additional awaited-rail and pending-departure witnesses tighten the same Save applicability requirement; no broader scope/ADR was needed.

## Static checks and source qualification

Final focused Ruff check and format check pass across all six touched Python files; changed Python compiles and git diff --check passes. Exact commands use the primary venv Python with `-X utf8 -m ruff check` / `-m ruff format --check` and all six paths listed in the source table. Original baseline debt is preserved in the earlier logs, then paid down only as explicitly authorized by root: unused SubjectError import in the screen, unused App import in the now-owned legacy screen tests, the cited single size-guard formatting drift, and Ruff's four/eight assertion layouts in the two legacy test modules. No reexport caller for either unused import was found. The Evals size row was lowered to the final4211 lines; no row rose.

All123 functions in test_evals_screen.py and107 functions in test_evals_bench_editor.py have identical AST before/after formatting and against exact base. Hashes and assertions are preserved in legacy-normalization.json. Product AST comparison preserves25 other bench-editor methods and76 other screen methods; changed definitions remain limited to Saved constructor/publication state/Save baseline/docs, select/region swap/Saved handler. The two legacy modules received only the explicitly authorized bootstrap_profile marker/comment, unused App removal and mechanical assertion formatting; their DB fixtures and behavior assertions are unchanged. Searches found no deliberate profile/env/source selection in either module body.

Static artifacts: `.test-tmp/textual-37031-37034/task1-static/{baseline.json,ast-preservation.json,legacy-normalization.json,final-lint.log,final-format.log,size-ratchet-baseline-format.log,guard-baseline-identity.json,final-source-hashes.json,source-hashes-before-static-cleanup.json,attempt-ledger.json}`. Guard/CSS source content is unchanged in Git from exact base; byte hashes explicitly distinguish native CRLF checkout bytes from Git LF bytes.

## Final verification after bounded static cleanup

`task1-green04` reran all five regression cases on final product import bytes:5parent PASS, five actual private children each1PASS, all0failures/errors/skips. `task1-final-controls01` reran exact legacy screen mount and ordinary Save on final bytes:2PASS/0failures/errors/skips. `task1-import-size02` passed4cases/0failures/errors/skips: final4211-line size ceiling/slack and both import guards. Final preimport558/558 modules,424356/425347 LOC, library130690/135111 LOC; first-send93/94 modules. Final lint/format check passes all6touched Python files. Earlier behavior-heavy runs are source-qualified only across the removed unused imports and AST-identical formatting, without repeated broad modules solely for whitespace.

`task1-legacy-control02` first proved the authorized source-bound bootstrap marker alone qualifies the original screen mount and Save nodes:2PASS, no App-prefix selector. This uses the existing Tests/conftest private bootstrap selection and per-test config reset. `task1-screen04` then runs the requested full Evals screen module plus five new private cases; `task1-bench02` runs only the34existing bench Save/validation/write-failure/Revert/dirty/prompt-mode cases. Bench result: 31 PASS / 3 FAIL / 0 errors / 0 skips. Screen result: 97 PASS / 2 FAIL / 0 errors / 0 skips across 99 cases (94 legacy plus five passing actual-child wrappers). Exact failure qualification is recorded below. Further app-booting runs are serial, and no temporary original-source restoration overlaps an active source-inspecting process.

## Governance and legacy qualification results

`task1-guards01` completed: parent30 cases,25 passed/5 failed/0 errors/0 skips;26 actual private children,21 passed/5 failed/0 errors/0 skips. All five failures are inherited native Windows path assumptions, with unchanged guard and CSS source inputs confirmed against the exact base in `guard-baseline-identity.json` (checkout CRLF/Git LF explicitly accounted for):

- `Tests/UI/test_design_token_governance.py::test_active_css_extension_is_in_hex_floor`: expected regex `components/active.css`, actual native `components\active.css`.
- `Tests/UI/test_component_pattern_governance.py::test_canonical_classes_defined_only_in_owning_sheet`: native backslash owner paths do not equal POSIX registry paths.
- `Tests/UI/test_component_pattern_governance.py::test_dimension_literal_ratchet`: native `core\_variables.tcss` misses the token-file exemption.
- `Tests/UI/test_component_pattern_governance.py::test_active_css_extension_is_in_dimension_floor`: expected POSIX source key, native backslash source key present.
- `Tests/UI/test_component_pattern_governance.py::test_pin_rejects_active_css_extension_without_writing_baseline`: expected POSIX text, native backslash diagnostic.

Both import-budget controls pass unchanged: preimport558/558 modules,424357/425347 LOC, fattest library route130690/135111 LOC; first-send93/94 modules. These have zero module headroom/one module headroom respectively; no budgets were raised. Governance receipt files and all actual child XML/logs remain retained under task1-guards01.

Root's exact legacy setup probe used:

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task1-legacy-control01 Tests/UI/test_app_query_one_fallback.py::test_query_one_reraises_screen_access_error Tests/UI/test_evals_screen.py::test_screen_mounts_the_three_pane_workbench
```

Result:1 passed control/1 failed legacy body/0 errors/0 skips, no private children. The collection-time App import eliminates late import setup errors, but source-bound runtime construction still refuses the per-test profile redirect: EvalsHarness.on_mount159 -> EvalsScreen.__init__266 -> LabScreen.__init__137 -> lab_rail_store.load_rail_layout51 -> get_cli_setting9874 -> raw_participants.py307 `RecoveryRequired(raw_source_selection_changed)`. The initial NoMatches for #lab-workbench on _default is secondary to that startup exception. At that stage root was notified and no legacy expansion, mark, fixture bypass or admission weakening was applied. The later explicit ruling authorized the two existing bootstrap_profile module markers described above; no fixture/admission implementation was changed.

## Final source byte SHA256

| File | Bytes | Lines | SHA256 |
|---|---:|---:|---|
| tldw_chatbook/UI/Evals/bench_editor.py | 82927 | 1557 | c9241a4972a508d522f4aee7e1438233ffd962b1165bd495ffe4c6550c2364da |
| tldw_chatbook/UI/Screens/evals_screen.py | 216542 | 4211 | 87c89b073c1f42b52d371d4f1ddd4c5a0b995fb0636d3dec9e75e8287dbdb259 |
| Tests/UI/test_evals_bench_save_publication.py | 7754 | 191 | d21868ee87ad5fd9d8d961f3e983be5a1d98f6c6215dd8b8d6b776462ecbf70c |
| Tests/UI/test_evals_screen.py | 185470 | 4030 | 34d690c01476e992ad212d1c1527ae56bc7cb5513b04fc02220ab8e603a46085 |
| Tests/UI/test_evals_bench_editor.py | 151347 | 3417 | 066892381d3061fa12b55cdc8310f12b9896be1252cf35d41c387f578fba4ba6 |
| Tests/Architecture/test_module_size_ratchet.py | 42058 | 647 | 783f694e9a238eeb70c805fcca50e423ecc8cb140af5f1c311311e440dc17b2a |
| Docs/Architecture/evals.md | 13149 | 133 | e979cf715e881a4484ce56a5c9915c2dc770116fd2062c406bdae91403c76d17 |

## Self-review

Traced every Saved caller and current region replacement owner. Raw snapshots retain invalid/whitespace drafts instead of parsing them as already persisted; originating editor identity prevents an old completion replacing a newly selected instance of the same bench. Checking after rail replacement protects the await interval. Sharing selection revision preserves pending user body publication; existing serialization and dirty-row accumulation remain the only mutation owner. Ordinary Save cleanup, Catalog/Inspector persisted labels, dirty state and a second explicit Save are asserted. No new architecture boundary or deliberate performance/security shortcut. No new general lesson file was needed; the encountered key-routing/profile-order traps are already recorded in the required lessons.

## Final legacy failure qualification and concerns

All source-inspecting test processes exited before baseline restoration. A single serial original-product-source run, task1-legacy-baseline01, retained only the authorized bootstrap markers and AST-identical legacy tests, temporarily installed the exact two product files from905d474e, ran only the five failed legacy nodes, and restored final bytes in finally. legacy-baseline-source-proof.json records exact baseline/final hashes and successful restoration. The same five controls then ran serially on restored final bytes in task1-legacy-final01. Both runs have exactly3PASS/2FAIL/0errors/0skips and no private children:

| Node | Broad current module result | Exact original / final result | Qualification |
|---|---|---|---|
| test_evals_bench_editor.py::test_add_from_picker_stages_a_row_and_save_persists_it | FAIL: saved target_ids lacks the added target | FAIL / FAIL, same missing target | Baseline-proven native legacy failure; staged row is visible before the old unchecked geometry Save click. No unrelated fix. |
| test_evals_bench_editor.py::test_is_dirty_flips_true_on_a_staged_target_add | FAIL: initial SelectCurrent #label NoMatches | FAIL / FAIL, same Select mount exception | Baseline-proven native initial-mount failure before Save publication; no unrelated fix. |
| test_evals_bench_editor.py::test_save_with_capture_continuations_toggled_on_via_the_ui_persists_the_flip | FAIL: stored flag False | PASS / PASS | Non-reproducing broad-run failure; not claimed as baseline-proven inherited. Assertions/functions unchanged. |
| test_evals_screen.py::test_pressing_delete_then_confirming_in_the_real_dialog_deletes_the_bench | FAIL: actual screen is EvalsScreen, not ConfirmationDialog | PASS / PASS | Non-reproducing broad-run click/modal failure; not claimed as a proven fixture-only cause. Test/production Delete function AST unchanged. |
| test_evals_screen.py::test_readiness_rows_with_several_continuations_paint_inside_the_inspector_viewport | FAIL: no #evals-inspector-target-0 | PASS / PASS | Non-reproducing broad-run readiness/viewport failure; not claimed as baseline-proven inherited. Test/readiness production functions unchanged. |

Critical existing AC3 controls passed in the scoped modules: invalid Top-K, blank name, taken-name conflict, deleted-row write failure, explicit Revert, dirty bench run completion, dirty mini-form run completion, and dirty sample-bench completion. Ordinary cleanup and explicit subsequent Save also pass in final core and legacy controls. Parent/actual-child XML receipts are kept separate in task1-static/qualified-result-summary.json. Final precommit-lint.log and precommit-format.log also confirm clean static checks after source restoration.

Five inherited Windows CSS/path governance failures remain separately named above; they are not part of the two persistent legacy failures. Existing Pydantic Field and pytest-asyncio fixture-loop-scope deprecations remain; optional Sounddevice/WebRTC/HuggingFace availability messages appear in setup. Output is not pristine. No full suite, budget increase, skip/xfail, timeout increase or guard bypass was used. Further app-booting runs were serial after root's process-custody ruling. Root owns current-head Linux CI, independent review, PR/integration and final task completion. The root-owned scripts/ui_pr_gate_census.txt integration edit is deliberately unstaged and outside this task's diff.

## Commits

`053a7900cea31ea8e04a26dd6cca6c38bd07e448` - fix(evals): preserve newer drafts after queued bench save. Exactly eight Task1-owned files committed; root-owned census remains unstaged. Task remains In Progress with all four ACs checked; independent review/current-head hosted CI remain root gates.

## Exact executed pytest command arrays

Each command.json below is the complete native pytest subprocess invocation, including unchanged timeout, plugins, private basetemp, XML and cache paths. The common runner selected native TEMP/profile before import.

### task1-red04

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_bench_save_publication.py -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-red04\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-red04\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-red04\result.xml
```

### task1-departure-red02

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_bench_save_publication.py -k departure-pending -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-departure-red02\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-departure-red02\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-departure-red02\result.xml
```

### task1-green04

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_bench_save_publication.py -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-green04\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-green04\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-green04\result.xml
```

### task1-screen04

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_bench_save_publication.py Tests/UI/test_evals_screen.py -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-screen04\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-screen04\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-screen04\result.xml
```

### task1-bench02

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_bench_editor.py -k "save or saving or revert or dirty or failure or renaming or prompt_mode" -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-bench02\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-bench02\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-bench02\result.xml
```

### task1-guards01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/evals_screen.py] Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/UI/Screens/evals_screen.py] Tests/UI/test_design_token_governance.py Tests/UI/test_component_pattern_governance.py Tests/Performance/test_screen_preimport_payload_budget.py -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-guards01\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-guards01\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-guards01\result.xml
```

### task1-final-controls01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_screen.py::test_screen_mounts_the_three_pane_workbench Tests/UI/test_evals_bench_editor.py::test_save_persists_every_field_and_reselects_the_bench -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-final-controls01\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-final-controls01\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-final-controls01\result.xml
```

### task1-import-size02

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/evals_screen.py] Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/UI/Screens/evals_screen.py] Tests/Performance/test_screen_preimport_payload_budget.py -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-import-size02\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-import-size02\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-import-size02\result.xml
```

### task1-legacy-baseline01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_bench_editor.py::test_add_from_picker_stages_a_row_and_save_persists_it Tests/UI/test_evals_bench_editor.py::test_is_dirty_flips_true_on_a_staged_target_add Tests/UI/test_evals_bench_editor.py::test_save_with_capture_continuations_toggled_on_via_the_ui_persists_the_flip Tests/UI/test_evals_screen.py::test_pressing_delete_then_confirming_in_the_real_dialog_deletes_the_bench Tests/UI/test_evals_screen.py::test_readiness_rows_with_several_continuations_paint_inside_the_inspector_viewport -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-legacy-baseline01\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-legacy-baseline01\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-legacy-baseline01\result.xml
```

### task1-legacy-final01

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 -m pytest Tests/UI/test_evals_bench_editor.py::test_add_from_picker_stages_a_row_and_save_persists_it Tests/UI/test_evals_bench_editor.py::test_is_dirty_flips_true_on_a_staged_target_add Tests/UI/test_evals_bench_editor.py::test_save_with_capture_continuations_toggled_on_via_the_ui_persists_the_flip Tests/UI/test_evals_screen.py::test_pressing_delete_then_confirming_in_the_real_dialog_deletes_the_bench Tests/UI/test_evals_screen.py::test_readiness_rows_with_several_continuations_paint_inside_the_inspector_viewport -p pytest_asyncio.plugin -p pytest_timeout -q --tb=short --timeout=180 -o cache_dir=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-legacy-final01\cache --basetemp=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-legacy-final01\pytest --junitxml=C:\Users\GDesktop-1\.codex\worktrees\textual-settings-patterns\tldw_tui\.test-tmp\textual-37031-37034\task1-legacy-final01\result.xml
```


## Approved review documentation minor

Corrected only Character Save handler docstring wording: it unconditionally re-selects its character bench and reloads persisted fields, separately from the now-conditional word-bench path. No Character behavior, dependency, CI census or task status changed. Physical source line count remains4211, so its size row is unchanged. Focused Ruff check/format and git diff --check pass; the whole product module AST is identical after removing docstrings. No app tests were repeated for this copy-only correction. Evidence: task1-static/review-doc-minor.json.

Before SHA256: 87c89b073c1f42b52d371d4f1ddd4c5a0b995fb0636d3dec9e75e8287dbdb259

After SHA256: 71f910ebe96bbbeec8baa0b8ab57661afdde0400188d7f424b05083d168f4cb0

Commit: `a7af77a99aab2da5132fbcbbedbcfbcdeacfd6cc` - docs(evals): clarify character save publication.

Task remains In Progress for controller current-head CI; broader legacy module clearance remains unclaimed. Existing environmental warning noise and native path/legacy failure classifications remain disclosed; no dependency upgrade or unrelated repair was made.
