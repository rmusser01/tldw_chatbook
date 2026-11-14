# TASK-37033 implementation report

Status: implementation and scoped local verification complete; task remains In Progress for coordinator independent review/CI. Initial owned-file commit: `6d6ba5b29f52a449c37abaedc114bcae081842c0`. Current I1 review-fix commit: `c375f164e6955c68c90c3d5e75a6c936023dacbc`.

## Scope and decisions

- Only the existing Prompt export function in `tldw_chatbook/UI/Library_Modules/library_file_export.py` changes product behavior. FileSave and Replace callbacks submit a native screen worker synchronously; only filesystem writing runs through `asyncio.to_thread`. Captured Markdown/destination and the existing Prompt-bound reporter remain operation-owned. Independent exports are non-exclusive so later exports cannot cancel earlier captured writes.
- No generic export framework, helper-seam behavior change, dependency, boot import, stylesheet, token, source cap, profile-admission bypass, or Notes/Report behavior change.
- Root explicitly authorized adapting the three direct Prompt helper controls in `Tests/Library/test_library_file_export.py` to actual native App worker ownership, and the narrow legacy-module bootstrap opt-in after its observed guarded source-selection failure. Root also authorized preserving malformed structured render failure/retry feedback in those same owned files.
- ADR required: no. ADR path: N/A. This repairs existing UI/service ownership under `033-application-session-state-ownership.md`, `036-application-service-composition-lifecycle.md`, `031-tui-keybinding-and-footer-hint-conventions.md`, `150-design-token-system-and-design-language.md`, and `161-component-pattern-library.md`.

## Reproduction and TDD

Explicit cwd: `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui`; Python: `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8`.

The committed audit probe `Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251/probes/library-research/test_export_observation.py.txt` was promoted into real SQLite/local-service/More-actions/FileSave tests. The first witness retained the audit's independent 0.5s observation window; its later failure is retained below and is not a passing certificate. The final minimal witness instead makes the actual App key handler acknowledge and release the held writer, with the existing 3s writer bound only as a watchdog. Logical event order proves key acknowledgement before physical write completion. The watchdog still lets incorrect original source finish its file/row checks and fail specifically on dispatch ordering. AC1 contains no 0.5s SLA; this is controlled responsiveness evidence, not a latency distribution or timeout increase.

Commands (prepend the explicit Python above):

```text
.test-tmp/textual-37031-37034/run-targeted.py task3-red01 Tests/UI/test_library_prompt_export_responsiveness.py
.test-tmp/textual-37031-37034/run-targeted.py task3-red02 Tests/UI/test_library_prompt_export_responsiveness.py
.test-tmp/textual-37031-37034/run-targeted.py task3-green01 Tests/UI/test_library_prompt_export_responsiveness.py Tests/UI/test_library_prompt_export_journeys.py::test_export_result_after_navigation_preserves_current_status[other-screen]
.test-tmp/textual-37031-37034/run-targeted.py task3-green02 Tests/UI/test_library_prompt_export_responsiveness.py Tests/Library/test_library_file_export.py
.test-tmp/textual-37031-37034/run-targeted.py task3-green03 Tests/UI/test_library_prompt_export_responsiveness.py Tests/UI/test_library_prompt_export_journeys.py Tests/UI/test_library_export_replace_confirm.py
```

The original-source `task3-baseline01` exact thirteen selectors and byte-restoration record are in `.test-tmp/textual-37031-37034/task3-source-qualification.json`; its full pytest argv is `task3-baseline01/command.json`.

- `task3-red01` is excluded setup evidence: omitted `request` fixture caused two parent `KeyError` failures before any private child ran. Signature was corrected before meaningful RED.
- `task3-red02` is the intended original-source RED: ordinary child passes, held child fails exactly `Library Prompt export blocked UI key dispatch until destination writer was released` (`[False] != [True]`). Both actual children have zero errors/skips; the held failure occurs after writing, source-row checks, worker drain and gate retirement.
- `task3-green01` is excluded as a GREEN claim: a CRLF-sensitive editing script failed before any source mutation but the shell continued to a no-change diagnostic. It repeats the original held failure and records the existing legacy other-screen fixture failure `RecoveryRequired: raw_source_selection_changed`. Later dependent shell steps use explicit fail-fast checks. A separate helper-edit syntax error also stopped before any file write or test run.
- The legacy chain is `test_library_prompt_export_journeys._export_host` -> `test_library_prompts_canvas._build_test_app` -> `Tests.UI.app_factory._build_test_app` -> real `TldwCli`/source-bound config loading after collection-time import. The module contains no deliberate profile/source-transition tests. Its exact existing `bootstrap_profile` opt-in retains admission and the supported per-test config reset.
- `task3-green02` proves candidate ordinary/held children and every adapted direct Prompt helper control pass. Its twelve unrelated native Windows helper failures are qualified below; the full helper batch is not claimed green.
- `task3-baseline01` reproduces all twelve Windows helper failures against exact original source and adds an intended RED at actual App dispatch for explicit Replace. The actual Replace child has zero errors/skips. Candidate bytes are restored in `finally` and checked by SHA256.

## Actual XML receipts

| Run | XML | Cases | Failures | Errors | Skips |
| --- | --- | ---: | ---: | ---: | ---: |
| task3-red01 | result.xml | 2 | 2 | 0 | 0 |
| task3-red02 | result.xml | 2 | 1 | 0 | 0 |
| task3-red02 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-red02 | pytest/test_prompt_export_dispatches_1/private-profile-process/pytest.xml | 1 | 1 | 0 | 0 |
| task3-green01 | result.xml | 3 | 2 | 0 | 0 |
| task3-green01 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green01 | pytest/test_prompt_export_dispatches_1/private-profile-process/pytest.xml | 1 | 1 | 0 | 0 |
| task3-green02 | result.xml | 47 | 12 | 0 | 0 |
| task3-green02 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green02 | pytest/test_prompt_export_dispatches_1/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-baseline01 | result.xml | 13 | 13 | 0 | 0 |
| task3-baseline01 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 1 | 0 | 0 |

Later aggregate/diagnostic receipts and exact exclusions are recorded below. Final minimal four-case receipt is 4 parent passes plus 4 actual child passes, zero failures/errors/skips.

## Inherited native Windows failures

All twelve failed on both original and candidate, with zero setup/teardown errors/skips. Full paired messages are retained in `.test-tmp/textual-37031-37034/task3-windows-baseline.json`. No tests were skipped, marked xfail, weakened, or changed to accept these failures.

- `test_write_export_text_replaces_the_file_and_keeps_its_mode`: POSIX mode expectation (0666 versus 0600) on native Windows.
- `test_a_dangling_symlink_destination_still_asks`: native symlink creation requires a privilege unavailable to this process (WinError 1314).
- `test_a_symlink_to_an_existing_file_is_written_through_and_named`: native symlink creation requires a privilege unavailable to this process (WinError 1314).
- `test_a_dangling_symlink_is_written_at_the_chosen_path`: native symlink creation requires a privilege unavailable to this process (WinError 1314).
- `test_receipts_name_the_folder_and_the_file`: POSIX slash/full-path receipt assumption on native Windows.
- `test_no_clobber_write_lands_a_new_file_where_hard_links_are_unsupported[PermissionError]`: native CRLF text-write bytes versus LF expectation.
- `test_no_clobber_write_lands_a_new_file_where_hard_links_are_unsupported[ENOTSUP]`: native CRLF text-write bytes versus LF expectation.
- `test_report_export_to_a_new_file_succeeds_where_hard_links_are_unsupported[PermissionError]`: POSIX slash/full-path receipt assumption on native Windows.
- `test_report_export_to_a_new_file_succeeds_where_hard_links_are_unsupported[ENOTSUP]`: POSIX slash/full-path receipt assumption on native Windows.
- `test_report_export_replace_writes_and_names_the_full_path`: POSIX slash/full-path receipt assumption on native Windows.
- `test_legacy_recovery_export_replace_writes_through_the_service`: POSIX slash/full-path receipt assumption on native Windows.
- `test_legacy_recovery_export_never_replaces_a_file_that_appeared_unasked`: POSIX slash/full-path receipt assumption on native Windows.

## Source custody and static checks

- Original export source SHA256: `496ea6a3a1371f7b18d28c70efd8c1f8b52eaad93b49dcd68ab9b2acb36853b8`.
- First formatted candidate SHA256: `d934d45ba7d60a3a95d10be883fd17c9e85fe592dd65e5b42cc0bafb2db0a80c` (restored exactly after baseline). Final production hash after preserving capture-error feedback: `782d6b9c3acdf6c965f4d959bda799a2f19807f50ce9ae8d5ba9a7fb4331145d`.
- The entire production module AST outside `export_library_prompt_file` matches the Git source. The shared `request_export_write`, Notes, Report, Collections, imports and all other functions remain unchanged.
- Focused Ruff check and format check pass on all four owned Python files before the final error-boundary control.
- Unchanged `scripts/check_textual_worker_contract.py` passed: no synchronous run_worker targets; 307 post-await lookups in 144 functions, none new; 73 wait-for-dismiss pushes reachable from 27 non-worker entries, none new. No census or allowances changed.
- Visual/token/component suites were not repeated: there are no stylesheet/token/style-property changes, per the common protocol and the coordinator-qualified Windows baseline.

## Warnings, limits and review

- Retained warnings include Pydantic `Field(exclude_if)` deprecation, Python audioop deprecation, pytest-asyncio default loop-scope deprecation, optional sounddevice/WebRTC availability logs and existing test-factory uninitialized Prompt/ChaChaNotes owner logs. Output is not claimed pristine.
- No full suite, remote operation, PR, CI dispatch, Qodo call, child agent, external message, configuration guard bypass, timeout increase or broad plugin was used.
- Existing tests and actual private-child teardown are inspected independently. Native gate threads and workers drain before test exit; SQLite closes in the new mounted tests.
- Self-review confirms captured data, no-clobber/Replace ownership and scoped reporter reuse. Independent review is explicitly owned by the coordinator (requesting-code-review skill), with status retained In Progress.

## Final result / commit

Final verification complete; committed exact six owned files as `6d6ba5b29f52a449c37abaedc114bcae081842c0`.


## Later verification and witness correction

- `task3-green03`: 23 pass / 4 fail, zero parent errors/skips. All six actual private children pass with zero errors/skips. The twelve fidelity combinations, four write-failure/retry combinations and other-screen scope case pass. Two scoped-navigation cases observed publication before the asynchronous receipt; they now wait only for the native `library-prompt-export` group. Their exact original assertions remain unchanged (AST assertion multiset proof: `task3-test-assertion-qualification.json`). The other two failures are native full-path separator assertions in the existing Prompt/Note confirmation controls.
- `task3-render-red01`: real malformed structured JSON serialization raises TypeError out of capture, 1 intended failure / 0 errors/skips. `task3-baseline02` preserves original source: the same render-error/retry control passes, while the two confirmation path-format failures reproduce unchanged (1 pass / 2 fail / 0 errors/skips). The candidate catch now retains original safe error feedback and retry. Flattening only that catch proves the entire valid-payload function AST is unchanged (`task3-capture-guard-qualification.json`).
- `task3-green04`: 102 pass / 3 fail / 0 errors/skips, 12 explicitly deselected helper cases already qualified against original source. All 34 selected helper cases (including render/retry), all three scoped-navigation cases, and all 64 Library size/wiring controls pass. Ordinary private child passes; held Replace fails the inherited independent 0.5s observation. The two additional failures are unchanged inventory/import guards, detailed below. This batch is not called green.
- `task3-baseline03`: exact original source fails the revised acknowledgement witness at the intended physical ordering assertion; 1 actual child failure / 0 child errors/skips. Both inventory/import guards also reproduce (3 parent failures / 0 errors/skips). Original writer/key-post occurs at 522166.875, watchdog release/physical completion at 522169.890, confirmation callback return at 522169.906, and App key acknowledgement at 522170.453 (`before_release=false`). This also shows about 0.55s of App queue/refresh time after callback return, so an unrelated 0.5s observer was not a reliable release criterion.
- `task3-diagnostic01` is diagnostic evidence, not a passing final receipt: parent 1 pass / 1 fail, actual child 1 fail / 0 errors/skips. On candidate source `_start_write` and `_answered` both return None/non-awaitable at 522332.531; writer enters at .656, first actual App acknowledgement at .968 is before release, physical completion at 522333.093. Textual forwards/bubbles the one F12 through App again at .984; the temporary non-idempotent observer recorded `[True, False]`. The final observer acknowledges the probe once, matching the original ThreadEvent contract. Callback profiling has been removed from normal tests. Native Python3.12 monotonic clock uses GetTickCount64 (15.625ms resolution), so final assertions use logical event order, not strict inequality of potentially equal timestamps.
- The unchanged native Performance guard passes in `task3-diagnostic01`: **558/558 modules**, 424352/425347 LOC, fattest Library route 130690/135111 LOC. Native HOME/USERPROFILE/XDG setup is the guard's own unchanged implementation. No caps, snapshots or allowances changed. Its zero-headroom warning is intentional and retained.

The original causal RED remains valid after diagnostic removal/idempotent acknowledgement: its first acknowledgement is after watchdog release and physical completion. No later duplicate can turn that first false acknowledgement into success. Product source has remained exact SHA782d6b9c since the capture-error guard.

## Additional unchanged baseline controls

- `Tests/UI/test_library_export_replace_confirm.py::test_prompt_export_onto_an_existing_file_asks_escape_keeps_it_replace_writes` and `::test_note_export_onto_an_existing_file_asks_escape_keeps_it_replace_writes`: original and candidate both report the existing mixed-separator receipt on Windows, while the assertion compares `str(destination)` with backslashes throughout. File preservation, Cancel/Replace and output are separately covered by the six new mounted cases. No confirmation-control assertion was changed.
- `Tests/Packaging/test_library_preimport_closure.py::test_preimport_defers_library_runtime_controllers`: original and candidate fail during initial config/default-data-root lock setup, before the route/deferred-controller assertions, at `create_private_text` with `PrivatePathError: unsafe_parent: missing_parent [\Users\...\test_preimport_defers_library_0]` (drive missing). The guard does not create its HOME/XDG directories or set USERPROFILE like the supported Performance probe does. Its source and config/private-path dependencies are byte-qualified unchanged in `task3-unchanged-guard-source.json`; no private-path or config guard was bypassed or altered.
- `Tests/Architecture/test_worker_exclusive_group_inventory.py::test_no_ungrouped_exclusive_workers`: original and candidate report the same eight existing sites, with no site in the Prompt export function: DirectoryNavigation._load; ChatScreen._console_environment; WatchlistsCollectionsScreen._resolve_daily_report_banner; SmartContentTree.on_mount; ConflictResolutionStep.on_show; ConsoleGoalSetupModal.launch; WorkspaceChangeReviewPanel._refresh_preparing; WorkspaceCreateModal._finish_or_retry_bindings. The DirectoryNavigation allowlist uses POSIX path spelling while native discovery formats backslashes. The remaining findings are unchanged existing inventory debt. No inventory/allowlist edits were made.

## Exact later commands and actual XML

All commands run with the explicit worktree cwd and native private runner described above. Original-source qualification uses checked original SHA496ea6a3 and restores exact candidate bytes in finally; receipts are `task3-source-qualification[-02/-03].json`.

```text
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-baseline01 Tests/Library/test_library_file_export.py::test_write_export_text_replaces_the_file_and_keeps_its_mode Tests/Library/test_library_file_export.py::test_a_dangling_symlink_destination_still_asks Tests/Library/test_library_file_export.py::test_a_symlink_to_an_existing_file_is_written_through_and_named Tests/Library/test_library_file_export.py::test_a_dangling_symlink_is_written_at_the_chosen_path Tests/Library/test_library_file_export.py::test_receipts_name_the_folder_and_the_file Tests/Library/test_library_file_export.py::test_no_clobber_write_lands_a_new_file_where_hard_links_are_unsupported[PermissionError] Tests/Library/test_library_file_export.py::test_no_clobber_write_lands_a_new_file_where_hard_links_are_unsupported[ENOTSUP] Tests/Library/test_library_file_export.py::test_report_export_to_a_new_file_succeeds_where_hard_links_are_unsupported[PermissionError] Tests/Library/test_library_file_export.py::test_report_export_to_a_new_file_succeeds_where_hard_links_are_unsupported[ENOTSUP] Tests/Library/test_library_file_export.py::test_report_export_replace_writes_and_names_the_full_path Tests/Library/test_library_file_export.py::test_legacy_recovery_export_replace_writes_through_the_service Tests/Library/test_library_file_export.py::test_legacy_recovery_export_never_replaces_a_file_that_appeared_unasked Tests/UI/test_library_prompt_export_responsiveness.py::test_prompt_export_dispatches_app_key_before_destination_release[replace-held-writer]
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-green03 Tests/UI/test_library_prompt_export_responsiveness.py Tests/UI/test_library_prompt_export_journeys.py Tests/UI/test_library_export_replace_confirm.py
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-render-red01 Tests/Library/test_library_file_export.py::test_prompt_render_failure_reports_error_and_allows_retry
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-baseline02 Tests/UI/test_library_export_replace_confirm.py Tests/Library/test_library_file_export.py::test_prompt_render_failure_reports_error_and_allows_retry
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-green04 Tests/Library/test_library_file_export.py Tests/UI/test_library_prompt_export_responsiveness.py::test_prompt_export_dispatches_app_key_before_destination_release[new-ordinary] Tests/UI/test_library_prompt_export_responsiveness.py::test_prompt_export_dispatches_app_key_before_destination_release[replace-held-writer] Tests/UI/test_library_prompt_export_journeys.py::test_export_result_after_navigation_preserves_current_status Tests/Architecture/test_library_modules_size_ratchet.py Tests/Architecture/test_library_prompts_wiring.py Tests/Architecture/test_worker_exclusive_group_inventory.py::test_no_ungrouped_exclusive_workers Tests/Packaging/test_library_preimport_closure.py --deselect Tests/Library/test_library_file_export.py::test_write_export_text_replaces_the_file_and_keeps_its_mode --deselect Tests/Library/test_library_file_export.py::test_a_dangling_symlink_destination_still_asks --deselect Tests/Library/test_library_file_export.py::test_a_symlink_to_an_existing_file_is_written_through_and_named --deselect Tests/Library/test_library_file_export.py::test_a_dangling_symlink_is_written_at_the_chosen_path --deselect Tests/Library/test_library_file_export.py::test_receipts_name_the_folder_and_the_file --deselect Tests/Library/test_library_file_export.py::test_no_clobber_write_lands_a_new_file_where_hard_links_are_unsupported[PermissionError] --deselect Tests/Library/test_library_file_export.py::test_no_clobber_write_lands_a_new_file_where_hard_links_are_unsupported[ENOTSUP] --deselect Tests/Library/test_library_file_export.py::test_report_export_to_a_new_file_succeeds_where_hard_links_are_unsupported[PermissionError] --deselect Tests/Library/test_library_file_export.py::test_report_export_to_a_new_file_succeeds_where_hard_links_are_unsupported[ENOTSUP] --deselect Tests/Library/test_library_file_export.py::test_report_export_replace_writes_and_names_the_full_path --deselect Tests/Library/test_library_file_export.py::test_legacy_recovery_export_replace_writes_through_the_service --deselect Tests/Library/test_library_file_export.py::test_legacy_recovery_export_never_replaces_a_file_that_appeared_unasked
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-baseline03 Tests/UI/test_library_prompt_export_responsiveness.py::test_prompt_export_dispatches_app_key_before_destination_release[replace-held-writer] Tests/Packaging/test_library_preimport_closure.py Tests/Architecture/test_worker_exclusive_group_inventory.py::test_no_ungrouped_exclusive_workers
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-diagnostic01 Tests/UI/test_library_prompt_export_responsiveness.py::test_prompt_export_dispatches_app_key_before_destination_release[replace-held-writer] Tests/Performance/test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget -rP
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-green05 Tests/UI/test_library_prompt_export_responsiveness.py::test_prompt_export_dispatches_app_key_before_destination_release
```

| Run | XML | Cases | Failures | Errors | Skips |
| --- | --- | ---: | ---: | ---: | ---: |
| task3-green03 | result.xml | 27 | 4 | 0 | 0 |
| task3-green03 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green03 | pytest/test_prompt_export_dispatches_1/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green03 | pytest/test_prompt_export_dispatches_2/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green03 | pytest/test_prompt_export_dispatches_3/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green03 | pytest/test_prompt_export_racing_file0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green03 | pytest/test_prompt_export_racing_file1/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-render-red01 | result.xml | 1 | 1 | 0 | 0 |
| task3-baseline02 | result.xml | 3 | 2 | 0 | 0 |
| task3-green04 | result.xml | 105 | 3 | 0 | 0 |
| task3-green04 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green04 | pytest/test_prompt_export_dispatches_1/private-profile-process/pytest.xml | 1 | 1 | 0 | 0 |
| task3-baseline03 | result.xml | 3 | 3 | 0 | 0 |
| task3-baseline03 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 1 | 0 | 0 |
| task3-diagnostic01 | result.xml | 2 | 1 | 0 | 0 |
| task3-diagnostic01 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 1 | 0 | 0 |

## Final minimal regression result

`task3-green05` passes all four ordinary/held New/Replace cases on exact final production source. Parent4 pass; four actual private children each1 pass; zero failures/errors/skips everywhere (253.06s). Held cases each record exactly one idempotent actual key acknowledgement before release/physical completion and no watchdog release. The two racing-file controls already passed in `task3-green03`; their assertions/source path are unchanged. No broad journey repetition was performed after the narrowly qualified capture-error guard.

| Run | XML | Cases | Failures | Errors | Skips |
| --- | --- | ---: | ---: | ---: | ---: |
| task3-green05 | result.xml | 4 | 0 | 0 | 0 |
| task3-green05 | pytest/test_prompt_export_dispatches_0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green05 | pytest/test_prompt_export_dispatches_1/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green05 | pytest/test_prompt_export_dispatches_2/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-green05 | pytest/test_prompt_export_dispatches_3/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |

Final focused Ruff check and format check pass on all four owned Python files; owned diff whitespace check passes. Final production/test SHA256 values:

- `tldw_chatbook/UI/Library_Modules/library_file_export.py`: `782d6b9c3acdf6c965f4d959bda799a2f19807f50ce9ae8d5ba9a7fb4331145d`
- `Tests/Library/test_library_file_export.py`: `5bd35440b0cbfbacbf6a39775b23c38646f9457a75dfa93fef9914b151c8c8d8`
- `Tests/UI/test_library_prompt_export_responsiveness.py`: `9b05c25c15bcb8906afee55e27d2ae5927a401d73e3b07e5241cd49781c292ea`
- `Tests/UI/test_library_prompt_export_journeys.py`: `8ef4dc8879d2bcc9b56cd3804b4ec1709a3744b35ad8d4fa79d2ddf238fd7b96`

Files owned by this task: the Prompt seam, new responsiveness test module, the two authorized existing helper/journey test files, `Docs/Architecture/library-shell.md`, and the TASK-37033 record. The local report/evidence remains in the assigned SDD/.test-tmp directories. Root-owned census, Evals namespace/plan/audit edits and all other work remain untouched.

## Commit and independent review handoff

- Base: `8a91bd5d85cdc7ab50190592d4a4439201d20759`
- Head / task commit: `6d6ba5b29f52a449c37abaedc114bcae081842c0` (`fix(library): keep saved Prompt exports responsive`)
- Post-commit verification: exactly the six owned files in the commit; owned working files match HEAD; final tested production/test byte hashes are unchanged. Root-owned changes remain unstaged and untouched.
- Review requirements: all four TASK-37033 ACs and `.superpowers/sdd/2026-10-27-textual-37031-37034/task-3-brief.md`. Focus on callback return ownership, captured data, late no-clobber collision, safe render/write failure feedback and scoped completion. Root owns independent review and current-head CI; task remains In Progress.


## Review fix round1/5 — Important I1

Fix start HEAD: `c37142d521e57fb203a34f01c2bedc684fe46560` (coordinator Evals namespace administration; Task3 product bytes unchanged). Pre-fix source SHA256 remains `782d6b9c3acdf6c965f4d959bda799a2f19807f50ce9ae8d5ba9a7fb4331145d`.

Verified finding: the no-clobber writer's FileExistsError handler performed a second destination metadata probe through `request_export_write` outside its recoverable error boundary. A PermissionError from that probe escaped the native worker and caused Textual's default fatal worker handling. The Prompt-only correction nests the collision/recheck path inside the existing Exception boundary and reuses the captured sanitized notifier. Confirmed Replace write errors still reach that same boundary. No `exit_on_error=False`, shared-helper, Notes/Report, ownership, dependency, import, allowance or visual changes were made.

The new low-level control keeps the native App and workers real. It creates a real precious file immediately before atomic no-clobber publication, arms only the subsequent destination Path.lstat fault after actual FileExistsError, and clears the fault in finally. It verifies sanitized PermissionError feedback, a running App, the precious file and temporary-file cleanup, explicit same-destination Replace retry, and output using the original captured content after the caller's mapping is edited. Production confirmation/ownership/state machines are not mocked; only filesystem I/O is controlled.

### TDD and exact commands

- `task3-r1-red01`: 1 intended failure / 0 errors/skips, on exact pre-fix source. Actual failure: `textual.worker.WorkerFailed: Worker raised exception: PermissionError(13, 'private metadata failure detail')`; the native App reaches fatal worker handling from the late Path.lstat probe. There are no private-profile children in this direct native App case and no teardown errors.
- A pre-apply AST verification script initially passed a statement list directly to ast.dump and stopped before writing source or launching GREEN. Fail-fast preserved exact pre-fix bytes; this is editing-tool exclusion evidence, not a test result.
- `task3-r1-green01`: all 10 parent cases pass, with two actual private children each1 pass and zero failures/errors/skips everywhere (173.42s). Seven helper cases cover the new collision/recheck/retry path, render/retry, direct late collision and both hard-link fallback refusals; the real picker write-failure/retry journey runs its wide dark variant; the two real SQLite/FileSave racing-file controls cover Cancel and Replace. All substantive existing assertions remain in place. No 19-case journey sweep was repeated.

Commands use the same explicit Python/worktree/native private runner as the initial implementation:

```text
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-r1-red01 Tests/Library/test_library_file_export.py::test_prompt_collision_recheck_failure_reports_error_and_allows_retry
& C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task3-r1-green01 Tests/Library/test_library_file_export.py::test_prompt_collision_recheck_failure_reports_error_and_allows_retry Tests/Library/test_library_file_export.py::test_prompt_render_failure_reports_error_and_allows_retry Tests/Library/test_library_file_export.py::test_prompt_export_wrapper_asks_when_a_file_appears_mid_write Tests/Library/test_library_file_export.py::test_prompt_export_to_a_new_file_succeeds_where_hard_links_are_unsupported Tests/Library/test_library_file_export.py::test_a_file_appearing_mid_export_still_asks_where_hard_links_are_unsupported Tests/UI/test_library_prompt_export_journeys.py::test_copy_and_export_failures_keep_editor_ready_for_retry[textual-dark-size0] Tests/UI/test_library_prompt_export_responsiveness.py::test_prompt_export_racing_file_requires_an_explicit_answer
```

| Run | XML | Cases | Failures | Errors | Skips |
| --- | --- | ---: | ---: | ---: | ---: |
| task3-r1-red01 | result.xml | 1 | 1 | 0 | 0 |
| task3-r1-green01 | result.xml | 10 | 0 | 0 | 0 |
| task3-r1-green01 | pytest/test_prompt_export_racing_file0/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |
| task3-r1-green01 | pytest/test_prompt_export_racing_file1/private-profile-process/pytest.xml | 1 | 0 | 0 | 0 |

### Source/static qualification and review handoff

- Final production SHA256: `dfe6f86946ff717678cd3009c60cd4cb3b8ba1f433e2dbded485295eb65e4608`.
- Helper-test SHA256: `9edafd42d1f35aac41a3938d2f9fb764449ee079e62c573676ce6803c86df42d`.
- `task3-r1-source-qualification.json` proves identical normal writer await, existing error logger/notifier/return, normal success receipt, outer capture/worker/modal ownership, every statement/import/function outside Prompt export, and every pre-existing helper test/class definition. Only the recheck failure boundary and the added regression change.
- Focused Ruff check and format check pass for `tldw_chatbook/UI/Library_Modules/library_file_export.py` and `Tests/Library/test_library_file_export.py`. No new imports in product code, source ceilings, headroom or boot behavior were introduced. The previous unchanged native558/558 measurement and baseline limitations remain disclosed; coordinator owns final combined-source/CI gates.
- ADR required: no; direct preservation of the existing AC3 failure contract under the canonical ADRs already listed. The required Backlog CLI recorded the review-fix plan before product/test edits and appends verified implementation notes afterward; status remains In Progress.
- M1 inherited Pydantic Field/audioop/pytest-asyncio warning debt remains deferred. The passing output is not described as pristine. No dependency/test-configuration cleanup, full suite, subagent, review dispatch, PR/CI or remote action was performed.
- R1 owned files are exactly the Prompt seam, `Tests/Library/test_library_file_export.py`, and TASK-37033 record. Root census/admin files stay untouched. Scoped rereview is requested from the coordinator after the exact owned-file commit.

R1 commit: `c375f164e6955c68c90c3d5e75a6c936023dacbc` (`fix(library): recover Prompt collision recheck failures`), parent `c37142d521e57fb203a34f01c2bedc684fe46560`. Exactly the three R1-owned files were committed; tested source/test bytes remain unchanged and owned working files match HEAD. Root census remains unstaged and untouched. Ready for coordinator scoped rereview; task remains In Progress.
