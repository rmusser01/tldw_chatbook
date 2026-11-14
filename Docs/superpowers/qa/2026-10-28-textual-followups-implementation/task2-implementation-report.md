# TASK-37267 implementer report

Status: **DONE_WITH_CONCERNS**. All seven new mounted regressions and all 13
selected existing behavior controls passed on the final product/test source.
The scoped controls/governance batch finished 46 passed / 8 failed / 0 errors /
0 skips. The eight static failures remain explicit below: baseline/current
scanner comparisons show no new timer edge, worker violation or required pin
change from this patch. The inherited real artifact-action signature mismatch
also remains outside this fix. Root owns independent review, current-dev
integration and hosted gates.

## Change and ownership

The Runs window now captures its source and selection revision before the
native interval's awaited read and refuses publication after either changes.
The existing selected-run setter advances the revision, including clearing
selection on a source switch. This covers return to the same source/run and
selected-run updates without introducing another state owner. The setter now
projects `None` as the existing empty-detail copy immediately on source switch,
before catalog success, failure, or emptiness. Accepted refresh still updates
the selected record/detail directly and retains the bundle/artifact/event
payload. The terminal/Server admission guards are unchanged.

Owned deliverables:

- `tldw_chatbook/UI/Research_Window.py`
- `Tests/UI/test_research_run_refresh_publication.py`
- `Docs/Architecture/research-workspace.md` (Runs section only)

`research_controller.py` is unchanged. Its `get_run` still caches the last
fetched result. Search found no current consumer requiring that cache to mirror
the UI selection: the window owns `selected_run` and all its run action methods
derive their ID from `_selected_run_id()`. This report makes no cache/selection
consistency claim. The Research Runs (`research`) and Research Workspace
(`research_workspace`) routes, scope executor, persistence and execution owners
remain separate and unchanged.

ADR required: no. ADR path: N/A; existing ADR-033, ADR-068, ADR-070 and ADR-078
cover the retained screen/service/execution boundaries. Design-language and
component-pattern instructions were read; no stylesheet, token or visual
structure change was needed. No schema, dependency, global owner, remote
fallback, runtime fallback or rollback guarantee was added.

## AC mapping and actual boundaries

| AC | Implementation and evidence |
| --- | --- |
| 1: newer B selection/highlight/detail and payload ownership | The revision check rejects A after real ListView Enter selects B. `test_refresh_preserves_newer_run_selection` asserts B's selected ID, highlighted row, detail, action-ID getter, exact payload object identities and rendered report/event; the same list/detail controls remain attached. |
| 2: Local completion after source departure cannot restore Local target | `test_local_refresh_cannot_restore_a_departed_source[server]` uses native Select Enter/down/Enter. Before release, Server is current and selection is clear; after release, the action-ID getter still refuses no selection and Local detail stays absent. The `source-aba` case returns through native Select, performs a real scoped SQLite pause/catalog read and real Enter reselect of A, then proves the older running A cannot overwrite the newly paused A. |
| 3: empty detail on source clear, later valid selection works | `_set_selected_run(None, reset_payload_state=True)` projects empty detail before list loading. `test_source_switch_clears_previous_detail[unavailable/empty]` asserts exact empty copy, no rows, no payload/event state, then returns through Select and Enter to render valid Local B. |
| 4: ordinary progress/payload and terminal/Server guards | `test_native_refresh_updates_progress_and_retains_current_payload` commits `A new SQLite progress` through the real Local service, observes the actual timer read and rendered new phase/progress, and retains all payload object identities. `test_native_refresh_skips_terminal_and_server_selections` observes the actual `Timer._tick` completion while delegating to its original implementation; both a completed Local record and a running Server record issue no scope get-run read. |
| 5: real mounted holds, transition witnesses, cleanup | `_HeldRead` calls the original `ResearchScopeService.get_run`, records the actual returned SQLite ID and native `Timer._run_timer` coroutine/task, and only then holds its return. The backend executor remains available during the hold. Every release is in a `finally` inside the App lifetime, and the actual child XML confirms no setup/teardown errors for intended RED or GREEN. |

The mounted harness pushes the real `ResearchScreen` with production bundled
CSS. It does not call or replace `_auto_refresh_selected_run`, shorten the
two-second interval, replace real persistence with constant get-run records, or
use archived hardcoded-output scripts. The canonical audit probe was read as
reference data only. Expected domain IDs, titles, progress, paused/running
control states and payload content are literal independent values.

Payload qualification: bundle and artifact **fixture state** is prepared from
the real scope/SQLite reads and projected with the existing renderers; stored
Local events travel through `watch_selected_run_events`. This is payload
ownership/retention evidence, not proof that the existing artifact action
succeeds. See the separately identified inherited mismatch below. The empty and
running Server catalog controls use small asynchronous backend doubles at the
external service seam; no real remote server or remote mutation was exercised.

## Commands, runs and failure accounting

All App runs used the project Python and mandated private-profile launcher,
with `require_escalated`. Every run retained `--timeout=180`. No full suite was
run. No profile/privacy/recovery guard, ACL, timeout, shared fixture, launcher,
CI file, census, task metadata or plan was modified by this implementer. No Git
staging, commit, branch, checkout, rebase or dispatch operation was performed.

Common invocation prefix:

```powershell
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37266-37268/run-targeted.py
```

Focused selectors, used unchanged in all three RED attempts:

```text
Tests/UI/test_research_run_refresh_publication.py::test_refresh_preserves_newer_run_selection
Tests/UI/test_research_run_refresh_publication.py::test_local_refresh_cannot_restore_a_departed_source[source-aba]
Tests/UI/test_research_run_refresh_publication.py::test_source_switch_clears_previous_detail[unavailable]
```

| Label and arguments after prefix | Actual outcome |
| --- | --- |
| `task2-red01 <three focused selectors>` | Invalid harness diagnostic: all 3 parent cases failed `KeyError: request` before private children or App mounts, because new test signatures omitted the wrapper's required fixture. Corrected test signatures only. No intended RED claim. Parent 3 failures/0 errors/0 skips; no child reports. |
| `task2-red02 <three focused selectors>` | Invalid evidence for this task: all 3 actual children hit the inherited artifact-loader `TypeError` before the intended ownership assertions. All 3 children had 1 failure/0 errors/0 skips; parent 3 failures/0 errors/0 skips, 155.14s. Holds released normally. Product remained unchanged. |
| `task2-red03 <three focused selectors>` | Intended RED: 3 parent and 3 actual child failures, 0 errors/skips, 92.85s. Actual failures were selected A vs expected B; older `running` vs expected `paused` after source ABA; and prior Local B detail vs expected empty detail after unavailable Server. Native task/transition witnesses are retained in child properties. |
| `task2-green01 Tests/UI/test_research_run_refresh_publication.py --durations=0 --durations-min=0` | All 7 parent cases and their 7 actual private children passed; no failures/errors/skips. Parent session 186.11s. This is the final product/test source. |
| `task2-controls01 -p task2_bootstrap_controls @task2Selectors --durations=0 --durations-min=0` | 46 passed / 8 failed / 0 errors / 0 skips, 358.17s; 13 legacy behavior controls passed, 33 governance controls passed, eight static failures retained below. Actual three private token/Python-style children passed with no errors/skips. Exact node IDs are retained in `task2/control-selectors.json`. This is not an all-green static batch. |

The initial invalid runs remain archived; none is relabeled as an intended
failure or green batch. The runtime source snapshots for each attempt are
retained separately before any product change.

The controls invocation uses the explicitly permitted supported
`bootstrap_profile` replay for the 13 exact legacy Research screen/controller
nodes in `task2/legacy-selectors.json`. The diagnostic-only plugin adds that
registered marker to those nodes and changes no test body, getter, shared
fixture, participant registry, guard or timeout. New controls retain the normal
`private_profile_test` subprocess wrapper. This is not a claim that the default
legacy fixture invocation was run or passed. Exact command:

```powershell
$env:PYTHONPATH = (Resolve-Path .test-tmp/textual-37266-37268/task2).Path
$task2Selectors = @(Get-Content .test-tmp/textual-37266-37268/task2/control-selectors.json | ConvertFrom-Json)
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37266-37268/run-targeted.py task2-controls01 -p task2_bootstrap_controls @task2Selectors --durations=0 --durations-min=0
```

Supporting read-command limits: an initial sandbox Git base read refused the
checkout as dubious ownership; the successful read used command-local
`git -c safe.directory=<exact worktree> rev-parse HEAD`, without modifying Git
configuration. Sandbox reads of launcher-created private folders were denied;
subsequent receipt/log inspection used `require_escalated`. No automatic
approval rejection occurred, and no ACL or guard was relaxed.

## Immutable source pins and actual receipts

BASE: `8fbc9d2e1d91c31654742736f652d173b07b3ea9` (read from this checkout).
No active-branch rebase was done by this implementer. Root owns later current-dev
integration and requalification.

| Source | SHA-256 |
| --- | --- |
| RED Research window (all RED attempts) | `026a6af5fa30f11e659cacec8d8cca95a6ee5720e2d76ef0281d0074fe75c359` |
| RED01 test | `cc6deed00d5e8f5d846f006b5a0715a58c3497abe4fa729cbd782597a6dc4bb0` |
| RED02 test | `7f97bd65b403167e1df849cb57dc9e49242dcf6faa3d5d9d04481675c9ad9d27` |
| Intended RED03 test | `7d2f91ebb7350568002153b3aa35d72459d95c701927b22f74afd293786171e1` |
| Final GREEN window | `4f6187501ca52af4aed63360cfa0ef711a31c29d79e89ba0ccbfc95c24c9e3dc` |
| Final GREEN test | `a5d9fad60f787de2de67a8a12a514faba38e89713fa4b341c76f723f48f19cef` |
| Unchanged controller | `28bbae033175802b9b82035bd9435f83dd2738f8110c4bbd9d2f68771f598bdd` |
| Final architecture doc | `a0787a22c51a0550578ff913ca14e06f7766a22c478f13c0fd3455b5e1eed7e2` |

Actual parent report hashes:

- RED01: `efedea15f63fe454a64cb8974fa093ed71d102d3e87675b9d18031de4c3524f8`
- RED02: `21d8be1012f8c47eb764325b036cae21e078bd9b5ccd8b3f1d901f5aa0911f8d`
- Intended RED03: `8f94d81a204dd1bf66393f8c0b3ff84ce74288b823dfeafcdd28145cd74e644a`
- Final GREEN01: `7ca5cf595b488518d82da7f0aa1b60c77deecbe48674da3bb779d3b0d75cadd2`
- Controls/governance: `540ae19ca28b66f8b351161f29a5e8d26d7daff2c8f2fb4418a5531a44324ee9`

All parent/child XML, child logs, command JSON, wrapper verification JSON,
parsed actual suite attributes, testcase outcomes/properties, raw hashes and
reported phase costs are archived under
`.test-tmp/textual-37266-37268/task2/<run-label>/`. The inspection is performed by
`task2/inspect-runs.py` and recorded in each `inspection.json`. Exact source
snapshots remain under `task2-red-sources`, `task2-red02-sources`,
`task2-red03-sources`, and `task2-green01-sources`; the complete manifest is
`task2/source-hashes.json`. These are supporting local artifacts for root to
archive, not additional product deliverables.

## Inclusive costs and limits

The GREEN command requests every parent setup/call/teardown row. Complete
reported parent phase rows, exact parent testcase inclusive totals, actual
child testcase totals/session totals and XML hashes are exported to
`task2/new-case-timings.json`; no timing-only rerun was done.

| New case | Parent testcase total (s) | Actual child testcase total (s) | Child session (s) |
| --- | ---: | ---: | ---: |
| newer B selection | 28.499 | 10.676 | 12.932 |
| Server departure | 26.176 | 15.414 | 17.437 |
| source ABA | 24.911 | 13.030 | 15.199 |
| unavailable catalog clear/reselect | 22.687 | 10.637 | 12.285 |
| empty catalog clear/reselect | 22.499 | 10.949 | 12.781 |
| normal progress/payload | 28.290 | 16.069 | 17.716 |
| native terminal/Server skips | 29.247 | 12.687 | 15.296 |

Parent totals include fixture setup/teardown and subprocess startup/waiting;
child testcase totals are the real child's setup/call/teardown, while child
session totals include collection/session overhead. Child processes keep their
existing default duration threshold, which can hide small split phase rows;
the archive preserves those notices rather than treating hidden phases as
zero. Parent displayed phase rows are rounded by pytest (including a reported
`0.00s` setup); the XML inclusive totals remain the independent receipt. These
are bounded headless harness costs on this host, not a natural race-frequency,
live-terminal performance or hosted CI capacity measurement.

## Static checks, self-review and concerns

Scoped Ruff baseline check returned `[]` for the unchanged window; the final
window and new test also returned `[]` with `--no-cache`. Baseline window format
check and final two-file format check passed. Scoped `git diff --check` passed.
No broad reformatting or inherited lint cleanup was done. Commands:

```powershell
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -m ruff check --no-cache --output-format json tldw_chatbook/UI/Research_Window.py
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -m ruff check --no-cache --output-format json tldw_chatbook/UI/Research_Window.py Tests/UI/test_research_run_refresh_publication.py
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -m ruff format --check tldw_chatbook/UI/Research_Window.py Tests/UI/test_research_run_refresh_publication.py
git -c safe.directory=C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui diff --check -- tldw_chatbook/UI/Research_Window.py Docs/Architecture/research-workspace.md
```

Self-review read the exact window/doc diff and complete new test. The revision
advances through the existing selected-run setter, not through controller
caching; the interval still accepts ordinary progress, returns silently on
transient read errors, and preserves its original admission guard and executor.
The only detail-rendering change is the explicit `None` projection. No shared
fixtures, census, plan, CI or task metadata were edited, and unrelated root
changes visible in the workspace were left untouched.

### Static failures and bounded baseline comparison

The 54-case controls batch included 13 existing behavior controls and 41
governance cases. All 13 behavior controls passed. Governance had 33 passes and
these eight failures; none was discarded or redefined as a pass:

| Failed static case | Actual result and baseline qualification |
| --- | --- |
| `test_no_timer_path_update_defaults_to_layout_true` | The Windows census uses backslash paths while `CLASSIFIED_SITES` uses slash paths. Raw comparison reports 108 unclassified sites. Diagnostic separator normalization still leaves four sites in unchanged `Widgets/Settings_Widgets/project_index_binding_card.py` (`_feedback`, `_launch`, `_paint`, `run`). The exact same 120 semantic census sites, explicit-layout flags and root associations occur with the unchanged RED Research AST and final Research AST; only this window's detail line shifts 787 → 791. Thus this patch adds no new obligation. Separator normalization was diagnostic only; the test/guard was not changed or rerun as a fabricated pass. |
| `test_clock_root_set_is_pinned` | Raw Windows paths disagree with slash pins. Diagnostic separator normalization additionally leaves one existing `ProjectIndexBindingCard._poll` root missing from the pins. Baseline/current have the same 75 roots and zero unresolved-root problems; no root was added, removed or rewired by this fix. |
| `test_wrapper_spelled_interval_is_censused` | Synthetic-only fixture expects `tldw_chatbook/timer_census_fixture.py`, while the actual keyed fixture path is `tldw_chatbook\\timer_census_fixture.py`; no Research source is used in this case. The guard's AST is unchanged from BASE. |
| `test_deferral_shim_callback_resolves_through` | The synthetic fixture's correct callback appears under the backslash path and fails the slash-key membership assertion. The guard's AST is unchanged from BASE. |
| `test_query_one_receiver_is_never_auto_classified` | The synthetic fixture's receiver correctly appears under the backslash path and fails the slash-key membership assertion. The guard's AST is unchanged from BASE. |
| `test_no_ungrouped_exclusive_workers` | Eight raw findings: one vendored directory-picker allowlist entry mismatches Windows separators, plus seven other group findings in unchanged Chat/watchlists/tree/import-wizard/goal/setup-review/workspace-create source. The real AST detector returns identical findings for every named BASE/current input. Research window has zero findings on both. No claim is made that the seven non-vendored findings are explained by separators. |
| `test_mutation_refreshes_dispatch_through_loader_group` | Three findings in unchanged scheduling/watchlists source: `SchedulesWorkbench._request_tasks_refresh._load`, `WatchlistsCollectionsScreen._follow_coordinated_briefing`, and `WatchlistsCollectionsScreen._request_briefings_refresh.refresh`. The same detector returns the exact same owner/detail findings on BASE and current source. These are baseline inventory failures, not a demonstrated runtime mutation regression from this task. |
| `test_allowlist_has_no_stale_entries` | The vendored directory-picker key is emitted with Windows backslashes while its allowlist key uses slashes. Baseline/current source and detector ASTs are the same, with the same live violation. |

No broad static battery was repeated. The baseline attribution uses the
guard's existing `census(modules=...)` seam to index the immutable RED Research
AST at the original module path, keeping the rest of the parsed package map
identical. The diagnostic compares semantic site keys, layout flags, associated
roots, root sets and unresolved-root problems. It does not change guards,
pins, classifications, product files or live source during a test.
`task2/timer-input-comparison.json` retains both complete actual observations;
`compare-timer-inputs.py` retains the exact comparator. The only actual site
record difference is the existing detail update's line 787 → 791.

The second comparator reads BASE blobs with `git show` and passes BASE/current
ASTs directly to the unmodified worker detector and mutation-contract helper.
It retains all external failing inputs, both guard sources and the owned
window under `task2/static-inputs/`, with raw hashes, AST equality and exact
findings in `worker-input-comparison.json`. All other compared source ASTs are
identical to BASE; raw Git blob/checkout hashes differ because of LF/CRLF.
The retained snapshots additionally compare equal after newline normalization
for every external input and both guards (`all_other_sources_normalized_text_equal: true`).
`git diff --name-only <BASE> -- tldw_chatbook <two guard paths>` independently
lists only `tldw_chatbook/UI/Research_Window.py`. This binds the unchanged
package inputs for the timer comparison as well. No shared artifact update or
pin increase is required by Task2; root may separately triage the pre-existing
platform/inventory debt.

Comparator commands (plain static AST reads, no App execution or profile
mutation):

```powershell
$env:PYTHONPATH = (Get-Location).Path
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37266-37268/task2/compare-timer-inputs.py
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37266-37268/task2/compare-worker-inputs.py
```

The final source/receipt consistency check confirmed the live product and new
test still exactly match the executed GREEN snapshots; scoped Ruff,
two-file format and tracked whitespace checks were rerun successfully before
handoff. No further App or timing-only batch was run. Final parsed summary:
`task2/final-summary.json`, SHA-256
`5680a87dcf4bc21172f19547cef9eb91cec06db66243650a55bbc7220396393d`.
Timer comparison SHA-256:
`c896f4f7f394eed3dd1fe51b93e2e17e71ad6f300c9e5baa8cd04a22096fd4c8`;
worker comparison:
`919ad8405e63aed6aa38c491d828c27be104d42e5284cdc2edf42ee1cbe7bcaa`;
new-case timing export:
`02ee2b90015f2ed786590a2e831a83b91261db886e3208c4cc93f1ae6b16f211`.

Known inherited concern: actual
`ResearchController.get_artifact(source, run_id, artifact_name)` calls
`service.get_artifact(run_id, artifact_name, mode=source)` positionally, while
real `ResearchScopeService.get_artifact` is keyword-only. The real mounted
bundle action auto-load reaches that TypeError; fake-service legacy tests mask
it. RED02 preserves the actual failures. Root independently confirmed the
ownership/history and directed leaving it untouched for this bounded timer
fix. Root separately filed medium-priority To Do
[TASK-37330](../../../../backlog/tasks/task-37330%20-%20Restore-Research-Runs-artifact-loading-through-the-real-scope-contract.md)
for that real-scope contract defect. No successful real artifact-action claim
is made here.

Warnings observed: existing pytest-asyncio unset fixture-loop-scope deprecation,
Pydantic `Field(exclude_if=...)` deprecation, and optional Sounddevice/WebRTC VAD/
datasets availability diagnostics. The static scanners additionally emitted
the existing invalid-escape SyntaxWarnings for `train_journey.py:31/32` (four
warning records across timer/worker scans). RED01 additionally emitted three
`record_property`/xunit2 warnings; subsequent diagnostics use ordinary testcase
user properties and did not emit those warnings. Final hosted gates,
independent review, current-dev integration, complete census/capacity
qualification, Git commit/PR, and Backlog closure remain root responsibilities.
