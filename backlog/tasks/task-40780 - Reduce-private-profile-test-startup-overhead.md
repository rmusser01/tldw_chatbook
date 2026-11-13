---
id: TASK-40780
title: Reduce private-profile test startup overhead
status: Done
created_date: 2026-11-13 19:15
assignee:
- '@codex'
updated_date: 2026-11-13 20:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove measured redundant collection and application construction from private-process test startup while preserving exact test selection, fresh profiles, fixtures, failure reporting and coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Private children collect only the selected test path while preserving parametrized selection, ancestor fixtures, and fresh process/profile isolation.
- [x] #2 Settings category enumeration avoids building a full app during collection and retains the same complete mounted category coverage.
- [x] #3 Paired fresh-process timings demonstrate reduced startup work; targeted isolation, timeout, failure-report and coverage checks pass.
- [x] #4 Alternating before/after benchmarks and targeted isolation/coverage checks run on the Ryzen Linux server in a window coordinated with the rollout chat, with raw results and limitations recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only optimization preserving existing process, profile and coverage boundaries; no new dependency or runtime choice.
1. Profile fresh private children and compare exact-node collection with unrelated paths ignored.
2. Add an actual subprocess regression for collector pruning, exact parametrization, ancestor fixtures and isolation; retain existing coverage/deadline checks.
3. Limit child collection using pytest native ignore-glob, and use the existing lightweight Settings app-stub pattern only for import-time category enumeration.
4. Run targeted tests and paired timings, review coverage equivalence and record measurements and remaining costs.
5. User-requested Linux verification: coordinate an idle window with the rollout chat, use a separate bounded guest, benchmark the same source/dependencies with alternating revisions and fresh profiles, and retain raw phase timings plus correctness checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Private-profile children now pass pytest's native --ignore-glob=* alongside their exact node ID. Explicit paths and their ancestors bypass pytest's ignore hook; unrelated files no longer get collectors. Fresh interpreter/profile creation, environment redirection, autouse fixtures, network/audio guards, cleanup, deadlines, failure reporting and coverage aggregation are unchanged.

Settings category enumeration now constructs an unmounted SettingsScreen with SimpleNamespace(app_config={}), a stub pattern already used by this repository's Settings tests. It no longer calls _build_test_app during import. Mounted tests still build real apps, visit every category and assert that their complete ordered category list matches the enumerated list.

ADR required: no; ADR path: N/A. This preserves the existing testing boundaries and uses the installed pytest feature, with no dependency, application interface or runtime-policy change.

## Scope

- Tests/private_profile.py: one native pytest option plus its rationale.
- Tests/UI/test_settings_category_sweep.py: lightweight collection-time app stub and mounted category equivalence assertion. No selectors or parametrization removed or reordered.
- Tests/CI/test_private_profile_collection.py: two real private children verify sibling collector pruning, package initialization, ancestor autouse fixture setup/teardown, class/parameter selection (including brackets in the ID), distinct process IDs and distinct private profiles/home directories. Automatically included by the existing Tests/CI PR lane.
- backlog/docs/lessons-testing-evidence.md: record the cold/warm profiling trap and the compatibility checks for native collector pruning.
- No workflow, runner, server, production code or dependency changes. Prior TASK-40750 changes remain in this worktree.

## Local measurements

Base: 1455ceeb96. Local Python 3.12.11, pytest 8.4.2, Textual 8.2.8. Plugin autoload disabled; asyncio and timeout plugins explicitly enabled. Every trial uses a fresh interpreter and Tests/conftest.py's fresh profile. Shared bytecode is warmed for both revisions. Local timings are not a hosted-CI throughput or p95 claim.

Representative node: Tests/UI/test_settings_connect_rows.py::test_provider_is_one_row_one_tab_stop_and_filters_by_name_or_id

Alternating before / after / after / before, warmed before each measured fresh child:

| Revision | Collection (s) | Body (s) | Total process (s) |
| --- | ---: | ---: | ---: |
| Before | 3.874 | 7.461 | 14.812 |
| After | 2.524 | 7.363 | 13.469 |
| After | 2.540 | 7.368 | 13.305 |
| Before | 3.858 | 7.063 | 14.307 |

Mean collection: 3.866 -> 2.532 s (1.334 s / 34.5% less). Mean total process: 14.560 -> 13.387 s (1.173 s / 8.1% less). Body times are similar; startup work was removed, not test actions. These measurements include both changes.

The separate native-filter-only experiment used the pending-save node below (which does not import the category sweep). In before / after / after / before order, process totals were 4.484 / 4.005 / 4.144 / 4.430 s; collection was 2.676 / 2.336 / 2.403 / 2.635 s. Mean process saving: 0.383 s. Do not add this saving again to the combined result above.

cProfile operation counts for Connect collection confirm the removed work:
- pytest_pycollect_makemodule: 1,768 -> 1 calls.
- app_factory._build_test_app during collection: 1 -> 0 calls. The original app construction consumed 1.354 profiled seconds.

The initial pending-save profile had a cold assertion rewrite (17 modules / 8,766 asserts; 9.31 s rewriting), with 19.94 s total collection. Warm profiled collection was 5.02 s. Cold/warm differences are not credited to these changes. Remaining work includes mandatory conftest/import initialization and actual UI interaction; sharing processes/apps or weakening config guards/GC was not attempted.

## Initial local verification

- New collector regression failed before the option because pytest visited an unrelated conftest.py file. With the option it passes, including actual package import and both fixture teardowns. An initial fixture-request test scaffold omitted node.name; fixed before the final run.
- 19 passed, 1 expected skip in 12.96 s: Tests/CI/test_private_profile_collection.py, Tests/CI/test_private_profile_child_deadline.py, Tests/test_private_profile_failure_report.py, Tests/test_private_profile_coverage.py. The skipped function is the intentionally child-only hung test; its parent actually launches it and verifies timeout/stack reporting.
- Coverage checks execute serial, xdist, standalone and disabled modes; serial and xdist retain all child-only lines and both branches (100% 2/2).
- 9 passed in 107.31 s: the category grouping check, all six mounted category sweeps (every category at 120x35 and 80x24), the Connect node above and pending-save [later-appearance].
- 106 passed in 27.98 s: both TASK-40750 governance cases plus Tests/CI/test_ui_pr_gate_census_checker.py and Tests/CI/test_ui_lane_shard_count_statements.py. Existing census coverage and shard contracts remain intact.
- Ruff passes for the new regression and the prior governance/census edits; all three startup files pass formatting. Direct linting of the two existing startup files has exactly the same three pre-existing async findings as HEAD (two ASYNC240, one ASYNC109), verified by code/message comparison. The project's required Ruff ratchet passes: 1,096 findings against a 1,105 baseline.
- Self-reviewed the complete diff, including pytest's explicit-path bypass in both Dir.collect and Package.collect. git diff --check and task ID/frontmatter checks pass.
- The initial local phase did not include a Linux/server benchmark; subsequent Linux verification is recorded below. No full-suite run, hosted workflow, deployment, push or merge was performed by this task.

## Reproduction

Use the project's Python environment and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. Functional commands:

python -m pytest Tests/CI/test_private_profile_collection.py Tests/CI/test_private_profile_child_deadline.py Tests/test_private_profile_failure_report.py Tests/test_private_profile_coverage.py -p pytest_asyncio.plugin -p pytest_timeout -p no:cacheprovider --timeout=90 -q

python -m pytest Tests/UI/test_settings_category_sweep.py::test_the_sweep_groups_cover_every_category_once Tests/UI/test_settings_category_sweep.py::test_every_settings_category_renders_and_restores_focus 'Tests/UI/test_settings_pending_save_drafts.py::test_staged_save_acknowledges_only_saved_values[later-appearance]' Tests/UI/test_settings_connect_rows.py::test_provider_is_one_row_one_tab_stop_and_filters_by_name_or_id -p pytest_asyncio.plugin -p pytest_timeout -p no:cacheprovider --timeout=180 -q

For a direct-child startup profile, set TLDW_TEST_PRIVATE_PROFILE_NODE to the exact node ID and run python -m cProfile -o /tmp/private-startup.prof -m pytest NODE '--ignore-glob=*' -p pytest_asyncio.plugin -p pytest_timeout -p no:cacheprovider --collect-only -q. The root conftest must establish the profile before application imports. Warm once before comparing revisions. Collection boundaries in the paired experiment were pytest_collection and pytest_collection_finish; body duration came from pytest_runtest_logreport; total used the surrounding subprocess wall clock.
## Ryzen Linux confirmation (2026-11-13)

Completed the user's requested benchmark in coordination with the chat “Plan self-hosted CI migration”. Waited for its failed full gate to stop and for all nine runners to become idle; it explicitly held the next gate. Used only a separate fresh overlay and tldw-ci-startup-benchmark-40780.service. No controller, deployed runner, image, PR, network policy or LLM service changed. The benchmark guest received no CI credentials.

Environment: Ryzen 7 7700X host, Debian 13; fresh Ubuntu 24.04.5 guest with 2 vCPU / 6 GiB RAM, host cgroup CPUQuota=200% / MemoryMax=8 GiB. System Python 3.12.3 venv; pytest 8.4.2, pytest-asyncio 1.2.0, pytest-timeout 2.4.0, pytest-cov 7.1.0, pytest-xdist 3.8.0, Textual 8.2.8. Both revisions use the same installed environment and source base 1455ceeb96c7c113a569a49a126fd287d4ac593b. This verifies Linux behavior, not the exact setup-python 3.12.16 deployment interpreter.

The harness runs the real wrapper, rather than adding the ignore flag manually to a direct child. Each trial first warms bytecode with collection only, then starts a fresh parent which launches a fresh private-profile child. JSON receipts assert exact selected node, child source path, expected before/after argv, separate parent/child PIDs/profiles, unique child profiles across trials, and passing setup/call/teardown phases. Order: before/after/after/before/after/before/before/after for each Settings case; before/after/after/before for governance.

| Case and metric | Before mean | After mean | Reduction |
| --- | ---: | ---: | ---: |
| Connect child collection | 2.968 s | 1.828 s | 38.4% |
| Connect parent pytest call (includes child lifetime) | 12.309 s | 11.348 s | 7.8% |
| Pending-save child collection (collector-only change) | 2.044 s | 1.930 s | 5.6% |
| Pending-save parent pytest call | 9.210 s | 9.071 s | 1.5% |
| CSS ownership parent pytest call (TASK-40750 plus startup change) | 25.490 s | 4.048 s | 84.1% |

All optimized Connect collection and call measurements were below every baseline measurement. Collector-only collection ranges also separate (before 2.033–2.067 s; after 1.903–1.949 s), but whole-test ranges overlap: its roughly 0.14 s mean call reduction is modest, not a demonstrated large whole-test speedup. The Mac collector-only estimate of 0.38 s does not carry over to Linux; Linux child collection saves about 0.115 s here.

Fresh-parent process totals also fell: Connect 17.290→15.189 s, pending-save 13.179→13.068 s, governance 27.061→5.625 s. Do not multiply those totals' savings by every CI case: ordinary CI amortizes its parent collection across many tests. The parent pytest call metric above is the closer comparison to the existing CI duration rows.

The added CSS regression cost 2.35 s in Linux verification. Against the measured 21.44 s reduction of the original CSS test, that leaves roughly 19.1 s net case-call savings for that test-only optimization in these runs.

Evidence: Docs/QA/2026-11-13-private-startup-linux/evidence.json retains all 20 measurements, all 40 parent/child process receipts, source hashes, package versions, host/guest metadata, verification output and exact benchmark/guest harness sources. Original raw console and compressed results also remain under /tmp/tldw-private-startup-linux-40780 locally; the host console is /var/lib/github-qemu-runner/startup-benchmark-40780/console.log.

All 20 measured trials passed. Final targeted verification: 27 passed, one intentional parent-side skip in 64.83 s. Coverage exercised serial, xdist, standalone and disabled modes; timeout/stack diagnostics, sibling pruning, package/ancestor fixtures, fresh profiles and complete Settings category coverage at both sizes passed. The skipped function is the intentionally child-only hang case, actually executed by its parent regression. No checks were relaxed to pass on Linux.

Setup started 19:41:02Z; timed phase started 19:42:24Z; verification started 19:48:42Z; guest completed with exit 0 and powered off at 19:49:48Z. Confirmed unit inactive at 19:50:25Z and explicitly released the window to the rollout chat. Host sampling during measurements showed no swap-in/out and mostly idle CPU outside the guest. After saving the evidence and confirming the unit was inactive, removed only this benchmark's disposable overlay.qcow2 and seed.img. Its console remains; no guest source or image was promoted.

Limits: one idle host/guest, four before and four after samples per Settings case, two each for governance; no loaded full-gate throughput, p95, full-suite or multi-worker contention claim. Settings startup and CSS savings clearly hold on this Linux setup; collector pruning is a smaller improvement. No production code or additional optimization was needed after Linux verification.
PR preparation: origin/dev remains 1455ceeb96. Independent read-only review found no actionable issues and reconciled all benchmark hashes, 20 trials and 40 process receipts. Fresh combined verification on the final tree: 134 passed, one expected skip in 81.14 s; five changed Python files pass formatting, new/clean files pass Ruff, and the required Ruff ratchet passes. Full-CI comparison is coordinated with rollout PR #3529: open this work as a draft and cancel only its own initial automatic runs, then rebase onto the merged rollout revision and measure a fresh pull_request run with the same lane layout/runner limits after a successful baseline. Cancelled runs are not measurements; no auto-merge is armed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
