# PR3346 Perf Guard investigation

## Outcome

The exact failing latency test passes unmodified on the rebased branch. The same original test with diagnostic timing observers passes on that branch and pinned latest dev. No reproducible performance defect or causally supported product fix was identified. This investigation made no production/test/budget/source edits, suppression, skip, timeout increase, commit, push or GitHub mutation. The rebased CI run still needs its own verdict; these Mac runs do not clear or explain away the original Linux failure.

Rebased source: 79e659dcde6f5360edb0070f9d184713bd8584d1. Target dev: 62a169f906508ed02bd7546ab278d916638d1607. The target was extracted with git archive into this private directory. Imported package/test paths in the probe receipts confirm the intended checkout/archive in each child.

## Original failure

Owned input log: /private/tmp/mixins-pr3346-perf-initial.log, copied here as perf-initial.log; SHA256 in investigation-context.json. Perf Guard run 41395825942, job 125116815632: 12 tests passed; Tests/Performance/test_ui_latency_guardrails.py::test_destination_tour_stays_under_switch_budgets failed with f2 -> LLMScreen: 10.2s > budget. Budget: 10.0s. Child test call: 100.68s. The supplied log does not split F2 press, first arrival, rendering/settlement, CPU or background-worker time, so it cannot identify a causal owner.

## Method

Read gh-fix-ci and systematic-debugging skills, repository/user AGENTS guidance, TASK38544, app-shell architecture, ADR097 boot ratchets and relevant testing/live/backlog lessons. Existing user authorization covers confirmed CI fixes. Root owns GH status and other CI failures; no duplicate GH investigation was performed.

Ran only the failing node with its actual private_profile_test wrapper, real TldwCli.run_test and original destination tour. The wrapper creates a fresh child interpreter and private HOME/config/data databases. All runs use repository Python 3.12, no pytest cache provider, PYTHONDONTWRITEBYTECODE=1, unchanged --timeout=300, and distinct private basetemp paths. Exact argv/cwd/exit/duration receipts: current-plain.json, current-probe.json, baseline-probe.json. Paired logs and original child JUnit/logs remain under their explicit run directories.

The two diagnostic runs load private latency_probe.py via PYTEST_PLUGINS/PYTHONPATH. It delegates unchanged Pilot.press/Pilot.pause, the existing arrival helper and Textual Stylesheet.parse/reparse, recording elapsed times in memory. It changes no data, screen, waiter, interval, result, assertion or worker behavior. TLDW_TEST_CSS_CACHE=0 disables the test-only CSS cache per the repository lesson, retaining production stylesheet behavior. The untouched plain run came first. Runs were serial; root held other tests until timed measurement completed. Observer overhead and different host/cache/load conditions remain limitations; these samples are diagnostics, not a performance ranking.

## Results

| Run | Result | Pytest total | F2 press | F2 arrival/settle | F2 total | CSS sources | F2 parse |
|---|---|---:|---:|---:|---:|---:|---:|
| Rebased, unchanged | 1 passed | 48.81s | not instrumented | not instrumented | under 10s | not instrumented | not instrumented |
| Rebased, observers, production CSS | 1 passed | 45.36s | 0.577s | 0.506s | 1.083s | 42 | 0.0036s |
| Pinned dev archive, same observers | 1 passed | 85.23s | 1.168s | 3.925s | 5.093s | 42 | 0.0044s |

Both observed F2 runs had LLMScreen active when the arrival helper started. No stylesheet reparse occurred during F2; 42 sources is below the 56-source soft limit and 64-source cache limit. Every destination passed its unchanged 10s budget. Complete per-destination records: timing-comparison.json and each *-probe-timings.json.

The guard times from before pilot.press, then polls the expected screen and executes six 50ms settle passes with Pilot pauses. It measures more than screen-class arrival. The longer dev sample includes 3.925s in the original arrival/settle helper although LLMScreen was already active. This establishes what the timer includes, not which component caused the historical CI overrun.

source-comparison.json proves the guard, private-profile wrapper, app.py, LLMScreen, LabScreen frame and LLMManagementWindow are byte-identical to target dev and verifies archived bytes against Git. The shared CSS delta is 18 lines of mix-in selectors; the tour does not open that Workbench and both observed tours retain 42 sources. Existing LabScreen deliberately mounts its body after first paint; LLMManagementWindow mounts its initial llama.cpp body and defers other providers. These owners were inspected, not altered. No CSS-cache cliff or new mount owner was found in the branch diff.

## Decision and limits

Do not weaken the budget, skip the destination, remove settlement or alter unrelated model-management code based on the single CI timing line. Retest through the rebased CI run. If the failure returns, capture phase timing on that runner before choosing a fix; the local observer is retained for that purpose. The original failure proves that run exceeded its budget; its mechanism remains unconfirmed.

This report does not assert a faster branch, flaky test, resolved CI, Linux timing guarantee or full-suite coverage. No full suite or broad performance sweep was run. The separate native TEMP/TMP/TMPDIR fixture collision is root-owned. TASK38544 and QA manifests were not edited here, avoiding concurrent root metadata work.
