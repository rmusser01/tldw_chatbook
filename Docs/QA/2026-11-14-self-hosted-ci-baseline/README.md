# Successful self-hosted CI baseline — PR #3529

Verified 2026-11-14. Normal pull_request Derived Artifacts run [43612634465](https://github.com/rmusser01/tldw_chatbook/actions/runs/43612634465), attempt 1, head `258ebec33d68a46a4f5d5a20838cce8dd6e2f627`. PR merged at 17:22:30Z (09:22:30 Pacific) as `d46f753d8287cffd48706f912bfd838dc8756edb`. The merged tree `e1c2f7c212e9dce84008dbd5f8bb17db952f0d33` equals the qualified head tree. Checkout logs identify the tested synthetic merge of that head into `1c14f0e9420b10e14084484b226bcd1ff9a0e381`.

| Metric | Measurement |
|---|---:|
| Workflow creation → required aggregate completion | **1h28m09s** |
| First job start → required aggregate completion | 1h26m50s |
| Workflow creation → last job completion | 1h29m42s |
| Workflow creation → final workflow update | 1h29m43s |
| First job start → last job completion | 1h28m23s |
| Executed jobs | 51 passed (48 test, 3 control) |
| Skipped jobs | 1 intentional event-specific skip |
| Total actual execution summed across jobs | 626.43 runner-minutes |
| Job-created → started queue delay, median / maximum | 33m18s / 1h12m01s |
| Test-job queue delay, median / maximum | 36m07s / 1h12m01s |
| UI shard 9 job / tests | 12m35s / 151 passed in 680.66s |
| Main project-chat UAT step | 2m28s; 74 passed in 146.02s |

Times are UTC in raw JSON. Workflow created 15:53:58Z, first preflight started15:55:17Z, preflight finished15:58:50Z, required aggregate finished17:22:07Z, last queue-tick job finished17:23:40Z. The 4m52s before fan-out creation is dependency/preflight delay, separate from each job's creation-to-start queue time. These durations overlap across jobs and must not be added to wall time. The sum includes the post-verdict queue tick. Every executed Derived Artifacts job used a `ghq-test-*` or `ghq-control-*` runner with the local labels; raw identities are retained.

UAT complete-evidence validation, missing-evidence negative control and artifact retention all passed. UI22/24/25 passed245/370/132 tests; UI9 includes both prior cadence and current coordinator fixes.

## Reproduction and comparison limits

Eight disposable 2-vCPU/6-GiB test guests plus one 1-vCPU/2-GiB control guest on the Ryzen7700X server; fleet CPU quota12 logical CPUs, memory cap64GiB, storage100GiB. Exact customized image SHA256 and upstream metadata are in resources.json and host-image.txt. setup-python installed CPython3.12.16 in the test jobs; some control work uses3.12.11. Per-job dependency install versions, pip cache keys and test summaries are retained in job-environments.json. Representative UI versions: Textual8.2.8, pytest9.2.0, pytest-asyncio1.5.0. Dependencies are not inferred from the local Mac environment.

The baseline overlapped its normal local Lint jobs and runner qualification43612630706. Qualification used the control slot15:53:59–15:54:56Z and a test slot15:54:00–15:55:17Z, then a renewed control16:02:15–16:02:19Z. The initial gate control queue includes that qualification. Hosted Perf Guard also ran; it did not consume local test slots. No optimization benchmark or PR3532 run was scheduled during this gate. This is one observation, not a p95 or causal optimization measurement. Compare final workflow topology, resources, image, dependencies, base source and overlapping work before attributing changes to3532.

The prior hosted7h52 run used another head and the migration repaired test fixtures and lane topology; it is historical context, not a controlled same-commit speed comparison. Earlier failed/cancelled rollout attempts remain separate and are not baselines.

Postmerge dev gate43620765153 began17:22:33Z and was still active at closeout; it is a different event/workload and is not included in the PR baseline. TASK-40998 subsequently extends local placement to the remaining dev jobs. This report remains historical; a fresh baseline after that deployment and qualification is required before the optimization comparison.

## Evidence

run.json and job-pages.json are raw API responses; jobs.json includes all52 jobs from attempt1. job-timings.json contains per-job dependency/creation delay, queue delay, execution and labels. summary.json, resources.json and job-environments.json hold derived facts. Complete original logs ZIP is retained at `/tmp/tldw-ci-success-43612634465/logs.zip`; selected extracted logs are there too. This closeout was held locally until #3529 merged to avoid restarting healthy CI, then included with TASK-40998.
