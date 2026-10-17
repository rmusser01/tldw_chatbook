# TASK-34406.11 — finite preparation handoffs

The original permission read is not one269ms disk operation. Qualified fresh-app profiles on current merged111f8e48 production record73.578ms wall/24.741ms thread CPU, then272.145/29.447ms. The latter includes313 posture stats,400 content stats,324 realpath lstats and27 mount-status calls. Individual GIL-releasing calls wait up to13ms while other interpreter threads execute. A temporary1ms handoff intervention inside that original read keeps every call count and yields26.175/16.253ms. It supports a scheduling convoy explanation; it does not isolate kernel I/O from GIL reacquisition or identify a specific native-lock holder. C monitoring misses BEFORE_WITH lock entry; that failed self-check is retained.

Those instrumented whole post-save spans1526.696ms and1523.393ms establish no release gain. Actual latency acceptance requires the separate unprofiled comparison below.

The production fix wraps only the existing finite synchronous callback in `Chat/console_preparation_reads.py`. It lowers handoffs to at most1ms while physical bodies run, reference-counts overlap using the existing bookkeeping lock, preserves already lower values, and restores the prior value after last exit only if still owned. A later different external value survives; identical external writes cannot be distinguished. The interval affects all interpreter threads during these windows. No permission/native source code, observation, freshness gate, read result, timer, deadline, worker pool or public setting changes.

Causal scheduling RED:6 failures/3 passes on original code (5ms instead of1ms). Initial GREEN26 passed. Final connected selection383 passed/1 structural skip, with stable selected sources and zero profile/network guard refusals. It includes original source/callback mutation, revocation, admission evidence, cancellation and physical lifetime controls. New actual-native cases preserve12/14 generation witnesses,15/19 execution-context checks,12 acquisitions and3 independent raw file admissions; actual issued leases and parent FDs retire. One existing UP049 on `await_finite_read[_Result]` reproduces on dev; new findings0, new test lint and touched formatting/diff checks pass.

Independent whole-branch review on111f8e48..18d02c7295 found0 Critical/Important/Minor issues. Exact unprofiled benchmark completed and fully qualified. The preregistered8-run B/C/C/B/B/C/C/B gate requires full matched settings/27 ordered schemas/headers, successful exact replies and verified traces, normal Quit/unmount, observed process/port closure and whole-Python source identity. Both post-save and Enter-to-HTTP medians must improve, with at least3/4 matched pairs improving for each. Otherwise keep unmerged. Required hosted CI and current merge policy apply after a passing gate; neither publication nor hosted CI was started because this gate failed.

Evidence is retained under the task's `pr-review-wave2` artifacts: `permission-wait-attribution-r1.json`, the three qualified diagnostic packets, `permission-scheduling-targeted-qualified-r1.json`, lint attribution and frozen benchmark protocol/manifest. Missing-key launch (no request), rejected C analyzer and failed lock-entry self-check remain separate failures; raw records are unchanged.

No full suite, population p95, Windows/native paint, OS-cold-cache, later-send/large-history/broad-provider, separate real-home census/app-child exit or arbitrary native-worker retirement claim.

## Actual first-send result — original gate held

All8 packets qualify: full supplied/expanded settings,27 exact ordered schemas/headers, HTTP200/exact replies, verified complete traces/checkpoints0, actual Quit/Chat+App unmount, driver/server exits0, observed owned PIDs absent/ports closed, unchanged whole-Python maps (baseline8607, candidate8609), no profiler/sampler. Expected Python difference is only the finite-read module and2 new tests.

| Boundary | Baseline median | Candidate median | Improving matched pairs |
| --- | ---: | ---: | ---: |
| Durable save → trace reservation |910ms|852ms|2/4|
| Enter → HTTP request |2.097606313s|1.688502500s|4/4|

Post-save pairs (baseline→candidate):943→1075,801→937,1197→763,877→767ms. Enter-to-HTTP pairs:2.302393→2.062940,1.866584→1.695522,2.943640→1.535327,1.892820→1.681483s. The whole first-send improvement is consistent in this small cohort; the post-save reduction is not. The preregistered gate requires3/4 improvements for each metric, so it fails. Do not relax that gate after inspecting the data. Keep the reviewed implementation on `codex/console-permission-filesystem-waits`, unpushed with no PR/merge, pending further post-save work or an explicit change of release criterion. Task remains In Progress.

Exact measured production/test head:18d02c72955b71f206313cb8f10187346fbf4b1e, compared with current merged111f8e48fc14dccd654f691396fca1a9d704236c. Root full qualification/statistics: `permission-scheduling-paired-qualified-r1.json`; all raw/adverse packets retained. Final closeout changes only Markdown; no new performance claim follows.

## Explicit release criterion revision — 2026-10-17

After reviewing all four matched pairs, the user explicitly accepted the consistent Enter-to-HTTP improvement and requested CI, rebase and merge before further post-save profiling. This supersedes the extra both-boundaries release hold; it does not change the preregistered gate result or any recorded observation. Enter-to-HTTP median improves by409.104ms (19.5%), with4/4 pairs faster. Post-save remains inconsistent (2/4 pairs faster); further profiling remains necessary. Required CI, resolved review threads and the reviewed implementation's permission/freshness/native-lifetime checks remain mandatory. Independent whole-branch review substitutes for unavailable Qodo under the user's existing batch authorization.
