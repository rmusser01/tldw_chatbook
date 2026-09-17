# Exact2031d Windows startup audit

Read-only local review of run34939974091/job104286174204, checkout2031d0c5261c2b4781aba40752f228da9d19b4eb. No apps/tests, repository edits, or network operations.

**31non-UI cases pass in10.54s; all four original UI cases time out at their unchanged60s pytest limit.** These are claim-authority/recompose, external-copy geometry, cpp keyboard, and llamafile keyboard. The primary step exits1. The failure-only diagnostic likewise exits1; continue-on-error does not constitute candidate success.

| Same keyboard case | Pinned dev | Candidate |
|---|---|---|
|llama.cpp|Passed with2warnings:51.17s session,42.32s call|Timeout; last60.031s sample still inside mount:114|
|llamafile|Passed with2warnings:43.20s session,39.39s call|Timeout; last60.016s sample still inside mount:114|

Dev revision4631b60f8dd9623fc55bf16f4a37e29fcb1240c7. There are23 optional monitoring records (6dev/6candidate/5dev/6candidate), allsource_matches=true,0observer/configerrors,0dropped-active maximum. The observer Git blob is byte-identical to a523 and63756, SHA256973e6d8bc973b29ac52f62a8da40328bbf45b33cbd1462ff632fc92a16207c48. --no-profile remains active: cProfile disabled, no cumulative native-call/self-time profile supplied. Monitoring timings include instrumentation overhead.

Both candidate test cases remain at test_llm_gguf_source_modes.py1367, waiting inside _mount_models114 (`await app.push_screen(screen)`). No completed mount, settle, focus or close is recorded for either candidate. Final config_operation counts480/456 show observed operations occurred; they are neither unique acquisitions nor native-open totals. Dev mounts complete13.437171s/8.890625threadCPU and13.174343s/6.453125threadCPU; later focus helpers finish42/38steps and the full cases pass.

The final cpp thread sample passes through Scheduler emergency-stop/config getters and fresh raw/native stat checks; the actual terminal cpp stack instead includes consume_pending_console_provider_intent→session default refresh→provider readiness config→admission qualification. The file sample is in native handle info/control-record observation; its actual terminal stack traverses local Skills scope/trust-service construction→config admission→private parent opening. Thread samples taken while a test awaits a task are **not the suspended test's direct synchronous call stack**. Different sampled callers show ongoing UI/worker activity; without per-caller cumulative measurements they cannot be ranked as the dominant cause.

The slowest individually completed config observations are1.755028s wall/.84375threadCPU (cpp, thread9360; acquisition.890962/body-and-release.864066) and2.229171s/.234375threadCPU (file main6300; acquisition2.228784/body-and-release.000387). Acquisition includes authority validation/admission and cannot be equated with pure mutex contention. Nested observations cannot be added to infer whole-process CPU or total backup overhead.

Compared with a523, whose cpp comparison reached Enter and file comparison reached post-push pause, this run has less observed test progress at timeout. That is a run-level fact, not proof the bounded-layout correction caused a slowdown. The unchanged observer and passing dev establish the remaining candidate acceptance gap; these sparse records do not justify another product optimization, permission cache, bypass, or deadline change.

Backup run34940059565 is separate and was still active at assignment time; no qualification inference is made. Exact log hash/counts and bounded records are saved in /private/tmp/uat-2031d-startup-independent-summary.json. Whole-PR acceptance remains unclaimed.
