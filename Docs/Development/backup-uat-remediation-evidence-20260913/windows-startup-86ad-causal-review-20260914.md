# Windows startup 86ad9993dc: bounded causal review

The context-policy lifetime correction has not resolved the Windows acceptance failure: all four fresh-process GGUF cases still reach the unchanged pytest60-second timeout; the enclosing step exits1. This retained log contains the primary test step, not a new cProfile diagnostic or successful completion. No repository changes, tests, network or remote operations were performed for this review.

## Exact terminal evidence

Run34864871343; supplied revision86ad9993dcffef710e9cd250665c1bb8314a67dc. Local log `/private/tmp/uat-windows-startup-86ad9993dc.log`, SHA25639d105f52f915b67c98f67f49749612de3e6a4786020376b6e76391f0cadf5b9. Lines4–9 enumerate the four cases and fixed `--timeout=60`; each collects one test with timeout method thread and func_only=False. Final step exit1 is line1230.

| Case in test_llm_gguf_source_modes.py | Main-thread work at terminal timeout | Evidence lines |
|---|---|---|
| test_claim_authority_survives_screen_recompose_and_not_window_selection | Asynchronous LLM recovery wrapper entering `_using`: operation.check → history/generation witnesses → fresh bootstrap/private-record observation → Windows native handle cleanup |274–299|
| test_external_copy_keyboard_geometry_and_unrelated_views_stay_stable | Deferred Collections capture wiring → CollectionsOfflineStore cursor initialization → LibraryCollectionsDB transaction/connection guard → fresh native storage check |565–600|
| test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamacpp-llama-cpp] | Deferred Collections capture wiring → get_user_data_dir → config_data operation acquisition → final fresh scope/bootstrap observation |866–903|
| same test[llamafile-llamafile] | Local-server setup-card discovery worker → ordinary LLM recovery operation initialization → generation/association observations → native private-path/ACL metadata |1199–1228|

These are snapshots of active work at the test deadline, not measured durations of those functions. They do not demonstrate deadlock, one native call taking60s, a policy/permission refusal, or a failed GGUF behavior assertion. The first asynchronous wrapper stack does not retain the caller sufficiently to label its particular operation. No completion receipt proves how far the test assertions themselves progressed. Some source-line text in the thread dump does not align with adjacent function frames; function/frame chains establish the boundary, not a claim about a particular displayed source statement.

Collections wiring is already scheduled by the existing post-mount timer (`app.py:17053–17056`), and its body remains synchronous (`10035–10094`). Local-provider discovery is an existing asynchronous Textual worker (`chat_screen.py:15231`) whose recovery wrapper performs synchronous native qualification before entering the coroutine. Thus current timeouts include deferred startup/readiness work, not merely the constructor.

## What changed, and what can be compared

Commit86ad changes only `_global_context_policy_overrides` product behavior: one existing `operation(config)` encloses the original nine consecutive getters; parsing still occurs after retirement. Its committed checkpoint is relative to562530e4b5 and records a real alternating local per-call comparison:9 acquisitions before,1 after, with equal output and native guard/lifetime regressions. That proves the local structural reduction, not total Windows savings. There is no operation-lifetime decision cache or removed check in this change.

Current unprofiled constructor logs are9.819/9.605/9.606/9.646s (lines64/345/646/949). Comparable retained5af primary-step logs are9.566/9.433/9.337/9.270s. Both primary runs time out in all four cases. The small cross-run constructor difference is not a controlled regression measurement and provides no speedup evidence; the changed method principally concerns later Console policy reads. Retained5af primary timeout stacks already include local-server discovery and deferred Collections wiring, so current snapshots do not establish newly achieved progress either.

Keep the separate5af instrumented result distinct: dev passed57.10s, candidate timed out; candidate last cumulative sample reported249 acquisitions,1275 registry reads and133414 native opens. Its23.634s constructor was under profiling, versus0.607s for instrumented dev, and its final sample remained in initial Console composition. Those inclusive totals overlap; they cannot be added or compared directly with the current unprofiled9.6s constructors. The86ad log has no new call counts or profiler records. The retained562530 checkpoint/review establishes the local baseline and correction, but supplies no paired native Windows before/after measurement.

## Decision

The remaining failure class is still failure to finish the full-app tests within60s while native backup authority work is active on the event-loop thread. The sampled callers narrow the outstanding measurement to deferred Collections configuration/DB checks and local-provider generation checks. They do not prove which dominates elapsed time or justify removing any check, extending a config lifetime across a different owner/await, adding a cache, or changing the limit.

Smallest next validation is inspection of an already-produced exact86ad diagnostic, if retained, against the existing5af diagnostic under identical observer settings, separating constructor, composition and deferred callback work and counting the changed policy method's calls. No new run is requested or performed here. In its absence, Windows benefit remains unmeasured and acceptance remains failed. Prior late-record/private-registry mutation proofs still rule out merging or skipping the independent fresh observations.
