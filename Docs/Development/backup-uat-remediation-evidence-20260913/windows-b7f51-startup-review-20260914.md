# Exact b7f51 Windows startup review

Read-only review of run34932736384/job104264181434, exact `b7f51bdc20e0355dce8ac630e0c1ddbe4fcdb8f3`. All22 extracted records match selected source, have empty observer/config-observer error lists, and report cProfile disabled. No app/test runs or repository edits. Complete extraction and bounded numeric/hash summary are adjacent `uat-b7f51-startup-profiles.json` and `uat-b7f51-startup-independent-review-summary.json`.

## Results

31 non-UI cases PASS9.72s. All four primary UI cases time out under the original60s limit: claim/recompose, external-copy, keyboard llama-cpp, keyboard llamafile. No completed primary passing call/session durations exist.

| Matched keyboard diagnostic | Dev call / pytest session | Candidate |
|---|---|---|
|llama-cpp|PASS40.99s /49.69s; two warnings|60s timeout|
|llamafile|PASS35.46s /38.55s; two warnings|60s timeout|

Dev mount completes in14.689919/10.632284s; focus traversal completes42steps in13.270312s and39steps in10.903951s, then both cases close and return. Counts mean completed press+pause appends, not attempted keys.

## Exact candidate progress

Both candidates still await `_mount_models` at enclosing testline1367 throughout the late records. Both progress from app context entry at helper112 (20–30s), through `push_screen` at114 (40–50s), to helper116 at60s. **Line116 is one of four `await pilot.pause()` calls after `push_screen` returned.** Thus Models-screen mounting has advanced beyond f311's final114, but the complete setup helper still has not returned. Neither has entered `test_settle`, `test_focus`, or `test_close`; actual keyboard assertions remain unexecuted.

This is more progress than f311 at the observation cutoff. It is less than the historical6b201 run, where both mounts completed in43.22/44.52s and both reached focus traversal1532. The recordings distinguish these boundaries; they do not support replacing them with one universal failure phase.

## Config work and final threads

Final candidate config-operation counts are772/695, user-directory28/28. f311 had561/514 and25/23 while still at114;6b201 had913/913 and32/32 after completed setup. These are entries including nested scopes and unequal completed work, not unique admissions or an aggregate performance comparison. There is no retained full-call timing for the changed guidance function in this observer.

At60.047s llama-cpp main records a running config operation aged0.029751s during `_sync_console_native_session_tabs → _ensure_active_console_session_settings → _maybe_refresh_stale_default_console_settings → _provider_readiness_app_config → startup/native authority`. At60.031s llamafile main records0.036752s during `_sync_console_chat_core_state → provider selection → fresh config/native checks`. The later timeout text catches both main threads in the Windows selector. Sampling differences indicate only different observations, not the duration or owner of a lock.

Diagnostic llama-cpp has seven default-pool workers in Textual timer waits and one DBStatus worker performing actual default-media-path/config/native checks. Diagnostic llamafile has six timer waits, one native lock/open worker whose truncated stack does not establish its owner, and a DBStatus Notes-path worker waiting at `config_participants.operation:343`. Primary pools likewise vary: keyboard llama-cpp has eight timer waits; the other primary cases include native work/config waiters. There is no proved queued timer cancellation, exhausted-pool deadlock, permanently retained config writer, or invisible-widget feedback loop. Full code-only extracted terminal frames are in `uat-b7f51-startup-terminal-frames.json`.

## Source-supported assessment

Git comparison with f311 confirms the only production file delta is `chat_screen.py`: `_sync_console_transcript_guidance` acquires one settings/readiness result and supplies it to its three presentation helpers; independent helper calls retain fresh acquisition. There is no native guard, policy-cache, deadline, or application-wide scope change. Source inspection supports removal of two repeated readiness derivations per such guidance pass, but this run does not directly count those calls or prove a lower total startup cost. Its unsuccessful end-to-end result also does not prove the bounded correction has no effect.

One remaining concrete adjacent pair is actually sampled, rather than inferred: llama-cpp's40.032s record reaches `_build_console_inspector_state:14277 → _console_provider_recovery_action → readiness`. Source14274 calls `_console_provider_blocker_copy()` and, when nonempty,14277 immediately calls `_console_provider_recovery_action()`, independently reacquiring readiness. This is a finite candidate for the same already-approved first-result/pure-presentation proof. Before changing it, establish that the first result and conditional presentation are equivalent with session convergence/errors preserved, no intervening effect, and a fresh next call. The artifact shows this exact caller executes; it does not establish an invocation frequency, expected Windows speedup, or sufficient overall fix.

The present failure is still setup/Pilot settling alongside native synchronization, not the prior post-setup focus traversal. Existing guards and60s limits remain intact. Separate current Windows backup/native completion and local semantic tests are independent evidence and are not reclassified here.
