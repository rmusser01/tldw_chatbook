# Console default-history selection verification

Task: TASK-34563.16. Base: 633e4faee1d6466a2fad0dfad8defa9e613e7e88.
ADR required: yes, existing [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md) applies; [ADR-126](../../backlog/decisions/126-complete-local-backup-and-recovery.md) source/native protections remain unchanged.

## Change

The exact default PromptHistory worker no longer repeats canonical selection and participant validation before the raw file operation performs them again. The worker retains its original source path and carries a private deny-only requirement into the existing raw scope. That requirement cannot grant authority and is checked before platform masking, including same-source nested scopes. Changing a queued default origin to a foreign custom path still refuses. Explicit paths and subclasses retain the original three-argument history call. No new owner, source cache, registry, permission bypass, history deferral or persistence mode was introduced.

## Verification

Original-body observation on Windows found four guarded canonical resolutions per warm append: one queued selection, one queued participant selection, one raw selection and one raw participant selection. The patch removes the queued pair; the original raw checks remain. The count regression first failed at queued_preselection=1 while the real append succeeded. Four refusal/compatibility controls passed. One profile fixture called an obsolete config owner before the intended boundary; letting the original history resolver observe the changed environment made the unchanged production control pass.

The first targeted integrated run (history-selection-once-green) passed 62 of 65 controls. The three failures asserted that the history method was never entered for a refused source; its raw scope now deliberately owns refusal inside that method. Their passive observer now follows exact original raw._file entry, preserving every byte, cache, draft, participant, pause and drain assertion. The corrected affected selection (history-selection-refusal-boundary-green) passed all 12 controls, including six new cases and six original default-history lifetime cases. All 65 behaviors are qualified across these runs. Actual held native reads/writes and repeated cancellation establish physical resource ownership; method-entry or process-tree observations alone do not.

All runs were sequential with private profiles and unchanged deadlines. Every listed native run retired normally, with zero forced retirement, identity overflow or lookup races. No full sweep or external provider request was run. Ruff reports zero findings in all seven changed Python files; AST and diff checks pass. Six files pass full formatter checks; raw_participants.py retains the same pre-existing formatter-only differences in unrelated sections. No new formatter difference was introduced. Independent source review is clear. The task remains In Progress under the repository's complete static/platform DoD; the overall latency target remains open.

## Measured limits and next boundary

An optional bounded scalar observer was added to the existing whole-Send diagnostic. It observes original code starts/returns/unwinds without replacing product callbacks, and pairs exact frame/thread/Task lifetimes. It retains only numeric IDs, times and whitelisted phase/effect names. The original default benchmark remains observer-free. Elapsed spans include awaits and overlap; they must not be summed as CPU or independent savings.

| Sample | Before history change | After history change |
| --- | ---: | ---: |
| Full Send 1 | 10.262235 s | 11.734794 s |
| Full Send 2 | 6.807843 s | 7.572805 s |
| Full Send 3 | 6.138163 s | 6.587664 s |
| History 1 | 1.5018 s | 0.3901 s |
| History 2 | 0.3678 s | 0.2936 s |
| History 3 | 0.4197 s | 0.4129 s |

These single diagnostic runs do not demonstrate total-latency improvement. The after run added six more original-code targets (12 total); neither instrumented run is speed acceptance. Both completed three actual saved turns, two non-streaming and one streaming, with full trace settlement, unchanged source, no unfinished/overflow/collision spans and a retired observer. Driver times: 63.328 s before, 68.516 s after. Run labels: received-intent-phase-before-1 and history-selection-phase-after-1.

The first diagnostic localizes a remaining stream-start to provider-entry interval of 3.933/2.814/2.244 seconds. Serialization itself takes milliseconds. The later diagnostic measures agent-reply orchestration at 5.936/3.098/2.269 seconds including reply/settlement; tool-provider composition alone occupies 1.049/0.698/0.641 seconds within that interval. These measurements direct the next source investigation to agent preparation, without changing required postcommit order.

Local evidence: .superpowers/sdd/2026-10-07-console-history-selection (static.json, source.json and baseline/current formatter diffs); shared checks/native-pairs directories contain the labels above, plus history-selection-once-red and history-selection-profile-original. No primary checkout edit, push or merge.
