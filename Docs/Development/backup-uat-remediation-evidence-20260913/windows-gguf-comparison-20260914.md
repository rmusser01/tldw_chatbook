# Windows GGUF CI comparison

Conclusion: this is not an observed dev baseline failure. The closest existing successful run has exactly the same source tree as dev 4631b60f8dd9623fc55bf16f4a37e29fcb1240c7. The current PR's four failures all terminate in backup admission/recovery filesystem checks. This implicates the backup integration/cold-start path, but the available single end-of-budget stacks do not identify a dominant call or separate the effect of changed test process isolation from runtime overhead. No timeout or guard change is justified by this evidence alone.

## Exact evidence

- PR revision 9e5914fca140e762b66ec4a7beb77cf2af450ca0: run 34825561902, Windows job 103916785039. Log /private/tmp/uat-ci-gguf-9e5914.log. Non-UI selection: 31 passed in 10.07 seconds. All four full-app GGUF cases hit the existing pytest 60-second case limit. Job completed failure after about 5m53s, below its unchanged 20-minute job limit.
- Closest preceding dev-ancestor run 34793585775, Windows job 103822304164, revision 70c817c2c36572eb111d0a60bd2824077817e8b6. Its tree and dev4631's tree are both fea617c3c540a44cdc6ec92889d3b8b2d5ed6fea; git diff is empty. Log /private/tmp/uat-ci-gguf-dev-70c817.log. All 35 cases passed in 140.82 seconds under the same per-case 60-second setting. Four UI call durations were 21.87s (claim), 19.73s (geometry), 42.16s (llamacpp), 34.90s (llamafile).
- Both runs: windows-latest resolved to windows-2025-vs2026 image 20260907.229.1, CPython 3.12.10, Textual 8.2.8. This is not the Windows 2022 support qualification job.
- Existing ae958 failure log /private/tmp/uat-ci-gguf-ae958.log also shows four full-app timeouts, including Console config admission/native_identity/NTFS at the deadline.
- Log SHA-256 receipt: /private/tmp/uat-ci-gguf-comparison-hashes.json. Exact run metadata: /private/tmp/uat-ci-gguf-dev-70c817.json; current check metadata: /private/tmp/uat-pr-checks-9e5914.json.

## Current 9e5914 timeout paths

1. Claim/recompose: main thread in LLM_Calls.recovery_review._using/check -> generation_witnesses._paired_witnesses -> bootstrap.startup_permission/_registry -> pinned directory/native fstat/info (log 734–759).
2. Geometry: recovery_review asynchronous -> _ordinary_operation -> acquire_storage -> _scope -> bootstrap._records/_control_records -> verified parent/native open (1010–1040).
3. llamacpp keyboard: ChatScreen coalesced control-bar sync -> operation(config) -> raw._scope -> acquire_storage/_scope -> bootstrap registry/records -> native verified-parent close (1346–1377).
4. llamafile keyboard: deferred collections-service setup -> get_library_collections_db_path/get_user_data_dir -> config_default_root operation -> acquire_storage -> admission_authority/qualified_for -> pinned-directory close (1644–1678).

These are concrete synchronous native admission/validation paths on the event-loop thread, not a GGUF model load or external process wait. The ordinary admission worker waiting on stop is expected lease lifetime behavior; it does not prove a deadlock. A single timeout stack does not establish that its leaf consumed the preceding 60 seconds, and no call-count/elapsed phase data exists to rank registry traversal, qualification, provider witness validation or UI scheduling.

## Comparison qualification and smallest next validation

Dev ran the four UI cases after the non-UI tests in one pytest interpreter. The PR intentionally runs each full-app case in a fresh interpreter with its profile selected before imports; test bodies retain their assertions. Therefore the existing successful dev log is a valid baseline pass, but not an identical cold-process benchmark.

The smallest controlled next validation is one actual GGUF full-app case (the claim/recompose case is sufficient initially) on the dev-equivalent tree and PR tree, using the same fresh-profile/process invocation and unchanged 60-second case ceiling. If detailed measurement is authorized, use bounded test-only monotonic phase checkpoints around import/construct/run_test entry/Models-ready/case/quit and aggregate entry count plus elapsed time for the already-observed acquire_storage, qualified_for, bootstrap records/registry and recovery witness boundaries. Preserve original return values/exceptions, restore wrappers, retain no paths/config/payload values, and emit no diagnostic IO while holding admission locks. This can distinguish repeated validation cost from one blocked native operation without disabling any guard. No such run or instrumentation was performed here.

No repository edits, dependency changes, new workflow runs, timeout changes, or authority changes were made.
