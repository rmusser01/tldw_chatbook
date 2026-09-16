# Actual three-role native contention probe

**Four of four bounded local cases passed; no new causal failure reproduced.** No repository/product edits, app boot, remote access or altered native results/guards/deadlines. One four-case run, no retries.

Source checkout HEAD was `238564276d7efc84e0954ace75f4c6e62493db3c`. Read-only diff confirms the exercised DB-size, config, raw/native admission, Scheduler, Notes transaction, trace and finite DB-call source files are unchanged from Windows bd6. This is local macOS mechanism evidence, not installed Windows qualification.

Probe `/private/tmp/uat-three-role-native-probe.py`, SHA256 `be47efd1b7f680c5941655e3d7d80901ff5e96948b814630c2899663242c1967`; driver `/private/tmp/uat-three-role-native-run.py`. Fresh synthetic private profile per child. Raw receipt `/private/tmp/uat-three-role-native-results.json`; independently checked hash and matching fixed log markers in `/private/tmp/uat-three-role-native-verified.json`.

| Original initializer leader | Variant | Child result | Duration |
|---|---|---|---:|
| Database-size Notes path getter | complete | PASS | 2.787s |
| Database-size Notes path getter | local pause | PASS | 2.556s |
| Trace current_graph_epoch | complete | PASS | 2.642s |
| Trace current_graph_epoch | local pause | PASS | 2.616s |

## Native behavior exercised

The real `DBStatusManager.update_db_sizes` invokes its real `_collect_db_sizes` through `asyncio.to_thread`. A minimal app holder receives its ordinary result; there is no UI or timer. Its callback wrapper only supplies a thread-role label and delegates the original bound method. All three real path getters, filesystem size formatters, error handling and result publication remain intact.

The trace callback is the exact `TraceGarbageCollector.current_graph_epoch` under actual `run_owned_db_call`, with a native Notes repository and real transaction entry/SQL/readback/worker-handle cleanup. The Scheduler uses its real `_offload`, `_record_heartbeat` and default config-derived heartbeat path; no alternate path is injected.

The original `_Acquisition.initializing` is delegated with the same instance/root/path. A bounded gate at the actual Notes path-getter call permits both initializer leader orders while DB-size config locks are held. The chosen leader is held only inside the original successfully entered context. In each case, the contender is directly observed in exact native `Condition.wait` code with `self is storage._changed` and immediate caller code identical to the original initializer body. Heartbeat is observed in the original config operation with exact config source and the same existing REBUILD lock, while the real DB-size caller holds that lock. Three distinct native worker threads are asserted. No Condition, lock or native result is replaced.

Scheduler close followed by a deliberately blocked20ms drain refuses and retains its actual worker task. Releasing the gate then completes or refuses the actual work through original code. Complete variants require a real nonnegative integer graph epoch, successful DB-size mapping and heartbeat timestamp readback. Pause variants require exact `RecoveryRequired(storage_locally_paused)` from the trace; DB-size's original handler yields Error for the blocked Notes/Media fields and returns None; heartbeat's original resolution handler returns None without creating a heartbeat. No synthetic successful result is substituted for either handled refusal.

After terminal worker completion, real Scheduler drain succeeds, its owned-task set is empty, and native operation/pending/raw sets are empty. Resume then performs fresh actual DB-size collection, graph-epoch read and heartbeat write/readback successfully. All cases assert unchanged selected environment, no test-mode override and zero blocked network attempts. Cleanup releases both timing gates, awaits the real tasks, restores the original initializer/callback and closes owned repositories/startup normally.

Logs are retained under `/private/tmp/uat-three-role-native-owtlyas2/{sizes-complete,sizes-pause,trace-complete,trace-pause}/run.log`.

## Causal limits

This strengthens the earlier two-role run_batch probe with the exact third config reader and actual current_graph_epoch call observed on Windows. It demonstrates that either deliberately ordered initializer admission can complete and unwind locally without a permanent lock cycle or stale task ownership. The pause trace callback is attempted but its transaction body is not claimed complete; transaction read success is established in complete and resumed phases only.

The probe does not replay the full Windows app, timer cadence, full Scheduler tick/runtime settlement graph, populated trace workload, 30-second drain window or native security costs. It exercises one representative contention edge in both orders, not every later reacquisition exchange or cancellation schedule. Per-child30s watchdogs and short gate waits bound the diagnostic; they do not change any product limit. No Windows timing improvement is inferred from local2.6s completion.

The Windows bd6 Library failure remains unresolved and preserved: exact Scheduler drain timed out while heartbeat waited for config and other native owners progressed. These passes do not justify a cache, acquisition shortcut, new producer hook, ownership bypass, deadline increase or speculative grouping patch. Further intervention needs evidence of the actual long-running native/CPU contention mechanism, not another repetition of these now-passing four local cases.
