# Windows b560 Settings/Library causal review

Read-only against HEAD5af7381f659b264e97e702a2f3f5f3f6640649b0 and the supplied b560 artifacts. admission.py, storage_admission.py and runtime_maintenance.py have no changes between the tested revision and current HEAD. No product fix is yet supported by this evidence.

Established:

- Both pre-capture inventories are Complete and unchanged, with no unavailable dependencies. Both terminal errors are original AdmissionTimeout at capture_service.py220 → Admission._admit600 → exclusive lease _lock173/_check188. These cases do not reproduce the earlier core snapshot scope change.
- Capture has acquired native maintenance gates and waits for ordinary leases to retire. In both fixtures, multiple retained samples show the native pause worker inside _local_pause_requested485 → pause_requested702 → _groups364 → _tokens331–333 → Windows private object/ACL checks, while capture remains in its exclusive-lease wait.
- pause_requested683–719 reads and validates the registry/group, including all current/proposed/historical root relationships, before checking contended native gates. _local_pause_requested478–486 evaluates its retained holds outside the coordinator lock. monitor_app699–715 awaits that off-loop observation before constructing RuntimeMaintenance and starting producer settlement. This makes delayed pause observation a concrete candidate dependency for the lease timeout.
- Neither fixture has a runtime-settlement log. This is consistent with no recorded settlement reaching its first stage; unlike081 Settings, there is no observed SchedulerLoop false drain here. Absence alone cannot exclude observer failure, delayed loop delivery, or a different monitor state.

Not established:

- Repeated samples from the same pool thread do not identify whether one long _groups call or several calls are represented. These logs lack invocation start/end, completed counts and exact worker-completion-to-loop-return timing.
- Busy event-loop samples in Textual dispatch, bounded-section reconciliation and tray height adjustment show execution, not endless recompose feedback. Library's final sample is even in the event-loop selector. Stable Mac layout/reconciliation controls do not prove the Windows mechanism, and neither these samples nor those controls justify changing widget behavior.
- There is no native token/root count, per-call worker CPU or admission cost breakdown in these particular logs. They do not prove an invalid registry, permanent lock inversion, forgotten owner, or a safely removable guard.

Smallest next causal probe:

Use the already-existing aggregate admission observer attachment with native_calls=False. Its individual _groups records retain caller code (distinguishing pause_requested), thread ID, start/completion/error, wall time, thread/process CPU, bounded token-call totals and numeric registry/root counts, with periodic samples independent of settlement. These directly distinguish an active long pre-gate scan from completed scans awaiting scheduling, without path/SID data, retries, caches or product changes. Pair with the existing bounded main-loop profile only as corroboration.

Check that the five-second loop profile window actually overlaps the native lease wait: delay2s from service.start_backup can profile rediscovery before maintenance (Settings snapshot0 is still inventory). If its window misses contention, do not infer the absence of a loop contribution; first use the admission timestamps to choose one bounded window in the same phase. If group calls finish well before timeout yet settlement never starts, the next smallest extension is fixed timestamps/boolean or original error class at native pause-worker completion and coroutine return, not a broader UI audit. If one verified scan consumes the interval, reproduce that actual numeric topology/phase before designing any optimization.

Preserve all existing deadlines and native group/gate checks. As with previous diagnostics, stop/write errors must not replace the original pending service.wait error. No Linux logs, keyboard fixture, live SQLite or product files were accessed or modified. Evidence root: /private/tmp/uat-windows-b560-support/backup-platform-windows-2022-py3.12-support-diagnostic-b560d806b4f5601d0f760a00864cba546d167ef8/test-logs/product-pytest/test_mounted_console_complete_{1,2}/home.
