# Windows bd6 support independent review

Exact run **34908697549**, job **104191144325**, tested clean source **bd6ba3b126588a59699308f71091e7a741bb83d3**. The Windows execution step ran 2026-09-14 23:25:50 through 2026-09-15 00:04:01 UTC. Run/head/status were checked by exact run ID; downloaded artifacts were independently verified with both existing verifiers. No app tests, source changes or Linux access were performed.

## Verified accounting and provenance

- All **163 artifact SHA256 entries** match. Source receipt is clean and exact bd6.
- **42/42 native passed** (1.551s); **303/305 product passed**, two failures (2238.416s). JUnit exists for both phases; **zero skips, errors or missing outcomes**. Intentionally unselected workflow jobs are not pytest skips.
- Two installed receipts each contain **2867 files / 2475 Python source comparisons**, all matching. Independent Git blob comparison matches **2443 CRLF-converted + 32 exact Python files**, no mismatch. This verifies the available receipt comparisons, not a fresh remote filesystem reread.
- Wheel hashes: fixture0 `574b0ffd97cdb7b94eb07b248953c13aee86961957d2d475004ae94c7e753171`; fixture1 `04222ee09c24b0504d243ea8a56768b0773c6ea34402c46a49cb91f884e40997`.
- All **12 new test_sqlite_inside_config_scope cases passed**: cached, reopened, getter, pause_before, pause_during, error, cancel, error_selector, cancel_selector, selector_after_open, failed_close, custom. Individual times are 4.664–6.314s and are retained in case-accounting.json. These native subprocess cases exercise actual scoped SQLite/config ownership, refusal, exception/cancellation preservation, restoration, uncertain-close retention and custom-helper rejection; they are source-scope regressions, distinct from the installed mounted cases.

| Installed mounted case | Outcome | JUnit duration |
|---|---|---:|
| Console | FAIL, preview unavailable | 212.575s |
| Settings | PASS | 218.580s |
| Library then Settings | FAIL, capture admission_timeout | 358.566s |

## Console: exact first refusal

`mounted-capture-review.log` retains two `ValueError(preview_sqlite_changed)` events at exact tested `storage_admission.py:1504`, through `observed_snapshot:514`. Both report **WAL / opened_state / mtime_ns and ctime_ns changed**. Device, inode and size do not appear among changed fields. This is the pre-copy comparison of the newly opened WAL descriptor with the previously observed five-field state, not the post-copy check and not the old post-archive WorkerFailed.

The ensuing single preview inventory has unavailable `db.chachanotes.primary` and seven unavailable dependencies: attachments, AgentRuns, Evals, notes sync bindings, builtin Persona identity, quiz and study. The embedded main then raises AssertionError at line70 with `dependency_unavailable, unavailable, undeclared_alias`. No archive capture was reached. `scope_changed=false` on this first inventory does not prove a stable source snapshot.

The alias issue is consistent with the existing downstream classification contract: physical identity comparison includes unavailable rows, while an unavailable core cannot claim a successfully validated shared cohort (`inventory.py:180–213`). It is not evidence of a separately introduced foreign alias or justification to preserve unproven cohort authority. The WAL writer and reason for its timestamps changing are **not identified** by these bounded observations. This matches the nearest refusal seen in 15c; it differs from 10cc's config/SQLite reopen failure. Passing the 12 repair regressions does not make this distinct snapshot refusal a pass.

Console has no runtime-settlement log because failure precedes capture. Its slowest completed group was only 29.1117ms (one root); all 2005 group observations completed without error. No slow-group explanation of the WAL mutation is established.

## Library: concrete drain owner, multiple advancing native callers

Library mounted and loaded, F4 completed, preview was Complete, and the two inventories (preview and capture before maintenance) have no deltas or unavailable dependencies. Capture then failed, with exact original `AdmissionTimeout` trace `capture_service.capture:220 → admission._admit:600 → _lock:173 → _check:188` (exclusive lease acquisition). The embedded assertion is line83; this is neither the 420s outer subprocess timeout nor a scope_changed result.

Runtime metadata independently identifies **SchedulerLoop._maintenance_drain**, returning false after **29.485s**, causing `RecoveryRequired(runtime_work_not_settled)` at `_settle_stage:54`. Startup retirement was false; cache retirement was never reached. At failure the recorded state remains five pending acquisitions, one core operation, one raw operation, ten live leases and one startup lease. These are observed retained counts, not proof that each is independently defective.

Four bounded thread snapshots provide a more specific three-role chain than the earlier two-role synthetic probe:

- **7008 heartbeat**: `_record_heartbeat:568 → default_heartbeat_path:165 → config_participants.operation:343`, waiting in the existing timed REBUILD/FILE lock loop throughout all four samples. Frames cannot identify which of those two locks from locals, so that distinction remains unknown.
- **7200 database-size worker**: `_collect_db_sizes:103 → _get_db_size:170 → get_chachanotes_db_path:9198 → _get_custom_database_path:9164 → get_cli_setting:8482`, within the native config operation. Initially sampled in native security/authority and scope work; later sampled in initializer condition wait. At settlement failure it had initializing_body active30.0s and authority_open20.406s. The original config operation holds its ordered config locks through this body.
- **3936 trace reader**: actual `current_graph_epoch:880 → Notes transaction.__enter__:23979 → native repository acquisition`, first in initializer condition wait, later progressing through native authority/scope checks. At settlement failure initializing_enter was active29.469s. This is **current_graph_epoch**, not the earlier probe's `LegacyTraceMaintenance.run_batch` callback.

The chain supports heartbeat completion being delayed behind a config operation doing native admission while another real Notes transaction competes. Samples show progress and changing wait roles; they do **not** prove a permanent lock cycle, identify every initializer owner continuously or establish the Windows scheduling cost's cause. The earlier local four-pass probe does not reproduce this exact three-role path or the Windows failure.

The slowest completed group observation is `_admit:541`, thread2600, 21 roots: **78.3885096s wall, 0.078125s own-thread CPU, 78.53125s process CPU**. Its nested token work is inclusive, not additional time. All2272 group calls completed without errors. This demonstrates long wall delay accompanied by work elsewhere in the process; it cannot alone assign that delay to a specific caller, algorithm, OS operation or guard.

If a further causal test is needed, the smallest source-specific extension is a bounded native interleaving with **the actual database-size config path getter, current_graph_epoch through run_owned_db_call, and Scheduler heartbeat**, recording actual initializer waits and config-lock ownership without replacing their results. Keep supported completion/local-pause paths and real Scheduler drain. A passing local probe still would not establish Windows performance. No guard weakening, cache, retry or deadline change follows from this evidence.

## Settings success and comparison limits

The exact installed Settings body passed all assertions: boot readiness, real saved note, F4, Complete preview, succeeded Complete capture, released maintenance, fresh resumed note write, before/after live content readback, coherent archive containing the pre-capture note and excluding the post-capture note, normal service teardown and no blocked network attempts. Five inventories are unchanged. Cache retirement reduced eight leases to the startup lease, then runtime resume recorded startup retired and zero live leases. The observer's `unrecognized_runtime_issue` label at successful resume is a fallback label, not a failed case.

Thus prior10cc Settings failure now passes, while its Library pass does not recur in bd6. The suite's two failures remain preserved. This run does not test subsequent Linux test-budget/import-only edits or any later source revision. Existing product and harness limits were not changed during this review.

## Receipts

Artifact root: `/private/tmp/uat-windows-bd6-support/backup-platform-windows-2022-py3.12-support-diagnostic-bd6ba3b126588a59699308f71091e7a741bb83d3`.

Verification: `/private/tmp/uat-windows-bd6-support/verification.json`, `git-blob-verification.json`, `case-accounting.json`, `bounded-causal-summary.json`; verifier stdout `/private/tmp/uat-windows-bd6-support-verify.log` and `/private/tmp/uat-windows-bd6-support-blobs.log`.

Full preserved JUnit failure bodies: `/private/tmp/uat-windows-bd6-support/junit-failures/6526b41a87beaaa0.txt` (Console), `/private/tmp/uat-windows-bd6-support/junit-failures/344f17c907ca7f60.txt` (Library). Bounded diagnostics are under the corresponding `test_mounted_console_complete_0/1/2/home` directories within `test-logs/product-pytest`.
