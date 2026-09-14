# Windows recovered-profile observation review

Disposition: evidence supports a narrowly test-only Windows wait correction under the already-existing 240-second subprocess ceiling. It does NOT establish that console_quit or console_edit passed, nor that the separate 60-second startup regression is acceptable. No repository edits or native reruns were performed.

## Preserved evidence

Exact e683efb86433d3598a99e31033da38f778b59896, run 34835166453, verified artifact root `/private/tmp/uat-windows-e683-support`. Existing review verifies 48 artifact hashes and the installed source receipt. Current `Tests/Backup_Recovery/test_profile_open.py` and `tldw_chatbook/Backup_Recovery/profile_open.py` have no diff against e683.

The two exact nodes are `test_profile_open_requires_actual_mounted_local_reads[console_quit]` and `[console_edit]`. JUnit failures are respectively `junit-failures/15d6d04f1f5bec94.txt` and `junit-failures/3f776e196d81632b.txt`. In the artifact's `test-logs/product-pytest/test_profile_open_requires_act{0,1}/home`, both `open-child-error.json.log` files record TimeoutError at embedded child mount line 89, through asyncio/timeouts.py __aexit__ line 115. This is the explicit test `asyncio.timeout(20)`; it is not a product exception or the outer subprocess timeout.

| Child phase (seconds from child entry) | console_quit | console_edit |
|---|---:|---:|
| Constructor complete / mount begin | 33.453 | 27.344 |
| Initial run_test yield / recovery wait starts | 93.218 | 83.875 |
| Original local-read function begins | 142.359 | 127.813 |
| Original local-read function returns normally | 147.015 | 132.641 |
| Wait-to-read scheduling gap | 49.141 | 43.938 |
| Local-read execution duration | 4.656 | 4.828 |

The observer emits `local_reads_and_receipt_complete` only after the original `_check_local_content` returns without exception. The unchanged product function checks selected process/config/database identity and activation generation, executes actual Notes/Prompts/Media reads inside their native lifetime, checks generation again, and writes or flushes the exact authenticated open receipt before returning. Successful return therefore proves completion of that receipt-writing path. No `open-local-read-error.json.log` was collected. The receipt bytes themselves are not in this artifact; do not claim an independent reread of them.

There is no `recovery_check_wait_complete`, `receipt_read_complete`, or quit phase in either run. The timeout prevents the test from reaching unchanged config, receipt installation identity, explicit rail edit, and normal quit assertions. Those functional outcomes remain unverified until a fresh run reaches them. A callback completing while timeout unwinding/run_test cleanup proceeds does not turn the failed scenario into a pass.

## Scheduling evidence and limits

`app.py` schedules `acknowledge_mounted` through `call_after_refresh` after post-mount work and `_schedule_deferred_startup_work`. `acknowledge_mounted` then dispatches `_check_local_content` using asyncio.to_thread. Returning from the initial run_test barrier does not guarantee this later callback has been dispatched or completed.

The retained rolling four 10-second stack samples put the main thread in Console provider/readiness/control/transcript and Canvas policy/config-admission work, including native Windows opens, bootstrap registry reads, and raw/config participant checks; one edit sample is in skill-context admission. These are concrete competing synchronous startup paths. The samples lack individual timestamps and do not continuously cover the whole gap, so they do not establish a single dominant lock, total cost per caller, or whether all delay occurred before callback dispatch versus thread-pool dispatch. They do not show a native local-content functional failure. This remains slow startup/scheduling evidence and must stay associated with the separate performance investigation.

## Smallest proposed correction

Use the existing external Windows child deadline as the sole bound for this functional recovery-check wait, e.g. `asyncio.timeout(None if sys.platform == 'win32' else 20)`. Keep the loop and all assertions unchanged. The Windows parent already executes the entire child with subprocess.run(timeout=240); service.wait remains 255 and the outer fixture remains 300. This consumes the remaining existing whole-child budget rather than adding a new 60/135/240-second allowance after mount. A stall or later edit/quit hang still fails at that existing whole-child deadline. The parent retains its current timeout diagnostic/error propagation and process termination semantics.

Leave the Windows initial-screen-only 135-second observation barrier, explicit rail-persistence 10-second wait, non-Windows 20/45/50/70 bounds, all receipt/content/config/read-failure/quit assertions, production code, and separate GGUF 60-second acceptance unchanged. There is no reason to introduce a timer helper, compute a fresh child-wide allowance at mount, or relax the completion predicate.

This changes where a stalled Windows functional test is reported (outer TimeoutExpired rather than the inner TimeoutError), deliberately and explicitly. It is justified only because this suite asserts native restored-profile functionality and already has the finite enclosing execution budget. If this 20-second wait is intended as an independently owned latency requirement, reject the correction instead; it cannot preserve that latency requirement.

## Required validation before acceptance

Verify only the one Windows branch changes; preserve the literal original behavioral assertions and nesting budgets. Exercise original successful receipt and injected read failure with existing cases; then run both actual Windows console_quit/edit cases through terminal success, including receipt reread, exact config-byte preservation, explicit edit persistence and normal quit. Keep observer phase records and error identity on failure. Do not mark their prior failures resolved merely because their read callbacks eventually completed. The independent GGUF startup check must still enforce 60 seconds and report its current failure separately.

## Independent implementation review

APPROVED within the test-only scope above. Root's final file SHA-256 is `68f29c1f1278f80416847cb9458b641a7244efccc0a879b41a81577548cfb538`. Exact diff is one explanatory comment and the proposed Windows-only inner timeout expression. An independent AST comparison proved the entire module and embedded child unchanged after normalizing that single timeout argument back to 20; the child compiles. Thus receipt, negative-read, persistence and quit assertions and the external 240/255/300 bounds are retained. Ruff passes. Independent Bandit baseline/current counts are identical; no new findings. Reports: `/private/tmp/uat-profile-open-observation-{baseline,current}-bandit.json`.

This is source/static approval only. Root's local actual Console cases were still running when reviewed; the mandatory fresh native Windows outcome remains pending. No all-pass or fixed-latency claim follows from this review.
