# Later-child observation review

APPROVED within the two test files. No actionable finding.

Independent focused suite: 11 passed in 1.01s; /private/tmp/uat-later-observer-independent.log. Two additional disposable guard probes passed: admission-observer setup failure retires the already acquired thread observer, and CancelledError identity survives both cleanup failures with both callbacks attempted. AST comparison proves the original main operation/assertion/timeout sequence is identical after removing diagnostic retain calls.

ExitStack registers each returned stop immediately; stop_observer preserves an active original error/cancellation while reverse cleanup continues. Observers span asyncio.run, including app teardown and executor shutdown. On an otherwise successful run, observer failure remains a test failure and reaches the existing bounded record_failure guard. No product behavior or native deadline changes.

The new timing file keeps only the latest 32 fixed checkpoint labels and numeric monotonic seconds. All actual labels are static test literals; no values/paths/credentials/results from retain kwargs enter this new file. Existing legacy output is unchanged. The *.log suffix matches the current sanitized collector. New markers distinguish catalog, selection, inspection, body, app-context and loop completion; periodic thread/admission logs survive a parent kill without relying on finally execution. Their last write can be partial if the process is killed mid-write; phase output is independent corroboration.

Limits: this is test-only observation approval, not Windows acceptance or a diagnosis of the remaining post-success wait. The parent retains the original 900s Windows bound. Hashes are in /private/tmp/uat-later-observer-independent-hashes.json. No source files edited by reviewer.
