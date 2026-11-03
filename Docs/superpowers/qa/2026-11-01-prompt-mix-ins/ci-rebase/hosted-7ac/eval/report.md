# Windows retained-eval capture scope investigation

Source7ac214599d4969a4d1a57db8bef41ff02eb433c9. Read-only investigation; no repository source, fixtures, policies, timeout, approval, seal, authority or performance pins changed. Parent owns task38544 and hosted rerun/publication.

## Observed failure

Hosted Windows Eval Recovery run41482684811, original job125404838288, failed only `Tests/Backup_Recovery/test_eval_retained_definitions.py::test_complete_rebackup_retains_eval_after_source_and_candidate_removal`;81 passed and1 skipped. The failure is **initial capture**, not rebackup: `_run(original, "temporary", "complete", script=_RETAIN_PUBLIC)` reaches generated script line52 and capture_service.py201. Restore, source/candidate removal and second backup have not begun.

The main-thread preview was complete. Capture rediscovered selected sources on the worker thread and its review digest differed, so CaptureReviewRequired(scope_changed) correctly refused before maintenance/acquisition/capture. Exact changed owner/field is not present in available logs. The full899-line initial-child log shows normal app startup, seeded video/message writes, an ordinary ChaChaNotes worker transaction/checkpoint, then refusal. A TTS shutdown CancelledError is later cleanup, not the capture refusal's cause. Four hosted pytest aliases were verified byte-identical by parent; supplied exact artifact child log SHA256aab77fb34fbd1b2b7b1e4c6c629c4df3abbcb11519277adb525f336c2b2b80ff.

## Local evidence

One exact real retained-eval test ran with a private bootstrap profile and an observational plugin injected into the initial child script. It records both existing `_review_digest` observations, returns the original result unchanged, reads no payload bodies and hashes paths. It does not retry, change capture ordering or grant authority. The exact command/cwd/exit and elapsed time are in observed.json; plugin/runner source retained.

Result:1 passed in44.46s (full initial capture, isolated restore/removal, reopen and rebackup journey). Both initial inventories are complete with193 rows; reviewed and current inventory/review digests equal; normalized owner/id/status/dependency/shared-group/metadata scope rows equal. Metadata times/modes are omitted from the observational comparison because the scope digest does not bind them. Original production digest equality is separately verified. Local macOS Python3.12.11 cannot establish Windows behavior; instrumentation may affect scheduling. This is non-reproduction, **not a demonstrated fix**.

Static review: no Backup_Recovery production or test source changes between f095 and7ac. Inventory scope intentionally excludes ordinary payload/timestamp growth; SQLite transient WAL/SHM paths are explicitly declared even absent. Therefore normal DB write/checkpoint activity alone is insufficient evidence to blame WAL lifecycle. Monitor maintenance does not run producer retirement before a capture intent; this refusal precedes that intent.

## Disposition

No root cause identified for actual Windows drift, so no speculative code change. Preserve strict CaptureReviewRequired and all source/seal/authority checks. Parent launched one same-head Windows rerun, with no flag/guard changes. That hosted result is outside this report and must be qualified separately. If it repeats, failure-only body-free reporting of reviewed/current owner and changed scope-field categories would provide evidence missing from the current logs; it must retain the original refusal and perform no retry or reapproval.

All owned subprocesses stopped. No broad tests or full suite. The process used TldwCli constructors and real backup owners without a mounted UI; it is not performance evidence. Parent/perf owner handles separate timing qualification.
