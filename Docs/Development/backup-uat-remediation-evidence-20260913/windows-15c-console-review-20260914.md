# Windows 15c Console preview causal review

Exact run34881797168/revision `15c1541f77780c7897a11ba136d738bad679e9b4`: existing verified receipts report146 artifact hashes,42 native passes,275 product outcomes (**272 passed,3 failed**, zero skips/missing), and two installed2867-file receipts with2475 matching Git Python blobs each. Console is the154.836s failure; Settings/Library are separate investigations. Product suite time2003.309s, native4.501s. I read the receipts and Console artifacts without executing tests, opening databases, or changing source.

Artifact: `/private/tmp/uat-windows-15c-support/backup-platform-windows-2022-py3.12-support-diagnostic-15c1541f77780c7897a11ba136d738bad679e9b4`. Console evidence is `test-logs/product-pytest/test_mounted_console_complete_0/home`; full JUnit failure is `/private/tmp/uat-windows-15c-support/junit-failures/6526b41a87beaaa0.txt`.

## Established failure

`mounted-child-failure.json.log` retains an ordinary `AssertionError` at embedded main line70, the Complete-preview assertion. `mounted-capture-review.log` has one preceding exact snapshot refusal:

- reason `preview_sqlite_changed`;
- terminal `storage_admission.py:1504`;
- member **wal**;
- phase **opened_state**;
- changed fields **mtime_ns, ctime_ns** only.

Thus the captured initial WAL stat tuple differs from the descriptor's opening stat tuple in timestamps, with device/inode/size unchanged in that comparison. This is before that WAL's payload copy. It does not prove a replaced file, growth, changed WAL bytes, or who changed its timestamps. The current diagnostic intentionally records no values or writer identity. Main-file private copying may already have occurred, but the failed source set is not accepted/cached as a verified snapshot.

The single resulting inventory marks `db.chachanotes.primary` unavailable. Seven explicit dependent owners reference it: `chat.attachments`, `db.agent_runs`, `db.evals`, `notes.sync_bindings`, `persona.visual_identity_builtin`, `quiz.local`, `study.local`. Preview returns `('dependency_unavailable','unavailable','undeclared_alias')`. There is no earlier inventory to compare: `scope_changed=false`/empty delta here means first observation, not stable bytes or Complete coverage. No capture/start/maintenance stage follows, so missing runtime-settlement/recovery-failure logs are expected. The child completed construction at21.250s and yielded mounted UI at47.156s; case duration includes subsequent work and cleanup, not one measured preview duration.

## Alias route and source limits

Reviewed exact15c `DB/recovery_core.py`, `Backup_Recovery/inventory.py`, and `storage_admission.py`; they are byte-identical to the inspected current files.

`_CoreAdapter.discover:27–45` validates the included Notes database and changes it to unavailable on validation issues. Its returned item retains the physical path but receives the explicit shared group **only when status is included** (`:113–118`). `_Attachments.discover` (`config_adapter.py:841–854`) independently declares the same configured Notes path and its explicit shared group. Other installed Notes cohort owners similarly share that store. `_merge_chachanotes_cohort` (`inventory.py:720–774`) only relabels included cohort rows, and skips unqualified core rows. `classify_entries:183–213` still places an unavailable path into its physical-identity bucket; multiple rows with a missing/different group yield `undeclared_alias`.

That is a concrete source route by which the refused core snapshot produces a secondary alias issue for a legitimate shared Notes file. It is not evidence of a newly created filesystem alias or grounds to weaken alias validation. The bounded inventory output does not serialize every row/status/physical bucket, so it does not identify the unique offending pair; the code route explains the observed aggregate, rather than proving a particular pair from a retained identity witness. The primary snapshot refusal remains independently sufficient to block capture.

## What remains unknown / minimum reproduction

Three retained thread snapshots show startup/config/native acquisition and later background waits; none identifies a Notes write/commit at the snapshot mismatch. They are periodic samples, not a transaction history. A concurrent legitimate writer or native timestamp update is plausible, but not established as the culprit. There is no evidence for an owner cache leak, a deadline failure, an authority defect or a need for an allowlist change. All1869 observed admission-group calls completed without errors; slowest was23.107ms, total3.579278s inclusive. Those counts do not explain the WAL timestamp mutation.

A bounded deterministic extension of the existing native main/WAL race regressions is justified if the aggregate behavior needs reproduction: run the real complete inventory/private-preview path on a synthetic native Notes store, schedule one actual native Notes transaction at the existing source-state→WAL-open boundary, and preserve the current refusal plus temporary-copy cleanup. Check the resulting core/dependency/shared-cohort classifications, then perform one separate fresh preview after the writer completes and require normal Complete classification. Do not inject an arbitrary alias, bypass owner validation, retry inside product code or relax timestamps. Existing `test_thread_diagnostics.py` already supplies the exact native opened-state seam; using it with real owner inventory distinguishes race classification from a made-up alias test. This would reproduce the mechanism, not identify the writer in this Windows run.

This differs from65a Console, which verified its archive and then failed through a wrapped unsupported raw-helper error. It is related to earlier789960 initial-opening and562530 post-copy snapshot refusals, but this is the first retained Console evidence here specifying **WAL timestamp fields at opening**. Prior failures remain preserved; no claim that they share a writer or are fixed follows. No product correction is proven by this read-only review.
