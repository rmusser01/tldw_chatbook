# UI26 readiness refresh retirement race

Source baseline7ac214599d4969a4d1a57db8bef41ff02eb433c9; simultaneous UI24 test-only fix is separate. Scope: `tldw_chatbook/UI/Console_Modules/console_spend_projection.py` and `Tests/UI/test_console_readiness_config_projection.py`. Parent task38544 records the diagnosis and approved fix. Routine lifetime bug fix within existing readiness ownership/source contracts; no new ADR. No timeout, admission, source, cancellation, proof, worker scheduling or size-pin changes.

## Confirmed cause

Hosted exact ancestry-size0 end-to-end response-rule journey closes its native session, then awaits a FULL presentation sync. An earlier scheduled readiness worker enters after that session is removed. `_refresh` calls `store.session_settings(key[5])` before its try/finally, so the actual store raises KeyError(Unknown Console chat session). The worker errors without clearing pending or signaling _settled. FULL's projection.warm waits indefinitely. Hosted log1850–2229 contains that worker traceback; the dictionary-backend error is a separate earlier diagnostic, not the cause of this wait. Later disposal awaits are secondary.

The exact unchanged native journey reproduced locally without fault injection:1 failed in45.38s, same WorkerFailed/KeyError. The private observation plugin records pytest phases and body-free async stacks; it changes no returns or source guards. Original command/cwd/results retained in original.json/log/xml and phases.jsonl.

A deterministic actual-store regression schedules warm, lets it create the real refresh coroutine task, then closes the actual session synchronously before worker entry. Against original code it times out waiting on the unset event after1s (1 expected failed in2.60s). No fake session lookup, owner mutation or source authority is used.

## Minimal correction

Move all existing `_refresh` setup into its existing try/except(cold refusal)/finally. The reader/tokenizer/settings lookup, read-request assignment, imports, runtime lookup and observer assignment retain exact original order before native admission. Existing exception handling returns cold and retryable; finally clears pending/read_request and sets _settled. CancelledError still propagates through finally. Physical reads retain their original lifetime/observer/currentness logic. Publication/proof checks and post-try publication body are unchanged.

AST proof confirms identical flattened setup/body, exception handlers, finally and post-try publication. The product file remains1866lines; no imports or thresholds added. No production log content changes.

## Verification and qualification

Final new race proves warm returnsFalse promptly, pendingFalse, settledTrue, no read request/physical reads, no value/native reader invocation/publication, and normal worker return. A subsequent current session with established canonical settings warms successfully and only that session publishes. Initial GREEN attempt created the retry session with settingsNone: existing pristine-default convergence advanced settings revision, correctly making warm returnFalse. The test setup now uses established settings/baseline, preserving the existing currentness contract. This intermediate1failed/6passed result is retained, not hidden.

- Original exact ancestry-size0:1 expected failed45.38s, same actual worker KeyError.
- Deterministic original race:1 expected failed2.60s (1s bounded unset-event wait).
- Unchanged physical cancellation (2), config-source retarget (1), runtime replacement (1), closed-owner (2):6passed in the intermediate7case command on final product bytes.
- Final deterministic race + unchanged exact native learn/repair/reopen/exclude ancestry-size0:2passed14.31s; original journey call10.92s. Original history/ancestry/rule/provider assertions remain unchanged.

No broader module/suite run. All owned subprocesses stopped and source frozen; perf owner notified for separate final five guards. All commands retain existing --timeout=180; only the new deterministic race has its own short wait-for failure bounds.

Static: product Ruff4→4 and test0→0; zero additions. Existing per-file formatting debt preserved, new test formatter-identical. git diff --check clean. Exact hashes, commands, source snapshots and semantic proof in static-proof.json, final-diff.patch and explicit report-files.json. Root owns gate enrollment, manifests, commits/push and hosted confirmation. Independent reviewer has settled source and evidence; separate review artifact records disposition.
