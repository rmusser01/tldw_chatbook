### Spec Compliance

- ✅ **Scoped fix compliant:** I1 is ADDRESSED. M1 and M2 disclosure is ADDRESSED. The corrected source satisfies the phase-local observation contract in the fix appendix; no new blocking source defect was found.
- ⚠️ This approval covers the I1 concurrency correction and M1/M2 disclosure only. The controller cap remains a blocking, separately planned structural gate. Publication, fresh-dev ancestry, external Qodo/CI/PerfGuard and overall merge readiness are not established by this review.

### Per-finding verdicts

- **I1 — ADDRESSED.** `tldw_chatbook/Chat/console_chat_controller.py:19600` captures the exact record/payload and source/runtime identities under the existing registry lock, releases it before `_read_chat_creation_source` (`:19306`), then reacquires it for `_chat_creation_record_locked` (`:19620`). The final validator rechecks record identity, exact payload, source/bridge/database identity, actor, incarnation, conversation, workspace, current primary/assistant/cancel ownership, Close and revocation against current memory while applying the original row predicates to the copied fields. Missing keys remain missing, preserving the original fail-closed missing-status behavior.
- **I1 — fresh phases and atomic decisions preserved.** The wrapper (`:19613`), remembered-grant path (`:19974`) and post-human-wait path (`:20108`) each obtain a fresh observation. Approval and grant insertion remain in the final lock section; token retirement remains outside. `_chat_creation_source_live` (`:19332`) reads anew for its existing preparation/native callers. The unchanged execution-entry and final-save callers use the corrected wrapper; no observation is persisted or passed across those boundaries. The requested frozen `console_agent_bridge.py:9321` excerpt confirms that `live_primary_run_id`, still used in the final section, is a pure dictionary lookup rather than storage or an external callback.
- **I1 — behavioral evidence is meaningful.** `Tests/Chat/test_console_chat_create_integration.py:1670` parks an actual AgentRuns connection in a worker while actual owner-thread Close/revocation proceeds, then requires denial, no retained record/grant and bounded worker termination. `:1745` checks unlocked reads for wrapper, ordinary confirmation and remembered confirmation, including surviving children. `:1848`, `:2060` and `:2098` cover runtime invalidations between row copy and final decision; `:1930` changes actual rows during the human wait. `:1786` characterizes the independent-writer observation boundary, and `:2021` separately proves terminal status at final save creates no durable row. The former is correctly not presented as a latest-row atomicity guarantee.
- **M1 — ADDRESSED as disclosure, not leak resolution.** `task-8-fix1-report.md:48` and `:49` explicitly retain the Task8 +265 and +233 descriptor warnings, their separate receipts and unresolved attribution. They are distinguished from the older Task7 +223 warning. The carried safe copies match their recorded hashes.
- **M2 — ADDRESSED as disclosure, not warning elimination.** `task-8-fix1-report.md:38` explicitly carries 58 startup/navigation passes with three warnings, records that they were not replayed, and lists the unchanged budget/headroom limitations. It does not claim that this earlier receipt covered the controller size row.

### Strengths

- `console_chat_controller.py:4767`, `:19306`, `:19620`: one private observation object makes the read boundary explicit without a new cache, authority flag, lock, getter contract or write policy. Copies contain only the existing row fields needed by validation.
- `test_console_chat_create_integration.py:1643`, `:2127`: both real primary and genuinely surviving-child positives remain covered, and the additional missing-status test closes the self-review regression while retaining its RED evidence.
- `task-8-fix1-final-validation.xml:1` and `task-8-fix1-final-callers.xml:1`: 24 plus 56 passes, zero failures/errors/skips; the corresponding before/after hashes are stable and match the frozen source map. Existing integration test functions are AST-identical in the supplied complete function map.

### New fix-round issues

#### Critical

- None found.

#### Important

- None found within the I1/M1/M2 repair scope. The existing controller cap gate below remains blocking for publication.

#### Minor

- **R1 — One immutable RED is described more precisely than its receipt supports.** `task-8-fix1-report.md:28` says all four blocked-read controls fail at the intended registry-lock barrier. In `task-8-fix1-final-immutable.log:602`, the subagent/revoke case fails first at `assert not errors`, and the exception contents are not printed. The other three cases do reach the explicit registry-lock assertion. Correct the report to distinguish three explicit barrier failures from the fourth worker-error failure whose precise cause is not exposed by this receipt. No rerun is needed to make that wording accurate. The three direct REDs, four corrected passing retirement controls and inspected source are sufficient for I1; this is an evidence-description issue, not a new production defect.

### Out-of-scope remaining gate

- **Controller size remains blocked:** `task-8-fix1-final-cap.log:9` records **33,027 > 29,367**. The immutable source already had 32,910 lines, and this fix adds 117. The report accurately acknowledges the remaining 3,660-line overage at `task-8-fix1-report.md:34`. This scoped approval neither waives that failure nor approves the planned structural repair, which has not been reviewed here. Full static/source-gate completion and publication readiness must not be claimed.

### Verification performed

- Read the re-review contract, fix appendix and report, then the complete immutable two-file diff in consecutive sections. Verified its **50,818-byte** size and SHA256 `ec64186f030200a33cb67cea7df759626ee55931eded9812aa0f0b2a6fd630fd`. The supplied source maps confirm BASE72c and reviewed40b9 have identical controller and integration-test hashes.
- Verified all **71** safe-copy hashes/sizes; checked the two final behavior receipts against the frozen source map and inspected their logs/XML. Checked the final immutable 5-failure/4-pass receipt, missing-field RED and explicit cap failure, with the R1 qualification above. No test was rerun.
- Checked the final fatal-Ruff, formatter, whitespace, diagnostic and worker reconstruction receipts. Maps retain all 34 Task7 function ASTs, 96 derived blobs and 11,792 existing QA blobs without mismatch. Startup/import/CSS carry remains explicitly bounded by the unchanged source/import maps rather than presented as a fresh sweep.
- One focused dependency check was requested: whether the final primary-run lookup could still perform storage under the lock. The frozen 805-byte excerpt SHA256 is `f6b2b4ef7127ec6df0655a01cdc7e027d38d1f6a2451cef4020a2f1a9cc4439d`; its body only reads `_live_primary_runs`.
- No mutable source/private-profile inspection, subagents, source/index/HEAD/branch mutation, test replay, installation or publication action occurred. Only this requested review report was written.

### Assessment

**Scoped task quality: Approved, with minor R1 wording correction.**

**Reasoning:** The correction removes SQLite reads from the registry critical section while preserving exact prepared authority, current runtime fences and independent fresh validation phases. The cap failure and retained warnings remain explicit separate limitations.
