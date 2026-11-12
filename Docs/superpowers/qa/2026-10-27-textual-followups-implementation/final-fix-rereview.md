**I1 — Older same-reminder acknowledgement overwrites a newer Save — ADDRESSED.** `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py:2978` snapshots every existing worker for the same captured owner/reminder and node before synchronous native admission. `:2988` awaits those predecessors before `update_reminder` at `:2994`. Later requests cannot reach either response publication or offline queue insertion before earlier responses settle. Each later admission includes earlier waiting workers, so third-and-later Saves remain ordered without relying on iteration order. The first response retains its normal success/failure handling; no later Save cancels it. Different identities have separate groups, creates have no predecessors, and `exclusive=False` remains at `:3012`.

**M1 — Remote explicit-null clearing — ADDRESSED as a retained qualification; underlying omission remains unchanged.** `Docs/Architecture/scheduling-workflows.md:96` explicitly states that durable queue null retention does not establish remote clearing and that the notification transport omits `None`. The fix report also retains this limit. This was an inherited disclosure requirement, not a request to repair the remote contract in this wave.

**M2 — Qualified static/test evidence — ADDRESSED as a retained qualification; inherited debt remains unchanged.** The task Implementation Notes and final-fix report retain the broader E402/E721, formatting, timing, worker/MRO, Windows visual/path and dependency-warning qualifications. `final-fix-static01/static-checks.json` claims clean Ruff/format only for the two owned Python files; it does not establish whole-repository clearance. Current-head hosted CI and timing remain controller obligations.

### New Breakage in the Fix Diff

None. Critical: 0; Important: 0; new Minor: 0.

### Checks

- Read the immutable `review-0e844e6e3a..372c20559c.diff` for base `0e844e6e3a7c6d3314a3f424f66ceea0c802bcc0` through head `372c20559c5271b0f6356ae95a533b5260d64009`, including the initially output-truncated task hunk. Reviewed all four changed files against the brief, exact I1 requirement, and added AC5. No Git diff was re-derived.
- Inspected the native Textual worker implementation for the new dependency: admission adds and starts the predecessor synchronously; start sets RUNNING before scheduling; `wait()` awaits the actual task. The snapshot excludes the current worker and other nodes. Ordinary save exceptions remain caught inside the predecessor, preserving later admission after recoverable failure.
- Inspected `Tests/UI/test_schedules_reminder_form_saves.py:230`: mounted Queue edit keys and visible form Save submit the same reminder twice through the real service/SQLite, hold the first accepted external response, and change shared active owner while waiting. Assertions at `:269` verify newest SQLite/Queue state; `:274` verifies the newer offline mutation survives; `:280` verifies the second request did not start before first release; `:282` verifies both truthful notices and no cancellation. Existing distinct-reminder/create/failure controls remain intact.
- Independently parsed both RED parent and actual child XML receipts: 2 expected stale-title failures (`First` versus `Second`), 0 errors/skips. Independently parsed GREEN parent and all 11 actual child XML receipts: 11 passes, 0 failures/errors/skips. Verified all 13 retained child SHA256 values and exact parent/child classname-plus-name set equality. These are 11 green scenarios, not 22.
- Independently parsed and hashed `final-fix-guards01/result.xml`: 5 passes, 0 failures/errors/skips, covering both workbench size checks and the three import/module checks. Recomputed all 10 current source SHA256 values and physical line counts from `source-proof.json`; every value matches. Workbench is 5530 lines under the unchanged 5532 cap. Reviewed recorded Ruff/format/diff-check exit-zero receipts; did not rerun them or claim independent static execution.
- Ran only read-only XML/hash/line-count inspection. No tests were rerun, no remote actions or subagents were used, and no product/index/HEAD/branch state was changed. This report is the sole authorized artifact write.

### Out-of-Scope Observations

- M1/M2 remain inherited, non-blocking qualifications; no additional outside-diff defect was identified.
- Inline/bulk/global service callers and screen-shutdown durability are unchanged contracts outside this full-form fix. The reviewed guarantee is same captured owner/reminder on the mounted owner.
- The controller still owns enrollment of the two new cases, census/capacity/QA reconciliation, PR publication and actual current-head hosted CI. Local receipts do not certify those gates.

### Verdict

**Fix round: All requested findings addressed, no new Critical/Important breakage.** I1 is closed. M1/M2 remain disclosed inherited limitations, not unresolved repair requests. The scoped fix is ready for controller integration and current-head CI; this verdict does not authorize a merge or certify hosted checks.
