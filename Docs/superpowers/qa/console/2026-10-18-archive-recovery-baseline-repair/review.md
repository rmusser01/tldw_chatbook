# Independent review — TASK35584 / PR3162

**Assessment: ready for the requested draft PR update.** No actionable critical or important findings remain in the reviewed repair.

Reviewed the working-tree repair against published parent `ccab42365855c5a37c17a2c6bed5c4bb6f2ee96f`, TASK35584, the baseline-repair plan, repository ownership/design guidance and the connected database, search, trace and runtime callers. The review covered all 31 changed Python/CSS paths, generated diffs, fixture consumers and diagnostic evidence. No source, Git index or refs were mutated by the reviewer; only this report and the requested temporary diagnostic record were written.

## Strengths

- Search reuses the existing operation-owned connection seam. Unchanged borrowed connections, native transactions, busy timeouts and ordinary close behavior retain their ownership. Native closure is checked while real references remain alive.
- Constructor and fixture repairs close their actual owners: failed Workspace initialization, local registry/media fixtures, replay workers and factory-attached mounted runtimes. Transaction success and garbage collection are not substituted for physical retirement.
- The ten baseline repairs retain meaningful observer, byte-bound, recovery-grace and migration assertions. CSS changes preserve all 262 old token values and resolved selector declarations; governance budgets are unchanged.
- The mounted fixture forwards only its unmounted app's `call_later` to the real host. Two tests wait for actual persisted receipt acknowledgement before installing spies or hiding the source. Original assertions and notification/navigation spies remain intact.

## Issues

**Critical:** none. **Important:** none open. No minor issue warrants expanding this repair.

Two P2 issues identified during review were repaired and verified:

1. **Cleanup could exceed the search deadline** — `tldw_chatbook/Chat/archive_recovery/search_service.py` and `tldw_chatbook/DB/base_db.py:40`. In the reviewer's real held-reader diagnostic, SQL finished in 0.0519s with 0.3988s left in a 0.5s budget, but ordinary WAL TRUNCATE cleanup then waited 1.1752s; search returned after 1.277s. The optional operation deadline now caps only the departing owned handle's busy timeout. Borrowed handles are untouched. The exact executed diagnostic and clearly labeled transcript copy are retained as `checkpoint-review-probe.py` and `checkpoint-review-observation.log` under the raw evidence root.
2. **Deadline cleanup could mask the body error after native retirement** — `tldw_chatbook/DB/base_db.py:92`. A stale thread-local cache can survive quiescence. The final helper checks the native SQLite descriptor before issuing cleanup PRAGMAs and still runs owning close in `finally`. Two real quiescence cases cover successful bodies and preservation of the original RuntimeError; no ordinary-close policy was broadened.

The first passive UI wait was insufficient because the factory app had no running callback pump. That failed intermediate result was retained. The fixture bridge now permits the existing generation-checked render/ack path to progress; no mark was forged or manually acknowledged.

## Evidence inspected

Counts overlap and must not be summed. Final targeted runs were performed by root/helpers; this reviewer inspected their logs, diagnostic code, source identities and native-owner records, and separately ran the held-reader diagnostic.

- Trace: 8 passed, covering all five original failures and connected controls.
- Migration/governance/generated CSS: 50 passed; selector/token equivalence checked.
- Integrated storage: 282 passed in 95.49s; final ownership controls: 18 passed in 3.59s.
- Final captured follow-up: 50 passed in 114.70s.
- Final mounted source: 9 passed in 51.97s; the unchanged delayed-real-ack diagnostic: 3 passed in 26.85s.

Storage/captured resource logs finish with 17 descriptors before and after diagnostic session GC, with no owned SQLite files. All nine final mounted teardowns and session teardown retain the same 17/zero-file plateau. The initial process counts were 14; these results establish the final plateau and retirement of owned SQLite resources, not zero total process-start descriptor growth. No warning threshold, ordinary GC policy or admission/privacy/permission gate was relaxed.

All 16 changed Python files pass recorded AST/duplicate-definition and critical Ruff checks, with zero introduced per-file lint findings. Existing formatter/lint debt is explicitly retained. The reviewer also ran the code-diff whitespace check successfully.

## Reviewed source identity and recommendation

The SHA256 of `git diff ccab42365855c5a37c17a2c6bed5c4bb6f2ee96f -- Tests tldw_chatbook` is `c65457d90ef0338f9cb041979dc6b10af48f0789d6a14822b2bfdc3910221939`.

| Final source | SHA256 |
| --- | --- |
| `tldw_chatbook/DB/base_db.py` | `56a8477ca207c3a160c5a4c7a74e1cbecd0af27cc91dab9cf4429b02ebffe687` |
| `tldw_chatbook/DB/Workspace_DB.py` | `8a9555401682db7d009750f7be19b0e022301faae7653b36985337b5b0236b38` |
| `tldw_chatbook/Chat/archive_recovery/search_service.py` | `031ae909cf8a5b19d4c2e45182113280db4699b474176fc51d22279219d4a75c` |
| `Tests/UI/test_console_archive_recovery_flow.py` | `5120f26ed2d6d6c031f93d28e8caddc99c10cf98df71ad922176f2d650267336` |

Keep the qualification manifest tied to these final bytes when completing the documentation and publishing. Existing ADR166/228/150/161 cover this repair; no additional architectural decision is introduced.

Skipped: full suite, installed build, full sidecar module, long five-cycle repeat and new live/model requests; the 30/30 model budget remains exhausted.
Remaining risks: two evidenced inherited voice failures, existing static debt and unselected UI/sidecar paths remain outside this qualification.
