# PR 3293 backlog integration, 2026-10-31

Rebased onto `dev` at `930fe423c26ef922b7421225feac011e109124c2`.
Pre-evidence head: `5d67001feaa9faa406fd66f45a48a43267c43982`.
All 18 production files are identical to the previously reviewed and tested
`3072e75141dae544b39ccfdd282053a449e3d8f3` integration. The rebase conflict
was additive testing lessons; both entries were retained.

The independent full-branch review's two Important custom-entry credential
findings were fixed and independently approved at `1fe1df8156d3170b9ade9d1bad1ac12b2569cf44`.
Registry entries retain their own credential authority, including an explicit
keyless result; unrelated stranded family credentials cannot block the entry.
Protect's persisted-state and exact draft-save repairs retain their reviewed
source and tests.

## Targeted evidence

The 97 Protect and 47 credential cases are retained evidence for unchanged
source, not claimed as new final-base executions. The 44 service/lifecycle
checks ran at the prior integration base; all affected production blobs survived
the final rebase. They cover the five non-chat consumers, context-window
resolution, Strata metadata/credentials, and encryption rollback boundaries.
The UI selection covers quick/full keyboard walks, resumed setup, and honest
failed-rollback copy. The final resume retry ran on the final base.

| Receipt | Testcase elements | Failures | Errors | XML SHA256 |
| --- | ---: | ---: | ---: | --- |
| Protect contracts | 97 | 0 | 0 | `faef65fbd13c2685009c015bef53651c5a2ac56a80785d2b448f4d90c9033209` |
| Entry credential boundary | 47 | 0 | 0 | `d5614c9e62db7224166debc08041d70d1c2b318a26faf4cd49d85decc17050d9` |
| Service/lifecycle integration | 44 | 0 | 0 | `1f9e39451c5a09cc1d7f3dafb9b3fc18314fed3d3da548756898d8dc4d523541` |
| UI integration with retirement error | 4 | 0 | 1 | `ae626109ebc16a1021ea1158b1daa1f2c6deed7f4d1ccf31fdecc2a414ec412f` |
| Stack-only retirement probe | 1 | 0 | 1 | `079563f00b0c7f441bb87126da429d98342ed3db43e83d28f4d43b80ccd74af3` |
| Corrected resume boundary | 1 | 0 | 0 | `4984a68cd211865b1434565e3bf985f82f2e84728cb1502f7a72af460abb36c6` |

Four UI test bodies passed, but the original resumed-target case errored during
native-executor retirement. The stack-only probe reproduced that failure and
identified `chachanotes-fts-backfill` still inside guarded database initialization
and builtin seeding. It did not establish a Provider discovery failure.

The correction is test-only: a forwarding observer retains the actual named
startup Worker, preserves the original starter call and return, and requires
completion and `WorkerState.SUCCESS` within the existing 30-second settle bound
before leaving the app mount. All original resume/Protect assertions and the
3-second executor-retirement guard remain. The exact-node retry passed in
73.15 seconds, including 1.37-second teardown. Start/end source hashes match.
No product behavior, worker cancellation deadline, or test guard changed.

Changed-owner diagnostic inventory entries are conserved. The four touched
module pins equal their measured lengths: gateway 8183, transcription service
4496, wizard 3106, Provider step 2454. The size limits are explicitly waived by
the user; measurements remain exact. The modified UI test has zero Ruff findings.

Raw command, XML, log, and source-manifest receipts are retained in this session's
isolated `pr3293-resync` sibling directories. Failed runs remain failed evidence;
the outer probe command's console encoding error occurred after its failure
receipt was written and is not counted as a separate product failure.

## Scope

Hosted CI and Qodo are replaced for this backlog clearance by the user's
explicit waiver, targeted local checks, and independent Superpowers review.
This is not a full-suite, live-provider, release, or complete key-encryption
lifecycle certification. TASK-34100.4 stays In Progress; the remaining outbound
readers and status surfaces remain in TASK-34100.4.1 and TASK-34100.4.2.
Existing pytest-asyncio, Pydantic and audioop deprecation warnings remain visible.
