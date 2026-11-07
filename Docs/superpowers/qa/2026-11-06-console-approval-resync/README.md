# Console approval PR #3036 resync

The existing PR is rebased onto `dev` at `5ca7ec6a03d92a9d8113468d37aac99cdb4d511c`. The owner requires native timing and actual Windows dispatch qualification before auto-merge. Source integration and actual dispatch are verified; presented-frame timing is still unqualified. The PR remains draft and auto-merge remains off.

## Integration

- Retained modern provider security, history, checkpoint and project-index checks, plus current runtime and served-relay ownership.
- Restricted phone choices to the intersection of original producer options and captured supported decisions. Kept the four-field relay contract and original arguments. Actual controller/host tests cover both winning origins, cancellation, FIFO, replacement and staged-card retirement. Delivery does not imply final acceptance.
- Corrected full `file_path`/`directory_path` target presentation and escaped Unicode formatting controls in paged Details while preserving ordinary Unicode, JSON meaning and original arguments.
- Preserved all 1,748 prior UI census members in order; appended six safety entries, raising the floor to 1,754. Enrolled relay regressions in the required lane without increasing job limits. Kept modern split-file census checks. Generated styles reproduce and existing token values are unchanged.
- Applied the current oldest-Date ADR ownership rule: approval ADR 235 stays; younger passkey ADR 235 becomes 354. Authentication Python behavior ASTs are unchanged. Approval ADR 235 and relay ADR 236 cross-reference the shared host contract.

The published predecessor `0b9ae198219290eea25b41561d0eeba0e260fa29` is preserved on `codex/pr3036-published-20261106`. Work continues in the isolated PR checkout on `codex/pr3036-rebase-20261106`; other worktrees are preserved.

## Verification

Counts below are separate phases with overlap; they must not be added into a whole-suite claim. The full suite was not run.

| Targeted phase | Result |
| --- | --- |
| Provider, gate and observation checks | 522 passed; one existing Windows chmod skip |
| MCP after dataclass import alias correction | 91 passed |
| Card, copy, lifecycle and geometry checks before final P2 fixes | 128 passed |
| Final P2 regressions: display / capture / Details / privacy | 18 / 19 / 24 / 10 passed (71 total) |
| Relay publisher / real controller-host-card integration | 21 / 6 passed |
| CI target pins / critical enrollment / final workflow nodes | 118 / 1 / 2 passed |
| Full census checker / design-token governance | 84 / 8 passed |
| Recorder / private-profile refusal / host wiring / feedback / disposable hook | 16 / 1 / 12 / 31 / 1 passed |
| SVG generator after redundant theme-mixin base removal | 5 passed; canonical SVG byte-identical |

The two P2 fixes had seven meaningful failing regressions before correction. Fresh complete-branch review and subsequent delta review found no remaining material source finding. Edited-file static comparison against the exact base has no new diagnostics; inherited diagnostics are not described as clean. CSS generation and diagnostic inventory checks pass.

## Actual Windows dispatch

[Content-free execution evidence](windows-dispatch.json) records genuine guarded Console `fs_read` execution through the named read-only binding, permission store, controller/host and isolated worker. Denial starts no worker. Allow once starts exactly one worker, pins the native root, reads the unchanged 89-byte fixture and proves cleanup. No remembered grant remains.

The main process and worker loaded 542 and 18 product modules respectively, all from this checkout and matching both identical 2,968-file source snapshots. Original-profile protection ran before home redirection. The final witness records zero socket/DNS attempts and no model gateway. Its fixture SHA256 is `62bc5b97ecede400a548347dec5b4659b798b0a45d402476de9f4c04d4365241`.

This proves execution. It provides no native UI timing. Earlier setup failures and an overbroad observer are retained privately; their durations are excluded from latency claims. An earlier socket audit omitted its address and remains unattributed. A separate control established that Windows Proactor self-pipe initialization can create loopback traffic; this does not explain that earlier event. The qualified final dispatch run recorded zero attempts.

## Mandatory timing gate

There are zero qualified native/browser product presentation samples. The 100 ms feedback and 200 ms actionable p95 budgets and 40 warm samples per transport are not claimed achieved. Compositor refresh, server frames, animation callbacks and terminal flushes alone do not establish presentation.

The owner authorized installed Playwright/CDP and QA-only native capture. Synthetic capture and clock capabilities are established, and an actual private native Console frame now has verified terminal handles and cell-to-pixel mapping. The complete workload has not yet reached an approval card; no product presentation distribution is qualified. Original failures and zero-sample results are retained. The PR stays draft with auto-merge off until the mandatory evidence, current-head Qodo review and resolved-thread requirements are met.

## Evidence

[Curated receipts](receipts.zip) are derived UTF8 exports with checkout/home paths, owner SIDs and terminal color escapes removed. The [receipt manifest](receipt-manifest.json) binds original and exported SHA256 separately. Original protected receipts remain in the isolated checkout. Profiles, virtual environments, databases, credentials and raw native identities are excluded. The [source manifest](source-manifest.json) binds the final changed Python files; the [independent review](independent-review.md) records findings and remaining limits.

## Subsequent hosted CI repair

Fresh CI identified startup-import and UAT integration regressions after the preceding record. The [follow-up](ci-followup.md) documents their repair, the measured Details restoration race, unchanged budgets and current source hashes. Timing qualification remains open; the earlier phase counts and source manifest retain their historical scope.


The [worker-bundle follow-up](artifact-followup.md) records the inherited hosted stamp failure, durable Windows LF repair, exact source hashes and focused verification limits.
