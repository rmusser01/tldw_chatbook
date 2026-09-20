# Console approval action ownership — TASK-32841

**Current integration:** see [current-dev review](CURRENT-DEV-REVIEW.md) and
[current gallery](GALLERY.md). The original-base evidence below is historical.

A queued Approve all, Deny all or Submit press now belongs to the batch shown
when it was published. Changed rounds/calls, clear and finishing transitions
invalidate it. A batch can publish its decision only once; unchanged identified
resyncs preserve its choices and submission state, while a fresh batch is usable.
The existing layout, decision scopes and raw-shell exclusions are preserved.

This independent follow-up starts at `dev` d6e2a46384. It does not depend on draft
PR2727 or PR2728 and does not implement the proposed MCP permission-matrix bulk
actions. No new ADR: existing ADR-032 permission boundaries and ADR-150 design
language apply. Final visual approval and current-head CI/review remain merge
gates; this evidence does not authorize merging.

## Verification

- [Final isolated run](tests/isolated-001.txt): **123 passed, 19 failed** across
  `Tests/UI/test_console_mcp_approval.py` and the new ownership module. All **22
  new cases pass**. [Passing-case census](qualified-cases.json).
- All 19 failures reproduce with unchanged `dev` production source in the same
  isolated checkout; names, messages and final traceback lines match exactly.
  [Comparison](baseline-comparison.json), [baseline run](tests/baseline-isolated-001.txt).
  These are 13 profile-setup refusals, three stale CSS-literal assertions and
  three controller expectation mismatches. They remain existing test debt;
  the full module is not green. No full-suite run was requested or performed.
- [Meaningful red run](tests/red-002.txt): 16 failures and five passes before
  the repair. The first fixture compared pre-mount Select defaults; it was
  corrected to set legal choices explicitly. Earlier per-case results remain
  under `tests/`; final evidence comes from the isolated run.
- [Static analysis](static-analysis.json): no new diagnostics; six existing
  production diagnostics, zero in the new test and runner.
  [Formatting](formatting.txt) covers the changed production class and new files.
- [Preflight](preflight-isolated.txt): six guards passed initially; the Canvas
  guard could not retrieve its pinned input in the sandbox. Its [network-enabled
  recheck](mermaid-recheck.txt) reproduced all six outputs. All seven guards
  therefore pass, without regenerating or changing any artifact.
- [Independent review](independent-review.txt): no production blocker. The
  requested submitted → unchanged resync → fresh-batch regression is included.

## Native evidence

[Eight historical captures](HISTORICAL-GALLERY.md) show keyboard Deny all, Approve all and
Submit in Textual dark/light at 120×40 and 170×48. Native run **003** uses the
real app, LinuxDriver, terminal streams, approval card and controller
request/resolve round trip. Two synthetic pending calls enter the controller;
bulk actions only change choices, and Submit resolves exactly those call IDs.
The returned decisions are never dispatched to a provider or tool executor.
No network attempts occurred. This is not connected-server execution coverage.

The runner checks positive geometry, clipping, center-point hit testing, full
painted button labels and focus before keyboard action. [Lifecycle evidence](lifecycle.json)
confirms normal exit, absent process, released lock, ten healthy private
databases, zero stored conversations/messages, unchanged default settings,
no app errors and matching source/runner hashes.

Earlier attempts are retained with explicit limits:

- **001** completed controller operations, but provider setup covered the card.
  Its raw `passed` flag does not establish visual qualification.
  [Unqualified overlay capture](earlier-native/001-overlay.svg).
- **002** showed the controls, but concurrent edits in the original checkout
  changed its CSS. It is superseded for this PR by isolated run 003.
  [Isolation receipt](isolation.json). The original checkout and unrelated edits
  were preserved; only owned approval files were copied.

Task allocation used a fresh all-ref/worktree census. The CLI's stale candidate
was corrected before implementation; the preserved original and isolated task
are two copies of one allocation. [Ownership check](allocation-owner-check.json).
[Export hashes](export-manifest.json) identify normalized receipts and originals.

Next review: MCP session grants and revocation, remaining approval workflows and
connected-runtime journeys. The wider component work stream remains open.
