# Console approval ownership — current-dev integration

PR2730 prevents queued approval actions from affecting a replacement batch and
prevents duplicate submissions. The production behavior matches saved head
`10747313ea6ec2c8d40381518ca31708cc94fa68`; Qodo requested a mechanical helper
rename to `ApprovalActionButton` and one shared native worker timeout constant. Existing scopes, raw-shell exclusions and layout stay
intact. ADR-032 and ADR-150 apply; no new ADR is required.

## Base and conflict choices

PR2728 merged at `de10a62e67124a2b21b78edf1a4887cea03ff139` after owner approval,
current-head CI and accumulated review. Its actual merge tree equals the approved
tree. [Closeout receipt](integration/pr2728-closeout.json).

Only PR2730's feature commit was replayed onto that merged dev. No product conflict
occurred. Both report conflicts retain every dev line in order and the complete
saved Console checkpoint; the lessons file merged automatically, preserving both
histories. [Conflict choices](integration/rebase-conflicts.json).

## Targeted verification

- **210 distinct passing cases**: 126 product cases (including all 25 ownership
  cases) and 84 native-runner boundary cases. [Census](integration/qualified-cases.json).
  The original 123 passing case IDs are all retained. The integrated product run
  reports [123 passed, 19 failed](integration/tests/integrated-001.txt); the added
  isolated cases and runner boundaries report [109 passed](integration/tests/followup-003.txt).
  The final runner boundary recheck reports [84 passed](integration/tests/runner-final-004.txt).
- **19 existing failures remain**: all 19 reproduce on unchanged current dev,
  with the same names, messages and final lines after normalizing only object
  addresses in two assertion expansions. [Comparison](integration/baseline-comparison.json),
  [baseline run](integration/tests/baseline-001.txt). These comprise 13 profile-setup
  refusals, three stale CSS assertions and three controller expectation mismatches.
  The full approval module is not green. No full test sweep was run.
- [All seven derived-artifact guards pass](integration/preflight.txt).
  [No new Ruff diagnostics](integration/static-analysis.json); six existing
  production diagnostics remain. New/changed test and runner files and every
  changed production range pass [formatting checks](integration/formatting.txt).
- [Independent review](integration/independent-review.txt) found no remaining
  blocker after three isolated tests and the native import-order correction.

## Native qualification

[Eight individually inspected captures](GALLERY.md) cover dark/light at 120×40 and
170×48 in integrated native run **006**. The real app uses LinuxDriver and TTY
streams, with a disposable profile. Keyboard Deny all and Approve all change
choices only; Submit completes the real controller round trip for exactly the
two synthetic pending calls. No tools are dispatched and no network attempts
occur. This does not qualify connected-server execution.

[Result](integration/native/result.json) records six loaded module paths under
this worktree, all eight source hashes and the runner hash. Positive geometry,
clipping, center-point hit testing, painted labels and focus are asserted.
[Lifecycle](integration/lifecycle.json) verifies normal exit, absent process,
released/reacquired lock, ten healthy private databases, zero persisted messages
or conversations, unchanged default settings, empty faulthandler and no app errors.

Earlier integration run **001 is unqualified**: its CLI parser cached the main
checkout package before the worktree path was pinned. File hashes alone did not
prove which code ran. [Import reproduction](integration/runner-import-provenance.json).
Runs **002/003 are superseded** by final 006 after the Qodo cleanup; 003 first
established explicit loaded-module receipts. Final 006 is pixel-identical to all
eight individually inspected 003 captures. Launches 004/005 were refused before
app construction because the prepared directories were incomplete. Their result
and lifecycle receipts are retained under `integration/earlier-native/`; none
is counted as final current-source qualification. Original-base evidence remains
historical. [Export hashes](integration/export-manifest.json) record whitespace
normalization only.

Qodo reported zero bugs and two maintainability findings on `f1af5ee17c`.
Both are addressed, with [109 affected passes](integration/tests/qodo-005.txt),
seven fresh guards and [eight pixel-identical final captures](integration/qodo-visual-comparison.json).
The production AST differs only in the helper name. [Follow-up receipt](integration/qodo-followup.json).

Current-head CI, accumulated Qodo review and PR2730's own final visual approval
remain merge gates. The wider work stream remains open: MCP inspector refresh,
long-path/compact presentation and connected-runtime journeys are follow-ups.
