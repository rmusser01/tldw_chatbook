# R1 completion validation and primary-provider offline diagnosis

The R1 completion gap is repaired offline. PR3286 remains draft and AC24 remains
unchecked. The previous failed live qualification is unchanged. This follow-up
used **zero live provider requests and zero product children**; cumulative actual
usage remains **412 requests / 28 children**. No qualification, budget request,
automatic retry or merge was performed.

## Completion repair

The previous run accepted a finding and then acknowledged completion with
`empty_findings:true`; strict final sealing correctly refused that contradiction.
`ReviewStore._publish` now checks that boolean against the same participant's
accepted findings before acknowledging or publishing new completion intent.
It returns the existing `invalid_record` refusal with the field, fixed mismatch
type and expected boolean. The existing bounded tool feedback tells the reviewer
to correct the same submission ID. No accepted finding or revision is rewritten.

The guard runs under existing locks after schema/reference checks and immutable
accepted-replay handling. Intermediate coverage may precede findings. Final
sealing still independently checks current coverage, terminal success, capture
receipts and outstanding rejections. No provider, accounting, permission,
continuation, schema, storage-limit or child-limit change was made. Existing
[ADR-281](../../../../../backlog/decisions/281-primary-agent-multi-model-code-review.md)
applies; no new architectural decision was needed.

Verification on the final production change:

- Four new regressions failed before the fix at the accepted-contradiction or
  incomplete native/fenced report boundary.
- Store, report, completion-recovery and conflict-recovery selection: **160 passed**.
- Final new regression file: **5 passed**, including immutable completion replay
  after later findings and refusal to seal stale coverage. Four overlap the prior
  selection: **161 unique R1 regression tests**.
- New test lint/format, changed product formatting, syntax and whitespace passed.
  Product lint retains the same five pre-existing findings; zero new findings.
- [Scoped review](review.md): no actionable findings.

One intermediate test run attempted to append nonempty raw output after terminal
settlement and correctly received `review_capture_closed`. The test now captures
text before settlement and supplies only the empty terminal-capture receipt
afterward. Production capture/sealing behavior was not changed. Initial pytest
cleanup also reported unrelated old temporary-directory warnings; subsequent
runs used dedicated task-owned basetemp directories.

## Separate primary-provider error

The saved final physical request returned HTTP200. The application reported
`unexpected provider error` with a fallback status502. That status is not proof
that the server returned502. Saved transport records contain status/metadata but
not response chunks or the original stream exception; the task-owned database
does not retain trace bodies that reconstruct them.

Five synthetic offline controls exercise the actual Console gateway, dispatcher,
custom-hosted handler, owned transport and SSE parser with only fake endpoint
resolution and HTTP responses. A valid HTTP200 stream succeeds. Malformed JSON,
a read failure after headers, and a missing clean terminal marker each produce
the same fallback error; a synthetic upstream HTTP502 produces the distinct
`provider unavailable` category. **All five passed.** These controls establish
that the saved message is ambiguous, not which failure occurred historically.

No provider production fix is justified by the retained evidence. The exact
historical exception remains unknown. Resolving it would require the original
stream exception or response chunks, which this run did not retain. No live
request was made to recreate it, and no diagnostic logging/privacy behavior was
changed. The private diagnosis and executable probe are preserved in the
[evidence manifest](retention.json), separately from the earlier live archive.

## Delivery limits

The current PR was confirmed draft with auto-merge disabled before publication.
Its preceding documentation-only head had UI Fast Lane9 and derived-check
failures plus a cancelled Windows GGUF check; this scoped follow-up does not
claim those checks green or repair unrelated CI. The production candidate before
the failed live run had passed55 checks with3 skips, as its retained receipt shows.
The new pushed head requires its own normal CI result.

No full test suite, new live qualification, upstream provider probe or CI rerun.
R1 model finding quality and the complete live verifier journey remain unproven;
offline validation and a green test selection do not complete AC24.
