# Scope and verification decisions

These decisions implement the three accepted follow-ups in the existing worktree.
They do not approve publishing, merging or a broader Settings rewrite.

- Used the filed task acceptance criteria as the specification for the accepted
  bounded work. A different intended scope would require a reversible task/code
  revision, rather than a new framework or product design being assumed.
- Kept the existing isolated worktree and preserved its prior branch and scratch
  artifacts. Work remains local and uncommitted; reviewable changes do not require
  triggering the unrelated commit-time external review hook. Integration remains
  a separate action.
- Kept production steps sequential, but overlapped independent verification tails,
  review and baseline preparation. Media's later scope correction touched separate
  files. This required tracking exact source versions and retaining failing runs;
  a conflicting source change would invalidate the affected receipt and require
  a focused rerun.
- Retained the historical UI gate capacity envelope instead of replacing it with
  a smaller recent three-run sample. Recovered per-entry timings retain a
  proportional reserve to the earlier 476.64-minute envelope. That distribution
  is an explicit modeling assumption. New Windows test durations are charged raw;
  observed unchanged controls were faster in CI, but this is not a guarantee for
  new tests. The historical 2.61 factor is only a sensitivity scenario, not a
  Windows calibration. A wrong estimate requires another measured rebalance or
  capacity adjustment. The concurrency cap stays 18; 38 shards introduce a third
  scheduling wave and can increase elapsed gate time.
- Preserved privacy/bootstrap and scanner guards. Native Windows cannot exercise
  the scanner's POSIX no-follow setup, so the cleanup test supplies a validated
  accepted-package boundary value and exercises the real UI cleanup owners. This
  establishes the migrated UI contract only; it cannot support a claim about
  native Windows scanner security or managed-runtime support.
- Preserved existing assertions and inherited formatting/lint debt rather than
  rewriting unrelated code. The legacy early-Clear assertion independently fails
  on the exact baseline, while the new settled Clear behavior passes. That legacy
  synchronization debt remains visible in the verification report.

No new ADR was required: production changes preserve existing ownership and commit
models under ADR-033/052/086/150/161/303/304. CI routing follows the existing
TASK-39144/ADR-103 capacity policy. The broader TASK-1378 remains open.
