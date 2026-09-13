---
id: TASK-20015
title: Support governed post-approval benchmark evidence correction
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 04:44'
updated_date: '2026-08-24 06:43'
labels:
  - performance
  - testing
  - evidence
  - governance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow an unreleased, already-promoted confirmatory evidence package to be reopened, re-reviewed, and atomically republished from the same immutable raw acquisition when final review finds a derived-manifest omission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirmatory manifests record the exact SHA resolved from the immutable implementation-base ref and fail closed if that ref is missing or drifts.
- [x] #2 An approved attempt can be reopened only by a fresh, numerically later `changes_required` receipt bound to its current approved digest, raw hash, verdict, and attempt ID.
- [x] #3 A distinct immutable `corrections/correction-NNN` artifact root requires and accepts a still-later independent approval structurally bound to its confined receipt location and changed digest while retaining every earlier receipt identity and the append-only attempt lineage.
- [x] #4 The unreleased repository package is recoverably removed without an intermediate commit and republished only through the existing atomic no-replace promotion path.
- [x] #5 The correction uses the same acquisition: the original approved five files and `review-001` remain byte-identical, and the versioned correction root copies the raw JSONL, machine summary, human report, and README byte-for-byte while changing only its derived manifest and later receipt.
- [x] #6 Focused and full harness tests, static checks, privacy scans, exact digest/receipt verification, and independent evidence review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add deterministic RED tests for the missing implementation-base manifest field and fail-closed post-approval review reopening.
2. Record and validate the immutable implementation-base SHA from the pinned ref in every campaign manifest.
3. Add a lock-held `reopen-review --correction-id correction-NNN` action that accepts only a fresh, numerically later eight-field `changes_required` receipt bound to the current approval, original artifact root, digest, raw hash, verdict, and attempt ID; record the correction ID in the append-only event while preserving every receipt identity.
4. Create a new immutable `corrections/correction-NNN` root under the same attempt by copying the original five reviewed artifacts, changing only the manifest to add the validated implementation-base SHA, and retaining the original approved root and `review-001` byte-for-byte.
5. Require a still-later eight-field approval at `corrections/correction-NNN/reviews/review-NNN.json`; derive and validate the correction identity and artifact root exclusively from that confined path, bind its changed digest, and preserve the original raw JSONL, machine summary, human report, README, verdict, and attempt ID. Registry markers bind normalized relative receipt path plus hash. Then recoverably remove the unreleased add-only destination without an intermediate commit and republish that correction root through the existing atomic no-replace promotion path.
6. Obtain independent `review-002` and `review-003`, verify exact original/correction byte identities, privacy, digest and receipt binding, focused and full harness gates, independent reviews, and task hygiene.

ADR required: no

ADR path: N/A

Reason: this extends benchmark-only evidence correction already required by the confirmation plan; it changes no production storage, provider, privacy, or runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the benchmark-harness portion of the governed correction workflow.
Confirmatory acquisition now resolves the fixed implementation-base ref and
records its exact SHA. An approved attempt can be reopened only through a
later, exact eight-field rejection receipt bound to the approved artifact
digest, raw hash, verdict, and attempt. The correction is built through an
atomic no-replace stage under `corrections/correction-NNN`; it preserves the
original five artifacts and earlier receipts, changes only the manifest's
implementation-base field, and accepts only a still-later approval whose
confined path determines the correction root.

The provider-free `prepare-correction` action resolves and verifies the fixed
implementation-base ref itself before it can create a correction namespace;
callers cannot supply an unverified revision. Correction approval and promotion
independently resolve that fixed ref again under the campaign lock and validate
the confined correction manifest against the resolved revision, so a manually
prebuilt root cannot bypass the authority check. Reopen and correction-approval
retries now reconcile the durable receipt marker with a missing append-only
lineage transition after either a pre-commit or post-commit failure, while
remaining idempotent after the transition commits. A visible marker is not by
itself treated as durable: reconciliation revalidates and fsyncs the exact
marker file and its containing directory before appending a missing transition
or returning an already-committed result.

Registry markers now bind normalized receipt path plus receipt hash and reject
duplicate numeric review identities. The correction path rejects symlinks,
lexical traversal, stale review numbers, implementation-ref drift, mutation of
any original or corrected artifact, and partial pre-rename creation. A failure
after the atomic rename retains one complete correction root and no stage.
Nested receipt-registry namespaces are created one component at a time without
following symlinks, and every new parent link is fsynced from the durable
registry root. Correction receipt publication also fsyncs its `reviews` child
link through the correction root and then fsyncs the parent `corrections/`
namespace before publishing the registry identity. This makes a complete root
retained after a post-rename parent-fsync failure safely retryable.
Promotion reuses the existing durable atomic no-replace publisher with the
source root derived solely from the approving receipt location. The pytest
cleanup fixture recursively unseals nested published/correction directories.
Ordinary synthetic correction-approval tests opt into a shared fixed-authority
fixture, so their outcome does not depend on whether the worktree-local
benchmark ref exists. The explicit missing/drift tests retain sole control of
their authority failures.

Verification completed for the implementation:

- review-finding RED shard: 12 expected failures;
- review-finding GREEN shard: 12 passed;
- marker-durability RED shard: 4 expected failures;
- marker-durability GREEN/reconciliation shard: 8 passed;
- correction-authority/durability RED shard: 4 expected failures;
- correction-authority/durability GREEN shard: 4 passed;
- missing-ref portability mutation: representative RED failed as expected,
  representative GREEN passed, and the correction shard passed 19 tests;
- correction/review regression shard: 94 passed;
- complete performance harness with the non-authority test ref pointed to a
  nonexistent in-process ref: 649 passed, 2 dependency warnings in 92.24s;
- Ruff, `py_compile`, and `git diff --check`: passed.

Modified files are the benchmark runner, its focused harness tests, and this
task record.

ADR required: no

ADR path: N/A

Reason: this is a benchmark-only correction protocol and does not change a
production storage, provider, privacy, security, or runtime boundary.

Operational closeout completed against retained campaign `attempt-0001`.
Independent `review-002` (SHA-256
`cdc00f8823e4202e88f7b260213cccf882ebbc97d6dacfe38efdb9ec356253a5`)
reopened the original digest for the missing implementation-base provenance.
`correction-001` preserved raw, summary JSON, report, and README byte-for-byte
and changed only the manifest by adding implementation base
`77c5e9f487af79391a479deb85e712163bfed909`. Independent `review-003` approved
corrected digest
`c04acca85762c5f2cbfe05113223049d907ad2c8436b0ce8909f7ae78267ee49` with
receipt SHA-256
`889b3f8382d7ac78fa931bd8ea50dda5aa78fc8e17cf08d193616702d6a2c95d`.
The unreleased six-file package was removed by `git revert --no-commit` and
atomically republished through the harness; the net canonical diff is only the
manifest plus receipt. Original artifacts and reviews remain immutable in
campaign history. Exact hash, digest, lineage, privacy, Ruff, `py_compile`, and
`git diff --check` gates passed, as did the full performance harness: 649
passed with 2 dependency warnings in 75.04s. Correction publication was
committed as `927eb0ea9`. No new ADR was required.
<!-- SECTION:NOTES:END -->
