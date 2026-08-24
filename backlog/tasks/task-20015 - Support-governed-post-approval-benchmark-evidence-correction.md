---
id: TASK-20015
title: Support governed post-approval benchmark evidence correction
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-24 04:44'
updated_date: '2026-08-24 04:45'
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
- [ ] #4 The unreleased repository package is recoverably removed without an intermediate commit and republished only through the existing atomic no-replace promotion path.
- [x] #5 The correction uses the same acquisition: the original approved five files and `review-001` remain byte-identical, and the versioned correction root copies the raw JSONL, machine summary, human report, and README byte-for-byte while changing only its derived manifest and later receipt.
- [ ] #6 Focused and full harness tests, static checks, privacy scans, exact digest/receipt verification, and independent evidence review pass.
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

Implemented the benchmark-harness portion of the governed correction workflow.
Confirmatory acquisition now resolves the fixed implementation-base ref and
records its exact SHA. An approved attempt can be reopened only through a
later, exact eight-field rejection receipt bound to the approved artifact
digest, raw hash, verdict, and attempt. The correction is built through an
atomic no-replace stage under `corrections/correction-NNN`; it preserves the
original five artifacts and earlier receipts, changes only the manifest's
implementation-base field, and accepts only a still-later approval whose
confined path determines the correction root.

Registry markers now bind normalized receipt path plus receipt hash and reject
duplicate numeric review identities. The correction path rejects symlinks,
lexical traversal, stale review numbers, implementation-ref drift, mutation of
any original or corrected artifact, and partial pre-rename creation. A failure
after the atomic rename retains one complete correction root and no stage.
Promotion reuses the existing durable atomic no-replace publisher with the
source root derived solely from the approving receipt location. The pytest
cleanup fixture recursively unseals nested published/correction directories.

Verification completed for the implementation:

- correction/review regression shard: 61 passed;
- complete performance harness: 630 passed, 2 dependency warnings in 107.25s;
- Ruff, `py_compile`, and `git diff --check`: passed.

Modified files are the benchmark runner, its focused harness tests, and this
task record. The recoverable removal/republish of the unreleased retained
package and the independent `review-002`/`review-003` evidence review remain
before AC #4 and #6, and task completion, can be recorded.

ADR required: no

ADR path: N/A

Reason: this is a benchmark-only correction protocol and does not change a
production storage, provider, privacy, security, or runtime boundary.
