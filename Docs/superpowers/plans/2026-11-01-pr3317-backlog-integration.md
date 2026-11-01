# PR3317 backlog integration

Goal: integrate the existing CI-cost report after PR3316 without changing application test coverage or qualification verdicts.
Spec: Docs/superpowers/specs/2026-10-29-ci-cost-reporting.md (source commit a60e027f5db751f57f0256d6b132d08ca32f5679).
ADR required: no additional ADR; source amends backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md.
Reason: this directly implements the reviewed diagnostic-only policy.

- Prepare only the single source commit a60e027f on the isolated PR3316 integration; its true parent is383bf3da799b4276fb124e9225e494a468ae9a6d. Preserve all newer architecture/runtime files, 32 shards, UAT steps, preflight and cancellation owners.
- Rebase this provisional preparation onto the confirmed PR3316 merge before publication; refresh source head and live non-draft eligibility. No draft PR may be merged or marked ready.
- Compare all parsed application jobs and aggregate verdict steps against that final base: only aggregate actions-read and the bounded diagnostic step may be added.
- Run focused actual-script cost/report, workflow, queue, cancellation and UAT/shard-count contracts; no full suite.
- Record exact local evidence and hosted limits, obtain independent requesting-code-review for the immutable final head, then exact-lease publish and merge under the user's hosted-CI waiver. Confirm merge commit and identical protection restoration.

User scope: oldest eligible first; independent requested skill replaces Qodo; hosted CI and size limits waived. PR3267/3297 excluded entirely. No release or main promotion.

Initial source review: no Critical/Important/Minor findings; pr3315-resync/pr3317-initial-review.md. This is not final-head approval.

Provisional base: e6563cc734411c84816b075e5d2da1f1cd0510e1. PR3316's pending baseline-selector and literal-notification repairs must arrive with its final merge before final qualification.
