# Bounded full CI qualifications implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, followed by one independent whole-change review.

**Goal:** Retain full regression feedback with one queue-front qualification and one background qualification across PRs.

**Architecture:** Preflight classifies the live front with read-only API calls. One caller job holds a native concurrency slot while a local reusable workflow runs the unchanged application lanes. The parent retains the required aggregate, each lane verdict, cost report and queue tick.

**Tech Stack:** GitHub Actions reusable workflows and native concurrency queues, github-script, Python/pytest/PyYAML and Node behavior harnesses.

**Spec:** `Docs/superpowers/specs/2026-10-29-ci-admission-options.md` (owner approved 2026-10-30).

ADR required: yes, amend existing decisions.
ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md` and `backlog/decisions/218-in-repo-merge-queue.md`.
Reason: cross-PR admission and read-only queue-front scheduling classification.

## Global constraints

- Preserve all application steps, selectors, dependencies, test timeouts and 28 UI shards.
- UI max-parallel is 17 for the front and 8 for background; six other jobs per qualification gives a 37-job application bound across the two slots.
- No same-head cancellation, new credentials, merge/protection changes or default-branch trigger.
- Preflight, aggregate and queue control stay outside the qualification slots.
- Every existing automatic/manual full-gate request still qualifies; no draft or path skipping.
- No full local suite. Hosted admission/retry semantics remain a qualification requirement.

## Review focus

- Missing, malformed, partial or stale queue metadata must never allocate the front slot; API errors fall back to background.
- A successful reusable-workflow wrapper with a skipped/missing lane must not satisfy the required aggregate.
- Failed-job retries must reacquire a slot and retain required PR-associated results; local contracts cannot certify GitHub's retry behavior.
- Source-based selection guards must read the actual called jobs, not a reconstructed graph that hides a bypass.
- Nested job-name prefixes must not break log replay, superseded-run safeguards or cost accounting.

## Task 1: Classifier and native admission graph

- [x] Read TASK-38101 and add failing tests in `Tests/CI/test_ci_qualification_admission.py` for the real classifier and raw caller/callee graph.
- [x] Extract application jobs unchanged into `.github/workflows/ci-qualification.yml`; expose seven lane results as outputs. Add a caller with fixed front/background groups, `queue: max`, no cancellation and successful-preflight dependency.
- [x] Add bounded, paginated front classification at the end of preflight. Match the queue's eligibility/order, then re-read the candidate's current identity, arming timestamp and head. Uncertainty reports background.
- [x] Preserve all lane verdicts using called-workflow outputs and add an explicit wrapper-result verdict. Run the new behavior/graph tests.

## Task 2: Preserve source guards and evidence tooling

- [x] Introduce a narrow test helper reading the two actual workflow documents without rewriting expressions/dependencies; update existing selection tests and mutation fixtures to follow the physical files.
- [x] Update capacity statements for 23 front / 14 background / 37 combined application jobs, preserving the 34-job eligible census per qualification.
- [x] Verify existing replay parsers accept nested job prefixes; add a regression witness if needed. Compare all application jobs against parent `a60e027f5db751f57f0256d6b132d08ca32f5679` apart from the documented dependency and UI parallelism changes.
- [x] Run focused CI/queue/selector/census/replay tests, static analysis, formatting and diff checks. Record exact evidence and limits.

## Task 3: Review and delivery

- [x] Obtain one independent whole-change review. Fix consequential findings with RED/GREEN tests; no second review pass.
- [x] Update task notes, ADRs and execution ledger. Keep the task In Progress until hosted qualification is complete.
- [ ] Publish a separate draft follow-up; preserve #3316 and #3317. Reconcile the stack as parents merge, without pushing to armed PRs.

## Execution ledger

- Started from cost-reporting head `a60e027f5db751f57f0256d6b132d08ca32f5679` on `codex/ci-bounded-qualifications` in the existing managed follow-up worktree. PR #3316 was armed using the user's explicit Qodo exception; that exception applies only to #3316.
- Ruling: use a read-only classifier and native queued reusable-workflow call, not a custom scheduler. The owner approved bounded full feedback; the cost if native hosted retry semantics differ is a failed qualification requiring repair, never a relaxed aggregate.

- Verification: 806 focused tests passed with 28 unchanged exhaustive shard cases deselected; after the review correction and cancellation-verdict strengthening, 150 focused admission/aggregation tests passed. Ruff lint/format and diff checks passed. Exact parsed seven-job conservation and live read-only GraphQL classification succeeded. Evidence and actionlint schema limitations are in `Docs/superpowers/qa/2026-10-30-ci-bounded-qualifications.md`.
- Independent review: one P2 malformed-metadata finding accepted. Eight real-script RED probes preceded strict UTC timestamp round-trip and nonempty eligibility validation; all passed afterward. No second review requested.
- Review dispositions: hosted concurrency, overflow and failed-job retry semantics remain required and unverified, so keep the draft and task In Progress. These were not waived. No review/merge authority changed; the Qodo waiver remains limited to #3316.
- Implementation ruling: forward the optional manual `pr` input to preserve the inherited lane condition. Use `always()` for the new wrapper verdict to reject cancellation without inheriting the individual lane verdicts' cancellation guard.
