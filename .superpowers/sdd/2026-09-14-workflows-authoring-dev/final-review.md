# Final whole-branch review — 2026-09-15

Reviewer: Ampere (`01a0a38d-2267-78c0-a6f2-7c2334081dae`), resumed original task reviewer.
Range: `77eb2601a6..e7a54a992f`. Recorded by coordinator from the full returned assessment; file references are repository-relative.

Current disposition: the historical static gate below was subsequently resolved
under the separately user-approved TASK-32601 no-new-static-debt qualification.
See [scoped follow-up](static-gate-review.md) and [evidence](static-gate.md).
The original stable-file limitation and nonblocking capture-cleanup minor remain.

## Spec compliance — finding disposition

Original Important alias finding: **CLOSED UNDER APPROVED SCOPE; not technically fixed.**

ADR-138:22 explicitly adopts the user-approved stable-file assumption, preserves
normal SQLite-managed activity and existing protections, and acknowledges the
remaining substitution/detached-inode risk. It does not authorize infrastructure
changes or waive other gates.

Within that contract, `tldw_chatbook/Workflows/authoring.py:150` rejects visible
aliases before generic opening. The regression verifies that a refused import
leaves a foreign writer blocked until transaction completion:
`Tests/DB/test_workflows_authoring_storage.py:159`.

The check/open race remains outside the approved operating contract. The reviewer
found no inconsistent or scope-expanding waiver in the amendment, and no new
production breakage from the fix.

## Strengths

- Storage remains bounded. `DB/Workflows_DB.py:26` uses the existing private
  connection factory, without execution APIs or new ownership machinery. All
  four migration blob identities match the preserved source.
- Draft durability and conflicts have substantive protection. Revision saving
  performs transactional generation/head checks; retained draft writes survive
  caller cancellation (`Workflows/document_service.py:973`, `draft_session.py:219`).
- Integration preserves ownership boundaries. App initialization is lazy, while
  Console updates touch stable children rather than replacing the editor
  (`app.py:19163`, `UI/Workflows_Modules/console_context.py:78`).
- Compatibility claims are limited. The compatibility spec:56 distinguishes local
  preservation from server acceptance and identifies envelope-field loss. Pinned
  server inspection corroborates it; all 130 discovery names match.

## Findings

### Critical

None within the approved operating scope.

### Important — existing merge gate

Whole-file static analysis remains unsuccessful and unwaived. `task-1-report.md:410`
records 713 Ruff findings and five formatter failures. These are baseline debt,
not newly introduced authoring defects. Nevertheless, they prevent a clean full
DoD/merge verdict. Resolve the gate through separately authorized cleanup or an
explicit task-specific exception. ADR-125/TASK-32160's exception does not apply
automatically.

### Minor

Capture cleanup is incomplete. `Docs/superpowers/qa/workflows-authoring-dev/capture.py:22`
imports a bootstrap sandbox whose removal depends on pytest's session-finish
hook, which this standalone script never runs. Its factory cleanup at line 99
is also skipped when capture fails. Repeated captures leave disposable profile
data behind; failures bypass factory cleanup. Give the bootstrap sandbox an
explicit owned lifetime and place factory cleanup in `finally`. This is
nonblocking for the authoring implementation.

Requests/Kokoro warning noise remains nonblocking baseline debt, disclosed at
`task-1-report.md:438`. It is not evidence of a new workflow regression and does
not justify dependency/shared-infrastructure cleanup in this review.

## Scope and evidence limitations

Reviewed the amended contract and broader branch integration, including coordinator
documentation, ADR, compatibility, ledger and capture harness. All 40 production/
test file identities match the previously reviewed task plus fix.

- 374 targeted passes preceded the guard fix.
- The appended 35-test run covers that fix; the coordinator independently reports
  another 35 passes after the documentation amendment.
- Visual qualification relies on inspected capture assertions and preserved
  reviewer evidence, not a fresh visual inspection. The ship verdict covers the
  two original visual corrections, with later interaction changes separately
  documented at the QA README:80.

Targeted outside-diff checks covered harness isolation/cleanup and pinned server
compatibility sources. No tests, app boots, subagents, mutations or regenerated
diffs were used by the reviewer.

## Assessment

Record the scope-based finding closure, retain the explicit race limitation, and
resolve the static-analysis gate before declaring completion. Track capture
cleanup separately.

**Ready to merge? No.** The technical authoring review passes under the approved
stable-file contract, with one nonblocking harness minor. Full DoD remains unmet
because existing lint/format failures are unwaived; the task stays In Progress.
