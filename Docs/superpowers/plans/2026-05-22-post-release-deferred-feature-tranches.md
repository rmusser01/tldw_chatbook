# Post-Release Deferred Feature Tranches

Date: 2026-05-22
Owner: `TASK-60.4`
Evidence source: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-22-cross-screen-workflow-validation.md`

## Purpose

This plan converts verified residual risks from the post-release actual-use audit into staged follow-up work. It does not approve broad implementation on its own; it makes the remaining future work visible, ordered, and testable.

No deferred implementation starts before open P0/P1 usability defects are triaged. Verified shipped behavior stays separate from deferred future work so roadmap status cannot imply that blocked service depth is already complete.

## Evidence Gate

All tranches below use `TASK-60.3` actual-use audit evidence as their entry condition:

- Home, Console, Library/Search-RAG, Artifacts/Chatbooks, Personas, Skills, Watchlists, Schedules, Workflows, ACP, MCP, and Settings were exercised through mounted Textual regressions and documented workflow evidence.
- No unresolved P0/P1 cross-screen workflow findings remain for the verified local-first scope.
- Service-depth gaps are recoverable or future-scoped, not hidden as shipped behavior.
- Each tranche must add actual-use QA evidence before completion; mounted assertions alone are not enough.

## Tranche 1: ACP Runtime Launch

Backlog owner: `TASK-60.4.1`
Audit evidence: ACP runtime payloads remain recoverably blocked because the UI can expose ACP session intent, but there is no real ACP-compatible runtime launch path yet.

Scope:

- Launch or attach to an ACP runtime from the ACP destination and Console handoff surfaces.
- Keep unavailable runtime states source-honest until launch support exists.
- Preserve Console as the task/run control surface once an ACP package is active.
- Validate ACP task/run package handoff as the first server-backed handoff target.

Exit evidence:

- Actual app QA proves an ACP task/run package can move from local planning into an ACP runtime or present a precise recovery path.
- Home, Console, and ACP agree on runtime status and active-work controls.

## Tranche 2: Write Sync Promotion

Backlog owner: `TASK-60.4.2`
Audit evidence: write sync remains deferred; current sync-related surfaces are intentionally read-only or dry-run so users are not misled into thinking mutation replay is safe.

Scope:

- Promote write sync only after dry-run, conflict, rollback, and authority labels are understandable.
- Gate mutation replay behind explicit user review.
- Preserve local-first control if server sync is unavailable or rejected.
- Keep source, artifact, workspace, and collection authority visible across Library, Home, Settings, and Console.

Exit evidence:

- Actual app QA proves users can preview, approve, recover, or abort write-sync operations without silent mutation.
- Test fixtures cover conflict and rollback recovery paths.

## Tranche 3: Workspaces And Library Depth

Backlog owner: `TASK-60.4.3`
Audit evidence: Workspace switching must not hide Library items. Users must be able to view and search all Library and Notes items while only the active workspace controls context eligibility, source authority, and Console manipulation.

Scope:

- Deepen workspace membership, Collections membership, and source authority without removing global Library visibility.
- Make Import/Export depth work under Library, not Artifacts.
- Show workspace ownership tags and eligibility states on Library items.
- Preserve single-user local workspace behavior for offline/shared-workspace fallback.

Exit evidence:

- Actual app QA proves users can view/edit/search across workspaces while only active-workspace items can be staged into Console context.
- Collections membership and deeper Import/Export are visible in Library and do not regress existing local Collections management.

## Tranche 4: Citation And Snippet Carry-Through

Backlog owner: `TASK-60.4.4`
Audit evidence: citation/snippet carry-through remains downstream future work; Library/Search-RAG can stage evidence, but downstream Chat answers, Artifacts, and exported Chatbooks still need durable source attribution.

Scope:

- Carry retrieved snippets, citations, and source authority from Library/Search-RAG into Console responses.
- Preserve citations/snippets in saved Artifacts and exported Chatbooks.
- Keep citation data readable to users and structured enough for future replay/export tooling.
- Avoid implying grounded answers when no source evidence is attached.

Exit evidence:

- Actual app QA proves source evidence survives the path from Library query to Console answer to saved Chatbook.
- Tests verify missing-source and stale-source recovery states.

## Tranche 5: Optional Dependency And Package Polish

Backlog owner: `TASK-60.4.5`
Audit evidence: optional dependency recovery remains source-honest; unavailable advanced features can identify missing setup, but install paths and package metadata still need polish.

Scope:

- Make missing dependency groups and recovery commands visible where optional media, RAG, MCP, web, or server features are unavailable.
- Clarify baseline local-first install versus advanced extras.
- Keep packaging metadata and setup docs aligned with the real optional feature matrix.
- Avoid presenting missing optional features as core app failure.

Exit evidence:

- Actual app QA proves missing optional features show the responsible dependency group and recovery action.
- Packaging and setup docs pass focused regression checks without machine-specific paths.

## Priority Rule

The tranches are ordered by workflow unlock and risk:

1. ACP runtime launch unlocks the first server-backed task/run package handoff.
2. Write sync promotion is safety-critical and must remain gated until mutation recovery is clear.
3. Workspaces and Library depth clarifies source authority for the operating context.
4. Citation and snippet carry-through increases answer trust and artifact durability.
5. Optional dependency and package polish improves recovery and contributor readiness without masking core workflow defects.

If future audits discover new P0/P1 defects, pause these tranches and address the usability breakage first.
