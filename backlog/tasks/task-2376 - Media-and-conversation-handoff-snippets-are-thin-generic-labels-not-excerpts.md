---
id: TASK-2376
title: 'Media and conversation handoff snippets are thin generic labels, not excerpts'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-04 20:07'
updated_date: '2026-09-17 02:07'
labels:
  - console
  - rag
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`capture_console_staged_evidence_for_chat`'s snippet formula prefers `display_summary` over `body` when both are present. Media and conversation handoff builders set `display_summary` to a short generic label (for example, "Media staged: {title}"), so the snippet the model actually receives is a label, not an excerpt of the real content. Notes are unaffected, since their builder does not set a competing `display_summary`.

PR-T1 Task 9 (task-2374) fixed the underlying zero-content bug for these handoff kinds and self-flagged this as a residual: the content delivered today is thin but honest (a real, correctly attributed reference), not silently empty. This task is about upgrading fidelity, not about a correctness regression.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media and conversation handoffs deliver a real content excerpt in the snippet sent to the model, not just a generic label
- [x] #2 Notes' existing (unaffected) snippet behavior is preserved
- [x] #3 A test pins the excerpt's actual content, not merely that a snippet is present
- [x] #4 Excerpts are bounded and sanitized; conversation content comes only from the matching fully loaded reader, and empty sources are described honestly.
- [x] #5 Native source staging preserves the existing draft and saved source records; targeted Notes and RAG behavior stays unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/005-console-workspace-server-readiness.md; backlog/decisions/147-conversation-archive-and-exact-resume.md.
Reason: repair content fidelity within the existing typed source handoff and reader eligibility boundaries; no schema, authority, or runtime changes.
1. Add failing content assertions through the actual builders and Console evidence capture, including bounds, reader identity, empty content, Notes and RAG compatibility.
2. Prefer source bodies for media/conversation evidence; put actual content first and construct a bounded conversation transcript from the matching loaded reader.
3. Run targeted regressions, static checks and private native media/conversation staging with draft/persistence assertions.
4. Review the diff, record evidence and update the continuation ledger before local commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Media and conversation evidence now uses actual source bodies. Library Media retains its 500-character excerpt and puts it before metadata; Conversations supplies a bounded 3000-character speaker-labeled transcript only from the matching complete reader. Speaker labels are separately capped at 80 characters after review exposed a text-starvation edge case. Empty sources and truncation are explicit. Notes and RAG summary behavior remains unchanged.
Production changes: Library conversation builder, media controller, handoff limits, and Console snippet choice. Regression coverage asserts exact content through capture, identity/version/generation fences, empty/large sources, sanitization, long metadata and long speakers.
Verification: 207 targeted tests plus 7 legacy media tests pass; changed tests/QA helpers pass Ruff and formatting; production lint adds no diagnostics to its documented baseline. Four native theme/size cells pass actual media/conversation capture, draft/session preservation, repeated source use, Un-stage and link Undo. Read-only checks confirm unchanged source records, 10 healthy private databases and unchanged default-profile files. Normal exit and exact process absence verified. No full suite or provider-generation claim.
ADR check: existing ADR-005 (backlog/decisions/005-console-workspace-server-readiness.md) and ADR-147 (backlog/decisions/147-conversation-archive-and-exact-resume.md); no new architecture decision. No CSS/token changes. Evidence, attempts, review, reproduction and limitations: Docs/superpowers/qa/2026-09-16-handoff-excerpts/README.md. Continuation ledger updated in Docs/superpowers/reports/2026-09-14-library-workflow-audit.md.
<!-- SECTION:NOTES:END -->
