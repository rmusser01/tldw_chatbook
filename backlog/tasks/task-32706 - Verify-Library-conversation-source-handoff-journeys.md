---
id: TASK-32706
title: Verify Library conversation source handoff journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 00:52'
updated_date: '2026-09-17 01:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qualify the separate Use as source journey, including workspace linking, Console staging and draft preservation, with accurate documentation of the current evidence-content limit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use as source links an eligible unlinked conversation to the active workspace and stages its exact source identity in the existing Console session.
- [x] #2 The current populated Console draft and original conversation/messages remain unchanged; staging makes no provider send.
- [x] #3 Repeated use of an already-linked source does not duplicate membership, and the original link receipt can undo only its own membership.
- [x] #4 Native dark/light compact/wide checks, targeted regressions, independent persistence checks and normal shutdown are recorded.
- [x] #5 User instructions distinguish Use as source from Resume and retain the separately tracked content-fidelity limitation.
- [x] #6 Review evidence, task notes and the continuation ledger identify remaining scope honestly.
- [x] #7 Related source-staging regression fixtures follow current startup and controller contracts without weakening their behavior assertions.
- [x] #8 Any reproduced shared source-staging defect is fixed or explicitly tracked with its remaining validation scope before closing the review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md; backlog/decisions/005-console-workspace-server-readiness.md (existing)
Reason: verify established source-staging and workspace membership contracts; no new architecture.
1. Inspect current handoff/link/Undo owners and TASK-2376 content-fidelity limits; run targeted existing regressions.
2. Exercise unlinked and already-linked Use as source in a guarded native profile across both themes and terminal sizes, with a populated current draft and unrelated blocked row.
3. Verify exact staged identity, no duplicate session/membership or send, original source preservation, Undo link, Unstage, process exit and database/config isolation.
4. Correct stale user-guide action names/instructions, document the known content limit, review captures and evidence, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified Conversations Use as source across dark/light and compact/wide terminals: link-on-use, repeated already-linked staging, exact source identity in the existing Console session, populated-draft preservation, Un-stage and retained-receipt Undo. Five source conversations/ten messages remain unchanged; only the original foreign workspace link persists after Undo. Sixteen final captures were rendered and inspected; normal exit and independent read-only checks cover all ten private databases and unchanged default-profile hashes.

Repaired stale splash/controller/geometry fixtures. Their cleanup exposed existing TASK-2502 on the separate live-work channel, which was repaired under that task after updating its plan/AC. Both first and replacement warm launches now paint; native verification includes eight such launches. The final focused selection passes130 tests. Eight adjacent suspend/roleplay-resume failures reproduce on unchanged production code and are documented as remaining validation limits.

Updated the guide to distinguish Use as source, Resume, Un-stage and Undo. TASK-2376 remains open: staged conversation evidence is a title-based label, not transcript text. This is the next bounded repair.

Files: user guide, live-work fixture/regression tests, resume census, source QA directory, continuation ledger and testing lessons. Existing ADR-005/147 apply; no new ADR or CSS change. Static checks add no production Ruff findings, with 212 inherited diagnostics retained; changed tests/helpers and changed production ranges are formatted. Independent review and the detailed evidence/limits are recorded in Docs/superpowers/qa/2026-09-16-conversation-source/README.md. No full suite, push or merge.
<!-- SECTION:NOTES:END -->
