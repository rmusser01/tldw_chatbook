---
id: TASK-32869
title: Specify Library Artifacts integration
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 21:50'
updated_date: '2026-09-20 06:11'
labels:
  - library
  - artifacts
  - design
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the approved Library-style Artifacts proposal and review into an explicit design, architectural decision, and staged implementation plan. Reports default to all with a Kept filter; the existing ZIP-pack manager stays linked from Library initially.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design records all reports by default with an explicit Kept filter and preserves manual keeping semantics.
- [x] #2 The design preserves kept reports after source deletion and documents catalog, scope, sharing, keyboard, and navigation behavior.
- [x] #3 A canonical ADR and executable implementation plan identify boundaries, alternatives, targeted verification, and atomic delivery stages.
- [x] #4 Documentation references and task metadata are verified without changing application code.
- [x] #5 The four final review findings are resolved in the specification and executable plan: Keep conflicts before writes, coherent read snapshots, suspend-aware dialog publication, and consistent reader/list keyboard behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Apply the approved final review corrections to the existing design and staged plan, preserving All reports with Kept filtering and the linked ZIP-pack manager.
2. Specify service-owned pre-write Keep conflict validation including concurrent-create fallback; add a no-mutation regression requirement and explicit source-owned read snapshot plus concurrent-write verification.
3. Specify Library suspend/resume presentation guards separately from retained data reads and approved share operations; make the Enter/Back/Escape focus contract and tests agree.
4. Strengthen the open implementation task acceptance criteria, verify document links and example syntax, check ADR-172 consistency and diff scope, then close the design task.
ADR required: no
ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md
Reason: These corrections make the implementation honor the existing accepted no-corruption, coherent-read, ownership, and focus contracts; they do not change the architectural decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Finalized the Library Artifacts specification and staged implementation plan with all four approved review corrections. Keep compatibility is checked by the service before parent/script mutations, including raced creates; browse opens an explicit deferred read snapshot with borrowed-transaction ownership preserved; pending dialog/focus effects are guarded across suspend and navigation while retained reads and accepted sharing keep their lifetime; Enter/Back/Escape behavior and its tests now agree.
Updated Docs/superpowers/specs/2026-09-19-library-artifacts-design.md and Docs/superpowers/plans/2026-09-19-library-artifacts.md, plus the relevant open implementation acceptance criteria. All reports with a Kept filter and the linked ZIP-pack manager remain the approved design. ADR-172 at backlog/decisions/172-library-artifacts-browse-and-navigation.md remains unchanged: these corrections implement its existing boundaries, so no new ADR is required.
Verification passed: 12 links in the revised documents resolve; all 8 Python examples parse; spec/plan global constraints match; all four findings and their regression requirements are present; four Backlog records have the expected states, criteria, and backward-only references; no placeholders or whitespace errors were found. The diff contains only the specification, plan, and three Backlog records. Application code and runtime tests remain for the open implementation stages; no runtime fix is claimed here.
<!-- SECTION:NOTES:END -->
