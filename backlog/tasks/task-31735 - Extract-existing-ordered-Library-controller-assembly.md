---
id: TASK-31735
title: Extract existing ordered Library controller assembly
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:29'
updated_date: '2026-09-05 19:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give existing controller construction one explicit assembly home while preserving initialization position, dependency names, late binding, and ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Six existing controller constructions retain identical ordered call ASTs and late-bound dependencies
- [x] #2 Library meets its unchanged screen line ceiling and all existing controller ownership checks remain enforced
- [x] #3 Relevant Conversations, Notes, reuse and architecture verification is recorded without behavioral assertion changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize existing construction, late binding and conversation UI before moving.
2. Move the contiguous six-controller construction block into Library_Modules/wiring.py at the identical initialization position; preserve every named callable and controller body.
3. Compare normalized constructor ASTs against the pre-move source, run scoped architecture and real UI coverage, static-check and request parent review.

ADR required: no
ADR path: N/A
Reason: Mechanical assembly move applies the existing DESIGN.md section 7 Console wiring precedent without changing runtime ownership or contracts.

Detailed plan: `Docs/superpowers/plans/2026-09-05-library-assembly-cleanup.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the contiguous six-controller construction block into Library_Modules/wiring.py at the identical initialization position. All six constructor Call ASTs match f8e63aea4f after self-to-screen normalization; every other screen method AST is unchanged. Named lambda ports, canonical controller classes, state owners and DOM/event bodies are preserved. Only the new helper is imported during construction, adding no new guarded preimport module.
Library is 41303 lines / 1301 methods; assembly is pinned at 338 lines. Characterization passed 8 checks before the move; 75 architecture/ratchet checks passed after it, with an independent final 46-check rerun after the helper import was made constructor-local. All 16 Notes responsive/identity/return checks passed. Seven-file controller/reuse UI selection: 276 passed, 8 failed (217.55s). Every failure reproduces at the exact pre-move baseline in an isolated worktree: 8 failed (13.67s); these are five stale Skills CSS censuses, a standalone Skills picker hit-test, a bare Skills fixture missing focused, and a RAG prompted-source-count expectation. No assertions were relaxed or runtime fixes mixed into the move.
New module/test and module-ratchet Ruff+format checks pass; screen retains the same 40 preexisting Ruff findings; diffcheck passes. Diagnostic statement comparison: screen 100 to 100 unchanged, new assembly zero. ADR required: no, existing DESIGN.md section 7 assembly precedent. Recipe section 26 updated. Parent final review pending.
Evidence: /private/tmp/library-assembly-architecture-after.xml; /private/tmp/library-assembly-notes-after.xml; /private/tmp/library-assembly-ui-after.xml; /private/tmp/library-assembly-eight-baseline.xml.

Final parent review approved the exact assembly and shared screen pin at 41303 / 1301. Constructor alias patch audit searched all 9 test files naming the six classes and AST-inspected patch/setattr calls: zero constructor alias patch targets. The final scoped commit preserves existing compatibility class names; tests do not depend on constructor lookup through the screen module. All acceptance criteria are complete; broader preexisting failures remain explicitly disclosed above.
<!-- SECTION:NOTES:END -->
