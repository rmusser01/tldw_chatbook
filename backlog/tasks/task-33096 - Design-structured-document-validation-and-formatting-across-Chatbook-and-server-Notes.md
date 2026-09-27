---
id: TASK-33096
title: >-
  Design structured document validation and formatting across Chatbook and
  server Notes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-27 20:25'
updated_date: '2026-09-27 20:30'
labels:
  - design
  - notes
  - editors
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define consistent live syntax feedback and explicit document formatting for Chatbook and tldw_server Notes while preserving source text, drafts, and portable language selections.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design records the approved YAML, JSON, and JSONL editing scope and save behavior.
- [x] #2 Source preservation, versioned language metadata transport, and native editor undo are explicit release acceptance criteria.
- [x] #3 Concrete code review findings are addressed with compatibility rules and targeted verification evidence.
- [x] #4 The design and proposed ADR are self-reviewed and presented for user review before implementation planning.
- [ ] #5 The user approves the written spec before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile the approved scope with source-level review findings in both repositories.
2. Write a cross-application design and proposed canonical ADR with measurable release criteria.
3. Review links, task identity, scope, compatibility, and document consistency; commit only the design artifacts.
4. Present the written spec for user review before invoking writing-plans.

ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md
Reason: Adds portable note language metadata and versioned sync behavior, and defines validation/formatting runtime and source-preservation boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Authored the revised cross-application spec and proposed ADR-194. The design retains local engines and whole-document syntax feedback, and adds explicit source preservation, negotiated language metadata, real editor undo, empty-draft recovery, source-safe previews/exports, and bounded diagnostic workers. Reviewed source paths in both repositories; this is documentation work, not runtime qualification.

Files: Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md; backlog/decisions/194-structured-note-language-and-local-editor-validation.md; backlog/decisions/README.md.

ADR required: yes. ADR-194 is proposed pending written-spec approval; existing ADR-027/029/073 ownership remains intact. Link and placeholder checks passed. User review remains pending; no implementation plan, product code, or dependency changes made.

Verification: spec/ADR relative links, placeholder scan, acceptance-criterion numbering, and whitespace passed. The Backlog guard passed for this task in an isolated check; committed-ref collision checks found no competing TASK-33096 or ADR-194 claim. The repository-wide Backlog ID guard fails on pre-existing unrelated duplicate IDs (including TASK-15665 and TASK-18311); none involves this task. The spec was submitted to the Codex file panel for review. Status remains In Progress pending explicit written-spec approval (AC5). No runtime tests were run for this documentation-only change.
<!-- SECTION:NOTES:END -->
