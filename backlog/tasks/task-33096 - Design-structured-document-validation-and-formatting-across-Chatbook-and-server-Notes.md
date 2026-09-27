---
id: TASK-33096
title: >-
  Design structured document validation and formatting across Chatbook and
  server Notes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 20:25'
updated_date: '2026-09-27 23:00'
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
- [x] #5 The user approves the written spec before implementation planning begins.
- [x] #6 The approved spec has linked implementation plans with exact integration targets, dependency gates, targeted tests, and a complete acceptance-criterion coverage map.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile the approved scope with source-level review findings in both repositories.
2. Write a cross-application design and proposed canonical ADR with measurable release criteria.
3. Review links, task identity, scope, compatibility, and document consistency; commit only the design artifacts.
4. Present the written spec for user review before invoking writing-plans.
5. After approval, map verified integration paths and write ordered foundation, engine, and editor implementation plans.
6. Self-review interface consistency and spec coverage, verify document links and scope, and present execution choices.

ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md
Reason: Portable note language, versioned sync, source-preserving formatting, and local validation runtime boundaries. Server ADR-031 also governs core-note sync.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the approved structured-document editing design and implementation planning. User approved the written spec on 2026-09-27; ADR-194 is Accepted.

The spec adds 15 measurable release criteria covering exact source preservation, versioned language metadata, true editor undo, scoped draft recovery, strict language behavior, source-safe exports, and bounded local workers. The programme plan splits execution into foundation, engine, and editor plans with 15 units, explicit dependency gates, actual integration paths, test examples, targeted commands, and full AC1-AC15 mapping. Existing server ADR-031 is linked and a server-local extension is required before its protocol changes.

Artifacts: Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md; Docs/superpowers/plans/2026-09-27-structured-document-editing.md and its foundations/engines/editors subplans; backlog/decisions/194-structured-note-language-and-local-editor-validation.md; ADR index.

Verification: all six spec/ADR/plan documents passed relative-link, placeholder, and whitespace checks; Python/JSON examples parse; 15 unique units and AC1-AC15 coverage verified; checked named integration paths against both repositories. Scoped Backlog ID/filename guard passed. Self-review corrected test fixture setup and removed an inferred nonexistent source-module path. Repository-wide Backlog guard has pre-existing unrelated duplicate IDs; no changes were made to those tasks.

ADR required: yes. ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md. No product code, dependency installs, runtime tests, or execution delegation occurred. Completion is for design/planning only; all implementation acceptance criteria remain unchecked. No generalizable new lesson beyond the existing source-verification guidance.
<!-- SECTION:NOTES:END -->
