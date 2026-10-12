---
id: TASK-34657
title: Expose structured source downloads and portable note exports
status: In Progress
created_date: 2026-10-07 03:19
dependencies:
- TASK-34653
- TASK-34656
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editors.md
updated_date: 2026-10-12 09:05
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make source downloads and portable notes distinct and preserve language through their real user actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Download source returns exact source with a matching known-mode extension; Export note returns a versioned portable envelope with source/language/metadata.
- [ ] #2 Real action wiring uses the reviewed import/export builders and scoped authority without passing structured source through Markdown conversion.
- [ ] #3 User guidance explains check/save separation, supported/refused constructs, limits and undo behavior; targeted action/static and source-byte controls are recorded.
- [ ] #4 Relevant ADR assessment, scoped static/security evidence and completion documentation are recorded without hiding unfinished release gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md (existing); server counterparts ADR065-069 apply to consumed boundaries.
Reason: direct source/portable action wiring and user guidance implement accepted owners/contracts without a new schema, transport, parser/runtime or ownership policy.
1. Read Task34657, task14 brief/context, final reviewed F6/I2/I3 reports, applicable AGENTS/lessons and design-language constitution. Capture action source and canonical mode/presence from the existing owner, including native representation fences and temporary views.
2. Add meaningful failing native action controls for exact JSON source bytes/known extension versus tldw.note.v1 source/language/metadata; current source/authority and refusal controls use the real existing builders.
3. Wire distinct Download source and Export note actions through incumbent Library/export boundaries, preserving current authority, overwrite confirmation and explicit Markdown conversion. Reuse the archive4/Canvas3 paths and pre-effect unsupported-version refusal.
4. Add Docs/User_Guide/structured-document-editing.md and extend the existing server guide specified by Ruling54. Explain temporary mode, save/check separation, refusal/limits/undo, empty recovery and syntax versus intended YAML structure; verify existing documentation links.
5. Run named action and focused F6 round-trip controls, actual owned native file-dialog/browser output-byte controls, scoped token/static/security/no-introduced formatter evidence against the immutable task bases. Use isolated temporary state, no full sweep or platform claims from synthetic controls.
6. Record exact source-bound commands/exits/raw outputs and qualification limits, self-review and separately commit explicit C/S unit paths including this task and the Root-corrected editors-plan reference. Leave In Progress for independent review; no nested agents.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented distinct Download source and Export note (.note.json) actions through complete Database Library session snapshots, existing transfer admission and native FileSave/path writer. Source writes UTF-8 bytes; portable output uses the existing tldw.note.v1 validator, retaining source/language/keywords. File Notes serializer/authority/excerpt paths are untouched. Added the structured editing guide and source/picker action controls. Existing ADR194 and server ADR065-069 govern; no new ADR is required.
Verification: native visible action/picker 5 passed; affected action/builder 60 passed; focused F6/archive4+Canvas/pre-effect+token 40 passed. Strict new test checks pass; immutable-base Ruff228->226 and Bandit17->17 with no introduced findings; actual edited-region Black checks pass. Scoped mypy remains blocked by the unchanged Screen '# type:' comment (BASE11106); no whole-suite or physical TTY/File Notes download claim. Full report/evidence: .superpowers/sdd/2026-09-27-structured-document-editing/task-14-report.md. Self-review complete; keep In Progress and all AC/DoD unchecked for Root review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
