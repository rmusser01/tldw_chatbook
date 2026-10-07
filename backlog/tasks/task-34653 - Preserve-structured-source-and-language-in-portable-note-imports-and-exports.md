---
id: TASK-34653
title: Preserve structured source and language in portable note imports and exports
status: In Progress
created_date: 2026-10-07 03:08
dependencies:
- TASK-34571
- TASK-34651
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-07 10:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users move source files and portable notes without losing document text or language.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Raw source exports retain exact text with a matching mode extension; portable notes retain source as a string plus language and metadata.
- [ ] #2 Versioned Chatbook/portable envelopes preserve Canvas and structured Notes together, with older supported readers retained.
- [ ] #3 Unknown archive/envelope versions fail before storage mutation; recognized source extensions carry mode without sniffing document text.
- [ ] #4 Targeted source/envelope/import/transaction tests verify exact-source round trips, atomic rejection and scoped static evidence.
- [ ] #5 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md (existing programme decision)
Reason: implement approved portable source/language archive and import service contracts; amend the existing decision before a new contract detail, with no new storage or transport owner.
1. Read task6 brief, owner preparation/Ruling29, current F5 reviewed exports, applicable instructions/lessons and governing ADR. Recheck archive versions and actual import/API/fingerprint owners.
2. Add meaningful failing exact raw source, portable envelope/language, Canvas-plus-Notes archive, old-reader/unknown-version and approval-fingerprint/retry controls.
3. Extend versioned archive/note records and actual source-aware import planning/fingerprint/execution; retain supported v1-v3 readers and preserve Canvas when selecting the verified new version.
4. Extend existing server Notes import schema/normalizer/endpoint and U types/hooks/export builders with explicit raw-source versus portable/legacy contracts, no source-language sniffing or synthetic empty content. Qualify whole versioned-batch refusal before mutation and truthful older import endpoint refusal while preserving ordinary legacy behavior and omitted/null semantics.
5. Run targeted C source/archive/Canvas/transaction/receipt tests and real S API serialization/materialization plus affected U import/export tests; whole W typecheck after shared edits and scoped immutable-base static/security/formatter checks. No full sweep, new service or editor adoption.
6. Update relevant docs/ADR and concise task evidence, self-review, append exact RED/GREEN/final commands/output and boundaries to task-6-report.md, commit explicit C/S unit paths separately; keep In Progress for controller independent review.
Rulings35/36 (task-6-contract-rulings.md and progress.md): exact portable format_version=tldw.note.v1 with portable_note/source import discriminators; explicit raw-source context defaults false for old JSON/YAML containers, JSONL may be source-only. Missing portable language is invalid; filename-less raw text uses a safe fallback title and omits inferred mode. All new Notes archives select verified v4 including null and unique stable-identity JSON carriers, retaining Canvas; conversation-only version selection remains. Amend governing ADR194/065 before these contract edits; preserve old readers, legacy omitted-field identity fingerprint and actual import path/authority validation.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented tldw.note.v1 note carriers and Chatbook4.0 for every new Notes archive under ADR194 and Rulings35/36, retaining Canvas and v1-v3 readers. Explicit raw source parsing preserves exact UTF-8 source and extension mode; legacy containers retain their default readers. Actual planner, approval fingerprint and receipt/executor owners carry language with omission/null distinction, preserving historical omitted fingerprints and binding approved source/mode effects. Targeted real SQLite/archive/approval verification and immutable-base/scoped static/security/formatter evidence is in .superpowers/sdd/2026-09-27-structured-document-editing/task-6-report.md. In Progress pending independent review; no new transport/store/service or editor adoption.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
