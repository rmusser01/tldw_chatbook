---
id: TASK-34656
title: Integrate structured document checking into Library Notes
status: In Progress
created_date: 2026-10-07 03:18
dependencies:
- TASK-34571
- TASK-34651
- TASK-34652
- TASK-34653
- TASK-34655
- TASK-34570
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- Docs/superpowers/reviews/2026-10-11-structured-note-editor-integration.md
updated_date: 2026-10-12 01:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users detect source errors and explicitly format whole Notes without losing saves or undo history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Known source modes select whole-document checking; database mode is durable and file overrides remain per-open without renaming or metadata injection.
- [ ] #2 Typing retires old results immediately and checks after400ms or open/mode change, with separate save status and navigable Problems in logical coordinates.
- [ ] #3 Qualified explicit formatting is one native undoable edit with captured-source/full-key/editability guards; errors do not veto saves.
- [ ] #4 Structured/future modes render source safely and read-only excerpts never report complete checks or accept formatting.
- [ ] #5 Actual Database Notes and File Notes flows plus targeted tests/token governance qualify compact60x20 and normal layouts.
- [ ] #6 Relevant ADR assessment, scoped static/security evidence and completion documentation are recorded without hiding unfinished release gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (existing accepted decision applies; no new ADR). ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md; UI governance backlog/decisions/150-design-token-system-and-design-language.md and 031-tui-keybinding-and-footer-hint-conventions.md. Reason: direct integration of the reviewed local worker/history and canonical note-language decisions without changing save/storage authority. Immutable implementation base: 43540c2c3ca2565b9f442493b2caccab8e369c7d. 1. Read Task12 brief/context/review constraints, actual reviewed E4 final APIs and I1 bridge, F5 canonical session/recovery owners, repository guidance and relevant bounded lessons. Map actual app service, note_session_port, library screen/controller/canvas and File Notes consumers before changes. 2. Add meaningful mounted RED scenarios for invalid JSON save, immediate retirement/debounced/open/mode checking, metadata-only durable language and stale navigation/authority protection. 3. Integrate one app-owned service with active leases and a no-save-authority validation controller; add token-backed language/Check/Format/status/accessible Problems while preserving exact draft/file source and existing autosave/recovery/conflict limits. 4. Verify synchronous live-widget/full-key captured-source admission and one native undo edit with positive and stale/read-only/no-op controls in real Database Notes and File Notes, including BOM/CRLF/emoji/tab/EOF/focus and compact60x20. 5. Rebuild source CSS and run new integration tests plus library_notes_session, file_notes_service, token governance and CSS bundle sync selections; affected dependencies and scoped static/security checks, no introduced legacy debt against immutable base. Retain exact argv/cwd/env-patch/exit/output/hash/source evidence in durable task12 evidence, no full sweep or user data. 6. Update task notes/docs/ADR qualification when needed, self-review and commit explicit unit paths; keep In Progress for fresh independent review and preserve I1/I3/I5 physicalIME/production/platform gates. Report blockers to controller before scope/architecture deviation.
Task12 Ruling43 secondary-source guard under existingAC4/spec: actual ResearchQuickNote and both adapter read mappings drop mode; use exact markup=False Source preview for this secondary surface without extending metadata/save contracts or guessing omitted null. Keep raw edit/save/provenance and main Notes Markdown preview. Amend ADR194 before this UI adoption, then targeted Quick Notes widget/service controls including literal markup/YAML/JSON/ordinaryMarkdown plus scoped static evidence. Cost: this secondary surface loses rendered Markdown preview until qualified mode-presence contract exists. See .superpowers/sdd/2026-09-27-structured-document-editing/task-12-secondary-source-ruling.md. No new ADR/storage/provider/diagnostics owner.
Task12 Ruling44 File native-representation correction within existingAC3/5/ADR194: real first format preserves BOM/CRLF but second Format creates spurious dirty/history because engine CRLF proposal and incumbent LF TextArea compare differently. Before adopting, amend ADR194; map proposal only at File adapter to incumbent LF representation while keeping captured opened facts and existing disk serializer. Detect mapped no-op before edit/history/dirty/autosave; preserve E1 typed construction/state/ranges and full-key/current-source guards. Qualify first-format/second-no-op/native undo/caret/byteexact BOMCRLF and unsupported mixed/loneCR profiles. No parser/generic bridge/canonical DB body/save serializer normalization. Distinct actual DB issues require controller scope review. See .superpowers/sdd/2026-09-27-structured-document-editing/task-12-file-representation-ruling.md.
Task12 Ruling45 existingAC3/4/5 source-loss fail-safe: actual Database mixed line endings do not round-trip TextArea; homogeneousCRLFformat/noop/save and loneCRroundtrip pass. Amend ADR194 before guard. Preserve canonical exact body/escaped preview, fence body edits/format with unsupported representation status, keep title/language/save using canonical original. Cover every existing widget-to-draft pre-recompose/queuedChanged/save-sync/transition caller; no mode can silently bypass loss fence. Verify realSQLite exact retention across open/title/mode/save, stale/navigation, canonical copy/export and supported native profile positives. No second persisted body/source normalization/engine/generic bridge/File mapping expansion. See .superpowers/sdd/2026-09-27-structured-document-editing/task-12-database-representation-ruling.md.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR194 I2 on canonical Database Notes and retained File Notes hosts: app-owned single worker service, permanent editor leases, mode/Check/Format/independent status/keyboard Problems, exact source previews and native one-transaction formatting through I1. Database mode stays durable; File mode is temporary and original BOM/newline/excerpt/hash/recovery owners remain intact. Rulings43/44/45 are amended in ADR194: secondary Source preview tradeoff, File-only CRLF mapping with truthful no-op, and lossy native Database source fence preserving title/mode/save/canonical copy-export fields.

Targeted covering213 passed; amended native-host34 passed; final actual-rendered-body/CSS16 passed. Exact commands, source manifests, RED/GREEN, raw output hashes, actual app-owned shutdown and scoped static comparisons are in .superpowers/sdd/2026-09-27-structured-document-editing/task-12-evidence and task-12-report.md. New production strict checks pass; expanded test-import mypy retains the immutable Screen type-comment parse barrier. FD warnings323/328 remain: immutable factory/BaseDB/allocator controls substantiate existing DB allocations but do not account for every descriptor. Physical OSIME/screen-reader/platform/clean-parent gates stay explicit. Docs/superpowers/reviews/2026-10-11-structured-note-editor-integration.md records qualifications and UX costs. Leave In Progress for fresh controller review; no full sweep/server/export-label expansion/push/merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
