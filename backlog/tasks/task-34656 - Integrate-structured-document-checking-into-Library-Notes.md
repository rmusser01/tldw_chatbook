---
id: TASK-34656
title: Integrate structured document checking into Library Notes
status: Done
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
updated_date: 2026-10-12 02:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users detect source errors and explicitly format whole Notes without losing saves or undo history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Known source modes select whole-document checking; database mode is durable and file overrides remain per-open without renaming or metadata injection.
- [x] #2 Typing retires old results immediately and checks after400ms or open/mode change, with separate save status and navigable Problems in logical coordinates.
- [x] #3 Qualified explicit formatting is one native undoable edit with captured-source/full-key/editability guards; errors do not veto saves.
- [x] #4 Structured/future modes render source safely and read-only excerpts never report complete checks or accept formatting.
- [x] #5 Actual Database Notes and File Notes flows plus targeted tests/token governance qualify compact60x20 and normal layouts.
- [x] #6 Relevant ADR assessment, scoped static/security evidence and completion documentation are recorded without hiding unfinished release gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (existing accepted decision applies; no new ADR). ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md; UI governance backlog/decisions/150-design-token-system-and-design-language.md and 031-tui-keybinding-and-footer-hint-conventions.md. Reason: direct integration of the reviewed local worker/history and canonical note-language decisions without changing save/storage authority. Immutable implementation base: 43540c2c3ca2565b9f442493b2caccab8e369c7d. 1. Read Task12 brief/context/review constraints, actual reviewed E4 final APIs and I1 bridge, F5 canonical session/recovery owners, repository guidance and relevant bounded lessons. Map actual app service, note_session_port, library screen/controller/canvas and File Notes consumers before changes. 2. Add meaningful mounted RED scenarios for invalid JSON save, immediate retirement/debounced/open/mode checking, metadata-only durable language and stale navigation/authority protection. 3. Integrate one app-owned service with active leases and a no-save-authority validation controller; add token-backed language/Check/Format/status/accessible Problems while preserving exact draft/file source and existing autosave/recovery/conflict limits. 4. Verify synchronous live-widget/full-key captured-source admission and one native undo edit with positive and stale/read-only/no-op controls in real Database Notes and File Notes, including BOM/CRLF/emoji/tab/EOF/focus and compact60x20. 5. Rebuild source CSS and run new integration tests plus library_notes_session, file_notes_service, token governance and CSS bundle sync selections; affected dependencies and scoped static/security checks, no introduced legacy debt against immutable base. Retain exact argv/cwd/env-patch/exit/output/hash/source evidence in durable task12 evidence, no full sweep or user data. 6. Update task notes/docs/ADR qualification when needed, self-review and commit explicit unit paths; keep In Progress for fresh independent review and preserve I1/I3/I5 physicalIME/production/platform gates. Report blockers to controller before scope/architecture deviation.
Task12 Ruling43 secondary-source guard under existingAC4/spec: actual ResearchQuickNote and both adapter read mappings drop mode; use exact markup=False Source preview for this secondary surface without extending metadata/save contracts or guessing omitted null. Keep raw edit/save/provenance and main Notes Markdown preview. Amend ADR194 before this UI adoption, then targeted Quick Notes widget/service controls including literal markup/YAML/JSON/ordinaryMarkdown plus scoped static evidence. Cost: this secondary surface loses rendered Markdown preview until qualified mode-presence contract exists. See .superpowers/sdd/2026-09-27-structured-document-editing/task-12-secondary-source-ruling.md. No new ADR/storage/provider/diagnostics owner.
Task12 Ruling44 File native-representation correction within existingAC3/5/ADR194: real first format preserves BOM/CRLF but second Format creates spurious dirty/history because engine CRLF proposal and incumbent LF TextArea compare differently. Before adopting, amend ADR194; map proposal only at File adapter to incumbent LF representation while keeping captured opened facts and existing disk serializer. Detect mapped no-op before edit/history/dirty/autosave; preserve E1 typed construction/state/ranges and full-key/current-source guards. Qualify first-format/second-no-op/native undo/caret/byteexact BOMCRLF and unsupported mixed/loneCR profiles. No parser/generic bridge/canonical DB body/save serializer normalization. Distinct actual DB issues require controller scope review. See .superpowers/sdd/2026-09-27-structured-document-editing/task-12-file-representation-ruling.md.
Task12 Ruling45 existingAC3/4/5 source-loss fail-safe: actual Database mixed line endings do not round-trip TextArea; homogeneousCRLFformat/noop/save and loneCRroundtrip pass. Amend ADR194 before guard. Preserve canonical exact body/escaped preview, fence body edits/format with unsupported representation status, keep title/language/save using canonical original. Cover every existing widget-to-draft pre-recompose/queuedChanged/save-sync/transition caller; no mode can silently bypass loss fence. Verify realSQLite exact retention across open/title/mode/save, stale/navigation, canonical copy/export and supported native profile positives. No second persisted body/source normalization/engine/generic bridge/File mapping expansion. See .superpowers/sdd/2026-09-27-structured-document-editing/task-12-database-representation-ruling.md.
Fix round1/5 (original review I1 only; Ruling46 recorded before adoption). ADR required: yes (existing governing ADR). ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md. Reason: qualify the same exact-source/native-round-trip fence at the existing secondary Quick Notes boundary without changing source ownership, metadata omission or accepted storage/provider rules. Amend secondary qualification before production adoption.
1. Retain the independent mixed-ending preview RED and trace current canonical ResearchQuickNote source, native load/preview and every affected widget-to-draft/title/save/navigation caller.
2. Add mounted mixed-source preview and unsafe-body admission controls with real temporary SQLite/canonical title-only/save and known metadata preservation, plus ordinary edited round-trippable preview/typing and queued/navigation positives.
3. Apply the smallest secondary representation fence/source fallback using existing owners; keep canonical body exact for preview/title/save, truthful body-edit restriction or existing main-source open action, no implicit null/read-contract extension/normalization/new provider/store/parser/bridge.
4. Run targeted Research Quick Notes widget/service controls covering amended code, scoped static/security checks against immutable cb6e6e5 fix base, retain source/command/output correspondence. No213/native-host/suite/build replay unless an actual amended dependency justifies it.
5. Self-review caller/lease/source behavior, update concise report/qualification/ADR/task notes, commit explicit fix paths and leave status/AC In Progress for fresh scoped re-review. M1/M2/M3 remain deferred for final branch triage. CV1/CV2/CV3 limits stay exactly recorded in task-12-review-scope-resolution.md.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR194 I2 on canonical Database Notes and retained File Notes hosts: app-owned single worker service, permanent editor leases, mode/Check/Format/independent status/keyboard Problems, exact source previews and native one-transaction formatting through I1. Database mode stays durable; File mode is temporary and original BOM/newline/excerpt/hash/recovery owners remain intact. Rulings43/44/45 are amended in ADR194: secondary Source preview tradeoff, File-only CRLF mapping with truthful no-op, and lossy native Database source fence preserving title/mode/save/canonical copy-export fields.

Targeted covering213 passed; amended native-host34 passed; final actual-rendered-body/CSS16 passed. Exact commands, source manifests, RED/GREEN, raw output hashes, actual app-owned shutdown and scoped static comparisons are in .superpowers/sdd/2026-09-27-structured-document-editing/task-12-evidence and task-12-report.md. New production strict checks pass; expanded test-import mypy retains the immutable Screen type-comment parse barrier. FD warnings323/328 remain: immutable factory/BaseDB/allocator controls substantiate existing DB allocations but do not account for every descriptor. Physical OSIME/screen-reader/platform/clean-parent gates stay explicit. Docs/superpowers/reviews/2026-10-11-structured-note-editor-integration.md records qualifications and UX costs. Leave In Progress for fresh controller review; no full sweep/server/export-label expansion/push/merge.
Fix round1/5 implements original review Important I1 under ADR194/Ruling46. Quick Notes detects native round-trip loss at canonical load and retains exact owner source for Source preview, snapshot/download/title-only Save and navigation flush; native body editing/Clear are fenced with an explicit existing-status explanation. Supported drafts reset the fence and preview/save live native edits. No mode-presence contract, implicit null, new metadata/save owner, normalization/parser/bridge/File mapping change. ADR secondary qualification was appended before adoption.

New mounted RED reproduced mixed JSON/Markdown preview loss and normalized title-only save request (stopped before any write). Current Tests/UI/test_research_quick_notes.py + Tests/Research_Workspace/test_quick_notes.py:85 passed exit0, including actual Save/controller/local adapter/Notes scope/interop/temporary SQLite title-only and navigation-flush retention of exact body and known json mode, supported native typing/edited-preview/actual save, queued/native-delete/navigation controls, supported CRLF/lone-CR source preview. Scoped immutable cb6e6e5 Ruff/mypy/formatter comparisons add no debt; security exits0. Durable separate task-12-evidence/fix1 retains exact invocations/outputs/source hashes and preserves the original522-artifact index. Report is appended at task-12-report.md end; qualification doc and ADR updated. M1/M2/M3 are deferred Minor findings for final whole-branch triage, CV1/CV2/CV3 and physical OSIME/TTY/screen-reader/platform/FD/clean-parent limits stay explicit. Leave status In Progress and all AC unchecked for scoped independent re-review; no full suite/build/engine replay, push or merge.
Final fix1 fixture self-review closes only the new temporary SQLite fixture's own cached NotesInteropService connection using its existing public close_all_user_connections API. Amended SQLite control:1 passed,25 deselected,exit0; production bytes are unchanged from the85-test covering run. Final scoped Ruff/mypy/formatter comparators and security/changed-line format checks all exit0. No existing FD harness repair/suppression or deferred Minor finding change. Preserve In Progress and unchecked AC for fresh independent review.
Canonical Database Notes and retained File Notes expose persisted/temporary language, bounded local Check/Format, independent status and keyboard Problems. Explicit formatting uses the qualified native history bridge while existing draft/autosave/recovery/hash/save owners retain authority. Syntax errors remain saveable. Source previews are escaped; lossy Database and secondary Quick Notes bodies preserve exact canonical source for metadata/title/save/copy/export snapshots while body editing is fenced. File-only proposal mapping preserves original BOM/CRLF serialization and truthful no-op. Four tokens/source CSS/generated bundles and actual compact rendering are qualified.
ADR194 governs and records Rulings43/44/45/46; ADR150/031 govern UI/tokens/key conventions. Original ASTRA task review identified one Important Quick Notes preview defect; fix33c356d and fresh scoped standard review close it with no new breakage. Full original and scoped reports plus all3 Cannot Verify dispositions were read before closure.
Verification:213 affected tests at recorded pre-ancestor source;34 amended native-host tests;16 final composited-body/token/bundle controls;85 affected secondary tests after fix;1 later fixture-cleanup control. Counts describe separate source revisions, not additive unique tests. New production static/security checks pass; immutable legacy comparisons add no debt. Original522+fix105 artifacts and current source hashes verify with zero mismatches. Actual module-entry mounted probe confirms app-owned worker/lease/process shutdown.
Precise limits remain: M1 pending read-only wording, M2 no-op feedback presentation and M3 +323/+328 test-host FD warnings/partial allocation attribution are deferred for final whole-branch triage. Expanded test-import mypy retains unchanged Screen comment parse barrier; no all-test/full-module clean claim. Textual8.2.8 public input API has no composition hooks; physical OSIME/TTY/accessibility/platform/clean-installed-parent/resource profiles remain explicit I5 gates, and overall I1 stays open. No full sweep, server/UI-export-label expansion, shared environment mutation, push or merge. Durable qualifications: Docs/superpowers/reviews/2026-10-11-structured-note-editor-integration.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
I2 complete after independent whole-task review, scoped fix/re-review and all CV dispositions. Source, save ownership, native history, worker retirement and compact rendering are qualified for the recorded mounted/native profiles. Deferred minor/static/resource/physical release limits remain documented; I3/I4/I5 and final whole-branch review are still required.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
