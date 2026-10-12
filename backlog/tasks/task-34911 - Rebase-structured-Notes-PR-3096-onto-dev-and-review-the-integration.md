---
id: TASK-34911
title: Rebase structured Notes PR 3096 onto dev and review the integration
status: In Progress
created_date: 2026-10-12 19:44
assignee:
- '@codex'
labels:
- notes
- structured-editing
- review
priority: high
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3096
updated_date: 2026-10-12 19:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the already reviewed local structured-document checks and reversible formatting with current dev so the existing draft PR has no merge conflicts. Preserve current upstream owners and source/metadata safety, then obtain a fresh independent review before updating the PR. Physical IME/spoken output qualification and separately deferred durable Notes sync remain outside this integration claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The PR branch is rebased onto the recorded current dev base, resolves its conflicts and retains the intended structured-Notes changes without deleting unrelated upstream behavior or documentation.
- [ ] #2 Targeted verification of affected real storage, source and language preservation, recovery, native editor and import/export paths qualifies the integrated source; conflict resolutions do not weaken authority, exact-source or undo safeguards.
- [ ] #3 Schema and migration allocations are compatible with current dev and scoped lint, formatting, type and security checks introduce no unexplained findings; real upgrade/bootstrap controls cover migration conflicts.
- [ ] #4 An independent requesting-code-review review of the rebased PR range is recorded; Critical and Important findings are fixed or explicitly reconciled with technical evidence before publication.
- [ ] #5 The existing PR is updated with the reviewed rebased head using an exact expected-head force-with-lease, retains dev as its base and documents actual checks and remaining qualification limits without enabling merge or claiming unverified physical coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for rebase/direct existing-owner integration; ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md. Reason: retain approved existing boundaries and adapt only verified conflict consequences; record a new ADR before adopting any genuinely new architecture.
1. Read this task, current AGENTS/Architecture references, original design, accepted final review, publication provenance and current dev conflict owners.
2. Preserve exact published head/backup; rebase onto recorded dev, replay only structured-feature changes, resolve conflicts by combining actual upstream/feature behavior, preserve original design/source safety and unrelated dev documents. Correct migration/identifier collisions with actual current allocations and upgrade controls.
3. Verify actual rebased module/config/dependency binding in private owned environments before targeted tests; run affected source/storage/recovery/native/worker/import/export and relevant scoped static/security checks with no broad sweep or user data.
4. Save exact resolution decisions, paths/commands/output/source hashes and limitations in a report; obtain Root-dispatched independent review of dev-to-rebased-head, address confirmed findings and verify amended controls.
5. Commit explicit integration/task paths, leave original physical qualification tasks open, and let Root publish the independently reviewed branch using exact expected-head force-with-lease; no merge or auto-merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased only the structured programme onto recorded dev `66a940d1a79673061ed5d3d0f16cb2430869ddb6`, retaining all current-dev documents and existing migration bodies. Dropped the old publication cleanup that would remove unrelated upstream documents; reconciled only feature server references (tasks 13586–13597, ADRs 070–074). Current app lifecycle, Notes canvas gates, import transaction/citation owner, sync maintenance, and atomic replace-confirmed export remain authoritative.

Canonical note language now follows current schema 80 as additive 80→81. ADR-194 and ADR-126 govern its exact canonical and shared Subscriptions recovery catalogs; Subscriptions stays schema 2, runtime.sync_state has no DDL change, and the entire device-private Notes database/WAL remains excluded. New real SQLite controls cover old/fresh catalogs, restricted upgrade, preserved source and unknown-schema refusal. Exact source/portable exports preserve newline bytes inside the current atomic owner; recovery participates in the current producer drain fence.

Focused storage, draft coalescing/original base, wire omission/null, import/archive, native history/layout, export, recovery/maintenance and real app lifecycle checks are recorded under `/private/tmp/structured-notes-pr-rebase-2026-10-12/c-evidence`; full report `c-rebase-report.md` records exact source and check manifests. Scoped lint/security/types are compared with current-dev and prior-feature debt, not claimed whole-file clean. The ProductionApp nested config-fixture collection refusal, aggregate FD/coroutine warnings, and LibraryScreen baseline mypy comment-parser limit remain qualified in that report. No full sweep, new transport/storage policy, push, merge or auto-merge.

Task intentionally remains In Progress with acceptance criteria unchecked for Root's independent review and exact expected-head publication. Physical OS IME/spoken output remain unqualified; durable server-readable Notes sync remains deferred to TASK-34818. Existing lessons-testing-evidence entries already cover frozen shared catalogs and immutable SQLite/WAL fixture staging; this integration preserved those rules.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
