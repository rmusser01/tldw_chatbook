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
updated_date: 2026-10-12 21:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the already reviewed local structured-document checks and reversible formatting with current dev so the existing draft PR has no merge conflicts. Preserve current upstream owners and source/metadata safety, then obtain a fresh independent review before updating the PR. Physical IME/spoken output qualification and separately deferred durable Notes sync remain outside this integration claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The PR branch is rebased onto the recorded current dev base, resolves its conflicts and retains the intended structured-Notes changes without deleting unrelated upstream behavior or documentation.
- [x] #2 Targeted verification of affected real storage, source and language preservation, recovery, native editor and import/export paths qualifies the integrated source; conflict resolutions do not weaken authority, exact-source or undo safeguards.
- [x] #3 Schema and migration allocations are compatible with current dev and scoped lint, formatting, type and security checks introduce no unexplained findings; real upgrade/bootstrap controls cover migration conflicts.
- [x] #4 An independent requesting-code-review review of the rebased PR range is recorded; Critical and Important findings are fixed or explicitly reconciled with technical evidence before publication.
- [x] #5 The existing PR is updated with the reviewed rebased head using an exact expected-head force-with-lease, retains dev as its base and documents actual checks and remaining qualification limits without enabling merge or claiming unverified physical coverage.
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
Rebased the structured programme onto dev 38654061, preserving unrelated upstream documents, prior migration bodies and the current lifecycle, Notes canvas/sync gates, import transaction/citation and atomic export owners. Removed the old publication cleanup that would delete upstream documents; retained feature server references to TASK-13586-13597 and ADR-070 through ADR-074.

Note language follows current schema 80 as additive 80-to-81. ADR-194 / ADR-126 govern strict canonical/shared recovery catalogs; Subscriptions remains schema 2, runtime.sync_state has no DDL change, and device-private Notes DB/WAL remains excluded. Exact-source exports use the existing atomic owner, and recovery participates in its existing producer-drain fence. Targeted storage, source/language, recovery, migration/catalog, import/archive, native history, Format, CSS and real app-lifecycle evidence is recorded in c-rebase-report.md and c-evidence.

Independent full rebased-PR review found 0 Critical / 0 Important / 1 Minor: misleading guide wording about autosave. The one-paragraph correction was independently confirmed. Subsequent dev advancement replayed 45 feature commits conflict-free as reviewed source head 9fc204f1c04b7b13e739dd2d302fd053f7e1fb5c. Root and scoped review independently verified byte-identical feature diff and unchanged runtime tree; only two upstream test files changed. Their initial five-case run returned 3 passes / 2 editor-arm timeouts; the exact two failures then passed serially on unchanged source, runner and original 3s / 30s deadlines. Matching 8,456-entry manifests and log hashes were verified. Both attempts remain preserved; cause is not established.

Published the reviewed source head to existing draft PR https://github.com/rmusser01/tldw_chatbook/pull/3096 with exact force-with-lease against original head 1d8ff566cfa8695a86012e272d7e02b984450cfd. The updated description records actual checks and limits. GitHub confirmed dev base, exact head, MERGEABLE, draft and no auto-merge; protection/CI gates remain separate. Subsequent task-only bookkeeping preserves the reviewed source and tests.

All integration acceptance criteria are accounted for. Status remains In Progress because repository static-analysis/formatting DoD is not wholly satisfied by the disclosed inherited feature/current-dev debt and analysis reach limits. No clean whole-file/type claim or waiver is made; broad debt repair was outside this rebase request. ProductionApp nested-fixture refusal, LibraryScreen mypy parser limit and FD/coroutine warnings remain qualified. Physical IME/spoken-output tasks remain open; durable Notes sync stays deferred to TASK-34818. Full report, static clarification, independent reviews and unchanged failed-case control are retained under /private/tmp/structured-notes-pr-rebase-2026-10-12. Existing testing-evidence lessons apply; no new architecture was introduced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed rebase published to draft PR3096 on dev38654061; no unresolved Critical/Important source finding. Guide correction and exact-source dev advancement confirmed. Incoming 3-pass/2-fail attempt and unchanged 2-pass retry retained; cause unproved. Merge, physical qualification and clean whole-file static DoD remain open.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
