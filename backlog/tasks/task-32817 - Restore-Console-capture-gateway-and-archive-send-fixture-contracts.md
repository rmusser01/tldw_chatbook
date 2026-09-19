---
id: TASK-32817
title: Restore Console capture-gateway and archive send fixture contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 19:12'
updated_date: '2026-09-18 20:31'
labels:
  - test-health
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32815 targeted archive verification found both restore/resume/send variants fail while mounting because CapturingGateway lacks cached_context_window. Both exact cases reproduce on saved source 2cc702b85c, before the footer guard. Preserve the real resume/send assertions while restoring the provider fixture contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Console capture gateway fixture supports the current context-window metadata contract with truthful deterministic test data.
- [x] #2 Both archive restore/resume/send variants pass with their original history, unrelated draft and persistence assertions intact.
- [x] #3 Targeted fixture consumers are checked and the saved-source failure evidence is linked.
- [x] #4 Archive fixture setup waits for the mounted composer to own the active session and for its unrelated draft to reach the store before starting recovery.
- [x] #5 Direct capture-gateway consumers use persisted provider defaults when creating later sessions and verify full rail status at the correct untruncated state boundary while retaining visible-row checks.
- [x] #6 Restored Console test screens load the same application stylesheets as initial screens, and the original lifecycle test can click both session tabs after recreation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconfirm the original two archive failures and inspect the gateway metadata contract. 2. Add matching cached/async resolution methods using the production pure fallback. 3. Replace fixed archive setup timing with bounded composer-ownership/stored-draft waits; preserve every original archive assertion. 4. Repair exposed direct-consumer assumptions: persist provider defaults for later sessions; verify visible active metadata and exact matching-row full status. 5. Restore APP_STYLESHEETS parity to RestoredConsoleHarness after the final-subset pointer failure exposed its unstyled layout; retain the real tab click, check its lifecycle consumer and the one adjacent real-service restore consumer. 6. Run targeted consumers serially, retain all failures, verify unchanged-case ASTs and final lint/format, obtain code review, update evidence/ledger and save draft PR2707. ADR required: no. ADR path: N/A. Reason: Fixture metadata, asynchronous setup and stylesheet contracts only; no production/provider/storage/UI behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored cached/async context-window fixture methods using the production pure catalog/fallback resolver. Archive setup now waits for real composer ownership and stored draft state. Later-session consumers persist provider defaults, rail checks preserve visible status plus matching-row full state, and restored harnesses load the initial application stylesheets. All 32 archive-module assertions and original lifecycle pointer actions remain. Fourteen distinct targeted cases pass serially in private profiles, covering both archive variants, all direct capture-gateway consumers and both restored-harness users; original/intermediate failures remain recorded. Source hashes/unchanged consumer ASTs verified, zero introduced Ruff diagnostics (22 existing), seven changed ranges formatted, independent review has no actionable findings. Evidence: Docs/superpowers/qa/2026-09-18-console-capture-gateway/README.md. Modified only Tests/UI/test_console_native_chat_flow.py and Tests/UI/test_console_conversation_archive_flow.py plus task/evidence/ledger documentation. No full suite or new native qualification. ADR required: no; ADR path: N/A, fixture contracts only. Broader component review and draft PR2707 merge gate remain open.
<!-- SECTION:NOTES:END -->
