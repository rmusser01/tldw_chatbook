---
id: TASK-31732
title: Resolve descriptor growth before Canvas V1 integration
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 19:46'
updated_date: '2026-09-05 19:59'
labels:
  - canvas
  - testing
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the descriptor-growth signal reported by the Canvas acceptance runs so integration is based on understood resource ownership rather than an unexplained warning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The original growth signal is reproduced or bounded with isolated, source-free evidence identifying resource categories and responsible test or runtime lifetimes.
- [ ] #2 Any confirmed in-scope resource leak is corrected with a failing regression and targeted passing controls without hiding the sentinel or weakening cleanup guarantees.
- [ ] #3 The final affected run and independent review document the outcome and retained limitations before the Canvas pull request proceeds.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md (renumber during integration). Reason: restore deterministic ownership of existing operation-local SQLite handles, no new storage or security policy.
1. Reproduce the ten-module signal with a source-free per-test diagnostic. Completed: 970 passed; regular descriptors +224; GC=1 thinking-only control still +83.
2. Add real SQLite regressions for ChatbookCreator._collect_conversations and ChatbookImporter._import_conversations: repeated success, post-construction service setup failure, malformed import and cancellation where applicable. Capture exact operation handles, verify they are closed at return, and preserve a separately owned same-file observer connection. Root must observe RED before production edits.
3. Give each operation-local CharactersRAGDB deterministic try/finally close_connection ownership across all work after construction. Do not change database-wide quiescence, connection policy, GC fixtures or sentinel limits.
4. Root runs GREEN and the original ten-module selection; compare aggregate lifetime results and narrow any remaining confirmed leak. Independent static review checks correction and retained limitations before closing ACs.
5. Record evidence and lessons, then use the authorized integration sequence in the Canvas plan: unique ADR number, recoverable ref, latest-dev rebase, targeted/preflight checks, PR, Qodo issues and checks, merge without bypass, then V2 design.
<!-- SECTION:PLAN:END -->
