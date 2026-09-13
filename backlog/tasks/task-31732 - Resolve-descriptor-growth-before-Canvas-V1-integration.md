---
id: TASK-31732
title: Resolve descriptor growth before Canvas V1 integration
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:46'
updated_date: '2026-09-05 20:12'
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
- [x] #1 The original growth signal is reproduced or bounded with isolated, source-free evidence identifying resource categories and responsible test or runtime lifetimes.
- [x] #2 Any confirmed in-scope resource leak is corrected with a failing regression and targeted passing controls without hiding the sentinel or weakening cleanup guarantees.
- [x] #3 The final affected run and independent review document the outcome and retained limitations before the Canvas pull request proceeds.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored deterministic operation-local DB ownership in ChatbookCreator and ChatbookImporter with try/finally around unchanged private processing helpers (d94c40755). Existing ADR-121 applies; no connection policy, GC or sentinel change. Four real SQLite regressions preserve a same-file observer and verify exact owned handles close across repeated success and post-construction service failure. Dedicated malformed/cancellation lifetime tests were omitted because the unconditional finally covers them structurally; existing thinking round trips remain controls. Root valid RED4failed1warning1.70s; GREEN plus thinking24passed1warning2.25s, plugin12→12. Original970selection passed1warning172.49s, plugin growth223→140 (83 fewer), no sentinelwarning; residual other-module retention is not claimed fixed. Independent static review6226f8a18..d94c40755 PASSspec/quality. Follow-up annotation/privacy run28passed1warning3.13s; new test Ruff/format pass, existing production lint debt retained without new findings. Evidence and limitations in Docs/Canvas/V1_VERIFICATION.md; ownership lesson in lessons-testing-evidence.md. TASK31741 separately owns preflight and rebase. No full suite or integration approval inferred from tests.
<!-- SECTION:NOTES:END -->
