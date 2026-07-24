---
id: TASK-174
title: Address PR 596 verified review findings
status: Done
assignee:
  - '@codex'
created_date: '2026-07-11 22:08'
updated_date: '2026-07-12 00:57'
labels:
  - pr-review
  - library
  - ingestion
  - navigation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct all reproducible correctness data-loss concurrency runtime search and test-isolation issues found while reviewing PR 596 so the dev-to-main promotion is safe to merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Abrupt parse-worker death fails only jobs owned by that pool generation and allows retry.
- [x] #2 Spawned parse workers do not preload the application or heavy ML stack before parsing.
- [x] #3 Ingest shutdown finishes only an already-claimed write and does not drain ready payloads.
- [x] #4 Library note navigation never discards edits after a failed save.
- [x] #5 App navigation vetoes when pending-work flush raises.
- [x] #6 Settings drafts survive fresh-screen navigation.
- [x] #7 Plain Library search queries with punctuation return matching FTS results.
- [x] #8 Auto document parsing falls back to the native parser when deferred Docling import is broken.
- [x] #9 Tests never write the user CLI config.
- [x] #10 Retired Notes mode-chip assertions are removed.
- [x] #11 All review threads are resolved and focused verification passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes

ADR path: `backlog/decisions/013-media-search-plain-text-fts-boundary.md`

Reason: ADR-013 records the narrow optional media-search contract that preserves raw `LIKE` text while supplying a separate safe FTS `MATCH` expression. The other changes implement existing contracts.

Spec: `Docs/superpowers/specs/2026-07-11-pr-596-review-fixes-design.md`

Detailed plan: `Docs/superpowers/plans/2026-07-11-pr-596-review-fixes.md`

1. Add pool generations, sentinel death monitoring, stale-callback isolation, and shutdown claim guards with red-green tests.
2. Add spawn-lightweight supported CLI launchers and a real spawn preload regression.
3. Make app and Library navigation fail closed and round-trip Settings drafts through fresh screens.
4. Quote plain-text input only at raw FTS boundaries and restore automatic native fallback for broken Docling imports.
5. Isolate test config writes and remove the retired Notes CSS contract.
6. Run focused and broad verification, then complete task and PR hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Contained parse-pool failures with generation ownership, worker-sentinel monitoring, stale-callback guards, targeted teardown, and shutdown claim refusal; added spawn-lightweight CLI entry points so child processes do not preload the application or ML stack.
- Made app and Library navigation fail closed around pending note saves, preserved Settings drafts across fresh screens, separated raw media search text from safe FTS expressions per ADR-013, and restored request-scoped native parsing fallback when automatic Docling loading fails.
- Isolated config writes before collection imports, retired the obsolete Notes mode-chip contract, and hardened two fast export tests to wait for terminal evidence instead of the initial idle state.
- Verified 341 focused affected tests and an independent 149-test cross-subsystem gate, plus compileall and diff checks. A 1,694-case broad run passed 1,693 cases and exposed only the export-test timing race fixed above; both affected cases then passed. All three PR review threads were answered and resolved, and an independent holistic review reported no findings.
- Modified the ingest runner and tests, supported launchers, navigation/Settings state, Library/media search, document parsing fallback, test isolation/contracts, README/packaging metadata, ADR-013, and the PR review plan/spec.
<!-- SECTION:NOTES:END -->
