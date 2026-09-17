---
id: TASK-32709
title: Resolve Console context windows from server then model defaults then 32000
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 01:07'
updated_date: '2026-09-17 03:28'
labels: []
dependencies: []
documentation:
  - Docs/Development/console-model-modal-investigation-2026-09-16.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Displayed context limits and request budgets must agree and prefer the configured server capacity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A valid selected-server context window takes precedence over model and API defaults
- [x] #2 Missing invalid unsupported or timed-out discovery falls back to model/API defaults then exactly 32000 tokens
- [x] #3 Discovery is asynchronous bounded cached and isolated by provider endpoint model and credential identity
- [x] #4 Console display request preparation and compaction use the same resolved limit with truthful provenance
- [x] #5 Nearby capacity-resolution defects are fixed and targeted regression and UI evidence are recorded
- [x] #6 Context metadata support preserves the existing startup module budget and generated diagnostic inventory matches the reviewed source
- [x] #7 Metadata requests enforce shared egress policy while allowing explicitly configured local origins, and public cache methods document their contracts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Amend ADR-052 for the approved fallback; implement one context resolver with provenance and bounded server metadata discovery; connect gateway request snapshots and UI estimates; preserve response reservation and explicit user budgets; test priority, unavailable/malformed servers, identity changes, cache and runtime/display agreement; inspect adjacent paths; run scoped tests/lint and update docs.
ADR required: yes
ADR path: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md
Reason: The approved estimated 32000-token fallback replaces the unknown-window automatic-budget block and introduces server-first capacity resolution.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented shared server-first context-window resolution under the amended ADR-052 (backlog/decisions/052-console-conversation-memory-and-compaction-policy.md). Server metadata wins over the existing model/API catalog; unknown windows fall back to exactly 32000. Provider/system fallback remains explicitly estimated and unverified. Gateway request snapshots, token estimates and compaction consume the same resolution, preserving output reservations and explicit conversation budgets.

Added bounded asynchronous metadata reads for llama.cpp serving props, selected Ollama running-model capacity and compatible model records. Reads have a one-second total deadline, 256 KiB response cap, positive bounded integer validation, no redirects, exact entry/endpoint/model/credential identity, bounded cross-loop single-flight/cache state and completion-based success/failure TTL. llama.cpp router lookup carries the selected model and autoload=false. Both model surfaces refresh through owned workers with generation and screen-stack fences, including A-B-A and dismissed-modal results. Removed stale Console token tables and aligned ordinary UI response reservation/safety margin with request capacity.

Nearby fixes: exact-model capacity before generic prefixes; custom endpoint execution aliases route through their API family; remote oversized integers and bodies fail safely; late results cannot update dismissed modals. Updated the user guide, investigation notes and incident-backed modal ownership lesson.

Verification: combined Console settings regression suite 277 passed (13 targeted modules, including real loopback HTTP transport); context/capacity/token/policy controls 139 passed; existing estimate/policy selection 13 passed; final changed-code context/subscription/mounted-modal run 39 passed. New files pass Ruff lint and format; modified Python lines have no Ruff findings, changed regions formatted and git diff --check passes. Broad legacy files retain existing unrelated lint debt. Three unrelated broader-test failures were reproduced against unchanged HEAD code/methods and documented in the investigation report. No full suite or real-account/provider validation was claimed. No new dependency, commit, merge or publication.

PR merge preparation: CI identified one eager context-window module at UI ready (1023 versus the unchanged 1022 limit). Plan: defer resolver imports to actual context actions, verify boot/import/preload guards and context regressions, and regenerate the diagnostic inventory after reviewing its sole delta: removal of the old token-counter lookup failure debug statement. ADR required: no new ADR; applies ADR-097 boot-budget policy and existing ADR-052. No sink or diagnostic argument is added.

Qodo findings 1/2: verify shared egress validation on the actual derived metadata URL, cover denied metadata addresses and authorized local origins with no-request assertions, and document cached/resolve parameters and outcomes. Startup trace confirms the pure fallback is needed during first paint; keep it with existing token-capacity helpers and defer only network cache construction/imports. ADR required: no new ADR; preserves ADR-052 and ADR-097 boundaries.

PR2703 review complete: shared async egress validation precedes all metadata HTTP streams; four metadata-destination regressions reproduced the bug and now pass with local-provider positive controls and the real loopback client. Public cache contracts documented. Pure fallback moved to existing token helpers and gateway metadata cache is created lazily under a lock; first-paint absence guard and unchanged boot budgets pass (1022/1022 ready, 669/686 import). Diagnostic statement review confirmed only one removed token-counter debug call and zero new sinks; regenerated inventory verifies. Final combined startup/context/token/Home run: 86 passed, one loopback deselected; separate loopback: 1 passed. Unchanged-dev preload module census remains 536 against 500 and is documented separately. ADR-052/097 apply; no new ADR. Ruff delta and diff checks clean.
<!-- SECTION:NOTES:END -->
