---
id: TASK-32709
title: Resolve Console context windows from server then model defaults then 32000
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 01:07'
updated_date: '2026-09-17 01:36'
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
<!-- SECTION:NOTES:END -->
