---
id: TASK-10.6.3
title: 'Gate 1.5.3: Console native transcript and composer surface'
status: Done
assignee: []
created_date: '2026-05-07 03:37'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies:
  - TASK-10.6.2
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the embedded ChatWindowEnhanced main surface with Console-owned transcript and composer widgets that reuse existing ChatSession and tab services where safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console transcript/event stream is a native region and no longer contains the full legacy ChatWindowEnhanced container.
- [x] #2 Console composer is a native action row with send stop attach and save-Chatbook affordances wired through existing chat handlers or documented compatibility adapters.
- [x] #3 Basic chat tabs session state handoff draft text and streaming fallback regressions continue to pass.
<!-- AC:END -->

## Implementation Plan

1. Convert the existing Gate 1.5.3 xfail coverage into deterministic failing regressions for a Console-native transcript/session surface and composer.
2. Add Console-owned session and composer widgets that reuse the existing tab/session chat handlers instead of duplicating chat behavior.
3. Update `ChatScreen` to mount the native surface, query it through compatibility seams, and keep state/handoff restoration working without a mounted full `ChatWindowEnhanced`.
4. Run focused Console/chat regressions, fix only migration-caused failures, and record QA evidence in the task notes.

## Implementation Notes

Moved the Console transcript from the full embedded `ChatWindowEnhanced` container to a cached `ConsoleSessionSurface` that hosts `ChatTabContainer` and task cards directly. Added `ConsoleComposerBar` with native send, stop, attach, and Save Chatbook compatibility affordances; send/stop/attach route through active chat-session handlers and Save Chatbook reports the current Artifacts/Chatbooks ownership seam.

Updated `ChatScreen` state/handoff compatibility seams so native Console tabs are discoverable through `_get_tab_container()`, active-session changes remain visible in composer status, and save/restore paths no longer require a mounted legacy chat window. Updated mounted regressions and the Gate 1 core-loop contract to assert native Console selectors and absence of `#chat-window`.

Verification: baseline red was observed by removing the Gate 1.5.3 xfails; final focused runs passed with `83 passed` for Console/chat state coverage, `70 passed` for Gate 1.5 verification, and `git diff --check` passed.
