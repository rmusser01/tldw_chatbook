---
id: TASK-16212
title: Admit OpenAI realtime loopback test server
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:20'
updated_date: '2026-08-14 00:29'
labels:
  - test-health
  - realtime
  - network
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow the OpenAI Realtime session test module to use its own in-process loopback WebSocket server under the repository network guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The module's localhost WebSocket connections are explicitly admitted by the network guard.
- [x] #2 External network access remains outside the test contract.
- [x] #3 The complete realtime module, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this declares an existing in-process integration-test requirement without changing product networking.

1. Preserve the 34 blocked loopback connections as RED evidence.
2. Apply the repository's existing module-level `allow_network` convention for fixture servers.
3. Run the full module with real loopback, then containing-chunk/static/diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the repository's module-level `allow_network` marker to the OpenAI Realtime session tests. Every admitted connection targets the module's own in-process `127.0.0.1` WebSocket server; no external endpoint was added. Removing the marker reproduced the network-guard failures. The full module passed 36 tests and the containing chunk passed 1,242 tests.
<!-- SECTION:NOTES:END -->
