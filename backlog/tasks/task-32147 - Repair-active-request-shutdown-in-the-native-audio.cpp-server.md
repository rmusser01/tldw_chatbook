---
id: TASK-32147
title: Repair active-request shutdown in the native audio.cpp server
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:34'
updated_date: '2026-09-09 07:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent native speech shutdown from destroying server or model state while accepted synthesis requests still use it. Qualify the repair on this macOS ARM host and retain an upstream-ready patch without silently changing Chatbook's approved runtime baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce active-request shutdown against identified upstream source and retain exact CPU and Metal failure evidence.
- [x] #2 An accepted request cannot outlive the server and model state it uses; shutdown rejects new work and does not hang forever on idle or partial client connections.
- [x] #3 Real CPU and Metal synthesis shutdown completes without the reproduced crash, and Chatbook successor or retry playback succeeds with owned resources joined.
- [x] #4 The minimal upstream patch, regression commands and runtime provenance are preserved separately from Chatbook's unchanged supported-runtime claims.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md and backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md govern the unchanged application runtime boundary.
Reason: Repair a native request-lifetime bug in a separately qualified upstream patch; do not upgrade or change Chatbook runtime provisioning or approved recipes.

1. Preserve tested binary UUIDs, crash-report extracts and pinned current upstream source to establish the active-request teardown race.
2. Add a small native HTTP shutdown regression that fails on active work, plus idle/partial-client controls; implement a minimal retained-request drain before handler/model destruction.
3. Build isolated patched CPU and Metal binaries and run serialized real synthesis shutdown, Chatbook recovery and physical playback validation.
4. Preserve a portable patch, exact source/build/runtime/model provenance and reproduction commands; record unsupported platform follow-ups. Run targeted checks and review the patch before marking done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved the reviewed audio.cpp request-drain patch and reproducible evidence in Docs/QA/tts-macos-burndown-2026-09-09/native/. The original CPU SIGSEGV and Metal SIGABRT were tied to the exact 0.5.1 source/binaries. Six native lifetime regressions failed before the repair; nine transport cases and eleven targeted CTests passed afterward. Real sampled-computation shutdown drained to exit zero in 7.398 s (CPU) and 3.526 s (Metal), followed by fresh-server Console playback; all four complete source/device transcripts matched exactly and owned resources joined. Two Metal observation misses remain recorded separately. No ADR was required: ADR-023/050 boundaries and approved provisioning remain unchanged. Shipping the repair in approved runtimes and sanitizer qualification remain separate recorded follow-ups. Native patch and QA receipts are the delivered changes; no binary or model is committed.
<!-- SECTION:NOTES:END -->
