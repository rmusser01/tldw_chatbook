---
id: TASK-1333
title: Reconcile stale dev-gate chat and audio tests
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-29 08:11'
updated_date: '2026-07-29 15:04'
labels:
  - testing
  - baseline
  - cleanup
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the mandatory dev test gate by aligning stale or nondeterministic tests with the current retired-Chat and audio-recording contracts, preserving the current dev shell-test repair, and safely refreshing the reviewed diagnostic inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The worker-events regression test retains its non-streaming failure coverage without importing retired message classes or duplicating the existing streaming-rejection contract.
- [ ] #2 The current dev chat-shell regression retains live session/persona label coverage without importing or replacing the retired `TabState` model.
- [ ] #3 The audio stream-error regression invokes one synchronous recording loop without VAD or thread races and proves the exact pre-error callback sequence, stream closure, and stopped state.
- [ ] #4 The PyAudio flow regression invokes one synchronous recording loop without VAD or thread races and proves exactly three callbacks, stream closure, and stopped state.
- [ ] #5 The SoundDevice flow regression disables VAD for its synthetic callback, waits boundedly for the mocked stream callback, and cleans up its recording thread even on failure before proving audio was queued.
- [ ] #6 The Llama.cpp and DeepSeek request tests patch the live runtime-config snapshot seam without restoring or emulating deleted mutable module-level settings.
- [ ] #7 Every real-seam Notes fixture creates its temporary trusted base directory as owner-only before constructing `NotesInteropService` and closes per-user Notes DB connections during teardown, without weakening production path verification.
- [ ] #8 Every changed diagnostic owner is reviewed against ADR-029, no unsafe payload logging is admitted, persistent sink topology remains unchanged, and the checked inventory matches production.
- [ ] #9 The affected modules and repository-wide suite collect and run without these baseline failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-29-dev-gate-test-contract-repair.md

ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: Reconciles tests with accepted production contracts and applies ADR-029's existing metadata-only inventory review requirement without making a new architectural decision.

1. Remove the retired StreamDone import and duplicate streaming assertion while preserving unique non-streaming failure coverage.
2. Preserve the current dev chat-shell repair rather than carrying a superseded branch edit.
3. Make both PyAudio loop tests synchronous, VAD-independent, and exact; keep the SoundDevice fixture VAD-independent with explicit cleanup.
4. Patch provider request tests through the live runtime-config snapshot seam instead of deleted module globals.
5. Create temporary trusted Notes roots in each stale real-seam fixture.
6. Review all changed production diagnostics and sink topology against ADR-029 before regenerating the checked inventory.
7. Run affected, static, inventory, and repository-wide gates; review and close only if the full Definition of Done is satisfied.
<!-- SECTION:PLAN:END -->
