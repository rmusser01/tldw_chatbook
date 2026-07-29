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
- [ ] #1 Worker-event regressions retain the live non-streaming delegation and failure coverage without importing retired message classes, recreating worker-owned citation/streaming behavior, or duplicating existing native Console and streaming-rejection contracts.
- [ ] #2 The current dev chat-shell regression retains live session/persona label coverage without importing or replacing the retired `TabState` model.
- [ ] #3 The audio stream-error regression invokes one synchronous recording loop without VAD or thread races and proves the exact pre-error callback sequence, stream closure, and stopped state.
- [ ] #4 The PyAudio flow regression invokes one synchronous recording loop without VAD or thread races and proves exactly three callbacks, stream closure, and stopped state.
- [ ] #5 The SoundDevice flow regression disables VAD for its synthetic callback, waits boundedly for the mocked stream callback, and cleans up its recording thread even on failure before proving audio was queued.
- [ ] #6 The Llama.cpp, DeepSeek, and local-LLM request tests patch the live runtime-config snapshot seam without restoring or emulating deleted mutable module-level settings.
- [ ] #7 Every real-seam Notes fixture creates its temporary trusted base directory as owner-only before constructing `NotesInteropService` and closes per-user Notes DB connections during teardown, without weakening production path verification.
- [ ] #8 Every changed diagnostic owner is reviewed against ADR-029, no unsafe payload logging is admitted, persistent sink topology remains unchanged, and the checked inventory matches production.
- [ ] #9 The large-batch RAG indexing regression retains its 1,000-item persistence and retrieval coverage without host-dependent wall-clock assertions.
- [ ] #10 The retargeted Evals profile regression creates its temporary trusted profile directory as owner-only before selecting the config path and closes its test-owned database connection, without weakening production path verification.
- [ ] #11 Local skill names cannot shadow the registered `search_run_log`, `run_log_stats`, or `run_log_slice` runtime tools, and the existing shadow-set drift guard passes.
- [ ] #12 The `quick_ingest()` fallback-path regression expects the canonical profile-aware `tldw_chatbook_media_v2.db` filename while retaining configured-path and traversal-rejection coverage.
- [ ] #13 The reusable RAG citation benchmark host context creates an owner-only isolated config profile before selecting `TLDW_CONFIG_PATH`, runs without reading or mutating host config/data/secrets, and retains its existing output-privacy assertions.
- [ ] #14 The production Console Stop regression drives a rendered visible button before clicking it, and proves that the user action cancels the provider stream and preserves the stopped partial response without relying on app teardown.
- [ ] #15 The affected modules and repository-wide suite collect and run without these baseline failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-29-dev-gate-test-contract-repair.md

ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: Reconciles tests with accepted production contracts and applies ADR-029's existing metadata-only inventory review requirement without making a new architectural decision.

1. Remove the retired StreamDone import, duplicate streaming assertion, and fully obsolete worker-local citation capture file while preserving unique retained-adapter and non-streaming failure coverage.
2. Preserve the current dev chat-shell repair rather than carrying a superseded branch edit.
3. Make both PyAudio loop tests synchronous, VAD-independent, and exact; keep the SoundDevice fixture VAD-independent with explicit cleanup.
4. Patch all stale provider request tests through the live runtime-config snapshot seam instead of deleted module globals.
5. Create temporary trusted Notes roots in each stale real-seam fixture.
6. Remove host-dependent timing assertions from the large-batch indexing test while retaining its functional coverage.
7. Create the retargeted Evals profile fixture directory before selecting its config path.
8. Synchronize the fixed Library skill collision set with registered run-log runtime names.
9. Align the Local Ingestion fallback assertion with the canonical media database filename.
10. Create the isolated RAG benchmark's trusted config profile directory before application imports.
11. Let the Textual pilot render the visible Console Stop state before issuing the pointer click, retaining the real user-action cancellation assertion.
12. Review all changed production diagnostics and sink topology against ADR-029 before regenerating the checked inventory.
13. Run affected, static, inventory, and repository-wide gates; review and close only if the full Definition of Done is satisfied.
<!-- SECTION:PLAN:END -->
