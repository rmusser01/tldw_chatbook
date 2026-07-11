# PR 596 Review Fixes — Design

**Status:** approved by the user on 2026-07-11. This document records the approved minimal-repair direction and the additional verified findings discovered before implementation finished.

## Goal

Make PR 596 safe to merge by correcting every reproducible review finding without changing its product scope, adding dependencies, or replacing the approved parallel-parse/single-writer architecture.

## Decisions

### Ingest process lifecycle

Keep the spawn-context `multiprocessing.Pool`. Assign every pool a monotonically increasing generation token and bind that token to success and error callbacks. A daemon monitor watches the generation's original worker sentinels; an unexpected exit fails only `PARSING` jobs submitted to that generation, terminates that pool, and permits lazy rebuild on retry. Callbacks from older generations are ignored.

The command-line and `python -m` launchers must remain spawn-safe: they import `tldw_chatbook.app` only inside their `main()` call. Spawned workers therefore import the lightweight worker target without first importing the Textual application or optional ML stacks.

When shutdown begins, no new parsed payload may be claimed. A write already claimed by the single writer finishes; ready but unclaimed payloads remain abandoned under the existing in-memory queue shutdown contract.

Alternatives rejected:

- `ProcessPoolExecutor` reliably reports broken workers but its interpreter-exit behavior can wait for long-running parses, violating the approved quit contract.
- Per-job timeouts cannot distinguish a dead worker from legitimate long transcription or OCR work.
- Replacing the pool with a custom process manager is a broader rewrite than the missing containment requires.

### Unsaved UI state

All navigation is fail-closed while user edits remain only in memory.

- A raised `flush_pending_work()` vetoes the app-level screen switch after logging and notifying.
- Library in-screen actions flush and continue only when `_library_note_dirty` is false afterward, regardless of whether the save failure is labelled `conflict` or `error`.
- Settings preserves its active category, search text, and deep-copied draft values through the existing `save_state`/`restore_state` seam used by fresh-screen navigation.

A discard-confirmation modal was rejected for this repair because it expands UX scope and still requires a safe retained copy while the modal is shown.

### Search and parser fallbacks

Library Search/RAG treats input as plain text, not as raw SQLite FTS syntax. Each raw FTS seam receives an expression produced by splitting on whitespace, double-quoting each term, and doubling embedded quotes. Adjacent quoted terms preserve FTS5's prior implicit-AND behavior for ordinary multiword queries while punctuation can no longer become syntax.

In automatic document-parser mode, a deferred Docling import failure is an availability failure and falls back to the native parser. Explicit Docling selection continues to report the Docling failure instead of silently changing parsers.

### Test hygiene

The autouse test-environment fixture sets `TLDW_CONFIG_PATH` to its per-test temporary directory, which is the runtime-supported override and prevents tests from touching the developer's real config. The focus-contract test stops asserting CSS for the retired Notes mode strip.

## Error handling and invariants

- Pool generations are isolated: no old callback can fail work owned by a replacement pool.
- One unexpected worker exit fails all and only in-flight parses from that generation as retryable.
- Shutdown never starts a new persistence write.
- Dirty note buffers and Settings drafts survive navigation failures or fresh-screen reconstruction.
- Search failures remain distinguishable from legitimate empty results.
- Native parsing is used only as the automatic-mode fallback.

## Testing

Use red-green TDD for each behavior:

1. A real spawned worker exits via `os._exit`; the sentinel monitor reports the failed generation and the registry becomes retryable.
2. A stale callback from generation A cannot fail a generation B job.
3. A spawn probe verifies `tldw_chatbook.app`, Torch, and Transformers are absent before the worker target imports parsing code.
4. Shutdown prevents a second payload claim.
5. App-level flush exceptions do not call `switch_screen`.
6. Every Library transition that can discard editor state remains in place after a failed save.
7. Settings state round-trips drafts through a fresh instance.
8. Public Library search returns seeded matches for punctuation, quotes, and hyphens.
9. Auto document parsing falls back when Docling is discoverable but broken; explicit Docling does not.
10. The previously failing config-isolation and focus-contract tests pass without access outside the test temp directory.

Run the focused ingestion, navigation, Library, Settings, Search/RAG, document-processing, and UI contract suites, followed by compile and diff hygiene checks. Recheck PR review threads and GitHub Actions after pushing.

## ADR check

ADR required: no  
ADR path: N/A  
Reason: these are regression fixes that implement the already approved pool/single-writer design and existing fresh-screen state seams. They do not introduce a new storage model, service contract, dependency, security policy, or runtime boundary.
