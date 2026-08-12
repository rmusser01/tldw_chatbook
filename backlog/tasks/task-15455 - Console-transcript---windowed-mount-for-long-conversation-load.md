---
id: TASK-15455
title: Console transcript: windowed mount for long-conversation load
status: Done
assignee:
  - codex
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: session resume loads the entire persisted tree (`depth_cap=10_000`) and the first `refresh_messages` mounts every row via individual awaited `mount()` calls — no batching, no windowing (`Widgets/Console/console_transcript.py:2283-2301`; old rows likewise removed one awaited `remove()` at a time). Height-watermark pruning runs only after first layout, so a long conversation pays full mount plus full-history Markdown parse (one Textual Markdown widget per assistant row, one child widget per markdown block) before anything is trimmed — and up to the 12k-20k-line watermarks stay mounted permanently, which also inflates every reconcile pass (task-15453) and layout.

Fix direction: mount a tail-first window (bottom N lines) and hydrate scrollback lazily on scroll; batch mounts. This is structural — stability first: anchor()/tail-follow semantics (`:1295/:1344/:1399-1408`), selection, pruning, and branch navigation must be pinned by tests before the windowing lands. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loading a 500+-message conversation mounts only the visible tail window initially (evidence + session-switch latency before/after)
- [x] #2 Scrollback hydrates on demand without breaking anchor/tail-follow, selection, or branch navigation (tests)
- [x] #3 Prune watermarks still bound total mounted height
<!-- AC:END -->

## Implementation Plan

1. Characterize the existing transcript refresh, pruning, selection, branch-navigation, and viewport-follow contracts with focused tests and latency instrumentation.
2. Add a bounded tail-first projection and batched transcript reconciliation in `UI/Console_Modules/`, preserving the transcript widget's public API and current region ownership.
3. Hydrate earlier messages when the viewport reaches the scrollback boundary while restoring the reading anchor and retaining selection and branch behavior.
4. Verify the 500+ message resume path, hydration behavior, and mounted-height watermarks with focused automated tests and before/after timing evidence.
5. Run affected Console suites, static analysis, and a self-review; document implementation notes and complete the acceptance criteria only when the evidence supports them.

ADR required: no

ADR path: N/A

Reason: This is an internal performance refinement of the existing Console transcript-region and message-store contracts; it does not change persistence, ownership, security, or a cross-module service interface.

## Implementation Notes

- Added a viewport-scaled, turn-aligned tail projection before row signatures or Markdown widgets are built. The complete history remains in the transcript model for export, persistence, selection, and branch operations.
- Added coalesced scrollback hydration for upward scrolling, wheel-at-boundary, and Page Up. Prepended rows compensate the scroll offset by their measured virtual height, preserving the reader's visible message and detached follow state.
- Reconciliation now removes stale direct children and mounts contiguous missing runs through Textual's batch DOM APIs. The existing reorder contract remains intact.
- Kept the implementation in the existing `ConsoleTranscript` leaf rather than adding a new `UI/Console_Modules/` controller: the window is internal view state of the actual scroll container, and moving it outward would split the established pruning, selection, row-signature, and reconciliation invariants. No screen-size ratchet growth was introduced.
- Identical 500-message Markdown session-swap probe at 100x30: merged baseline `19.8457s`, 500 mounted messages; windowed implementation `0.2411s`, 22 mounted messages and 478 lazily retained messages. This is an 82x speedup and 98.8% lower elapsed time on the test host.
- Verification: 4 focused windowing tests, 17 combined windowing/pruning/tail-follow contracts, and 127 broader native-transcript/Markdown/selection/diff/jump/region test bodies passed; two mutation checks killed removal of initial windowing and hydration; Ruff, `py_compile`, and `git diff --check` passed. The repository pytest wrapper cannot bootstrap asyncio on this Windows host because its process-wide network guard intercepts Python's loopback `socketpair`; the same test bodies were therefore executed directly in isolated Textual harnesses.
- Added the reorder/window lesson to `backlog/docs/lessons-testing-evidence.md` after the broader regression pass caught a boundary-preservation defect that focused tests missed.
- Modified: `tldw_chatbook/Widgets/Console/console_transcript.py`, `Tests/UI/test_console_transcript_windowing.py`, this task, and the testing-evidence lessons document.
- ADR required: no. The existing Console transcript-region and leaf-widget ownership remain unchanged.
