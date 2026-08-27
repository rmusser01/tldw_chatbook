---
id: TASK-22508
title: Prevent multi-EPUB imports from exhausting memory
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 15:33'
updated_date: '2026-08-26 16:57'
labels:
  - ingest
  - ebook
  - reliability
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bound EPUB parse concurrency and reject archives whose expanded structure can exhaust process memory, so one batch cannot freeze or crash the application.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EPUB jobs run in a one-process parse-pool generation; ordinary light jobs retain configured parallelism in separate generations.
- [x] #2 EPUB archives that exceed documented member-count, per-member, markup-expanded-size, total-expanded-size, or compression-ratio limits fail as one isolated ingest job before ebooklib extraction.
- [x] #3 Ordinary EPUB imports and EPUB chunking continue to work.
- [x] #4 A three-EPUB spawned-pool reproduction stays well below the prior multi-worker peak, and targeted coordinator, EPUB parser, and real spawned-worker tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/091-bounded-epub-archive-admission.md
Reason: EPUB ZIP expansion and process ownership are untrusted resource boundaries; fixed archive limits plus isolated pool generations reject plausible alternatives.

1. Preserve the red archive-boundary tests and replace the initial admission-only regression with one proving EPUB and ordinary jobs use separate pool generations.
2. Extend the central-directory guard with the fixed markup-expanded-size boundary, preserving ordinary ebooklib format errors.
3. Create EPUB pool generations with one OS worker, keep queued EPUBs on that generation, and retire it after the batch so high-water RSS is released; retain configured parallelism for ordinary generations.
4. Keep EPUB admission exclusive from ordinary parse generations and resume queued work after off-loop pool teardown.
5. Run focused coordinator, ebook parser/chunking, folder queue, pool lifecycle, and real spawned-worker verification; record comparative RSS evidence, self-review, and update task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented isolated one-process EPUB pool generations while preserving configured ordinary-import parallelism, plus gated off-loop pool retirement and durable retryable failure handling when teardown cannot be proven complete. Added central ZIP/manifest admission limits and a checked ebooklib reader boundary that validates the source path and fails closed before extraction for unsafe paths, member counts, expanded sizes, compression ratios, markup, and unresolved package paths. Review follow-up bounded terminate/join waits, routed teardown-thread construction/start failures through the same fail-closed handler, prevented late teardown completion from releasing the gate, repaired the canvas ingest harness, and added focused regressions. The persistent diagnostic inventory was regenerated only after reviewing all six changed statements: the two duplicated EPUB exception logs preserve the preexisting interpolation behavior after exception splitting, while the four shutdown diagnostics use static messages; no sink topology changed. On a clean branch from current origin/dev, targeted verification completed with 305 passed and 1 Windows-only skip; py_compile, Ruff, repository derived-artifact preflight, and git diff --check passed. The spawned three-EPUB chunking reproduction completed 3/3 with no errors or resident workers, a 62.3 ms maximum event-loop gap, and 496.2 MiB peak tree RSS versus roughly 750 MiB before. Full suite was not run locally per repository policy; GitHub's required test shards provide the merge gate. ADR: backlog/decisions/091-bounded-epub-archive-admission.md. Added the retained-worker-RSS incident to backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
