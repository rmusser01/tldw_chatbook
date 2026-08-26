# ADR-091: Bound EPUB archive admission and parse concurrency

Status: Accepted
Date: 2026-08-26
Related Task: TASK-22508

## Decision

Chatbook will isolate ebook ingestion in one-process Library parse-pool
generations. Ebook and ordinary jobs do not share a pool generation: queued
EPUBs reuse the one ebook worker sequentially, then the ebook pool is retired
off the UI thread so its high-water resident memory is released. Ordinary jobs
continue to use the configured parallel worker count in their own generations.
This boundary is independent of the audio/video STT lane.

Before any EPUB reaches `ebooklib`, Chatbook will inspect its ZIP central
directory with Python's standard `zipfile` module. After the central-directory
bounds pass, it will bounded-read `META-INF/container.xml` and the referenced
OPF package metadata so manifest-declared markup is counted even when a
document uses a nonstandard filename extension. Every EPUB reader uses this
same checked-open seam. The archive is rejected when any of these fixed safety
limits is exceeded:

- 10,000 members;
- 64 MiB for one expanded member;
- 16 MiB total expanded markup bytes (recognized filename extensions plus
  manifest-declared HTML/XML media types);
- 128 MiB total expanded bytes;
- 200:1 expanded-to-compressed ratio for any non-empty member.

Negative declared member sizes, when surfaced by the ZIP reader, are rejected
by the same guard. Rejection is isolated to that ingest job and uses the stable
guard error `EPUB archive exceeds safety limits.` The guard only expands the
bounded container/package metadata needed for classification; it does not
extract content documents and does not replace ebooklib's ordinary EPUB-format
validation for non-ZIP inputs or archives without package metadata. When
container/package metadata is present but cannot be safely classified, the
guard fails closed with the safety error so a parser disagreement cannot bypass
custom-extension markup accounting.

These values are safety invariants, not user-tunable preferences. Raising them
requires revisiting measured parser memory amplification and this decision.

## Context

The Library parse pool defaults to three processes. EPUB was classified as a
light document, although `ebooklib` retains the archive model while the EPUB
extractors build full BeautifulSoup trees and a complete text string. Optional
chunking adds another representation of the same text, and the structured
result crosses multiprocessing IPC before the single writer persists it.

In the TASK-22508 reproduction, three 59 KiB EPUB files expanded to about
4.13 MiB each. The full queue-to-SQLite path peaked at 638 MiB without
chunking and 743-750 MiB with chunking. Admission limiting initially allowed
only one EPUB at a time, but did not lower that peak: the shared three-process
pool assigned successive EPUBs to different long-lived workers, which retained
about 410 MiB of combined high-water RSS after the batch. Running the same
batch through one physical worker reduced peak tree RSS to about 529 MiB.
Counting only compressed filesystem bytes or logical concurrency therefore did
not predict the resource cost; process ownership is part of the boundary.

The existing audio/video heavy set also participates in STT-specific routing,
so adding `ebook` to that set would couple EPUB admission to transcription
state. A logical one-job lane inside the shared pool was also insufficient
because pool scheduling rotated sequential jobs among resident workers.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Set the entire parse pool to one worker | Prevents useful parallelism for small text and document imports. |
| Limit EPUB admission to one job inside the shared pool | Successive EPUBs can rotate across long-lived workers, retaining one high-water heap per process. |
| Add `ebook` to the audio/video heavy set | That set is coupled to local-STT capacity and routing, which EPUB does not use. |
| Trust compressed file size or folder preflight totals | ZIP compression can hide the expanded bytes that ebooklib must retain. |
| Make archive limits configurable | A safety boundary should fail closed consistently; a hidden tuning knob would recreate the crash condition. |
| Stream or rewrite EPUB parsing | Could reduce the per-book cost, but is a broader parser replacement unnecessary to contain this incident. |
| Only reduce concurrency | Protects batches but leaves one extreme or malicious archive able to exhaust memory. |

## Consequences

- Multi-EPUB batches retain per-file progress and failure isolation while one
  physical ebook worker parses the batch sequentially.
- Ordinary light jobs retain configured parallelism, but wait at the pool-mode
  boundary rather than sharing resident workers with an ebook batch.
- Switching between ebook and ordinary work incurs one spawn-pool teardown and
  creation; teardown remains off the UI thread.
- Broken pool generations use the same teardown gate, so queued work resumes
  only after the old workers have joined and cannot overlap a replacement.
- If termination or join fails, the gate remains asserted and queued local jobs
  fail retryably with restart guidance; a replacement generation is not created
  while old workers may still exist.
- Very large image-heavy EPUBs may be refused even when structurally valid;
  the refusal is explicit and does not crash the application.
- MOBI, AZW, AZW3, and FB2 share the one-at-a-time ebook lane but do not use
  the ZIP-specific EPUB admission guard.
- The result payload remains unchanged. Removing duplicate full-content and
  chunk representations is a separate optimization, not required for this
  containment boundary.

## Links

- [TASK-22508](../tasks/task-22508%20-%20Prevent-multi-EPUB-imports-from-exhausting-memory.md)
- [Library parallel parse design](../../Docs/superpowers/specs/2026-07-10-library-f3-parallel-parse-design.md)
- [Library heavy-lane design](../../Docs/superpowers/specs/2026-07-12-library-ingest-heavy-lane-design.md)
