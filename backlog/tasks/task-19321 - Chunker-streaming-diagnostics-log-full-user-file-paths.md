---
id: TASK-19321
title: Chunker streaming diagnostics log full user file paths
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - security
  - diagnostics
  - chunking
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from TASK-19191's independent review; re-verified at dev `a542fd463`.
Three diagnostic sites in `tldw_chatbook/Chunking/engine/chunker.py`'s
file-streaming path record user file paths (ADR-029 private data) in log
records:

- `:2363` — `logger.info(f"Stream processing file: {file_path} ({file_size}
  bytes)")` — the full user path at INFO on every streamed file, the
  hottest of the three.
- `:2485` — `logger.error(f"File stream decoding failed for {file_path}:
  {e}")` — full path, plus a `UnicodeDecodeError` whose own text carries
  byte-context from the file.
- `:2490` — `logger.error(f"File stream processing failed: {e}")` — no
  explicit path, but an `OSError` in `_CHUNKER_NONCRITICAL_EXCEPTIONS`
  stringifies with the filename embedded, so the path leaks through `{e}`.

Repair to the TASK-15103/15600 programme bar (ADR-029,
`backlog/decisions/029-local-private-data-boundary.md`): keep the
diagnostic useful while redacting the private value — basename-or-hash
instead of full path, lengths/counts instead of content, and
`type(e).__name__` (or an otherwise path-free rendering) where the
exception message itself can carry the private value, as at `:2490`.
Owner ruling applies (stability-over-quick-wins, 2026-08-11): repair the
call sites in the established programme idiom; do not reach for a
sink-level filter or log-formatter trick that other emit paths can bypass.

Two knock-ons the fix must chase: (1) `chunker.py` is a TASK-494 owner row
in `Docs/security/production-diagnostic-inventory.json` (call_count 49,
digest `fae786d5d91b387794b3`) — repairing call shapes changes the digest,
so the inventory must be regenerated with only the reviewed delta in the
same PR and `scripts/check_persistent_diagnostic_inventory.py` must pass
(the exact step TASK-19042/19043 initially missed; see the 2026-08-20
lesson in `backlog/docs/lessons-testing-evidence.md`). (2) The adjacent
raised exception messages (`InvalidInputError` at `:2487-2488` embeds the
full path; `ChunkingError` at `:2491` embeds `str(e)`) flow to callers,
not the log — check whether any downstream handler logs them verbatim
before deciding they are out of scope, and record the verdict either way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No diagnostic emitted by the chunker file-streaming path records a full user file path or user file content; the repaired records still identify the file (basename or stable hash) and size/failure class well enough to debug
- [ ] #2 The `:2490` shape no longer interpolates raw exception text that can embed the filename; the record retains the exception type and path-free context
- [ ] #3 The persistent diagnostic inventory's `chunker.py` owner row is regenerated with only the reviewed delta in the same PR and `scripts/check_persistent_diagnostic_inventory.py` passes
- [ ] #4 Regression coverage pins the repaired shapes so a full path re-entering these streaming diagnostics turns a test red
- [ ] #5 Production behavior other than log-record content is unchanged (no sink filters, no behavioral workarounds)
<!-- AC:END -->
