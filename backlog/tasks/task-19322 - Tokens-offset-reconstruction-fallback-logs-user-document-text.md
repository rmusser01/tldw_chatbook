---
id: TASK-19322
title: Tokens offset-reconstruction fallback logs user document text
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20'
updated_date: '2026-08-20'
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
`tldw_chatbook/Chunking/engine/strategies/tokens.py:779-782` — when a token
piece cannot be located during offset reconstruction, the fallback logs
`piece={repr(piece[:50] + '...' if len(piece) > 50 else piece)}` — up to 50
characters of raw user document text — at debug level.

DEBUG is not a defense under the TASK-15103/15600 programme bar (ADR-029,
`backlog/decisions/029-local-private-data-boundary.md`): users run with
debug sinks enabled, and content redaction is level-independent. Repair in
the house idiom — replace the content with what actually diagnoses the
mismatch: piece length, `pos`, and, if correlation across records matters,
a short stable hash of the piece. The diagnostic's job is "the tokenizer
produced a piece the text search could not align, at this position" — none
of that requires the characters themselves.

Owner ruling applies (stability-over-quick-wins, 2026-08-11): redact at
the call site in the established idiom; no formatter/sink cleverness.

Knock-on: `tokens.py` is a TASK-494 owner row in
`Docs/security/production-diagnostic-inventory.json` (call_count 28,
digest `9d6d1bc7cce0c3cc6d4b`) — the repair changes the digest, so
regenerate the inventory with only the reviewed delta in the same PR and
keep `scripts/check_persistent_diagnostic_inventory.py` green (the step
TASK-19042/19043 initially missed; 2026-08-20 lesson in
`backlog/docs/lessons-testing-evidence.md`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The offset-reconstruction fallback diagnostic records no user document text at any log level; it still records enough to diagnose the misalignment (piece length, position, and/or a stable hash)
- [x] #2 The persistent diagnostic inventory's `tokens.py` owner row is regenerated with only the reviewed delta in the same PR and `scripts/check_persistent_diagnostic_inventory.py` passes
- [x] #3 Regression coverage pins the repaired shape so document text re-entering this diagnostic turns a test red
- [x] #4 Offset-reconstruction behavior itself is unchanged — only the log record content changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the surrounding `_reconstruct_offsets_by_decoding` to fix what the
   diagnostic must still convey (which piece failed to align, where).
2. Write the regression pin FIRST against the unrepaired source (born-red:
   the log record contains the sentinel document text), using the
   TASK-15103-era loguru sink-capture idiom, placed in
   `Tests/Chunking/test_tokens_offsets.py`.
3. Repair in the house idiom: fixed event text + `piece_len`, `pos`, and a
   short stable `piece_sha256` digest; no characters of the piece.
4. Sweep the rest of `tokens.py` for same-class (document-text) leaks.
5. Re-derive the `tokens.py` inventory row with the checker's own
   `_scan_file`/`diagnostic_digest`, census ALL stored-vs-generated drift
   before regenerating, then restamp the summarization privacy fixture's
   two projection-hash fields under an isolated HOME.
6. Gates: checker exit 0; architecture inventory suite; summarization
   privacy suite; tokens-strategy Chunking tests; repo-wide collect-only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Repair** (`tldw_chatbook/Chunking/engine/strategies/tokens.py`): the
not-found fallback in `_reconstruct_offsets_by_decoding` now logs
`piece_len=<n>, piece_sha256=<10-hex>, pos=<n>` with the same fixed event
text; the `repr(piece[:50])` echo of decoded document text is gone.
`hashlib` import added; the digest is computed only when the fallback
fires. Reconstruction behavior is untouched (AC#4; the born-red test also
pins `offsets == [(0, len(piece))]`).

**Same-class sweep**: the other 27 logger calls in `tokens.py` carry
tokenizer/model names, counts, sizes, and exception captures from
encode/decode/init failures — no other direct interpolation of document
text; matches the TASK-19191 review's single finding. (The `{e}` captures
on encode/decode failures are the exception-capture class, out of scope
here and typically tokenizer-internal.)

**Born-red evidence**: `test_offset_reconstruction_fallback_logs_no_document_text`
run against the unrepaired source failed with the sentinel visible in the
captured record (`piece='SECRET-DOCUMENT-CONTENT-19322'`); green after the
repair with `piece_len=29, piece_sha256=3baeeb8112, pos=0` in the record.

**Inventory**: re-derived with the checker's `_scan_file`/`diagnostic_digest`:
`tokens.py` call_count 28 (unchanged), digest `9d6d1bc7cce0c3cc6d4b` →
`0127372c4bf1b469c5bd`. The pre-regeneration census found the checker RED
at base: dev PR #1880 (TASK-18310, merged after 19191's regen) added two
diagnostics without syncing the inventory — `UI/Console_Modules/workspace.py`
29→30 (`aa327c8e543ff0802849`) and `UI/Screens/chat_screen.py` 156→157
(`8759b4073f8408304b19`). Both reviewed under ADR-029: fixed-event
`logger.opt(exception=True).debug(...)` on internal registry-reconcile
errors, no user content interpolated; absorbed as an explicitly reviewed
delta so the gate is green. Summary `task_494_calls` 7128→7130; owner
files 517 and sink topology byte-identical; per-owner sums re-verified
against the summary buckets.

**Fixture**: `Tests/fixtures/summarization_diagnostic_review.json`'s two
normalized-projection hash fields restamped `c463bf02…` → `2852e14c…`
via the test module's own `_canonical_sha256(_normalized_inventory_projection(...))`
under an isolated HOME (exactly 2 occurrences replaced).

**Gates**: checker exit 0 (517 owners / 1210 / 7130 / 7 sinks);
`Tests/Architecture/test_persistent_diagnostic_inventory.py` 65 passed;
`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py` 257 passed;
tokens-strategy Chunking files 116 passed / 9 skipped / 1 failed —
`test_chunker_v2.py::test_process_text_tokenizer_override`, an HF-network
download failure reproduced bit-for-bit on a clean origin/dev worktree
(pre-existing, sandboxed network); repo-wide `--collect-only -q` clean
(52,297 collected, 0 errors).
<!-- SECTION:NOTES:END -->
