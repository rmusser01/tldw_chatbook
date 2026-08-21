---
id: TASK-19321
title: Chunker streaming diagnostics log full user file paths
status: Done
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
- [x] #1 No diagnostic emitted by the chunker file-streaming path records a full user file path or user file content; the repaired records still identify the file (basename or stable hash) and size/failure class well enough to debug
- [x] #2 The `:2490` shape no longer interpolates raw exception text that can embed the filename; the record retains the exception type and path-free context
- [x] #3 The persistent diagnostic inventory's `chunker.py` owner row is regenerated with only the reviewed delta in the same PR and `scripts/check_persistent_diagnostic_inventory.py` passes
- [x] #4 Regression coverage pins the repaired shapes so a full path re-entering these streaming diagnostics turns a test red
- [x] #5 Production behavior other than log-record content is unchanged (no sink filters, no behavioral workarounds)
<!-- AC:END -->

## Implementation Plan

1. Repair the three call sites in `chunk_file_stream` in the TASK-15103 idiom
   (call-site repair, no sink filter): compute one stable content-free file
   handle (`path_sha256=<sha256(resolved path)[:12]>`) next to the existing
   `stat()` and use it in all three records; keep byte size at `:2363`; at
   `:2485` replace `{file_path}: {e}` with the handle + `type(e).__name__` +
   encoding + byte offset (ints/codec name only — the raw
   `UnicodeDecodeError` text carries byte context from the file); at `:2490`
   replace `{e}` with the handle + `type(e).__name__` (an `OSError` in
   `_CHUNKER_NONCRITICAL_EXCEPTIONS` stringifies with the filename embedded).
2. Rule on the adjacent raise text (`InvalidInputError` `:2486-2488`,
   `ChunkingError` `:2491`): trace `chunk_file_stream` callers; record the
   verdict in Implementation Notes with the 15103 precedent (the programme
   repaired logger call sites and left raise messages caller-facing).
3. Born-red regression test `Tests/Chunking/test_chunker_stream_diagnostic_privacy.py`
   using the programme's loguru-capture idiom (`logger.add(lambda m: ...)`,
   assert the fixed event IS logged and the path/exception text is NOT):
   one test per site — happy-path INFO, `UnicodeDecodeError` via cp1252
   bytes, `OSError`-embeds-path via streaming a directory
   (IsADirectoryError/PermissionError). Verify each fails at base.
4. Re-derive the `chunker.py` owner row in
   `Docs/security/production-diagnostic-inventory.json` via the checker's own
   `_scan_file` + `diagnostic_digest` (call_count stays 49 — reworded, not
   added/removed — so summary buckets are untouched; only
   `diagnostic_digest` changes); hand-edit that one string; checker exit 0.
5. Restamp `Tests/fixtures/summarization_diagnostic_review.json`: import the
   privacy test module via importlib with `sys.modules["privmod"]` registered
   BEFORE `exec_module`, compute
   `_canonical_sha256(_normalized_inventory_projection(inv, set(MODULE_COUNTS)))`,
   raw-string-replace the old hash (assert exactly 2 occurrences: checked +
   origin_dev_generated fields). Isolated HOME throughout.
6. Gates: checker exit 0; `Tests/Architecture/test_persistent_diagnostic_inventory.py`
   65/65; `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py` all
   green; new test green; `Tests/Chunking/` chunker-path suites green;
   repo-wide `--collect-only -q` clean. All with worktree-pinned
   PYTHONPATH+cwd and the `tldw_chatbook.__file__` assert.

## Implementation Notes

Repaired the three `chunk_file_stream` diagnostic sites in the TASK-15103
call-site idiom, and — a discovery the filing did not anticipate — carried
the repair through the Chunking engine's **vendoring contract**, which
otherwise turns any hand-edit of `chunker.py` red.

**The repair (`tldw_chatbook/Chunking/engine/chunker.py`).** One stable
content-free handle is computed next to the existing `stat()`:
`path_ref = sha256(str(file_path.resolve()))[:12]` (surrogatepass encode for
undecodable POSIX names). The three records became:
- `:2363` INFO → `Stream processing file: path_sha256=<ref> (<n> bytes)`
  (hash chosen over basename: a user's file NAME is itself private under
  ADR-029; the hash still correlates records and is recomputable from a
  candidate path).
- `:2485` ERROR → handle + `type(e).__name__` + codec + `e.start` byte
  offset; no path, no `str(e)` (a `UnicodeDecodeError`'s text carries byte
  context from the file).
- `:2490` ERROR → handle + `type(e).__name__` only; an `OSError` in
  `_CHUNKER_NONCRITICAL_EXCEPTIONS` stringifies with the filename embedded,
  so the message is dropped while the raised `ChunkingError` still carries
  full detail to the caller.

**Raise-text verdict (description knock-on 2).** `chunk_file_stream` has
ZERO production callers at dev `0f5cba2f7` (grep: the definition plus
`Tests/Chunking/` only), so the `InvalidInputError`/`ChunkingError` raise
texts (and the `File not found: {file_path}` raise at `:2334`) flow to
library callers/tests only and never reach a logger. ADR-029's boundary
governs persistent log records, and the 15103 repair commits
(`d8a0d5234`/`566d6f9be`/`03c71d053`) consistently rewrote logger call
sites while leaving raise messages caller-facing. Raise texts kept as-is;
any future production caller's handler must log type-only per the same bar.

**Vendoring (the surprise).** The engine is vendored from tldw_server @ pin
`385afa951` (spec §5.2: "vendored files are never hand-edited");
`Tests/Chunking/test_sync_script.py` diffs every vendored file against the
pin and my edit turned it red (dev's `chunker.py` is byte-identical to the
pin — verified against a pinned clone). A subclass would duplicate the whole
200-line generator and leave the leaky original importable, so instead
`sync_chunking_engine.py` gained `ENGINE_PATCHES` — the exact mechanism the
script already uses for ported tests: sync = upstream-at-pin + rewrite +
patches, the local-modification check compares against the PATCHED state,
and upstream drift under a patch anchor fails loudly (`_replace_once`).
Patch output verified byte-identical to the committed `chunker.py`;
`test_sync_script.py` 4/4. Recorded in `VENDOR_MANIFEST.toml` `[patches]`
and a spec §5.2 amendment note. NOTE for tasks 19322/19323: your files
(`strategies/tokens.py`, `security_logger.py`) are vendored too — add your
repairs as `ENGINE_PATCHES` entries, not bare edits.

**Inventory / fixture method (for merge-time re-derivation).** The
`chunker.py` row was re-derived with the checker's own
`_scan_file`+`diagnostic_digest`: call_count 49 unchanged (3 reworded, none
added), digest `fae786d5d91b387794b3` → `4703a3a31544d0a4ae94`; summary
buckets untouched by this task. The same commit folds REVIEWED dev drift:
TASK-18310 (#1880, merged 22:44) landed after the 19191 regen (#1878,
22:24), leaving dev's checker red — `workspace.py` 29→30 and
`chat_screen.py` 156→157 (both new calls are fixed-string
`logger.opt(exception=True).debug` events, no interpolated user data),
`task_494_calls` 7128→7130. Fixture
`Tests/fixtures/summarization_diagnostic_review.json` restamped via
importlib-loading the privacy test module (`sys.modules["privmod"]` set
before `exec_module`), computing
`_canonical_sha256(_normalized_inventory_projection(inv, set(MODULE_COUNTS)))`,
and raw-replacing the old hash `c463bf02…` → `4dc9f467…` in exactly its 2
fields; isolated HOME throughout.

**Born-red evidence.** New
`Tests/Chunking/test_chunker_stream_diagnostic_privacy.py` (loguru-capture
idiom from the 15103-era `Tests/RAG/test_fusion.py` pins): 3/3 FAIL against
`origin/dev`'s chunker.py (assertions show the leaked tmp paths verbatim,
including the OSError `[Errno 21] Is a directory: '<path>'` embed), 3/3
pass repaired. The OSError case streams a directory — a real
`IsADirectoryError`/`PermissionError`, no monkeypatching.

**Gates (final committed state).** Checker exit 0 (`517 owners, 1210
TASK-492 calls, 7130 TASK-494 calls, 7 sink files`); architecture suite
65/65; privacy suite 257/257; new test 3/3; `Tests/Chunking/` 428 passed /
32 skipped / 1 xfail with exactly 2 pre-existing environment reds
(`test_process_text_tokenizer_override`, `test_golden_parity[tokens-cjk]`
— both fail IDENTICALLY with base chunker.py swapped in: offline HF-cache
lookups, untouched by this diff); repo-wide `--collect-only -q` 52,299
collected, exit 0. All runs cwd+PYTHONPATH-pinned to the worktree with the
`tldw_chatbook.__file__` assert.

**Files.** `tldw_chatbook/Chunking/engine/chunker.py`,
`Helper_Scripts/sync_chunking_engine.py`,
`tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml`,
`Docs/superpowers/specs/2026-08-18-chunking-engine-parity-design.md`,
`Docs/security/production-diagnostic-inventory.json`,
`Tests/fixtures/summarization_diagnostic_review.json`,
`Tests/Chunking/test_chunker_stream_diagnostic_privacy.py` (new), this
task file, and a lessons entry.
