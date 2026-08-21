---
id: TASK-19191
title: Per-row-reviewed regeneration of the persistent diagnostic inventory (dev red)
status: Done
assignee: ['@claude']
created_date: '2026-08-20'
labels:
  - test-health
  - diagnostics
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_persistent_diagnostic_inventory.py` is red on dev:
`scripts/check_persistent_diagnostic_inventory.py` exits 1 at dev `7877defba`
(reproduced 2026-08-20 in a pristine worktree, PYTHONPATH+cwd pinned,
`tldw_chatbook.__file__` asserted inside the worktree). The checker's own
guidance applies: review the diff row by row before running `--write` —
regenerating without reading it is the exact failure mode the content-keyed
digest exists to prevent (task-3750).

Current rebuild-vs-committed diff (copy committed → `--write` → structured
diff → committed file restored):

Rows only in COMMITTED (1):
- `RAG_Search/enhanced_chunking_service.py` (count=6) — file deleted, row kept.

Rows only in REBUILD (22):
- 19 `Chunking/engine/` files from the chunking-engine landing (base 13,
  chunker 49, multilingual 9, process_text/options 1, regex_safety 1,
  security_logger 7, strategies: code 3, code_ast 5, ebook_chapters 17,
  fixed_size 1, json_xml 25, paragraphs 11, rolling_summarize 2, semantic 11,
  sentences 18, structure_aware 6, tokens 28, words 15; utils/metrics 2)
- `RAG_Search/parent_child_adapter.py` (1)
- `UI/Library_Modules/library_media_browse_controller.py` (2) — the known
  pre-existing 3-row library drift (post-d64608b84 regen, 1ba3d4755/b4ebe85e8
  era)
- `Widgets/Console/console_changed_files_section.py` (2) — NEW since the
  wave-3 queue was written: the changed-files rail landing (12d621071 et seq.)

Rows changed (10):
- `Chat/console_chat_controller.py` 45→45 (digest only)
- `Chunking/Chunk_Lib.py` 100→31
- `DB/Client_Media_DB_v2.py` 354→338 (library drift)
- `Event_Handlers/STTS_Events/stts_events.py` 30→29 (TASK-19043's merged
  deletion — implementer and reviewer both missed the hand-edit playbook step)
- `RAG_Search/chunking_service.py` 5→3
- `RAG_Search/simplified/enhanced_rag_service.py` 11→10
- `UI/Screens/change_review_screen.py` 1→13 (changed-files rail landing)
- `UI/Screens/chat_screen.py` 153→156
- `UI/Screens/library_screen.py` 110→109 (library drift)
- `Widgets/enhanced_file_picker.py` 6→6 (digest only)

persistent_sink_topology: +1 row in rebuild —
`Chunking/engine/security_logger.py` (an addHandler-kind sink; sink additions
are exactly what this inventory exists to review).

Summary: owner_files 494→515, persistent_sink_files 6→7, task_492_calls
1209→1209, task_494_calls 6974→7122.

The drift has GROWN past the contributors known at wave-3 close-out (the
changed-files rail rows and the chunking_service/enhanced_rag_service/
chat_screen deltas are new), so re-produce the diff at the implementation
commit before reviewing. A red gate left standing stops guarding — the next
unreviewed diagnostic lands invisibly behind it (same failure shape as the
task-19044 lesson).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The rebuild-vs-committed diff is re-produced at the implementation commit and every row delta is individually reviewed and dispositioned in the task's Implementation Notes (new diagnostics accepted or challenged, deleted rows confirmed against deleted/moved code, digest-only changes explained), with particular attention to the new persistent sink in `Chunking/engine/security_logger.py`.
- [x] #2 The committed inventory is regenerated via the checker's `--write` only after that review, and `Tests/Architecture/test_persistent_diagnostic_inventory.py` passes on the result.
- [x] #3 Any row the review REJECTS (a diagnostic that should not exist, e.g. one that could log sensitive content) is filed or fixed rather than silently accepted into the inventory. (No whole row was rejected; three call-level findings — F1–F3 in the Implementation Notes — are recorded there and handed to the controller for filing rather than silently absorbed.)
- [x] #4 No hand edits to the regenerated JSON beyond what the review justifies; the summary counts match the accepted rows. (The JSON is the untouched `--write` output; all four summary invariants re-verified by probe.)
<!-- AC:END -->

## Implementation Plan

1. Worktree off origin/dev (`555298a33`, includes 19190's stts_events row fix); baseline both gates
   (arch 10 red / summ 3 red) and the checker (exit 1) with PYTHONPATH+cwd pinned.
2. Re-produce the rebuild-vs-committed diff at the implementation commit via an in-memory
   `build_inventory()` vs the committed JSON (committed file untouched): structured
   rows-only-in-committed / rows-only-in-rebuild / changed / sink deltas.
3. Review every row: for NEW rows print and read every diagnostic call's source segment; for
   CHANGED rows walk git history to the revision whose scan matches the committed digest and diff
   added/removed call segments; for the REMOVED row verify the file's current state on disk.
   Classify against the script's TASK-492/494 rules and the 15103/15600 privacy bar; record any
   unsafe diagnostics as findings without blocking the truthful regeneration.
4. Read `Chunking/engine/security_logger.py` end-to-end and judge the new addHandler-kind sink.
5. Run `--write`; verify checker exit 0 and the four summary invariants with a printed probe.
6. Reconcile `Tests/fixtures/summarization_diagnostic_review.json`: recompute the
   `_normalized_inventory_projection` sha256 with the test module's own helpers (checked ==
   generated required) and stamp both boundary fields, per the d64608b84/5899c31a1 precedent.
7. Gates: arch 65/65, summarization all pass, repo-wide `--collect-only -q` clean; commit.

## Implementation Notes

Regenerated `Docs/security/production-diagnostic-inventory.json` via the checker's `--write`
after a per-row review of every delta, and reconciled the summarization boundary fixture.
Base: origin/dev `555298a33`. Summary moved 494→517 owner files, 6→7 sink files,
task_492_calls 1209 (unchanged), task_494_calls 6972→7128. The drift had grown past the
filing-time diff: three Scheduling count changes and two new Scheduling migration rows joined,
and the stts_events row from the filing diff was already resolved by task-19190's hand-fix.

The review is organized BY FILE (not by absolute counts), so a mechanical re-regen on a rebase
does not invalidate it: each disposition below is about that file's diagnostic content, and a
future merge that touches other files changes only rows outside this review.

### Removed row (1) — accepted

- `RAG_Search/enhanced_chunking_service.py` (was 6 calls): correction to the filing-time diff,
  which said "file deleted". The file still EXISTS — `a762a9731` (Phase B retirement) rewrote it
  as a diagnostic-free delegating shim over the vendored engine + `parent_child_adapter`;
  scanning it yields 0 logger calls, so the row rightly leaves the inventory. Verified by
  reading the file, not just the git log.

### New rows (24) — all TASK-494, all accepted

Classification: every new path is outside the TASK-492 prefixes (`Chat/`, `LLM_Calls/`,
`MCP/`, `Tools/`) and not `Agents/mcp_tool_provider.py`, so TASK-494 with the standard reason
is correct for all 24.

- 19 `Chunking/engine/**` rows (vendored tldw_server engine, `72a295a7e`): every diagnostic
  call segment was printed and read. The corpus is overwhelmingly counts, method/strategy
  names, config keys/defaults, overlap/max_size adjustments, and optional-dependency
  availability at debug/info/warning — squarely metadata. Borderlines accepted deliberately:
  `chunker.py`'s "Suspicious control characters found: {control_char_samples}" logs at most 10
  repr'd control characters (category Cc) from user input — control chars carry no prose, so
  this cannot reconstruct user text; `strategies/ebook_chapters.py` logs the user-CONFIGURED
  custom chapter regex at debug (config, not content). Three call-level findings recorded
  below (F1, F2) rather than silently absorbed; none rises to rejecting a row.
- `RAG_Search/parent_child_adapter.py` (1 debug, %d counts only) — new seam from `a762a9731`.
- `Scheduling/db/migrations/v1_to_v2.py`, `v2_to_v3.py` (1 constant debug each) — schema
  migration landings (task-18937/18939 era).
- `UI/Library_Modules/library_media_browse_controller.py` (2 warnings: operation +
  exception_type only, the 15600 house style) — the known library drift.
- `Widgets/Console/console_changed_files_section.py` (2 constant-message
  `opt(exception=True).warning`) — changed-files rail landing.

### Changed rows (12) — every one diffed at CALL level, all accepted

Method: for each row, walked `git log` back to the revision whose scan reproduces the
committed digest, then diffed the (method, segment) multisets old→new — stronger evidence
than the required 2–3 spot-checks.

- `Chat/console_chat_controller.py` (digest-only, 45→45): four debug messages reworded
  (approval/skill "clear" → "remount" during teardown/revocation); constant strings.
- `Chunking/Chunk_Lib.py` (100→31): compat-shim rewrite (`70542dbef`). The 76 removed calls
  include two former user-content leaks now GONE ("Extracted JSON metadata: {…}", "Extracted
  header text: {…}" at debug) — a privacy improvement; the 7 added are counts/option messages.
- `DB/Client_Media_DB_v2.py` (354→338): the library privacy rework. Removed calls logged DB
  paths (`self.db_path_str`), user search queries and result titles at info; added calls are
  error_type/count/mode-only structured events — a clear privacy improvement.
- `RAG_Search/chunking_service.py` (5→3) and `RAG_Search/simplified/enhanced_rag_service.py`
  (11→10): engine-migration residue; counts-only messages removed/reworded.
- `Scheduling/scheduler/loop.py` (4→6), `Scheduling/services/scheduling_service.py` (6→8),
  `UI/Screens/scheduling/schedules_workbench.py` (10→11): task-18939 handler timeout + manual
  reminder run; new calls log task ids/types, timeout seconds, callback `__qualname__` —
  metadata only ( `logger.exception` in the scheduler loop matches the pre-existing pattern
  there).
- `UI/Screens/change_review_screen.py` (1→13): annotate-loop rail; 12 constant-message
  `opt(exception=True).warning` calls.
- `UI/Screens/chat_screen.py` (153→156): changed-files rail; adds log conversation_id (an
  internal DB id, consistent with pinned precedent) and one constant-message warning.
- `UI/Screens/library_screen.py` (110→109): removed one warning that logged conversation_id.
- `Widgets/enhanced_file_picker.py` (digest-only, 6→6): the same
  "Failed to persist file-picker recent/last-dir state" error call reflowed (whitespace inside
  the source segment changes the content digest); method and fields unchanged, still
  `type(e).__name__` only — matches its TASK-14651 metadata-only registry pin.

### New persistent sink — accepted with a caveat

`Chunking/engine/security_logger.py`, `loguru_sink` kind, scope `SecurityLogger.__init__`:
`logger.add(log_file, filter="security" in extra, format=time|level|event_type|message,
rotation 100 MB, retention 30 days)`. Judgment: acceptable. (a) It is DORMANT in production —
the sink registers only when a `log_file` is passed, the global `get_security_logger()` path
passes none, and `configure_security_logging` has zero production callers. (b) When
configured, the format persists only the message; messages are constant summaries plus
sizes/category labels. The user-content payloads (`xml_sample` ≤500 chars of user XML from
`json_xml.py`, blocked regex pattern ≤200 chars) live in the `details` dict, which is stored
in the in-memory `_events` list and never reaches the sink format (extra carries only
`security` + `event_type`). The latent export path is finding F3.

### Findings for filing (recorded, regenerated truthfully — not fixed here)

- F1 `Chunking/engine/chunker.py`: logs the FULL user file path at info ("Stream processing
  file: {file_path} ({file_size} bytes)") and error ("File stream decoding failed for
  {file_path}: {e}") in the file-streaming path — against the 15103/15600 "no full paths of
  user files" bar if app code ever drives `chunk_file`/streaming.
- F2 `Chunking/engine/strategies/tokens.py`: offset-reconstruction fallback logs up to 50
  chars of the document text at debug ("Token piece not found … piece={repr(piece[:50]…)}") —
  user text in a diagnostic.
- F3 `Chunking/engine/security_logger.py`: latent `export_events()` writes the in-memory event
  list — including `xml_sample` user content — to an arbitrary path with a plain `open()`
  (invisible to the sink topology, which does not track bare `open`). No production caller
  today; worth redacting `details` at capture or gating the export.

### Fixture reconciliation (handoff from 19190)

`Tests/fixtures/summarization_diagnostic_review.json` pins
`manifest_boundary.checked_normalized_inventory_sha256` /
`origin_dev_generated_normalized_inventory_sha256`: sha256 over
`_normalized_inventory_projection` (the WHOLE inventory with only the two summarization
owners' call_count/digest and `summary.task_492_calls` masked), so ANY reviewed inventory
change outside those two owners legitimately moves it. Following the d64608b84/5899c31a1
precedent: recomputed the projection hash with the test module's own helpers on the
regenerated inventory (checked == generated == `bcd5b70ec5e17256…` asserted before stamping)
and stamped BOTH fields. The two owner rows themselves were untouched by this regeneration
(1209 TASK-492 calls unchanged), which is exactly the boundary the fixture protects. The three
red tests were all downstream of the stale hash (the two mutant tests failed early on the
projection assert with a non-matching message); no test was modified.

### Gates (exact, read from output files)

- `Tests/Architecture/test_persistent_diagnostic_inventory.py`: 65 passed (was 10 failed/55).
- `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`: 257 passed (was 3 failed/254).
- Repo-wide `pytest --collect-only -q`: 52228 tests collected, exit 0.
- Checker: exit 0 — "517 owners, 1209 TASK-492 calls, 7128 TASK-494 calls, 7 sink files";
  probe-verified len(owners)==owner_files, per-owner sums == summary, len(topology)==sink files.

### Controller caveat (concurrency)

Dev merges touching diagnostics re-drift this inventory constantly. At merge time, re-run the
checker; if red, a mechanical re-regen (`--write` + re-stamp of the two fixture boundary
fields via `_normalized_inventory_projection`/`_canonical_sha256`) is justified for rows
already dispositioned above — the review is per-file, so only rows for files NOT covered here
need fresh eyes.

Files changed: `Docs/security/production-diagnostic-inventory.json` (regenerated),
`Tests/fixtures/summarization_diagnostic_review.json` (two boundary hashes), this task file.
No lessons entry added: the 19042/19043 deletion-direction lesson in
`lessons-testing-evidence.md` already records this incident and names task-19191.
