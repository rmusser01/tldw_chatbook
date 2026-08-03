# Library Ingest task-2043 Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Executed inline by the planning session; TDD per item, grouped commits, targeted suites (standing rule: no full-suite sweeps).

**Goal:** Land the round-2 critique P2 batch (task-2043) on `fix/library-ingest-p2-2043` (base origin/dev @ `34a02cb1d`).

**Architecture:** Display copy/state in `library_ingest_state.py`; row-detail expansion threaded screen→builder→queue-panel (in-place updates via the task-2042 dynamic-regions path, so detail toggling never recomposes the form); pre-flight duplicate detection in the existing pre-flight worker using thread-local media-db reads.

## Items → decisions (surveyed 2026-08-03)

1. **Inline error details (replaces the 10s toast)** — screen set `_library_ingest_expanded_details`; details button toggles + `_update_library_ingest_dynamic_regions()`. Builder kwarg `expanded_details`; row gains `details_expanded` + `detail_lines` (category, full UNWRAPPED message via a new public `unwrap_ingest_error`, and — for retryable failures — a hint line "A retry can succeed if the failure was transient or after installing missing tooling.", satisfying item 1's OR-arm for corrupt-file Retry). Queue panel renders the lines; button label flips Show/Hide. Dismiss/clear prune the set.
2. **Retry copy + prefix chain** — covered by (1): details show the unwrapped single-prefix message + the retry hint; the toast handler is deleted.
3. **Rail re-entry preserves the staged form** — drop the `_reset_library_ingest_transient_state()` call from `_select_library_rail_row` (:7300); the deep-link entry (:1990) keeps its documented reset. Update any tests pinning the rail-switch wipe; new test pins preservation.
4. **Select labels** — `_compose_type_group` select branch yields a `type-group-field-label` Static like the input branch (task-2012 missed Selects).
5. **Checkbox state glyph** — `StateGlyphCheckbox(Checkbox)` in the canvas: instance `BUTTON_INNER` = "✓"/" " tracked in `watch_value` (per-instance shadow of the class attr the renderer reads). Monochrome-distinguishable.
6. **Panel filler rows** — scoped TCSS: `LibraryIngestCanvas Collapsible { padding-bottom: 0; }` + `LibraryIngestCanvas Collapsible Contents { padding: 0 0 0 3; }` (Textual defaults supply the blank rows). Live-verified.
7. **Counts line honesty** — `_queue_counts_line` appends " — all ingests" (registry restores prior sessions from the jobs DB, so "session" would lie). Update pinned copy tests.
8. **Pre-flight duplicate warning** — DB dedups on sha256(PARSED content), not raw bytes, so scope to the generic group (text read ≈ parse): in `_run_library_ingest_preflight`, for ≤20 generic files ≤8MB each, sha256 the decoded text and `media_db.get_media_by_hash` (thread-local connections; read-only). New defaulted `PreflightResult.already_in_library: int`; state renders a quiet line "N file(s) appear to already be in your Library — they'll be matched, not re-imported." inside the summary child (non-structural). Soft copy absorbs parse-normalization mismatches.
9. **Casing** — `FileOpen(title="Import media")`.
10. **Considered, deliberately closed**: `.md` under "plain text file(s)" stays (accurate to the pipeline group; renaming the group label is copy churn without a correctness gain); a persistent ingest-history surface is deferred — the registry already restores from the jobs DB, a dedicated history view is its own feature (note in task).

Verification: targeted files (`test_library_ingest_state.py`, `test_library_ingest_canvas.py`, `test_library_ingest_runner.py`, shell ingest subset) + `--collect-only` sweep + live pass (details expansion, glyphs, filler rows, dup warning, preserved form) + PR.
