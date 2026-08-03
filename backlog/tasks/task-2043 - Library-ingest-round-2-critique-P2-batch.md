---
id: TASK-2043
title: >-
  Library ingest round-2 critique P2 batch
status: Done
assignee: []
created_date: '2026-08-03 02:00'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description (the why)

Remaining P2-grade findings from the round-2 dual-agent critique
(snapshot `2026-08-03T01-33-07Z…`, 24/40) not covered by TASK-2041/2042.

## Acceptance Criteria (the what)

- [x] Corrupt-file extraction failures: Retry stays offered (missing
      tooling genuinely can fix an extraction miss) and the expanded
      details now explain what a retry could fix; the details never chain
      two "Failed to …" prefixes (shared `unwrap_ingest_error`).
- [x] "Show details" presents error details in a readable, copyable,
      non-expiring surface (inline expandable row section), not a ~4s
      toast.
- [x] Rail re-entry either preserves the staged form for the session or
      warns before discarding it (today it silently wipes path, pre-flight
      and metadata — deliberate but destructive).
- [x] Select fields (PDF engine, Encoding) get visible labels like the
      value inputs did in task-2012.
- [x] Checkbox on/off is distinguishable without color (custom glyph or
      suffix; stock Textual renders "X" for both states).
- [x] Collapsed panels no longer carry ~3 blank filler rows; expanded
      panels no trailing blank region.
- [x] The queue counts line reflects the current batch (or labels itself
      as all-history) so batch outcomes don't blur together.
- [x] Pre-flight warns "already in your Library" for byte-identical files
      (the content hash exists at pre-flight time) instead of the match
      being an after-the-fact discovery.
- [x] "Import media" (canvas) vs "Import Media" (picker) casing aligned.
- [x] Considered and deliberately closed: `.md` stays under "plain
      text file(s)" (accurate to the pipeline group; renaming is copy
      churn without a correctness gain). A dedicated persistent
      ingest-history surface is deferred — the registry already restores
      prior sessions from the jobs DB; a history view is its own feature
      if wanted.

## Implementation Notes

Shipped on `fix/library-ingest-p2-2043`:

- **Inline error details** replace the auto-expiring toast: screen set
  `_library_ingest_expanded_details` toggles per-row lines (category +
  full message via the new shared `unwrap_ingest_error` + an honest
  retry hint when Retry is offered) through the task-2042 in-place queue
  update; button flips Show/Hide. Live: "Category: unsupported file
  type" + full details rendered under the row.
- **Session-persistent form**: rail switches run
  `_pause_library_ingest_transient_ui` (debounce stop + clear disarm)
  instead of the full wipe; the deep-link entry keeps its documented
  reset. Live: path survived a Media round-trip.
- **Pre-flight duplicate forecast**: the worker hashes staged generic
  files (20-file/8MB caps, thread-local db reads — the DB dedups on
  sha256 of PARSED content, so only read≈parse types are checkable);
  new `PreflightResult.already_in_library` renders a quiet line. Live:
  "1 file appears to already be in your Library — it will be matched,
  not re-imported." at dwell time, pre-submit.
- Select labels; `StateGlyphCheckbox` (glyph carries on/off — live:
  `▐ ▌` vs `▐✓▌`); counts line "— all ingests"; FileOpen casing;
  panel filler rows removed (Textual Collapsible defaults + TWO unscoped
  `_conversations.tcss` rules — the `CollapsibleTitle height:3` needed a
  child-combinator specificity bump because it sits LATER in the bundle).
- **Unmasked latent bug fixed (pre-existing since PR #717)**: the options
  loader called `get_cli_setting(prefix, name)` — with a dotted first
  arg the second positional is the DEFAULT, so fresh profiles loaded the
  field NAME string for every option: analyze truthy-flipped ON,
  type_options filled with junk. The old wipe-on-entry masked it; form
  persistence exposed it. Explicit `None` defaults now; pinned by a
  call-shape recording test (the task-687/698 recorder lesson).

Verified: canvas+state+runner 235/235; shell subset 97/97; full-tree
collect-only clean; live pass on an isolated profile (defaults correct,
zero filler rows, inline details, glyphs, dup forecast, preserved form).
