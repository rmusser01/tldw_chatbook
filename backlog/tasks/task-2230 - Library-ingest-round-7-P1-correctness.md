---
id: TASK-2230
title: >-
  Library ingest round-7 P1s (latest-batch regression, unresolvable-source gating, severity contrast)
status: Done
assignee: []
created_date: '2026-08-04 16:00'
labels:
  - library
  - ingest
  - ux
  - a11y
priority: high
dependencies: []
---

## Description (the why)

Round-7 dual-agent critique (snapshot `2026-08-04T15-38-09Z…`, 22/40;
B's 14 mechanical probes on the shipped rulings ALL passed — the score
reflects materially deeper coverage, not regressed rulings). Three
correctness defects:

1. **[P1, my own task-2221 regression]** `latest_batch_line` is computed
   only from groups with a `batch_id`, so a single-file run never
   updates it — the line keeps reporting the last MULTI-FILE batch (3×
   repro; confirmed in code). Separately `— all ingests` is
   queue-scoped: after Clear it drops (7→2) while Recent still lists 10,
   so the word "all" is a lie.
2. **[P1]** A nonexistent path renders "Path not found: …", BLANKS the
   gate line, and leaves Start styled exactly like a valid selection;
   pressing it yields a transient toast and NO queue row. A 404 URL is
   similar but does write a row. Unsupported/empty/invalid-option all
   correctly gate — three behaviours for one situation, and the most
   common user error (a typo) has the least recovery.
3. **[P1, a11y]** Validation error text measures 3.06:1 on the canvas
   background (AA fail) while ordinary help text is 7.21:1 — the
   importance hierarchy is inverted. The invalid field's border renders
   only while focused (1.05:1 unfocused) although the gate line says
   "Fix the highlighted options". Failed/skipped queue rows carry no
   colour distinction from done rows.

## Acceptance Criteria (the what)

- [x] The latest-run line updates for EVERY submission including a
      single file; it is suppressed when the queue holds only one run
      (the group header already says it).
- [x] The lifetime tally line does not claim "all" for a queue-scoped
      count (renamed, or sourced from the durable ledger).
- [x] A path that cannot be resolved gates Start with an explanatory
      gate line, consistent with unsupported/empty/invalid-option; if
      any unresolvable source stays submittable, the gate line says so
      and the attempt always leaves a queue record.
- [x] Validation/error text meets WCAG AA (≥4.5:1) against the canvas
      background.
- [x] An invalid field stays visibly marked when it loses focus (the
      gate line's "highlighted" is true).
- [x] Failed and skipped rows are distinguishable from done rows by
      colour IN ADDITION to their existing glyph+word (never
      colour-only).

## Implementation Notes

- **Latest-run regression (mine, task-2221):** the line filtered to
  groups with a `batch_id`, so single-file runs never became "latest".
  Now computed from the newest submission across ALL groups, relabelled
  "Latest run:", and suppressed when the queue holds one run (the group
  header already says it). Two regression tests: a later single-file run
  wins over an older batch; a one-run queue hides the line.
- **Lifetime tally:** "— all ingests" → "— in queue". The count is
  queue-scoped and drops on Clear while Recent keeps the history, so the
  old label denied its own number.
- **Unresolvable source:** `errors_are_path_problem` now gates Start and
  gets its own gate line ("Can't find that path — check it, or use
  Browse… to pick a file or folder."), consistent with
  unsupported/empty/invalid-option. Live-verified: Start renders
  disabled (115,115,115) with the explanation.
- **A11y:** validation text `$error` (3.06:1, AA fail, less readable
  than help text at 7.21:1) → `#ff8fa3` (7.71:1). Invalid fields keep a
  persistent `-ingest-option-invalid` border so the gate line's
  "highlighted" is true without focus. Failed rows carry `#ff8fa3` and
  skipped rows muted colour IN ADDITION to their glyph+word (the
  monochrome contract from round 4 holds). Live ANSI: failed
  `255,143,163` vs done `226,227,229`.

**Verification.** 297 core + 55 shell-subset green; collect clean.
Live-verified all three fixes on a fresh isolated profile.

**Qodo round (fixed in `2e27b4dd7`):** REAL bug — the persistent invalid
marker was compose-time only, so it never toggled on the in-place
text/number edit path (stayed marked when valid, never marked when
invalid): exactly this arc's recurring "the in-place updater must own
every conditional" family. Now toggled in
`handle_library_ingest_option_value_changed`, regression-tested
valid→invalid→valid. Plus: `$ds-status-error-readable` named token
instead of a repeated literal; Google-style docstrings on the new tests.
