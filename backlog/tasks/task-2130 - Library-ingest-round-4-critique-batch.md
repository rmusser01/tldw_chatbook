---
id: TASK-2130
title: >-
  Library ingest round-4 critique batch (options trust, error experience, session ledger, mechanical minors)
status: In Progress
assignee: []
created_date: '2026-08-04 00:00'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description (the why)

The round-4 dual-agent critique (snapshot `2026-08-03T23-40-32Z…`, 25/40 —
trend 21 → 24 → 29 → 25; the dip is new coverage, not regression: every
round-3 fix was verified holding by both agents) found the remaining
defects cluster in three areas plus mechanical minors. Owner priority:
options trust first, then everything in this pass.

1. **Options trust (P1, leads):** "Reset to defaults" resets Selects but
   not text Inputs (Chunk size kept "10009999" through two presses); the
   collapsible-header settings receipt misreports Input values (claims
   "1000" while the field holds "10009999"); invalid option values are
   neither validated nor gate Start ("abc" chunk size ran a job; only
   signal an orange border visible while focused — color-only).
2. **Error experience (P1):** failure "Show details" repeats the summary
   byte-for-byte; the advisory says "installing missing tooling" naming
   none; "Unsupported: photo.jpg." appears exactly when the
   supported-formats sentence is hidden, and never says what would work.
3. **Session ledger (P2):** the armed "Press again to clear N finished"
   confirm can land off-viewport when the button sits at the bottom edge
   (reads as broken); a confirmed clear destroys Recent ingests including
   failure records; the post-activity empty state claims "No ingest jobs
   yet." after a ten-job session.
4. **Mechanical minors (both agents):** completion toasts overdraw both
   canvas bottom-border rows; disabled "Start ingest" label ≈1.5:1
   contrast; Browse dialog ignores the typed path fragment and opens at
   home; during an 80-file batch the tally read "3 done — all ingests"
   with no in-flight signal; the async duplicate-forecast line lands late
   and shifts layout; the expanded details row collapses when Retry is
   pressed; the status line above Start is blank for valid selections
   where a commit summary would close the distance from the forecast;
   Encoding Select opened on the second click, not the first.

## Acceptance Criteria (the what)

- [ ] Reset to defaults returns every control in the panel — Selects,
      checkboxes, AND text Inputs — to defaults, including the top-level
      generic fields and any persisted option values.
- [ ] The panel-header receipt always matches the actual field values:
      editing a text Input updates the receipt the same way editing a
      Select does.
- [ ] An invalid option value (non-numeric chunk size/overlap) shows an
      inline text message (not color-only) and gates Start with an honest
      gate line, the same way a bad path does.
- [ ] Failure details are never a verbatim repeat of the summary: the
      expanded row carries the underlying error (and names the missing
      dependency when that is the category); advisory copy is
      per-category.
- [ ] The unsupported-files line says what IS supported (or why the file
      is not), without requiring the user to clear the path.
- [ ] The armed clear-finished confirm is scrolled into view on first
      press.
- [ ] A confirmed Clear finished does not erase the Recent-ingests
      ledger; failure records survive there.
- [ ] The queue empty-state copy after a session with activity does not
      claim "No ingest jobs yet."
- [ ] Disabled Start-ingest label is readable (≥3:1 contrast against its
      background).
- [ ] Browse opens at the typed path's directory when one is present and
      valid.
- [ ] The queue tally shows in-flight work during a batch (not just the
      done count).
- [ ] The duplicate-forecast area does not shift layout when the async
      annotation lands (placeholder reserved or equivalent).
- [ ] An expanded details row stays expanded across a Retry press.
- [ ] A commit-summary line renders beside/above Start for a valid
      selection ("N will import · M will match · K will fail").
- [ ] Toast/border overdraw and the Encoding two-click open are
      investigated with notes; fixed if the root cause is cheap, and
      documented as stock/upstream behavior otherwise.
