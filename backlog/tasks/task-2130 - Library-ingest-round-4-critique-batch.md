---
id: TASK-2130
title: >-
  Library ingest round-4 critique batch (options trust, error experience, session ledger, mechanical minors)
status: Done
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

- [x] Reset to defaults returns every control in the panel — Selects,
      checkboxes, AND text Inputs — to defaults, including the top-level
      generic fields and any persisted option values.
- [x] The panel-header receipt always matches the actual field values:
      editing a text Input updates the receipt the same way editing a
      Select does.
- [x] An invalid option value (non-numeric chunk size/overlap) shows an
      inline text message (not color-only) and gates Start with an honest
      gate line, the same way a bad path does.
- [x] Failure details are never a verbatim repeat of the summary: the
      expanded row carries the underlying error (and names the missing
      dependency when that is the category); advisory copy is
      per-category.
- [x] The unsupported-files line says what IS supported (or why the file
      is not), without requiring the user to clear the path.
- [x] The armed clear-finished confirm is scrolled into view on first
      press.
- [x] A confirmed Clear finished does not erase the Recent-ingests
      ledger; failure records survive there.
- [x] The queue empty-state copy after a session with activity does not
      claim "No ingest jobs yet."
- [x] Disabled Start-ingest label is readable (≥3:1 contrast against its
      background).
- [x] Browse opens at the typed path's directory when one is present and
      valid.
- [x] The queue tally shows in-flight work during a batch (not just the
      done count).
- [x] The duplicate-forecast area does not shift layout when the async
      annotation lands (finding: the annotation is computed synchronously
      INSIDE the pre-flight worker before apply — there is no async
      landing; the observed late line was a re-triggered second pre-flight
      generation applying, inherent to re-analysis. The related real bug —
      the 20-candidate cap presenting as the total — is fixed with
      "at least N" copy).
- [x] An expanded details row stays expanded across a Retry press
      (finding: the expanded set already retains the job id through retry;
      the observed collapse is the inherent no-error interim while the job
      re-runs — the row re-expands when it re-fails).
- [x] A commit-summary line renders beside/above Start for a valid
      selection ("N will import · M will match · K will fail").
- [x] Toast/border overdraw and the Encoding two-click open are
      investigated with notes: toasts lifted two rows via the
      task-1562-proven Toast-margin lever (rack margins are resisted);
      Encoding "two-click" is upstream Select anatomy — only the value row
      posts Toggle, the field's border rows focus without toggling, so a
      value-row click opens first-click (documented, not app-fixable
      cheaply).

## Implementation Plan (the how)

Owner priority: options trust first, then everything. Four commit
clusters: (1) shared validator + gate + inline error Statics + in-place
receipt + honest Reset; (2) never-verbatim details + exception chain +
named dependencies + supported-formats guidance; (3) durable Recent
ledger + honest empty copy + scroll-visible confirm; (4) mechanical
minors (commit summary, capped forecast, browse fragment, contrast,
toast clearance) + pins.

## Implementation Notes

**Cluster 1 (options trust).** `validate_ingest_option_value` /
`collect_ingest_option_errors` in library_ingest_state are the single
source for the gate AND the canvas's inline per-field error Statics
(display-managed; text, never color-only). Text/number edits update the
receipt title, the error line, and the gate in place (recompose stays
Select/checkbox-only for cursor survival). Reset resets the generic
mirror fields (the state builder re-injects them, which is how Inputs
survived two presses) and wipes the persisted section; uses the
context-preserving recompose.

**Cluster 2 (error experience).** The parse worker captures up to three
distinct `__cause__`/`__context__` messages; the expansion renders
Category (+exception type), a Details line ONLY when the structured
message differs from the job's own error, Underlying lines, and a
retry advisory that names a missing dependency
(`_missing_dependency_from`) or gives honest transient/corrupt guidance.
Blocked-selection unsupported line appends `SUPPORTED_FORMATS_COPY`.

**Cluster 3 (ledger).** Screen snapshots terminal jobs into
`_library_ingest_recent_ledger` before `registry.clear_finished()`;
builder merges ledger into `recent_jobs` (dedup by id, cap 10);
`queue_empty_line` says "Queue is empty." after activity; the armed
confirm is `scroll_visible()`d (querying the post-recompose button).

**Cluster 4 (minors).** `commit_summary_line` beside Start; duplicate
forecast says "at least N" when the 20-candidate cap was hit (an
80-duplicate folder read "20 files appear…" — the cap presented as
truth, found via round-4 evidence); Browse honors the typed fragment's
directory; disabled Start = 3.15:1 measured live (the TASK-1801 stack
lesson applied: neutralize `opacity: 50%` before any color matters;
`$text NN%` alpha blends toward BLACK, not the surface); toasts lifted
2 rows via `Toast { margin-bottom: 2 }`.

**Qodo round (7 findings, fixed in `94f5bceef`; 1 declined).** Real bug:
option errors gated Start from HIDDEN groups and disabled fields (stale
persisted values with nothing visible to fix) — collect now scopes to
rendered groups and skips gated-off/dep-missing fields. Real mismatch:
UI bounds now mirror `clamp_chunk_size` [100, 5000] so the gate never
blesses what submit would rewrite. Reliability: visited-identity guard
on the exception-chain walk (cyclic `__cause__` could spin a worker).
Compliance: typed browse path through `validate_path_simple`; named
caps; Args/Returns docstrings. Declined: delegating to
`input_validation.validate_number_range` (float-based, boolean — '1.5'
would pass and no message could be built).

**Verification.** 241 core + 114 shell-subset tests green; 29,327
collect. Live on a fresh isolated profile: inline "Chunk size must be a
whole number." renders while typing, receipt shows the actual value,
gate line "Fix the highlighted options to start: …" with Start disabled;
disabled-contrast measured via ANSI. Batch-tally live report pinned as a
sampling artifact (the counts line already names queued/parsing work).
