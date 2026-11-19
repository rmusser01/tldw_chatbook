---
id: TASK-19869
title: >-
  Chatbook import non-happy paths understate what was written
status: Done
assignee:
  - '@t1-chatbook'
created_date: '2026-08-22'
labels:
  - ux
  - honesty
  - chatbooks
priority: medium
dependencies:
  - TASK-279
  - TASK-19734
---

## Description

Source: surfaced by the reviewer of **TASK-19734**, which corrected the
chatbook import wizard's *successful* paths. These are the two remaining paths
where the wizard still tells the user less than it knows. Re-verified at
`3605bd52d`.

**1. A fatal error hides a completed write.**

When the import worker raises, the handler at
`UI/Wizards/ChatbookImportWizard.py:1116-1119` calls `_show_error` (`:1203`),
which updates only the finalize row and retitles the panel "❌ Import Failed".
The per-type rows that had already been set to `⟳ Preparing notes…` /
`⟳ Preparing characters…` / `⟳ Preparing media…` (`:1065-1073`) are never
updated again and freeze there, and **no stats panel renders at all**.

Meanwhile the items the run had already written are permanently in the
database. There is no rollback (TASK-279). So the screen reads as "nothing
happened, it failed while preparing", and the truth is "an unknown number of
items were committed and will still be there when you retry". A user who then
retries the import gets duplicates or conflicts they were given no reason to
expect.

**2. Server mode reports a 4-item chatbook as empty.**

At `:1017-1028` the server path builds a fresh `ImportStatus` from the job
payload:

- `successful_items` defaults to `0` when the key is absent
- `failed_items` defaults to `0`
- `total_items` is their sum
- `skipped_items` is **hard-set to 0**, regardless of what the server did

A payload that omits `successful_items` therefore yields `attempted_items == 0`
with `left_out_items == 0`, which `ImportStatus.outcome`
(`chatbook_importer.py:293-309`) resolves to `IMPORT_OUTCOME_EMPTY`, and the
panel says:

> ⊘ Nothing was imported — this chatbook contained no items.

…about a chatbook whose four-item manifest the wizard is holding in memory at
that moment. The hard-set `skipped_items = 0` is a second, independent false
claim: it is not something the server reported, it is a value the client
invented.

Both belong to the family TASK-19550, TASK-19734 and TASK-19861 are in — *the
app asserts an outcome it did not produce* — and both are on the paths where a
user most needs the truth, because both are paths where they are about to
decide whether to retry.

## Acceptance Criteria

- [x] When the import fails partway through, the completion surface states that
      items may already have been written and cannot be rolled back
- [x] Per-type rows do not remain frozen on a "preparing" state after a fatal
      error — each row says what is known about that type, including "unknown"
- [x] The wizard never reports "this chatbook contained no items" for a
      chatbook whose manifest it holds and which lists items
- [x] Server mode does not invent a `skipped_items` value the server did not
      report; an unreported count is presented as unknown, not as zero
- [x] A server payload missing `successful_items` produces a message that
      distinguishes "the server told us nothing" from "there was nothing to
      import"
- [x] Tests drive both paths — a fatal error raised after at least one item is
      committed, and a server payload with no counts — and assert the rendered
      text; both are mutation-checked against the current behaviour
- [x] The one-vocabulary rule TASK-19734 established (empty = a claim about the
      file, excluded = a claim about the run) is extended to cover these two
      cases rather than bypassed

## Implementation Plan

1. Rewrite `_show_error` into a real failure surface: per-type rows from the
   import's recorded results where they exist, "❓ Unknown — the import
   stopped before X were reported" for types the manifest lists but the run
   never reported on, the opted-out-media wording where that applies, and
   "— No X in this chatbook" only when the manifest has none; render the
   stats panel with the counts recorded before the failure; banner states
   what was written, what is unknown, and that the import cannot be rolled
   back.
2. Server branch: record only counts the payload actually carries
   (`successful_items`/`failed_items`/`skipped_items`); track
   `_server_counts_unknown` when none were; include skipped in the total.
3. `_show_completion`: a dedicated unknown-counts outcome (title, banner,
   `outcome-unknown` state, "?" stats) when the server reported no counts;
   and a manifest guard on the EMPTY outcome so "this chatbook contained no
   items" cannot render for a chatbook whose manifest lists items (the
   server-zeros-for-a-listed-chatbook edge).
4. Born-red tests for both paths with mutation checks; run the wizard
   suites for regressions.

## Implementation Notes

- Approach: both defects fixed on their shared surface (the progress step),
  extending the task-19734 result-driven machinery rather than adding a
  parallel vocabulary.
- Fatal-error path (`_show_error` + `_error_surface_row` +
  `_fatal_error_banner`): per-type rows report recorded results first;
  types the manifest lists but the run never reported on read
  "❓ Unknown — the import stopped before X were reported"; media the user
  opted out of keeps the excluded wording; the stats panel renders the
  counts recorded before the failure (Total = attempted at that point);
  the banner names written items ("N item(s) were already written and stay
  written"), mid-flight items ("may also have landed"), or — when nothing
  at all was recorded — "it stopped before reporting any result", always
  with "cannot be rolled back" and a retry warning. Applies to both the
  exception path and the `success=False` return path.
- Server path: only payload-carried counts are recorded; a payload with no
  counts at all yields the "⚠️ Import Finished — Item Counts Unknown"
  outcome (custom title, banner naming the manifest's item count, "?" in
  every stat slot, `outcome-unknown` styling added to
  `_wizards.tcss` and the bundle rebuilt); a payload reporting
  `skipped_items` now displays it (Total includes skips) instead of the
  invented zero; and a payload of explicit zeros for a manifest-listing
  chatbook renders "Nothing was imported — the server reported no items
  imported. This chatbook contains N item(s)." rather than the file claim
  "contained no items" (the EMPTY outcome is manifest-guarded).
- Tests: `Tests/Chatbooks/test_chatbook_import_wizard_failure_honesty.py`
  (9 tests, green, `bootstrap_profile` marker). Mutation checks performed:
  restoring the base server-count handling (default-0 + hard-zero skipped)
  and the base `_show_error` (finalize row + title only) turns 6 of the 9
  red; the fixed code passes all 9. No regressions across the wizard
  suites (validation, backup honesty, conflict honesty, embeddings
  honesty, failure honesty, design-token governance: 45 passed).
- ADR required: no -- UI honesty fix on an existing surface; no schema,
  sync, boundary or contract change (the server payload contract is read
  more defensively, not changed).
- Modified files: `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`,
  `tldw_chatbook/css/features/_wizards.tcss`,
  `tldw_chatbook/css/tldw_cli_modular.tcss` (rebuilt),
  `Tests/Chatbooks/test_chatbook_import_wizard_failure_honesty.py` (new).

## Notes

Filed as one task because both are the same surface's remaining non-happy-path
claims and a fix for either will touch the same completion-panel code. The
rollback gap itself is TASK-279 and is not in scope here — what is in scope is
that the UI currently conceals it at exactly the moment it matters.
