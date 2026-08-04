---
id: TASK-2231
title: >-
  Library ingest round-7 P2s (forecast/receipt vocabulary reconciliation, Retry feedback)
status: In Progress
assignee: []
created_date: '2026-08-04 17:00'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description (the why)

Round-7 critique, the two P2s. Theme: the pre-commit surface makes
promises the post-commit surface won't honour.

1. **Vocabulary doesn't reconcile.** Pre-flight promises
   `1 will import · 1 will match · 2 will skip`; the receipt says
   `2 done · 2 skipped` — "match" silently folds into "done", and the
   two `✓ done` rows are byte-identical in glyph and colour, so only the
   sub-line distinguishes an import from a dedup match. The user cannot
   audit the promise against the outcome, which cancels the value of
   having made it. Related: the all-match consent line renders on a
   selection where only SOME files are matches.
2. **Retry is indistinguishable from a dead button.** Three clicks on a
   failed row, polled at 0.35s for 3s each: no spinner, no attempt
   counter, no timestamp change, no toast. Zero visible change at the
   flow's highest-anxiety moment.

## Acceptance Criteria (the what)

- [ ] The receipt uses the forecast's vocabulary: matched outcomes are
      reported as "matched", distinct from "imported", in the tally, the
      group header, and the completion toast.
- [ ] A dedup-matched row is distinguishable from a fresh import at a
      glance (its own glyph), not only by its sub-line.
- [ ] The all-match consent line only claims "everything" when every
      importable file in the selection is a predicted match.
- [ ] Pressing Retry produces immediate visible feedback (the row shows
      it is re-attempting) and the resulting row carries an attempt
      count, so a repeat failure is visibly a NEW attempt.
