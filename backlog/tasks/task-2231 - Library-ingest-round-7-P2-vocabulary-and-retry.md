---
id: TASK-2231
title: >-
  Library ingest round-7 P2s (forecast/receipt vocabulary reconciliation, Retry feedback)
status: Done
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

- [x] The receipt uses the forecast's vocabulary: matched outcomes are
      reported as "matched", distinct from "imported", in the tally, the
      group header, and the completion toast.
- [x] A dedup-matched row is distinguishable from a fresh import at a
      glance (its own glyph), not only by its sub-line.
- [x] The all-match consent line only claims "everything" when every
      importable file in the selection is a predicted match.
- [x] Pressing Retry produces immediate visible feedback (the row shows
      it is re-attempting) and the resulting row carries an attempt
      count, so a repeat failure is visibly a NEW attempt.

## Implementation Notes

- **Vocabulary:** a dedup match is recognised by the writer's progress
  marker (the same predicate the tally uses) and now renders
  `≡ matched · <name>` — its own glyph AND word, so an import and a
  match are distinguishable at a glance rather than only by the
  sub-line. `_batch_outcome_parts` splits matched out of done, so group
  headers and the latest-run line read "1 done · 1 matched"; the
  completion toast says "N matched" instead of "N already in Library".
  The forecast's three words (import/match/skip) are now the receipt's
  three words.
- **Consent scope:** the "Everything here…" line additionally requires
  zero predicted skips, so it can no longer claim "everything" for a
  selection that also contains unsupported files.
- **Retry feedback:** requeue always created a new QUEUED job with an
  incremented count, but the in-flight rows never showed it — so a
  retry was visually identical to nothing happening. Queued/parsing/
  writing rows now carry the attempt marker, and the suffix reads
  "· attempt N" (N = retry_count + 1) on every row state, which reads
  as progress and is unambiguous mid-flight.

**Verification.** 300 core + 56 shell-subset green; 29,815 collect
clean. Live: forecast "1 will match" + consent line → `≡ matched ·
copy_of_report.txt` row → "Latest run: 1 matched".

**Qodo round (fixed in `76dc40154`):** two internal inconsistencies from
this PR's own changes. (1) The attempt suffix was appended BEFORE
`detected_type`, so only rows WITH a type mis-ordered
("… · attempt 2 · pdf") — my test used typeless jobs and missed it; the
marker is now the trailing element in every state and the test carries a
type. (2) `_queue_counts_line` bucketed purely by state, so the queue
line said "2 done" while a group header below said "1 done · 1 matched";
the tally now applies the same dedup predicate as the headers and rows.
**Lesson: when a new distinction is introduced, EVERY surface that
aggregates the old bucket has to learn it — the row, the group header,
the tally, and the toast were four separate places.**
