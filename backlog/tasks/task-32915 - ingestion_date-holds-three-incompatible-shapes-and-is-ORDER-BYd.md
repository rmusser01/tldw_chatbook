---
id: TASK-32915
title: ingestion_date holds three incompatible shapes and is ORDER BY'd
status: To Do
assignee: []
created_date: '2026-09-22 15:30'
labels:
  - tier2-review
  - review-time
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed at **P3** by the tier-2 review as a timestamp-format nit. It is not one — it is a data-contract defect
with existing rows at stake, found while burning down the P3 tail.

`ingestion_date` is a `DATETIME` column that is **`ORDER BY`'d and range-filtered**, and it holds **three
mutually incompatible shapes** written by at least four different ingest paths:

| shape | example |
|---|---|
| date only | `2026-09-22` |
| space-separated | `2026-09-22 10:00:00` |
| ISO-8601 with `Z` | `2026-09-22T10:00:00.123Z` |

Because these are compared as **strings**, and `' '` (0x20) sorts before `'T'` (0x54), and a shorter prefix
sorts before a longer one:

- Two rows ingested at the **same instant** by different paths sort in an order determined by *which path
  wrote them*, not by time.
- A range filter using one shape silently excludes rows written in another.

This is the same class as the `+00:00` vs `Z` split that TASK-32897 found in the timestamp guard, and the
same consequence TASK-32803.3 names ("cutoffs that skip rows") — but on a different column, with a third
shape, and already-written rows.

## Why it is not a P3

The P3 framing implies "swap the writer and move on". That is exactly the fix that **must not** be applied
alone: existing rows are never rewritten (TASK-32803.5's own AC#2 states this), so converting one writer
makes the column *more* mixed, not less, and correctness then depends on read-side normalization that would
have to cover every comparison site.

The work is: enumerate the writers, enumerate the readers that compare or order on this column, decide
whether to normalize on read or migrate the rows, and only then touch a writer.

Source: tier-2 code review 2026-09-21, S11. Re-rated while implementing TASK-32902.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every writer of `ingestion_date` and every reader that orders or range-filters on it is enumerated
- [ ] #2 A decision is recorded: normalize on read, or migrate existing rows
- [ ] #3 No single writer is converted before that decision, since that widens the drift
- [ ] #4 A test asserts that two rows written at the same instant by different ingest paths compare equal
<!-- AC:END -->
