---
id: task-25813
title: ADR-097 exception ledger does not record the one raise that happened
status: To Do
assignee: []
labels:
  - governance
  - performance
created_date: '2026-08-30'
priority: low
---

## Description (the why)

ADR-097 makes the four boot budgets ratchets and requires that any raise of
a constant be recorded as **a row in its exception ledger, in the same
commit**, on the grounds that this is "loud and auditable by construction:
a raised constant with no ledger row is a defect".

`MAX_TLDW_MODULES_AT_UI_READY` currently reads **972**. ADR-097's own table
records it as **970**. It was raised deliberately by the owner on
2026-08-29 (`6fac5dbf95`, *"raise ui-ready census ratchet 970->972 for
tls_trust (PR #2223, ADR-097 deliberate refresh)"*) — so the decision was
made and the cause named. But the ledger still reads
*"(none granted yet)"*.

The process worked; the audit trail did not. This matters because the
ledger's whole value is that a reader can trust it: one silently missing
row makes it evidence of nothing, and the next reader comparing the ADR
table (970) against the code (972) has no way to tell a sanctioned raise
from an unsanctioned one.

## Acceptance Criteria (the what)

- [ ] The ledger carries a row for the 970 → 972 raise with its date,
      guard, constant, named cause (`tls_trust`), and the PR that
      carried it (#2223)
- [ ] ADR-097's context table is reconciled with the current constants, or
      states explicitly that it is a historical snapshot
- [ ] Check whether any of the other three constants have moved since the
      ADR was written, and add rows for those too — this was found by
      comparing one constant against the table, so the others are unaudited
- [ ] Consider a cheap guard: a test that fails when a ratchet constant
      differs from the value recorded in the ADR without a matching ledger
      row. Without it this recurs, which is the exact pattern ADR-097 was
      written about

## Notes

Found incidentally while taking the ratchet baseline for the 2026-08-30
holistic review. Not a runtime performance defect — filed low priority as
process hygiene, but filed rather than dropped because a ledger nobody
maintains is worse than no ledger.
