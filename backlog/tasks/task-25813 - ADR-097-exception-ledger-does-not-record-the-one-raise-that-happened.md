---
id: TASK-25813
title: ADR-097 exception ledger does not record the one raise that happened
status: Done
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

- [x] The ledger carries a row for the 970 → 972 raise with its date,
      guard, constant, named cause (`tls_trust`), and the PR that
      carried it (#2223)
- [x] ADR-097's context table is reconciled with the current constants, or
      states explicitly that it is a historical snapshot
- [x] Check whether any of the other three constants have moved since the
      ADR was written, and add rows for those too — this was found by
      comparing one constant against the table, so the others are unaudited
- [x] Consider a cheap guard: a test that fails when a ratchet constant
      differs from the value recorded in the ADR without a matching ledger
      row. Without it this recurs, which is the exact pattern ADR-097 was
      written about

## Notes

Found incidentally while taking the ratchet baseline for the 2026-08-30
holistic review. Not a runtime performance defect — filed low priority as
process hygiene, but filed rather than dropped because a ledger nobody
maintains is worse than no ledger.


## Implementation Notes

The ledger now carries the 970 → 972 raise, transcribed from the owner's own
commit (`6fac5dbf95`, PR #2223) rather than re-decided here — the decision
was deliberate and cited this ADR; only the required row was missing.

**Audited the other four constants at the same time**, which the original
filing flagged as unknown: `MAX_TLDW_MODULE_COUNT` 660,
`MAX_BOOT_PARSED_CSS_BYTES` 860,000, `MAX_PASS_ADDED_MODULES` 500,
`MAX_PASS_ADDED_LOC` 380,000, `MAX_SINGLE_ROUTE_ADDED_LOC` 145,000 — all
unchanged from the ADR's table. Exactly one raise had gone unrecorded.

The ADR's context table is explicitly a historical snapshot (its columns are
dated review points), so it is now labelled as such and points at the ledger
as the authoritative record, rather than being edited in place to track
current values.

**The guard: considered, and NOT built — an owner call.** A test comparing
each constant against a recorded value would work, but it needs a new
artifact to compare against: the ADR's context table is historical by
design, so the guard would require a separate "current limits" record that
someone has to keep in sync. That trades one drift problem for another
unless the record is generated rather than hand-written. Recommendation:
have the ratchet helper emit the current limits into
`Tests/Performance/boot_budget_snapshots/` (where the per-guard snapshots
already live) and assert the ADR ledger has a row for any difference. That
is a real design decision about process tooling, not a cleanup, so it is
left for the owner rather than added unasked.
