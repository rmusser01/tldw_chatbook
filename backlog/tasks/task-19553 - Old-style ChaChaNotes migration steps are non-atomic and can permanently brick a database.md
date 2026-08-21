---
id: TASK-19553
title: >-
  Old-style ChaChaNotes migration steps are non-atomic and can permanently brick
  a database
status: To Do
assignee: []
created_date: '2026-08-21 20:03'
labels:
  - db
  - migrations
  - data-loss
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F1**, and the lane's only finding rated SEVERE. **CONFIRMED BY LIVE
EXPERIMENT**, not by reading.

The old-style migration steps in `tldw_chatbook/DB/ChaChaNotes_DB.py` are
**non-atomic and non-idempotent**, so a partial apply leaves the database in a
state that can never be repaired from inside the app.

Mechanism: `executescript` commits the pending transaction and then autocommits
each statement individually. **22 of 38 steps use `executescript`, and 22 of 38
have no entry-version guard**, several with bare unguarded DDL — the v12→v13
step is 13 `ALTER`s and 4 `CREATE TRIGGER`s with nothing to make a re-run safe.

The lane's live experiment: it ran the chain on a genuine v11 database with one
of four columns already applied. Two `ALTER`s **stayed committed** while the
schema version stamp **stayed at 11**; "rolled back cleanly" evaluated to
`False`. Every subsequent launch re-raises on the already-applied column,
`CharactersRAGDB.__init__` fails, and conversations, notes and characters all
become unreachable — with **no in-app recovery path**. The control case (a
new-style step using `self.transaction()`) rolled back cleanly.

Reachability is not hypothetical and is not the newest users: databases in the
field sit at v4–v25, i.e. exactly the users who must replay the longest chain
to reach the current `_CURRENT_SCHEMA_VERSION` (42).

**Fix (the lane's, and it fits the owner's durable-over-clever standing
ruling):** port the remaining `executescript` steps to the
`transaction()` + per-statement pattern already used at
`ChaChaNotes_DB.py:5203`. That pattern is in-repo, proven by the control, and
does not require inventing anything.

Note the version-guard machinery itself is *not* broken — the lane verified
that the `executescript`-commits-the-transaction hazard is explicitly handled
where the code re-issues `BEGIN`. The defect is the steps that never adopted
the safe pattern.

## Acceptance Criteria

- [ ] Every migration step in `ChaChaNotes_DB.py` applies atomically: an
      interrupted or failing step leaves the database at its entry version with
      no partially-applied DDL
- [ ] Every step is idempotent or entry-version guarded, so re-running a step
      after an interruption is safe rather than fatal
- [ ] The remaining `executescript` steps are ported to the
      `transaction()` + per-statement pattern at `ChaChaNotes_DB.py:5203`
      (durable, already proven in-repo) rather than a new mechanism
- [ ] A test reproduces the lane's experiment: a genuine older-version database
      with one column of a multi-`ALTER` step pre-applied migrates to current
      **or** rolls back cleanly to its entry version — it never lands
      half-applied with a stale version stamp
- [ ] The test covers the long-chain case (a v4-era database replayed to
      current), since that is the reachable population
- [ ] If any already-bricked shape is recoverable, a recovery path exists that
      does not require the user to hand-edit SQLite
