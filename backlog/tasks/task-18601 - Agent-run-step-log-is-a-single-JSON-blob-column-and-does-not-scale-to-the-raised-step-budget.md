---
id: TASK-18601
title: >-
  Agent run step log is a single JSON blob column and does not scale to the
  raised step budget
status: To Do
assignee: []
created_date: '2026-08-18 20:30'
labels:
  - agents
  - database
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentRunsDB.append_steps` stores a run's entire step log as one JSON string in
the `agent_runs.steps` column: it SELECTs the existing blob, `json.loads` it,
extends the list, and `json.dumps` the whole thing back. `AgentService._persist`
calls it once at end of run, so the write itself is O(n), not O(n^2).

That design was sized for the Console's old 96-step budget. TASK-18600 raised
the shipped step budget to 25000 (owner decision: allow long-running, expensive
sessions). Each step's `result` is capped at 2000 characters by
`agent_runtime.run_agent_loop`, so a worst-case run can now serialize a
tens-of-megabytes JSON blob into one column -- and re-parse all of it every time
the run log is opened, on the UI thread.

Nothing observed failing yet: this is a scaling limit reached by a deliberate
config change, filed at the time the ceiling was raised rather than after a user
hits it. Realistic runs are far smaller (most steps are a few hundred bytes),
which is why TASK-18600 shipped the number as specified instead of lowering it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A run with 25000 recorded steps persists and re-opens without a user-visible stall in the run-log viewer.
- [ ] #2 Reading a run's metadata (status, budget, result) does not require parsing its full step log.
- [ ] #3 The run-log viewer can render a long run without holding every step in memory at once.
- [ ] #4 Existing runs stored in the current blob format remain readable after the change.
<!-- AC:END -->

## Measurement (2026-08-21) — the premise, quantified

`append_steps` (`DB/AgentRuns_DB.py`) reads the ENTIRE step log, JSON-parses
it, extends the list, re-serializes it and rewrites the whole column -- on
every append. That is O(n) per step and O(n^2) per run.

Measured against a real `AgentRunsDB`, one ~200-byte step per append:

    append #    1:    0.05 ms
    append #  100:    0.13 ms
    append #  500:    0.49 ms
    append # 1000:    0.98 ms
    append # 1500:    1.42 ms
    append # 2000:    2.18 ms
    2,000 appends total: 2.09s

Per-append cost grows linearly with log size (44x from the 1st to the
2,000th), confirming the quadratic total. Extrapolating the same curve to the
25,000-step budget AC #1 names gives **~5.4 minutes** of pure database churn
for one run -- and that is write cost alone, before the viewer reads it back.

**Scope.** This is an arc, not a single change: a child `agent_run_steps`
table plus a schema bump (AgentRuns_DB is at v4, and its migration mechanism
is `CREATE TABLE IF NOT EXISTS` on every open), a compatibility read path so
existing blob-format runs stay readable (AC #4), and a viewer change to page
steps rather than hold them all in memory (AC #3). Suggested split:

* **A** -- child table + dual-read (new writes go to rows, reads prefer rows
  and fall back to the blob). Closes AC #1, #2, #4 and is independently
  shippable.
* **B** -- viewer paging over the new table. Closes AC #3.
* **C** -- optional backfill of historical blobs, once A has soaked.

Not started here; the measurement is recorded so the arc can be planned
against a number rather than an adjective.
