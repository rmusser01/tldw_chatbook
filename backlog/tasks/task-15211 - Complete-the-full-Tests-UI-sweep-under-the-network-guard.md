---
id: TASK-15211
title: >-
  Complete the full Tests/UI sweep under the network guard
status: Done
assignee: []
created_date: '2026-08-11 07:00'
labels:
  - tests
  - test-infrastructure
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Debt acknowledged by task-15111 rather than papered over. That task blocked test network I/O by default and verified the guard against every module its socket shim had proven was reaching the network, plus the twelve modules that legitimately stand up in-process loopback servers (`903 passed`). What it could **not** finish was a full `Tests/UI` sweep under the guard: the machine was carrying four or more concurrent pytest sessions from other agents and neither half of the split run passed roughly 6%.

So the guard is verified where it was known to matter, and unverified across the rest of `Tests/UI`. The specific risk is not a false block — it is a module that stands up a fixture server, lacks `@pytest.mark.allow_network`, and therefore **hangs until the 300s timeout** (client blocked while the server thread sits in `accept()`) instead of failing fast. Under this repo's `timeout_method = thread` a hang kills the whole process, which is exactly the failure mode task-14912 was created to eliminate.

Run it on a quiet machine, in disjoint halves if needed, and record READ counts for each.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The full `Tests/UI` suite completes under the network guard with READ pass counts (disjoint halves acceptable, stated as such)
- [x] #2 Any module that hangs or blocks is fixed at its source — marked `allow_network` with a reason if it legitimately needs a loopback server, or stubbed if it does not
- [x] #3 A fixture-server module missing the marker fails fast rather than hanging, so this cannot reintroduce the run-killing hang class
<!-- AC:END -->

## Implementation Plan

1. Run all 503 Tests/UI modules in 16 checkpointed, resumable chunks against a frozen worktree
2. Fix guard-class catches immediately on separate branches; leave the frozen tree untouched
3. Attribute every failure row to a cluster; file the inventory

## Implementation Notes

**Complete: 10,811 passed / 117 failed rows across 16/16 chunks** — the first
end-to-end run of the whole suite. Full inventory:
`Docs/Design/2026-08-13-tests-ui-sweep-inventory.md`.

**The egress answer**: ONE distinct source in the entire suite —
`llm_screen._probe_local_server` -> 127.0.0.1:11434 at teardown of any test
mounting the Lab/LLM screen. Fixed mid-sweep (PR #1596, worker lifetime + the
task-15111 harness-stub pattern). The "tests POST real inference" class from
task-15111 did not reappear. One possible sibling flagged for follow-up (the
settings provider-Test teardown pair, task-15791).

**Three catches fixed and merged during the run**: #1591 (four NEW unbounded
background waits, the 14912 hang class, caught by the guard on chunk 0);
#1596 (the probe — which also cured the sweep's own chunk-5 FAILURE-TO-EXIT:
pytest printed its summary then sat forever, two event loops parked in kevent;
re-run on post-fix dev exits clean with zero egress. That failure mode is the
best explanation for all three earlier monolithic sweep attempts dying);
#1603 (35 task files whose lowercase `id:` frontmatter the maturity harness
counts as no id).

**Why this attempt finished when three before it did not**: checkpointed
chunks with per-chunk logs and resume-by-skipping; a hung chunk diagnosed
live (zero CPU accrual + native thread sample), killed, its already-printed
summary still recorded; the worktree frozen for the sweep's whole life with
fixes shipped from a second worktree. The run survived one environment
process-kill and one TCC lockout.

**Inventory filed as** task-15790 (file-notes arcs, ~35 tests, StopIteration
pair triaged first) and task-15791 (console/destination drift, ~38 tests,
incl. the probe-sibling check). Frozen-tree residue of #1554/#1591/#1596 is
recorded as such, not re-filed. Negative result recorded on task-15741: the
blank-note ConflictError did not reproduce anywhere in the full sweep.
