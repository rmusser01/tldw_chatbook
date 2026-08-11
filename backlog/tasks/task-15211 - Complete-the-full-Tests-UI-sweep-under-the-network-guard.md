---
id: TASK-15211
title: >-
  Complete the full Tests/UI sweep under the network guard
status: To Do
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
- [ ] #1 The full `Tests/UI` suite completes under the network guard with READ pass counts (disjoint halves acceptable, stated as such)
- [ ] #2 Any module that hangs or blocks is fixed at its source — marked `allow_network` with a reason if it legitimately needs a loopback server, or stubbed if it does not
- [ ] #3 A fixture-server module missing the marker fails fast rather than hanging, so this cannot reintroduce the run-killing hang class
<!-- AC:END -->
