---
id: TASK-1465
title: >-
  CI rework: parallel directory shards replace the 27-file -m unit matrix; dedupe python-app.yml (owner sign-off)
status: To Do
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - ci
priority: high
dependencies: [task-1453]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CI's `-m unit` job matrix (8 OS/Python runs) selects 27 of 900 test files while installing torch/chromadb/playwright each time; `-m integration` selects 40; ~590 files are exercised by no PR-triggered job in test.yml — only by `python-app.yml`'s duplicate, serial, unbounded `pytest ./Tests/` on main. No CI job uses parallelism. Restructure onto xdist directory shards with a nightly deep job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] PR gate: parallel (`-n auto --dist loadscope`) jobs covering ALL of Tests/ (e.g. core = Tests minus UI, ui = Tests/UI); marker-based unit/integration jobs removed or reduced to a deliberate, documented subset
- [ ] `python-app.yml` deleted or collapsed after confirming branch-protection required-check wiring with the owner
- [ ] Nightly/dispatch job: serial full run + `HYPOTHESIS_PROFILE=thorough` + `--run-slow` + coverage; OS/Python matrix breadth moved here per owner decision
- [ ] PR-gate wall time before/after recorded
