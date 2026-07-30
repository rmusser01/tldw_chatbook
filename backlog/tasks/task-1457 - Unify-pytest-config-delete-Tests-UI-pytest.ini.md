---
id: TASK-1457
title: >-
  Unify pytest config: delete Tests/UI/pytest.ini so all invocations share one rootdir, asyncio mode, and timeout
status: To Do
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - infra
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/pytest.ini` is a second pytest config that wins whenever pytest is invoked as `pytest Tests/UI` (CI does): rootdir flips, `asyncio_mode=auto` and `--strict-markers` switch on, the pyproject `timeout=300` switches OFF, and `Tests/conftest.py`'s autouse isolation fixtures stop loading. Consequence found in the audit: repo-root runs use pytest-asyncio strict mode, and ~46 Tests/UI files rely on auto mode — their async tests are silently NOT executed in a full run from the repo root. One config must govern every invocation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] `Tests/UI/pytest.ini` is gone; `pytest Tests`, `pytest Tests/UI`, and CI invocations all resolve the same rootdir and settings
- [ ] `asyncio_mode = "auto"` applies suite-wide; the previously-dormant async tests either pass or are individually quarantined via `xfail(strict=False)` + a follow-up task each
- [ ] `--strict-markers` is on at root with every in-use marker registered (collect-only run is clean)
- [ ] The `optional_deps`/`optional` marker mismatch is resolved (conftest gate points at the real marker; Tests/README.md matches)
- [ ] `--collect-only` count delta vs baseline is itemized in the PR; junit outcome diff shows no unexplained regressions
