---
id: TASK-1457
title: >-
  Unify pytest config: delete Tests/UI/pytest.ini so all invocations share one rootdir, asyncio mode, and timeout
status: Done
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

- [x] `Tests/UI/pytest.ini` is gone; `pytest Tests`, `pytest Tests/UI`, and CI invocations all resolve the same rootdir and settings
- [x] `asyncio_mode = "auto"` applies suite-wide; the previously-dormant async tests either pass or are individually quarantined via `xfail(strict=False)` + a follow-up task each
- [x] `--strict-markers` is on at root with every in-use marker registered (collect-only run is clean)
- [x] The `optional_deps`/`optional` marker mismatch is resolved (conftest gate points at the real marker; Tests/README.md matches)
- [x] `--collect-only` count delta vs baseline is itemized in the PR; junit outcome diff shows no unexplained regressions

## Implementation Plan

1. Census dormant async tests and anyio markers (found: only 2 files / 3 tests dormant; zero anyio)
2. Absorb the ini into pyproject (asyncio_mode=auto, --strict-markers, -ra, marker registry incl. snapshot/notes/requires_display/smoke); delete Tests/UI/pytest.ini
3. Point the --run-optional gate at the real `optional` marker; drop the unused `optional_deps` registration
4. Strict-markers collect-only sweep; register what it surfaces
5. Run activated async tests + formerly-ini-rooted invocations; fix or quarantine per protocol

## Implementation Notes

Single config in pyproject now governs every invocation (verified: `pytest Tests/UI`
resolves rootdir to the repo root; single-UI-file runs load the root conftest and
its fixtures — 163 passed on formerly-ini-rooted files). Strict markers immediately
caught `benchmark` as a silent no-op in three Transcription files (registered as
labeling-only). Of the audit's feared dormant async tests, only 3 remained; 2 pass,
and `test_console_sync_records_worker_lifecycle` — rotted while dormant, its
12-stub ChatScreen skeleton now 13 stages behind the production method — is the
suite's first `xfail(strict=False)` quarantine (rewrite: task-1469). The ini's
blanket DeprecationWarning ignores were deliberately not carried over. Collection:
24,332 / 0 errors. Modified: `pyproject.toml`, `Tests/conftest.py`,
`Tests/README.md`, `Tests/UI/test_ui_responsiveness.py`; deleted
`Tests/UI/pytest.ini`; filed task-1469.
