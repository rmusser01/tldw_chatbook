---
id: TASK-1453
title: >-
  Adopt pytest-xdist: per-worker config sandboxes and parallel local runs
status: Done
assignee: []
created_date: '2026-07-30 09:05'
labels:
  - testing
  - performance
priority: high
dependencies: [task-1450, task-1451, task-1452]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The suite (~11,600 tests, 23,793 collected items, 1+ hour) runs fully serially: pytest-xdist is not installed, not declared in any extra, and no CI job or config passes `-n`. The dominant cost (Textual app mounts) is CPU-bound and parallelizes near-linearly. Compatibility prerequisites landed separately: the RAG_Search session autouse model fixture is gated (task-1451), Hypothesis profiles no longer depend on collection order (task-1452). Remaining blocker fixed here: all xdist workers inherited `TLDW_TEST_CONFIG_ROOT` from the controller and would share one sandbox (one HOME/XDG/config.toml — a write-race). Real test servers all bind port 0 (verified — no fixed-port binds in Tests/), so no port work is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] `pytest-xdist` installable via both `.[dev]` and `requirements-test.txt` (plus `pytest-mock`/`pytest-cov`, which the docs and tests already assume)
- [x] Each xdist worker gets its own config sandbox subtree (own HOME/XDG/TLDW_CONFIG_PATH); the controller still owns and removes the root
- [x] Works for both rootdirs: repo root and `pytest Tests/UI`
- [x] A full `-n auto --dist loadscope` run completes; its junit outcome set diffed against the serial baseline shows no unexplained regressions (new failures get fix-or-`xfail(strict=False)`+task, never silent skips)
- [x] Serial behavior is unchanged when xdist is not used (no `-n` in addopts — opt-in via CLI/CI)

## Implementation Plan

1. Add deps to `[dev]` extra + `requirements-test.txt`
2. Per-worker sandbox suffix (`PYTEST_XDIST_WORKER`) in both conftest bootstrap blocks, guarded against double-suffixing (root conftest republishes the suffixed path; Tests/UI conftest and subprocess children must not suffix again)
3. Verify: fixed-port grep (clean), parallel full run + junit diff vs baseline
4. Recommended invocation: `pytest -n auto --dist loadscope --max-worker-restart=3` (loadscope preserves module-scoped fixtures and intra-module ordering; worker restarts bound the blast radius of pytest-timeout's thread-method process kills)

## Implementation Notes

Sandbox suffixing: controller creates the root (unsuffixed, owner) → workers see
`PYTEST_XDIST_WORKER` and nest `<root>/<worker>` with their own home/data/config
dirs; the name-guard makes re-entrant conftest loads idempotent. `-n` is
deliberately NOT in addopts yet — serial stays the default until the parallel
outcome diff has soaked; CI adoption is task-1465.
Modified: `pyproject.toml`, `requirements-test.txt`, `Tests/conftest.py`,
`Tests/UI/conftest.py`.
