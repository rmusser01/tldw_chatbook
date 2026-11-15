---
id: TASK-40316
title: "test_console_variant_stream.py runs on no PR lane, though it pins that a failed regenerate keeps the prior good answer"
status: Done
assignee: ['@claude']
updated_date: 2026-11-13 02:56
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- console
- lanes
priority: low
references:
- Tests/Chat/test_console_variant_stream.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cred stream). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** `test_regenerate_empty_stream_retains_failed_sibling_and_restores_anchor` (`Tests/Chat/test_console_variant_stream.py:366`, #2164, 473e7c9298) guards that a failed regenerate keeps the prior good answer in provider context, a TASK-571 regression class. Nothing under `.github/` or `scripts/` runs the file; `test.yml` runs on pushes to main only. The file has 14 tests and no profile marker; TASK-33621.22 (2026-10) recorded 4 reds in it, and its state today is unknown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file passes under its plain command at `-n 4`, each red first classified with its cause
- [x] #2 The file runs on a PR lane (never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->

## Implementation Plan

1. Run the file under its plain command at `-n 4` on base 26b2e677dd and classify each red.
2. Fix the cause, then add the file whole to the PR lane that runs its siblings (`recovered-suites` in `derived-artifacts.yml`, which runs `test_console_regenerate_branching.py`; not `pr-fast-lane`), and keep that lane's exact-target contract in step.
3. Time the lane's command on the file three times.

## Implementation Notes

**Reds, classified.** On base the plain command (`-n 4`) gave 4 failed and 15 passed of 19. All four were the same test-harness cause: `RecoveryRequired: raw_source_selection_changed`. Three controller-driven regenerate tests (`..._delegates_and_streams_incrementally`, `..._empty_stream_retains_failed_sibling_and_restores_anchor`, `..._new_sibling_carries_its_own_generation_usage`) raised it from `controller.regenerate_message`, which reads config inside the per-test sandbox. The fourth, `..._stop_mid_stream_leaves_anchor_untouched_new_sibling_stopped`, reached it as `TimeoutError` on `gateway.started`: its background regenerate task died of the same refusal before the provider (the log records the RecoveryRequired read). None was a product defect.

**Fix.** The file was the one controller-driven Console regenerate file without `pytestmark = pytest.mark.bootstrap_profile`. 7970fbcada added that mark to 12 sibling Console files, `test_console_regenerate_branching.py` among them, and missed this one. With the mark, the plain command passes 19/19.

**Lane.** The file is appended whole to the `recovered-suites` job's "Run the recovered suites" step, after the other bootstrap_profile suites. That job is serial, under `--storage-lease-guard --timeout=180 --tb=short`, and its own job, not the fast PR contract. `RECOVERED_SUITES_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py` gains the same row, which that contract requires. `Tests/CI/test_ci_queue_pressure_contract.py` + `Tests/CI/test_derived_artifacts_workflow.py`: 193 passed.

**Timings** of the lane's command on the file (`pytest -p no:xdist Tests/Chat/test_console_variant_stream.py --storage-lease-guard --timeout=180 --tb=short`, xdist off locally as the lane has none), all 19 passed:
- run 1: 10.29 s pytest, 22 s wall, load 62.6
- run 2: 11.30 s pytest, 25 s wall, load 62.5
- run 3: 13.43 s pytest, 28 s wall, load 64.5

Files: `Tests/Chat/test_console_variant_stream.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
