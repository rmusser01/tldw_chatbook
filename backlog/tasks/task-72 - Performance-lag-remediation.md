---
id: TASK-72
title: Performance lag remediation
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-28 23:58
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce startup, navigation, and console transcript lag by removing eager work and adding performance guardrails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Startup and navigation measurements are captured before and after remediation
- [x] #2 Embeddings and heavy optional dependencies are not checked during plain app import
- [x] #3 Console transcript updates avoid full transcript remounts during streaming
- [x] #4 Nonessential startup services do not block UI readiness
- [x] #5 Focused UI and performance tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add subprocess startup/import guards for optional dependency checks and app import graph behavior.
2. Remove import-time embeddings dependency checks while preserving explicit eager dependency mode.
3. Lazy-load screen route resolution and legacy feature windows that are not needed for plain app import.
4. Add safe repeat-navigation reuse for allowlisted destination screens with context-change invalidation.
5. Incrementalize native Console transcript reconciliation and skip unchanged transcript refresh polls.
6. Defer nonessential startup services and startup cleanup work so UI readiness is not blocked.
7. Gate high-volume metrics/logging overhead behind explicit enablement.
8. Capture before/after measurement deltas and run focused UI/performance verification before closing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the approved performance-lag remediation in the isolated worktree. Added subprocess startup/import guards and removed the import-time embeddings/RAG dependency check while preserving explicit `TLDW_EAGER_DEPENDENCY_CHECK=true` behavior. Added lazy screen route resolution, lazy `UI.Screens` exports, and lazy splash effect discovery so guarded legacy feature windows and splash effect modules are not loaded during plain app import.

Added allowlisted destination screen caching with invalidation on runtime/workspace context changes, and fixed Notes navigation so initial scope refresh has one owner per entry. Incrementalized `ConsoleTranscript` row reconciliation, added row signatures/build counters for tests, and skipped unchanged native transcript refreshes in `ChatScreen`. Deferred TTS/STTS initialization, DB-size polling, token-count timers, and startup media cleanup until after UI readiness or first use. Gated normal metric log emission behind `TLDW_METRICS_LOGGING` and lowered high-volume select-change logging.

Captured before/after measurements in `Docs/superpowers/qa/performance/2026-05-28-lag-remediation-closeout.md`. Baseline was about 4.5s import, 6.5s to `_ui_ready`, and 0.26s-0.61s route switches. After remediation, isolated probes measured 4.365s app import, 0.492s to `_ui_ready`, route switches of 0.274s-0.547s, and 0 `METRIC` log lines during import/startup probes. Residual risks are documented in the closeout: plain app import remains broad, splash card definitions still stay in the app import graph, large user DBs were not replayed, and transcript wall-clock budgets remain structural rather than timed.

Modified files include `tldw_chatbook/app.py`, `tldw_chatbook/UI/Navigation/screen_registry.py`, `tldw_chatbook/UI/Screens/__init__.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Screens/notes_screen.py`, `tldw_chatbook/Widgets/Console/console_transcript.py`, `tldw_chatbook/Utils/optional_deps.py`, `tldw_chatbook/Utils/Splash_Screens/__init__.py`, `tldw_chatbook/Widgets/splash_screen.py`, `tldw_chatbook/Metrics/metrics_logger.py`, focused UI/performance/unit tests, and the performance plan/closeout docs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the critical startup path and repeat-interaction lag by lazy-loading heavy modules, deferring nonessential services until after readiness or first use, reusing cacheable destination screens safely, removing duplicate Notes refresh ownership, and making native Console transcript refreshes keyed/incremental instead of full-list remounts. Focused verification passed for the startup, route navigation, destination smoke, metrics gating, and console transcript suites; static checks passed with targeted `py_compile` and `git diff --check`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria are checked off.
- [x] #2 Implementation plan is added after the task moves to In Progress.
- [x] #3 Focused automated tests cover changed behavior.
- [x] #4 Static analysis / formatting checks are run or documented if not applicable.
- [x] #5 Relevant docs or QA artifacts are updated.
- [x] #6 Implementation notes summarize approach, modified files, measurement deltas, and residual risks.
- [x] #7 Self-review completed.
- [x] #8 Task status is set to Done via Backlog CLI only after all DoD items are complete.
<!-- DOD:END -->
