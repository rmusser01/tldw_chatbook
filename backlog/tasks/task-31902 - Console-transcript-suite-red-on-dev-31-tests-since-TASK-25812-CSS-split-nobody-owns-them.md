---
id: TASK-31902
title: >-
  Console transcript suite red on dev (31 tests) since TASK-25812 CSS split -
  nobody owns them
status: To Do
assignee: []
created_date: '2026-09-07 22:04'
labels:
  - console
  - tests
  - tech-debt
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
31 tests in Tests/UI/test_console_native_transcript.py fail identically on clean dev. Bisected (fresh venv from dev requirements, macOS): first bad commit b62407e25 'perf(css): split the agentic-terminal module off the pre-first-paint parse (TASK-25812) (#2281)'. Suite was fully green at 18384c80d (Aug 29) and red on current dev. The split moved Console styles into css/screen_agentic_console.tcss which production loads via TldwCli.CSS_PATH/ChatScreen.CSS_PATH, but the suite's TranscriptHarness CSS_PATH (_BUNDLE) still points only at tldw_cli_modular.tcss. CI never caught it: test.yml runs the full UI suite only on dev/main pushes, its last dev run (Sep 4, 33894387395) concluded failure on infra with zero recorded test failures in shard artifacts. Adding screen_agentic_console.tcss to the harness bundle is NOT a one-line fix: it regressed the count 31->48 locally, so the harnesses need rewiring onto ConsolidatedCSSApp's production-order bracketing per the lessons-testing-evidence 'same stylesheet sources, same order' lesson. Failure groups: compositor-painting (roleplay tints, rule spans, empty state), More-menu lifecycle (capture/dismiss/keyboard traversal), action-row width budgets (reference terminals), media card fit. Pattern task: TASK-31249 (Library UI test debt census).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 31 tests pass on dev, or are rewritten/removed with reasons recorded (no bare skips),Root cause recorded: exactly which harness/CSS seam diverges from production after TASK-25812,Suite runs green in a fresh venv on clean dev in a single process,test.yml's dev/main UI-shard runs are green again or the infra failure is reported
<!-- AC:END -->
