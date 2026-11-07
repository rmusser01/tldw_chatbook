---
id: TASK-37120
title: 'CI: stop pytest''s major version reaching the PR lanes unreviewed'
status: To Do
assignee: []
created_date: '2026-10-27 17:45'
labels:
  - ci
  - testing
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR lanes in .github/workflows/derived-artifacts.yml install pytest unpinned (`-e . pytest pytest-asyncio pytest-timeout packaging`), which resolved pytest 9.1.1 on 2026-10-27, while the repo venv agents test in has pytest 8.4.2. TASK-33620.15.15 found a census-bound test (Tests/UI/test_console_first_send_preimport.py) that passed every local run and failed only under the lanes' pytest 9, because pytest 9 resolves monkeypatch dotted paths through importlib.import_module. The next pytest major bump will reach every lane on its next install with no reviewed change, and local green will again not predict lane green. Found in review of TASK-33620.15.15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The PR lanes and the repo's development venv resolve the same pytest major version, or a recorded decision says why they differ and how a contributor reproduces the lane's version locally
- [ ] #2 A new pytest major version reaches the PR lanes only through a reviewed change (for example a pin or constraints file), not on the next unpinned install
- [ ] #3 The decision is recorded with the TASK-33620.15.15 incident that prompted it
<!-- AC:END -->
