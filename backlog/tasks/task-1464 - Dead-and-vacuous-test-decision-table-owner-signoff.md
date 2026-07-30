---
id: TASK-1464
title: >-
  Decision table for dead/vacuous tests: rotted skips, swallowed assertions, assertion-free and mock-only tests (owner sign-off)
status: To Do
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - cleanup
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The audit (`backlog/docs/test-suite-audit-2026-07-30.md` §5) found ~416 tests that verify little or nothing and ~226 unconditionally dead tests. What to delete vs rewrite vs unskip is a policy decision. This task delivers the per-category decision table with file:line inventories and executes only the approved subset. Notably: 27 tests wrap all assertions in swallowing `except Exception` (including the two Evals integration tests the docs cite as flagship coverage); `Tests/RAG/simplified/test_vector_stores.py` tests in-file stubs against a module that does not exist; 25 `@slow` tests never run anywhere because nothing passes `--run-slow`; the suite has zero xfail usage so known-broken tests rot invisibly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Decision table covers: 27 exception-swallowed tests; ~174 assertion-free; ~99 trivial-assert (incl. the 3 placeholder security tests in Tests/Web_Scraping/test_security.py); 143 mock-callgraph-only; module-level skips with contradicted reasons; test_vector_stores.py (delete vs rewrite); @slow policy (proposal: scheduled --run-slow job)
- [ ] Each category has an owner decision recorded before any deletion lands
- [ ] The xfail(strict=False) quarantine convention is documented in Tests/README.md
- [ ] Approved subset implemented with itemized collect-only deltas
