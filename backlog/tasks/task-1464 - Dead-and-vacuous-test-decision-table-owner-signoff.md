---
id: TASK-1464
title: >-
  Decision table for dead/vacuous tests: rotted skips, swallowed assertions, assertion-free and mock-only tests (owner sign-off)
status: Done
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

- [x] Decision table covers: 27 exception-swallowed tests; ~174 assertion-free; ~99 trivial-assert (incl. the 3 placeholder security tests in Tests/Web_Scraping/test_security.py); 143 mock-callgraph-only; module-level skips with contradicted reasons; test_vector_stores.py (delete vs rewrite); @slow policy (proposal: scheduled --run-slow job)
- [x] Each category has an owner decision recorded before any deletion lands
- [x] The xfail(strict=False) quarantine convention is documented in Tests/README.md
- [x] Approved subset implemented with itemized collect-only deltas

## Implementation Plan

1. Refresh every inventory on current dev before asking for rulings (audit data was 15 PRs stale; one whole category had self-resolved via foreign deletions)
2. Present four category rulings via the decision table; implement the approved subset
3. Verify per file; attribute any surrounding failures against the unmodified base tip

## Implementation Notes

Owner rulings (all four recommended options): un-swallow all; delete all
placeholders; delete the stub suite + follow-up; inventory-only for bulk.

Deleted: `test_worldbook_ui.py` + `test_chat_dictionaries_ui.py` (whole files —
assert-True checklists, commented-out example bodies, a prose-dict
"documentation test"); the 3 placeholder security tests (+ the emptied
`TestSecureDefaults`); `test_vector_stores.py` (stub suite; real coverage filed
as task-1600); the legacy AppTest block in `test_tools_settings_window.py`
(guard + always-skip fixture + its 16 permanently-skipped consumers).

Un-swallowed: of 16 flagged sites, 4 were verified-legitimate on inspection
(collect-then-assert, known-type handler asserts, hypothesis assume) and left
alone; 12 genuine swallows narrowed to the domain-error classes each contract
permits, so crash classes now bite. The flagship
`test_budget_monitoring_integration` immediately surfaced what its swallow hid:
BudgetMonitor raises EAGERLY from update_cost and the old API
(`is_budget_exceeded`/`check_budget`) is gone — the feature was integrated all
along; the test was skipping itself. Rewritten to the real contract. The
kokoro narrowing needed one widening round: under the suite's HF-offline
default (task-1451) a missing model surfaces as OSError/LocalEntryNotFoundError,
matched by class now.

Category 4: `backlog/docs/test-suite-vacuous-inventory-2026-07-30.md` (249
assertion-free / 119 mock-only candidates, with the helper-assert
false-positive caveat front and center).

Also observed, NOT from this change (identical on unmodified base): 11
pre-existing failures in `Tests/TTS/test_profile_backup_integration.py` +
`test_tts_preferences.py` — the in-flight speech workstream's area; reported in
the PR rather than filed over their active work.
