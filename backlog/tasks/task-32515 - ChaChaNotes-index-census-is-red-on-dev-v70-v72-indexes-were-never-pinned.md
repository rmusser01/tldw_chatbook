---
id: TASK-32472
title: 'ChaChaNotes index census is red on dev: v70-v72 indexes were never pinned'
status: To Do
assignee: []
created_date: '2026-09-11 20:21'
labels:
  - db
  - tests
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/ChaChaNotesDB/test_index_census.py::TestIndexCensus::test_no_unexpected_indexes fails on plain dev (tip be380a1a6f): the live schema defines 14 named indexes added by the v70-v72 migrations that EXPECTED_CHACHANOTES_INDEXES never pinned. The census exists to make every new index a reviewed decision; while it is red, a migration can add an unreviewed index and nobody notices, and every branch that touches the DB inherits a red it cannot tell from its own. Surfaced by the wave-3 backlinks group (task-32186), which pins its own idx_note_links_target but cannot own the fourteen strangers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_no_unexpected_indexes passes on dev with every v70-v72 index pinned in EXPECTED_CHACHANOTES_INDEXES (table, unique, columns), each reviewed as intended rather than pasted
- [ ] #2 Any index found to be redundant or unintended is dropped by a migration instead of pinned, and the decision is recorded in the task notes
- [ ] #3 The test's failure message still prints paste-ready IndexPin lines for the next drift
<!-- AC:END -->
