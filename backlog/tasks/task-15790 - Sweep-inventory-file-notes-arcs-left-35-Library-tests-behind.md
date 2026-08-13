---
id: TASK-15790
title: 'Sweep inventory: file-notes arcs left ~35 Library tests behind'
status: To Do
assignee: []
labels:
  - test-health
  - library
priority: medium
---

## Description

From the task-15211 full-suite sweep (`Docs/Design/2026-08-13-tests-ui-sweep-inventory.md`,
chunks 7-8): ~35 Library tests are red on dev, all consistent with the
file-notes feature arcs shipping without their module contracts re-run:

- 10x one color contract in `test_library_file_notes_git.py`
  (`Color(51,66,78)` vs `Color(81,103,126)`) — a theme change unpropagated.
- 3x push copy: `'Push checking'` vs `'Review session changes (2) · Checking push'`.
- 15x `test_library_shell.py` notes cluster: focus-target drift
  (`'library-note-preview-region'` vs `'library-note-save'`), `NoMatches` on
  notes rows, id-set diffs.
- Stragglers in workspace/export-receipt/choice-strips/multiselect/prompts
  modules, including 2x stale doubles missing new production attributes and
  2x `coroutine raised StopIteration` in `test_library_prompts_canvas.py` —
  **triage the StopIteration pair first**: a PEP-479 conversion can mask a
  real exhausted-iterator bug in production, and "possible real bug" outranks
  every stale contract here.

Per the 15512 precedent: attribute each cluster to its causing commit before
adjusting any expectation, and treat "the test is old" as evidence about the
test only after the product behaviour is confirmed intended.

## Acceptance Criteria

- [ ] The StopIteration pair is attributed (real bug vs test artifact) with evidence, before any contract is updated
- [ ] Each cluster is attributed to its causing commit
- [ ] Genuine product breaks are fixed rather than absorbed into expectations
- [ ] The listed modules pass whole on dev
