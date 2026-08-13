---
id: TASK-15782
title: Repair the generic ingest-options snapshot test asserting a stale hardcoded dict
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - test-health
  - library
priority: low
---

## Description

Found and flagged as pre-existing and unrelated in task-15470's
Implementation Notes (input-latency burn-down's config-persistence task):
`test_options_persist_to_config` fails on a content mismatch — a schema
drift, not a threading bug. Task-15470's notes record that this failure was
"exposed only once the `run_worker` crash this task fixed stopped masking
it," meaning the test has likely been silently failing-or-skipped-via-crash
for a while and nobody noticed once the underlying worker exception stopped
swallowing it. The test itself asserts against a hardcoded dict of expected
generic-ingest options that has drifted out of sync with whatever the ingest
options form/save path actually persists today.

## Acceptance Criteria

- [ ] `test_options_persist_to_config`'s expected dict is reconciled against
      current production behavior — diagnosed as either a genuinely stale
      test expectation (update the test) or a real regression in what gets
      persisted (fix production and keep the test's original intent)
- [ ] The specific drifted key(s)/value(s) are identified and documented in
      the task notes, not just "test updated"
- [ ] The test passes on dev without weakening its coverage (it should still
      catch a genuine future persistence regression, not just always pass)
