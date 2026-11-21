---
id: TASK-42133
title: "Unused hashlib import in test_config_load_settings_table_guard.py"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:20
labels:
- follow-up
- tests
- lint
priority: low
references:
- Tests/test_config_load_settings_table_guard.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the providers stream (round 18). At d85e90bc9e `ruff check --select F401` reports `Tests/test_config_load_settings_table_guard.py:716:12: F401 'hashlib' imported but unused` (the round report said line 717). `test_config_toml_content_is_the_data_file_with_placeholders_resolved` imports `hashlib` and never uses it; the import came with 90f239f9c2 (2026-11-03). The Ruff F gates (`.github/workflows/lint.yml`, `derived-artifacts.yml`) check only `tldw_chatbook/`, so nothing fails on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `ruff check --select F Tests/test_config_load_settings_table_guard.py` reports no findings
- [x] #2 The file's tests still pass
<!-- AC:END -->

## Implementation Plan

1. Delete the unused `import hashlib` in `test_config_toml_content_is_the_data_file_with_placeholders_resolved`.
2. Run `ruff check --select F` on the file and the file's tests.

## Implementation Notes

Removed the unused `import hashlib` from `test_config_toml_content_is_the_data_file_with_placeholders_resolved` (the sibling `test_default_config_template_bytes_are_unchanged` keeps its own, used import).
- Before: `ruff check --select F Tests/test_config_load_settings_table_guard.py` reported `F401 'hashlib' imported but unused` at 717:12. After: "All checks passed!".
- The file's tests: 12 passed (`-n 4`, through the test wrapper).

File: `Tests/test_config_load_settings_table_guard.py`.
