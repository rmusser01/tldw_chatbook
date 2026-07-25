---
id: TASK-634
title: Align Settings config tests with batched persistence boundary
status: Done
assignee:
  - '@codex'
created_date: '2026-07-25 19:20'
updated_date: '2026-07-25 19:26'
labels:
  - settings
  - tests
  - configuration
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Settings behavior tests aligned with the centralized configuration persistence boundary so they verify the active batched adapter instead of requiring a deleted, unused module import.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings tests no longer monkeypatch the removed `settings_screen.save_setting_to_cli_config` symbol.
- [x] #2 Batched saves are still asserted through `SettingsConfigAdapter`, while invalid and reverted drafts are proven not to persist.
- [x] #3 The focused failures, full Settings configuration-hub file, and resumed Settings block pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the 12 full-block failures and confirm they all fail before reaching behavior because the obsolete module symbol is absent.
2. Remove no-op legacy patches and route persistence assertions through the active `SettingsConfigAdapter` seam.
3. Run the focused tests, full configuration-hub file, resumed Settings block, and static checks.

ADR required: no
ADR path: N/A
Reason: This is a test-contract correction following an existing persistence-boundary decision; production configuration ownership does not change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Removed two no-op patches from staged-toggle tests and replaced the remaining
  dead singular-helper spies with one recorder at the active
  `SettingsConfigAdapter.save_sections` boundary.
- Preserved the batched payload assertion and strengthened invalid/revert
  coverage so those paths prove that the active persistence adapter receives
  no calls.
- Verification: all 12 original failures passed, followed by the complete
  295-test Settings block. Ruff check, Ruff format, and diff checks passed.
- No production code or configuration ownership changed.
<!-- SECTION:NOTES:END -->
