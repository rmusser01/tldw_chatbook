---
id: TASK-81
title: Functionalize Settings Storage defaults
status: Done
assignee: []
created_date: 2026-06-08 00:32
labels:
- settings
- storage
- configuration
- ux
dependencies: []
documentation:
- Docs/superpowers/plans/2026-06-08-settings-storage-defaults.md
- backlog/decisions/004-settings-storage-defaults-restart-boundary.md
- Docs/superpowers/specs/2026-05-24-settings-configuration-hub-design.md
- Docs/superpowers/qa/product-maturity/screen-qa/settings/notes.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn Settings > Storage from a validation-only status panel into a guided configuration category for persisted local database path defaults, while preserving active storage services and file migration as restart-required or future dedicated flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings > Storage loads current persisted database path defaults from config.
- [x] #2 Users can edit, validate, save, and revert storage path defaults without moving files or reconnecting live database handles.
- [x] #3 Invalid or unsafe paths block save and show visible recovery copy.
- [x] #4 Storage copy clearly states changes take effect on next launch and active handles remain unchanged.
- [x] #5 Focused automated tests cover load, edit, validation, save, revert, ownership copy, and no live migration claims.
- [x] #6 Actual Textual-web/CDP screenshots verify baseline, focused input, invalid recovery, and saved restart-required state before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/004-settings-storage-defaults-restart-boundary.md
Reason: Storage defaults define persisted database path ownership and explicitly reject live file migration/reconnection in Settings; this is a storage/runtime boundary future contributors are likely to revisit.

Use `Docs/superpowers/plans/2026-06-08-settings-storage-defaults.md` as the implementation plan. Keep this PR scoped to persisted database path defaults and restart-required copy. Do not add live migration, file movement, automatic directory creation on save, active DB reconnection, or sync/server storage policy changes in this slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Settings > Storage as a guided persisted-defaults category for local database paths. Added `settings_storage_defaults.py` for loading, validation, save payload construction, and non-mutating path check rows; wired SettingsScreen draft state, field-specific guidance, Save/Revert, Check Storage, and an exclusive background save worker. Storage save semantics intentionally update config defaults only: no files are moved, directories are not auto-created, and active database handles remain unchanged until restart per ADR 004.

Added mounted and helper regressions covering config load, validation, edit dirty state, invalid recovery, save/revert, exclusive worker usage, ownership records, and restart/no-live-migration copy. Captured actual textual-web/CDP evidence for baseline, focused input, invalid recovery, and saved restart-required states under `Docs/superpowers/qa/product-maturity/screen-qa/settings/storage-defaults-2026-06-07/`; the screenshot pass was approved after rendered review.

Verification: `python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short` passed with `191 passed, 1 warning` from the existing requests dependency warning.
<!-- SECTION:NOTES:END -->
