---
id: TASK-258
title: config.py import hygiene: single TOML parse, consolidated load, lazy chardet
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, startup]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
config.py import parses the 1,285-line embedded default TOML twice (2753 + 3840, the latter only for providers); load_cli_config_and_ensure_existence() and load_settings() each independently re-open/re-parse/re-merge the same user config at import (3865-3866); Utils/Utils.py:48 imports chardet (~21ms) for two rarely-used functions. Also (lower value): tldw_chatbook/__init__'s shim imports textual.widgets (~53ms) for every non-UI consumer. Watch _CONFIG_CACHE/_SETTINGS_CACHE semantics — unmocked integration tests required for any loader consolidation (T229 rule). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P2 C3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Embedded default TOML parsed exactly once per process
- [ ] #2 User config read+parsed once at startup, both caches serving from it (unmocked real-loader tests pin behavior)
- [ ] #3 chardet imports only inside its two callers
- [ ] #4 Measured config-import delta reported
<!-- AC:END -->
