---
id: TASK-34918
title: Add-a-cheap-snapshot-gate-to-the-Console-credential-poll
status: To Do
assignee: []
created_date: '2026-10-13 14:46'
labels:
  - performance
  - ui-perf
  - wave-1-timers
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console 0.25s _poll_console_credential_readiness timer rebuilds _active_console_settings_readiness() every tick before the (revision, status, version) snapshot compare (chat_screen.py:16648-16663). The in-tree comment already clocks this at ~8ms per tick (task-32804.3) — roughly 3% of a core while idling on the Console screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tick cost dominated by a cheap fingerprint compare and full readiness build only runs when it changes,Credential status still updates within one tick of an actual change (test)
<!-- AC:END -->
