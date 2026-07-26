---
id: TASK-693
title: Surface the configurable file-tool sandbox root to the user
status: To Do
assignee: []
created_date: '2026-07-26 06:06'
labels:
  - tools
  - agents
  - ux
dependencies:
  - TASK-545
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
write_file (and read_file/list_directory) are confined to <user data dir>/tool_sandbox by default. The root is already configurable via [tools] file_sandbox_root in config.toml, but nothing in the app surfaces that setting or the sandbox's actual location to the user, so an agent that can only write where the user never looks is of limited practical use. Identified in the design spec for TASK-545 P2 (Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md, 'Known limitations carried, not fixed'), which explicitly deferred it to keep that phase scoped to porting the tool behind the permission gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can discover, from within the app, where the file-tool sandbox root currently resolves to
- [ ] #2 The [tools] file_sandbox_root config option is discoverable (documented and/or exposed in a settings surface) rather than only readable by someone who already knows to look for it
<!-- AC:END -->
