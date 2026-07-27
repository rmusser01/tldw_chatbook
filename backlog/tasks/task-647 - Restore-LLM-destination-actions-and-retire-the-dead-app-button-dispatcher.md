---
id: TASK-647
title: Restore LLM destination actions and retire the dead app button dispatcher
status: To Do
assignee: []
created_date: '2026-07-26 23:48'
labels:
  - architecture
  - state
  - llm
  - reliability
dependencies: []
references:
  - backlog/decisions/026-application-session-state-ownership.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore truthful production LLM controls and remove the unused root dispatcher so destination view state and actions have one live owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every visible actionable control in the production LLM window is either handled exactly once by the destination or removed when no runtime contract exists.
- [ ] #2 The unsupported custom Transformers server-launch block is absent without adding a new process lifecycle.
- [ ] #3 LLM navigation remains owned by LLMManagementWindow while TldwCli.llm_active_view, its watcher, llm_nav_events root routing, button_handler_map, and _build_handler_map are removed.
- [ ] #4 Normal production TldwCli tests cover destination navigation and a fault-injected safe action without test or simplified application classes.
- [ ] #5 Focused static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
