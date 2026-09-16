---
id: TASK-32687
title: Qualify Cursor and Codex package and hook adapters
status: To Do
assignee: []
created_date: '2026-09-16 04:30'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32686
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make supported foreign packages useful without claiming runtime equivalence for unsupported semantics.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Version-pinned portable/OpenAI overlay, standalone Codex and Cursor fixtures cover deterministic interpretation, explicit-path replacement and execution-affecting catalog overlays.
- [ ] #2 Vendor metadata, variables, manual-only behavior and rule/agent constraints map explicitly or remain unsupported with preserved deliberate exclusions.
- [ ] #3 Hook mappings qualify timing, payload, matchers, input/output, cwd, shell-free argv and timeout together; required unknown guards block affected behavior.
- [ ] #4 Each fixture records upstream revision and license with independently expected inventory; parsing evidence is separated from exercised Chatbook behavior and original-host comparison.
<!-- AC:END -->
