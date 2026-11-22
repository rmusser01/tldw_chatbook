---
id: draft-38943
title: Design bounded plugin UI contributions from proven Console panels
status: Draft
created_date: 2026-11-05 16:09
updated_date: 2026-11-22 15:51
labels:
- plugins
- console
- design
priority: low
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- https://code.claude.com/docs/en/plugins/mods/overview
- TASK-38935
- TASK-38940
- TASK-38941
- TASK-38942
documentation:
- Docs/Architecture/console-run-hooks.md
- Docs/Architecture/tool-catalog.md
- backlog/decisions/162-managed-agent-plugins.md
- backlog/decisions/163-expanded-console-hook-runtime.md
- backlog/docs/design-language.md
- backlog/docs/component-patterns.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Determine whether reusable plugin UI contributions would serve demonstrated Console customization needs after concrete panels have been qualified. Define a narrow, reviewable extension boundary before committing to a new UI plugin API or runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A written comparison uses evidence from concrete output/context/validation/replay panel work to identify which contributions require an extension seam and which are adequately served by built-ins, settings, commands, or hooks.
- [ ] #2 The proposed first release specifies allowed data/event access, UI slots and component vocabulary, activation scope, lifetime, budgets, error isolation, and unload/shutdown behavior.
- [ ] #3 The contract preserves permission floors, workspace exclusions, instruction-body visibility, runtime ownership, accessibility, keyboard conventions, and design-token governance.
- [ ] #4 Alternatives include retaining built-in panels and using existing plugin/hook facilities; no JavaScript runtime or compatibility promise is assumed without a justified decision.
- [ ] #5 An ADR records the accepted or deferred direction, unresolved questions, and the smallest separately reviewable implementation release; existing plugin and hook governance is explicitly reconciled.
- [ ] #6 The deliverable is reviewed design and linked evidence only; it neither enables third-party UI code nor changes permissions or the live plugin runtime.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
- [ ] #4 Design-only completion includes a reviewed written contract and ADR disposition; implementation, installation, or activation is not part of this draft.
<!-- DOD:END -->
