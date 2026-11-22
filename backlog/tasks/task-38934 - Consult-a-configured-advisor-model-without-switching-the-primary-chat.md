---
id: TASK-38934
title: Consult a configured advisor model without switching the primary chat
status: To Do
created_date: 2026-11-05 16:08
labels:
- agents
- console
- model-routing
priority: high
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-35582
- TASK-32477
documentation:
- Docs/Architecture/agent-runtime.md
- Docs/Architecture/llm-providers.md
- backlog/decisions/344-agent-provider-routing.md
- backlog/decisions/281-primary-agent-multi-model-code-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users obtain focused strategic advice from a chosen second model while their primary model remains responsible for the ongoing task. Make the consultation's scope, destination, and resource use visible so its value can be assessed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can explicitly request one advisory consultation using an existing configured provider/model or preset; the primary chat's selected provider/model remains unchanged.
- [ ] #2 The consultation includes the user question and a bounded, inspectable authorized context selection, subject to ordinary provider-disclosure, routing, permission, and budget gates.
- [ ] #3 The advisor returns guidance without tools, file effects, recursive consultation, automatic task execution, or implied approval of its recommendations.
- [ ] #4 The response identifies its actual provider/model, source consultation, available token/cost accounting, and any unknown pricing; unavailable destinations are not silently substituted.
- [ ] #5 Cancellation, provider failure, timeout, and exhausted capacity leave an explicit outcome without duplicate dispatch or loss of the primary conversation.
- [ ] #6 Targeted checks and a controlled real-provider example demonstrate a primary/advisor exchange, resource accounting, context limits, and rejection of unauthorized destinations.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
