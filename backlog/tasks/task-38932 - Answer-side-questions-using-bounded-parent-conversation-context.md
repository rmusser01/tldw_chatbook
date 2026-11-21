---
id: TASK-38932
title: Answer side questions using bounded parent conversation context
status: To Do
created_date: 2026-11-05 16:08
labels:
- console
- side-chat
priority: high
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-17167
documentation:
- Docs/Architecture/chat-pipeline.md
- Docs/Architecture/console.md
- backlog/decisions/068-console-text-selection-and-annotations.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users ask a contextual question about ongoing work without selecting and copying all of its surrounding conversation or redirecting the primary task. Preserve the existing ephemeral side-chat experience while making its available context explicit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can open a side question from the composer without a selection or from the existing transcript-selection actions.
- [ ] #2 The side question uses a bounded snapshot of the authorized parent conversation captured for that side chat; omitted or truncated context is disclosed.
- [ ] #3 Context preparation respects the selected side-chat model's capacity, provider-destination disclosure rules, workspace scope, and existing instruction visibility restrictions.
- [ ] #4 Side-chat questions and replies do not enter parent history, change its draft, interrupt its run, execute tools, or update Personal Context.
- [ ] #5 Stop, retry, dismissal, and parent navigation retain exact side-chat ownership; late responses cannot attach to another conversation.
- [ ] #6 Targeted service and mounted checks cover both entry points, continued primary streaming, oversized context, provider failures, and parent-context changes.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Applicable targeted automated checks, formatting/lint checks, and live UI or real-provider evidence are recorded before completion; full test sweeps require user opt-in.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
