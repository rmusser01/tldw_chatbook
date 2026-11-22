---
id: TASK-38937
title: Stage a child-chat result back into its originating parent
status: To Do
created_date: 2026-11-05 16:08
labels:
- console
- agents
- handoff
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-34215
- TASK-38936
documentation:
- Docs/Architecture/console.md
- Docs/Architecture/agent-runtime.md
- backlog/decisions/219-console-chat-destinations-and-bounded-starts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user bring a completed child conversation's useful result back to the conversation that requested it without copying unrelated history or silently redirecting ongoing work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An eligible child result offers an explicit return-to-parent action resolved from authoritative creation provenance, with the exact parent and selected result shown before staging.
- [ ] #2 The action stages a bounded, attributed summary or selected result in the parent for user review; it never automatically sends, interrupts a running turn, or overwrites an existing draft.
- [ ] #3 Failed, cancelled, partial, or superseded results retain their actual outcome labels and cannot be presented as verified completion.
- [ ] #4 Unavailable/deleted parents, changed scopes, revoked access, or unresolved provenance refuse the action without falling back to the current chat.
- [ ] #5 Repeated activation cannot duplicate the same pending handoff, and a staged result remains tied to its exact source revision/result identity.
- [ ] #6 Targeted and mounted checks cover running parents, draft conflicts, stale results, duplicate activation, and navigation during handoff.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
