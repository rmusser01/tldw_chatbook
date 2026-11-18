---
id: TASK-41046
title: "ADR-250 contradicts itself on the MCP server's SubscriptionsDB"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
- adr
- mcp
priority: low
references:
- backlog/decisions/250-application-services-bundle.md
- tldw_chatbook/MCP/local_server_tools.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-250 contradicts itself on where the MCP server's read-only `SubscriptionsDB` (`MCP/local_server_tools.py:116`) belongs:
- **§2 rule 3** says that when a second composition root opens a database, it moves into the services bundle's `storage` "when that root migrates", and it names the MCP server's `SubscriptionsDB` as the one case.
- **§6** pins the exact `open_profile_storage(...)` signature, which has no subscriptions path.

TASK-36947.7 migrated the MCP root under §6 and left `SubscriptionsDB` opened directly. The owner question it raised on 2026-11-07 is still open. Until one rule wins, the next change to either root has two binding answers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-250 states one rule for the MCP server's SubscriptionsDB and records the owner's decision
- [ ] #2 The code follows that rule
<!-- AC:END -->
