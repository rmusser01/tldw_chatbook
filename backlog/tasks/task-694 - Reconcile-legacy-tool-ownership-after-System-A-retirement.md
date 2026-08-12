---
id: TASK-694
title: Reconcile legacy tool ownership after System A retirement
status: In Progress
assignee: []
created_date: '2026-07-26 06:30'
updated_date: '2026-08-12 20:27'
labels:
  - tools
  - agents
  - security
dependencies:
  - TASK-545
references:
  - ADR-030
  - ADR-032
documentation:
  - Docs/superpowers/specs/2026-08-12-task-694-legacy-tool-ownership-reconciliation-design.md
  - Docs/superpowers/plans/2026-08-12-task-694-legacy-tool-ownership-reconciliation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-545 originally left `rag_search`, `web_search`, `search_notes`, and
`code_audit` for a later built-in port. System A has since been removed, and
the four capabilities no longer share one runtime destination: `web_search`
ships through the local provider; Console Library retrieval ships through the
direct or RAG Library provider; and the audit subsystem is unwired pending a
separate keep/redesign/delete decision.

Close the stale port promise without duplicating those providers or pretending
the unwired audit is a security control. Pin the current ownership and default
built-in inventory, preserve the tested legacy Python imports as compatibility
surfaces, and correct the related current and historical governance records.
No runtime provider, permission, schema, flag, or tool behavior changes in this
task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focused ownership test proves the four legacy names are absent from the gateable built-in table, the default built-in catalog remains exactly calculator/datetime, and the current provider catalogs contain `web_search`, `library_search_notes`, and `search_library_rag` under their authoritative owners
- [ ] #2 Fresh-process tests prove `WebSearchTool`, `RAGSearchTool`, and `SearchNotesTool` remain importable through `tldw_chatbook.Tools` without invoking a tool, opening an application database, or using the network
- [ ] #3 Current governance records no longer promise a four-tool built-in port, no longer call `web_search` public-target-only, and scope profile-driven agent retrieval to the Library provider while retaining MCP `perform_rag_search` as separate follow-up work
- [ ] #4 The complete unwired audit subsystem and every live built-in/local file-mutation seam are assigned to the audit follow-up, and the live audit guide prominently states that the current code is not wired, monitoring, enforcement, or a security control
- [ ] #5 Focused tests and changed-file static/security checks pass, and the implementation range contains no production Python change, new tool registration, gate, risk tag, alias, warning, or compatibility deletion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/030-local-library-agent-tool-boundary.md; backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: this task changes no runtime, storage, provider, permission, egress, or security boundary. It records the provider boundaries already accepted by ADR-030/032; any retained audit redesign must perform its own ADR check.

Detailed plan: `Docs/superpowers/plans/2026-08-12-task-694-legacy-tool-ownership-reconciliation.md`

1. Add one read-only ownership/import test module and mutation-prove every asserted owner and compatibility mapping.
2. Correct the authoritative task and ADR records without changing production behavior.
3. Preserve historical observations while appending the current audit and RAG ownership outcome, and mark the audit guide as unwired/non-enforcing.
4. Rebase once onto latest dev, run the focused behavioral/static/security gates, verify no production Python diff, self-review, and close this task only with complete evidence.
<!-- SECTION:PLAN:END -->
