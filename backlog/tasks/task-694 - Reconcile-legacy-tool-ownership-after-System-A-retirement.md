---
id: TASK-694
title: Reconcile legacy tool ownership after System A retirement
status: Done
assignee: []
created_date: '2026-07-26 06:30'
updated_date: '2026-08-12 22:08'
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
  - >-
    Docs/superpowers/specs/2026-08-12-task-694-legacy-tool-ownership-reconciliation-design.md
  - >-
    Docs/superpowers/plans/2026-08-12-task-694-legacy-tool-ownership-reconciliation.md
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
- [x] #1 A focused ownership test proves the four legacy names are absent from the gateable built-in table, the default built-in catalog remains exactly calculator/datetime, and the current provider catalogs contain `web_search`, `library_search_notes`, and `search_library_rag` under their authoritative owners
- [x] #2 Fresh-process tests prove `WebSearchTool`, `RAGSearchTool`, and `SearchNotesTool` remain importable through `tldw_chatbook.Tools` without invoking a tool, opening an application database, or using the network
- [x] #3 Current governance records no longer promise a four-tool built-in port, no longer call `web_search` public-target-only, and scope profile-driven agent retrieval to the Library provider while retaining MCP `perform_rag_search` as separate follow-up work
- [x] #4 The complete unwired audit subsystem and every live built-in/local file-mutation seam are assigned to the audit follow-up, and the live audit guide prominently states that the current code is not wired, monitoring, enforcement, or a security control
- [x] #5 Focused tests and changed-file static/security checks pass, and the implementation range contains no production Python change, new tool registration, gate, risk tag, alias, warning, or compatibility deletion
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Pinned the current ownership map without changing runtime behavior: `LocalToolProvider` owns `local:web_search`; `LibraryToolProvider.library_search_notes` owns direct note retrieval; `LibraryRagToolProvider.search_library_rag` owns fallback profile-driven retrieval; and no live provider owns `code_audit`.
- Preserved the lazy compatibility imports for `WebSearchTool`, `RAGSearchTool`, and `SearchNotesTool`. The fresh-child test resolves only those imports and independently blocks/probes `socket.bind`, `socket.connect`, `socket.getaddrinfo`, `socket.gethostbyaddr`, `socket.gethostbyname`, `socket.gethostname`, `socket.getnameinfo`, `socket.sendmsg`, `socket.sendto`, and `sqlite3.connect`; it never instantiates or invokes a legacy tool. Mutations removing `socket.gethostbyname`, `socket.sendto`, or `sqlite3.connect` from the blocked set, or removing the caught-event final failure, each made the compatibility node RED.
- Expanded TASK-743 to own the complete audit keep/redesign/delete decision across the retained implementation, hooks, demo, tests, docs, built-in `write_file`, and local `fs_write`/`fs_edit`/`fs_patch`. Narrowed TASK-3500 to the remaining MCP `perform_rag_search` parity gap because the Library providers already own agent retrieval.
- Corrected ADR-032 and TASK-1354 to distinguish permission from transport: both web tools remain permission-gated; only `web_fetch` enforces public HTTP(S) target and redirect validation. `web_search` sends the query to the caller/model-selected allowlisted engine and may use an operator-configured local Searx endpoint; it does not apply public-target validation.
- Marked the audit guide prominently as unwired historical/reference material, not monitoring, enforcement, or a security control, while preserving the historical System A and RAG observations with explicit current dispositions.
- Verification: the clean-tree fetch left `origin/dev` unchanged at `706105a2f7e1406231f513a601718dab721cd997` (old = new); at the single no-op rebase, HEAD was `1c402aef17ae4f5fd7c5c21a3db78d1f6a4996dc`, merge-base was `706105a2f7e1406231f513a601718dab721cd997`, ahead/behind was 12/0, and the upstream delta and overlap were empty. The exact focused suite passed `205 passed, 1 warning in 11.51s`; the warning was the characterized Requests dependency-version warning. Ruff format/check, mypy, Bandit (with B101 skipped), compileall, range diff-check, and bare diff-check all exited zero.
- Scope evidence: both `git diff --name-only origin/dev...HEAD -- 'tldw_chatbook/**/*.py'` and the packaging/data-migration path scan emitted nothing. The range adds no registration, gate, risk tag, shadow entry, alias, runtime warning, compatibility deletion, or successful legacy invocation test.
- ADR required: no. This records existing boundaries under [ADR-030](../decisions/030-local-library-agent-tool-boundary.md) and [ADR-032](../decisions/032-local-agent-tool-permission-boundary.md). Approved design: `Docs/superpowers/specs/2026-08-12-task-694-legacy-tool-ownership-reconciliation-design.md`; implementation plan: `Docs/superpowers/plans/2026-08-12-task-694-legacy-tool-ownership-reconciliation.md`.
- Added the TASK-694 Markdown hard-break incident to [`lessons-backlog-hygiene.md`](../docs/lessons-backlog-hygiene.md): stripping diff-check-invalid trailing spaces without a render comparison changed the intended visual separation, so governed metadata now uses explicit list structure.
<!-- SECTION:NOTES:END -->
