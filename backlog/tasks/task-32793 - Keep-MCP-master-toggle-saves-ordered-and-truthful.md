---
id: TASK-32793
title: Keep MCP master-toggle saves ordered and truthful
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 14:55'
updated_date: '2026-09-18 15:32'
labels:
  - mcp
  - ui
  - lifecycle
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep explicit local-tools master-switch choices and their displayed outcomes consistent through overlapping writes, read-only refresh, navigation and shutdown across both MCP controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tools and Servers master-switch choices persist in submission order; cancellation or screen destruction cannot reorder or abandon admitted writes.
- [x] #2 Pending latest choices survive refresh and recreation; failures restore saved truth, partial cache-publication outcomes stay truthful, and old configuration receipts cannot overwrite current state.
- [x] #3 Normal shutdown fences admission and drains saves before dependent teardown without changing tool authority or unrelated settings.
- [x] #4 Targeted overlap/recovery/context tests and real private dark/light compact/wide journeys qualify the changes; review ledgers and draft PR evidence are updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce overlapping writes, pending refresh and partial save outcomes. 2. Record ADR-169 extending the existing MCP root-save owner to the two explicit local configuration controls; route both master-switch entry points through it. 3. Preserve latest requested state and scoped receipts through refresh/recreation; keep catalog presentation independent of persistence. 4. Verify ownership/order/config identity, existing root behavior and exact shutdown. 5. Review native private theme/size journeys and update ledgers and PR2707. ADR required: yes. ADR path: backlog/decisions/169-mcp-local-config-save-lifetime.md. Reason: Extend existing cross-screen app ownership from root saves to the same master setting in Tools and Servers; no runtime authority change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both master controls now capture activation intent and admit saves synchronously to the shared app-owned local-config FIFO (ADR-169, extending ADR-168). Per-key receipts preserve pending/partial outcomes and advance only across verified sibling mutations; normal shutdown drains both settings. Master polling leaves root drafts untouched. Servers wraps the master label and keeps dependent controls and saved-state guidance consistent while pending. Runtime authority and unrelated settings are unchanged.

Validation: 177 distinct targeted cases across scoped lifecycle, master, canvas, paint, adjacent and governance runs; final replay corrects a harness-only widget-ID typo. Sixteen final dark/light 80x24/170x48 native captures were inspected; run004 confirms actual ordered file/cache writes, unchanged permission profiles/defaults, ten healthy private databases and clean shutdown. Independent review found no remaining issue. Ruff has no introduced findings; new files/changed methods formatted, CSS rebuilt, Backlog and diagnostic guards pass. The diagnostic inventory intentionally removes one obsolete warning and adds no sink/call. The Servers fixture now retains the bootstrap private config identity like adjacent MCP modules.

Evidence and failed-attempt disposition: Docs/superpowers/qa/2026-09-18-mcp-master-settings/README.md. Updated MCP/completion ledgers and the queued-admission testing lesson. No full suite; PR2707 remains draft for separate visual review/merge approval. Remaining tool refresh/execution, non-master Servers labels/lifecycles and Audit are recorded in the MCP ledger.
<!-- SECTION:NOTES:END -->
