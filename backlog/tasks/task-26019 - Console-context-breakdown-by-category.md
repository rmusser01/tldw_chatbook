---
id: TASK-26019
title: 'Console: context breakdown by category'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 17:13'
labels:
  - console
  - context
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user cannot see what is consuming the context window. Verified on origin/dev: Widgets/Console/console_context_controls.py:105,118 shows a request row and a conversation row plus overhead, and a named grep for breakdown across Chat/ and Widgets/Console/ returns only cost-modal rows - so when a window fills, there is no way to tell whether tool schemas, RAG results, attachments or history are responsible. Hermes splits the window into eight named categories with a glyph grid. Chatbook's PreparedConsoleRequest accounting already separates memory, compactable and overhead, so the data is partly assembled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The context surface reports usage split by named category covering at minimum: system prompt, tool schemas, retrieved context, attachments, memory summary and live conversation
- [x] #2 Category figures are derived from the same accounting used to build the request, not estimated separately - a mismatch between the two is impossible by construction
- [x] #3 Categories that cannot be attributed are shown as an explicit unattributed bucket rather than silently folded into another
- [x] #4 The existing model-window honesty is preserved: an unverified window is still labeled as estimated
- [x] #5 The breakdown updates without a model call
- [x] #6 Where a category is large enough to act on, the surface names the action that would reduce it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: accounting split tests (tools step, attachments, provenance-gated RAG)\n2. ConsoleRequestTokenAccounting: tool_schema/rag/attachment fields via the SAME cumulative wire counts; non_compactable preserved\n3. Pure build_context_breakdown (partition sums to total; explicit unattributed bucket; action hints)\n4. Controller captures prepared_before.accounting at the send preflight; popover renders rows
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Accounting extension in console_prepared_request._account_categories: a 'tools' step joins the cumulative wire-count order (system, memory, TOOLS, mandatory, compactable, active) so tool_schema_tokens is its own delta instead of hiding in mandatory (the one legacy pin updated: its intent 'tools are counted' unchanged; non_compactable_tokens now includes the tools bucket so its single consumer is byte-identical). rag_context_tokens ⊂ mandatory via the same construction, ONLY when request provenance labels rows (rag_attributed; capture-off = unknowable); attachment_tokens = per_image share of conversation rows (frozen tuples handled). All appended defaulted fields. Pure build_context_breakdown in console_context_controls: 7-way partition summing exactly to total_input_tokens, capture-off mandatory shown as 'Instructions & context (unattributed)' (AC#3), zero rows dropped, action hints on Conversation/Attachments/Memory/Tools (AC#6). Controller captures prepared_before.accounting at the send preflight (the request's OWN accounting — AC#2 by construction; AC#5 free) into _context_accounting_by_session; chat_screen passes it to build_console_context_control_state; the model popover renders 'Last request by category' rows. Window honesty untouched (AC#4). 6 new tests (3 accounting + 3 builder); prepared-request 32 passed; context-controls/session-settings failures verified pre-existing (worktree bisect at 7e67a0070). Gap noted: the breakdown reflects the LAST prepared send — before any send the popover simply omits the block.
<!-- SECTION:NOTES:END -->
