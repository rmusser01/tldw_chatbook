---
id: TASK-99
title: 'Tokenize hardcoded #6f6f6f borders on Schedules/Workflows/ACP panes'
status: Done
assignee: []
created_date: '2026-06-11 17:32'
updated_date: '2026-06-16 14:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
css/components/_agentic_terminal.tcss uses hardcoded #6f6f6f borders for schedules/workflows/acp pane ids, breaking theme portability; replace with $ds-grid-line like the personas fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No hardcoded hex borders remain for those screens.
- [x] #2 Bundle regenerated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Small CSS tokenization fix for existing pane borders; no architectural, storage, provider, or runtime boundary decision.

1. Add a focused regression that fails while Schedules/Workflows/ACP pane borders use hardcoded #6f6f6f instead of $ds-grid-line.
2. Update the source agentic terminal TCSS to use $ds-grid-line for the affected pane borders.
3. Regenerate the modular TCSS bundle from the source modules.
4. Run focused regression and diff checks.
5. Update task notes and mark Done if verification passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the Schedules, Workflows, and ACP pane border color in `tldw_chatbook/css/components/_agentic_terminal.tcss` from hardcoded `#6f6f6f` to `$ds-grid-line`, then regenerated `tldw_chatbook/css/tldw_cli_modular.tcss` with the CSS build script.

Added `Tests/QA/test_agentic_terminal_css_tokens.py` so the source module and generated bundle fail if the hardcoded pane border color returns.
<!-- SECTION:NOTES:END -->
