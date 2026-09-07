---
id: TASK-2241
title: 'MCP: lead the header with the plain-English explainer (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 19:44'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The reassurance line is third under 'MCP' + jargon purpose line; acronym never expanded. Post-fix re-review P2. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plain-English line (with 'MCP (Model Context Protocol)' expanded once) leads,Jargon purpose line demoted or removed,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — header copy change only.

1. mcp_screen.py compose_content(): delete the jargon purpose line ('Manage MCP servers, scoped tools, permissions, and audit readiness.') — the mode chips already enumerate what the screen manages; the plain-English explainer leads directly under the title, expands the acronym once ('MCP (Model Context Protocol) lets chatbook use external tools — most people never need to change anything here.'), and keeps the #mcp-purpose id so the destination parity harness (#mcp-purpose strip checks) still finds it.
2. Update tests asserting the old structure/copy: test_destination_shells.py (3 spots), test_latest_dev_core_app_usability_smoke.py, test_product_maturity_phase6_recovery_docs.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plain-English explainer (acronym expanded once) now leads the header under #mcp-purpose; jargon purpose line deleted; user guide + 4 test files updated; commit also lands task-2240 first-paint test realignments (entangled hunks).
<!-- SECTION:NOTES:END -->
