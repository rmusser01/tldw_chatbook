---
id: TASK-2063
title: 'MCP: plain-language onramp and de-jargoned labels (F-052, F-060)'
status: In Progress
assignee: []
created_date: '2026-08-03 16:24'
updated_date: '2026-08-03 17:05'
labels:
  - mcp
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCP screen has no plain-language onramp and jargon labels: purpose line is a tautology, form labels say 'Profile id' and 'Transport', the rail has no empty state at zero servers, and the disabled 'No scope entities' Select has no reason tooltip. Add a one-line explainer, rename UI labels only (Profile id -> Name, Transport -> Connection), add a rail empty state, and add a tooltip explaining the disabled scope Select.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mcp_screen purpose line gains a one-line plain-language explainer,Profile form UI labels renamed: Profile id to Name and Transport to Connection (internal identifiers unchanged),Rail shows an empty state at zero servers,Disabled 'No scope entities' Select has a tooltip explaining why,Existing tests asserting old strings updated; MCP tests passing plus ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI label/copy-only; internal ids and payload keys unchanged). Steps: 1. RED tests: profile form label is 'Name' not 'Profile id'; overview table columns use 'Connection'; MCP screen has plain-language explainer line; rail shows empty state at zero servers; disabled scope-ref Select carries a why-tooltip. 2. mcp_screen.py: add #mcp-purpose-explainer line. 3. mcp_profile_form.py: 'Profile id' -> 'Name'. 4. 'Transport' -> 'Connection' in mcp_servers_mode.py (_TABLE_COLUMNS, detail line) and mcp_server_mutations.py (form label, required error). 5. mcp_rail.py: #mcp-rail-empty Static at zero servers; tooltip on disabled #mcp-rail-scope-ref. 6. Update column assertions in test_mcp_servers_mode.py and purpose-copy block in phase6 recovery test; run MCP tests + ruff.
<!-- SECTION:PLAN:END -->
