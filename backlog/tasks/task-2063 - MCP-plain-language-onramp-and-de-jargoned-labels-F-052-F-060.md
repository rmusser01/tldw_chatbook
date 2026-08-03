---
id: TASK-2063
title: 'MCP: plain-language onramp and de-jargoned labels (F-052, F-060)'
status: Done
assignee: []
created_date: '2026-08-03 16:24'
updated_date: '2026-08-03 17:17'
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
- [x] #1 mcp_screen purpose line gains a one-line plain-language explainer
- [x] #2 Profile form UI labels renamed: Profile id to Name and Transport to Connection (internal identifiers unchanged)
- [x] #3 Rail shows an empty state at zero servers
- [x] #4 Disabled 'No scope entities' Select has a tooltip explaining why
- [x] #5 Existing tests asserting old strings updated; MCP tests passing plus ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI label/copy-only; internal ids and payload keys unchanged). Steps: 1. RED tests: profile form label is 'Name' not 'Profile id'; overview table columns use 'Connection'; MCP screen has plain-language explainer line; rail shows empty state at zero servers; disabled scope-ref Select carries a why-tooltip. 2. mcp_screen.py: add #mcp-purpose-explainer line. 3. mcp_profile_form.py: 'Profile id' -> 'Name'. 4. 'Transport' -> 'Connection' in mcp_servers_mode.py (_TABLE_COLUMNS, detail line) and mcp_server_mutations.py (form label, required error). 5. mcp_rail.py: #mcp-rail-empty Static at zero servers; tooltip on disabled #mcp-rail-scope-ref. 6. Update column assertions in test_mcp_servers_mode.py and purpose-copy block in phase6 recovery test; run MCP tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: copy/label-only changes, no internal identifier or payload-key changes. (1) mcp_screen.py: added #mcp-purpose-explainer Static ('MCP lets chatbook use external tools — most people never need to change anything here.') under the untouched purpose line. (2) mcp_profile_form.py: 'Profile id' label -> 'Name' (input id mcp-form-id, payload key profile_id unchanged). (3) 'Transport' -> 'Connection' across the MCP hub's user-facing labels: mcp_servers_mode.py _TABLE_COLUMNS and the server-record detail line, mcp_server_mutations.py form label and its 'Connection is required.' validation error. (4) mcp_rail.py: #mcp-rail-empty Static ('No servers yet — Add server to connect one.', dimmed via DEFAULT_CSS) at zero servers; disabled #mcp-rail-scope-ref Select gets a tooltip ('No scope entities to pick for this scope — the select stays disabled until one exists.'). Tests: new test_form_labels_use_plain_language (profile form), test_rail_shows_empty_state_at_zero_servers, test_disabled_scope_ref_select_explains_why; updated column assertions in test_mcp_servers_mode.py and explainer assertion in the phase6 recovery test. TDD: 6 tests RED before implementation. Verification: 428 passed across all Tests/UI MCP files; 311 passed (workbench + visual parity incl. the two MCP geometry tests, confirming the extra header row breaks nothing); ruff clean on all touched files (one pre-existing F841 in test_mcp_rail.py at HEAD left alone — outside edited ranges, fails identically on unmodified file). ADR: not required (UI copy/labels only). Not done: MCP/server.py's developer-facing 'Transport {t} not implemented' exception left as-is (protocol term, not hub UI); no restyle of the purpose line itself; commit a7dc7e46b.
<!-- SECTION:NOTES:END -->
