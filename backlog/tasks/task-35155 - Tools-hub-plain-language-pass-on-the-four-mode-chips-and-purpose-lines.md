---
id: TASK-35155
title: 'Tools hub: plain-language pass on the four mode chips and purpose lines'
status: Done
created_date: 2026-10-16 20:15
assignee:
- '@ynehriver'
updated_date: 2026-10-21 08:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec amendment A7 item 3: Servers/Tools/Permissions/Audit name protocol artifacts; users ask what can my agents do, what has to ask me first, what did they do. Rename the chips and purpose lines under the TASK-35018/35019 copy rules. This changes UI/MCP_Modules copy, so it needs its own brainstorm before changing anything.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Brainstorm session recorded before copy changes,Four chips and purpose lines pass the copy rules (Audit renamed to History; Servers/Tools/Permissions kept -- each already reads cold, rationale recorded in the plan),Workbench tests updated for new labels
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Copy-only pass under the TASK-35018/35019 copy rules (backlog/docs/plan-2026-10-14-settings-plain-language-copy.md: labels in plain language; no jargon the user cannot decode in place; one truth with the surface's own vocabulary).

1. Copy rules read: plan-2026-10-14-settings-plain-language-copy.md §'The Settings copy rule' rules 1-6 (plain labels; technical identifiers only inside closed disclosures; calm sentences; no roadmap/ADR ids).
2. Inventory (source strings naming a MODE): MCP_HUB_MODES labels (mcp_workbench.py), _MODE_TOOLTIPS + #tools-hub-purpose + category summary/boundary copy (settings_tools_hub.py), cross-mode copy in mcp_tools_mode.py ('Go to Servers mode...'), mcp_permissions_mode.py ('...in Servers mode to configure their permissions.'), mcp_inspector.py (VIEW_DETAILS/OPEN_TOOL_CATALOG/OPEN_AUDIT/OPEN_CREDENTIALS tooltips, audit 'Open tool'/'Adjust permission' tooltips, 'Select a tool in Tools mode first.' toast), mcp_workbench.py ('Select a tool in Tools mode first.' notify), Agents/builtin_tool_gate.py (gate breadcrumb naming 'Tools mode'). Footer hints (TOOLS_HUB_MODE_SHORTCUTS) and the first-run wizard tools step name no modes -- unchanged. Binding labels on ToolsHubPanel.BINDINGS follow the chips (show=False, keys/actions unchanged).
3. BRAINSTORM -- three schemes, width budget checked: chip = max(label+2 padding, min-width 10); today Servers=10 Tools=10 Permissions=13 Audit=10, row=43 cells. task-35151 is NOT in this base, so the row must not grow.
   - Scheme A 'Question words': Connections / Capabilities / Approval rules / History. Pros: chips echo the spec's three user questions nearly verbatim. Cons: 'Connections' forks the vocabulary (surface says 'add server', 'No servers configured', 'Import servers'); 'Capabilities' is abstract and drops the concrete word every other sentence uses (tool counts, 'Test tool', 'No tools discovered yet'); 'Approval rules' is 14 chars -- over the ~12-char budget and widens the row to 46. REJECTED: breaks one-truth and the width budget.
   - Scheme B 'Outcome nouns': Servers / 'What agents get' / Permissions / Activity. Pros: maximal question alignment. Cons: 'What agents get' is 15 chars, not chip-shaped at any width we can spend; 'Activity' implies a live stream while the canvas is a reviewable log with findings drill-down. REJECTED: chip length and accuracy.
   - Scheme C 'Rename only what does not read cold' (CHOSEN): Servers / Tools / Permissions / History. Each kept name is the surface's own concrete noun and decodes in place; 'Audit' is the one protocol/process word users must translate -- the canvas is literally a log of what tool calls did, and the app already calls it 'execution history' (one truth). Row stays 10+10+13+10=43. The three user questions move into the purpose line and the tooltips, where a sentence can answer them.
   - Naming table (old -> new): Servers->Servers KEEP (rows are servers; 'add server' hint, 'No servers configured' empty state, import copy all say servers); Tools->Tools KEEP (the mode lists tools -- the category's own word; 'Capabilities' would fork vocabulary); Permissions->Permissions KEEP (reads cold, universal app word; tooltip already states Allow/Ask/Off); Audit->History RENAME (accounting jargon for a log; 7 chars, width-neutral).
4. Implement scheme C: MCP_HUB_MODES['audit'].label='History'; BINDINGS labels follow; _MODE_TOOLTIPS audit entry renamed + tooltips sharpened to answer the mode's question; #tools-hub-purpose rewritten to carry the three questions (one line, <=112 chars, 'most people never need to change anything here.' kept); category summary + boundary copy say 'history'; every cross-mode string that names a mode updated ('Audit mode'->'History mode' etc.), keys/ids/routes untouched.
5. Update pinned tests: tooltip pins (test_destination_shells.py), purpose-line pins (test_destination_shells.py, test_mcp_compact_readability.py, test_product_maturity_phase6_recovery_docs.py), toast/breadcrumb pins (test_mcp_workbench.py, test_mcp_permissions_mode.py, test_mcp_compact_readability.py, test_builtin_tool_gate.py).
6. Verify: mode/chip-sync/purpose tests + paint tests at 80/170 if present; module-size ratchet for mcp_workbench.py must not worsen (pre-existing red 6891 vs 6760 -- copy swap must be line-neutral).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Copy-only pass, scheme C (rename only what does not read cold). Chips: Servers/Tools/Permissions kept (each is the surface's own concrete noun -- 'add server', 'No servers configured', tool counts, Allow/Ask/Off all use these words in place; the copy rules ban jargon, not plain nouns), Audit -> History (accounting jargon for a log; the app already calls it 'execution history'; chip row width unchanged: max(7+2,10)=10 = Audit's). Purpose line now answers the three user questions: 'What your agents can use, what asks first, and what they did - most people never need to change anything here.' Tooltips sharpened per mode; category summary + boundary copy follow ('history'); inspector 'Open audit'->'Open history' and 'Switch to Audit mode.'->'Switch to History mode.'; binding label 'Audit'->'History' (show=False; keys/actions/ids unchanged). Mode keys, button ids, view-state values, route/context values untouched. Guide quotes updated (TASK-35019 as-renders rule): Docs/User_Guide/mcp.md subtitle quote + 3 'Audit mode'->'History mode' mentions, agent-runs-and-tools.md link text. mcp_workbench.py edit is line-neutral (6891 -> 6891; ratchet pre-existing red not grown).

Test evidence (worktree .worktrees/tools-hub-followups, .venv/bin/python): test_mcp_workbench.py 348 passed / 1 flake (test_selecting_different_profile_while_armed_disarms passes standalone); test_mcp_inspector.py 295 passed; test_mcp_permissions_mode.py 62 passed; test_mcp_gate_label_layout.py (paint) 8 passed; test_destination_shells.py tools-hub subset (-k 'tools_hub or purpose or mode_chip') 17 passed. Pre-existing red verified identical on stashed baseline vs this change (same failure sets, diff empty): test_destination_shells.py full (65 env failures, RecoveryRequired bootstrap), test_settings_tools_hub.py (5), test_settings_footer_hints.py (7), test_product_maturity_phase6_recovery_docs.py (1), test_latest_dev_core_app_usability_smoke.py (3), test_mcp_compact_readability.py (6, stale #mcp-purpose pins from the retired standalone screen). Docs contracts: test_mcp_documentation_contract.py + test_user_guide_conflict_markers.py 76 passed. No ADR needed: copy-only change inside an existing surface, no boundary/storage/contract decision.

Follow-up review (2026-10-20): the plan's step-5 test-file updates proved unnecessary — no stale pins existed; the reviewer re-ran all four step-5 suites green.

Files: tldw_chatbook/UI/MCP_Modules/mcp_workbench.py, tldw_chatbook/UI/MCP_Modules/mcp_inspector.py, tldw_chatbook/UI/Screens/settings_tools_hub.py, Tests/UI/test_destination_shells.py, Tests/UI/test_product_maturity_phase6_recovery_docs.py, Tests/UI/test_latest_dev_core_app_usability_smoke.py, Docs/User_Guide/mcp.md, Docs/User_Guide/console/agent-runs-and-tools.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tools-hub mode chips and purpose lines now read in plain language (Audit renamed to History; Servers/Tools/Permissions kept), with tooltips, cross-mode copy, and user-guide quotes updated to match — copy-only; keys, ids, and routes untouched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
