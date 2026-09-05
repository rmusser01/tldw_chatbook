---
id: TASK-31667
title: Keep retained Library workspace actions in sync
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:00'
updated_date: '2026-09-05 18:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure retained Library rail actions reflect current source availability and recovery policy without replacing their widgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Available, failed and custom-policy source transitions update the retained handoff button tooltip and blocked state using existing policy derivation.
- [x] #2 Button identity and DOM structure remain unchanged across both directions of the transitions.
- [x] #3 Existing Library destination recovery assertions and focused rail regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the existing service-failure and custom-policy tooltip assertions as reproductions; add available-to-failure-to-custom-to-available retained-button identity coverage.
2. Move the existing pure handoff-state policy into Workspaces/display_state.py, retaining a thin screen adapter so initial construction and retained updates share one derivation.
3. Pass only the derived blocked/tooltip value at the three existing rail.sync_state callsites; patch the mounted button in place without widget construction or recomposition.
4. Verify initial and retained policy transitions, original Library destination failures and rail tests; measure screen ceilings and static checks.
ADR required: no
ADR path: N/A
Reason: Routine retained-projection defect repair and mechanical relocation into the existing workspace display-state boundary; no new policy, event, DOM or lifecycle contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The retained rail path refreshed source-error text but left the workspace handoff button at its initial empty-source tooltip. Preserved the original destination failure/custom-policy assertions. Moved the existing pure handoff policy into Workspaces/display_state.py and retained the screen adapter; initial composition and all three retained sync callsites now share that derivation. Rail.sync_state receives only the derived blocked/tooltip pair and patches the same mounted button, preserving its pressable-while-blocked behavior. No temporary widget construction, DOM replacement or recompose was added.
Verification: original failure/custom/taxonomy plus new identity transition test4passed. Full Library selection across both destinationfiles and rail tests51passed193deselected24.69s; workspace display-state and Library ratchets26passed3deselected1.49s. New regression proves initial availability, failure, custom policy and restored availability while preserving exact button identity. All original timeout/copy assertions remain unchanged. Scoped Ruff/format and diffcheck pass; formatter also normalized one adjacent workspace typeannotation. Parent reviewed rail/screen changes with no blocking finding. Screen41324lines/1301methods stays below unchanged ceilings.
ADR required:no; existing policy and display-state ownership, mechanically relocated. Library decomposition recipe section22 documents both retained-policy and blocking-adapter incidents. Textual testing guidance informed actual mounted identity checks.
<!-- SECTION:NOTES:END -->
