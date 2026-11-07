---
id: TASK-34900
title: Triage the consumerless Interop packages
status: To Do
created_date: 2026-10-12 17:53
labels:
- simplification
- product-decision
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
22 of 31 Interop packages (~22.4k LOC) are wired at boot and called by nothing: their only production importer is app_service_wiring.py and their scope-service attributes have zero references from UI Event_Handlers Widgets Agents or Tools. The server-parity roadmap lists the domains as deliberate so this is a product decision: keep freeze or prune and at minimum stop adding consumerless domains.

Evidence (verified on dev df4985e): Import-graph verified on dev df4985e. Consumerless packages include Kanban (4279 LOC - no UI or tool calls any operation) Prompt_Studio (1706) Auth_Account (1313) MCP_Governance (1303) Voice_Assistant (660) Meetings (694) Feedback (601) and 14 more; Evaluations core service IS used (app.py:1824). Evidence the seam was never exercised: UI/Outputs_Panel.py:75 reads a never-set attribute. Tests keeping them green: Tests/RuntimePolicy/test_unsupported_capabilities.py imports all 41 scope services. Reference: Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision recorded as an ADR or roadmap update,Per-package disposition list produced (keep with named consumer freeze or delete),Boot budget re-measured after any removals,Kanban backup-participant registration preserved or migrated,New Interop domain additions gated on a named consumer
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
