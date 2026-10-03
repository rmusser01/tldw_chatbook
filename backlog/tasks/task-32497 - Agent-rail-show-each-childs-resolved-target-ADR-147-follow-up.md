---
id: TASK-32497
title: 'Agent rail: show each child''s resolved target (ADR-147 follow-up)'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-12 07:55'
updated_date: '2026-09-29 19:35'
labels:
  - agents
  - console
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32477 (agent provider routing, ADR-147, final-review finding I-1). Spec §Error handling promises "The rail summary shows each child's resolved target (e.g. 'qwen-local · qwen3.8-27b')" and the user guide repeats the claim — but nothing implements it: SubAgentSummary (console_agent_bridge.py:1079-1107) carries only text/status/run_id/handle_id, and _fleet_row_from_summary/_fleet_row_from_record (UI/Console_Modules/agent.py:397-443) render no target (secondary_text=""). Plumb the resolved target (from the run row's v16 resolved_* snapshot, falling back to the live definition for legacy rows) through FleetHandle/SubAgentSummary into the rail row's secondary_text, or correct the guide sentence if the rail format is deliberately deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each live/finished child's rail line shows its resolved target (provider · model) or the guide claim is corrected
- [x] #2 Legacy (NULL-snapshot) rows render something honest (no fabricated target)
- [x] #3 Widget/pilot test pins the rendered target for a routed child
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/147-agent-provider-routing.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: Carry the existing resolved run snapshot into the existing rail row; no new ownership, persistence or styling boundary.
1. Trace child target capture and rail projections.
2. Add failing routed-child and mounted rail tests, including NULL legacy snapshots.
3. Carry frozen provider/model through FleetHandle and SubAgentSummary; render provider · model or Target unavailable in existing row secondary text.
4. Run targeted runtime/bridge/widget tests and source checks; record evidence for root review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented frozen child target display using existing FleetHandle, SubAgentSummary and InspectorSectionRow metadata. Fleet reservation copies only resolved provider/model; historical projection reads only saved resolved fields. The existing wrapped secondary text shows provider · model first, with Target unavailable for NULL/missing snapshots. Inline children publish the persisted pair through the existing model scope before the first provider call; a pending-SPAWN check prevents unrelated skill calls from borrowing another row. No schema, CSS, credentials or endpoint-URL changes.

Evidence: five initial RED failures, then five new routing/paint checks GREEN; stronger inline provider-boundary pin RED, then GREEN under private_profile_test. Final inline + four narrow/saved/legacy widget cases: 5 passed. Three directly affected full-Console usage/survivor cases: 3 passed under private_profile_test. Coordinator/routing broader targeted run: 64 passed, with three in-flight root refusal-copy failures subsequently verified by root and three profile-selection failures resolved by the private wrappers. New target test Ruff lint/format clean; existing production files retain baseline lint/format debt, with no whole-file reformat. Scoped diff whitespace check passed.

Updated the Console user guide and recorded the first-step/provider-boundary lesson in lessons-testing-evidence.md. ADR required: no; implements ADR-147 and composes existing ADR-150/161 row grammar. Status and acceptance checkboxes remain In Progress/unchecked pending root review.

Combined fallback integration caught the live/saved rail keeping original target after a fallback switch. Added exact live run target publication, associated inline summaries with real run IDs, and selected active frozen fallback target for saved rows while preserving original audit columns. Three new RED failures became five passing target paint/history checks (/private/tmp/agent-burndown-active-target-red.log, /private/tmp/agent-burndown-active-target-green.log). Service switch callback and inline actual adapter revalidation pending.

Actual Console inline regression now covers plain inheritance and explicit same-provider alternate-model fallback: 2 passed, observing the correct rail target before each child provider call. Inline rows carry their actual saved run ID for exact updates; fleet rows use exact attached-run metadata. Persistence happens before a bounded never-raise target observer publishes the new pair. The root MODEL_ERROR on_step experiment was replaced because lifecycle trace callbacks intentionally do not use the legacy step channel. Mounted/history target tests passed5. Added saved trace integrity assertion after finding duplicate context assembly records; its shared callback repair is handled in TASK32508.

Final disposition 2026-09-29: Independent target/fallback re-review approved active fallback and resumed target metadata while original audit fields remain unchanged. Its 50-case fallback/painted-target run passed in 19.90s; the actual inline fleet-off dispatch and complete saved context trace checks previously passed both cases. No remaining actionable target-display finding. All acceptance criteria are checked; scoped tests, changed-code static checks, documentation and independent review are complete. Task is Done. Inherited source formatting debt is preserved; no full-suite/live-provider result is claimed. This disposition supersedes earlier pending-review notes.
<!-- SECTION:NOTES:END -->
