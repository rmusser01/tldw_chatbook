---
id: TASK-32499
title: >-
  Agent-routing polish bundle: unnamed-refusal loop counter, error level
  surfacing, helper placement (ADR-147 follow-up)
status: Done
assignee:
  - '@codex'
created_date: '2026-09-12 07:55'
updated_date: '2026-09-29 19:28'
labels:
  - agents
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32477 (agent provider routing, final-review minors M-1/M-2/M-3). Three small items, one PR: (1) agent_runtime.py:1473-1474 increments the loop's secondary spawn counter on `result.ok or not agent_name`, so refused UNNAMED spawns burn the loop bound and after max_subagents refused generic spawns, legal spawns see "budget exhausted" with zero children admitted (fail-closed, turn-bounded, pre-existing shape — but the routing feature made refusals common; the authoritative service counter is untouched). (2) RoutingError.level (override/preset/default) exists but is not surfaced in refusal strings (agent_service.py SpawnAdmissionRefusal f"[{code}] {err}"; settings panel f"{label} -> [{code}] {exc}") — spec says each error names the failing level; include it, which also softens the inherit-readiness UX (an inherit refusal then reads as "the parent's provider just became unready"). (3) parse_params_text lives in Widgets/settings_agents_panel.py and is imported by Widgets/Console/console_endpoint_template_modal.py — move it to Chat/sampling_params.py or a forms helper and update both import sites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refused unnamed spawns do not consume the loop's secondary spawn bound (test pinned)
- [x] #2 Refusal messages surface the failing level (override/preset/default/inherit) in both spawn and settings-panel paths
- [x] #3 parse_params_text moved to a shared module; both widgets import it; suites green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the existing unnamed-admission refusal regression. 2. Pin routing error levels in spawn and Settings reports, move the unchanged params parser into the existing shared module. 3. Run targeted routing, Settings and endpoint tests and static checks. ADR required: no; ADR path: backlog/decisions/147-agent-provider-routing.md; reason: direct repair of existing routing and shared sampling contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the existing runtime already excludes admission refusals from the secondary spawn bound; expanded the regression to named and unnamed refusals at max_subagents=1 without adding another counter. Spawn and Settings refusal strings now include RoutingError.level. Moved the unchanged params parser into Chat/sampling_params.py and imported it from both editors. Focused refusal/parser/report checks passed 8; full affected Settings routing and sampling selection passed 20. The Settings config fixture now preserves its already-bound bootstrap source and restores original bytes. ADR check: direct repair under ADR-147, no new ADR. Evidence: /private/tmp/agent-burndown-routing-polish-green.log and /private/tmp/agent-burndown-routing-editor-green.log. Independent review pending.

Final disposition 2026-09-29: independent read-only implementation review approved the scoped repair with no actionable findings. The targeted acceptance checks and changed-line static checks recorded above pass; inherited whole-file lint/format debt remains outside this correctness task. All acceptance criteria are checked and this task is Done. No full-suite or live-provider qualification is claimed. This disposition supersedes earlier pending-review notes.
<!-- SECTION:NOTES:END -->
