---
id: TASK-32678
title: Integrate hook input transformations and post-event barriers
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:24'
updated_date: '2026-10-01 00:10'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32677
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply structured hook effects at the real tool boundary without bypassing review or allowing later model steps to overtake required context.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transformers run once in order, final arguments are frozen, qualified guards and context handlers run, then ordinary permission review binds the actual call.
- [x] #2 Preauthorized and durable tool paths retain their existing exemptions while all controlling restrictions and fresh dispatch checks apply.
- [x] #3 Dispatched results establish required PostToolUse and known-error PostToolUseFailure checkpoints before subsequent model admission or normal settlement.
- [x] #4 Validated effect acceptance and checkpoint release are atomic; failure, cancellation, master-off or stale results cannot erase a requirement or replay settled tool work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/197-console-hook-configuration-review.md
Reason: Integrate accepted reviewed H3 tool preparation and checkpoint contracts on current dev; preserve current approval provenance, sensitive-result projections and consent owners. No new authority or context owner.
1. Read task/spec H3 and reviewed checkpoint 8192f1fb1f; trace current common agent/service/catalog dispatch and result/model/terminal boundaries. Establish targeted unchanged neighbors and exact import/profile provenance.
2. Reuse reviewed pipeline/checkpoint tests, adapted to the existing private profile and actual current API. Demonstrate public behavior RED plus successful same-entry controls before production integration.
3. Reuse reviewed tool_pipeline/checkpoints and scoped engine dependency-planning changes. Integrate through three-way task diffs, preserving newer current tool review provenance, context serialization and sensitive-result projections. Keep plugin discovery and H4/H5 lifecycle producers out.
4. Carry the reviewed host-neutral attributed text types and exact-send budget check into agent_models; qualify the existing host carry operation directly while native Plugin context wrapper integration remains its foundation task. Preserve current ToolResult approval metadata alongside honest dispatch-state provenance. Use immutable tool/schema snapshots and existing permission owners; declare required offline schema packages in core.
5. Qualify ordered transformations, frozen review identity, fresh dispatch checks, Canvas/durable/runtime/inline-skill paths, no-I/O schema validation, effect-free requirements, known/uncertain dispatch outcomes and atomic required post-event context acceptance. Gate both next model admission and final settlement without replaying settled tool effects.
6. Update H3 spec/ADR and task evidence; run exact new pipeline/checkpoint/execution files plus affected agent/catalog/tool review/Console neighbors and core packaging check. Targeted only. Authored Ruff/format/syntax, shared-file baseline parity, dependency metadata and whitespace.
7. Self-review all four ACs, record limits and actual results, close through Backlog CLI and commit separately before H4. Required plugin/MCP/native adapter dependencies follow; no full suite or automatic merge is assumed.
8. Repair the unchanged real skill-spawn fixture to retain the selected private bootstrap profile; the exact node fails against exported pre-H3 d7432e3396 source with raw_source_selection_changed. Add only Tests/Agents/test_skill_tool_spawn.py marker/import; preserve admission guards and source selection. Qualify existing worker/capacity/timeout neighbors under their intended private profile rather than bypassing recovery.
9. The same real-config fixture failure is confirmed against pre-H3 source in Tests/Agents/test_tool_worker_capacity.py. Retain its selected private profile with the existing marker only; include the exact capacity file in final qualification. No production recovery guard changes.
10. Correct the capacity test physical-resource precondition: recovery admission may outlast its 20ms caller timeout, so await each real worker-entry event before filling the next slot and keep controlled workers held until explicit finally release. The old private-profile control took 8.59s against a 10s artificial hold; the integrated run crossed that hold and no longer tested a full pool. Use a bounded 30s hold/5s startup check; retain exact 8/6 capacity and terminal ownership assertions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated reviewed H3 checkpoint 8192f1fb1f into the current tool/runtime/service boundaries. Ordered schema-validated transformations produce frozen candidates; legacy guards, context acceptance, existing permission review and fresh catalog/argument checks retain their original owners. Required postevents precede observers, model admission and terminal persistence, with owning/dependency failure scope and honest settled/uncertain/not-started provenance. Current ToolReviewDecision/approval metadata, sensitive tool-record projections, worktree tool gating and actual guarded worker execution are preserved.

Reused host-neutral attributed text and complete-send limits; standalone carry/copy/multimodal behavior is qualified directly through the host carrier. Native Plugin wrapper/graph composition remains its prerequisite task rather than importing unavailable plugin owners. Added core jsonschema/referencing declarations with offline no-retrieval schema validation. H3 only consumes an exact pinned H2 session; lifecycle producers are H4.

Baseline: initial unchanged neighbors stopped at 11 passed/1 failed because real hook admission read a switched profile; retaining the existing private bootstrap profile gives 135 passed. Public barrier and base-dependency RED: 3 failed/2 passed, with no-requirement and Pydantic controls. Core qualification: 116 passed. Existing skill/capacity fixture failures were reproduced against exported pre-H3 d7432e3396 source. Only test markers were added. The capacity fixture also crossed its artificial 10s hold; it now awaits each actual worker-entry event and retains controlled workers until explicit finally release (bounded 30s hold). No production recovery guard was bypassed.

Final frozen covering run across 18 exact affected feature/regression files: 514 passed in 88.47s, zero skips. Full Ruff/format on seven hook/new test files; Console regression file full Ruff; all changed source parses and TOML/whitespace pass. Existing shared Ruff diagnostics remain 3/20/51/10/27/30 in models/runtime/service/catalog/bridge/ConsoleRuntime; skill fixture improves 2 to 1. Exact command and evidence: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.

Self-review covers all four ACs. Existing ADR-163/162/197 apply; H3 spec and ADR163 interface notes updated. Controlled Darwin Console/SQLite/subprocess and runtime metadata qualification only. No full suite, live provider, graphical UI, native Plugin composition, Windows/Linux runtime or fresh full dependency resolution is claimed.
<!-- SECTION:NOTES:END -->
