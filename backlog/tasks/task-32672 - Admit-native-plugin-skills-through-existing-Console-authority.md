---
id: TASK-32672
title: Admit native plugin skills through existing Console authority
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:19'
updated_date: '2026-09-16 15:45'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32671
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first usable local plugin path through existing skill, tool and context services with workspace-specific eligibility.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed native skill can install disabled, enable in a named workspace and run through the existing Console skill/tool path under ordinary permissions.
- [x] #2 Stable workspace and installation generations are rechecked before injection, launch, approval acceptance and invocation; one installed revision applies across scopes.
- [x] #3 Plugin context stays attributed and untrusted with whole-block limits; manual-only and inline/fork metadata, empty tool allowlists and dependencies preserve their meaning.
- [x] #4 Owned skill listings expose package provenance while service-level standalone edit/delete/overwrite paths refuse package mutations; unrelated standalone skills remain usable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md; extends backlog/decisions/009-local-skill-trust-boundary.md
Reason: Direct implementation of accepted native plugin admission, existing Console authority and package-owned Skills projections; preserves one plugin mutation/execution owner and existing permission/runtime boundaries.
1. Read TASK-32672, task5 brief, reviewed F4 handoff and relevant ADR/lessons. Establish RED through the native plugin/Console path using the isolated worktree interpreter and real owned registry/trust/coordinator.
2. Under R20/R22, minimally extend Plugins/coordinator.py and Plugins/review.py with typed explicit trust and activation reviews through the same authenticated commit pipeline. Bind current exact package/material, installation/workspace and mapping/configuration generations. Never directly toggle rows or expose arbitrary authority replacement.
3. Implement Plugins/admission.py immutable admitted revision/selection/dependency/mapping/scope snapshots and current-authority revalidation at injection, invocation, launch and approval acceptance. Validate captured named workspace existence/archive state, distinguish global/default sentinels, and prevent namespace-only changes from cancelling unrelated work or mid-run enablements from widening a snapshot.
4. Implement Plugins/skill_provider.py and Plugins/context.py using existing Skills/ToolProvider/Console ownership, attributed untrusted whole-block context limits and exact manual-only/inline/fork/empty-tool constraints. Missing/unready dependencies remain blockers; aliases cannot shadow existing tools or other skills.
5. R3/R7 authorize minimal Chat/console_agent_bridge.py, Chat/console_chat_controller.py and app.py changes in addition to console_runtime.py, console_skill_resolver.py and Agents/tool_catalog.py. Wire the actual frozen catalog, skill runner/substitution and bundle-definition checks, ordinary permissions, captured run identity and lazy app-owned persistent non-UI worker marshalling. SQLite construction/use/close stays on its owner thread/loop; keep trust/keyring/scans off startup and retain service lifetime across navigation.
6. Integrate package-owned listing/detail provenance and service-level guards in Skills_Interop/local_skills_service.py and skills_scope_service.py. Refuse standalone edits, deletion, import overwrite and inapplicable execution/promotion against owned IDs while preserving standalone successful controls. Do not create standalone trust projections or a parallel skill runtime.
7. Test real install-disabled then explicit trust/workspace enable and Console submit/provider payload; cover scope A/B/default, invalid/archived identity, stale/tampered material, unrelated namespace advance, new capability mid-run, dependencies, manual/fork/empty tools, alias collisions and owned mutation refusal. Preserve raw transcript and actual final untrusted attributed provider input. Extend real fixture only as necessary.
8. Run exact feature and affected neighboring regression files, startup/preimport guards for changed construction seams, Ruff/Black/syntax/whitespace checks. For pre-existing large-file violations establish baseline parity and clean changed hunks under R10; no unrelated formatting/full suite.
9. Update relevant authoring/operation documentation, self-review and write task5 report with actual RED/GREEN, changed paths, API handoff and limits. Controller handles exact-path commits, independent task review, AC/Done closeout. F6 owns selective cancellation, F7 revision drain, and M4 actual plugin MCP connection ownership; preserve required captured ownership integration points without duplicating those runtimes.
10. R25: add Plugins/service.py as the lazy app-owned persistent worker and typed facade/marshalling lifetime; extend Tests/Plugins/conftest.py for the real Console stack and update Plugins/README.md. Keep admission/context/provider focused. Ensure storage scheduling does not prevent the later required immediate live seal/host cancellation while the worker is blocked.
11. R26: preserve synchronous configuration capture as an immutable worker-published eligibility maximum only. Async send acceptance revalidates current authenticated authority, material and workspace, intersects rather than widens that maximum, and binds exact accepted run identity before plugin effects. Repeat current checks at each later use/approval; cold/locked/uninitialized state is empty, not an implicit grant. Do not block UI on worker Future.result or put source bodies/secrets into persisted configuration metadata.
12. R27: document strict native string metadata context/user_invocable/disable_model_invocation/argument_hint without loosening the portable parser. Invalid recognized metadata blocks admission; empty allowed-tools stays no tools, not inheritance. Vendor spelling normalization remains its assigned adapter increment.
13. R28: add the next plugin-owned migration for persistent stable installation aliases and minimally extend Plugins/authority.py, registry.py, recovery.py, coordinator.py and review.py as needed to bind/reconstruct them. Allocate at first install from authenticated package name, suffix/check collision and retain across update/restart/recovery. Preserve old authenticated snapshot digest/canonical bytes on read with explicit backward qualification; no silent grant/retrust or prior migration edits. Add focused registry/authority/store/recovery and actual wheel-resource checks for the migration. Update ADR-162/spec authoring/identity details as needed; this implements the accepted stable identity contract, not a new authority owner.
14. R30: give each frozen logical turn a fresh pending plugin admission identity, preserving actual request.turn_id association when available. Bind pending ownership to actual root/child run IDs via existing lifecycle callbacks before dispatch/spawn; never infer from the viewed session, reuse a consumed pending ID for an unrelated retry, or bypass inherited maxima. Refusal/cancellation/teardown must retire pending custody without losing retained running work.
15. F5 review fix round1: revalidate direct-provider plugin authority after all async dispatch barriers and before chunk/result/final acceptance; capture every supported explicit workspace generation including global/default sentinels; retain real direct-provider work until actual terminal evidence rather than consumer exit. R33 permits a minimal Chat/console_provider_gateway.py adapter and targeted gateway tests only if its existing register_provider_work signal cannot be reused directly. Preserve the existing provider-work owner, trace signal privacy and cancellation semantics.
16. Controller-confirmed aggregate context gap: enforce whole-block8KiB/send32KiB at the final agent-provider boundary for plugin-origin instructions and repeated skill_file results, using host attribution and preserving standalone/user content. Add real AgentService/provider payload controls for a valid aggregate and oversized repeated file results; no helper-only assertion or lifetime-total substitute for the per-send requirement. Extend Tests/Plugins/test_native_skill_flow.py, test_admission.py and exact affected Console/gateway/skill-file tests. Record any additional necessary owner path before editing; maintain isolated fixtures and targeted tests only.
17. R34: minimally extend Agents/agent_models.py, agent_service.py and agent_runtime.py to carry host-owned typed attribution from an authenticated plugin skill_file read through ToolResult and the current live message row to the existing final Console provider check; extend exact Tests/Agents/test_skill_file_runtime_tool.py controls. Keep the sidecar ephemeral, content-free and out of durable transcript/configuration/trace projections and provider JSON. Never infer ownership from untrusted text or model-supplied metadata, and preserve existing standalone formatting/truncation semantics while plugin blocks remain whole. Verify user/standalone marker-looking text controls and per-send aggregate behavior through real AgentService payloads.
18. F5 fix round2: preserve or reject whole live plugin blocks through actual Console world-info/evidence transformations after skill substitution and before the final AgentService send. Resolve the remaining finding5 recorded in task-5-fix1-review.md; current _prepend_evidence_context string concatenation erases attribution. Extend the real Console-to-AgentService/provider test in Tests/Plugins/test_native_skill_flow.py with retrieved evidence and under/over-limit initial-plus-file-results controls. Minimal affected Tests/Chat/test_console_world_info_application.py and test_console_local_citation_boundary.py/test_console_agent_swap.py controls are allowed where their actual helpers change. Keep standalone/multimodal behavior and privacy projections intact; a formatter losing the type never converts live plugin material into historical output. Run only covering feature/helper regressions and changed-file static parity against887afe3811, then append exact RED/GREEN and final output to the same task report.
19. R35: for opaque chat-dictionary/world-info transformations after plugin substitution, retain attribution only when the result contains one exact whole copy of the host-carried input; otherwise fail this send closed. Preserve known whole prefix/suffix joins without marker parsing, ensure refusal is not swallowed by existing optional-processor fallback, retain ordinary standalone/multimodal behavior, and document the conservative embedded-user-text limitation. Add changed/duplicated/carrier-preserved controls through the actual send path as relevant.
20. Record the actual TASK-32672 producer-to-final-send testing trap briefly in backlog/docs/lessons-testing-evidence.md once the full Console regression is green: a bridge-only typed input test missed later Console string processors, and a resumed pending call needs fresh attribution distinct from historical completed output. Reuse the existing producer/reconstruction lesson section if appropriate; state observed RED/GREEN evidence and the concrete fixture boundary, not a new generic checklist.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Native packages now install disabled, receive explicit trust and scoped activation reviews, and enter the existing Console skill/tool path through one lazy PluginService worker. Immutable run ceilings, current authority/definition checks, stable authenticated aliases, retained exact root/child/provider-work ownership and package-owned listing/mutation guards preserve ordinary permission and standalone behavior. Live plugin instruction and skill-file output stay attributed through actual Console/AgentService assembly with whole-block 8 KiB and final-send 32 KiB limits; opaque edits or duplication refuse the send.

Implemented in Plugins admission/context/provider/service and the existing Console, AgentService/runtime, tool catalog, Skills services and app ownership seams; migration 003 persists stable installation aliases. ADRs: backlog/decisions/162-managed-agent-plugins.md, 163-expanded-console-hook-runtime.md and 009-local-skill-trust-boundary.md. Minimal integration scope and conservative opaque-transform behavior are recorded in the Implementation Plan and delivery ledger.

Validation: actual native install/trust/enable/Console behavior and targeted ownership, current-authority, continuation, standalone, gateway and aggregate-context controls passed. Final fix1 selections: 135 feature, 33 bridge and 16 gateway cases, plus focused warning/continuation checks (overlap documented). Final fix2 selections: 19 actual-submit/aggregate and 84 neighboring cases, 103 distinct passes. Changed-file syntax and Ruff/Black baseline parity passed with zero new debt; whitespace clean. Actual isolated wheel importlib.resources verified SQL001/002/003 against Git bytes. No full suite or remote provider run. Startup qualification retains the unchanged baseline preimport failure (515 modules/382451 LOC) and zero UI-ready headroom (973/973); no all-startup-green claim. macOS local APFS and controlled marker fixtures remain the qualified environment.

Independent task review plus two fresh scoped re-reviews cleared all five blocking findings. Out-of-scope reference-expansion interaction was recorded for final whole-branch review. F6 selective revocation, F7 drain/archived-resumption/retention and M4 connection ownership consume the recorded public handoff. Updated plugin documentation and an actual Console context-evidence testing lesson. Commits: 1dfc37a908, 887afe3811, 5dcc1f6f34.
<!-- SECTION:NOTES:END -->
