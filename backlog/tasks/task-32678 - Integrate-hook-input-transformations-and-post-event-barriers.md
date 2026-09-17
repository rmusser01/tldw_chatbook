---
id: TASK-32678
title: Integrate hook input transformations and post-event barriers
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:24'
updated_date: '2026-09-17 00:28'
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
- [ ] #1 Transformers run once in order, final arguments are frozen, qualified guards and context handlers run, then ordinary permission review binds the actual call.
- [ ] #2 Preauthorized and durable tool paths retain their existing exemptions while all controlling restrictions and fresh dispatch checks apply.
- [ ] #3 Dispatched results establish required PostToolUse and known-error PostToolUseFailure checkpoints before subsequent model admission or normal settlement.
- [ ] #4 Validated effect acceptance and checkpoint release are atomic; failure, cancellation, master-off or stale results cannot erase a requirement or replay settled tool work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/148-console-run-hooks.md
Reason: Direct implementation of the accepted common tool-preparation, permission-bound candidate and required post-event checkpoint contracts. Preserve the legacy observational callback and existing permission/runtime owners.

1. Read task11 brief, actual TASK32678 and hook-seam-research.md's H3 mapped entries and controller clarifications. Recheck final H1/R41 and H2 APIs/reports before implementation. Identify the exact common tool branch, schema/identity generation, next-input and normal terminal boundaries; notify controller before extra source/test/doc paths, including any minimal tool_catalog exposure or fleet neighbor required by the brief.
2. Establish public-entry behavioral RED in the exact pipeline/checkpoint tests. Build post_hook_case around the actual agent dispatch callback and next-model/root-settlement guards using deterministic held events. A visible result and valid same-entry success must be preconditions; an unrelated readiness helper or import failure alone cannot prove the required barrier.
3. Add hooks_v2/tool_pipeline.py and checkpoints.py. Run ordered transformers once, validate every candidate against the actual resolved tool schema, freeze final arguments and owner/definition generations, then run final validators and context handlers. Consume R41 phase/identity/matcher distinctions and preserve effect-free requirements. Use the final H2 owned synchronous facade where the existing agent loop is synchronous.
4. Wire one preparation path above runtime/provider branch fan-out and before Canvas preauthorization, permission exemptions and ordinary review. Recompute exact exemptions against final arguments; retain existing durable invocation and permission-store semantics. Revalidate the final candidate and owner/definition generations immediately before real dispatch. Hook output never supplies permission or dispatches directly.
5. Install required PostToolUse and known-error PostToolUseFailure checkpoints at actual result classification before result completion can enable another model request or normal root settlement. Keep existing optional post_tool_call behavior unchanged. Not-dispatched denials produce no tool-result hook; uncertain remote cancellation remains distinct from a known tool failure. Never replay settled tool effects.
6. Atomically accept attributed context and settle its checkpoint with current authority/generations. Cancelled, revoked, malformed, disabled or stale results cannot satisfy controlling requirements. Gate both next input and terminal persistence/notification, including a final tool with no following model loop; do not hold checkpoint/admission/fleet locks while hook work awaits. Reuse the current context owner rather than introducing a parallel ledger.
7. Run the exact four brief test files plus any controller-recorded affected runtime/fleet neighbors. Qualify ordered A/B/final-C execution, stale approval identity, Canvas and durable paths, context-only/effect-free handlers, known and uncertain errors, next-model and final-settlement races, and valid success controls. Use isolated task profiles and worktree package/data-root provenance, targeted tests only and source/test freeze during final covering runs.
8. Record public API contracts and H4/H5/H6 checkpoint/producer handoffs in task-11-report.md, Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md and backlog/decisions/163-expanded-console-hook-runtime.md. Run syntax/Ruff/Black with shared-file baseline parity and whitespace; preserve final commands/output/frozen hashes and report exact changed paths. Controller owns commit, independent review, all four ACs, final notes and Done after gates pass.

9. Extend only Agents/tool_catalog.py with the minimal public immutable current name/tool-ID/source/schema/catalog-generation snapshot needed by H3, captured consistently against the existing catalog lock/generation and rechecked immediately before dispatch. Do not expose mutable schema aliases or introduce another catalog/permission owner; preserve current nominal provider identity/classification. Qualify changed generation/schema refusal and same-entry valid dispatch in the two approved new pipeline/checkpoint test files, together with actual service/fleet/model/terminal races there. Record this additional source in final hashes and shared-file static baseline parity. Existing attributed context carriers and host-injected publication remain authoritative; notify controller before any separate context-owner/type path change. This directly implements existing ADR163/162 and AC1-4, no new ADR.

10. Extend Chat/console_runtime.py only with a read-only accessor to the already-pinned exact-session H2 engine under its existing lock; H3 must not create/reconfigure sessions, omit retained requirements because an engine is sealed/disabled, or add lifecycle producers owned by H4. Extend Agents/hooks_v2/engine.py with minimal host dependency-status/phase admission access bound to its existing engine-issued event scope. Capture dependency status and fixed failure once in bounded engine-controlled per-scope state and reuse it for planning, execution and resulting failures; a caller-supplied false Boolean must not erase a requirement. Preserve exact event/handler/issuer identity, original deadlines, no global scope registry/new requirement owner, current-authority checks and explicit versus dependency-only failure scopes. Add corresponding regression coverage to Tests/Agents/test_hooks_v2_execution.py and include the full execution file in final targeted coverage; prove raise-then-false planning/execution cannot downgrade or turn into an optional notification, with valid stable controls and foreign/closed-scope refusal. Record exact API in the already-approved spec/ADR/report; direct existing ADR163/162 implementation, no new ADR.

11. Apply R48 from h3-dependency-ruling.md: separate owning-event and dependency-only checkpoint queries with exact host-resolved required handler IDs; empty selection is only positively independent/standalone work, not a missing-resolver fallback. Normal settlement still joins all pending required post-events, then preserves failure scope. Current F5 graph excludes hook-dependent components; I1 owns actual graph composition, M4/H6 consumers and I7 qualification. Preserve requirements on unknown mapping; no Plugins import or duplicate graph owner. Qualify the dependency/independent/explicit-required/unknown/terminal matrix in approved tests and amend existing hook spec/ADR163 interface notes before code reliance.

12. Apply R49 from h3-dispatch-provenance-ruling.md by adding only minimal optional host dispatch_state provenance to Agents/agent_models.py ToolResult, with actual existing service/runtime/catalog owners marking not_started, settled or uncertain. No payload-string inference or untrusted result-body authority. Preserve local refusal versus completed error versus unresolved post-start cancellation, existing custody and legacy callbacks; never replay. Qualify actual wrapper/dispatch controls in approved tests and run relevant unchanged timeout-wall-clamp, tool-call-abandon and execution-capacity neighbors. Record any extra neighbor edit before change, include agent_models.py in frozen/static baseline evidence, and amend existing spec/ADR163 interface notes before implementation reliance.

13. Apply R50 from h3-context-carrier-ruling.md: extend existing PluginContextText with distinct typed hook origins and preserve both via existing host carry/transform operations; no fake installation ID or parallel string/ledger owner. Add the minimal shared host-neutral final-send budget check in existing agent_models.py and delegate from Plugins/context.py while retaining its compatibility error surface. Enforce hook4KiB/block/16KiB/event and package8KiB/block/shared32KiB/send, counting present blocks once per occurrence, through text/multimodal/copy/transform paths. Required acceptance failure cannot clear checkpoints. Additional exact source Plugins/context.py and affected Tests/Plugins/test_native_skill_flow.py/Tests/Agents/test_agent_models.py are recorded before edits; qualify actual standalone/plugin/mixed final requests and legacy behavior, frozen/static parity and accepted hook spec/ADR notes. Deferred F5 unrelated reference-expansion issue stays outside scope.

14. Apply R51 from h3-schema-reference-ruling.md in the already-approved pipeline and test paths: candidate schema validation uses a no-I/O local registry and refuses missing external references without dropping constraints or exposing URI/payload text. Preserve internal references and ordinary schema controls; qualify public preparation with a retrieval spy so no real network/filesystem access occurs. Installed jsonschema default remote retrieval was confirmed by source inspection; no fetch was attempted. Use current dependency APIs, no new retrieval owner/dependency. Document final API/unsupported boundary in existing hook spec/ADR/report and include frozen covering/static evidence.

15. Apply R52 from h3-runtime-dependencies-ruling.md: add exact additional paths pyproject.toml and Tests/Packaging/test_hooks_v2_dependencies.py before changes. Declare jsonschema>=4.26,<5 and directly imported referencing>=0.37,<1 in core dependencies, remove the obsolete dev-only jsonschema declaration/comment, preserve unrelated pins/extras, and amend existing ADR163/spec dependency notes. Qualify real base distribution metadata without extra markers and the built wheel's isolated no-network import/validation using available controlled dependencies; distinguish this from a fresh full dependency resolution. Capture the pre-change metadata regression and positive control, include the packaging test in frozen targeted coverage and all added paths in hashes/static checks. No package install, new validator/retrieval owner, or optional validation bypass.

16. Address initial review Important I1 in the existing runtime/service and approved H3 test paths: assign honest not_started provenance to common runtime and inline-skill refusals before their dispatch callbacks execute, retaining settled/uncertain semantics after actual host invocation. Inspect other pre-dispatch exits in the same common runtime fan-out and observer boundary; never infer child-start state from result text. Add actual loop/service refusal and successful controls in Tests/Agents/test_hooks_v2_tool_pipeline.py and/or test_hooks_v2_post_checkpoints.py. Final covering is those two files plus Tests/Agents/test_post_tool_dispatch_hook.py, Tests/Chat/test_console_run_hooks_regressions.py, Tests/Agents/test_agent_runtime.py and Tests/Agents/test_skill_tool_spawn.py; additional unchanged neighbors only if changed behavior warrants them, and any extra edited path must be recorded first. Freeze amended source/tests, preserve RED/GREEN and full covering/static output with shared baseline parity and updated hashes; append Fix1 report. Deferred Minor M1/M2 stay final-branch triage; actual later-owner integration/platform limits remain explicit. Existing ADR163/R49/AC3 apply, no new ADR.
<!-- SECTION:PLAN:END -->
