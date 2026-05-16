# Phase 4.7 Agent Execution QA Closeout

Date: 2026-05-16
Branch: `codex/phase4-7-agent-execution-qa-closeout`
Backlog task: TASK-11.7

## Scope

Close Product Maturity Phase 4 after replaying the agent configuration and execution surfaces that were changed in TASK-11.2 through TASK-11.6:

- Personas local behavior selection and Console handoff readiness.
- Skills local Agent Skills validation and selected-skill Console handoff.
- MCP server-first source scope and action readiness.
- ACP runtime/session blocked and handoff states.
- Schedules and Workflows run-control recovery states.

This closeout does not add server parity, ACP runtime launch, real retry/pause/resume services, or full workflow execution. Those remain Phase 5/server-parity risks.

## Workflow Matrix

| Surface | Goal | Steps replayed | What worked | Blockers / residual risk | Severity | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Personas | Select a local behavior target and stage it into Console. | Mounted Personas with local character/persona fixture, selected character/profile targets, pressed Attach to Console. | Local character/profile targets expose selected metadata, readiness copy, and Console handoff payloads. | Full character/persona runtime launch and server parity remain future work. | P2 residual | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-2-personas-runtime-launch.md`; `Docs/superpowers/qa/product-maturity/phase-4/personas-selected-polish-2026-05-12.png` |
| Skills | Validate local Agent Skills and attach only a valid selected skill. | Mounted Skills with valid and invalid local skill fixtures, selected valid skill, inspected invalid metadata state, pressed Attach to Console. | Valid `SKILL.md` records are selectable, invalid records expose validation errors, and Console receives only the selected valid skill context. | Import and script execution are intentionally not wired in Phase 4. | P2 residual | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-3-skills-attach-validation.md`; `Docs/superpowers/qa/product-maturity/phase-4/skills-valid-invalid-2026-05-12.png` |
| MCP | Confirm MCP remains server-first with scoped tools and honest action readiness. | Mounted MCP and Unified MCP panel, switched local/server scope in focused tests, inspected empty/policy-blocked action states. | Servers, scoped tools, permissions, audit readiness, action payload readiness, and route aliases remain visible. | Runtime server management and policy editing depth remain Phase 5/server-parity work. | P2 residual | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-4-mcp-source-scope.md`; `Docs/superpowers/qa/product-maturity/phase-4/mcp-source-scope-final-real-viewport-2026-05-14.png` |
| ACP | Distinguish missing runtime, configured runtime, and session handoff states. | Mounted ACP runtime/session fixtures, verified disabled Launch/Follow states, verified session payload can stage Console live work. | ACP owns runtime setup copy; missing runtime and no-session states are blocked honestly; session payloads hand off through the shared Console seam. | Actual ACP runtime launch is not implemented. | P1 accepted residual for Phase 5 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-5-acp-runtime-session.md`; `Docs/superpowers/qa/product-maturity/phase-4/acp-runtime-session-2026-05-14.png` |
| Schedules | Show selected/empty run state with recoverable control copy. | Mounted empty state and failed/pending active-work fixtures, verified selected-run detail, disabled retry/pause/approval controls, and Console follow seam. | Empty and failed/pending states expose state, next action, disabled reason, and Console follow where context exists. | Real retry/pause/resume services are not implemented. | P1 accepted residual for Phase 5 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-15-phase-4-6-schedules-workflows-run-control.md`; `Docs/superpowers/qa/product-maturity/phase-4/schedules-run-control-2026-05-15-polish.png` |
| Workflows | Show selected/empty workflow-run state with recoverable approval/control copy. | Mounted empty state and pending-approval active-work fixture, verified selected-run detail, disabled approval controls, and Console launch seam. | Empty and pending-approval states expose state, next action, disabled reason, and Console launch where context exists. | Real approval-review/run-control services are not implemented. | P1 accepted residual for Phase 5 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-15-phase-4-6-schedules-workflows-run-control.md`; `Docs/superpowers/qa/product-maturity/phase-4/workflows-run-control-2026-05-15-polish.png` |

## Screenshot Evidence

Actual running-app or textual-web screenshot evidence exists for every visible screen changed during Phase 4:

- Personas: `Docs/superpowers/qa/product-maturity/phase-4/personas-selected-polish-2026-05-12.png`
- Skills: `Docs/superpowers/qa/product-maturity/phase-4/skills-valid-invalid-2026-05-12.png`
- MCP: `Docs/superpowers/qa/product-maturity/phase-4/mcp-source-scope-final-real-viewport-2026-05-14.png`
- ACP: `Docs/superpowers/qa/product-maturity/phase-4/acp-runtime-session-2026-05-14.png`
- Schedules: `Docs/superpowers/qa/product-maturity/phase-4/schedules-run-control-2026-05-15-polish.png`
- Workflows: `Docs/superpowers/qa/product-maturity/phase-4/workflows-run-control-2026-05-15-polish.png`

## Regression Evidence

Initial broad closeout replay before this closeout patch:

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase4_agent_execution_plan.py Tests/UI/test_destination_shells.py --tb=short
```

Result: `5 failed, 95 passed, 7 warnings in 446.46s`.

Failure triage:

- `test_phase4_agent_execution_plan_splits_parent_into_reviewable_child_tasks`: expected because TASK-11.7 was moved from `To Do` to `In Progress` for this closeout.
- `test_personas_selected_persona_profile_updates_console_handoff_target`: did not reproduce in isolation; the test passed by itself in `66.23s`, indicating suite-order/timing sensitivity rather than a broken Personas workflow.
- `test_library_destination_lists_local_source_snapshot_from_services`: root cause was a fixed `pilot.pause(0.2)` in the test. The test now uses `_wait_for_library_snapshot`.
- MCP route-boundary failures: root cause was stale expected copy, `tools and servers`, after approved MCP copy changed to `MCP servers, scoped tools`.

Focused isolated replay:

```bash
python -m pytest -q Tests/UI/test_destination_shells.py::test_personas_selected_persona_profile_updates_console_handoff_target --tb=short
```

Result: `1 passed, 1 warning in 66.23s`.

Focused regression replay after the closeout patch:

```bash
python -m pytest -q Tests/UI/test_destination_shells.py::test_watchlists_collections_initial_load_uses_distinct_loading_copy Tests/UI/test_destination_shells.py::test_personas_destination_does_not_enqueue_retry_while_blocking_snapshot_runs Tests/UI/test_destination_shells.py::test_library_destination_labels_plain_list_notes_as_sample_snapshot Tests/UI/test_destination_shells.py::test_library_destination_lists_local_source_snapshot_from_services --tb=short
```

Result: `4 passed, 1 warning in 11.69s`.

Final closeout replay after the closeout patch:

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase4_agent_execution_plan.py Tests/UI/test_destination_shells.py --tb=short
```

Result: `100 passed, 1 warning in 51.15s`.

Diff hygiene:

```bash
git diff --check
```

Result: passed with no output.

## Findings

- P0 blockers: none found in Phase 4 surfaces during closeout.
- P1 issues fixed in Phase 4: false-ready or contradictory run-control copy for Schedules and Workflows, missing ACP runtime/session ownership copy, MCP action readiness ambiguity, and missing selected-target validation for Personas/Skills.
- P1 accepted residuals: ACP runtime launch, real Schedules retry/pause/resume, and real Workflows approval/run-control services are explicitly deferred to Phase 5 server parity/live integrations.
- P2 residuals: deeper server parity, skill import/execution sandboxing, MCP server-management depth, and direct runtime launch are tracked as later workflow-depth work.

## Closeout Decision

Phase 4 is verified for local agent configuration and execution control surfaces. The app is usable for the Phase 4 target workflows as honest local/blocked/readiness surfaces, and remaining runtime/server-parity work is explicit rather than hidden behind enabled controls.
