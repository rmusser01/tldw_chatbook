# Phase 4 Agent Configuration And Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Personas, Skills, MCP, ACP, Schedules, and Workflows understandable and controllable as agent execution surfaces that can hand off to Console.

**Architecture:** Keep Phase 4 contract-first and screen-native: each destination owns its local state, honest blocked states, and Console handoff seam without pretending full server parity exists. Build on the existing destination workbench, scope services, Home active-work adapter, and Console live-work contracts instead of introducing parallel runtime registries. Visible UI changes require actual screenshot evidence before approval.

**Tech Stack:** Python 3.12, Textual, pytest, Backlog.md, existing destination screens, `CharacterPersonaScopeService`, `SkillsScopeService`, `UnifiedMCPPanel`, Home active-work adapter, Console live-work handoffs, and product-maturity QA docs.

---

## Source Of Truth

- Product roadmap: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Parent backlog task: `TASK-11`
- Child execution slices: `TASK-11.1` through `TASK-11.7`
- Existing UX contracts:
  - `Docs/superpowers/specs/2026-04-19-characters-persona-runtime-alignment-vertical-design.md`
  - `Docs/superpowers/specs/2026-04-21-unified-mcp-control-plane-parity-design.md`
  - `Docs/superpowers/specs/2026-05-01-local-skills-kanban-parity-design.md`
  - `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
  - `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- Existing screen QA evidence:
  - `Docs/superpowers/qa/product-maturity/screen-qa/personas/`
  - `Docs/superpowers/qa/product-maturity/screen-qa/skills/`
  - `Docs/superpowers/qa/product-maturity/screen-qa/mcp/`
  - `Docs/superpowers/qa/product-maturity/screen-qa/acp/`
  - `Docs/superpowers/qa/product-maturity/screen-qa/schedules/`
  - `Docs/superpowers/qa/product-maturity/screen-qa/workflows/`

Use the repo virtualenv from any worktree:

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

## Scope

Included:

- Personas can select a local character/persona, expose readiness and policy state, and stage usable Console context.
- Skills can list local Agent Skills, validate `SKILL.md` readiness, and stage a selected skill into Console without mutating `~/.codex/skills`.
- MCP keeps the Unified MCP server-first control plane, with tools/actions under servers and explicit local/server readiness.
- ACP owns runtime/session setup and shows honest blocked states until an ACP-compatible runtime exists.
- Schedules and Workflows expose selected run state, approvals/retry/follow readiness, and Console handoff from active work.
- Home and Console remain the cross-loop control surfaces; destination screens should not fork alternate launch paths.
- Each visible UI slice includes actual screenshot QA when the layout changes.

Excluded:

- Full `tldw_server2` parity or sync engine implementation. That remains Phase 5.
- Full ACP runtime implementation if the runtime contract is not available.
- Flattening MCP into generic Tools or Settings.
- Moving ACP setup into Settings beyond global defaults.
- Rewriting `ChatWindowEnhanced` or changing Console internals outside the handoff/readiness seams needed by this phase.
- Broad visual redesign without evidence from mounted screenshots.

## File Structure

### Create

- `Docs/superpowers/qa/product-maturity/phase-4/README.md`
  - Phase 4 QA index and screenshot policy.
- `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-agent-execution-planning.md`
  - Planning-gate evidence once this plan PR lands.
- `Tests/UI/test_product_maturity_phase4_agent_execution_plan.py`
  - Regression coverage for roadmap, plan, and child task tracking.
- Backlog child tasks:
  - `backlog/tasks/task-11.1 - Phase-4.1-Agent-execution-baseline-and-contracts.md`
  - `backlog/tasks/task-11.2 - Phase-4.2-Personas-runtime-launch-and-Console-context.md`
  - `backlog/tasks/task-11.3 - Phase-4.3-Skills-attach-validation-and-local-execution-contract.md`
  - `backlog/tasks/task-11.4 - Phase-4.4-MCP-source-scope-and-action-readiness.md`
  - `backlog/tasks/task-11.5 - Phase-4.5-ACP-runtime-session-contract.md`
  - `backlog/tasks/task-11.6 - Phase-4.6-Schedules-and-Workflows-run-control.md`
  - `backlog/tasks/task-11.7 - Phase-4.7-Agent-execution-QA-closeout.md`

### Modify

- `backlog/tasks/task-11 - Product-Maturity-Phase-4-Agent-Configuration-And-Execution.md`
  - Mark parent in progress and point to child execution sequence.
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Track Phase 4 plan, QA path, child tasks, and residual risks.
- Destination screens as their child tasks execute:
  - `tldw_chatbook/UI/Screens/personas_screen.py`
  - `tldw_chatbook/UI/Screens/skills_screen.py`
  - `tldw_chatbook/UI/Screens/mcp_screen.py`
  - `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
  - `tldw_chatbook/UI/Screens/acp_screen.py`
  - `tldw_chatbook/UI/Screens/schedules_screen.py`
  - `tldw_chatbook/UI/Screens/workflows_screen.py`
- Services and state seams as needed:
  - `tldw_chatbook/Character_Chat/character_persona_scope_service.py`
  - `tldw_chatbook/Skills_Interop/local_skills_service.py`
  - `tldw_chatbook/Skills_Interop/skills_scope_service.py`
  - `tldw_chatbook/Home/active_work_adapter.py`
  - `tldw_chatbook/Chat/console_live_work.py`

## Risk Controls

- Do not mark a Phase 4 child task Done unless the mounted app was QA-walked and the task file records the evidence path.
- Do not rely on rendered SVGs, screenshots of mockups, or code-only layouts for UI approval. Capture the actual running screen.
- Avoid fixed `pilot.pause()` sleeps in new UI tests; wait for selectors, state text, or enabled controls.
- Keep route IDs stable.
- Keep blocked states honest: disabled controls must say why and what the user can do next.
- Keep server parity out of Phase 4 unless an existing local adapter already exposes the seam.
- Edit source TCSS only; do not hand-edit generated CSS.
- Prefer selector/action-state assertions over brittle full-paragraph copy checks.

---

### Task 1: Phase 4.1 Agent Execution Baseline And Contracts

Backlog: `TASK-11.1`

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-agent-execution-planning.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-4/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Test: `Tests/UI/test_product_maturity_phase4_agent_execution_plan.py`

- [ ] **Step 1: Write the failing tracking regression**

Add assertions that `TASK-11.1` through `TASK-11.7`, this plan, and the Phase 4 QA path are present in the roadmap and backlog.

- [ ] **Step 2: Run the red test**

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_phase4_agent_execution_plan.py --tb=short
```

Expected: FAIL because the plan and child tasks do not exist yet.

- [ ] **Step 3: Add planning files and child tasks**

Create the child task files, update `TASK-11`, add Phase 4 QA README, and update the roadmap.

- [ ] **Step 4: Run the focused tracking test**

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_phase4_agent_execution_plan.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md Docs/superpowers/qa/product-maturity/phase-4 Docs/superpowers/trackers/product-maturity-roadmap.md backlog/tasks/task-11* Tests/UI/test_product_maturity_phase4_agent_execution_plan.py
git commit -m "docs: plan phase 4 agent execution"
```

### Task 2: Phase 4.2 Personas Runtime Launch And Console Context

Backlog: `TASK-11.2`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/Character_Chat/character_persona_scope_service.py` only if a missing service contract is proven
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/Character_Chat/test_character_persona_scope_service.py`
- QA: `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-2-personas-runtime-launch.md`

- [ ] **Step 1: Add red tests for selected persona Console readiness**

Cover local snapshot loaded, selected character/persona ID, attach button state, policy-denied state, and Console handoff payload.

- [ ] **Step 2: Implement minimal readiness and handoff fixes**

Use the existing `character_persona_scope_service`; do not create a second persona registry.

- [ ] **Step 3: Capture actual screenshots**

Use textual-web/CDP or the approved terminal screenshot flow. Record baseline, fixed selected state, and blocked state in the QA artifact.

- [ ] **Step 4: Run focused verification**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py Tests/Character_Chat/test_character_persona_scope_service.py --tb=short
git diff --check
```

### Task 3: Phase 4.3 Skills Attach Validation And Local Execution Contract

Backlog: `TASK-11.3`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/skills_screen.py`
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/Skills/test_local_skills_service.py`
- Test: `Tests/Skills/test_skills_scope_service.py`
- QA: `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-3-skills-attach-validation.md`

- [ ] **Step 1: Add red tests for Agent Skills validation and attach state**

Cover valid `SKILL.md`, invalid metadata, missing local backend, policy-denied server action, and Console staged context.

- [ ] **Step 2: Implement minimal validation/readiness fixes**

Follow the Agent Skills spec: directory name matches `name`, required YAML frontmatter exists, and body instructions stay available through progressive disclosure.

- [ ] **Step 3: Capture actual screenshots**

Record valid, invalid, and attached-to-Console states.

- [ ] **Step 4: Run focused verification**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py Tests/Skills/test_local_skills_service.py Tests/Skills/test_skills_scope_service.py --tb=short
git diff --check
```

### Task 4: Phase 4.4 MCP Source Scope And Action Readiness

Backlog: `TASK-11.4`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py`
- Test: `Tests/UI/test_unified_mcp_panel.py`
- Test: `Tests/UI/test_destination_shells.py`
- QA: `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-4-mcp-source-scope.md`

- [ ] **Step 1: Add red tests for server-first hierarchy and readiness copy**

Assert tools remain nested under servers, local/server scope is visible, disabled actions explain runtime/policy blockers, and route aliases still work.

- [ ] **Step 2: Implement minimal MCP panel readiness fixes**

Preserve the existing `UnifiedMCPPanel` instead of moving MCP into Settings or generic Tools.

- [ ] **Step 3: Capture actual screenshots**

Record local empty, server blocked, and action-ready states where available.

- [ ] **Step 4: Run focused verification**

```bash
$PY -m pytest -q Tests/UI/test_unified_mcp_panel.py Tests/UI/test_destination_shells.py --tb=short
git diff --check
```

### Task 5: Phase 4.5 ACP Runtime Session Contract

Backlog: `TASK-11.5`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/acp_screen.py`
- Create or modify ACP interop state files only if required by tests
- Test: `Tests/UI/test_destination_shells.py`
- QA: `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-5-acp-runtime-session.md`

- [ ] **Step 1: Add red tests for ACP runtime/session display contracts**

Cover runtime missing, runtime configured, session selected, disabled Console follow, and compatibility copy.

- [ ] **Step 2: Implement minimal state model**

ACP owns runtime setup and session state. Settings may expose global defaults later but must not become the runtime owner.

- [ ] **Step 3: Capture actual screenshots**

Record blocked runtime and configured/session states if a runtime fixture is available.

- [ ] **Step 4: Run focused verification**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py --tb=short
git diff --check
```

### Task 6: Phase 4.6 Schedules And Workflows Run Control

Backlog: `TASK-11.6`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/schedules_screen.py`
- Modify: `tldw_chatbook/UI/Screens/workflows_screen.py`
- Modify: `tldw_chatbook/Home/active_work_adapter.py` only if active-work payloads need normalized control metadata
- Test: `Tests/UI/test_destination_shells.py`
- Test: `Tests/Home/test_active_work_adapter.py`
- QA: `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-6-schedules-workflows-run-control.md`

- [ ] **Step 1: Add red tests for run state, approval, retry, and Console follow controls**

Cover running, paused, failed, approval-pending, no active run, and latest digest output where Schedules supports it.

- [ ] **Step 2: Implement minimal control state fixes**

Inspector state must come from the same item status shown in the detail pane. Disabled buttons must describe the next recovery action.

- [ ] **Step 3: Capture actual screenshots**

Record Schedules and Workflows selected-run, blocked, and failed/retry states if fixtures support them.

- [ ] **Step 4: Run focused verification**

```bash
$PY -m pytest -q Tests/UI/test_destination_shells.py Tests/Home/test_active_work_adapter.py --tb=short
git diff --check
```

### Task 7: Phase 4.7 Agent Execution QA Closeout

Backlog: `TASK-11.7`

**Files:**
- Modify: `Docs/superpowers/qa/product-maturity/phase-4/README.md`
- Create: `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-agent-execution-closeout.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-11 - Product-Maturity-Phase-4-Agent-Configuration-And-Execution.md`
- Modify: child task files that are completed
- Test: `Tests/UI/test_product_maturity_phase4_agent_execution_plan.py`

- [ ] **Step 1: Replay the mounted workflows**

Verify Personas, Skills, MCP, ACP, Schedules, and Workflows in the running app. Capture actual screenshots for every visible screen changed during the phase.

- [ ] **Step 2: Record the workflow matrix**

Document goal, steps attempted, what worked, blockers, severity, screenshots, and exact verification commands.

- [ ] **Step 3: Update roadmap and task state**

Mark only verified child tasks Done. Leave incomplete runtime/server parity work as explicit residual risk.

- [ ] **Step 4: Run focused closeout verification**

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_phase4_agent_execution_plan.py Tests/UI/test_destination_shells.py --tb=short
git diff --check
```

- [ ] **Step 5: Commit closeout**

```bash
git add Docs/superpowers/qa/product-maturity/phase-4 Docs/superpowers/trackers/product-maturity-roadmap.md backlog/tasks/task-11* Tests/UI/test_product_maturity_phase4_agent_execution_plan.py
git commit -m "docs: close phase 4 agent execution"
```
