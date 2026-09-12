# Per-definition child wall-time cap implementation

ADR required: yes
ADR path: backlog/decisions/157-per-definition-child-wall-time-caps.md; backlog/decisions/158-agent-runs-migration-order-after-worktree-qualification.md
Reason: durable optional policy, audit identity, and retained runtime bound supplement ADR-134/135; ADR-131 durable accounting is unchanged.

**Spec:** Docs/superpowers/specs/2026-09-12-definition-wall-cap-design.md.

Preconditions: root has accepted the design, recorded ADR-157 without modifying accepted ADR-134, and used Backlog CLI to put TASK-13154.7 In Progress and record this plan before source changes. Work on the existing isolated branch. Re-read actual schema before code: ADR-158 moves this independent feature to v18→v19 while worktree execution is unresolved. Coordinate Settings and RunBudget denial-field edits rather than overlapping ownership. Read only targeted test guidance and actual fixture APIs.

## Global Constraints

- Work only in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr, branch codex/agent-orchestration-remaining. Root owns allgit and Backlog status; workers leave edits unstaged and dispatch no subagents.
- Use .superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python under pytest isolation; targeted tests only, no liveconfig/network/newdependencies or unrelated static debt cleanup.
- This executes after Settings preset integration, independently of unresolved worktree recovery. ADR-158 assigns v18→v19; recheck the actual version before source edits.
- An optional definition cap can tighten but never widen the baseline budget applicable to that branch. Preserve threaded outlive-parent and inline parent-remainder semantics. Do not turn automatic elapsed deadlines into pausable per-run clocks.
- Capped continuations receive fresh run time but retain their lineage's admitted ceiling even if the definition later raises/removes its cap. Never-capped/generic behavior stays unchanged.
- Preserve every non-wall/non-spawn RunBudget field through helper reconstruction, including denial limit, retry count and warning fraction. Keep uncapped definition fingerprints byte-identical.
- Canonical Settings only; preserve ownership/save-feedback/preset repairs. Follow DESIGN.md and Impeccable craft-floor; inspect wide/narrow painted form once, fix observed defects then confirm.

### Task 1: Definition identity and real SQLite

Files: Agents/agent_models.py; DB/AgentRuns_DB.py; new DB/migrations/agent_runs_v18_to_v19_definition_wall_seconds.sql; Tests/Agents/test_agent_models.py; Tests/DB/test_agent_runs_db.py.

Add appended optional max_wall_seconds with None default; shared validation rejects bool/invalid/nonfinite/nonpositive values. definition_from_row tolerates missing optional key. Fingerprint conditionally adds normalized float only for present cap, proving pre-field byte compatibility for None. Fresh schema, guarded upgrade, version constant/table and parameterized create/update store nullable REAL. Existing SELECT * row conversion needs no special cast.

Verification prerequisite: the schema18 baseline has one inherited failure in `test_runtime_tool_names`; its exact expected set omits the already-merged `report_to_supervisor` and `read_agent_messages` entries. Reconcile those two expectations without changing production catalog names or weakening exact-set coverage. Record the110-pass/1-fail baseline separately from feature RED/GREEN.

Tests first: exact old fingerprint, valid fractional value, int/float fingerprint equivalence, invalid input no-write, file-backed v18->v19/reopen/idempotency and standalone SQL (using real preceding schema). Preserve existing definitions and all existing schema18 data; recovery ownership records are not implemented yet. Update current-version assertions in older migration tests, retaining their structural/data checks. Run only model + AgentRunsDB modules and formatter/linter on changed Python.

### Task 2: Frozen spawn restriction and retained continuation

Files: Agents/agent_models.py; Agents/agent_service.py; Agents/fleet_coordinator.py; targeted Agents tests for fleet runtime/coordinator/continuation/automatic child scope.

Replace manual child RunBudget reconstruction with dataclasses.replace while retaining exact wall/spawn formulas; keep all inherited dimensions including newly integrated denial limit. Apply post-helper definition minimum in spawn. Add appended optional definition_wall_seconds to FleetHandle and RetainedTranscript, keyword-only default to reserve and shared _launch_fleet_child; initialize at reservation and copy during retention. Resume reuses current frozen definition identity, combines current budget/current cap/retained bound, and propagates final bound only for cap-controlled lineages. Preserve existing generic and never-capped behavior and fresh continuation time allowance.

Tests first: actual persisted budgets and gated runtime termination for tighter, non-widening, subsecond, inline parent bound, uncapped and sibling cases; mutate DB after planning and forbid reread; continue with raised/lowered/removed cap across at least two generations; disabled/deleted refusal. Test all non-wall/non-spawn budget fields survive both helpers, including the new denial dimension. Add actual gated human wait/cancel and automatic elapsed-deadline test with worker lease retained until release. Always release events/join threads in finally. Reuse current automatic ledger acceptance fixtures, do not mock its check into unconditional success. Run only touched runtime/coordinator/automatic modules or explicit scenarios plus their existing neighboring regressions.

### Task 3: Canonical Settings and documentation closeout

Files: Widgets/settings_agents_panel.py; Tests/UI/test_settings_agents_category.py; existing user-facing Agents documentation and governing design/plan references identified with rg.

Add agents-wall-seconds-input; parse blank None/float then existing validation; select renders numeric value, New/every preset clears to None. Preserve sibling DB ownership and omitted-tool feedback fixes. Mounted tests drive Save and read real DB after valid, clear and invalid edits; check form reset on all preset selections and selected definition unchanged until Save. Documentation distinguishes optional tightening from replacement; describes per-run continuation bound and current cooperative/human-wait semantics without promising remote termination.

Run targeted Settings module, impacted models/DB/runtime regressions and changed-file static checks using the isolated existing interpreter. Self-review diff for dropped budget fields, misplaced floors, stale cap UI, schema collisions, and cap removal widening continuation. Root performs required reviews and final Backlog AC/notes/status updates only with recorded evidence. No full sweep or live external certification claimed.

## Interface and validation patterns

Task 1 appends `max_wall_seconds: float | None = None` to AgentDefinition. Validation checks bool before numeric conversion, guards OverflowError, and requires math.isfinite(value) and value>0. None preserves the old fingerprint payload; only a present cap adds `max_wall_seconds: float(value)` before hashing. Existing CRUD validates before writes.

Task 2 preserves helpers through dataclasses.replace(child, max_wall_seconds=existing_formula, max_subagents=0). After the existing branch helper:

```python
if resolved is not None and resolved.max_wall_seconds is not None:
    child_budget = replace(child_budget,
        max_wall_seconds=min(child_budget.max_wall_seconds, resolved.max_wall_seconds))
```

Add `definition_wall_seconds: float | None = None` at the end of FleetHandle and RetainedTranscript; extend reserve with a keyword-only default and retain it before child execution can complete. For continuation, minimum the normal budget, any current definition cap and any retained admitted ceiling; preserve None for a never-capped lineage. `_launch_fleet_child` passes coordinator-only metadata to reserve, never into `_run_one` kwargs.

Task 3 adds a compact Input `agents-wall-seconds-input`. Blank maps to None; nonblank parses float then uses shared model validation. Selection fills it, New and generic preset loading clear it. Existing Save errors remain visible and leave the stored row untouched.

Each task writes its own exact red/green commands, counts, warnings, changed paths and scoped static evidence. Root commits and independently reviews before the next task. Root performs final acceptance/notes/Done updates through Backlog CLI only after all three slices pass.
