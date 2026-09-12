# PR 2631 integration and review

Source of authority: accepted ADR-129/130/131/132/134/135/136 and existing workstream task acceptance criteria. User requests rebase onto latest dev, address Qodo review, then merge.

ADR required: no new contract; amend existing ADR-131/134/135 migration numbering before DB edits.
ADR paths: backlog/decisions/131-durable-agent-budget-accounting.md, backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md, backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md.
Reason: preserve approved storage semantics while assigning unoccupied migration versions.

Global constraints: preserve all current dev features and lifecycle fences; no wholesale historical file replacement. Keep shared checkout untouched. No full local test sweep without user opt-in. No ratchet loosening. Preserve upstream canonical task records and renumber only colliding workstream records. Root owns git index, rebase, push and merge. Workers edit only assigned files.

Pinned integration base: d30d8c516cc901b4b017f5214483c56ae10ccda8. Historical PR commit a0f9a90d4b4e2a905764223d666ff2bb52783682, historical common base 2c4c657d015b13b48a5d98712e8767ddcc01003f.

1. Agent runtime integration: combine lifecycle, worktree, steering, capacity and progress contracts in Agents production files and matching Tests/Agents.
2. Chat integration: combine durable automatic work, acceptance, wake delivery, provider and runtime changes in Chat and matching Tests/Chat; repair obsolete budget probes.
3. Console integration: port progress/history presentation and automatic admission onto upstream controllers; preserve newer controller extractions. Own UI, Widgets/Console, Workspaces, css, UI/Wizards and matching Tests/UI/Architecture/Wizards.
4. Persistence and governance: reconcile AgentRuns schema with upstream v15 using v16 budget tokens, v17 automatic work, v18 runtime ownership; preserve all upstream migrations. Test upgrades from old and current dev databases. Reconcile task IDs and docs.
5. Integration verification: fresh targeted suites, privacy/static/architecture checks, independent task reviews then whole-branch review. Fix proven regressions.
6. Publish: update PR with force-with-lease, mark ready, obtain Qodo review, address actionable comments and required checks, update to any subsequent dev changes, merge and verify.

Task5 amendment, TASK-32493: add eight bounded provider-free modules to the existing serial PR fast lane under ADR-103, including the six local SemLock-blocked callback cases. Existing required gate, dependencies, and event cadence remain unchanged. Verify exact non-overlapping target contract and clean-runner results.

## Latest-dev boot reconciliation

Dev d8516accde3a39c30cf41b06709cc359d2a9034d adds the registered Improve-My-Prompt module; the combined ready census rises to 974 against the unchanged 973 cap. Defer the workstream's execution-capacity module until actual execution rather than changing either registry behavior or the cap.

ADR required: no new ADR. ADR paths: backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md and backlog/decisions/097-boot-budget-ratchets.md. Reason: preserve the existing runtime-wide admission authority and defaults while completing its intended lazy construction; no resource, ownership, or scheduling policy changes.

1. Keep WorkOrigin as canonical lightweight model data, re-exported from execution_capacity for compatibility. Localize service imports used only at execution.
2. Give bridges a lazy capacity supplier bound to the owning runtime. Serialize first allocation, preserve explicit capacity injection and replacement checks, and refuse allocation after disposal. Keep runtime disposal noncreating and close the same shared capacity if it exists.
3. Prove ordinary imports/construction/inspection do not load or allocate capacity; prove real execution, parallel first use, replacement and disposal retain one owner and limits. Rerun targeted runtime/admission/lifecycle and boot tests, then independent review before pushing the rebase.
