# Interim agent worktree safety

**Goal:** Stop known unqualified agent-driven Git mutation and automatic pathname cleanup while TASK-31210/31211's complete recovery execution boundary is unresolved.

**Spec:** TASK-31210 AC5/6/7 and TASK-31211 AC5/6/8; Docs/superpowers/specs/2026-09-12-agent-worktree-recovery-design.md (execution qualification checkpoint). These functional recovery tasks remain incomplete after this prerequisite.

ADR required: no new ADR
ADR path: backlog/decisions/155-agent-worktree-recovery.md; backlog/decisions/158-agent-runs-migration-order-after-worktree-qualification.md
Reason: implement the existing unsupported-boundary refusal and no automatic/forced pathname deletion requirements. No replacement Git backend is adopted.

## Global Constraints

- Work only in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr. Root owns Git/Backlog; no worker subagents or overlapping source workers.
- Use .superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python under pytest isolation. Targeted tests, temporary real repositories/SQLite, no live configuration, providers, dependencies or user filesystem mutations. Unique basetemp paths and captured raw outputs belong in this plan's workspace. No foreign cleanup.
- Explicitly requested worktree isolation must refuse before any Git helper or child execution; never silently run the child in a shared workspace. Ordinary non-worktree agent execution stays available.
- Confirmed merge/discard cannot reach the old pathname helpers even when an embedding caller supplies a confirmation callback. Preview/live disclosure uses the same capability gate. Do not claim a qualified platform exists.
- Existing checkout contents and branches stay untouched. No worktree removal, prune, forced cleanup or speculative adoption. Provider routing retirement must still run independently of handle-map presence.
- Low-level explicit-path helper APIs may remain for compatibility/manual callers; these tests do not qualify them for run-admitted automatic authority. Do not describe this prerequisite as implemented worktree recovery.

### Task 1: Refuse unqualified execution and retain existing work

Files: Agents/agent_worktree.py (small closed refusal/capability seam if needed); Agents/agent_service.py (request-plan disclosure, admission, merge/discard wrappers and cleanup); related focused Agents worktree/runtime tests; Tests/Chat/test_console_worktree_merge_confirm.py for Console preview parity; Docs/User_Guide/console/agent-runs-and-tools.md. Avoid adding a general backend registry, configuration switch or new schema.

Read actual admission and teardown routes and the original race report in .superpowers/sdd/2026-09-12-agent-worktree-recovery/task-1-report.md. The old _admit_agent_worktree calls create_agent_worktree(provider.workspace_root,...) and its failure cleanup force-discards; _retire_agent_worktree(discard=True) force-removes by pathname; end-of-turn _sweep_stale_agent_worktrees can delete clean unmerged data using DB terminal status alone. Neither caller confirmation nor terminal status repairs root/physical-owner authority.

1. Write failing behavioral tests before source edits. Fleet and inline/no-fleet AgentService requests with isolation=worktree must use the same refusal. A real fleet scripted spawn must return an actionable unsupported_execution_boundary refusal, persist its failed precreated run/settle its reserved handle and owner, never call create/admit helpers and never execute the child model. Non-isolated sibling/next execution must still work. Read the real caller unwind rather than mocking it away.
2. Add tests proving even an explicitly supplied confirm callback cannot enable/execute old merge/discard helpers. Shared first-request planning must omit those two schema IDs for preview and live paths while retaining the other fleet tools. Keep runtime refusal defense if closures can be invoked independently of disclosure. Do not retain unreachable legacy mutation bodies merely as a future template.
3. Seed actual clean-unmerged and dirty temporary worktree checkouts plus an unrelated sentinel. End-of-turn cleanup and failed-start retirement must retain bytes, branch and directory. Require provider routing retirement even if the service handle map is absent. Track temporary resource cleanup explicitly in test finally; any fixture Git removal is only its own temporary repositories.
4. Implement a small uniform reason-coded refusal with copy explaining that safe agent worktree execution is unavailable and existing work is retained for manual review. It must not imply the work was merged, discarded or automatically recovered. Gate creation and landing/destruction before filesystem reads/writes from the old helpers. Do not fall back to normal/shared isolation.
5. Stop production automatic root cleanup and failed-admission/failed-start forced removal; preserve associated records where they already exist. Update docstrings and targeted old expectations to the new retention policy, preserving independent provider unadmission. Replace superseded service integration assertions with active refusal/retention tests and remove unreachable legacy tests; do not leave skipped templates for the removed unsafe behavior. Preserve separate low-level helper coverage. No fabricated durable recovery record or drain proof.
6. Document this temporary limitation and retained data honestly. New Console confirmation/recovery features remain unavailable; ordinary non-isolated agents and explicitly user-operated Git remain available. Do not tell the model to perform unapproved manual Git as a workaround.
7. Run focused new/regression nodes for request planning, worktree spawn refusal, retired routing and preservation. Use real temporary Git/DB evidence plus bounded owned process/thread cleanup. Legacy low-level helper success tests are separate, not evidence of the new admitted boundary. Capture exact RED/GREEN commands, stdout/stderr and exits under this plan workspace.
8. Compare exact per-file baseline/current Ruff diagnostic sets and format every changed hunk using the root scoped_static.py utility; no whole-file format/fix of legacy modules. Run git diff --check. Leave source unstaged, report, root commits and independently reviews. This prerequisite does not close TASK-31210/31211.

## Completed prerequisite

Implemented in `93ee16a144` and `534ad2c890`, with independent review and one scoped fix round approved. Final affected Console/service selection passed 25 tests (246 deselected, no new skips); changed-file diagnostic sets added nothing and every edited hunk passed formatting. Initial RED and intermediate test corrections remain separately qualified in the task report. The obsolete service behavior tests were replaced with active refusal/retention coverage; manual helper tests remain separate. TASK-31210/31211 remain In Progress because safe functional execution and durable recovery are still unresolved. ADR-155 and migration-order ADR-158 remain applicable.
