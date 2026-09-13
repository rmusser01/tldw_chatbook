# Agent worktree creation restoration

> **For agentic workers:** Use subagent-driven-development to implement and independently review this slice. Root owns Git and Backlog changes.

**Goal:** Restore isolated child creation in the exact authorized repository while preserving unfinished work.

**Architecture:** Thread the existing frozen selected `RunAdmittedWorkspaceRoot` through the Console bridge into AgentService. Reuse Git lifecycle helpers and existing child path routing with current binding/root checks. Keep all automatic cleanup disabled and merge/discard unavailable until their later confirmation and ownership slices are complete.

**Tech Stack:** Python 3.12+, existing Git executable, SQLite-backed AgentService and pytest.

**Spec:** Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md, Creation and Verification sections.

ADR required: yes, existing amendment
ADR path: backlog/decisions/155-agent-worktree-recovery.md
Reason: correct the blanket execution restriction and explicitly limit ordinary Git identity guarantees while preserving authorization.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`, branch `codex/agent-orchestration-remaining`. Root owns index, commits, Backlog and governance files. No shared-checkout edits or overlapping source workers.
- Existing interpreter: `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`. Product imports only under Tests/conftest isolation. Run targeted tests via this plan's `run_pytest.py`, which owns unique basetemp/evidence. No dependency installs, live config, external providers, network, foreign cleanup, worktree pruning or new checkouts for verification.
- Ordinary local Git is the selected implementation. No new backend registry, worker protocol, sandbox, metadata alias experiment or full concurrent-replacement proof. Do not claim application identity checks provide atomic Git confinement. Existing public fs/read-only Git containment remains unchanged.
- Creation needs an exact selected writable named binding; no provider fallback root, scratch, inferred first folder or model-supplied destination. No-fleet/inline isolation refuses without child/shared fallback.
- Existing work and partial/failed-start checkouts survive. Keep `_retire_agent_worktree` routing retirement and no-op `_sweep_stale_agent_worktrees`. Do not restore automatic force deletion from historical code.
- Merge/discard closures and disclosure remain unavailable in this slice. No task is marked Done merely because creation is restored.

### Task 1: Restore selected-authority creation and child routing

**Files:**
- Modify `tldw_chatbook/Chat/console_chat_controller.py`: accepted-turn selected authority and bridge call.
- Modify `tldw_chatbook/Chat/console_agent_bridge.py`: optional argument forwarded to service.
- Modify `tldw_chatbook/Agents/agent_service.py`: optional constructor authority, admission, inline refusal and failure retention.
- Modify `tldw_chatbook/Agents/agent_worktree.py`: minimal creation validation/unique destination and generated-hook suppression needed by restoration.
- Tests: `Tests/Agents/test_fleet_runtime.py`, `Tests/Agents/test_fleet_continuation.py`, `Tests/Agents/test_agent_worktree.py` where applicable; `Tests/Chat/test_console_turn_execution_context.py`/bridge tests for exact authority forwarding. New focused tests may live in `Tests/Agents/test_agent_worktree_creation.py`.

**Interfaces:**
- Consume existing `RunAdmittedWorkspaceRoot` and `capture_run_admitted_workspace_roots`, `create_agent_worktree(repo_root, run_id)`, and `LocalToolProvider.admit_run_workspace_root(run_id, authority)`.
- Produce optional `worktree_repo_authority: RunAdmittedWorkspaceRoot | None = None` on bridge `run_reply` and service constructor; existing callers default to refusal for worktree isolation.
- `_admit_agent_worktree(handle, child_run_id)` continues returning None on success or an actionable reason-coded string on refusal. It stores a created AgentWorktree before any later failing admission step.

- [ ] Write a real service regression before production edits. Seed a real temporary Git repository and an unrelated fallback repository; bind the selected authority to the first. Script a fleet child to write using the real local tool path. Assert child bytes exist only in its registered checkout; both source and fallback working files remain unchanged. Observe current blanket refusal as RED. Example assertions:
  ```python
  assert child_row["status"] == "success"
  assert created.worktree_path.joinpath("child.txt").read_text() == "child work"
  assert not selected_root.joinpath("child.txt").exists()
  assert not fallback_root.joinpath("child.txt").exists()
  ```
  Use actual status constants from existing tests where their serialized name differs. Do not stub admission or Git.
- [ ] Test no selection, read-only selection, removed/retargeted binding, replaced root and kill-switch refusal before child execution; ordinary sibling execution still works. Cover post-create drift by a deterministic test barrier at the application recheck, preserving the created checkout and refusing child admission. Cover failed thread start and provider admission retention. Preserve the existing raw-shell/virtual-CLI exclusions.
- [ ] Thread only the exact accepted-turn selected binding into the bridge/service. Build the current authority guard from existing selection/identity checks and a fresh fail-closed tool-kill reader. Default None for absent selection, private scratch or read-only selection. Direct service tests inject explicit immutable authority fixtures. Do not call live viewed-session selection from a child thread.
  ```python
  # Optional argument on both call boundaries:
  worktree_repo_authority: RunAdmittedWorkspaceRoot | None = None
  # Service stores the already admitted turn snapshot:
  self._worktree_repo_authority = worktree_repo_authority
  ```
- [ ] Restore the old admission's necessary routing logic from `93ee16a144^`, replacing provider.workspace_root with the injected authority. Validate authority/root before Git and again afterward. Capture child identity and compose a child guard that requires both source and child to remain current. Preserve the handle record before routing registration so failure cannot silently lose known work. Do not restore old cleanup bodies.
- [ ] Keep fixed application Git args, validate run identifiers before branch/path use, allocate a unique child directory instead of colliding on eight-character prefixes, disable checkout hooks for generated creation, and retain partial material on errors. Preserve ordinary helper compatibility where it does not weaken service authorization. Distinguish unsupported no-fleet admission from missing/revoked source authority.
- [ ] Adapt continuation tests to the restored behavior: a retained worktree child resumes into a fresh isolated checkout only with the same newly authorized repository selection; absence/revocation refuses. Do not revive prior worktree contents as continuation state. Retained previous work remains on disk.
- [ ] Run focused RED/GREEN and affected neighbors using `python3 .superpowers/sdd/2026-09-12-agent-worktree-creation-restoration/run_pytest.py <unique-label> Tests/... -q`. Inspect all output and qualify inherited dependency warnings. No full suite. Use the existing scoped static utility against source base `9861170ff9`; compare diagnostic identities and format only edited hunks. Run `git diff --check`.
- [ ] Write the implementation report with evidence paths and limitations. Root commits this scoped source change, dispatches independent task review, resolves concrete findings and updates user documentation/status. Later recovery slices consume the restored authority/creation route.
