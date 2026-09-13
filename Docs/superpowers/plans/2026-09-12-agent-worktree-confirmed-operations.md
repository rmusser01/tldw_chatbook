# Confirmed agent worktree operations

> **For agentic workers:** Use subagent-driven-development. Root owns Git, Backlog and governance. Workers leave source unstaged.

**Goal:** Provide one tested apply/merge/discard path for current-turn model tools and later-turn Console recovery.

**Architecture:** A narrow recovery module validates durable records against a freshly captured selected authority, previews changes, asks for exact consent, revalidates, transactionally claims the record, and invokes fixed local Git lifecycle operations. Reuse existing lifecycle types and helpers where sound; public read-only Git tools remain unchanged. No alternate Git backend or platform qualification project is part of this work.

**Tech Stack:** Python 3.12+, SQLite, ordinary local Git, existing ExecutionOwner and confirmation callable.

**Spec:** Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md

ADR required: no new ADR; clarify known-no-effect claim release within ADR-155
ADR path: backlog/decisions/155-agent-worktree-recovery.md
Reason: direct implementation of confirmed recovery; release to unresolved is allowed only after positively verified absence of destination mutation, while ambiguous effects stay protected.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`. Root owns Git/Backlog/docs. Workers do not spawn subagents.
- Product imports only under Tests/conftest, using the existing `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`. Copy the stdlib runner into this plan workspace, use unique evidence labels and owned temporary Git/SQLite fixtures. No full suite, network/providers/live config/dependencies or foreign cleanup.
- Exact selected writable named authority, full root/common metadata/child identity chains, original base and matching run/conversation/binding are mandatory. Stored paths never grant permission. Guard checks bracket each operation; ordinary Git does not guarantee atomic protection against external concurrent metadata replacement.
- Only terminal, unresolved, durably drained records are eligible. No branch-prefix adoption, missing-base reconstruction, automatic cleanup, forced worktree-root removal, or automatic uncertain replay.
- Fixed application commands only, hooks disabled for generated mutations, no model argv/executable/environment/network. Do not widen the public read-only Git allowlist. Do not change process cwd or use worker-only fchdir pinning in the multithreaded app.
- Preview max8192 characters. Patch capture uses an owned spool capped at32MiB, streaming rather than an unbounded Python string. Git output and execution are bounded. On interruption or ambiguous Git/persistence completion, retain the checkout and protect the record.

### Task 1: Shared confirmed operation engine and same-turn adapters

**Files:**
- Create `tldw_chatbook/Agents/agent_worktree_recovery.py`: authority/record validation, preview and confirmation orchestration, claim/outcome lifecycle.
- Modify `tldw_chatbook/Agents/agent_worktree.py`: strengthen ordinary lifecycle operations for bounded machine-safe preview/patch, disabled generated hooks, safe merge conflict handling, logical discard. If a private process helper needs its own file, use `Agents/agent_worktree_git.py` with no public model-callable surface.
- Modify `tldw_chatbook/DB/agent_worktrees.py`: exact-owner `finish_operation(..., state="unresolved")` for a caller-proven no-effect refusal; preserve all existing guarded transition requirements.
- Modify `tldw_chatbook/Agents/agent_service.py`: replace the disabled merge/discard closures and restore their schema gate under the real confirmation callable.
- Modify `tldw_chatbook/Agents/tool_catalog.py`: accurate current-turn handle plus Console recovery and retained baseline descriptions.
- Tests: new `Tests/Agents/test_agent_worktree_confirmed_recovery.py`, existing `test_agent_worktree.py`, selected `test_fleet_runtime.py` disclosure/operation nodes, and repository completion-transition nodes.

**Public shared interface (used unchanged by later Console wiring):**

```python
@dataclass(frozen=True)
class WorktreeRecoveryOutcome:
    action: str
    message: str
    state: str
    commit_sha: str | None = None

def recover_agent_worktree(
    db: AgentRunsDB, *, authority: RunAdmittedWorkspaceRoot,
    conversation_id: str, run_id: str, action: str,
    request_confirmation: Callable[[dict], dict],
    should_cancel: Callable[[], bool],
) -> WorktreeRecoveryOutcome | WorktreeRefusal: ...
```

This function runs on an owning worker thread, never the UI thread. It borrows db and does not close an existing caller connection. The caller owns the ExecutionOwner; the synchronous operation retains that owner until every launched Git operation has completed or its cleanup outcome is known. An unexpected failed process-tree cleanup marks the current owner cleanup-unproven before returning. Cancellation before the mutation claim returns without changes. Cancellation after possible effects never reports a clean cancellation; it protects uncertain state.

The exact confirm payload contains `run_id`, `action` (apply/merge/discard), `branch`, `worktree` and `source` (child path), `destination` (selected repository path), and bounded `diffstat`. Add `retains_checkout=True` and clear retained-baseline text for discard. The service may add the current handle ID via its callback wrapper. Only `decision.get("allow") is True` consents. Missing callback, exception, denial or cancellation leaves the unresolved source unchanged.

1. Validate record ownership and eligibility using metadata-only reads. Require exact workspace/binding/fingerprint/root/identity against the supplied writable authority and a fresh guard. Resolve fixed source and child Git common-directory queries; match the recorded common directory and identity. Validate child checkout's exact branch and original base ancestry. Refuse nonstandard external child gitdir/submodule layouts for mutations rather than adopting them.
2. Prepare read-only preview and a content fingerprint. Include child HEAD/index/diff against original base, all untracked file contents/paths/symlink targets, and ignored entries when discard would remove them. Use bounded machine-safe Git output (`--no-ext-diff --no-textconv --no-color`), NUL-delimited paths and no-follow filesystem reads. Large or special unsupported entries return a specific refusal; no silent truncated fingerprint. Include destination HEAD/status in the confirmation snapshot so changes while waiting demand a fresh preview. Preview never stages or commits.
3. Ask the caller's exact confirmation. Revalidate current authority/identities and recompute the content fingerprint after Allow. On drift or cancellation, refuse without claiming or mutating. Transactionally claim with a generated operation ID only when all eligibility conditions still hold.
4. Perform the confirmed action using the original base. Always retain the source checkout. Apply captures child changes with a fixed agent identity and disabled hooks, streams a binary patch to the capped owned spool, runs check before applying, and lands unstaged edits without disturbing unrelated parent index/working files. Merge requires a clean destination and no existing merge/rebase/cherry-pick operation, uses a real explicit merge commit with hooks disabled, and aborts only a merge this operation demonstrably started. On conflict preserve child content and positively verify destination restoration before declaring a clean refusal. Never abort a pre-existing operation.
5. Discard is logical cleanup: verify POSIX no-follow descriptor-relative primitives before any mutation, restore the recorded base in the child, remove remaining child changes/untracked/ignored entries without following symlinks, detach HEAD at base, and delete exactly `refs/heads/<recorded branch>` with expected-old-SHA CAS. Retain the top checkout directory and its administrative link. Refuse nested repositories/submodules rather than deleting unknown ownership. Check source/destination authority throughout. A replacement symlink must never lead cleanup outside the child. No `git worktree remove --force`, `shutil.rmtree(root)` or pathname root deletion.
6. Persist exact operation success as applied/merged/discarded_cleanup_pending and return an honest receipt. `finish_operation(..., state="unresolved")` is allowed only for verified no-destination-effect failures (including an oversized patch after a disclosed source capture commit). Stale/mismatched completion remains refused. A crash, timeout after potential effects, unexpected exception, partial cleanup, or failed persistence remains in-flight/uncertain, non-actionable and never automatically retried. Do not report a successful receipt if its completion record could not be confirmed.

Service closures resolve only their current turn's durable handle map and verify its run is the corresponding created child in the same conversation. They call this shared function with the frozen source authority and existing confirmation callback. No provider fallback root. The same predicate (`fleet_active`, primary agent, real confirmation callable) controls schemas and runtime callback exposure. Authority absence yields an explicit refusal; it never selects an arbitrary binding. Tool copy explains older work is available through Console Recover agent work and that discard retains a baseline checkout.

- [ ] Write RED real Git+SQLite tests before implementation for denial/no callback, original-base apply after parent HEAD advances, new/untracked/binary changes, explicit merge commit, same-file conflict preserving parent and source, changed preview refusal, wrong conversation/binding/identity, held/uncertain refusal and two competing claims.
- [ ] Add RED logical discard tests for clean and dirty child, ignored entries, external symlink target survival, retained detached baseline, exact-ref mismatch, nested-repository refusal, and unsupported primitive refusal before mutation. No tests touch user repositories.
- [ ] Implement the shared operation function and narrow fixed lifecycle changes. Test interruption before consent and mutation-completion persistence failure: reopening must not allow automatic replay. Test known-no-effect claim release separately from uncertain completion.
- [ ] Wire real same-turn closures and disclosure with targeted actual run_turn tests; include waiting for actual owner drain, not only fleet terminal status. Existing tests that expected the superseded blanket refusal should now test missing authority/card explicitly.
- [ ] Run focused new engine tests and directly affected helper/service/schema nodes. Record output, compare added Ruff identities and edited-hunk formatting against the source base, and run diff-check. Keep inherited warning qualification precise.
- [ ] Report behavior/evidence and public interfaces for Console integration. Leave source unstaged for independent review; the visible card, preview call-site parity and retained recovery list are the next slice, so neither task is Done yet.
