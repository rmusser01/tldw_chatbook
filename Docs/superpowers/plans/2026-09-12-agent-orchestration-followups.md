# Agent orchestration follow-up reliability plan

> **For agentic workers:** Use subagent-driven-development to implement this plan task by task.

**Goal:** Make navigation's busy-session read observational and record the remaining orchestration work against merged dev.

**Architecture:** Keep the conversation-owned coordinator and retained AgentService ownership. Remove retained-owner cleanup from the read-only fleet snapshot; the existing rail, cancellation, and lifecycle paths keep their cleanup behavior. Terminal coordinator handles remain pruned at turn start.

**Tech Stack:** Python, Textual controller, threading, SQLite, pytest.

**Spec:** `backlog/tasks/task-15666 - busy_fleet_session_count-prunes-the-fleet-as-a-side-effect-of-a-read.md` (acceptance criteria), `backlog/decisions/129-fleet-mailbox-and-wake-reliability.md` (existing lifecycle contract).

ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md
Reason: restore the documented read-only interface; no new ownership, storage, pruning, or execution policy.

## Global Constraints

- Work only in the existing isolated `agent-orchestration-pr` worktree on `codex/agent-orchestration-followups`; preserve the dirty main checkout.
- Preserve conversation ownership, cancellation authority, conservative execution caps, and between-turn coordinator pruning.
- Use targeted tests only; do not run a full suite or change architecture, import, or CSS guard thresholds.
- Do not add dependencies, new runtime abstractions, or new settings.
- Use the isolated interpreter `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`; the shared interpreter can import the dirty main checkout.

### Task 1: Make fleet snapshots observational (TASK-15666)

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`fleet_snapshot`, `_prune_settled_fleet_survivors` documentation).
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (stale `fleet_has_unsettled_children` pruning comment only).
- Test: `Tests/Chat/test_console_agent_bridge.py` (existing survivor/count/cleanup tests).
- Update: the TASK-15666 file above (implementation notes and AC after verified review).

**Interfaces:**
- Consumes `ConsoleChatController.busy_fleet_session_count() -> int`, which calls `fleet_teardown_split()` and the real bridge's `fleet_snapshot(conversation_id) -> list[FleetHandle]`.
- Produces the same counts and snapshots without changing coordinator handles or retained owner membership.
- Existing `live_snapshot`, cancellation, and lifecycle cleanup continue dropping settled owners. `_conversation_fleet_coordinator` continues pruning terminal handles at the next turn start.

- [x] Extend `test_busy_fleet_session_count_sees_a_session_whose_only_work_is_a_survivor` to capture the real coordinator snapshot and retained owner list before reading the count while the child is live and after releasing its gate and joining it. Assert the count is respectively 1 and 0, and both snapshots remain unchanged. Core assertions:

```python
owners_before = bridge._retained_fleet_owners(session.id)
handles_before = bridge._conversation_fleet_handles(session.id)
assert owners_before
assert controller.busy_fleet_session_count() == expected_count
assert bridge._retained_fleet_owners(session.id) == owners_before
assert bridge._conversation_fleet_handles(session.id) == handles_before
```

  After the terminal read, call `live_snapshot` and assert the owner is released while the terminal handle remains until the next turn. Reuse real bridge/service/coordinator and the existing gated provider fake; do not mock the method under test.
- [x] Run that exact test before production changes. The intended failure is retained owners disappearing on the terminal busy-count read, not a provider/setup error.
- [x] Remove the `_prune_settled_fleet_survivors(conversation_id)` call from `fleet_snapshot` only. Keep the service lookup and survivor handle filtering unchanged. Update its and the cleanup helper's documentation to describe the current lifecycle. Remove the controller comment saying the navigation count prunes coordinator state.
- [x] Update `test_a_survivor_is_visible_and_stoppable_after_its_turn_returns` so its retained-owner cleanup assertion follows `live_snapshot`, the existing cleanup path. The read-only snapshot still returns an empty live fleet after settlement.
- [x] Run the existing targeted selection:

```sh
.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q -k 'busy_fleet_session_count or survivor or finished_childs_row or fleet_coordinator_factory'
```

- [x] Run changed-line Ruff/lint and formatting checks without reformatting pre-existing large-file debt; record exact commands, failures, and baseline comparisons. Run `git diff --check`.
- [x] Record implementation notes, red/green evidence, and any limitations. Commit only task-scoped files. Root arranges independent task review and whole-branch review, then sets Done via Backlog CLI once all criteria and checks are satisfied.

### Task 2: Fence late approval arms and serialize final verdicts (TASK-13215)

**Files:**
- Modify: `tldw_chatbook/Chat/console_interrupt_rounds.py` (`InterruptRoundHost` revocation/registration only).
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (approval verdict snapshot and obsolete revoke documentation).
- Test: existing revocation/host tests identified in `.superpowers/sdd/2026-09-12-agent-orchestration-followups/revocation-revalidation.md`; add focused cases there rather than a new framework.
- Update: `backlog/tasks/task-13215 - Fleet-approval-revocation-add-a-revoked-run-tombstone-and-close-the-residual-arm-read-windows.md`, `backlog/decisions/067-indefinite-human-approval-waits.md`.

**Spec:** TASK-13215 acceptance criteria; existing ADR-067 cancellation authority.

ADR required: no new ADR
ADR path: backlog/decisions/067-indefinite-human-approval-waits.md
Reason: clarify the existing per-run cancellation contract, including its physical-thread lifetime limit, without introducing a new permission boundary.

**Interfaces:**
- `revoke_for_run(run_id, stamps)` remembers revocation for those exact kinds even if no round exists; its returned count still counts rounds actually swept.
- `run_round(..., check_revoked=True)` rejects a matching revoked owner before publishing a card or pending badge. `check_revoked=False` kinds keep their existing primary-only contract.
- Empty run IDs still are not swept. Log one content-free warning when arming a revocable unowned round; no task text, paths, tool arguments, or credentials in the warning.
- Keep tombstones for the host lifetime. No TTL, size eviction, session-close clearing, or logical-run completion can forget revocation: an abandoned provider daemon may still reach its fallback. Document memory proportional to distinct revoked runs; safe reclamation needs proof every physical invocation drained and is outside this fix.
- Serialize the entire MCP verdict snapshot with revocation using the existing lock. Either a completed approval snapshot precedes revocation or revocation precedes an all-deny snapshot; never produce a mixed batch. Revocation cannot retract a previously completed approval or tool side effect.

- [ ] Read the investigation report and preserve its distinction between real late-arm/mixed-snapshot gaps and the obsolete cross-lock/session-keyed payload diagnosis. Review the current host and controller paths before choosing the smallest implementation.
- [ ] Add a deterministic regression that calls `revoke_approval_rounds_for_run(run_id)` before invoking an actual local `fs_write` approval fallback under `use_run_id(run_id)`. A mounted callback that would approve must never be called; the target file must not exist. Add corresponding late-arm MCP and skill-script assertions with empty registries/badges/payloads afterwards and a nonrevoked sibling still answerable.
- [ ] Add a warning assertion for a revocable arm outside any run context, without changing legacy unowned-call approval behavior. Add sibling retained-payload and switch/remount assertions for tool approval and skill-script revoke/teardown. Prove a destructive same-session payload removal mutation fails those assertions.
- [ ] Add controlled snapshot/revocation ordering coverage: revocation before final snapshot yields all unresolved deny; a competing revocation cannot alter individual verdicts partway through the snapshot. Use Events/barriers and bounded joins, not arbitrary sleeps or an assertion about source text.
- [ ] Run each new regression on unchanged production code and record the intended failures. Update the existing test which currently treats revocation of a not-yet-armed owner as permission to arm later; retain its zero-round count and unrelated-owner checks.
- [ ] Implement host-owned tombstones and registration checks under the existing lock. The intended mechanism is a per-kind set of run IDs, updated by the same sweep lock:

```python
with self.lock:
    for kind in stamps:
        self._revoked_runs[kind].add(run_id)
    # Existing per-kind round sweep remains under this lock.
```

  During registration, combine the tombstone and state stamp checks. Remove a rejected preregistered entry by identity, stamp it revoked, and return the normal revoked outcome outside the lock. Do not resurrect a swept object or publish a dead card.
- [ ] Take the complete controller approval result snapshot under `_approval_state_lock`, including a fresh revoked check. Keep audit logging and UI callbacks outside the lock; do not hold the non-reentrant host lock across arbitrary hooks. Preserve unresolved-denial provenance. Existing exact-round sibling cleanup should need tests/documentation, not a redundant locking layer.
- [ ] Run focused revocation, host wiring, local approval, and human-wait tests named by the report. Run changed-line lint, formatting checks, and `git diff --check`; compare pre-existing whole-file debt without reformatting unrelated code. Add an ADR-067 clarification and task notes with exact evidence and the lifetime tradeoff.
- [ ] Commit task-scoped changes and report red/green/mutation evidence. Root performs independent task and final branch reviews before marking Done.

### Task 3: Reconcile the remaining old harness failures (TASK-2155, TASK-22720, TASK-19642.8.3)

**Files:**
- Modify: `Tests/UI/test_console_dictionary_send_integration.py` (real durable-policy test setup).
- Modify: `Tests/Chat/test_console_local_citation_boundary.py` (remove the one TASK-22720 xfail marker only).
- Modify: `Tests/UI/test_console_mcp_approval.py` (one assertion from Task 2's approved review).
- Update: the three existing task files for TASK-2155, TASK-22720 (now named `Agent-bridge-placeholder-replacement-test-trips-the-unresolved-recovery-guard`), and TASK-19642.8.3.

**Spec:** The three task acceptance criteria. No production behavior changes.

ADR required: no
ADR path: N/A for test repairs; existing ADR-134 for wake-budget verification
Reason: align old harnesses and records with already-shipped durable acceptance and wake authority.

**Interfaces:**
- The real dictionary test's manually bound conversation must have a durable Console Library policy before accepting a turn. Use the real policy repository and hydrate the real holder, as the adjacent world-info integration harness already does. Do not bypass durable commit or mark the session ephemeral.
- Keep the provider/agent payload substitution assertions and verify the stored user text stays raw.
- The citation replacement test already passes all its assertions on merged dev. Remove only its stale expected-failure marker; do not weaken recovery guards or delete assertions.

- [ ] Read root's `stale-harness-probes.log`, `dictionary-durable-probe.log`, `dictionary-hydration-probe.log`, and `dictionary-policy-probe.log` in the plan workspace. The exact exception is `RuntimeError: Durable Console Library policy no longer matches acceptance.` Hydration alone still failed because the policy row did not exist. Inserting the policy and hydrating made the agent dictionary test pass; no production source was changed in those probes.
- [ ] Update both dictionary test setups using the existing real repository seam after binding their conversation, before submitting:

```python
store = screen._ensure_console_chat_store()
session = _active_native_session(screen)
session.persisted_conversation_id = conv_id
policy = store.persistence.console_library_policy_repository.insert(
    conv_id, store.session_library_policy_candidate(session.id)
)
assert policy.snapshot.policy_revision == 1
await store.hydrate_session_library_policy(session.id)
```

  A small helper within this test module may share this setup. Do not import from the world-info test, which already imports this module. Keep production gateway/controller behavior intact; if another harness defect surfaces, identify its cause before fixing it and record the evidence.
- [ ] Run the dictionary module before and after its harness edit. Extend the agent path with the same raw stored user-text assertion already present in the provider test, so fixing dispatch cannot accidentally endorse substituting persisted text.
- [ ] Remove the `pytest.mark.xfail` attached to `test_citation_repair_agent_missing_placeholder_keeps_runtime_row_without_repair`. Run that exact test and nearby `citation_repair_agent` cases normally. The expected replacement content/status/visible-output assertions remain unchanged.
- [ ] Address the approved Task 2 review's minor precision finding: in `test_revoked_arm_is_refused_before_configuration_read`, replace `assert all(not registry for registry in observed)` with `assert observed == []`. Run both parametrized cases. This pins the already-implemented early return before any configuration callback, not merely an empty registry inside it.
- [ ] Reconcile TASK-19642.8.3 using the four exact nodes from root's passing baseline and inspect the assertions in the two wake modules. Same manual authority gates and frozen run budget still apply; automatic-chain budgets add restrictions under ADR-134. Correct the old headless test docstring only if necessary for accuracy; no production changes or new budget behavior.
- [ ] Run the targeted dictionary module, the citation agent selection, and the four named wake nodes. Do not add the entire Chat/UI suites. Run changed-line lint/format checks and whitespace checks, recording any pre-existing debt precisely.
- [ ] Update all three task implementation notes with current root causes, unchanged production contracts, verification commands/results, and ADR decisions. Do not mark Done until independent review. Commit the batch's scoped files and write `task-3-report.md` with exact evidence.

## Revalidation and durable inventory

In parallel with this bounded fix, root records the remaining workstream in `backlog/docs/agent-orchestration-followups-2026-09-12.md`. TASK-13215 is independently revalidated against the current interrupt host. Investigation does not authorize speculative code changes or closing an older ticket without evidence. Newly confirmed fixes receive a task plan before implementation.

Preflight: no open PR or remote branch names matched TASK-15666/TASK-13215; current source changes are included in merged PR #2631. Open PR #2427 is broad test cleanup and may overlap older harness tickets, so those require file-level comparison before repair. These checks were made against `origin/dev` at `8ab21ecaf3` on 2026-09-12 UTC.
