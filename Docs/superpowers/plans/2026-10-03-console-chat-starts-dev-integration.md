# Console chat starts on current dev implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the user-authorized Console workspace/casual chat-start PR against dev with current integration evidence and only feature-related changes.

**Architecture:** Apply the reviewed feature delta to a new dev-based branch in the existing managed worktree. Preserve dev's newer migrations, durable hook receipts, root-fork metadata, recovery catalogs and native lifetime owners. The shared automatic allowance remains the original canonical owner; the bounded start does not gain new Stop-scheduler eligibility.

**Tech Stack:** Python 3.12, Textual 8, real SQLite, pytest, Ruff, Git/GitHub CLI.

**Spec:** `Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md` at immutable reviewed source `53065745187aaf4d47fc4357bb9096aa28fd5673`; snapshot supplied in this plan's SDD directory until the feature patch restores the canonical file. Existing feature tasks TASK-34202/TASK-34203 and follow-ups TASK-34203.1/TASK-34203.2 bind the deliverable.

ADR required: no
ADR path: `backlog/decisions/211-console-chat-destinations-and-bounded-starts.md` (existing), with current ADR-163, ADR-126, ADR-158 and both exact ADR-147 provider-routing/archive contracts retained.
Reason: integrate the approved design with already-shipped owners and migrations. No new continuation policy, runtime, dependency or permission authority is introduced.

## Global Constraints

- Frozen dev is `f0ffcf9e819b577bd38c416f38550969c75fb5a0`; reviewed source is `53065745187aaf4d47fc4357bb9096aa28fd5673`. Keep the original `codex/console-chat-starts` branch intact. Publishing branch is `codex/console-chat-starts-dev` in `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`.
- Apply only the delta after `c49be0691652557c4a32db54a1604e4ea90cac97` through the reviewed source. This includes the three feature design commits and all feature/baseline work, excluding the earlier 43 unrelated local documentation commits.
- Preserve destination=same_workspace|casual, mode=draft|start and same_workspace/draft defaults; source focus/workspace/composer custody, fresh destination defaults/scratch/grants, full supplied-body disclosure and explicit override semantics remain binding.
- Started requires both exact durable stores before provider dispatch. Exact destination/runtime/source/target checks, draft revisions, immutable machine provenance, no replay, conservative uncertainty, original canonical allowance/deadline and physically retained cleanup remain binding. Manual Send/Retry stays explicit user work; fork_chat retains its existing contract.
- AgentRuns uses 21→22 for chat starts; ChaChaNotes uses 75→76. Preserve shipped AgentRuns 19/20/21 and ChaChaNotes 74/75 meanings, data, SQL artifacts, hook receipts and current exact recovery schema validation. Do not weaken validation or relabel a stale catalog.
- Stop scheduler receipt chain IDs are distinct from automatic allowance IDs. Native agent_chat_start does not become a queue-owned Stop parent. Preserve refusal of unsupported automatic follow-ups rather than minting a fresh human allowance. Existing ordinary queue-owned hook continuation behavior remains governed by ADR-163.
- Use current native DB ownership, maintenance and hook admission owners. No direct raw-thread DB escape, uncounted cleanup, replacement-owner publication, second approval, fresh capacity claim or awaited gap at the source withdrawal/acceptance cutoff is allowed.
- All 21 historically failing baseline nodes and their seven complete modules require current evidence, plus feature/migration/continuation/recovery/maintenance owners named below. Preserve observable assertions; no new skip/xfail, warning suppression, raised FD limit or full repository sweep. All21 recorded baseline nodes must pass. Record the unchanged dev strict XFAIL (TASK32873 bare-screen timer harness) and historical diagnostic archaeology SKIP separately; neither result qualifies that behavior.
- Reuse `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`; retain current test bootstrap/offline/null-keyring/profile isolation. Do not manually repurpose HOME or opt into live services. Every worktree write/test/cache/commit requires scoped exec_command require_escalated because the managed root was not added to the sandbox.
- Root owns task/plan/QA closure, final task-ID reconciliation, push/PR and attachment. Worker owns port/code/test fixes and implementation commits. Never touch the shared stash, source checkout/user files, main/dev branches, or push/merge from a worker.
- Original QA and its archives remain historical evidence for their original SHA. New integration results go into this plan's scratch and final QA; never rewrite historical version names/results to imply current-dev qualification.

---

### Task 1: Port the approved feature into dev's current native contracts

**Files:**
- Apply: every feature file in the immutable binary patch `/private/tmp/console-chat-feature-only-01a0fa6c.patch` (manifest `/private/tmp/console-chat-feature-only-01a0fa6c.json`).
- Additional Modify: `tldw_chatbook/DB/recovery_core_schema.py`, `tldw_chatbook/DB/recovery_operations.py`, and existing constructor/SQL-validation/schema-derived artifacts affected by the installed schema.
- Rename: `tldw_chatbook/DB/migrations/agent_runs_v18_to_v19_chat_starts.sql` → `agent_runs_v21_to_v22_chat_starts.sql`; `chachanotes_v73_to_v74_agent_chat_starts.sql` → `chachanotes_v75_to_v76_agent_chat_starts.sql`; `Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py` → `test_chachanotes_v76_agent_chat_starts_migration.py`.
- Test: original feature/21-node owners and the existing continuation, recovery and maintenance modules below. Add narrowly scoped integration regressions beside the native start and migration owners.
- Controller-owned: Backlog/plan/QA currentness, task-ID collision, final publication.

**Interfaces:**
- Consumes: ADR-211 native start/allowance/draft contracts; dev `ContinuationReceipt`/`ContinuationAdmission`, `ConsoleDurableTurnAcceptance.continuation_receipt/user_root_fork`, `run_owned_db_call`/`operation_owned_connection`, hook preflight and maintenance/native shutdown owners.
- Produces: fresh and migrated schemas22/76; combined exact acceptance and lifecycle behavior with current tests, retained historical recovery variants and no extra Stop scheduling authority.

- [x] **Step 1: Read the requirements, immutable source and compatibility assessment before edits.** Read this task's Backlog file, supplied spec/ADR snapshots and `/private/tmp/console-chat-dev-compatibility-01a0fa6c.md`. Read current relevant ADRs, design-language/component-patterns and relevant test/lifetime lessons. Inspect the actual helper contracts rather than treating the assessment as authority.

Snapshot old-file formatting against frozen dev before edits. Derive existing Python paths from the supplied patch's Git path inventory; missing new files require full formatting. Capture command argv, env, exit code and output for every check.

```python
from pathlib import Path
import json, subprocess
scratch = Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration')
manifest = json.loads(Path('/private/tmp/console-chat-feature-only-01a0fa6c.json').read_text())
assert manifest['sha256'] == '31099e87d4c252475d366ac413f7bf311d2e11e338d921026a8eea93896a98db'
assert subprocess.check_output(['git', 'merge-base', '--is-ancestor', manifest['dev'], 'HEAD']) == b''
```

- [x] **Step 2: Apply the complete feature-only patch once and reconcile actual conflicts.** Keep the dev side's unrelated/new functionality and add the feature semantics from the immutable source. Do not replace a whole controller/store/repository with the old source. Both original branches remain reachable.

```sh
git apply --3way --index /private/tmp/console-chat-feature-only-01a0fa6c.patch
git diff --name-only --diff-filter=U
```

Resolve each reported path, then verify there are no unmerged index entries. Record how each production conflict preserves both contracts. The original unmerged design TASK33802 collided with a landed dev task. Controller reconciliation preserves the landed identity and maps the complete unmerged chain to TASK34201/34202/34203 before final preflight/publication. Do not edit historical archives.

- [x] **Step 3: Qualify and reconcile schema/storage compatibility with RED/GREEN evidence.** Rename only the feature's unshipped migrations/test as listed; use AgentRuns current22 and ChaChaNotes current76. Retain dev's 73→74/74→75 runner/map and existing AgentRuns constructor fields/history; add 75→76 and22. Update the rebuilt checkpoint temporary table suffix to76. Preserve every old checkpoint field/index, continuation receipt, auxiliary failure reason, definition cap, routing snapshot, worktree row and reservation.

The feature historical bootstrap test uses75 instead of73 and standalone SQL advances75→76. Add real predecessor rows and compare them after constructor/SQL upgrade and repeated reopen; use the existing dev v75 receipt fixture to carry a settled hook receipt. Preserve old73/74 upgrade assertions in dev's receipt tests while changing only their final installed-version expectation to76. Genuine old AgentRuns fixtures must drop the new attempt table before referenced chain/run tables when reconstructing historical schemas.

```python
# Literal final expectations for the existing real constructor tests:
assert db._get_db_version(db.get_connection()) == 76
assert not db.get_connection().execute('PRAGMA foreign_key_check').fetchall()
# Existing predecessor dict equality excludes only the new nullable field:
assert migrated_checkpoint.pop('agent_chat_start_attempt_id') is None
assert migrated_checkpoint == predecessor_checkpoint
```

Recapture exact current core/combined-subscriptions/AgentRuns schema SQL from real installed constructors. Add v76/v22 catalog entries and preserve qualified historical variants and explicit supported migration routes. Never substitute a number on old SQL or relax exact-schema comparisons. Run the schema/recovery owners listed below and retain pre-fix failures.

- [x] **Step 4: Preserve the combined durable receipt and machine-input contracts.** Merge both dataclass/fingerprint sets. The native agent-start, hook-continuation and ordinary root-fork metadata cases must be exclusive; incompatible combinations refuse before publication. Retain hook gate consumption in the same real transaction, exact receipt dedupe, original root-fork metadata, agent-start retry equality, exact draft CAS and checkpoint fencing. Exercise successful controls for all three legitimate cases and refusal controls for mixed provenance, mismatched receipt and either-store failure.

Keep current hook continuation text/context carriers, skipped human UserPromptSubmit/trusted-input/history authority and native agent-start exclusions together. Neither machine origin may become manual by a merged default branch.

- [x] **Step 5: Integrate native finite operations and hooks with current lifetime owners.** Use the actual current `run_owned_db_call`/`operation_owned_connection` contracts for start prepare/refusal/outcome/settlement. Capture exact DB/ledger/bridge/owner across waits. Include native-start tasks in shutdown/maintenance drains and refuse both preparation/acceptance and shared primary admission when maintenance is paused. Preserve retained/shielded tasks until actual worker/resource exit.

Retain the final source withdrawal/target checks and durable acceptance cutoff without an unowned awaited gap. A synchronous finite acceptance must use the actual native owned connection/participant boundary; an awaited acceptance must use existing native admission serialization and committed-result ownership. Do not introduce a new lock/lifecycle framework.

Audit actual v2 provisional initialization: retain current grant/currentness/checkpoint/capacity ceilings and cleanup, refuse a preflight needing another target consent, and recheck runtime/destination/bridge after awaits. A native start consumes its shared capacity claim once. Add held-worker/cancel/maintenance/replacement controls plus a success control.

The Stop boundary is a literal native invariant, alongside configured-hook success/denial controls:

```python
# After real native start acceptance/completion using _native_start_rig:
assert target.id not in controller.prompt_queue_coordinator._chains
assert target.id not in controller.prompt_queue_coordinator._stop_parents
assert runs.automatic_work.snapshot(chain).used['generation'] == 1
assert store.persistence.db.get_connection().execute(
    'SELECT COUNT(*) FROM console_hook_continuation_receipts'
).fetchone()[0] == 0
```

Use a configured v2 Stop proposal to verify no escaped follow-up/provider send/new allowance. Preserve current ordinary human queue Stop controls (one initial+at most3 continuations, original elapsed budget). Children/wakes/recursive chat starts retain the canonical allowance root; hook scheduler IDs never become ledger IDs. Manual Send/Retry stays a fresh explicit user allowance.

- [x] **Step 6: Verify every historical failure and complete affected owners on the integrated bytes.** Use literal argv arrays, separate basetemps, current root conftest isolation, `TLDW_TEST_GC_EVERY=1`, and recorded return codes. First run all21 nodes from the original baseline inventory. If upstream legitimately renamed an owner, document an exact one-to-one node mapping and prove the same behavior; never drop a failing selection.

Run the complete groups in `Targeted verification` below. Add a focused integration regression for each newly repaired incompatibility. Investigate any failure before repair, retain RED/GREEN, preserve production strictness and report warnings without suppression. Run affected static/dependent inventories with the installed interpreter; full preflight is controller-owned after task-ID reconciliation.

- [x] **Step 7: Self-review and commit only owned integration changes.** Use `git -c gc.auto=0`. Report exact BASE/HEAD, all changed paths, per-conflict dispositions, migration/catalog/receipt/native controls, literal commands/exit codes/output, known warnings and tests not run. Verify committed-source equality and committed-HEAD formatter ratchets. No helpers/reviewers, full sweep, shared stash, installation, source-checkout edit, push, PR or merge. Return DONE/DONE_WITH_CONCERNS/NEEDS_CONTEXT/BLOCKED plus SHA and one-line summary; controller independently reviews.

- [ ] **Step 8: Controller review, qualification and authorized publication.** Independent task review precedes the final whole-dev-branch review; fix loops follow SDD. Qualify actual isolated Console/provider destination/mode/start/refusal/reopen behavior on the integrated tree. Verify the reconciled design TASK34201 and its foundation/feature chain per the landed-keeps-id rule, with current inbound references and provenance; re-sweep immediately before push. Preserve historical artifacts and publish current QA separately. After all required gates pass, push `codex/console-chat-starts-dev`, create the PR with base `dev`, and attach its URL. User authorization is already explicit; no repeated approval question.

## Targeted verification

### Every recorded baseline failure and complete original owners

Use `Docs/superpowers/qa/2026-10-03-console-baseline-remediation/baseline-nodes.json` after the patch restores it. Complete owners:

- Tests/Agents/test_agent_chat_create_tools.py
- Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py
- Tests/Chat/test_console_agent_project_instructions.py
- Tests/Chat/test_console_chat_fork.py
- Tests/Chat/test_console_prompt_queue_coordinator.py
- Tests/UI/test_console_launch_wake.py
- Tests/UI/test_console_runtime_ownership.py

### Feature/native/allowance/schema owners

- Tests/Chat/test_console_chat_start.py
- Tests/Chat/test_console_chat_create_confirm.py
- Tests/Chat/test_console_chat_create_integration.py
- Tests/Chat/test_chat_create_confirm_card.py
- Tests/Chat/test_message_metadata.py
- Tests/Chat/test_automatic_work_lineage.py
- Tests/Chat/test_automatic_provider_budget.py
- Tests/Chat/test_automatic_wake_budget.py
- Tests/Chat/test_console_fleet_wake.py
- Tests/UI/test_console_prompt_queue.py
- Tests/DB/test_automatic_chat_starts.py
- Tests/DB/test_automatic_work_budget.py
- Tests/DB/test_automatic_work_deadlines.py
- Tests/DB/test_automatic_work_migration.py
- Tests/DB/test_automatic_wake_attempts.py
- Tests/DB/test_automatic_runtime_owner.py
- Tests/DB/test_agent_orchestration_dev_migration.py
- Tests/DB/test_chachanotes_v76_agent_chat_starts_migration.py
- Tests/DB/test_chachanotes_v75_hook_continuation_receipts_migration.py
- Tests/DB/test_sql_validation.py

### Current dev seams changed by this integration

- Tests/Chat/test_hooks_v2_continuations.py
- Tests/Chat/test_hooks_v2_lifecycle.py
- Tests/Chat/test_hooks_v2_teardown.py
- Tests/Chat/test_console_dispatch_continuation_handoff.py
- Tests/Backup_Recovery/test_agent_runs_recovery_schema.py
- Tests/Backup_Recovery/test_core_owners.py
- Tests/Backup_Recovery/test_console_artifacts_roundtrip.py
- Tests/Backup_Recovery/test_console_maintenance.py
- Tests/Backup_Recovery/test_console_sync_maintenance.py
- Tests/Backup_Recovery/test_remaining_sqlite_participant_lifetimes.py

If a listed file is absent on the frozen tree, record it and select the exact existing owner covering that contract; do not invent a passing no-op. Counts across selections overlap and must not be summed as unique tests.

### Current-dev routing compatibility

Preserve current-dev optional provider/model/preset routing on prepared new_chat: resolve from fresh destination generation defaults, retain ADR-147 override enablement/allowlist/final-provider guards and enabled routed preset parameters, capture/disclose before approval, persist the exact resolved snapshot, and retain prepared source/destination/runtime currentness. Never fill routing from source-session settings; chat creation ignores subagent default routing.

Migrate the six legacy raw-executor new_chat routing tests through genuine preparation/approval while retaining all refusal and success assertions. Add disclosure/durable snapshot and destination-fallback controls. Existing fork routing remains unchanged.

### Latest-dev cost-display qualification

The complete telemetry owners exposed stale fixture protocols and one inherited missing idle-refresh cancellation. Preserve the 2026-09-04 current/next-send spend spec: unaccepted optimistic rows are excluded from both request context and Current; accepted/dispatched rows retain request context. Fix the existing canonical send owner to cancel its pending idle refresh before awaited send work. Retain exact keyboard-send, active-edit and refusal/idle-rearm controls; rerun complete modified owners. ADR required: no. ADR path: existing ADR-052/095 and current/next-send spend spec. Reason: restore an existing lifecycle contract without new authority or owners.
