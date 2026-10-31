# Task 17 recovery preflight — Qodo 4179836584

Head inspected: `06cfe2f6a30236dcd8f81893ebf49bc3d0b78036` (verified with `git rev-parse HEAD`). Read-only technical investigation; no product/test source, index or HEAD changes.

## Verdict

**Not verified as a correctness defect under the accepted policy.** The narrower factual observation is verified: a conversations-only backup/isolated restore can retain a ChaChaNotes v77 `agent_chat_start_attempt_id` without its AgentRuns attempt. This retains provenance and recovery input, not automatic execution authority. The suggested requirement to preserve authority for “continued automatic work” contradicts ADR-126 decision 11 and ADR-219 decision 7. Restoring both stores also does not authorize automatic replay.

There is an availability difference: omitting the workspaces group omits AgentRuns run history, allowance accounting and attempts. That is the selected group's data, not a broken foreign key within the restored conversation database. Existing explicit retry works without the original attempt and runs as new manual work while retaining machine provenance. No lost transcript/input, automatic replay, allowance bypass, or blocked explicit retry was demonstrated.

Recommendation to root: dismiss the proposed authority-preservation bug with policy and test evidence. Optional bounded regression coverage may document independent history recovery; do not couple stores solely to satisfy this comment.

## End-to-end evidence

1. **Backup owners and selection.** `DB/recovery_core.py:498–516` declares ChaChaNotes dependencies on content/assets, not AgentRuns. `discover` at 40–69 conditionally removes unused asset dependencies. `DB/recovery_operations.py:1299–1307` declares the opposite dependency: AgentRuns requires ChaChaNotes. Its default location is a sibling `agent_runs.db` (79–80). `Backup_Recovery/data_groups.py:135–166` places ChaChaNotes in `conversations` and AgentRuns in `workspaces`.
2. `Backup_Recovery/group_selection.py:11–48` narrows payloads using `resolve_inventory_groups` while retaining intentional exclusion evidence. `data_groups.py:307–329` closes actual selected dependencies; it does not reverse them for archive creation. `capture_service.py:233–249` still discovers all owners under maintenance. `DB/recovery_core.py:220` and `recovery_operations.py:295` capture real SQLite snapshots through installed adapters; candidate dependency validation runs at `Backup_Recovery/capture.py:480`.
3. **Validation.** `DB/recovery_core.py:251–299` checks exact catalog/version, local foreign keys and integrity. ChaChaNotes dependency validation at 380–425 checks actual asset references. It contains no AgentRuns lookup. The checkpoint migration `DB/migrations/chachanotes_v76_to_v77_agent_chat_starts.sql` makes the attempt field bounded and unique for machine-origin rows, but has no cross-database FK. `DB/recovery_operations.py:106–116` explicitly says historical IDs/leases/pending states are evidence, not local claims; relocation does not rewrite them to appear locally approved.
4. **Selected restore.** `Backup_Recovery/restore_plan.py:765–776` resolves archive groups with the current target only for replacement. `restore_groups.py:102–161` follows the archive's directed dependency graph. Thus isolated conversations restore may omit AgentRuns. For replacement, `required_target_groups` at 70–99 expands an active target dependent group; `resolve_archive_groups` at 150–156 rejects a required target group unavailable in the archive. An included target AgentRuns store declares the ChaChaNotes dependency, so replacement is not the same case as a fresh isolated restore. Missing/unused target stores do not create active dependencies.
5. **Staging and publication.** `Backup_Recovery/staging.py:412–446` invokes installed candidate dependency checks; publication validates installed artifacts/dependencies at `publication.py:2551–2600`. Finalization at `publication.py:591–610` binds the restored generation to activation-required installed owners before completing recovery. It does not derive activation from imported receipts.
6. **Restored execution gates.** `Agents/activation.py:26–60` observes actual AgentRuns, ChaChaNotes and workspace/config sources; its `RecoveryAdmissionGuard` at 91–97 gates installed agent entry points. `Agents/agent_service.py:28` and `Chat/console_agent_bridge.py:46` use this shared guard. Imported history remains inspectable while provider/run creation is refused until local owner review.
7. **Ordinary restart recovery.** `Chat/console_runtime.py:4235` starts the runtime audit. `console_fleet_wake.py:940–1008` runs it once, waits for readiness, and retains an admission fence on failure. `DB/automatic_work.py:1207–1261` replaces the runtime owner, marks foreign prepared/accepted chat-start attempts `review_required`, retains uncertain reservations/charges and pauses their allowance roots. Therefore including AgentRuns does not resume accepted starts automatically.
8. **Live automatic authority.** `Chat/console_chat_start.py:51–54,112–126` requires a live coordinator-owned object/session incarnation; saved provenance cannot construct it. `Agents/automatic_work_runtime.py:70–112` requires a process-local accepted latch plus the exact persisted attempt/owner and active ledger. `DB/automatic_work.py:727–749` rejects missing attempts/owner mismatch. There is no path that reconstructs live start authorization merely from the ChaChaNotes receipt.
9. **Opening/retrying recovered conversations.** `Chat/console_chat_store.py:3115,3222–3283` hydrates dispatch recovery from the conversation repository before publishing a restored session. `console_dispatch_repository.py:447` reconciles the checkpoint; parsing at 1699–1714 verifies machine provenance internally against the saved user metadata. `console_chat_controller.py:13555–13675` handles explicit retry under `manual_work_scope`; live machine continuations become `explicit_manual_retry`, and reopened recovery streams as manual work. UI caller: `UI/Console_Modules/prompt_queue.py:635`. No old AgentRuns attempt is required for this explicitly requested new work.

## Policy and lessons

- ADR-126: September 15 selectable-group amendment (30–39), decision 7 (176: completing recovery is not permission to start), decision 11 (227–238: definitions/history restored without active authority; fresh local review each generation).
- ADR-219: decisions 5–7 (52–77): one immediate attempt, no pending automatic continuation/retry queue, restart review without automatic resend. Native receipt compatibility clarification (117–139) retains exact historical catalogs and a narrow metadata-only staged migration; it does not authorize continuation.
- ADR-158 governs migration allocation/order, not backup dependency policy. ADR-208 makes migration SQL the executed source; neither warrants rewriting a shipped migration for this feedback.
- Read recovery-related lessons: `lessons-testing-evidence.md:303–343` (native profile admission), 17547–17556 (isolate all profile roots), 17999–18008 (exact recovery schemas after migration rebases), 18383–18387 (postcommit recovery cleanup). Used existing canonical pytest profile isolation and real SQLite tests, not constructors mocked away.

## Caller census

No production function is proposed for editing. For the possible shared-boundary alternative, grepped all production callers:

- `resolve_inventory_groups`: `data_groups.py` definition; `models.py:56`; `group_selection.py:24,85`; `restore_groups.py:113,125`; `capture.py:209`; `recovery_service.py:1179`; `later_rollback.py:80,123,2105`.
- `resolve_archive_groups`: `restore_plan.py:770`; `destinations.py:199,286`; `recovery_service.py:501,582`.
- Core `validate_dependencies` entry consumers: `capture.py:480`, `replacement.py:82`, `staging.py:437`, `publication.py:2593` (the latter two discover the optional method dynamically).
- `read_chat_start_attempt`: only automatic context `mark_accepted` and `check` at `Agents/automatic_work_runtime.py:73,94` outside its definition.
- `retry_dispatch_recovery`: only product UI caller `UI/Console_Modules/prompt_queue.py:635`; implementation and tests above.
- Runtime wake `start_recovery`: `Chat/console_runtime.py:4235` (other same-name methods belong to backup service).

## Actual targeted verification

Command run from the inspected worktree:

```sh
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:randomly --basetemp=/private/tmp/pr2995-task17-recovery-preflight \
  Tests/Chat/test_console_chat_start.py::test_reopened_machine_retry_is_explicit_and_retains_provenance \
  Tests/Chat/test_console_chat_start.py::test_live_machine_retry_starts_manual_work_without_reassigning_old_allowance \
  Tests/DB/test_automatic_chat_starts.py::test_recovery_marks_foreign_starts_and_fences_old_owner \
  Tests/Backup_Recovery/test_restore_data_groups.py::test_target_reverse_dependency_adds_required_group_to_review \
  Tests/Backup_Recovery/test_restore_data_groups.py::test_unavailable_reverse_dependency_refuses_instead_of_retiring_unselected_data \
  Tests/Backup_Recovery/test_activation_agents.py::test_restored_agent_execution_is_inert
```

**10 passed in 17.21s, exit 0.** One preserved `PytestCacheWarning`: sandbox denied `.pytest_cache/v/cache/nodeids` in the worktree; this occurred after test execution, not at admission or assertion. No full suite, installs, network service calls or private data exports.

The first test (903–966) writes an actual machine-origin checkpoint/provenance through the real ChaChaNotes repository, restores a new store without AgentRuns, asserts zero provider calls before the explicit retry, then verifies manual work origin and preserved provenance. It patches retry context resolution and disables agent runtime: it proves the repository/retry behavior, not a complete ZIP-to-fresh-process restore. The second uses the native controller and real AgentRuns to prove explicit retry does not reassign/refund the old allowance. The DB restart case expands to prepared/accepted. Activation cases expand to turn/resume/bridge/inspection in fresh subprocesses with real stored sources. The two group tests prove the pure dependency resolver, not a complete filesystem restore.

## Bounded follow-up proposal, if root wants a direct archive regression

**No product changes required by demonstrated behavior.** Smallest meaningful added test should reuse the existing native receipt creation and backup restore harnesses. Create actual ChaChaNotes v77 and AgentRuns v22 with an accepted start, close/drain, capture a conversations-only archive (and separately select only conversations from a full archive), restore isolated, and assert: exact receipt/provenance survives; AgentRuns attempt is absent; installed activation is required; no provider/network call occurs on open; explicit manual retry after normal local review works. Keep the existing old-attempt charge test as its sibling control. A replacement case should assert existing AgentRuns reverse dependency adds workspaces or refuses when unavailable. Do not fake the adapter validation or infer full restore from manifest-only grouping.

Bounded optional affected files: one existing `Tests/Backup_Recovery` restore test module (prefer `test_selective_restore_service.py` after choosing its closest actual harness), plus `Tests/Chat/test_console_chat_start.py` only if sharing its exact native fixture is necessary. No dependency, schema, migration or production edits recommended. Proposed new IDs: `test_conversations_only_restore_keeps_machine_receipt_inert`, `test_conversations_restore_requires_existing_target_agent_runs_group`.

If the owner instead chooses mandatory pairing as a new data-availability policy, amend **ADR-126's September 15 selectable-group amendment / decision 2 dependency contract**, and clarify **ADR-219 decision 7** that backup pairing preserves historical attempt/allowance evidence but never execution authority. Then bound implementation to `_CoreAdapter.discover` and `validate_dependencies` in `DB/recovery_core.py` plus focused real-SQLite backup/restore tests. Reuse existing dependency closure and installed candidate validation; require matching referenced attempts rather than mere file presence. Do not change the v77 migration, revive an imported runtime owner, or add a generic cross-store framework. This policy choice is not implied by the existing ADRs and remains root-owned.
