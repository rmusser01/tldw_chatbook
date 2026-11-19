# Console chat starts: frozen-dev compatibility assessment

Read-only planning assessment; no application, branch, index, or Backlog edits and no tests executed.

- Dev: `f0ffcf9e819b577bd38c416f38550969c75fb5a0`
- Reviewed feature: `53065745187aaf4d47fc4357bb9096aa28fd5673`
- Feature code base: `9ba96ebb626dd010f14d093b0e8d40f37c715d32`
- Worktree inspected: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`

## Conclusion

Port ADR-211 into dev’s current owners. The shared allowance implementation has no competing dev edits since the feature code base, but both schema versions collide, acceptance metadata now has additional independent contracts, and dev added hook lifecycle and backup-maintenance ownership. A mechanical merge of the old controller/coordinator code is insufficient. Preserve the approved destinations, draft lifecycle, source/target cutoff, shared allowance and no-replay semantics; qualify the additional native seams explicitly.

ADR required: no new decision for the compatibility changes described here.
ADR path: `backlog/decisions/211-console-chat-destinations-and-bounded-starts.md`, with existing ADR-163, ADR-126, ADR-158 and ADR-147 constraints retained.
Reason: adapting the approved design to already-shipped schema and runtime owners. A choice to add a new hook continuation policy or generic controller framework would exceed this compatibility assessment.

## 1. Exact migration collisions and edits

| Store | Reviewed feature | Frozen dev | Integration migration |
| --- | --- | --- | --- |
| AgentRunsDB | current 19; `agent_runs_v18_to_v19_chat_starts.sql` | current 21; 19 definition wall caps, 20 worktree recovery, 21 routing snapshots | `agent_runs_v21_to_v22_chat_starts.sql`, current 22 |
| ChaChaNotes | current 74; `chachanotes_v73_to_v74_agent_chat_starts.sql` | current 75; 74 auxiliary failure reason, 75 hook continuation receipts | `chachanotes_v75_to_v76_agent_chat_starts.sql`, current 76 |

AgentRuns uses idempotent constructor DDL/ALTERs and an append-only version audit, not a runner keyed by version. Port the allowance column, immutable/direct-root triggers, attempt table and indexes into the current constructor, retaining dev’s definition, worktree, resolved routing and recovery changes. Add the version-22 audit row and rename/update the standalone SQL. Preserve historical version meanings under ADR-158. Check `Tests/DB/test_agent_orchestration_dev_migration.py`: its genuine old-schema fixture drops automatic tables and will need to remove the new attempt table before dropping its referenced chains/runs.

ChaChaNotes uses an actual versioned runner. Keep dev’s `_migrate_from_v73_to_v74` and `_migrate_from_v74_to_v75`, add 75→76 and a dispatch-map entry, and update the feature rebuild table name from `_v74` to `_v76`. Retain every manual/queued checkpoint and dev’s existing hook receipt table. The rebuilt checkpoint adds `agent_chat_start_attempt_id` and the third origin; preserve its uniqueness and origin/queue/attempt constraints. Update the feature test filename and historical fixture entry point to v75. Existing v75 migration tests that assert the current constructor ends at exactly 75 must be adapted without erasing their 73/74 history checks.

Recovery schema catalogs are another required compatibility edit, not just test constants. Dev `DB/recovery_core_schema.py` captures exact ChaChaNotes v75 SQL; `DB/recovery_operations.py` has `_AGENT_RUNS_SCHEMA` v21 variants, historical catalogs/migration adapters, and subscriptions/core combined coverage. Recapture current catalogs from real constructors at v76/v22, preserve qualified historical variants, and update only explicitly supported upgrade paths. Do not merely relabel versions or weaken exact-schema validation.

## 2. Shared allowance contracts

`DB/automatic_work.py`, `Agents/automatic_work_budget.py`, and `Agents/automatic_work_runtime.py` have **no dev delta from the feature code base**. The feature changes remain the relevant implementation:

- `allowance_root_chain_id` is nullable for existing/manual roots, immutable and direct to a null-root chain. Local chains and runs retain their own conversation; a cross-conversation start is provenance in the launch attempt, never a run-parent link.
- `_snapshot` aggregates reservations over the root and its descendants but returns the requested member’s conversation identity. `_check_admission`, `_start_automatic`, refusal rollback, pause, settlement and recovery mutate/check the canonical root.
- `prepare_chat_start` creates the target-local descendant and generation reservation; acceptance consumes once; only proven prepared refusal refunds. Recovery fences stale owners and retains unknown charges without resending.
- `AutomaticWorkContext.attempt_kind` distinguishes accepted wake and chat-start authorities; it must stay an app-owned context, not metadata-derived authority.

Do not use dev’s manual chain creation to submit chat starts. Dev controller `_stream_assistant_response_inner` around lines 27238–27258 calls `ledger.create_chain` for `WorkOrigin.MANUAL`; automatic work instead checks the supplied local chain’s conversation. Preserve feature automatic context through both submit and stream wrappers, and preserve explicit manual retry as a new user allowance.

## 3. Receipt coexistence at native acceptance

Dev added `ConsoleDurableTurnAcceptance.continuation_receipt` and `user_root_fork`. Feature adds agent-start provenance/attempt fields. All must survive the merged dataclass and acceptance fingerprint.

Dev `ConsoleDispatchRepository.insert_with_messages` validates an exact `ContinuationReceipt`, queued origin, parent assistant identity and 1–3 admitted turns; inserts the durable dedupe record in `console_hook_continuation_receipts`; and writes user-row metadata `{origin: hook, initiator: hook_continuation, parent_turn_id, stop_event_id}`. Its ordinary branch also preserves `MessageMetadata(root_fork=True)`. Feature’s replacement expression only writes agent-start metadata or `None`, so choosing that expression wholesale would erase both dev behaviors.

Use explicit exclusive branches for native chat-start metadata, hook-continuation metadata, and ordinary root-fork metadata; reject incompatible combinations. Retain the dev one-use `ContinuationAdmission` transaction contribution and the feature exact-attempt idempotence check, draft CAS consumption and request/checkpoint receipt. Receipt presence never supplies live execution authority. Keep the two databases as separate durable fences, with `started` only after both have succeeded.

## 4. Stop continuations: a compatibility edge requiring explicit qualification

Dev’s Stop continuation is a queue-owned machine entry, not an automatic wake or chat-start attempt. `Agents/hooks_v2/continuations.py` enforces fewer than three admitted follow-ups and less than 120 seconds, with foreground/revocation/drain/veto checks. `ConsolePromptQueueCoordinator` consumes one host `(parent_turn_id, event_id)`, inherits the settled AgentBudget and its elapsed wall deadline, and issues the exact live gate plus receipt. Its `ContinuationReceipt.chain_id` is a hook scheduler chain identity (initially the parent turn ID), **not** an `automatic_work_chains.id`.

Keep those identities separate. Do not feed the hook receipt’s `chain_id` into the shared allowance ledger or mint a chat-start/wake claim to represent a hook proposal.

Evidence of the integration hazard: dev `_submit_draft_inner` clears automatic context for every non-wake origin; queued hook turns therefore use the manual stream defaults unless a new exact inherited authority path supplies otherwise. Feature only preserves automatic context for wakes/chat starts, so simply layering the feature origin on dev does not establish inherited automatic allowance for a hook follow-up. A Stop follow-up that is enabled after a launched chat must not replenish the ADR-211 balance.

There is also an eligibility boundary: feature `turn_accepted` explicitly skips `AGENT_CHAT_START`, and its coordinator directly calls `submit_draft`, not `run_prompt_chain`. Dev captures Stop parents only for a queue chain with `accepted_live_turn` and a custody request. Thus the existing feature route does not itself enroll starts in dev’s Stop scheduler. Do not silently add queue ownership to make tests pass. The integration plan should state and test the intended native behavior: preserve current eligibility, and where a machine descendant is admitted, preserve the originating allowance through exact native authority rather than manual defaults. If qualification shows a broader new Stop behavior is necessary, isolate that policy question instead of conflating hook scheduler receipts with launch receipts.

Retain dev’s hook continuation text carrier, skipped UserPromptSubmit/trusted-human authority and prompt-history exclusion. Add the agent-start exclusions alongside these; neither machine origin should become a human request because the other branch was merged.

## 5. Native lifetime and maintenance owners

Dev `console_fleet_wake.py` now uses `run_owned_db_call`/`operation_owned_connection`, captures the exact ledger/database in async callbacks, and rejects admission during `_maintenance_paused`. Feature `console_chat_start.py` still uses raw `asyncio.to_thread` for prepare/abort/complete/outcome publication and a synchronous `accept_chat_start` at the source-ownership cutoff.

Adapt new finite DB work to dev’s existing owned DB helpers and maintenance lifecycle. Preserve the feature’s retained worker tasks/shielded drain so cancellation is not mistaken for resource settlement. Retain exact database/bridge references across waits; a replaced runtime must not publish into a replacement owner. Include coordinator tasks in native shutdown/maintenance drain, and add maintenance refusal to shared automatic-primary admission as well as start preparation/acceptance.

Do not naively move the synchronous acceptance cutoff to an awaited worker: feature deliberately performs the final source/target checks and durable acceptance without an intervening await. An off-loop port needs the existing operation/admission owner to serialize withdrawal vs acceptance and to publish the actual committed result after cancellation. This deserves an explicit plan step and race test, not an unreviewed implementation shortcut.

Dev’s v2 hook initialization is another new awaited preflight with capability views, provisional input ownership and cleanup obligations. Audit it for `AGENT_CHAT_START`: no second target approval dialog, no retained paused continuation on pre-acceptance refusal, and no false user-origin events. Preserve actual hook owners, currentness rechecks and cleanup joins. Verify capacity claims are counted exactly once through hook initialization and stream entry; starts and wakes share the feature’s automatic-primary claim map and manual reserve.

## 6. Targeted evidence for the integration plan

No tests were run for this assessment. Recommended targeted groups:

1. **Schema and historical preservation:** `Tests/DB/test_automatic_work_migration.py`, `test_agent_orchestration_dev_migration.py`, the renamed chat-start migration test, `test_chachanotes_v75_hook_continuation_receipts_migration.py`, and relevant `Tests/ChaChaNotesDB/test_migration_atomicity.py` cases. Real v21→v22 and v75→v76 fixtures must carry root chains, reservations, worktree rows, definition caps/routing snapshots, manual/queued checkpoints, hook receipts and auxiliary failure reasons. Include constructor/standalone SQL parity, rollback, foreign-key checks, repeated reopen and fresh/migrated shape.
2. **Allowance and attempts:** `Tests/DB/test_automatic_chat_starts.py`, `test_automatic_work_budget.py`, `test_automatic_work_deadlines.py`, `test_automatic_wake_attempts.py`, `test_automatic_runtime_owner.py`; `Tests/Chat/test_automatic_work_lineage.py`, `test_automatic_provider_budget.py`, `test_automatic_wake_budget.py`. Cover recursive starts, target children/wakes, late descendant unknown usage/overage, sibling refusal, original deadline and manual-send fresh allowance without reparenting survivors.
3. **Combined acceptance:** `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`, `Tests/Chat/test_console_chat_start.py`, `test_hooks_v2_continuations.py`, `test_message_metadata.py`. Verify ordinary root-fork metadata, hook receipt dedupe after checkpoint settlement/reopen, exclusive machine provenance, exact start retry idempotence, draft revision consumption, and either-store failure with no provider dispatch.
4. **Native hook/queue interaction:** actual v2 hook-enabled start with success control, preflight needing consent, pending required checkpoint, cancellation/revocation and delayed cleanup. Exercise Stop scheduler eligibility explicitly; for any admitted automatic descendant, assert ledger root identity, no `create_chain` renewal, original budget/deadline, finite hook cap, and foreground priority. Keep existing hook failure matrix/remaining-budget tests.
5. **Capacity/lifetime:** `Tests/Chat/test_console_fleet_wake.py`, `Tests/UI/test_console_runtime_ownership.py`, `test_console_launch_wake.py`, `test_console_prompt_queue.py`, and selected `Tests/Backup_Recovery/test_console_maintenance.py`, `test_console_sync_maintenance.py`, `test_remaining_sqlite_participant_lifetimes.py`. Race source cancellation/manual send across durable cutoff, runtime replacement, active start plus wake/manual work, maintenance during preparation and late DB completion.
6. **Recovery catalogs:** `Tests/Backup_Recovery/test_agent_runs_recovery_schema.py`, `test_core_owners.py`, `test_console_artifacts_roundtrip.py`, and the tests using `test_schema_policy_matches_installed_store`. Validate exact fresh/migrated catalogs and combined subscriptions/core shape; preserve historical acceptance.

The existing reviewed feature tests and live qualification remain historical evidence for that feature SHA. Re-run the relevant targeted groups and actual Console/provider destination/start/blocked/reopen cases on the integration result; no full suite is implied.
