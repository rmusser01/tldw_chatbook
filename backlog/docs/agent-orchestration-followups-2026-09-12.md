# Agent orchestration remaining work — 2026-09-12

Baseline: `origin/dev` at `8ab21ecaf372ad0b5cc8bca98dd12c427f4aef87`.
[PR #2631](https://github.com/rmusser01/tldw_chatbook/pull/2631) merged the
25-item audit scope. The supervisor and its managed sub-agents remain one
agent-orchestration workstream; these older open tickets are part of that area.

This inventory distinguishes current code observations from old ticket claims.
It does not treat every old failure as a current product defect.

## Reliability and resource follow-ups

| Task | Current evidence | Remaining work |
| --- | --- | --- |
| [TASK-13215](../tasks/task-13215%20-%20Fleet-approval-revocation-add-a-revoked-run-tombstone-and-close-the-residual-arm-read-windows.md) | `InterruptRoundHost.revoke_for_run` sweeps registered rounds but does not remember revoked runs. A later round starts with `revoked=False`. The old skill-script cross-lock premise is stale: the maps now share the host lock and payloads are keyed by round. | Reject prompts armed after their run was revoked; prove sibling payload preservation; warn on unowned revocable rounds. Independently revalidate the return-time race before changing that contract. |
| [TASK-15666](../tasks/task-15666%20-%20busy_fleet_session_count-prunes-the-fleet-as-a-side-effect-of-a-read.md) | Coordinator terminal pruning was already at turn start; the remaining defect was retained-owner cleanup during the count. | **Done on this branch** (`726409de10`): observational snapshot, real controller/bridge regression, preserved cleanup; 10 targeted tests passed and independent review approved. |
| [TASK-18601](../tasks/task-18601%20-%20Agent-run-step-log-is-a-single-JSON-blob-column-and-does-not-scale-to-the-raised-step-budget.md) | DB child-table storage and metadata-only reads already shipped; three AC are checked. `ConsoleRunLogModal` still receives/stores a complete `log_text` and builds one `TextArea`. | Finish viewer paging/bounded memory. Do not redo the DB migration. |
| [TASK-18929](../tasks/task-18929%20-%20Agent-loop-consecutive-denial-circuit-breaker.md) | Existing run budgets bound execution; a separate consecutive-denial streak guard remains an open proposal. | Add a per-run breaker with honest terminal messaging, reset and sibling-isolation tests. Resolve the ticket's suggested small default versus its “0 or absent disables” AC before implementation. |
| [TASK-31511](../tasks/task-31511%20-%20Agent-run-webhook-delivery-spawns-a-thread-and-event-loop-per-run.md) | `schedule_run_webhook` starts a daemon thread running `asyncio.run` per delivery. | Reuse bounded delivery workers and avoid repeated unchanged-settings reads while keeping finalization nonblocking. Requires a runtime-lifecycle ADR check. |

## User-facing capabilities and recovery

| Task | Current evidence | Remaining work |
| --- | --- | --- |
| [TASK-31210](../tasks/task-31210%20-%20Console-confirm-card-for-agent-worktree-merge.md) | Merge/discard backend and controller confirmation seam exist; the Console does not wire `set_pending_worktree_merge`. Tool disclosure fails closed. | Add the visible diffstat Allow/Deny card, correct disclosure, parked/remounted state, and a real controller-to-agent-run test. |
| [TASK-31211](../tasks/task-31211%20-%20Cross-turn-persistence-for-agent-worktrees.md) | `AgentService.run_turn` initializes a fresh `_agent_worktrees` map. | Recover previous-turn unmerged work through a confirmed path while preserving DB-backed live-owner protections. Persistent handles versus a recovery UI needs an explicit design/ADR decision. |
| [TASK-18923](../tasks/task-18923%20-%20Agent-rail-live-per-run-status-line-elapsed-and-streaming-tokens.md) | Elapsed/activity rendering already exists. Fleet rows document that token totals arrive on finish, rather than providing growing live usage. | Reconcile the partial implementation, then add honest live usage where observable and verify the requested cadence/idle teardown. |

## Older verification and program records

The following six exact old failure nodes were run on the merged baseline using
the isolated worktree interpreter: **4 passed, 1 failed, 1 XPASS**. Counts are
not added to the merged PR's CI results.

| Task | Fresh evidence | Disposition |
| --- | --- | --- |
| [TASK-19642.8.3](../tasks/task-19642.8.3%20-%20Restore-Console-fleet-and-headless-wake-authority-tests.md) | All three named `test_console_fleet_wake_safety.py` nodes and `test_a_headless_wake_takes_the_same_agent_dispatch_and_budget` pass. | Reconcile and close after reviewing their assertions against current authority and automatic-budget policy. No reproduced defect from the four original nodes. |
| [TASK-22720](../tasks/task-22720%20-%20Agent-bridge-placeholder-replacement-test-trips-the-unresolved-recovery-guard.md) | `test_citation_repair_agent_missing_placeholder_keeps_runtime_row_without_repair` XPASSes. The task's own correction already rejects its original swallowed-exception premise. | Remove the stale expected-failure marker, verify the intended recovery behavior, and close with evidence. |
| [TASK-2155](../tasks/task-2155%20-%20Agent-branch-console-send-never-invokes-agent-bridge-pre-existing-dev-failure.md) | `test_native_send_applies_conversation_dictionary_agent_branch` fails before dispatch because the manually bound conversation lacks a durable Library policy. Hydration alone failed; inserting the real policy and hydrating it made the probe pass. | Repair the test setup, preserving production durable acceptance and dictionary behavior. |
| [TASK-13154](../tasks/task-13154%20-%20Supervisor-agent-fleet-program.md) | Parent remains In Progress; discovered definition, wake, and steering children are Done. Older deferred notes include spawn validation, Settings DB ownership, and tool-filter feedback. | Reconcile the original six-phase acceptance criterion and each deferred note with merged work before closing the parent. |

Open PR #2427 was checked for overlap: it lists none of the four test files
above. No open PR or remote branch name matched TASK-13215 or TASK-15666 when
this wave started. Repeat the check before later changes or integration.

The parent program's deferred notes were also inspected:

- The mutually exclusive named-agent/skill spawn guard now raises `ValueError`;
  that old implementation note is superseded. A supplied roster has a real
  no-reread regression (`test_run_turn_reuses_planned_agent_roster_without_db_reread`).
  The older `test_definitions_load_once_per_turn_roster_in_protocol` checks
  roster disclosure, not a read count; direct-load call-count coverage still
  needs reconciliation before closing the parent's testing note.
- Settings still constructs an `AgentsSettingsPanel` on category rendering;
  its `_derive_runs_db` opens an owned `AgentRunsDB`, and the panel has no
  explicit close hook. This ownership cleanup remains open.
- `_form_definition` still drops `RUNTIME_TOOL_NAMES` from the typed allowlist;
  the Save path does not explain those omissions. Per-save feedback remains open.
- Settings category placement is a product preference, not a correctness bug.

## Additional approval-harness failures observed in this wave

The broad affected-file run for TASK-13215 exposed these **seven inherited
failure nodes**, each reproduced with the original production files restored.
They remain verification follow-ups; these observations alone do not establish
seven product defects:

| Test file | Exact node | Observed failure |
| --- | --- | --- |
| `Tests/UI/test_console_parked_payload_rekey.py` | `test_bridges_do_not_share_a_head` | Legacy mounted-payload readiness times out; failing test leaves a waiter alive. |
| Same | `test_promoted_round_mounts_with_remaining_time_not_original_timeout` | Expects wall-clock countdown, sees the original 30-second answerable-time budget. Reconcile with current decision visibility before changing assertions. |
| `Tests/UI/test_console_mcp_approval.py` | `test_finishing_card_is_not_counted_and_keyboard_focuses_the_card` | Expected card is hidden. |
| Same | `test_alt_a_focuses_the_pending_approval_decision_select` | Focus stays on `console-setup-modal-action`. |
| Same | `test_alt_a_reaches_the_card_at_80_columns_with_inspector_closed` | Same setup-modal focus mismatch. |
| Same | `test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css` | Expected row header has zero geometry. |
| Same | `test_single_row_fast_buttons_have_nonzero_geometry_and_do_not_overlap_under_bundled_css` | Expected fast-approve button has zero geometry. |

`test_falsy_run_id_is_normalized` also fails after those leaking harness tests,
but passes alone; it is not counted as an independent human-wait defect.
The affected same-session skill-script test's preregistration/publication race
was repaired within TASK-13215 by waiting for its actual badge/payload and
ensuring worker shutdown in `finally`.

The final approval selection passed **197 tests, 5 deselected** (the five mounted
MCP cases above); a separate approval/payload selection passed **30**, and the
local-provider selection passed **16**. These selections overlap and are not
summed. Both parked-payload failures above remain outside that green selection.
Whole-file formatter/lint debt and environment dependency warnings are recorded
separately; changed lines introduce no static-check findings.

## Boundaries and order

Burn down confirmed cancellation/read-path defects first, reconcile stale tests,
then address bounded log rendering and denial behavior. Worktree confirmation
precedes durable worktree recovery; live status and webhook delivery can follow
as separate, reviewable tasks.

Other existing agent-area programs (persistent memory, scheduled server work,
MCP resources/prompts, code-execution tools, and broader Settings work) require
their own scope review. They are not additional defects proven by this audit.
Module-wide formatting debt also stays separate from correctness repairs.

Direct peer addressing, progress-triggered wakes, and durable progress inboxes
remain optional designs outside [ADR-136](../decisions/136-scoped-child-progress-and-supervisor-relay.md).
The implemented communication remains bounded process-local steering/progress,
explicit supervisor relay, and versioned session tasks.
