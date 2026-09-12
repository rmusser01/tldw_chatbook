# Scoped agent messaging implementation plan

> **For agentic workers:** Use subagent-driven-development to implement this plan task by task, with targeted tests and a task review before the next implementation task. Do not create nested agents, commit, stage, reset, or alter unrelated work. Steps use checkbox syntax for tracking.

**Goal:** Let a live child report bounded progress to its conversation's supervisor, with explicit collection, user inspection/discard, and existing steering for relay.

**Architecture:** One native-runtime message store owns bounded conversation inboxes. Each child receives only a bound sender; each active primary receives a bound reader. Runtime dispatch enforces scope, cancellation, continuation policy, cycle handling, and body-free metadata; the Console inspects queues without consuming them.

**Tech Stack:** Python 3.11+, stdlib locking/dataclasses/JSON, existing agent runtime, Textual 8, pytest. No dependencies or schema changes.

**Spec:** [Scoped messaging design](../specs/2026-09-08-scoped-agent-messaging-design.md)

ADR required: no new ADR; implement the accepted contract.
ADR path: [ADR-136](../../../backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md).
Reason: Direct implementation of the reviewed cross-agent contract. [Review M1–M7](../reviews/2026-09-09-scoped-agent-messaging-review.md) is part of the acceptance authority.

## Global constraints

- The user's relay-first approval persists. No direct peer addressing, progress wake, automatic approval, fresh chain allowance, durable inbox, or implicit retry.
- Only live threaded fleet children report; only live primaries collect. Children receive no coordinator/reader/arbitrary-address API.
- Automatic readers collect only their exact known immutable work chain. Manual readers can collect earlier-turn reports in the same conversation.
- Report bounds: 2,000 body characters, 4,000 compact serialized envelope characters; valid UTF-8; reject ASCII controls except tab/newline/carriage return. Agent display label at most 80 characters; service identities at most 128 characters.
- Pending per child: 8 entries / 16,000 body characters. Lifetime per child run: 32 entries / 64,000 body characters. Per conversation: 32 entries / 64,000 body characters. Per native runtime: 256 entries / 512,000 body characters.
- Collection: at most 4 whole reports / 8,000 serialized result characters, further bounded by the run's effective tool-result cap. Zero/negative legacy result cap means use the 8,000 message cap, not an unbounded result.
- Never truncate an admitted/collected report silently. If the first eligible report cannot fit a consumer's smaller cap, refuse without removing it; explicit user discard recovers capacity.
- Coordinator lock precedes progress-store lock where both are required. No callback, SQLite, provider, or policy work under queue locks. Terminalization/revocation/admission serialize; observers refresh outside locks.
- Session close invalidates the inbox before worker cancellation; runtime disposal closes all inboxes. Screen detach does neither. Late callbacks and old capabilities cannot mutate replacement queues.
- Productive reader calls must not trigger false loop detection. Only a trusted positive collected count resets no-progress history; empty/refused reads retain cycle protection. All calls retain run/model limits.
- Refuse every restored pending messaging call before dispatch. Completed native results replay as recorded; executing remains ambiguous. Persistent-primary reader calls retain ADR-063 executing/result barriers. Children gain no durable-primary checkpoint path.
- Metadata projections, AgentStep args/results/summaries, DB steps, and live/resumed marker/rail previews are body-free for these tools. Full provider results and permitted private history/capture remain separate.
- Queue availability is session-only. Primary native continuation history can retain collected content with full capture disabled. Inspection/discard does not erase existing history copies.
- Keep new UI behavior in Console modules/widgets, preserve keybinding conventions, and do not raise the existing screen-size ceiling. Explicit user discard is not a model tool or an approval flow.
- Targeted tests only. No full-suite run, provider network request, commit, staging, or integration is authorized by this plan. Existing unrelated working changes remain intact.

## Task 1: Bounded inbox owner and coordinator lifecycle (TASK-32489)

**Read first:** the task file and ADR-136 queue/identity/lifecycle sections.

**Files:**
- Create `tldw_chatbook/Agents/fleet_messages.py`.
- Modify `tldw_chatbook/Agents/fleet_coordinator.py` only for optional inbox binding and sender revocation.
- Create `Tests/Agents/test_fleet_messages.py`.
- Extend `Tests/Agents/test_fleet_coordinator.py` for actual finish/post/prune behavior.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class MessageIdentity:
    handle_id: str
    run_id: str
    parent_run_id: str
    chain_id: str | None
    agent: str

@dataclass(frozen=True)
class ProgressMessage:
    message_id: str
    identity: MessageIdentity
    body: str

@dataclass(frozen=True)
class CollectedMessages:
    content: str
    collected_count: int
    remaining_count: int
```

`MessageError(ValueError)` exposes a fixed `code` and never interpolates a body
or caller ID. `MessageStore()` provides `open_inbox(conversation_id: str) ->
MessageInbox`, `close_inbox(conversation_id: str) -> None`, `close() -> None`,
and `pending_counts() -> dict[str, int]` (nonzero counts only).

`MessageInbox` provides `sender(identity: MessageIdentity) -> MessageSender`,
`reader(run_id: str, *, chain_id: str | None, automatic: bool) -> MessageReader`,
`revoke_sender(handle_id: str) -> None`, `snapshot() -> tuple[ProgressMessage, ...]`,
and `discard(message_ids: Sequence[str]) -> int`. Only one reader may be live per
inbox; closing it permits the next primary. Automatic missing-chain admission
refuses. Unknown/foreign discard IDs cannot affect another inbox.

`MessageSender.send(message: object) -> str` returns the generated message ID;
`close() -> None` revokes the capability and retires its lifetime counter.
`MessageReader.collect(max_chars: int = 8000) -> CollectedMessages`,
`pending_count() -> int`, and `close() -> None` operate only on that exact live
capability. Capability validation uses object identity and owner membership,
not knowledge of a string ID. Every returned envelope is immutable/detached.

Coordinator constructor gains optional `message_inbox: MessageInbox | None =
None`. `bind_progress_sender(handle_id: str, *, parent_run_id: str, chain_id:
str | None) -> MessageSender | None` checks the exact live handle and attached
run ID while holding its lock, derives the immutable identity, and binds once.
`finish` revokes the sender in the same critical section before releasing the
live handle. Pruning retains admitted progress independently of handle history.

- [x] Write failing tests for actual FIFO, immutable identity, child share, lifetime exhaustion after drain/discard, aggregate contention, owner replacement, reader exclusivity, and automatic-chain filtering. Start with a concrete consumer assertion:

```python
store = MessageStore()
inbox = store.open_inbox("conversation-a")
sender = inbox.sender(MessageIdentity("h", "r", "p", "chain-a", "reader"))
sender.send("Cursor pagination is required.")
reader = inbox.reader("primary", chain_id="chain-a", automatic=True)
batch = reader.collect()
assert batch.collected_count == 1
assert "Cursor pagination is required." in batch.content
assert batch.remaining_count == 0
```

- [x] Run the new targeted file; verify failure is the missing capability, not a malformed fixture. Record the red command/output.
- [x] Implement the stdlib-only owner. One store lock guards its inbox queues, counts, capability membership and closed state; inbox operations use that lock. Enqueue validates body/envelope before locking, then checks every allowance before appending/incrementing. Reject before any mutation. Use fixed errors such as `invalid_message`, `message_too_large`, `queue_full`, `sender_limit`, `unavailable`, `reader_busy`, and `result_limit_too_small`.
- [x] Serialize the final collection response under the bounded lock before removal; reserve space for the accurate remaining count and wrapper. Skip ineligible chains without consuming them. Return compact JSON `{status: "collected", messages: [...], remaining: N}` with whole envelopes. Empty collection returns an empty array and zero eligible count.
- [x] Add exact-boundary tests with Unicode, escaping, 2,001-character input, a small consumer cap, concurrent last-slot admission, and a discarded snapshot racing a new post. Exercise real coordinator `finish` versus send; late callbacks refuse and accepted reports survive prune.
- [x] Run `pytest Tests/Agents/test_fleet_messages.py Tests/Agents/test_fleet_coordinator.py Tests/Agents/test_fleet_steering_mailbox.py -q` using `.venv/bin/python -m pytest`, then Ruff check/format on new files. Review the scoped diff, record evidence in TASK-32489, and leave changes uncommitted.

## Task 2: Live runtime tools and ownership (TASK-32490)

**Consumes:** all Task 1 interfaces exactly as specified above. Read TASK-32489's
implementation report before editing callers; do not reimplement its queue.

**Files:**
- Modify `Agents/agent_models.py`, `Agents/tool_catalog.py`, `Agents/agent_runtime.py`, and `Agents/agent_service.py` under `tldw_chatbook/`.
- Create `tldw_chatbook/Agents/fleet_message_tools.py` for schemas, bounded receipts, trusted collected-result type and body-free projections.
- Modify `Chat/console_agent_bridge.py`, `Chat/console_runtime.py`, and `Chat/console_chat_controller.py` for store ownership and close hooks.
- Create `Tests/Agents/test_fleet_message_tools.py` and `Tests/Chat/test_fleet_message_lifecycle.py`; extend `Tests/Agents/test_provider_continuation_runtime.py` for restored messaging refusal and execution barriers. Update affected steering assertions in `Tests/Agents/test_fleet_send_to_agent.py` and `test_fleet_steering_mailbox.py` to include the existing continuation metadata where appropriate. Align the related `Tests/Chat/test_fleet_execution_ownership.py` automatic-owner fixture with the required real accepted context if its pre-existing missing-context failure is confirmed, preserving all ownership assertions.

**Interfaces produced:** `REPORT_TO_SUPERVISOR_TOOL_NAME = "report_to_supervisor"`
and `READ_AGENT_MESSAGES_TOOL_NAME = "read_agent_messages"`, reserved in the
runtime name set. Define schemas with exact arguments (report: one `message`
string; reader: empty object; `additionalProperties: false`). `MessageToolResult`
extends the existing `ToolResult` with a trusted `collected_count: int = 0`.

`LoopDeps.report_to_supervisor: Callable[[dict], ToolResult] | None = None` and
`LoopDeps.read_agent_messages: Callable[[dict], ToolResult] | None = None` keep
the original parsed dict for strict service validation. Their dedicated branches
must refuse when callbacks are absent, not fall through to catalog invocation.

`ConsoleAgentBridge` owns/injects one `message_store: MessageStore` shared by all
its conversation coordinators. It exposes `progress_snapshot(conversation_id:
str) -> tuple[ProgressMessage, ...]`, `progress_counts() -> dict[str, int]`,
`discard_progress(conversation_id: str, message_ids: Sequence[str]) -> int`,
`close_progress(conversation_id: str) -> None`, and `close_all_progress() -> None`.
These inspection methods never allocate an absent inbox, launch work, or touch
completion attention. Add a non-creating lookup on MessageStore if needed for
these production consumers; document its exact name in the report.

- [x] Write a service-level two-child regression with controlled provider replies. Child A reports; primary collects; an explicit existing `send_to_agent` relay steers B; exact next provider payload contains the report/steering. Assert a child's forged reader/steering call is refused, with no sibling access.
- [x] Run to see the missing tool disclosure/dispatch failure before implementation.
- [x] Bind the child sender only after `attach_run` and capture immutable parent/child/chain identity. Preserve the supervisor-only fleet variable; give the child only its sender callback. Check cancellation, kill switch, live execution owner and automatic context before each call, and close the primary reader in the run's `finally` path.
- [x] Bind the primary reader independently of spawn allowance and avoid creating/disclosing messaging for a deliberately tool-less primary with no existing coordinator/inbox. A coordinator with existing reports remains readable even with zero spawn allowance. Put capability-gated instructions in the actual request prompt; align preview/first-request schema planning where those plans claim parity.
- [x] Strictly validate arguments and use fixed body-free failures. Map `CollectedMessages` into `MessageToolResult(content=batch.content, collected_count=batch.collected_count)`. Reporting success names the generated ID, says queued/session-only/no wake, and never echoes the body.
- [x] In the pure loop, defer cycle evaluation only for the reader, using the trusted result type/count. Productive read clears no-progress history; empty/refused read retains the detector. Test more than three productive batches and repeated empty/mixed calls. No other tool gains an exception.
- [x] Refuse restored pending messaging calls before invoking either callback. Complete that refusal via the existing continuation failure-result path; no executing/queue side effect occurs. Preserve committed-result replay and ambiguous executing behavior. Test a replacement inbox containing a new report remains untouched by restored pending read.
- [x] Separate provider tool data from AgentStep metadata. Before `add` persists or invokes a hook, message-tool args/results/summaries become fixed IDs/count/reason projections. Test with full capture off that the report appears in the provider result/private primary checkpoint and not DB step JSON, rail, or live/restored markers. Do not erase the protocol result or claim child checkpoint persistence.
- [x] Wire actual session close to `close_progress` before cancel callbacks and runtime disposal to `close_all_progress` before worker shutdown. Keep screen detach inert. Old sender/reader objects remain closed after new inbox creation. Use real bridge/controller gates for navigation and replacement tests; no fresh chain or wake event on a report.
- [x] Run targeted new tests plus existing fleet send, steering, automatic-work ownership, preparation, provider-continuation and runtime tests touched by the patch. Ruff-check new files; compare legacy diagnostics to the preserved baseline instead of sweeping unrelated style changes. Record task evidence and leave changes uncommitted.

Baseline note (2026-09-10): existing steering payloads carry
`EXCHANGE_CONTINUATION_KEY=True`, consumed/stripped by `Chat/local_reasoning.py`.
Five old fleet-send whole-dict expectations omit that key. Preserve production
metadata and align the fixture's exact role/content/metadata expectation; do not
weaken ordering, approval, or cancellation assertions. The pre-implementation
run is recorded at `/tmp/scoped-messaging-implementation-baseline.txt`.

## Task 3: Console inspection and user discard (TASK-32491)

**Consumes:** Task 2 bridge `progress_snapshot`, `progress_counts`, and
`discard_progress`. Bodies are available only by explicit snapshot read; count
projections do not include them. Use actual resolved conversation identity,
never the screen's current session after an async operation has begun.

**Files:**
- Create `tldw_chatbook/Widgets/Console/console_agent_progress_modal.py`. Add its styles to `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate `css/tldw_cli_modular.tcss` using the repository CSS builder, preserving unrelated style changes.
- Modify `tldw_chatbook/UI/Console_Modules/agent.py`, `fleet.py`, and `left_rail.py`; extend the existing inspector section/action plumbing in `Widgets/Console/console_inspector_section.py` if required. Add minimal `UI/Screens/chat_screen.py` callback wiring only if needed; do not raise the screen size ceiling.
- Modify the existing conversation navigation count projection in `UI/Console_Modules/agent.py` and its direct consumers `UI/Console_Modules/workspace.py`, `Workspaces/conversation_browser_state.py`, and `Widgets/Console/console_workspace_context.py` as needed to expose pending progress in existing navigation. Preserve historical sub-agent counts; progress counts are a separate field and include no bodies.
- Create `Tests/UI/test_console_agent_progress.py`; extend the affected fleet/inspector tests.
- Update `Docs/User_Guide/console/agent-runs-and-tools.md`, ADR-136 delivery status, and the orchestration review ledger.

**Interface produced:** `ConsoleAgentProgressModal` accepts a captured
conversation identity plus `load: Callable[[], tuple[ProgressMessage, ...]]` and
`discard: Callable[[Sequence[str]], int]`. It renders literal text with source
labels, session-only queue wording, and separate selection/discard controls.
Selection IDs come from the last displayed immutable snapshot, not a fresh
select-all during discard. UI callbacks refresh only if their captured owner is
still current. Poll while mounted using existing bounded refresh conventions;
stop timers on unmount. No report-triggered toast or model wake.

- [x] Write a rendered test: enqueue two reports through a real store; inspect without consumption; select one; concurrently enqueue a third; discard; assert exactly the selected ID is removed and the remaining two are visible. Verify completion attention and lifetime count do not change.
- [x] Run that test before introducing the modal to establish red evidence.
- [x] Add the modal and a visible existing-Agent-section action/count. Keep access available after terminal handles are pruned and when agent mode is off. Render message content as literal text and suppress control effects; do not mount hidden body previews in navigation metadata.
- [x] Update count refresh on model collection/user discard and navigation selection. A saved stale view cannot discard from another conversation; terminal/pruned rows are not required for access. Runtime-full copy directs the user to pending counts in existing conversation navigation.
- [x] Add rendered tests for zero/live/pruned counts, mode-off access, model collection while modal open, stale close/reopen, navigation away/return, and sidebar/inspector metadata with no body leakage. Respect existing non-terminal-convention keybindings.
- [x] Document queued versus collected, no wake during idle/wait, explicit relay, discard scope, no durable inbox, and ADR-063 private history. Exercise targeted UI tests and backend integration; no full suite. Verify only this feature's files, close task notes/criteria via CLI after review, and leave changes uncommitted.

### Task 3 implementation evidence

The Console extension is implemented and independently reviewed.
The task report is `.superpowers/sdd/2026-09-10-scoped-agent-messaging/task-3-report.md`.
It records real-owner rendered tests, source-snapshot diffs, static baseline
comparison, and wide/narrow screenshots. The Agent action polls body-free counts
and refreshes existing navigation only when those counts change. The modal captures
one exact inbox and distinguishes highlighting for inspection from selected discard
IDs. Existing historical counts and fail-closed recovery remain intact.

Review corrections before handoff: SelectionList retains only its prompt's first
line, so literal content uses a separate scrollable detail pane. Navigation evidence
crops the owning row's compositor strips; matching the same phrase in the Agent
action is insufficient. Progress-bearing rows preserve precomputed line breaks to
prevent Textual rewrapping metadata over the count line. Affected legacy fleet and
controller harnesses attach the existing real DB helper before recovery.

Task 3 review found the behavior compliant and visuals ready to ship, with one
Important lint/evidence correction (I1). The original static JSON contained SIM102
in the new test helper despite the report and orchestration ledger claiming zero.
The nested condition is flattened while preserving bounds checks before hit testing.
Fresh focused Ruff and formatter checks pass on both new UI files; the regenerated
baseline comparison has zero introduced diagnostics and exits nonzero if any remain.
The original JSON is preserved as `task-3-fix-1-static-before.json`; exact commands
and output are in `task-3-fix-1-report.md` beside the task report. This mechanical
test-only correction required no behavioral rerun or new capture. Scoped re-review
approved the correction. The final combined identity finding was subsequently
fixed and approved as recorded below.

## Review and delivery

### Final review correction: stable progress ownership

ADR required: no new ADR
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md
Reason: Clarifies the accepted live conversation owner's identity and disposal;
no schema, provider authority, queue migration, or new messaging capability.

The final combined review reproduced reports stranded by temporary-conversation
Save through the actual submit/child/SQLite path. TASK-32490 and TASK-32491 include
an added acceptance criterion for this correction before implementation.

1. Preserve final-fix before-images and reproduce the saved identity failure with
   the actual controller/bridge/threaded-child harness, not a synthetic inbox seed.
2. Give native sessions an eager, nonserialized progress-owner token. Share it
   atomically among restored live siblings; separate it from persisted run/fleet
   identities. Under a narrow identity lock, validate/open/bind and release the
   owner without allowing late setup to reopen it after final close.
   Register one typed current MessageStore reference with the native store so
   direct close, pristine rollback, and state replacement invalidate owners too;
   replacing this registration closes the previous store, never its successor.
   Capture expected ownership before controller preparation/worker scheduling and
   validate it at bridge entry and inbox binding, including reused native IDs.
3. Resolve preview, primary collection, count projection, inspection, and cleanup
   using that owner. Keep modal fencing on the exact native session and inbox.
   Preserve one sibling's queue when another closes; last binding closes before
   cancellation. Runtime replacement keeps exact old capabilities invalid.
4. Verify real Save success, rollback, cancelled await, close during write, live
   producer across Save, later actual primary read, shared siblings, lifetime and
   chain bounds, and rendered modal/navigation behavior. Run affected tests and
   before-image static checks only; record all output.
5. Generate one final-fix review package, run one scoped independent re-review of
   the confirmed finding and fix-induced regressions, then record dispositions
   and complete task criteria/status only when supported by that evidence.

Detailed implementation checkpoint:
`.superpowers/sdd/2026-09-10-scoped-agent-messaging/final-fix-1-proposal.md`.

- [x] Before Task 1, record a plan preflight matrix: Task 1→2 sender/reader contracts, Task 2→3 bridge methods, all shared files and task-local test/implementation consistency.
- [x] Preserve before-images of every existing file touched. Since the baseline is an intentionally dirty shared checkout, generate scoped review diffs from those snapshots, not `HEAD` (which includes unrelated work). Never stage to manufacture a review artifact.
- [x] Review each task's spec compliance and quality before proceeding to its dependent implementation. Use bounded fix/re-review rounds; record corrections in the plan ledger and owning task.
- [x] At completion, review the combined messaging patch, run only the affected test groups and static checks, and update implementation status with actual evidence. Full feature completion requires all three tasks and their acceptance criteria, not just the pure queue.

### Completion evidence

The final combined review found one Important identity defect. The single final
fix dispatch preserved the eager owner across Save and fenced stale queued/setup
work against replacement. Its final affected run passed 400 tests, including 14
identity and 7 styled UI cases; a separate queue/coordinator/tool run passed 84.
The nine corrected Python files have zero introduced Ruff diagnostics and no
remaining formatter changes overlapping the correction. The independent scoped
re-review closed I1 with zero new findings. Commands, limitations, dispositions,
and retained review artifacts are linked from the
[implementation review](../reviews/2026-09-10-scoped-agent-messaging-implementation.md).
Source/tests remain frozen after verification; completion edits update only task
status and documentation. No full suite, provider network call, staging or commit.
