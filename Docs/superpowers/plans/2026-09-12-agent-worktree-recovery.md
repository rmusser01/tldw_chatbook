# Confirmed agent worktree recovery implementation

> **For agentic workers:** Use subagent-driven-development; implement and independently review each task sequentially.

**Goal:** Complete TASK-31210 and TASK-31211 with exact visible confirmation and durable recovery under current filesystem authority.

**Architecture:** Extend the existing contained pinned worker with closed internal worktree operations, persist base/ownership and positive physical drain, then expose same-turn cards and a controller-owned recovery list.

**Spec:** Docs/superpowers/specs/2026-09-12-agent-worktree-recovery-design.md.

ADR required: yes
ADR path: backlog/decisions/155-agent-worktree-recovery.md
Reason: durable ownership and recovery lifetime, closed mutating Git authority, physical drain and explicit discard retention.

## Execution qualification checkpoint

Task1 is incomplete. Commit `32557b80f2` repairs and independently verifies the
nested-interpreter prerequisite (four affected tests passed). Actual temporary
Git races show the proposed nested source/child pins do not retain linked
administrative metadata: a replaced parent receives the child commit. The
mutation implementation is held while that boundary is redesigned; later card
and recovery tasks must not enable it based on the fixture success. See
`backlog/docs/agent-orchestration-followups-2026-09-12.md`.

## Global Constraints

- Work only in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr, branch codex/agent-orchestration-remaining. Root owns allgit and Backlog status; workers leave changes unstaged and dispatch no subagents.
- Use .superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python under pytest isolation. Targeted tests only, real temporary Git/SQLite, no user repository operation, network, provider calls, shared environment edits, new dependencies or guard increases.
- Only exact current named writable admitted binding plus per-call confirmation authorizes mutation. Persisted paths, branch names and DB terminal status never suffice. Scratch remains nonpersistent/nonrecoverable.
- One-shot contained pinned workers, strict closed operations, no arbitrary Git argv/shell/executable. Reuse nested POSIX pins. Unsupported Windows/descriptor capabilities refuse before admission.
- Reuse ExecutionOwner; persist held/drained/uncertain writer proof and make uncertainty sticky. No new ownership registry. No automatic replay after uncertain effects.
- Logical discard removes confirmed changes but retains detached baseline checkout with an honest cleanup-pending receipt. No forced pathname root deletion or root-removal GC.
- Bound previews to 8192 characters and binary patch spools to 32 MiB before destination mutation. A confirmed child capture commit may precede an oversized-patch refusal; preserve that source commit and its original base as unresolved work, and explain the outcome. Preserve prior merge state and user destination changes on refusal.
- Follow DESIGN.md and Impeccable craft-floor before UI editing. Existing terminal tokens, no new global keybindings; wide/narrow painted inspection once, observed fixes then confirmation.

### Task 1: Closed admitted worktree authority and operations

Files:
- `Agents/local_tool_provider.py`: public capture_worktree_authority() -> RunAdmittedWorkspaceRoot | WorktreeRefusal; preserve explicit selected authority on construction; reject ambiguous/legacy/scratch/read-only/stale routes.
- `Chat/console_chat_controller.py`: `_compose_local_provider`/admitted-root construction passes exact selected binding marker; recovery later consumes same capture/guard functions.
- `Agents/agent_service.py`: `_admit_agent_worktree`, merge/discard closures use captured authority, retain associated object per handle, revalidate after consent; never provider.workspace_root for mutation.
- `Tools/workspace_tool_protocol.py`: closed internal worktree schemas/intents/companion identity validation; no general Git argv.
- `Tools/workspace_tool_executor.py`: worktree argument normalization and fixed optional identity frames; retain existing containment/result validation; propagate uncertain outcome codes.
- `Tools/workspace_tool_worker.py` / `workspace_tool_dispatch.py`: lazy closed worktree dispatch; no arbitrary executable or path target chosen by user/model.
- `Tools/workspace_root_pin.py`: support retaining secondary child while destination stays pinned, or explicit unsupported refusal. Never process-global app chdir.
- New `Agents/agent_worktree_operations.py`: actual pinned create/preview/apply/merge/discard implementations; keep old low-level helper wrappers only for compatibility tests until production routes moved.

Tests before implementation:
- Repair the root-qualified nested-interpreter fixture before treating its results as product evidence: include resolved dependency site paths inherited through the parent virtualenv .pth, preserve the pytest-owned HOME/config/data/keyring/offline environment in the outer test harness, and install the test network guard there before product imports. Keep the actual contained worker environment contract unchanged. The four affected subprocess nodes pass with this test-only control; evidence is in the plan ledger.
- The host currently refuses even a standalone multiprocessing Event with ENOSPC inside and outside the sandbox despite free disk space. Nineteen inherited semaphore-gated race tests fail before product execution. New deterministic process tests must use owned Pipe barriers with bounded waits and finally cleanup, proving the real root operations without depending on named semaphore availability. Do not skip assertions, replace subprocesses with threads, kill foreign processes, or change host kernel limits. Retain the inherited baseline limitation separately from new product evidence.
- Extend `Tests/Tools/test_workspace_tool_protocol.py`: reject unknown fields/options, invalid branch/base/UUID/companion identities, wrong intent, arbitrary external child paths; ordinary operations unchanged.
- Extend `Tests/Tools/test_workspace_root_pin.py`, `test_workspace_tool_executor.py`: root replacement before admission refuses; rename after successful pin acts only on admitted root; companion replacement refuses; no process CWD change; Git child containment/timeout/cleanup-unproven result.
- Add `Tests/Agents/test_agent_worktree_authority.py`: real temp repo, explicit authority, read-only/removed/retargeted binding/scratch/ambiguous roots/legacy fallback refusal; post-confirm downgrade no mutation.
- Update `Tests/Agents/test_agent_worktree.py` targeted Git helpers to assert source and destination bytes/index/HEAD. Tests for merge conflict preserve user's preexisting merge state; preview no staging; changed fingerprint demands fresh consent.
- Qualify physical discard implementation against a deterministic root-replacement gate. If selected logical discard, assert source agent changes/untracked files gone, base unchanged, detached checkout retained, exact branch deleted, unrelated tree untouched; tool copy/receipt honest.

Run only these targeted nodes, changed-line lint and whitespace. Leave edits unstaged with exact red/green and static evidence for root commit and independent review before Task2.

### Task 2: Durable ownership, physical drain and uncertainty

Files:
- `DB/AgentRuns_DB.py`: version 19 and migration hook; narrow record/list/CAS APIs, structural projections only.
- New `DB/migrations/agent_runs_v18_to_v19_worktree_recovery.sql` (recheck number first).
- Optional `DB/agent_worktrees.py` repository class to keep large DB module small; no new database/file.
- `Agents/agent_service.py`: record before starting child; normal/admission-failure/abandoned-owner transitions, no run when persistence fails.
- `Agents/agent_worktree.py`: GC preserves all recorded unresolved/in-flight/uncertain and unknown work; stop destructive pathname cleanup for recorded trees.
- `Agents/execution_capacity.py`: one-shot on_drained(callback(cleanup_proven)) dispatched outside lock; mark_cleanup_unproven sticky bit. `Agents/agent_service.py`: bind existing child_owner to precreated run before worktree admission, register DB drain callback. `Agents/local_tool_provider.py`: optional admitted-child cleanup-unproven observer at WorkspaceToolExecutionError catch. `Chat/console_runtime.py`: reuse existing RuntimeCapacity for recovery operations, no second owner registry.
- Relevant AgentRuns export/sync projections: exclude new locators/identities from exported/model-facing run payloads.

Tests:
- New `Tests/DB/test_agent_worktree_recovery.py`: v18 reopen→v19, idempotence, rollback/corrupt state, FK conversation ownership, bounded pages, exact expected-state CAS, no text blob query.
- Real two-turn/temp-Git tests: durable original base survives parent advancing; dirty and clean-unmerged preserved; failed apply leaves recoverable/uncertain durable record; another conversation's live run protected.
- Crash-order injection around creation, DB initial record, worker admission, Git effect and DB completion: no automatic reapply, in-flight→uncertain on reopen, unknown legacy rows never adopted/deleted.
- Gated `_settle_fleet` abandonment: DB terminal with still-live thread remains non-actionable; no owner-fence eviction during turn pruning.
- Privacy tests verify raw scratch locator never recorded and new worktree roots not exposed by unrelated run metadata export.

#### Owner/drain implementation and test nodes

Before storage implementation, add bounded tests to existing execution-capacity tests: callback runs once on root+last operation drain in either order, outside capacity lock (callback can call snapshot), late registration, failed-start completion, one callback exception does not suppress later callbacks, reserve after finished refuses, sticky cleanup-unproven reaches callback. Do not perform DB work under lock.

Integration tests gate an actual `_call_with_timeout` worker past logical timeout: run DB terminal, writer_state held, list non-actionable, release worker, callback writes drained, list actionable. Gate `_settle_fleet` child thread beyond terminalization with the same assertions. Gate delayed model lifeline cleanup too. Inject actual WorkspaceToolExecutionError(cleanup_unproven) at provider handler boundary and assert writer_state uncertain after owner drains; successful sibling unaffected. Close/reopen DB with drained versus held records: only drained recovers. Mutation state uncertainty and writer uncertainty remain independent.

Recovery driver uses runtime_capacity.begin_execution(MANUAL, conversation, child=False), reserve_tool immediately before admitted helper and finish in actual worker finally; finish_root after driver exits. The original child run is historical ownership, not this execution's run ID. Separate recovery cancel_event persists in runtime-owned operation state.

Logical discard acceptance: restore pinned child tracked files to recorded base, fd-relative no-follow untracked cleanup, detach at base, exact expected-old-SHA update-ref deletion, retain baseline checkout with honest receipt. POSIX capability required; Windows refuses until equivalent handle-based clean/secondary-root pin is implemented and qualified. New `Tests/Agents/test_agent_worktree_operations.py` must gate root rename/replacement and nested symlink races, verify unrelated external directory bytes untouched, preserve `.git` link and reject foreign common-dir/submodule layouts. No pathname worktree remove fallback.

Required exact negative platform evidence: a Windows-capability fixture must refuse all unqualified multi-root mutations before worker admission and leave source/destination unchanged. This is a refusal test, not Windows pinning qualification. Actual POSIX tests must run the real helper/Git in temporary repositories. Persist writer_state held/drained/uncertain and register owner callback before starting child. Sticky uncertain wins every drained CAS, including late callbacks.

Leave edits unstaged and write evidence before root independent review.

### Task 3: Visible same-turn confirmation and exact disclosure (TASK-31210)

Files:
- New `Widgets/Chat_Widgets/worktree_confirm_card.py`.
- `Widgets/Chat_Widgets/chat_task_cards.py`: compose, visibility, iterators.
- `UI/Screens/chat_screen_state.py`: optional pending worktree payload field/helpers.
- `UI/Screens/chat_screen.py`: `console_view_hooks`, exact message adapter, `CONSOLE_DECISION_CARD_SELECTORS`; narrow helpers may go in new Console_Modules/worktree.py instead of bloating screen.
- `Chat/console_runtime.py`: explicit disposable slot for actual card; detach clear/attach remount. Do not introduce inert always-present router.
- `Chat/console_chat_controller.py`: worktree round final decision lock/owner-signal correctness as needed, preview caller capability.
- `Chat/console_agent_bridge.py`: `build_project_instruction_preview_request` and `build_personal_context_preview_snapshot` flag signature/forwarding into first-request plan.
- `css/components/_agentic_terminal.tcss`: existing token card selectors; rebuild bundle with build_css.py.

Tests:
1. New `Tests/UI/test_worktree_confirm_card.py`: markup literal, action effects, bounded preview display, missing/stale exact request IDs, each button and hide state.
2. Extend `Tests/Chat/test_console_worktree_merge_confirm.py`: Allow/Deny/cancel/switch/remount/close with real runtime binding and host; detached capability disables disclosure and existing pending payload remounts on attach.
3. Extend `Tests/Agents/test_merge_discard_worktree_runtime_tool.py`: a real AgentService.run_turn wired through actual bridge/controller callback, scripted provider spawn→wait→merge/discard, visible mounted card click. Assert real temporary destination Git/bytes before and after, no side effect on Deny, real child has durable terminal/drained proof from Task2 before request.
4. Preview parity through both public preview calls, fleet/tool enable/disabled matrix, native/non-native plan if applicable. Existing surface unavailable case remains fail closed.
5. UI design-token/bundle and affected startup/preload guards only. Do not raise limits.

Record red/green evidence plus real mounted screenshot/visible-state verification. Parent reviews then closes task via Backlog CLI. No external Git operations or user checkout mutation during verification.

### Task 4: Controller-owned Console recovery (TASK-31211)

Files:
- New `Chat/console_worktree_recovery.py`: bounded candidate listing, prepare/confirm/execute domain flow and retained operation state.
- `Chat/console_runtime.py`: own recovery service/operations until settlement; exact session-close cancellation and disposal.
- `Chat/console_chat_controller.py`: capture explicit current project selection independent of previous turn, extend worktree confirmation with explicit operation cancellation owner; no reset/reuse old turn event.
- `UI/Console_Modules/worktree.py`: list/refresh/action adapters; worker scheduling, result projection only.
- New `Widgets/Chat_Widgets/worktree_recovery_dialog.py` or screen-local compact existing ListView modal pattern; no new permanent panel required.
- `UI/Screens/chat_screen.py`: action/palette discoverability, view hook wiring.
- `Agents/tool_catalog.py`: accurate same-turn tool IDs, Console path for old work and honest selected discard cleanup semantics.
- Existing user docs/help plus task implementation notes/ADR links.

Tests:
- Real SQLite close/reopen, new controller/session bound to same conversation and selected repo: previous dirty work offers apply and discard; apply lands exact bytes; denial preserves all state; successful operation cannot replay.
- Source running, foreign conversation, wrong binding, stale fingerprint, missing base, legacy unknown, owner-uncertain and mutation-uncertain rows are non-actionable with honest copy.
- Recovery owner does not bind stale primary cancel event. Session switch parks/remounts exact card; close cancels only recovery operation; double-click/second session loses DB CAS; sibling primary approval unaffected.
- Authority/source changes during user wait refuse and require new preview. Before-admission cancellation makes no changes; after-admission interruption either proven result or uncertain record.
- Mounted command→list→Apply/Merge/Discard→card→result at normal and narrow terminal sizes; actual Git fixtures, no mocked operation success.

Finish with scoped review, targeted regression nodes, lint/formatter and CSS/import checks required for touched paths. No full test sweep. Set tasks Done only after chosen mutation/deletion/owner boundary and relevant platform evidence are complete; otherwise report exact refused capability rather than declaring all task criteria complete.


Leave edits unstaged with implementation notes, exact targeted evidence and remaining limitations. Root closes both tasks only after independent review of all four slices.
