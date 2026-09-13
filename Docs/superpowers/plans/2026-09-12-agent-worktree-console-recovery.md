# Console worktree confirmation and recovery

> **For agentic workers:** Use subagent-driven-development. Root owns Git, Backlog and governance; workers leave source unstaged.

**Goal:** Make same-turn worktree actions visibly confirmable and recorded earlier work recoverable through Console.

**Architecture:** A real inline card projects the retained controller round. A small Console adapter opens a paged recovery list; a runtime-owned helper captures current session authority and owns each recovery worker/cancellation event independently of agent turns and disposable views. Both paths use the same shared recovery operation engine.

**Tech Stack:** Python 3.12+, Textual 8, current ConsoleRuntime/interrupt host, existing design tokens, SQLite/Git off the UI thread.

**Spec:** Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md

ADR required: no new decision
ADR path: backlog/decisions/155-agent-worktree-recovery.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: implements the approved retained confirmation/recovery design and existing UI language.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`. Root owns Git/Backlog/docs. Workers do not spawn agents.
- Existing test interpreter and Tests/conftest isolation; this plan's copied runner uses unique evidence labels/basetemp. Targeted tests only, no whole UI/full-suite sweeps, dependency changes, live configuration/provider calls or foreign cleanup. Every launched temporary thread/task must be joined or observably owned through completion.
- Exact owning conversation and current selected writable named repository, never viewed workspace fallback or scratch. Mutation needs fresh post-consent authority and source validation through the shared engine.
- Recovery operations belong to retained ConsoleRuntime, with their own Event and ExecutionOwner. Unmount/navigation must not drop an active worker or borrow another turn's Stop Event. Session close cancels only that session's recovery; app disposal cancels all owned recovery. A stopped asyncio waiter cannot erase the still-running physical worker owner.
- Worktree rounds remain on the existing separate interrupt-host path; no new approval queue. Exact request IDs and Allow/Deny only, no remember setting. The real surface hook controls preview/live tool disclosure.
- This is a local extension of the established Console, in Operate mode. Preserve PRODUCT.md/DESIGN.md and use `backlog/docs/design-language.md`, existing cards/modals and `$ds-*` tokens. Read Impeccable craft-floor immediately before UI edits; no new identity/concept interview or unrelated design-file repair. Root already ran Impeccable context once; do not rerun it.

### Task 1: Visible card and retained manual recovery

**Files and responsibilities:**
- Create `Widgets/Chat_Widgets/worktree_confirm_card.py`: explicit source, destination, action, bounded diffstat and exact-ID Allow/Deny. Create lazily when first needed to preserve startup import budgets. Put its decision Message on the already-loaded ChatTaskCards if needed, following QuestionAnswered.
- Modify `Widgets/Chat_Widgets/chat_task_cards.py` and `UI/Screens/chat_screen_state.py`: pending_worktree_merge volatile state, lazy card routing and visibility. Snapshot restore must not resurrect stale confirmation; do not serialize automatic source bodies/locators into durable resume data.
- Create `UI/Console_Modules/worktree.py`: narrow disposable UI adapter (task state projection, exact decision forwarding, recovery-list launch). Keep ChatScreen changes to composition/hook/delegation, no new policy in its giant class.
- Create `Widgets/Chat_Widgets/worktree_recovery_dialog.py`: paged recorded-work list, read-only status/reason, Apply/Merge/Discard actions and Close. No action secretly approves; selecting one dismisses the list and starts the existing inline confirmation flow. Use a standard modal shell, vertically scrollable content and existing Button states. Long paths wrap without displacing actions. Disable mutation for held/uncertain/in-flight/resolved records with a readable reason. Empty state identifies the current conversation/repository; missing authority states how to select a writable named repository.
- Create `Chat/console_worktree_recovery.py`: retained helper for list operations and independent recovery task/owner/event lifecycle. ConsoleRuntime owns it lazily and delegates start/cancel/close; do not store it on the screen.
- Modify `Chat/console_runtime.py`: add real `set_pending_worktree_merge` disposable hook slot; remount worktree rounds independently before unified-approval early returns; runtime helper lifecycle integration for session close/dispose.
- Modify `Chat/console_chat_controller.py`: optional explicit operation cancel Event for request_worktree_merge_confirm; capture manual recovery authority from the exact owning session selection with the same fresh binding/root/kill-switch checks as accepted turns; thread real surface flag into both preview calls.
- Modify `Chat/console_agent_bridge.py`: both preview builder entry points accept/forward `worktree_merge_enabled=False` consistently with live planning. Reuse public `runs_db` and `runtime_capacity` for recovery.
- Modify `UI/Screens/chat_screen.py` and `UI/console_command_provider.py`: hook the narrow adapter and add `Console: Recover agent work…` with direct help text. No new global keybinding; no screen-size budget increase.
- Modify `css/components/_agentic_terminal.tcss` using existing tokens, rebuild `css/tldw_cli_modular.tcss` through `css/build_css.py` only.
- Tests: new `Tests/UI/test_console_worktree_recovery.py`, existing `Tests/Chat/test_console_worktree_merge_confirm.py`, relevant controller/bridge preview fixtures, runtime attachment/session-close and task-state nodes, targeted CSS/startup/screen-size guards including `Tests/Performance/test_screen_preimport_payload_budget.py` and `Tests/Architecture/test_screen_size_ratchet.py`.

**Engine contract consumed:**

```python
recover_agent_worktree(db, *, authority, conversation_id, run_id, action,
                      request_confirmation, should_cancel)
# returns WorktreeRecoveryOutcome(action, message, state, commit_sha=None)
# or WorktreeRefusal(reason_code, message)
```

`AgentWorktreeRepository(db).list_for_conversation(conversation_id, workspace_id=..., binding_id=..., limit=50, after_run_id=...)` returns metadata-only rows. No step/transcript hydration. For listing, capture the exact session identity/selected binding, perform blocking DB/filesystem validation off the UI thread and refuse drift. A later click re-captures authority; list metadata is never mutation authority. Do not revive old fleet handles.

The retained helper may expose `list_work(session_id, after_run_id=None)` and `start(session_id, run_id, action)` with stable typed outcomes; keep its task registry local to recovery lifecycle, not a duplicate physical-capacity ledger. Allocate existing runtime capacity as a manual execution, retain the owner until the worker finishes, and close only worker-thread-created DB connections. Its cancellation Event is passed explicitly into the controller confirmation method and the shared engine. Do not use `_active_cancel_events[session_id]` for manual recovery. Track shielded physical-worker completion so closing a disposable list/card or cancellation of an asyncio waiter cannot prematurely call finish_root. No unrelated turn is stopped. Retain a bounded operation receipt for remount or deliver it to the owning conversation only; do not append another conversation's receipt to the currently viewed transcript.

The new confirmation DTO field is volatile. The visible card renders agent-influenced values with markup disabled and escapes control characters in single-line fields. Show a brief action sentence, separately labeled source and destination, a scrollable diffstat, and the discard retained-baseline consequence. Two buttons: Allow once and Deny. Construction defaults hidden without an on_mount hide race. A stale/missing request ID cannot authorize anything. Clear/disable decision controls after one click; re-syncing the same payload must not undo a pending decision.

- [ ] Write RED mounted card tests: markup-like filenames stay text, exact IDs survive Allow/Deny, duplicate/stale/missing IDs cannot decide a newer round, hide/show construction order works, long source/diffstat remains usable at wide and narrow terminal sizes.
- [ ] Write RED real controller→bridge→AgentService.run_turn test using real temporary Git and SQLite: a spawned child writes, physically drains, requests merge, card becomes visible, exact Allow applies changes; Deny preserves them. Verify preview/live schema equality with real hook wired and absent.
- [ ] Implement narrow card/hook/preview wiring and separate remount before unified-decision early returns. Add session-switch and suspend/reattach tests retaining exact request ID and independently pending worktree/unified rounds.
- [ ] Write RED two-turn/reopened-DB recovery test: new service/runtime lists recorded prior work by exact conversation/current binding, then user-confirmed apply/discard reaches the real Git engine. Verify wrong conversation, read-only/scratch selection, held/uncertain rows, changed source/authority during human wait and duplicate action refusal.
- [ ] Implement retained manual helper and recovery dialog/palette action. Verify a pre-set old primary Stop Event and another active turn's Stop cannot deny recovery; owning session close cancels it and leaves other session work intact. Gate the physical worker and verify owner is retained after cancellation until actual completion; post-effect uncertainty remains protected on reopen.
- [ ] Verify interface in one batched wide/narrow mounted screenshot round under Tests fixtures, fix all material layout gaps once, then at most one confirming round. Export owned screenshots/artifacts to this plan's evidence directory. Root performs visual inspection; no live user app/config imports. Use native Textual evidence, not the web detector.
- [ ] Run focused tests and directly affected CSS bundle/token/import/screen-size checks. Compare scoped static diagnostics to the source base, qualify inherited failures separately, never raise an architectural budget to silence a new regression. Write report with evidence, actual UI action and retained lifecycle behavior; leave source unstaged for independent code and visual review.
