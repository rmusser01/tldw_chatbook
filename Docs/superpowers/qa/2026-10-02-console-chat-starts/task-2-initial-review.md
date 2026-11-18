### Spec Compliance

- ❌ Issues found: worker cleanup releases automatic capacity early; human preflight decisions occur after acceptance; prepared starts still disable the visible Send button; one preparation-refund path reports a false not_started; blocked outcomes never reach chat-row/activity state.
- ✅ Both durable fences precede mark_accepted() and started: console_chat_controller.py:10870/:10919; console_chat_start.py:327.
- ✅ Blank-provider ABSENT and explicit human Retry rulings are consistent with spec:66/:248.
- ⚠️ Required recursive start→child/wake integration case is not established. Native-start test verifies one target generation; capacity tests exercise claims directly. Tests/Chat/test_console_chat_start.py:287/:542; task-2-brief.md:229.

### Strengths

- Approval records bind frozen defaults to live incarnation/primary ownership; explicit comparisons retain authority despite compare=False: console_chat_controller.py:16750/:16782; console_chat_store.py:1378.
- Revision-checked draft edits use operation-owned connections; consumption/request receipt commit together. Real SQLite edit/clear/reopen cases: chat_persistence_service.py:1403/:1532; test_console_chat_start.py:235/:253.
- Machine provenance is durable and malformed provenance stays untrusted; duplicate acceptance checks saved body/identities/authority: message_metadata.py:451; console_dispatch_repository.py:141.
- Baseline supports checkpoint-fixture repairs, preserving production SQL/receipt guards: task2-baseline.log:4400/:5699; test_console_dispatch_checkpoint_repository.py:55/:1084.
- Existing evidence supports 700-pass Chat group, scoped lint/ratchets; separate real PTY/local-provider QA. Streaming uncertainty qualified honestly: task2-chat-final.log:37; task2-static-final.log:2; task2-postcommit-ratchets.log:1; qualification-receipts.json:84/:149/:214/:432/:761.

### Issues

#### Critical

None identified.

#### Important

1. **Target Stop releases physical automatic capacity while its worker is still running.**
tldw_chatbook/Chat/console_chat_start.py:298 releases the claim after submit_draft returns. Existing agent path awaits asyncio.to_thread at console_chat_controller.py:26959; cancellation returns stopped at:27168 without joining the thread; stream wrapper removes ownership at:24919.
Focused check held actual bridge worker behind Event, stopped target, awaited coordinator tasks: worker_exited=false, automatic_claims=0, target stopped, attempt completed. Another start/wake can enter early. Retain exact worker-completion owner; release only after settlement. Mounted Stop test releases barrier immediately after click and cannot establish this requirement (Tests/UI/test_console_runtime_ownership.py:2889).

2. **Project-instruction decisions occur after the draft is consumed and started is reported.**
Acceptance/receipts at tldw_chatbook/Chat/console_chat_controller.py:10870/:10919. Later _run_agent_reply can await binding selection at:26479 or dispatch consent at:26599. Multiple eligible folders or first-use instruction consent consume draft/generation before discovering required human decision.
Named risk/check inspected existing project-instruction entry points for pre-acceptance refusal. Spec:165 requires not_started, retained draft, no dialog continuation. Perform admission before both fences with bounded required-action reason.

3. **Prepared native starts still disable the actual Send control.**
tldw_chatbook/UI/Console_Modules/prompt_queue.py:614 overrides only accepted starts. Prepared remains Preparing.../send_enabled=False through:127; UI/Screens/chat_screen.py:21871 disables Send. Controller/dispatcher withdrawal fixes cannot help disabled button.
Focused check paused native readiness and called actual presentation: is_prepared=true, controller refusal None, Preparing..., enabled false. Project prepared starts manually sendable before ordinary busy presentation; mounted click must reach withdrawal.

4. **Withdrawal during ledger preparation ignores an unconfirmed refund.**
tldw_chatbook/Chat/console_chat_start.py:206 ignores abort_chat_start boolean and returns not_started/source_unavailable; exception becomes not_started/preparation_refused. Bypasses later _run uncertainty handling because task not created yet.
Focused check paused preparation after reservation, stopped source, forced abort=False: not_started/source_unavailable, one generation reserved, zero coordinator tasks. Report review_required for unconfirmed settlement; retain preparation ownership through cancellation/shutdown. Existing settlement tests only later _run path (Tests/Chat/test_console_chat_start.py:1322).

5. **Blocked/review launch outcomes are lost from chat rows and activity.**
tldw_chatbook/Chat/console_chat_controller.py:16929 saves only pending text/revision/provenance, not requested mode/outcome. Early disabled/capacity refusals console_chat_start.py:145 set no target activity. Observer only refreshes/toasts (UI/Screens/chat_screen.py:24144).
Named risk/check existing projections: rows ordinary active/open-session (UI/Console_Modules/workspace.py:2773); activity queue/run state (Chat/console_prompt_queue_coordinator.py:160). Refused indistinguishable from deliberate draft after toast/restart; actual REFUSED_START receipt lacks outcome (qualification-receipts.json:374). Persist bounded outcome and project existing status owners as spec:280 requires.

#### Minor

- Inherited diagnostics remain evidence debt: baseline FD+1080; later recovery+325/+305, inherited invalid-escape. Not established Task2 regressions; clean GC-enabled runs do not prove cleanup. Track separately: task2-baseline.log:6803/:6822, task2-ledger-recovery.log:3915, task2-recovery-fixed2.log:299.
- Document plan deviation: brief lists modifications to test_chat_create_confirm_card, test_console_chat_store, test_chat_persistence_service, test_console_prompt_queue_coordinator, test_console_dispatch_recovery but no patch. Much coverage consolidated in new module; record substitution/resolve missing scenarios rather than imply every owner changed (task-2-brief.md:35).

### Assessment

**Task quality: Needs fixes.** Approval/persistence foundations well evidenced, but acceptance ordering, manual control, physical concurrency and truthful outcomes need fixes. Task2 remains In Progress.
