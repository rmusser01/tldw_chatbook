# Console Prompt Queue Implementation Plan

> Execute one Backlog task at a time. Before changing production code for a task,
> read its task file, move it to `In Progress`, and add that task's implementation
> plan through the Backlog CLI. Tasks 14803-14806 intentionally remain `To Do`
> until their prerequisite work is complete.

**Goal:** Let a Console user queue up to ten text prompts behind an accepted agent
turn, manage them visibly, and drain them as separate FIFO turns with explicit
pause, failure, Stop, context-change, close, and quit recovery.

**Architecture:** `ConsoleChatStore` owns a provider-context epoch;
`ConsoleTurnExecutionContext` freezes one owning session's configuration for one
turn; a pure `ConsolePromptQueueRegistry` owns process-memory queue state; a
controller-side coordinator is the sole next-turn authority; Textual widgets render
immutable snapshots and emit typed intents. Queue text is never persisted.

**Tech stack:** Python 3.11+, Textual 8.x, asyncio/Textual workers, frozen
dataclasses, pytest/pytest-asyncio, Ruff.

**Design authority:**
`Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/098-visible-bounded-console-prompt-queue.md`

**Reason:** ADR-098 already accepts the bounded visible queue, controller
ownership, context epoch, turn-context boundary, lifecycle guards, and
process-memory scope. This plan implements that decision; it does not introduce a
new architecture choice. ADR-033 and ADR-031 remain binding for application-state
ownership and keybindings.

## Dependency order

1. TASK-14802 - store-owned conversation-context epoch.
2. TASK-14803 - immutable owning-session turn execution context. This can be
   developed independently of TASK-14802.
3. TASK-14804 - pure bounded queue registry. This can be developed independently
   of TASK-14802 and TASK-14803.
4. TASK-14805 - controller coordinator, after TASK-14802, TASK-14803, and
   TASK-14804.
5. TASK-14806 - close, navigation, and quit loss guards, after TASK-14805.
6. TASK-14808 - mounted UX, documentation, and live proof, after TASK-14806.

## Global constraints

- Use the repository interpreter at
  `C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe`.
  App-importing probes belong under `Tests/` and run through pytest; a bare Python
  probe does not receive the suite's profile isolation.
- Write behavior-changing tests red first. For every guard called out below,
  temporarily break the protected production line with bytecode caches disabled or
  cleared and confirm the named new test turns red.
- Exercise real production signatures at controller/provider seams. A permissive
  fake must not redefine a keyword-only signature or omit a real pre-ready refusal.
- Queue registry mutation is synchronous and event-loop-thread confined. Worker
  callbacks marshal to the app thread before touching it. Do not add a widget lock
  or mutate queue state from a Textual widget.
- Do not persist queue entries in the database, `TaskResumeState`, native screen
  snapshots, prompt history, diagnostics, or logs. Never log prompt bodies.
- Prompt-bearing dataclasses and exceptions must use redacted representations.
  Render snapshots contain precomputed safe previews and never copy full bodies;
  only the selected edit request may fetch one exact body.
- Do not snapshot API keys, approval grants, skill trust, or permission verdicts.
  Authority is revalidated by the existing runtime owners.
- Preserve the no-argument `on_submission_accepted` callback for manual-origin sends
  only. Queued acceptance uses a distinct content-free event.
- Keep `UI/Screens/chat_screen.py` under its one-way size and method ratchets. New
  Console UI/controller code belongs in `UI/Console_Modules/` and construction in
  `wiring.py`.
- TASK-14801 is concurrently redesigning the Console left rail. Before TASK-14808,
  read its final task notes and current files, then adapt queue markers to the
  resulting rail/session hierarchy without reverting or duplicating that work.
- If source TCSS changes, edit the source partial, regenerate the bundle with
  `tldw_chatbook/css/build_css.py`, and run the bundle-sync check. Do not hand-edit
  the generated bundle.
- UI assertions must include painted content and neighboring geometry, not only
  widget values or `display`. Re-query widgets after structural recomposition and
  poll for worker-mounted controls.
- Before each PR, run focused suites, every full test file that calls a changed
  seam, full-tree collection, the Console architecture/ratchet gates, and Ruff.
  Compare ambient failures using identical commands and failure sets.
- Live verification happens only in TASK-14808. Set `TLDW_TEST_MODE`, `HOME`,
  `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`, and `[paths] data_dir` to
  one scratch profile before importing or launching the app. Validate the scratch
  TOML after every boot and confirm the running process has no real-profile file
  handles.

---

## TASK-14802 - Add a Console conversation-context epoch

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Add: `Tests/Chat/test_console_conversation_context_epoch.py`
- Exercise existing: `Tests/Chat/test_console_chat_store_tree.py`
- Exercise existing: `Tests/Chat/test_console_chat_store_summary.py`
- Exercise existing: Console variant, retry/regenerate, rewind, persistence, and
  session-close test files that call the changed store seams
- Update: `backlog/tasks/task-14802 - Add-a-Console-conversation-context-epoch.md`

### Implementation steps

- [ ] Characterize every mutation seam with red tests. Pin the exact distinction:
      active-path content/lineage/text-variant/attachment/summary changes advance;
      ordinary append, streaming/status/persistence/feedback, off-path edits, and
      idempotent writes do not.
- [ ] Characterize failed-message retry separately: a successful retry changes an
      existing failed row into provider-visible history and advances once; a stopped
      retry does likewise because its partial becomes history, while a failed retry
      remains excluded and does not authorize unrelated epoch adoption. Stopped
      regeneration keeps its existing sibling/branch semantics.
- [ ] Add a private per-session epoch map, a read-only
      `conversation_context_epoch(session_id)` API, and one internal increment seam.
      Initialize on create/restore and purge on close.
- [ ] Add semantic before/after guards to `update_message_content`,
      `set_active_leaf`, `set_session_context_summary`, `select_variant`, branch
      creation, deletion, and generation-attachment mutations. Avoid double increments
      when a higher-level mutation delegates to another instrumented seam.
- [ ] Verify edit-resend, regeneration, sibling selection, and rewind obtain their
      increments through the store operations they already use; do not add a second
      controller-owned epoch.
- [ ] Run the focused and reached store/controller suites. Mutation-check at least
      active-path edit and same-value stability.
- [ ] Complete the task ACs, notes, ADR reference, and DoD before marking Done.

### Exit contract

No user-facing queue exists. The store exposes a tested, process-local change token
whose value cannot be advanced by normal response streaming or harmless UI
re-selection.

---

## TASK-14803 - Capture immutable owning-session Console turn context

**Files:**

- Add: `tldw_chatbook/Chat/console_turn_context.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py` only if a shared public model
  belongs there rather than in the focused module
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: the smallest compatibility wrapper in
  `tldw_chatbook/UI/Screens/chat_screen.py`
- Add: `Tests/Chat/test_console_turn_execution_context.py`
- Extend: provider selection, payload, run-state-per-session, workspace isolation,
  RAG, retry/regenerate, and send-draft snapshot suites
- Update: `backlog/tasks/task-14803 - Capture-immutable-owning-session-Console-turn-context.md`

### Implementation steps

- [ ] Write joined red tests with two sessions using different provider/model,
      system prompt, generation settings, workspace roots, Auto-RAG defaults, and
      tool configuration. Switch the viewed tab and change settings during an
      awaited validation probe; assert the first turn stays internally consistent
      and the next turn sees the change.
- [ ] Define frozen `ConsoleTurnExecutionContext` configuration records. Keep
      credentials, approval/trust decisions, cancel events, live streams, and staged
      manual evidence out of the model.
- [ ] Deep-copy mutable mappings, sequences, roots, and settings into immutable
      values. Do not retain a live `ConsoleWorkspaceContext`, session settings
      object, or mutable callback; mutate the sources after capture in tests.
- [ ] Add one owning-session resolver seam. Resolve from the session's stored
      `ConsoleSessionSettings`, workspace identity/context, current app/provider
      configuration, and capability catalog exactly once per turn.
- [ ] Replace active/viewed controller projections in attachment gates, provider
      selection, leading system messages, payload image budgets, windowing,
      fingerprinting/cache inputs, RAG defaults, direct dispatch, and agent dispatch
      with the captured context.
- [ ] Thread the same context instance from validation through provider resolution,
      payload construction, capability checks, and stream execution. Do not
      reconstruct a partial selection after an await.
- [ ] Preserve session one-shot/pinned-prefill behavior and runtime credential/tool/
      skill checks. Prove revocation still reaches the existing live authority seam.
- [ ] Run the joined tests plus every full file referencing `_provider_selection`,
      `_leading_system_message`, `_provider_message_payloads`, and
      `_stream_assistant_response_inner`; run the screen ratchet and import gates.
- [ ] Mutation-check the owning-session selection and mid-validation settings guard.
      Complete task notes and DoD.

### Exit contract

All existing Console turn types use one immutable target-session configuration.
Manual UX is unchanged and no queue is yet reachable.

---

## TASK-14804 - Add the pure bounded Console prompt queue registry

**Files:**

- Add: `tldw_chatbook/Chat/console_prompt_queue.py`
- Add: `Tests/Chat/test_console_prompt_queue.py`
- Update: `backlog/tasks/task-14804 - Add-the-pure-bounded-Console-prompt-queue-registry.md`

### Implementation steps

- [ ] Write red pure tests for immutable entry IDs, FIFO order, per-session isolation,
      `queued + claimed <= 10`, stale revisions, and session cleanup.
- [ ] Define bounded enums/dataclasses for entry, claim, queue mode, pause reason,
      reservation, mutation result, and body-free render snapshot. Precompute a
      sanitized one-line preview at admission/edit using a fixed maximum cell budget
      independent of viewport width, redact prompt-bearing representations, and
      inject stable ID and monotonic-time producers where tests require determinism.
- [ ] Implement revision-checked admission, edit, move, remove, clear-waiting, claim,
      settle, return-to-head, pause-after-turn, keep-draining, pause, resume,
      reservation, context baseline/adoption, closing tombstone, shutdown, and
      session removal.
- [ ] Keep claimed work locked. `Clear waiting` must never remove the claimed/starting
      entry. A paused queue accepts new text only at its tail.
- [ ] Make admission and final queue-empty release one registry decision that can
      return either admitted or reroute-normal-send. Pin both race winners in tests.
- [ ] Capture the creating thread and reject foreign-thread mutations, matching the
      application-state owner convention without adding async or locks.
- [ ] Assert queue models have no imports from Textual, provider gateways, database,
      prompt history, diagnostics, or screen modules. Assert snapshots contain no
      full prompt bodies.
- [ ] Prove unchanged revisions reuse a body-free snapshot without walking ten
      100,000-character entries. Cover ANSI/control sequences, Rich markup,
      multiline text, wide glyphs, and combining characters in pure preview tests.
- [ ] Mutation-check capacity and stale revision rejection. Complete task notes and
      DoD.

### Exit contract

A deterministic pure state machine owns every queue transition. It has no worker,
provider, widget, or persistence side effect.

---

## TASK-14805 - Coordinate sequential Console queued turns

**Dependencies:** TASK-14802, TASK-14803, TASK-14804

**Files:**

- Add: `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_prompt_queue.py`
- Extend: `Tests/Chat/test_console_prompt_queue.py`
- Add: `Tests/Chat/test_console_prompt_queue_coordinator.py`
- Extend: run-state-per-session, provider isolation, RAG capture, approval, skill,
  retry/regenerate, Stop, hands-free, close, shutdown, marker, and notification tests
- Update: `backlog/tasks/task-14805 - Coordinate-sequential-Console-queued-turns.md`

### Implementation steps

- [x] Add explicit manual/queued submission origin and an accepted-turn outcome that
      records session ID, assistant/user identities, terminal result, and committed
      conversation epoch without carrying prompt text into UI events.
- [x] Preserve `on_submission_accepted()` for accepted manual origin only. Add a
      separate content-free queued-acceptance event keyed by session and entry ID.
- [x] Add coordinator admission behind an accepted live turn or existing queue. Reuse
      canonical draft validation, refuse attachments/manual staged evidence intact,
      and leave recognized slash commands on their existing path.
- [x] Implement one per-session `run_prompt_chain` that awaits the active turn,
      evaluates the terminal result, honors pause-after-turn, compares context epoch,
      claims FIFO, resolves one immutable turn context, rechecks riders, submits, and
      repeats.
- [x] Retain the original global agent reservation across successful queued turns.
      Release it when empty or paused; Resume/Retry must reacquire visibly and must not
      register a hidden global waiter.
- [x] Implement failure, Stop, context-review, pre-accept refusal, and unexpected
      exception pauses. Wire Retry, Skip and resume, Resume next, Retry stopped turn,
      Keep draining, and Use current context and resume to the exact approved
      transitions.
- [x] Authorize queue recovery through a narrow typed internal capability at the
      existing generation gate. Do not add a general `force=True` bypass available
      to unrelated controller actions.
- [x] Make queued RAG capture session-targeted and origin-aware. Auto-RAG may generate
      owning-session evidence; manually staged evidence remains screen-owned and is
      never read or cleared by queued origin.
- [x] Add one immutable controller activity projection and migrate busy count, cap,
      markers, fleet summary, polling, navigation warnings, and notification
      eligibility to it. Distinguish slot occupancy, validation/preparation, accepted
      live work, approval wait, queue presence, and pause state. Intermediate
      completions remain running and do not toast.
- [x] Gate competing Continue, Regenerate, Edit and resend, Summarize, transcript
      recovery, and hands-free entry points through the coordinator whenever future
      queue work exists.
- [x] Tombstone a closing session or global shutdown before Stop/cancel. Verify
      cancellation-driven terminal callbacks cannot claim another prompt.
- [x] Add joined async tests for three ordered turns, queue-empty/admission races,
      refusal before and exception after acceptance, approval waits, global cap
      reacquisition, cross-session concurrency, context review staleness, and shutdown.
- [x] Prove accepted queued prompts enter normal persistence and prompt history once,
      in accepted order; admission, edit, reorder, and refused starts write neither.
      Assert the queue-empty/admission race emits exactly one final notification.
- [x] Mutation-check legacy-hook isolation, owning-session context, shutdown
      suppression, and intermediate notification suppression. Complete task notes and
      DoD.

### Exit contract

The controller can safely coordinate queued turns through direct APIs and tests. No
new visible shelf or manager is mounted until TASK-14806 also supplies loss guards.

---

## TASK-14806 - Guard Console queues across close, navigation, and quit

**Dependency:** TASK-14805

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` only for thin lifecycle
  delegation compatible with the size ratchet
- Modify: `tldw_chatbook/app.py` for a thin non-blocking, reentrancy-safe pre-quit
  dispatcher
- Extend: session close, navigation guard, quit, controller shutdown, and screen
  teardown suites
- Update: `backlog/tasks/task-14806 - Guard-Console-queues-across-close-navigation-and-quit.md`

### Implementation steps

- [x] Derive one content-free, revisioned lifecycle aggregate from TASK-14805's
      controller activity projection. Report exact live-run, queued-session, and
      unsent-prompt counts, including claimed pre-accept entries. Do not introduce a
      second mutable projection; keep paused queues distinct from running agents.
- [x] Replace session close with one combined transcript/live/queue confirmation.
      Extend Console leave warnings to separate live-run, queued-session, and queued-
      prompt counts, including an empty transcript with live/queued work.
- [x] Pin session-close confirmation to the requested session ID and capture the
      lifecycle revision. After any close, leave, or quit approval, re-read the
      impact; if its revision or counts changed, fail closed and present updated copy
      instead of destroying newly admitted or newly live work.
- [x] Make `TldwCli.action_quit()` dispatch one exclusive async confirmation worker
      guarded by `_quit_in_progress`. Set shutdown state and run cleanup only after
      approval; Stay or errors fail closed and preserve all Console state.
- [x] Split approved quit cleanup so blocking cache/config persistence and timed
      joins run off the Textual event loop, while app state, timers, audio ownership,
      and final `exit()` remain on or marshal back to the app thread. Preserve current
      cleanup ordering and exactly-once behavior.
- [x] Inventory every user-initiated app exit entry point and route it through that
      guard. Keep startup password cancellation and signal/forced termination outside
      the interactive guarantee and document those exclusions in tests.
- [x] Add lifecycle tests for repeated quit, confirmation failure, Stay preservation,
      combined close, cancellation-after-tombstone, and absence from persistence,
      snapshots, history, diagnostics, and logs. Prove the app loop remains responsive
      while approved blocking persistence is in progress.
- [x] Prove the controller shutdown/tombstone ordering by waking the real terminal
      transition after cancellation and observing that no subsequent queue claim or
      provider submission occurs.
- [x] Run the reached application navigation, screen teardown, session-close,
      controller shutdown, and configuration-encryption exit suites plus the screen
      ratchet. Complete task notes and DoD.

### Exit contract

Programmatically staged queue work is protected by the complete user-initiated loss
boundary before the queue becomes reachable from the composer.

---

## TASK-14808 - Deliver the visible Console prompt queue experience

**Dependency:** TASK-14806

**Files:**

- Add: `tldw_chatbook/UI/Console_Modules/prompt_queue.py`
- Add: `tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` only for thin compatibility
  wrappers/delegation
- Modify: Console composer/session surface widgets and source TCSS needed to mount the
  one-row shelf and background count labels
- Add: `Tests/UI/test_console_prompt_queue.py`
- Add: `Tests/UI/test_console_prompt_queue_modal.py`
- Extend: composer undo, send snapshot, key routing, dictation/hands-free,
  parallel-runs, lifecycle-dialog integration, geometry, and ratchet suites
- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: F1 help, fleet-marker legend, collapsed-composer, leave, and quit docs
- Update: `backlog/tasks/task-14808 - Deliver-the-visible-Console-prompt-queue-experience.md`

### Implementation steps

- [x] Re-read TASK-14801 and the resulting left-rail/session-surface code. Record the
      exact non-overlapping mount points before editing Console UI.
- [x] Build `ConsolePromptQueueUIController` for exact draft transactions and one
      per-session Textual chain worker. Keep `_dispatch_console_draft_send` as a thin
      compatibility wrapper and preserve or reduce `chat_screen.py` ratchets.
- [x] Build an always-mounted, display-managed one-row `ConsolePromptQueueRegion`.
      Render only on revision changes; use `Preparing...`, `Queue`, `Queue full`,
      `10/10 - Manage to make room`, `Starting...`, and `Send` at their exact state
      boundaries.
- [x] Route Enter, Send/Queue button, mouse, voice, hands-free, and programmatic sends
      through the same typed `sent | queued | refused` dispatcher. Refusals restore
      the exact text/cursor/selection/undo transaction.
- [x] Build the focused manager modal pinned to its owning session ID and queue
      revision, with stable selection and typed edit, move,
      remove, clear-waiting, pause/resume/recovery, review, and confirm-current-context
      intents. Fetch full text only when one selected entry enters edit mode.
- [x] Render the registry's precomputed previews without fetching queue bodies.
      Let widget layout crop the safe fixed-budget string further after terminal
      resize. Collapsed/background surfaces show count and state only; session
      switches while the manager is open never retarget its intents to the newly
      viewed session.
- [x] Integrate the already-landed session-close, leave, and quit guards with the
      mounted shelf/manager. Stay must preserve current focus and edit text.
- [x] Update F1 help, fleet/session markers, both Console guide pages, collapsed-
      composer behavior, and lifecycle copy. Add no global shortcut or configuration
      setting.
- [x] Add mounted tests with the real app stylesheet. Assert painted dynamic labels,
      key paths, safe previews, privacy, focus after recompose, unchanged-revision
      no-op polling, and every neighboring control inside 80x24, 100x30, and 160x40.
- [x] Add end-to-end mounted coverage proving the UI dispatcher, controller
      coordinator, lifecycle projection, and normal message persistence are joined;
      component-only coverage is not sufficient.
- [x] Run the isolated live walkthrough: two sessions with distinct system prompts
      and roots; queue/edit/reorder three prompts; park approval; switch sessions;
      drain; pause/resume; Stop; retry stopped; Resume next; verify one final
      notification; exercise close/leave/quit; inspect compositor output at 80x24 and
      a normal terminal size.
- [x] Run the full verification matrix, mutation checks, self-review, documentation
      review, task notes, AC completion, and DoD.

### Exit contract

The prompt queue is visible, bounded, manageable, keyboard-first, session-correct,
and protected at every user-initiated loss boundary. Accepted entries become normal
turns; unsent entries remain process-memory-only.

## Final program verification

- [x] All six task files are Done with checked ACs, implementation notes, exact test
      evidence, and ADR links.
- [x] Queue-focused pure, controller, UI, lifecycle, architecture, and documentation
      suites pass.
- [x] Full-tree pytest collection succeeds; focused Ruff passes for the new queue
      modules and tests, with no new violations in changed legacy files.
- [x] The Console screen size/method ratchets do not increase.
- [x] The eight required mutation checks turn the intended new tests red.
- [x] Live verification uses an isolated profile and confirms no real user config or
      data path changed.
- [x] No queued prompt body appears in Git diffs, logs, diagnostics, persistence,
      snapshots, or prompt history before acceptance.
