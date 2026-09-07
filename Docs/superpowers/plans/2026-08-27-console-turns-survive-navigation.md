# Console Turns Survive Screen Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task with the review checkpoints below.

**Goal:** Make every accepted Console turn and human-decision wait continue safely while the user navigates to another app screen, with exact cancellation boundaries and durable, privacy-preserving attention on hidden terminal outcomes.

**Architecture:** Extend the existing app-owned `ConsoleRuntime` into the lifetime owner for accepted-turn tasks while leaving `ConsoleChatController` and `ConsoleChatStore` as the sole execution and transcript authorities. A disposable `ChatScreen` captures immutable send-time inputs, transfers custody before clearing the composer, and later reattaches by reconciling the live store. Existing decision registries become view-independent, ChaChaNotes writes terminal rows and exact local receipt marks atomically, and the shell derives a boolean Console-attention projection from pending decisions plus durable unseen receipts.

**Tech Stack:** Python 3.11+, Textual 8.x, asyncio, frozen dataclasses, SQLite/FTS5 through the existing ChaChaNotes transaction layer, pytest/pytest-asyncio, existing Loguru and Textual test harnesses. No new dependency and no schema migration.

---

## Planning references and constraints

- **Backlog task:** `backlog/tasks/task-22514 - Console-turns-survive-screen-navigation.md`
- **Approved design:** `Docs/superpowers/specs/2026-08-27-console-turns-survive-navigation-design.md`
- **ADR required:** yes
- **ADR path:** `backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md`
- **Reason:** This changes the long-lived task-ownership, cancellation, decision-clock, and attention-persistence boundaries shared by the app, runtime, controller, screen, and shell navigation.
- **Applicable lessons:** `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`, and `backlog/docs/lessons-backlog-hygiene.md`.
- Preserve the fresh-screen invariant: never cache or hide `ChatScreen`.
- Do not grow `chat_screen.py`; new non-visual behavior belongs in `UI/Console_Modules/` or `Chat/`, as enforced by `Tests/Architecture/test_screen_size_ratchet.py`.
- Preserve existing dirty user work. Before Task 1, record `git status --short` and classify every pre-dirty overlapping path. The `git add ...` examples below assume a task-clean path; use `git add -p` for every pre-dirty/shared path, then inspect both `git diff --cached --name-only` and the full `git diff --cached` before each commit so unrelated hunks never enter this feature.
- Before Task 1's first code edit, record `git rev-parse HEAD` in TASK-22514 as the implementation base. Every task commit must contain only this feature's paths, so the final `<recorded-task-base>..HEAD` Python manifest is safe to use for lint/format verification.
- Run targeted tests throughout. Ask the user before any full-suite run.

## Acceptance mapping

| Task AC | Plan coverage |
| --- | --- |
| AC 1: ordinary and agent turns survive real navigation | Tasks 1–4 and 9 |
| AC 2: all three human-decision types notify and wait detached | Task 5 |
| AC 3: exact Stop/session-close/quit scopes | Task 8 |
| AC 4: reattach reconciliation has no duplicate/missing content | Tasks 3 and 9 |
| AC 5: deterministic, live-provider, restart, privacy, and narrow-layout evidence | Tasks 6–10 |
| AC 6: User Guide is current | Task 10 |
| AC 7: one notice plus durable exact unseen receipt | Tasks 6 and 7 |
| AC 8: frozen context/attachments without payload or privacy expansion | Tasks 1, 2, 4, and 9 |

## Task 1: Characterize the old ownership boundary and add the custody value objects

**Files:**

- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_turn_context.py`
- Modify: `Tests/Chat/test_console_runtime_lifetime.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`

### Steps

- [ ] Add failing runtime-level tests for the desired custody contract: synchronous admission registers a sensitive request before scheduling, returns a stable turn ID, retrieves task failures, and releases custody at terminal cleanup. Leave the prompt-queue launcher change for Task 2 so this task can end green.

  Run:

  ```bash
  pytest -q Tests/Chat/test_console_runtime_lifetime.py Tests/UI/test_console_runtime_ownership.py -k "custody or runtime_owned"
  ```

  Expected: new assertions fail because `ConsoleRuntime` has no custody registry or admission API.

- [ ] Add a frozen, slots-based `ConsoleTurnCustodyRequest` next to the existing immutable turn-context types. Reuse `ConsoleTurnConfigurationSnapshot`; do not introduce a second provider/settings model.

  The request should carry only the data needed after detachment:

  ```python
  @dataclass(frozen=True, slots=True)
  class ConsoleTurnCustodyRequest:
      turn_id: str
      session_id: str
      draft: str = field(repr=False)
      configuration: ConsoleTurnConfigurationSnapshot = field(repr=False)
      attachment_ids: tuple[str, ...] = ()
      staged_evidence_launch: ConsoleLiveWorkLaunch | None = field(
          default=None,
          repr=False,
      )
  ```

  Keep accepted attachment objects in a runtime-only record if controller APIs require them; do not deep-copy bytes or put names/content into `repr`.

- [ ] Add a minimal private custody record and registry to `ConsoleRuntime`. The registry owns task lifetime only: stable turn ID, owning session ID, task handle, and sensitive request references marked `repr=False`. It must not mirror run state, queue state, or terminal state already owned by the controller/store.

- [ ] Add a synchronous runtime admission method with registration-before-scheduling semantics:

  ```python
  def accept_turn(self, request: ConsoleTurnCustodyRequest) -> str:
      self._raise_if_disposed_or_session_fenced(request.session_id)
      record = self._register_custody(request)
      try:
          record.task = asyncio.create_task(self._run_custodied_turn(record))
      except BaseException:
          self._release_custody(record.turn_id)
          raise
      record.task.add_done_callback(self._finish_custodied_turn)
      return record.turn_id
  ```

  The done callback must retrieve `task.result()` so exceptions cannot become “Task exception was never retrieved,” run terminal cleanup once, and release request/context/attachment references.

- [ ] Add tests with deterministic scheduling barriers that assert:

  - custody exists before the coroutine is allowed to start;
  - task creation failure removes custody and leaves the caller in control of draft recovery;
  - request and custody `repr` omit prompt, attachment, RAG, credential, and tool data;
  - terminal cleanup removes the record and permits weak-reference collection of sensitive owned objects;
  - `ConsoleRuntime` does not duplicate controller run-state fields.

- [ ] Run the focused tests:

  ```bash
  pytest -q Tests/Chat/test_console_runtime_lifetime.py Tests/UI/test_console_runtime_ownership.py
  ```

  Expected: PASS.

- [ ] Commit only this task's files:

  ```bash
  git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_turn_context.py Tests/Chat/test_console_runtime_lifetime.py Tests/UI/test_console_runtime_ownership.py
  git diff --cached --name-only
  git commit -m "feat: add Console turn runtime custody"
  ```

## Task 2: Transfer prompt custody before clearing the composer

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/prompt_queue.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/retrieval.py`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_fleet_wake.py`
- Modify: `tldw_chatbook/Chat/console_prompt_queue.py`
- Modify: `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_fleet_wake.py`
- Modify: `Tests/Chat/test_console_prompt_queue.py`
- Modify: `Tests/Chat/test_console_prompt_queue_coordinator.py`
- Modify: `Tests/Chat/test_console_run_state_per_session.py`
- Modify: `Tests/UI/test_console_prompt_queue.py`
- Modify: `Tests/UI/test_console_composer_improvement_transaction.py`
- Modify: `Tests/UI/test_console_send_draft_snapshot.py`
- Modify: `Tests/UI/test_chat_screen_worker_groups.py`
- Modify: `Tests/UI/test_console_composer_undo.py`
- Modify: `Tests/UI/test_console_fleet_wake_ui_freshness.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/UI/test_console_parallel_runs.py`
- Modify: `Tests/UI/test_console_regenerate_feedback.py`
- Modify: `Tests/UI/test_console_skill_commands.py`
- Modify: `Tests/Chat/test_console_turn_execution_context.py`

### Steps

- [ ] Write failing prompt-queue tests for the handoff transaction:

  - immutable configuration and exact staged-evidence launch are captured for the selected session;
  - runtime custody succeeds before the composer revision is cleared;
  - a synchronous admission refusal leaves the composer and staged attachments untouched;
  - a failure after custody but before durable acceptance creates a recovery keyed by turn ID with the exact draft and attachment references, without replacing an older recovery for that session;
  - a failure after durable acceptance becomes transcript/run output and never restores unsent text;
  - queued follow-up prompts retain their own immutable input snapshot rather than inheriting the first turn's configuration or consulting a later view.

  Run:

  ```bash
  pytest -q Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_send_draft_snapshot.py -k "custody or composer or revision or staged"
  ```

  Expected: FAIL because `_stage_normal_chain` currently returns after launching a screen worker and `_submit_console_native_draft` owns recovery on the screen.

- [ ] Replace the keyboard path's destructive pre-launch `stash_draft_for_send()` with a two-step composer transaction: `capture_draft_for_send()` takes a non-destructive revision-pinned snapshot, and `commit_captured_draft(stash)` clears only that accepted revision after runtime custody succeeds while preserving any newer typing. Clicking Send and pressing Enter must use the same transaction. A synchronous refusal performs no restore because nothing was cleared.

- [ ] Change `ConsolePromptQueueUIController`'s launch seam to return a synchronous custody result instead of a Textual worker. Remove screen-owned in-flight stash ownership after the composer transaction above becomes authoritative; only the runtime's turn-keyed recovery owns a post-custody/pre-acceptance draft.

- [ ] In `wiring.py`, replace:

  ```python
  screen.run_worker(screen._submit_console_native_draft(...))
  ```

  with a small binding that synchronously captures:

  - `ConsoleTurnConfigurationSnapshot` from `_build_console_turn_execution_context(session_id)`;
  - the exact `ConsoleLiveWorkLaunch` from `_snapshot_console_staged_evidence()`;
  - the exact staged attachment IDs expected for the handoff;
  - the prompt-history/follow-intent values required by the current dispatch contract;
  - the runtime's stable turn ID and session ID.

  Then call the runtime admission method. The binding may read the mounted screen during this synchronous capture only; the resulting request and runtime task must not retain the screen or bound screen methods.

- [ ] Add one store operation that atomically transfers the exact expected pending-attachment sequence to a stable turn ID and returns the original `PendingAttachment` objects without copying bytes. `ConsoleRuntime.accept_turn(...)` performs that transfer, registers custody, and schedules on the same owner loop. If validation, registration, or task creation refuses custody, roll the exact objects back in their original order without overwriting attachments staged after the snapshot. After custody, only the custody record/controller owns those references; the controller must no longer re-read `store.pending_attachments()` for that turn.

- [ ] Add an in-memory, content-redacted `ConsoleTurnRecoveryEntry` registry to `ConsoleRuntime`, keyed by turn ID and secondarily indexed by session. If a custody task fails before the controller reports durable acceptance, move its exact draft and transferred attachment references into a new recovery entry; never overwrite or collapse an older entry for the same session. Expose ordered, exact restore/discard actions to the fresh-screen prompt-queue/composer projection. Restore re-stages only that entry's attachments and only into a live matching session; discard releases references. These entries are navigation-lifetime recovery only and must not create a second restart scanner or replace durable dispatch recovery.

- [ ] Define queue ownership explicitly as **one runtime-owned chain task plus one immutable request per queued entry**. Extend `QueuedPrompt`/the registry so queue admission captures a repr-redacted `ConsoleTurnCustodyRequest`-equivalent snapshot for that entry (attachments and staged evidence remain prohibited by the existing staged-rider gate). Change `ConsolePromptQueueCoordinator._submit_queued` and `ConsoleChatController._submit_queued_entry` to consume the claimed entry's own snapshot. Do not pass the initial request's configuration through `run_prompt_chain`, and do not resolve queued configuration from a mounted screen when the chain drains headlessly.

- [ ] Route the existing main-agent and fleet-wake submission entry points in `ConsoleAgentBridge`/`ConsoleFleetWakeCoordinator` through the same runtime custody API with their already app-owned, screen-free inputs. Preserve their wake ledger and controller origins; do not create a second agent task or duplicate bridge run state. Tests must prove ordinary, main-agent, and wake turns all appear in the runtime custody registry before scheduling and release their records independently.

- [ ] Move the domain body currently in `ChatScreen._submit_console_native_draft` behind `ConsoleRuntime._run_custodied_turn`. Keep controller calls authoritative and pass the frozen configuration and frozen staged-evidence launch explicitly into `run_prompt_chain`/`submit_draft`.

- [ ] Extend the existing controller signatures narrowly, for example:

  ```python
  await controller.run_prompt_chain(
      session_id=session_id,
      initial_turn=lambda: controller.submit_draft(
          request.draft,
          session_id=session_id,
          configuration=request.configuration,
          accepted_attachments=record.attachments,
          staged_evidence_launch=request.staged_evidence_launch,
      ),
  )
  ```

  Do not re-resolve a mounted view's `_turn_context_provider` after custody. Each later queue claim supplies its own snapshot through the coordinator contract above. Preserve the controller's queue, cap, durable preparation, dispatch recovery, and terminalization logic.

- [ ] Use `_capture_frozen_console_staged_rag(draft, context, launch)` and `_release_frozen_console_staged_rag(launch, result)` with the exact admitted launch. Refactor those operations into app-owned callables/services or values that the runtime can invoke without retaining `ChatScreen`; do not query newer staged UI state.

- [ ] Remove the obsolete screen-owned submit coroutine after its logic has moved. Migrate every direct caller or test seam found by `rg -n "_submit_console_native_draft" Tests tldw_chatbook`: behavior tests should admit through the runtime and await the specific custody task via a narrow runtime test seam, skill-command spies should patch runtime admission, worker-group assertions should no longer classify ordinary sends as screen workers, and explanatory comments should name the runtime handoff. End with no executable production reference to the removed method. Ensure the screen-size ratchet does not increase.

- [ ] Add negative architecture assertions that the accepted-turn path contains neither `screen.run_worker` nor a bound `ChatScreen._submit_console_native_draft` callback.

- [ ] Run the focused tests:

  ```bash
  pytest -q Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_prompt_queue.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_composer_improvement_transaction.py Tests/UI/test_console_send_draft_snapshot.py Tests/UI/test_chat_screen_worker_groups.py Tests/UI/test_console_composer_undo.py Tests/UI/test_console_fleet_wake_ui_freshness.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_parallel_runs.py Tests/UI/test_console_regenerate_feedback.py Tests/UI/test_console_skill_commands.py Tests/Chat/test_console_run_state_per_session.py Tests/Chat/test_console_turn_execution_context.py Tests/Architecture/test_screen_size_ratchet.py
  ```

  Expected: PASS.

- [ ] Commit only this task's files after inspecting the staged list:

  ```bash
  git add tldw_chatbook/UI/Console_Modules/prompt_queue.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/retrieval.py tldw_chatbook/UI/Console_Modules/message.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_fleet_wake.py tldw_chatbook/Chat/console_prompt_queue.py tldw_chatbook/Chat/console_prompt_queue_coordinator.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_prompt_queue.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_run_state_per_session.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_composer_improvement_transaction.py Tests/UI/test_console_send_draft_snapshot.py Tests/UI/test_chat_screen_worker_groups.py Tests/UI/test_console_composer_undo.py Tests/UI/test_console_fleet_wake_ui_freshness.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_parallel_runs.py Tests/UI/test_console_regenerate_feedback.py Tests/UI/test_console_skill_commands.py Tests/Chat/test_console_turn_execution_context.py
  git diff --cached --name-only
  git commit -m "refactor: hand Console sends to app runtime"
  ```

## Task 3: Remove surviving screen dependencies, then make navigation a pure detach

**Files:**

- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_turn_context.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/retrieval.py`
- Modify: `Tests/Chat/test_console_runtime_lifetime.py`
- Modify: `Tests/Chat/test_console_viewless_hooks.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`
- Modify: `Tests/UI/test_console_store_continuity.py`
- Modify: `Tests/UI/test_console_sync_outlives_screen.py`
- Modify: `Tests/UI/test_screen_navigation.py`

### Steps

- [ ] Replace the old-policy assertions in `test_console_runtime_lifetime.py` with failing tests that navigation:

  - clears only view hooks;
  - does not set the controller visit-cancel event;
  - does not tombstone prompt queues or cancel ordinary submit/stream tasks;
  - does not deny or resolve pending decisions;
  - leaves the same runtime/controller/store instances alive.

- [ ] Add a failing real-navigation test that starts a barrier-controlled stream, invokes the production navigation handler to Library or Settings, lets the stream complete with no Console mounted, returns through production navigation, and observes the terminal row exactly once.

  Run:

  ```bash
  pytest -q Tests/Chat/test_console_runtime_lifetime.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_console_store_continuity.py Tests/UI/test_screen_navigation.py -k "navigation or detach or reattach"
  ```

  Expected: FAIL because runtime `leave_console()` still calls controller `leave_console()` and the screen still asks for fleet-loss confirmation.

- [ ] Before enabling pure detach, enumerate every `CONSOLE_VIEW_HOOK_SLOTS` entry in a failing owner test and move every post-custody domain dependency to either the immutable request or an app-owned service. This includes dictionary/world-info application, frozen RAG capture/release, Library policy, identity/history, project/workspace authority, settings/tool policy, and accepted attachments. After this step, the only detachable hooks are repaint/focus/card projections; a custody task must retain no `ChatScreen`, widget, or bound screen method.

- [ ] Narrow `CONSOLE_VIEW_HOOK_SLOTS` to those disposable projection callbacks and add an architecture assertion that detaching all hooks during preparation, provider readiness, or streaming leaves the runtime request sufficient to continue. This ordering is mandatory: do not remove navigation cancellation while any accepted-turn domain operation still depends on a view hook.

- [ ] Split attachment identity from app-disposal generation:

  - `attach_view(view)` increments and returns a monotonic attachment generation;
  - the screen stores that token;
  - `detach_view(view, generation)` clears hooks only when both claimant and generation match;
  - a late outgoing unmount cannot detach a freshly attached successor.

- [ ] Make navigation call pure detach. Remove ordinary cancellation, queue tombstoning, decision denial, and teardown-notice behavior from `ConsoleRuntime.leave_console()` / `ConsoleChatController.leave_console()`; retain permanent cancellation only under explicit Stop, session close, and app disposal.

- [ ] Move the runtime detach to the beginning of `ChatScreen.on_unmount` and guarantee it with `try/finally`. View-owned cleanup (microphone/audio, timers, modals, animations, local workers) may still run, but a cleanup failure must not leave runtime hooks bound.

- [ ] Remove the Console navigate-away confirmation by making `ChatScreen.confirm_navigation()` return immediately or removing the Console-specific delegate. Preserve confirmation for actual destructive actions such as session close and app quit.

- [ ] On attach, run one full store-driven reconciliation, re-derive the active session's pending decision projection, and then restart ordinary view-only refresh timers. Never replay per-token deltas accumulated while detached. Receipt-aware acknowledgement is wired in Task 7 after Task 6 introduces the durable receipt field; Task 3 only establishes the successful-render reconciliation boundary it will use.

- [ ] Add failure-injection tests for unmount cleanup, stale outgoing detach, sync/render failure, and notification failure. Assert the runtime/store remain intact and the old screen can be garbage-collected.

- [ ] Run the focused tests:

  ```bash
  pytest -q Tests/Chat/test_console_runtime_lifetime.py Tests/Chat/test_console_viewless_hooks.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_console_store_continuity.py Tests/UI/test_console_sync_outlives_screen.py Tests/UI/test_screen_navigation.py
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_turn_context.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/retrieval.py Tests/Chat/test_console_runtime_lifetime.py Tests/Chat/test_console_viewless_hooks.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_console_store_continuity.py Tests/UI/test_console_sync_outlives_screen.py Tests/UI/test_screen_navigation.py
  git diff --cached --name-only
  git commit -m "fix: detach Console views without cancelling turns"
  ```

## Task 4: Harden frozen-input, privacy, and viewless-owner coverage

**Files:**

- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_turn_context.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/retrieval.py`
- Modify: `Tests/Chat/test_console_viewless_hooks.py`
- Modify: `Tests/Chat/test_console_turn_execution_context.py`
- Modify: `Tests/UI/test_console_sync_outlives_screen.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`

### Steps

- [ ] Audit the Task 3 hook inventory with failing owner tests that classify every former/current hook as one of:

  1. frozen into `ConsoleTurnCustodyRequest` before acceptance;
  2. app-owned domain service safe while viewless; or
  3. disposable UI projection callback cleared at detach.

  Assert that no task reachable from runtime custody retains a `ChatScreen`, widget, DOM query, screen timer, cost timer, or `run_worker` callback.

- [ ] Close any residual gaps in the frozen screen-derived inputs established in Task 3:

  - provider/model/session settings and provider payload controls;
  - scratch snapshot and selected workspace roots;
  - selected dictionary and world-info application results;
  - staged RAG launch plus source/top-k defaults;
  - Library/direct-tool policy and tool configuration;
  - global display identity and prompt-history values;
  - accepted attachment identities/objects and their session ownership.

  Prefer extending `ConsoleTurnConfigurationSnapshot` only when the value is configuration. Keep large/transient accepted objects in the private custody record with `repr=False`.

- [ ] Verify `_console_chat_dictionary_applier`, `_console_world_info_applier`, frozen RAG capture/release, and Library policy resolution now execute through narrow app-owned helpers wherever post-handoff asynchronous work is required. Close any remaining screen reach-back; helpers may use app services and captured IDs/configuration, but never `ChatScreen` or widgets.

- [ ] Assert `CONSOLE_VIEW_HOOK_SLOTS` contains only repaint/focus/card hooks and all are explicitly detached. Remove any residual old “viewless ordinary hooks are inert/fail closed” branch for domain work, because an accepted turn already holds everything it needs.

- [ ] Add tests that mutate UI settings, staged RAG, workspace selection, identity, and attachments after handoff and verify the running turn uses the original frozen values. Also verify a later turn uses the new values.

- [ ] Add provider-payload regression assertions: runtime bookkeeping and attention IDs never appear in the outbound provider request, and accepted attachments follow the exact pre-feature vision/non-vision filtering policy.

- [ ] Run:

  ```bash
  pytest -q Tests/Chat/test_console_viewless_hooks.py Tests/Chat/test_console_turn_execution_context.py Tests/UI/test_console_sync_outlives_screen.py Tests/UI/test_console_runtime_ownership.py
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_turn_context.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/retrieval.py Tests/Chat/test_console_viewless_hooks.py Tests/Chat/test_console_turn_execution_context.py Tests/UI/test_console_sync_outlives_screen.py Tests/UI/test_console_runtime_ownership.py
  git diff --cached --name-only
  git commit -m "refactor: freeze Console turn inputs at handoff"
  ```

## Task 5: Retain every human-decision round while Console is detached

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Console_Modules/skill.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`
- Modify: `Tests/UI/test_console_headless_approval.py`
- Modify: `Tests/UI/test_console_mcp_approval.py`
- Modify: `Tests/UI/test_console_skill_install_confirm.py`
- Modify: `Tests/UI/test_skill_install_concurrent_confirms.py`
- Modify: `Tests/Chat/test_console_skill_script_confirm.py`
- Modify: `Tests/Chat/test_skill_script_concurrent_confirms.py`

### Steps

- [ ] Add failing parameterized tests for MCP/tool approval, skill-install confirmation, and skill-script confirmation. For each decision type prove:

  - the stable round/request ID remains registered after detach;
  - no default allow/deny result is produced by navigation;
  - one sanitized app notice is emitted per stable ID while hidden;
  - a fresh Console view re-derives and mounts the correct active-session head card;
  - two rounds of the same type and mixed decision types retain FIFO order without overwriting each other;
  - explicit response, Stop, session close, and app disposal each resolve only their documented scope.

- [ ] Replace visit-cancel-based decision waiting with a small active-time clock stored on the existing round record:

  ```python
  remaining_active_seconds: float | None
  active_since: float | None
  ```

  Under the existing round lock, time begins only after the active session's head card reports a successful mount. Detach, session switch, card unmount/replacement, head resolution, and card-render failure subtract elapsed monotonic time and pause before changing projection state. A configured timeout of zero remains unbounded. Do not retain a wall-clock deadline that advances while no answerable control exists.

- [ ] Use an injected monotonic clock and explicit events/barriers in tests; do not use timing sleeps. Verify detach just before expiry preserves the remaining duration; switching sessions pauses the outgoing round and resumes only a successfully mounted incoming head; resolving one FIFO head does not start the next until its mount succeeds; and a mount/render failure consumes zero decision-active time.

- [ ] Reuse the controller's existing MCP, skill-install, and skill-script registries and FIFO maps. Add one derived `pending_decision_projection(session_id)` that selects the ordered answerable card; do not build a parallel decision state machine in the runtime.

- [ ] Have runtime attachment changes, Console session switches, head-card transitions, and `ChatTaskCards.sync_state`/unmount report through one narrow controller method such as `set_answerable_decision(session_id, decision_id | None)`. Pass a non-`None` decision ID only after the exact MCP/install/script card is successfully populated and mounted; clear it before unmount/replacement and in render-failure cleanup. Exactly that mounted head accrues time; detached, background-session, non-head, and failed-render rounds remain paused. The runtime may track “notice already sent” by stable decision ID, but must clear that memory when the round resolves.

- [ ] Ensure a round that is not answerable—because Console is detached or another session/card is active—produces the same one-time app-wide notice with only safe type/session labels and a return-to-Console instruction. Never include arguments, scripts, paths, exception bodies, credentials, or raw approval payloads.

- [ ] Run:

  ```bash
  pytest -q Tests/UI/test_console_headless_approval.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_console_skill_install_confirm.py Tests/UI/test_skill_install_concurrent_confirms.py Tests/Chat/test_console_skill_script_confirm.py Tests/Chat/test_skill_script_concurrent_confirms.py
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Console_Modules/skill.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py Tests/UI/test_console_headless_approval.py Tests/UI/test_console_mcp_approval.py Tests/UI/test_console_skill_install_confirm.py Tests/UI/test_skill_install_concurrent_confirms.py Tests/Chat/test_console_skill_script_confirm.py Tests/Chat/test_skill_script_concurrent_confirms.py
  git diff --cached --name-only
  git commit -m "feat: keep Console decisions alive while detached"
  ```

## Task 6: Persist exact terminal receipts in the transcript transaction

**Files:**

- Modify: `tldw_chatbook/Chat/message_metadata.py`
- Modify: `tldw_chatbook/Chat/conversation_local_marks_service.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- Modify: `tldw_chatbook/Chat/console_dispatch_repository.py`
- Modify: `tldw_chatbook/Chat/console_conversation_hydration.py`
- Modify: `tldw_chatbook/Video_Generation/video_metadata.py`
- Modify: `Tests/Chat/test_conversation_local_marks_service.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`
- Modify: `Tests/Chat/test_console_durable_turn_acceptance.py`
- Modify: `Tests/Chat/test_console_dispatch_recovery.py`
- Modify: `Tests/Chat/test_console_generate_image.py`
- Modify: `Tests/Chat/test_console_generate_video.py`
- Modify: `Tests/Chat/test_console_video_message.py`
- Add: `Tests/Chat/test_console_terminal_attention.py`

### Steps

- [ ] Add failing service tests for a bounded `console_unseen:<opaque-receipt-id>` mark namespace:

  - valid construction and exact parsing;
  - rejection of empty, malformed, oversized, or non-opaque IDs;
  - uncached prefix/list/boolean queries for startup shell attention, so receipt writes made inside another repository transaction cannot be hidden by an exact-mark cache;
  - exact acknowledgement deletes only one receipt;
  - existing `starred` and `fleet_unseen` behavior remains unchanged.

- [ ] Extend `MessageMetadata` with a bounded local-only `terminal_receipt_id: str = ""`. Update strict parsing/degrading load tests and ensure it remains only in local `metadata_json`, never provider message content.

- [ ] Expose one marks-service SQL primitive that accepts an existing ChaChaNotes cursor, while keeping the public transactional wrapper:

  ```python
  def set_mark_with_cursor(
      self,
      cursor: sqlite3.Cursor,
      conversation_id: str,
      mark_type: str,
      *,
      created_at: str,
      updated_at: str,
  ) -> None:
      ...
  ```

  Validation and SQL must live in one place. Do not open a second transaction from inside terminal persistence.

- [ ] Add optional terminal-receipt parameters to `ChatPersistenceService.create_message` and `update_message_content`. When present, force the existing `transaction(immediate=True)` path, persist the terminal row/metadata, and insert the namespaced mark through the same cursor before commit.

- [ ] Cover the authoritative durable-dispatch settlement path separately. Extend `ConsoleAssistantSettlement` with the validated terminal receipt mark/ID needed by `ConsoleDispatchRepository.settle_with_assistant()`. In the repository's existing `transaction(immediate=True)`, update the assistant row (including receipt-bearing metadata), insert the exact `console_unseen:*` mark through the shared cursor helper, and delete/settle the dispatch checkpoint before the same commit. Complete/failed store settlements supply a receipt; stopped/discarded settlements supply none. Do not let `_settle_owned_dispatch_terminal()` short-circuit around receipt creation.

- [ ] Centralize receipt minting in `ConsoleChatStore.mark_message_complete` and `mark_message_failed`. Reuse the same receipt on idempotent terminal retry of the same row. Explicit Stop, session close, and app-shutdown cancellation must call stopped/cancelled paths without minting a receipt.

- [ ] Cover immediate terminal media commits that bypass those markers:

  - when `append_generation_message(..., persist=True)` creates an already-complete image row, mint the receipt first, attach it through `MessageMetadata`, and pass it into the same atomic `create_message(attachments=..., generation_metadata=..., terminal_mark=...)` transaction;
  - when `append_video_message(..., persist=True)` creates an already-complete video row, preserve the full `VideoGenerationMetadata` payload and its local-only receipt together. Use the smallest backward-compatible representation: add a bounded optional `terminal_receipt_id` to `VideoGenerationMetadata` serialization/hydration and expose one common `ConsoleChatMessage` receipt accessor that reads ordinary `MessageMetadata` or video metadata. Do not overwrite video metadata with a second `metadata_json` document or leak the receipt into remote video-provider payloads.

  Image/video rows created without persistence do not publish durable unseen attention until their durable create succeeds.

- [ ] Add rollback tests for both persistence routes. Inject failure after the generic row update, after the durable-dispatch assistant update, after mark insertion, and before checkpoint deletion; assert terminal state, receipt metadata, mark, and checkpoint are all old or all new. Add race tests showing two results in one conversation receive distinct marks and acknowledging the older receipt leaves the newer one intact.

- [ ] Add restart-style tests using a temporary on-disk ChaChaNotes DB for ordinary, immediate image, and immediate video terminal rows: reopen services, hydrate the media metadata and receipt without losing either, discover the unseen receipt by namespace, render/acknowledge its matching row, reopen again, and confirm only that receipt is gone.

- [ ] Verify cancellation reasons: user Stop/session close/app shutdown create no receipt; unexpected cancellation without an explicit reason terminalizes as failure and does create one.

- [ ] Run:

  ```bash
  pytest -q Tests/Chat/test_conversation_local_marks_service.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_generate_image.py Tests/Chat/test_console_generate_video.py Tests/Chat/test_console_video_message.py Tests/Chat/test_console_terminal_attention.py
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/message_metadata.py tldw_chatbook/Chat/conversation_local_marks_service.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_dispatch_checkpoint.py tldw_chatbook/Chat/console_dispatch_repository.py tldw_chatbook/Chat/console_conversation_hydration.py tldw_chatbook/Video_Generation/video_metadata.py Tests/Chat/test_conversation_local_marks_service.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_generate_image.py Tests/Chat/test_console_generate_video.py Tests/Chat/test_console_video_message.py Tests/Chat/test_console_terminal_attention.py
  git diff --cached --name-only
  git commit -m "feat: persist exact Console terminal receipts"
  ```

## Task 7: Project hidden outcomes into app notices and shell attention

**Files:**

- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/UI/Navigation/nav_overflow_menu.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_master_shell_navigation.py`
- Modify: `Tests/UI/test_console_narrow_layout.py`
- Add: `Tests/UI/test_console_turn_attention.py`

### Steps

- [ ] Write failing projection tests that derive one boolean Console-attention state from:

  - any durable `console_unseen:*` mark; or
  - any unresolved hidden human decision.

  Assert that opening Console alone does not clear terminal attention, resolving a decision clears only decision attention, and rendering/acknowledging the matching terminal row clears only that receipt.

- [ ] Add one runtime-owned attention recompute method. It may query local mark IDs and controller pending-decision summaries, then notify the current shell projection. It must expose only a boolean and sanitized text—never counts, raw IDs, prompts, tool arguments, paths, or exception bodies.

- [ ] Emit one app-wide toast for each completion/failure receipt whose matching row has not been rendered—whether Console is detached or viewing another session—and for each non-answerable decision ID. Completion uses informational severity and terminal failure uses error severity. Record “notified” only after the app notification call succeeds; a notification-render failure must leave durable attention intact and must not affect turn finalization.

- [ ] Wire exact receipt acknowledgement into the successful-render boundary established in Task 3. Transcript synchronization returns or publishes only the receipt IDs whose matching ordinary/image/video rows mounted successfully; acknowledge that exact set after render completion. A mount, media-card hydration, or render failure acknowledges nothing, and opening Console alone never clears a mark.

- [ ] Add the same non-color-only glyph to Console in both navigation surfaces. Preserve the existing hotkey label and ghost-button geometry, and set a tooltip such as “Console needs attention.” Use the production destination metadata/hierarchy; do not fork a test-only label path.

- [ ] On app/runtime initialization and fresh navigation-bar mount, recompute from durable marks so attention survives restart or an unavailable prior bar. On mark/decision changes, update the currently mounted main and overflow projections if present; no per-token shell updates.

- [ ] Add Textual tests at `80x24`, `100x30`, and `160x48` for:

  - visible main-navigation glyph when Console fits;
  - overflow-menu glyph when Console is in overflow;
  - accessible text/tooltip not dependent on color;
  - no clipped hotkey or shifted ghost geometry;
  - glyph clears only after all receipt and decision sources clear.

  Use the consolidated production CSS and enable app notifications when asserting toasts.

- [ ] Run:

  ```bash
  pytest -q Tests/UI/test_console_turn_attention.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_console_narrow_layout.py
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/UI/Navigation/main_navigation.py tldw_chatbook/UI/Navigation/nav_overflow_menu.py tldw_chatbook/app.py Tests/UI/test_console_turn_attention.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_console_narrow_layout.py
  git diff --cached --name-only
  git commit -m "feat: surface hidden Console outcomes in navigation"
  ```

## Task 8: Fence Stop, session close, and app quit to their exact scopes

**Files:**

- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_fleet_wake.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Chat/test_console_runtime_lifetime.py`
- Modify: `Tests/Chat/test_console_agent_swap.py`
- Modify: `Tests/Chat/test_console_agent_bridge_cancel_all.py`
- Modify: `Tests/Chat/test_console_automatic_library_preparation.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_close_session_fleet.py`
- Modify: `Tests/Chat/test_console_fleet_wake_safety.py`
- Modify: `Tests/Chat/test_console_dispatch_recovery_fix_round1.py`
- Modify: `Tests/Chat/test_console_dispatch_recovery_fix_round4.py`
- Modify: `Tests/Chat/test_console_durable_turn_fix_round1.py`
- Modify: `Tests/Chat/test_console_local_citation_boundary.py`
- Modify: `Tests/Chat/test_console_prompt_queue_coordinator.py`
- Modify: `Tests/Chat/test_console_run_markers.py`
- Modify: `Tests/Chat/test_console_run_state_per_session.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/UI/test_app_quit_guard.py`
- Modify: `Tests/UI/test_console_prompt_queue.py`
- Modify: `Tests/UI/test_console_send_draft_snapshot.py`
- Add: `Tests/Chat/test_console_runtime_shutdown.py`

### Steps

- [ ] Add failing deterministic race tests for the cancellation matrix:

  | Action | Required result |
  | --- | --- |
  | Navigate | detach only; no stop signal, task cancel, queue tombstone, decision resolution, receipt, or outcome toast |
  | Stop | cancel selected turn/chain and only its decision rounds; no receipt/toast |
  | Session close | fence new/queued admissions, revision-confirm impact, cancel and await only that session's turns/rounds/fleet, then delete session; no receipt/toast |
  | Confirmed app quit | revision-pin whole-runtime impact, fence all new admissions, cancel and bounded-drain all custody, then dispose; no receipt/toast |

- [ ] Replace the synchronous destructive `ConsoleChatController.close_session()` boundary with a two-phase controller contract used by an async `ConsoleRuntime.close_session(...)` seam:

  1. `begin_session_close(session_id, expected_revision)` atomically sets the per-session admission fence, rechecks the revision-pinned impact, tombstones queued claims, revokes decisions, and signals/cancels the session's turn/fleet work without deleting the store session;
  2. the runtime awaits that session's custody tasks **and delegated fleet** up to the named cooperative timeout, retrieves task exceptions, and advances the session/conversation generation fence on timeout;
  3. `finalize_session_close(ticket)` verifies the close ticket/generation, releases preparation/recovery/attachment/fleet-wake owners, and only then deletes the session and activates its neighbor.

  `_close_console_session_tab()` must await this runtime seam after its confirmation loop. Completion-vs-close tests must prove completion committed before the close gate is preserved, while cancellation recorded first rejects late terminal output. Reuse the existing queue, preparation, stream, decision, and sub-agent cleanup internals; do not duplicate them in the runtime.

- [ ] Add one bounded async fleet-drain seam to `ConsoleAgentBridge`, built on its existing live fleet/coordinator callbacks rather than polling sleeps. Register a per-conversation waiter before the post-registration live snapshot to close the child-finishes-during-registration race; resolve it when `fleet_snapshot(conversation_id)` has no live handle. `cancel_all_subagents()` remains the signal path, and `await_fleet_terminal(...)` becomes the drain path.

- [ ] Extend the revisioned `ConsoleLifecycleImpact` with a content-free live delegated-child count. `lifecycle_impact(session_id=...)` counts `ConsoleAgentBridge.fleet_snapshot(conversation_id)` even when the parent turn/custody task is already terminal; the whole-runtime impact sums survivors across sessions. Register one bridge fleet-activity listener that advances the global and owning-session lifecycle revisions on child spawn and terminal settlement, including retained survivors. Session-close/quit copy names delegated children separately. Tests must prove a lone surviving child triggers confirmation and that a child spawn/settle between dialog snapshot and fencing refreshes the revision-pinned confirmation instead of silently widening or discarding consent.

- [ ] On session close, cancel the exact conversation fleet and await its terminal waiter before deleting the session. On app disposal, snapshot all live conversation fleets, fence late delivery, cancel them, and drain them concurrently within the single global shutdown deadline—not one full timeout per conversation. A delegated child may remain live after the parent custody task exits, so parent-task completion is never accepted as fleet-drain evidence.

- [ ] Fence late child settlement and auto-wake with the session/conversation generation carried by the close ticket. After close/timeout or app-disposal fencing, a stale child's wake, transcript callback, receipt mark, notification, and approval resolution are discarded before reaching `ConsoleChatStore` or shell attention. Add a deterministic test where the parent custody task has already completed, its child ignores cancellation through the grace window, and the child settles later; close/quit remains bounded and no deleted-session row, receipt, or wake appears.

- [ ] Migrate every direct `controller.close_session(...)` caller found by `rg -n "controller\.close_session|_ensure_console_chat_controller\(\)\.close_session" Tests tldw_chatbook` to the async runtime seam or, for pure controller unit tests, the explicit begin/finalize ticket sequence. End with no public synchronous path that can delete a live session before its tasks drain.

- [ ] Move Console quit confirmation to `TldwCli`, because quit may begin while Library/Settings—not `ChatScreen`—is mounted. Preserve other screens' own unsaved-work confirmation, then revision-pin `controller.lifecycle_impact()` in an app-level Console confirmation loop.

- [ ] Extend `Tests/UI/test_app_quit_guard.py`, the existing production-shaped `TldwCli.action_quit` suite, for quit from Console and from a non-Console screen, revision changes between confirmation and fence, duplicate-dialog prevention, admission rejection after confirmation, bounded drain, and no shutdown receipt/toast.

- [ ] After confirmation succeeds, fence runtime admission before setting the final shutdown state. Remove or neutralize duplicate ChatScreen Console-quit confirmation so a quit from Console does not show two dialogs.

- [ ] Add bounded runtime disposal:

  1. fence admission and bump a terminal generation;
  2. signal controller shutdown and cancel custody tasks;
  3. await the task set up to one named timeout;
  4. retrieve all completed exceptions;
  5. bounded-drain delegated fleets under the same deadline and abandon uncooperative wrappers without permitting later transcript/wake/receipt mutation;
  6. close provider gateway and release references.

  Do not use an unbounded `gather`. A thread may finish after the timeout, but its generation/atomic terminal gate must reject late writes.

- [ ] Reuse or add one atomic terminalization compare-and-set at the controller/store boundary so completion and cancellation cannot both publish terminal state. Test completion-wins, cancellation-wins, and provider-ignores-cancellation interleavings with barriers rather than sleeps.

- [ ] Run:

  ```bash
  pytest -q Tests/Chat/test_console_runtime_lifetime.py Tests/Chat/test_console_runtime_shutdown.py Tests/Chat/test_console_agent_swap.py Tests/Chat/test_console_agent_bridge_cancel_all.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_close_session_fleet.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/Chat/test_console_dispatch_recovery_fix_round4.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_local_citation_boundary.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_run_markers.py Tests/Chat/test_console_run_state_per_session.py Tests/UI/test_screen_navigation.py Tests/UI/test_app_quit_guard.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_send_draft_snapshot.py
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_fleet_wake.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/app.py Tests/Chat/test_console_runtime_lifetime.py Tests/Chat/test_console_runtime_shutdown.py Tests/Chat/test_console_agent_swap.py Tests/Chat/test_console_agent_bridge_cancel_all.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_close_session_fleet.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/Chat/test_console_dispatch_recovery_fix_round4.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_local_citation_boundary.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_run_markers.py Tests/Chat/test_console_run_state_per_session.py Tests/UI/test_screen_navigation.py Tests/UI/test_app_quit_guard.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_send_draft_snapshot.py
  git diff --cached --name-only
  git commit -m "fix: fence Console cancellation and shutdown scopes"
  ```

## Task 9: Prove real navigation, multi-session isolation, privacy, and dead-view safety

**Files:**

- Add: `Tests/UI/test_console_turn_navigation_continuity.py`
- Modify: `Tests/UI/test_console_store_continuity.py`
- Modify: `Tests/UI/test_console_sync_outlives_screen.py`
- Modify: `Tests/UI/test_console_headless_approval.py`
- Modify: `Tests/UI/test_probe_headless_approval_behaviour.py`
- Modify: `Tests/test_probe_import_provenance.py`
- Modify: `Tests/Architecture/test_screen_size_ratchet.py` only if its measured ceiling can be lowered; never raise it

### Steps

- [ ] Build one production-shaped mounted harness that starts `TldwCli` with isolated temp data/config paths, uses the real navigation action to leave and return, and controls the provider via async barriers. Do not call screen internals as a substitute for navigation.

- [ ] Cover ordinary chat and main-agent turns in the two outer race windows:

  - navigation immediately after runtime custody but before the task starts;
  - navigation after streaming begins but before terminalization.

  Assert separately that the live store advances while detached, no detached DOM callback fires, the fresh screen renders the correct final row, and no duplicate/missing chunks appear.

- [ ] Add deterministic navigation barriers for every required interior phase: asynchronous preparation, provider-readiness resolution, streaming, queued-continuation claim/drain, turn-launched tool execution, image/video generation work, delegated-agent work, human approval wait, and the terminal transaction between row update and commit. Parameterize the common navigation harness, but use origin-specific assertions so a tool/media/delegated path cannot pass merely because an ordinary stream survived.

- [ ] For each returning-view case, assert all four approved evidence layers independently:

  1. the app-owned live store contains the expected identified rows/state;
  2. the fresh mounted transcript renders uniquely identifying content once;
  3. ChaChaNotes contains the expected durable rows/receipt state;
  4. a subsequent send's captured provider payload continues the same conversation lineage without duplicating the prior turn.

  A database append or store assertion alone is not sufficient UI evidence.

- [ ] Add a multi-session case: two sessions run concurrently, one reaches approval and the other completes; returning to either session mounts only its own card/result and acknowledging one does not clear the other's attention.

- [ ] Add weak-reference/dead-view tests: after navigation and garbage collection the outgoing `ChatScreen` is collectible while custody continues, and stale attachment-generation callbacks cannot mutate or detach the incoming screen.

- [ ] Instrument the production-shaped harness with counters around DOM queries, transcript-sync polling, and reason-tagged main/overflow navigation updates. Reset after detach and release several stream/tool/media deltas. Assert exactly zero detached DOM queries and transcript polls, and zero **per-token or non-attention** shell updates. A hidden decision/receipt transition may cause one bounded/coalesced attention-projection update while detached, as Task 7 requires; assert stream chunks alone never drive it. On fresh Console attach, assert one bounded transcript reconciliation rather than replayed per-token updates.

- [ ] Add privacy owner tests that search runtime custody records, `repr`, captured Loguru records, notifications, navigation labels, sync payloads, exports, and provider payloads. Permit only opaque turn/session/receipt IDs and sanitized status text. Assert raw prompts, attachment paths/names/bytes, RAG context, tool arguments/results, credentials, and exception bodies are absent.

- [ ] Run the provenance gate first so a passing subprocess never accidentally imports an installed package or stale checkout:

  ```bash
  pytest -q Tests/test_probe_import_provenance.py
  ```

  Expected: PASS and reported module paths resolve inside this checkout.

- [ ] Run the integrated targeted group:

  ```bash
  pytest -q Tests/UI/test_console_turn_navigation_continuity.py Tests/UI/test_console_store_continuity.py Tests/UI/test_console_sync_outlives_screen.py Tests/UI/test_console_headless_approval.py Tests/UI/test_probe_headless_approval_behaviour.py Tests/Architecture/test_screen_size_ratchet.py
  ```

  Expected: PASS.

- [ ] Perform fault-injection validation by temporarily reversing one named invariant at a time in the test seam—not by committing mutations—and show the focused test fails for:

  - screen-owned launch restored;
  - hidden decision clock allowed to advance;
  - exact-receipt acknowledgement replaced with conversation-wide clear;
  - terminal row and receipt written in separate transactions.

  Restore the implementation after each check and record the commands/results in the task Implementation Notes.

- [ ] Commit:

  ```bash
  git add Tests/UI/test_console_turn_navigation_continuity.py Tests/UI/test_console_store_continuity.py Tests/UI/test_console_sync_outlives_screen.py Tests/UI/test_console_headless_approval.py Tests/UI/test_probe_headless_approval_behaviour.py Tests/test_probe_import_provenance.py Tests/Architecture/test_screen_size_ratchet.py
  git diff --cached --name-only
  git commit -m "test: verify Console turns survive navigation"
  ```

## Task 10: Update documentation and run isolated live verification

**Files:**

- Modify: `Docs/User_Guide/console/chat-basics.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/superpowers/specs/2026-08-27-console-turns-survive-navigation-design.md`
- Modify: `backlog/tasks/task-22514 - Console-turns-survive-screen-navigation.md`
- Modify only if a real reusable incident occurred: `backlog/docs/lessons-live-verification.md` or `backlog/docs/lessons-testing-evidence.md`

### Steps

- [ ] Update the User Guide to state:

  - all accepted Console turns continue when switching app screens;
  - unsent composer text, staged-but-unaccepted inputs, screen modals, microphone/audio, and view timers remain screen-scoped;
  - hidden approvals notify once, wait, and consume decision time only while their card is answerable;
  - hidden completion/failure creates a Console glyph until the matching result is rendered;
  - Stop affects the selected turn/chain, session close affects that session, and confirmed app quit ends the whole runtime;
  - forced process exit/restart does not continue a live task.

  Remove the obsolete section around `agent-runs-and-tools.md` lines 1247–1255 that says leaving Console cancels every in-flight turn and denies approvals. Preserve unrelated existing edits in both guide files by staging only this task's hunks.

- [ ] Mark the design status as implemented only after all targeted evidence is green. Add implementation notes and exact test/live evidence to TASK-22514; check every acceptance criterion only when its evidence exists.

- [ ] After every task-specific command above has passed for **all files modified by that task**, rerun this consolidated critical-path targeted regression set. This command complements rather than replaces the per-task suites (especially Task 2's direct-caller migration set and Task 8's complete close/quit caller set). Do not run the full repository suite unless the user opts in:

  ```bash
  pytest -q \
    Tests/Chat/test_console_runtime_lifetime.py \
    Tests/Chat/test_console_runtime_shutdown.py \
    Tests/Chat/test_console_turn_execution_context.py \
    Tests/Chat/test_console_turn_preparation.py \
    Tests/Chat/test_console_viewless_hooks.py \
    Tests/Chat/test_console_terminal_attention.py \
    Tests/Chat/test_conversation_local_marks_service.py \
    Tests/Chat/test_chat_persistence_service.py \
    Tests/Chat/test_console_durable_turn_acceptance.py \
    Tests/Chat/test_console_dispatch_recovery.py \
    Tests/Chat/test_console_generate_image.py \
    Tests/Chat/test_console_generate_video.py \
    Tests/Chat/test_console_video_message.py \
    Tests/Chat/test_console_prompt_queue.py \
    Tests/Chat/test_console_prompt_queue_coordinator.py \
    Tests/Chat/test_console_agent_bridge_cancel_all.py \
    Tests/Chat/test_console_close_session_fleet.py \
    Tests/Chat/test_console_fleet_wake_safety.py \
    Tests/Chat/test_console_skill_script_confirm.py \
    Tests/Chat/test_skill_script_concurrent_confirms.py \
    Tests/UI/test_console_prompt_queue.py \
    Tests/UI/test_console_send_draft_snapshot.py \
    Tests/UI/test_console_turn_navigation_continuity.py \
    Tests/UI/test_console_runtime_ownership.py \
    Tests/UI/test_console_store_continuity.py \
    Tests/UI/test_console_sync_outlives_screen.py \
    Tests/UI/test_console_headless_approval.py \
    Tests/UI/test_console_mcp_approval.py \
    Tests/UI/test_console_skill_install_confirm.py \
    Tests/UI/test_skill_install_concurrent_confirms.py \
    Tests/UI/test_console_turn_attention.py \
    Tests/UI/test_master_shell_navigation.py \
    Tests/UI/test_console_narrow_layout.py \
    Tests/UI/test_screen_navigation.py \
    Tests/UI/test_app_quit_guard.py \
    Tests/Architecture/test_screen_size_ratchet.py \
    Tests/test_probe_import_provenance.py
  ```

  Expected: PASS. If the user explicitly opts into a full sweep, run `pytest` afterward and record it separately.

- [ ] Derive and inspect the exact task-owned Python/test manifest, then run both lint and formatter checks over every path in it:

  ```bash
  git diff --name-only <recorded-task-base>..HEAD -- '*.py'
  git diff --name-only <recorded-task-base>..HEAD -- '*.py' | xargs python -m ruff check
  git diff --name-only <recorded-task-base>..HEAD -- '*.py' | xargs python -m ruff format --check
  ```

  Expected: the inspected manifest contains only this feature's source and test files, and both commands PASS. Replace the placeholder with the recorded commit literally; do not run the placeholder or include unrelated dirty files. If `ruff` is not installed in the active environment, record that and run the repository's configured equivalent rather than installing a new dependency without approval.

- [ ] Perform the isolated live-provider journey from `backlog/docs/lessons-live-verification.md`:

  1. create temporary profile/data/config directories and start the app from this checkout;
  2. verify import provenance before trusting the run;
  3. prefer an available local provider and send one uniquely identifiable streamed prompt; if only an external or billed provider is available, pause and obtain explicit user approval for that call before using it;
  4. navigate to Library or Settings using the real shell while tokens are still arriving;
  5. observe the sanitized completion notice and Console attention glyph;
  6. return to Console and verify one complete, non-duplicated transcript result;
  7. repeat with one approval-bearing tool flow, confirm the hidden wait does not expire, return, decide, and observe completion;
  8. quit with one slow turn active and verify the revision-pinned confirmation and bounded shutdown.

  Record timestamps, provider/model, navigation path, screenshots for visible UI only, and transcript/store/receipt assertions. Do not put secrets or raw sensitive context into the evidence. If no local provider is usable and the user does not approve external/billed use, record the missing live evidence, leave the applicable AC unchecked, and do not mark TASK-22514 Done.

- [ ] Self-review the exact diff against all eight acceptance criteria, ADR-094, privacy boundaries, and the cancellation matrix. Do not mark Done if any targeted test, lint check, documentation item, or live journey is missing.

- [ ] Finish Backlog hygiene:

  ```bash
  backlog task edit 22514 --notes "Implemented app-owned Console turn custody, navigation-safe decisions, exact terminal attention receipts, shell attention, and scoped shutdown. Targeted and live evidence: <fill in exact results>. ADR: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md."
  backlog task edit 22514 -s Done
  ```

- [ ] Stage only the documentation/task hunks, inspect them, and commit:

  ```bash
  git add -p Docs/User_Guide/console/chat-basics.md Docs/User_Guide/console/agent-runs-and-tools.md
  git add Docs/superpowers/specs/2026-08-27-console-turns-survive-navigation-design.md 'backlog/tasks/task-22514 - Console-turns-survive-screen-navigation.md'
  git diff --cached --name-only
  git diff --cached --check
  git commit -m "docs: document navigation-safe Console turns"
  ```

## Final review checkpoint

- [ ] Invoke `superpowers:requesting-code-review` after all implementation tasks are complete and before merge/PR work.
- [ ] Address correctness, privacy, cancellation, and test-evidence blockers through `superpowers:receiving-code-review`.
- [ ] Invoke `superpowers:verification-before-completion`; rerun the relevant commands and cite their fresh outputs before claiming completion.
- [ ] Use `superpowers:finishing-a-development-branch` to present merge/PR/cleanup choices only after the task is genuinely Done.
