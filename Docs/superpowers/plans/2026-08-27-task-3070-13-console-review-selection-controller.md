# TASK-3070.13 Console Review and Selection Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the seven reviewed Console review/selection policy methods and the trajectory snapshot adapter out of `ChatScreen` into one explicit non-DOM `ConsoleReviewSelectionController`, while preserving the three Textual delegates, six screen-owned presentation/ADR-068 methods, persistence authority, privacy boundaries, and user-visible behavior.

**Architecture:** `tldw_chatbook/UI/Console_Modules/review_selection.py` will own review provider/root policy, annotation discovery state, selection feedback and note policy, trajectory snapshot loading, and trajectory launch sequencing. The existing `build_console_controllers()` seam will construct `screen._review_selection` from fine-grained late-bound callables; `ChatScreen` will retain three complete five-line-or-shorter Textual delegates and three fail-loud `_ControllerState` descriptors with no shadow storage. Persistence remains in existing stores/repositories/DB services, Textual modal/screen construction remains in wiring or `ChatScreen`, and the feedback worker returns an immutable result before controller state is updated on the event-loop thread.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Ruff, stdlib AST source inspection, Backlog.md CLI, existing persistent-diagnostic inventory tooling.

**Spec:** `Docs/superpowers/specs/2026-08-27-task-3070-13-console-review-selection-controller-design.md`

## Global Constraints

- Implementation base: `origin/dev` `ee8dc24115ba14cddef9de2b262e1cbf7bfa32ca`; if the executable drift gate finds a changed 7/3/6 family, stop and amend before continuing.
- Exact ownership: seven moves, three complete delegates of at most five physical lines each, six screen stays.
- Conservative screen reduction: at least 399 physical lines and seven direct methods; the trajectory adapter's additional removal is not counted toward this minimum.
- No `ChatScreen` reference, DOM query, sibling-controller object, dependency container, dynamic facade, mixin, Git authority, SQL/schema ownership, stale screen re-export, or compatibility shadow state.
- Preserve ADR-068 screen ownership of review-note modal/fetch/mutate/forced-reload flow.
- Keep `TrajectoryScreen` lazily imported at presentation time; importing the controller module must not widen Chat first paint.
- Never mutate `annotation_previews` from the blocking feedback-persistence thread.
- Never add selected text, note content/title, prompt content, raw DB rows, or provider failures to logs/diagnostics.
- Do not run a local full test suite. Use only the focused tests and gates named below; GitHub Actions remains the broad integration gate.

---

## Scope, baseline, and ADR check

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-3070-13-console-review-selection`
- Planning branch: `codex/task-3070-13-console-review-selection`
- Current screen: 17,599 physical lines / 539 direct methods.
- Frozen current-base family: 16 methods / 840 lines = seven moves / three delegates / six stays.
- Conservative final screen: no more than 17,200 physical lines / 532 direct methods.
- Focused baseline: 111/115 pass. The four known failures are stale unit traversals in `Tests/UI/test_console_annotation_markers.py`; mounted annotation behavior is green and production must not be changed to satisfy those assertions.

ADR required: no

ADR path: `backlog/decisions/068-console-text-selection-and-annotations.md`

Reason: this task applies the existing controller ownership rule in `DESIGN.md` section 7 and preserves ADR-068's explicit screen-owned review-note boundary. It changes no storage/schema, Git/database authority, service contract, security/privacy policy, dependency, or long-lived UX structure.

## Task 0: Record the implementation plan before production changes

**Files:**

- Add: `Docs/superpowers/plans/2026-08-27-task-3070-13-console-review-selection-controller.md`
- Modify: `backlog/tasks/task-3070.13 - Extract-Console-review-and-selection-workflow-ownership.md`

**Interfaces:**

- Consumes: the approved/hardened design spec and the task's five acceptance criteria.
- Produces: a Backlog `## Implementation Plan` section that links this exact execution order and ADR decision.

- [ ] **Step 1: Attach the plan to the In Progress Backlog task**

  Run before changing production code:

  ```bash
  backlog task edit 3070.13 --plan "1. Freeze the exact current-base 7/3/6 boundary with RED architecture tests\n2. Establish ConsoleReviewSelectionController through build_console_controllers and fail-loud descriptors\n3. Move and characterize change-review and annotation policy\n4. Move selection-feedback and selection-note policy with an explicit worker/event-loop boundary\n5. Move trajectory loading/launch policy, install the three complete delegates, and retarget focused callers\n6. Prove five high-risk branches with bounded manual mutations\n7. Run focused behavior, architecture, static, privacy, diagnostic, backlog-ID, and diff gates\n8. Rebase on latest dev, revalidate drift, refresh task evidence, and complete delivery\n\nADR required: no\nADR path: backlog/decisions/068-console-text-selection-and-annotations.md\nReason: applies DESIGN.md section 7 while preserving ADR-068 review-note ownership and all existing durable authorities."
  ```

- [ ] **Step 2: Verify and commit the planning artifacts**

  ```bash
  backlog task 3070.13 --plain
  git diff --check
  git add "backlog/tasks/task-3070.13 - Extract-Console-review-and-selection-workflow-ownership.md" Docs/superpowers/plans/2026-08-27-task-3070-13-console-review-selection-controller.md
  git commit -m "docs(console): plan review selection extraction"
  ```

## Task 1: Freeze the reviewed boundary with executable RED architecture tests

**Files:**

- Create: `Tests/Architecture/test_console_review_selection_controller_boundary.py`
- Reference: `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- Reference: `tldw_chatbook/UI/Screens/chat_screen.py`
- Reference: `tldw_chatbook/UI/Console_Modules/wiring.py`

**Interfaces:**

- Consumes: the exact seven move names, three delegate names, six stay names, three state names, and current-base spans from the design spec.
- Produces: `MOVE_METHODS`, `DELEGATE_METHODS`, `STAY_METHODS`, `TASK_BASE`, AST helpers, and a live `origin/dev` drift test used again after the final rebase.

- [ ] **Step 1: Add path-first AST tests without importing the missing module**

  Define the exact sets in the new architecture file:

  ```python
  MOVE_METHODS = frozenset({
      "_console_change_review_provider",
      "_console_change_review_workspace_roots",
      "_console_selection_feedback_flow",
      "_create_console_selection_note",
      "_load_console_annotation_previews",
      "_record_console_feedback_event",
      "_sync_console_annotation_discovery",
  })
  DELEGATE_METHODS = frozenset({
      "action_open_trajectory_view",
      "on_console_selection_feedback_requested",
      "on_console_selection_note_requested",
  })
  STAY_METHODS = frozenset({
      "_console_change_review_run_id",
      "_console_review_notes_flow",
      "_console_selection_quote_requested",
      "_dismiss_console_selection_menus_outside_transcript",
      "_open_change_review",
      "on_console_review_notes_requested",
  })
  ```

  Assert the final contract:

  - all seven move names are direct methods on `ConsoleReviewSelectionController` and absent from `ChatScreen`;
  - all six stays remain direct `ChatScreen` methods and are absent from the controller;
  - all three delegates retain their current Textual decorator/binding surfaces, call one `self._review_selection` entrypoint, contain no policy, and span at most five physical lines excluding decorators;
  - `_build_trajectory_snapshot` is a module-level function in `review_selection.py`, absent from `chat_screen.py`, and the review module has no module-scope import of `trajectory_screen`;
  - `ChatScreen` declares three `_ControllerState` descriptors pointing to `("_review_selection", "annotation_loaded_conversation")`, `("_review_selection", "annotation_previews")`, and `("_review_selection", "selection_feedback_inflight")` and has no instance assignment to the compatibility names;
  - descriptor access before wiring raises `RuntimeError`;
  - the controller has no `query`, `query_one`, `focus`, `push_screen`, `ChatScreen`, `__getattr__`, `__getattribute__`, mixin base, SQL literal/execute call, Git subprocess, or stored sibling-controller accessor/object;
  - `build_console_controllers()` assigns `screen._review_selection` exactly once and is the sole production constructor;
  - the constructor has only named keyword-only callables and no `screen`, `app_instance`, dependency object, `*args`, or `**kwargs` escape hatch;
  - `chat_screen.py` is at most 17,200 lines / 532 direct methods and the immutable ratchet is not raised.

- [ ] **Step 2: Add current-base arithmetic and durable rebase drift coverage**

  Read the frozen `TASK_BASE` and live `origin/dev` sources with bounded `git show`. For the frozen base, assert 17,599/539, the 16 exact candidate names, 840 total lines, seven/three/six, 280 move lines, 426 stay lines, and conservative 399/7 removal. For live `origin/dev`, accept only one of two complete states:

  ```python
  if MOVE_METHODS <= origin_screen_methods:
      assert exact_current_family(origin_screen) == reviewed_family
      assert projected_counts(origin_screen) <= (17_200, 532)
  else:
      assert final_owner_contract(origin_screen, origin_review_module)
  ```

  Any partial family, span change, classification change, missing lazy-import guarantee, or higher projection fails and requires a written amendment.

- [ ] **Step 3: Run the new test and observe RED**

  ```bash
  ../../.venv/bin/python -m pytest Tests/Architecture/test_console_review_selection_controller_boundary.py -q
  ```

  Expected: FAIL because `review_selection.py` and `ConsoleReviewSelectionController` do not exist and the seven methods/state still belong to `ChatScreen`.

- [ ] **Step 4: Commit the RED contract**

  ```bash
  git add Tests/Architecture/test_console_review_selection_controller_boundary.py
  git commit -m "test(console): freeze review selection boundary"
  ```

## Task 2: Establish the controller, state, and wiring seam

**Files:**

- Create: `tldw_chatbook/UI/Console_Modules/review_selection.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_controller_wiring.py`
- Modify: `Tests/Architecture/test_console_review_selection_controller_boundary.py`

**Interfaces:**

- Consumes: current screen/app/store services only through late-bound wiring lambdas.
- Produces: `ConsoleReviewSelectionController`, `screen._review_selection`, three owner state fields, and these exact public controller entrypoints for retained framework callers: `request_selection_feedback(self, action: str, quote: str, anchor_message_id: str | None) -> None`, `request_selection_note(self, quote: str) -> None`, and `open_trajectory_view(self) -> None`.

- [ ] **Step 1: Extend the wiring tests first**

  Import `ConsoleReviewSelectionController`, add `("_review_selection", ConsoleReviewSelectionController)` to `_ALL_CONTROLLER_SLOTS` immediately before `_send_price`, and rename the count-specific test to seventeen. Add `test_review_selection_controller_is_late_bound_without_sibling_objects` that replaces each permitted screen target after construction, calls the stored seam, and observes the replacement.

  The test must also assert initial owner state:

  ```python
  assert controller.annotation_loaded_conversation is None
  assert controller.annotation_previews == {}
  assert controller.selection_feedback_inflight is False
  ```

  Run and expect import/collection RED:

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_controller_wiring.py -q
  ```

- [ ] **Step 2: Add the smallest importable controller skeleton**

  Create a non-Textual module with a documenting 7/3/6 ownership header and this constructor contract:

  ```python
  class ConsoleReviewSelectionController:
      def __init__(
          self,
          *,
          store_accessor: Callable[[], Any],
          agent_conversation_id_accessor: Callable[[], str | None],
          change_review_provider_accessor: Callable[[str], Any | None],
          run_active_accessor: Callable[[], bool],
          run_active_for_root: Callable[[str], bool],
          workspace_roots_accessor: Callable[[], tuple[str, ...] | None],
          agent_runs_db_accessor: Callable[[], Any | None],
          native_messages_accessor: Callable[[], list[Any]],
          run_worker: Callable[..., Any],
          show_feedback_comment: Callable[[str, str], Awaitable[str | None]],
          dispatch_prompt: Callable[[str], Awaitable[Any]],
          marshal_to_ui: Callable[..., None],
          present_trajectory: Callable[..., None],
          notify: Callable[..., None],
      ) -> None:
          self.annotation_loaded_conversation: str | None = None
          self.annotation_previews: dict[str, tuple[str, ...]] = {}
          self.selection_feedback_inflight = False
  ```

  Store only these callables and state fields. Do not accept or cache a screen, app, Console chat controller, agent controller, prompt-queue controller, dependency dataclass, or generic callback mapping.

- [ ] **Step 3: Add presentation helpers and construct through existing wiring**

  In `wiring.py`, add two module-level presentation helpers:

  ```python
  async def _show_console_feedback_comment(screen: Any, action: str, quote: str) -> str | None:
      return await screen.app.push_screen_wait(
          ConsoleFeedbackCommentModal(action=action, quote=quote)
      )

  def _present_console_trajectory(screen: Any, launch: ConsoleTrajectoryLaunch) -> None:
      from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen
      screen.app.push_screen(
          TrajectoryScreen(
              launch.snapshot,
              screen_title=launch.screen_title,
              conversation_id=launch.conversation_id,
              revision_provider=launch.revision_provider,
              snapshot_builder=launch.snapshot_builder,
          )
      )
  ```

  `ConsoleTrajectoryLaunch` is a frozen, slotted dataclass defined in `review_selection.py` carrying `snapshot`, `screen_title`, `conversation_id`, `revision_provider`, and `snapshot_builder`. This is data, not a dependency container. The lazy import must stay inside `_present_console_trajectory`.

  Construct `screen._review_selection` immediately before `_send_price`. Wire each fine-grained callable with a late-bound lambda that resolves the exact current operation/fact at call time. No callable may return `_agent`, `_prompt_queue`, `_session`, or `_console_chat_controller`; only the store/service object, scalar/tuple fact, or operation result may cross.

- [ ] **Step 4: Move the three mutable fields and install fail-loud compatibility**

  Reuse the existing descriptor class:

  ```python
  _console_annotation_loaded_conversation = _ControllerState(
      "_review_selection", "annotation_loaded_conversation"
  )
  _console_annotation_previews = _ControllerState(
      "_review_selection", "annotation_previews"
  )
  _console_selection_feedback_inflight = _ControllerState(
      "_review_selection", "selection_feedback_inflight"
  )
  ```

  Remove the three `__init__` assignments. Preserve mutable-map identity: reads and writes both forward; no screen or descriptor shadow is permitted.

- [ ] **Step 5: Run seam tests**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_controller_wiring.py Tests/Architecture/test_console_review_selection_controller_boundary.py -q
  ```

  Expected: wiring/state tests GREEN; architecture remains RED only for not-yet-moved methods, adapter, delegates, and final count.

- [ ] **Step 6: Commit the seam**

  ```bash
  git add tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_controller_wiring.py Tests/Architecture/test_console_review_selection_controller_boundary.py
  git commit -m "refactor(console): establish review selection controller seam"
  ```

## Task 3: Move change-review and annotation-discovery policy

**Files:**

- Create: `Tests/UI/test_console_review_selection_controller.py`
- Modify: `tldw_chatbook/UI/Console_Modules/review_selection.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_change_review_opener_roots.py`
- Modify: `Tests/UI/test_console_annotation_markers.py`

**Interfaces:**

- Consumes: `agent_conversation_id_accessor`, `change_review_provider_accessor`, live run probes, `workspace_roots_accessor`, `native_messages_accessor`, `run_worker`, and an explicit store argument for discovery.
- Produces these exact moved controller signatures: `_console_change_review_provider(self) -> Any | None`, `_console_change_review_workspace_roots(self) -> tuple[str, ...] | None`, `_sync_console_annotation_discovery(self, store: Any) -> None`, and async `_load_console_annotation_previews(self, database: Any, store: Any, conversation_id: str) -> None`.

- [ ] **Step 1: Write isolated RED tests with plain fakes**

  Cover:

  - missing/raising agent identity returns no change-review provider;
  - a provider receives the live `run_active` and `run_active_for_root` callables, not captured boolean state;
  - workspace roots return the exact tuple and degrade to `None` on missing/raised collaborators;
  - missing persisted conversation clears loaded id/previews, same conversation dispatches no duplicate load, transition clears previews and dispatches one named worker;
  - annotation DB read occurs off the event-loop thread;
  - persisted annotation ids re-key to current native ids and preserve comment order;
  - a stale conversation result and a load failure leave current state untouched and do not leak note content.

  Use the exact node names `test_annotation_loader_discards_stale_conversation_result`, `test_annotation_loader_rekeys_on_event_loop_after_worker_read`, and `test_change_review_provider_uses_live_run_probes`, because Task 6 invokes them directly.

  Run and observe RED on missing methods:

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py -q
  ```

- [ ] **Step 2: Move the four methods without screen delegates**

  Preserve existing bodies and exception posture, replacing only ambient accesses with the constructor callables and owner state. The loader may update `annotation_previews` only after `await asyncio.to_thread(...)` resumes. Remove all four definitions from `ChatScreen`; do not leave same-name delegates.

- [ ] **Step 3: Retarget production callers to the owner**

  Update:

  ```python
  self._review_selection._sync_console_annotation_discovery(store)
  transcript.set_change_review_provider_factory(
      self._review_selection._console_change_review_provider
  )
  provider = self._review_selection._console_change_review_provider()
  workspace_roots = self._review_selection._console_change_review_workspace_roots()
  ```

  In the screen-owned ADR-068 reload branch, set the compatibility descriptor as today, then await `self._review_selection._load_console_annotation_previews(...)`; retain the screen-owned transcript sync immediately afterward.

- [ ] **Step 4: Retarget focused test seams only**

  - Change direct `screen._sync_console_annotation_discovery(store)` calls to `screen._review_selection._sync_console_annotation_discovery(store)`.
  - In `test_change_review_opener_roots.py`, patch or inspect the owning controller methods while keeping mounted `v`/opener expectations unchanged.
  - Keep state assertions through the three fail-loud compatibility descriptors to prove read/write forwarding and map identity.

- [ ] **Step 5: Run focused change-review/annotation tests**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py Tests/Architecture/test_console_review_selection_controller_boundary.py -q
  ```

  At this checkpoint, only the four previously documented nested-row assertions may remain red; no new failure is accepted.

- [ ] **Step 6: Commit the coherent slice**

  ```bash
  git add tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py
  git commit -m "refactor(console): move review annotation policy"
  ```

## Task 4: Move selection feedback and note policy with a safe thread boundary

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/review_selection.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_review_selection_controller.py`
- Modify: `Tests/UI/test_console_selection_end_to_end.py`
- Modify: `Tests/UI/test_console_annotation_markers.py`

**Interfaces:**

- Consumes: `store_accessor`, `show_feedback_comment`, `dispatch_prompt`, `run_worker`, `notify`.
- Produces the three remaining exact moved signatures plus event-free request entrypoints: `_record_console_feedback_event(self, store: Any | None, session_id: str | None, action: str, quote: str, comment: str, anchor_message_id: str | None) -> bool`, async `_console_selection_feedback_flow(self, action: str, quote: str, anchor_message_id: str | None = None) -> None`, and async `_create_console_selection_note(self, quote: str) -> None`.

  `_record_console_feedback_event` returns `True` only when the comment annotation was created; it catches every persistence error and never mutates controller state.

- [ ] **Step 1: Extend isolated RED coverage**

  Add plain-fake tests for blank/duplicate request guards, cancel, exact message composition, no-anchor dispatch, non-fatal audit failure, annotation-only-for-nonempty-comment, audit-before-dispatch, `finally` guard release, note validation/title/provenance/date cap, missing DB, markup-safe success, and privacy-safe failure.

  Add a thread-bound preview map:

  ```python
  class _EventLoopOnlyMap(dict):
      def __init__(self, owner_thread: int) -> None:
          super().__init__()
          self.owner_thread = owner_thread
      def __setitem__(self, key: str, value: tuple[str, ...]) -> None:
          assert threading.get_ident() == self.owner_thread
          super().__setitem__(key, value)
  ```

  Use it in `test_feedback_preview_map_changes_only_on_event_loop`; record the store write thread and assert it differs from the map mutation/event-loop thread.

  Add exact mutation nodes named `test_feedback_audit_finishes_before_dispatch`, `test_feedback_guard_releases_after_dispatch_error`, and `test_note_write_failure_never_logs_selected_text`.

- [ ] **Step 2: Move feedback policy and harden the worker boundary**

  In the event-loop coroutine, capture the current store and session id with a non-fatal guarded lookup before `to_thread`. Then:

  ```python
  annotation_created = await asyncio.to_thread(
      self._record_console_feedback_event,
      store,
      session_id,
      action,
      quote,
      comment,
      anchor_message_id,
  )
  if annotation_created and anchor_message_id:
      existing = self.annotation_previews.get(anchor_message_id, ())
      self.annotation_previews[anchor_message_id] = existing + (comment,)
  await self._dispatch_prompt("\n".join(lines))
  ```

  The blocking helper owns only store writes and a boolean result. It must not call sibling-controller accessors, inspect the screen, or mutate `annotation_previews`. Preserve audit-before-send and non-fatal audit failure.

- [ ] **Step 3: Move note policy and install the two complete Textual delegates**

  Move `_create_console_selection_note` and the validation/provenance/privacy behavior to the controller. Add request entrypoints that validate blank input, arm/schedule the reviewed worker, and capture immutable event values.

  Reduce the screen handlers to complete framework boundaries:

  ```python
  # Import the long Textual message names with these local type aliases so
  # Ruff formatting cannot expand either delegate past the five-line ceiling.
  FeedbackRequested = ConsoleSelectionFeedbackRequested
  NoteRequested = ConsoleSelectionNoteRequested

  @on(ConsoleSelectionFeedbackRequested)
  def on_console_selection_feedback_requested(self, event: FeedbackRequested) -> None:
      event.stop()
      self._review_selection.request_selection_feedback(
          event.action, event.quote, event.anchor_message_id
      )

  @on(ConsoleSelectionNoteRequested)
  def on_console_selection_note_requested(self, event: NoteRequested) -> None:
      event.stop()
      self._review_selection.request_selection_note(event.quote)
  ```

  Keep the shown short type aliases and format each definition to at most five physical lines excluding decorators; do not move policy back onto the screen to satisfy formatting.

- [ ] **Step 4: Run focused feedback/note regressions**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_annotation_markers.py Tests/Architecture/test_console_review_selection_controller_boundary.py -q
  ```

  Expected: all feedback/note/controller nodes GREEN; only documented stale nested-marker assertions may remain red.

- [ ] **Step 5: Commit the persistence/threading slice**

  ```bash
  git add tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_annotation_markers.py
  git commit -m "refactor(console): move selection feedback policy"
  ```

## Task 5: Move trajectory ownership, finish delegates, and repair focused ownership handles

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/review_selection.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_review_selection_controller.py`
- Modify: `Tests/UI/test_trajectory_live.py`
- Modify: `Tests/UI/test_console_turn_undo_all.py`
- Modify: `Tests/UI/test_console_annotation_markers.py`
- Modify: `Tests/Architecture/test_console_review_selection_controller_boundary.py`

**Interfaces:**

- Consumes: current store/session metadata, agent-runs DB seam, `run_worker`, `marshal_to_ui`, `present_trajectory`, and existing store/repository read APIs.
- Produces: module function `_build_trajectory_snapshot`, `ConsoleTrajectoryLaunch`, `open_trajectory_view()`, and the final screen action delegate.

- [ ] **Step 1: Retarget the trajectory helper tests RED-first**

  Change only the helper import:

  ```python
  from tldw_chatbook.UI.Console_Modules.review_selection import (
      _build_trajectory_snapshot,
  )
  ```

  Keep `TrajectoryScreen` imports and behavioral assertions where presentation tests need them. Run the helper nodes and observe RED until the adapter moves:

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_trajectory_live.py -q -k "build_trajectory_snapshot or snapshot_builder"
  ```

- [ ] **Step 2: Move the adapter unchanged and keep projection purity**

  Move `_build_trajectory_snapshot` from `chat_screen.py` to `review_selection.py` with its existing store/repository calls, `capture_failed` diagnostics, paging, citation verification, and `derive_trajectory` inputs unchanged. Move `ProviderUsage`/`derive_trajectory` imports with it; keep `ActiveCitationTraceState` in `chat_screen.py` if its independent citation-count path still uses it. Do not move service reads into `Chat/trajectory.py`.

- [ ] **Step 3: Implement trajectory launch sequencing on the controller**

  `open_trajectory_view()` resolves the current persisted conversation/title and current agent-runs DB, emits the same two notifications, builds off-thread, marshals back through `marshal_to_ui`, and passes a `ConsoleTrajectoryLaunch` to the presenter. Preserve `thread=True`, `exclusive=True`, and group `"trajectory-launch"`.

  The launch dataclass carries live callbacks:

  ```python
  @dataclass(frozen=True, slots=True)
  class ConsoleTrajectoryLaunch:
      snapshot: TrajectorySnapshot
      screen_title: str
      conversation_id: str
      revision_provider: Callable[[], int]
      snapshot_builder: Callable[[], TrajectorySnapshot]
  ```

- [ ] **Step 4: Install the final action delegate and retarget patch handles**

  Replace `ChatScreen.action_open_trajectory_view` with:

  ```python
  def action_open_trajectory_view(self) -> None:
      """Open Trace for the active Console conversation (``y``)."""
      self._review_selection.open_trajectory_view()
  ```

  In `test_console_turn_undo_all.py`, retarget the two monkeypatches from `screen._console_change_review_provider` to `screen._review_selection._console_change_review_provider`. Do not add a callable compatibility facade. Retarget any remaining direct moved-method test call similarly.

- [ ] **Step 5: Correct the four stale nested-marker assertions without production changes**

  Add one test-only recursive row helper:

  ```python
  def _all_rows(transcript: ConsoleTranscript) -> list[Any]:
      rows: list[Any] = []
      def visit(row: Any) -> None:
          rows.append(row)
          for nested in row.nested_rows:
              visit(nested)
      for row in transcript._transcript_rows():
          visit(row)
      return rows
  ```

  Use it only where the four stale assertions currently search top-level rows for `kind == "annotations"`. Keep the same marker count/content expectations; do not alter transcript grouping/rendering.

- [ ] **Step 6: Run the complete focused behavior/ownership suite**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_turn_undo_all.py Tests/UI/test_trajectory_live.py Tests/UI/test_console_controller_wiring.py Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/Architecture/test_console_wave6_closeout_inventory.py -q
  ```

  Expected: GREEN. This is the focused suite; do not add other repository paths merely to increase the count.

- [ ] **Step 7: Commit the completed extraction checkpoint**

  ```bash
  git add tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_trajectory_live.py Tests/UI/test_console_turn_undo_all.py Tests/UI/test_console_annotation_markers.py Tests/Architecture/test_console_review_selection_controller_boundary.py
  git commit -m "refactor(console): complete review selection ownership"
  ```

## Task 6: Prove high-risk tests with bounded manual mutations

**Files:**

- Temporarily modify, then restore with `apply_patch`: `tldw_chatbook/UI/Console_Modules/review_selection.py`
- Test: `Tests/UI/test_console_review_selection_controller.py`

**Interfaces:**

- Consumes: a clean committed extraction checkpoint.
- Produces: five observed RED/GREEN pairs with no mutation residue.

For every mutation: make one semantic edit with `apply_patch`, run only the named node and observe RED, restore the exact inverse with `apply_patch`, rerun GREEN, then require `git diff --exit-code`. Never use `git checkout`, `git reset`, a mutation dependency, or an unbounded test sweep.

- [ ] **Mutation 1: stale annotation result rejection**

  Temporarily remove/invert the `annotation_loaded_conversation` comparison before loader state replacement.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py::test_annotation_loader_discards_stale_conversation_result -q
  ```

  Expected mutated result: RED because a prior conversation paints its preview map. Restore and expect GREEN.

- [ ] **Mutation 2: feedback inflight release**

  Temporarily move `selection_feedback_inflight = False` out of the flow's `finally` block.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py::test_feedback_guard_releases_after_dispatch_error -q
  ```

  Expected mutated result: RED because an exception latches the feature. Restore and expect GREEN.

- [ ] **Mutation 3: audit-before-dispatch ordering**

  Temporarily dispatch the composed prompt before awaiting feedback persistence.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py::test_feedback_audit_finishes_before_dispatch -q
  ```

  Expected mutated result: RED on observed call order. Restore and expect GREEN.

- [ ] **Mutation 4: event-loop-only preview mutation**

  Temporarily move the `annotation_previews` assignment back inside `_record_console_feedback_event`.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py::test_feedback_preview_map_changes_only_on_event_loop -q
  ```

  Expected mutated result: RED from `_EventLoopOnlyMap` on the worker thread. Restore and expect GREEN.

- [ ] **Mutation 5: selection-note privacy logging**

  Temporarily include `title` or `quote` in the note write-failure log.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py::test_note_write_failure_never_logs_selected_text -q
  ```

  Expected mutated result: RED because the sentinel selected text appears in captured logs. Restore and expect GREEN.

- [ ] **Step 6: Confirm exact restoration**

  ```bash
  git diff --exit-code
  git status --short
  ```

  Retain the five RED/GREEN outcomes in the active evidence for final Implementation Notes; commit no mutation.

## Task 7: Run targeted static, privacy, diagnostic, and repository gates

**Files:**

- Modify only if reviewed statement ownership requires it: `Docs/security/production-diagnostic-inventory.json`
- Test: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Test: `Tests/CI/test_backlog_task_id_uniqueness.py`

**Interfaces:**

- Consumes: the clean extraction checkpoint and exact modified Python paths.
- Produces: focused Ruff/format/compile, privacy, diagnostic, task-ID, count, and diff evidence.

- [ ] **Step 1: Run targeted Ruff, format, compile, and diff checks**

  ```bash
  ../../.venv/bin/python -m ruff check tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_turn_undo_all.py Tests/UI/test_trajectory_live.py
  ../../.venv/bin/python -m ruff format --check tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_turn_undo_all.py Tests/UI/test_trajectory_live.py
  ../../.venv/bin/python -c 'from pathlib import Path; paths = ["tldw_chatbook/UI/Console_Modules/review_selection.py", "tldw_chatbook/UI/Console_Modules/wiring.py", "tldw_chatbook/UI/Screens/chat_screen.py"]; [compile(Path(path).read_bytes(), path, "exec") for path in paths]'
  git diff --check
  ```

- [ ] **Step 2: Run exact architecture/count/privacy gates**

  ```bash
  ../../.venv/bin/python -m pytest Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/Architecture/test_console_wave6_closeout_inventory.py Tests/Architecture/test_persistent_diagnostic_inventory.py Tests/CI/test_backlog_task_id_uniqueness.py -q
  rg -n "logger\.|log\.|notify\(|quote|comment|title|api[_ -]?key|provider.*error" tldw_chatbook/UI/Console_Modules/review_selection.py
  ```

  Manually verify each log/notification preserves the reviewed privacy boundary. The selected quote/comment/title may appear only in intended user-visible content or persistence arguments, never new diagnostics.

- [ ] **Step 3: Validate and reconcile persistent diagnostics only if needed**

  First run the non-write checker:

  ```bash
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
  ```

  If it reports moved statement ownership, inspect the exact source pair against the inventory base:

  ```bash
  git log -1 --format=%H -- Docs/security/production-diagnostic-inventory.json
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --statements tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/review_selection.py --since <inventory-base-sha>
  ```

  Only after confirming metadata relocation with no widened payload, regenerate through the supported writer and rerun:

  ```bash
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --write
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
  ```

- [ ] **Step 4: Run repository task hygiene and no-placeholder scans**

  ```bash
  ../../.venv/bin/python scripts/check_backlog_task_ids.py
  ../../.venv/bin/python -c 'from pathlib import Path; p = [Path("Docs/superpowers/plans/2026-08-27-task-3070-13-console-review-selection-controller.md"), Path("backlog/tasks/task-3070.13 - Extract-Console-review-and-selection-workflow-ownership.md")]; needles = [bytes.fromhex(value).decode() for value in ("544244", "544f444f", "696d706c656d656e74206c61746572", "66696c6c20696e2064657461696c73")]; hits = [(str(path), needle) for path in p for needle in needles if needle in path.read_text()]; assert not hits, hits'
  ```

  Expected: task-ID checker GREEN and placeholder scan empty.

- [ ] **Step 5: Commit diagnostic inventory relocation only if present**

  ```bash
  git add Docs/security/production-diagnostic-inventory.json
  git commit -m "docs(security): relocate review selection diagnostics"
  ```

  Skip this commit when the inventory is unchanged.

## Task 8: Rebase, revalidate, record evidence, and deliver

**Files:**

- Modify: `backlog/tasks/task-3070.13 - Extract-Console-review-and-selection-workflow-ownership.md`
- Modify only for a real reusable incident: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-backlog-hygiene.md`

**Interfaces:**

- Consumes: clean focused-green branch, five mutation proofs, and latest `origin/dev`.
- Produces: post-rebase evidence, checked acceptance criteria, Implementation Notes, Done task status, and PR-ready branch.

- [ ] **Step 1: Self-review before rebase**

  ```bash
  git diff --stat origin/dev...HEAD
  git diff --check origin/dev...HEAD
  git diff origin/dev...HEAD -- tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py
  ```

  Verify no behavior redesign, new durable authority, sibling-controller object, broad screen facade, stale re-export, shadow state, eager trajectory import, privacy widening, ratchet increase, or unrelated edit entered the branch.

- [ ] **Step 2: Rebase onto latest dev with a clean tree**

  ```bash
  git status --short
  git fetch origin dev
  git rebase origin/dev
  ```

  If the rebase changes the reviewed family or makes the architecture drift test red, stop and amend the spec/task rather than absorbing the change.

- [ ] **Step 3: Run the executable live-base drift gate first**

  ```bash
  ../../.venv/bin/python -m pytest Tests/Architecture/test_console_review_selection_controller_boundary.py::test_origin_dev_review_selection_family_still_matches_review -q
  ```

  Expected: GREEN in either complete pre-extraction or complete post-extraction state; partial/unrecognized state is a blocker.

- [ ] **Step 4: Rerun final focused evidence after rebase**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_turn_undo_all.py Tests/UI/test_trajectory_live.py Tests/UI/test_console_controller_wiring.py Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/Architecture/test_console_wave6_closeout_inventory.py -q
  ../../.venv/bin/python -m ruff check tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_turn_undo_all.py Tests/UI/test_trajectory_live.py
  ../../.venv/bin/python -m ruff format --check tldw_chatbook/UI/Console_Modules/review_selection.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_review_selection_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_review_selection_controller.py Tests/UI/test_change_review_opener_roots.py Tests/UI/test_console_annotation_markers.py Tests/UI/test_console_selection_end_to_end.py Tests/UI/test_console_turn_undo_all.py Tests/UI/test_trajectory_live.py
  ../../.venv/bin/python -c 'from pathlib import Path; paths = ["tldw_chatbook/UI/Console_Modules/review_selection.py", "tldw_chatbook/UI/Console_Modules/wiring.py", "tldw_chatbook/UI/Screens/chat_screen.py"]; [compile(Path(path).read_bytes(), path, "exec") for path in paths]'
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
  ../../.venv/bin/python scripts/check_backlog_task_ids.py
  git diff --check
  ```

  Expected: all targeted gates GREEN. Do not run a local full suite.

- [ ] **Step 5: Record post-rebase implementation evidence**

  Use Backlog.md CLI to check all five acceptance criteria and add concise `## Implementation Notes` covering:

  - the final one-owner extraction and exact 7/3/6 boundary;
  - final `ChatScreen` line/direct-method counts and conservative reduction;
  - named fine-grained wiring and three fail-loud descriptors;
  - preserved ADR-068 screen-owned review-note flow and existing Git/database authority;
  - worker-thread persistence versus event-loop preview mutation evidence;
  - focused pytest/Ruff/format/compile/privacy/diagnostic/task-ID/diff results;
  - all five mutation RED/GREEN outcomes;
  - ADR required: no, ADR-068 path/reason, plan deviations, and whether a real lessons entry was warranted.

  Do not mark Done until every Definition-of-Done item is satisfied.

- [ ] **Step 6: Commit task evidence and complete the Backlog task**

  ```bash
  git add "backlog/tasks/task-3070.13 - Extract-Console-review-and-selection-workflow-ownership.md"
  git commit -m "docs(backlog): record review selection evidence"
  backlog task edit 3070.13 -s Done
  git add "backlog/tasks/task-3070.13 - Extract-Console-review-and-selection-workflow-ownership.md"
  git commit -m "docs(backlog): complete TASK-3070.13"
  ```

  Add a lessons file to the first commit only if implementation surfaced a genuinely reusable incident; do not invent one.

- [ ] **Step 7: PR and merge handoff**

  Push normally, or use `--force-with-lease` only if rebase rewrote the remote branch. Open/update the PR, wait for Qodo, inspect every top-level and inline review comment, verify each finding against current code, implement only technically valid minimal fixes, reply in original inline threads when applicable, rerun only affected focused tests plus required gates, and merge only when review is fully addressed, latest dev is included, required GitHub Actions are green, and GitHub permits the merge.

## Plan self-review

- Spec coverage: Tasks 1-5 map every goal, non-goal, ownership rule, state contract, runtime flow, compatibility retarget, and verification requirement to an executable step.
- Placeholder scan: the plan contains no implementation placeholder; `<inventory-base-sha>` is an explicit runtime value obtained by the immediately preceding command, not missing design work.
- Type consistency: the controller constructor, seven moved method signatures, three event-free entrypoints, `ConsoleTrajectoryLaunch`, descriptor targets, and caller spellings are consistent across all tasks.
- Scope check: this remains one atomic controller extraction/PR; change review, annotations, feedback, notes, and trajectory are one current screen ownership family and share state/wiring/verification boundaries.
