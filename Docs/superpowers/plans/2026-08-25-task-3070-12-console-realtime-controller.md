# TASK-3070.12 Console Realtime Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** Move the Console's 56 reviewed realtime orchestration methods from `ChatScreen` into one explicit non-DOM `ConsoleRealtimeController` while preserving every characterized transport, audio, transcript, fallback, cancellation, remount, teardown, privacy, and visible behavior.

**Architecture:** `tldw_chatbook/UI/Console_Modules/realtime.py` will own the realtime session model, controller-only constants, mutable realtime state, and the exact 56-method move set frozen by the Wave 6 inventory. The existing `build_console_controllers()` function will construct `screen._realtime` from named, late-bound callables; `ChatScreen` will retain only `_repaint_console_realtime_chip`, the `on_key` framework boundary, `on_unmount`, and two fail-loud `_ControllerState` compatibility descriptors with no shadow storage. No second wiring API, dependency object, screen reference, sibling-controller reference, DOM access, mixin, or behavior change is introduced.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Ruff, AST source inspection, Backlog.md CLI, existing diagnostic inventory scripts.

---

## Scope and frozen evidence

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-3070-12-console-realtime`
- Design: `Docs/superpowers/specs/2026-08-24-task-3070-12-console-realtime-controller-design.md`
- Characterization owner: `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- Reviewed classification: 57 realtime methods = 56 moves, 0 delegates, 1 screen stay (`_repaint_console_realtime_chip`).
- Planning base: `7f38cb6ef09ade054c755600c5b48435571482da`; current `ChatScreen` = 20,017 physical lines / 633 direct methods.
- Conservative task result: no more than 18,039 physical lines / 577 direct methods.
- Local full-suite execution is prohibited. GitHub Actions remains the broad integration gate.
- Two focused Buddy tests are already red at the planning base because their skeletal app does not install a lazily constructed Buddy controller. Correct only that test fixture by assigning a real `PersonaBuddyController`; do not change production lazy-loading behavior.
- The Wave 6 inventory has one unrelated pre-existing red ratchet (`test_first_chat_task_ratchet_is_earned` at 20,017 > 19,922). This extraction must earn that assertion back without raising or weakening a ceiling.

## ADR check

ADR required: no

ADR path: N/A

Reason: this task directly applies the accepted controller/region ownership rules in `DESIGN.md` section 7 and the approved Wave 6 closeout amendment. It changes no storage, schema, sync policy, service contract, provider boundary, dependency, security/privacy policy, or long-lived application structure.

## Task 0: Record the approved plan before implementation

**Files:**

- Modify: `backlog/tasks/task-3070.12 - Extract-Console-realtime-orchestration-ownership.md`
- Add: `Docs/superpowers/plans/2026-08-25-task-3070-12-console-realtime-controller.md`

- [ ] **Step 1: Attach the implementation plan to the In Progress task**

  The task must already be `In Progress`. Record the approved plan summary through Backlog.md CLI before any production change:

  ```bash
  backlog task edit 3070.12 --plan "1. Freeze the exact realtime ownership boundary with RED architecture tests\n2. Establish ConsoleRealtimeController through build_console_controllers and fail-loud descriptors\n3. Move and characterize the 56 policy methods with plain-fake tests\n4. Retarget mounted ownership seams and correct the two stale Buddy fixtures\n5. Prove five high-risk branches with bounded manual mutations\n6. Run focused architecture, static, privacy, diagnostic, and inventory gates\n7. Rebase on latest dev, revalidate drift, refresh evidence, and complete delivery\n\nADR required: no\nADR path: N/A\nReason: applies DESIGN.md section 7 and the approved Wave 6 closeout boundary without changing a durable architecture contract."
  ```

- [ ] **Step 2: Commit the planning artifacts**

  ```bash
  git add "backlog/tasks/task-3070.12 - Extract-Console-realtime-orchestration-ownership.md" Docs/superpowers/plans/2026-08-25-task-3070-12-console-realtime-controller.md
  git commit -m "docs: plan Console realtime controller extraction"
  ```

## Task 1: Freeze the controller boundary with RED architecture tests

**Files:**

- Create: `Tests/Architecture/test_console_realtime_controller_boundary.py`
- Reference: `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- Reference: `tldw_chatbook/UI/Screens/chat_screen.py`
- Reference: `tldw_chatbook/UI/Console_Modules/wiring.py`

- [ ] **Step 1: Add a path-first architecture test for the new owner**

  Parse files with `ast`; do not import a module that does not exist yet. Import `REALTIME_MOVE_METHODS` and `REALTIME_STAY_METHODS` from the frozen Wave 6 inventory and assert:

  ```python
  assert REALTIME_PATH.is_file()
  realtime_owner = _class_node(REALTIME_PATH, "ConsoleRealtimeController")
  assert REALTIME_MOVE_METHODS <= _direct_method_names(realtime_owner)
  ```

  Add exact post-extraction assertions in the same file for later GREEN:

  - all 56 move names are direct methods on `ConsoleRealtimeController` and absent from `ChatScreen`;
  - `_repaint_console_realtime_chip` remains a direct `ChatScreen` method and is absent from the controller;
  - no callable `ChatScreen` delegate carries a moved name;
  - no dynamic facade (`__getattr__`/`__getattribute__`) or realtime mixin bypasses ownership;
  - `_console_realtime` and `_console_realtime_close_worker` are `_ControllerState` descriptors pointing at `("_realtime", "session")` and `("_realtime", "close_worker")` with no instance shadow assignments;
  - descriptor access before wiring raises `RuntimeError` rather than being swallowed as a missing optional attribute;
  - controller source has no `query`, `query_one`, `focus`, `push_screen`, modal construction, `ChatScreen`, or AST attribute access to the sibling-controller slots `self._dictation`, `self._hands_free`, or `self._session`;
  - `build_console_controllers()` assigns `screen._realtime` exactly once and no other production function constructs the controller;
  - `chat_screen.py` is at most 18,039 lines and 577 direct methods.

  Add an executable `test_origin_dev_realtime_family_and_projection_still_match_review` which reads `origin/dev:tldw_chatbook/UI/Screens/chat_screen.py` with bounded `git show` and asserts:

  - the exact direct-method names containing `console_realtime` equal `REALTIME_METHODS` (57 total);
  - the exact move/delegate/stay classification remains 56/0/1;
  - realtime candidate and stay spans remain 1,997 and 19 physical lines;
  - subtracting the 1,978-line / 56-method extraction from the current `origin/dev` counts projects no higher than the reviewed 18,039-line / 577-method result.

  This is the current-base/rebase drift gate. The frozen amendment tests remain historical evidence and do not substitute for it.

- [ ] **Step 2: Run the new architecture file and observe RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest Tests/Architecture/test_console_realtime_controller_boundary.py -q
  ```

  Expected: FAIL because `UI/Console_Modules/realtime.py` and `ConsoleRealtimeController` do not exist and all 56 methods still belong to `ChatScreen`.

- [ ] **Step 3: Commit the executable RED contract**

  ```bash
  git add Tests/Architecture/test_console_realtime_controller_boundary.py
  git commit -m "test(console): freeze realtime controller boundary"
  ```

## Task 2: Establish the controller seam and existing wiring path

**Files:**

- Create: `tldw_chatbook/UI/Console_Modules/realtime.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_controller_wiring.py`
- Test: `Tests/Architecture/test_console_realtime_controller_boundary.py`

- [ ] **Step 1: Extend the controller wiring test first**

  Import `ConsoleRealtimeController`, add `_realtime` to `_ALL_CONTROLLER_SLOTS` after `_hands_free`, rename the count-specific test from fourteen to fifteen, and assert the installed instance type and the named seam callables. Keep the existing test's construction-order and late-binding assertions.

  Run:

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_controller_wiring.py -q
  ```

  Expected: collection FAIL because the new owner does not exist yet.

- [ ] **Step 2: Add the smallest importable controller skeleton**

  Move `ConsoleRealtimeSession` into `realtime.py` first, import that model back into `chat_screen.py` for the temporary transition, then add the production imports that the moved methods will own, a module docstring explaining the 56/0/1 boundary, and:

  ```python
  class ConsoleRealtimeController:
      def __init__(self, *, ...named keyword-only late-bound callables...) -> None:
          self.session: ConsoleRealtimeSession | None = None
          self.close_worker: Worker | None = None
          # Store only the named callables; never store ChatScreen or sibling controllers.
  ```

  The constructor signature must name each real dependency instead of hiding them in a dependency object. Use late-bound callables for mutable framework/application seams, including chat-store/runtime/dictation access, provider-session/recorder/sink factories, notifications, UI marshaling/scheduling, chip repaint, voice-chip restore, and pipeline fallback. Snapshot only an identity whose lifetime is demonstrably stable and document that exception; otherwise use a callable.

- [ ] **Step 3: Construct the controller in the existing builder**

  In `build_console_controllers()`:

  - import `ConsoleRealtimeController` from `.realtime`;
  - construct `screen._realtime` immediately after `_hands_free` and before `_message`;
  - pass explicit late-bound lambdas which resolve screen services and cross-controller actions at call time;
  - retarget dictation's `realtime_adopt_transcript` and `realtime_session_accessor` to `screen._realtime`;
  - retarget hands-free's `realtime_session_accessor` and `enter_realtime_loop` to `screen._realtime`;
  - retarget the auto-speak realtime-active read to `screen._realtime.session`;
  - update the builder docstring and attachment list from fourteen to fifteen controllers.

  Do not add `wire_console_screen()`, a factory, a dependency container, or a second construction call.

- [ ] **Step 4: Install fail-loud compatibility descriptors**

  Reuse the existing descriptor class in `chat_screen.py`:

  ```python
  _console_realtime = _ControllerState("_realtime", "session")
  _console_realtime_close_worker = _ControllerState("_realtime", "close_worker")
  ```

  Remove the old `__init__` assignments for those two state names so there is no shadow state. At this stage the old screen methods may remain temporarily while behavior is moved, but all state reads/writes must forward to the controller-owned slots.

- [ ] **Step 5: Run the seam tests**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_controller_wiring.py Tests/Architecture/test_console_realtime_controller_boundary.py -q
  ```

  Expected: wiring tests GREEN; architecture tests remain RED only for the not-yet-moved 56 methods and final size count.

- [ ] **Step 6: Commit the controller seam**

  ```bash
  git add tldw_chatbook/UI/Console_Modules/realtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_controller_wiring.py Tests/Architecture/test_console_realtime_controller_boundary.py
  git commit -m "refactor(console): establish realtime controller seam"
  ```

## Task 3: Characterize controller policy with plain fakes

**Files:**

- Create: `Tests/UI/test_console_realtime_controller.py`
- Modify: `tldw_chatbook/UI/Console_Modules/realtime.py`
- Reference: `Tests/UI/test_console_realtime_wiring.py`
- Reference: `Tests/Chat/test_console_realtime_loop.py`

- [ ] **Step 1: Write isolated RED tests around high-risk orchestration**

  Use plain fake callables/sessions/taps/sinks/workers; do not mount Textual and do not mock `ChatScreen`. Pin at least:

  1. construction starts with `session is None` and `close_worker is None`;
  2. loop entry installs the session model before the FSM emits its first `connecting` intent;
  3. callback marshaling accepts the current session and rejects a stale session/attempt;
  4. connect creates the tap and provider session once, preserves first-words buffering, and seeds in the existing order; the audio sink remains lazy and is created per reply only on the first audio delta;
  5. reconnect reuses the tap, replaces the provider session, reseeds, and exhausts after one attempt into the existing fallback/exit path;
  6. input transcript, pending reply row, transcript deltas, audio, playback completion, and usage persistence preserve existing ordering and metadata;
  7. key handling consumes Escape only, while another key reports a keypress barge without stealing the event;
  8. active teardown stops tap, closes provider session, then stops sink, clears state, and releases the exact Buddy generation;
  9. teardown awaits a retained close worker even after active session state is cleared;
  10. failure sanitization prevents API-key fragments, raw provider payloads, audio bytes, and private transcript text from diagnostics/notifications.

  Run:

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_controller.py -q
  ```

  Expected: RED on the first missing moved method/behavior.

- [ ] **Step 2: Move the controller-only constants**

  Move constants used only by the realtime controller from `chat_screen.py` into `realtime.py`. Keep the chip message mapping available to `chat_screen.py` for `_repaint_console_realtime_chip`, but do not re-export controller implementation details through the screen module.

- [ ] **Step 3: Move the exact 56 methods in coherent lifecycle slices**

  Move the methods named by `REALTIME_MOVE_METHODS` without renaming or delegating them. Mechanically replace screen-owned state access with `self.session` / `self.close_worker`, and replace the limited external screen calls with the named constructor dependencies. Preserve bodies and ordering unless a mechanical receiver change is required.

  Implement only two small public boundaries in addition to the frozen private move set:

  ```python
  def handle_key(self, key: str) -> bool:
      """Handle realtime key policy and report whether Escape was consumed."""

  async def teardown(self) -> None:
      """Release an active loop and await any retained close worker."""
  ```

  Preserve these invariants exactly:

  - install `ConsoleRealtimeSession` before `RealtimeLoopController.enter()`;
  - preserve attempt/session identity guards in every marshaled callback;
  - preserve connect/seed/ready order and first-words buffering;
  - preserve one reconnect, same tap, new provider session, reseed, and loud fallback;
  - preserve transcript row status, metadata, audio/playback, usage, and error sanitization;
  - preserve remount-safe Buddy generations and exact-owner release;
  - preserve off-UI-thread tap stop and retained close-worker cleanup.

- [ ] **Step 4: Keep only framework/presentation boundaries on the screen**

  - Keep `_repaint_console_realtime_chip` byte-for-byte except for importing the chip message mapping from `realtime.py`.
  - In `ChatScreen.on_key`, call `self._realtime.handle_key(event.key)`; only when it returns `True`, call `event.stop()`, `event.prevent_default()`, and return.
  - In `ChatScreen.on_unmount`, replace the embedded realtime cleanup with `await self._realtime.teardown()`.
  - Remove all 56 moved definitions from `ChatScreen`; do not leave same-name delegates.

- [ ] **Step 5: Run isolated and architecture tests until GREEN**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_controller_wiring.py Tests/Architecture/test_console_realtime_controller_boundary.py Tests/Architecture/test_console_wave6_closeout_inventory.py Tests/Architecture/test_console_wave6_inventory.py -q
  ```

  Expected: GREEN, including the previously over-budget first-chat ratchet now earned by the extraction. Do not weaken any budget.

- [ ] **Step 6: Commit the extraction checkpoint**

  ```bash
  git add tldw_chatbook/UI/Console_Modules/realtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_realtime_controller.py Tests/Architecture/test_console_realtime_controller_boundary.py
  git commit -m "refactor(console): extract realtime orchestration owner"
  ```

## Task 4: Retarget mounted ownership seams and restore baseline fixture validity

**Files:**

- Modify: `Tests/UI/test_console_realtime_wiring.py`
- Possibly modify only if receiver access requires it: `Tests/UI/test_console_button_routing.py`
- Test: `Tests/Chat/test_console_realtime_loop.py`
- Test: `Tests/Audio/test_realtime_mic_tap.py`
- Test: `Tests/LLM_Calls/test_realtime_protocol.py`

- [ ] **Step 1: Retarget patches to the owning module**

  Add:

  ```python
  from tldw_chatbook.UI.Console_Modules import realtime as realtime_module
  ```

  Retarget only symbols now owned by `realtime.py`: `get_api_key`, realtime provider/model/voice/idle/acoustic settings, VAD helpers, realtime timeout/controller constants, transcript constants, and failure markers. Keep hands-free engine/pipeline patches on `hands_free_module`. Do not preserve stale `chat_screen_module` re-exports merely to keep old patch handles alive.

  Retarget direct private method calls from `console._moved_name(...)` to `console._realtime._moved_name(...)`; leave assertions and mounted user actions unchanged.

- [ ] **Step 2: Correct the two pre-existing Buddy fixture defects**

  In only the two Buddy realtime tests, install the app-owned test controller explicitly:

  ```python
  from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController

  app.persona_buddy_controller = PersonaBuddyController()
  ```

  This mirrors existing repository test injection and exercises the same FSM/generation contract. Do not add `local_character_persona_service` to all skeletal apps and do not make production construct a controller for a disabled/missing-service profile.

- [ ] **Step 3: Run the two corrected baseline nodes first**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_wiring.py::test_persona_buddy_realtime_fsm_replaces_generation_and_releases_on_exit Tests/UI/test_console_realtime_wiring.py::test_persona_buddy_realtime_generation_survives_screen_replacement -q
  ```

  Expected: GREEN with only fixture changes.

- [ ] **Step 4: Run the complete focused realtime regression set**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_wiring.py Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_button_routing.py::test_mic_button_exits_the_realtime_loop_instead_of_toggling Tests/Chat/test_console_realtime_loop.py Tests/Audio/test_realtime_mic_tap.py Tests/LLM_Calls/test_realtime_protocol.py -q
  ```

  Expected: GREEN. Do not run `pytest` without explicit paths.

- [ ] **Step 5: Commit test ownership updates**

  ```bash
  git add Tests/UI/test_console_realtime_wiring.py Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_button_routing.py
  git commit -m "test(console): verify extracted realtime orchestration"
  ```

  Omit `Tests/UI/test_console_button_routing.py` from `git add` if it needed no edit.

## Task 5: Prove the policy tests are non-vacuous with bounded manual mutations

**Files:**

- Temporarily modify, then restore with `apply_patch`: `tldw_chatbook/UI/Console_Modules/realtime.py`
- Test: `Tests/UI/test_console_realtime_controller.py`
- Test: `Tests/UI/test_console_realtime_wiring.py`

Perform this task only after the extraction checkpoint is committed and the worktree is clean. For each mutation: apply one semantic edit with `apply_patch`, run only the named node(s) and observe RED, restore that exact edit with `apply_patch`, rerun the node(s) and observe GREEN, then confirm `git diff --exit-code`. Do not use `git checkout`, `git reset`, a mutation dependency, or an unbounded sweep.

- [ ] **Mutation 1: session-before-FSM ordering**

  Temporary edit: move assignment of `self.session = state` from before to after `state.controller.enter()`.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_controller.py::test_enter_installs_session_before_fsm_emits_connecting -q
  ```

  Expected mutated result: RED because the emitted `connecting` intent cannot observe the active state. Restore and expect GREEN.

- [ ] **Mutation 2: stale-session rejection**

  Temporary edit: invert/remove the current-session identity guard in `_console_realtime_marshal`.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_controller.py::test_marshal_rejects_stale_session_callback -q
  ```

  Expected mutated result: RED because a stale callback mutates the replacement loop. Restore and expect GREEN.

- [ ] **Mutation 3: reconnect exhaustion and fallback**

  Temporary edit: allow a second reconnect instead of exiting after the one reviewed retry.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_wiring.py::test_transport_drop_reconnects_once_and_reseeds Tests/UI/test_console_realtime_wiring.py::test_auth_failure_during_reconnect_gives_up_instead_of_hanging -q
  ```

  Expected mutated result: RED because the second drop/reconnect failure no longer exits into the loud failure path. Restore and expect GREEN.

- [ ] **Mutation 4: generation-safe Buddy release**

  Temporary edit: release Buddy voice without checking the state's captured generation/owner.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_wiring.py::test_persona_buddy_realtime_generation_survives_screen_replacement -q
  ```

  Expected mutated result: RED because the stale screen releases its successor. Restore and expect GREEN.

- [ ] **Mutation 5: retained close-worker teardown**

  Temporary edit: make `teardown()` ignore `self.close_worker` after active state has cleared.

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_wiring.py::test_unmount_right_after_exit_still_closes_the_session -q
  ```

  Expected mutated result: RED because immediate unmount leaves the provider session open. Restore and expect GREEN.

- [ ] **Step 6: Record the mutation evidence without committing mutations**

  Retain the five RED/GREEN command outcomes in the active implementation-session evidence; do not edit a worktree file yet. Task 7 Step 5 records them in the final post-rebase Implementation Notes. Confirm the restored production/tests tree has no mutation residue:

  ```bash
  git diff --exit-code
  ```

## Task 6: Run targeted static, privacy, diagnostic, and inventory gates

**Files:**

- Modify if generated statement ownership changes: `Docs/security/production-diagnostic-inventory.json`
- Modify: `backlog/tasks/task-3070.12 - Extract-Console-realtime-orchestration-ownership.md`
- Test: `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- Test: `Tests/Architecture/test_console_wave6_inventory.py`

- [ ] **Step 1: Run targeted Python/static gates**

  Use the exact modified Python-file list from `git diff --name-only --diff-filter=ACM origin/dev...HEAD`:

  ```bash
  ../../.venv/bin/python -m ruff check tldw_chatbook/UI/Console_Modules/realtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_realtime_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_realtime_wiring.py
  ../../.venv/bin/python -m ruff format --check tldw_chatbook/UI/Console_Modules/realtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_realtime_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_realtime_wiring.py
  ../../.venv/bin/python -c 'from pathlib import Path; paths = ["tldw_chatbook/UI/Console_Modules/realtime.py", "tldw_chatbook/UI/Console_Modules/wiring.py", "tldw_chatbook/UI/Screens/chat_screen.py"]; [compile(Path(path).read_bytes(), path, "exec") for path in paths]'
  git diff --check
  ```

  Include `Tests/UI/test_console_button_routing.py` if modified.

- [ ] **Step 2: Run architecture and privacy-focused gates**

  ```bash
  ../../.venv/bin/python -m pytest Tests/Architecture/test_console_realtime_controller_boundary.py Tests/Architecture/test_console_wave6_closeout_inventory.py Tests/Architecture/test_console_wave6_inventory.py -q
  rg -n "api[_ -]?key|audio_frames|transcript|provider.*error|logger\.|log\." tldw_chatbook/UI/Console_Modules/realtime.py
  ```

  Inspect each diagnostic/log/notification field manually against the existing redaction boundary. Raw API keys, provider payloads, PCM bytes, and private transcript text must not be newly persisted or logged.

- [ ] **Step 3: Review and update the diagnostic inventory only if required**

  Find the inventory comparison base:

  ```bash
  git log -1 --format=%H -- Docs/security/production-diagnostic-inventory.json
  ```

  Run validation first:

  ```bash
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
  ```

  If statement ownership moved, review only the affected files against the printed/base revision:

  ```bash
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --statements tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/realtime.py --since <inventory-base-sha>
  ```

  After confirming only ownership paths/line locations changed and no durable diagnostic widened, regenerate through the supported writer:

  ```bash
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --write
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
  ```

- [ ] **Step 4: Run repository task hygiene gate**

  ```bash
  ../../.venv/bin/python scripts/check_backlog_task_ids.py
  ```

- [ ] **Step 5: Commit diagnostic inventory changes if present**

  ```bash
  git add Docs/security/production-diagnostic-inventory.json
  git commit -m "docs(security): relocate realtime diagnostics inventory"
  ```

  Skip the commit when the inventory is unchanged.

## Task 7: Complete task records, rebase, and rerun exact focused evidence

**Files:**

- Modify: `backlog/tasks/task-3070.12 - Extract-Console-realtime-orchestration-ownership.md`
- Modify if needed: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Self-review the final diff before declaring completion**

  ```bash
  git diff --stat origin/dev...HEAD
  git diff --check origin/dev...HEAD
  git diff origin/dev...HEAD -- tldw_chatbook/UI/Console_Modules/realtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py
  ```

  Verify no behavior/UX fix, dependency change, public API, stale screen patch alias, sibling-controller reach-back, shadow state, logging expansion, or unrelated edit entered the branch.

- [ ] **Step 2: Rebase onto the latest `origin/dev`**

  ```bash
  git status --short
  git fetch origin dev
  git rebase origin/dev
  ```

  The status must be clean before rebase. Do not record final counts or final test evidence before this step because a valid rebase can change both.

- [ ] **Step 3: Run the executable current-base drift gate first**

  ```bash
  ../../.venv/bin/python -m pytest Tests/Architecture/test_console_realtime_controller_boundary.py::test_origin_dev_realtime_family_and_projection_still_match_review -q
  ```

  This test inspects the freshly fetched `origin/dev`, not the historical amendment revision. If the exact 57-method family, 1,997/19 spans, 56/0/1 classification, or conservative 18,039/577 projection changed, stop and amend the design/task before continuing. Never rewrite frozen evidence or raise a ratchet to absorb drift.

- [ ] **Step 4: Run final focused evidence after rebase**

  ```bash
  ../../.venv/bin/python -m pytest Tests/UI/test_console_realtime_wiring.py Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_button_routing.py::test_mic_button_exits_the_realtime_loop_instead_of_toggling Tests/Chat/test_console_realtime_loop.py Tests/Audio/test_realtime_mic_tap.py Tests/LLM_Calls/test_realtime_protocol.py Tests/Architecture/test_console_realtime_controller_boundary.py Tests/Architecture/test_console_wave6_closeout_inventory.py Tests/Architecture/test_console_wave6_inventory.py -q
  ../../.venv/bin/python -m ruff check tldw_chatbook/UI/Console_Modules/realtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_realtime_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_realtime_wiring.py
  ../../.venv/bin/python -m ruff format --check tldw_chatbook/UI/Console_Modules/realtime.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Architecture/test_console_realtime_controller_boundary.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_realtime_controller.py Tests/UI/test_console_realtime_wiring.py
  ../../.venv/bin/python -c 'from pathlib import Path; paths = ["tldw_chatbook/UI/Console_Modules/realtime.py", "tldw_chatbook/UI/Console_Modules/wiring.py", "tldw_chatbook/UI/Screens/chat_screen.py"]; [compile(Path(path).read_bytes(), path, "exec") for path in paths]'
  ../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
  ../../.venv/bin/python scripts/check_backlog_task_ids.py
  git diff --check
  ```

  Expected: all targeted gates GREEN. No local full suite.

- [ ] **Step 5: Refresh Backlog task acceptance criteria and final notes from post-rebase evidence**

  Use Backlog.md CLI to add concise Implementation Notes containing:

  - one-owner extraction and named dependency wiring;
  - exact post-rebase 56/0/1 result and final line/method counts;
  - preserved realtime behavior and compatibility descriptors;
  - post-rebase focused pytest/Ruff/format/isolated-compile/privacy/diagnostic results;
  - five mutation RED/GREEN outcomes;
  - ADR required: no / ADR path: N/A / reason;
  - any deviation from this plan;
  - whether a generalizable lessons entry was warranted (do not invent one).

  Check all four acceptance criteria only after this post-rebase evidence is green. Do not mark Done yet.

- [ ] **Step 6: Commit the refreshed task records**

  ```bash
  git add "backlog/tasks/task-3070.12 - Extract-Console-realtime-orchestration-ownership.md" backlog/docs/lessons-testing-evidence.md
  git commit -m "docs(backlog): record realtime extraction evidence"
  ```

  Omit the lessons file if no real reusable incident occurred. If this documentation commit changes a checked file, rerun its exact targeted gate.

- [ ] **Step 7: Mark TASK-3070.12 Done only after every Definition-of-Done item is satisfied**

  ```bash
  backlog task edit 3070.12 -s Done
  git add "backlog/tasks/task-3070.12 - Extract-Console-realtime-orchestration-ownership.md"
  git commit -m "docs(backlog): complete TASK-3070.12"
  ```

- [ ] **Step 8: Delivery handoff**

  Push the rebased branch with `--force-with-lease` only if history changed, open/update the PR, inspect all top-level and inline review feedback, implement only technically valid minimal fixes, reply in original inline threads when applicable, rerun only affected focused tests plus the required gates, wait for required GitHub Actions, and merge only when review is fully addressed, the branch is current, checks are green, and merge is allowed.
