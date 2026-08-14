# Console Image and H3 Controller Implementation Plan

> **For Codex:** Execute this plan in order with test-driven development and verify each slice before continuing.

**Goal:** Move the reviewed non-DOM Console image-generation and H3 lifecycle cluster from `ChatScreen` into one `ConsoleImageController` while preserving H3 identity, persistence, remount, and UI-settlement behavior.

**Architecture:** `ConsoleImageController` owns the 25 methods recorded by the approved Wave 6 inventory and the three compatible state attributes. `ChatScreen` retains the five DOM/modal/composer methods and one framework-required slash-command delegate. `wiring.py` supplies named keyword-only late-bound callables; `ConsoleMessageController` calls the image controller directly for regenerate/select/keep/toggle actions.

**Tech Stack:** Python 3.11, Textual 8.x, asyncio/threading, pytest, Ruff.

**ADR required:** no

**ADR path:** N/A

**Reason:** This implements the already-approved Wave 6 ownership boundary and introduces no new storage, security, provider, runtime, or cross-module policy.

---

### Task 1: Lock the baseline and RED contracts

**Files:**
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`
- Create: `Tests/Chat/test_console_image_controller.py`
- Test: `Tests/Chat/test_console_h3_image_edit.py`
- Test: `Tests/Chat/test_console_image_view.py`

1. Run the existing architecture, image-view, H3, and generation-action tests on the unmodified implementation.
2. Add isolated controller-construction tests that do not mount Textual.
3. Add RED assertions for controller ownership, all-at-once compatibility descriptors, direct MessageController wiring, exact cancel-event identity, remount settlement, and persistence ordering.
4. Run only the named RED tests and confirm they fail because the controller/ownership boundary is absent.

### Task 2: Add the controller seam and compatible state

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/image.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Chat/test_console_image_controller.py`

1. Add one small reusable read/write controller-state descriptor at module scope.
2. Put `_imagegen_inflight_sessions`, `_imagegen_inflight_message_ids`, and `_console_h3_ui_generations` defaults on `ConsoleImageController`.
3. Construct `screen._image` from `wiring.py` using named keyword-only late-bound dependencies.
4. Wire MessageController regenerate/select/keep/toggle callbacks directly to `screen._image`.
5. Run the isolated seam and compatibility tests until GREEN.

### Task 3: Move projection and remote-image policy

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/image.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Chat/test_console_image_controller.py`
- Test: `Tests/Chat/test_console_image_view.py`

1. Move image spec, remote fetch, generation-card spec, pending decode, and view-toggle methods without changing their decisions.
2. Replace DOM/framework reaches with the named callables supplied by wiring.
3. Rewire staying screen callers directly to `self._image`.
4. Run isolated projection tests and the existing image-view suite.

### Task 4: Move H3 lifecycle ownership

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/image.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/Chat/test_console_image_controller.py`
- Test: `Tests/Chat/test_console_h3_image_edit.py`

1. Move registry, immutable reference snapshot, attachment filtering, live-origin, cleanup, reconciliation, failure hydration, settlement, and command-runner methods.
2. Preserve the exact `threading.Event` object through registry, worker, and adapter boundaries.
3. Preserve durable-message-first ordering, byte-free completion records, remount adoption, source cleanup, and safe failure copy.
4. Rewire mount/unmount/resume/cancel call sites to the controller.
5. Run isolated lifecycle tests and the existing H3 suite; mutate event identity and cleanup ordering once to prove the tests fail.

### Task 5: Move ordinary generation and variant actions

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/image.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Test: `Tests/Chat/test_console_image_controller.py`
- Test: existing generate-image and generation-card tests under `Tests/Chat/`

1. Move conversation context, provider resolution, ordinary generation, regenerate, select, and keep behavior.
2. Retain `_console_command_generate_image` as a physical-span-bounded screen delegate to `self._image`.
3. Keep the five approved DOM/modal/composer methods on `ChatScreen` and route their policy calls to the controller.
4. Run action and command tests until GREEN.

### Task 6: Verify, review, and close out

**Files:**
- Modify: `backlog/tasks/task-3070.2 - Extract-Console-image-and-H3-controller.md`

1. Run the Wave 6 architecture test and all focused image/H3/generation tests.
2. Run Ruff check and format-check on every changed Python file, plus isolated `py_compile` for production modules.
3. Run the full pytest suite and record any pre-existing/platform skips separately.
4. Run `git diff --check`, privacy/artifact/scope checks, and inspect the complete diff for unnecessary abstraction.
5. Add concise Implementation Notes, check all ACs only after evidence is green, and set TASK-3070.2 to Done.
