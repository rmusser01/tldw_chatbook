# Console Video Controller Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the reviewed generated-video state and lifecycle policy from `ChatScreen` into a no-DOM `ConsoleVideoController` without changing cancellation, persistence, publication, remount, or presentation behavior.

**Architecture:** Add one controller in `UI/Console_Modules/video.py`, constructed in the existing wiring graph with explicit late-bound callbacks. Move the 31 reviewed methods byte-faithfully, retain only the two registry-bound command delegates on `ChatScreen`, and preserve the eight assignable screen names with the existing `_ControllerState` descriptor.

**Tech Stack:** Python 3.11+, asyncio/threading, Textual 8, pytest, Ruff.

---

### Task 0: Lock the approved post-image baseline amendment

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`
- Modify: `backlog/tasks/task-3070*.md`

- [x] Record post-image base `8d806b71d9c5ae7ed333ccb42780f6b2ea68acd0`
  independently from the immutable original implementation base.
- [x] Add the source-inspected fleet/wake, first-chat, browser-unseen, and
  auto-speak drift inventories and conservative residue budgets.
- [x] Project the remaining work against 22,172 lines / 712 methods and prove
  at least 76 lines / 12 methods of ratchet margin.
- [x] Preserve TASK-3070.3 as video-only; add serial atomic drift children and
  move final ratchet closeout to TASK-3070.11.
- [x] Run only `Tests/Architecture/test_console_wave6_inventory.py` and record
  the exact result before writing video production code.

### Task 1: Freeze the focused baseline and controller contract

**Files:**
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`
- Create: `Tests/Chat/test_console_video_controller.py`

- [x] Add no-mount tests that construct `ConsoleVideoController` with plain fakes and pin all eight defaults, explicit video-store override/app-store fallback, video-card storage identity, shared cancellation-event identity, shielded completion, persistence-before-sync, and drain behavior.
- [x] Add direct MessageController regenerate-routing evidence and keep the existing play/save assertions unchanged.
- [x] Run the new nodes before production changes and record the expected RED caused only by the missing controller/direct regenerate seam (4 failed).
- [x] Keep the amended Wave 6 AST manifest stable while implementing video; it
  names the exact 31 moves, two delegates, eight compatibility fields,
  zero-DOM rule, and five-line delegate ceiling.

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_console_wave6_inventory.py \
  Tests/Chat/test_console_video_controller.py \
  Tests/Chat/test_console_video_actions.py
```

Expected: the new controller/routing nodes fail before production movement; existing characterization nodes remain green.

### Task 2: Move video ownership into one minimal controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/video.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [x] Define `ConsoleVideoController` with exactly the controller-owned defaults:
  `_console_videogen_inflight`, `_console_videogen_cancels`,
  `_console_video_store`, `_pending_video_artifacts`,
  `_pending_video_artifacts_closed`,
  `_pending_video_operation_cancels`,
  `_pending_video_active_operations`, and
  `_pending_video_deferred_closes`.
- [x] Name only the dependencies the moved methods actually use: stable
  `app_instance`; chat-store access; default session settings; transcript
  append/sync; composer read/clear; screen-result presentation; and OS-player
  presentation.
- [x] Move the reviewed 31 method definitions without behavior changes. Replace
  internal `ChatScreen.<video method>` references with
  `ConsoleVideoController.<video method>`; keep screen presentation calls
  behind the named callbacks.
- [x] Construct `screen._video` in `build_console_controllers` immediately
  after `screen._image`, using late-bound lambdas for every screen/sibling
  dependency.
- [x] Add the eight `_ControllerState("_video", ...)` class assignments and
  remove their old `ChatScreen.__init__` assignments. Setters must write
  through and pre-build reads/writes must remain fail-loud.
- [x] Replace the two registry-bound screen methods with at-most-five-line
  delegates to `screen._video`; rewire all other screen lifecycle/card/sync
  callers directly to the controller.

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_console_wave6_inventory.py \
  Tests/Chat/test_console_video_controller.py
```

Expected: all selected nodes pass; AST proves the controller owns all 31 methods, has no `query_one`, and both delegates satisfy the physical-span ceiling.

### Task 3: Wire message actions directly to the video owner

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `Tests/Chat/test_console_video_actions.py`

- [x] Add the missing named `regenerate_console_video_message` constructor
  dependency alongside existing play/save video dependencies.
- [x] Route MessageController play/save/regenerate callables directly to
  `screen._video` in `wiring.py`; no call detours through a screen method.
- [x] Verify play/save/regenerate pass the durable persisted storage ID and keep
  the existing action result behavior.

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_video_actions.py
```

Expected: PASS.

### Task 4: Retarget private tests without changing assertions

**Files:**
- Modify: `Tests/Chat/test_console_generate_video.py`
- Modify: `Tests/Chat/test_console_video_capacity.py`
- Modify: `Tests/Chat/test_console_video_message.py`
- Modify only if direct private access requires it:
  `Tests/Widgets/test_console_video_card.py`,
  `Tests/Widgets/test_console_video_card_rows.py`,
  `Tests/Widgets/test_console_video_preview.py`

- [x] Retarget direct calls/patches from `ChatScreen` to the defining
  `ConsoleVideoController` module/class while preserving each assertion.
- [x] Keep mounted screen tests unchanged unless their only dependency is the
  moved private owner; an assertion or user-visible copy change is a finding,
  not accommodation.
- [x] Mutation-check one delegate, the shared cancellation handoff,
  persistence-before-cleanup, and MessageController regenerate wiring; restore
  every mutation.

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_actions.py \
  Tests/Chat/test_console_video_capacity.py \
  Tests/Chat/test_console_video_message.py \
  Tests/Widgets/test_console_video_card.py \
  Tests/Widgets/test_console_video_card_rows.py \
  Tests/Widgets/test_console_video_preview.py \
  Tests/Video_Generation/test_video_store.py \
  Tests/Video_Generation/test_video_formats.py
```

Expected: PASS. This is the complete modified-functionality test boundary; do not run the repository-wide suite per owner direction.

### Task 5: Static verification and task closeout

**Files:**
- Modify: `backlog/tasks/task-3070.3 - Extract-Console-video-controller.md`
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Modify: this plan

- [x] Run Ruff check on every modified Python file and format the two new files;
  preserve the legacy `chat_screen.py` formatter/lint baseline rather than
  rewriting unrelated code.
- [x] Compile only the modified production modules into a validated temporary
  cache root and remove that root after inspection.
- [x] Run `git diff --check`, regenerate the diagnostic owner manifest after
  reviewing the 15 moved metadata-only calls, run the persistent-diagnostic
  non-write verifier,
  and focused privacy/artifact scans over the branch diff.
- [x] Record exact focused test counts, mutation evidence, modified files,
  review findings, and the owner-directed full-suite exclusion.
- [x] Check every acceptance criterion and set TASK-3070.3 to Done only after
  review and verification are complete.

ADR required: no

ADR path: N/A

Reason: this is a behavior-preserving implementation of the approved Wave 6
boundary in `2026-08-13-console-decomposition-wave6-design.md` and
`DESIGN.md` section 7; it changes no storage, provider, security, runtime, or
cross-module policy.
