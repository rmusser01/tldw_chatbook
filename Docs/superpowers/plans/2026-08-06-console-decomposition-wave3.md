# Console Decomposition — Wave 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the message cluster together with the visual surfaces that reflect its state, plus the prompt cluster — then lower the size ratchet so the gain is locked in rather than re-consumed.

**Architecture:** Per `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` and the four controllers + two region widgets already in `UI/Console_Modules/`. Message and its surfaces go together because the control bar and transcript reflect message/generation state; splitting them across waves would force a cross-wave seam.

**Tech Stack:** Python ≥3.11, Textual, pytest.

**Measured on dev at `22c08f958` (post-wave-2) — re-locate by anchor, never by line number:**
- `chat_screen.py` 20,964 lines, `ChatScreen` 612 methods.
- **message: 1,577 lines / 35 methods** — the largest remaining cluster. `handle_console_message_action` is 294 lines.
- **prompt: 853 / 18** — contains `_open_console_prompts_modal` (397), the biggest non-`__init__` method on the class.
- **transcript: 607 / 14** — the visual surface coupled to message.
- The monsters now: `__init__` (603), `_open_console_prompts_modal` (397), `on_button_pressed` (381), `on_key` (309), `compose_content` (309).
- **`__init__` has grown to 603 lines, largely from wiring four controllers.** That is a real cost of this pattern and is wave 4's problem, not this wave's — do not attack it here, but do not make it worse: keep new wiring blocks tight and grouped.

## Global Constraints

The spec's six migration rules bind every task (see its "Migration safety" section). Plus, hard-won across waves 1–2:

**THE CONTROLLER BINDING RULE** — canonical example is `ConsoleDictationController.__init__`'s docstring:
- Framework services (`run_worker`, `set_timer`, `post_message`, `push_screen`, `call_after_refresh`, `is_mounted`) — live-read from the screen via `@property`.
- App-level dependencies — **named keyword-only callable parameters**, wired at the construction site as late-binding lambdas (`x_accessor=lambda: self._x()`), never bound methods, never constructor snapshots.
- `app_instance` — snapshot only, justified in the docstring.
- Workers — preserve pre-move group names; never invent a flat cluster group.
- **Controller-to-controller traffic goes through named callables the screen wires**, resolved at CALL time (a construction-time capture leaves whichever controller is built second as `None`). Never a back-door through screen attributes.
- **A write-only proxy getter raises `RuntimeError`, not `AttributeError`** — `hasattr()`/`getattr(_, _, default)` swallow exactly that exception, so a defensive read would silently take the default forever.
- **Zero `query_one`/DOM access in a controller.** All four existing controllers have zero. Match them.
- **A region widget never stores a child-widget instance the screen may replace** — pass a zero-arg builder (DESIGN.md §7).

**Process rules, each paid for:**
- **`git push` after EVERY commit.** An entire unpushed wave was destroyed by the `/private/tmp` cleaner. Work lives in `.worktrees/`, never `/private/tmp`; the interpreter is `<repo>/.venv/bin/python`.
- **Characterise BEFORE extracting**, always. Two wave-2 tasks did it retroactively and a reviewer ruled that count-level evidence only — it cannot detect an assertion-weakening retarget when the test files change in the same commit.
- **Commit the extraction the moment it exists**, before any sweep.
- Tests FOREGROUND with Bash `timeout: 600000`; the dispatch names the verification files AND requires running every file the task edits.
- Gate before each task: **open PRs touching `chat_screen.py`** (`gh pr list` + per-PR file check), never branch last-commit age. Branch age is what let #1350's 2,285 lines blindside wave 2 mid-flight.
- Rebase onto `origin/dev` before each task; CSS-bundle conflicts resolve by `git checkout --theirs` then regenerating via `.venv/bin/python tldw_chatbook/css/build_css.py`.
- **A scripted rename across this boundary needs an ambiguity check first.** Several methods exist in BOTH the screen (real implementation) and a controller (same-named accessor property); a blind rename turns correct prose wrong. Check for exactly one definition site before rewriting any reference.

**Three defect classes that have already bitten this programme:**
1. **Dead bodies** — a moved method left behind with zero callers (cost a fix round in wave 1, another in wave 2). Audit every moved method: gone, or an intentional ≤14-line delegation WITH real callers.
2. **Silent drops outside diff markers** — an orphaned `import threading` (runtime NameError) and a config symbol needing a dual import, neither visible in any hunk. Run pyflakes on `chat_screen.py` and the new module; compare to the pre-move baseline.
3. **Single-module test helpers** — a helper pinning a config symbol on `chat_screen_module` stops governing the moved path once a controller reads its own copy. Three instances found so far. Grep helpers for every symbol you move.

## File Structure

- `tldw_chatbook/UI/Console_Modules/message.py` (new) — `ConsoleMessageController`
- `tldw_chatbook/UI/Console_Modules/transcript.py` (new) — the transcript region widget
- `tldw_chatbook/UI/Console_Modules/prompts.py` (new) — `ConsolePromptsController`
- `tldw_chatbook/UI/Screens/chat_screen.py` — delegations shrink
- `Tests/Architecture/test_screen_size_ratchet.py` — budget lowered in Task 4
- Tests mirror each under `Tests/UI/`

---

### Task 1: The message controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_message_controller.py` (new)

**Interfaces:**
- Consumes: the binding rule; seams to the session and workspace controllers (messages belong to a session, which lives in workspace context) — named callables in both directions, wired by the screen.
- Produces: `ConsoleMessageController` owning message state and lifecycle.

- [ ] **Step 1: Map the cluster.** Every `*message*` method (35, ~1,577 lines) plus everything reading message state; per-method verdict (moves / ≤14-line delegation / stays with reason). `handle_console_message_action` (294 lines) is the risk centre — it dispatches user actions across regenerate/branch/delete/copy and will have the largest stays-list. Map message↔session and message↔transcript traffic in both directions. If the boundary has no clean answer, **report BLOCKED with the map**.
- [ ] **Step 2: Characterise first.** Grep `Tests/` for message coverage; run what exists against unmodified code and record counts. Send/receive and at least one `handle_console_message_action` branch need tests driving REAL interactions before the move, asserting the persisted result (message rows in the store), not widget state. Commit separately, push, green pre-move.
- [ ] **Step 3: Extract** per the binding rule.
- [ ] **Step 4: Commit, push, then verify** in one foreground call (timeout 600000): the new test file + `Tests/UI/test_console_native_chat_flow.py` + `Tests/UI/test_console_shell_regions.py` + `Tests/Architecture/test_screen_size_ratchet.py`; then a second call for every other file edited.
- [ ] **Step 5: Report** the map, the `handle_console_message_action` decision, the seams, the dead-body audit, pyflakes before/after, the test-helper grep.

Commit: `refactor(console): message controller (wave 3)`

---

### Task 2: The transcript region widget

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/css/features/_console.tcss` if any CSS moves
- Test: `Tests/UI/test_console_transcript_region.py` (new)

**Interfaces:**
- Consumes: `frame_console_region` from `Console_Modules/frame.py`; `ConsoleMessageController` (Task 1) via named callables.
- Produces: the transcript region widget, composing the block `compose_content` currently builds inline around `#console-main-column`'s transcript area — **ids verbatim, same nesting**.

This is a region, not a controller: it owns pixels. Follow `left_rail.py`/`right_rail.py`, not the controllers.

- [ ] **Step 1: Map the block** in `compose_content` (309 lines) — every id, widget class, and `self.*` reference. The transcript cluster is 607 lines / 14 methods; decide per method whether it is region-owned (composition, its own event handlers) or controller-owned (state that survives a recompose).
- [ ] **Step 2: Characterise first** — a real `pilot` interaction against the mounted transcript, asserting persisted state; committed, pushed, green pre-move. The geometry baseline (`Tests/UI/test_console_shell_regions.py`) must stay byte-identical throughout.
- [ ] **Step 3: Extract**, ids verbatim, `@on` handlers for what stays inside, messages upward for cross-region effects, zero-arg builders for any child the screen replaces at runtime.
- [ ] **Step 4: Commit, push, verify** (same two-call shape as Task 1).
- [ ] **Step 5: Report** including whether any CSS moved and the bundle regeneration.

Commit: `refactor(console): transcript region widget (wave 3)`

---

### Task 3: The prompts controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/prompts.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_prompts_controller.py` (new)

**Interfaces:**
- Produces: `ConsolePromptsController` owning the prompt cluster (853 lines / 18 methods), including `_open_console_prompts_modal` (397 lines — the largest non-`__init__` method on the class).

`push_screen` for the modal is a framework service, not a reason to stay. The modal's own widget is not in scope; only the screen-side orchestration moves.

- [ ] **Step 1: Map the cluster**, per-method verdicts, and the prompt↔composer/message traffic.
- [ ] **Step 2: Characterise first** — driving the real modal-open path and asserting the persisted selection; committed, pushed, green pre-move.
- [ ] **Step 3: Extract** per the binding rule.
- [ ] **Step 4: Commit, push, verify** (two calls).
- [ ] **Step 5: Report** as above.

Commit: `refactor(console): prompts controller (wave 3)`

---

### Task 4: Close the wave and lower the ratchet

**Files:**
- Modify: `Tests/Architecture/test_screen_size_ratchet.py`, `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`, `DESIGN.md` if the wave established a new rule

- [ ] **Step 1: Measure** the new `chat_screen.py` line count and `ChatScreen` method count with `ast`.
- [ ] **Step 2: Lower the ratchet** in `_BUDGETS` to exactly those numbers. This is the step the whole wave exists to earn — the second ratchet test fails if you skip it, by design.
- [ ] **Step 3: Annotate the spec** — mark message/transcript/prompt done, and update the Progress section's numbers honestly (both the raw figures and the fact that concurrent growth continues).
- [ ] **Step 4: `DESIGN.md`** — add any rule this wave established that waves 1–2 did not already cover; if none, say so in the report rather than inventing one.
- [ ] **Step 5: Serial gate**, at most two foreground calls (timeout 600000), then commit and push.

Commit: `refactor(console): wave 3 closed — ratchet lowered, honest numbers`

---

## Wave 3 exit criteria

- Message, transcript, and prompt clusters live in `UI/Console_Modules/`; the screen holds delegations.
- Every controller follows the binding rule; the transcript region follows the region rules; zero DOM access in controllers.
- The geometry baseline is byte-identical and green.
- No behaviour change; every pre-existing DOM-driven test passes unchanged.
- **The ratchet is lowered to the new measurement** — the wave's gain is locked in.

## Not in wave 3 (deliberate)

`__init__` (603 lines, much of it controller wiring — a real cost of this pattern and wave 4's problem), `on_button_pressed` (381), `on_key` (309), and the agent/character/image/skill/citation clusters. Settings and Library screens remain their own future efforts under the same spec.
