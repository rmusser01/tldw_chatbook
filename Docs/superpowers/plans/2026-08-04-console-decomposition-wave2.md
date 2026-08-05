# Console Decomposition — Wave 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the three largest non-visual clusters — hands-free, workspace, session — as controllers following the binding rule wave 1 settled, retiring the dictation controller's one disclosed exception along the way.

**Architecture:** Per `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` and the wave-1 precedents in `UI/Console_Modules/`. Controllers only, this wave: the remaining visual surfaces (main column, mode bar, control bar, transcript) are coupled to the message cluster and go together in wave 3.

**Tech Stack:** Python ≥3.11, Textual, pytest.

**Why this scope:** wave 1 (2 regions + 1 controller) cost a full day with multi-round fix loops on the hottest file in the repo. Wave 2 is pure controllers — the pattern is now settled and templated — sized to land before it rots. Measured on post-wave-1 dev (`2fbd5571e`): `chat_screen.py` is 21,238 lines / 617 `ChatScreen` methods.

## Global Constraints

The spec's six migration rules bind every task (see the spec's "Migration safety" — by reference, they are: one cluster per change; ids verbatim; geometry assertions before/after; CSS with its region via `tldw_chatbook/css/features/` + bundle regeneration; no behaviour change rides an extraction; characterise before moving uncovered behaviour).

House rules, sharpened by wave 1's cost:

- **EXECUTION GATE (owner ruling, standing):** before Task 1 is dispatched, run the churn gate — `git log origin/dev --oneline --since="24 hours ago" -- tldw_chatbook/UI/Screens/chat_screen.py` plus a scan of open `origin/*console*` branches — and get the owner's explicit go.
- **The wait-trap rule (burned five implementers in wave 1):** every pytest call is FOREGROUND with Bash `timeout: 600000`. Dispatches name the exact 1-2 verification files; extra runs are forbidden. If a call auto-backgrounds, the timeout parameter was wrong — fix it, never wait.
- **Commit the extraction the moment it exists**, before any sweep.
- Rebase onto `origin/dev` before each task; CSS-bundle conflicts resolve by `git checkout --theirs` the bundle then regenerating via `/private/tmp/tldw-venv/bin/python tldw_chatbook/css/build_css.py`; re-run the geometry baseline (`Tests/UI/test_console_shell_regions.py`, 30 tests) after every rebase.
- Never `git stash`; never `git checkout --` on uncommitted work; line numbers below were measured at dev `2fbd5571e` — re-locate by anchor.
- Existing DOM-driven tests pass unchanged; tests reaching moved private methods get mechanically retargeted with assertions byte-for-byte — and any test fake that intercepts `query_one` must assert the selector/type it receives (wave 1's finding 5).

**THE CONTROLLER BINDING RULE (wave 1's hardest-won lesson — two fix rounds; the canonical example is `ConsoleDictationController.__init__`'s docstring):**
- Framework services (`run_worker`, `set_timer`, `post_message`, `is_mounted`, …) — live-read from the screen via `@property`.
- App-level dependencies — **named keyword-only callable parameters**, passed as late-binding lambdas at the construction site (`composer_accessor=lambda: self._console_composer_or_none()`), never bound methods, never constructor snapshots.
- `app_instance` — snapshot only, justified in the docstring (identity-stable).
- Workers under a group named for the controller (`group="console-<name>"`).
- A controller never stores a widget instance the screen may replace — builders or fresh queries (DESIGN.md §7).

## Verified facts (measured at dev `2fbd5571e` — re-verify anchors, not conclusions)

- Post-merge truth: 21,238 lines / 617 methods. `compose_content` is 13479–13783 (305 lines); the frame shim has 7 call sites there (wave-3 removal).
- **Hands-free cluster: 28 methods, 730 lines.** Largest: `_install_console_hands_free_store_tap` (77), `_console_hands_free_marshal` (62), `_repaint_console_hands_free_chip` (51), `_console_hands_free_try_claim_reply` (51), `_console_hands_free_request_stop_and_send` (45), `_deliver_console_hands_free_capture_ended` (40), `_enter_console_hands_free_loop` (36). Init state: `_console_hands_free_store_tap_installed`, `_console_hands_free_vad_degraded`, plus whatever `_console_hands_free` holds — map it.
- **The dictation controller reaches back into hands-free today** via disclosed named screen properties (`_console_hands_free`, `_enter_console_hands_free_loop`, `_console_undo_histories`, `_console_visible_draft_session_id` — see `dictation.py`'s module docstring). Task 1 must land a clean seam between the two controllers and DELETE the "disclosed temporary exception" language from both docstrings — that exception exists only because hands-free was not yet extracted.
- **Workspace cluster: 45 methods, 1,552 lines.** Largest: `_resume_console_workspace_conversation` (205), `_current_console_workspace_context` (83), `_sync_console_workspace_context` (78), `_open_console_workspace_scope_picker` (75), `_open_console_workspace_switcher` (74). Init state: `_console_workspace_conversation_query` / `_search_token` / `_search_error`. Note `on_console_workspace_conversation_search_changed` (64) is an EVENT HANDLER — it stays on the screen as a one-line delegation.
- **Session cluster: 37 methods, 1,212 lines.** Largest: `_start_character_console_session` (215), `_console_session_from_state` (91), `_swap_console_session_character` (77), `_sync_console_session_draft` (71), `_promote_console_temporary_session` (64).
- The `_repaint_console_hands_free_chip` name says hands-free touches a CHIP — a visual. A controller may not own pixels: the repaint method either delegates through an accessor the screen provides (acceptable: the chip belongs to the composer bar, an existing widget) or is evidence the cluster needs a region seam — Task 1's map decides, honestly.
- Wave-1 templates: `UI/Console_Modules/dictation.py` (controller), `left_rail.py` / `right_rail.py` (regions, zero-arg builders), `frame.py`.

## File Structure

- `tldw_chatbook/UI/Console_Modules/hands_free.py` (new) — `ConsoleHandsFreeController`
- `tldw_chatbook/UI/Console_Modules/workspace.py` (new) — `ConsoleWorkspaceController`
- `tldw_chatbook/UI/Console_Modules/session.py` (new) — `ConsoleSessionController`
- `tldw_chatbook/UI/Screens/chat_screen.py` — delegations shrink; `UI/Console_Modules/dictation.py` — exception language retired (Task 1)
- Tests: `Tests/UI/test_console_hands_free_controller.py`, `..._workspace_controller.py`, `..._session_controller.py` (new characterisation + unit files)

---

### Task 1: The hands-free controller — and the end of dictation's exception

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/hands_free.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Console_Modules/dictation.py`
- Test: `Tests/UI/test_console_hands_free_controller.py` (new)

**Interfaces:**
- Consumes: the binding rule; `ConsoleDictationController`'s existing reach-back list (its module docstring names each one).
- Produces: `ConsoleHandsFreeController` owning the hands-free state and lifecycle; `ConsoleDictationController` re-pointed at it through named late-binding callables instead of screen properties; the "disclosed temporary exception" language deleted from both docstrings.

- [ ] **Step 1: Map the cluster.** All 28 `*hands_free*` methods plus every method reading `self._console_hands_free*` state; per-method verdict (moves / stays as delegation / stays because it touches non-hands-free state). Resolve the chip question from the Verified facts honestly: if `_repaint_console_hands_free_chip` writes widget state, it delegates through a screen-provided accessor and the map says so. Map the dictation↔hands-free traffic BOTH directions.
- [ ] **Step 2: Characterise first.** Grep `Tests/` for hands-free coverage; run what exists against unmodified code and record counts. The hands-free loop's entry/exit (`_enter_console_hands_free_loop`, capture-ended delivery) must have a test driving the REAL path before the move — add one if missing, committed separately, passing pre-move.
- [ ] **Step 3: Extract** per the binding rule. The dictation controller's four reach-backs become named callable parameters wired to the new controller (or to the screen where the state genuinely stays); both docstrings lose the exception language; the screen builds both controllers in `__init__` and wires them with late-binding lambdas.
- [ ] **Step 4: Commit immediately, then verify** — ONE call: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_console_hands_free_controller.py Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_shell_regions.py -p no:randomly` (timeout 600000). Known: the dictation flake (`test_the_transcribing_indication_reverts_on_a_mid_capture_stop`) is order-dependent and pre-existing — report the honest count.
- [ ] **Step 5: Report** the map, the seam design, both docstring diffs, counts.

Commit: `refactor(console): hands-free controller; dictation's exception retired (wave 2)`

---

### Task 2: The workspace controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_workspace_controller.py` (new)

**Interfaces:**
- Consumes: the binding rule; `ConsoleHandsFreeController` seam if the map shows traffic (state it either way).
- Produces: `ConsoleWorkspaceController` owning workspace context, conversation search state (`_console_workspace_conversation_query`/`_search_token`/`_search_error`), and the resume/switcher/scope-picker flows.

Steps mirror Task 1: map (45 methods — the largest cluster; `_resume_console_workspace_conversation` at 205 lines is the risk centre: it touches sessions, drafts, and the transcript, so expect a stays-list, and a BLOCKED-with-map report beats forcing it); characterise the resume flow and the search debounce through real interactions BEFORE the move (search-token/error state is exactly where a snapshot-vs-live binding bug would hide); extract; commit immediately; verify with ONE call: `Tests/UI/test_console_workspace_controller.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_shell_regions.py -p no:randomly` (timeout 600000); report.

Modal-opening methods (`_open_console_workspace_switcher`, `_open_console_workspace_scope_picker`) push screens — that is a framework service (`push_screen` live-read via property), not a reason to stay.

Commit: `refactor(console): workspace controller (wave 2)`

---

### Task 3: The session controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_session_controller.py` (new)

**Interfaces:**
- Consumes: the binding rule; the workspace controller's seam (sessions live inside workspace context — the two controllers may need a named callable between them; design it deliberately, never a back-door through the screen).
- Produces: `ConsoleSessionController` owning session lifecycle: start/activate/swap/promote/rename, draft sync.

Steps mirror Task 1: map (37 methods; `_start_character_console_session` at 215 lines crosses characters, personas, and drafts — expect the largest stays-list of the wave); characterise session activation and the temporary-session promotion through real flows pre-move (promotion writes durable rows — assert the DB, not the widget); extract; commit immediately; verify with ONE call: `Tests/UI/test_console_session_controller.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_shell_regions.py -p no:randomly` (timeout 600000); report.

Commit: `refactor(console): session controller (wave 2)`

---

### Task 4: Close the wave

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`, `DESIGN.md`
- Test: the serial gate

- [ ] Measure and record honest numbers (old → new, both baselines, no spin — Task 6 of wave 1 is the template).
- [ ] Annotate the spec's chat table rows (workspace, session; add a hands-free row — the plan-time table never listed it because the cluster hid inside dictation's orbit).
- [ ] `DESIGN.md` §7: add the controller-to-controller seam rule this wave establishes (named callables between controllers, wired by the screen at construction; never back-doors through screen attributes) — if that is in fact how Tasks 1-3 landed; write what actually shipped.
- [ ] Serial gate, TWO calls max (timeout 600000 each): `Tests/UI/test_console_hands_free_controller.py Tests/UI/test_console_workspace_controller.py Tests/UI/test_console_session_controller.py Tests/UI/test_console_shell_regions.py -p no:randomly`, then `Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py -p no:randomly`.
- [ ] Commit: `refactor(console): wave 2 closed — three controllers, honest numbers`

---

## Wave 2 exit criteria

- Hands-free, workspace, and session state each live in one controller; the screen holds delegations; the dictation controller's exception language is gone because the exception is gone.
- Every controller follows the binding rule; controller-to-controller traffic goes through named callables wired by the screen.
- The geometry baseline is byte-identical to its wave-1 commits and green.
- No behaviour change anywhere; every pre-existing DOM-driven test passes unchanged.
- The spec and DESIGN.md record what actually shipped, with honest numbers.

## Not in wave 2 (deliberate)

The message cluster (1,548 lines / 34 methods, plus `handle_console_message_action` at 294), the transcript/main-column/mode-bar/control-bar surfaces, and the agent/character/image clusters — wave 3, where the message controller and its visual surfaces land together. The monsters (`_open_console_prompts_modal` 397, `on_button_pressed` 369, `__init__` 367, `on_key` 242) shrink as their callers move; direct assault on the remainder is wave 3+. Jump mode and border key hints stay separate work.
