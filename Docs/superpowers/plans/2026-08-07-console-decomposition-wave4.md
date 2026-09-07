# Console Decomposition — Wave 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the three costs the previous three waves created or left behind — the controller wiring that swelled `__init__`, the button dispatcher whose branches belong to controllers that already exist, and the agent cluster — then lower the size ratchet so the gain is locked in.

**Architecture:** Per `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` and the nine controllers/regions already in `UI/Console_Modules/`. Wave 4 is different in kind from waves 1–3: two of its three extractions move code that is *already* owned elsewhere but still physically lives on the screen. Only Task 3 is a conventional cluster extraction.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest.

**Measured on dev at `391b7bf69` (post-wave-3) — re-locate by anchor, never by line number:**
- `chat_screen.py` **18,909 lines**, `ChatScreen` **598 methods**.
- **`__init__` is now 782 lines** (it was 603 when wave 3 was planned). **411 of those lines are six controller-construction statements**: `_workspace` (90 lines / ~25 kwargs), `_message` (83 / ~20), `_session` (77 / ~23), `_prompts` (73 / ~19), `_dictation` (62 / ~13), `_hands_free` (26 / ~11). Everything else in `__init__` is ~250 short attribute assignments.
- `on_button_pressed` **381 lines, 19 top-level branches**. The five largest — `console-workspace-conversation-` (81), `console-conversation-star-` (65), `console-close-session-tab-` (39), `console-workspace-conversations-toggle` (35), `console-dictation` (31) — belong to controllers that **already exist**.
- `on_key` **309 lines, 24 top-level branches**, almost all composer text-editing (backspace/delete/arrows/home/end/ctrl+w/ctrl+u/ctrl+z/ctrl+y/enter). **Not in this wave** — see "Not in wave 4".
- `compose_content` **312 lines**. Not in this wave.
- **agent: 617 lines / 15 methods** — the largest remaining named cluster. Largest members: `_console_agent_section_lines` (114), `_sync_console_agent_section` (80), `_console_agent_full_log_available` (61), `_apply_fleet_agent_section_auto_open` (56), `_inject_resume_agent_markers` (47).

## Global Constraints

The spec's six migration rules bind every task. Plus, hard-won across waves 1–3:

**THE CONTROLLER BINDING RULE** — canonical example is `ConsoleDictationController.__init__`'s docstring in `Console_Modules/dictation.py`:
- Framework services (`run_worker`, `set_timer`, `post_message`, `push_screen`, `call_after_refresh`, `is_mounted`) — live-read from the screen via `@property`.
- App-level dependencies — **named keyword-only callable parameters**, wired at the construction site as late-binding lambdas (`x_accessor=lambda: self._x()`), never bound methods, never constructor snapshots.
- `app_instance` — snapshot only, justified in the docstring.
- Workers — preserve pre-move group names.
- **Controller-to-controller traffic goes through named callables the screen wires**, resolved at CALL time.
- **Zero `query_one`/DOM access in a controller.** Regions are the opposite: DOM is theirs.
- **A proxy property standing in for a baseline plain attribute must be read-WRITE, and its setter must write THROUGH to the controller** (DESIGN.md §7). Wave 3 shipped one getter-only proxy and turned **41 tests red in a file the branch never touched**. A genuinely *write-only* proxy getter raises `RuntimeError`, not `AttributeError`.
- `action_*` and `@on` handlers that Textual resolves **by name on the Screen** stay on the Screen. Their bodies may move; their definitions may not.

**Process rules, each paid for:**
- **`git push` after EVERY commit.** An entire unpushed wave was destroyed by the `/private/tmp` cleaner. Work lives in `.worktrees/wave4`; the interpreter is `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.
- **Never `git stash`** — the stash stack is shared across 100+ worktrees.
- **Never `cd` to the parent repo.** Another session works there with hundreds of uncommitted files; a wave-3 implementer's compound `cd <root> && … git commit` swept them into a commit on their branch.
- **Characterise BEFORE extracting**, always. Retroactive characterisation is count-level evidence only.
- **Commit the extraction the moment it exists**, before any sweep.
- Tests FOREGROUND with Bash `timeout: 600000`. A backgrounded pytest strands the implementer — it has done so six times on this programme.
- **`pytest` reporting "no tests ran" is a FAILED gate, not a pass.** Verify each path exists before trusting a green line. Two test paths were mistyped during wave 3 and produced a silent no-op run.
- **Scope verification to every file REFERENCING what you move, not the files you edited.** That error let a wave-3 task ship 42 failing tests past a green run. A plain diff CANNOT see moved-but-delegated methods — intersect `defs(new module)` with `defs(baseline screen)` instead.

**Three defect classes that have bitten every wave:**
1. **Dead bodies** — every moved method is *gone*, or a ≤14-line pure-forwarding delegation **with a real caller**. Verify with `ast`. An `ast`-lineno delete also misses decorator lines above `@staticmethod`/`@classmethod`, and can take the first line of the *next* declaration's comment block — sweep for both.
2. **Silent drops outside the diff hunks** — an orphaned import causing a runtime `NameError`; a config symbol needing a dual import. pyflakes both files, compare to baseline.
3. **Hand-built screen fixtures.** **Every wave has shipped exactly one to dev** — wave 1 via call sites, wave 2 via `MagicMock(spec=ChatScreen)` (spec reads the CLASS, so `__init__`-wired attributes are invisible), wave 3 via six bare `ChatScreen.__new__()` fixtures. Assume you will too. Sweep `grep -rl 'ChatScreen.__new__\|spec=ChatScreen' Tests/`. `Tests/UI/console_controller_stubs.py` exists — reuse or extend it; note its `NO_APP` sentinel, which exists because an inferred `app_instance=None` is a silent-default hole.

## File Structure

- `tldw_chatbook/UI/Console_Modules/wiring.py` (new) — `build_console_controllers(screen)`, the six constructions verbatim.
- `tldw_chatbook/UI/Console_Modules/agent.py` (new) — `ConsoleAgentController`.
- `tldw_chatbook/UI/Screens/chat_screen.py` — `__init__` and `on_button_pressed` shrink; delegations added.
- `Tests/Architecture/test_screen_size_ratchet.py` — budget lowered in Task 4.
- Tests mirror each new module under `Tests/UI/`.

---

### Task 1: The controller wiring module

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_controller_wiring.py` (new)

**Interfaces:**
- Produces: `build_console_controllers(screen: "ChatScreen") -> None` — constructs all six controllers and assigns them to `screen._workspace`, `screen._session`, `screen._dictation`, `screen._hands_free`, `screen._message`, `screen._prompts`, **in that order**. Import `ChatScreen` only under `TYPE_CHECKING`; the function takes the screen as a parameter, so there is no import cycle.
- Consumes: nothing from later tasks.

**Why this is not the dependency-object anti-pattern.** A code reviewer on PR #1408 proposed collapsing the wiring into a per-controller dependency object; that was declined because it hides the explicit signatures. This task does the opposite: **every named keyword argument stays exactly as written**, character for character. Only the call site moves. The gain is that the whole controller graph becomes readable in one place, which is where construction-order bugs are visible — and construction order is load-bearing here (`app_instance` is set before all six; every cross-controller lambda resolves its sibling at CALL time precisely so the build order cannot matter).

- [ ] **Step 1: Map.** Record each construction's exact line span and kwarg count, and every name the six statements read off `screen` or the enclosing `__init__` scope. **If any construction reads a local variable of `__init__` rather than an attribute of `screen`, that is the one real hazard in this task** — report it explicitly with the variable name, and pass it as an explicit parameter of `build_console_controllers` rather than reconstructing it.
- [ ] **Step 2: Characterise first.** A test that constructs a real `ChatScreen` and asserts all six controller attributes exist, are of the right class, and that a representative named dependency on each **resolves to the same object** as before the move (call the lambda, compare identity — not just "is not None"). Run against unmodified code, record counts, commit separately, push, green pre-move.
- [ ] **Step 3: Extract.** Move the six statements verbatim into `build_console_controllers`, replacing `self` with the `screen` parameter. `__init__` calls `build_console_controllers(self)` at the same point in its sequence — **not earlier and not later**, since ~250 attribute assignments surround it and some are read by the lambdas.
- [ ] **Step 4: Commit, push, then verify** in one foreground call (`timeout: 600000`): the new test file + `Tests/UI/test_console_shell_regions.py` + `Tests/UI/test_ui_responsiveness.py` (its `_console_controller_slots()` helper reads `ChatScreen.__init__`'s source via `ast` to find `Console*Controller(` assignments — **this task moves those assignments out of `__init__`, so that helper WILL break and must be repointed at `wiring.py`**) + `Tests/Architecture/test_screen_size_ratchet.py`. Then a second call for every other file edited.
- [ ] **Step 5: Report** the map, the local-variable hazard finding, before/after line and method counts, and the three audits.

Commit: `refactor(console): move controller wiring out of __init__ (wave 4)`

---

### Task 2: Route the button dispatcher to its owners

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Console_Modules/workspace.py`, `tldw_chatbook/UI/Console_Modules/session.py`, `tldw_chatbook/UI/Console_Modules/dictation.py`
- Test: `Tests/UI/test_console_button_routing.py` (new)

**Interfaces:**
- Consumes: `build_console_controllers` from Task 1 (the wiring may need new named callables if a moved branch reaches something the controller cannot).
- Produces: no new module. Branch bodies become methods on the controller that already owns their state.

**`on_button_pressed` stays on the Screen** — Textual resolves it by name. Only the bodies move. The result should read as a routing table.

- [ ] **Step 1: Map all 19 top-level branches** — button id, line count, and the controller that owns the state it mutates. Per-branch verdict: moves to `<controller>`, or stays with a reason. A branch that mutates state across two controllers **stays on the screen** — that is coordination, and wave 2 set this precedent with `on_console_workspace_conversation_search_changed`. Report the split.
- [ ] **Step 2: Characterise first.** For each branch you intend to move, a `pilot` test pressing the REAL button and asserting the PERSISTED result (a store/DB row, not widget state). The five largest branches are mandatory; smaller ones need coverage only if none exists. Commit separately, push, green pre-move.
- [ ] **Step 3: Move** each branch body to its controller as a named method. The screen's branch becomes a call. Where a moved body used `query_one`, the DOM work stays on the screen behind a named callable — **zero DOM in a controller**, no exceptions.
- [ ] **Step 4: Commit, push, verify** (same two-call shape as Task 1).
- [ ] **Step 5: Report** the per-branch table, the stays-list with reasons, and the three audits.

Commit: `refactor(console): route button dispatch to its owning controllers (wave 4)`

---

### Task 3: The agent controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/agent.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_agent_controller.py` (new)

**Interfaces:**
- Produces: `ConsoleAgentController` owning the agent cluster (617 lines / 15 methods), including `_console_agent_section_lines` (114) and `_sync_console_agent_section` (80).

- [ ] **Step 1: Map the cluster**, per-method verdicts, and the agent↔session/message traffic in both directions. Waves 2 and 3 both found `*name*`-matched methods that were **false positives** (realtime-speech "transcript", image-gen "message") — check each of the 15 against what it actually touches, and report any it excludes.
- [ ] **Step 2: Characterise first** — driving the real agent-section sync and asserting persisted run state; committed, pushed, green pre-move. `Tests/UI/test_console_agent_rail.py` and `Tests/UI/test_console_parallel_runs.py` already exercise parts of this cluster; run them pre-move and record counts.
- [ ] **Step 3: Extract** per the binding rule.
- [ ] **Step 4: Commit, push, verify** (two calls; include `Tests/UI/test_console_agent_rail.py` and `Tests/UI/test_console_parallel_runs.py`).
- [ ] **Step 5: Report** as above.

Commit: `refactor(console): agent controller (wave 4)`

---

### Task 4: Close the wave and lower the ratchet

**Files:**
- Modify: `Tests/Architecture/test_screen_size_ratchet.py`, `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`, `DESIGN.md` if the wave established a new rule

- [ ] **Step 1: Rebase onto `origin/dev` FIRST, then measure.** Wave 3 set its budget before the final rebase and dev's own commits pushed the file 5 lines past it; the budget had to be re-measured twice. Measure `chat_screen.py`'s line count and `ChatScreen`'s method count with `ast` **after** the rebase.
- [ ] **Step 2: Lower the ratchet** in `_BUDGETS` to exactly those numbers. This is the step the whole wave exists to earn; the second ratchet test fails if you skip it, by design.
- [ ] **Step 3: Annotate the spec** — mark wiring/button-routing/agent done, update the Progress section's numbers honestly, and record what the wave did NOT achieve as well as what it did.
- [ ] **Step 4: `DESIGN.md`** — add any rule this wave established that waves 1–3 did not already cover; **if none, say so in the report rather than inventing one.**
- [ ] **Step 5: Serial gate**, at most two foreground calls (`timeout: 600000`), then commit and push.

Commit: `refactor(console): wave 4 closed — ratchet lowered, honest numbers`

---

## Wave 4 exit criteria

- Controller construction lives in `UI/Console_Modules/wiring.py` with every named kwarg preserved verbatim.
- `on_button_pressed` reads as a routing table; every moved branch's body lives with the state it mutates.
- The agent cluster lives in `UI/Console_Modules/agent.py`.
- Zero DOM access in any controller; the geometry baseline is byte-identical and green.
- No behaviour change; every pre-existing DOM-driven test passes unchanged.
- **The ratchet is lowered to the post-rebase measurement.**

## Not in wave 4 (deliberate)

- **`on_key` (309 lines, 24 branches).** Almost all of it is composer text-editing — backspace, delete, arrows, home/end, ctrl+w/u/z/y, enter. That is a **composer keymap**, and `ConsoleComposerBar` already exists at `tldw_chatbook/Widgets/Console/console_composer_bar.py`. Routing 24 five-line branches into delegations risks the wave-3 transcript outcome (net zero), so it wants its own design pass deciding whether the keymap belongs to the composer widget rather than a controller. Wave 5.
- **`compose_content` (312).** More region extraction; wave 3 showed a region can net zero lines. Wave 5, and it should be planned together with task-2767 (the transcript's two access idioms).
- The image (638/21), character (412/12), rail (515/27), and skill (339/16) clusters.
- Settings and Library screens remain their own future efforts under the same spec.
