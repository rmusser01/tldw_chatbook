# Console Decomposition — Wave 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the composer's keymap with the widget that already implements every operation it maps to, extract the two largest remaining feature clusters, and lower the size ratchet.

**Architecture:** Per `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` and the eleven controllers/regions in `UI/Console_Modules/`. Task 1 is unusual and is the wave's most interesting change: it moves code **into an existing widget**, not into a new module, because the ownership is already there and only the mapping is misplaced.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest.

**Measured on dev at `fdaf1379f` (post-wave-4 + follow-ups) — re-locate by anchor, never by line number:**
- `chat_screen.py` **17,727 lines**, `ChatScreen` **593 methods**. Ratchet budget 17727/593.
- Largest methods: `compose_content` (314), `on_key` (309), `__init__` (285 — down from 782, wave 4 did its job), `on_button_pressed` (206).
- **`on_key`'s 24 top-level branches, classified by what they touch: 18 touch ONLY the composer** (~120 lines: arrows, home/end, backspace/delete, ctrl+w/u/z/y, shift+enter, select-all). The other 6 reach further: `enter` (72 lines, composer + `query_one` + store), `pageup/pagedown` (17), the hands-free branch (23), the realtime branch (8), the command-popup branch (16), and the setup-modal guard (4).
- **image: 638 lines / 21 methods.** Largest: `_console_command_generate_image` (162), `_paste_console_clipboard_image` (76), `_console_generate_image_llm_context_options` (68), `_extend_specs_with_remote_images` (59).
- **character: 412 / 12.** Largest: `_refresh_active_character_avatar_if_scope_changed` (108), `_build_character_avatar_widget` (74), `_apply_console_character_choice_async` (65), `_render_character_avatar_into_section` (33).

**The finding that shapes Task 1.** `tldw_chatbook/Widgets/Console/console_composer_bar.py` is 4,231 lines and **already implements every operation those 18 branches call**: `insert_text`, `delete_left`, `delete_right`, `delete_word_left`, `move_cursor_left/right/up/down/home/end`, `undo`, `redo`, `select_all_draft`. It defines **no `on_key` and no `BINDINGS`**. So the screen holds a key→method table for methods the widget owns. That is not a switch worth relocating for its own sake — it is a mapping living apart from the thing it maps to.

## Global Constraints

The spec's six migration rules bind every task. Plus, hard-won across waves 1–4:

**THE CONTROLLER BINDING RULE** — canonical example is `ConsoleDictationController.__init__`'s docstring in `Console_Modules/dictation.py`:
- Framework services (`run_worker`, `set_timer`, `post_message`, `push_screen`, `call_after_refresh`, `is_mounted`) — live-read from the screen via `@property`.
- App-level dependencies — **named keyword-only callable parameters**, wired as late-binding lambdas **in `Console_Modules/wiring.py`** (wave 4 moved construction there; a new controller is added to `wiring.py`, never back into `__init__`).
- `app_instance` — snapshot only, justified in the docstring.
- Workers — preserve pre-move group names. `Tests/UI/test_chat_screen_worker_groups.py` scans `chat_screen.py` **and** `Console_Modules/*.py`; a new module is in scope automatically.
- Controller↔controller traffic through named callables resolved at **CALL time**.
- **Zero `query_one`/DOM access in a controller.** Every existing controller has zero. Regions and widgets are the opposite — DOM is theirs.
- **A proxy property standing in for a baseline plain attribute must be read-WRITE and write THROUGH to the controller** (DESIGN.md §7). Wave 3 shipped a getter-only one and turned **41 tests red in a file the branch never touched**.
- **A region owns its behaviour, not its children's API** (DESIGN.md, wave-5 rule from task-2767): route through a region only when the invariant is the region's; querying a child widget by id is idiomatic in Textual, not a boundary violation.
- `action_*` and `@on` handlers Textual resolves **by name on the Screen** stay on the Screen. Their bodies may move; their definitions may not. **`on_key` itself stays on the screen** — it is the focus-routing policy ("treat the composer as the default printable text target"), which is a screen concern.

**Process rules, each paid for:**
- **`git push` after EVERY commit.** An entire unpushed wave was destroyed by the `/private/tmp` cleaner. Work lives in `.worktrees/wave5`; the interpreter is `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.
- **Never `git stash`** — the stash stack is shared across 100+ worktrees.
- **Never `cd` to the parent repo.** A wave-3 implementer's compound `cd <root> && … git commit` swept another session's ~500 uncommitted files into a commit on their branch.
- **Characterise BEFORE extracting**, always. Retroactive characterisation is count-level evidence only.
- **Commit the extraction the moment it exists**, before any sweep.
- Tests FOREGROUND with Bash `timeout: 600000`. A backgrounded pytest strands the implementer — it has done so six times on this programme.
- **`pytest` reporting "no tests ran" is a FAILED gate, not a pass.** Verify every path exists before trusting a green line; two paths were mistyped in wave 3 and produced a silent no-op.
- **If a test fails, run BOTH arms — with and without the change — before calling it causal.** A correct deletion looked causal across three signals in wave 4 and nearly got reverted; a two-arm A/B showed the test was nondeterministic on its own.
- **Scope verification to every file REFERENCING what you move, not the files you edited.** That error let a wave-3 task ship 42 failing tests past a green run. A plain diff CANNOT see moved-but-delegated methods — intersect `defs(new module)` with `defs(baseline screen)` instead.
- Rebase onto `origin/dev` before each task, and **re-measure the ratchet after the FINAL rebase** — wave 3 set its budget twice from a stale base and both landed red.

**Four defect classes that have bitten this programme:**
1. **Dead bodies / over-deletion** — every moved method gone, or a ≤14-line pure-forwarding delegation **with a real caller** (`ast`, not grep). An `ast`-lineno delete misses decorator lines AND can take the first line of the *next* declaration's comment block; both have happened.
2. **Silent drops outside the hunks** — orphaned imports causing runtime `NameError`. pyflakes both files, compare to baseline (currently **25** on `chat_screen.py`).
3. **Imports that look unused but are load-bearing for tests.** Tests reach symbols as `chat_screen_module.X` or `setattr(chat_screen_module, "X", ...)` — invisible to any import-grep, and the quoted form is not even an identifier. Wave 4 deleted five such imports and turned 28 tests red. task-3023 repointed the known set, but **check before deleting any import your extraction orphans**.
4. **Hand-built screen fixtures.** Every wave has shipped exactly one to dev. Sweep `grep -rl 'ChatScreen.__new__\|spec=ChatScreen' Tests/`. `Tests/UI/console_controller_stubs.py` exists — reuse or extend it, and note its `NO_APP` sentinel, which exists because an inferred `app_instance=None` is a silent-default hole.

## File Structure

- `tldw_chatbook/Widgets/Console/console_composer_bar.py` — gains the keymap it already has the operations for.
- `tldw_chatbook/UI/Console_Modules/image.py` (new) — `ConsoleImageController`.
- `tldw_chatbook/UI/Console_Modules/character.py` (new) — `ConsoleCharacterController`.
- `tldw_chatbook/UI/Console_Modules/wiring.py` — two more constructions.
- `tldw_chatbook/UI/Screens/chat_screen.py` — `on_key` and both clusters shrink.
- `Tests/Architecture/test_screen_size_ratchet.py` — budget lowered in Task 4.

---

### Task 1: The composer keymap moves to the composer

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`, `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_composer_keymap.py` (new)

**Interfaces:**
- Produces: a single public entry point on `ConsoleComposerBar` — `handle_console_key(event: Key) -> bool` — returning True when it consumed the key. The screen's `on_key` calls it and returns on True.
- Consumes: nothing from later tasks.

**The split is ownership, not line count.** The composer takes the 18 branches that touch **only** the composer, because each maps to a method it already implements. The screen keeps everything that is *routing policy* or reaches beyond the composer: the setup-modal guard, the hands-free branch, the realtime branch, the command-popup branch, `_should_capture_console_input`, `enter` (72 lines — it sends, touching the store and the DOM), and `pageup/pagedown` (it scrolls the transcript, not the composer).

- [ ] **Step 1: Map all 24 branches** — key(s), line count, every method called, and a per-branch verdict (moves to the composer / stays with reason). **If a branch you expect to move calls anything other than a `ConsoleComposerBar` method, it stays** — report it rather than widening the widget's API to accommodate it. Confirm from the source that every operation the moving branches call already exists on the widget; **if any does not, that branch stays too** (adding operations is a feature change, forbidden in an extraction).
- [ ] **Step 2: Characterise first.** Mount `ConsoleComposerBar` in a minimal host app and drive REAL key presses through `pilot`, asserting the resulting `draft_text` and cursor position — not internal calls. Cover at minimum: backspace, delete, ctrl+w, ctrl+u, arrows, home/end, undo/redo, select-all. Also pin the SCREEN-level routing that must not change: a printable key typed while the composer is not focused still reaches it. Commit separately, push, green pre-move.
- [ ] **Step 3: Move** the 18 branch bodies into `handle_console_key`, verbatim. `on_key` keeps its own definition, its docstring's stated policy, and every branch that stays, and gains one call to the composer.
- [ ] **Step 4: Commit, push, then verify** in one foreground call (`timeout: 600000`): the new test file + `Tests/UI/test_console_composer_undo.py` + `Tests/UI/test_console_command_composer.py` + `Tests/Architecture/`. Then a second call for every other file your referencing sweep flags.
- [ ] **Step 5: Report** the branch table, the stays-list with reasons, whether any operation had to be added to the widget (it should not), the four audits, and before/after line counts.

Commit: `refactor(console): move the composer keymap to the composer (wave 5)`

---

### Task 2: The image controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/image.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Console_Modules/wiring.py`
- Test: `Tests/UI/test_console_image_controller.py` (new)

**Interfaces:**
- Produces: `ConsoleImageController` owning the image cluster (638 lines / 21 methods), constructed in `wiring.py`.

- [ ] **Step 1: Map the cluster**, per-method verdicts, and the image↔message/session traffic in both directions. Waves 2, 3 and 4 all found `*name*`-matched methods that were **false positives**; check each of the 21 against what it actually touches and report any you exclude, plus any image-cluster method whose name lacks "image".
- [ ] **Step 2: Characterise first** — drive a real generate-image path and assert the persisted result (message rows / stored specs), not widget state. `Tests/Chat/test_console_generation_actions.py` and `Tests/Chat/test_console_generation_card.py` already cover parts of this cluster: run them pre-move and record counts. Commit separately, push, green pre-move.
- [ ] **Step 3: Extract** per the binding rule; construct in `wiring.py`.
- [ ] **Step 4: Commit, push, verify** (two calls; include both generation test files).
- [ ] **Step 5: Report** as above.

Commit: `refactor(console): image controller (wave 5)`

---

### Task 3: The character controller

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/character.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Console_Modules/wiring.py`
- Test: `Tests/UI/test_console_character_controller.py` (new)

**Interfaces:**
- Produces: `ConsoleCharacterController` owning the character cluster (412 lines / 12 methods), constructed in `wiring.py`.

**One boundary is already known.** `_build_character_avatar_widget` (74) and `_render_character_avatar_into_section` (33) build and mount a widget into `ConsoleLeftRail`, which takes it as a zero-arg builder (`character_avatar_widget_builder`, DESIGN.md §7). **Widget construction is DOM work and must not enter the controller.** Expect the avatar builders to stay on the screen with the controller owning the *decision* (which character, whether the scope changed) — but map it and say what you found rather than assuming this shape.

- [ ] **Step 1: Map the cluster**, per-method verdicts with reasons, and the character↔session/message traffic. Note that `{{char}}`/`{{persona}}` refer to the CHARACTER and `{{user}}` to the human — do not "fix" any copy that uses them.
- [ ] **Step 2: Characterise first** — drive a real character choice and assert the persisted selection; committed, pushed, green pre-move.
- [ ] **Step 3: Extract** per the binding rule; construct in `wiring.py`.
- [ ] **Step 4: Commit, push, verify** (two calls).
- [ ] **Step 5: Report** as above, including the avatar-builder boundary decision.

Commit: `refactor(console): character controller (wave 5)`

---

### Task 4: Close the wave and lower the ratchet

**Files:**
- Modify: `Tests/Architecture/test_screen_size_ratchet.py`, `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`, `DESIGN.md` if the wave established a new rule

- [ ] **Step 1: Rebase onto `origin/dev` FIRST, then measure** `chat_screen.py`'s line count and `ChatScreen`'s method count with `ast`. Wave 3 measured before its final rebase twice and both budgets landed red.
- [ ] **Step 2: Lower the ratchet** in `_BUDGETS` to exactly those numbers. This is the step the whole wave exists to earn; the second ratchet test fails if you skip it, by design.
- [ ] **Step 3: Annotate the spec** — mark the composer keymap, image and character done, update the Progress numbers honestly, and record what the wave did NOT achieve as well as what it did.
- [ ] **Step 4: `DESIGN.md`** — Task 1 likely establishes a rule worth recording (a keymap belongs with the operations it maps to, not with the focus policy that routes to it). Add it if the wave earned it; **if not, say so in the report rather than inventing one.**
- [ ] **Step 5: Serial gate**, at most two foreground calls (`timeout: 600000`), then commit and push.

Commit: `refactor(console): wave 5 closed — ratchet lowered, honest numbers`

---

## Wave 5 exit criteria

- The composer's key→operation mapping lives on `ConsoleComposerBar`; `on_key` keeps only routing policy and the branches that reach beyond the composer.
- The image and character clusters live in `UI/Console_Modules/`, constructed in `wiring.py`.
- Zero DOM access in any controller; the geometry baseline is byte-identical and green.
- No behaviour change; every pre-existing DOM-driven test passes unchanged.
- **The ratchet is lowered to the post-rebase measurement.**

## Not in wave 5 (deliberate)

- **`compose_content`'s workbench header.** Measured: ~62 composed lines plus a 230-line/15-method sync cluster of which several are `action_*` handlers that must stay on the Screen by name. That is the shape that made wave 3's transcript region net **+1 line**; extracting it would buy structure at no size cost, and this wave has better targets. Revisit only with a reason beyond "it is still inline".
- `on_button_pressed` (206) — wave 4 already routed its six worthwhile branches; the remaining thirteen were judged coordination or event-passing, with reasons recorded.
- The rail (515/27), rag (405/17), skill (339/16) and citation (333/9) clusters.
- Settings and Library screens remain their own future efforts under the same spec.
