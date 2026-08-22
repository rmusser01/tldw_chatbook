# Persona Buddy UAT Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan sequentially. Every production behavior starts with `superpowers:test-driven-development`; use `superpowers:verification-before-completion` before each commit and completion claim. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore persisted Persona Buddy preferences on real app startup, size resolved portrait frames to the visible frame slot, and clear the project-instructions setup rejection so full-application UAT is faithful and can exercise trusted live states.

**Architecture:** Keep the existing ownership boundaries. `config.py` projects the raw Buddy table but does not parse it; `parse_persona_buddy_preferences()` remains the strict field boundary used by the app-owned controller. `PersonaBuddyWidget` derives one exact visible frame-slot size and uses it both for resolution authority and the controller call. The Console controller owns terminal run state when project-instructions setup rejects a turn. No renderer, schema, CSS, state-priority, or preference-model change is authorized.

**Tech Stack:** Python 3.11+, Textual 8, TOML, existing Persona Buddy/Persona Visual runtime, pytest/Pilot, tmux for isolated real-app UAT.

**ADR required:** no

**ADR path:** Existing [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)

**Reason:** ADR-074 already defines Buddy preference persistence, native Textual rendering, exact runtime authority, and verification boundaries. This task fixes two implementation regressions without changing those decisions.

---

## Fixed contracts and file map

- Modify `tldw_chatbook/config.py` only to pass through the effective `persona_buddy` table in the normalized application settings dictionary.
- Create `Tests/Persona_Buddy/test_persona_buddy_config_projection.py` for isolated real-TOML projection, parser defaults, controller startup, and first-write preservation.
- Modify `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py` only to derive resolution dimensions from the mounted, visible `#persona-buddy-frame` content region.
- Modify `Tests/UI/test_persona_buddy_widget.py` for exact real-CSS frame-region, crop, resize-authority, and hidden/collapsed controls.
- Modify the focused Console project-instructions/controller test and `tldw_chatbook/Chat/console_chat_controller.py` only if actual-app UAT proves that disabling unavailable project instructions leaves the rejected run non-terminal.
- Reuse `Tests/UI/test_persona_buddy_app_mount.py`, `Tests/Persona_Buddy/test_persona_buddy_preferences.py`, `Tests/Persona_Buddy/test_persona_buddy_resolution.py`, `Tests/UI/test_personas_workbench_state.py`, and `Tests/Architecture/test_persona_buddy_boundary.py` as regression gates; do not broaden them unless a born-RED test proves an adjacent contract is missing.
- The existing `/private/tmp/tldw-buddy-uat-profile.GLxMvx` fixture is diagnostic input only. Copy it to a new disposable root before final UAT; never run against it in place or against the real user profile.
- Do not change `NO_COLOR` behavior. The final child process explicitly unsets it so color evidence is meaningful.
- Do not add CSS, a new live-test framework, provider transport, Persona Visual schema/runtime, Workbench behavior, or geometry compensation logic.

## Task 1: Project persisted Buddy preferences into app startup

**Files:**

- Create: `Tests/Persona_Buddy/test_persona_buddy_config_projection.py`
- Modify: `tldw_chatbook/config.py:1759-1785`

- [ ] **Step 1: Write the real-config projection RED**

  Add `test_real_toml_persona_buddy_table_reaches_controller_startup` using the cache save/restore pattern from `Tests/Video_Generation/test_config_projection.py`. Point `TLDW_CONFIG_PATH` at a private scratch TOML containing:

  ```toml
  [persona_buddy]
  enabled = true
  source = "local"
  local_persona_id = "persona-uat"
  open = true
  collapsed = true
  x = 7
  y = 5
  width = 31
  height = 14
  ```

  Load through `load_settings(force_reload=True)`, pass `settings["persona_buddy"]` through `parse_persona_buddy_preferences()`, construct `PersonaBuddyController`, and assert the exact selection, open/collapsed state, and geometry. Assert the scratch TOML bytes are unchanged after construction.

- [ ] **Step 2: Run the projection RED**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Buddy/test_persona_buddy_config_projection.py -k real_toml_persona_buddy_table_reaches_controller_startup
  ```

  Expected: fail because `load_settings()` omits the `persona_buddy` key (or the controller receives defaults if the test uses `.get`).

- [ ] **Step 3: Add malformed/default and first-action RED controls**

  Add:

  - `test_projected_malformed_fields_keep_independent_safe_defaults`
  - `test_first_persist_after_restart_preserves_loaded_geometry`
  - `test_projection_does_not_expose_unrelated_private_tables`

  The first-action test constructs the controller from the projected table, changes only `open`, persists through an injected writer, and asserts the serialized write retains `(x=7, y=5, width=31, height=14)` rather than `(1000000, 1000000, ...)`.

- [ ] **Step 4: Run all config REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Buddy/test_persona_buddy_config_projection.py Tests/Persona_Buddy/test_persona_buddy_preferences.py
  ```

  Expected: the new projection/startup cases fail; existing preference parser and persistence cases remain green.

- [ ] **Step 5: Implement the one-table pass-through**

  In the full-table section of `_load_settings_uncached()` add only:

  ```python
  "persona_buddy": copy.deepcopy(toml_config_data.get("persona_buddy", {})),
  ```

  Do not parse fields in `config.py`, synthesize a default table, perform a startup write, or expose any additional table.

- [ ] **Step 6: Run GREEN and mutation proof**

  Run Step 4. Then temporarily remove the new projection line and rerun the three new tests; the startup and first-write tests must fail. Restore the line and rerun Step 4 green.

- [ ] **Step 7: Run the adjacent startup/Workbench gate**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench_state.py -k 'restart or persona_buddy or explicit'
  ```

  Expected: pass, with no startup write and no loss of selection/geometry.

- [ ] **Step 8: Verify and commit Task 1**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/config.py Tests/Persona_Buddy/test_persona_buddy_config_projection.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/config.py Tests/Persona_Buddy/test_persona_buddy_config_projection.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/config.py Tests/Persona_Buddy/test_persona_buddy_config_projection.py
  git diff --check
  git add tldw_chatbook/config.py Tests/Persona_Buddy/test_persona_buddy_config_projection.py
  git commit -m "fix: restore Persona Buddy startup preferences"
  ```

## Task 2: Resolve portraits for the visible frame slot

**Files:**

- Modify: `Tests/UI/test_persona_buddy_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py:252-337`

- [ ] **Step 1: Extend the fake controller with exact call evidence**

  Record each `(cols, lines)` passed to `_FakeController.resolve_current_visual()` without changing its returned visual. Keep the evidence content-free and local to the test.

- [ ] **Step 2: Write the exact frame-region RED**

  Add `test_resolution_uses_visible_frame_content_region_not_window_region`. Mount `_BuddyApp` with the real bundled stylesheet at `80x24`, wait for a resolution, then assert:

  ```python
  buddy = app.screen.query_one(PersonaBuddyWidget)
  frame = buddy.query_one("#persona-buddy-frame", Static)
  assert controller.resolve_sizes[-1] == (
      frame.content_region.width,
      frame.content_region.height,
  )
  assert controller.resolve_sizes[-1] != (
      buddy.content_region.width,
      buddy.content_region.height,
  )
  ```

- [ ] **Step 3: Run the frame-region RED**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_widget.py -k resolution_uses_visible_frame_content_region_not_window_region
  ```

  Expected: fail because the controller currently receives the whole Buddy content region.

- [ ] **Step 4: Add crop, resize, and refusal RED controls**

  Add:

  - `test_prepared_portrait_fits_complete_visible_frame_slot`
  - `test_frame_slot_resize_changes_resolution_authority_once`
  - `test_hidden_collapsed_or_zero_frame_slot_does_not_resolve`

  The crop test uses a production-shaped colored frame with distinct top and bottom markers, asserts both paint inside the frame Static, and asserts no marker is clipped into the status/hints rows. The resize test changes the actual frame-slot height and waits for exactly one new `(cols, lines)` call. The refusal test proves no call occurs while the frame Static is hidden, collapsed, detached, or zero-sized.

- [ ] **Step 5: Run all frame REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_widget.py -k 'frame_content_region or frame_slot or prepared_portrait or hidden_collapsed'
  ```

  Expected: exact-size/crop/authority cases fail under the whole-window sizing; pre-existing hidden/collapsed safety remains green where already covered.

- [ ] **Step 6: Implement one shared frame-size helper**

  Add a private helper used by both `_resolution_loop()` and `_resolution_authority()`:

  ```python
  def _resolution_size(self) -> tuple[int, int] | None:
      if not self.is_attached:
          return None
      frame = self.query_one("#persona-buddy-frame", Static)
      if not frame.display:
          return None
      region = frame.content_region
      if region.width < 1 or region.height < 1:
          return None
      return region.width, region.height
  ```

  `_resolution_loop()` passes the returned exact values without `max(1, ...)`; `_resolution_authority()` embeds the same values and returns `None` when unavailable. Preserve every existing controller/view/post-await/unavailable fence.

- [ ] **Step 7: Run GREEN and mutation proof**

  Run Step 5 and the full widget file. Then temporarily change `_resolution_size()` back to `self.content_region`; the exact-size/crop test must fail. Temporarily omit the size tuple from `_resolution_authority()`; the resize-authority test must fail. Restore both guards and rerun green.

- [ ] **Step 8: Run Buddy lifecycle/runtime regressions**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/UI/test_persona_buddy_widget.py \
    Tests/UI/test_persona_buddy_app_mount.py \
    Tests/Persona_Buddy/test_persona_buddy_resolution.py
  ```

  Expected: pass, including animation, reduced motion, fallback, cancellation, modal resume, navigation, unavailable authority, and compact/collapsed behavior.

- [ ] **Step 9: Verify and commit Task 2**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py
  git diff --check
  git add tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py
  git commit -m "fix: size Persona Buddy portraits to frame slot"
  ```

## Task 3: Terminalize a project-instructions setup rejection

**Files:**

- Modify: `Tests/Chat/test_console_agent_project_instructions.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`

- [ ] **Step 1: Write the actual disable-path RED**

  Exercise a real controller submission whose enabled project instructions have no eligible binding and whose setup callback returns `disable`. Assert the rejected result is non-accepted, the session preference is disabled, and the run state is terminal rather than `VALIDATING`; then submit again and prove the provider gateway is reached.

- [ ] **Step 2: Run the focused RED**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_agent_project_instructions.py -k 'disable_terminalizes or disable_allows_retry'
  ```

  Expected: the first result returns `project_instructions_disabled` while the run remains `VALIDATING`, blocking the retry.

- [ ] **Step 3: Apply the narrow controller-owned terminal transition**

  When setup returns `disable`, clear project-instruction delivery and return through the controller's existing blocked/terminal path with fixed content-free copy. Do not auto-resubmit, call the provider, or alter project-instruction persistence ownership.

- [ ] **Step 4: Run GREEN, mutation, and adjacent project-instructions tests**

  Run the focused REDs and the complete project-instructions controller test file. Temporarily restore the raw `ConsoleSubmitResult` return; the terminal-state/retry test must fail. Restore the fix, run scoped Ruff/format/compile/diff checks, and commit the two-file repair.

## Task 4: Verify the integrated repair in the actual application

**Files:**

- Modify: `backlog/tasks/task-20938 - Repair-Persona-Buddy-restart-and-frame-sizing.md`
- Modify only if this incident adds durable new knowledge: `backlog/docs/lessons-live-verification.md` or `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Run the complete touched-component gate in a pre-import isolated environment**

  ```bash
  UAT_TEST_ROOT="$(mktemp -d /private/tmp/task20938-tests.XXXXXX)"
  env -u NO_COLOR \
    HOME="$UAT_TEST_ROOT/home" \
    XDG_CONFIG_HOME="$UAT_TEST_ROOT/config" \
    XDG_DATA_HOME="$UAT_TEST_ROOT/data" \
    XDG_CACHE_HOME="$UAT_TEST_ROOT/cache" \
    TLDW_CONFIG_PATH="$UAT_TEST_ROOT/config/tldw_cli/config.toml" \
    TLDW_TEST_MODE=1 \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
      Tests/Persona_Buddy/test_persona_buddy_config_projection.py \
      Tests/Persona_Buddy/test_persona_buddy_preferences.py \
      Tests/Persona_Buddy/test_persona_buddy_resolution.py \
      Tests/UI/test_persona_buddy_widget.py \
      Tests/UI/test_persona_buddy_app_mount.py \
      Tests/UI/test_personas_workbench_state.py \
      Tests/Architecture/test_persona_buddy_boundary.py \
      Tests/test_probe_import_provenance.py
  ```

  Expected: all selected tests pass. Record any unchanged branch baseline separately; do not mark the task Done if a modified-component failure remains.

- [ ] **Step 2: Run static, privacy, and scope gates**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
    tldw_chatbook/config.py \
    tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py \
    Tests/Persona_Buddy/test_persona_buddy_config_projection.py \
    Tests/UI/test_persona_buddy_widget.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check \
    tldw_chatbook/config.py \
    tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py \
    Tests/Persona_Buddy/test_persona_buddy_config_projection.py \
    Tests/UI/test_persona_buddy_widget.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile \
    tldw_chatbook/config.py \
    tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py \
    Tests/Persona_Buddy/test_persona_buddy_config_projection.py \
    Tests/UI/test_persona_buddy_widget.py
  git diff --check origin/dev...HEAD
  git diff --name-only origin/dev...HEAD
  ```

  Expected: static checks pass; the file list contains only TASK-20938 planning/closeout plus the two production and two focused test files unless an explicitly documented lesson is warranted.

- [ ] **Step 3: Clone the diagnostic fixture into a disposable UAT root**

  ```bash
  UAT_SOURCE=/private/tmp/tldw-buddy-uat-profile.GLxMvx
  UAT_ROOT="$(mktemp -d /private/tmp/task20938-uat.XXXXXX)"
  cp -R "$UAT_SOURCE/." "$UAT_ROOT/"
  UAT_CONFIG="$UAT_ROOT/effective/config.toml"
  perl -pi -e "s#\Q$UAT_SOURCE\E#$UAT_ROOT#g" "$UAT_CONFIG"
  chmod 700 "$UAT_ROOT" "$UAT_ROOT/effective" "$UAT_ROOT/data"
  chmod 600 "$UAT_CONFIG"
  shasum -a 256 "$UAT_CONFIG" > "$UAT_ROOT/config.before.sha256"
  ```

  Expected: the copied config points only inside the new disposable root. Confirm the real user config hash before launch and do not continue if any path resolves outside `UAT_ROOT`.

- [ ] **Step 4: Launch the real app without monochrome mode**

  ```bash
  UAT_SOCKET="$UAT_ROOT/buddy.sock"
  env -u NO_COLOR \
    HOME="$UAT_ROOT/home" \
    XDG_CONFIG_HOME="$UAT_ROOT/xdg-config" \
    XDG_DATA_HOME="$UAT_ROOT/data" \
    XDG_CACHE_HOME="$UAT_ROOT/cache" \
    TLDW_CONFIG_PATH="$UAT_CONFIG" \
    TLDW_TEST_MODE=1 \
    tmux -S "$UAT_SOCKET" new-session -d -x 120 -y 40 \
      '/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app'
  ```

  Expected: the actual `TldwCli` process starts with the selected local Persona Buddy visible at the persisted geometry. Verify the child environment contains no `NO_COLOR` and the process reads the copied `TLDW_CONFIG_PATH`.

- [ ] **Step 5: Perform and capture actual-app UAT**

  Use `tmux -S "$UAT_SOCKET" capture-pane -p -e` plus real key/mouse input to verify:

  - startup already shows the configured Buddy without a Workbench action;
  - the complete top and bottom of the portrait fit between header and status/hints;
  - captured ANSI cells include at least one RGB foreground/background where `r`, `g`, and `b` are not all equal;
  - idle animation changes painted cells without changing the selected Persona;
  - the disposable local provider drives thinking, speaking, tool-running, approval-needed, error, and recovery state labels/paint;
  - navigation remount, Fold/Open, Close/Show, keyboard move/resize/reset, and viewport resize remain operable;
  - no first action rewrites the saved width/height or replaces saved `x/y` with the sentinel.

  Capture before-action, after-action, and post-restart panes under `$UAT_ROOT/evidence/`; record predicates rather than relying on visual memory.

- [ ] **Step 6: Restart and prove durable restoration**

  Exit the first child cleanly, record the copied config, relaunch with the exact Step 4 environment, and verify selection, open/collapsed state, geometry, full frame, and color restore before any input. The expected config delta is only the deliberately performed geometry/open/collapse actions; no startup-only rewrite is allowed.

- [ ] **Step 7: Shut down and verify containment**

  ```bash
  tmux -S "$UAT_SOCKET" kill-server
  shasum -a 256 "$UAT_CONFIG" > "$UAT_ROOT/config.after.sha256"
  git status --short
  git worktree list --porcelain
  ```

  Expected: no live UAT process remains, the assigned worktree contains only intended task closeout edits, other worktrees are untouched, and the real user config hash still equals its recorded pre-UAT hash.

- [ ] **Step 8: Record closeout truthfully**

  Update TASK-20938 through Backlog CLI: check all seven ACs only if every scoped gate is green, replace the provisional plan with the final plan link plus deviations, add concise Implementation Notes with RED/GREEN/mutation/static/full-app evidence, and set status `Done`. If the UAT reveals another independent defect, keep this task In Progress or file a separate collision-safe task before claiming completion.

- [ ] **Step 9: Verify and commit closeout**

  ```bash
  backlog task 20938 --plain
  git diff --check
  git add \
    'backlog/tasks/task-20938 - Repair-Persona-Buddy-restart-and-frame-sizing.md' \
    Docs/superpowers/plans/2026-08-22-task-20938-persona-buddy-uat-repairs.md
  git commit -m "docs: complete Persona Buddy UAT repairs"
  git status --short --branch
  ```

  Expected: task is Done with seven checked ACs, implementation notes and exact evidence; the worktree is clean after commit.
