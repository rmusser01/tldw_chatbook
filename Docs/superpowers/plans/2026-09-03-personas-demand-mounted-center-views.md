# Personas Demand-Mounted Center Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Personas settle into its usable default Characters surface without constructing four inactive heavy center views, then mount and cache each view only when its workflow first needs it.

**Architecture:** `PersonasScreen` composes four hidden, zero-demand slot containers in the historical center-stack order. A screen-owned async `_ensure_center_view()` population boundary constructs, mounts, hydrates, and caches one requested body with per-view concurrency and lifecycle-generation guards; workflow admission points await that boundary before touching editor/detail state.

**Tech Stack:** Python 3.11+, Textual 8.x, asyncio, pytest/pytest-asyncio

**Spec:** `Docs/superpowers/specs/2026-09-03-personas-demand-mounted-center-views-design.md`

## Global Constraints

- Preserve the four existing heavy root IDs: `ccp-character-editor-view`, `ccp-persona-editor-view`, `personas-dictionary-detail`, and `personas-lore-detail`.
- The initial Characters card, library, preview, inspector, attachments, conversation, and Try-It chrome remain eager.
- Switching modes alone must not mount a heavy view.
- Mounted views remain cached until the `PersonasScreen` instance unmounts; never recompose them on mode switches.
- Apply selection/editor state only after the requested body is mounted and the screen lifecycle generation is current.
- A transient mount failure must remain retryable and must not crash the application.
- Event-loop gaps while opening Personas under the production CSS bundle must stay below 250 ms.
- No storage, provider/runtime, Console-handoff, information-architecture, or dependency changes.
- Use targeted verification only; do not run the full repository suite without explicit user opt-in.
- ADR required: yes.
- ADR path: `backlog/decisions/115-personas-demand-mounted-center-views.md`.
- Reason: the change defines the long-lived lifecycle and restore/admission contract shared by the Personas screen and its four authoring widgets.

---

### Task 1: Pin the initial-load and first-use lifecycle

**Files:**

- Modify: `Tests/UI/test_personas_deferred_center_views.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:600-625`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:1146-1215`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:1396-1445`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:1681-1735`

**Interfaces:**

- Consumes: existing heavy widget constructors and `_show_center(visible_id: str | None) -> None`.
- Produces: `_ensure_center_view(view_key: str) -> Widget | None`, stable slot IDs, and per-screen cached body identity used by later tasks.

- [ ] **Step 1: Replace the old after-load batch assertion with a failing settled-load contract**

  Update `Tests/UI/test_personas_deferred_center_views.py` so a real settled load expects all heavy roots to remain absent while the direct children of `#personas-detail-stack` contain the four slots in historical order:

  ```python
  _HEAVY_VIEW_IDS = (
      "ccp-character-editor-view",
      "ccp-persona-editor-view",
      "personas-dictionary-detail",
      "personas-lore-detail",
  )

  _EXPECTED_STACK_ORDER = [
      "ccp-character-card-view",
      "personas-character-editor-slot",
      "personas-character-attachments",
      "ccp-persona-card-view",
      "personas-persona-editor-slot",
      "personas-conversation-actions",
      "personas-dictionary-detail-slot",
      "personas-lore-detail-slot",
      "personas-conversation-transcript-view",
      "personas-mode-placeholder",
      "personas-characters-empty",
  ]

  async def test_settled_initial_load_keeps_heavy_views_unmounted():
      # Push the real screen and drain its initial worker.
      assert all(not list(screen.query(f"#{view_id}")) for view_id in _HEAVY_VIEW_IDS)
      stack = screen.query_one("#personas-detail-stack")
      assert [child.id for child in stack.children] == _EXPECTED_STACK_ORDER
  ```

- [ ] **Step 2: Run the settled-load test and confirm the current batch mount fails it**

  Run:

  ```bash
  python -m pytest -q Tests/UI/test_personas_deferred_center_views.py::test_settled_initial_load_keeps_heavy_views_unmounted --tb=short
  ```

  Expected: FAIL because `_load_after_mount()` currently calls `_mount_deferred_center_views()` and all four roots exist after settle.

- [ ] **Step 3: Add failing first-use cache and isolation tests**

  Add a mounted test that calls the new boundary directly, records the returned character editor, edits `#personas-char-editor-name`, switches through a second call, and proves only that root exists and the same object/value returns:

  ```python
  async def test_first_use_mounts_only_requested_view_and_caches_it():
      first = await screen._ensure_center_view("character-editor")
      assert isinstance(first, PersonasCharacterEditorWidget)
      first.query_one("#personas-char-editor-name", Input).value = "Keep me"
      second = await screen._ensure_center_view("character-editor")
      assert second is first
      assert second.query_one("#personas-char-editor-name", Input).value == "Keep me"
      assert not list(screen.query(PersonaProfileEditorWidget))
      assert not list(screen.query(PersonasDictionaryDetailWidget))
      assert not list(screen.query(PersonasLoreDetailWidget))
  ```

- [ ] **Step 4: Run the first-use test and confirm the boundary is absent**

  Run:

  ```bash
  python -m pytest -q Tests/UI/test_personas_deferred_center_views.py::test_first_use_mounts_only_requested_view_and_caches_it --tb=short
  ```

  Expected: FAIL with `AttributeError` for `_ensure_center_view`.

- [ ] **Step 5: Implement stable slots and the minimal population boundary**

  In `personas_screen.py`, import `Widget` from `textual.widget`, define the four view/slot identities near `_CENTER_VIEW_IDS`, initialize a lifecycle generation and one `asyncio.Lock` per key, and replace `_mount_deferred_center_views()` with the first-use boundary:

  ```python
  _CENTER_VIEW_ROOTS = {
      "character-editor": "#ccp-character-editor-view",
      "persona-editor": "#ccp-persona-editor-view",
      "dictionary-detail": "#personas-dictionary-detail",
      "lore-detail": "#personas-lore-detail",
  }

  _CENTER_VIEW_SLOTS = {
      "character-editor": "#personas-character-editor-slot",
      "persona-editor": "#personas-persona-editor-slot",
      "dictionary-detail": "#personas-dictionary-detail-slot",
      "lore-detail": "#personas-lore-detail-slot",
  }
  ```

  Compose each slot as a `Vertical` with `display = False`, natural container
  height, and `width = "100%"`. Do not force `height = "auto"`: the editor roots
  own `height = "100%"`, and the wrapper must preserve their full-height layout and
  hit-testing. Implement the factory as a total key match and implement:

  ```python
  async def _ensure_center_view(self, view_key: str) -> Widget | None:
      selector = _CENTER_VIEW_ROOTS.get(view_key)
      if selector is None:
          raise ValueError(f"Unknown Personas center view: {view_key}")
      existing = self.query(selector).first(None)
      if existing is not None:
          return existing
      async with self._center_view_mount_locks[view_key]:
          existing = self.query(selector).first(None)
          if existing is not None:
              return existing
          generation = self._center_view_lifecycle_generation
          body = self._build_center_view(view_key)
          body.display = False
          try:
              slot = self.query_one(_CENTER_VIEW_SLOTS[view_key], Vertical)
              await slot.mount(body)
              if generation != self._center_view_lifecycle_generation or not self.is_mounted:
                  await body.remove()
                  return None
              self._hydrate_center_view(view_key, body)
              slot.display = True
              return body
          except Exception:
              if body.is_mounted:
                  await body.remove()
              logger.opt(exception=True).warning(
                  "Personas center view mount failed (view_key={}).", view_key
              )
              self._notify("Couldn't open this Personas view. Try again.", "error")
              return None
  ```

  `_hydrate_center_view()` calls `set_runtime_source()` only for a mounted `PersonaProfileEditorWidget`. Remove the batch mount call from `_load_after_mount()` and update comments/docstrings to describe first-use mounting.

- [ ] **Step 6: Run the lifecycle tests and make them pass**

  Run:

  ```bash
  python -m pytest -q Tests/UI/test_personas_deferred_center_views.py --tb=short
  ```

  Expected: PASS for initial absence, slot order, first-use isolation, and cached identity.

- [ ] **Step 7: Commit the lifecycle seam**

  ```bash
  git add Tests/UI/test_personas_deferred_center_views.py tldw_chatbook/UI/Screens/personas_screen.py
  git commit -m "perf(personas): add first-use center view lifecycle"
  ```

---

### Task 2: Admit every editor/detail workflow through the boundary

**Files:**

- Modify: `Tests/UI/test_personas_deferred_center_views.py`
- Modify: `Tests/UI/test_personas_workbench_state.py:500-590`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:4662-4930`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:6996-7860`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:9520-9605`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:14540-14920`

**Interfaces:**

- Consumes: `_ensure_center_view(view_key: str) -> Widget | None` from Task 1.
- Produces: exact workflow admission for character/persona create/edit and dictionary/lore select/create/edit/restore.

- [ ] **Step 1: Add failing mounted admission tests for all four bodies**

  Parameterize the existing production-screen harness so each real entry path starts from a settled screen, triggers one action, and asserts the requested root exists while the other three remain absent. Cover:

  ```python
  (
      ("characters", PersonaActionRequested("create"), "ccp-character-editor-view"),
      ("personas", PersonaActionRequested("create"), "ccp-persona-editor-view"),
      ("dictionaries", dictionary_row_selection, "personas-dictionary-detail"),
      ("lore", lore_row_selection, "personas-lore-detail"),
  )
  ```

  Reuse the service fakes already present in `test_personas_workbench.py`,
  `test_personas_dictionaries.py`, and `test_personas_lore.py`; do not introduce a
  surrogate screen.

- [ ] **Step 2: Run each new admission case and confirm direct queries fail**

  Run:

  ```bash
  python -m pytest -q Tests/UI/test_personas_deferred_center_views.py -k "workflow_mounts_only" --tb=short
  ```

  Expected: FAIL with `NoMatches` at the first direct editor/detail query because the batch mount was removed in Task 1.

- [ ] **Step 3: Gate selection and create paths before state mutation**

  At the beginning of the body-dependent portions of these async methods, await the exact key and return when it is unavailable:

  ```python
  detail = await self._ensure_center_view("dictionary-detail")
  if not isinstance(detail, PersonasDictionaryDetailWidget):
      return
  ```

  Apply the pattern to:

  - `_select_dictionary` and `_select_lore_entry`, after service data succeeds but before `state.select_entity()`;
  - `_begin_create_character` and `_begin_create_profile`, before clearing selection/edit state;
  - `_begin_create_dictionary` and `_begin_create_lore`, before clearing selection/edit state.

  Use the returned typed widget instead of immediately querying it again.

- [ ] **Step 4: Gate character/persona edit paths and reuse returned widgets**

  Update the two production edit handlers (`EditCharacterRequested` and
  `EditPersonaProfileRequested`) to await `character-editor` or `persona-editor`
  before advancing editor generations, clearing state, loading record data, or
  starting visual-identity workers. Abort cleanly on `None`.

- [ ] **Step 5: Preserve dictionary/lore restore ordering**

  Extend the existing round-trip restore test in `test_personas_workbench_state.py`:

  ```python
  assert screen2.query_one("#personas-dictionary-detail").display is True
  assert not list(screen2.query("#ccp-character-editor-view"))
  assert not list(screen2.query("#ccp-persona-editor-view"))
  assert not list(screen2.query("#personas-lore-detail"))
  assert screen2.state.selected_entity_id == saved_dictionary_id
  ```

  Add the symmetric lore case. The existing `_apply_pending_restore()` dispatch remains the owner; its `_select_dictionary()` / `_select_lore_entry()` call now awaits body readiness before applying the exact saved record.

- [ ] **Step 6: Run focused workflow and restore tests**

  Run:

  ```bash
  python -m pytest -q \
    Tests/UI/test_personas_deferred_center_views.py \
    Tests/UI/test_personas_workbench_state.py \
    Tests/UI/test_personas_editor_save_in_place.py \
    Tests/UI/test_actor_pack_creation_workflow.py \
    Tests/UI/test_personas_dictionaries.py \
    Tests/UI/test_personas_lore.py \
    --tb=short
  ```

  Expected: PASS with each workflow mounting only its requested heavy body.

- [ ] **Step 7: Commit workflow admission**

  ```bash
  git add Tests/UI/test_personas_deferred_center_views.py Tests/UI/test_personas_workbench_state.py tldw_chatbook/UI/Screens/personas_screen.py
  git commit -m "fix(personas): admit heavy workflows through lazy views"
  ```

---

### Task 3: Prove concurrency, retry, teardown, and responsiveness

**Files:**

- Modify: `Tests/UI/test_personas_deferred_center_views.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:1146-1215`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:1681-1765`

**Interfaces:**

- Consumes: `_ensure_center_view(view_key: str) -> Widget | None`, `_build_center_view(view_key: str) -> Widget`, and `_hydrate_center_view(view_key: str, body: Widget) -> None`.
- Produces: deterministic failure/retry and lifecycle-generation behavior plus the 250 ms responsiveness gate.

- [ ] **Step 1: Add a failing concurrent first-use test**

  Instrument `_build_center_view` while running two calls together:

  ```python
  first, second = await asyncio.gather(
      screen._ensure_center_view("character-editor"),
      screen._ensure_center_view("character-editor"),
  )
  assert first is second
  assert build_count == 1
  assert len(list(screen.query(PersonasCharacterEditorWidget))) == 1
  ```

- [ ] **Step 2: Add a failing transient mount retry test**

  Patch the character slot's `mount` method to raise once, then delegate to the
  real bound method. Assert the first ensure returns `None`, the second returns a
  mounted editor, one bounded notification was emitted, and only one editor remains.

- [ ] **Step 3: Add a failing teardown-generation test**

  Gate the slot mount with two `asyncio.Event` objects. Start ensure, wait until
  the fake mount reaches its controlled await, increment the screen lifecycle via
  `on_unmount`, release the mount, and assert the method returns `None`, performs no
  persona runtime hydration/focus, and leaves no mounted heavy root attached to the
  retired screen.

- [ ] **Step 4: Run the concurrency/retry/teardown tests and confirm the missing guards**

  Run:

  ```bash
  python -m pytest -q Tests/UI/test_personas_deferred_center_views.py -k "concurrent or retry or teardown" --tb=short
  ```

  Expected: at least one FAIL before the final guards, with duplicate construction,
  consumed failure state, or stale post-mount hydration.

- [ ] **Step 5: Complete lifecycle invalidation and cleanup**

  Increment `_center_view_lifecycle_generation` at the start of `on_unmount()`.
  Ensure the population boundary double-checks the mounted root inside its per-view
  lock, marks no separate ready state before mount/hydration succeeds, hides slots
  until success, and removes a partially mounted body on failure or stale generation.
  Catch cancellation separately and re-raise it after cleanup:

  ```python
  except asyncio.CancelledError:
      if body.is_mounted:
          await body.remove()
      raise
  ```

- [ ] **Step 6: Add and run the production-CSS heartbeat regression**

  Patch `PersonasCharacterEditorWidget.compose` to `time.sleep(0.35)` before
  yielding a `Static`, run a 5 ms heartbeat while pushing a real `PersonasScreen`,
  settle the initial worker, and assert the default card is mounted, the editor is
  absent, and `max(gaps) < 0.25`:

  ```python
  def blocking_character_editor_compose(self):
      time.sleep(0.35)
      yield Static("deliberately slow inactive editor")

  assert screen.query_one("#ccp-character-card-view").is_mounted
  assert not list(screen.query("#ccp-character-editor-view"))
  assert max(gaps) < 0.25
  ```

  Run:

  ```bash
  python -m pytest -q Tests/UI/test_personas_deferred_center_views.py::test_inactive_editor_compose_cannot_stall_personas_navigation --tb=short
  ```

  Expected: PASS only when initial load never constructs the inactive editor.

- [ ] **Step 7: Run the complete lifecycle module**

  ```bash
  python -m pytest -q Tests/UI/test_personas_deferred_center_views.py --tb=short
  ```

  Expected: PASS.

- [ ] **Step 8: Commit hardening and responsiveness coverage**

  ```bash
  git add Tests/UI/test_personas_deferred_center_views.py tldw_chatbook/UI/Screens/personas_screen.py
  git commit -m "test(personas): harden lazy view lifecycle"
  ```

---

### Task 4: Verify the affected Personas surface and close TASK-31215

**Files:**

- Modify: `backlog/tasks/task-31215 - Personas-mount-heavy-center-views-on-first-use.md`
- Modify only if the implementation exposed a generalizable incident: `backlog/docs/lessons-testing-evidence.md`

**Interfaces:**

- Consumes: all Task 1-3 behavior.
- Produces: focused verification evidence, completed acceptance criteria, and concise implementation notes.

- [ ] **Step 1: Run the focused Personas regression set**

  ```bash
  python -m pytest -q \
    Tests/UI/test_personas_deferred_center_views.py \
    Tests/UI/test_personas_workbench.py \
    Tests/UI/test_personas_workbench_state.py \
    Tests/UI/test_personas_editor_save_in_place.py \
    Tests/UI/test_actor_pack_creation_workflow.py \
    Tests/UI/test_personas_dictionaries.py \
    Tests/UI/test_personas_lore.py \
    Tests/UI/test_personas_inspector_pane.py \
    --tb=short
  ```

  Expected: PASS. Do not broaden to the full suite without user opt-in.

- [ ] **Step 2: Run the architectural message-pump guard**

  ```bash
  python -m pytest -q Tests/Architecture/test_no_blocking_io_on_message_pump.py --tb=short
  ```

  Expected: PASS.

- [ ] **Step 3: Run scoped static and diff checks**

  ```bash
  python -m compileall -q tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_deferred_center_views.py
  ruff check tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_deferred_center_views.py
  ruff format --check tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_deferred_center_views.py
  git diff --check origin/dev...HEAD
  ```

  Expected: all commands exit 0.

- [ ] **Step 4: Review the diff against the approved spec**

  Confirm the diff changes no storage/provider/Console contract, retains every
  existing heavy root ID, introduces no body unmount-on-switch path, and contains no
  direct initial-load call that populates a heavy view. Search explicitly:

  ```bash
  rg -n "_mount_deferred_center_views|_ensure_center_view|personas-.*-slot" \
    tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_deferred_center_views.py
  ```

- [ ] **Step 5: Update the Backlog task through the CLI**

  Check all five acceptance criteria, add implementation notes with measured widget
  and heartbeat evidence, list targeted test counts, record any plan deviation, and
  set TASK-31215 to Done only after every Definition-of-Done requirement is met:

  ```bash
  backlog task edit 31215 \
    --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
    --notes "Implemented screen-owned first-use mounting for all four heavy Personas center views; verification evidence recorded here." \
    --status Done --plain
  ```

- [ ] **Step 6: Commit the closeout records**

  ```bash
  git add 'backlog/tasks/task-31215 - Personas-mount-heavy-center-views-on-first-use.md' backlog/docs/lessons-testing-evidence.md
  git commit -m "docs(personas): close demand-mounted view task"
  ```
