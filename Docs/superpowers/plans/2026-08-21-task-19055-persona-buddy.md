# Persona Buddy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off, explicitly selected, app-wide floating companion for one eligible local Persona, driven only by trusted application lifecycle state and the immutable Persona Visual runtime.

**Architecture:** `TldwCli` owns one screen-independent `PersonaBuddyController` and one UI-thread mount reconciler. Every ordinary `BaseAppScreen` dynamically mounts or removes a disposable Textual view; the view polls immutable controller snapshots and the controller retains no screen/widget references. Personas Workbench and app-owned Console runtime seams publish exact identity and trusted state leases, while Persona Visual database, asset, decode, frame preparation, and config writes run off-loop behind one serialized shield-and-drain boundary.

**Tech Stack:** Python 3.11+, Textual 8, asyncio, SQLite/Persona Visual repository and runtime, Pillow/Rich Pixels, pytest/Pilot.

**Execution skills:** Use @superpowers:test-driven-development for every behavior change, @textual-tui and @impeccable for Task 3, and @superpowers:verification-before-completion before each completion claim.

---

## File map and fixed contracts

- `tldw_chatbook/Persona_Buddy/preferences.py`: strict `[persona_buddy]` config codec and atomic persistence for `enabled`, `source`, `local_persona_id`, `open`, `collapsed`, `x`, `y`, `width`, and `height`.
- `tldw_chatbook/Persona_Buddy/controller.py`: app-owned identity, generations, source-scoped leases, priority resolution, async serialization, and immutable public snapshots; no Textual imports or view references.
- `tldw_chatbook/Persona_Buddy/rendering.py`: selected-frame Pillow decode and bounded Rich `Pixels` preparation; no Textual imports.
- `tldw_chatbook/Persona_Buddy/console_adapter.py`: content-free mapping from trusted Console lifecycle events into controller leases.
- `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py`: the disposable floating view, mouse/keyboard geometry, timers, and painted output only.
- `tldw_chatbook/app.py`: construct/shutdown the controller and reconcile the current `BaseAppScreen` mount on explicit preference/lifecycle changes.
- `tldw_chatbook/UI/Navigation/base_app_screen.py`: screen-local dynamic mount/unmount; releases mouse capture and removes only its own widget generation.
- Canonical state keys use underscores: `idle`, `listening`, `thinking`, `speaking`, `approval_needed`, `tool_running`, `wake_armed`, `offline`, `error`. Hyphens are labels only.
- Priority is exactly: `error`, `approval_needed`, timed explicit/custom, authored trigger, `tool_running`, `wake_armed` only while live state is absent/idle, trusted voice, `offline`, `idle`.
- The view uses `position: absolute` plus `overlay: screen`, starts bottom-right, and never consumes parent flow or `fr` budget.
- The controller retains no screen/widget references. `TldwCli.reconcile_persona_buddy_view()` calls the current screen; each mounted view polls immutable snapshots and fences screen identity plus view generation.
- The Buddy never consumes model prose or streaming `Emote:` directives.
- User instruction: run touched/modified component tests only, never the full suite.

Baseline before Buddy code: the prescribed touched suite produced `727 passed, 1 failed`; the sole failure was the pre-existing `widget_defaults_self.tcss` bundle mismatch in `Tests/UI/test_css_bundle_sync_guard.py`. Prospective legacy formatter baseline: Ruff would reformat 17 of the 22 existing files listed in Task 6 Step 6 and already accepts the other 5; preserve that exact file-set result instead of bulk-formatting unrelated legacy code. Because this task adds widget CSS, regenerate and review the exact bundle during Task 3; do not mark the task Done unless the branch's final scoped CSS gate and every other touched gate pass.

## Task 1: Freeze preferences and controller state

**Files:**

- Create: `tldw_chatbook/Persona_Buddy/__init__.py`
- Create: `tldw_chatbook/Persona_Buddy/preferences.py`
- Create: `tldw_chatbook/Persona_Buddy/controller.py`
- Create: `Tests/Persona_Buddy/test_persona_buddy_preferences.py`
- Create: `Tests/Persona_Buddy/test_persona_buddy_controller.py`

- [ ] **Step 1: Write the preference REDs**

  Add named tests `test_preferences_default_off_without_selection`, `test_preferences_round_trip_exact_local_selection_and_geometry`, `test_preferences_reject_malformed_fields_independently`, and `test_preference_failure_is_path_free`. Use the public shape:

  ```python
  prefs = parse_persona_buddy_preferences({"enabled": True, "source": "local", "local_persona_id": "p-1", "x": 9, "y": 4, "width": 28, "height": 12})
  assert prefs.selection == PersonaBuddySelection("local", "p-1")
  assert serialize_persona_buddy_preferences(prefs)["local_persona_id"] == "p-1"
  ```

- [ ] **Step 2: Run the preference REDs**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Buddy/test_persona_buddy_preferences.py
  ```

  Expected: collection fails with `ModuleNotFoundError: tldw_chatbook.Persona_Buddy`.

- [ ] **Step 3: Implement the strict config contracts**

  Implement exact frozen/slotted types and bounded parsing:

  ```python
  @dataclass(frozen=True, slots=True)
  class PersonaBuddySelection:
      source: Literal["local"]
      local_persona_id: str

  @dataclass(frozen=True, slots=True)
  class PersonaBuddyGeometry:
      x: int
      y: int
      width: int
      height: int

  def parse_persona_buddy_preferences(section: Mapping[str, object]) -> PersonaBuddyPreferences: ...
  def serialize_persona_buddy_preferences(prefs: PersonaBuddyPreferences) -> dict[str, object]: ...
  def persist_persona_buddy_preferences(prefs: PersonaBuddyPreferences) -> bool:
      return save_settings_to_cli_config({"persona_buddy": serialize_persona_buddy_preferences(prefs)})
  ```

  Each malformed field falls back to its safe default without accepting bool-as-int, unknown source, controls, or an empty/oversized Persona ID. Public errors and logs use fixed categories only.

- [ ] **Step 4: Run the preference GREENs**

  Run the Step 2 command. Expected: all preference tests pass.

- [ ] **Step 5: Write the controller REDs**

  Add `test_priority_is_exact_for_overlapping_sources`, `test_release_requires_exact_source_owner`, `test_wake_armed_yields_to_non_idle_live_voice`, `test_timed_custom_state_expires`, and `test_selection_never_changes_from_observed_persona`. Drive this API:

  ```python
  token = controller.acquire_state(source="approval", owner="session:round", state="approval_needed")
  controller.acquire_state(source="tool", owner="run:call", state="tool_running")
  assert controller.snapshot().state == "approval_needed"
  assert controller.release_state(source="approval", owner="wrong") is False
  assert controller.release_state(token=token) is True
  ```

- [ ] **Step 6: Run the controller REDs**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Buddy/test_persona_buddy_controller.py
  ```

  Expected: import or missing `PersonaBuddyController` failures.

- [ ] **Step 7: Implement minimal lease/priority state**

  Implement:

  ```python
  class PersonaBuddyController:
      def snapshot(self) -> PersonaBuddySnapshot: ...
      def select_local_persona(self, persona_id: str) -> int: ...
      def acquire_state(self, *, source: str, owner: str, state: str, expires_at: float | None = None) -> PersonaBuddyLeaseToken: ...
      def release_state(self, *, token: PersonaBuddyLeaseToken | None = None, source: str | None = None, owner: str | None = None) -> bool: ...
      def set_timed_state(self, *, owner: str, state: str, ttl_seconds: float) -> PersonaBuddyLeaseToken: ...
      def set_authored_trigger(self, *, owner: str, state: str) -> PersonaBuddyLeaseToken: ...
  ```

  Use a lock around source/owner lease maps, a monotonic generation, normalized safe state grammar, and frozen/slotted snapshots with no paths, bytes, prompts, or exception text in repr.

- [ ] **Step 8: Run Task 1 GREEN and mutations**

  Run both Task 1 test files. Then temporarily reverse priority, weaken exact owner release, and permit observed Persona retargeting one at a time; each named test must fail before restoration.

- [ ] **Step 9: Commit Task 1**

  ```bash
  git add tldw_chatbook/Persona_Buddy Tests/Persona_Buddy/test_persona_buddy_preferences.py Tests/Persona_Buddy/test_persona_buddy_controller.py
  git commit -m "feat: define Persona Buddy controller state"
  ```

## Task 2: Resolve Persona Visual frames under app-owned lifetime

**Files:**

- Modify: `tldw_chatbook/Persona_Buddy/controller.py`
- Create: `tldw_chatbook/Persona_Buddy/rendering.py`
- Modify: `tldw_chatbook/app.py:5772-5775`
- Modify: `tldw_chatbook/app.py:6340-6395`
- Modify: `tldw_chatbook/app.py:11530-11555`
- Create: `Tests/Persona_Buddy/test_persona_buddy_resolution.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`

- [ ] **Step 1: Write the real-SQLite resolution RED**

  Add `test_resolve_selected_local_persona_from_real_active_binding`, `test_disabled_deleted_missing_or_unbound_selection_preserves_enabled_but_hides`, and `test_state_idle_portrait_fallback_never_blanks`. Create a real TASK-19053 graph and assert exact `PersonaVisualIdentity`, `requested_state`, `resolved_state`, `animation_id`, cache-asset tuple, frame asset ID, manifest-frame index, and selected-frame identity.

- [ ] **Step 2: Write selected-frame and painted-render REDs**

  Add `test_sprite_frames_prepare_distinct_painted_frames`, `test_reduced_motion_prepares_only_frame_zero`, `test_decode_failure_keeps_previous_or_portrait_frame`, and `test_render_snapshot_repr_is_byte_and_path_free`. The renderer contract is:

  ```python
  prepared = prepare_persona_buddy_frame(
      resolved_frame,
      resolution_cache_identity=resolution.cache_identity,
      cols=24,
      lines=10,
  )
  assert prepared.cells
  assert prepared.graph_identity == resolution.cache_identity.graph
  assert prepared.asset_id == resolved_frame.asset_id
  assert prepared.selected_frame == resolved_frame.selected_frame
  ```

- [ ] **Step 3: Run the resolution/render REDs**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Buddy/test_persona_buddy_resolution.py
  ```

  Expected: missing resolution/render API failures.

- [ ] **Step 4: Implement selected-frame preparation**

  In `rendering.py`, Pillow-decode the selected embedded frame, fully load it, bound dimensions/cells, and return a repr-hidden Rich `Pixels` renderable plus immutable cell/cache metadata. Never return source paths or raw bytes publicly. Decode/format failures return `persona_buddy_frame_unavailable` without replacing a previously accepted frame.

- [ ] **Step 5: Define the serialized drain result and operation seam**

  Implement one registered inner task before the first await:

  ```python
  T = TypeVar("T")

  @dataclass(frozen=True, slots=True)
  class BuddyDrainResult(Generic[T]):
      completed: bool
      value: T | None = None
      error_category: str | None = None
      cancellation: BaseException | None = field(default=None, repr=False)

  async def _drain_owned(self, awaitable: Awaitable[T], *, name: str) -> BuddyDrainResult[T]: ...
  ```

  Shield the inner task, retain the first outer cancellation, absorb repeated outer cancellations until the child settles, distinguish child cancellation from successful `None`, and release the operation slot only after draining and outcome handling.

- [ ] **Step 6: Implement resolution with an explicit fence tuple**

  Snapshot the complete validated `PersonaVisualIdentity` object and complete `PersonaVisualCacheIdentity` object, not a hand-picked partial tuple. Their equality therefore fences every existing identity field, including Persona/binding/binding-version, pack/revision, version/version-number/manifest hash, requested/resolved state, animation, reduced-motion flag, complete cache-asset tuple, and portrait identity. Add `(profile_generation, persona_revision, controller_generation, preferences_generation, viewport_generation, frame asset_id, manifest_frame_index, selected_frame)` around those exact objects. Run local Persona lookup, repository graph read, `resolve_active_persona_visual`, Pillow decode, and Rich preparation via `asyncio.to_thread`. Insert `is_current(snapshot)` immediately after each await and before each state mutation. A late view is never targeted; it reads the resulting snapshot on its own poll. Keep runtime vocabulary strictly Persona Visual.

- [ ] **Step 7: Write await-barrier/cancellation REDs**

  Add separate barriers `test_stale_after_persona_read_cannot_apply`, `test_stale_after_graph_read_cannot_apply`, `test_stale_after_runtime_resolve_cannot_apply`, `test_stale_after_frame_prepare_cannot_apply`, `test_repeated_cancel_drains_before_next_owner`, and `test_shutdown_drains_before_profile_db_closes`.

- [ ] **Step 8: Wire app construction and shutdown**

  Construct the controller immediately after `_wire_character_persona_services()` with the local Persona service, profile database/repository factory, config snapshot, and loop-safe app scheduler. In `_shutdown_app_owned_lifecycles`, await Buddy shutdown before closing the database or screen stack.

- [ ] **Step 9: Run Task 2 GREEN and mutations**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Buddy Tests/Persona_Visual/test_persona_visual_runtime.py Tests/UI/test_console_runtime_ownership.py
  ```

  Expected: pass. Remove each post-await fence, replace shield/drain with raw `to_thread`, and collapse child-cancel into successful `None` one at a time; the corresponding tests must fail.

- [ ] **Step 10: Commit Task 2**

  ```bash
  git add tldw_chatbook/Persona_Buddy tldw_chatbook/app.py Tests/Persona_Buddy Tests/UI/test_console_runtime_ownership.py
  git commit -m "feat: resolve Persona Buddy visual states"
  ```

## Task 3: Mount the native floating view dynamically

**Files:**

- Create: `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py`
- Modify: `tldw_chatbook/UI/Navigation/base_app_screen.py:202-230`
- Modify: `tldw_chatbook/UI/Navigation/base_app_screen.py:350-375`
- Modify: `tldw_chatbook/app.py` near the screen-navigation helpers
- Run: `tldw_chatbook/css/build_css.py`
- Regenerate: `tldw_chatbook/css/widget_defaults_self.tcss`
- Regenerate: `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Review/restore-or-stage: `tldw_chatbook/css/screen_css_self.tcss`
- Review/restore-or-stage: `tldw_chatbook/css/screen_css_scoped.tcss`
- Review/restore-or-stage: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_persona_buddy_widget.py`
- Create: `Tests/UI/test_persona_buddy_app_mount.py`
- Create: `Tests/Live/persona_buddy_terminal_probe.py`
- Modify: `Tests/UI/test_shell_chrome_contract.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Load the final UI craft references**

  Immediately before editing visible UI, read Impeccable `reference/new-work.md` and `reference/craft-floor.md`. Do not rerun the already-completed context command. Preserve the terminal-native Neon Workbench direction and semantic tokens.

- [ ] **Step 2: Write dynamic-mount REDs**

  Add `test_enable_mounts_on_current_screen_without_navigation`, `test_disable_unmounts_without_navigation`, `test_close_removes_only_current_generation`, `test_reopen_mounts_without_navigation`, and `test_recompose_unsubscribes_and_remounts_both_directions`. Drive:

  ```python
  await app.reconcile_persona_buddy_view()
  assert len(screen.query(PersonaBuddyWidget)) == 1
  ```

  The app, not the controller, identifies the active `BaseAppScreen`. `BaseAppScreen.reconcile_persona_buddy_view()` mounts/removes one view, releases capture before removal, and verifies its own screen/view generation after every await.

- [ ] **Step 3: Run dynamic-mount REDs**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_app_mount.py
  ```

  Expected: missing widget/reconciler failures.

- [ ] **Step 4: Write geometry/input/paint REDs**

  Add `test_overlay_paints_without_flow_or_fr_budget`, `test_drag_and_resize_are_viewport_bounded`, `test_keyboard_move_resize_reset_collapse_close`, `test_tiny_viewport_uses_labelled_compact_control`, `test_state_repaint_never_steals_focus`, `test_animation_cells_change_then_freeze_hidden_collapsed`, `test_reduced_motion_paints_static_frame`, and `test_modal_covers_buddy_hit_target`. Use real bundled CSS, predicate-based waits with bounded deadlines, compositor cell/glyph assertions, and no bundle-less harness.

- [ ] **Step 5: Implement the lightweight view and mount reconciler**

  Implement:

  ```python
  class PersonaBuddyWidget(Widget):
      BINDINGS = [("h", "move_left", "Move left"), ("j", "move_down", "Move down"), ("k", "move_up", "Move up"), ("l", "move_right", "Move right"), ("H", "shrink_width", "Narrower"), ("L", "grow_width", "Wider"), ("J", "grow_height", "Taller"), ("K", "shrink_height", "Shorter"), ("0", "reset_geometry", "Reset"), ("c", "toggle_collapse", "Collapse"), ("x", "close", "Close")]
      def refresh_from_controller(self) -> None: ...
      def on_mouse_down(self, event: events.MouseDown) -> None: ...
      def on_mouse_move(self, event: events.MouseMove) -> None: ...
      def on_mouse_up(self, event: events.MouseUp) -> None: ...
  ```

  Use `absolute_offset`, capture/release, semantic tokens, labelled buttons, `position: absolute`, and `overlay: screen`. Snapshot/reconciliation polling stays active for every mounted view so restore and state changes remain observable; only frame advancement pauses while hidden or collapsed. If unavailability unmounts the view, Persona restore/publication explicitly calls the app reconciler after controller refresh. Geometry changes update the in-memory controller immediately; config persistence occurs once on mouse-up or one completed keyboard action through the same serialized off-loop preference operation, never on every move.

- [ ] **Step 6: Implement modal/navigation/recompose safety**

  `BaseAppScreen.on_mount` reconciles once; navigation creates a fresh view. `on_unmount` releases capture and removes only that screen's view. `TldwCli.reconcile_persona_buddy_view()` no-ops for splash/auth/recovery and when the active screen is modal/non-`BaseAppScreen`; Textual modal layering remains above the Buddy. Late reconcile removes only the stale view it created.

- [ ] **Step 7: Regenerate and review CSS**

  First preserve the known baseline mismatch, then run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_css_bundle_sync_guard.py
  ```

  Expected after regeneration: pass. Review all five generated outputs. Stage every content change attributable to the known baseline reconciliation or Buddy declaration; explicitly restore timestamp-only `tldw_cli_modular.tcss` churn and any byte-identical screen sheet. Require `git status --short tldw_chatbook/css` to show only reviewed generated changes. Do not edit `Tests/UI/test_css_bundle_sync_guard.py` unless a new guard is genuinely necessary.

- [ ] **Step 8: Add the real-terminal probe**

  Create a bounded POSIX PTY subprocess probe that starts a minimal production-CSS Buddy app, sends real SGR mouse down/move/up and keyboard bytes, opens a modal, navigates/replaces the screen, resizes the PTY, restarts with the same isolated config, and writes a JSON report. It must print `PASS persona_buddy_terminal` only when drag, resize, keys, focus, modal hit testing, navigation, viewport clamp, and geometry restore all pass; capability-skip Windows only.

- [ ] **Step 9: Run Task 3 GREEN and mutations**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_widget.py Tests/UI/test_persona_buddy_app_mount.py Tests/UI/test_shell_chrome_contract.py Tests/UI/test_screen_navigation.py Tests/UI/test_css_bundle_sync_guard.py
  ```

  Expected: pass. Mutate `overlay: screen`, clamping, focus guard, modal exclusion, screen/view generation, and capture release individually; each named barrier must fail.

- [ ] **Step 10: Commit Task 3**

  ```bash
  git add tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py tldw_chatbook/UI/Navigation/base_app_screen.py tldw_chatbook/app.py tldw_chatbook/css/widget_defaults_self.tcss tldw_chatbook/css/widget_defaults_scoped.tcss tldw_chatbook/css/screen_css_self.tcss tldw_chatbook/css/screen_css_scoped.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_persona_buddy_widget.py Tests/UI/test_persona_buddy_app_mount.py Tests/Live/persona_buddy_terminal_probe.py Tests/UI/test_shell_chrome_contract.py Tests/UI/test_screen_navigation.py
  git commit -m "feat: mount floating Persona Buddy view"
  ```

## Task 4: Add explicit Personas Workbench ownership actions

**Files:**

- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py:164-250`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:3922-4010`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:5822-5915`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py:6922-7030`
- Modify: `Tests/UI/test_personas_inspector_pane.py`
- Modify: `Tests/UI/test_personas_workbench_state.py`
- Modify: `Tests/UI/test_personas_persona_visual_authoring.py`
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Write inspector/action REDs**

  Add typed `PersonaBuddyActionRequested(action, source, persona_id, revision)` tests for explicit `Use for Buddy`, `Show Buddy`, `Close Buddy`, and `Disable Buddy`. Assert active local Persona eligibility and exact server tooltip `Save a local copy first`; ordinary row/highlight selection must emit no Buddy action.

- [ ] **Step 2: Write lifecycle/restart/export REDs**

  Add `test_workbench_highlight_never_retargets_buddy`, `test_disabled_deleted_missing_persona_hides_but_preserves_enabled_selection`, `test_restore_reresolves_same_selection`, `test_explicit_replacement_is_required`, `test_restart_restores_selection_open_collapsed_and_geometry`, `test_persona_json_export_excludes_buddy_preferences`, and `test_visual_publication_invalidates_bound_buddy_old_and_new_identity_only`.

- [ ] **Step 3: Run the Workbench REDs**

  Run focused named files. Expected: missing typed messages/actions and no controller integration.

- [ ] **Step 4: Implement explicit Workbench commands**

  Add labelled inspector buttons and typed messages. In `PersonasScreen`, snapshot profile/source/Persona ID/revision, validate active local authority, then call `await app.persona_buddy_controller.update_preferences(...)`. That controller method registers the sole owned operation before its first await, shield/drains `asyncio.to_thread(persist_persona_buddy_preferences, ...)`, revalidates profile/selection/preferences generations before and after persistence, applies the in-memory state only after a confirmed durable result, reconciles a late commit after outer cancellation, and releases serialization only after outcome handling. After it returns current, await `app.reconcile_persona_buddy_view()`. Server-backed/ineligible actions remain disabled with the exact tooltip; failures use fixed path-free copy.

- [ ] **Step 5: Integrate lifecycle and publication refresh**

  Persona disable/delete notifies the controller to mark the exact selection unavailable without clearing enabled/selection. Restore/replacement re-resolves only after authority checks. After Persona Visual publication, invalidate the Buddy only when old or new full identity matches its exact binding; unrelated entries remain, and failure/cancel invalidates nothing.

- [ ] **Step 6: Run Task 4 GREEN and mutations**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_inspector_pane.py Tests/UI/test_personas_workbench_state.py Tests/UI/test_personas_persona_visual_authoring.py Tests/UI/test_personas_workbench.py -k "buddy or persona_visual"
  ```

  Expected: pass. Mutate explicit-selection, server eligibility, preservation, config restart, export exclusion, and targeted invalidation guards one at a time and record failures.

- [ ] **Step 7: Commit Task 4**

  ```bash
  git add tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_inspector_pane.py Tests/UI/test_personas_workbench_state.py Tests/UI/test_personas_persona_visual_authoring.py Tests/UI/test_personas_workbench.py
  git commit -m "feat: manage Persona Buddy from Workbench"
  ```

## Task 5: Publish trusted Console lifecycle leases

**Files:**

- Create: `tldw_chatbook/Persona_Buddy/console_adapter.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py:628-706`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:3747-3840`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:13057-13160`
- Modify: `tldw_chatbook/Chat/console_fleet_wake.py:305-350`
- Modify: `tldw_chatbook/Chat/console_fleet_wake.py:470-650`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:8218-8255`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:19403-19440`
- Create: `Tests/Persona_Buddy/test_persona_buddy_console_adapters.py`
- Modify: `Tests/UI/test_console_run_gate.py`
- Modify: `Tests/UI/test_console_realtime_wiring.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`
- Modify: `Tests/Chat/test_console_realtime_loop.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_fleet_wake.py`

- [ ] **Step 1: Write the pure adapter REDs**

  Pin this producer mapping:

  | Producer | Source/owner | State | Release |
  | --- | --- | --- | --- |
  | `ConsoleChatController._set_run_state` | `console-run` / session+run generation | validating/checking/retrying→`thinking`, streaming→`speaking`, failed→`error` | idle/completed/stopped/blocked or exact run replacement |
  | approval registry changes | `approval` / session+round id | `approval_needed` | exact round settle/revoke/terminal |
  | `ConsoleAgentBridge.on_step` | `tool` / run id+tool-call sequence | `tool_running` on `STEP_TOOL_CALL` | matching `STEP_TOOL_RESULT`, error, cancellation, or run terminal |
  | fleet-wake pending/delivery | `wake` / conversation+run id | `wake_armed` | delivered/cleared/disposed; priority applies only when live absent/idle |
  | realtime FSM | `voice` / session+loop generation | live→`listening`, thinking→`thinking`, speaking→`speaking`, connecting/reconnecting→`offline`, idle→release | `ExitLoop`/replacement/unmount |
  | public trusted API | `explicit`/`authored` + caller owner | safe custom/built-in | expiry or exact token release |

  Test exact underscore keys, overlapping sessions/runs/tools/rounds, replacement, and no model text fields.

- [ ] **Step 2: Run adapter REDs**

  Run the new adapter file. Expected: missing adapter/events.

- [ ] **Step 3: Implement content-free adapter events**

  Define frozen/slotted events carrying only source, owner token, safe state, terminal flag, and optional monotonic expiry. Mapping functions call controller acquire/release and never accept prompt, assistant text, tool args/results, paths, provider payloads, or model directives.

- [ ] **Step 4: Wire actual run/approval producers**

  Give the app-owned `ConsoleChatController` a Buddy sink at creation through `ConsoleRuntime`; do not capture a screen. Emit run events at `_set_run_state` and approval events at the registry's actual insert/remove/revoke/terminal points. A controller-only construction with no sink remains a no-op.

- [ ] **Step 5: Wire actual tool and wake producers**

  `ConsoleRuntime.ensure_agent_bridge` supplies an app-owned thread-safe sink. `ConsoleAgentBridge.on_step` emits tool start/result with run identity; all terminal/finally paths release remaining tokens. `ConsoleFleetWakeCoordinator` mirrors exact pending/delivering membership into per-conversation/run wake leases and releases only settled owners.

- [ ] **Step 6: Wire realtime and trusted public APIs**

  In `_console_realtime_mode_changed`/exit, publish exact session+loop generation events and release the preceding exact token before acquiring its replacement. Screen unmount releases only that screen's realtime token. Keep `set_timed_state` and `set_authored_trigger` as explicit trusted app APIs; do not wire streaming `Emote:` parsing.

- [ ] **Step 7: Run actual-producer GREENs**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Buddy/test_persona_buddy_console_adapters.py Tests/UI/test_console_run_gate.py Tests/UI/test_console_realtime_wiring.py Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_realtime_loop.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_agent_bridge.py -k "persona_buddy or tool_step"
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_fleet_wake.py -k "persona_buddy or pending or deliver"
  ```

  Expected: pass. Tests must drive the real producer callbacks, not only pure mappers.

- [ ] **Step 8: Run Task 5 mutations**

  Remove exact owner matching, terminal release, tool-result release, wake idle gating, realtime generation, and no-sink no-op one at a time; each corresponding producer test must fail.

- [ ] **Step 9: Commit Task 5**

  ```bash
  git add tldw_chatbook/Persona_Buddy/console_adapter.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_fleet_wake.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Persona_Buddy/test_persona_buddy_console_adapters.py Tests/UI/test_console_run_gate.py Tests/UI/test_console_realtime_wiring.py Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_realtime_loop.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_fleet_wake.py
  git commit -m "feat: drive Persona Buddy from trusted lifecycle state"
  ```

## Task 6: Isolated verification, live proof, and closeout

**Files:**

- Modify: `backlog/tasks/task-19055 - Add-opt-in-app-wide-floating-Persona-Buddy.md`
- Create: `Tests/Architecture/test_persona_buddy_boundary.py`
- Verify: `Tests/Live/persona_buddy_terminal_probe.py`
- Verify: `tldw_chatbook/css/widget_defaults_self.tcss`
- Verify: `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Verify: `tldw_chatbook/css/screen_css_self.tcss`
- Verify: `tldw_chatbook/css/screen_css_scoped.tcss`
- Verify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify only if a reviewed touched delta requires it: `Docs/security/production-diagnostic-inventory.json`
- Modify a relevant `backlog/docs/lessons-*.md` only for a genuine incident-backed reusable lesson

- [ ] **Step 1: Add architecture/privacy RED then GREEN**

  Add tests proving `Persona_Buddy/controller.py`, `preferences.py`, and `rendering.py` have no Textual/UI imports; controller source contains no model/Emote parser; Buddy preferences do not occur in Persona/Actor Pack exporters; public snapshots/logs exclude path/bytes/prompt/provider/token fields. Temporarily add one forbidden UI import to prove RED, then restore and record GREEN.

- [ ] **Step 2: Establish isolated roots before interpreter import**

  ```bash
  TASK19055_ROOT="$(mktemp -d /private/tmp/tldw-task19055.XXXXXX)"
  mkdir -p "$TASK19055_ROOT/home" "$TASK19055_ROOT/config" "$TASK19055_ROOT/data" "$TASK19055_ROOT/cache"
  env HOME="$TASK19055_ROOT/home" XDG_CONFIG_HOME="$TASK19055_ROOT/config" XDG_DATA_HOME="$TASK19055_ROOT/data" XDG_CACHE_HOME="$TASK19055_ROOT/cache" TLDW_CONFIG_PATH="$TASK19055_ROOT/config/config.toml" TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/test_probe_import_provenance.py
  ```

  Expected: pass and every imported `tldw_chatbook` module path starts with this assigned worktree.

- [ ] **Step 3: Run the combined touched-component gate only**

  In one shell, define a literal isolated runner and run:

  ```bash
  : "${TASK19055_ROOT:?Run Step 2 and retain its isolated root}"
  run_task19055() { env HOME="$TASK19055_ROOT/home" XDG_CONFIG_HOME="$TASK19055_ROOT/config" XDG_DATA_HOME="$TASK19055_ROOT/data" XDG_CACHE_HOME="$TASK19055_ROOT/cache" TLDW_CONFIG_PATH="$TASK19055_ROOT/config/config.toml" TLDW_TEST_MODE=1 "$@"; }
  run_task19055 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Persona_Buddy \
    Tests/Persona_Visual/test_persona_visual_runtime.py \
    Tests/UI/test_persona_buddy_widget.py \
    Tests/UI/test_persona_buddy_app_mount.py \
    Tests/UI/test_personas_inspector_pane.py \
    Tests/UI/test_personas_workbench_state.py \
    Tests/UI/test_personas_persona_visual_authoring.py \
    Tests/UI/test_console_run_gate.py \
    Tests/UI/test_console_realtime_wiring.py \
    Tests/UI/test_console_runtime_ownership.py \
    Tests/UI/test_shell_chrome_contract.py \
    Tests/UI/test_screen_navigation.py \
    Tests/UI/test_css_bundle_sync_guard.py \
    Tests/Chat/test_console_realtime_loop.py \
    Tests/Architecture/test_persona_buddy_boundary.py
  ```

  Then run the large touched files with their focused Buddy nodes:

  ```bash
  run_task19055 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py -k "persona_buddy or persona_visual"
  run_task19055 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_agent_bridge.py -k "persona_buddy or tool_step"
  run_task19055 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_fleet_wake.py -k "persona_buddy or pending or deliver"
  ```

  Expected: all pass. Do not run the full suite.

- [ ] **Step 4: Run the bounded real-terminal probe**

  ```bash
  env HOME="$TASK19055_ROOT/home" XDG_CONFIG_HOME="$TASK19055_ROOT/config" XDG_DATA_HOME="$TASK19055_ROOT/data" XDG_CACHE_HOME="$TASK19055_ROOT/cache" TLDW_CONFIG_PATH="$TASK19055_ROOT/config/config.toml" TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Tests/Live/persona_buddy_terminal_probe.py --report "$TASK19055_ROOT/persona-buddy-terminal.json"
  ```

  Expected: exit 0, stdout `PASS persona_buddy_terminal`, and JSON booleans true for drag, resize, keyboard, focus, modal, navigation, viewport clamp, and restart restore.

- [ ] **Step 5: Run Impeccable detector exactly once**

  After the final visible UI edit, run once:

  ```bash
  node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --target tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py
  ```

  Expected: exit 0 and `[]`. Address every finding without rerunning the detector, then rerun only normal affected UI tests.

- [ ] **Step 6: Run exact static/governance gates**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Persona_Buddy tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/Persona_Buddy Tests/UI/test_persona_buddy_widget.py Tests/UI/test_persona_buddy_app_mount.py Tests/Architecture/test_persona_buddy_boundary.py Tests/Live/persona_buddy_terminal_probe.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Persona_Buddy tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/Persona_Buddy Tests/UI/test_persona_buddy_widget.py Tests/UI/test_persona_buddy_app_mount.py Tests/Architecture/test_persona_buddy_boundary.py Tests/Live/persona_buddy_terminal_probe.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/app.py tldw_chatbook/UI/Navigation/base_app_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_fleet_wake.py Tests/UI/test_personas_inspector_pane.py Tests/UI/test_personas_workbench_state.py Tests/UI/test_personas_persona_visual_authoring.py Tests/UI/test_personas_workbench.py Tests/UI/test_console_run_gate.py Tests/UI/test_console_realtime_wiring.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_shell_chrome_contract.py Tests/UI/test_screen_navigation.py Tests/Chat/test_console_realtime_loop.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_fleet_wake.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/app.py tldw_chatbook/UI/Navigation/base_app_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_fleet_wake.py Tests/UI/test_personas_inspector_pane.py Tests/UI/test_personas_workbench_state.py Tests/UI/test_personas_persona_visual_authoring.py Tests/UI/test_personas_workbench.py Tests/UI/test_console_run_gate.py Tests/UI/test_console_realtime_wiring.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_shell_chrome_contract.py Tests/UI/test_screen_navigation.py Tests/Chat/test_console_realtime_loop.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_fleet_wake.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Persona_Buddy tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py tldw_chatbook/app.py tldw_chatbook/UI/Navigation/base_app_screen.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_fleet_wake.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Architecture/test_persona_buddy_boundary.py Tests/Architecture/test_persistent_diagnostic_inventory.py Tests/test_persistent_diagnostic_boundary.py Tests/test_database_path_privacy.py
  git diff --check
  ```

  Expected: new/small files pass full Ruff and format; every touched legacy file passes fatal Ruff and compile. The legacy formatter command must reproduce exactly the recorded 17-would-reformat/5-already-formatted baseline with no new file entering the would-reformat set; any branch-owned formatter regression blocks Done. If the diagnostic inventory differs, review the semantic delta; regenerate only a Buddy-owned fixed-category delta. Any failing scoped gate blocks Done.

- [ ] **Step 7: Review baseline versus branch and close only when green**

  Compare final scoped results with the preserved `727 passed / 1 pre-existing CSS mismatch` baseline. The CSS mismatch must now be repaired because Buddy touched its source bundle. If any touched gate, real-terminal assertion, detector finding, mutation, static/privacy/architecture/governance gate, or branch-owned regression remains, keep TASK-19055 In Progress and report the blocker.

- [ ] **Step 8: Record evidence and mark Done**

  Only after Step 7 is green: check all eight ACs, add concise Implementation Notes with RED→GREEN nodes, mutation results, worktree/isolation evidence, real-terminal JSON, Impeccable output, exact focused counts, baseline handling, ADR-074, scope exclusions, and the explicit no-full-suite instruction; then set status Done.

- [ ] **Step 9: Commit closeout**

  ```bash
  git add "backlog/tasks/task-19055 - Add-opt-in-app-wide-floating-Persona-Buddy.md" Tests/Architecture/test_persona_buddy_boundary.py
  git commit -m "docs: complete Persona Buddy delivery"
  ```
