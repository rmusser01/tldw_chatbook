# Persona Buddy Pet-Only Chrome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Every behavior starts with `superpowers:test-driven-development`; use `superpowers:verification-before-completion` before each commit and completion claim. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Persona Buddy's labelled floating panel with a tightly fitted pet, two icon controls, fixed actionable-alert text, and a real folded thumbnail while preserving the existing controller, persistence, lifecycle, and rendering authority.

**Architecture:** Keep the change in the existing `PersonaBuddyWidget` view boundary. The widget accepts only the direct, post-fence result of `resolve_current_visual()`, stores a frozen render ticket with its exact authority and stable maximum frame box, and derives display-only geometry from that ticket without persisting or feeding fitted dimensions back into resolution. Existing Textual `Button` controls, controller preference operations, renderer output, and app-owned cancellation/reconcile lifecycle remain authoritative.

**Tech Stack:** Python 3.11+, Textual 8, Rich Pixels, existing Persona Buddy/Persona Visual runtime, pytest/Pilot, consolidated Textual CSS builder, POSIX PTY UAT.

**ADR required:** no

**ADR path:** Existing [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)

**Reason:** ADR-074 already decides local Persona Visual ownership, immutable runtime identity, app-owned Buddy state, persistence, and native Textual rendering. This task only distills disposable view chrome and fit behavior.

---

## Fixed contracts and file map

- Modify `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py` for accepted-render authority, fitted display geometry, pet-only composition, icon controls, alert replacement, folded thumbnail, drag hit testing, and compact fallback. Do not create another production module.
- Modify `Tests/UI/test_persona_buddy_widget.py` for born-RED real-CSS/Pilot behavior, authority barriers, mutations, animation geometry, alerts, controls, folded rendering, and constrained viewports.
- Modify `Tests/Live/persona_buddy_terminal_probe.py` only to report and exercise the approved normal/alert/folded/constrained visual states and to save stable screenshot-ready terminal captures.
- Regenerate and review `tldw_chatbook/css/widget_defaults_self.tcss`; the builder may touch all five generated CSS files, but keep only semantic output changes and restore timestamp-only churn in `tldw_chatbook/css/tldw_cli_modular.tcss`.
- Reuse `Tests/UI/test_persona_buddy_app_mount.py`, `Tests/Persona_Buddy/test_persona_buddy_resolution.py`, `Tests/Architecture/test_persona_buddy_boundary.py`, and `Tests/Live/test_persona_buddy_terminal_probe.py` as regression gates. Discover any additional canonical paths with `rg --files Tests | rg 'persona_buddy'`; do not invent a missing path.
- No controller, renderer, preference schema, database, Persona Visual manifest, provider, Workbench, Console, icon dependency, tooltip subsystem, or window-manager change is authorized unless a born-RED test proves the approved contract cannot be met at the widget boundary; stop and report before expanding scope.
- Before the first UI edit, read `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/reference/craft-floor.md`. The Impeccable context command has already been run for this design and must not be rerun.

## Task 1: Pin accepted-render authority and stable fit geometry

**Files:**

- Modify: `Tests/UI/test_persona_buddy_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py:143-430,521-595`

- [ ] **Step 1: Write the stable-box and direct-result REDs**

  Extend the fake prepared frames with exact `width`, `height`, `paint_digest`, and cache fields. Add:

  - `test_accepted_visual_uses_one_maximum_frame_box_without_jitter`
  - `test_single_frame_fits_to_content_without_persisting_dimensions`
  - `test_prior_budget_snapshot_visual_cannot_refit_current_view`
  - `test_prior_viewport_direct_result_cannot_replace_accepted_render`

  Use frames with `(width, height)` values `(8, 6)` and `(12, 10)` and assert the content box stays `(12, 5)` terminal cells across animation. Record controller preference writes and assert auto-fit causes none. Block `resolve_current_visual()`, change preferred geometry or viewport generation, publish a stale `snapshot.visual`, release the blocked call, and assert neither frame nor region changes.

- [ ] **Step 2: Run the authority REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_widget.py -k 'accepted_visual or maximum_frame_box or single_frame_fits or prior_budget or prior_viewport'
  ```

  Expected: fail because polling currently paints `snapshot.visual`, the fitted widget has no accepted result ticket, and display geometry follows saved panel dimensions.

- [ ] **Step 3: Add the smallest frozen accepted-render record**

  In the widget module add one private record, not a new abstraction layer:

  ```python
  @dataclass(frozen=True, slots=True)
  class _AcceptedRender:
      authority: tuple[object, ...]
      visual_identity: tuple[object, ...]
      visual: PersonaBuddyVisualSnapshot = field(repr=False, compare=False)
      content_width: int
      content_height: int
  ```

  Add `_visual_identity(visual)` by extracting the incumbent identity tuple, and `_stable_content_box(visual)`:

  ```python
  width = max(_OPERABLE_WIDTH, *(frame.width for frame in visual.frames))
  height = max(_OPERABLE_HEIGHT, *((frame.height + 1) // 2 for frame in visual.frames))
  return width, height
  ```

  Reject non-exact positive integer dimensions as the existing fail-soft visual path. Keep the record repr/content-free.

- [ ] **Step 4: Accept only the direct post-fence result**

  In `_resolution_loop()`, capture one exact authority that includes controller/view generation, semantic state, preference/profile/viewport generation, selection, collapsed mode, and the deterministic requested render budget. After the existing await and all currentness checks, build `_AcceptedRender` from the local `visual` return value and assign it once. Do not call `refresh_from_controller()` to obtain the visual being accepted. Polling may refresh state/preferences but may not replace `_accepted_render` from `snapshot.visual`.

- [ ] **Step 5: Separate requested budget from fitted display size**

  Replace `_resolution_size()` with a deterministic budget derived from preferred geometry, one-cell boundary cost, viewport, and folded mode. Add `_display_geometry()` that clamps saved `x/y` using the accepted content box plus boundary only. `_apply_geometry()` uses display geometry; persistence continues to use `_working_preferences.geometry`. Never include the fitted widget region in resolution authority.

- [ ] **Step 6: Run GREEN and mutation proof**

  Run Step 2 and the full widget file. Then independently mutate (a) acceptance back to `snapshot.visual`, (b) removal of viewport/budget/mode from the authority, and (c) use of fitted region as the next render budget. Each named stale/feedback test must fail. Restore and rerun green.

- [ ] **Step 7: Commit Task 1**

  ```bash
  git diff --check
  git add tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py
  git commit -m "fix: pin Persona Buddy render authority"
  ```

## Task 2: Remove resting chrome and overlay icon controls

**Files:**

- Modify: `Tests/UI/test_persona_buddy_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py:41-240,430-660,596-812`
- Modify: `tldw_chatbook/css/widget_defaults_self.tcss` (generated)
- Review after build: `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Review after build: `tldw_chatbook/css/screen_css_self.tcss`
- Review after build: `tldw_chatbook/css/screen_css_scoped.tcss`
- Review/restore timestamp-only churn: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write the pet-only composition REDs**

  Add:

  - `test_resting_buddy_contains_pet_and_icons_without_default_words`
  - `test_single_frame_touches_every_inner_content_edge`
  - `test_icon_controls_have_exact_labels_tooltips_and_hit_regions`
  - `test_keyboard_focus_exposes_only_exact_action_label`
  - `test_pet_surface_drags_but_icon_buttons_do_not`

  Assert the mounted tree has no header, drag-handle, status, or hints widgets; compositor text excludes `Persona Buddy`, `Drag`, `Fold`, `Close`, `State`, `Visual pending`, and movement hints at rest. Assert button glyphs are `▾` and `×`, tooltips are `Fold` and `Close`, each button is at least three cells wide, the prepared frame occupies the complete inner content region, and any non-button pet cell starts drag.

- [ ] **Step 2: Run the chrome REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_widget.py -k 'resting_buddy or touches_every_inner or icon_controls or keyboard_focus or pet_surface_drags'
  ```

  Expected: fail on the current title/header/status/hints, word buttons, padded layout, and handle-only drag target.

- [ ] **Step 3: Reduce composition to the frame and two buttons**

  Replace the `Horizontal` header and text rows with:

  ```python
  yield Static("", id="persona-buddy-frame")
  yield Button("▾", id="persona-buddy-collapse", classes="persona-buddy-control")
  yield Button("×", id="persona-buddy-close", classes="persona-buddy-control")
  ```

  Set exact native tooltips (`Fold`/`Open`, `Close`) as mode changes. Remove `border_title`, default pending/unavailable prose, drag-label writes, and status/hint updates. Keep fixed alert text for Task 3 only.

- [ ] **Step 4: Replace panel CSS with pet-owned overlay CSS**

  Set widget/frame padding to zero, remove the header/status/hint rules, keep one-cell boundary, make the frame fill the content box, and absolutely overlay the two controls in the upper corners. Use at least width 3 for native button line padding. Match repository focus behavior (`background: $accent; text-style: bold underline; outline: none`) so the glyph is not obscured. Add a one-line transient focus-help `Static` only if Textual cannot expose the exact action label through the focused button without changing resting composition; it must be hidden unless a control has keyboard focus and must overlay rather than consume layout.

- [ ] **Step 5: Make all non-control pet cells draggable**

  In `on_mouse_down()`, retain the current button-region early return and lower-right resize test, then treat any remaining point inside the widget content region as drag. Remove the drag-handle query. Preserve focus, capture release, modal exclusion, navigation fences, and one persistence write on mouse-up.

- [ ] **Step 6: Run GREEN and control mutations**

  Run Step 2 plus the existing keyboard/drag/resize/capture/modal tests. Mutate each of: reintroduce one default word, reduce button width below three, remove the button early return, and restore heavy outline. Each corresponding test must fail. Restore and rerun green.

- [ ] **Step 7: Build and verify CSS outputs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_css_bundle_sync_guard.py Tests/UI/test_widget_css_consolidation.py
  git diff --check
  git status --short
  ```

  Expected: CSS synchronization passes. Review all five generated outputs; retain the Buddy block change in `widget_defaults_self.tcss`, retain any semantically required generated delta, and restore only timestamp-only modular churn.

- [ ] **Step 8: Commit Task 2**

  ```bash
  git add tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py tldw_chatbook/css/widget_defaults_self.tcss
  git commit -m "feat: distill Persona Buddy to pet-only chrome"
  ```

  Add another generated CSS path only if Step 7 proves its content changed semantically.

## Task 3: Add fixed actionable alerts and real folded thumbnail

**Files:**

- Modify: `Tests/UI/test_persona_buddy_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py`
- Modify if generated CSS changes: `tldw_chatbook/css/widget_defaults_self.tcss`

- [ ] **Step 1: Write the alert allow-list REDs**

  Add `test_only_actionable_states_replace_pet_with_fixed_path_free_text`, parametrized over:

  ```python
  {
      "approval_needed": "Approval needed",
      "error": "Error",
      "offline": "Offline",
  }
  ```

  Assert the exact label replaces—not overlays—the portrait, uses at most two centered rows inside the existing stable box, and does not resize the widget. Parametrize idle/thinking/speaking/listening/tool_running/wake_armed/explicit/authored/custom controls and assert all remain wordless. Feed hostile `state_source`, owner, reason, and arbitrary state text and assert none reaches compositor output. Clear the alert and assert the latest current accepted frame returns.

- [ ] **Step 2: Run the alert REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_widget.py -k 'actionable_states or non_actionable_states or alert_restores'
  ```

  Expected: fail because the current widget always paints a status row and has no replacement alert surface.

- [ ] **Step 3: Implement one exact alert mapping**

  Add only:

  ```python
  _ACTIONABLE_ALERTS = {
      "approval_needed": "Approval needed",
      "error": "Error",
      "offline": "Offline",
  }
  ```

  `_paint_frame()` chooses the mapped fixed label before the accepted frame and otherwise paints the accepted frame or an empty string. Do not interpolate reason/source/owner/provider/tool/assistant content. Apply semantic classes for color while retaining the text as the carrier.

- [ ] **Step 4: Write the folded-thumbnail and constrained-fallback REDs**

  Add:

  - `test_folded_mode_resolves_real_thumbnail_under_distinct_authority`
  - `test_folded_thumbnail_is_static_and_uses_open_close_icons`
  - `test_only_effective_area_below_10x4_uses_two_button_fallback`
  - `test_undersized_asset_uses_only_10x4_operable_minimum`

  Assert folding triggers one resolution at a fixed folded budget distinct from full mode, carries the same graph/cache provenance, paints a real frame, freezes animation, changes `▾` to `▴` with tooltip `Open`, and retains `×`. Verify preferred and viewport constraints independently: `10x4` still paints a thumbnail, while `9x4` and `10x3` paint only both icon buttons. A one-pixel fixture centers inside exactly `10x4` without extra widget growth.

- [ ] **Step 5: Run the folded REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_persona_buddy_widget.py -k 'folded_mode or folded_thumbnail or effective_area or undersized_asset'
  ```

  Expected: fail because current collapsed mode hides the portrait and uses a text strip; compact fallback thresholds are based on the old 28x8 panel.

- [ ] **Step 6: Implement folded budget and exact fallback threshold**

  Make mode part of resolution authority. For collapsed mode return the fixed 10x4 content budget; for normal mode use the preferred/viewport-derived content budget. Keep animation frozen while collapsed. Replace the old default-panel compact test with `available_width < 10 or available_height < 4`; in compact fallback keep both icon buttons operable and hide only the pet/alert surface. Remove the old one-button minimal path.

- [ ] **Step 7: Run GREEN and mutations**

  Run Steps 2 and 5 plus the full widget test. Mutate (a) alert mapping to accept arbitrary state, (b) folded authority to reuse full mode, (c) folded paint to blank/text, and (d) the threshold to trigger at `10x4`. Each named test must fail. Restore and rerun green.

- [ ] **Step 8: Rebuild CSS if needed and commit Task 3**

  Run the Task 2 CSS builder/sync commands when `BUNDLED_CSS` changed. Then:

  ```bash
  git diff --check
  git add tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py
  git add tldw_chatbook/css/widget_defaults_self.tcss  # only if semantically changed
  git commit -m "feat: render Persona Buddy alerts and thumbnail"
  ```

## Task 4: Prove lifecycle regressions and actual-terminal presentation

**Files:**

- Modify: `Tests/Live/persona_buddy_terminal_probe.py`
- Modify if required by its existing focused tests: `Tests/Live/test_persona_buddy_terminal_probe.py`
- Test only: canonical Persona Buddy app-mount, runtime, architecture, CSS, and widget files discovered with `rg --files Tests | rg 'persona_buddy|css_bundle_sync|widget_css_consolidation'`

- [ ] **Step 1: Write PTY report REDs for four visual states**

  Extend the existing atomic report with content-free booleans and numeric regions for:

  - `pet_only_normal`
  - `fixed_alert_replaces_pet`
  - `real_folded_thumbnail`
  - `constrained_two_icons`

  Add screenshot/capture paths for normal, alert, folded, and constrained states. The CLI gains `--capture-dir`; it atomically writes exactly `normal.ansi`, `alert.ansi`, `folded.ansi`, and `constrained.ansi` under that caller-owned directory before returning success. Preserve the existing early-failure artifact, child return code, bounded sanitized tail, and all incumbent interaction checks.

- [ ] **Step 2: Run the PTY report RED**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Live/test_persona_buddy_terminal_probe.py
  ```

  Expected: fail because the report lacks the four approved visual-state proofs.

- [ ] **Step 3: Drive exact terminal states without sleeps**

  In the child app, expose only fixed test controls for the approved state transitions. The parent must wait on report predicates/compositor regions, not unconditional sleeps; use the existing bounded PTY draining. Capture terminal output after each predicate. Do not include paths, raw content, or provider/tool payloads in report fields.

- [ ] **Step 4: Run the isolated real PTY three times**

  ```bash
  UAT_ROOT="$(mktemp -d /private/tmp/tldw-task21000.XXXXXX)"
  mkdir -p "$UAT_ROOT/captures-1"
  env -u NO_COLOR HOME="$UAT_ROOT/home" XDG_CONFIG_HOME="$UAT_ROOT/config" XDG_DATA_HOME="$UAT_ROOT/data" XDG_CACHE_HOME="$UAT_ROOT/cache" TLDW_CONFIG_PATH="$UAT_ROOT/config/config.toml" TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Tests/Live/persona_buddy_terminal_probe.py --report "$UAT_ROOT/report-1.json" --capture-dir "$UAT_ROOT/captures-1"
  ```

  Repeat with `report-2.json`/`captures-2` and `report-3.json`/`captures-3`. Expected: exit 0, every existing and new check true, no traceback, and all four named ANSI captures non-empty per run. Preserve `$UAT_ROOT` through human screenshot review; do not commit captures.

- [ ] **Step 5: Run focused product regression gates**

  Discover the canonical test paths first, then run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/UI/test_persona_buddy_widget.py \
    Tests/UI/test_persona_buddy_app_mount.py \
    Tests/Persona_Buddy/test_persona_buddy_resolution.py \
    Tests/Architecture/test_persona_buddy_boundary.py \
    Tests/Live/test_persona_buddy_terminal_probe.py \
    Tests/UI/test_css_bundle_sync_guard.py \
    Tests/UI/test_widget_css_consolidation.py
  ```

  If a listed path is absent, use the canonical `rg --files` result and record the correction; do not silently skip the behavior. Expected: all touched/component gates green. Any scoped failure leaves TASK-21000 In Progress.

- [ ] **Step 6: Run scoped static, privacy, and governance gates**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py Tests/Live/persona_buddy_terminal_probe.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py Tests/Live/persona_buddy_terminal_probe.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py Tests/UI/test_persona_buddy_widget.py Tests/Live/persona_buddy_terminal_probe.py
  rg -n 'provider|prompt|assistant|tool_(arguments|result)|Traceback|/Users/|/private/' tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py
  git diff --check
  ```

  Expected: static checks green; privacy scan has no user/provider content or local path emission in the widget. Record any inherited baseline distinctly; do not call the task Done while a changed-component gate fails.

- [ ] **Step 7: Run the one permitted Impeccable review**

  The design-time Impeccable context command already ran and must not run again. After the final visible edit, run exactly once:

  ```bash
  node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --json tldw_chatbook/Widgets/Persona_Widgets/persona_buddy_widget.py
  ```

  Expected: exit 0 and JSON `[]`. Record stdout verbatim and do not rerun the detector. Address an actionable result through RED tests; retain the one-shot output and verify the repair with focused tests/static checks only.

- [ ] **Step 8: Present final screenshots for human UAT**

  Convert the final ANSI captures to the repository's incumbent image format/harness if needed and show normal, alert, folded, and constrained screenshots. Confirm visually: no resting words, only the pet and `▾`/`×`; alert replaces the pet; folded still reads as the pet; no decorative side gutters; controls remain legible and unobscured.

- [ ] **Step 9: Commit Task 4**

  ```bash
  git add Tests/Live/persona_buddy_terminal_probe.py Tests/Live/test_persona_buddy_terminal_probe.py
  git commit -m "test: prove pet-only Persona Buddy UAT"
  ```

  Stage only paths that actually changed.

## Task 5: Close out TASK-21000 only after every gate is green

**Files:**

- Modify: `backlog/tasks/task-21000 - Distill-Persona-Buddy-to-pet-only-chrome.md`
- Modify only if this task produced a genuinely reusable incident: `backlog/docs/lessons-live-verification.md` or `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Recheck task-ID and worktree provenance**

  Fetch current refs before integration, then verify TASK-21000 remains collision-free across reachable refs and worktrees. Confirm `git status --short` contains only this task's intended closeout edits.

- [ ] **Step 2: Record concise implementation notes directly**

  TASK-21000 is a five-digit ID, so do not use `backlog task 21000` or `backlog task edit 21000`: this repo's CLI silently no-ops or creates `task-task- - .md`. Use `apply_patch` on the exact task file to add `## Implementation Notes` covering the pet-only composition, exact alert allow-list, accepted-render/fit authority, folded thumbnail, generated CSS, focused tests, mutation proof, Impeccable one-shot result, PTY report paths, screenshots, and any explicitly inherited baseline. Link the approved spec and ADR-074; note that no ADR was added because boundaries did not change.

- [ ] **Step 3: Mark all acceptance criteria and Done only conditionally**

  Use `apply_patch` on the exact task file to change all seven AC boxes to checked and frontmatter `status: Done` only if every changed-component test/static/CSS/privacy/UAT gate passed and human screenshot UAT was accepted. Otherwise leave it In Progress and record the exact blocker. Immediately run `git status --short backlog/` and verify no `backlog/tasks/task-task- - .md` artifact exists.

- [ ] **Step 4: Verify and commit closeout**

  ```bash
  sed -n '1,220p' 'backlog/tasks/task-21000 - Distill-Persona-Buddy-to-pet-only-chrome.md'
  test ! -e 'backlog/tasks/task-task- - .md'
  git diff --check
  git status --short
  git add 'backlog/tasks/task-21000 - Distill-Persona-Buddy-to-pet-only-chrome.md'
  git commit -m "docs: complete pet-only Persona Buddy chrome"
  ```

  Expected: the direct task file shows Done, seven checked ACs, Implementation Plan and Implementation Notes, exact dependency/reference metadata, and no unrelated file.
