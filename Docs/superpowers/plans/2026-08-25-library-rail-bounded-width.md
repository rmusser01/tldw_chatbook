# Library Rail Bounded Width Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Library destination one stable bounded-fractional navigation-rail contract while preserving custom widths, adaptive collapse behavior, and a usable route-general narrow-terminal escape path.

**Architecture:** A new pure Library width-policy module will be the single source of arithmetic and ordinary-layout contracts. `LibraryScreen` will own one normalized reader-preference snapshot, ordinary stage/eligibility state, and equality-guarded application; `LibraryRail` will only apply the reversible style contract it receives. Adaptive readers will retain their existing resolver and shell ownership but consume the same per-width requested rail value. One screen-owned pinned emergency return bar will serve every ordinary route below 64 cells.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/Textual Pilot, TCSS, Backlog.md

---

**Approved design:** `Docs/superpowers/specs/2026-08-25-library-rail-bounded-width-design.md`

**ADR required:** yes  
**ADR path:** `backlog/decisions/086-library-adaptive-reader-shell.md`  
**Reason:** This changes the long-lived cross-route Library shell geometry and responsive-stage contract; ADR-086 already owns that boundary and has been amended rather than duplicated.

## Task 0: Record the executable plan in Backlog before production work

**Files:**
- Modify: `backlog/tasks/task-22301 - Bound-Library-rail-width-across-modes.md`
- Add: `Docs/superpowers/plans/2026-08-25-library-rail-bounded-width.md`

- [ ] **Step 1: Add the concise in-task Implementation Plan**

Run:

```bash
backlog task edit 22301 --plan "1. Add a pure shared 3:13 bounded rail-width policy and exact custom-width resolver.\n2. Make adaptive readers consume one projected or custom requested rail width in every branch.\n3. Apply reversible equality-guarded ordinary rail contracts from one normalized settings snapshot.\n4. Add the route-general below-64 single-stage layout and one guarded pinned Library return action.\n5. Lock production box-model geometry, accessibility, and resize no-work behavior with mounted tests.\n6. Align fresh/reset defaults, Settings copy, and user documentation without migrating stored widths.\n7. Run targeted static/test verification and approved real-PTY UAT, then close the task.\n\nADR required: yes\nADR path: backlog/decisions/086-library-adaptive-reader-shell.md\nReason: ADR-086 owns the long-lived cross-route Library responsive shell contract and has been amended for this change."
```

- [ ] **Step 2: Verify Backlog rendered the plan and ADR block correctly**

```bash
backlog task 22301 --plain
```

Expected: status is `In Progress`, the seven-step `Implementation Plan` is present after Acceptance Criteria, and the amended ADR path is explicit.

- [ ] **Step 3: Commit the approved design/plan checkpoint**

```bash
git add 'backlog/tasks/task-22301 - Bound-Library-rail-width-across-modes.md' Docs/superpowers/plans/2026-08-25-library-rail-bounded-width.md
git commit -m "docs(library): plan bounded rail implementation"
```

## Task 1: Establish the pure shared width policy

**Files:**
- Create: `tldw_chatbook/Library/library_rail_width.py`
- Modify: `tldw_chatbook/Library/library_adaptive_reader_state.py`
- Create: `Tests/Library/test_library_rail_width.py`
- Modify: `Tests/Library/test_library_adaptive_reader_state.py`
- Modify: `Tests/Library/test_library_media_reader_state.py`

- [ ] **Step 1: Write failing projection and ordinary-contract tests**

Cover these exact oracles in `Tests/Library/test_library_rail_width.py`:

```python
@pytest.mark.parametrize(
    ("content_width", "expected"),
    [(1, 24), (127, 24), (128, 24), (152, 29), (163, 31), (165, 31), (178, 33),
     (181, 34), (10_000, 34)],
)
def test_project_default_library_width_uses_three_sixteenths_with_half_up_rounding(
    content_width: int, expected: int
) -> None:
    assert project_default_library_width(content_width) == expected
```

Also assert:

- `0` and negative widths raise `ValueError` rather than fabricating a rendered width.
- custom width normalization accepts/stores 24–48 independently of the 34-cell default ceiling.
- ordinary custom boundaries are exact: saved 35 at `W=74 -> 34`, `W=75 -> 35`; saved 48 at `W=64 -> 24`, `80 -> 40`, `87 -> 47`, `88 -> 48`.
- `ordinary_emergency_required(W)` is true exactly for positive `W < 64`; requesting an `ALONGSIDE` style contract there raises `ValueError` so the screen cannot accidentally render an impossible co-present state.
- default alongside mode expresses `display=True`, `width="3fr"`, minimum 24, maximum 34; custom alongside mode expresses one exact width/min/max.
- rail-only expresses `display=True`, `width="1fr"`, `min_width=0`, and `max_width=None`; hidden expresses `display=False`, `width=None`, `min_width=None`, and `max_width=None`.

- [ ] **Step 2: Run the focused tests and confirm they fail for missing policy**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_rail_width.py -q
```

Expected: FAIL during collection because `library_rail_width` and its public API do not exist.

- [ ] **Step 3: Implement the minimum pure policy**

Create immutable, UI-free policy types and helpers. Keep names explicit enough that callers cannot confuse saved and effective widths:

```python
LIBRARY_REFERENCE_WIDTH = 31
LIBRARY_MIN_WIDTH = 24
LIBRARY_DEFAULT_MAX_WIDTH = 34
LIBRARY_CUSTOM_MAX_WIDTH = 48
LIBRARY_CANVAS_MIN_WIDTH = 40
LIBRARY_EMERGENCY_WIDTH = LIBRARY_MIN_WIDTH + LIBRARY_CANVAS_MIN_WIDTH


def project_default_library_width(content_width: int) -> int:
    if content_width <= 0:
        raise ValueError("content_width must be positive")
    fractional = (3 * content_width + 8) // 16
    return max(LIBRARY_MIN_WIDTH, min(fractional, LIBRARY_DEFAULT_MAX_WIDTH))
```

Add one immutable `OrdinaryRailStyleContract` and one `OrdinaryRailPresentation` with mutually exclusive `ALONGSIDE`, `RAIL_ONLY`, and `HIDDEN` values. `ordinary_emergency_required(content_width)` is a separate pure predicate; `LibraryScreen`, not this style type, chooses rail-only versus hidden in emergency mode. `resolve_ordinary_rail_contract(...)` must:

- return the native `3fr`/24/34 contract for default alongside mode;
- transiently compress custom widths with `max(24, min(saved, W - 40))` when `W >= 64`;
- reject `ALONGSIDE` when `ordinary_emergency_required(W)` is true rather than applying the alongside formula;
- never mutate or return a replacement saved preference.

For absent inline declarations, the adapter will call Textual's `styles.clear_rule("width")`, `clear_rule("min_width")`, or `clear_rule("max_width")`; `None` in the pure contract is a clear instruction, not a CSS scalar.

Re-export compatibility names from `library_adaptive_reader_state.py` only where existing callers/tests need them. Change the unresolved/reference target from 28 to 31 without making it the rendered default projection.

- [ ] **Step 4: Run the pure policy and existing adaptive-state suites**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_rail_width.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the pure policy**

```bash
git add tldw_chatbook/Library/library_rail_width.py tldw_chatbook/Library/library_adaptive_reader_state.py Tests/Library/test_library_rail_width.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py
git commit -m "feat(library): centralize rail width policy"
```

## Task 2: Make adaptive resolution consume one requested width

**Files:**
- Modify: `tldw_chatbook/Library/library_adaptive_reader_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py`
- Modify: `Tests/Library/test_library_adaptive_reader_state.py`
- Modify: `Tests/Library/test_library_media_reader_state.py`
- Modify: `Tests/UI/test_library_adaptive_reader_shell.py`
- Modify: `Tests/UI/test_library_media_reader_shell.py`

- [ ] **Step 1: Add failing resolver tests for the shared requested-width invariant**

Add table-driven tests proving that, for each positive `W`, the resolver computes exactly one `requested_library_width`:

- custom off uses `project_default_library_width(W)` in full-fit, auto-collapse, explicit-priority, hysteresis, and final allocation paths;
- custom on uses the normalized saved 24–48 value in every one of those paths;
- an existing dormant saved 28 has no effect while custom is off and becomes an exact 28 request when custom is enabled;
- a zero-width pre-layout call returns the all-zero sentinel before priority inheritance or hysteresis and never seeds either.

Add the approved adaptive behavior matrix:

- default Notes Navigator Items priority gives Items widths `56/42/32/32` at terminal widths `120/100/80/60` in the production box model;
- Notes editor/work keeps Items open at 120/100 and collapsed at 80/60;
- explicit Library priority at `W=34` yields Library 24, Items 0, Work 0; at `W=33` it yields Library 23, Items 0, Work 0;
- explicit Items reopen uses the same projected Library request, transfers focus into Items when it fits, and transfers focus to the Items grip when responsive resolution collapses it;
- two grips continue to consume ten cells where both collapsed grips exist.

Parameterize custom widths `24`, `34`, `35`, and `48` across Notes, Conversations, and Media resolver profiles. Include below-minimum, above-maximum, non-integer, and missing inputs so normalization clamps/falls back to the documented 24–48 range without applying the 34-cell automatic ceiling.

- [ ] **Step 2: Run the focused tests and confirm stale fixed-width behavior fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py -q
```

Expected: FAIL where the resolver still reads the historical `preferences.library_width` or fixed 28 in default mode.

- [ ] **Step 3: Refactor the resolver without changing its ownership boundary**

At the start of each positive-width resolution, compute:

```python
requested_library_width = (
    preferences.library_width
    if preferences.custom_widths_enabled
    else project_default_library_width(width)
)
```

Return the all-zero sentinel before this calculation and before reading `previous.priority_pane`. Replace every requested-library-width use in full-fit checks, priority sizing, collapse decisions, hysteresis thresholds, and final layout construction with that local value. Preserve existing requested-versus-effective open state, five-cell grips, automatic/explicit priority, focus transfer, and bounded settling. Keep `LibraryAdaptiveReaderShell.sync_layout()` as the exact-width owner and equality-guard its display/width/min/max writes.

- [ ] **Step 4: Run adaptive unit and mounted shell tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py -q
```

Expected: PASS with no new recompose or preference write during `sync_layout()`.

- [ ] **Step 5: Commit adaptive integration**

```bash
git add tldw_chatbook/Library/library_adaptive_reader_state.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py
git commit -m "fix(library): project adaptive rail widths per shell"
```

## Task 3: Apply the reversible ordinary declaration adapter

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_rail.py`
- Modify: `Tests/UI/test_library_shell.py`

- [ ] **Step 1: Add failing mounted tests for declaration application and owner restoration**

Use `Tests.UI.consolidated_css.ConsolidatedCSSApp` and mount a production `LibraryRail` in a minimal grid host. Apply contracts directly and assert compositor regions, not just declarations:

- ordinary custom-off rail declares `3fr`, min 24, max 34 and renders the projected value within one compositor cell;
- simulate ordinary → adaptive → ordinary by applying the bounded contract, overwriting exact width/min/max as the adaptive shell does, invalidating ownership, and applying the same bounded contract again; it restores native declarations rather than leaking exact width;
- apply alongside → rail-only → hidden → alongside contracts and verify each clears/reapplies the exact declarations;
- applying the same contract twice performs no second style write, but returning from an adaptive owner restores the declarations even when the cached ordinary tuple is unchanged.

- [ ] **Step 2: Run the mounted tests and confirm current width drift/leakage**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py -q -k "ordinary_declaration or geometry_owner"
```

Expected: FAIL because `LibraryRail` has no declarative adapter and can retain an adaptive exact width.

- [ ] **Step 3: Add a declarative, equality-guarded rail adapter**

In `LibraryRail`, add one method that accepts the immutable ordinary contract and writes only changed values among:

- `display`
- `width`
- `min_width`
- `max_width`

It must not read configuration, resolve stages, recompose, or load data. Cache the last applied tuple, but skip writes only when both the cached tuple and the rail's current inline declarations match. Expose `invalidate_width_contract_owner()` for the screen/adaptive shell transition seam. When a contract member is `None`, use `styles.clear_rule(...)`. This prevents an unchanged cached ordinary tuple from hiding mutations made by the adaptive shell.

- [ ] **Step 4: Run the declaration/owner tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py -q -k "ordinary_declaration or geometry_owner"
```

Expected: PASS, including same-target ordinary → adaptive → ordinary restoration.

- [ ] **Step 5: Commit the adapter**

```bash
git add tldw_chatbook/Widgets/Library/library_rail.py Tests/UI/test_library_shell.py
git commit -m "feat(library): add reversible rail declarations"
```

## Task 4: Orchestrate ordinary width contracts from one preference snapshot

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_library_notes_reader.py`
- Modify: `Tests/UI/test_library_conversation_reader.py`

- [ ] **Step 1: Write failing tests for the normalized reader snapshot**

Assert each settings generation reads and normalizes `[library.reader]` exactly once for all ordinary/adaptive routes. A route switch within the generation reuses the snapshot; a settings-generation change refreshes it once without persisting anything.

- [ ] **Step 2: Run the snapshot tests and verify duplicate reads fail**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py -q -k "reader_snapshot"
```

Expected: FAIL because current route loaders normalize the shared section independently.

- [ ] **Step 3: Add the one-generation screen snapshot**

Replace per-route duplicate reader preference reads with one normalized snapshot. Do not change saved values. Route adapters consume the same snapshot and refresh it only when the existing settings generation changes.

- [ ] **Step 4: Run the snapshot tests and verify they pass**

Run the command from Step 2. Expected: PASS with one read and zero persistence calls.

- [ ] **Step 5: Write failing route-matrix and custom-restoration tests**

At one settled width, switch through Media, Chats, Notes, Prompts, Skills, Collections, Search/RAG, Import, Export, Study handoffs, and landing. Assert every co-present custom-off rail edge is stable within one compositor cell. For custom values `24`, `34`, `35`, and `48`, cover ordinary and adaptive routes, transient ordinary compression, adaptive collapse, and exact restoration after width returns. Initial mount, navigation settle, and route recompose must converge on the same result.

- [ ] **Step 6: Run the route matrix and verify current orchestration fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py -q -k "route_rail_matrix or custom_width_restoration"
```

Expected: FAIL until the screen applies the shared contract at every lifecycle seam.

- [ ] **Step 7: Apply contracts at existing lifecycle seams**

The screen chooses `ALONGSIDE`, `RAIL_ONLY`, or `HIDDEN`, resolves against settled `#library-shell-grid.content_region.width`, invalidates the rail adapter when geometry ownership changes, and applies the result on existing mount, route, manual-collapse, compact-stage, resize, and settings-generation seams. Add no worker or polling loop.

- [ ] **Step 8: Run the route matrix and verify it passes**

Run the command from Step 6. Expected: PASS.

- [ ] **Step 9: Write a failing resize no-work test**

Instrument rail/canvas recompose, destination loading, configuration reads/writes, worker creation, and persistence after initial settle. Pure width changes may patch declarations but must increment none of those counters; an unchanged effective tuple performs no declaration writes.

- [ ] **Step 10: Run the resize test, close any unintended side effect, and rerun**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py -q -k "resize_width_contract_no_work"
```

Expected before the final orchestration guard: FAIL on at least one redundant seam. Expected after minimal equality guards: PASS.

- [ ] **Step 11: Commit screen orchestration**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py
git commit -m "feat(library): bound ordinary rail geometry"
```

## Task 5: Add the route-general emergency single-stage escape path

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_emergency_return.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_library_honesty_accessibility.py`
- Modify: `Tests/UI/test_library_notes_reader.py`

- [ ] **Step 1: Write failing emergency-stage geometry and precedence tests**

Mount every ordinary route below 64 content cells and prove:

- landing/unactivated state enters rail-only; an active destination preserves canvas focus where possible;
- activating a rail row opens a canvas-only stage and focuses the route's normal entry target;
- rail, canvas, and return bar never overlap or leave blank reserved rail space.

Define and test precedence exactly:

1. modal/route guard/cancel/dirty/destructive/conflict/running state retains first refusal;
2. at positive `W < 64`, emergency geometry overrides ordinary co-presence and the visual geometry of Notes' `<120` compact takeover, but does not mutate either Notes requested compact stage or manual-collapse preference;
3. the emergency entry stage follows safe current focus, then an active destination, then rail-only fallback;
4. at `W >= 64`, emergency geometry ends and the still-requested Notes compact/manual-collapse state is reapplied before ordinary alongside fallback.

Cover entry and recovery with both manual rail collapse and each Notes compact stage.

- [ ] **Step 2: Run the geometry/precedence tests and confirm the missing emergency state**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_notes_reader.py -q -k "emergency_geometry or emergency_precedence"
```

Expected: FAIL because ordinary routes currently attempt impossible 24+40 co-presence below 64.

- [ ] **Step 3: Implement only the emergency stage resolver and screen state**

Add a screen-owned `RAIL_ONLY`/`CANVAS_ONLY` emergency stage. Use the pure threshold predicate, preserve requested Notes/manual state, apply the precedence above, and reuse the width contracts from Tasks 1–4. Do not mount the return widget yet.

- [ ] **Step 4: Run the geometry/precedence tests**

Run the command from Step 2. Expected: PASS with one visible stage and no overlap.

- [ ] **Step 5: Write failing pinned return-widget tests**

Prove exactly one bar exists in shared `#library-canvas`, outside route scroll; it is hidden with no reserved row at 64+; it is visible/focusable in canvas-only emergency mode; it uses `‹ Library` or ASCII `< Library`; and Enter/pointer activation posts one return request rather than changing state directly. Repeatedly switch ordinary routes and assert the same bar instance remains mounted while route content alone is replaced.

- [ ] **Step 6: Run the widget tests and confirm the widget is absent**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py -q -k "emergency_return_widget"
```

Expected: FAIL because no shared pinned return bar exists.

- [ ] **Step 7: Implement one presentation-only pinned return widget**

The widget owns label substitution plus focusable/enabled/visible presentation. It posts a single return-request message; it does not inspect route state or bypass screen guards.

Give `#library-canvas` two retained direct children: the emergency bar and a new `#library-canvas-route-content` host. Mount the bar once before that route-content host and hide it with `display: none` outside canvas-only emergency mode so it reserves no row. Update `_library_entry_canvas_owner()`, `_repair_library_entry_canvas_owner()`, `_replace_library_canvas_child()`, and every direct-child query/replacement seam to inspect or replace only children of `#library-canvas-route-content`; the retained bar must never be treated as route content or removed during navigation.

- [ ] **Step 8: Run the widget tests**

Run the command from Step 6. Expected: PASS.

- [ ] **Step 9: Write failing eligibility, Escape, footer, and F1 tests**

Add parameterized safe/unsafe top-level states. Safe canvas state advertises and enables `esc rail`; modal, nested Back, cancellation, dirty draft, destructive confirmation, conflict, and running mutation do not. Pointer/Enter and eligible Escape must reach the same guarded request. Binding-order tests prove emergency Escape is after route-specific bindings and before `library_list_focus_rail`.

- [ ] **Step 10: Implement one screen-owned eligibility projection and guarded transition**

In `LibraryScreen`, compute one immutable eligibility result used by all four surfaces:

1. return-bar visible/enabled/guarded state;
2. emergency Escape `check_action`;
3. footer shortcut copy;
4. F1 help copy.

Place the emergency Escape binding after route-specific Back/cancel/dirty/destructive bindings and before the broad list-focus-rail action. Route-specific handlers and modal screen-stack handling keep first refusal. Both pointer and keyboard requests call the same guarded transition. A successful return focuses the selected rail row and falls back to rail search.

- [ ] **Step 11: Run eligibility/accessibility tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py -q -k "emergency_eligibility or escape_rail or narrow_return_help"
```

Expected: PASS, with truthful action availability and no hidden-pane focus.

- [ ] **Step 12: Write failing generation/restoration tests**

Add one route-general `LibraryScreen._library_stage_interaction_generation` oracle. Every rail activation, canvas entry, emergency return, and Notes compact-stage interaction advances it through `_advance_library_stage_interaction()`. At 64+, restore the pre-entry focus/scroll tuple only if its captured generation is still current; a later interaction defeats restoration.

- [ ] **Step 13: Implement the route-general generation guard and rerun all emergency tests**

Keep Notes' existing internal focus generation for Notes-only callbacks, but use the new route-general generation as the sole validity owner for emergency restoration. Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py -q -k "emergency or narrow or escape or footer or help or route"
```

Expected: PASS, including stale-restoration rejection.

- [ ] **Step 14: Commit emergency navigation**

```bash
git add tldw_chatbook/Widgets/Library/library_emergency_return.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_library_notes_reader.py
git commit -m "feat(library): add narrow emergency return path"
```

## Task 6: Lock production box-model geometry and resize performance

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_css_build_integrity.py`

- [ ] **Step 1: Add the complete failing production-width matrix**

Under `ConsolidatedCSSApp`, first assert actual `#library-shell-grid.content_region.width` (`W`), then assert containment and the approved state:

| Terminal width | Expected `W` | Neutral ordinary | Neutral/work-owned adaptive |
| --- | ---: | --- | --- |
| 235 | 231 | rail 34 + canvas | Library 34, all panes open |
| 170 | 166 | rail 31 + canvas | Library 31, all panes open |
| 120 | 116 | rail 24 + canvas | Library collapsed; Items + Work open |
| 100 | 100 | rail 24 + canvas | Library collapsed; Items + Work open |
| 80 | 80 | rail 24 + canvas | Library + Items collapsed |
| 60 | 60 | emergency single-stage | Library + Items collapsed; Work escape |

At `<120`, explicitly assert the compact class removes the grid border/padding so `W == terminal width`. At every width assert no rail/canvas/pane/footer intersection and no compositor overflow.

For each matrix row, exercise custom widths `24`, `34`, `35`, and `48` in one ordinary and one adaptive destination. Assert ordinary compression/restoration, adaptive collapse/explicit-priority behavior, and unchanged saved preferences. Include invalid persisted values below 24, above 48, non-integer, and absent through the real normalization path.

Add a high-frequency resize test that alternates widths across projection, compression, adaptive collapse/hysteresis, emergency, and restoration boundaries. Spy on `recompose`, configuration reads/writes, route data loads, worker creation, and preference persistence; all remain zero for geometry-only resizes after initial settle. Also assert unchanged effective tuples perform no style assignments.

- [ ] **Step 2: Run the matrix and integrity tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_css_build_integrity.py -q -k "production_width_matrix or resize_geometry or modular_css"
```

Expected: FAIL until source TCSS, compact box model, and generated bundle match the contract.

- [ ] **Step 3: Make the minimum source-TCSS changes and regenerate**

Keep presentation/overflow rules in `_agentic_terminal.tcss`; do not put competing persistent width declarations there. Add only the return-bar presentation and any compact border/padding rule needed by the approved box model. Never hand-edit generated CSS.

Regenerate:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.css.build_css
```

- [ ] **Step 4: Run the complete geometry/performance matrix**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_css_build_integrity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit production geometry**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_shell.py Tests/UI/test_css_build_integrity.py
git commit -m "test(library): lock bounded rail geometry matrix"
```

## Task 7: Align Settings, defaults, migration semantics, and user copy

**Files:**
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/test_config_library_defaults.py`
- Modify: `Tests/UI/test_settings_appearance_defaults.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Docs/User_Guide/library.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `Docs/User_Guide/library/media-and-conversations.md`

- [ ] **Step 1: Write failing Settings/default/migration tests**

Prove:

- a fresh config and explicit reset use reference value 31 through the real template/load path;
- `[library.reader]` remains empty/absent in the shipped template until the user explicitly saves shared reader settings; the fresh legacy `[library.media_reader].library_width` fallback changes to 31;
- loading an existing explicit `[library.reader].library_width = 28` or legacy `[library.media_reader].library_width = 28` performs no migration/write and resolves 28 when custom mode is later enabled;
- custom off leaves any saved 24–48 width dormant;
- enabling custom later exposes the unchanged stored value;
- Settings validation remains 24–48, not 24–34;
- visible copy says shared “Library rail”, explains bounded automatic sizing separately from explicit custom width, and truthfully warns that an explicit preference may shrink temporarily to preserve 40 content cells;
- neither resize, mode switching, nor settings-generation refresh modifies the Settings field or calls persistence.

- [ ] **Step 2: Run focused Settings/config tests and confirm old default/copy failures**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py -q -k "library or reader or width"
```

Expected: FAIL on the historical 28 default and route-specific wording.

- [ ] **Step 3: Update defaults and copy without adding migration logic**

Use the shared constants instead of duplicating 31/24/48. Change the fresh legacy fallback/template value to 31 while leaving the shipped `[library.reader]` table empty; the existing precedence `environment → explicit shared reader → explicit legacy media reader → code default` preserves stored 28 values. Do not deep-merge 31 into an explicitly stored 28 and do not write configuration during load. Update docs with the distinction among:

- automatic `3:13` sizing bounded 24–34;
- explicit custom 24–48 preference;
- ordinary temporary compression to preserve 40 cells;
- adaptive collapse/priority exceptions;
- ordinary `<64` emergency single-stage and `‹ Library`/`< Library` return.

- [ ] **Step 4: Run Settings/config/docs-related tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Settings and documentation**

```bash
git add tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py Docs/User_Guide/library.md Docs/User_Guide/settings.md Docs/User_Guide/library/media-and-conversations.md
git commit -m "docs(library): explain bounded rail preferences"
```

## Task 8: Run targeted verification and real-PTY UAT

**Files:**
- Create: `Docs/superpowers/qa/library-rail-bounded-width-2026-08/README.md`
- Create: `Docs/superpowers/qa/library-rail-bounded-width-2026-08/width-matrix.txt`
- Modify: `backlog/tasks/task-22301 - Bound-Library-rail-width-across-modes.md`
- Modify only if this task produced a reusable incident: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run focused static analysis**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Library/library_rail_width.py tldw_chatbook/Library/library_adaptive_reader_state.py tldw_chatbook/Widgets/Library/library_rail.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_emergency_return.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Library/test_library_rail_width.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/UI/test_library_shell.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_honesty_accessibility.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/Library/library_rail_width.py tldw_chatbook/Library/library_adaptive_reader_state.py tldw_chatbook/Widgets/Library/library_rail.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_emergency_return.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Library/test_library_rail_width.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/UI/test_library_shell.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_honesty_accessibility.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Library/library_rail_width.py tldw_chatbook/Library/library_adaptive_reader_state.py tldw_chatbook/Widgets/Library/library_rail.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_emergency_return.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: Run the complete focused regression suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_rail_width.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/UI/test_library_shell.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_css_build_integrity.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py -q
```

Expected: PASS.

- [ ] **Step 3: Ask whether the user wants the optional full repository sweep**

AGENTS.md requires explicit opt-in before a full sweep. Present the already-passing focused suite and ask once. If the user declines or does not opt in, record `not run — targeted verification selected` in the task evidence and continue with PTY UAT; the full sweep is not a closeout prerequisite.

If and only if the user opts in, run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
```

Expected when approved: PASS, or any failure is rerun and documented with exact evidence. Do not mark the task Done while an in-scope failure remains.

- [ ] **Step 4: Exercise the production app in isolated real PTYs at the six approved widths**

Create a scratch profile outside the repository and launch the real module from this worktree with one unique tmux session per width: `tldw22301-235`, `-170`, `-120`, `-100`, `-80`, and `-60`. Substitute the literal width/height in both tmux and `stty` (do not leave `WIDTH`/`HEIGHT` placeholders at execution time):

```bash
/usr/bin/env -i TLDW_TEST_MODE=1 HOME=/tmp/tldw-task22301/home XDG_DATA_HOME=/tmp/tldw-task22301/xdg-data XDG_CONFIG_HOME=/tmp/tldw-task22301/xdg-config TLDW_CONFIG_PATH=/tmp/tldw-task22301/config.toml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/library-rail-bounded-width PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin LANG=en_US.UTF-8 tmux -L tldw22301 new-session -d -x 235 -y 52 -s tldw22301-235 '/bin/zsh -c "stty cols 235 rows 52; exec /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app"'
```

Before trusting each capture, record `#{pane_width}x#{pane_height}`, process command, and current path. Resize detached sessions with `tmux -L tldw22301 resize-window -t SESSION -x WIDTH -y HEIGHT` and re-check pane size; if attaching, use `window-size latest` so client and server agree. Do not redirect stderr because Textual renders there. Kill all six sessions after evidence capture with `tmux -L tldw22301 kill-server`.

Drive Library routes only through keyboard/tmux input—no CUA. Capture the pane after:

- initial Library landing;
- Collections and at least one ordinary reading route;
- Media/Chats/Notes adaptive states where the viewport supports them;
- custom 35 at `W=74/75` and custom 48 at `W=64/80/87/88`, configured through Settings before capture; verify the displayed effective width changes while the saved field does not;
- 60-cell activation into canvas-only, guarded `Escape`, Enter return, then `resize-window` to 64 and back to 60 to verify restoration;
- ASCII substitution enabled through its production Settings control, followed by a 60-cell capture showing `< Library`.

Pointer activation remains covered by production-mounted Pilot tests unless an actual SGR mouse event is deliberately injected and documented; do not infer it from keyboard PTY input. Write commands, observed dimensions, focus/return results, and captures to `Docs/superpowers/qa/library-rail-bounded-width-2026-08/README.md` and `width-matrix.txt`. Label evidence as detached tmux PTY unless a named terminal client was actually attached; do not claim native iTerm2 or Windows Terminal acceptance.

- [ ] **Step 5: Self-review the complete diff against every acceptance criterion**

```bash
git status --short
git diff --check
git diff origin/dev...HEAD --stat
git diff origin/dev...HEAD -- tldw_chatbook/Library tldw_chatbook/Widgets/Library tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/css Tests/Library Tests/UI Docs/User_Guide backlog
```

Check for width constants duplicated outside the pure policy, stale fixed 28 behavior, hand-edited generated CSS, resize-time service/config work, hidden-pane focus, dishonest Escape/footer/F1 copy, and route omissions.

- [ ] **Step 6: Close TASK-22301 only after evidence is complete**

Update the task file so all seven acceptance criteria are checked. Add concise Implementation Notes covering approach, files, testing, PTY evidence, trade-offs, and the amended ADR. Add a lessons entry only if implementation produced a new reproducible incident that generalizes beyond this task.

Then run:

```bash
backlog task edit 22301 -s Done --notes "Implemented the shared bounded-fractional Library rail policy, reversible ordinary and adaptive adapters, guarded narrow-terminal return path, Settings/docs alignment, production geometry coverage, and real-PTY UAT. ADR: backlog/decisions/086-library-adaptive-reader-shell.md"
git add 'backlog/tasks/task-22301 - Bound-Library-rail-width-across-modes.md' Docs/superpowers/qa/library-rail-bounded-width-2026-08
git commit -m "chore(library): close bounded rail task"
```

- [ ] **Step 7: Verify the final commit, not the pre-commit working tree**

```bash
git status --short
git log -1 --oneline
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_rail_width.py Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/UI/test_library_shell.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_css_build_integrity.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py -q
```

Expected: clean worktree and focused suite PASS on the committed tree.
