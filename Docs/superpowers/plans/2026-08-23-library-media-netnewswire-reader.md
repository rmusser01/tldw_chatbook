# Library Media NetNewsWire Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Library ▸ Media into a terminal-native, NetNewsWire-shaped reader where the Library and Items panes collapse independently, Items and Reader remain visible together when width permits, and complete stored media content stays authoritative.

**Architecture:** Extend the existing Library Media browse controller, canvas, viewer state, scope service, focus settlement, and delete receipt instead of adding a second media stack. A small pure `library_media_reader_state` module owns preferred-versus-effective layout and selected-versus-loaded request fencing; a Library-local shell widget owns geometry and two five-column grips; `LibraryScreen` remains the service/persistence orchestrator. Watchlists shares only the approved interaction grammar in this slice—no shared pane framework or imports.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich, Pillow plus the existing terminal-image capability ladder, TOML configuration, pytest/pytest-asyncio, production-bundle Textual geometry harnesses.

**ADR required:** yes

**ADR path:** `backlog/decisions/084-library-media-reader-ia.md`

**Reason:** ADR-084 owns the long-lived three-role information architecture, responsive-versus-preferred persistence boundary, local-only Items backend, and the deferred Watchlists sharing decision. ADR-067 continues to own authoritative Media pagination; ADR-055 continues to own Media delete receipt/Undo semantics; ADR-031 owns keybindings and footer-hint truthfulness.

---

## Approved inputs and baseline

- Design: `Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md`
- Architecture decision: `backlog/decisions/084-library-media-reader-ia.md`
- Applicable decisions: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, `backlog/decisions/055-library-destructive-action-reversibility-rule.md`, `backlog/decisions/067-library-top-level-pagination-contracts.md`
- Required lessons: `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`, `backlog/docs/lessons-backlog-hygiene.md`
- Baseline branch: current `origin/dev`, not the older feature branch from which the design conversation began.
- Baseline evidence already recorded in the planning worktree:
  - 133 passed: `Tests/Library/test_library_media_state.py`, `Tests/Library/test_library_media_viewer_state.py`, `Tests/UI/test_library_media_browse_controller.py`
  - 84 passed: `Tests/UI/test_library_media_side_by_side.py`, `Tests/UI/test_library_multiselect_media.py`
  - Existing pytest temporary-directory cleanup warnings are environmental and occurred after exit code 0.

`origin/dev` already has 20-row authoritative Media pages, exact totals/facets, requested/applied scope separation, stale mutation recovery, compact rows, semantic focus/scroll restoration, multi-select, bulk export/delete, Trash, and the shared receipt/Undo seam. Do not implement any of those again. Tests for “beyond the first 50” should advance through existing 20-row pages and prove records above index 50 are reachable.

## Delivery and Backlog discipline

Each numbered task below is one atomic Backlog task and one independently reviewable PR. Before starting each task:

1. Rebase its branch on the latest `origin/dev`.
2. Sweep every remote ref and worktree for the current maximum Backlog ID; never reuse a number carried from this plan.
3. Create/read the task file, mark it In Progress, add the task-specific plan and ADR links, then implement only its acceptance criteria.
4. Use the task’s exact tests plus the cross-cutting gates listed in Task 7. Do not mark Done until ACs, notes, docs, status, and review are complete.

The planning-time `backlog task list --plain` command did not return, so this document deliberately does not guess or reserve task IDs. File each task immediately before its implementation, following `lessons-backlog-hygiene.md`.

Do not add a wall-clock expiration to Media Undo. “Bounded” uses the existing ADR-055 receipt boundary: the receipt is dismissed, consumed, or superseded by a later delete, while Trash remains the durable recovery path. A timer would fork the established single/bulk reversibility contract.

## File map

**New production files**

- `tldw_chatbook/Library/library_media_reader_state.py` — pure session identity, request generation, preferred/effective layout, custom-width normalization, hysteresis.
- `tldw_chatbook/Widgets/Library/library_media_reader_shell.py` — Media-only shell and five-column pane grips.
- `tldw_chatbook/Widgets/Library/library_media_image_preview.py` — narrow local-image eligibility/decode/widget helpers over existing rendering primitives.

**Primary production modifications**

- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- `tldw_chatbook/Widgets/Library/library_media_viewer.py`
- `tldw_chatbook/Widgets/Library/library_media_content.py`
- `tldw_chatbook/Library/library_media_state.py`
- `tldw_chatbook/Library/library_media_viewer_state.py`
- `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/config.py`
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Generated only: `tldw_chatbook/css/tldw_cli_modular.tcss`, `tldw_chatbook/css/widget_defaults_scoped.tcss`

**Focused tests**

- New: `Tests/Library/test_library_media_reader_state.py`
- New: `Tests/UI/test_library_media_reader_shell.py`
- New: `Tests/UI/test_library_media_reader_flow.py`
- New: `Tests/UI/test_library_media_image_preview.py`
- Modify: `Tests/Library/test_library_media_viewer_state.py`
- Modify: `Tests/UI/test_library_media_browse_controller.py`
- Modify: `Tests/UI/test_library_media_side_by_side.py`
- Modify: `Tests/UI/test_library_multiselect_media.py`
- Modify: `Tests/UI/test_settings_appearance_defaults.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify only when its owned seam changes: `Tests/UI/test_library_canvas_sync_defects.py`, `Tests/UI/test_library_entry_compose_once.py`, `Tests/UI/test_library_media_trash.py`

**Docs/task records**

- `Docs/User_Guide/library/media-and-conversations.md`
- `Docs/User_Guide/settings.md` if that guide inventories Appearance groups
- One new Backlog task record per numbered task
- This plan and ADR-084 linked from every task record

## Constants fixed by this plan

Put the constants in `library_media_reader_state.py`; do not scatter them through screen/CSS code:

```python
LIBRARY_TARGET_WIDTH = 28
LIBRARY_MIN_WIDTH = 24
LIBRARY_MAX_WIDTH = 48
ITEMS_TARGET_WIDTH = 40
ITEMS_MIN_WIDTH = 32
ITEMS_MAX_WIDTH = 72
READER_COMFORT_WIDTH = 44
PANE_GRIP_WIDTH = 5
LAYOUT_HYSTERESIS_WIDTH = 4
SELECTION_SETTLE_SECONDS = 0.12
```

The maximums are validation bounds, not normal wide-mode widths. Reader uses `min-width: 0`; 44 is a resolver comfort target, not a CSS minimum.

## Task 1: Ship the pure reader session and responsive layout contract

**Outcome:** The state rules are independently testable before any UI is moved: manual preferences never collapse into responsive state, explicit opens receive temporary priority, custom widths clamp, and stale detail responses cannot overwrite the current selection.

**Files:**

- Create: `tldw_chatbook/Library/library_media_reader_state.py`
- Create: `Tests/Library/test_library_media_reader_state.py`
- Modify: the Task 1 Backlog record

- [ ] **Step 1: File the atomic Backlog task and add the design/ADR links**

Acceptance criteria must name the pure resolver, request fencing, hysteresis, and no UI/service changes. Record:

```text
ADR required: yes
ADR path: backlog/decisions/084-library-media-reader-ia.md
Reason: implements ADR-084's preferred/responsive/effective state contract.
```

- [ ] **Step 2: Write RED layout preference and clamping tests**

Create table-driven tests for defaults, fixed targets, and malformed/custom config:

```python
def test_default_preferences_are_both_open_and_fixed(): ...
def test_custom_widths_clamp_to_declared_minimums_and_maximums(): ...
def test_fixed_mode_ignores_saved_custom_width_values(): ...
def test_responsive_collapse_does_not_mutate_preferences(): ...
```

The proposed immutable contracts are:

```python
ReaderMode = Literal["read", "analysis", "highlights", "info"]
PaneName = Literal["library", "items"]

@dataclass(frozen=True)
class MediaReaderLayoutPreferences:
    library_open: bool = True
    items_open: bool = True
    custom_widths_enabled: bool = False
    library_width: int = LIBRARY_TARGET_WIDTH
    items_width: int = ITEMS_TARGET_WIDTH

@dataclass(frozen=True)
class MediaReaderEffectiveLayout:
    library_open: bool
    items_open: bool
    library_width: int
    items_width: int
    reader_width: int
    priority_pane: PaneName | None
```

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Library/test_library_media_reader_state.py -k 'preference or width or responsive'
```

Expected: import/collection failure because the module does not exist.

- [ ] **Step 3: Implement strict normalization**

Add `normalize_media_reader_preferences(raw: Mapping[str, Any])`. Reuse simple local coercion; do not import Settings or `config.py` into this dependency-root module. Fixed mode returns target widths. Custom mode clamps Library to 24–48 and Items to 32–72. Unsupported values fall back safely.

- [ ] **Step 4: Write RED resolver geometry tests**

Cover exact shell widths and grip accounting:

```python
@pytest.mark.parametrize(
    ("width", "library_open", "items_open"),
    [(160, True, True), (120, False, True), (80, False, False)],
)
def test_normal_resolution_collapses_library_then_items(...): ...

def test_explicit_open_collapses_other_pane_first_and_uses_requested_minimum(): ...
def test_reader_can_drop_below_comfort_after_explicit_open_without_overflow(): ...
def test_two_grips_and_reader_remain_reachable_at_sixty_columns(): ...
def test_returning_width_restores_target_widths_not_intermediate_widths(): ...
def test_hysteresis_prevents_one_column_resize_thrashing(): ...
def test_shrink_expand_cycles_are_idempotent(): ...
```

Use available shell width, including a fixed five columns for each grip whether its pane is open or collapsed. Normal resolution uses target widths and collapse order Library then Items. Explicit priority may use only the requested pane’s minimum; other normal open panes never interpolate.

- [ ] **Step 5: Implement `resolve_media_reader_layout`**

Use one small pure function. Inputs: available width, normalized preferences, previous effective layout, and optional priority pane. Output only the effective layout. Do not put Textual `Size`, Widgets, timers, or persistence in this module.

Pseudocode:

```python
def resolve_media_reader_layout(width, preferences, *, previous=None, priority=None):
    targets = normalized_target_widths(preferences)
    desired = preferred_open_set(preferences)
    if priority:
        desired.add(priority)
        collapse_the_other_pane_first_until_requested_minimum_fits(...)
    else:
        collapse_library_then_items_until_targets_plus_reader_comfort_fit(...)
    apply_hysteresis_only_when_reopening_a_responsive_collapse(...)
    return exact_targets_plus_reader_remainder(...)
```

- [ ] **Step 6: Write RED request/session tests**

Pin backend-qualified identity and selected-versus-loaded truth:

```python
def test_begin_selection_updates_selected_before_loaded(): ...
def test_pending_banner_can_name_selected_and_loaded_titles(): ...
def test_enter_can_force_immediate_load_generation(): ...
def test_only_matching_generation_and_backend_qualified_id_can_settle(): ...
def test_stale_success_and_stale_failure_are_rejected(): ...
def test_selected_and_loaded_can_differ_only_while_pending(): ...
def test_mode_persists_when_new_item_settles(): ...
def test_external_server_session_cannot_collide_with_local_id(): ...
```

Use `local:media:<id>` and `server:media:<id>` as canonical identities. Keep backing IDs separately for service calls.

- [ ] **Step 7: Implement the immutable session reducer**

Add a frozen `LibraryMediaReaderSessionState` and small transition functions (`begin_selection`, `settle_success`, `settle_failure`, `set_mode`, `enter_external_detail`, `leave_external_detail`). A request carries monotonically increasing generation plus requested canonical id. Reject rather than silently accept impossible combinations in `__post_init__`.

Do not store database rows or duplicate `LibraryMediaViewerState`; the session keeps identities/status only.

- [ ] **Step 8: Run Task 1 tests and inverse checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Library/test_library_media_reader_state.py
```

Temporarily reverse collapse priority; the table test must fail. Temporarily accept a stale generation; the stale-success and stale-failure tests must fail. Restore both mutations.

- [ ] **Step 9: Close and commit Task 1**

Update the task ACs/notes/status, then:

```bash
git add tldw_chatbook/Library/library_media_reader_state.py Tests/Library/test_library_media_reader_state.py 'backlog/tasks/<task-1-file>.md'
git commit -m "feat(library): define media reader session layout"
```

## Task 2: Mount the permanent three-role Media shell and collapsible grips

**Outcome:** Media renders Library + Items + Reader together, both five-column grips work by pointer/keyboard, responsive collapse is geometry-driven, and other Library destinations retain their current shell.

**Files:**

- Create: `tldw_chatbook/Widgets/Library/library_media_reader_shell.py`
- Create: `Tests/UI/test_library_media_reader_shell.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`, `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Modify: Task 2 Backlog record

- [ ] **Step 1: File Task 2 against Task 1’s merged task and write RED shell tests**

Use `Tests.UI.consolidated_css.ConsolidatedCSSApp` or the existing production-shaped Library harness with the exact `TldwCli.CSS_PATH`. Test the real hierarchy, not a bare widget. Add:

```python
def test_media_shell_mounts_library_items_reader_and_two_five_column_grips(): ...
def test_expanded_and_collapsed_grip_copy_names_its_action(): ...
def test_grips_are_focusable_clickable_and_geometry_stable(): ...
def test_reader_is_never_a_collapse_target(): ...
def test_non_media_library_routes_keep_the_existing_shell(): ...
```

Assert compositor text and containment, not only `styles.width`.

- [ ] **Step 2: Implement the smallest Media-local shell**

`LibraryMediaReaderShell` receives already-built Library, Items, and Reader widgets plus an effective layout. It owns no service calls or persistence. Add one message:

```python
class PaneToggleRequested(Message):
    def __init__(self, pane: PaneName) -> None: ...
```

Add a minimal `LibraryMediaPaneGrip(Button)` that paints horizontal `<---` or `--->` centered vertically inside a fixed five-column region, with tooltip/accessibility copy “Collapse/Expand Library/Items pane.” Do not subclass or modify the current three-column `LibraryNavigationRailHandle`; its vertical-label contract serves other routes.

`sync_layout()` must patch display and exact widths in place. Resize cannot recompose the Media item list or trigger reads.

- [ ] **Step 3: Convert only the Media compose branch**

In `LibraryScreen.compose_content`, detect `shell.canvas_kind == "media"` before yielding the normal rail/handle/canvas trio. Build:

```text
Library rail (optional) | Library grip | Items child (optional) | Items grip | permanent Reader
```

Items child is the existing `LibraryMediaCanvas` or `LibraryMediaTrashCanvas`; Reader is always a `LibraryMediaViewer`, built from `build_library_media_viewer_state(None)` when no detail is loaded. Remove the list→viewer replacement behavior only for Media. Keep all other Library destination branches unchanged.

The current 3-column Library handle and in-rail collapse button stay in use outside Media. Hide the in-rail collapse control inside the Media shell so the five-column grip is the single pane-control grammar.

- [ ] **Step 4: Keep current selection usable with immediate loads**

Before Task 3 adds traversal settlement, a normal row activation should:

1. update the selected canonical local id;
2. keep Items mounted;
3. begin the current exclusive detail worker immediately;
4. leave Reader’s prior detail or empty state mounted;
5. settle the loaded detail without replacing the shell.

Preserve current multi-select row toggling and Trash child transitions. The Reader can show a simple “Loading media…” status in this task; selected-versus-loaded textual markers land in Task 3.

- [ ] **Step 5: Drive resize through the pure resolver**

On shell resize, measure the shell’s settled region width, resolve from session/preferences, and call `sync_layout`. Do not use raw terminal width. Manual grip messages update preferred state and temporary priority; responsive collapse does not. Add a generation/veto guard so deferred focus cannot steal focus after the user moves elsewhere.

- [ ] **Step 6: Add production geometry cases**

At 160×50, 120×35, 100×30, and 80×24, assert expected open panes from the resolver, each grip exactly five columns, Reader containment, no horizontal overflow, and reachable collapsed grips. Add a 60-shell-column direct harness case proving no compositor exception and 50 columns left after grips.

At every size, assert resizing causes zero `search_media`, facet, detail, progress, or mutation calls.

- [ ] **Step 7: Add source CSS and regenerate bundles**

Define only Media-shell selectors in `_agentic_terminal.tcss`. Reader must use `width: 1fr; min-width: 0`. Panes receive exact inline cell widths from the resolver. Focus styles may change color/outline but never width/height.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```

- [ ] **Step 8: Run Task 2 tests and incumbent shell regressions**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_reader_shell.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_entry_compose_once.py
```

Required inverse: temporarily make Reader `min-width: 44`; the 80×24/60-shell containment test must fail. Temporarily resize via recompose; the zero-call or identity-preservation test must fail.

- [ ] **Step 9: Close and commit Task 2**

```bash
git add tldw_chatbook/Widgets/Library/library_media_reader_shell.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/Widgets/Library/library_media_viewer.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  tldw_chatbook/css/widget_defaults_scoped.tcss \
  Tests/UI/test_library_media_reader_shell.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_entry_compose_once.py \
  'backlog/tasks/<task-2-file>.md'
git commit -m "feat(library): mount collapsible media reader shell"
```

## Task 3: Make Items filtering and Reader loading continuous and truthful

**Outcome:** Row traversal highlights immediately, settles detail loading after 120 ms, Enter loads immediately, stale responses cannot repaint Reader, selected/loading/loaded states are textual, and Filter media uses the existing authoritative controller across records above index 50.

**Files:**

- Create: `Tests/UI/test_library_media_reader_flow.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` only if append support cannot be expressed by its existing requested/applied result contract
- Modify: `tldw_chatbook/Library/library_media_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- Modify: `Tests/UI/test_library_media_browse_controller.py`
- Modify: `Tests/UI/test_library_media_side_by_side.py`
- Modify: Task 3 Backlog record

- [ ] **Step 1: Write RED selection settlement and stale-response tests**

Use condition/event-controlled fake service calls, never arbitrary sleeps:

```python
async def test_arrow_traversal_updates_selection_immediately_but_loads_only_settled_row(): ...
async def test_enter_bypasses_selection_settle_window(): ...
async def test_pending_banner_names_selected_b_and_loaded_a(): ...
async def test_late_success_for_a_cannot_replace_loaded_b(): ...
async def test_late_failure_for_a_cannot_replace_loaded_b_or_show_error(): ...
async def test_detail_failure_keeps_items_usable_and_reader_retryable(): ...
```

Expose a screen-level `Timer` for the 120 ms settle window. Cancel it when selection changes, Media exits, bulk selection starts, or an external detail opens. The session generation is the correctness fence; Textual worker cancellation is only resource cleanup.

- [ ] **Step 2: Add selected/loading/loaded row state**

Extend `LibraryMediaRow` with explicit text-state inputs such as `loading: bool` and `loaded: bool`; do not infer loaded from focus or color. Render stable two-line rows in Items:

```text
Selected · loading preview  Title
Loaded in Reader            Title
                              type · author/source · age
```

Use shorter equivalent copy at the 32-column minimum while preserving the words “Loading” and “Loaded.” Decorative markers may supplement the text.

- [ ] **Step 3: Wire session transitions into `LibraryScreen`**

Replace `_library_media_view == "viewer"` as the load-validity fence with canonical id + request generation. Keep `_library_media_view` only for Items subviews that still need it (`list`/`trash`), or rename it to make that scope truthful.

When B begins loading while A is loaded, do not clear A. Store the pending title from the applied row so the banner is truthful without waiting for detail. On matching success, atomically store detail/highlights/content mode, set loaded id, clear pending/error, then sync Reader and Items. On matching failure, retain loaded detail and show Reader-local Retry/Open original availability.

- [ ] **Step 4: Add `Filter media` to Items**

Mount an `Input` with placeholder/label `Filter media`. Debounce through the existing Media controller’s `MediaBrowseScope(query=...)`; do not client-filter `retained_items`. Existing controller `_search` already calls `search_media(mode="local", query, limit=20, offset=...)` and owns exact totals/stale fallback.

Add a clear-filter control and copy that includes the active query when zero results are returned.

- [ ] **Step 5: Preserve filter/unfiltered anchors and page identity**

Session state keeps the last unfiltered selected canonical id **and its last applied unfiltered `MediaBrowseScope` (including page)**. A filter result selects its first matching row only after the new applied snapshot arrives. Clearing the query first requests that captured unfiltered scope; after that authoritative page applies, restore the prior id when it is still present. If the page clamps because the collection changed, follow the controller's authoritative clamp and choose the deterministic first row on the clamped page. Do not briefly select page 1 or start its detail load while the captured page is pending.

Use existing Previous/Next pages unless product review explicitly calls for append. The accepted requirement is incremental authoritative page access, not a new infinite-scroll abstraction. Add a test that pages 1→2→3, opens a record whose ordinal is above 50, applies a filter, then clears it and returns to that same record on page 3. This proves selection/Reader continuity, unfiltered-scope restoration, and no duplicate ids. Do not change the 20-row page size.

- [ ] **Step 6: Preserve bulk selection semantics**

Entering Select mode cancels pending single-item settlement. Reader keeps the last loaded item and shows no wording that implies it represents the checked set. Existing Select all visible, Clear, Export, Move to trash, Cancel, partial failure, receipt, and Trash behaviors remain reachable. Exiting Select resumes ordinary row selection without automatically loading a random row.

- [ ] **Step 7: Run Task 3 tests and inverses**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Library/test_library_media_reader_state.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py
```

Required inverses:

- temporarily client-filter only `retained_items`; the above-50/search call-ledger test must fail;
- temporarily gate detail response on selected id alone; the reused-id/generation stale test must fail;
- remove settle cancellation; the traversal call-count test must fail.

- [ ] **Step 8: Close and commit Task 3**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py \
  tldw_chatbook/Library/library_media_state.py \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py \
  'backlog/tasks/<task-3-file>.md'
git commit -m "feat(library): make media traversal continuously readable"
```

Omit `library_media_browse_controller.py` from staging if no production change was necessary.

## Task 4: Recompose Reader into Read, Analysis, Highlights, and Info modes

**Outcome:** Reader is content-first, mode persists across items, all existing item capabilities remain reachable through a focused toolbar/More menu, provenance is explicit, and the one-off server-ingest detail is read-only and never merged into Items.

**Files:**

- Modify: `tldw_chatbook/Library/library_media_viewer_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_content.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/Library/test_library_media_viewer_state.py`
- Modify: `Tests/UI/test_library_media_reader_flow.py`
- Modify: `Tests/UI/test_library_media_side_by_side.py`
- Modify: Task 4 Backlog record

- [ ] **Step 1: Write RED mode and provenance tests**

```python
def test_first_reader_entry_defaults_to_read_and_mode_persists_across_items(): ...
def test_missing_analysis_and_highlights_show_item_specific_empty_states(): ...
def test_exactly_one_mode_surface_is_painted(): ...
def test_find_targets_only_loaded_item_content(): ...
def test_info_names_backend_id_source_stored_representation_and_console_payload(): ...
def test_toolbar_overflow_keeps_every_action_reachable_from_more(): ...
```

Test at wide and 80×24 widths using compositor text/containment.

- [ ] **Step 2: Extend viewer state only with display facts**

Add provenance fields needed by Info—backend, canonical id, original source, stored representation/completeness, and Console handoff representation. Keep mode and More-open state in `LibraryMediaReaderSessionState`, not `LibraryMediaViewerState`. Continue deriving authoritative content/analysis/read-later/metadata from the existing detail record.

- [ ] **Step 3: Recompose the Reader hierarchy**

Change `LibraryMediaViewer.compose` to:

1. compact identity line;
2. title + author/source;
3. primary toolbar;
4. four mode buttons;
5. exactly one active mode body;
6. pending/error/receipt banners.

Read reuses `LibraryMediaContentBody` and its Markdown/raw toggle. Analysis reuses current analysis display/edit. Highlights reuses current list/add/delete controls. Info owns metadata, provenance, representation status, and metadata edit controls.

Do not mount four full content bodies and hide three; compose only the active body to avoid duplicate Markdown parsing and focus targets.

- [ ] **Step 4: Build the toolbar and a simple inline More region**

Primary actions: Find, Read Later, Use in Console, More. At narrow widths, keep Find and Read Later visible as long as they fit; More exposes everything else. Use the existing toolbar buttons and an inline/collapsible region, not a new popover framework.

More contains only already-supported secondary actions: Edit metadata, Open original, existing copy/export if present, Move to trash, and the legacy Media manager escape hatch if still necessary. Pane grips never appear in More.

Find means `Find in item` and searches only the loaded detail. Preserve current match navigation/search body implementation.

- [ ] **Step 5: Make the server-ingest compatibility route explicit**

Replace `_library_media_detail_is_remote: bool` with the session’s backend-qualified external detail transition. Fix the current route so `handle_library_ingest_view_on_server` cannot have its remote mode cleared by `_open_library_item_by_id`.

The external session must:

- show `Server item · not in local Media list`;
- leave the local Items snapshot visible with no selected/loaded row claim for the server id;
- offer only Find, Open original, and Use in Console when supported;
- hide/disable Read Later, progress, Highlights, metadata editing, rich preview, and delete with truthful reasons;
- clear incompatible generations on entry and exit;
- exit when any local row is selected or Back is requested.

No backend selector, server paging, or merged ids are added.

- [ ] **Step 6: Preserve Console handoff truth**

Route Use in Console through the existing authoritative payload builder. Info’s “Use in Console sends …” line must be built from the same payload decision, not duplicate wording. Tests compare the actual handoff payload to the Info description and prove preview/rendered Markdown is not silently substituted.

- [ ] **Step 7: Run Task 4 tests and inverses**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Library/test_library_media_viewer_state.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_library_media_side_by_side.py
```

Temporarily expose Read Later in external mode; the external-capabilities test must fail. Temporarily mount all four bodies; the one-mode DOM/paint test must fail.

- [ ] **Step 8: Close and commit Task 4**

```bash
git add tldw_chatbook/Library/library_media_viewer_state.py \
  tldw_chatbook/Widgets/Library/library_media_viewer.py \
  tldw_chatbook/Widgets/Library/library_media_content.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Library/test_library_media_viewer_state.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_library_media_side_by_side.py \
  'backlog/tasks/<task-4-file>.md'
git commit -m "feat(library): organize media reader modes and actions"
```

## Task 5: Preserve mutation, reading-progress, focus, and Escape contracts

**Outcome:** Single delete advances to the correct adjacent row and shares the existing receipt/Undo seam, progress belongs only to the loaded identity, focus graduates through effective panes, and footer hints remain truthful.

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_reader_shell.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py`
- Modify: `Tests/UI/test_library_media_reader_flow.py`
- Modify: `Tests/UI/test_library_multiselect_media.py`
- Modify: `Tests/UI/test_library_media_trash.py` if its mounted shell assertions change
- Modify: Task 5 Backlog record

- [ ] **Step 1: Write RED delete adjacency and restore tests**

```python
async def test_delete_selects_following_row_then_previous_then_empty(): ...
async def test_single_delete_uses_existing_shared_receipt_and_undo_worker_group(): ...
async def test_undo_reinserts_and_reselects_when_item_matches_active_scope(): ...
async def test_undo_succeeds_with_restored_outside_current_filter_message(): ...
async def test_bulk_actions_and_existing_bulk_undo_contract_do_not_regress(): ...
```

Use the current `restore_media_item(mode="local")`, controller mutation refresh, one in-flight interlock, and one receipt. Do not add a second single-item undo flag/timer/path.

- [ ] **Step 2: Adapt the current delete seam to the permanent shell**

After committed single delete:

1. reconcile/remove the row through `LibraryMediaBrowseController`;
2. choose following row, else previous row, based on the pre-delete applied ordering;
3. begin that row’s detail request or settle Reader empty;
4. render the existing ADR-055 receipt in Reader/Items without obscuring navigation.

Undo restores through the service, reconciles according to current query/type/sort, and reselects only if the restored summary belongs in the active applied scope. Otherwise leave selection stable and say it was restored outside the filter.

- [ ] **Step 3: Write and implement loaded-identity progress tests**

Pin:

```python
async def test_progress_restores_after_loaded_content_mounts(): ...
async def test_stale_detail_never_writes_progress_under_new_selected_id(): ...
async def test_mode_change_preserves_per_item_read_scroll_for_session(): ...
async def test_external_server_detail_does_not_use_local_progress_seam(): ...
```

Store per-loaded-id scroll snapshots in screen/session memory. Write progress using `loaded_id`, never `selected_id`. Reuse the current reading-progress service calls and off-event-loop isolation.

- [ ] **Step 4: Write RED focus and Escape graduation tests**

At each effective layout, assert focus order and Escape behavior:

```python
def test_escape_closes_more_find_confirmation_before_leaving_reader(): ...
def test_escape_moves_reader_to_items_then_library_then_screen_back(): ...
def test_escape_skips_responsively_collapsed_panes(): ...
def test_hidden_panes_have_no_focusable_descendants_but_grips_remain_reachable(): ...
def test_deferred_focus_restore_yields_to_newer_user_focus(): ...
def test_footer_advertises_only_working_current_actions(): ...
```

No new `ctrl+` bindings. Reuse screen-level single-letter conventions only if an action genuinely needs a key; pointer/Tab/grip activation can ship without new printable bindings.

- [ ] **Step 5: Implement one outward Escape handler**

Give transients first refusal: More, Find, confirmation, edit/highlight forms. Then inspect focused widget ancestry and effective layout to move Reader→Items→Library. If a pane is hidden, skip to the next effective region. At Library, delegate to the existing screen back behavior. Re-register footer shortcuts whenever transient/effective state changes.

- [ ] **Step 6: Run Task 5 tests and inverse checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_side_by_side.py
```

Required inverses: choose previous before following after delete; adjacency test fails. Write progress using selected id; stale-progress test fails. Permit focus inside hidden Items; focusability test fails.

- [ ] **Step 7: Close and commit Task 5**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_media_reader_shell.py \
  tldw_chatbook/Widgets/Library/library_media_viewer.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_trash.py \
  'backlog/tasks/<task-5-file>.md'
git commit -m "fix(library): preserve media reader recovery and focus"
```

Omit unchanged optional test files from staging.

## Task 6: Add capability-gated local PNG/JPEG/WebP preview

**Outcome:** Eligible local images render above unchanged complete text when capability exists, while capability-off, decode, and render failures fall back honestly without changing the item’s load status.

**Files:**

- Create: `tldw_chatbook/Widgets/Library/library_media_image_preview.py`
- Create: `Tests/UI/test_library_media_image_preview.py`
- Modify: `tldw_chatbook/Library/library_media_viewer_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/Library/test_library_media_viewer_state.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`, `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Modify: Task 6 Backlog record

- [ ] **Step 1: Write RED pure eligibility and fallback tests**

Preview eligibility is local-only and requires both original-file availability and normalized PNG/JPEG/WebP type/MIME. Add:

```python
@pytest.mark.parametrize("mime", ["image/png", "image/jpeg", "image/webp"])
def test_eligible_local_original_image_types(mime): ...

@pytest.mark.parametrize("mime", ["image/gif", "application/pdf", "audio/mpeg", "video/mp4"])
def test_ineligible_types_never_download_or_render(mime): ...

def test_remote_url_or_server_detail_never_fetches_for_preview(): ...
def test_capability_off_keeps_complete_stored_text(): ...
def test_decode_failure_keeps_complete_text_and_local_retry(): ...
```

Mandatory headless default is capability-off unless a fake widget factory is injected.

- [ ] **Step 2: Implement the narrow helper module**

Do not create a general media framework or abstract class hierarchy. Add pure/simple functions:

```python
def image_preview_eligibility(detail, file_check, *, backend: str) -> PreviewEligibility: ...
def decode_media_image(content: bytes) -> PIL.Image.Image: ...
def build_media_image_widget(image, *, app_config, box_cols, box_lines) -> Widget: ...
```

Reuse:

- `MediaReadingScopeService.check_media_file(mode="local", file_type="original")`
- `download_media_file(mode="local", file_type="original")`
- `Chat.console_image_view.fit_image_cell_size`
- `Chat.console_image_view.resolve_default_mode`
- `textual_image.widget.Image` when graphics mode is usable
- `Utils.mosaic_render.mosaic_from_image` fallback

Inject a plain callable `preview_widget_factory` for tests. No remote fetch, cache service, plugin, or new dependency.

- [ ] **Step 3: Load and render preview without blocking or replacing text**

After the matching local detail settles, run eligibility/file/download/decode off the event loop under the same canonical id + generation fence. Store only ephemeral per-item preview state/image. In Read mode:

1. identity/title/toolbar;
2. preview widget or preview status;
3. Hide preview / Show preview;
4. complete stored text/Markdown body.

Default shown per eligible item for the current screen session. Failure copy is exactly scoped—“Image preview unavailable” or “Image preview failed — showing complete stored text”—with item-local Retry. It never becomes the item’s load error.

- [ ] **Step 4: Add mounted preview-order and generation tests**

Use a fake preview widget factory and controlled file/download calls:

```python
async def test_preview_mounts_above_byte_for_byte_unchanged_complete_text(): ...
async def test_hide_show_is_per_item_session_state_and_does_not_reload_detail(): ...
async def test_preview_failure_keeps_item_loaded_and_retry_is_item_local(): ...
async def test_late_preview_for_a_cannot_mount_over_loaded_b(): ...
async def test_external_server_detail_never_calls_local_file_seams(): ...
```

The fake renderer proves widget placement; headless tests must not assert a real terminal-image protocol.

- [ ] **Step 5: Add preview CSS, regenerate, and run Task 6 tests**

Keep the preview contained inside Reader at narrow widths and leave the complete text scrollable below it. Regenerate and verify:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_image_preview.py \
  Tests/Library/test_library_media_viewer_state.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_css_build_integrity.py
```

Required inverses: replace stored text with the preview; unchanged-text test fails. Remove the preview generation fence; the late-preview test fails. Permit GIF; ineligible-call-ledger test fails.

- [ ] **Step 6: Close and commit Task 6**

```bash
git add tldw_chatbook/Widgets/Library/library_media_image_preview.py \
  tldw_chatbook/Library/library_media_viewer_state.py \
  tldw_chatbook/Widgets/Library/library_media_viewer.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  tldw_chatbook/css/widget_defaults_scoped.tcss \
  Tests/UI/test_library_media_image_preview.py \
  Tests/Library/test_library_media_viewer_state.py \
  'backlog/tasks/<task-6-file>.md'
git commit -m "feat(library): preview local media images"
```

## Task 7: Persist Media layout preferences and complete visual QA

**Outcome:** Manual pane/default-width preferences persist through canonical Settings while responsive state remains session-only, and the complete reader is documented, production-shaped, visually verified, and reviewed.

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_settings_appearance_defaults.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_library_media_reader_shell.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`, `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Modify: `Docs/User_Guide/library/media-and-conversations.md`
- Modify: `Docs/User_Guide/settings.md` only if it inventories Appearance controls
- Modify: Task 7 Backlog record

- [ ] **Step 1: Write RED Settings normalization/persistence tests**

Extend `SettingsAppearanceDefaults` with:

```python
library_media_library_open: bool = True
library_media_items_open: bool = True
library_media_custom_widths_enabled: bool = False
library_media_library_width: int = 28
library_media_items_width: int = 40
```

Tests must prove load from `app_config["library"]["media_reader"]`, strict validation, width bounds, deep-merge preservation of unrelated `[library]` keys, reset-to-default draft, Save/Revert behavior, and app-config refresh. Do not add another settings model/module unless the existing Appearance module becomes demonstrably unmanageable.

- [ ] **Step 2: Implement minimal Appearance-defaults support**

Extend `load_appearance_defaults`, `validate_appearance_defaults`, and `build_appearance_save_sections`. Use the constants and normalization from `library_media_reader_state.py`; do not duplicate width bounds. Deep-copy/merge the current `library` mapping and update only `library["media_reader"]`.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_appearance_defaults.py -k 'library_media'
```

- [ ] **Step 3: Add the compact canonical Settings group**

In `SettingsScreen` Appearance, add “Library Media layout” controls:

- remember Library open/collapsed;
- remember Items open/collapsed;
- Fixed default widths / Custom widths toggle;
- Library width and Items width inputs enabled only in custom mode;
- Reset layout to defaults.

Extend existing Appearance draft, validation, save sections, field guidance, search fields, widget sync, and Save/Revert path. `build_appearance_save_sections` deep-merges the existing `library` section and writes nested `media_reader`. Screen grip actions use the same normalized keys through `save_setting_to_cli_config("library.media_reader", ...)`; responsive overrides are never written.

In `config.py`, preserve `library.media_reader` in normalized `app_config["library"]` and add the documented default TOML block. Do not touch deprecated settings surfaces.

- [ ] **Step 4: Add end-to-end preference tests**

Prove:

- grip collapse persists preferred state and a new Library screen loads it;
- responsive collapse never writes config;
- Settings Save updates the next/current Media shell through the existing app-config refresh mechanism;
- Reset restores both panes open and fixed targets;
- malformed config cannot cause overflow or a crash.

- [ ] **Step 5: Implement one live preference refresh seam**

After a successful Appearance save, increment an app-owned Library layout refresh generation, mirroring the existing Console appearance signal. Mounted Library screens normalize the new preferred values and re-resolve layout without a media read or whole-screen recompose. Grip actions optimistically update the same app-config mapping and persist only the changed preferred key off-thread. Failed persistence reports a warning and restores the previously loaded preference.

Run the Settings and shell end-to-end tests; temporarily persist a responsive collapse and verify the no-write ledger test fails.

- [ ] **Step 6: Finish source CSS, regenerate, and run bundle gates**

Polish information hierarchy, focus contrast, narrow toolbar/More containment, preview containment, empty/error banners, and no-color-only states. Use source CSS only, then:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_css_build_integrity.py
```

- [ ] **Step 7: Update user documentation**

Document the three roles, grip arrows, preferred versus responsive collapse, Filter media versus Find in item, Reader modes, local-only Items scope, server compatibility label, complete-text authority, eligible image formats, preview fallback, delete/Undo/Trash, and Settings controls.

Do not imply that Library rail search searches Media unless its real service semantics do. If current copy is overbroad, narrow the label/help copy in this task and test it; extending search semantics requires a separate accepted task.

- [ ] **Step 8: Run the complete focused regression matrix**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Library/test_library_media_state.py \
  Tests/Library/test_library_media_viewer_state.py \
  Tests/Library/test_library_media_reader_state.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/UI/test_library_media_reader_shell.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_library_media_image_preview.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_settings_appearance_defaults.py \
  Tests/UI/test_settings_configuration_hub.py
```

Also run:

```bash
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/Library/library_media_reader_state.py \
  tldw_chatbook/Widgets/Library/library_media_reader_shell.py \
  tldw_chatbook/Widgets/Library/library_media_image_preview.py \
  tldw_chatbook/Widgets/Library/library_media_viewer.py \
  tldw_chatbook/UI/Screens/library_screen.py
```

- [ ] **Step 9: Perform production-shaped visual verification**

Use the exact app CSS stack and a scratch `TLDW_CONFIG_PATH`; never touch the real profile. Capture the real Media route at 160×50, 120×35, 100×30, and 80×24 with a deterministic local fixture. Verify compositor text plus screenshots for:

- correct open panes and five-column grips;
- Reader priority and no horizontal overflow;
- selected versus loaded wording during a controlled pending detail;
- all four Reader modes;
- narrow More reachability;
- empty filter and detail error recovery;
- delete receipt/Undo;
- capability-off complete-text fallback;
- one real local PNG/JPEG/WebP when the installed renderer permits it.

For any configured real server, optionally verify the finished-ingest external detail identifies the exact server media id and remains read-only. Do not make a live server a completion requirement for the local-only feature.

- [ ] **Step 10: Run the required final review**

Use `superpowers:requesting-code-review` after all focused evidence passes. Review specifically for stale-response races, persistence leakage of responsive state, hidden focus targets, service calls on resize, accidental server catalogue claims, loss of complete stored text, CSS bundle drift, and Watchlists coupling.

Address findings with focused RED tests first, rerun affected commands, then rerun the complete focused matrix.

- [ ] **Step 11: Close and commit Task 7**

Complete task ACs, notes, ADR links, docs, review, and status. Add a lesson only if implementation exposed a genuinely new incident-backed trap.

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/UI/Screens/settings_appearance_defaults.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/config.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  tldw_chatbook/css/widget_defaults_scoped.tcss \
  Tests/UI/test_settings_appearance_defaults.py \
  Tests/UI/test_settings_configuration_hub.py \
  Tests/UI/test_library_media_reader_shell.py \
  Docs/User_Guide/library/media-and-conversations.md \
  Docs/User_Guide/settings.md \
  'backlog/tasks/<task-7-file>.md'
git commit -m "feat(library): persist media reader layout preferences"
```

Omit `Docs/User_Guide/settings.md` if it did not require modification.

## Programme completion gate

After every atomic task has merged, rebase a verification branch on the latest `origin/dev` and rerun Task 7’s full focused matrix, CSS gates, `git diff --check`, and the four production-shaped captures. Audit every programme task from `origin/dev`; do not rely on local summaries. Verify ADR-084 and the approved design still match the shipped behavior, especially local-only Items, fixed-default/custom-opt-in widths, non-persisted responsive overrides, and no shared Watchlists implementation.
