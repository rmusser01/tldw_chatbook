# Compact Library Media Browsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make populated Library Media pages scan efficiently at 100×30 with at least five truthful one-line rows while preserving the existing wide preview, authoritative paging, focus, recovery, and mutation contracts.

**Architecture:** Reuse the screen’s existing measured `<120` compact state and pass it through the existing Media presentation kwargs. `LibraryMediaCanvas` owns one pure label grammar plus an in-place compact-presentation patch so breakpoint crossings do not rebuild the row tree. Extend the existing bounded list-entry focus settlement with an optional Media semantic target and scroll snapshot so viewer Back restores the activated row without adding a second focus system.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich markup, TCSS, pytest/pytest-asyncio, existing Library Media controller and canvas state.

**ADR required:** no

**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`

**Reason:** ADR-067 already owns authoritative Media paging, stale recovery, and mutation refresh. This task changes only responsive presentation and bounded semantic focus restoration.

---

## Scope and File Ownership

**Production files**

- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — compact row grammar, mounted in-place presentation, preview/focus visibility.
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — transport measured compact state, breakpoint focus transfer, viewer return snapshot, bounded semantic list-entry restore.
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` — compact row/list/pager allocation.
- Modify: the mirrored Media fallback rules in `LibraryScreen.BUNDLED_CSS` — keep no-bundle/test fallback behavior identical to the component stylesheet.
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` and `tldw_chatbook/css/widget_defaults_scoped.tcss` — generated stylesheet outputs; never hand-edit.

**Tests and docs**

- Modify: `Tests/UI/test_library_media_side_by_side.py` — returning-user harness correction plus compact/wide geometry, focus, resize, viewer return, and state regressions.
- Modify only if a direct state fixture is required: `Tests/UI/test_library_multiselect_media.py` — Select/mutation compact regression.
- Modify only if the targeted sync seam requires a direct regression: `Tests/UI/test_library_canvas_sync_defects.py` — compact kwarg transport through canvas sync.
- Modify: `Docs/User_Guide/library/media-and-conversations.md` — compact Media scanning and viewer/Back behavior.
- Modify: `backlog/tasks/task-19579 - Optimize-compact-Library-Media-browsing.md` — implementation plan link, checked ACs, notes, and status.

**Do not modify** Media database/service/controller/state contracts, page size, breakpoint value, user preferences, navigation routes, or unrelated Library canvases.

## Task 1: Repair the returning-user Media harness baseline

**Files:**

- Modify: `Tests/UI/test_library_media_side_by_side.py:15-45`

- [ ] **Step 1: Add the explicit returning-user harness helper**

```python
def _build_media_test_app():
    app = _build_test_app()
    app.library_new_profile_admission = False
    return app
```

`test_library_media_side_by_side.py` imports `_build_test_app` from `Tests.UI.app_factory`, not the returning-user wrapper in `test_library_shell.py`; therefore its real app object retains the session’s fresh-profile admission fact. Replace this file’s direct calls with `_build_media_test_app()`. Do not change production admission behavior or stored lifecycle state.

- [ ] **Step 2: Run the exact pre-feature owner baseline**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_media_side_by_side.py
```

Expected: the eleven incumbent owner tests pass. Record that merged `dev` previously failed all eleven at the hidden `#library-row-browse-media` lookup because the harness accidentally modeled a fresh Starter profile.

- [ ] **Step 3: Commit the test-fixture correction separately**

```bash
git add Tests/UI/test_library_media_side_by_side.py
git commit -m "test(library): model returning media harness"
```

## Task 2: Add compact row presentation and vertical allocation

**Files:**

- Modify: `Tests/UI/test_library_media_side_by_side.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py:23-515`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:10617-10632`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:1241-1260,2587-2650`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`, `tldw_chatbook/css/widget_defaults_scoped.tcss`

- [ ] **Step 1: Write RED compact and wide presentation tests**

Use `LibraryProductionCSSHarness` from `Tests/UI/test_library_shell.py` (its `CSS_PATH` is exactly `TldwCli.CSS_PATH`) and a deterministic helper that seeds at least 45 canonical Media summary rows. Add:

In `test_compact_media_paints_five_one_line_rows_and_hides_preview`, open the production Media route at 100×30, wait for its exact first page, collect compositor-painted `.library-media-row` buttons, and assert at least five rows, one-cell height, ` · ` metadata, no newline, zero-area preview, and a non-focusable `#library-media-open-viewer`.

In `test_wide_media_keeps_two_line_rows_and_preview`, open the same fixture at 170×48 and assert two-cell rows, a newline in the label, painted side-by-side preview geometry, and a focusable viewer action.

Also assert compact normal rows have neither the `▸` glyph nor `library-media-row-selected`, while compact Select rows retain `☐/☑`.

- [ ] **Step 2: Run the new nodes and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_side_by_side.py \
  -k 'compact_media_paints or wide_media_keeps'
```

Expected: compact assertions fail because rows are two lines and the preview is painted; wide regression remains green.

- [ ] **Step 3: Implement one row-label source of truth**

In `library_media_canvas.py`, add a small module-level helper or private method with this contract:

```python
def _media_row_label_rest(row: LibraryMediaRow, *, compact: bool) -> str:
    title = _visible_row_title(row.title)
    if compact:
        return f" {title} · {row.secondary}"
    return f" {title}\n    {row.secondary}"
```

Use it during compose and mounted breakpoint patches. Store raw row title, secondary text, selected state, and `_library_row_label_rest` on each Button so `_apply_library_row_toggle` preserves the current density.

- [ ] **Step 4: Add compact input and mounted presentation patch**

Extend `LibraryMediaCanvas.__init__` and `sync_state` with `compact: bool = False`. Add:

```python
def apply_compact_presentation(self, compact: bool) -> None:
    """Patch row density and preview participation without recomposing."""
```

The method must:

- update `self.compact`;
- patch row labels, `_library_row_label_rest`, height, and minimum height;
- suppress normal preview-selected glyph/class in compact mode and restore them wide;
- preserve `☐/☑` in Select mode;
- set preview paint/focus participation consistently;
- never call `refresh(recompose=True)`.

Pass `"compact": self._library_notes_compact` from `_library_media_canvas_presentation()` so every compose, replacement, and targeted sync path shares the same source.

- [ ] **Step 5: Allocate compact list space in source CSS**

In `_agentic_terminal.tcss`, update only compact Media selectors so the workbench/list fill remaining height, the row scroll owns `1fr`, rows use height/min-height `1`, and the preview/detail placeholder are hidden. Mirror those same no-bundle Media rules in `LibraryScreen.BUNDLED_CSS`. Keep the pager outside `#library-media-row-scroll` and contained.

Do not change wide selectors or Trash row density.

- [ ] **Step 6: Regenerate and verify CSS**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```

Expected: bundle regeneration succeeds and parity check passes.

Inspect `git diff --name-only` after regeneration and include both generated outputs (`tldw_cli_modular.tcss` and `widget_defaults_scoped.tcss`) when changed. The component source and `LibraryScreen.BUNDLED_CSS` fallback must express the same compact Media allocation before continuing.

- [ ] **Step 7: Run the focused compact/wide nodes**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_side_by_side.py::test_compact_media_paints_five_one_line_rows_and_hides_preview \
  Tests/UI/test_library_media_side_by_side.py::test_wide_media_keeps_two_line_rows_and_preview
```

Expected: both nodes pass, with at least five compositor-painted rows at 100×30 and unchanged two-pane preview at 170×48.

- [ ] **Step 8: Run presentation inverses one at a time**

Temporarily restore two-line compact row height; the five-row/one-line node must fail. Restore immediately. Temporarily keep preview-selected styling in compact mode; the compact marker/style node must fail. Restore immediately.

- [ ] **Step 9: Commit compact presentation**

```bash
git add Tests/UI/test_library_media_side_by_side.py \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  tldw_chatbook/css/widget_defaults_scoped.tcss
git commit -m "feat(library): compact media browse rows"
```

## Task 3: Make breakpoint focus transfer lossless and read-free

**Files:**

- Modify: `Tests/UI/test_library_media_side_by_side.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:4826-4870,5185-5250`
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py`

- [ ] **Step 1: Write RED breakpoint-transition tests**

Add these exact mounted nodes using `LibraryProductionCSSHarness`, 45 Media rows, and the production-shaped service call ledger:

- `test_media_resize_preserves_scope_focus_scroll_without_reads`
- `test_media_preview_focus_moves_to_selected_row_on_compact_resize`
- `test_media_resize_focus_restore_yields_to_newer_user_focus`

Cover compact→wide→compact on a mounted page. Assert:

- applied scope/page/filter/selection are unchanged;
- row-scroll offset is unchanged and contained;
- zero new `search_media` and facet calls occur;
- row focus remains on the same row;
- if `#library-media-open-viewer` is focused wide, wide→compact transfers focus to the preview-selected row;
- if the user moves focus after transition intent is captured, the deferred fallback does not steal it.

- [ ] **Step 2: Run the new nodes and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_side_by_side.py::test_media_resize_preserves_scope_focus_scroll_without_reads \
  Tests/UI/test_library_media_side_by_side.py::test_media_preview_focus_moves_to_selected_row_on_compact_resize \
  Tests/UI/test_library_media_side_by_side.py::test_media_resize_focus_restore_yields_to_newer_user_focus
```

Expected: the current canvas has no compact patch hook, and focused preview action becomes invalid/does not transfer deterministically.

- [ ] **Step 3: Wire the mounted canvas into the existing responsive transition**

In `_apply_library_notes_stage_visibility()`, query the optional mounted `LibraryMediaCanvas` and call `apply_compact_presentation(self._library_notes_compact)` exactly as the Notes canvas already receives its presentation update.

Capture whether current focus belongs to the preview before hiding it. Transfer only that disappearing preview focus to the semantic row whose canonical `media_id` matches the preview selection; resolve that row directly from the mounted canvas and do not assume row index zero or introduce viewer-return snapshot state yet. Other controls retain focus. If a callback is required after refresh, capture the focus-intent generation and original preview owner, then re-check both the generation and current live focus before applying the fallback.

- [ ] **Step 4: Run breakpoint tests and inverses**

Expected: focused nodes pass. Temporarily trigger `_sync_library_canvas` or a Media request on resize; the zero-call node must fail. Temporarily remove the newer-user-focus guard; the focus-veto node must fail. Restore after each.

- [ ] **Step 5: Commit responsive focus wiring**

```bash
git add Tests/UI/test_library_media_side_by_side.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_media_canvas.py
git commit -m "fix(library): preserve media focus across resize"
```

## Task 4: Restore semantic row and scroll after viewer Back

**Files:**

- Modify: `Tests/UI/test_library_media_side_by_side.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:3420-3450,6434-6600,16829-16875,17684-17730,30104-30130`

- [ ] **Step 1: Write RED viewer-return tests**

Add these exact mounted nodes, all using `LibraryProductionCSSHarness` at 100×30:

- `test_compact_media_viewer_back_restores_semantic_row_and_scroll`
- `test_compact_media_viewer_back_survives_authoritative_recompose`
- `test_compact_media_viewer_back_falls_back_after_row_removed`
- `test_compact_media_viewer_back_follows_single_page_clamp`
- `test_compact_media_viewer_back_empty_page_focuses_recovery_control`
- `test_compact_media_viewer_back_restore_yields_to_newer_user_focus`

For the ordinary/recompose cases, use a compact page with enough rows to scroll:

1. focus and activate a non-first visible row;
2. record canonical `media_id`, applied page, and `#library-media-row-scroll.scroll_y`;
3. enter the viewer and return with Back/Escape;
4. force an additional authoritative list recompose inside the current settle window;
5. assert the same Media identity remains focused and the scroll offset is restored.

Add deterministic fallbacks:

- captured row removed but page still valid → first authoritative row;
- restore saved applied scope page 2 so the initial mount reads offset 20 from a production-shaped service with total 21; viewer Back itself must perform no read because the applied page is retained; after Back, explicitly invoke the existing `_request_library_media_browse(controller.mutation_refresh_scope, focus_identity=None)` authoritative-refresh seam, whose service responses are coherent page-2 total 20/empty and page-1 total 20; assert exact read offsets `[20, 20, 0]`, first row on applied page 1, and no fourth read;
- an authoritative exact empty result → nearest enabled recovery control in the order type filter, clear-filter/import empty action, then Retry (the fixture should assert the actually mounted `#library-media-empty-import` fallback);
- newer user focus after Back intent → no focus steal.

- [ ] **Step 2: Run the new nodes and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_restores_semantic_row_and_scroll \
  Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_survives_authoritative_recompose \
  Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_falls_back_after_row_removed \
  Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_follows_single_page_clamp \
  Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_empty_page_focuses_recovery_control \
  Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_restore_yields_to_newer_user_focus
```

Expected: `_exit_library_media_viewer()` arms the unconditional first-row focus and loses the activated row/scroll after remount or repeated recompose.

- [ ] **Step 3: Add a minimal semantic return snapshot**

Before `_open_library_media_viewer()` replaces the list, capture the activated canonical Media id and current row-scroll offset. Do not persist it and do not alter controller state.

Extend the existing bounded list-entry focus settlement rather than creating a second timer/controller. The armed focus request may carry an optional Media target id and scroll offset. `_focus_library_list_entry()` should:

- prefer the target Media row when it exists;
- restore a valid bounded scroll offset;
- fall back to the first authoritative row when the target disappeared or the page clamped;
- when the authoritative page is empty, focus the nearest mounted enabled Media recovery control instead of leaving focus on the removed viewer Back button;
- preserve the existing checked-row preference in Select mode;
- clear semantic target state when the existing disarm path fires.

Viewer Back supplies the snapshot to this arm instead of calling the unconditional first-row form.

Update `_sync_library_media_browse_state()` in this task so `_library_pending_list_entry_focus` preserves and retries that optional semantic Media target instead of overwriting it with `#library-media-row-0`. In the authoritative-recompose test, hold the deferred restoration callback, move focus to `#library-media-type-filter`, then release it; callback ordering must preserve the newer user focus.

- [ ] **Step 4: Run focused tests and required inverse**

Expected: all viewer-return nodes pass. Temporarily restore unconditional first-row arming; the non-first-row/repeated-recompose node must fail. Restore immediately.

- [ ] **Step 5: Commit viewer-return restoration**

```bash
git add Tests/UI/test_library_media_side_by_side.py \
  tldw_chatbook/UI/Screens/library_screen.py
git commit -m "fix(library): restore compact media browse position"
```

## Task 5: Protect state variants, document behavior, and close the task

**Files:**

- Modify: `Tests/UI/test_library_media_side_by_side.py`
- Modify if directly required: `Tests/UI/test_library_multiselect_media.py`
- Modify if directly required: `Tests/UI/test_library_canvas_sync_defects.py`
- Modify: `Docs/User_Guide/library/media-and-conversations.md`
- Modify: `backlog/tasks/task-19579 - Optimize-compact-Library-Media-browsing.md`

- [ ] **Step 1: Add focused compact state regressions**

Assert compact behavior through:

- Select mode and checkbox toggling;
- loading with retained rows;
- stale read-only rows and enabled Retry/type recovery;
- first/middle/final pager states and visible disabled reason;
- delete confirmation/receipt and mutation gate;
- type chooser;
- zero-result distilled empty state.

Required feedback may reduce the visible-row count, but the pager/actions/reasons must remain painted, contained, and keyboard reachable.

Use exact node names prefixed `test_compact_media_` for each added state case so the final selector cannot silently omit them. At minimum include `test_compact_media_select_markers_and_toggle_survive_density_patch`, `test_compact_media_stale_and_retry_actions_remain_truthful`, and `test_compact_media_pager_receipt_and_empty_states_remain_contained`.

- [ ] **Step 2: Run only touched/direct-owner tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_canvas_sync_defects.py \
  -k 'media and (compact or wide or preview or resize or viewer or focus or scroll or select or loading or stale or retry or pager or mutation or empty)'
```

If an optional test file was not modified and selects no direct owner, omit it and record the narrower command. Do not run the full suite.

- [ ] **Step 3: Run CSS and static gates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_css_build_integrity.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_canvas_sync_defects.py
git diff --check origin/dev...HEAD
git diff --check
```

Omit an optional test file from Ruff only when `git diff --name-only origin/dev...HEAD` proves it is unchanged. Run Ruff check on every changed Python path in that inventory. Run Ruff format-check only on changed Python files that are already format-clean at `origin/dev`; do not bulk-format legacy files.

- [ ] **Step 4: Commit the final focused regressions before docs**

```bash
git add Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_canvas_sync_defects.py
git commit -m "test(library): cover compact media recovery states"
```

Stage only files that actually changed; do not create an empty commit.

- [ ] **Step 5: Update the user guide**

Document:

- compact one-line `title · type · age` rows;
- preview available at wide widths and row activation at compact widths;
- viewer Back restoring page/row position;
- unchanged exact pager, stale/Retry, filter, Select, and mutation behavior.

Use ASCII diagrams only.

- [ ] **Step 6: Perform bounded UAT at both supported geometries**

Using the production hierarchy and exact `TldwCli.CSS_PATH`, verify 20+ populated Media rows at 100×30 and 170×48. At 100×30, capture compositor evidence for five painted rows and contained pager; at 170×48, verify the unchanged two-pane preview. Exercise row activation/Back, resize crossing, Select, stale/Retry, and focus veto. Use an isolated scratch config/data profile; do not touch the real profile.

- [ ] **Step 7: Request final spec and quality/minimality reviews**

Review the full task range against TASK-19579 and the design spec. Resolve all Critical/Important findings and re-run only affected owner gates.

- [ ] **Step 8: Close TASK-19579**

Use the Backlog CLI to mark each AC complete, add concise Implementation Notes with exact test/UAT/inverse/static evidence, record ADR-067/no-new-ADR, and set status Done. Re-read with `backlog task 19579 --plain`.

- [ ] **Step 9: Commit docs and closeout**

```bash
git add Docs/User_Guide/library/media-and-conversations.md \
  'backlog/tasks/task-19579 - Optimize-compact-Library-Media-browsing.md'
git commit -m "docs(library): close compact media browsing task"
```

## Required Inverse Summary

Run each mutation separately and restore immediately:

1. Restore two-line compact rows → five-row/one-line geometry test fails.
2. Keep compact preview-selection glyph/style → compact truthfulness test fails.
3. Issue a Media read on breakpoint crossing → zero-call resize test fails.
4. Remove newer-user-focus veto → resize focus test fails.
5. Restore unconditional first-row viewer Back arm → semantic row/scroll restoration test fails.

## Completion Boundary

The branch is complete only when the focused changed-component gates, CSS parity, static checks, both exact geometries, final reviews, task hygiene, and docs are green. Repository-wide pytest and unrelated Library work are explicitly out of scope.
