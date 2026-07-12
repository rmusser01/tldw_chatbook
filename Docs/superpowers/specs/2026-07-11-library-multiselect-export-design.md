# Library multi-select row export (task 159)

**Status:** Design approved (brainstorm), pending spec review.
**Backlog:** task-159 — "Multi-select row export for the Library".
**Builds on:** F4 whole-scope export (PR #597) + task-157 export progress/cancel (PR #607).

## Problem

F4 shipped current-scope export: the whole library, or one source (optionally
media-type-filtered). There is no way to export an *arbitrary subset* of rows.
The Library browses Media, Conversations, and Notes as three separate canvases,
each a `Vertical` of one compact `Button` per row (the record id stashed as
`button.media_id`/`.conversation_id`/`.note_id`); selection today is a single
highlighted id per source, and pressing a row opens its viewer/editor. We want
per-row multi-select so a user can check specific rows and export exactly those.

## Goal / Acceptance

- **AC1** — In a browse canvas the user can enter a selection mode, check
  individual rows, and export exactly the checked rows as a chatbook.
- **AC2** — The export form accepts an explicit id set as its scope (counts,
  scope label, and the actual export all reflect the chosen ids).

## Chosen approach

**Per-source selection** (decided in brainstorm): multi-select operates within
the active canvas; selection is a set of ids for that source and is cleared when
the source's filter changes or the user leaves the canvas ("what you see is what
you export"). Cross-source accumulation was rejected for v1 (needs persistent
cross-canvas state + a review surface).

**Reuse `ExportScope.kind`, add an `ids` override** rather than a new kind: an
`ExportScope(kind="media", ids=(...))` means "export exactly these media ids";
an empty `ids` keeps today's whole-source behavior byte-for-byte.

## Components

### 1. `ExportScope` — new `ids` field (`tldw_chatbook/Library/library_export_scope.py:24-50`)

```python
@dataclass(frozen=True)
class ExportScope:
    kind: str
    media_type: str | None = None
    ids: tuple[str, ...] = ()   # explicit id subset; empty ⇒ whole source

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(...)
        if self.ids and self.kind == "everything":
            raise ValueError("ids may only scope a single source, not 'everything'.")
```
`ids` is a `tuple` (hashable) so `ExportScope` stays `==`-comparable and hashable
— required by the counts worker's stale-scope guard (`scope != self._library_export_scope`).

### 2. Resolvers branch on `scope.ids` (same file)

- `resolve_export_selections` (`:104-143`): for each single-source `kind`, if
  `scope.ids` is non-empty return `{ContentType.X: list(scope.ids)}` directly
  (no `get_all_*` DB query); else the existing whole-source query. The
  ContentType is the one already mapped to that kind (media→MEDIA,
  conversations→CONVERSATION, notes→NOTE).
- `count_export_scope` (`:83-101`): symmetric — `{source: len(scope.ids)}` when
  `scope.ids`, else the existing `len(query)`.
- `export_scope_label` (`:146-169`): when `scope.ids`, render
  `"Selected {source} · {n} items"` (e.g. `"Selected media · 3 items"`); else
  the existing per-kind label. `media_type` is ignored when `ids` is set (the
  ids already encode the exact subset).

### 3. Row `checked` state (`Library/library_{media,conversations,notes}_state.py`)

Each row dataclass gains `checked: bool = False`. The pure state builder accepts
the current selection set and a `select_mode` flag and sets `checked` =
`row_id in selection` (only meaningful in select mode). The existing single-id
`selected` field and its `▸` marker are unchanged and used only in normal mode.

### 4. Canvas widgets (`Widgets/Library/library_{media,conversations,notes}_canvas.py`)

- Row `Button` label prefix: in select mode show `☑`/`☐` from `row.checked`;
  in normal mode the existing `▸`/space marker. (Keep it a single `Button`;
  only the leading glyph changes.)
- Toolbar: a **Select** toggle button (`#library-{source}-select-toggle`).
- A selection action row, shown only in select mode, with `N selected` text and
  buttons: **Select all** (`#library-{source}-select-all`), **Clear**
  (`#library-{source}-select-clear`), **Export selected**
  (`#library-{source}-export-selected`, disabled when the set is empty).
  Rendered via the same always-yield-then-`.display`-toggle pattern the canvas
  already uses, so it can be shown/hidden without a full recompose.

### 5. Screen wiring (`UI/Screens/library_screen.py`)

- New per-source state: `self._library_{media,conversations,notes}_selection: set[str]`
  and `self._library_{source}_select_mode: bool` (mirrors the existing
  `_selected_*_id` fields at `:561-579`).
- **Select toggle handler** flips `_library_{source}_select_mode` and recomposes
  the canvas; entering fresh clears any stale set.
- **Row press dispatch:** the existing per-row handlers
  (`handle_library_conversation_row` `:5271`, the media `:5311` and notes
  `:6359` equivalents) check the source's `select_mode`: if ON, toggle the row's
  id in the selection set and update just the row + the `N selected` text
  (targeted update, no viewer/editor open); if OFF, today's open-viewer path.
- **Select all** adds every currently-rendered row id to the set; **Clear**
  empties it.
- **Export selected** builds `ExportScope(kind=<source>, ids=tuple(sorted(sel)))`
  and calls the existing `_open_library_export_canvas(scope)` (`:3192`). The
  export canvas, counts worker, scope label, progress, and cancel all already
  flow from `ExportScope` — no export-side changes beyond the resolver branches
  in §2.
- **Lifecycle:** leaving the canvas / changing the source filter clears the set
  and turns select mode off (fresh WYSIWYG). Selection covers only the rendered
  (capped) rows — `LIBRARY_SOURCE_PAGE_SIZES` (notes 100, media/conv 50).

## Data flow

```
Select toggle → select_mode = True → canvas recomposes with ☐ rows
row press (mode ON) → toggle id in _library_media_selection → update row + "N selected"
Export selected → ExportScope(kind="media", ids=(…)) → _open_library_export_canvas
   → counts worker: count_export_scope sees ids → "Selected media · 3 items"
   → run export: resolve_export_selections returns {MEDIA: [ids]} → creator exports those
```

## Error handling

- **Export selected** disabled while the set is empty.
- An id deleted between select and export is harmless: `ChatbookCreator`'s
  `_collect_*` already logs "not found" and continues for missing ids.
- Server-mode export stays disabled exactly as today (the existing
  `_library_export_is_server_mode` guard in `_open_library_export_canvas`).

## Testing

- **Pure/unit (`Tests/Library/`):**
  - `ExportScope(kind="media", ids=("1","2"))` constructs; `ids` with
    `kind="everything"` raises; equality/hash hold (two equal scopes compare
    equal, usable as a dict key).
  - `resolve_export_selections` with `ids` returns exactly `{ContentType.X:
    [ids]}` and issues NO `get_all_*` call (assert via a stub id-source that
    records calls); without `ids` behaves as before.
  - `count_export_scope` with `ids` returns `len(ids)`.
  - `export_scope_label` with `ids` returns `"Selected {source} · N items"`.
  - State builder: `checked` is set for ids in the selection and false otherwise;
    select-all/clear semantics.
- **Screen handlers:** a row press in select mode toggles the id and does NOT
  open the viewer; "Export selected" builds the expected `ExportScope`
  (`SimpleNamespace`-fake `self` where feasible, mirroring the task-157 tests).
- **Canvas smoke (`app.run_test()`):** in select mode the Select-all/Clear/Export
  buttons render and "Export selected" is disabled at 0 selected.

## Scope / non-goals

- **Per-source only** — no cross-source selection tray.
- **Visible rows only** — selection is over rendered (capped) rows; changing the
  filter or leaving the canvas clears it. "Export all matching a filter" is
  already served by the existing whole-source/filtered export.
- No new keyboard multi-select shortcuts beyond the Select toggle.
- No change to the export execution/progress/cancel path beyond the two
  resolver branches — that machinery already consumes `ExportScope`.
