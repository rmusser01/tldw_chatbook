# Console rail conversation rows — flush-left, width-aware two-line names

**Date**: 2026-07-20
**Status**: Approved design, pending implementation plan
**Scope anchor**: `tldw_chatbook/Widgets/Console/console_workspace_context.py`

## Why

Conversation names in the Console left rail are hard-truncated at 20 characters
(`_MAX_CONVERSATION_ROW_TITLE`) and indented ~4 cells by a selection-marker
prefix, a hard-coded second-line indent, and button padding — while the rail
itself is `3fr` wide (typically 40+ cells). Most of the row width is wasted and
most of the name is lost. Target look: names flush left (Claude Desktop
sidebar style), using the full rail width, wrapping onto a second line when
long instead of truncating.

## Decisions (user-approved)

1. **Row layout**: the name wraps to up to 2 lines; the metadata line
   (`workspace - status - age`) stays, rendered as the row's final line.
2. **Active marker**: the `► ` / two-space marker prefix is removed. Selection
   is shown by the existing `console-workspace-conversation-row-selected`
   styling (background + bold + underline) alone.
3. **Wrap method**: width-aware — the wrap point is computed from the rail's
   actual measured width and recomputed on terminal resize, not a fixed
   character budget.

## New row anatomy

```
 Debugging the RAG splat bug in     ← name line 1 (flush left, no marker)
 the retrieval leg fusion path…     ← name line 2 (only when needed; … if still over)
 Chats - saved - 2d              *  ← metadata line (dim, flush left, ellipsized at budget)
 [2 Sub-Agents]                     ← badge line (unchanged, only when count > 0)
```

- Rows remain `Button` widgets. All press handlers, widget ids
  (`console-workspace-conversation-{index}`), attached attributes
  (`conversation_id`, `row_key`, `native_session_id`, `scope_type`,
  `workspace_id`), star buttons, focus and keyboard behavior in
  `chat_screen.py` are untouched.
- Row height becomes variable: 2-line button (short name), 3-line (wrapped
  name), +1 line when the sub-agent badge renders. Plus the existing 1-line
  bottom margin.
- The tooltip keeps the full untruncated title (unchanged).
- Both row paths get the change — the grouped browser
  (Starred/Workspaces/Chats) and the legacy single-workspace section — since
  they share `_conversation_button`.

## Wrapping

New pure helper in `console_workspace_context.py`:

```
wrap_console_conversation_title(title: str, budget: int) -> tuple[str, ...]
```

- Word-wraps the title into at most 2 lines of `budget` terminal cells.
- Measures in cells via `rich.cells.cell_len` (CJK/emoji titles must not
  overflow); wide-character-safe truncation for the ellipsis cut.
- If the name still overflows 2 lines, line 2 ends with `…`.
- Spaceless tokens longer than the budget (URLs, hashes) hard-break at the
  budget.
- Budget floor of 10 cells to avoid degenerate wraps on absurdly narrow rails.
- **Order matters**: wrap the *raw* title first, then markup-escape each line,
  then join with `\n` and append the badge markup. Escaping first inflates
  lengths (`[` → `\[`) past real cell width and misplaces break points.

The metadata line is cell-truncated to the same budget with `…` (today it is
unbounded and clips at the rail edge — the same overflow class that forced the
sub-agent badge onto its own line, task-226). The same ordering applies:
truncate the raw metadata string, then escape.

`_MAX_CONVERSATION_ROW_TITLE` is retired.

## Budget derivation

- Base width = the tray's measured content width (`content_region.width`),
  which already excludes the rail-body scrollbar and the tray's own padding.
- Browser rows (share the line with the 3-cell star button + 1-cell margin,
  plus 2 cells button padding): `budget = width − 6`.
- Legacy-section rows (no star column): `budget = width − 2`.
- Measuring `content_region.width` is deliberate: the width chain includes
  frame borders that vary — the rail carries a solid frame (2 cells) and the
  tray's own frame variant flips between `solid` (2 cells) and `none` (0)
  with workspace state (`_frame_console_region` /
  `_workspace_context_frame_variant`). `content_region` accounts for border
  and padding automatically; no hand-derived arithmetic.
- First compose runs before layout has measured anything: use one
  conservative named fallback constant (exact value finalized during
  implementation against the real layout chain; it only affects the
  pre-measurement first frame). The first fit pass
  (`on_mount` → `call_after_refresh(_fit_height_to_content)`) measures the
  real width and corrects it (see Relabel triggers). One extra recompose at
  mount is accepted — consistent with `sync_state`'s existing unconditional
  recompose on every state change.

## Relabel triggers

- The tray stores the budget it last rendered with.
- The budget recheck lives in `_fit_height_to_content` — the single choke
  point that already runs after mount, after every resize, and after every
  `sync_state` recompose. When the freshly measured budget differs from the
  stored one, it schedules the existing recompose path (the `sync_state`-style
  refresh with scroll-restore machinery already in the widget); when it
  matches, nothing happens.
- `on_resize` alone would be insufficient: the tray's frame-variant flip
  (solid ↔ none) changes its *content* width without changing its outer
  size, so no resize event fires for it. The fit pass runs in that path
  regardless.
- **Oscillation guard**: the budget depends on tray width; tray width depends
  on whether `#console-left-rail-body`'s scrollbar shows; wrapping changes the
  content height that decides whether the scrollbar shows. To break that loop
  at the source, set `scrollbar-gutter: stable` on `#console-left-rail-body`
  (supported in installed Textual 8.2.7) so the scrollbar cell is always
  reserved and width is independent of scroll state.
- Contingency (not built up front): if live verification shows recompose jank
  during continuous resize drags, add a ~100 ms debounce timer around the
  relabel. The budget-change guard alone is expected to suffice.

## Heights stay honest

- `_conversation_row_render_height` gains the wrapped-name line count:
  `height = name_lines + 1 (metadata) + (1 if badge)`. The star button's
  matched height (set per-row so the star spans the full row) uses the same
  new signature.
- `_conversation_browser_rows_height` and `_legacy_conversation_list_height`
  sum per-row heights (button height + 1 margin) using the **same wrap
  helper with the same budget** used to build each label — the helper is pure
  and deterministic, so label building and height summing cannot disagree.
- `CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT = 3` in
  `Workspaces/display_state.py` also feeds the visible-row-count heuristic
  (`body_height // row_height`). It stays as the nominal minimum row height
  with a comment; the slight overestimate for lists containing wrapped rows
  only affects how many rows are loaded (the list scrolls), not layout.

## CSS changes (`css/components/_agentic_terminal.tcss`)

- `#console-left-rail-body`: add `scrollbar-gutter: stable;`.
- `.console-conversation-browser-row-line`: add `height: auto;`.
  **This fixes a latent bug the change would otherwise trigger**: Textual's
  `Horizontal` defaults to `height: 1fr`, and the current CSS never overrides
  it, so the list height is divided *equally* among row-lines. That happens
  to be correct only while every row is the same height (and is likely
  already subtly wrong for mixed badge/non-badge lists, the task-226 case).
  With 3- and 4-line rows routinely adjacent, equal division would stretch
  short rows and crop tall ones; `height: auto` makes each line size to its
  explicitly-heighted button.
- `.console-workspace-conversation-row`: keep `padding: 0 1`,
  `text-align: left`, `content-align: left middle`; `min-height: 2` remains
  correct (shortest row is name + metadata).
- No bundle edits by hand — the app rebuilds `tldw_cli_modular.tcss` at boot.

## Cleanups riding along

- `GLYPH_ACTIVE` import removed from `console_workspace_context.py` (both
  marker sites go away; the glyph remains in use elsewhere).
- The `text.lstrip('> ')` fallback in `_conversation_button`'s tooltip becomes
  moot (both call sites always pass `tooltip_label`); simplify.

## Testing

- **Unit tests** for `wrap_console_conversation_title`: short, exact-fit,
  two-line, two-line-with-ellipsis, spaceless long token, budget floor,
  wide-character (CJK/emoji) titles.
- **Update existing assertions** on the old marker/20-char format across
  `Tests/UI/test_console_agent_rail.py`,
  `Tests/UI/test_console_workspace_context_rail.py`, and
  `Tests/UI/test_console_native_chat_flow.py`.
- **Mount test** via `app.run_test()` at two terminal widths asserting
  (a) labels wrap at the expected budget and (b) the precomputed list height
  equals the sum of rendered row heights.
- The Console visual-snapshot gate
  (`Tests/UI/test_workbench_visual_snapshots.py`) asserts chrome copy only —
  verified unaffected.
- **Live verification** (verify skill, tmux): real Console screen at wide and
  narrow terminal sizes; resize across a budget boundary; confirm no
  recompose oscillation with a scrolling rail.

## Out of scope

- Selected-row styling changes (bold/underline treatment stays as is).
- The Library conversations list and session-switcher modal (separate
  surfaces; can adopt the wrap helper later if wanted).
- Reordering or changing metadata content (`workspace - status - age`).
