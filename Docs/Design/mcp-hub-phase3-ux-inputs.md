# MCP Hub — Phase 3 UX Inputs (from Phase 2 senior UX/HCI review, 2026-07-14)

Source: post-implementation UX/HCI review of the Phase 2 screens (QA captures in
`Docs/superpowers/qa/mcp-hub-phase2-2026-07/`). The Phase 2 A-batch (import
action-row stranding, status-colored error/warning text, left-aligned inspector
action stack, state-driven Save enablement, neutral aggregate text with a
colored worst-state glyph, adaptive rail count padding) was applied in Phase 2
(`fix(mcp-hub): six Phase 2 UX polish fixes (A1-A6)`). The items below are
design inputs for the Phase 3 (Tools mode) spec/plan.

## Structural (fold into Phase 3 design)

1. **Canvas vertical void, round two.** The servers table owns the full column
   height; with few rows there is a sea of empty space between the last row and
   the callouts pinned at the container bottom. Real fix: `height: auto` capped
   growth (max-height to the pane) so callouts hug the last row. This interacts
   with the table-growth requirement and Tools mode's own canvas layout —
   design them together.

2. **Focused-empty-scroll dashed outline.** A focused `VerticalScroll` with
   little content draws a full dashed rectangle around dead space (visible in
   the lifecycle capture). Adopt the design system's quieter focus treatment
   for content panes, or suppress the outline when the pane has no scrollable
   overflow.

3. **CHECKING should carry a time expectation.** "Working — connect…" gives no
   elapsed time or bound; show "up to {timeout}s" or an elapsed counter so the
   user can decide wait-vs-cancel.

4. **The inspector idles while forms are open.** While add/edit/import panels
   are up, the inspector shows the empty-state copy. It is the natural home
   for contextual field help (env placeholder examples, import format notes,
   per-candidate detail for import preview).

5. **Selective import.** The import preview is all-or-nothing; per-candidate
   checkboxes (default checked) with the apply button reflecting the count.

## Polish (ride with any Phase 3 task touching the area)

- Cancel button underline focus artifact in the profile form.
- Hover tooltips occasionally render as a visible text line below the toolbar
  in captures — verify tooltip layering.
- Import preview entries could show the source filename when loaded from file.
- Table Status column stays monochrome (documented limitation); revisit with
  Rich per-cell `Text` styles using ANSI theme colors if Phase 3 renders the
  tools catalog as a DataTable too.

## Engineering backlog carried from Phase 2 reviews (file as tasks post-merge)

- Mutations-panel remount on passive resync wipes typed input (edit-mode
  external-record path lacks the `_form_visible()` guard family).
- Slot delete / secret clear fire on single click — adopt the arm-then-confirm
  pattern (T7 precedent).
- Slot secret Input clears before the in-flight guard may swallow the submit —
  clear only after acceptance.
- Deleted-profile runtime-state resurrection (lifecycle completion re-writes
  state for a deleted id; recreated same-id profile inherits stale badge).
- Catalog read amplification: one refresh = 2N+1 full JSON store loads — add a
  batch accessor on `LocalMCPStore`.
- User cancel records as `ok=False`/"Cancelled" and renders as a red failure —
  cancel deserves neutral semantics.
- All-failed import batch closes the panel and discards the pasted JSON — keep
  the panel open with the error summary.
- `mcp_import._PLACEHOLDER_RE` accepts unbalanced braces (form regex was
  fixed; align the importer).
- `notify()` toasts interpolate profile ids without markup escaping
  (style-injection only; also pre-existing in "Saved {id}.").
- Gate-copy precedence in the genuine no-target server-source state shows the
  scope message instead of "select a target".
