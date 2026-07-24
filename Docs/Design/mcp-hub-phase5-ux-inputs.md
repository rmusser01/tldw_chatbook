# MCP Hub — Phase 5 UX inputs (from the Phase 4 sr-UX/HCI review)

Deferred structural items from the Phase 4 Permissions-mode review (2026-07-16),
plus design tensions surfaced by the Console/Library idiom study. Feed into
Phase 5 (chat bridge + Audit mode) planning alongside the spec.

## B-batch (structural, deferred)

1. **Kill-switch redesign.** The A-batch shipped a Library-style toggle Button
   ("block MCP tools in chat: yes/no ▸") fixing the polarity trap, but the
   control still persists a store field with no runtime effect until the chat
   bridge. Phase 5: wire it into send-time assembly the moment the bridge
   lands, and consider the affirmative form ("Allow MCP tools in chat") with
   the store mapping inverted at the seam. Until wired, consider disabling
   with an explanatory suffix.
2. **Inspector inversion.** The permission rule explanation is the point of
   the mode, yet "Advanced (legacy control plane)" is expanded by default and
   owns ~80% of the pane. Collapse Advanced by default in Permissions/Tools
   modes; give the freed space to the rule detail. (Phase 6 retires Advanced
   from default view entirely — this is the interim step.)
3. **Cascade provenance display.** Replace the single origin sentence with a
   three-line cascade (Tool override → Server default → Global default) with
   the winning rung highlighted — also the cheapest way to teach what
   Space's "Inherit" rung means.
4. **Cross-mode jump.** "Change in Permissions" affordance from a blocked or
   gated Test Tool result (and from Tools-mode State cells) that switches
   mode AND selects the governing row. Today the user must remember, switch,
   re-find.
5. **Space-cycle discoverability, properly.** The A-batch put the cycle hint
   in the preview-strip legend line. The real fix is the persistent bottom
   key-hint strip both mockups show (MCP.png: "Space toggle  A approval  D
   deny  T test tool …") — blocked on the app-wide dead `AppFooterStatus`
   mounting defect (backlog task; affects every BaseAppScreen). Decide:
   fix the mounting, or ship an in-screen hint strip widget.
6. **Mutation feedback + undo.** Space instantly persists with no echo and no
   undo; an accidental Off is silent. Direction: status-strip echo
   ("list_characters → Allow") plus single-step undo.
7. **Semantic state colors — DESIGN TENSION, decide in Phase 5.** The
   Console/Library idiom (and current MCP rendering) is monochrome state text
   + glyph-only differentiation; the New_UI mockups (MCP.png, Settings.png)
   color the state words themselves (green/amber/red). Pick one direction for
   the whole app; if coloring, use the existing `$ds-status-*` tokens and
   keep glyphs as the color-blind-safe channel.
8. **Result payload readability.** "OK · 1.2s" is followed by one wrapped
   raw-JSON paragraph mixing result and governance metadata. Pretty-print;
   split governance metadata into its own dimmed block. (Feeds directly into
   Phase 5's Audit ▸ Executions detail view — design once, use twice.)
9. **Matrix scale posture.** 15 rows leave the canvas mostly empty, but 100+
   tools need what Tools mode has: a text filter, plus collapsible server
   groups; decide what the preview strip summarizes under a filter.
10. **Server-source empty-state routing.** Connect/refresh recovery actions
    are disabled for the server source — per-source action wiring and
    source-appropriate guidance needed (from the Phase 3 review, re-confirmed).

## Idiom notes for Phase 5 implementers (from the Console/Library study)

- Counts lines: lowercase state vocabulary, no noun, " · " separators
  (`library_ingest_state.py:416` precedent) — now used by the permissions
  preview strip.
- Toggles: cycling Buttons, never Textual Checkbox/Switch (they render
  unreliably; `library_notes_canvas.py:407` rationale) — the kill switch now
  follows this.
- Hierarchy: two-space label indents + structurally separate dimmed group
  headers; `▸ ` marker reserved for the active row.
- Toasts: verb-first with parenthetical context ("Test already running for
  search_docs (docs-server)."); internal errors show exception class name,
  not raw message.
- Empty-state copy: "No {noun} selected." / "Select {noun} to {verb}."
- No legend precedent existed before the permissions strip added one — if
  legends spread, standardize the format (dimmed, " · " separated).

## Cosmetic carry-overs

- The by-key unverifiable-tool arm notice ("can't be verified against the
  catalog") vs genuine config-changed copy — shipped in the A-batch; keep the
  distinction when Audit mode renders these events.
- "1 asks" fixed via count-vocabulary; keep the same builder for any Audit
  summaries.
