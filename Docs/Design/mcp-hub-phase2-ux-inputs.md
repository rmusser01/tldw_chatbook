# MCP Hub — Phase 2 UX Inputs (from Phase 1 senior UX/HCI review, 2026-07-13)

Source: post-implementation UX/HCI review of the Phase 1 screens (QA captures in
`Docs/superpowers/qa/mcp-hub-phase1-2026-07/`). The A-batch (focus-vs-active chip
treatment, legible disabled actions, human copy for reason codes and built-in
exposure flags, rail left-alignment + width-aware truncation, inspector "Why ·"
framing) was applied in Phase 1 (`fix(mcp-hub): apply five approved UX fixes`).
The items below were explicitly deferred to Phase 2 by the user. They are design
inputs for the Phase 2 spec/plan, not yet backlog tasks.

## Structural (fold into Phase 2 design)

1. **Advanced escape hatch must become a collapsed disclosure.** It currently
   occupies most of the inspector at all times, and its section dumps can
   describe a different object than the selected server (e.g. built-in inventory
   shown while docs-server is selected) — two objects' facts in one pane with no
   boundary. Default collapsed ("▸ Advanced"), remember open state, and label
   the object its content describes. Phase 2's structured forms shrink its role.

2. **Canvas vertical field is wasted; master-detail loses the master.** Overview
   content occupies the top quarter; the detail view replaces the table entirely
   with no path back except knowing "All servers" in the rail. Add a breadcrumb
   ("← All servers") or keep a slim table above the detail. Let the table grow
   into available height.

3. **Recovery callouts: compact + actionable.** Three separately-bordered
   three-row boxes for three one-line facts, none actionable. Target: one-line
   strips, each with its remedy action (mockup precedent: "Open connector
   settings"), jump-to-server on activate.

4. **Informational callout variant.** Phase placeholders ("Tools mode arrives in
   a later phase") currently reuse the orange warning chrome — semantic dilution
   of the single alarm color. Add a neutral/informational `ds-` callout variant;
   every phased rollout will need it.

5. **Status color semantics.** Ready/Stale/Needs-setup are glyph+word only.
   Use theme tokens (green/amber/red) redundant with the glyphs (colorblind-safe)
   across rail badges, table Status column, and inspector badge.

6. **Keyboard discoverability.** Modes switch on 1-4 but nothing on screen says
   so. Ship the spec §5 shortcut bar (`[1-4] mode · [a] add server · [t] test
   tool · [r] refresh`) or Console-style contextual footer hints.

## Polish (ride with any Phase 2 task touching the area)

- "env (1)" → "1 env var" in the Auth column.
- Scope column is inert in Local source (all "Personal"/"—") — hide it until
  Server source makes it meaningful.
- Aggregate line could carry `ds-status-badge` treatment to anchor it.
- Rail tool counts ("docs-server · 3") should right-align in a consistent
  column rather than trail the name.
- Inspector zero-action state lost the legacy guidance copy ("Select Section:
  Inventory to inspect runnable tools") — restore an equivalent hint (also
  flagged by the final branch review).

## Engineering watch-items already recorded for Phase 2 (from reviews)

- Rail mount-echo guard must become one-shot-consume before real multi-option
  scopes land (A→B→A change would be swallowed as an echo today).
- `asyncio.Lock` hardening around the inspector's remove→mount window
  (worker-vs-pump interleave).
- Textual 8.2.7 gotchas now proven on this surface: `Select` posts `Changed`
  for its constructor value at mount; `Select.BLANK` is not a real API; Message
  classes in MCP-acronym widgets need explicit `namespace=`; round borders need
  ≥2 lines (height-1 buttons collapse); app-loaded CSS beats widget DEFAULT_CSS
  unconditionally.
