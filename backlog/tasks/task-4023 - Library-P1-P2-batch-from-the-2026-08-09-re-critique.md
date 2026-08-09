---
id: TASK-4023
title: Library P1/P2 batch from the 2026-08-09 re-critique
status: To Do
assignee: []
created_date: '2026-08-09 20:30'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library re-critique 2026-08-09, snapshot
`.impeccable/critique/2026-08-09T20-15-07Z__tldw-chatbook-ui-screens-library-screen-py.md`
(22/40; trend 23 → 21 → 27 → 22 → 22). The P0 (Escape crash) shipped in PR #1464; the three
highest-value findings are tasks 4020/4021/4022. This is the remainder — grouped for one pass,
split if any item grows.

**Accessibility / honesty (highest value in this batch)**
1. RC-07 — disabled state is colour-only, measured: Select-mode bulk buttons **1.08:1** when 0
   selected (the very buttons task-2853 added — present, focusable, meaningful, unreadable);
   Media `Select` when empty 1.45:1 and silent on click; Export ~1.4–1.51:1; Collections' three
   buttons **2.30:1 even when enabled**. Floor disabled contrast at 3:1, add a non-colour marker,
   and attach the reason to the control. The product's non-colour vocabulary already exists
   (`☐/☑`, `▸`, `┃…┃`, `(selected)`, `✓/○`) — disabled state simply never joined it.
2. RC-06 — the Notes canvas copy says "switch to Files" but the `Database | Files` strip
   (`library_screen.py:7333`) does not render on first paint: the fast rail-click path
   (`_replace_library_browse_canvas`) swaps only the inner canvas and never composes the strip;
   only a full recompose renders it. Named `task-3317` in an unmerged sibling branch's test
   docstrings — reconcile rather than duplicate.
3. RC-09 — DB sizes in the Details disclosure are computed once and never refreshed: UI showed
   `Prompts 148.0KB / Media 476.0KB` while disk incl. sidecars was 180.0KB/508.0KB; a recompose
   with no disk change corrected both. task-2859's WAL-inclusive helper is correct and stale.
4. RC-10 — F1 lists Escape 2–3× with contradictory labels (`- esc: focus rail` / `- escape: Back`
   / `- escape: Focus rail`), omits F6 on Search/RAG though the footer advertises it, says nothing
   about Collections on the Collections panel, and does not close on a second F1.

**Interaction grammar (the score's current ceiling)**
5. Four footer dialects across seven canvases (different separators, different key names:
   `F6 panes` vs `F6 next pane`, `/ Find` vs `/ focus search`); the hub's `i`/`n` shortcuts vanish
   elsewhere with no statement of whether they still work.
6. Three active-state markers (`▸` prefix, `┃…┃` bars, `(selected)` text) and three toolbar
   layouts (Media vertical, Notes 3×2 grid, Prompts/Skills single row).
7. `▸` carries two incompatible meanings: disclosure (`Details ▸`) vs silent cycler
   (`type: All ▸`, `mode: Search ▸`) that advances with no menu — the option set is undiscoverable.
8. Escape still inert on Export, Collections, and the Study staging canvas; the staging canvas has
   no back path at all; `Export…` from within Media navigates away with no return.

**Search**
9. RC-08 — results land ~30 rows below the fold behind the configuration panel; clicking `Run`
   leaves the visible half of the canvas pixel-identical. Enter in the rail search navigates and
   pre-fills but does not run. Two search inputs are live with different values and navigation
   silently overwrites one with the other; never-executed strings still enter `Recent searches`.

**Layout**
10. Media list renders in a ~30-char column on a 170-col terminal with the detail below it, and
    truncates titles at 17 chars while ~115 columns sit blank; the media viewer gives a 33-line
    document a 7-row viewport while spending 2 lines on a `file://` temp path.
11. At ≤100 cols the landing canvas vanishes entirely while the rail still reads "pick a section
    on the left".

**Copy**
12. "opens staging canvas" printed three times in the primary nav (a truthfulness fix that traded
    a lie for internal jargon); Collections stacks four "nothing here" sentences and still offers
    no "Add to collection" anywhere; export scope "Everything" excludes Prompts/Skills/Collections;
    `Type: plaintext` for a `.md` the viewer renders as markdown, with extensions stripped from
    every list title.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disabled controls meet a 3:1 floor, carry a non-colour marker, and state their reason at the control
- [ ] #2 The Files strip renders on first paint; the Notes copy and the rendered controls agree
- [ ] #3 Details DB sizes refresh rather than reporting a stale first reading
- [ ] #4 F1 lists each binding once with one label per key, includes the keys its own footer advertises, and closes on a second press
- [ ] #5 One footer grammar, one active-state marker vocabulary, and one meaning per glyph across the Library's canvases
- [ ] #6 Search results are visible at the point of action, Enter runs the search, and one query model backs both inputs
- [ ] #7 Each remaining copy/layout item is fixed or declined with a one-line reason in the notes
<!-- AC:END -->
