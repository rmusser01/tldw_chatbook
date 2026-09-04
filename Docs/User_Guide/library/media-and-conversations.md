# Library Media & Conversations — browsing source material and staging it into Console

## What this screen is for

These two Library panels are where you browse what the app has collected:
**Media** holds everything you have imported (documents, transcripts, web
pages), and **Conversations** lists every chat you have had in Console. Both
are read-mostly browsers — you come here to re-read content, search inside
it, mark it up, and hand pieces of it to Console as context or package them
into a bundle. Media uses a reading-first three-pane shell; Conversations
keeps the existing list-and-preview browser.

## Getting there

Press **Ctrl+3** (or click **⌃3 Library** in the nav bar, or **Ctrl+P** →
"Library"), then in the left rail's **Browse** section click **Media** or
**Conversations**. Each rail row shows its count. The selected row's canvas
fills the center of the screen.

## Layout tour

```text
Wide

+ Library +--->+ Items +--->+ Reader -------------------------------+
| Browse        | Filter     | title · source · date                 |
| Media         | item rows  | Find · Read later · Use in Console   |
| ...           | ...        | Read · Analysis · Highlights · Info  |
+---------------+------------+ complete stored content               |

Narrow

+--->+--->+ Reader -----------------------------------------------+
| both pane grips remain reachable; Reader gets the available width |
```

Media has three stable roles:

- **Library** — the normal Library navigation rail.
- **Items** — a local-only list with **Filter media**, type selection,
  paging, bulk actions, and balanced two-line rows.
- **Reader** — a permanent reading surface. Selecting another row updates
  Reader in place; the Items list is not replaced.

Library and Items each have a five-column, full-height grip. **`<---`**
collapses the pane to its left and **`--->`** expands it. The grips are
clickable and keyboard-operable. Reader has no grip and never collapses.
Your manual pane choices are remembered. If the terminal is too narrow, the
screen temporarily collapses Library first and then Items; widening the
terminal restores the remembered layout instead of saving the temporary
responsive state.

While another row is loading, Items distinguishes a **Loading ·** row
prefix from the settled **Loaded ·** one. Reader may keep the prior item visible,
but names both items until the new detail settles. Late or failed loads cannot
replace a newer selection. Conversations retains its existing paged
list-and-preview layout.

## Features & controls

### The shared select / export grammar

- **"Select"** switches the list into select mode: every row gains a **☑/☐**
  checkbox, and a strip appears showing "N selected", "Select all N shown",
  "Clear", and "Export selected" (Media also adds "Delete selected" — see
  below). The button relabels to **"Done"** to exit; entering or leaving
  select mode clears the selection, and leaving with items still checked
  shows a quiet "Selection discarded (N items)" notice so exiting is never a
  silent no-op.
- **Disabled actions announce themselves.** While nothing is checked,
  "Export selected"/"Delete selected" read **"○ Export selected"** /
  **"○ Delete selected"** — the leading **○** is the Library's disabled
  marker (the same ✓/○ pair the ingest toggles use), so the state never
  depends on colour alone — and their tooltips say what to do ("Select one
  or more items…"). The same goes for **"○ Select"** when the list is
  empty ("Nothing here to select yet."). Checking the first row flips the
  labels back in place.
- **"Export…"** (hidden while selecting) exports the whole current scope —
  for Media that means the current type filter — and **"Export selected"**
  exports just the checked rows. Both open the same "Export bundle (.zip)"
  form, covered in [import & export](import-and-export.md).

**Media's "Delete selected"** (Media only — Conversations has no delete;
see its own row below) is a second bulk action next to "Export selected".
Pressing it swaps the strip for a confirmation naming the count — "Delete N
selected items? You can undo right away, or restore later from Trash." —
with "Delete" / "Cancel", the same in-place armed-button pattern
as the media viewer's own single-item "Delete" (never a popup modal).
Confirming moves every checked item to trash (the same soft-delete the
viewer's Delete uses) and updates the list and the rail's "Media N" count
immediately; if any item fails, the rest of the batch still completes and a
notice names how many could not be deleted. Row checkboxes are frozen while
the confirmation is showing, so the count you confirm is always the count
that gets deleted.

A successful delete leaves a receipt in the same spot — "✓ deleted · N
items · in Trash" with "Undo" and "Dismiss" — until you act on it or start
another delete. The viewer's single-item "Delete" leaves the same receipt
("✓ deleted · 1 item · in Trash") in the list it returns you to — single
and bulk delete share one undo story. "Undo" restores every item the
receipt names (or just the ones still outstanding, if a prior undo
partially failed); "Dismiss" clears the receipt without restoring anything.
"Undo" is the at-point convenience; the durable way back is the **Trash
view** the receipt points at (see "Media Trash" below), which lists every
deleted item — including ones from earlier sessions — and restores them
per item. (Re-importing the same file from
[Import & export](import-and-export.md) also still restores a trashed
match instead of refusing.)

### Media Trash

Press **"Trash"** on the Media toolbar to swap the list for the Trash
view: "‹ Media" (back), a "Trash (N)" heading, and one two-line row per
deleted item — the title, then a dim "type · trashed 2h" line saying when
it was deleted, newest first. Press a row to select it (the **▸** marker
moves), then **"Restore"** to put it back: the row leaves the Trash, the
rail's "Media N" count goes up in place, a "Restored 'Title'." line
confirms it, and the item is back in the media list (and in search
results) exactly as it was — restore never rewrites the item. "‹ Media" or
Escape returns to the list.

Notes on the edges: with nothing deleted the view says "Trash is empty.
Items you delete from Media land here." and "Restore" reads "○ Restore"
with a reason tooltip; if the trash holds more items than one fetch page,
a status line says "showing X of N" honestly. Entering Trash clears any
"✓ deleted…" receipt still showing on the list — the Trash view is the
durable path that receipt pointed at. Trashed items are **excluded from
search** (Library search and RAG keyword retrieval both skip them) until
restored.
There is no permanent-delete or empty-trash action here yet — restoring is
the only operation, and nothing is ever removed from the Trash except by
restoring it.

### Media list

| Control | What it does |
|---|---|
| "Filter media" / "Clear filter" | Searches the complete local Media source before paging; it is separate from Find in item. Clearing restores the unfiltered selection when it is still available. |
| "type: All types" | Opens one bounded keyboard list containing the complete type set, with ✓ on the active choice. "All types" means no filter; a stored type literally named "All" remains a separate selectable value. Press Escape (or pick the current choice) to cancel. |
| "Previous" / "Next" | Moves through exact 20-item pages after the active query, type, and sort are applied. The final page may contain fewer rows; disabled buttons explain why they cannot move. |
| "Retry" | Repeats a failed page request. If retained rows may be out of date, unsafe row and bulk actions stay disabled until recovery succeeds. |
| "Export…" / "Select" | The shared grammar above; Export… is scoped to the active type filter. |
| "Trash" | Opens the Trash view — every deleted media item, restorable per item (see "Media Trash" above). Hidden while selecting, like "Export…". |
| Row press / Enter | Selects the item and loads it into the permanent Reader; Enter bypasses the short traversal-settle delay. In Select mode, it toggles the row's checkbox instead. |
| Library / Items grip | Collapses or expands that pane and remembers the manual choice. Responsive collapses caused by terminal width are not saved. |

Empty states: with nothing imported, "No media in your Library yet. Import
something to see it here."; with a filter that matches nothing, "No media
of type 'pdf'."

The pager reports the exact visible range, total, and page. Changing page or
type clears current-page selection with a visible "Selection cleared."
notice. A failed request keeps the last applied page visible instead of
silently replacing it with a partial or broad Library snapshot.

### Media Reader

Reader stays mounted beside Items and keeps one mode visible at a time:
**Read**, **Analysis**, **Highlights**, or **Info**. The chosen mode persists
while you move through items. Missing analysis or highlights produces an
item-specific empty state; it does not silently switch modes.

- **Read** — the complete stored text ("No stored content." when empty). For
  markdown-flavored media (a `.md`/Obsidian-style item whose content has a
  real heading, table, or fenced code block), a "Rendered (selected) |
  Raw" toggle appears above the box and defaults to **Rendered** — headings,
  tables, and code render properly instead of showing literal `#`/`##`/`|`
  characters, using the same renderer as Notes' own "Preview". Press
  "Raw" to see the plain source instead. Below the toggle (or directly
  above Content for everything else) is a "Search content…" box — its
  placeholder reads "Search content (raw text)…" whenever the toggle is
  present, since search always matches the raw stored text regardless of
  which view is showing. Typing a query shows "Match 1 of 4 matches" (or
  "No matches") and a "◀ Prev" / "Next ▶" pair that steps through matches
  and wraps at either end — in both views. Only the visual highlighting of
  the current match is Raw-only; Rendered shows the same step count with
  no on-screen mark. While a search is active, the search box, match
  count, and "◀ Prev" / "Next ▶" pin to the top of the viewer pane so
  they stay visible while you step through matches — even in a small
  terminal, and no matter how far you scroll. Clearing the query
  (submit an empty box) unpins them back into the Read section. Eligible local
  PNG, JPEG, and WebP files can also show an inline image above the complete
  text. **Hide preview** / **Show preview** affects only that item for this
  session. An unavailable or failed preview reports the problem and leaves
  every character of stored text readable; GIF, PDF, audio, video, remote URL,
  and server-item previews are not fetched or rendered here.
- **Analysis** — stored analysis text you can view and edit ("Edit
  analysis", or "Add analysis" when empty; "No analysis yet." otherwise).
  This section only edits text — it never calls a model; analysis is
  produced at import time (the "Analyze after import" option) or written by
  hand here.
- **Highlights** — saved quotes from this item ("No highlights yet." when
  empty). Expand the collapsed **"Add highlight"** section, fill "Quote"
  (required), optionally "Note (optional)" and "Color (optional)", and
  press "Add highlight". Each saved highlight shows the quote with a
  color swatch, its color/note details, and a "✕ Delete" button.
- **Info** — metadata and provenance: backend-qualified ID, original source,
  stored representation, preview status, and the representation **Use in
  Console** will send. The Items catalogue is local-only. A finished server
  import may open one read-only compatibility detail labelled **Server item ·
  not in local Media list**; it does not become a local Items row.
- **Primary toolbar**:

| Button | What it does |
|---|---|
| "Find" | Opens Find in item for the loaded representation; this never filters Items. |
| "Use in Console" | Stages this item as context for your next Console message. |
| "Read later" ↔ "Remove later" | Toggles the loaded item's persisted reading-list state. |
| "More" | Keeps secondary actions reachable: Edit metadata, Open original when available, Open manager, and Move to trash. Narrow layouts retain these actions here rather than hiding them. |
| "Move to trash" | Two-step, title-specific confirmation. Success selects the adjacent item and leaves a bounded Undo receipt; Trash remains the durable recovery path. |

### Review sets

A review set pins an ordered snapshot of media items so you can work through
them one by one, with your place and progress saved between visits.

- **Create** — **Review these** on the media list pins the whole filtered
  result (in the current sort order, capped at 500 items with a notice when
  trimmed); in Select mode, **Review selected** pins just the checked items in
  list order and leaves Select mode. Creating a set activates it and opens its
  first item in the Reader. If another set was mid-walk, a notice names it and
  its progress ("Paused 'Read later' at 1 of 2 · 0 reviewed. Resume from
  Sets.") — creating never silently strands a walk.
- **Resume on entry** — opening the media area with a set active loads its
  current item into the Reader automatically, on every entry, so the banner
  and the open document always agree (Escape shows the list until the next
  entry).
- **Walk** — while a set is active the Reader carries a banner naming the
  set, your place, and the open item's own state ("Reviewing: All media — 2
  of 14 · 1 reviewed · ✓ reviewed"), and the footer shows the same place. `]` advances and marks the item you leave as reviewed;
  `[` goes back without marking; `m` toggles the loaded item's reviewed mark;
  a final `]` on the last item marks it done in place. **Escape** leaves the
  Reader but keeps the set active — re-entering resumes at your cursor.
  **R** (Exit review) deactivates the set without deleting it.
- **Resume / switch / dismiss** — **Sets** on the media list title row opens
  the saved-set picker: each row shows the set's name, live progress, and
  created date (so two "2 selected items" sets stay distinguishable), with
  the active set marked `✓`. Picking a set activates it (deactivating any
  other) and lands at its saved cursor; picking a completed set reopens it.
  **Dismiss** soft-deletes a set and leaves a "✓ dismissed · name" receipt on
  the media list with **Undo** (restores the set — cursor, reviewed marks,
  and active state intact) and **Dismiss** to clear the receipt. **Review
  read-later** in the same picker builds a new set from your read-later
  queue, newest saves first.
- Deleted media items become skipped tombstones: progress counts only items
  that still exist, and a set whose items were all removed reports "No items
  to review" instead of completing.

*Verified against fix/media-review-continuity — 2026-09-04 (task-31233/34/36/38:
live create-from-selection → every-entry resume → dismiss-undo pass in tmux).*

### Conversations

Conversations use a fixed **20-item page**. The row list scrolls independently
while its pager stays visible underneath it. The pager always states the exact
range, total, and page: with 45 conversations it reads **"1-20 of 45 · Page 1
of 3"**, then **"21-40 of 45 · Page 2 of 3"**, and finally **"41-45 of 45 ·
Page 3 of 3"**. Previous and Next show a visible reason when unavailable, such
as "Already on the first page." or "No more results."

Filtering searches the **full conversation source before paging**, so a match
on the oldest page is still found; clearing the filter returns to unfiltered
page 1. Moving to another page or changing the filter leaves Select mode,
clears its current-page checkboxes, and shows "Selection cleared." If a page
cannot be refreshed safely, the last good rows remain visible but read-only,
the pager explains that the list may be out of date, and **Retry** repeats the
requested load.

| Control | What it does |
|---|---|
| "Filter conversations… (Enter)" | Type and press Enter to search conversation titles, stable IDs, and indexed message content before the 20-item result page is chosen. Clearing it restores unfiltered page 1. |
| "Previous" / "Next" | Moves through complete 20-item pages; the final page may contain fewer rows. Disabled buttons state why they cannot move. |
| Row press | Selects the row and shows the preview (title, "Messages: N", "Updated: age"). |
| "Open in Console" | Stages the conversation as **source context** in Console — see below. |
| "Export…" / "Select" | The shared grammar; export packages conversations into a bundle. |

Empty state: "No conversations yet. Chat in Console and it appears here."
There is no create, rename, or delete here — this panel treats your chats
as source material, not sessions.

**"Open in Console" does not resume the chat.** It hands the conversation
to Console as staged context, pre-filling the prompt "Use this conversation
as source context for my next question." — your next message continues in
the *current* Console session, grounded by the old conversation. To switch
back into a past session and keep chatting in it, use Console's own
conversation rail instead; that one resumes sessions, this one quotes them.
This explicit handoff **does not change the conversation's Auto or Assistant
setting**; it supplies one bundle of **staged context** for the next send.

## Common tasks

### Filter media by type
1. In **Media**, click "type: All types" — a bounded list of every stored
   type appears in place of the toolbar, with ✓ on the active one.
2. Click the type you want. The list narrows and the status line reads
   e.g. "2 of 5 · type: pdf". Pick "All types" to clear the filter, or
   press Escape to close the list without changing anything. A stored type
   literally named "All" is distinct from the unfiltered choice.

### Open a media item and search inside it
1. Click a row; it loads into Reader without replacing Items.
2. Press **Find**, type into "Search content…", and press Enter. The status shows
   "Match 1 of N matches", the matches are highlighted in the content, and
   the whole search bar pins to the top of the viewer for as long as the
   search is active.
3. Step through with "◀ Prev" / "Next ▶"; the current match is emphasized
   and the pinned bar keeps the count and both controls in view while you
   navigate.

### Highlight a passage
1. In Reader, choose **Highlights** and expand "Add highlight".
2. Paste the passage into "Quote"; optionally add "Note (optional)" and a
   color name or hex value in "Color (optional)".
3. Press "Add highlight" — it appears in the list with a ● swatch.

### Stage a conversation as Console context
1. In **Conversations**, click a row, then "Open in Console" in the preview.
2. Console opens with the conversation staged and the prompt "Use this
   conversation as source context for my next question." ready to go — edit
   or replace it, then send.

### Export selected items as a bundle
1. Click "Select", check the rows you want ("Select all N shown" grabs the
   whole visible list), then "Export selected".
2. The "Export bundle (.zip)" form opens — name, destination, and options
   are covered in [import & export](import-and-export.md).

### Recover something you deleted last week
1. In **Media**, click "Trash" on the toolbar.
2. Find the item ("Trash (N)" lists everything deleted, newest first, each
   row saying "type · trashed 3d"), press its row, then "Restore".
3. "Restored 'Title'." confirms it; "‹ Media" (or Escape) takes you back to
   the list, where the item — and the rail's "Media N" count — are back.

### Delete selected media items
1. In **Media**, click "Select", check the rows you want to remove.
2. Click "Delete selected" — the strip becomes "Delete N selected items?
   This moves them to trash." with "Delete" / "Cancel".
3. Click "Delete" to confirm (or "Cancel" to back out without deleting
   anything). The rows disappear and the rail's "Media N" count drops
   immediately; the items are trashed, not permanently destroyed.

## Keyboard & commands

The Media type chooser supports **Up/Down**, **Home/End**, and **Enter**;
**Escape** cancels without applying a choice and returns focus to its opener.
The Media grips accept **Enter** and **Space**. Arrow-key traversal moves the
Items selection with a short settle delay; **Enter** loads immediately.
**Escape** closes transient Reader state before moving outward through Items
and Library. While a review set is active, the Reader adds **]** (next in
set, marking the item you leave), **[** (previous, never marks), **m**
(toggle reviewed), and **R** (exit review, keeping the set resumable); the
footer shows these alongside your "X of M · N reviewed" progress. **Enter** submits the "Filter conversations… (Enter)", **Filter
media**, and "Search content…" boxes. Global
navigation keys live in the [guide index](../index.md).

## Related settings & docs

- Appearance settings remember the preferred Library/Items pane states.
  Automatic Library-rail width follows 3:13, bounded to 24–34 cells. When
  explicitly enabled, custom widths remain Library 24–48 and Items 32–72;
  ordinary layouts may temporarily compress the rail to preserve 40 content
  cells, while these adaptive readers may collapse or prioritize panes.
  **Reset layout** restores both panes open, automatic width, a dormant
  31-cell Library preference, and 40-cell Items preferences. Below 64 columns,
  ordinary Library routes switch between full-width rail and canvas stages via
  **‹ Library** (or **< Library** with ASCII glyphs). Responsive changes never
  overwrite saved preferences.
- Media arrives via
  [import & export](import-and-export.md) (Import media, including the
  "Analyze after import" and chunking options that shape what the viewer
  shows). Note that imported text is lightly sanitized on the way in (null
  bytes and unusual control characters become spaces; see
  [import & export](import-and-export.md) for the details), and every chunk
  is stamped with the chunking engine version that produced it.
- Both panels are retrieval sources for
  [Search / RAG](search-and-rag.md) — the "Media" and "Conversations"
  scope toggles there search exactly what you browse here.
- Console's **Save as...** action can file a reply into Media; the viewer's
  "Read it later" toggle feeds the same reading list used elsewhere in the
  app.
- [Library overview](../library.md) — the rail, the other panels, and how
  the pieces fit together.

## Quirks & troubleshooting

- **Highlight colors must parse.** "Color (optional)" accepts standard
  color names or hex values (e.g. `yellow`, `#ffcc00`); anything else is
  saved but renders as plain text with no tint on the swatch.
- **"Open in Library ▸ Media" stays in Library.** The separate Media
  screen it used to jump to was retired; the button now returns to
  Library's own Media surface.
- **Conversations are paged 20 at a time**, newest first. The filter searches
  the complete source before paging, so older matches remain reachable.
- **Media is paged 20 at a time after filtering.** Its type chooser comes
  from the complete distinct type set, not only the currently visible rows.
  If a refresh fails, Retry remains available while stale rows are read-only.
- **Prompts use the same fixed 20-item page grammar.** Their search, sort, and
  collection scopes apply before paging; selections retain captured versions
  across pages, and a failed refresh keeps the last applied rows read-only with
  an exact Retry action. See [Library prompts](prompts.md).
- **"Open in Console" can refuse with "Copy or link blocked Library
  sources into the active workspace before using them in Console."** The
  handoff requires the conversation to be eligible for the active
  workspace; until your sources are linked into it, staging is blocked
  (the same gate guards the other "Use in Console" actions).
- **Staging now actually reaches the model.** "Use in Console" (media)
  and "Open in Console" (conversations) used to stage content that
  displayed as attached but never made it into what the model was sent
  — that's fixed. What arrives today is a real, correctly attributed
  reference to the item, but a thin one: a short generic label (e.g.
  "Media staged: \<title\>"), not an excerpt of the actual content —
  upgrading that to a real excerpt is still open (task-2376).
- **No conversation delete here** — by design, this panel never modifies
  chats. Manage sessions from Console itself.
- **"Media item is unavailable."** — the item was removed while you had it
  open (for example from the Media screen); the viewer drops back to the
  list.

—
*Verified against c2cbb8081 — 2026-08-04 (PR-T1: media and conversation
handoffs delivering a real, attributed reference to the model on send is
covered by capture round-trip tests, task-2374; the live check's own
handoff scenario was blocked on that profile by an unrelated Library
workspace-eligibility gate).*
*Verified against dev @ 6b38a13b8 — 2026-08-07 (task-2858 Task 3, LIB-13:
the media viewer's Content section renders markdown-flavored media
through the same renderer Notes' "Preview" uses, behind a "Rendered |
Raw" toggle defaulting to Rendered).*

*Re-stamped against dev @ 4acb17a0b — 2026-08-07 (TASK-2857: "Open in
Media manager" is now "Open in Library ▸ Media" — task-2851 had already
retired the separate Media screen it used to jump to; "Export…"/"Export
selected" now open the "Export bundle (.zip)" form, not "Export
chatbook"; "Analyze after ingest" is now "Analyze after import").*

*Verified against dev @ 023a04a48 — 2026-08-07 (task-2859: the
Conversations canvas now opens with a "Conversations (N)" title header
matching Media's, and its filter box renders above the empty-state text
instead of below it).*

*Verified against dev @ 023a04a48 — 2026-08-07 (task-3020: the media
viewer's single-item "Delete" now drops the rail's "Media N" count
immediately too, matching "Delete selected"; Library-wide Escape/keyboard
behavior for the bulk-delete confirmation is covered in the
[Library overview](../library.md#keyboard--commands), not duplicated
here).*

*Verified against dev @ e13608106 — 2026-08-09 (task-4022: a re-critique
found the confirm copy promised a Trash that didn't exist anywhere in the
product, a deleted file could never be re-imported (the dedup match didn't
exclude trashed rows), and bulk delete had no receipt or undo. Fixed: a
trashed match now restores on re-import instead of silently refusing;
"Delete selected"'s confirm copy and the single-item viewer's "Delete"
copy both now say what actually happens instead of promising a Trash view;
"Delete selected" leaves a "✓ deleted · N items" receipt with Undo/Dismiss
at the point of action).*
*Verified against dev @ 642567627 — 2026-08-10 (task-4023 AC#1, RC-07:
the select-mode bulk buttons and the empty-list Select toggle described
above rendered at 1.39:1/1.45:1 — legible now (7.25:1 measured live via
ANSI decode), with the "○" disabled marker and reason tooltips added).*
*Verified against fix/settings-appearance-crash @ 57ad075de — 2026-08-10
(task-4023 AC#5/#7: the Media toolbar is one horizontal row (type filter,
Export…, Select) matching the sibling canvases; cycle buttons use the "⇄"
glyph with an option-listing tooltip; list titles render in full at wide
terminals instead of the old 17-character cut; the viewer's metadata says
"Type: markdown (stored as plaintext)" for items it renders as markdown;
and Export… now remembers where you came from — Escape on the Export
canvas returns to this canvas.)*
*Verified against feat/library-queue-batch @ a899cbf6a — 2026-08-11
(task-14901 / ADR-055: the viewer's single-item "Delete" now leaves the
same "✓ deleted · 1 item" receipt with Undo/Dismiss as "Delete selected" —
single delete is one-item bulk, sharing the bulk path's undo and its
in-flight interlock — and its confirm copy promises the undo instead of
pointing at re-import.)*
*Verified against feat/library-queue-batch @ db733c62b — 2026-08-11
(task-4025: the browsable Media Trash view described above — the "Trash"
toolbar action, per-item Restore through the existing restore seam, both
delete confirm copies and the receipt re-pointed at Trash per ADR-055
Pattern A, and the explicit search decision: trashed items stay out of
Library search and RAG retrieval until restored.)*
*Verified against feat/library-queue-batch @ 345da0422 — 2026-08-11
(task-14900: the Media list's wide side-by-side list | preview split
described in the layout tour — live at 170/121 cols, back to the stacked
flow at 119/100, keyboard traversal, Select mode and the bulk toolbar
checked in both layouts, "No preview in Select mode." placeholder wide
only.)*
*Verified against feat/library-queue-batch @ 0662e09f5 — 2026-08-11
(task-14902: the type filter converged on the Notes Sort chooser pattern —
pressing "type: All" swaps the toolbar for a one-row strip of every type
with ✓ on the active one; a pick applies directly (no more press-to-cycle),
Escape cancels and refocuses the opener, and the footer/F1 read
"enter choose type / esc cancel" while the strip is open. Checked live in
both the wide side-by-side and stacked layouts, mouse and keyboard-only.)*
*Verified against task/15774-burn @ 76e2b6c7e — 2026-08-15 (task-15774:
an active content search now pins the search box, match count, and
"◀ Prev" / "Next ▶" to the top of the viewer pane — at 80x24 they used to
sit below the visible fold exactly while being used; compositor-strip
evidence at exactly 80x24 before/after, plus a 120x40 pin that an
inactive search keeps today's in-flow layout with no reserved space.)*

*Verified against the chunking-engine-parity worktree — 2026-08-19
(chunking-engine-parity, doc-only on this page): the chunking-engine note
above reflects the engine swap only — the media viewer and Conversations
panel behavior is unchanged. Sanitization and the engine-version stamp live
in the import pipeline; details and verification pointers are on the
[import & export](import-and-export.md) page.*

*Verified against codex/library-top-level-pagination — 2026-08-20
(TASK-18914 / ADR-067: complete-source Media type filtering and exact 20-item
database pages, pinned pager at 100x30 and 170x48, bounded complete facet
chooser with an unambiguous "All types" choice, retained stale recovery,
selection clearing, and metadata-only diagnostics).*
