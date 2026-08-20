# Library Media & Conversations — browsing source material and staging it into Console

## What this screen is for

These two Library panels are where you browse what the app has collected:
**Media** holds everything you have imported (documents, transcripts, web
pages), and **Conversations** lists every chat you have had in Console. Both
are read-mostly browsers — you come here to re-read content, search inside
it, mark it up, and hand pieces of it to Console as context or package them
into a bundle. Both panels share one interaction grammar, so learning one
teaches you the other.

## Getting there

Press **Ctrl+3** (or click **⌃3 Library** in the nav bar, or **Ctrl+P** →
"Library"), then in the left rail's **Browse** section click **Media** or
**Conversations**. Each rail row shows its count. The selected row's canvas
fills the center of the screen.

## Layout tour

![Media viewer](../images/library/media-viewer.svg)

Both list canvases follow the same shape, top to bottom:

- **Toolbar** — the panel heading with a count ("Media (3)" /
  "Conversations (3)" — task-2859: Conversations previously had no
  heading at all, so its top row read as bare "Export…"/"Select" with
  nothing naming the canvas), plus "Export…" and "Select". Media adds a
  "type: All types" chooser right after the heading (press it to open a
  bounded list of every type — ✓ marks the active one — and pick directly) and a
  "Trash" action (the browsable Trash view — see "Media Trash" below);
  Conversations instead has a "Filter conversations… (Enter)" text box,
  which now renders above the empty-state text (task-2859: it used to sit
  below "No conversations yet.", reading as an afterthought).
- **Row list** — one two-line row per item: the title with a **▸** marker,
  then a dimmer second line (Media: type and age; Conversations:
  "3 messages - 4h"). Hovering a row shows its full title as a tooltip.
  Media rows scroll independently above a pinned 20-item pager, so paging
  controls remain visible after moving through a full page.
- **Preview block** — a few summary lines plus one action ("Open in
  viewer" for media, "Open in Console" for conversations). On a wide
  terminal the **Media** list shows it **beside** the row list — list on
  the left, preview on the right, each half scrolling on its own — the
  same split shape as the Collections workbench; below the Library's one
  width breakpoint (the same crossing that compacts Notes) it returns to
  the stacked under-the-list flow. Hidden in Media while Select mode is
  active (see below) — it never shows an item outside the current
  selection; in the wide split the right half then says "No preview in
  Select mode." instead of sitting blank. Conversations keeps the stacked
  shape at every width, and the Trash view and media viewer are single
  surfaces that always use the full canvas width.

Opening a media item swaps the list for the **media viewer** (pictured
above): "‹ Back to list", the title, metadata lines, then the "Content",
"Analysis", and "Highlights" sections, and an action row at the bottom.

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
| "type: All types" | Opens one bounded keyboard list containing the complete type set, with ✓ on the active choice. "All types" means no filter; a stored type literally named "All" remains a separate selectable value. Press Escape (or pick the current choice) to cancel. |
| "Previous" / "Next" | Moves through exact 20-item pages after the active query, type, and sort are applied. The final page may contain fewer rows; disabled buttons explain why they cannot move. |
| "Retry" | Repeats a failed page request. If retained rows may be out of date, unsafe row and bulk actions stay disabled until recovery succeeds. |
| "Export…" / "Select" | The shared grammar above; Export… is scoped to the active type filter. |
| "Trash" | Opens the Trash view — every deleted media item, restorable per item (see "Media Trash" above). Hidden while selecting, like "Export…". |
| Row press | Selects the row and shows the preview (title, "Type: …", "Updated: …"). |
| "Open in viewer" | Opens the selected item in the media viewer. |

Empty states: with nothing imported, "No media in your Library yet. Import
something to see it here."; with a filter that matches nothing, "No media
of type 'pdf'."

The pager reports the exact visible range, total, and page. Changing page or
type clears current-page selection with a visible "Selection cleared."
notice. A failed request keeps the last applied page visible instead of
silently replacing it with a partial or broad Library snapshot.

### Media viewer

- **Metadata** — lines for "Type:", and when present "Author:", "URL:",
  "Keywords:", "Updated:".
- **"Content"** — the stored text ("No stored content." when empty). For
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
  (submit an empty box) unpins them back into the Content section.
- **"Analysis"** — a stored analysis text you can view and edit ("Edit
  analysis", or "Add analysis" when empty; "No analysis yet." otherwise).
  This section only edits text — it never calls a model; analysis is
  produced at import time (the "Analyze after import" option) or written by
  hand here.
- **"Highlights"** — saved quotes from this item ("No highlights yet." when
  empty). Expand the collapsed **"Add highlight"** section, fill "Quote"
  (required), optionally "Note (optional)" and "Color (optional)", and
  press "Add highlight". Each saved highlight shows the quote with a
  color swatch, its color/note details, and a "✕ Delete" button.
- **Action row**:

| Button | What it does |
|---|---|
| "Edit" | Opens an inline form (Title, Author, URL, Keywords) with "Save" / "Cancel". |
| "Use in Console" | Stages this item as context for your next Console message. |
| "Read it later" ↔ "Remove from read-it-later" | Toggles the item on your read-it-later list. |
| "Open in Library ▸ Media" | Returns to Library's own Media surface (the separate Media screen was retired). |
| "Delete" | Two-step: shows "Delete this media? You can undo right away, or restore later from Trash." with "Delete" / "Cancel". Confirming trashes the item, returns to the list, and drops the rail's "Media N" count by one immediately, the same as the list's own "Delete selected". The list you land on shows the same receipt as a bulk delete — "✓ deleted · 1 item · in Trash" with "Undo" / "Dismiss" — "Undo" restores in place, and the Trash view holds the item for later either way. |

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

## Common tasks

### Filter media by type
1. In **Media**, click "type: All types" — a bounded list of every stored
   type appears in place of the toolbar, with ✓ on the active one.
2. Click the type you want. The list narrows and the status line reads
   e.g. "2 of 5 · type: pdf". Pick "All types" to clear the filter, or
   press Escape to close the list without changing anything. A stored type
   literally named "All" is distinct from the unfiltered choice.

### Open a media item and search inside it
1. Click a row, then "Open in viewer".
2. Type into "Search content…" and press Enter. The status shows
   "Match 1 of N matches", the matches are highlighted in the content, and
   the whole search bar pins to the top of the viewer for as long as the
   search is active.
3. Step through with "◀ Prev" / "Next ▶"; the current match is emphasized
   and the pinned bar keeps the count and both controls in view while you
   navigate.

### Highlight a passage
1. In the viewer, scroll to **Highlights** and expand "Add highlight".
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
The other panels are mouse/arrow-driven buttons and inputs. **Enter** submits
the "Filter conversations… (Enter)" and "Search content…" boxes. Global
navigation keys live in the [guide index](../index.md).

## Related settings & docs

- Neither panel owns any config.toml keys; media arrives via
  [import & export](import-and-export.md) (Import media, including the
  "Analyze after import" and chunking options that shape what the viewer
  shows).
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

*Verified against codex/library-top-level-pagination — 2026-08-20
(TASK-16483 / ADR-067: complete-source Media type filtering and exact 20-item
database pages, pinned pager at 100x30 and 170x48, bounded complete facet
chooser with an unambiguous "All types" choice, retained stale recovery,
selection clearing, and metadata-only diagnostics).*
