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
  cycling "type: All ▸" filter right after the heading; Conversations
  instead has a "Filter conversations… (Enter)" text box, which now
  renders above the empty-state text (task-2859: it used to sit below
  "No conversations yet.", reading as an afterthought).
- **Row list** — one two-line row per item: the title with a **▸** marker,
  then a dimmer second line (Media: type and age; Conversations:
  "3 messages - 4h"). Hovering a row shows its full title as a tooltip.
- **Preview block** — appears under the list once a row is selected: a few
  summary lines plus one action ("Open in viewer" for media, "Open in
  Console" for conversations). Hidden in Media while Select mode is active
  (see below) — it never shows an item outside the current selection.

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
- **"Export…"** (hidden while selecting) exports the whole current scope —
  for Media that means the current type filter — and **"Export selected"**
  exports just the checked rows. Both open the same "Export bundle (.zip)"
  form, covered in [import & export](import-and-export.md).

**Media's "Delete selected"** (Media only — Conversations has no delete;
see its own row below) is a second bulk action next to "Export selected".
Pressing it swaps the strip for a confirmation naming the count — "Delete N
selected items? You can undo right away — there's no Trash view to browse
later." — with "Delete" / "Cancel", the same in-place armed-button pattern
as the media viewer's own single-item "Delete" (never a popup modal).
Confirming moves every checked item to trash (the same soft-delete the
viewer's Delete uses) and updates the list and the rail's "Media N" count
immediately; if any item fails, the rest of the batch still completes and a
notice names how many could not be deleted. Row checkboxes are frozen while
the confirmation is showing, so the count you confirm is always the count
that gets deleted.

A successful delete leaves a receipt in the same spot — "✓ deleted · N
items" with "Undo" and "Dismiss" — until you act on it or start another
bulk delete. "Undo" restores every item the receipt names (or just the
ones still outstanding, if a prior undo partially failed); "Dismiss" clears
the receipt without restoring anything. There is no persistent Trash view
to browse later — restoring an item you've dismissed, or deleted in an
earlier session, means re-importing the same file from
[Import & export](import-and-export.md): it now restores the item instead
of refusing, whereas before it silently reported the file as already in
your Library with no way back.

### Media list

| Control | What it does |
|---|---|
| "type: All ▸" | Cycles through All plus each media type present in your Library. While filtered, a status line reads "2 of 5 · type: pdf". |
| "Export…" / "Select" | The shared grammar above; Export… is scoped to the active type filter. |
| Row press | Selects the row and shows the preview (title, "Type: …", "Updated: …"). |
| "Open in viewer" | Opens the selected item in the media viewer. |

Empty states: with nothing imported, "No media in your Library yet. Import
something to see it here."; with a filter that matches nothing, "No media
of type 'pdf'."

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
  no on-screen mark.
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
| "Delete" | Two-step: shows "Delete this media? Re-import the same file later to bring it back — there's no Trash view to browse." with "Delete" / "Cancel". Confirming trashes the item, returns to the list, and drops the rail's "Media N" count by one immediately, the same as the list's own "Delete selected". Unlike the list's bulk delete, there's no in-place Undo here — the way back is re-importing the same file. |

### Conversations

| Control | What it does |
|---|---|
| "Filter conversations… (Enter)" | Type and press Enter to filter by title substring (case-insensitive, over the loaded list). Status shows "2 matches for 'demo'". |
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
1. In **Media**, click "type: All ▸" — each press cycles to the next type.
2. The list narrows and the status line reads e.g. "2 of 5 · type: pdf".
   Cycle back around to "type: All ▸" to clear.

### Open a media item and search inside it
1. Click a row, then "Open in viewer".
2. Type into "Search content…". The status shows "Match 1 of N matches" and
   the matches are highlighted in the content.
3. Step through with "◀ Prev" / "Next ▶"; the current match is emphasized.

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

### Delete selected media items
1. In **Media**, click "Select", check the rows you want to remove.
2. Click "Delete selected" — the strip becomes "Delete N selected items?
   This moves them to trash." with "Delete" / "Cancel".
3. Click "Delete" to confirm (or "Cancel" to back out without deleting
   anything). The rows disappear and the rail's "Media N" count drops
   immediately; the items are trashed, not permanently destroyed.

## Keyboard & commands

Nothing screen-specific: these panels are mouse/arrow-driven buttons and
inputs. The only key worth naming is **Enter** to submit the
"Filter conversations… (Enter)" and "Search content…" boxes. Global
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
- **Conversations shows at most 75 rows**, newest first, and the filter
  only matches within what is loaded. Very old conversations may not
  appear; find them via [Search / RAG](search-and-rag.md) instead.
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

*Verified against dev @ 8bb6dd730 — 2026-08-09 (task-4022: a re-critique
found the confirm copy promised a Trash that didn't exist anywhere in the
product, a deleted file could never be re-imported (the dedup match didn't
exclude trashed rows), and bulk delete had no receipt or undo. Fixed: a
trashed match now restores on re-import instead of silently refusing;
"Delete selected"'s confirm copy and the single-item viewer's "Delete"
copy both now say what actually happens instead of promising a Trash view;
"Delete selected" leaves a "✓ deleted · N items" receipt with Undo/Dismiss
at the point of action).*
