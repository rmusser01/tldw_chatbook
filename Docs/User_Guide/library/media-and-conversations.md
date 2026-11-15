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
Wide (Media)

+ Library ----+<---+ Items ------+<---+ Reader ---------------------------+
| Browse       | Filter        | title · source · date              |
| Media        | item rows     | Find · Read later · Use in Console |
| ...          | ...           | Read · Analysis · Highlights · Info|
+--------------+---------------+ complete stored content            |

Narrow (Media)

+--->+ Items ----------------------------------+--->+ Reader -------------+
| ‹ Library                                  | Select a media item  |
| filter · type · sort · item rows           | to read it here.     |
```

Every Library reader uses two full-height, five-column collapse controls.
The full width is clickable, including the space around the arrows.

Media has three stable roles:

- **Library** — the normal Library navigation rail.
- **Items** — a local-only list with a filter over title, content and
  keyword (its box is narrow, so it reads "Title/keyword…" and its tooltip
  spells out "Filter by title, content or keyword"), type selection,
  paging, bulk actions, and balanced two-line rows.
- **Reader** — a permanent reading surface. Selecting another row updates
  Reader in place; the Items list is not replaced.

Library and Items each have a full-height, five-column grip. **`<---`**
collapses the pane to its left and **`--->`** expands it. The grips support
clicks, Enter, and Space. Reader has no grip and never collapses. Your manual
pane choices are remembered.

The default Library and Items columns prefer five and ten extra cells,
respectively. At narrower widths those additions yield before the screen
collapses Library, then Items. This preserves the earlier collapse boundaries
with the five-column controls. Widening restores the preferred widths without
rewriting your settings. Explicitly reopening a pane takes effect immediately.
Where Library and Items do not both fit beside the Reader, reopening Library
closes Items, and selecting **Media** in the rail again keeps that choice.

- **The Items column grows with the terminal.** Once the Reader is
  comfortable the surplus is split between them, up to a 56-cell ceiling, so
  a 98-character title paints 46 characters at 235 columns (it painted 31
  before, in a fixed 40-cell column).
- **With no item open, Items takes the empty Reader's width.** When the
  Reader is only showing "Select a media item to read it here.", the columns
  it would give a document go to the Items list instead — so on a wide
  terminal a long title reads in full rather than truncating beside empty
  space. Opening an item restores the Reader's width, and closing back to the
  list widens Items again.
- **Each item is two rows** — its title, then its type and age — with no
  blank row between items, so a 52-row terminal lists 15 items rather than
  11.
- **The navigation rail joins Media at 112 columns.** Below that the screen
  shows Items and the Reader only, and the Items pane itself collapses below
  88 columns — except below **64 columns with nothing open in the Reader**,
  where the list is all there is to show: Items stays, the Reader keeps what
  is left for its placeholder, and a **‹ Library** (or **< Library** with
  ASCII glyphs) control at the top of the Items pane returns to the rail.
  It is the only width that shows that control; opening an item hands the
  width back to the Reader as usual. Going back to the rail gives it the
  width while you are there — selecting **Media** again returns to the list
  and its **‹ Library** control, so the two are a round trip rather than a
  one-way door.
- **Opening Items beside an open item gives the list the stage.** Where the
  terminal is too narrow for the list and a usable Reader side by side (up to
  82 columns), the Items grip opens the list across the whole width
  instead of leaving the item a few columns beside it. The item stays open
  behind the list, and the same grip closes the list again and returns to the
  item with focus inside it. In Media, opening another item from that list
  with Enter or a click shows it across the whole width with focus in it;
  **]** / **[** (the next or previous item) also show it across the whole
  width, and moving through the list with the arrow keys keeps the list.
  Widening the terminal ends the list's stage and keeps focus where it was in
  the list. Conversations, Collections, Prompts and Skills share two of these:
  Enter or a click opens the item across the whole width with focus in it,
  and widening keeps focus in the list. In Artifacts
  the list takes the stage too, and **F6**, or **Tab** onto the artifact's
  controls, returns to the open artifact. Below 64 columns with nothing open, each of these
  lists keeps the stage, as Media's does.

While another row is loading, Items distinguishes a **Loading ·** row
prefix from the settled **Loaded ·** one. Reader may keep the prior item visible,
but names both items until the new detail settles. Late or failed loads cannot
replace a newer selection. Conversations retains its existing paged
list-and-preview layout.

**The Reader tracks the visible set.** When the item open in the Reader
leaves the list — a filter that matches nothing, or deleting the loaded item
(from the viewer or as the sole item in a bulk delete) — the Reader stops
painting it and falls back to "Select a media item to read it here." rather
than showing a now-absent item. A filter that still has matches moves the
Reader to its first result instead. Undoing a delete re-adds the row, which
opens normally again.

**Row markers.** An item's second row says what it is and when it last
changed, and adds **· analysed** when that item already carries an analysis —
so you can see what is worth generating without opening anything
(`document · updated 5m · analysed`). The age is labelled, because a bare
"10m" on an `audio` or `video` row reads as the item's *length*; under a
minute it reads **updated just now**. The word is **updated**, not "added":
the value is the item's last-modified time — the same one the Reader's
preview line calls "Updated:" — so saving an analysis moves it. It shares the
Trash list's grammar (`pdf · trashed 3m`), which is also why it is
"updated 5m" and not "updated 5m ago": on this pane four cells are the
difference between a keyword row showing its term and hiding it.

The row that is open in the Reader ends **· loaded** (**· loading** while it
is fetching) — the state is a fact about the row, so it sits with the other
facts rather than in front of the title.

The row re-reads its state from the database the moment an
analysis is saved — from the Reader's Generate, or from a bulk Analyze run —
without re-paging the list; until task-31942 lands (the save does not yet
commit durably), the mark can lag the Reader on a real profile. Its title row
starts with a single one-cell **state slot**. With
no review set active it carries only the current-row pointer **▸** in the wide
layout (blank otherwise); once a set is active (see "Review these" below),
every item in the set carries **·** until you review it and **✓** afterwards,
and items outside the set keep the pointer or stay blank — the current row is
still highlighted, so nothing is lost when its **▸** gives way to a mark. The
marks update the moment you mark an item reviewed in the Reader, without
leaving it. Select mode
reuses that same cell for its **☑/☐**, so a row never carries two markers.

While a filter is active, a row it found only through one of its **keywords**
adds **· keyword: \<term\>** — the filter searches titles, item text and
keywords, so without it a hit whose title and body hold nothing you typed
reads as a mistake (`article · updated 2m · keyword: notes`). A row whose
title or text carries the term already shows you why it is there and says nothing
extra. Long tags are cut to ten columns (`keyword: quokkasand…`; five
wide CJK characters, or five flag emoji) to keep the line short — the cut
counts a flag by the two columns it paints and never leaves half of one, so
the row frame does not drift. It can still be too long for a narrow Items
pane, and the labelled age made that tighter: at the pane's narrowest the row
ends in an ellipsis a few characters into `keyword:` itself, before the term
starts. A row that is both analysed and a keyword hit can run out of room at
the default width too, and the row open in the Reader spends a further nine
columns on its `· loaded`, so at the automatic narrow width (100 columns) that
row is the first to clip.

The Reader's **Info** tab "Keywords:" line applies the same guard: it shows the
full stored keyword but drops a dangling half-flag so that surface's frame
does not drift either (the edit form still prefills the stored keyword
verbatim).

## Features & controls

### The shared select / export grammar

- **"Select"** switches the list into select mode: every row gains a **☑/☐**
  checkbox, and a strip appears showing "N selected", "Select all N shown",
  "Clear", and "Export selected" (Media also adds "Delete selected" — see
  below). The button relabels to **"Done"** to exit; entering or leaving
  select mode clears the selection, and leaving with items still checked
  shows a quiet "Selection discarded (N items)" notice so exiting is never a
  silent no-op. In Media, entering select mode this way (or with **s**, see
  below) also puts keyboard focus on a row, so Down and Space work right
  away, and clicking anywhere on a row — not just its checkbox — toggles it,
  including a fast second click on the same row. "Done" renders on its own
  row below the other select-mode actions rather than sharing a browse-mode
  slot such as "sort:".
- **Disabled actions announce themselves.** While nothing is checked,
  "Export selected"/"Delete selected" read **"○ Export selected"** /
  **"○ Delete selected"** — the leading **○** is the Library's disabled
  marker (the same marker every gated Library action carries — see
  [State glyphs](../library.md#state-glyphs)), so the state never
  depends on colour alone — and their tooltips say what to do ("Select one
  or more items…"). The same goes for **"○ Select"** when the list is
  empty ("Nothing here to select yet."). Checking the first row flips the
  labels back in place — the word itself does not move: an enabled
  select-mode action reserves the marker's own two cells, so "Export
  selected" starts in the same column whether or not the "○" is showing.
  You can see that on Media, Conversations and Prompts. Notes reserves the
  same width, but its select-mode row overflows the notes pane at every
  width today, so "Clear" and "Export selected" are off-screen there until
  that row is reworked. Conversations now splits its select toolbar into
  content-width rows the same way Media does — a summary row ("N selected",
  "Select all N shown") above a bulk-action row ("Clear", "Export selected")
  — so every label paints in full at every supported width (no more the old
  "Selec"/"Exp" clipping), and the word still holds its column across the
  disabled flip.

**Media's "Analyze"** (Media only) generates an analysis for every checked
item in one run, in list order, on its own row under Clear/Export/Review:

- Pressing it leaves select mode and reports progress **in the list**:
  "Analyzing 3 of 40 · 2 failed" while it runs, then "✓ analyzed · 38 of 40
  · 2 failed" when it settles ("✗ analyzed · 0 of 3 · 3 failed" if nothing
  succeeded). **Retry failed** re-runs only the items that failed;
  **Dismiss** clears the receipt. A clean run says just "✓ analyzed · 40 of
  40" with no failure count and no Retry.
- **Items that already have an analysis are never overwritten silently.**
  If any checked item has one, the first press runs nothing and offers
  "N of M already analyzed" with **Skip them** (analyze only the rest) and
  **Overwrite** (analyze everything, replacing what is there) — no
  Dismiss on this row; "Skip them" already is the change-nothing outcome
  and retires the card. Changing the filter, type or page, or toggling
  Select mode, withdraws the offer.
- One run at a time: a second press while one is in flight says "Analysis
  already running" rather than starting a second.
- With no analysis provider configured the action reads **"○ Analyze"** and
  an always-visible line beneath it states the reason and the next step
  ("No analysis provider is configured · Set one in Settings ▸ Providers &
  Models."), the same wording the Reader's Generate gives — so a
  keyboard-first user reads why without hovering. The tooltip still carries
  it too.
- The run belongs to the Library screen: leaving Library stops it, and a
  notice says where it got to ("Analysis stopped at 3 of 40 · reopen Select
  ▸ Analyze to continue; finished items are skipped"). Items already
  analyzed are skipped by a fresh run, so continuing is just re-selecting
  them and pressing Analyze again.

- **"Export…"** (hidden while selecting) exports the whole current scope —
  for Media that means the current type filter — and **"Export selected"**
  exports just the checked rows. Both open the same "Export bundle (.zip)"
  form, covered in [import & export](import-and-export.md).

**Media's "Delete selected"** (Media only — Conversations has no delete;
see its own row below) is a second bulk action next to "Export selected".
Pressing it swaps the strip for a confirmation naming the count — "Delete N
selected items? You can undo right away, or restore later from Trash." —
with "Delete" / "Cancel", the same in-place armed-button pattern
as the media viewer's own single-item "Delete" (never a popup modal). The
viewer's confirmation names the one item it will move — "Move
"team-handbook" to Trash? You can undo right away, or restore later from
Trash." — so you can tell which document is about to go even when the
Items list has another row highlighted.
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
Focus moves straight to "Undo" the moment the receipt appears, so pressing
**Enter** undoes immediately — the confirmation's "You can undo right
away" is literally true at that instant. "Undo" stays live even if the
list behind the receipt goes stale in the meantime (a later page change,
say): it restores exactly the ids the receipt already names, not whatever
the list happens to show now, so a stale page can never be the reason
Undo is unavailable. If a restore itself fails, the receipt becomes
"✗ undo failed · n of m · \<reason\>" and "Undo" becomes "Retry undo",
retrying only the items still outstanding; a later full success clears
the receipt as normal. A restore that comes back with nothing usable
(not just one that errors) counts toward that "n of m" and stays in the
retry set, so the receipt total never understates what still needs redoing. "Undo" is the at-point convenience; the durable way
back is the **Trash view** the receipt points at (see "Media Trash"
below), which lists every deleted item — including ones from earlier
sessions — and restores them per item. (Re-importing the same file from
[Import & export](import-and-export.md) also still restores a trashed
match instead of refusing.)

### Media Trash

Press **"Trash"** on the Media toolbar to swap the list for the Trash
view: "‹ Media" (back), a **"Local Trash · N items"** heading ("· 1 item"
for one; "· N matching" while a Trash search is active; the bare "Local
Trash" at widths too narrow for the suffix), and one two-line row per
deleted item — the title, then a dim "type · trashed 2h" line saying when
it was deleted, newest first. Press a row to select it (the **▸** marker
moves), then **"Restore"** to put it back: the row leaves the Trash, the
rail's "Media N" count goes up in place, a "Restored 'Title'." line
confirms it, and the item is back in the media list (and in search
results) with its title, content and analysis untouched. Restore brings the
item back and marks it changed now, so it returns at the top of a Newest
sort — the same is true of "Undo" on a delete receipt. "‹ Media" or
Escape returns to the list.

Notes on the edges: with nothing deleted the view says "Trash is empty.
Items you delete from Media land here." and "Restore" reads "○ Restore"
with a reason tooltip; if the trash holds more items than one fetch page,
a status line says "showing X of N" honestly. The pager under the list is
drawn **only when a second page exists**: a trash that fits on one page keeps
the item range ("1-1 of 1") and drops "Page 1 of 1" along with both
**Previous** / **Next**, the same rule the rest of the Library follows (see
[Library](../library.md)), so Restore sits directly under the last row. Entering Trash clears any
"✓ deleted…" receipt still showing on the list — the Trash view is the
durable path that receipt pointed at. Trashed items are **excluded from
search** (Library search and RAG keyword retrieval both skip them) until
restored.
A selected row offers two actions and no others: **"Restore"** (key `r`),
and **"Delete forever"** (key `x`), which arms an inline "Cancel | Delete
permanently" confirmation and, once confirmed, removes that one item for
good — there is no undo and no receipt afterwards. In that confirmation the
committing **"Delete permanently"** is painted in the theme's readable error
colour (not the ordinary body ink) and sits at least three cells clear of the
focused **"Cancel"**, so the most destructive control is the most marked and
never a mis-click away from the safe one. The two row actions sit two cells
apart, the destructive one set apart in the quiet danger styling the
select-mode "Delete" uses; both keys are advertised in the footer, and
only while the actions they stand for are genuinely pressable. The
confirmation names the item, its type and when it was trashed in the same
words the row does ("trashed 6h"), with the exact timestamp on the
tooltip. There is no empty-trash or bulk permanent delete; each item is
deleted on its own.
Whichever Trash action you take, "‹ Media" returns you to a
current list: the app refreshes the page it fenced for its own write, so it
never asks you to press "Retry" for a change you just made here.
The Reader beside the Trash list keeps whatever live item you had open —
your reading position is not thrown away — and says which list that item
belongs to: **Showing a Media item · not in Trash**, above its title.

### Media list

| Control | What it does |
|---|---|
| Scope line | The quiet line under the "Media (N)" header states the scope the rows in front of you actually came from: the count as **N of M**, the applied filter in quotes, the type ("all types" until you pick one), and the sort — for example `Media · 1 of 11 · filter “notes” · all types · sort: Newest`. The filter box keeps a *draft* until you press Enter, so the box and the list can legitimately disagree; this line always describes the applied scope and never echoes an unsubmitted draft. When the Media total is not known exactly, the count stands alone (`Media · 12 · …`). |
| "Clear" (on the scope line) | Appears only while a filter or a type is applied. It drops the whole scope this line states — filter *and* type — and empties the filter box with it, so no draft is left behind. The toolbar's "Clear filter" still clears only the filter it names. |
| "Title/keyword…" / "Clear filter" | Searches the complete local Media source before paging — titles, item text, and the keywords an item is tagged with, so a tag you filed items under finds them even when it appears in no title. It is separate from Find in item, and "Review these" pins exactly what it returned. Clearing restores the unfiltered selection when it is still available. |
| "type: All types" | Opens one bounded keyboard list containing the complete type set. The row your arrow keys are on carries a leading `█` bar (the same cue the list rows use); ✓ marks the value currently in force, so the row you are on and the row that is active are told apart — on opening, both marks sit on the active row (`█ ✓ All types`). "All types" means no filter; a stored type literally named "All" remains a separate selectable value. Press Escape (or pick the current choice) to cancel. |
| "sort: Newest" | Opens the same kind of bounded keyboard list with all four orders (Newest, Oldest, Title A-Z, Title Z-A) fully visible, the same leading `█` on the row you are on and ✓ on the active one. Escape cancels. |
| "Previous" / "Next" | Moves through exact 20-item pages after the active query, type, and sort are applied. The final page may contain fewer rows; disabled buttons explain why they cannot move. With only one page, the controls do not render at all — just the item range. |
| "Retry" | Repeats the failed load. When a load fails, the reason and this Retry sit together in one bordered callout above the rows ("Couldn't load page 1 · database is locked"; red for a hard failure, amber for a timeout) — that is the only Retry on screen, and it also reloads the type list when that is what failed. The reason is the failure's own words only for an operating-system or database error; anything else is named by the kind of failure it is ("the connection failed", "the database could not be read", or "an unexpected error" when it is none of those), so a private path — and the exception's own text — never reaches the screen. If retained rows may be out of date, rows stay open (a row press is a read, never disabled by staleness) but Select, Export, Delete, sort, and Select all stay disabled with a reason until recovery succeeds. A Retry that fails again shows "Couldn't retry · \<reason\>" so a second failed attempt reads differently from the first, instead of repeating the unchanged staleness copy. When the fault callout itself fails the same way on a consecutive Retry, the message stops repeating one sentence and names the recovery step — "Couldn't load page 1 · database is locked · reopen Chatbook to reconnect to the media database" — so a persistent fault points somewhere rather than looping. If only the type list fails ("Couldn't load media types"), the rows that loaded keep all their actions. |
| "Export…" / "Select" | The shared grammar above; Export… is scoped to the active type filter. |
| "Review these" | Pins the whole filtered list as an ordered review set (see "Review sets" below). Like "Export…" — the other action that acts on the whole filtered list — it renders as "○ Review these" with the failure on its tooltip when the first load failed with nothing behind it, and stays live over rows a later failure retained. |
| "Trash" | Opens the Trash view — every deleted media item, restorable per item (see "Media Trash" above). Hidden while selecting, like "Export…". It is never disabled by a failed Media load — it is the route to your deleted items exactly when the list is unhappy — whereas "Export…" and "Review these" render as "○ Export…" / "○ Review these" with the reason on their tooltip when the list has nothing to select. |
| Row press / Enter | Selects the item and loads it into the permanent Reader; Enter bypasses the short traversal-settle delay. In Select mode, it toggles the row's checkbox instead. |
| "Sets" (title row) | Opens the saved-set picker (see "Review sets" below). It stays put on an empty or filtered-to-zero list — it is navigation, not a result, and it is the way back to a saved set when the list itself has nothing to offer. Hidden only in Select mode, like the other list-level actions. |
| Library / Items grip | Collapses or expands that pane and remembers the manual choice. Responsive collapses caused by terminal width are not saved. |

A filter that narrows to exactly one row loads that row into the Reader for
you, and the list says so: **1 result · Enter opens** under the toolbar.
(Clearing the filter puts your previous selection back.) With several hits
nothing is auto-loaded and the line does not appear.

Empty states: with nothing imported, "No media in your Library yet. Import
something to see it here."; with a type that matches nothing, "No media
of type 'pdf'."; with a filter query that matches nothing, "No media matched
“day2” in titles, content or keywords." beside a live "Clear filter"; with
both — a query that misses inside a type — the sentence names both facets:
"No media of type 'pdf' matched “day2” in titles, content or keywords."

An empty page keeps the toolbar it was filtered with. `type:`, `sort:`,
"Export…", "Trash", "Select" and "Review these" all stay where they were, so
the facet that produced the empty page can be read and reset without leaving
the canvas. "Select" and "Review these" carry the `○` marker and a reason
there — with no rows, neither has anything to act on; "Export…" stays live,
because it exports the type, not the query. One recovery action rides with
the sentence: "Show all types" whenever a type is in force, and "Import
media" only when the source itself is empty — a filter miss never suggests
importing, because clearing the filter is the honest way back. Keyboard focus
lands back in the filter box, which is where you retype.

The pager reports the exact visible range, total, and page. Changing page or
type clears current-page selection with a visible "Selection cleared."
notice. A failed request keeps the last applied page visible instead of
silently replacing it with a partial or broad Library snapshot.

### Media Reader

Reader stays mounted beside Items and keeps one mode visible at a time:
**Read**, **Analysis**, **Highlights**, or **Info**. The chosen mode persists
while you move through items. An open item also survives a trip through the
Library rail: press Notes (or any other rail row) and come back, and the
Reader shows the same item on the same tab at the same place, with its Items
row still marked `loaded` — only **‹ Back** (or Escape, in the narrower
layouts where it leaves the Reader) returns you to the list. Leaving Library
for another destination and coming back reopens that item too, or shows its
Retry state if it was deleted meanwhile. Missing analysis
or highlights produces an item-specific empty state; it does not silently
switch modes. Leaving **Read** for another mode and returning drops you back
at the same place in the text —
the reading position is restored even though the rendered body has to lay out
again first.

Its header is deliberately short: **‹ Back**, the title, the action row, and
the mode row — five rows above the reading surface, border included. A byline
row appears only when the item actually has an author or a URL — an author
stored as the literal "Unknown", which is what most local imports write when
a file names none, counts as no author and paints no byline — and an
identity line ("Server item · not in local Media list") only for a server
item a local Media list cannot show. The mode row is the only label for the
open mode; no section header repeats it. Body text wraps at a reading measure
of about 90 columns however wide the terminal is, while the box around it
still spans the pane.

- **Read** — the complete stored text ("No stored content." when empty).
  Whenever the stored text itself has a real heading, table, or fenced code
  block — whatever the item's media type, because the stored type does not
  tell you what the text contains — a "Rendered (selected) |
  Raw" toggle appears above the box and defaults to **Rendered** — headings,
  tables, and code render properly instead of showing literal `#`/`##`/`|`
  characters, using the same renderer as Notes' own "Preview". Press
  "Raw" to see the plain source instead. A rendered heading starts in the
  same column as the prose beneath it. Any item with content but no
  markdown gets no toggle, whatever its media type; **Info** then carries the
  one-line reason, "No Markdown formatting to render — showing the stored
  text", so the reading surface is not banner-ed with it on every open — it
  names what THIS item has, so a plain-prose transcript is not told that
  rendering is "for transcripts". An item with no stored content shows no such
  note anywhere — the box already says "No stored content." and there is no
  rendered view to explain away. Below the toggle (or directly
  above Content for everything else) is a "Search content…" box — its
  placeholder reads "Search content (raw text)…" whenever the toggle is
  present, since search always matches the raw stored text regardless of
  which view is showing. Typing a query shows "Match 1 of 4" (or
  "No matches") and a "◀ Prev" / "Next ▶" pair that steps through matches
  and wraps at either end — in both views. With no matches both controls
  are disabled and read "○ Prev" / "○ Next", the same marker the Media
  pager uses, so there is nothing live to press. Only the visual highlighting of
  the current match is Raw-only; Rendered shows the same step count with
  no on-screen mark. The search bar stays exactly where Find opened it,
  directly under the Read/Analysis/Highlights/Info row: submitting a query
  only reveals the match count and "◀ Prev" / "Next ▶" beneath the box, and
  clearing the query (submit an empty box) hides them again. Nothing above
  the bar moves. Eligible local PNG, JPEG, and WebP files can also show an
  inline image above the complete text. **Hide preview** / **Show preview**
  affects only that item for this session. An unavailable or failed preview reports the problem and leaves
  every character of stored text readable; GIF, PDF, audio, video, remote URL,
  and server-item previews are not fetched or rendered here.
- **Analysis** — stored analysis text you can view and edit ("Edit
  analysis", or "Add analysis" when empty; "No analysis yet." otherwise).
  The actions sit directly beneath the analysis text, however short it is —
  a two-line analysis puts them on the next line, not at the bottom of the
  pane. A long analysis scrolls the tab, carrying its actions to the end of
  the text rather than clipping them.
  A Markdown analysis renders like the Read tab, with the same
  Rendered/Raw toggle. While a Find query is active the analysis shows its
  stored text instead — only that view can mark the matches — and the strip
  says so ("Showing the stored text so matches can be marked · clear the
  search to read it rendered."); clearing the search hands the rendered view
  straight back.
  Analysis is produced at import time (the "Analyze after import" option),
  written by hand here, or generated in place: **"Generate"** (**"Regenerate"**
  once one exists) calls the configured analysis provider — the `provider`
  and `model` named under `[analysis_defaults]` in `config.toml`, the same
  table bulk **Analyze** and "Analyze after import" use — without leaving
  the reading flow. With no provider configured it reads **"○ Generate"**
  and an always-visible line beneath the button names the reason and the
  next step ("No analysis provider is configured · Set one in Settings ▸
  Providers & Models.") — the same wording the Select-mode bulk **Analyze**
  gate and the Import "Analyze N skipped" gate use — so the gap is readable
  before you click, without hovering. The tooltip still carries it too.
- **Highlights** — saved quotes from this item ("No highlights yet." when
  empty). Expand the collapsed **"Add highlight"** section, fill "Quote"
  (required), optionally "Note (optional)" and "Color (optional)", and
  press "Add highlight". Each saved highlight shows the quote with a
  color swatch, its color/note details, and a "✕ Delete" button.
- **Info** — metadata and provenance: backend-qualified ID, original source,
  stored representation, preview status, the representation **Use in
  Console** will send, and — for an item with content that has no Markdown —
  the "No Markdown formatting to render — showing the stored text" line that
  explains why Read offers no Rendered|Raw toggle. The Items catalogue is local-only. A finished server
  import may open one read-only compatibility detail labelled **Server item ·
  not in local Media list**; it does not become a local Items row. A
  `plaintext` item whose text renders as Markdown lists its type as
  "Type: markdown (stored as plaintext)".
- **Primary toolbar**:

| Button | What it does |
|---|---|
| "Find" | Opens the search bar for the tab you are reading — the transcript on Read, the analysis on Analysis — focused and ready to type; a second press or Escape closes it. Walking with `]`/`[` keeps an active query but never moves your cursor into the field. This never filters Items. `Ctrl+F` opens it from the keyboard. When the tab you are on has nothing to search — Highlights and Info have no text — it reads "○ Find" and prints that tab's own reason on the line directly under the toolbar, not only in a tooltip (task-32362). On Highlights and Info that reason reads "This tab has no text to search · switch to Read or Analysis." |
| "Note" | Opens a new note that names this document: the note is titled after the item and its first line is a `[title](media://…)` link back to it, with the caret on the line below, ready to type. `n` does the same from the keyboard (not while the Find box or the Items filter has focus — there it types an `n`). The Reader keeps your place: pressing the Media rail row brings back the same item, on the same tab, at the same scroll position, and the note stays open on your next visit to Notes. For a server item the button reads "○ Note" with the reason "Notes beside server items need server support". |
| "Use in Console" | Stages this item as context for your next Console message, linking it to the active workspace on the way if it is not a member yet — the same one-press link that notes and conversations do. A receipt under the toolbar says which workspace by name ("Linked to Local Default · staged in Console", or "Already linked to …" when it was a member already) with an **Undo link** beside it that removes the membership a press in this reader session added (a link made by Search ▸ evidence ▸ `u` has no Undo there; remove it from the workspace's own sources). An Undo pressed after the active workspace changed says so instead of removing anything. The receipt stands only while that workspace is the active one; after a switch it is withheld and the next press links into the new workspace. `c` does the same from the keyboard. If the link itself fails, nothing is staged and the message names the workspace and the next step; a raw workspace id is never shown. |
| "Read later" ↔ "Remove later" | Toggles the loaded item's persisted reading-list state. |
| "More" | Keeps secondary actions reachable: Edit metadata, Open original when available, Open manager, and Move to trash. Narrow layouts retain these actions here rather than hiding them. Opening it adds one toolbar row directly beneath this one — the tab row and the reading body shift down a single line, never off the fold. The button reads "More ▴" while the row is open, and focus stays on it so a second press closes the row. |
| "Move to trash" | Two-step, title-specific confirmation: the armed copy reads "Move "team-handbook" to Trash? You can undo right away, or restore later from Trash.", naming the item the Reader shows. The one destructive action in this strip is set apart from the neutral ones — a left margin and the Library's quiet danger ink — so it never ends the row flush and unmarked. Success selects the adjacent item and leaves a bounded Undo receipt; Trash remains the durable recovery path. The rail's "Media N" count drops at once, as it does for "Delete selected". |

Read later, Edit metadata › Save, Move to trash and Use in Console all act on
the item the Reader shows, whichever row the Items list has highlighted. That
matters after opening an item by link — an import receipt's "Open in
Library", a landing "Open", or a Search/RAG result: the list cursor lands on
that item (and on no row at all if it is not on the current page), and the
actions never fall back to the first row.

The primary row holds five actions (Find, Note, Read later, Use in Console,
More). On a Reader too narrow to fit all five side by side — counting
"More ▴" so the row never reflows when More opens — the row is already two
rows (Find · Note · Read later over Use in Console · More), More open or
closed.

### Review sets

A review set pins an ordered snapshot of media items so you can work through
them one by one, with your place and progress saved between visits.

- **Create** — **Review these** on the media list pins the whole filtered
  result (in the current sort order, capped at 500 items with a notice when
  trimmed); in Select mode, **Review selected** pins just the checked items in
  list order and leaves Select mode. Creating a set activates it and opens its
  first item in the Reader. If another set was mid-walk, a notice names it and
  its progress ("Paused 'Read later' at 1 of 2 · 0 reviewed. Resume from
  Sets.") — creating never silently strands a walk. A set made with
  **Review these** is named after what it pinned: `Search: "day2"` for a
  filter, `pdf items` for a type, otherwise **All media**. There is no separate
  "Reviewing N items." notice: the banner the Reader opens with says the
  same thing and more, and in a narrow terminal that notice landed on the
  Reader's own border.
- **Resume on entry** — opening the media area with a set active loads its
  current item into the Reader automatically, on every entry, so the banner
  and the open document always agree. Escape behaves here exactly as it does
  everywhere else in the Reader — see [Keyboard & commands](#keyboard--commands)
  for the one rule and the chip that states it.
- **Walk** — while a set is active the Reader carries a banner naming the
  set, your place, and the open item's own state ("Reviewing: All media — 2
  of 14 · 1 reviewed · ✓ reviewed"), and the footer shows the same place —
  and the footer chip says what the key does: **`]` next (marks reviewed)**.
  `]` advances and marks the item you leave as reviewed;
  `[` goes back without marking; `m` toggles the loaded item's reviewed mark;
  a final `]` on the last item marks it done in place. Clicking a row of the
  set in the Items pane moves both readouts to that item (it does not commit
  a new resume point — `]`/`[` walk from what the Reader is showing anyway). **Escape** steps out
  of the Reader — to the loaded Items row, and to the list itself in
  narrower layouts — and keeps the set active; re-entering resumes at your
  cursor.
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

### Conversations

Conversations use a fixed **20-item page**. The row list scrolls independently
while its pager stays visible underneath it. The pager always states the exact
range, total, and page: with 45 conversations it reads **"1-20 of 45 · Page 1
of 3"**, then **"21-40 of 45 · Page 2 of 3"**, and finally **"41-45 of 45 ·
Page 3 of 3"**. Previous and Next show a visible reason when unavailable, such
as "Already on the first page." or "No more results." A list that fits one page
shows only its range ("1-2 of 2") — the page counter, those reasons and the
Previous/Next controls all return the moment a second page exists, exactly as
on the Media list. The canvas is headed **Conversations (N)**, and each row's
second line gives its message count and age (`5 messages · 16m`).

The reader above names the open conversation by **title**, not by its stored
id ("Loaded Design review notes · 30 of 30 messages · complete."), and each
message is headed with its sender and the same compact age the list uses
("user · 27m"), never a raw timestamp.

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
| Row press | Selects the row and loads it into the **Conversation reader** — not a preview. See below. |
| "Resume conversation" / "Restore and resume" | In the reader header, beside **Read** and **Info** (keyboard: `c`). Reopens the original conversation in Console — it does **not** stage it as source context, and it does not depend on workspace membership. |
| "Use as source" | In the reader header. Stages the open conversation as **source context** in Console. If the conversation is not in the active workspace, the press **links it first**, then continues — see below. |
| "Link to workspace" | Appears in the same header row only while the open conversation is not in the active workspace. Press it to take on the membership **without** handing anything to Console. |
| "Undo link" | Appears under the **"✓ linked · \<workspace\>"** receipt after a link, only while that workspace is active, and removes exactly the membership that press added. |
| "Export…" / "Select" | The shared grammar; export packages conversations into a bundle. |

**The detail pane is a transcript reader.** Pressing a row loads the whole
conversation into a pane headed **"Conversation reader"** with:

- a **Read** / **Info** mode row (the same shape the media Reader uses);
- a status line reading `Loaded <conversation id> · N of M messages ·
  complete.`;
- a **"Find in complete transcript…"** box; and
- the transcript itself, one block per message, each headed
  `user` / `assistant` and the message's timestamp.

**The reader stays in step with Console.** Each time the Conversations list
reads again — when you come back to Library from Console, page, or filter —
the open conversation's saved transcript is checked. If messages were deleted
(or brought back with Undo), edited, or added in Console since the reader
loaded it, the reader reloads it in place, so its message count matches the
list. Read/Info and your Find text are kept.

One rough edge is known and tracked separately: the status line shows the
raw conversation UUID and each message shows a full ISO timestamp rather
than an age string. ("Open in Console" moved into the reader header,
beside Read/Info, in task-32056.)

Empty state: "No conversations yet. Chat in Console and it appears here."
There is no create, rename, or delete here — this panel treats your chats
as source material, not sessions.

**Use as source** stages an excerpt of the selected conversation in the
*current* Console session. It preserves an existing draft; if the draft is
empty, it fills in "Use this conversation as source context for my next
question." It does not send automatically or change the conversation's Auto
or Assistant setting. The excerpt is the first 3,000 characters of its
transcript, each message labelled with its sender, followed by the
conversation's title, ID, message count, workspace, and last update. A longer
conversation is cut at 3,000 characters and its later messages are left out;
nothing on screen says so. When you send, the model receives the excerpt.

**Resume conversation** (keyboard: `c`) instead opens the original saved
conversation and its active branch. **Restore and resume** first recovers
an archived conversation, with confirmation where required.

## Common tasks

### Filter media by type
1. In **Media**, click "type: All types" — a bounded list of every stored
   type appears in place of the toolbar, with a leading `█` on the row your
   arrow keys are on and ✓ on the active one.
2. Click the type you want. The list narrows and the status line reads
   e.g. "2 of 5 · type: pdf". Pick "All types" to clear the filter, or
   press Escape to close the list without changing anything. A stored type
   literally named "All" is distinct from the unfiltered choice.

### Open a media item and search inside it
1. Click a row; it loads into Reader without replacing Items.
2. Press **Find**, type into "Search content…", and press Enter. The status shows
   "Match 1 of N", the matches are highlighted in the content, and the
   count plus "◀ Prev" / "Next ▶" appear directly beneath the box — the bar
   itself does not move, and neither does the header above it.
3. Step through with "◀ Prev" / "Next ▶"; the current match is emphasized and
   the bar keeps the count and both controls in view while you navigate.

### Highlight a passage
1. In Reader, choose **Highlights** and expand "Add highlight".
2. Paste the passage into "Quote"; optionally add "Note (optional)" and a
   color name or hex value in "Color (optional)".
3. Press "Add highlight" — it appears in the list with a ● swatch.

### Stage a conversation as Console context
1. In **Conversations**, open a row and choose **Use as source** in the reader
   header. When the inline notice says it will add the conversation to the
   active workspace, the same press links it and continues. **Link to
   workspace** adds membership without staging; `c` invokes Resume instead.
2. Console opens with the conversation's excerpt staged and your existing
   draft preserved. An empty draft receives the suggested context prompt. Edit the
   draft before sending; staging itself sends nothing. See the excerpt limit
   above before relying on the conversation's contents.
3. **Un-stage** clears the staged evidence without changing the draft or
   workspace link. Return to the same Library reader to use **Undo link** for
   a membership this action added. Undo link removes membership; it does not
   clear evidence already staged in Console.

### Export selected items as a bundle
1. Click "Select", check the rows you want ("Select all N shown" grabs the
   whole visible list), then "Export selected".
2. The "Export bundle (.zip)" form opens — name, destination, and options
   are covered in [import & export](import-and-export.md).

### Recover something you deleted last week
1. In **Media**, click "Trash" on the toolbar.
2. Find the item ("Local Trash · N items" lists everything deleted, newest
   first, each row saying "type · trashed 3d"), press its row, then
   "Restore".
3. "Restored 'Title'." confirms it; "‹ Media" (or Escape) takes you back to
   the list, where the item — and the rail's "Media N" count — are back.

### Delete selected media items
1. In **Media**, click "Select", check the rows you want to remove.
2. Click "Delete selected" — the strip becomes "Delete N selected items? You
   can undo right away, or restore later from Trash." with "Delete" /
   "Cancel". The sentence wraps inside the Items pane at every width,
   including its narrowest.
3. Click "Delete" to confirm (or "Cancel" to back out without deleting
   anything). The rows disappear and the rail's "Media N" count drops
   immediately; the items are trashed, not permanently destroyed.

## Keyboard & commands

The Media type chooser supports **Up/Down**, **Home/End**, and **Enter**;
**Escape** cancels without applying a choice and returns focus to its opener.
The Media grips accept **Enter**; in Select mode **Space** is reserved for
toggling the focused row, so the grips take Enter only there. Arrow-key
traversal moves the Items selection with a short settle delay; **Enter**
loads immediately.
**Escape** closes transient Reader state first — the Find bar, then the More
strip — and then steps out of the Reader to the Items list; the footer chip
always names the next step it will take. The chip is the authority, and it
has five readings: **`esc close`** over anything transient (the Find bar, the
More strip, an open type/sort strip, an open edit form), **`esc cancel`** over
an armed "Move to trash" confirm, **`esc focus Items`** from the open document,
**`esc focus Library`** from the
Items row, and **`esc back`** wherever the next pane out is collapsed rather
than on screen — the Items row at 100x30 and narrower, and the rail row in any
layout the Reader still has a real exit from. In the three-pane layout the rail
row is the last stop, and the footer drops the `esc` chip there rather than
advertise a key that does nothing. No step lands in a text box.

**Unsaved text is never dropped by a key.** While the Analysis form or the
Edit metadata form holds text that differs from what is saved, Escape keeps
the form and says so ("Unsaved analysis — Save or Discard changes first." /
"Unsaved media details — Save or Discard changes first."), and the footer
reads `esc save or discard first` instead of `esc close`. Save keeps the
text; the form's Cancel reads **Discard changes** and is the one way to drop
it while keeping the item (Ctrl+Q's "Discard and quit" and deleting the item
from the Items list -- Select ▸ Delete selected -- drop it too, each after its
own question). Switching Reader tabs keeps what
you typed, and "‹ Back", another Items row (clicked, or reached with
Up/Down), a rail row, Export…, a review set, Use in Console, Move to trash
and the nav bar all refuse the same way until you save or discard; Ctrl+Q
asks before it quits. Every refusal brings the form back on screen, on its
own tab, with Save and Discard changes. The Items filter still filters the
list, but it never moves the Reader off the form: the item you are editing
stays, even when the filter hides its row or matches nothing, and Clear
brings the list back around it. An untouched form still closes on one
Escape or Cancel, and over an armed "Move to trash" confirm the chip reads
`esc cancel` and Escape cancels the confirm first.

Stepping out of the Reader moves focus, not the document: in the three-pane
layout (verified at 235x52) the Items pane is already showing the list, so
the item keeps painting beside it and `]`/`[` keep working from the row.
Where the Library pane is collapsed but the Items pane still shows the list
(verified at 100x30) the "‹ Back" control returns you to the list, and so
does Escape from the Items row. Between 64 and 88 columns both panes are
collapsed: the control and the key still register the exit, but nothing on
screen changes yet — the Reader keeps painting the item it had, and `]`/`[`
stop working until you re-enter Media from the rail. Below 64 columns, with
nothing open in the Reader, the Items pane is the stage instead (see the
layout tour above).
**F6** cycles Library → Items → the Reader's content box, which draws a
heavy border while it holds focus, so the state is visible in a plain-text
capture and not by colour alone (no overlay, so the text stays readable).
From a focused Items row beside the Reader, the Reader's own keys stay
live and are advertised with it: **]** / **[** walk items, **n** takes a
note from the open document, **l** toggles
read-later, **c** sends the item to Console, **t** arms Move to trash, and
**s** enters Select mode — focus lands on the first (or first
still-checked) row, so Down and Space work immediately, and Space toggles
a row while **s** again is Done. On the last item of an active review set
**]** reads "finish review" and marks it done in place. Focus survives a
Reader recompose (loading a new item, a background refresh, an Undo
receipt) and never lands on a pane's collapse/expand grip.
While a review set is active, the Reader adds **]** (next in
set, marking the item you leave), **[** (previous, never marks), **m**
(toggle reviewed), and **R** (exit review, keeping the set resumable); the
footer shows these alongside your "X of M · N reviewed" progress. **Enter** submits the "Filter conversations… (Enter)", **Filter
media**, and "Search content…" boxes. In **Conversations**, **/** focuses the
filter box whenever the list is on screen, and entering Conversations from the
rail puts focus on its first row, so **Up/Down** walks the list straight away.
Global navigation keys live in the [guide index](../index.md).

## Related settings & docs

- Appearance settings remember the preferred Library/Items pane states.
  Automatic Library-rail width follows 3:13 plus five cells, bounded to 29–39
  cells when space allows. When
  explicitly enabled, custom widths remain Library 24–48 and Items 32–72;
  ordinary layouts may temporarily compress the rail to preserve 40 content
  cells, while these adaptive readers may collapse or prioritize panes.
  **Reset layout** restores both panes open, automatic width, a dormant
  36-cell Library preference, and 50-cell Items preferences. Below 64 columns,
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
- **A conversation outside the active workspace links itself on use.**
  Workspace membership decides which items a Console turn may read, so a
  conversation that is not in the active workspace cannot simply be handed
  over in silence — but it is not refused either. **"Use as source" stays
  pressable**, and one sentence beneath it says what the press will do:
  "This conversation is not in this workspace. Pressing this adds it to the
  active workspace first, and you can undo that." The press links it,
  continues to Console, and leaves a receipt — **"✓ linked ·
  \<workspace\> · this conversation can now be used in Console"** — with
  **"Undo link"** beside it, which removes exactly that membership. Switching
  workspaces hides the receipt and Undo. Returning to the named workspace
  while the same conversation stays loaded restores them, provided the
  conversation can still be used there. An old Undo press cannot remove a
  link after the workspace changes or the list becomes stale. The separate
  **"Link to workspace"** button is still there for taking on the
  membership without a hand-off. (The action name is painted once, on the
  button; the line under it is the explanation, not a second control.)
  A block that a link cannot resolve still refuses: the button dims to
  "○ Use as source", states its own remedy (with no active workspace:
  "Select an active workspace before using this item in Console.") and
  offers no link, never naming a button that is not on screen. Pressing `c`
  while the load fence blocks Resume says that same sentence as a message
  rather than doing nothing. The rail's "Use in Console" keeps its own
  pressable-with-reason grammar: it acts on a SET whose members can be
  blocked for different reasons, so no single link would unblock it.
- **The reader takes the width once a conversation is open.** With nothing
  loaded the list absorbs the empty reader's columns (the density rule); as
  soon as a row is open, the split is restored and the reader gets the
  majority of a wide terminal — measured at 235x52: reader 132 columns,
  list 49. At 100x30 the rail steps aside and the two panes share the stage
  (reader 44, list 45); at 60x24 the reader is the single stage.
- **Media links on use, like notes and conversations.** The Media
  reader's "Use in Console" (and `c`) used to refuse every item that was
  not already a member of the active workspace, naming a raw workspace id
  and a "copy or link" action that does not exist for media, while Search
  ▸ evidence ▸ `u` staged the same item with no workspace rule at all.
  Both now take the same path: the item is linked to the active workspace
  and staged in one step, and the receipt names the workspace by its display
  name. Only a link the reader's own press made offers "Undo link" there; a
  link made from Search evidence has no Undo in the reader (remove it from
  the workspace's sources). An item that is never used stays unlinked, so
  the Default workspace still means what it meant.
- **Staging sends an excerpt, not the whole item.** "Use in Console"
  (media) sends the first 500 characters of its stored text, followed by
  its title, ID, and type, author, and keyword lines. Reader ▸ Info's "Use in
  Console sends:" line tells you which you will get: "Complete stored text
  excerpt", "Stored text excerpt (truncated)", or "Metadata only (no stored
  text)". "Use as source" (conversations) sends the first 3,000 characters of
  its transcript, each message labelled with its sender; see
  [Conversations](#conversations).
- **No conversation delete here** — by design, this panel never modifies
  chats. Manage sessions from Console itself.
- **"Media item is unavailable."** — the item was removed while you had it
  open (for example from the Media screen); the viewer drops back to the
  list.
