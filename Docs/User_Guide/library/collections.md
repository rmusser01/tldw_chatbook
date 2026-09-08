# Library Collections — a reading list of saved web captures

## What this screen is for

Collections is a **reading list**. You save a web page to it with **Quick
Capture**, the app extracts the readable article text, and you come back to
read it, highlight it, note it, and mark it Read — without leaving the
terminal and without keeping the browser tab open.

It is **not** a folder manager. There is no "create a collection, then put
things in it" here: nothing else in the Library has an "Add to collection"
action, and this page does not create named containers. If you are looking
for grouping, the surfaces that actually group content today are Notes
folders ([Notes](notes.md)) and Prompt collections
([Prompts](prompts.md)).

> **This page was rewritten.** Until 2026-09-08 it described a
> create/rename/delete Collections manager. That surface no longer ships:
> the generic Collections tables are read-only recovery data (see
> [Legacy Collections data](#legacy-collections-data) below), and the row
> opens the captures reading list documented here. Whether the row keeps
> the name "Collections" or becomes "Captures" is an open product
> decision (task-32057 AC#1) — the name may change; the surface described
> here is what ships.

## Getting there

Press **Ctrl+3** (or click **⌃3 Library** in the nav bar, or **Ctrl+P** →
"Library"), then in the left rail's **Browse** section click
**Collections**.

The rail row reads **Collections (N)** — N is the number of captures in the
currently selected scope, and it is filled in before you ever open the row,
from the same enumerator the list itself uses (task-32057). At narrow rail
widths the row abbreviates to **Captures (N)**.

Selecting the row mounts six scope sub-rows underneath it — **All
Captures**, **Saved**, **Reading**, **Read**, **Archived**,
**Favorites** — plus one row per saved search. Those sub-rows are part of
the Collections destination; selecting the row does not change any other
rail section's open/collapsed state, and nothing about visiting it is
written to `[library.rail_state]`.

## Layout tour

Three panes, the standard Library reader topology: the rail on the left,
the **capture list** in the middle, and a permanent **reader** on the
right.

**Capture list (middle):**

- **Quick Capture** — opens the save form (see
  [Common tasks](#common-tasks)). Disabled with its reason in the tooltip
  when the active authority cannot capture.
- **Filters** — a disclosure holding **Domain**, **Tags, comma separated**,
  **From date (YYYY-MM-DD)**, **To date (YYYY-MM-DD)**, then **Apply
  filters** and **Clear**.
- **Sort: saved desc** — one button that cycles the sort: saved desc,
  saved asc, updated desc, updated asc, title asc, title desc, relevance.
- **Filter captures** — free-text search over the current scope. Press
  Enter to apply.
- **Capture rows**, two lines each: `▸ <title>` on the first, then
  `<domain> · <date> · <Status>` on the second, with `Favorite` and
  `Extraction failed` / `Extraction interrupted` appended when they apply.
  The row you have loaded in the reader is prefixed `Loaded in Reader`; a
  row you just clicked whose detail is still arriving reads `Selected ·
  loading`.
- **Range line** — `1–20 of 57`, or `0–0 of 0` when the scope is empty, or
  `Page N · total unavailable` when a refresh failed.
- **Previous** / **Next** — 20 captures per page. Each carries its reason
  as a tooltip when it is not pressable ("No current next page is
  available.").

**Reader (right):** empty until you select a capture ("Select a capture to
read it here."), then:

- **`<Local|Server> Collections · <domain>`**, the capture title, and a
  byline: author or publication date, estimated `N min read`, the status,
  and the authority.
- **Mark Read · Favorite · Move to Archive**, then **Open Original ·
  More**.
- **Mode row** — **Read · Highlights · Notes · Info**; the active mode is
  prefixed `✓`.
- The mode body. **Read** shows the extracted text (or "No readable
  content is stored for this capture."); **Highlights** is a quote box, an
  optional note, **Add highlight**, and the existing highlights each marked
  `Active` or `Detached · reattach needed` with **Delete highlight**;
  **Notes** is a free-text **Capture note** with **Save capture note**,
  then **Linked Notes** with **Unlink** per link and a **Note ID** box with
  **Link Note**; **Info** lists canonical URL, submitted URL, tags, status,
  extraction state, word count and authority, plus the backing Media item
  and its availability when there is one.
- **More** reveals the lower-frequency actions: **Summarize**, **Listen**,
  **Save Offline Copy**, **Retry Extraction**, **Delete Permanently…**.

## Features & controls

| Control | What it does |
|---|---|
| Quick Capture | Opens the save form: a URL box, optional Title, optional comma-separated Tags, and a note box, then **Save capture** / **Cancel**. |
| Filters / Apply filters / Clear | Narrow the scope by domain, tags, and a saved-date range. Applying always returns to page 1. |
| Sort: … | Cycles the sort order in place; the label always names the order in force. |
| Filter captures | Free-text search inside the current scope. |
| Scope sub-rows | All Captures, Saved, Reading, Read, Archived, Favorites, then your saved searches. The selected scope carries the count. |
| Previous / Next | Move by exact 20-capture pages. |
| Mark Read / Favorite / Move to Archive | Status actions on the loaded capture. Archiving leaves a `Moved to Archive · was <status>.` receipt with **Undo**. |
| Open Original | Opens the capture's original URL in your browser. |
| Read / Highlights / Notes / Info | Reader modes over the one loaded capture. |
| Summarize / Listen | Produce a summary or an audio rendering, when the active authority supports them. |
| Save Offline Copy | Stores a managed local copy of the capture. |
| Retry Extraction | Re-runs article extraction after a failed or interrupted attempt. |
| Delete Permanently… | Two-step: it reveals a sentence naming the capture, its highlights and its offline copy, then **Delete permanently** / **Cancel**. This cannot be undone. |

Every disabled action carries a text reason, never colour alone: a leading
**○** marker plus a tooltip — either the capability's own reason from the
service, "Availability has not been checked.", or "Wait until the selected
capture is loaded and current."

### Legacy Collections data

If your profile still holds records from the superseded generic
Collections tables, the reader shows:

> Legacy Collections are read-only on this profile · Use the legacy
> Collections inspector or JSON recovery export.

followed by a **Legacy Collections data… (N)** button. Opening it shows a
bounded, read-only inspector of those records with **Export complete
JSON…** and **Close inspector**. Those records cannot be created, renamed,
deleted, restored, or added to — the local Collections service refuses
every one of those writes — so read and export are the only two things you
can do with them (task-32057).

Profiles with no legacy records never see any of this.

## Common tasks

1. **Save a page to read later** — Open Collections, press **Quick
   Capture**, paste the URL (a title, tags and a note are optional), then
   press **Save capture**. The new capture is selected and the app starts
   extracting its readable text in the background.
2. **Read something you saved** — Click a row. The reader loads it in
   **Read** mode. Use **Mark Read** when you are done, or **Move to
   Archive** to file it away (with **Undo** available on the receipt).
3. **Keep a quote** — With a capture loaded, switch to **Highlights**,
   paste or type the quote, add an optional note, and press **Add
   highlight**.
4. **Connect a capture to your Notes** — Switch to **Notes**, paste the
   Note's exact ID into **Note ID**, and press **Link Note**. Links show
   the Note's availability, and **Unlink** removes one.
5. **Find one capture among many** — Pick a scope sub-row (or **All
   Captures**), then either type into **Filter captures** or open
   **Filters** for domain / tags / date range. **Previous** and **Next**
   page through the result, 20 at a time.
6. **Recover legacy Collections records** — See
   [Legacy Collections data](#legacy-collections-data): open the inspector
   and use **Export complete JSON…**.

## Keyboard & commands

This canvas has no screen-specific keys. **/** focuses the rail search box
and **Escape** returns focus to the rail, as on every Library canvas;
**F6** cycles the panes. Global keys live in the
[guide index](../index.md).

## Related settings & docs

- Captures are stored per profile in the Library Collections database
  (`library_collections_db_path`). This canvas owns no other config.toml
  keys.
- [Library overview](../library.md) — the rail, the other Browse
  destinations, and the runtime source note.
- [Guide index](../index.md) — global keys and navigation.

## Quirks & troubleshooting

- **A stale page is readable but inert.** If a refresh fails, the last good
  page stays on screen under "Showing the last good page. Refresh failed;
  totals and page actions are paused.", the exact total is withheld, and
  the paging controls are disabled. Press **Retry**.
- **A failed load names its reason.** "Captures could not be loaded:
  \<reason\>." with **Retry**; the reader's own equivalent is "Capture
  could not be loaded: \<reason\>." with its own **Retry**.
- **Switching captures keeps the old one readable.** While a newly selected
  capture's detail is arriving, the reader says `Loading "<new>"… showing
  "<old>" until ready.` rather than blanking.
- **A save whose outcome is unknown does not auto-retry.** The form keeps
  your draft, says "Save outcome unknown. Refresh before retrying.", and
  offers **Refresh capture list**; a deliberate retry against a Server
  first warns that it may reapply Saved status and clear Favorite on an
  existing canonical URL.
- **The Chunking Lab strip** above the canvas ("Chunking Lab | Try selected
  text") is not part of Collections — it is a Library-wide developer tool
  that paints on every canvas. See [Library overview](../library.md).

—
*Verified against fix/library-crit8-docs — 2026-09-08 (task-32057 /
task-32073: whole-page rewrite. The previous page described a
create/rename/delete Collections manager that no longer ships; the row
opens the Quick Capture reading list documented here. Live at 235x52 and
100x30 on a seeded profile: the rail read `Collections (0)` before the row
was ever visited, selecting it mounted the six scope sub-rows and left the
Create section open with no `[library.rail_state] sections` write, and the
canvas painted `Quick Capture` / `Filters` / `Sort: saved desc` /
`Filter captures` / "No captures match this scope. Clear filters or save a
URL with Quick Capture." / `0–0 of 0`. The `legacy_read_only` reason and
its recovery path are now on the canvas. AC#1 — what the row should BE —
is a product decision and is deliberately left open.)*
