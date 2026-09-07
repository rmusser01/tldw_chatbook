# Library Collections — named records for content you plan to read and review

## What this screen is for

Collections are named containers for saved content. The panel's own status
line says it plainly: "Collections hold saved items for review — adding
items is coming; you can create and name collections now." Today, though,
this surface is early: you can create, rename, and delete Collection
records, but you cannot yet put items into them or read items from them
("Available now: create, rename, delete records" is the only action-status
line shown). Reach for this panel now only to stake out named Collections
ahead of those features.

## Getting there

Press **Ctrl+3** (or click **⌃3 Library** in the nav bar, or **Ctrl+P** →
"Library"), then in the left rail's **Browse** section click
**Collections**. The rail row shows the current count.

## Layout tour

![Collections](../images/library/collections.svg)

The canvas is titled **"Collections (N)"** (task-2859: dropped the
redundant "Library " prefix and matches the sibling "Name (n)" pattern
Media/Notes/Prompts/Skills already use). Top to bottom:

- **Delete receipt** (after a confirmed deletion): a persistent
  `✓ deleted · Collection · name` toolbar with **Undo** and **Dismiss**.
  It remains visible even when the deleted Collection was the last row.

- **Empty state** (before your first Collection) — "No Collections yet.",
  "Create a local Collection record to start reviewing saved content.",
  "No stored collection items are available locally yet. Collections are
  for reading, reviewing, and reusing saved content." (shown once), and
  "No Collection selected."
- **Collections** — the list pane. Collections are shown 20 at a time in
  creation-time order, with name and stable identity breaking ties. Each
  row reads "name - N items" (item counts are always 0 today); hovering a
  row shows its sync status as a tooltip. The exact range and page appear
  below the independently scrolling rows, followed by **Previous** and
  **Next**. When a server sync profile is present, a short read-only status
  banner appears above the list.
- **Stored collection content** — the detail pane for the selected row:
  "Selected: name", the description (or "No description."), the plain
  status line quoted above, "Action status" / "Available now: create,
  rename, delete records", and a collapsed-by-default **Details**
  disclosure (click to expand) holding the item count, the sync status
  label, its detail sentence (when there is one), and the "Updated … UTC"
  line.
- **Create / Rename** — the form: inputs "Collection name" and "Optional
  description", then the action buttons. While the typed name can't yet
  create a Collection, a single guidance sentence appears above the name
  field explaining why (e.g. "Enter a Collection name." or "A Collection
  with this name already exists."); it disappears once the name is valid.

## Features & controls

| Control | What it does |
|---|---|
| "Collection name" | Name for a new Collection, or the new name when renaming. Required; 120 characters max. |
| "Optional description" | Free-text description shown in the detail pane. |
| Create Collection | Adds a Collection record. Enabled once a valid, unused name is typed. |
| Rename Collection | Renames the selected Collection to the typed name. |
| Delete Collection | First press of the two-press delete: it arms deletion and reveals "Confirm delete". |
| "Confirm delete" | Second press: deletes the selected Collection. Its items stay in the Library, and the tooltip promises the Undo that appears in this panel. |
| Undo | Restores the deleted Collection and its existing membership, then selects it again. |
| Dismiss | Removes the recovery receipt without restoring the Collection. |
| Collection rows | Click to select; the detail pane fills in. Row tooltip shows the sync status label. |
| Previous / Next | Loads the adjacent 20-item page. The control that remains available keeps focus when an edge page disables the control you used. |
| Retry | Reloads the requested page, or repeats stable-ID placement after a follow-up read failed. |

Disabled buttons always carry their reason as a tooltip — for example
"Enter a Collection name.", "A Collection with this name already exists.",
"Select a Collection before renaming it.", or "Select a Collection before
deleting it."

**Sync labels** (shown on row tooltips and inside the selected Collection's
**Details** disclosure) are strictly read-only — every detail sentence ends by promising that no
writes will be queued:

| Label | Detail line |
|---|---|
| "Sync: local-only" | "This Collection is local-only. No sync writes will be queued." (the usual state; shown without a detail line in the pane) |
| "Sync: sync-unavailable" | "Sync dry-run is unavailable for this Collection. No writes will be queued." |
| "Sync dry-run: ready" | "Read-only mirror check: N mapped records. No writes will be queued." |
| "Sync dry-run: conflicts" | "Read-only mirror check: N conflicts need review. No writes will be queued." |
| "Sync dry-run: orphaned mappings" | "Read-only mirror check: orphaned local or remote mappings need review. No writes will be queued." |
| "Sync dry-run: unsupported" | "Read-only mirror check unavailable: (reasons). No writes will be queued." |

## Common tasks

1. **Create a Collection** — Open Collections from the rail. Type a name
   into "Collection name" (and a description if you want one); the
   Create Collection button enables. Press it — the app opens the page
   that owns the new Collection and selects its "name - 0 items" row.
2. **Rename a Collection** — Click its row in the Collections list, type
   the new name into "Collection name", then press Rename Collection. If
   its ordered position changes, the owning page opens automatically.
3. **Delete a Collection** — Click its row, press Delete Collection, then
   press the "Confirm delete" button that appears beside it. Deleting is
   deliberately two presses; nothing is removed on the first press. After
   deletion, choose **Undo** to restore the Collection and its membership,
   or **Dismiss** to leave it deleted and remove the receipt. Undo opens
   and selects the restored Collection's current owning page.
4. **Move through a long list** — Use **Previous** and **Next** beneath the
   rows. The header reports an exact range such as "21-40 of 45" and an
   exact page such as "Page 2 of 3". Returning to Collections restores the
   last successfully applied page; an unfinished or failed page request is
   never saved as your position.

## Keyboard & commands

None — this panel has no screen-specific keys or slash commands. Global
keys live in the [guide index](../index.md).

## Related settings & docs

- This panel owns no config.toml keys; Collections are stored locally per
  profile.
- [Library overview](../library.md) — the rail, the other Browse panels,
  and the "Server sync WIP · local only" runtime note.
- [Guide index](../index.md) — global keys and navigation.

## Quirks & troubleshooting

- **Item-level features are not wired yet.** No item reader, no Search/RAG
  over Collections, no Study or Console handoff, no server sync, and no
  "Add to collection" affordance anywhere else in the app — exactly as the
  panel's status line says ("adding items is coming"). Item counts stay
  at 0 until that lands.
- **Names are capped at 120 characters** ("Collection names must be 120
  characters or fewer."); descriptions are capped at 500. Duplicate names
  are refused ("A Collection with this name already exists.").
- **A greyed-out button explains itself** — it reads with a leading **○**
  (the Library's disabled marker, so the state never depends on colour
  alone), and hovering it gives the exact requirement that is not yet met
  in its tooltip.
- **Sync is display-only.** Every sync label describes a read-only check;
  no state on this panel ever queues a server write.
- **A stale page is readable but inert.** If a Collection write commits but
  the follow-up page read fails, the known result remains visible, the
  exact total/range is hidden, and row, mutation, Previous, and Next actions
  are disabled. Press **Retry** to recover a current page. A successful
  create, rename, delete, or restore is not reported as failed merely
  because that follow-up read failed.
- If the source shrinks past the current page, Collections probes that page
  and clamps to the new final page once. If it changes again during that
  recovery, the last known rows remain visible with Retry instead of
  walking through more pages or inventing a total.
- If the Collections storage layer fails to load, actions report
  "Couldn't load Collections. Check the local Library and retry."; a
  failed delete reports "Failed to delete Collection."

—
*Verified against dev @ e3d0d2c9d — 2026-08-06 (TASK-2855: plain-language
status line replaces the spec/roadmap block, sync-safety/internal detail
moved behind a collapsed-by-default Details disclosure, empty-state
message deduplicated, three enable-Create sentences collapsed into one)*
*Verified against dev @ 642567627 — 2026-08-10 (task-4023 AC#1, RC-07:
the three form buttons measured 2.30:1 while disabled — legible now
(5.91:1 measured live via ANSI decode), with the "○" disabled marker;
typing a valid name flips Create back in place without the marker).*
*Verified against fix/settings-appearance-crash @ 57ad075de — 2026-08-10
(task-4023 AC#5/#7: the empty state is two lines — "No Collections yet."
plus one create-one-below sentence; the selected Collection row carries
the shared leading "▸ " marker; Escape on the Collections canvas moves
focus to the rail search box, matching every other list canvas, and the
footer advertises "esc focus rail".)*
*Verified against feat/library-queue-batch @ a899cbf6a — 2026-08-11
(task-14901 / ADR-055: the "Confirm delete" tooltip now states the
consequence — member items survive, the Collection itself cannot be
restored from Library.)*
*Verified against codex/collection-delete-undo-receipt — 2026-08-12
(TASK-15102 / ADR-055: deleting a Collection now leaves a named receipt
with Undo and Dismiss actions; Undo restores the Collection and its
membership, while member items always remain in the Library.)*
*Verified against codex/task-18916-collections-pagination — 2026-08-28
(TASK-18916 / ADR-067: exact 20-item pages, deterministic mutation
placement, one-clamp shrink recovery, applied-page restoration, and stale
Retry posture.)*
