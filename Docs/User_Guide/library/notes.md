# Library notes — write, sync, and reuse notes stored in your Library

## What this screen is for

The Notes canvas is where you create and edit the notes stored in your
Library: quick captures, meeting notes, research summaries — anything you
want to keep, search, and later hand to the Console as context. Notes
autosave as you type, can be started from templates, imported from files,
exported as Markdown or text, and mirrored to a folder on disk with the
built-in sync panel. For the rail, landing canvas, and the other Library
sources, start with the [Library overview](../library.md).

## Getting there

Press **Ctrl+3** to open Library, then click **Notes** in the rail's
"Browse" section (the row shows a live count). **Ctrl+P** →
"Tab Navigation: Switch to Library" works too. To jump straight into writing, click **New note**
under the rail's "Create" section.

## Layout tour

Wide Database Notes keeps Library navigation beside the list while you scan:

```text
+----------------------+-----------------------------------------------+
| Library              | Database | Files                              |
| Browse               | Notes (N)                                     |
|   Notes              | Filter...   Sort   Sync   Import   Export     |
|   Media              |                                               |
|   Conversations      |  Note title                           age     |
| ...                  |  Note title                           age     |
+----------------------+-----------------------------------------------+
```

Opening a database note—or switching to Files—turns the workbench into one
focused task. The Library rail stays mounted but yields its width, and one
stable cue names the return destination:

```text
+-----------------------------------------------------------------------+
| ‹ Library / Notes                                                     |
|                                                                       |
|  Focused note editor or retained Files workspace                      |
|                                                                       |
+-----------------------------------------------------------------------+
```

Below 120 columns, Notes keeps the existing navigation-first, one-stage
layout. Choose Notes in the rail, then work in the full canvas; the compact
editor's own Back control returns to its list.

```text
+----------------------------+     +----------------------------+
| Library rail               | --> | Notes list or editor       |
|   Notes                    |     | ‹ Back to list (editor)    |
+----------------------------+     +----------------------------+
```

- **Source strip** — a "Database | Files" toggle above the canvas. This
  page covers the Database side; see below for Files.
- **Notes list** — the default view: a "Notes (N)" header, the
  "Filter notes… (Enter)" field, a toolbar (sort / Sync / Import note /
  Export… / Select), and one row per note showing its title and age.
- **Editor** — opens when you click a note: title, body, keywords, a meta
  line with the autosave status, and an action row. On wide terminals the
  top `‹ Library / Notes` cue returns to the exact prior list row, scope, and
  scroll positions; on compact terminals use `‹ Back to list`.
- **New note view** — opens from the rail's "New note": a "Blank note"
  button plus a "From a template" list.
- **Notes sync panel** — opens from the toolbar's "Sync" button: folder,
  direction, conflict policy, auto-sync, and an activity log.

### Database vs. Files vs. Sync

Three different notes worlds meet here, and each surface now says so in
place: the strip above the canvas switches between two of them.
**Database** (this page) keeps notes inside the Library itself — its own
placement line points to Files or Sync for notes that live in a folder on
disk. **Files** swaps the whole canvas for the File Notes workspace, which
edits plain files under a folder you choose directly and has its own
Session Git panel — see [File notes](file-notes.md). **Sync** (opened from
this page's toolbar, below) is the third: it mirrors a folder's notes INTO
the Library's database, unlike Files mode, which edits that folder
directly without mirroring it in.

## Features & controls

### Notes list

| Control | What it does |
|---|---|
| "Filter notes… (Enter)" | Type and press Enter to filter; the status line then reads "filter: \<text\> · N results". |
| "Sort: Newest" | Opens a one-row strip of Newest / Oldest / Title (✓ on the active one) in place of the action row; pick one directly, or press Escape to cancel. |
| "Sync" | Opens the Notes sync panel (below). |
| "Import note" | Opens a file picker, "Import Note (TXT, MD, JSON, YAML)". The imported file becomes a new note. |
| "Export…" | Opens the "Export bundle (.zip)" canvas scoped to notes — bundle notes into a .zip. |
| "Select" / "Done" | Toggles select mode: rows grow ☑/☐ checkboxes, and a row appears with "N selected", "Select all N shown", "Clear", and "Export selected". "Export…" hides while selecting. |

With no notes at all, the list reads "No notes yet. Create one to see it
here."

### Editor

| Control | What it does |
|---|---|
| "‹ Back to list" | Returns to the list (your text is already saved — see autosave below). |
| Title, body, keywords | The note's fields; keywords are comma-separated. |
| Meta line | Created/Modified/version plus the autosave status: "N words · saved", "saving…", "changed elsewhere", or "save failed". |
| "Save" | Saves immediately, without waiting for autosave. |
| "Preview" / "Edit" | Toggles the body between editing and rendered Markdown. |
| "Use in Console" | Hands the note to the Console as staged context, with the suggested prompt "Use this note as context and help me work with it." |
| "Export .md" / "Export .txt" | Saves the note to a file you pick; success shows "Note exported successfully to \<name\>". |
| "Copy" | Copies the note to the clipboard as Markdown — "Note copied to clipboard as markdown!" |
| "Delete" | Asks inline first: "Delete this note? Undo will be available in the Notes list." Confirm with "Delete" or back out with "Cancel". A successful delete returns to the list with a named "✓ deleted · …" receipt offering **Undo** and **Dismiss**. |

**Autosave** runs about two seconds after you stop typing; the meta line
flips to "saving…" and back to "saved". If the same note was changed
somewhere else while you were editing, a banner appears: "This note
changed elsewhere — Overwrite saves your text; Reload discards it." —
pick **Overwrite** or **Reload**.

The wide `‹ Library / Notes` cue and Escape use the same guarded return as
the compact Back control. A dirty save, sync, conflict, reload confirmation,
or running mutation can therefore keep the focused task open until it is safe
to leave. A successful return restores the Database/Files source, filter,
sort, selected note or placement, Notes-list scroll, Library-rail scroll, and
semantic keyboard focus instead of starting over at the first row.

After a confirmed delete, the receipt stays in the Notes list until you
choose **Undo**, choose **Dismiss**, or complete a newer note deletion.
**Undo** restores that exact database note and immediately returns its row
and the Notes rail count. **Dismiss** removes only the receipt; the note
remains deleted. Notes do not currently expose a separate Trash browser, so
the receipt is the in-Library recovery action.

### New note view

"Blank note" drops you straight into the editor with an empty title (shown
as an "Untitled" placeholder — just start typing) and an empty body. If you
leave again via "‹ Back to list" without typing anything, the blank note is
quietly discarded rather than left behind as a stray "Untitled" row; typing
anything, or pressing "Save", keeps it. That includes naming it "Untitled"
yourself: once you have touched the title field the note is yours, and it
is kept even with an empty body. The "From a template" list
pre-fills title, body, and keywords instead; each row shows the template
name with the title the note will get. Available templates: Brainstorming
session, Bug report, Code review, Daily journal entry, Meeting notes,
Project planning, Research notes, Todo list.

### Notes sync panel

Mirrors notes between a folder on disk and the Library ("Mirror notes
between a folder on disk and the Library.").

| Control | What it does |
|---|---|
| "‹ Back to notes" | Returns to the list. |
| folder + "Browse…" | The folder to mirror; Browse opens "Select Notes Sync Folder". |
| "Direction" choices | An always-visible choice row — "Bidirectional", "Disk → Library", "Library → Disk" — with ✓ on the active one; click to pick. |
| "Conflicts" choices | The same choice-row shape for the conflict policy: "Newer wins", "Disk wins", "Library wins". |
| "auto-sync: every 5m ✓/○" | Toggles a background sync every five minutes. |
| "Sync now" | Runs a sync immediately; the button reads "Syncing…" while it runs. |

The status line below reports "idle", "syncing · 3/12",
"done · no changes", "done · N changes · M conflicts", or
"failed · \<reason\>", and an activity log keeps the last 20 entries,
most recent first.

#### What a conflict policy does to the copy that loses

A conflict is when the same note changed **both** in the Library and in
the file on disk since the last sync. One copy has to give way, and the
one that does is always saved first:

| Policy | Which copy is kept as the note/file | What happens to the other one |
|---|---|---|
| **Newer wins** | Whichever was edited more recently | Saved beside the file, then replaced |
| **Disk wins** | The file on disk | The Library's version is saved beside the file, then replaced |
| **Library wins** | The note in the Library | The file's version is saved beside the file, then replaced |

The saved copy is a plain file next to the original, named
`your-note.md.conflict-20260821T203015Z-disk.bak` (`-disk` for the file's
version, `-db` for the Library's). It holds the replaced text exactly, so
recovering it is a rename — nothing is added to it. The sync never picks
these files up again, so they will not turn into extra notes.

If that copy cannot be written for any reason, **the sync does not
overwrite anything**: both versions are left exactly as they are and the
run reports an error instead.

The activity log tells you which happened: "1 conflict resolved (Disk
wins)" and "Replaced copy saved as …", or "1 conflict left unresolved —
both copies kept as they are" when the run did not change either side.

## Common tasks

### Create a note from a template
1. In the rail, click **New note** under "Create".
2. Under "From a template", click **Meeting notes** (or any other row).
3. The editor opens pre-filled; just start typing — autosave handles the
   rest.

### Import a Markdown file as a note
1. In the notes list, click **Import note**.
2. Pick the file in the "Import Note (TXT, MD, JSON, YAML)" dialog.
3. The new note appears at the top of the list; open it to edit.

### Set up folder sync
1. In the notes list, click **Sync**.
2. Enter a folder (or click **Browse…**), then pick a Direction and a
   Conflicts policy — each is a choice row with ✓ on the active value;
   click the one you want.
3. Click **Sync now** and watch the status line; optionally turn on
   "auto-sync: every 5m" to keep it running every five minutes.

### Use a note in Console
1. Open the note and click **Use in Console**.
2. You land in the Console with the note staged as context and the
   prompt "Use this note as context and help me work with it." ready to
   send or rewrite.

### Export a note as Markdown
1. Open the note and click **Export .md**.
2. Choose a destination in the "Export Note as Markdown" dialog — the
   toast confirms "Note exported successfully to \<name\>".

### Undo a deleted note

1. Confirm **Delete** in the note editor.
2. In the Notes list, find the "✓ deleted · \<title\>" receipt.
3. Click **Undo** to restore the note, or **Dismiss** to leave it deleted.

## Keyboard & commands

| Key | Action |
|---|---|
| Enter (in "Filter notes… (Enter)") | Apply the filter |

That is the only screen-specific key — everything else here is
click-driven. Global navigation keys live in the [guide index](../index.md).

## Related settings & docs

- `[notes]` in config.toml — the sync panel reads and writes these:
  `sync_direction` (default `bidirectional`), `sync_conflict_resolution`
  (default `newer_wins`), `auto_sync` (default `false`), and
  `sync_directory` (default `~/Documents/Notes`).
- [Notes bidirectional sync](../../Features/notes_bidirectional_sync.md) —
  deep dive on the sync engine behind the panel.
- [File notes](file-notes.md) — the "Files" side of the source strip.
- [Library overview](../library.md) — the rail, landing canvas, and the
  other Library sources.

## Quirks & troubleshooting

- **Import failures are deliberately unspecific** — whatever goes wrong
  (unreadable file, unsupported content, a file over the ~8 MB guard),
  the message is always "Could not import that file." Check the file's
  type and size if it keeps failing.
- **Sync never asks about conflicts** — there is no "ask me" policy by
  design; conflicts are always resolved by the "conflicts" setting, and
  the count is reported in the status line afterwards. The copy that
  loses is never simply discarded — see
  ["What a conflict policy does to the copy that loses"](#what-a-conflict-policy-does-to-the-copy-that-loses).
- **`.conflict-…bak` files appear in your sync folder** — those are the
  replaced copies from a conflict, kept on purpose. Delete them once you
  are happy with the merge; rename one back over the original to restore
  what it holds. Sync ignores them.
- **Notes rows have no ▸ marker** — unlike media rows, note rows show
  only the title and age; they still open on click.
- **Notes cap at 2,000,000 characters** — longer content is rejected
  rather than truncated.
- **"Use in Console" now actually delivers the note.** It used to stage
  the note so it displayed as attached while sending nothing to the
  model — that's fixed: your next send now carries a real excerpt of the
  note body, not just its title.

—
*Verified against c2cbb8081 — 2026-08-04 (PR-T1: "Use in Console"
delivering the note's real content on send is covered by capture
round-trip tests, task-2374).*
*Verified against dev @ 6b38a13b8 — 2026-08-07 (task-2858 Task 3, LIB-14:
"Blank note" no longer leaves a stray "Untitled" row if abandoned
untouched, and the title shows an "Untitled" placeholder instead of
literal editable text).*
*Verified against dev @ 6b38a13b8 — 2026-08-07 (task-2858 Task 4, LIB-19:
Database mode, Files mode, and the Sync panel each now carry a one-line
placement sentence in-app relating them to each other).*

*Re-stamped against dev @ 4acb17a0b — 2026-08-07 (TASK-2857: "Export…"
now opens the "Export bundle (.zip)" canvas, not "Export chatbook").*

*Re-verified 2026-08-09 (task-3315): the LIB-14 untouched-blank discard
and the empty-title → "Untitled" save fallback described above had
regressed on dev (the notes-adaptive session-coordinator refactor read
the seeded snapshot title instead of the presented-empty editor, and
dropped the save-seam fallback); both are restored, and Esc from the
note editor no longer dead-ends (it routes through the same guarded
Back seam as the "‹ Back to list" button).*

*Verified against feat/media-ingest-followups — 2026-08-09 (xhigh review
+ live-verify round): a note you deliberately title "Untitled" with an
empty body now survives navigating away — the discard used to compare the
title against the seed's spelling, so typing that exact word destroyed
the note with no prompt and no undo. It now keys on whether you touched
the title field at all, and the "Untitled" a blank title is saved under
matches what the notes list shows for that row.*

*Re-verified against fix/library-recritique-p1s — 2026-08-09 (task-4021:
the "Blank note no longer leaves a stray Untitled row" behavior described
above had silently regressed on dev -- the GC's emptiness check compared
against the create seam's literal seeded title, which is never blank, so
the branch was unreachable by any exit path. Restored: the check now
treats that literal seed as blank too, and the fix is proven at every exit
seam (Back, Escape, rail switch, screen leave), not just the two this
paragraph's prose already covered).*
*Verified against fix/settings-appearance-crash @ 57ad075de — 2026-08-10
(task-4023 AC#5: the Notes footer speaks the shared per-key grammar —
"ctrl+n new note | / find note | esc focus rail" on the list, "ctrl+s
save note | esc back to notes" in the editor (with shorter labels at
compact widths); locked states such as a running sync advertise no dead
keys.)*
*Verified against feat/library-queue-batch @ 0662e09f5 — 2026-08-11
(task-14902: the Sort chooser described above is this page's pre-existing
pattern — it became the Library-wide one; the table copy above was
re-verified against the live control (its label reads "Sort: Newest",
not the stale "sort: Newest ▸"), and the sync panel's Direction/Conflicts
rows were corrected to describe the always-visible ✓ choice groups the
panel actually shows.)*

*Verified on codex/notes-delete-undo-receipt — 2026-08-11 (TASK-15100:
confirmed Database Note deletion now leaves a named inline Undo/Dismiss
receipt; Undo restores the exact soft-deleted row and Notes rail count through
the version-checked service seam.)*

*Verified 2026-08-21 (TASK-19026): wide Database browsing retains the Library
rail; database editing and Files use one focused workbench with a guarded
`‹ Library / Notes` return; exact browse identity and independent scroll
positions survive return and compact/wide breakpoint crossings.*

*Verified against dev @ 5f720a404 — 2026-08-21 (TASK-19554): the conflict
policies now describe what actually happens to the losing copy. "Disk wins"
applies the disk copy (it previously applied nothing at all while reporting
the conflict as resolved), and every policy that overwrites a side saves that
side as a `.conflict-…bak` file next to the note first — fail-closed, so no
overwrite happens when the copy cannot be saved. Covered end-to-end by
`Tests/Notes/test_sync_conflict_preservation.py`.*
