# Library notes — write, sync, and reuse notes stored in your Library

## What this screen is for

The Notes canvas is where you create and edit the notes stored in your
Library: quick captures, meeting notes, research summaries — anything you
want to keep, search, and later hand to the Console as context. Notes
autosave as you type, can be started from templates, imported from files,
exported as Markdown or text, and connected to reviewed local folder sync.
For the rail, landing canvas, and the other Library
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

- **Source strip** — a "Library notes | Folder files" toggle above the canvas.
  This page covers the Library notes side; see below for Folder files.
- **Notes list** — the default view: a "Notes (N)" header, the
  "Filter notes… (Enter)" field, a toolbar (sort / Add from files… /
  Export… / Select), and one row per note showing its title and age.
- **Editor** — opens when you click a note: title, body, keywords, a meta
  line with the autosave status, and an action row. On wide terminals the
  top `‹ Library / Notes` cue returns to the exact prior list row, scope, and
  scroll positions; on compact terminals use `‹ Back to list`.
- **New note view** — opens from the rail's "New note": a "Blank note"
  button plus a "From a template" list.
- **Add from files…** — asks whether this is an **Import once** or a lasting
  **Keep a folder synced** relationship before reading a source.
- **Manage sync folders** — appears when roots or migration candidates exist;
  it shows text-explicit status and the valid action for each root.

### Library notes vs. Folder files vs. lasting sync

Three different notes worlds meet here, and each surface now says so in
place: the strip above the canvas switches between two of them.
**Library notes** (this page) keeps notes inside the Library database.
**Folder files** swaps the whole canvas for the File Notes
workspace, which edits plain files under a folder you choose directly and
has its own Session Git panel — see [File notes](file-notes.md). **Keep a
folder synced** creates a reviewed, lasting relationship between one local
folder and a managed Library Notes folder. Unlike Folder files, both sides
remain distinct authorities and every reconciliation is reviewed or recovered
through the lasting-sync runtime.

## Features & controls

### Notes list

| Control | What it does |
|---|---|
| "Filter notes… (Enter)" | Type and press Enter to filter; the status line then reads "filter: \<text\> · N results". |
| "Sort: Newest" | Opens a one-row strip of Newest / Oldest / Title (✓ on the active one) in place of the action row; pick one directly, or press Escape to cancel. |
| "Add from files…" | Choose **Import once** or **Keep a folder synced** before selecting a source. |
| "Manage sync folders" | Appears only when roots or paused migration candidates exist; opens root status and contextual controls. |
| "Last import" | Reopens the latest import receipt from this app session after you return to the Notes list. |
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

### Add from files and lasting sync

**Add from files…** first asks what relationship you want:

- **Import once** copies supported files into Database Notes and ends after its
  reviewed receipt. Later changes to the originals are not tracked.
- **Keep a folder synced** creates a lasting local relationship. Choose the
  folder, direction, and local Library destination, then choose **Check
  changes**. Checking is mutation-free. Review safe actions, attention items,
  skips, filesystem effects, and deletion-like effects before **Activate
  reviewed root** is enabled.

If files or notes change after checking, activation is refused as stale and the
nearest valid action is **Check again**. Conflicts and deletion choices are not
silently settled by a global winner policy. Server setup is visibly disabled
with **Unavailable - server sync-folder capability not installed**.

For an eligible conflict where the same bound note and file both changed,
choose **View comparison** to inspect their current text and metadata, then
stage exactly one choice:

- **Keep file** updates the bound Library note from that file.
- **Keep note** replaces that file with the bound Library note.
- **Keep both** first preserves the original Library-note text as a new,
  unbound manual note, then updates the original bound note from the file.
- **Skip for now** changes nothing. The conflict stays in **Needs attention**
  and can be reviewed later.

Staging or changing a choice does not alter either authority. **Apply
reviewed** rechecks the whole review against fresh file and note state, applies
the safe actions and selected conflict resolutions that are still valid, and
can finish partially when some rows were skipped. If anything changed since
Check, Apply refuses the stale review and offers **Check again** instead.

Every completed resolution leaves an at-action receipt with **Undo** and
**Dismiss**. Dismiss hides only that receipt. Undo is offered for up to 30 days
while the exact private recovery payload is still present and both
authorities still match the applied outcome; it never overwrites later edits.
**Resolution history** survives restart and records the explicit choice and
bounded state without storing note text, hashes, or absolute paths. An expired,
changed, failed, interrupted, or unsupported resolution remains visible with
its safe next action instead of guessing or writing through the failure.

Deletion, identity, move, representation, duplicate-authority, managed-folder,
pause, capability, and activation attention remain blocked; this conflict
review does not turn them into content choices. Unsupported filesystem writes
fail closed, leave the root in attention or recovery, and do not escape the
configured sync folder.

**Manage sync folders** lists active, paused, passive, offline, attention,
recovery, stopped, and migrated-candidate states. Use **Check changes** to scan
an available root. **Review** appears when its changes need attention;
legacy candidates use **Review migration**. **Pause** and **Resume** control an
active root. **Retarget** and **Disconnect** remain visibly disabled with an
unavailable-in-this-release reason; no files or notes change.

### Import once

**Import once** copies supported note files into local Database Notes. It is
not the same as **Keep a folder synced**: the import ends after this reviewed
batch, while lasting sync retains a root relationship.

Choose files one at a time with **Add another file**, or choose one folder.
A folder is exclusive; it cannot be combined with selected files. Selected
files also need an existing-or-new destination path such as
`Research / Interviews`. The destination is only a proposal during checking;
no folder or note is created yet.

Choose **Check selection** to build a read-only review. Review groups explain
whether each source is new, an unchanged or changed repeat, an uncertain
match, unsupported, or failed. You can skip an item, create a new note, or,
when an existing match is authorized, update its content and/or add its folder
placement. Uncertain matches must be confirmed. If the imported top-level
folder already exists, choose whether to use it, create a unique sibling, or
enter another name.

Only **Import selected items** approves and executes the exact choices shown.
Progress remains visible and **Cancel import** stops cooperatively after the
current item; completed items are not rolled back. A partial receipt states
what finished. Retryable failures show **Retry N failures**; a cancelled batch
with unfinished items shows **Retry unfinished items**. **Back to Notes** may
hide a running import without stopping it; the list then offers **View import**
or **Continue import** until it settles. **Last import** reopens the same-session
receipt afterward.

## Common tasks

### Create a note from a template
1. In the rail, click **New note** under "Create".
2. Under "From a template", click **Meeting notes** (or any other row).
3. The editor opens pre-filled; just start typing — autosave handles the
   rest.

### Import Markdown files or a folder

1. In the notes list, click **Add from files…**, choose **Import once**, and
   pick the first file or one folder.
2. For files, click **Add another file** as needed and enter the Database Notes
   destination. A folder already supplies its proposed hierarchy.
3. Click **Check selection** and review classifications, actions, matches, and
   any top-level folder collision.
4. Click **Import selected items**. You can cancel cooperatively, retry work
   identified by the receipt, or return to Notes and reopen **Last import**.

### Set up lasting folder sync

1. Close any older Chatbook version using this profile, then restart the
   cutover release.
2. In the notes list, click **Add from files…** and choose **Keep a folder
   synced**.
3. Choose a local folder, direction, and local destination. Server sync remains
   unavailable until its separate capability is installed.
4. Choose **Check changes** and review the exact safe, attention, skipped, and
   deletion-like effects.
5. Choose **Activate reviewed root**. If the review is stale, choose **Check
   again** instead.

Existing legacy evidence appears as a paused candidate. Open **Manage sync
folders**, choose **Review migration**, inspect the current dry-run, and
activate explicitly. The migration never inherits a legacy conflict winner or
automatic-sync setting.

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

- Lasting root paths, bindings, operations, and recovery state live in the
  private device sync store, not ordinary `config.toml` settings.
- [Lasting Notes folder sync](../../Features/notes_bidirectional_sync.md) —
  runtime, cutover, ownership, and recovery details.
- [File notes](file-notes.md) — the **Folder files** side of the source strip.
- [Library overview](../library.md) — the rail, landing canvas, and the
  other Library sources.

## Verification evidence

TASK-19012 verifies this journey through the real `LibraryScreen` hierarchy
and the shipped CSS bundle. The mounted matrix covers Database Notes and its
Add-from-files chooser at wide and 60×20 sizes, the visibly unavailable server
destination, lasting-root attention/recovery after a fresh screen, and Folder
files with Session Git at its supported 40×20 layout. It checks painted text,
focus, compositor containment, disabled-action contrast, and the physical
messages that enter Import once.

For a local smoke check, run:

```bash
python Helper_Scripts/verify_notes_files_sync_tui.py
```

The helper creates a disposable HOME, XDG roots, config, and data directory
before importing the app. It disables model downloads, scrubs caller
credentials, proxies, SSH, and Git configuration, launches the TUI under a
unique tmux socket, and writes a bounded evidence directory containing
checksummed Library, New note, and Notes list frames at wide, 60×20, and 40×20
sizes. It never opens or migrates the caller's Chatbook databases. The
temporary profile is removed after its decoy config checksum is rechecked; the
evidence directory remains for inspection.

## Quirks & troubleshooting

- **Checking does not change Notes** — source discovery, parsing, prior-receipt
  lookup, and collision analysis are read-only. If checking fails, review the
  selected paths and destination and try again.
- **Cancellation is partial, not undo** — work already completed remains in
  Database Notes and is reported honestly in the receipt. Retry resumes only
  unfinished or explicitly retryable work from that same app session.
- **Another Chatbook process blocks activation** — close it and restart before
  activating folder sync. The cutover does not hot-swap or run two writers.
- **Unknown cutover state fails closed** — a future or unrecognized private
  marker is not repaired or downgraded. No migration or sync work starts.
- **Migration candidates stay paused** — choose **Review migration** and approve
  a current dry-run; legacy conflict/automatic settings are never replayed.
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
compact widths); locked operations advertise no dead
keys.)*

*Verified on codex/notes-delete-undo-receipt — 2026-08-11 (TASK-15100:
confirmed Database Note deletion now leaves a named inline Undo/Dismiss
receipt; Undo restores the exact soft-deleted row and Notes rail count through
the version-checked service seam.)*

*Verified 2026-08-21 (TASK-19026): wide Database browsing retains the Library
rail; database editing and Files use one focused workbench with a guarded
`‹ Library / Notes` return; exact browse identity and independent scroll
positions survive return and compact/wide breakpoint crossings.*
