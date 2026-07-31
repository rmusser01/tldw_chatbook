# File Notes — plain files on disk, edited in place

## What this screen is for

File Notes edits ordinary files that live in a folder you choose on disk —
what you see in the editor is exactly what's in the file, and saves write
straight back to it. It is a separate system from [Database notes](notes.md):
nothing here is stored in the Library database, there are no templates or
autosync, and no "Use in Console" handoff. Reach for it when your notes are a
folder of Markdown files (a wiki, a repo's docs, an Obsidian vault) and you
want to read, edit, search, and — if the folder is a Git repository — stage
and commit this session's edits without leaving the app.

## Getting there

Open [Library](../library.md) (**Ctrl+3**), pick **Notes** in the rail's
Browse section, then use the source strip at the top of the canvas: it reads
**Database** | **Files**. Click **Files** — while the workspace loads you'll
briefly see "Opening File Notes…". Click **Database** to switch back; either
switch first saves any unsaved edits on the side you're leaving.

## Layout tour

![File Notes workspace](../images/library/file-notes.svg)

- **Folder link row** (top) — before setup the status reads "Choose a notes
  folder." with buttons **Details** and **Choose folder…**. Once linked, the
  status becomes "Linked — \<folder\>" (or "Checking — …" / "Offline — …"
  when the folder can't be verified) and the button relabels to **Change…**.
  **Details** opens the read-only "File Notes folder details" dialog.
- **Navigator** (left) — a "Search file contents…" input, the **Files** tree
  of everything under the linked folder, a **Search results** tree that
  appears only while a query is active, and the **Session Git (N)** button —
  N counts the files changed in this session.
- **Editor pane** (right) — a breadcrumb ("No file selected" until you open
  one; "Recently deleted: \<path\>" right after a delete), a save-status
  label (Idle / Dirty / Saving / Saved / Conflict / Error, sometimes with a
  detail after a dash), a path input with placeholder "relative/path.md",
  two toolbars (**New Move Delete Restore Protect** and
  **Reload Save Copy Refresh**), the text editor itself, and an action-status
  line where results like "Deleted. Restore remains available." appear.
- **Session Git panel** — pressing **Session Git (N)** swaps the whole
  workspace for the staging-and-commit panel described below; **Esc** or
  **Back to navigator** returns to the files.

## Features & controls

### Folder link

| Control | What it does |
|---|---|
| **Choose folder…** / **Change…** | Opens the "Choose File Notes Folder" picker; the choice is saved to `[file_notes] root` in config.toml |
| **Details** | Opens "File Notes folder details" — a read-only status report; **Close** or **Esc** dismisses it |

### Editor toolbar

The path input ("relative/path.md") is the target for **New**, **Move**, and
**Save Copy** — type where you want the file to go, then press the button.

| Control | What it does |
|---|---|
| **New** | Creates an empty file at the typed path and opens it — there is no file picker for creating content |
| **Move** | Moves the open file to the typed path |
| **Delete** | Two-press: first press shows "Click Delete again to confirm.", second deletes ("Deleted. Restore remains available.") |
| **Restore** | Brings back the most recently deleted file |
| **Protect** / **Unprotect** | Toggles protection on the open file ("Protected." / "Unprotected."); every save to a protected file first stores a checkpoint of its previous contents in the local recovery database |
| **Reload** | Re-reads the open file from disk, replacing the editor contents |
| **Save Copy** | Writes the editor text as-is to the typed path; only enabled while the save status is Dirty, Conflict, or Error |
| **Refresh** | Re-scans the folder and rebuilds the **Files** tree |

Saving is automatic — edit and the status walks Dirty → Saving → Saved. If
the file changed on disk underneath you the status shows Conflict; **Reload**
takes the disk version.

### Session Git — stage and commit this session's edits

The panel is headed "Prepare session for commit" with the scope line
"Session paths only · stages complete file state" and the keyboard guide
"Up/Down Select | Tab Actions | Enter Run | Esc Back". Before anything runs
it shows "Repository: not checked" / "Status: NOT CHECKED".

**Trust first.** Press **Trust and check status** and a confirmation dialog
titled "Trust Session Git repository?" appears:

> Repository: \<path\>
>
> Trust lasts only for this application process. Git status and staging may
> execute configured Git filters, including arbitrary programs with side
> effects outside Chatbook.
>
> Continue only if you trust this repository and its Git configuration.

**Cancel** is focused first, so Enter alone runs nothing. Confirming with
**Trust and check status** checks the repository and, from then on, a
**Refresh** button takes the trust button's place.

**Rows.** Each file edited this session gets a row whose second line states
where it stands:

| Row status | Meaning |
|---|---|
| READY TO STAGE · Git: unstaged | Your edit can be staged |
| STAGED · by Chatbook | Already staged from here |
| UPDATE AVAILABLE · newer note edits are not staged | You edited again after staging |
| UPDATE REQUIRED · stage the moved note before unstaging | A move needs restaging first |
| NO ACTION · matches HEAD | The file equals the last commit |
| BLOCKED · already staged outside Chatbook; manage this path in Git, then Refresh | Hands off — you staged it yourself |
| BLOCKED · ignored by Git / Git conflict / Git unavailable | Fix the condition outside Chatbook, then **Refresh** |

With no session edits the panel says "No current-session Git changes."

**Staging and committing.** Use **Stage** / **Unstage** on a row, or
**Stage all (N)** / **Unstage all (N)**. **Commit staged (N)** stays disabled
behind the gate "Stage at least one session note to commit" until something
is staged. It then opens the commit form — **Subject** (placeholder
"Required commit subject") and **Body (optional)** — and **Review commit**
runs a pre-check ("Checking commit...") before showing the review: the
"Exact commit message" as Git will record it, a "Show included notes (N)"
toggle listing every file going in, and branch/identity details. Finish with
**Confirm commit**, or step back with **Edit message** / **Cancel commit**.
If a result comes back uncertain, **Check again** re-checks it.

## Common tasks

1. **Link a notes folder.** Open Files (source strip), press
   **Choose folder…**, pick the folder in "Choose File Notes Folder". The
   status becomes "Linked — \<folder\>" and the **Files** tree fills in.
2. **Create a file.** Type its location — e.g. `ideas/today.md` — into the
   "relative/path.md" input and press **New**. The file is created on disk
   and opened; start typing and it saves automatically.
3. **Find text across files.** Type a query into "Search file contents…" —
   a **Search results** tree appears under the **Files** tree; pick a result
   to open that file. Clear the query and the tree disappears.
4. **Stage and commit this session's edits.** Press **Session Git (N)**,
   trust the repository if asked, press **Stage all (N)**, then
   **Commit staged (N)**. Fill in **Subject**, press **Review commit**,
   read the "Exact commit message", and press **Confirm commit**.
5. **Restore a deleted file.** After a delete the breadcrumb shows
   "Recently deleted: \<path\>" — press **Restore** and the file is back on
   disk and in the tree.

## Keyboard & commands

| Key | Action |
|---|---|
| Up / Down (Session Git panel) | Select a row |
| Tab (Session Git panel) | Move into the selected row's actions |
| Enter (Session Git panel) | Run the highlighted action |
| Esc (Session Git panel) | Step back safely: from the row list, back to Files; from the commit form, cancel the commit; from the review, back to editing the message |
| Esc (dialogs) | Close "File Notes folder details" or cancel the trust dialog |

## Related settings & docs

- **config.toml `[file_notes]`** — `root` is the linked folder; written
  whenever you use **Choose folder…** / **Change…**.
- [Database notes](notes.md) — the Library-stored notes system, with
  templates, the Notes sync panel, and Console handoff. The sync panel there
  mirrors DB notes to a folder; File Notes is different — the files *are*
  the notes.
- [Library](../library.md) — the parent screen; [guide index](../index.md)
  for global keys.
- There is no deeper Docs/Features write-up for File Notes or Session Git —
  this page is the reference.

## Quirks & troubleshooting

- **Per-file caps: 8 MB and 2,000,000 characters.** Edits that would push a
  file past either limit are refused at save time.
- **Staging is session-scoped and whole-file.** Only files touched in this
  session appear in the panel, and staging records each file's complete
  current state — not a partial diff. A path you already staged outside
  Chatbook shows BLOCKED here on purpose: finish it in Git, then **Refresh**.
- **Trust doesn't persist.** The "Trust Session Git repository?" dialog
  returns after every app restart — trust lasts only for the running
  process, by design.
- **No Console handoff.** Unlike Database notes, media, and prompts, this
  workspace has no "Use in Console" — copy text out manually if you need it
  in a chat.
- **Toolbar briefly refuses during Git work.** While a stage/unstage/commit
  is running you may see "Session Git mutation in progress; structural
  actions are busy." — wait a moment and retry.
- **Search and Restore need the recovery database.** Both are backed by a
  local index; if it can't be opened you'll see "Recovery unavailable:
  \<error\>", search falls back to a slower direct scan, and **Restore** may
  be unavailable.

—
*Verified against dev @ bd05a692a — 2026-07-31*
