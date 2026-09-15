# Browsing large folders

File and folder pickers show entries as the filesystem discovers them. You can
start browsing before the scan finishes. The status line shows **Scanning…**
and the number of entries found, then **Loaded** when the listing is ready.
While scanning, the count is what has been FOUND — filters have not been
applied yet, and it is not a percentage. The settled line counts what is on
screen, and says so when they differ: "Loaded · 4 of 6 entries shown" when a
dot-entry or a file filter is holding something back, and "Loaded · 6
entries" when nothing is (task-32622: it used to report the found total over
a shorter list, so counting the rows disagreed with the line above them).
Enter does nothing while only the parent row is available during scanning. The
first discovered entry receives the initial highlight; you can still explicitly
navigate to the parent folder.

Use **Sort** during loading or afterward to choose folders first, discovery
order, name, last modified, last accessed, created, or size. The adjacent
control selects ascending or descending order. Discovery order keeps incoming
entries at the end and has no direction. Finishing a scan does not
automatically reorder it. The parent-folder entry stays first, and sorting
preserves your selected file.

**Folders first** lists every folder before every file, each group in name
order; **Descending** reverses the names and keeps the folders on top. It is
the order every picker that can hand back a *folder* opens on, because there
the folders are the choice. That is the three Notes doors — "Import once"
(which can return either a file or one folder), "Keep a folder synced" and
Folder files' "Choose File Notes Folder" — and also every other
folder-choosing dialog in the app: importing a skill folder, binding a
workspace, choosing a TTS model or voice directory, an external model
directory, a podcast export folder. Pickers that can only return a *file*
still open on discovery order, and both orders are on the menu either way
(task-32611).

In those three Notes folder pickers each folder row also says how many notes
sit directly inside it ("12 notes"), and a folder holding an Obsidian vault is
marked "· vault". The count never descends into sub-folders and never reads
more than 500 entries of one folder, so it cannot delay the listing; a folder
that has not been reached yet, or cannot be read, simply shows nothing. A
folder with more than 500 entries is counted as far as that budget goes and
says so with a "+" — "48+ notes" means at least 48, counted from the first 500
things in the folder. Only the Notes doors show this; a model-file or
character-card picker has no use for it and asks for nothing.
**Ctrl+R** in those pickers offers the folders you last chose through that
same door, with the list focused, so **Enter** on one uses it without browsing
to it (task-32643).

Size and modification details load as rows approach the visible area. Sorting
by size or a timestamp needs metadata for all matching entries, so this work
runs in the background; the status shows **Sorting…** when scanning has finished
but ordering is still being prepared. You can still navigate away or cancel.

**Created** uses filesystem birth time when available. Unknown timestamps go
last in either direction; inode-change time is never presented as creation time.
**Last accessed** reflects the value reported by the filesystem, which may update
it infrequently. Missing or unreadable row details appear as a dash.

Search results grow as more matching files are discovered. Refresh rereads the
folder and its metadata; changing a filter or sort uses the current scan's data.
