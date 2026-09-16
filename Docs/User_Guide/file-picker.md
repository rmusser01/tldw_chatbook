# Browsing large folders

File and folder pickers show entries as the filesystem discovers them. You can
start browsing before the scan finishes. The status line shows **Scanning…**
and the number of entries found, then **Loaded** when the listing is ready.
The count includes entries hidden by the current filters; it is not a percentage.
Enter does nothing while only the parent row is available during scanning. The
first discovered entry receives the initial highlight; you can still explicitly
navigate to the parent folder.

Use **Sort** during loading or afterward to choose discovery order, name,
last modified, last accessed, created, or size. The adjacent control selects
ascending or descending order. Discovery order keeps incoming entries at the
end and has no direction. Finishing a scan does not automatically reorder it.
The parent-folder entry stays first, and sorting preserves your selected file.

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

## Choosing a folder in a compact terminal

The folder picker keeps its listing, editable path and **Select**/**Cancel**
actions visible at 80×24. Use the arrow keys and Enter to open a folder or its
parent, or enter a path directly. Resizing preserves the typed path, selection
and keyboard focus.

Invalid paths leave the picker open so you can correct them. Submitting a valid
path clears the error even when it names the current folder. Long diagnostics
are shortened on screen in compact terminals to leave room for folder rows;
the full typed path stays in the editable field.
