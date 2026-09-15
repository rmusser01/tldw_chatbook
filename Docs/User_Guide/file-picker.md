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
