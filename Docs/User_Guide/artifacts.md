# Artifacts in Library

Library's **Artifacts** section contains **All artifacts**, **Chatbooks**, and
**Reports**. These views browse local content even when the Library's other
sources use a server. Press **Ctrl+6**, or choose **Ctrl+P → Tab Navigation:
Library — Artifacts**, to open All artifacts. Existing `artifacts` routes and
configured defaults reach the same view.

## Browse and read

The layout follows the rest of Library: navigation rail, Items, and reader.
Search titles, switch between Newest and A–Z, and use First, Prev, Next, or Last
to reach the complete inventory. Each page contains up to 20 copies. Selecting
an item loads its Preview; Details shows its provenance and available files.
Reports include local timestamps and labels distinguishing Watchlist copies
from independent kept copies.

Press **Enter** in Items to focus the reader. Reader arrows scroll the body
without changing the selected item. **‹ Items** or **Escape** returns to the
selected row. Escape in a populated search field first clears the query.
**/** focuses the artifact search. Pane handles follow Library's adaptive
layout, and each artifact view remembers its query, selection, and scroll
position for the current visit.

## Reports and kept copies

Reports opens on **All reports**. It includes live Watchlist reports and
independent saved copies. **Kept** shows only saved copies and remains usable
when the Watchlists database is unavailable. Reading a report does not keep it.

**Keep in Library** saves a complete report and its complete scripts. The saved
copy remains readable and exportable after deleting its Watchlist. Live and
kept copies remain separate rows. A conflicting imported source ID cannot
silently replace or relabel an existing kept copy; Keep refuses that conflict.

**Export…** writes Markdown through the file picker. **Scripts…** opens the
existing kept-report manager on that copy. **Play** is available when the live
report has usable audio. **Watchlists** returns to the selected live report's
source. Empty Reports and failed runs offer **Try report demo**; this uses the
existing demo service and configured provider.

If a source fails, Library shows an error and Retry instead of claiming that
its last successful count is current. Other healthy artifact views remain
usable.

## Chatbooks and the pack manager

Chatbooks shows registered bundles and responses saved from Console. A saved
response previews its full stored text; if the original save was truncated,
the reader labels it **Saved response excerpt**. Other records preview their
registry metadata. Missing export files do not hide saved text or metadata.

**Use in Console** preserves the selected artifact's provenance. **Open source**
is offered when its original local conversation is available. **Manage
Chatbook packs…** opens the existing manager, including Create, Import,
Templates, Export, and Delete. The manager remains a separate workflow linked
from Library.

The inventory covers registered Chatbooks and reports. It does not scan folders
for unregistered exports or invent empty Datasets/Drafts sections.

## Share exported Chatbooks

For a Chatbook with a usable exported ZIP, **Share…** opens the existing
multi-selection dialog with that item selected. The dialog includes every
eligible registered bundle, not just the current page. Saved responses without
an exported ZIP cannot be shared through this workflow; the reader explains
why.

Sharing requires the optional web extras (`pip install 'tldw_chatbook[web]'`).
Choose a share name, optional shared username/password, and reachability:

- **This computer only (localhost)** keeps the page on this machine.
- **Local network (all interfaces)** allows other devices on your LAN. An
  unprotected LAN share requires typing `share` to confirm.
- Leave Port blank to choose a free port automatically.

Starting a share stages immutable copies of the selected ZIPs. Library shows
its URLs and count in a persistent strip with **Manage** and **Stop**, including
while reading Notes or collapsing artifact panes. The share belongs to the app
and survives navigation. Starting another share replaces it; later edits to the
original bundles do not change the staged downloads.

Recipients can download individual packs or a combined bundle, inspect
`index.json` metadata, and import packs through **Manage Chatbook packs… →
Import**. Traffic uses plain HTTP: a password controls access but does not
encrypt traffic. Use a trusted local network or a TLS reverse proxy.

**Stop** or closing the app stops the server and removes staged copies.
