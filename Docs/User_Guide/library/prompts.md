# Library prompts — save and reuse system and user prompts

## What this screen is for

The Prompts canvas stores Prompts and Recipes with a name, description,
keywords, author, and reusable System/User blocks. A Prompt can go directly to
Console; a Recipe first becomes an unsaved Prompt copy for review. You can also
apply saved Prompts from Console with `/prompt` and `/system`. For the rail,
landing canvas, and the other Library sources, start with the
[Library overview](../library.md).

## Getting there

Press **Ctrl+3** to open Library, then click **Prompts** in the rail's
"Browse" section (the row shows a live count). **Ctrl+P** →
"Tab Navigation: Switch to Library" works too. To jump straight into writing one, click
**New prompt** under the rail's "Create" section.

## Layout tour

The editor opens in **Basic** for an eligible Prompt. Both views edit the same
working copy; switching views does not convert, flatten, or save it.

```text
Basic                                    Advanced
+----------------------------------+     +----------------------------------+
| [Basic] [Advanced]               |     | [Basic] [Advanced]               |
| Name                             |     | Name                             |
| Description                      |     | Description                      |
| Instructions                     |     | Structured System/User blocks    |
| Message template                 |     | Keywords · Author · Collections  |
|                                  |     | Retained history                  |
| [lifecycle action] [secondary]   |     | [lifecycle action] [secondary]   |
+----------------------------------+     +----------------------------------+
```

- **Prompts list** — the default view: an exact "Prompts (N)" header, the
  "Filter prompts… (Enter)" field, a local collection selector, a toolbar
  ("sort: Newest" / "sort: Name", "Select", "Import…", and "Export…"), and
  one row per prompt showing its name, artifact/source/lane summary, and
  description and age when present.
  Lists longer than one 20-row page have **Previous** and **Next** controls
  plus an exact range, total, and page line, such as **"21-40 of 45 · Page
  2 of 3"**.
- **Import row** — appears inline under the toolbar when you press
  "Import…": a "File or folder path…" field, then "Browse…", "Import",
  and "Cancel", with an outcome line underneath.
- **Editor** — opens when you click a Prompt or Recipe (or create one):
  "‹ Back to list", a Basic/Advanced switch, shared identity fields, a muted
  meta line ("Modified 4h · v2"; "New prompt" before the first save), a
  save-status line, and a fixed lifecycle action area. Basic shows the concise
  Instructions and Message template fields. Advanced adds structured block,
  metadata, collection, and retained-history controls.

## Features & controls

### The prompts list

| Control | What it does |
|---|---|
| Filter prompts… (Enter) | Filters local prompt names and descriptions after a short pause; **Enter** applies the pending text immediately without sending a second request |
| collection: All prompts | Opens the local-only collection manager; choose **All prompts** or one collection |
| sort: Newest / sort: Name | Press to open a one-row strip of Newest / Name (✓ on the active one) in place of the toolbar; pick one directly (changing sort returns to page 1), or press Escape to cancel |
| Select | Enters selection mode so Prompt and Recipe rows can be checked across searches and pages |
| Previous / Next | Requests the adjacent exact 20-row page without changing search, collection, or sort |
| Import… | Opens the inline Import row (below) |
| Export… | Opens the local **Export bundle (.zip)** canvas scoped to every active Prompt and Recipe; this is separate from the editor's one-item Markdown export |
| A prompt row | Opens that prompt in the editor |

Search, collection, sort, and page form one exact local request. Search,
collection, and sort apply to the complete source before its page is chosen and
successful changes return to page 1. The header and page line use the matching
total, including results beyond the first page. While an adjacent page loads,
the last applied rows remain visible and the pager explains why it is disabled.
If a request fails, those rows remain read-only and **Retry** repeats the failed
request; the filter still shows the text you tried while the rows and range
continue to describe the last applied scope. Empty outcomes remain distinct:

- **No prompts yet. Create or import a prompt to begin.** — the local library is empty.
- **This collection has no prompts. Choose another collection or add prompts.** —
  the selected collection is valid but empty.
- **No prompts match "…". Clear the search or try different words.** — the
  current search has no matches, whether browsing all prompts or a collection.

### Selecting Prompts and Recipes

Press **Select** to enter selection mode. Each Prompt or Recipe row becomes a
literal unchecked/checked row; press it to add or remove that item. Every
selection captures the item's version, so a later delete/export can reject a
changed record instead of acting on a different revision. The summary reports
both the complete basket and the current settled page, for example `7 selected
· 2 on this page`.

- **Select page** adds every valid row on the currently settled page without
  replacing items selected elsewhere.
- **Clear all** empties the complete basket, including items hidden by another
  search, page, sort, or collection.
- **Done** leaves selection mode and discards the complete basket.
- **Export selected** opens the existing local Chatbook export canvas for
  exactly the selected active Prompt and Recipe IDs.
- **Delete selected** confirms and soft-deletes the complete selected set as
  one atomic operation.

The basket persists while you search, page, change sort or collection, and
while you visit and return from the Export canvas. It clears when you press
**Done** or **Clear all**, after a successful selected delete, when you enter an
editor or create a Prompt, when you switch to another Library source, or when
you leave Library. Selection is session-only and is not restored after an app
restart.

Delete checks the captured version of every selected item before changing any
of them. If one item is missing, changed, invalid, or cannot be deleted, nothing
in the batch is deleted and the basket remains available. A successful single
or selected delete leaves one in-place **Undo** / **Dismiss** receipt. Undo
restores the whole receipt atomically; if restore fails, nothing is partially
restored and the receipt remains available.

Bulk tagging is not part of selection mode. Use Prompt collections for the
current organization workflow.

### Local Prompt collections

The list selector and the editor's **Manage collections** action open the same
**Manage Prompt collections** surface. It is explicitly **Local only**: there is
no source or server selector. Collection rows are ordered by the local catalog's
stable exact order. Type in **Search collections… (Enter)** and press **Enter**
to search names. The manager fetches at most 100 rows at a time; **Load more**
appends the next bounded page and remains available until the complete catalog
has loaded. A failed initial, search, or later-page request shows **Retry** for
that exact catalog request.

Use **New collection** to create a name. Focus one concrete collection and use
**Rename selected** to rename it; **All prompts** is not renameable. Names are
shown literally, including Unicode and text such as `[bold]`. If two names differ
only in a way that would otherwise look identical, their rows retain literal
ID-qualified labels such as `Planning · #17`. A deterministic name collision says
**Name already exists — choose another.** and does not offer a misleading Retry.
Other failures use bounded create/rename copy and may be retried without exposing
service details. The Prompt collection manager deliberately has no **Delete**
action.

Searching the collection catalog does not silently change the active Prompt
filter. Choose **All prompts** or a collection row, then **Done**, to change it;
**Cancel** keeps the prior choice.

### Importing prompts

Press **Import…**, enter a path, press **Import**. The path can be a
single file or a whole folder; **Browse…** opens a file picker for the
single-file case only — a folder path must be typed by hand. Supported
formats: `.json`, `.yaml`, `.yml`, `.md`, and `.txt`.

The outcome line reports, for example, "2 imported · 1 skipped
(duplicate name)", adding "· 1 failed" when a file could not be parsed.
A prompt whose name already exists is always **skipped** — imports never
overwrite or rename. A bad path shows "Could not find that file or
folder."

### Bulk export with Chatbooks

From the normal Prompts list, press **Export…** to open the existing **Export
bundle (.zip)** canvas with the scope line `Prompts · N items`. The count is a
fresh, uncapped query over all active local Prompts and Recipes, not just the
visible 20-row page or current search/collection filter. In selection mode,
**Export selected** opens the same canvas for exactly the basket's active IDs,
including selections hidden by the current page or filter. Returning from the
canvas preserves the basket after success, cancellation, or failure.

Choose a destination, then press **Export bundle (.zip)**. Progress, Cancel,
Retry, overwrite disclosure, and the session's **Last export: …** receipt behave
exactly like the Library's other bundle exports. In server mode this action is
refused because Chatbook export packages local databases only.

The `Everything` scope includes Prompts alongside media, conversations, and
notes. Each new Chatbook Prompt record preserves the current portable artifact:
name, author, details, separate System and User lanes, canonical keywords,
Prompt or Recipe type, prompt format, schema version, and the exact stored
definition (including compatibility-only definitions). Multiline text,
Unicode, and markup-looking text remain literal.

The bundle deliberately does not carry source row IDs, UUIDs, client IDs,
versions, source timestamps, deleted rows, retained history, collection
memberships, or usage state. Import creates ordinary destination-owned identity,
timestamps, version, and lifecycle state. Older Chatbooks with the legacy
single-`content` Prompt payload still import; that content remains the legacy
Prompt's System lane. An unknown or invalid Prompt-record version fails closed.
If any selected Prompt disappears or cannot be represented while exporting, the
archive aborts instead of claiming a partial success.

### The prompt editor

- **Name** — required, and unique across all prompts (up to 300
  characters).
- **Description** — what the prompt is for; shown under the name in the
  list.
- **Basic: Instructions and Message template** — edits the exact existing
  System/User block when each lane has at most one block. The stored block ID,
  order, syntax, wrapper, mapping hint, and version behavior are preserved.
- **Advanced: System and User blocks** — exposes the complete structured
  block editor plus compiled previews, keywords, author, Collections, and
  retained history.
- **Remembered view** — choosing Basic or Advanced is saved for this profile.
  A Recipe, multi-block Prompt, compatibility/conversion state, version
  conflict, or record that cannot be safely updated opens in Advanced with a
  reason. This temporary safety override does not replace the remembered view.
- **Compatibility artifacts** — an unsupported structured definition is
  read-only. If its lanes can be recovered, **Convert and save as new Prompt**
  creates a detached, editable Prompt working copy: **Save Prompt** is enabled,
  the meta line reads "New prompt · Unsaved changes", and the original artifact
  is untouched.
- **Recipe starter content** — controls whether the current block text is
  included when saving a new Recipe.
- **Keywords (comma-separated)** and **Author**.

Saved local Prompts also show **Collections** with their current membership
names. If a referenced collection cannot be found, its honest fallback is
`Collection #ID`. **Manage collections** opens the shared manager in multi-select
mode. Check zero, one, or many collections; those checks are staged only. Focusing
one row also sets the separate, persistent **Rename target: …** line, so renaming
one collection never changes the staged membership set.

Returning from the manager shows current and staged memberships in the editor.
Only **Apply memberships** writes the staged set. It is independent of **Save
Prompt**: applying memberships does not save prompt content or clear its unsaved
marker, and saving content does not claim that staged memberships were applied.
An Apply failure keeps the staged set and offers another explicit Apply. If the
initial membership load fails, the manager cannot open on an unknown empty set;
use **Retry memberships** to load the exact current memberships first.

Membership controls are disabled for an unsaved Prompt, a non-local/foreign
identity, or an editor that is no longer current. After a successful Apply, the
Prompts rail count is refreshed and the hidden exact list is invalidated. **‹ Back
to list** then reloads the current search/collection/sort/page scope before showing
it; the membership outcome never says that the Prompt itself was saved.

Nothing autosaves here. While you have unsaved edits the meta line shows an "Unsaved
changes" marker, and leaving the editor (Back, another row, another screen) is
blocked until you save or resolve the edit. The save-status line reports the
outcome:

- "Saved."
- "Name already in use — pick another or open the existing prompt." —
  with an **Open existing** button that discards your edit and opens the
  prompt holding that name.
- "A deleted prompt holds this name — restore it or choose another."
- "Couldn't save this prompt. Try again."

If the Prompt or Recipe changed elsewhere while you edited, the editor shows
the conflict explanation and replaces the normal actions with **Save as new**
and **Reload**. Reload restores the current version; Save as new keeps your
blocks in a new item.

### Retained history

Saved local Prompts and Recipes include a collapsed **Retained history (…)**
disclosure. Selecting an item loads only its exact retained count; opening the
disclosure lazily loads the newest bounded page. Use **Load older versions** to
request another page. A failed count or page load offers **Retry** without
changing the editor.

Retained history contains create and update snapshots, not a complete audit
log. Each row shows its version, timestamp, artifact type, and changed-field
summary. Selecting a row reveals literal, read-only metadata plus its stored
System and User lanes. Version 1 is labeled `Created`; when pruning removed the
immediate predecessor, the summary says `Earlier baseline unavailable` rather
than guessing what changed.

Some retained rows are preview-only. Malformed, mismatched, unknown, future,
unsupported, or over-limit structured records show the exact compatibility
reason and cannot be restored. Legacy Recipe snapshots are preview-only too;
legacy Prompt snapshots remain supported. You can still inspect history while
the current editor is dirty or compatibility-only, but **Restore selected
version…** remains disabled until the working copy is clean and the current
editor is compatible.

Restore always asks for confirmation and creates a new current version; it
does not rewrite the retained row. The confirmation also calls out a
Prompt↔Recipe type change. If the Prompt changed after history was loaded, use
the editor's **Reload** conflict action before retrying. Restoring content that
already matches the current version reports `no_change` and creates no extra
version.

Newer snapshots restore their captured keywords. Older snapshots that predate
keyword capture keep the current keywords and disclose that choice. Validation,
duplicate-name, keyword-persistence, or unexpected failures leave the Prompt
and selected retained row unchanged so you can correct the issue or retry.

### The action area

The fixed action area shows only actions valid for the current lifecycle:

| State | Visible actions |
|---|---|
| New | **Save prompt**, **Cancel** |
| Saved and clean | **Use in Console**, **More actions** |
| Saved and changed | **Save changes**, **Discard changes** |
| Version conflict | **Save as new**, **Reload** |
| Mutation in progress | The relevant actions remain in place but are disabled with a readable reason |

**More actions** expands inline for a saved, clean item. It contains Export…,
Copy Markdown, Duplicate, Collections, History, and Delete. Press **Escape** to
close it and return focus to More actions.

- **Use in Console** opens the shared variable/System authorization dialog
  when needed, appends the selected User text to the Console composer, and can
  replace the session System prompt with confirmation.
- **Export…** saves a representable Prompt or Recipe as Markdown.
- **Copy Markdown** copies the exact live working copy.
- **Duplicate** opens a new unsaved copy named `<name> (copy)`.
- **Delete** confirms before soft-deleting the saved item and leaves an
  Undo/Dismiss receipt after success.

For a Prompt, **Use in Console** works differently from the notes and media
actions: instead of staging a source for retrieval, it appends selected User
text to the current Console draft — never replacing it. It refuses a dirty
Prompt ("Save your changes before using this prompt in Console."). A System-only
Prompt is allowed only through the explicit System authorization described
below. For an editable Recipe, the action stays in Library and converts the
Recipe into a detached unsaved Prompt copy for review; it does not stage text,
change the saved Recipe, or switch to Console.

### Prompt variables at insertion time

System and User lanes may contain variables such as `{customer}`. Names are
case-sensitive and must match `[A-Za-z_][A-Za-z0-9_]*`; `{customer}` and
`{Customer}` are different names. The shared **Prompt variables** dialog lists
each name once, in first-occurrence order across System then User, and uses the
one entered value for every occurrence in both lanes. Blank values are valid,
and braces inside an entered value stay literal rather than becoming another
variable.

Use `{{` for a literal `{` and `}}` for a literal `}`. Thus
`{{customer}}` inserts `{customer}`, while `{{{customer}}}` inserts the value
inside literal braces. Invalid and unmatched forms such as `{first-name}`,
`{ name }`, `{name`, and ordinary JSON object braces remain literal. Names are
limited to 64 characters and one insertion to 64 unique variables. If a limit
is exceeded, **Apply** is disabled and the dialog shows the specific bounded
message `A Prompt variable name exceeds 64 characters.` or
`This Prompt has more than 64 variables.`; **Use original placeholders** stays
enabled.

A System lane is always a separate choice. The checkbox reads
`Replace the current session System prompt with this System lane` and starts
**Off**. Turning it on may add System-only fields without discarding values you
already entered. **Apply** fills all active lanes; **Use original placeholders**
applies the selected lanes unchanged; **Cancel** applies nothing. A System-only
Prompt, including one whose User lane is blank, has no active lane until you
turn on System replacement. A Prompt with no applicable User text and no
authorized System lane makes no change.

Library appends to the active Console draft settled when Console consumes the
handoff; `/prompt` and Console's picker instead replace the complete draft they
captured when their flow opened. Library never guesses a destination: if it has
no prior Console target, it says `Open Console once, then retry Use in Console.`
If the target session or authorized System prompt changes, the application is
refused without modifying either lane. A transient composer remount is retried
only while the handoff is valid.

The 120-second expiry starts when you confirm **Apply** or **Use original
placeholders**, not while you are filling the dialog. If Console has not
consumed the application by the boundary, it expires and must be retried.
Variable values and pending applications remain memory-only and are never saved
as reusable defaults. Applied draft/System text then follows its normal
lifecycle. **Menu → Undo Prompt change** restores only the prior draft, not the
System prompt. If the live System replacement succeeds but durable persistence
fails, Console warns: `System prompt applied for this session, but the change
could not be saved -- it may not survive a reload.`

### Where prompts surface in Console

In the Console composer, `/prompt <name>` replaces the draft with a
saved prompt's user text, and `/system <name>` applies its system part
to the session; Console's "Edit system prompt" modal can also save a new
prompt to the Library. See
[Console: Context & RAG](../console/context-and-rag.md).

## Common tasks

1. **Create a prompt** — click **New prompt** in the rail's "Create"
   section, fill in **Name** and the System and/or User blocks, then press
   **Save Prompt**; the status line reads "Saved."
2. **Import a folder of prompts** — press **Import…**, type the folder's
   path into "File or folder path…" (Browse… only picks single files),
   press **Import**, and read the "N imported · N skipped" outcome.
3. **Use a prompt in Console** — open it, press **Use in Console**; you
   land in Console with its selected User text appended to the composer (and
   its System lane applied only if you explicitly authorize it). Or, from
   Console, type `/prompt <name>` to replace the current draft.
4. **Duplicate and tweak** — open a Prompt or losslessly representable Recipe,
   press **Duplicate prompt**, rename the "<name> (copy)" editor that opens,
   adjust the blocks, then use the enabled save action — the original is untouched.
5. **Bulk-export all local Prompts and Recipes** — from the Prompts list, press
   **Export…**, confirm `Prompts · N items`, choose a destination, then press
   **Export bundle (.zip)**. Use the rail's **Export** row and `Everything` when
   you also want media, conversations, and notes.
6. **Export one Prompt or Recipe as Markdown** — open a losslessly representable
   artifact, press **Export…**, and pick a location; a notice confirms the export.
   A compatibility artifact or legacy Recipe that fails this check requires
   **Convert and save as a new Prompt** before Copy, Export, or Duplicate.
7. **Browse one collection** — press **collection: All prompts ▸**, choose a
   collection, then press **Done**. Search and paging now stay inside that exact
   collection until you explicitly choose **All prompts** or another collection.
8. **Change a Prompt's memberships** — open a saved local Prompt, press **Manage
   collections**, stage the checks you want, press **Done**, then press **Apply
   memberships**. Save any content edits separately with **Update original**.

## Keyboard & commands

| Key | Action |
|---|---|
| Enter (in the Prompt filter) | Apply the pending debounced Prompt search immediately |
| Enter (in collection search) | Run the collection-name search from page 1 |
| Escape (in the collection manager) | Cancel the manager unless a create/rename is still settling |

`/prompt` and `/system` are Console commands, documented in
[Console: Context & RAG](../console/context-and-rag.md).

## Related settings & docs

- No `config.toml` keys belong to this panel.
- [Library overview](../library.md) — the rail, counts, and the other
  sources.
- [Library skills](skills.md) — skills are also created and imported
  from the rail, but go through a trust review before use.
- [Console: Context & RAG](../console/context-and-rag.md) — `/prompt`,
  `/system`, and the "Insert prompt" picker.
- [Guide index](../index.md) — global keys and navigation.

## Quirks & troubleshooting

- **Imports skip duplicates silently by design** — a name that already
  exists (even on a deleted prompt) is counted as "skipped (duplicate
  name)", never overwritten or auto-renamed.
- **The filter ignores keywords** — it matches only the name and
  description, so a keywords-based search will come up empty.
- **Collections here are local-only** — this manager does not browse or mutate
  server collections, and it does not offer collection deletion.
- **Size caps**: names up to 300 characters; the system prompt, user
  prompt, and description up to 2,000,000 characters each.
- **Two Export… actions serve different formats** — the Prompts-list action
  writes a multi-item Chatbook `.zip`; the editor action writes one Prompt or
  Recipe as Markdown.

—
*TASK-198, TASK-202, and TASK-196 behavior in this guide was reverified against
the real Textual compositor and focused automated suites on 2026-08-09.
Verification is tied to branch history rather than a self-referential
documentation SHA.*
*Verified against feat/library-queue-batch @ a899cbf6a — 2026-08-11
(task-14901 / ADR-055: the delete confirmation copy now states the
deletion cannot be undone from Library, on every variant — clean, dirty,
and multi-item.)*

*Verified against feat/library-queue-batch @ 0662e09f5 — 2026-08-11
(task-14902: the sort control converged on the Library chooser-strip
pattern — press opens Newest / Name with ✓ on the active one, a pick
requests that exact scope at page 1, Escape cancels; the collection
control's label dropped the cycle glyph — it opens the collection
manager, a direct-pick surface, and never cycled.)*
