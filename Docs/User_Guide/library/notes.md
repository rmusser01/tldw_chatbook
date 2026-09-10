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

Database Notes uses three side-by-side roles when there is room: Library
navigation, the Notes list, and the note you are working on. The Library
navigation and Notes list each have their own slim collapse grip. Collapsing
one does not collapse the other, and each grip remembers its own choice.

Wide Database Notes keeps Library navigation beside the list while you scan:

```text
+----------------------+-----------------------------------------------+
| Library              | Database | Files                              |
| Browse               | Notes (N)                                     |
|   Notes              | Filter...  New  Select  Add from files… Export |
|   Media              | New folder   Move   Remove                    |
|   Conversations      |  Note title · age                             |
| ...                  |  Note title · Folder · age                    |
+----------------------+-----------------------------------------------+
```

The first time you open an editable note during a wide Notes work session,
Library navigation closes automatically once to give the note more room. This
temporary close does not change your saved pane choice. Use the visible grip
to reopen Library navigation; after you do, it stays open for the rest of that
work session. Opening another note, switching between Edit, Preview, and Info,
saving, resolving a conflict, or resizing the terminal does not close it
again. The automatic close becomes available again only after you clear the
selected Database note, switch between Database Notes and Folder Files,
change the linked Folder Files root, leave Notes, or close the open Folder
Files file. Folder Files' compact **Back to navigator** action is not a reset.

When Library navigation is closed, one stable cue names the return
destination:

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
  This page covers the Library notes side; see below for Folder files. At
  wide sizes both switches stay on the strip in either mode, so switching
  back from Folder files is the same control you switched in with. On
  compact terminals the strip becomes a **‹ Library / Notes** cue instead.
- **Notes list** — the default view: a "Notes (N)" header, the
  "Filter notes… (Enter)" field, a toolbar (**New** / Select / Add from
  files… / Export, plus the folder and placement actions — two rows on a
  wide list, three when the list is too narrow to seat two groups on one),
  the folder tree, and one row per note showing its title and how long ago
  it changed
  ("3m", "1d"). When two notes in the same folder share a title, each row
  also names its folder — "Reading list · Unfiled · 2h". While no note is
  open the list takes the width the empty work area would otherwise waste,
  so long titles are not truncated on a wide terminal; opening a note hands
  that width back. Its own grip collapses or restores the list without
  changing the Folder Files tree choice. Renaming a note does not repaint its
  list row live while the note stays open: a Notes refresh that lands while
  the title field holds focus is skipped rather than queued, so tabbing or
  clicking to another field does not by itself catch it up. The row shows the
  new title the next time the canvas refreshes with focus outside the title
  and body — in practice, that means returning to the list (the note work
  area's own **‹ Notes** / **‹ Back to list** control, below), which always
  repaints immediately; no filter re-query is needed once you are back.

  The folder tree lists notes **newest first by default** — every folder's
  notes, and the automatic **Unfiled** group, in the order they were last
  changed. **Sort** switches that to **Oldest** or to **Title** (by name),
  and your choice is remembered. The order you pick is the order the
  database pages notes in, so choosing one reloads the tree — including
  the folders you have open — and a note really does move between pages
  rather than only shuffling within the page you can see; jumping straight
  to a note lands on the page that order puts it on. While a filter is
  active, results come back in the search's own order (grouped by folder),
  so **Sort** is disabled there and says so.

  In a narrow list pane the toolbar's action groups stack one action per
  line rather than running off the pane edge, so every action stays
  pressable.
- **Note work area** — opens when you click a note. **Edit** shows the title
  and body, **Preview** renders the Markdown, and **Info** holds keywords,
  dates, version details, copy/export actions, and Delete. Save status and
  frequent actions remain in the header. On wide terminals the
  top `‹ Library / Notes` cue returns to the exact prior list row, scope, and
  scroll positions; on compact terminals use `‹ Back to list`.
- **New note view** — opens from the rail's "New note": a "Blank note"
  button plus a "From a template" list.
- **Add from files…** — asks whether this is an **Import once** or a lasting
  **Keep a folder synced** relationship before reading a source. Both buttons
  sit together, each directly under its own description; the bar below holds
  only **Back to Notes**.
- **Manage sync folders** — appears when roots or migration candidates exist;
  it shows text-explicit status and the valid action for each root.

### Library notes vs. Folder files vs. lasting sync

Three different notes worlds meet here, and each surface now says so in
place: the strip above the canvas switches between two of them.
**Library notes** (this page) keeps notes inside the Library database.
**Folder files** is a mode of this same Notes screen: it swaps the work area
for the File Notes workspace — which edits plain files under a folder you
choose directly and has its own Session Git panel — while the Library rail
and the source strip stay where they were. See
[File notes](file-notes.md). **Keep a
folder synced** creates a reviewed, lasting relationship between one local
folder and a managed Library Notes folder. Unlike Folder files, both sides
remain distinct authorities and every reconciliation is reviewed or recovered
through the lasting-sync runtime.

### Portable organization with Sync v2

For an eligible local-first server profile, Manual Sync can also carry the
logical organization of Library notes: keywords, keyword collections, Database
Notes folders, and their memberships. These six organization types enroll as
one capability; Chatbook will not synchronize only part of the group.

The first run may pause for an adoption review when a local and server object
have the same visible name or path but different identities. Open **Settings**,
find Manual Sync, and choose whether to merge, rename the local object, or keep
it local. Until review and initial inventory finish, organization stays local
and publication is blocked. Interrupted enrollment and sync can be retried from
the same Manual Sync surface; committed checkpoints and pending changes are
resumed rather than rebuilt.

Deleting a Database Notes folder synchronizes only that explicit deletion. Its
descendants and memberships stay dormant and become effective again after the
folder is restored. If a note belongs to a folder through both manual and
source-managed placement, removing one placement does not remove the portable
membership while the other remains effective.

This does not synchronize filesystem paths or grant filesystem access. Folder
Files and lasting folder sync remain device-private authorities. For the
identity, dependency, suppression, conflict, and recovery contract, see the
[Sync-v2 client runtime](../../Development/Sync-v2-client.md).

### Let a permitted agent find and organize notes

Console agents and local in-app MCP clients use the same bounded Notes tools.
`library_search_notes` can combine a normal literal text query with an exact
folder or whole-keyword filter. At least one of these is required:

- `query` searches note title, body, and keywords case-insensitively;
- `keyword` matches the trimmed spelling exactly, so `agent-lesson` does not
  match `Agent-Lesson`;
- `folder_id` uses the stable opaque folder identity returned by an earlier
  read; and
- `folder` resolves an exact relative Database Notes path. It is not a local
  filesystem path. Ambiguous, deleted, or conflicting folder selections are
  refused instead of guessed.

Search and read results include bounded folder and keyword metadata, exact
totals, and an opaque `organization_version`. Folder/keyword changes advance
that token independently of note-body versions. Returned note content and
organization names are explicitly untrusted reference data: they cannot grant
an agent permission, authorize an action, or override your instructions.

With the **`library.notes.save.local`** permission, `library_save_note` can
create a note or update the exact `note_id` plus `expected_version` returned by
a prior read. `ensure_keywords` only adds missing whole keywords; it never
removes your existing keywords. A `folder_id` attaches the note to that exact
portable folder, while `folder` can ensure one root-level folder by name. Both
forms are additive: they do not move the note out of its other folders.
Organization-changing updates must also supply the latest
`expected_organization_version`. If either content or organization changed,
the agent must re-read, show the new state, and retry rather than overwrite it.

If Notes organization is offline or still enrolling, a permitted save can
remain locally visible as **pending organization**. Chatbook keeps it out of
normal sync dispatch until readiness finalization commits the note and its
organization publication together. A folder-name collision becomes
**placement review** once note and keyword publication are safe; the note stays
usable while you resolve its folder. These states survive restart. Policy
denial creates no note, folder, keyword, receipt, or hidden write.

Because note titles are not unique, an agent should search before creating and
update a confirmed match instead of blindly retrying a create. The canonical
Agent Lessons marker is the spelling-exact `agent-lesson` keyword; the
`Agent_Lessons` folder is visible organization, not the discovery authority.

### Reuse solutions with Agent Lessons

Chatbook creates one conventional root folder named `Agent_Lessons` after the
applicable Notes organization readiness boundary. It remains an ordinary,
user-owned Database Notes folder: you may rename, move, or delete it, and
Chatbook does not recreate a folder you deliberately changed. On synchronized
profiles, only a provably untouched empty seed race converges automatically;
edited, acknowledged, used, differently spelled, or colliding candidates wait
for your existing organization review.

The folder is for browsing. Agents discover lessons by searching the
spelling-exact whole keyword `agent-lesson`, so a marked lesson remains
discoverable after folder changes. A lesson uses one note for one reusable
solution and records:

- applicability and symptoms;
- feedback or the trigger that exposed the issue;
- privacy-preserving provenance and the independently observed root cause;
- the verified solution;
- real failed attempts and why they failed, or an honest statement that the
  first tested approach succeeded;
- verification evidence;
- one generalizable principle and its rationale;
- caveats and related public `note:` IDs; and
- optionally, a promotion candidate for later human review.

When troubleshooting, a capable foreground agent is guided to search first,
read likely matches, verify their versions and applicability, and update an
existing lesson with the same root cause instead of creating a duplicate. A
subagent may search and return evidence or a structured draft to the foreground
agent, but it cannot save an Agent Lesson.

Before any classified lesson save, the foreground primary must show the exact
proposed title, complete content, organization, target note identity, and
expected versions. The approval card offers only **Approve once** or **Deny**.
Approval applies to that exact call and observed Note/organization state; a
changed note, marker, pending placement receipt, payload, role, or replay is
refused without a partial write. This extra review still applies when ordinary
Notes saves are broadly allowed. A denied or abandoned preview creates no Note,
folder, keyword, receipt, or hidden draft.

Lesson content that resembles a high-confidence live credential is refused and
is not echoed in the error. Redacted values, clearly fake examples, long hashes,
UUIDs, stack traces, and error IDs remain recordable. Every retrieved lesson is
still **untrusted reference data**: it can suggest a solution to verify, but it
cannot grant permission, authorize a command, expand filesystem/network scope,
or override your current instructions.

### Promote a verified lesson into reusable instructions

When a lesson contains independently verified, procedural, reusable evidence,
the foreground agent may nominate one small instruction improvement. One strong
signal can be enough; repetition alone does not make weak or contradictory
evidence authoritative. The proposal should preserve the general principle and
why it works, state unknowns, and avoid accumulating incident-specific rules.

Repository instruction promotion is limited to `AGENTS.md` or
`AGENTS.override.md` inside the selected writable folder binding. Console first
asks whether it may prepare an exact read-only proposal. The result identifies
the binding, current applicable instruction chain, target digest or absent
state, complete replacement, bounded diff, evidence, and verification. Applying
that exact result is a second **Approve once** / **Deny** decision. If the file,
binding, or applicable instruction chain changed, the application writes
nothing and requires a fresh proposal; intervening user edits are never reset.

For a Chatbook-managed local skill, Console can prepare and show exact replacement
text but cannot apply it. Use **Library ▸ Skills**, open the named skill, paste the
accepted text, and save against the current version. The edited skill remains
blocked until you review and approve its new trust baseline. The agent may
re-read it afterward to verify the result. Raw workspace tools never gain access
to Chatbook's managed skill store.

A rejected, stale, failed, or applied outcome is durable only if you separately
approve updating an ordinary Agent Lesson Note. That outcome is historical
evidence for later searches; it never authorizes a later write or another
device.

## Features & controls

### Notes list

Database Notes are presented as a folder tree rather than one flattened
snapshot. Each level loads independently in fixed pages of 20:

- A folder-row **More folders** control loads the next 20 direct children of
  that exact folder. The root has its own folder control.
- A note-row **More notes** control loads the next 20 visible placements in
  that exact folder. **Unfiled** has its own independent note control.
- When a link, Back action, or restored selection lands in a middle page, the
  branch shows its truthful range and a **Load earlier** control. Loading an
  adjacent page preserves the tree's scroll position unless the activated
  pager still owns keyboard focus; in that case focus advances to the first
  newly loaded row.
- A failed branch request leaves the existing rows in place and changes only
  that branch's control to **Retry**. Retrying keeps keyboard focus on the
  control while loading and moves it only after the requested rows arrive.
- Collapsing a folder keeps its fresh branch pages for a quick re-expand.
  Mutations and stale results refresh only the affected folder branches.

Filtering uses the same placement-aware hierarchy and bounded pages: matching
notes retain the ancestors needed to understand their location, duplicate
placements remain distinct, and the result count is the exact query total.
Long folder and note titles are clipped inside the Items pane rather than
creating horizontal terminal overflow. The Library and Items grips remain
independently collapsible; closing Library gives its width to the title tree,
and closing Items gives its width to the note work area. These are durable
choices: resizing, refreshing a branch, or opening and closing Library does
not reopen Items after you intentionally close it; if you close both panes,
both stay closed until you choose to reopen one.

| Control | What it does |
|---|---|
| "Filter notes… (Enter)" | Type and press Enter to filter; the status line then reads "filter: \<text\> · N results". |
| "New" | Creates a note directly from the list — the same destination as the rail's **New note** row, without the template picker. Disabled while another notes operation is running. |
| "New folder" | Creates a folder in the tree beneath the toolbar. Disabled, with the reason in its tooltip, when the selected folder is sync-managed ("This folder is managed by sync; change its sync root instead.") or its branch is stale ("This branch may be out of date; retry it before changing it."). |
| Folder selected: "Rename" / "Move" / "Remove" | Act on the selected folder. Same two disabled reasons as **New folder**. |
| Note selected: "Add to folder" / "Move note" / "Remove placement" | File the selected note into a folder, move its placement, or take it out again. A sync-managed placement is refused with "This placement is managed by sync; change its sync root instead."; a note sitting in the automatic **Unfiled** group cannot have its placement removed ("Unfiled is shown automatically; move the note into a folder."). |
| "Restore folder" | Appears after a folder removal, to put it back. |
| "Sort: Newest" | Opens a one-row strip of Newest / Oldest / Title (✓ on the active one) in place of the action row; pick one directly, or press Escape to cancel. **Newest is the default.** The value is the order the folder tree is paged in, so choosing a new one reloads the tree (open folders included). Disabled while a filter is showing, with "Filter results keep their own order. Clear the filter to sort." |
| "Add from files…" | Choose **Import once** or **Keep a folder synced** before selecting a source. |
| "Manage sync folders" | Appears only when roots or paused migration candidates exist; opens root status and contextual controls. |
| "Last import" | Reopens the latest import receipt from this app session after you return to the Notes list. |
| "Export…" | Opens the "Export bundle (.zip)" canvas scoped to notes — bundle notes into a .zip. |
| "Select" / "Done" | Toggles select mode: rows grow ☑/☐ checkboxes, and a row appears with "N selected", "Select all N shown", "Clear", and "Export selected". "Export…" hides while selecting. |

With no notes at all, the list reads "No notes yet. Create your first note."
above the tree — even when the seeded **Agent_Lessons** folder (see "Reuse
solutions with Agent Lessons" above) is the only row showing, so a
first-time user is never left staring at one unexplained folder with no
other cue. While the library holds zero notes, that folder row itself also
carries a one-line gloss: "Agent_Lessons — where Console agents file
reusable lessons (empty)".

### Edit, Preview, and Info

| Control | What it does |
|---|---|
| "‹ Notes" / "‹ Back to list" | Returns to the list (your text is already saved — see autosave below). One wording across Edit, Preview, and Info: "‹ Notes" at wide sizes, "‹ Back to list" on a compact terminal. |
| **Edit** | Shows the editable title and body. This is the default view when you open a note. |
| **Preview** | Shows the note's title above the body, rendered as Markdown, without replacing your draft. |
| **Info** | Shows Properties (including comma-separated keywords, note dates/version, and **Linked from**), Reuse & Export, and Danger sections. |
| **Linked from (N)** (Info → Properties) | Lists the notes whose bodies link to this one, newest import or not — the `[title](note://…)` links Import once writes for an Obsidian vault's `[[wikilinks]]` (see "Obsidian vaults"). Click an entry to open that note. While the lookup runs the line reads "Linked from — checking…", and "Linked from — couldn't check" if it failed, so a count is only claimed once the answer is in. When nothing points here the line reads "Linked from (0) — no notes link here yet". The list is capped at 50 entries; past that the count reads "50+". Links you type by hand in the body count too, as long as they use the same `note://` form. |
| Status line | Shows the autosave state: "Saved", "Saving…", "Unsaved changes", "Conflict — …", "Save failed — …", or "Unavailable — …". It does not carry a word count. Created/Modified/version details are under Info → Properties, each with an absolute local timestamp beside its relative age and the word count (e.g. "Created 2026-09-08 21:14 · 3m ago · Modified … · v1 · 6 words"). "Saved" appears once per view, not repeated in Info. |
| **Save** | Saves immediately, without waiting for autosave. It remains visible beside the mode controls. |
| **Use in Console** | Hands the note to the Console as staged context, with the suggested prompt "Use this note as context and help me work with it." It remains visible beside **Save**. |
| **Copy** (Info) | Copies the note to the clipboard as Markdown — "Note copied to clipboard as markdown!" |
| **Export Markdown** / **Export text** (Info) | Saves the note to a file you pick; success shows "Note exported successfully to \<name\>". |
| **Delete** (Info → Danger) | Asks inline, in place — Info stays open, the prompt renders where Delete was pressed: "Delete this note? Undo will be available in the Notes list." Tab / Shift+Tab cycle only between **Cancel** and **Delete** while it is open, and the footer names whichever one is focused ("enter cancel" or "enter delete"). Every other Info action — including "‹ Notes" / "‹ Back to list" — is disabled until you choose Cancel or Delete. A successful delete returns to the list with a named "✓ deleted · …" receipt offering **Undo** and **Dismiss**. |

Opening a note shows "Loading note…" only while the note is being read. If a
read takes longer than about three seconds the editor stops waiting and shows
"Unable to load note — timed out after 3 s. Press Retry." with a **Retry**
button; **‹ Notes** takes you back to the list, and opening another note still
works.

**Autosave** runs about two seconds after you stop typing; the meta line
flips to "saving…" and back to "saved". If the same note was changed
somewhere else while you were editing, a banner appears: "This note
changed elsewhere — Overwrite saves your text; Reload discards it." —
pick **Overwrite** or **Reload**.

While any editor field — the title, the body, or either keyword box — has
keyboard focus, nothing repaints the editor underneath you: a refresh that
arrives mid-sentence — a save landing, the
first note reaching the list, the Library graduating to its full rail — leaves
the editor alone, so keystrokes never land in the wrong box and focus never
jumps away as you type or after you move to another field. The next refresh
that arrives once your hands are off the field paints normally.

Notes does not use **Ctrl+S**, and there is no replacement Notes save
shortcut. Use the visible **Save** button when you want an immediate Database
save; normal Tab navigation and **F6** can reach it. Autosave continues to
handle ordinary typing.

When the note body has keyboard focus, only its boundary becomes more
prominent. The body background and editor size stay unchanged, so focusing
the editor does not flash or fill the writing surface. Small fields such as
Title and Keywords still use their usual filled focus treatment.

The wide `‹ Library / Notes` cue and Escape use the same guarded return as
the compact Back control. A dirty save, sync, conflict, reload confirmation,
or running mutation can therefore keep the focused task open until it is safe
to leave. A successful return restores the Database/Files source, filter,
sort, selected note or placement, Notes-list scroll, Library-rail scroll, and
semantic keyboard focus instead of starting over at the first row.

If a save is blocked — most commonly a title that starts or ends with a
space — Escape does not leave silently: it notifies "Can't leave yet — fix
the title or press Discard new note." so there is always a visible way
forward, either fixing the field or discarding a still-new note.

After a confirmed delete, the receipt stays in the Notes list until you
choose **Undo**, choose **Dismiss**, complete a newer note deletion, or leave
the list for **Add from files** or **Folder files** — either of those also
dismisses a still-open receipt, since it is scoped to this list session. Its
"✓ deleted · \<title\>" line sits above the two actions rather than beside
them, so **Undo** and **Dismiss** stay reachable however narrow the list is.
**Undo** restores that exact database note and immediately returns its row —
in its folder, or under Unfiled — along with the Notes rail count, and moves
the selection to the restored row. **Dismiss** removes only the receipt; the
note remains deleted. *(This page previously said Notes expose no separate
Trash browser, so the receipt was the only in-Library recovery action —
superseded by task-32144: see "Recently deleted" below, which recovers a note
whose receipt was dismissed.)*

*Verified against fix/library-notes-list — 2026-09-09 (task-32123: the
receipt's actions are no longer composed off the pane; task-32124: Undo
returns the row to the folder tree, not only the count).*

### New note view

"Blank note" drops you straight into the editor with an empty title (shown
as an "Untitled" placeholder — just start typing) and an empty body. Its
status reads "Draft — not saved yet" until you type the first character or
press **Save**, rather than "Saved" before anything you have written is
actually kept. If you leave again via "‹ Back to list" without typing
anything, the blank note is quietly discarded rather than left behind as a
stray "Untitled" row.
Pressing "Save" keeps it, and so does typing anything **that is not only
whitespace** — a title of nothing but spaces, with an empty body and no
keywords, still counts as blank and is discarded on the way out. That includes naming it "Untitled"
yourself: once you have touched the title field the note is yours, and it
is kept even with an empty body. The "From a template" list
pre-fills title, body, and keywords instead; each row shows the template
name with the title the note will get. Available templates: Brainstorming
session, Bug report, Code review, Daily journal entry, Meeting notes,
Project planning, Research notes, Todo list.

Opening this view parks keyboard focus on **Blank note**, so Enter creates
a note straight away without tabbing to find it; ↑/↓ move between Blank
note and the template rows, and the focused row carries the same left-edge
bar the Notes list rows use. The footer's "enter create note" appears only
while one of those rows genuinely has focus — move to "‹ Notes" and it
drops, because Enter there goes back rather than creating anything.

### Add from files and lasting sync

**Add from files…** first asks what relationship you want. Until you choose,
the header names neither relationship — it reads "Add from files" and its next
action is to pick one:

- **Import once** copies supported files into Database Notes and ends after its
  reviewed receipt. Later changes to the originals are not tracked.
- **Keep a folder synced** creates a lasting local relationship. Choose the
  folder, direction, and local Library destination, then choose **Check
  changes**. Checking is mutation-free. Review safe actions, attention items,
  skips, filesystem effects, and deletion-like effects before **Activate
  reviewed root** is enabled.

Both folder pickers remember where you were. Each reopens at the directory it
last picked in *that* flow, so Import once and Keep a folder synced never move
each other's starting point, and neither borrows the Library ingest browser's.
The first use of either — or a remembered folder that has since been moved or
deleted — opens at your home directory instead. **Folder files** keeps its own
separate memory, see [File notes](file-notes.md).

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
A folder is exclusive; it cannot be combined with selected files, so a folder
selection offers no **Add another file**. Either way, **Change selection**
reopens the picker and replaces what you chose, and **Clear** drops the
selection without leaving Import once. Selected files also need an
existing-or-new destination path such as `Research / Interviews`. The
destination is only a proposal during checking; no folder or note is created
yet.

The picker — from **Add another file**, from a folder choice, and from
**Change selection** — reopens at the directory Import once last picked, or at
your home directory the first time. Its **Folder path** field can be typed into
directly: press **Enter** to browse into the typed path, or click **Select
folder** to use it immediately without pressing Enter first — either way,
whatever the field currently holds is what gets picked, not merely the
directory being browsed. An invalid path shows an inline reason and leaves the
dialog open. Once a folder is picked, the confirmation line shows its full path
(elided in the middle for long paths, keeping the folder name itself visible),
not just its name.

Choose **Check selection** to build a read-only review. Each source is one
line — path · what will happen · where it lands — with its **Skip** and
**Create new** controls beside the path. Rows are grouped by outcome (**New**,
**Unchanged repeat**, **Changed repeat**, **Uncertain match**, **Unsupported**,
**Skipped**, **Empty**, **Failed**) and each group header carries **Skip all**
and, where the group can create notes, **Create all** for the rows it counts.
An empty or whitespace-only file is reported as "Empty file — nothing to
import." and an application configuration file (a JSON or YAML document with no
note body) as "Not a note file (app configuration)." A well-formed document
that simply holds no note — an empty JSON array, a CSV with only headers —
reads "This source does not contain any notes." None of these is a failure. A
document that mixes note records with other records is still a failure ("This
source could not be parsed as notes."), so a damaged export is never presented
as harmless configuration. A structured source states how many notes it will
create, so a two-row CSV reads "create 2 new notes". You can still skip an
item, create a new note, or, when an existing match is authorized, update its
content and/or add its folder placement; **Confirm this match**, **Replace note
content** and **Add folder placement** sit on their own line under the row, so
they stay reachable in a narrow pane. Uncertain matches must be confirmed. If
the imported top-level folder already exists, choose whether to use it, create
a unique sibling, or enter another name.

Only **Import selected items** approves and executes the exact choices shown.
Progress remains visible and **Cancel import** stops cooperatively after the
current item; completed items are not rolled back. The receipt states what
happened in plain words — "Import finished · 61 notes created · 11 files
skipped" — and a **Skipped (N)** disclosure lists each skipped path with its
reason. A file the app skipped for you — an unchanged repeat, an empty or
unsupported source — keeps its own reason there; only a row you set to Skip
yourself reads "Skipped by you." A partial receipt states what finished. Retryable failures show
**Retry N failures**; a cancelled batch with unfinished items shows **Retry
unfinished items**. **Back to Notes** may hide a running import without
stopping it; the list then offers **View import** or **Continue import** until
it settles. **Last import** reopens the same-session receipt afterward.

#### Obsidian vaults

If the folder you chose holds an `.obsidian/` directory, the review shows an
**Obsidian vault** toggle, on by default, and one line saying what it does.

**Not on Windows.** Vault detection runs in the POSIX discovery pass only, so
on Windows a vault imports as an ordinary folder: no toggle appears, the vault's
own folders are walked, frontmatter stays in the body and wikilinks stay as
text. Tracked as task-32178.

With it on:

- `.obsidian/`, `.trash/` and `Templates/` are listed under **Skipped** as one
  row each, naming the vault reason ("Obsidian configuration — skipped…"),
  rather than one row per file inside them.
- YAML frontmatter is read: `title` becomes the note title, and `tags` and
  `aliases` become keywords (there is no separate alias field, and keeping them
  as keywords is what makes the note findable by its alternate names). The
  frontmatter block is removed from the note body — unless it is the whole file,
  in which case the note keeps it and still takes its title and keywords from it.
- `[[wikilinks]]` and `[[link|alias]]` whose target is imported in the same
  batch become note links; a link to anything else stays as plain text, and a
  `[[link]]` written inside a code block or backticks is left alone.

Turn the toggle off to import the vault exactly as any other folder — every
directory walked, frontmatter left in the body, links left as text. The config
files inside are then listed one by one, still as **Skipped** ("Not a note file
(app configuration).") rather than as failures. One rule
applies either way: a title that is only a template placeholder, such as a
`# {{date:YYYY-MM-DD}}` heading, is never used, and the file name is used
instead. Toggling re-runs the read-only check, so nothing is written either way
and Import once never modifies the vault on disk; it does rebuild the review,
so any per-item Skip/Create choices you had already made are reset.

Review rows for new notes state what will be created — the resulting title, its
keywords, and how many links it carries — before you approve anything.

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
   destination. A folder already supplies its proposed hierarchy. Picked the
   wrong source? Use **Change selection** or **Clear**.
3. Click **Check selection** and review classifications, actions, matches, and
   any top-level folder collision. For an Obsidian vault, check the
   **Obsidian vault** toggle first — see "Obsidian vaults" above.
4. Click **Import selected items**. You can cancel cooperatively, retry work
   identified by the receipt, or return to Notes and reopen **Last import**.

### Set up lasting folder sync

1. Close any older Chatbook version using this profile, then restart the
   cutover release.
2. In the notes list, click **Add from files…** and choose **Keep a folder
   synced**.
3. Click **Choose folder…**; type into the **Folder path** field and either
   press Enter (browses into it) or click **Select folder** (uses it right
   away). Choose a direction and local destination. Server sync remains
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
1. Open the note, choose **Info**, and click **Export Markdown**.
2. Choose a destination in the "Export Note as Markdown" dialog — the
   toast confirms "Note exported successfully to \<name\>".

### Undo a deleted note

1. Confirm **Delete** in the note editor.
2. In the Notes list, find the "✓ deleted · \<title\>" receipt.
3. Click **Undo** to restore the note, or **Dismiss** to leave it deleted.

### Recently deleted

The receipt is the immediate way back, but it is not the only one. Under the
folder tree, **Recently deleted (N)** counts the notes that have been deleted
and not yet restored; it is absent while that count is zero.

1. Click **Recently deleted (N)** to open the list, newest deletion first.
   Each row shows the note's title and how long ago it went.
2. Click that row's **Restore** — or press **r** on the focused row — to put
   the note back in its folder (or Unfiled). The rail count and the tree row
   return exactly as **Undo** returns them; it is the same restore.
3. Press **Escape** (or **‹ Notes**) to go back to the list.

The view holds the 20 most recent deletions and says so when there are more.
Nothing here deletes anything for good: the Trash offers **Restore** and no
permanent delete.

## Keyboard & commands

| Key | Action |
|---|---|
| **Ctrl+N** | New note. Works on the Library landing (no row selected yet) as well as inside the Notes workflow — the landing's bare **n** still works too, but the footer advertises Ctrl+N in both places now. |
| **/** | Focus the note filter ("find note"), without typing a literal "/" into it. Once the filter has focus, "/" is an ordinary typeable character rather than an accelerator — a second "/" adds a literal slash, since a filter can legitimately target a folder-style path such as "Work/Q3". |
| **Escape** | Focus the rail (in **Recently deleted**, go back to the list) |
| **r** (in **Recently deleted**) | Restore the focused row |
| Enter (in "Filter notes… (Enter)") | Apply the filter |
| ↑ / ↓ (New note view) | Move between **Blank note** and the template rows |
| Enter (New note view) | Create from the focused row |

The footer advertises these as `ctrl+n new note | / find note | esc focus
rail`. Notes does not register **Ctrl+S** and does not replace it with
another save shortcut. Use the visible
Database **Save** button for an immediate save; Folder Files saves
automatically. Global navigation keys live in the [guide index](../index.md).

## Related settings & docs

- Lasting root paths, bindings, operations, and recovery state live in the
  private device sync store, not ordinary `config.toml` settings. The one
  exception is where each folder picker reopens, immediately below — that is
  an ordinary `config.toml` setting.
- **config.toml `[library.notes_import] last_directory`** and
  **`[library.notes_sync] last_directory`** — the directory the **Import
  once** and **Keep a folder synced** pickers respectively reopen at; each is
  written whenever a selection is made in that flow. Separate keys, so neither
  flow moves the other's starting point; **Folder files** has its own
  `[file_notes] browse`. A key naming a folder that no longer exists is
  ignored and the picker opens at your home directory.
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

## Verified against

*Verified against c2cbb8081 — 2026-08-04 (PR-T1: "Use in Console"
delivering the note's real content on send is covered by capture
round-trip tests, task-2374).*

*Verified against dev @ 6b38a13b8 — 2026-08-07 (task-2858 Task 3, LIB-14:
"Blank note" no longer leaves a stray "Untitled" row if abandoned
untouched, and the title shows an "Untitled" placeholder instead of
literal editable text).*

*Re-stamped against dev @ 4acb17a0b — 2026-08-07 (TASK-2857: "Export…"
now opens the "Export bundle (.zip)" canvas, not "Export chatbook").*

*Re-verified against dev @ 71f15ff76f — 2026-08-09 (task-3315): the LIB-14
untouched-blank discard and the empty-title → "Untitled" save fallback
described above had regressed on dev (the notes-adaptive session-coordinator
refactor read the seeded snapshot title instead of the presented-empty
editor, and dropped the save-seam fallback); both are restored, and Esc from
the note editor no longer dead-ends (it routes through the same guarded
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

*Verified on codex/notes-delete-undo-receipt — 2026-08-11 (TASK-15100:
confirmed Database Note deletion now leaves a named inline Undo/Dismiss
receipt; Undo restores the exact soft-deleted row and Notes rail count through
the version-checked service seam.)*

*Verified against dev @ 1bda754fa1 — 2026-08-21 (TASK-19026): wide Database
browsing retains the Library rail; database editing and Files use one
focused workbench with a guarded `‹ Library / Notes` return; exact browse
identity and independent scroll positions survive return and compact/wide
breakpoint crossings.*

*Verified against dev @ 38b2704b36 — 2026-08-30 (TASK-24309): Agent Lessons
folder ownership, exact marker discovery, evidence template, foreground
approval, subagent draft boundary, credential refusal, and untrusted-retrieval
contract added. See
[ADR-105](../../../backlog/decisions/105-portable-notes-organization-and-agent-lessons.md)
and [ADR-106](../../../backlog/decisions/106-human-reviewed-agent-lesson-promotion.md).*

*Verified against fix/library-uat-31796-31797 — 2026-09-06 (task-31796: the
list row no longer keeps the pre-rename "Untitled" title until a filter
re-query. Superseded in mechanism, not outcome, by task-32062/PR #2531
below: rather than repainting the instant a rename saves, a refresh that
lands while the title field holds focus is skipped and only replays on the
next refresh that finds focus outside the field — in practice, on returning
to the list. See the Notes list description above.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (task-32063: the
work pane no longer restates the list pane's "Library notes · Library
database …" authority sentence, a "Next:" clause only appears when it names a
control on screen, and the Add-from-files header is one sentence ("Add files to
Library notes.") instead of a run-on stacked under three more. task-32061:
Escape from the editor leaves the Notes list at the visibility it had.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (task-32062: a
Notes refresh that arrives while the title or body has focus leaves the editor
alone, and a snapshot that is a keystroke behind no longer rewrites the
focused field — measured live, a title and body typed within ~0.4 s used to be
stored as one scrambled title with an empty body. task-32061 re-checked on a
fresh profile: the list pane survives the first note's Escape.)*

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (review of
PR #2531: the keyword boxes get the same protection as the title and body — a
refresh landing while you type keywords no longer rebuilds the editor or
rewrites the box from an older snapshot. Fix-round regression caught by the
scoped re-review, task-32062: the deferred-blur replay this same PR first
shipped could yank focus back out of a field the reader had deliberately
moved to, so the mechanism changed from "replay on blur" to "skip and let
the next out-of-field refresh paint" -- the behavior the rename-propagation
stamp above now describes.)*

*Verified against fix/library-crit8-docs — 2026-09-08 (task-32073,
docs-vs-live pass from critique #8; live at 235x52 on a seeded profile):
the **New** and **New folder** toolbar actions, the folder-selected
**Rename / Move / Remove** set, the note-selected **Add to folder / Move
note / Remove placement** set, and the footer's **ctrl+n new note** and
**/ find note** keys all ship and were undocumented here. The
whitespace-title rule was corrected: the abandon-discard check tests
`title.strip()`, so a spaces-only title with an empty body does NOT keep
the note, contrary to the "typing anything keeps it" claim.)*

*Verified against fix/library-crit8-keyboard — 2026-09-08 (task-32052: the
New note view focuses **Blank note** on entry, ↑/↓ walk it and the template
rows with a visible cursor, the footer's "enter create note" follows the
focused control, and Tab no longer leaves the Library screen for the
navigation bar. Pinned in `Tests/UI/test_library_crit8_keyboard.py`.)*

*Verified against fix/library-crit8-notes-loader — 2026-09-08 (task-32050:
opening a stored note no longer stays on "Loading note…" for ever, and a
stuck load now reaches a failed state with Retry).*

*Verified against fix/library-notes-file-notes — 2026-09-09 (task-32136:
Folder files is a mode of Notes — once a folder is linked, the Library rail
and the "Library notes | Folder files" strip stay visible inside it at wide
sizes, and the strip, not the back cue, is the way back. The empty state
shown before any folder is linked is full-width without the rail; see
[File notes](file-notes.md).)*

*Verified against fix/library-notes-pickers — 2026-09-09 (task-32122: the
Import once and Keep-synced folder pickers used to commit the directory
being browsed and silently ignore a typed-but-unsubmitted path when
**Select folder** was pressed. Both now resolve the **Folder path** field
first — Enter still browses into it, and Select/Select folder use it
immediately, with an inline error and the dialog left open for an invalid
path. The confirmation line shows the full picked path, not just its
basename. Pinned in `Tests/UI/test_file_open_select_folder.py`,
`Tests/UI/test_select_directory_typed_path.py`,
`Tests/UI/test_enhanced_select_directory.py`.)*

*Verified against fix/library-notes-onboarding — 2026-09-09 (task-32126:
the empty-state line previously never rendered once the seeded
Agent_Lessons folder existed, because the tree projection had a row and
the "no rows" check never fired; the guide's copy also disagreed with the
code's. Fixed to render "No notes yet. Create your first note." above the
tree whenever the library holds zero notes, matching the code copy
exactly, and to gloss the Agent_Lessons folder row while it does. Pinned
in `Tests/Widgets/Library/test_library_notes_canvas.py`.)*

*Verified against fix/library-notes-import-ux — 2026-09-09 (task-32125: the
chooser shows **Import once** and **Keep a folder synced** together, each under
its own description, and the header no longer says "Lasting sync" before you
have chosen. task-32130: empty files and application config files are reported
as empty/skipped instead of failed, the receipt discloses **Skipped (N)** with
every path and reason, and the completion line counts what happened.
task-32134: **Change selection** and **Clear** ship beside the selection.
task-32135: review rows are one line each, grouped, with per-group **Skip all**
/ **Create all**.)*

*Verified against fix/library-notes-list — 2026-09-09 (task-32127: the list
no longer sits at 38 columns beside an empty work area, and its action
groups stack rather than clip in a narrow pane; task-32137: rows
carry an age, and same-folder duplicate titles name their folder;
task-32128: the tree's title order is the database's, so Sort is not
offered there — superseded by task-32172 below; task-32123: the delete
receipt's Undo/Dismiss actions are no longer composed off the pane;
task-32124: Undo returns the row to the folder tree, not only the count.)*

*Verified against fix/library-notes-editor-keys — 2026-09-09 (task-32131: `/`
no longer types itself into the filter it focuses; a same-round controller
ruling then made a second `/` — once the filter already has focus — an
ordinary typeable character rather than an accelerator, since a filter can
legitimately target a folder-style path such as "Work/Q3".
task-32132: Delete's confirmation now renders in place — Info stays open, Tab
is trapped between Cancel and Delete, and the footer names the focused
button. task-32133: a refused Escape now notifies "Can't leave yet — fix the
title or press Discard new note.", and a fresh blank note reads "Draft — not
saved yet" instead of "Saved" until the first save lands. task-32138: ctrl+n
now works on the Library landing too (the footer advertises it there instead
of a bare `n`). task-32139: Edit, Preview, and Info share one back-cue
wording, sized by terminal width. task-32142: Preview shows the title above
the body, Info shows "Saved" once and an absolute timestamp beside each
relative age, and a delete receipt no longer survives into Add from files or
Folder files.)*

*Verified against fix/library-notes-obsidian — 2026-09-09 (task-32129: Import
once detects an Obsidian vault, skips `.obsidian/`, `.trash/` and `Templates/`
with reasons, reads frontmatter titles and tags, and links wikilinks resolved
within the batch. Merged with the import-ux wave above: the Obsidian toggle
lives in that wave's single review options slot, and an Obsidian skip is
listed in the receipt's **Skipped (N)** disclosure with its own reason.)*

*Verified against fix/library-notes-docs — 2026-09-09 (task-32141: guide
sweep after the Notes critique wave; 13 claims verified, 5 corrected).*

*Verified against fix/library-notes-import-ux — 2026-09-09 (review of
task-32130 and task-32135: a note-free structured document is skipped rather
than failed while a mixed one stays a failure, an automatic skip keeps its own
reason on the receipt instead of "Skipped by you.", the receipt keeps its
skipped paths after another selection starts, and the follow-on review choices
moved onto their own line so nothing is clipped out of reach.)*

*Verified against fix/library-notes-editor-keys — 2026-09-09 (PR #2547
review round, task-32132/task-32133: "‹ Notes" / "‹ Back to list" is now
disabled, not just the other Info actions, while a delete confirmation is
open — pressing it used to silently displace the prompt into Edit; and a
keyword typed only through Info's Properties field on a fresh blank note
now clears the "Draft — not saved yet" status once it autosaves, matching
the main keywords field.)*

*Verified against fix/library-notes-docs — 2026-09-09 (PR #2549 review, at
the re-merged wave: an unterminated ``` or ~~~ fence now keeps the rest of a
note as code, so a `[[link]]` after it is neither recorded nor rewritten;
Obsidian vault detection is stated as POSIX-only, since the Windows
discovery adapter never reports a vault (task-32178).)*

*Verified against fix/library-notes-i-trash — 2026-09-09 (task-32144: a
"Recently deleted (N)" row under the folder tree opens a Trash view of the
soft-deleted notes, newest first, with a per-row Restore and `r` on the
focused row. Restore commits through the same seam the delete receipt's Undo
uses, so the row returns to its folder and the rail count moves identically.
The view offers no permanent delete.)*

*Verified against fix/library-notes-i-backlinks — 2026-09-09 (task-32145:
Info → Properties now lists "Linked from (N)" — the notes whose bodies carry
this note's `note://` link — and each entry opens that note. Checked live on a
fresh profile after importing the review vault: "Zettelkasten — overview" read
"Linked from (2)" and listed both linking notes, activating one opened it, and
an unlinked note read "Linked from (0) — no notes link here yet".)*

*Verified against fix/library-notes-r-pickers — 2026-09-09 (task-32174:
**Import once**'s and **Keep a folder synced**'s folder pickers each now
reopen at the directory they were last successfully browsed in, falling
back to home when nothing is recorded yet, or when the recorded value no
longer names a real folder — independently of each other and of the Library
ingest browser's own last-used directory. Stored in
`config.toml` as `[library.notes_import] last_directory` and
`[library.notes_sync] last_directory`; Folder files' own picker does the
same for its `[file_notes] browse` setting, see
[File notes](file-notes.md).)*

*Verified against fix/library-notes-r-list — 2026-09-09 (task-32172: the
placement order is a repository parameter now — folder paging AND the
deep-link locator's page arithmetic both take it — so Sort is offered on the
folder tree again and Newest/Oldest really re-page it; it stays disabled,
with its reason, while a filter window is showing. The folder tree's default
order changes from title to newest-first with this, which is what the Sort
control has always claimed.)*
