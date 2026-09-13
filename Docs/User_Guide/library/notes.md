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

Library notes uses three side-by-side roles when there is room: Library
navigation, the Notes list, and the note you are working on. The Library
navigation and Notes list each have their own slim collapse grip. Collapsing
one does not collapse the other, and each grip remembers its own choice.

Wide Library notes keeps Library navigation beside the list while you scan:

```text
+----------------------+-----------------------------------------------+
| Library              | Library notes | Files                         |
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
selected Library note, switch between Library notes and Folder files,
change the linked Folder files root, leave Notes, or close the open Folder
files file. Folder files' compact **Back to navigator** action is not a reset.

On a wide terminal the strip above the canvas keeps both source switches —
**Library notes | Folder files** — whether Library navigation is open or
closed, and the in-canvas **‹ Notes** control is the way back to the list.
The **‹ Library / Notes** cue this page used to describe here belongs to
compact terminals, where the strip collapses to it — see the Source strip
bullet below. (Was "When Library navigation is closed, one stable cue names
the return destination" — superseded by task-32270 below.)

```text
+-----------------------------------------------------------------------+
| ‹ Library / Notes            (compact terminals only)                 |
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
  also names its folder — "Reading list · Unfiled · 2h". When they share
  the age as well, the row adds a third part that is not shared: the time
  of day it was last changed ("Reading list · Unfiled · 2m · 09:14"), or,
  for two notes written inside the same minute, a short id
  ("Reading list · Unfiled · 2m · #0f3a"). Only the rows that would
  otherwise be identical carry it. While no note is
  open the list takes the width the empty work area would otherwise waste,
  so long titles are not truncated on a wide terminal; opening a note hands
  that width back. The same rule holds below 64 columns, where there is no
  room for two panes at all: with nothing open the list is the whole stage
  instead of sharing it with an empty work area, and opening a note gives
  the stage to the note. Its own grip collapses or restores the list without
  changing the Folder files tree choice. Renaming a note does not repaint its
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
  frequent actions remain in the header. On wide terminals the editor's own
  `‹ Notes` control returns to the exact prior list row, scope, and
  scroll positions; on compact terminals use `‹ Back to list`. While you are
  typing, the title, body and keyword fields are each their own authority: a
  background refresh never rewrites the field under your hands, and it never
  moves your place in the Items list beside it. Tab out of any of them —
  title, body or keywords — takes effect before the next keystroke, so
  typing straight through a Tab puts the rest where you meant it, and it
  lands on the next control **inside the editor** rather than in the
  browse chrome above it (see "Editor keys" below).
- **New note view** — opens from the rail's "New note": a "Blank note"
  button and a "From a template…" row that unfolds the eight templates.
  (**Ctrl+N** / **n** skip this view: they make the blank note itself.)
- **Add from files…** — asks whether this is an **Import once** or a lasting
  **Keep a folder synced** relationship before reading a source. Both buttons
  sit together, each directly under its own description; the bar below holds
  only **Back to Notes**.
- **Manage sync folders** — appears when roots or migration candidates exist;
  it shows text-explicit status and the valid action for each root.

### Library notes vs. Folder files vs. lasting sync

Three different notes worlds meet here, and each surface now says so in
place: the strip above the canvas switches between two of them.
**Library notes** (this page) keeps notes inside the Library's own database.
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
logical organization of Library notes: keywords, keyword collections, Library
notes folders, and their memberships. These six organization types enroll as
one capability; Chatbook will not synchronize only part of the group.

The first run may pause for an adoption review when a local and server object
have the same visible name or path but different identities. Open **Settings**,
find Manual Sync, and choose whether to merge, rename the local object, or keep
it local. Until review and initial inventory finish, organization stays local
and publication is blocked. Interrupted enrollment and sync can be retried from
the same Manual Sync surface; committed checkpoints and pending changes are
resumed rather than rebuilt.

Deleting a Library notes folder synchronizes only that explicit deletion. Its
descendants and memberships stay dormant and become effective again after the
folder is restored. If a note belongs to a folder through both manual and
source-managed placement, removing one placement does not remove the portable
membership while the other remains effective.

This does not synchronize filesystem paths or grant filesystem access. Folder
files and lasting folder sync remain device-private authorities. For the
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
- `folder` resolves an exact relative Library notes path. It is not a local
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
user-owned Library notes folder: you may rename, move, or delete it, and
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

Library notes are presented as a folder tree rather than one flattened
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
- Opening a note reveals where it lives: while the note loads, the status line
  shows **Locating note…** and the tree expands the folders on the way to it
  and marks its row. Open a second note before the first reveal lands and only
  the older reveal is dropped — the note you just chose still opens.

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
| "New" | Opens the **New note** view — the same destination as the rail's **New note** row: **Blank note**, or **From a template…**. (**Ctrl+N** skips the view and makes the blank note itself.) Disabled while another notes operation is running. |
| "New folder" | Creates a folder in the tree beneath the toolbar. Disabled, with the reason in its tooltip, when the selected folder is sync-managed ("This folder is managed by sync; change its sync root instead.") or its branch is stale ("This branch may be out of date; retry it before changing it."). |
| Folder selected: "Rename" / "Move" / "Remove" | Act on the selected folder. Same two disabled reasons as **New folder**. |
| Note selected: "Add to folder" / "Move note" / "Remove placement" | File the selected note into a folder, move its placement, or take it out again. A sync-managed placement is refused with "This placement is managed by sync; change its sync root instead."; a note sitting in the automatic **Unfiled** group cannot have its placement removed ("Unfiled is shown automatically; move the note into a folder."). |
| "Restore folder" | Appears after a folder removal, to put it back. |
| "Sort: Newest" | Opens a one-row strip of Newest / Oldest / Title (✓ on the active one) in place of the action row; pick one directly, or press Escape to cancel. **Newest is the default.** The value is the order the folder tree is paged in, so choosing a new one reloads the tree (open folders included). Disabled while a filter is showing, with "Filter results keep their own order. Clear the filter to sort." |
| "Add from files…" | Choose **Import once** or **Keep a folder synced** before selecting a source. |
| "Manage sync folders" | Appears only when roots or paused migration candidates exist; opens root status and contextual controls. |
| "Last import" | Reopens the latest import receipt from this app session after you return to the Notes list. |
| "Export…" | Opens the "Export bundle (.zip)" canvas scoped to notes — bundle notes into a .zip. |
| "Select" / "Done" | Toggles select mode: rows grow ☑/☐ checkboxes, and a row appears with "N selected", "Select all N shown", "Clear", and "Export selected". The count is also repeated on its own line below the row, and the two always read the same number. "Export…" hides while selecting. On a compact terminal the row shortens to "Done", "All N", "Clear" and "Export" and drops its own copy of the count, keeping the line below it — all four actions stay on the pane. |

With no notes at all, the list reads "No notes yet. Create your first note."
above the tree — even when the seeded **Agent_Lessons** folder (see "Reuse
solutions with Agent Lessons" above) is the only row showing, so a
first-time user is never left staring at one unexplained folder with no
other cue. While the library holds zero notes, that folder row itself also
carries a one-line gloss: "Agent_Lessons — where Console agents file
reusable lessons (empty)".

On a terminal narrower than 64 columns the list pane is narrow enough that
two things used to be cut off mid-word: the status line lost its last word
("…or add from", without "files."), and the toolbar's third action painted
as "Sel". At that width — and only there; at 64 columns and up the line
keeps its full wording — the status line drops its "Library notes ·"
prefix, since the source strip above it already says which notes these
are, and the toolbar moves the action that does not fit onto a row of its
own. Nothing is ever painted as half a word.

### Edit, Preview, and Info

| Control | What it does |
|---|---|
| "‹ Notes" / "‹ Back to list" | Returns to the list (your text is already saved — see autosave below). One wording across Edit, Preview, and Info: "‹ Notes" at wide sizes, "‹ Back to list" on a compact terminal. |
| **Edit** | Shows the editable title and body. This is the default view when you open a note. |
| **Preview** | Shows the note's title above the body, rendered as Markdown, without replacing your draft. It takes the whole work pane, and it takes keyboard focus when you open it, so `pgup`/`pgdn` page the rendered note straight away — no click inside the box first. An Obsidian callout (`> [!note] Title`) renders as a quoted block headed "Note: Title" rather than printing its `[!note]` marker. The status line does not offer to keep editing while Preview is showing; it names **Edit** instead. |
| **Info** | Shows Properties (including comma-separated keywords, note dates/version, and **Linked from**), Reuse & Export, and Danger sections. |
| **Linked from (N)** (Info → Properties) | Lists the notes whose bodies link to this one, newest import or not — the `[[target\|title]](note://…)` links Import once writes for an Obsidian vault's `[[wikilinks]]` (see "Obsidian vaults"). Click an entry to open that note. While the lookup runs the line reads "Linked from — checking…", and "Linked from — couldn't check" if it failed, so a count is only claimed once the answer is in. When nothing points here the line reads "Linked from (0) — no notes link here yet". The list is capped at 50 entries; past that the count reads "50+". Links you type by hand in the body count too, as long as they use the same `note://` form. The answer is looked up rather than searched for: each note records the links its body carries when it is saved or imported, so the lookup costs what a note's own inbound links cost rather than growing with the size of your vault. An existing library picks its links up the first time this version opens it — nothing to re-import. |
| Status line | Shows the autosave state: "Saved", "Saving…", "Unsaved changes", "Conflict — …", "Save failed — …", or "Unavailable — …". It does not carry a word count. Created/Modified/version details are under Info → Properties, each with an absolute local timestamp beside its relative age and the word count (e.g. "Created 2026-09-08 21:14 · 3m ago · Modified … · v1 · 6 words"). "Saved" appears once per view, not repeated in Info. |
| Chrome strip | One row directly under the body, right-aligned: "N words · L:C" — the words in the note and the caret's line and column, both counted from 1. It follows your typing and your arrow keys with no save and no reload. Nothing on it is a control; Tab never stops there. It appears while **Edit** is the open view on a terminal 80 columns or wider; **Preview** and **Info** have no caret, and a narrower terminal gives the row back to the body. The strip does not repeat the save state — that stays on the status line above the mode controls. The editor's Created/Modified/version line lives in one place, Info → Properties; the editor pane no longer builds a second, never-shown copy of it. |
| **Save** | Saves immediately, without waiting for autosave. It remains visible beside the mode controls. |
| **Use in Console** | Hands the note to the Console as staged context, with the suggested prompt "Use this note as context and help me work with it." It remains visible beside **Save**. |
| **Copy** (Info) | Copies the note to the clipboard as Markdown — "Note copied to clipboard as markdown!" |
| **Export Markdown** / **Export text** (Info) | Saves the note to a file you pick; success shows "Note exported successfully to \<name\>". |
| **Delete** (Info → Danger) | Asks inline, in place — Info stays open and the prompt renders inside the Info box, on the row directly under the Delete button that raised it: "Delete this note? Undo will be available in the Notes list." Tab / Shift+Tab cycle only between **Cancel** and **Delete** while it is open, and the footer names whichever one is focused ("enter cancel" or "enter delete"). Every other Info action — including "‹ Notes" / "‹ Back to list" — is disabled until you choose Cancel or Delete. Cancelling puts focus back on Delete and leaves Info scrolled exactly where it was. A successful delete returns to the list with a named "✓ deleted · …" receipt offering **Undo** and **Dismiss**. |

Opening a note shows "Loading note…" only while the note is being read. If a
read takes longer than about three seconds the editor stops waiting and shows
"Unable to load note — timed out after 3 s. Press Retry." with a **Retry**
button; "‹ Notes" / "‹ Back to list" (the same compact-vs-wide wording as
Edit/Preview/Info) takes you back to the list, and opening another note still
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
shortcut. Use the visible **Save** button when you want an immediate Library
notes save; normal Tab navigation and **F6** can reach it. Autosave continues to
handle ordinary typing.

#### Editor keys

| Key | What it does |
|---|---|
| **Tab** / **Shift+Tab** | Move between the editor's own controls and stay there: "‹ Notes", Edit, Preview, Info, Save, Use in Console, Title, Body, and back round to "‹ Notes". Tab out of the body no longer wraps round to the "Library notes / Folder files" switch above the pane, where typed characters went nowhere. The move lands before the next keystroke, so typing straight through a Tab puts the rest where you meant it. |
| **F6** / **Shift+F6** | Leave the editor for the Notes list or the Library rail. This is the way out of the editor's Tab cycle; **Escape** is the other (it returns to the list). |
| **Ctrl+End** / **Ctrl+Home** | Jump the caret to the end or the start of the note body. `End` and `Home` still move within the current line. |
| **Escape** | Returns to the list — one press, from Edit, Preview, or Info. From Info it goes back to the editor first. |

Arriving in the **Title** by keyboard puts the caret at the end of the
existing title; it does not select the title, so one keystroke can no longer
replace it. Select the text yourself (Shift+Home, or drag) when you do want
to overwrite. The keyword boxes behave the same way.

The footer names whichever editor control has focus as an "enter …" chip, so
focus is never unaccounted for: Tab onto "‹ Notes" and the footer reads
"enter back to list", onto Save and it reads "enter save note". While the
body or a field has focus there is no enter chip, because Enter types.

When the note body has keyboard focus, only its boundary becomes more
prominent. The body background and editor size stay unchanged, so focusing
the editor does not flash or fill the writing surface. Small fields such as
Title and Keywords still use their usual filled focus treatment.

The editor's `‹ Notes` control, the compact `‹ Library / Notes` cue and
Escape all use the same guarded return. A dirty save, sync, conflict, reload confirmation,
or running mutation can therefore keep the focused task open until it is safe
to leave. A successful return restores the Library notes / Folder files source, filter,
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
the selection to the restored row. If that folder is collapsed, Undo opens
it, so the row you were promised back is one you can see; a folder whose
contents fail to reload still opens, with its own retry row inside, rather
than staying shut over a note that is already restored. **Dismiss** removes only the receipt; the
note remains deleted. *(This page previously said Notes expose no separate
Trash browser, so the receipt was the only in-Library recovery action —
superseded by task-32144: see "Recently deleted" below, which recovers a note
whose receipt was dismissed.)*

*Verified against fix/library-notes-list — 2026-09-09 (task-32123: the
receipt's actions are no longer composed off the pane; task-32124: Undo
returns the row to the folder tree, not only the count).*

*Verified against fix/library-notes-w3-list-tree — 2026-09-11 (task-32255:
Undo opens the restored note's folder, including when a branch reload
fails; task-32254: a third part tells apart two rows that share title,
folder and age; task-32272: one selection count, not two that disagree).*

*Verified against fix/library-notes-w3-editor-keys — 2026-09-11 at 235x52
and 100x30 (task-32246: Tab out of the body stays in the editor and the
footer names where focus is; task-32247: Ctrl+End reaches the end of a
37 KB note and is on the footer; task-32253: Shift+Tab into the Title no
longer selects it; task-32268: the delete prompt renders inside the Info
box under Delete).*

### New note view

**Ctrl+N** (and the bare **n**) do not open this view at all: they create the
blank note and drop you in its editor, because that is the answer nearly
every time. The view below is the way to a template, and it opens from the
rail's **Create ▸ New note** row or the Notes list's **New** button.

"Blank note" drops you straight into the editor with an empty title (shown
as an "Untitled" placeholder — just start typing) and an empty body. The note
itself already exists at that point — it is in the list and in the rail's
count — so its status reads "Empty note — type to keep it" rather than
"Saved": what is not yet safe is not the note, but the fact that you have
written nothing in it. If you leave again via "‹ Back to list" without typing
anything, the blank note is quietly discarded rather than left behind as a
stray "Untitled" row.
Pressing "Save" keeps it, and so does typing anything **that is not only
whitespace** — a title of nothing but spaces, with an empty body and no
keywords, still counts as blank and is discarded on the way out. That includes naming it "Untitled"
yourself: once you have touched the title field the note is yours, and it
is kept even with an empty body. **From a template…** unfolds eight template
rows that pre-fill title, body, and keywords instead; each row shows the
template name with the title the note will get. Available templates:
Brainstorming session, Bug report, Code review, Daily journal entry, Meeting
notes, Project planning, Research notes, Todo list. The rows stay folded
until you ask for them, and fold again the next time you open the view.

Opening this view parks keyboard focus on **Blank note**, so Enter creates
a note straight away without tabbing to find it; ↑/↓ move between Blank
note, **From a template…** and (once it is open) the template rows, and the
focused row carries the same left-edge bar the Notes list rows use. The footer's "enter create note" appears only
while one of those rows genuinely has focus — move to "‹ Notes" / "‹ Back to
list" (the same compact-vs-wide wording as Edit/Preview/Info) and it drops,
because Enter there goes back rather than creating anything.

### Add from files and lasting sync

**Add from files…** first asks what relationship you want. Until you choose,
the header names neither relationship — it reads "Add files to Library notes."
over "Choose how files should relate to Library notes.", and its next action is
to pick one. (Was "reads **Add from files**" — superseded by task-32271 below:
that was the toolbar button's label, never the heading's.)

- **Import once** copies supported files into Library notes and ends after its
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
deleted — falls through to the folder `[notes] sync_directory` names, and only
then to your home directory. **Folder files** keeps its own separate memory,
see [File notes](file-notes.md).

**What a folder has to be before it can be checked.** The folder itself must be
a real folder (not a link to one) on a local disk, outside Chatbook's own data
directory, not already connected as a sync folder, and not the Folder files
root. Every `.md` file inside it must also pass, and one bad file stops the
whole folder: each must be one you own or share a group with and can write,
UTF-8 text, 10 MB or smaller, an ordinary file with a single name on disk, and
consistent in its line endings — all Unix or all Windows, not a mix, and not
the carriage-return-only style old Mac editors wrote.

**Check changes** refuses anything else and names which rule it was, in the
setup pane's status line, with the next action — for example "That folder is
inside Chatbook's own data directory. Pick a folder outside it, then Check
again", "Another Chatbook window is using that folder", "That folder is
already connected", "Some files there use a mix of line endings. Save them
with one style, then Check again", or "Some files there are larger than 10 MB".
When several files fail for different reasons, the message names the most
common one. A refusal changes nothing: pick a different folder, or fix the
cause, and **Check changes** again in the same session.

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
active root — though today **Resume** does not bring a paused root back at
all, whether or not anything changed while it was paused: its row reads
"✕ Failed · Next: Review changes", **Check changes** answers "Manual check
failed", and **Review** says the folder is still paused. A restart does not
recover it either: the row comes back as "Ⅱ Paused · Next: Resume", Check
changes fails while it is paused, Resume fails the same way, and a file
edited on disk after the restart never syncs — so pause only if you can live
with the root staying paused; nothing in this release resumes it
(task-32519). (Was "so pause only when you can live with a restart" —
superseded by task-32271 below: the restart was asserted, then walked.)
**Retarget** and **Disconnect**
remain visibly disabled with an unavailable-in-this-release reason; no files
or notes change.

### Import once

**Import once** copies supported note files into local Library notes. It is
not the same as **Keep a folder synced**: the import ends after this reviewed
batch, while lasting sync retains a root relationship.

Import once and **Add from files** are whole tasks, not a second reading
pane: while one of them is open the Notes list beside it closes to its grip
and the task takes the pane's width, and the list comes back the moment you
leave. While you are reviewing or running an import, Library navigation
closes too, so the review has the whole canvas. **Check selection** stands
directly under the selection summary it acts on rather than at the pane
floor; the review, import and receipt steps keep their action pinned under
the scrolling list it approves.

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
your home directory the first time. Its **File name** field ("File name or
path") can be typed into directly: press **Enter** to browse into the typed
path, or click **Select folder** to use the folder being browsed. The field
arrives empty and stays empty as you browse — the "Folder path" field that
arrives pre-filled with the directory being browsed, and selects that value on
the click that focuses it, is the Folder files root picker's
([File notes](file-notes.md)); here `Ctrl+A` selects whatever you typed. An
invalid path shows its reason on a row under the field ("The file must exist")
and leaves the dialog open. **Select folder** picks the folder row you last
clicked if you clicked one, otherwise the folder being browsed. (Was "Its
**Folder path** field … The field arrives pre-filled with the directory being
browsed, and the click that puts the cursor in it selects that value" —
superseded by task-32271 below: Import once and Keep a folder synced open the
files-or-one-folder dialog, whose field is "File name" and starts empty; only
Folder files' "Choose File Notes Folder" has the pre-filled "Folder path".)
Once a folder is picked, the confirmation line shows its full path
(elided in the middle for long paths, keeping the folder name itself visible),
not just its name.

Choose **Check selection** to build a read-only review. The review takes the
pane while it is open — the Notes list steps aside and comes back when you
leave. The status line above it states the total ("Review 66 sources before
import.") before you approve anything.

Each source is one line — path · what will happen · where it lands — with its
**Skip** and **Create new** controls beside the path. The path gives way
first if the line is too long (elided in the middle), because the half that
decides anything is the outcome: the resulting title, its keywords and its
link count.

Rows are grouped by outcome (**New**, **Unchanged repeat**, **Changed
repeat**, **Uncertain match**, **Unsupported**, **Skipped**, **Empty**,
**Failed**) and each group header carries **Skip all on this page** and, where
the group can create notes, **Create all on this page** — both act on exactly
the rows that page's heading counts, so a later page keeps its own choices.
A group's header states its whole size ("New (58)"); only when the group is
too big for one page does it read "New (25 of 58 on this page)", so the count
on a page always says which number it means.

A run of interchangeable rows — forty-five archived notes in one folder, all
headed for the same place — collapses to one summary row with a disclosure
("▶ vault/Archive · 45 files · Create all · Create in vault / Archive").
Open it to reach every individual **Skip** / **Create new**; the group's own
bulk actions settle the whole run without opening it. Pages are filled by the
rows they *show*, so a collapsed run costs a page one line and never has a
page break through the middle of it. When there is more than one page,
three lines close the page under its last group — **Previous page**, "Page 1
of 4", **Next page** — and the button at either end says which end you are at
("Previous page unavailable — this is the first page", "Next page unavailable
— this is the last page") rather than merely greying out; a group that
continues on another page says so in its header ("New (25 of 73 on this
page)"). A single-page review has no pager at all.
A `.git` folder is never walked, whatever the source and whatever the Obsidian
toggle says: it is listed once under **Skipped** as "Git repository data —
skipped. Nothing in it becomes a note." A git-backed vault would otherwise
review its own repository internals — several hundred sources that can never
be notes.

A file type that cannot become a note is named rather than dismissed: an
Obsidian canvas reads "Obsidian canvas — not a note.", and an image, document
or media file points at where it does belong ("Image — not a note. Add it in
Library ▸ Media."). Anything else keeps the plain "This file type is not
supported."
An empty or whitespace-only file is reported as "Empty file — nothing to
import." and an application configuration file (a JSON or YAML document with no
note body) as "Not a note file (app configuration)." A well-formed document
that simply holds no note — an empty JSON array, a CSV with only headers —
reads "This source does not contain any notes." None of these is a failure. A
document that mixes note records with other records is still a failure ("This
source could not be parsed as notes."), so a damaged export is never presented
as harmless configuration — and the reason names the record that failed
("Record 2 of 3 has no note content.", or "Row 3 could not be read as a note."
for a CSV), so a 200-note export does not have to be bisected by hand. The
whole file is refused, not partly imported: fix the named record and import
again. A structured source states how many notes it will
create, so a two-row CSV reads "create 2 new notes". You can still skip an
item, create a new note, or, when an existing match is authorized, update its
content and/or add its folder placement; **Confirm this match**, **Replace note
content** and **Add folder placement** sit on their own line under the row, so
they stay reachable in a narrow pane. **Update existing** works on an unchanged
repeat too — it replaces the note's content and leaves its folder placement
alone. The difference between the stored note and the file is shown on the one
row that would write it: choose **Update existing** with **Replace note
content** to see it. A row that says "Content: no change." never carries a
diff, because the two answer different questions — whether this *file* changed
since it was last imported, and whether the *note* now differs from it. Uncertain matches must be confirmed. If
the imported top-level folder already exists, the review opens with the
non-destructive default already chosen — a unique sibling — and says where the
notes will go ("A folder with this name already exists. These notes will go
into Imported (2) instead — choose another option to change that."). Choose
**Use existing folder** or **Use another name** to change it; the name field
starts empty, with no error painted against a name you have not typed.

Only **Import selected items** approves and executes the exact choices shown.
While it is unavailable it carries its reason as its own text ("Import
selected items unavailable — Choose how to handle the folder name
collision."), as **Check selection** does before a source is chosen.

Progress remains visible and **Cancel import** stops cooperatively after the
current item; completed items are not rolled back. Progress counts *planned
changes* — one per note a source creates, one per source otherwise — so a
two-note CSV is two of them; the line says so ("67 of 67 planned changes
complete").

The receipt states what happened once, in plain words — "59 notes created ·
8 files skipped · 54 links resolved" — with the two counts reconciled in the
line beneath it ("67 planned changes from 66 reviewed sources."), and a
**Skipped (N)** disclosure lists each skipped path with its reason. A file the app skipped for you — an unchanged repeat, an empty or
unsupported source — keeps its own reason there; only a row you set to Skip
yourself reads "Skipped by you." A partial receipt states what finished. Retryable failures show
**Retry N failures**; a cancelled batch with unfinished items shows **Retry
unfinished items**. **Back to Notes** may hide a running import without
stopping it; the list then offers **View import** or **Continue import** until
it settles. **Last import** reopens the same-session receipt afterward.

#### Obsidian vaults

If the folder you chose holds an `.obsidian/` directory, the review shows an
**Obsidian vault** toggle, on by default, and one line saying what it does.

Windows works the same way: the Windows discovery adapter detects the vault and
skips its own folders exactly as the POSIX one does.

With it on:

- `.obsidian/`, `.trash/` and `Templates/` are listed under **Skipped** as one
  row each, naming the vault reason ("Obsidian configuration — skipped…"),
  rather than one row per file inside them.
- YAML frontmatter is read: `title` becomes the note title, and `tags` and
  `aliases` become keywords (there is no separate alias field, and keeping them
  as keywords is what makes the note findable by its alternate names). An alias
  is an alternate *name*, not a tag, so it is stored as `alias: <name>` — in
  Info you can tell the two apart, and searching for the name still finds the
  note. The frontmatter block is removed from the note body — unless it is the
  whole file, in which case the note keeps it and still takes its title and
  keywords from it. Any other property in that block (`mood`, `status`,
  `rating`) is not imported, and the review row says which ones ("… · not
  imported: status"), so nothing disappears silently.
- `[[wikilinks]]` and `[[link|alias]]` whose target is imported in the same
  batch become note links; a link to anything else stays as plain text, and a
  `[[link]]` written inside a code block or backticks is left alone. A linked
  note keeps its wikilink and shows the linked note's title, with the
  identifier behind it — `[[Reading/Zettelkasten|Zettelkasten — overview]]`
  followed by `(note://…)`. Preview renders it as the title alone, an exported
  file is still a working Obsidian link, and importing that file again
  recovers the same link rather than stacking a second identifier on it.

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
keywords, and how many links it carries — before you approve anything. When the
import finishes, the receipt adds how many of those links actually resolved
("59 notes created · 12 links resolved"); links to notes outside the batch
stayed as text and are not counted.

## Common tasks

### Create a note from a template
1. In the rail, click **New note** under "Create".
2. Under "From a template", click **Meeting notes** (or any other row).
3. The editor opens pre-filled; just start typing — autosave handles the
   rest.

### Import Markdown files or a folder

1. In the notes list, click **Add from files…**, choose **Import once**, and
   pick the first file or one folder.
2. For files, click **Add another file** as needed and enter the Library notes
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
3. Click **Choose folder…**; type a path into the **File name** field ("File
   name or path") and either press Enter (browses into it) or click **Select
   folder** (uses the folder being browsed, or the folder row you last
   clicked). (Was "type into the **Folder path** field" — superseded by
   task-32271 below: this is the files-or-one-folder dialog, whose field is
   "File name" and starts empty.) Choose a direction and local destination.
   Server sync remains unavailable until its separate capability is
   installed.
4. Choose **Check changes** and review the exact safe, attention, skipped, and
   deletion-like effects. If the folder cannot be used, the status line under
   the pane's "Add files to Library notes" heading says which rule it broke
   and what to do; choose **Choose folder…** again and check the new one.
5. Choose **Activate reviewed root**. If the review is stale, choose **Check
   again** instead. **Manage sync folders** appears in the notes toolbar once
   a root is active. The synced notes and their **⇄ Sync managed** folder are
   in the database as soon as the receipt says "N applied", but on a profile
   that already held notes the list beside you does not pick them up — its
   count and tree stay as they were, through a manual check and a source
   round trip — until you restart the app (task-32518).

Existing legacy evidence appears as a paused candidate. Open **Manage sync
folders**, choose **Review migration**, inspect the current dry-run, and
activate explicitly. The migration never inherits a legacy conflict winner or
automatic-sync setting.

### Use a note in Console
1. Open the note and click **Use in Console**.
2. You land in the Console with the note staged as context and the
   prompt "Use this note as context and help me work with it." ready to
   send or rewrite.

### Capture a Console answer as a note
The return leg of **Use in Console**. In the Console, select an assistant
reply, click **More…**, then **Capture as note**. The reply is saved here
immediately — titled with its first line of text (a leading code fence or
heading mark is dropped), holding the answer verbatim — and
a "Saved to Notes" receipt offers **Open note**, which lands you on that note
in the editor.

A captured note carries three keywords: `console`, `conversation:<id>` and
`message:<id>`. They are ordinary keywords, so filtering the list on
`conversation:` finds every answer kept from one chat. Capturing is blocked
while the Console chat is temporary — a temporary chat promises nothing is
written locally, and a note is a local write.

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
| **Ctrl+N** | Makes a new blank note and opens it — no chooser in between. Works on the Library landing (no row selected yet) as well as inside the Notes workflow — the landing's bare **n** does the same, but the footer advertises Ctrl+N in both places now. |
| **/** | Focus the note filter ("find note"), without typing a literal "/" into it. Once the filter has focus, "/" is an ordinary typeable character rather than an accelerator — a second "/" adds a literal slash, since a filter can legitimately target a folder-style path such as "Work/Q3". |
| **Escape** | Focus the rail (in **Recently deleted**, go back to the list) |
| **r** (in **Recently deleted**) | Restore the focused row |
| Enter (in "Filter notes… (Enter)") | Apply the filter |
| ↑ / ↓ (New note view) | Move between **Blank note**, **From a template…** and the template rows it opens |
| Enter (New note view) | Create from the focused row |

The footer advertises these as `ctrl+n new note | / find note | esc focus
rail`. Notes does not register **Ctrl+S** and does not replace it with
another save shortcut. Use the visible
Library notes **Save** button for an immediate save; Folder files saves
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
  ignored and the picker falls back to `[notes] sync_directory`, then to your
  home directory.
- [Lasting Notes folder sync](../../Features/notes_bidirectional_sync.md) —
  runtime, cutover, ownership, and recovery details.
- [File notes](file-notes.md) — the **Folder files** side of the source strip.
- [Library overview](../library.md) — the rail, landing canvas, and the
  other Library sources.

## Verification evidence

TASK-19012 verifies this journey through the real `LibraryScreen` hierarchy
and the shipped CSS bundle. The mounted matrix covers Library notes and its
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
  Library notes and is reported honestly in the receipt. Retry resumes only
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
confirmed Library note deletion now leaves a named inline Undo/Dismiss
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
shown before any folder is linked was full-width without the rail —
superseded by task-32173 below; see [File notes](file-notes.md).)*

*Verified against fix/library-notes-pickers — 2026-09-09 (task-32122: the
Import once and Keep-synced folder pickers used to commit the directory
being browsed and silently ignore a typed-but-unsubmitted path when
**Select folder** was pressed. Both now resolve the **Folder path** field
(on the dialog both flows open today that field is **File name** — task-32271
below) first — Enter still browses into it, and Select/Select folder use it
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
Obsidian vault detection was stated as POSIX-only, since the Windows
discovery adapter never reported a vault — superseded by task-32178 below,
which taught that adapter to detect one.)*

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

*Verified against fix/library-notes-r-editor — 2026-09-09 (task-32177: the
New-note view's and the note-loading/retry view's own Back buttons had been
left out of task-32139's back-cue unification and stayed hard-coded
"‹ Notes" at every width; both now follow the same "‹ Notes" (wide) /
"‹ Back to list" (compact) rule as Edit, Preview, and Info.)*

*Verified against fix/library-notes-r-import — 2026-09-09 (task-32176: the
group bulk actions say **Skip all on this page** / **Create all on this page**;
a structured source that fails names the record or row that failed; and
**Update existing** on an unchanged repeat now updates the note instead of
aborting the run with no receipt.)*

*Verified against fix/library-notes-r-import — 2026-09-09 (task-32178: the
receipt counts the Obsidian links it resolved; the Windows discovery adapter
now detects a vault and skips `.obsidian/`, `.trash/` and `Templates/` like the
POSIX one, so the "Not on Windows" caveat is gone; and an `aliases:` entry is
stored as `alias: <name>` so it is distinguishable from a tag.)*

*Verified against fix/library-notes-r-list — 2026-09-09 (task-32172: the
placement order is a repository parameter now — folder paging AND the
deep-link locator's page arithmetic both take it — so Sort is offered on the
folder tree again and Newest/Oldest really re-page it; it stays disabled,
with its reason, while a filter window is showing. The folder tree's default
order changes from title to newest-first with this, which is what the Sort
control has always claimed.)*

*Verified against fix/library-crit8-riders-notes — 2026-09-10 (task-32100:
opening a note now reveals and marks its row in the folder tree — the reveal
used to be abandoned by the row click's own focus change, every time.)*

*Verified against fix/library-crit8-riders-notes — 2026-09-10 (task-32106: a
Notes refresh landing mid-edit repaints the Items list, keeps its scroll
offset, and leaves the focused editor field alone; and Tab out of any editor
field — title, body or keywords — now moves focus before the next keystroke
is delivered, so typing straight through a Tab no longer appends what follows
to the field you just left.)*

*Verified against fix/library-notes-r-file-notes — 2026-09-09 (task-32173:
Folder files now keeps the Library rail before a folder is linked as well as
after, so the empty state is a mode of Notes rather than a full-width
onboarding step. Compact terminals — under about 120 columns — collapse the
rail either way, as before. See [File notes](file-notes.md).)*

*Verified against fix/library-crit9-notes — 2026-09-10 (task-32233: Escape on
the Notes list now goes where the footer chip says — out of the Filter box to
the rail's **Search Library…** box, and from there to the canvas, the same
two-step ladder Media has. task-32218: one noun per source — this page and the
canvas call the Library's own notes **Library notes** and the on-disk ones
**Folder files**; "Library database", "Database Notes" and "Folder Files" are
retired as user-visible names. task-32215: the three placement verbs (**Add to
folder**, **Move note**, **Remove placement**) are pinned to a selected row;
**Sort** on a populated list arrived with task-32172. task-32217: the note
editor's Body box takes the height its pane has spare.)*

*Verified against fix/library-crit10-notes-details — 2026-09-11 (task-32356:
**Ctrl+N** and **n** create the blank note and open it instead of posing a
nine-row chooser; the eight templates fold behind one **From a template…**
row on the New note view, which is still where the rail's **New note** and
the list's **New** go. task-32358: a fresh blank note's status reads "Empty
note — type to keep it", which agrees with the list and the rail count it
sits beside — it refines task-32133's wording, not its rule that an untouched
blank note must never claim "Saved". task-32360 AC#2, handed over from the
layout branch: below 64 columns the Notes status line and the browse toolbar
no longer clip mid-word.)*

*Verified against fix/library-crit10-docs — 2026-09-11 (task-32366: 14
critique-10 claims reconciled; surface fixes in task-32346, 32348, 32349,
32354, 32355). This page needed no correction — the note editor's autosave
story and its guarded return were already stated here; the Library overview is
what had drifted.*

*Verified against fix/library-notes-w3-sync — 2026-09-11 (task-32269, and the
task-32243 fix it waited on): the lasting-sync chapter was written from the
design and had never been walked, because no folder could be admitted — a
refused **Check changes** crashed over its own refusal and then poisoned the
folder for the whole session. Every step in this chapter has now been walked
on this branch at 235x52, in two sessions.

Session one, a 179-file vault under `$HOME`: refusal copy on a folder inside
the profile → **Choose folder…** → the `$HOME` vault → 60 safe · 0 attention →
**Activate reviewed root** → "Sync root activated. 60 applied · durable
receipt recorded" → the notes appear under a **⇄ Sync managed** folder (on
that fresh profile; on a seeded profile the list stays at its old count with
no such folder until the app restarts — superseded by task-32271 below,
rider task-32518) →
**Manage sync folders** (which only exists once a root is active) → **Check
changes** → "Manual check finished."

Session two made a real conflict — edit the note in Chatbook, edit the same
file on disk — and walked the half no earlier run could reach: **Check
changes** → "⚠ Needs attention · Next: Review changes" → **Review** →
**View comparison** (a real `--- Note / +++ File` diff with both sides' line
and character counts) → **Keep file** → **Apply reviewed** → an at-action
receipt with **Undo** and **Dismiss** → **Undo** → **Resolution history**,
where the entry is recorded "undone" → **Pause** (the action becomes Resume)
→ **Resume**. **Retarget** and **Disconnect** stay visibly disabled
throughout, as this chapter says. (The Resume step was recorded as done, but
that walk's own capture ends on "✕ Failed · Next: Review changes", and the
docs sweep reproduced it with nothing changed on either side: Resume leaves
the root paused and every later Check changes fails — superseded by
task-32271 below, rider task-32519.)

Added in this pass: what a folder and its files have to be before they can be
checked, and the named refusals. Known gap, not fixed here: the root row in
**Manage sync folders** reads "Sync folder (name unavailable before cutover)"
rather than the display name you typed — task-32451.)*

*Verified against fix/library-notes-w3-import-review — 2026-09-11 (task-32250,
task-32256, task-32257, task-32258, task-32262, task-32263, and its fix round
1): the Import once review pages by rendered rows so no group or run is cut in
two, collapses an interchangeable run to one summary row with a disclosure,
spends the path budget last so the outcome survives, and takes the pane while
it is open; a disabled primary carries its reason as text; the receipt states
its outcome once and reconciles the two denominators; dropped frontmatter
properties, the no-change/diff basis, the pre-selected collision default and
vault-aware unsupported copy are all stated on the surface; and a resolved
wikilink is stored as `[[target|title]](note://<id>)`. Fix round 1: `.git` is
skipped at the walker for every source and both platform adapters; the update
diff reduces both sides to one link spelling before comparing, so an unchanged
source shows no diff; a collapsed run's summary names the whole run's size
when a page shows only part of it; and a collapsed run's title is no longer
parsed as Textual markup, so a vault folder named `[bold]Archive` renders as
itself.*

*Verified against fix/library-notes-w3-layout — 2026-09-11 (wave-3 group
`layout`, live at 235x52 / 100x30 / 60x24 on a seeded scratch profile with a
Markdown-showcase note). task-32249: Preview fills the work pane, takes focus
on arrival so `pgup`/`pgdn` page without a click, renders an Obsidian callout
instead of printing its `[!note]` marker, and its status line names **Edit**
rather than offering to keep editing. task-32259: **Check selection** stands
with the selection summary, and Import once / Add from files close the Notes
list beside them while they are the task in hand. task-32261: the compact
select strip drops its own copy of the count and keeps the line below it, so
Done, All N, Clear and Export all stay on a 42-column pane; both counts track
the selection (task-32272, landed first). task-32270: the `‹ Library / Notes` cue is a compact
control — the wide sentences that promised it are corrected above.
task-32389: below 64 columns an empty work pane hands the whole stage to the
list.*

*Verified against fix/library-notes-w3-pickers-git — 2026-09-11 (task-32251:
every path field in these pickers now selects its pre-fill on the click that
focuses it — clicking in and typing an absolute path used to leave
`/Users/you/Users/you/vault` — and an invalid path reports on a row under the
field instead of inside the dialog's bottom border. With nothing remembered, a
picker opens at `[notes] sync_directory` before falling back to home. Verified
live at 235x52.)*

*Verified against fix/library-notes-w3-capture-console — 2026-09-11
(task-32146 and its fix round 1: Console's **More… ▸ Capture as note** walked
live at 235x52 and 100x30 — the receipt's **Open note** landed on the new note
in this screen's editor, and the note's `console` / `conversation:<id>` /
`message:<id>` keywords were read back from the database. The title wording
above gained the code-fence / heading rule in fix round 1, copy-only.)*

*Verified against fix/library-notes-w3-chrome-strip — 2026-09-11 (task-32143:
the note editor gained a chrome strip — one right-aligned row under the body
reading "N words · L:C", live at 235x52 and 100x30, hidden at 79 columns and
in Preview/Info. The editor's second, never-displayed Created/Modified/version
line was removed with it, leaving Info → Properties as its one home. The save
state was NOT moved onto the strip: it stays on the status line above the mode
controls, so nothing on screen reports saving twice — moving it is rider
task-32513.)*

*Verified against fix/library-notes-w3-backlinks-table — 2026-09-11
(task-32186: "Linked from" now reads a persisted link relation instead of
scanning every note body on every note open. Same rows, same "checking…" /
"couldn't check" / count states; a note in Trash drops out of the list and
comes back with it when restored. Measured on throwaway vaults of 1,000 /
3,000 / 10,000 notes: the lookup went from 0.85 / 3.07 / 8.79 ms — growing
with the vault — to 0.08 ms flat. Upgrading an existing database backfills
the relation from the bodies it already holds. task-32467: a "Linked from"
answer that lands while the work pane is mid-recompose is now held for the
next paint instead of terminating the app.)*

*Verified against fix/library-notes-wave3-docs — 2026-09-12 (task-32271, the
wave-3 docs sweep, on dev 7159fc0b99 merged into this branch: every wave-3
claim on this page re-walked live at 235x52, the compact claims at 100x30 and
the below-64-column claim at 60x24, on two seeded scratch profiles with
git-backed vaults under `$HOME`. Contradictions found and superseded above:
the Import once and Keep a folder synced pickers are the files-or-one-folder
dialog with an empty "File name" field, not the pre-filled "Folder path" one
(that is Folder files'); the review's pager is three stacked lines under a
page's last group, present only past one page ("Page 1 of 4" on an 81-source
vault — a 67-source vault with a collapsed run fits one page); the chooser
heading reads "Add files to Library notes." over "Choose how files should
relate to Library notes."; and the Session Git trust dialog is "Trust
repository for session changes?" (see [File notes](file-notes.md)). Two
defects found and filed rather than fixed: activating a lasting-sync root on
a profile that already holds notes leaves the Notes list at its old count
with no ⇄ Sync managed folder until the app restarts (task-32518), and Resume
after Pause always lands the root in "✕ Failed" with every later Check
changes failing (task-32519 — the paused bindings are refused by the
observation Resume runs before it re-activates them; the task-32269 walk's
own Resume capture shows the same row). Everything else on this page held:
the third-part row disambiguator, one selection count, Tab staying inside the
editor with the footer naming focus, Ctrl+End, Shift+Tab into the Title, the
delete prompt inside the Info box, Preview taking focus, the chrome strip at
235x52 and 100x30 and gone at 60x24, Linked from after import, trash and
undo, the `.git` skip, the collision default, the receipt's two
denominators, the export destination refusal, the ingest browser opening at
`[notes] sync_directory`, and Capture as note's Open note landing in this
editor.)*
